import os
import json
import math
import importlib.util
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import numpy as np
import random
import argparse

from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, LlamaTokenizer, LlamaForCausalLM, AutoModelForCausalLM

from llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
from transformers.cache_utils import Cache
import types
from flash_attn import flash_attn_func

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None, choices=["llama2-7b-chat-4k", "longchat-v1.5-7b-32k", "xgen-7b-8k", "internlm-7b-8k", "chatglm2-6b", "chatglm2-6b-32k", "chatglm3-6b-32k", "vicuna-v1.5-7b-16k", "pangu-1b-32k"])
    parser.add_argument('--e', action='store_true', help="Evaluate on LongBench-E")
    parser.add_argument('--block-size', type=int, default=32, help="Block size used for sparse block search")
    return parser.parse_args(args)


def load_solution_functions(base_dir: str):
    """Dynamically load the contestant provided solution module."""

    solution_path = os.path.join(base_dir, "solution.py")
    spec = importlib.util.spec_from_file_location("longbench_solution", solution_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to locate solution.py at {solution_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    if not hasattr(module, "prepare") or not hasattr(module, "search"):
        raise AttributeError("solution.py must define `prepare` and `search` functions")
    return module.prepare, module.search


class SparseSearchContext:
    """Runtime context for executing contestant provided sparse block search."""

    def __init__(self, search_fn):
        self.search_fn = search_fn
        self.enabled: bool = False
        self.kv_representations: List[Optional[torch.Tensor]] = []
        self.num_blocks_total: List[int] = []
        self.prefill_lengths: List[int] = []
        self.block_size: int = 1
        self.head_counts: List[int] = []
        self.current_step: Optional[int] = None
        self.layer_last_step: Dict[int, int] = {}
        self.density_sum: float = 0.0
        self.density_count: int = 0
        self.num_layers: int = 0
        self.generated_tokens: int = 0

    def set_num_layers(self, num_layers: int) -> None:
        self.num_layers = num_layers

    def configure(
        self,
        kv_representations: List[Optional[torch.Tensor]],
        num_blocks_total: List[int],
        prefill_lengths: List[int],
        block_size: int,
    ) -> None:
        self.kv_representations = kv_representations
        self.num_blocks_total = num_blocks_total
        self.prefill_lengths = prefill_lengths
        self.block_size = max(block_size, 1)
        self.head_counts = [int(kv.shape[1]) if kv is not None else 0 for kv in kv_representations]
        self.density_sum = 0.0
        self.density_count = 0

    def enable(self) -> None:
        self.enabled = True
        self.current_step = None
        self.layer_last_step.clear()
        self.generated_tokens = 0

    def disable(self) -> None:
        self.enabled = False
        self.current_step = None
        self.layer_last_step.clear()
        self.generated_tokens = 0

    def start_step(self, step_index: int) -> None:
        if not self.enabled:
            return
        self.current_step = step_index
        self.layer_last_step.clear()

    def finish_step(self) -> None:
        if not self.enabled:
            return
        self.current_step = None

    def set_generation_tokens(self, count: int) -> None:
        self.generated_tokens = max(count, 0)

    def should_process(self, layer_id: int, hidden_states: Optional[torch.Tensor]) -> bool:
        if not self.enabled or self.current_step is None:
            return False
        if layer_id >= len(self.kv_representations):
            return False
        if self.kv_representations[layer_id] is None:
            return False
        if hidden_states is None or hidden_states.size(1) != 1:
            return False
        return True

    def _normalize_tables(
        self, layer_id: int, block_tables: Optional[Iterable], num_blocks_total: int
    ) -> List[List[int]]:
        head_count = self.head_counts[layer_id] if layer_id < len(self.head_counts) else 0
        normalized: List[List[int]] = [[] for _ in range(head_count)]
        if not block_tables:
            if num_blocks_total <= 0:
                return normalized
            for head_idx in range(head_count):
                normalized[head_idx] = list(range(num_blocks_total))
            return normalized

        for head_idx in range(min(len(block_tables), head_count)):
            head_table = block_tables[head_idx]
            if isinstance(head_table, torch.Tensor):
                head_list = head_table.detach().cpu().tolist()
            elif isinstance(head_table, np.ndarray):
                head_list = head_table.tolist()
            else:
                head_list = list(head_table)
            deduped: List[int] = []
            seen: set = set()
            for block_id in head_list:
                block_int = int(block_id)
                if block_int < 0:
                    continue
                if block_int >= num_blocks_total:
                    block_int = num_blocks_total - 1
                if block_int in seen:
                    continue
                seen.add(block_int)
                deduped.append(block_int)
                if len(deduped) >= num_blocks_total:
                    break
            normalized[head_idx] = deduped
        return normalized

    def _record_density(self, layer_id: int, tables: List[List[int]], num_blocks_total: int) -> None:
        if self.current_step is None:
            return
        if self.layer_last_step.get(layer_id) == self.current_step:
            return
        self.layer_last_step[layer_id] = self.current_step
        head_count = len(tables)
        if num_blocks_total <= 0:
            num_blocks_total = 1
        for head_idx in range(head_count):
            selected = min(len(tables[head_idx]), num_blocks_total)
            self.density_sum += float(selected) / float(num_blocks_total)
            self.density_count += 1

    def _build_indices(
        self,
        layer_id: int,
        tables: List[List[int]],
        kv_seq_len: int,
        device: torch.device,
    ) -> List[torch.Tensor]:
        if layer_id >= len(self.prefill_lengths):
            return []
        head_count = len(tables)
        prefill_len = self.prefill_lengths[layer_id]
        if kv_seq_len <= 0 or head_count == 0:
            return []

        decode_tokens: List[int] = []
        if kv_seq_len > prefill_len:
            decode_tokens = list(range(prefill_len, kv_seq_len))

        head_indices: List[torch.Tensor] = []
        for head_idx, blocks in enumerate(tables):
            token_indices: List[int] = []
            seen: set[int] = set()
            limit = min(prefill_len, kv_seq_len)
            for block_id in blocks:
                start = block_id * self.block_size
                end = min(start + self.block_size, limit)
                if start >= end:
                    continue
                for position in range(start, end):
                    if position in seen:
                        continue
                    seen.add(position)
                    token_indices.append(position)

            if decode_tokens:
                token_indices.extend(decode_tokens)

            if not token_indices:
                if limit > 0:
                    token_indices.extend(range(limit))
                if decode_tokens:
                    token_indices.extend(decode_tokens)

            indices_tensor = (
                torch.tensor(token_indices, dtype=torch.long, device=device)
                if token_indices
                else torch.zeros(0, dtype=torch.long, device=device)
            )
            head_indices.append(indices_tensor)

        return head_indices

    def compute_head_indices(
        self,
        layer_id: int,
        query_states: torch.Tensor,
        kv_seq_len: int,
        device: torch.device,
    ) -> List[torch.Tensor]:
        if layer_id >= len(self.kv_representations):
            return []
        kv_repr = self.kv_representations[layer_id]
        if kv_repr is None:
            return []
        num_blocks_total = self.num_blocks_total[layer_id] if layer_id < len(self.num_blocks_total) else 0
        if num_blocks_total <= 0:
            return []
        try:
            if query_states.dim() == 3:
                query_for_search = query_states.detach()
            elif query_states.dim() == 2:
                query_for_search = query_states.detach().unsqueeze(0)
            else:
                query_for_search = query_states.detach().view(1, -1, query_states.shape[-1])
            block_tables = self.search_fn(kv_repr, query_for_search, num_blocks_total, layer_id)
        except Exception:
            block_tables = None
        tables = self._normalize_tables(layer_id, block_tables, num_blocks_total)
        self._record_density(layer_id, tables, num_blocks_total)
        return self._build_indices(layer_id, tables, kv_seq_len, device)

    @property
    def sparsity(self) -> float:
        if self.density_count == 0:
            return 0.0
        density = self.density_sum / float(self.density_count)
        density = max(0.0, min(1.0, density))
        return 1.0 - density


def make_llama_sparse_forward(layer_id: int, context: SparseSearchContext):
    """Create a sparse-aware forward pass for LLaMA attention layers."""

    def sparse_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        original_forward = getattr(self, "_sparse_original_forward")
        if (not context.enabled) or (not context.should_process(layer_id, hidden_states)):
            return original_forward(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        bsz, q_len, _ = hidden_states.size()

        if self.config.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if position_embeddings is None:
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        kv_seq_len = key_states.shape[-2]
        indices_per_head = context.compute_head_indices(
            layer_id,
            query_states[:, :, -1, :],
            kv_seq_len,
            key_states.device,
        )

        if len(indices_per_head) == 0:
            indices_per_head = [
                torch.arange(kv_seq_len, device=key_states.device, dtype=torch.long)
                for _ in range(self.num_key_value_heads)
            ]

        attn_output = torch.zeros(
            (bsz, self.num_heads, q_len, self.head_dim),
            dtype=query_states.dtype,
            device=query_states.device,
        )

        dropout_p = 0.0 if not self.training else self.attention_dropout

        for kv_head_idx in range(self.num_key_value_heads):
            group_start = kv_head_idx * self.num_key_value_groups
            group_end = group_start + self.num_key_value_groups
            q_group = query_states[:, group_start:group_end, :, :]

            if kv_head_idx < len(indices_per_head):
                head_indices = indices_per_head[kv_head_idx]
            else:
                head_indices = torch.arange(kv_seq_len, device=key_states.device, dtype=torch.long)

            if head_indices.numel() == 0:
                head_indices = torch.arange(kv_seq_len, device=key_states.device, dtype=torch.long)

            k_head = torch.index_select(key_states[:, kv_head_idx, :, :], 1, head_indices).contiguous()
            v_head = torch.index_select(value_states[:, kv_head_idx, :, :], 1, head_indices).contiguous()

            k_flash = k_head.unsqueeze(2)
            v_flash = v_head.unsqueeze(2)
            q_flash = q_group.transpose(1, 2).contiguous()

            attn_group = flash_attn_func(
                q_flash,
                k_flash,
                v_flash,
                dropout_p=dropout_p,
                softmax_scale=self.scaling,
                causal=False,
            )

            attn_output[:, group_start:group_end, :, :] = attn_group.transpose(1, 2)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, -1)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        attn_weights_to_return = attn_weights if output_attentions else None
        return attn_output, attn_weights_to_return, past_key_value

    return sparse_forward


def patch_sparse_attention(model, context: SparseSearchContext):
    from transformers.models.llama.modeling_llama import LlamaAttention

    layer_index = 0
    for module in model.modules():
        if isinstance(module, LlamaAttention):
            if not hasattr(module, "_sparse_original_forward"):
                module._sparse_original_forward = module.forward
                module.forward = types.MethodType(make_llama_sparse_forward(layer_index, context), module)
            layer_index += 1
    context.set_num_layers(layer_index)
    return layer_index


def get_eos_token_id(tokenizer) -> Optional[int]:
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if isinstance(eos_token_id, (list, tuple)):
        return eos_token_id[0] if eos_token_id else None
    return eos_token_id


def run_prefill(model, input_kwargs: Dict[str, torch.Tensor]):
    with torch.no_grad():
        return model(**input_kwargs, use_cache=True, return_dict=True)


def prepare_kv_representations(
    past_key_values, block_size: int, prepare_fn
) -> Tuple[List[Optional[torch.Tensor]], List[int]]:
    kv_repre_list: List[Optional[torch.Tensor]] = []
    prefill_lengths: List[int] = []
    for layer_id, past in enumerate(past_key_values):
        if not isinstance(past, (tuple, list)) or len(past) < 2:
            kv_repre_list.append(None)
            prefill_lengths.append(0)
            continue
        key_cache, value_cache = past[0], past[1]
        if key_cache is None or value_cache is None:
            kv_repre_list.append(None)
            prefill_lengths.append(0)
            continue
        k_full = key_cache.squeeze(0).permute(1, 0, 2).contiguous()
        v_full = value_cache.squeeze(0).permute(1, 0, 2).contiguous()
        prefill_lengths.append(k_full.shape[0])
        try:
            kv_repr = prepare_fn(k_full, v_full, block_size=block_size)
        except Exception:
            kv_repr = None
        kv_repre_list.append(kv_repr)
    return kv_repre_list, prefill_lengths


def run_sparse_generation(
    model,
    input_tensors: Dict[str, torch.Tensor],
    max_new_tokens: int,
    eos_token_id: Optional[int],
    min_decode_tokens: int,
    block_size: int,
    prepare_fn,
    context: SparseSearchContext,
):
    if "input_ids" not in input_tensors:
        raise ValueError("Sparse generation requires tokenized input ids")
    input_kwargs = {key: value for key, value in input_tensors.items()}
    input_ids = input_kwargs["input_ids"]
    attention_mask = input_kwargs.get("attention_mask")
    prefill_outputs = run_prefill(model, input_kwargs)
    past_key_values = prefill_outputs.past_key_values
    kv_repre_list, prefill_lengths = prepare_kv_representations(past_key_values, block_size, prepare_fn)
    seq_len = input_ids.shape[1]
    effective_block = max(block_size, 1)
    num_blocks_total = [math.ceil(max(length, 1) / effective_block) for length in prefill_lengths]
    context.configure(kv_repre_list, num_blocks_total, prefill_lengths, block_size)
    context.enable()

    generated_ids = input_ids.clone()
    if attention_mask is not None:
        current_attention_mask = attention_mask.clone()
    else:
        current_attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=input_ids.device)
    past_key_values_current = past_key_values

    for step in range(max_new_tokens):
        context.set_generation_tokens(step)
        context.start_step(step)
        last_token = generated_ids[:, -1:]
        ones = torch.ones((current_attention_mask.size(0), 1), dtype=current_attention_mask.dtype, device=current_attention_mask.device)
        current_attention_mask = torch.cat([current_attention_mask, ones], dim=-1)
        decode_kwargs = {
            "input_ids": last_token,
            "attention_mask": current_attention_mask,
            "past_key_values": past_key_values_current,
            "use_cache": True,
            "return_dict": True,
        }
        with torch.no_grad():
            outputs = model(**decode_kwargs)
        context.finish_step()
        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated_ids = torch.cat([generated_ids, next_token], dim=-1)
        past_key_values_current = outputs.past_key_values
        if eos_token_id is not None and next_token.item() == eos_token_id and (step + 1) >= min_decode_tokens:
            break

    context.disable()
    return generated_ids, context.sparsity

# This is the customized building prompt for chat models
def build_chat(tokenizer, prompt, model_name):
    if "chatglm3" in model_name:
        prompt = tokenizer.build_chat_input(prompt)
    elif "chatglm" in model_name:
        prompt = tokenizer.build_prompt(prompt)
    elif "longchat" in model_name or "vicuna" in model_name:
        from fastchat.model import get_conversation_template
        conv = get_conversation_template("vicuna")
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
    elif "llama2" in model_name:
        prompt = f"[INST]{prompt}[/INST]"
    elif "xgen" in model_name:
        header = (
            "A chat between a curious human and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the human's questions.\n\n"
        )
        prompt = header + f" ### Human: {prompt}\n###"
    elif "pangu" in model_name:
        messages = [
            {"role": "system", "content": "你是华为聊天助手。"},
            {"role": "user", "content": prompt}
        ]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    elif "internlm" in model_name:
        prompt = f"<|User|>:{prompt}<eoh>\n<|Bot|>:"
    return prompt

def post_process(response, model_name):
    if "xgen" in model_name:
        response = response.strip().replace("Assistant:", "")
    elif "internlm" in model_name:
        response = response.split("<eoa>")[0]
    return response

def get_pred(rank, world_size, data, max_length, max_gen, prompt_format, dataset, device, model_name, model2path, out_path, block_size):
    device = torch.device(f'cuda:{rank}')
    model, tokenizer = load_model_and_tokenizer(model2path[model_name], model_name, device)
    prepare_fn, search_fn = load_solution_functions(os.path.dirname(os.path.abspath(__file__)))
    search_context = SparseSearchContext(search_fn)
    patch_sparse_attention(model, search_context)
    eos_token_id = get_eos_token_id(tokenizer)

    for json_obj in tqdm(data):
        prompt = prompt_format.format(**json_obj)
        # truncate to fit max_length (we suggest truncate in the middle, since the left and right side may contain crucial instructions)
        tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
        if "chatglm3" in model_name:
            tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt", add_special_tokens=False).input_ids[0]
        if len(tokenized_prompt) > max_length:
            half = int(max_length/2)
            prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True)+tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
        if dataset not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]: # chat models are better off without build prompts on these tasks
            prompt = build_chat(tokenizer, prompt, model_name)
        if "chatglm3" in model_name:
            if dataset in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]:
                input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
            else:
                input = prompt.to(device)
        else:
            input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
        context_length = input.input_ids.shape[-1]
        if dataset == "samsum":
            output_full = model.generate(
                **input,
                max_new_tokens=max_gen,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
                min_length=context_length + 1,
                eos_token_id=[tokenizer.eos_token_id, tokenizer.encode("\n", add_special_tokens=False)[-1]],
            )[0]
        else:
            output_full = model.generate(
                **input,
                max_new_tokens=max_gen,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
            )[0]
        baseline_pred = tokenizer.decode(output_full[context_length:], skip_special_tokens=True)
        baseline_pred = post_process(baseline_pred, model_name)

        sparse_pred = baseline_pred
        sparsity = 0.0
        if hasattr(input, "items"):
            input_tensors = {key: value for key, value in input.items()}
            min_decode_tokens = 1 if dataset == "samsum" else 0
            try:
                generated_sparse_ids, sparsity = run_sparse_generation(
                    model,
                    input_tensors,
                    max_gen,
                    eos_token_id,
                    min_decode_tokens,
                    block_size,
                    prepare_fn,
                    search_context,
                )
                sparse_pred_raw = tokenizer.decode(generated_sparse_ids[0, context_length:], skip_special_tokens=True)
                sparse_pred = post_process(sparse_pred_raw, model_name)
            except Exception:
                sparse_pred = baseline_pred
                sparsity = 0.0

        with open(out_path, "a", encoding="utf-8") as f:
            json.dump(
                {
                    "pred_full": baseline_pred,
                    "pred_sparse": sparse_pred,
                    "sparsity": sparsity,
                    "answers": json_obj["answers"],
                    "all_classes": json_obj["all_classes"],
                    "length": json_obj["length"],
                },
                f,
                ensure_ascii=False,
            )
            f.write('\n')
    dist.destroy_process_group()

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

def load_model_and_tokenizer(path, model_name, device):
    if "chatglm" in model_name or "internlm" in model_name or "pangu" in model_name:
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device)
    elif "llama2" in model_name:
        replace_llama_attn_with_flash_attn()
        tokenizer = LlamaTokenizer.from_pretrained(path)
        model = LlamaForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16).to(device)
    elif "longchat" in model_name or "vicuna" in model_name:
        from fastchat.model import load_model
        replace_llama_attn_with_flash_attn()
        model, _ = load_model(
            path,
            device='cpu',
            num_gpus=0,
            load_8bit=False,
            cpu_offloading=False,
            debug=False,
        )
        model = model.to(device)
        model = model.bfloat16()
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
    model = model.eval()
    return model, tokenizer

if __name__ == '__main__':
    seed_everything(42)
    args = parse_args()
    world_size = torch.cuda.device_count()
    mp.set_start_method('spawn', force=True)

    model2path = json.load(open("config/model2path.json", "r"))
    model2maxlen = json.load(open("config/model2maxlen.json", "r"))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name = args.model
    # define your model
    max_length = model2maxlen[model_name]
    if args.e:
        datasets = ["qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "gov_report", "multi_news", \
            "trec", "triviaqa", "samsum", "passage_count", "passage_retrieval_en", "lcc", "repobench-p"]
    else:
        datasets = ["gov_report"]
        # datasets = ["narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh", "hotpotqa", "2wikimqa", "musique", \
        #             "dureader", "gov_report", "qmsum", "multi_news", "vcsum", "trec", "triviaqa", "samsum", "lsht", \
        #             "passage_count", "passage_retrieval_en", "passage_retrieval_zh", "lcc", "repobench-p"]
    # we design specific prompt format and max generation length for each task, feel free to modify them to optimize model output
    dataset2prompt = json.load(open("config/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("config/dataset2maxlen.json", "r"))
    # predict on each dataset
    if not os.path.exists("pred"):
        os.makedirs("pred")
    if not os.path.exists("pred_e"):
        os.makedirs("pred_e")
    for dataset in datasets:
        if args.e:
            data = load_dataset('THUDM/LongBench', f"{dataset}_e", split='test')
            if not os.path.exists(f"pred_e/{model_name}"):
                os.makedirs(f"pred_e/{model_name}")
            out_path = f"pred_e/{model_name}/{dataset}.jsonl"
        else:
            data = load_dataset('THUDM/LongBench', dataset, split='test')
            if not os.path.exists(f"pred/{model_name}"):
                os.makedirs(f"pred/{model_name}")
            out_path = f"pred/{model_name}/{dataset}.jsonl"
        prompt_format = dataset2prompt[dataset]
        max_gen = dataset2maxlen[dataset]
        data_all = [data_sample for data_sample in data]
        data_subsets = [data_all[i::world_size] for i in range(world_size)]
        processes = []
        for rank in range(world_size):
            p = mp.Process(
                target=get_pred,
                args=(
                    rank,
                    world_size,
                    data_subsets[rank],
                    max_length,
                    max_gen,
                    prompt_format,
                    dataset,
                    device,
                    model_name,
                    model2path,
                    out_path,
                    args.block_size,
                ),
            )
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
