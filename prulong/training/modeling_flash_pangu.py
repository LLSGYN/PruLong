# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch Pangu model with PruLong mask support."""
from typing import List, Optional, Tuple, Union, Any

import importlib.util
import os
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss

import torch.distributed as dist

from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging, ModelOutput

from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update

from flash_attn import flash_attn_kvpacked_func, flash_attn_varlen_kvpacked_func
from flash_attn.bert_padding import unpad_input, pad_input

try:
    from flash_attn.layers.rotary import apply_rotary_emb_func
except ImportError:
    raise ImportError('Please install RoPE kernels: `pip install git+https://github.com/HazyResearch/flash-attention.git#subdirectory=csrc/rotary`')
from block_sparse_attn import block_streaming_attn_func

from dataclasses import dataclass

from training.distributed_attention import DistributedAttention
from training.l0 import deterministic_z_from_log_alpha, sample_z_from_log_alpha


_PANGU_CONFIG_SPEC = importlib.util.spec_from_file_location(
    "prulong.pangu.configuration",
    Path(__file__).resolve().parents[1] / "model" / "openPangu-Embedded-1B" / "configuration_openpangu_dense.py",
)
if _PANGU_CONFIG_SPEC is None or _PANGU_CONFIG_SPEC.loader is None:
    raise ImportError("Could not locate configuration_openpangu_dense.py for Pangu configuration")
_pangu_config_module = importlib.util.module_from_spec(_PANGU_CONFIG_SPEC)
_PANGU_CONFIG_SPEC.loader.exec_module(_pangu_config_module)
PanguEmbeddedConfig = _pangu_config_module.PanguEmbeddedConfig

logger = logging.get_logger(__name__)


class PawPanguConfig(PanguEmbeddedConfig):
    def __init__(self, **kwargs):
        self.local_window_size = kwargs.pop("local_window_size", 1024)
        self.disable_linear_regularization_term = kwargs.pop("disable_linear_regularization_term", False)
        self.suggested_sparsity = kwargs.pop("suggested_sparsity", None)

        # Streaming (optional)
        self.toggle_type = kwargs.pop("toggle_type", None)
        self.sink_size = kwargs.pop("sink_size", 128)

        super().__init__(**kwargs)

def get_mask(log_alpha, training=False, threshold_for_deterministic=None, apply_one=False):
    if training:
        mask = sample_z_from_log_alpha(log_alpha)
    else:
        mask = deterministic_z_from_log_alpha(log_alpha, apply_one=apply_one)
        if threshold_for_deterministic is not None:
            mask = (mask > threshold_for_deterministic).to(mask.dtype)
    return mask

def generate_streaming_info_blocksparse_flash_attn(
    sink_block_num, local_block_num, n_query_heads, device
):
    streaming_info = torch.tensor(
        [sink_block_num, local_block_num] * n_query_heads,
        device=device,
        dtype=torch.int32,
    )
    return streaming_info

def streaming_attn_varlen_kvpacked_func(
    q: torch.Tensor, kv: torch.Tensor,
    streaming_info_kwargs: dict,
    cu_seqlens: torch.Tensor, max_seqlen: int,
    dropout_p: float = 0.0,
    causal: bool = True,
    return_attn_probs: bool = False,
    window_size: Tuple[int, int] = (0, 0),
) -> Optional[torch.Tensor]:
    # kv is of shape [total_seqlen, k_or_v, num_heads, head_dim]
    k, v = kv[:, 0, :, :], kv[:, 1, :, :]
    
    total_seqlen, query_heads, head_dim = q.size()
    key_value_heads = k.size(1)
    
    # Since all heads are streaming heads
    head_mask_type = torch.tensor(
        [-1] * query_heads, device=q.device, dtype=torch.int32
    )
    
    streaming_info_kwargs["n_query_heads"] = query_heads
    streaming_info_kwargs["device"] = q.device
    streaming_info = generate_streaming_info_blocksparse_flash_attn(**streaming_info_kwargs)
    
    attn_output = block_streaming_attn_func(
        q, k, v,
        cu_seqlens, cu_seqlens,
        head_mask_type, streaming_info,
        max_seqlen, max_seqlen,
        p_dropout=dropout_p, 
        is_causal=causal,
    )
    
    return attn_output

def streaming_attn_kvpacked_func(
    q: torch.Tensor, kv: torch.Tensor,
    streaming_info_kwargs: dict,
    dropout_p: float = 0.0,
    causal: bool = True,
    return_attn_probs: bool = False,
) -> Optional[torch.Tensor]:
    # kv is of shape [bsz, kv_seq_len, k_or_v, num_heads, head_dim]
    
    bsz, seqlen, query_heads, head_dim = q.size()
    k, v = kv[:, :, 0, :, :], kv[:, :, 1, :, :]
    
    
    key_value_heads = k.size(2)
    kv_seqlen = k.size(1)
    
    q_unpad = q.view(bsz * seqlen, query_heads, head_dim)
    k_unpad = k.view(bsz * kv_seqlen, key_value_heads, head_dim)
    v_unpad = v.view(bsz * kv_seqlen, key_value_heads, head_dim)
    
    cu_seqlens_q = torch.arange(0, (bsz + 1) * seqlen, step=seqlen, dtype=torch.int32, device=q_unpad.device)
    cu_seqlens_kv = torch.arange(0, (bsz + 1) * kv_seqlen, step=kv_seqlen, dtype=torch.int32, device=k_unpad.device)
    
    # Since all heads are streaming heads
    head_mask_type = torch.tensor(
        [-1] * query_heads, device=q.device, dtype=torch.int32
    )
    
    streaming_info_kwargs["n_query_heads"] = query_heads
    streaming_info_kwargs["device"] = q_unpad.device
    streaming_info = generate_streaming_info_blocksparse_flash_attn(**streaming_info_kwargs)
    
    attn_output = block_streaming_attn_func(
        q_unpad, k_unpad, v_unpad,
        cu_seqlens_q, cu_seqlens_kv,
        head_mask_type, streaming_info,
        seqlen, seqlen,
        p_dropout=dropout_p,
        is_causal=causal,
    )
    
    return attn_output.reshape(bsz, seqlen, query_heads, head_dim)
    


def rmsnorm_func(hidden_states, weight, variance_epsilon):
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
    return (weight * hidden_states).to(input_dtype)


class PanguRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        PanguRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.register_buffer(
            "variance_epsilon",
            torch.tensor(eps),
            persistent=False,
        )

    def forward(self, hidden_states):
        return rmsnorm_func(hidden_states, self.weight, self.variance_epsilon)

class PanguEmbeddedRotaryEmbedding(nn.Module):
    def __init__(self, config: PanguEmbeddedConfig, device=None):
        super().__init__()
        # BC: "rope_type" was originally "type"
        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class PanguMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


@torch.jit.script
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    final_shape = list(hidden_states.shape[:-2]) + [-1] + [hidden_states.shape[-1]]
    expand_shape = [-1] * (len(hidden_states.shape) - 1) + [n_rep] + [-1]
    hidden_states = hidden_states.unsqueeze(-2).expand(expand_shape)
    return hidden_states.reshape(final_shape)

class PanguAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        config: PawPanguConfig,
        context_window_toggle: Optional[int] = 1024,
    ):
        """
        @context_window_toggle: if not None, the attention will be limited to a context window specified by this value
        """
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads)
        self.num_key_value_heads = getattr(config, "num_key_value_heads", self.num_heads)
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.bias)

        self.register_buffer(
            "norm_factor",
            torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32)).to(torch.get_default_dtype()),
            persistent=False,
        )

        self.rotary_emb = PanguEmbeddedRotaryEmbedding(config=self.config)

        self.distributed_attn_func = DistributedAttention(self.interpolated_attention)


        self._dtype = self.q_proj.weight.dtype
        self.attn_mask_log_alphas = nn.Parameter(torch.empty(self.num_key_value_heads, dtype=self._dtype))
        self.attn_mask_log_alphas.data.normal_(mean=4.5, std=0.01)
        self.threshold_for_deterministic = None

        self.context_window_toggle = context_window_toggle
        

        self.toggle_type = getattr(config, "toggle_type", None)
        self.sink_blocks = (config.sink_size + 127) // 128 if config.sink_size is not None else 0
        self.local_blocks = (config.local_window_size + 127) // 128 if config.local_window_size is not None else 0

        if self.toggle_type == "streaming":
            self.streaming_info_kwargs = {
                "sink_block_num": self.sink_blocks,
                "local_block_num": self.local_blocks,
            }
            self.context_window_toggle = (self.sink_blocks + self.local_blocks) * 128
        elif self.toggle_type == "local":
            self.streaming_info_kwargs = None
            self.context_window_toggle = config.local_window_size
        elif self.toggle_type is None:
            self.streaming_info_kwargs = None
            self.context_window_toggle = None
        else:
            raise ValueError(f"Unknown toggle type: {self.toggle_type}")

    @torch.no_grad()
    def set_threshold_for_deterministic(self, threshold_for_deterministic):
        self.threshold_for_deterministic = threshold_for_deterministic

    @torch.no_grad()
    def get_masks(self):
        z = get_mask(self.attn_mask_log_alphas, training=self.training, threshold_for_deterministic=self.threshold_for_deterministic)
        return z

    @torch.no_grad()
    def reset_masks(self, value=4.0):
        self.attn_mask_log_alphas.data.normal_(mean=value, std=0.01)

    @torch.no_grad()
    def fill_masks_with_value(self, value):
        if isinstance(value, float) or isinstance(value, int) or (isinstance(value, torch.Tensor) and value.numel() == 1):
            self.attn_mask_log_alphas.data.fill_(value)
        else:
            if isinstance(value, list):
                value = torch.tensor(value, dtype=self._dtype, device=self.attn_mask_log_alphas.device)
            value = value.reshape(-1)
            assert value.shape[0] == self.attn_mask_log_alphas.numel(), "Value shape does not match mask shape"
            self.attn_mask_log_alphas.data.copy_(value)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def interpolated_attention(self, q, kv, unpadded_lengths, z):
        q_varlen = q
        kv_varlen = kv

        if unpadded_lengths is not None:
            cu_seqlens, max_seqlen = unpadded_lengths
            prefix_shape = q.shape[:-2]
            # Flatten leading dims for FlashAttention varlen API: (total_q, num_heads, head_dim)
            q_varlen = q.reshape(-1, q.shape[-2], q.shape[-1])
            # (batch, seq, 2, num_heads, head_dim) -> (total_kv, 2, num_heads, head_dim)
            kv_varlen = kv.reshape(-1, kv.shape[-3], kv.shape[-2], kv.shape[-1])

            attn_output = flash_attn_varlen_kvpacked_func(
                q_varlen, kv_varlen,
                cu_seqlens, cu_seqlens,
                max_seqlen, max_seqlen,
                dropout_p=0.0, softmax_scale=1.0/self.norm_factor,
                causal=True, return_attn_probs=False,
            )
            attn_output = attn_output.reshape(*prefix_shape, attn_output.size(-2), attn_output.size(-1))
        else:
            attn_output = flash_attn_kvpacked_func(
                q, kv,
                dropout_p=0.0,
                softmax_scale=1.0/self.norm_factor,
                causal=True, return_attn_probs=False,
            )

        if self.toggle_type is None:
            return attn_output

        if self.toggle_type == "streaming":
            if unpadded_lengths is not None:
                cu_seqlens, max_seqlen = unpadded_lengths
                cw_attn_output = streaming_attn_varlen_kvpacked_func(
                    q_varlen, kv_varlen,
                    self.streaming_info_kwargs,
                    cu_seqlens, max_seqlen,
                    dropout_p=0.0, causal=True, return_attn_probs=False,
                )
                cw_attn_output = cw_attn_output.reshape(*prefix_shape, cw_attn_output.size(-2), cw_attn_output.size(-1))
            else:
                cw_attn_output = streaming_attn_kvpacked_func(
                    q, kv,
                    self.streaming_info_kwargs,
                    dropout_p=0.0, causal=True, return_attn_probs=False,
                )
                
        elif self.toggle_type == "local":
            if unpadded_lengths is not None:
                # varlen, ignore padding tokens, efficient for large batch with many paddings
                cu_seqlens, max_seqlen = unpadded_lengths

                cw_attn_output = flash_attn_varlen_kvpacked_func(
                    q_varlen, kv_varlen,
                    cu_seqlens, cu_seqlens,
                    max_seqlen, max_seqlen,
                    dropout_p=0.0, softmax_scale=1.0/self.norm_factor,
                    causal=True, return_attn_probs=False,
                    window_size=(self.context_window_toggle - 1, 0),
                )
                cw_attn_output = cw_attn_output.reshape(*prefix_shape, cw_attn_output.size(-2), cw_attn_output.size(-1))
            else:
                cw_attn_output = flash_attn_kvpacked_func(
                    q, kv,
                    dropout_p=0.0,
                    softmax_scale=1.0/self.norm_factor,
                    causal=True, return_attn_probs=False,
                    window_size=(self.context_window_toggle - 1, 0),
                )
        else:
            raise ValueError(f"Unknown toggle type: {self.toggle_type}")


        effective_attn_output = attn_output * z.unsqueeze(-1) + cw_attn_output * (1 - z).unsqueeze(-1)

        return effective_attn_output


    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        unpadded_lengths: Optional[Tuple[torch.Tensor]] = None,
        seq_parallel_group: Optional[Any] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        added_batch_dim = False
        if hidden_states.dim() == 2:
            # Handle flattened (seq_len, hidden) inputs from sequence-parallel sharding.
            hidden_states = hidden_states.unsqueeze(0)
            added_batch_dim = True
            if position_ids is not None and position_ids.dim() == 1:
                position_ids = position_ids.unsqueeze(0)

        q_len, h_size = hidden_states.size(-2), hidden_states.size(-1)


        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q = q.view(*q.shape[:-1], self.num_heads, self.head_dim)
        k = k.view(*k.shape[:-1], self.num_key_value_heads, self.head_dim)
        v = v.view(*v.shape[:-1], self.num_key_value_heads, self.head_dim)

        has_layer_past = past_key_value is not None

        if has_layer_past:
            past_kv = past_key_value[0]
            past_len = past_key_value[1]
        else:
            past_len = 0

        if position_embeddings is None:
            rotary_position_ids = position_ids
            if rotary_position_ids is None:
                rotary_position_ids = torch.arange(q.size(1), device=q.device).unsqueeze(0)
            cos, sin = self.rotary_emb(hidden_states, rotary_position_ids)
        else:
            cos, sin = position_embeddings

        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        q, k = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        kv = torch.stack([k, v], -3)
        kv = repeat_kv(kv, self.num_key_value_groups)

        # Cache QKV values
        if has_layer_past:
            new_len = past_len+q.size(1)
            if new_len > past_kv.size(1):
                past_kv = torch.cat([past_kv, torch.empty(hidden_states.size(0), 256, 2, kv.size(3), kv.size(4), dtype=kv.dtype, device=kv.device)], 1)
            past_kv[:, past_len:new_len] = kv
            kv = past_kv[:, :new_len]
        else:
            past_kv = kv
        past_key_value = (past_kv, past_len+q.size(1)) if use_cache else None

        if self.toggle_type is None:
            z = torch.ones(
                self.num_heads,
                device=hidden_states.device,
                dtype=q.dtype,
            )
        else:
            z_kv = get_mask(
                self.attn_mask_log_alphas,
                training=self.training,
                threshold_for_deterministic=self.threshold_for_deterministic,
            )  # (num_key_value_heads,)
            z = z_kv.unsqueeze(-1).expand(-1, self.num_key_value_groups).reshape(-1)

        if seq_parallel_group is not None and dist.is_initialized() and dist.get_world_size(seq_parallel_group) > 1:
            attention_func = self.distributed_attn_func
            kwargs = {"group": seq_parallel_group, "gather_idx": (0 if unpadded_lengths is not None else 1)}

            z = torch.split(z, self.num_heads // dist.get_world_size(seq_parallel_group), dim=0)[dist.get_rank(seq_parallel_group)]
        else:
            attention_func = self.interpolated_attention
            kwargs = {}

        attn_output = attention_func(q, kv, unpadded_lengths, z, **kwargs)
        attn_output = attn_output.reshape(*attn_output.shape[:-2], h_size)
        attn_output = self.o_proj(attn_output)

        if added_batch_dim:
            attn_output = attn_output.squeeze(0)

        attn_weights = None


        return z.sum(), attn_output, attn_weights, past_key_value

class PanguDecoderLayer(nn.Module):
    def __init__(
        self,
        config: PawPanguConfig,
        context_window_toggle: Optional[int] = 4096,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = PanguAttention(config=config, context_window_toggle=context_window_toggle)
        self.mlp = PanguMLP(config)
        self.input_layernorm = PanguRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = PanguRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self._fsdp_wrap = True

    @torch.no_grad()
    def set_threshold_for_deterministic(self, threshold_for_deterministic):
        self.self_attn.set_threshold_for_deterministic(threshold_for_deterministic)

    @torch.no_grad()
    def get_masks(self):
        return self.self_attn.get_masks()

    @torch.no_grad()
    def reset_masks(self, value=4.0):
        self.self_attn.reset_masks(value)

    @torch.no_grad()
    def fill_masks_with_value(self, value):
        self.self_attn.fill_masks_with_value(value)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        unpadded_lengths: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        seq_parallel_group: Optional[Any] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        z_sum, hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            unpadded_lengths=unpadded_lengths,
            seq_parallel_group=seq_parallel_group,
            position_embeddings=position_embeddings,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (z_sum, hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class PanguPreTrainedModel(PreTrainedModel):
    config_class = PawPanguConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["PanguDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

@dataclass
class BaseModelOutputWithPastAndSparsity(ModelOutput):
    last_hidden_state: torch.FloatTensor
    past_key_values: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    model_sparsity: Optional[torch.FloatTensor] = None
    target_sparsity: Optional[torch.FloatTensor] = None
    sparsity_loss: Optional[torch.FloatTensor] = None

class PanguModel(PanguPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`PanguDecoderLayer`]

    Args:
        config: PawPanguConfig
    """

    def __init__(
        self,
        config: PawPanguConfig,
    ):
        super().__init__(config)
        context_window_toggle = config.local_window_size
        disable_linear_regularization_term = config.disable_linear_regularization_term

        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.rotary_emb = PanguEmbeddedRotaryEmbedding(config=config)
        self.layers = nn.ModuleList(
            [
                PanguDecoderLayer(config, context_window_toggle=context_window_toggle)
                for _ in range(config.num_hidden_layers)
            ]
        )
        self.norm = PanguRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False

        self.total_num_heads = config.num_attention_heads * config.num_hidden_layers
        self.total_num_kv_heads = config.num_key_value_heads * config.num_hidden_layers

        self._dtype = self.norm.weight.dtype
        if disable_linear_regularization_term:
            self.sparsity_lambda_1 = torch.tensor([0.0], dtype=self._dtype)
        else:
            self.sparsity_lambda_1 = nn.Parameter(torch.tensor([0.0], dtype=self._dtype))
        self.sparsity_lambda_2 = nn.Parameter(torch.tensor([0.0], dtype=self._dtype))

        self.threshold_for_deterministic = None
        if config.suggested_sparsity is not None:
            self.round_masks_for_sparsity(config.suggested_sparsity)

        # Initialize weights and apply final processing
        self.post_init()

    @torch.no_grad()
    def set_threshold_for_deterministic(self, threshold_for_deterministic):
        for layer in self.layers:
            layer.set_threshold_for_deterministic(threshold_for_deterministic)

    @torch.no_grad()
    def get_masks(self):
        masks = []
        for layer in self.layers:
            masks.append(layer.get_masks())
        return masks

    @torch.no_grad()
    def reset_masks(self, value=4.0):
        for layer in self.layers:
            layer.reset_masks(value)
        self.sparsity_lambda_1.data.zero_()
        self.sparsity_lambda_2.data.zero_()

    @torch.no_grad()
    def get_sparsity(self):
        masks = self.get_masks()
        total_sum = 0
        for mask in masks:
            total_sum += mask.sum()
        return 1 - (total_sum / self.total_num_kv_heads)

    @torch.no_grad()
    def _pre_save_get_threshold(self):
        orig_threshold = self.threshold_for_deterministic

        sparsity_target = self.get_sparsity()
        l = 0
        r = 1
        while r - l > 1e-8:
            m = (l + r) / 2
            self.set_threshold_for_deterministic(m)
            if self.get_sparsity() > sparsity_target:
                r = m
            else:
                l = m
        m = (l + r) / 2

        self.config.suggested_threshold = m

    @torch.no_grad()
    def reset_masks_with_stripe_pattern(self, width_1, width_2, start_with_keep=True):
        if start_with_keep:
            value_1 = 10.0  # Some high value
            value_2 = -10.0  # Some low value
        else:
            value_1 = -10.0
            value_2 = 10.0
        for l, layer in enumerate(self.layers):
            value = value_1 if l % (width_1 + width_2) < width_1 else value_2
            layer.fill_masks_with_value(value)

    @torch.no_grad()
    def load_masks(self, masks):
        for l in range(len(masks)):
            self.layers[l].fill_masks_with_value(masks[l])

    @torch.no_grad()
    def round_masks_for_sparsity(self, target_sparsity):
        masks = self.get_masks()
        # masks is a list of tensors, each tensor is of shape (num_key_value_heads,)
        # First find the number of high values
        num_high = int(sum([mask.shape[0] for mask in masks]) * (1-target_sparsity))
        
        # Find the top-num_high values
        # Break ties randomly
        rng = torch.Generator()
        rng.manual_seed(42)
        value_list = [
            (i, j, masks[i][j], torch.rand(1, generator=rng).item())
            for i in range(len(masks))
            for j in range(masks[i].shape[0])
        ]
        # Sort by the random variable then resort by the value
        value_list.sort(key=lambda x: x[3])
        value_list.sort(key=lambda x: x[2], reverse=True)
        for i, j, _, _ in value_list[:num_high]:
            masks[i][j] = 10.0
        for i, j, _, _ in value_list[num_high:]:
            masks[i][j] = -10.0
            
        # Load the new masks
        self.load_masks(masks)
        
        # Return the sparsity
        return self.get_sparsity()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        unpadded_lengths: Optional[Tuple[torch.Tensor]] = None,
        seq_parallel_group: Optional[Any] = None,
        target_sparsity: Optional[float] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is None and inputs_embeds is None:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        # position_ids = None

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds
        if hidden_states.dim() == 2:
            hidden_states = hidden_states.unsqueeze(0)
        seq_len = hidden_states.size(1)
        batch_size = hidden_states.size(0)
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0)
        if position_ids.dim() == 1:
            position_ids = position_ids.unsqueeze(0)
        position_ids = position_ids.expand(batch_size, seq_len)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)


        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        z_sum = 0

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                layer_outputs = torch.utils.checkpoint.checkpoint(
                    decoder_layer,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    None,
                    unpadded_lengths,
                    output_attentions,
                    False,
                    seq_parallel_group,
                    position_embeddings,
                    use_reentrant=False,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    unpadded_lengths=unpadded_lengths,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    seq_parallel_group=seq_parallel_group,
                    position_embeddings=position_embeddings,
                )

            z_layer_sum, hidden_states = layer_outputs[0], layer_outputs[1]
            z_sum = z_sum + z_layer_sum

            if use_cache:
                next_decoder_cache += (layer_outputs[3 if output_attentions else 2],)

            if output_attentions:
                all_self_attns += (layer_outputs[2],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None

        if seq_parallel_group is not None and dist.is_initialized() and dist.get_world_size(seq_parallel_group) > 1:
            # Collect z_sum across GPUs in sequence parallel group (i.e., across all the heads during attention)
            z_sums = [torch.zeros_like(z_sum) for _ in range(dist.get_world_size(seq_parallel_group))]
            dist.all_gather(z_sums, z_sum, group=seq_parallel_group)
            z_sum = sum(z_sums)

        model_sparsity = 1 - (z_sum / self.total_num_heads)

        if target_sparsity is None:
            z_loss = None
        else:
            z_loss = self.sparsity_lambda_1.reshape([]) * (
                model_sparsity - target_sparsity
            ) + self.sparsity_lambda_2.reshape([]) * (
                model_sparsity - target_sparsity
            )**2

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, model_sparsity, target_sparsity, z_loss] if v is not None)
        return BaseModelOutputWithPastAndSparsity(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            model_sparsity=model_sparsity,
            target_sparsity=target_sparsity,
            sparsity_loss=z_loss,
        )

@dataclass
class CausalLMOutputWithPastAndSparsity(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    model_sparsity: Optional[torch.FloatTensor] = None
    target_sparsity: Optional[torch.FloatTensor] = None
    sparsity_loss: Optional[torch.FloatTensor] = None

class PawPanguForCausalLM(PanguPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(
        self,
        config,
    ):
        super().__init__(config)
        self.model = PanguModel(
            config,
        )
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.logit_block_size = int(os.environ.get("LOGIT_BLOCK_SIZE", 0))

        # Initialize weights and apply final processing
        self.post_init()

    @torch.no_grad()
    def set_threshold_for_deterministic(self, threshold_for_deterministic):
        self.model.set_threshold_for_deterministic(threshold_for_deterministic)

    @torch.no_grad()
    def get_masks(self):
        return self.model.get_masks()

    @torch.no_grad()
    def reset_masks(self, value=4.0):
        self.model.reset_masks(value)

    @torch.no_grad()
    def get_sparsity(self):
        return self.model.get_sparsity()

    @torch.no_grad()
    def reset_masks_with_stripe_pattern(self, width_1, width_2, start_with_keep=True):
        self.model.reset_masks_with_stripe_pattern(width_1, width_2, start_with_keep)
        
    @torch.no_grad()
    def load_masks(self, masks):
        self.model.load_masks(masks)
        
    @torch.no_grad()
    def round_masks_for_sparsity(self, target_sparsity):
        return self.model.round_masks_for_sparsity(target_sparsity)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def compute_loss(self, hidden_states, labels):
        logits = self.lm_head(hidden_states)
        if len(logits.shape) > 2:
            logits = logits.transpose(-1, -2)
        return F.cross_entropy(
            logits, labels,
            ignore_index=-100,
            reduction=("sum" if getattr(self, "token_scaled_loss", False) else "mean")
        )

    def save_pretrained(self, *args, **kwargs):
        # First save the suggested threshold
        self.model._pre_save_get_threshold()
        return super().save_pretrained(*args, **kwargs)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        seq_lengths: Optional[torch.Tensor] = None,
        return_token_losses: bool = False,
        shifted_labels: Optional[torch.LongTensor] = None,
        seq_parallel_group: Optional[Any] = None,
        target_sparsity: Optional[float] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer
        >>> from training.modeling_flash_pangu import PawPanguForCausalLM

        >>> model = PawPanguForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if seq_lengths is not None:
            if inputs_embeds is not None:
                assert len(inputs_embeds.shape) == 2, "inputs_embeds should be a 2D tensor with `seq_lengths`"
                # assert inputs_embeds.size(0) == seq_lengths.sum(), "inputs_embeds and seq_lengths should have the same batch size"
            else:
                assert len(input_ids.shape) == 1, "input_ids should be a 1D tensor with `seq_lengths`"
                # assert input_ids.size(0) == seq_lengths.sum(), "input_ids and seq_lengths should have the same batch size"

            assert attention_mask is None or attention_mask.all().item(), "attention_mask should be None or all ones for `seq_lengths`"
            assert not use_cache, "use_cache is not supported with `seq_lengths`"

            cu_seqlens = F.pad(torch.cumsum(seq_lengths, dim=0, dtype=torch.torch.int32), (1, 0))
            max_seqlen = seq_lengths.max().item()

            unpadded_lengths = (cu_seqlens, max_seqlen)
        elif (
            ((attention_mask is not None) and (not attention_mask.all().item()))
            and not use_cache
        ):
            if inputs_embeds is not None:
                bsz = inputs_embeds.size(0)
                inputs_embeds, unpad_indices, cu_seqlens, max_seqlen = unpad_input(inputs_embeds, attention_mask)
            else:
                bsz = input_ids.size(0)
                input_ids, unpad_indices, cu_seqlens, max_seqlen = unpad_input(input_ids.unsqueeze(-1), attention_mask)
                input_ids = input_ids.squeeze(-1)
            unpadded_lengths = (cu_seqlens, max_seqlen)
        else:
            unpadded_lengths = None

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            unpadded_lengths=unpadded_lengths,
            seq_parallel_group=seq_parallel_group,
            target_sparsity=target_sparsity,
        )
        hidden_states = outputs[0]

        if seq_lengths is None and unpadded_lengths is not None:
            hidden_states = pad_input(hidden_states, unpad_indices, bsz, max_seqlen)

        if labels is not None or shifted_labels is not None:
            if shifted_labels is not None:
                labels = shifted_labels.reshape(-1)
                hidden_states = hidden_states.reshape(-1, hidden_states.size(-1))
            else:
                labels = labels[..., 1:].reshape(-1)
                hidden_states = hidden_states[..., :-1 ,:].reshape(-1, hidden_states.size(-1))

            if self.logit_block_size > 0:
                num_valid_labels = (labels != -100).sum()
                hidden_states = torch.split(hidden_states, self.logit_block_size, dim=0)
                labels = torch.split(labels, self.logit_block_size, dim=0)

                if getattr(self, "token_scaled_loss", False):
                    loss = sum(
                        torch.utils.checkpoint.checkpoint(self.compute_loss,
                                                          hidden_state_block,
                                                          label_block,
                                                          use_reentrant=False)
                        for hidden_state_block, label_block in zip(hidden_states, labels)
                    )
                else:
                    loss = sum(
                        ((label_block != -100).sum() / num_valid_labels) *
                        torch.utils.checkpoint.checkpoint(self.compute_loss,
                                                            hidden_state_block,
                                                            label_block,
                                                            use_reentrant=False)
                        for hidden_state_block, label_block in zip(hidden_states, labels)
                    )
            else:
                loss = self.compute_loss(hidden_states, labels)

            logits = None
        else:
            logits = self.lm_head(hidden_states)
            loss = None

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPastAndSparsity(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            model_sparsity=outputs.model_sparsity,
            target_sparsity=outputs.target_sparsity,
            sparsity_loss=outputs.sparsity_loss,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past
