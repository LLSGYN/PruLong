import os
import json
import csv
import json
import torch
import argparse

from training.modeling_flash_llama import PawLlamaForCausalLM, get_mask as get_llama_mask
from training.modeling_flash_pangu import PawPanguForCausalLM, get_mask as get_pangu_mask

def parse_args():
    parser = argparse.ArgumentParser(description="Save mask values")
    
    parser.add_argument("--checkpoint", "-ckpt", required=True)
    parser.add_argument(
        "--model_family",
        choices=["llama", "pangu"],
        default="llama",
        help="Which model family to load when exporting masks.",
    )
    parser.add_argument("--out_path", "-o", default=None)
    parser.add_argument("--sparsity", "-sp", default=None, type=float)
    
    args = parser.parse_args()
    
    if args.out_path is None:
        args.out_path = os.path.join(
            args.checkpoint,
            "masks.tsv" if args.sparsity is None else f"masks_sp{args.sparsity}.tsv"
        )
        
    return args

@torch.no_grad()
def main():
    args = parse_args()
    
    if args.model_family == "pangu":
        ModelClass = PawPanguForCausalLM
        mask_fn = get_pangu_mask
    else:
        ModelClass = PawLlamaForCausalLM
        mask_fn = get_llama_mask

    model = ModelClass.from_pretrained(args.checkpoint)
    if args.sparsity is not None:
        print("Set to", model.round_masks_for_sparsity(args.sparsity))
        threshold = 0.0
    else:
        threshold = 0.5
    
    masks = []
    n = 0
    c = 0
    for layer in model.model.layers:
        # Stretch from 
        log_alpha = layer.self_attn.attn_mask_log_alphas
        mask = mask_fn(log_alpha, training=False, threshold_for_deterministic=threshold)
        n += mask.numel()
        c += mask.sum().item()
        masks.append(mask.tolist())
    
    print(f"Sparsity: {1-c/n}")
    
    with open(args.out_path, "w") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerows(masks)
    
    
if __name__ == "__main__":
    main()
