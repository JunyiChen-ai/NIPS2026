"""
Activation patching on Qwen2.5-7B-Instruct, following GoT (COLM 2024).

For each true/false statement pair with IDENTICAL tokenization length:
1. Wrap in few-shot prompt ending with "This statement is:"
2. Run true prompt → save all hidden states
3. Run false prompt → get baseline logit_diff = logit(TRUE) - logit(FALSE)
4. For each (layer, token_pos): run false prompt but patch true hidden state
   → measure logit_diff recovery

This localizes where truth information is encoded in the model.

Output: patching_results/ with per-(layer, token) recovery heatmap.
"""

import os
import json
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
DATASETS_BASE = "/data/jehc223/NIPS2026/datasets"
OUTPUT_DIR = "/data/jehc223/NIPS2026/extraction/patching_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Few-shot prompt template (GoT style)
FEW_SHOT_PREFIX = """\
The city of Paris is in France. This statement is: TRUE
The city of Beijing is in Japan. This statement is: FALSE
The city of Berlin is in Germany. This statement is: TRUE
The city of Cairo is in Brazil. This statement is: FALSE
"""


def load_equal_length_pairs(tokenizer):
    """Load GoT Cities pairs where true and false versions tokenize to same length."""
    import csv
    samples = []
    base = os.path.join(DATASETS_BASE, "knowledge_factual/geometry_of_truth/cities")
    for split in ["train", "val"]:
        path = os.path.join(base, f"{split}.csv")
        with open(path) as f:
            for row in csv.DictReader(f):
                samples.append({"text": row["statement"], "label": int(row["label"])})

    # Group by city to find true/false pairs
    by_city = {}
    for s in samples:
        parts = s["text"].split(" is in ")
        if len(parts) == 2:
            city = parts[0]
            if city not in by_city:
                by_city[city] = {"true": None, "false": None}
            if s["label"] == 1:
                by_city[city]["true"] = s["text"]
            else:
                by_city[city]["false"] = s["text"]

    # Keep only pairs with equal tokenization length
    pairs = []
    for v in by_city.values():
        if v["true"] is None or v["false"] is None:
            continue
        true_prompt = FEW_SHOT_PREFIX + v["true"] + " This statement is:"
        false_prompt = FEW_SHOT_PREFIX + v["false"] + " This statement is:"
        true_ids = tokenizer(true_prompt, return_tensors="pt").input_ids
        false_ids = tokenizer(false_prompt, return_tensors="pt").input_ids
        if true_ids.shape[1] == false_ids.shape[1]:
            pairs.append((true_prompt, false_prompt, v["true"], v["false"]))

    print(f"Found {len(pairs)} equal-length pairs (out of {len(by_city)} cities)")
    return pairs


def run_patching(model, tokenizer, pairs, n_pairs=100):
    """Run patching experiment.

    For each pair:
    1. Forward true prompt, save hidden states at all layers
    2. Forward false prompt, get baseline logit_diff
    3. For each (layer, token): patch true hidden into false, measure recovery

    Returns per-layer average recovery at last N token positions.
    """
    device = next(model.parameters()).device
    n_layers = model.config.num_hidden_layers
    # hidden_states from output_hidden_states=True for Qwen2:
    # index 0 = embedding output
    # index 1..n_layers-1 = output of layers 0..n_layers-2
    # index n_layers = final norm output (after last layer)
    # Total: n_layers + 1 entries
    n_hidden = n_layers + 1

    # Find TRUE/FALSE token ids
    true_token = tokenizer.encode(" TRUE", add_special_tokens=False)[-1]
    false_token = tokenizer.encode(" FALSE", add_special_tokens=False)[-1]
    print(f"TRUE token: {true_token} ('{tokenizer.decode([true_token])}')")
    print(f"FALSE token: {false_token} ('{tokenizer.decode([false_token])}')")

    pairs_to_use = pairs[:n_pairs]
    n_positions = 15  # last 15 token positions

    # Accumulate recovery per (hidden_state_index, position_from_end)
    recovery_sum = np.zeros((n_hidden, n_positions))
    counts = np.zeros((n_hidden, n_positions))

    model.eval()

    for true_prompt, false_prompt, true_text, false_text in tqdm(pairs_to_use, desc="Patching"):
        true_ids = tokenizer(true_prompt, return_tensors="pt").input_ids.to(device)
        false_ids = tokenizer(false_prompt, return_tensors="pt").input_ids.to(device)
        seq_len = true_ids.shape[1]
        assert true_ids.shape == false_ids.shape, "Token lengths must match"

        # 1. True forward: save hidden states
        with torch.no_grad():
            true_out = model(true_ids, output_hidden_states=True)
        true_hidden = [h.detach().clone() for h in true_out.hidden_states]
        true_logit_diff = (true_out.logits[0, -1, true_token] - true_out.logits[0, -1, false_token]).item()

        # 2. False forward: baseline
        with torch.no_grad():
            false_out = model(false_ids, output_hidden_states=True)
        false_logit_diff = (false_out.logits[0, -1, true_token] - false_out.logits[0, -1, false_token]).item()

        # Baseline: how much logit_diff needs to recover
        # true_logit_diff should be positive, false_logit_diff should be negative
        total_gap = true_logit_diff - false_logit_diff
        if abs(total_gap) < 0.1:
            continue  # Model can't distinguish, skip

        # 3. Patch each (layer, position)
        n_pos = min(n_positions, seq_len)

        for hi in range(n_hidden):
            for pos_offset in range(n_pos):
                pos = seq_len - 1 - pos_offset  # from last token backwards

                patch_val = true_hidden[hi][0, pos, :].clone()

                # Determine which module to hook
                if hi == 0:
                    target_module = model.model.embed_tokens
                elif hi < n_layers:
                    # hi = 1..n_layers-1 → model.layers[0..n_layers-2]
                    target_module = model.model.layers[hi - 1]
                else:
                    # hi = n_layers → final norm
                    target_module = model.model.norm

                def make_hook(target_pos, val):
                    def hook_fn(module, input, output):
                        if isinstance(output, tuple):
                            out = list(output)
                            out[0] = out[0].clone()
                            out[0][0, target_pos, :] = val
                            return tuple(out)
                        else:
                            output = output.clone()
                            output[0, target_pos, :] = val
                            return output
                    return hook_fn

                handle = target_module.register_forward_hook(make_hook(pos, patch_val))

                with torch.no_grad():
                    patched_out = model(false_ids)

                handle.remove()

                patched_logit_diff = (patched_out.logits[0, -1, true_token] - patched_out.logits[0, -1, false_token]).item()

                # Recovery: fraction of the gap recovered
                rec = (patched_logit_diff - false_logit_diff) / total_gap
                rec = max(0.0, min(1.0, rec))  # clamp to [0, 1]

                recovery_sum[hi, pos_offset] += rec
                counts[hi, pos_offset] += 1

        # Clean up
        del true_hidden, true_out, false_out

    # Average
    mask = counts > 0
    avg_recovery = np.zeros_like(recovery_sum)
    avg_recovery[mask] = recovery_sum[mask] / counts[mask]

    return avg_recovery, counts


def main():
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto",
        attn_implementation="eager",
    )
    model.eval()
    n_layers = model.config.num_hidden_layers
    print(f"Model: {n_layers} layers")

    print("\nLoading equal-length pairs...")
    pairs = load_equal_length_pairs(tokenizer)

    print(f"\nRunning patching (n_pairs={min(100, len(pairs))})...")
    recovery, counts = run_patching(model, tokenizer, pairs, n_pairs=100)

    # Save
    results = {
        "model": MODEL_NAME,
        "n_layers": n_layers,
        "n_hidden_states": n_layers + 1,
        "recovery": recovery.tolist(),
        "counts": counts.tolist(),
        "position_meaning": "index 0 = last token (end of 'is:'), 1 = second-to-last, etc.",
        "hidden_state_meaning": "index 0 = embedding, 1..n_layers-1 = layers 0..n_layers-2, n_layers = final norm",
        "metric": "logit_diff recovery: (patched_diff - false_diff) / (true_diff - false_diff), clamped to [0,1]",
        "prompt_format": "few-shot GoT style ending with 'This statement is:'",
    }

    out_path = os.path.join(OUTPUT_DIR, "patching_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")

    # Summary: recovery at last token per layer
    print("\n=== Recovery at last token (pos=0) per hidden state ===")
    for hi in range(recovery.shape[0]):
        if hi == 0:
            name = "embedding"
        elif hi < n_layers:
            name = f"layer_{hi-1}"
        else:
            name = "final_norm"
        print(f"  {name:>15}: {recovery[hi, 0]:.4f}")

    # Best transformer layer for last token (exclude embedding and final_norm)
    layer_recovery = recovery[1:n_layers, 0]
    best_layer = layer_recovery.argmax()
    print(f"\nBest transformer layer (last token): layer_{best_layer} "
          f"(recovery={layer_recovery[best_layer]:.4f})")

    # Top-5 layers
    top5 = np.argsort(layer_recovery)[::-1][:5]
    print(f"Top-5 layers: {['layer_' + str(l) for l in top5]}")
    print(f"Recoveries:   {[f'{layer_recovery[l]:.4f}' for l in top5]}")


if __name__ == "__main__":
    main()
