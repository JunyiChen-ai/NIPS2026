"""
Feature extraction for new datasets (v2).

Reads all.jsonl from datasets_prepared/ and extracts 13 features in a single
pass per dataset. Split-level feature directories are created afterward by
split_features.py using index slicing (no redundant extraction).

Feature semantics (13 per sample):
  INPUT-SIDE (from Pass 1: model.generate prefill):
    1. input_last_token_hidden  — hidden state at last input token, per layer
       Shape: (n_layers+2, hidden_dim). Layers = [embed, layer_0..layer_{n-1}, final_norm].
       Semantic: the model's representation of the full input at the last position.
       Used by: LR Probe, PCA+LR, ITI, KB MLP (layer-selected).
    2. input_mean_pool_hidden   — mean-pooled hidden states over all input tokens, per layer
       Shape: (n_layers+2, hidden_dim).
       Semantic: average representation across all input positions.
    3. input_per_head_activation — o_proj input at last token, reshaped to per-head
       Shape: (n_layers, n_heads, head_dim).
       Semantic: what each attention head contributes at the last position.
       Used by: ITI per-head probes, Attention Satisfies.
    4. input_logit_stats        — statistics of logit distribution at last input position
       Dict: {logsumexp, max_prob, entropy, top5_values, top5_indices}.
       Semantic: model's prediction confidence before generation starts.
       Used by: LLM-Check, SeaKR energy score.
    5. input_attn_stats         — attention pattern statistics at last input token
       Shape: (n_layers, n_heads, 3) = [skewness, entropy, diag_logmean].
       Semantic: how focused/spread the attention is at each head.
       Used by: Attention Satisfies probe.
    6. input_attn_value_norms   — attention-weighted value norms per head
       Shape: (n_layers, n_heads, seq_len). Variable seq_len, zero-padded.
       Semantic: contribution magnitude of each input token through each head.

  GENERATION-SIDE (from Pass 2: replay forward on prompt+generated):
    7. gen_last_token_hidden    — hidden state at last generated token, per layer
       Shape: (n_layers+2, hidden_dim).
       Semantic: model's final state after full generation.
    8. gen_mean_pool_hidden     — mean-pooled hidden states over generated tokens only
       Shape: (n_layers+2, hidden_dim).
       Semantic: average representation of what the model generated.
    9. gen_per_token_hidden_last_layer — per-token hidden states from last layer
       Shape: (gen_len, hidden_dim). Variable gen_len, zero-padded.
       Semantic: token-level representations for sequence-level analysis.
       Used by: LID (intrinsic dimensionality).
   10. gen_logit_stats_last     — logit statistics at last generated token
       Dict: same structure as input_logit_stats.
       Semantic: model's confidence at the end of generation.
   11. gen_attn_stats_last      — attention statistics at last generated token
       Shape: (n_layers, n_heads, 3).
       Semantic: attention patterns during final generation step.
   12. gen_step_boundary_hidden — hidden states at paragraph boundaries ("\\n\\n")
       List of (n_layers+2, hidden_dim) tensors at each "\\n\\n" position.
       Semantic: representations at reasoning step transitions.
       (Currently not consumed by any of the 12 methods; stored for future use.)
   13. gen_step_boundary_indices — token indices where "\\n\\n" occurs in generation.

Model: Qwen2.5-7B-Instruct, bfloat16, 2×A100, device_map='auto'.
Method: 2-pass (generate with prefill hooks → replay forward with hooks).
"""

import os
import sys
import json
import torch
from tqdm import tqdm

# Reuse the existing FeatureExtractor and save logic
sys.path.insert(0, os.path.dirname(__file__))
from extract_features import (
    FeatureExtractor, save_split_features, is_split_done,
    MODEL_NAME, MAX_SEQ_LEN, BATCH_SIZE,
)

OUTPUT_DIR = "/data/jehc223/NIPS2026/extraction/features"
PREPARED_DIR = "/data/jehc223/NIPS2026/datasets_prepared"

# Datasets to extract (fava/ragtruth each extracted once, label variants share features)
DATASETS = [
    "common_claim_3class",
    "when2call_3class",
    "fava",
    "ragtruth",
]


def load_all_samples(dataset_name):
    """Load all.jsonl for a dataset."""
    path = os.path.join(PREPARED_DIR, dataset_name, "all.jsonl")
    samples = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            samples.append({
                "text": d["text"],
                "label": d["label"],
                "label_multi": d.get("label_multi"),
            })
    return samples


def main():
    extractor = FeatureExtractor(MODEL_NAME)

    for ds_name in DATASETS:
        # Check if already done
        if is_split_done(OUTPUT_DIR, ds_name, "all"):
            print(f"\nSkipping {ds_name}/all (already done)")
            continue

        samples = load_all_samples(ds_name)
        print(f"\nProcessing {ds_name}/all: {len(samples)} samples")

        current_batch_size = BATCH_SIZE
        results = {k: [] for k in [
            "input_last_token_hidden", "input_mean_pool_hidden",
            "input_per_head_activation", "input_logit_stats",
            "input_attn_stats", "input_attn_value_norms",
            "gen_last_token_hidden", "gen_mean_pool_hidden",
            "gen_per_token_hidden_last_layer", "gen_logit_stats_last",
            "gen_attn_stats_last", "gen_step_boundary_hidden",
            "gen_step_boundary_indices",
            "labels", "texts", "gen_texts", "input_seq_lens", "gen_lens",
        ]}

        sample_idx = 0
        pbar = tqdm(total=len(samples), desc=f"{ds_name}/all")
        while sample_idx < len(samples):
            batch_end = min(sample_idx + current_batch_size, len(samples))
            batch_samples = samples[sample_idx:batch_end]
            batch_texts = [s["text"] for s in batch_samples]

            try:
                batch_features = extractor.extract_batch(batch_texts)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                extractor._clear_all()
                old_bs = current_batch_size
                if old_bs <= 1:
                    raise RuntimeError(f"OOM even with batch_size=1 at sample {sample_idx}")
                current_batch_size = max(1, old_bs // 2)
                print(f"\n  OOM at batch_size={old_bs}, reducing to {current_batch_size}")
                continue

            for sample, feat in zip(batch_samples, batch_features):
                results["labels"].append(sample["label"])
                results["texts"].append(sample["text"])
                results["gen_texts"].append(feat["gen_text"])
                results["input_seq_lens"].append(feat["input_seq_len"])
                results["gen_lens"].append(feat["gen_len"])
                results["gen_step_boundary_indices"].append(feat["gen_step_boundary_indices"])
                for k in ["input_last_token_hidden", "input_mean_pool_hidden",
                           "input_per_head_activation", "input_logit_stats",
                           "input_attn_stats", "input_attn_value_norms",
                           "gen_last_token_hidden", "gen_mean_pool_hidden",
                           "gen_per_token_hidden_last_layer", "gen_logit_stats_last",
                           "gen_attn_stats_last", "gen_step_boundary_hidden"]:
                    results[k].append(feat[k])

            pbar.update(len(batch_samples))
            sample_idx = batch_end

        pbar.close()
        save_split_features(results, OUTPUT_DIR, ds_name, "all", MODEL_NAME)

        # Also save labels_multi to meta.json if present
        labels_multi = [s.get("label_multi") for s in samples]
        if any(m is not None for m in labels_multi):
            meta_path = os.path.join(OUTPUT_DIR, ds_name, "all", "meta.json")
            with open(meta_path) as f:
                meta = json.load(f)
            meta["labels_multi"] = labels_multi
            with open(meta_path, "w") as f:
                json.dump(meta, f, ensure_ascii=False)
            print(f"  Added labels_multi to meta.json")

    print("\nAll done!")


if __name__ == "__main__":
    main()
