"""
Split extracted features from 'all' into train/val/test using pre-computed indices.

Reads:
  - features/{dataset}/all/   (extracted features for all samples)
  - datasets_prepared/{dataset}/split_indices.json  (train/val/test indices)

Creates:
  - features/{dataset}/train/
  - features/{dataset}/val/
  - features/{dataset}/test/

Each split directory contains the same feature files as 'all', but sliced to
only include the samples in that split.
"""

import os
import json
import torch
import numpy as np

FEATURES_DIR = "/data/jehc223/NIPS2026/extraction/features"
PREPARED_DIR = "/data/jehc223/NIPS2026/datasets_prepared"

DATASETS = [
    "common_claim_3class",
    "when2call_3class",
    "fava",
    "ragtruth",
]

# Tensor fields that are stacked (N, ...) and can be sliced by index
TENSOR_FIELDS = [
    "input_last_token_hidden",
    "input_mean_pool_hidden",
    "input_per_head_activation",
    "input_attn_stats",
    "input_attn_value_norms",
    "gen_last_token_hidden",
    "gen_mean_pool_hidden",
    "gen_per_token_hidden_last_layer",
    "gen_attn_stats_last",
]

# JSON list fields
JSON_FIELDS = [
    "input_logit_stats",
    "gen_logit_stats_last",
]


def slice_and_save(dataset, split_name, indices):
    """Slice features from 'all' directory by indices and save to split directory."""
    all_dir = os.path.join(FEATURES_DIR, dataset, "all")
    split_dir = os.path.join(FEATURES_DIR, dataset, split_name)
    os.makedirs(split_dir, exist_ok=True)
    idx = np.array(sorted(indices))

    # Tensor fields
    for fname in TENSOR_FIELDS:
        src_path = os.path.join(all_dir, f"{fname}.pt")
        if not os.path.exists(src_path):
            continue
        data = torch.load(src_path, map_location="cpu")
        if isinstance(data, torch.Tensor):
            torch.save(data[idx], os.path.join(split_dir, f"{fname}.pt"))
        elif isinstance(data, list):
            torch.save([data[i] for i in idx], os.path.join(split_dir, f"{fname}.pt"))

    # gen_step_boundary_hidden: always a list of variable-size tensors (per save_split_features)
    sbh_path = os.path.join(all_dir, "gen_step_boundary_hidden.pt")
    if os.path.exists(sbh_path):
        sbh = torch.load(sbh_path, map_location="cpu")
        torch.save([sbh[i] for i in idx], os.path.join(split_dir, "gen_step_boundary_hidden.pt"))

    # JSON fields
    for fname in JSON_FIELDS:
        src_path = os.path.join(all_dir, f"{fname}.json")
        if not os.path.exists(src_path):
            continue
        with open(src_path) as f:
            data = json.load(f)
        sliced = [data[i] for i in idx]
        with open(os.path.join(split_dir, f"{fname}.json"), "w") as f:
            json.dump(sliced, f)

    # Meta
    with open(os.path.join(all_dir, "meta.json")) as f:
        meta = json.load(f)

    n_all = meta["n_samples"]
    new_meta = {
        "model": meta["model"],
        "dataset": meta["dataset"],
        "split": split_name,
        "n_samples": len(idx),
    }
    for key in ["labels", "texts", "gen_texts", "input_seq_lens", "gen_lens",
                "gen_step_boundary_indices"]:
        if key in meta:
            new_meta[key] = [meta[key][i] for i in idx]

    # labels_multi if present
    if "labels_multi" in meta:
        new_meta["labels_multi"] = [meta["labels_multi"][i] for i in idx]

    with open(os.path.join(split_dir, "meta.json"), "w") as f:
        json.dump(new_meta, f, ensure_ascii=False)

    print(f"  {split_name}: {len(idx)} samples")


def main():
    for dataset in DATASETS:
        print(f"\nSplitting {dataset}")

        # Load split indices
        indices_path = os.path.join(PREPARED_DIR, dataset, "split_indices.json")
        with open(indices_path) as f:
            splits = json.load(f)

        # Verify all dir exists
        all_dir = os.path.join(FEATURES_DIR, dataset, "all")
        if not os.path.exists(os.path.join(all_dir, "meta.json")):
            print(f"  Skipping (features/all not extracted yet)")
            continue

        for split_name, indices in splits.items():
            slice_and_save(dataset, split_name, indices)

    print("\nDone!")


if __name__ == "__main__":
    main()
