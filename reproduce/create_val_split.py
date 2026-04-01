"""
Create validation splits from existing train splits.
For each dataset, split train into train_sub + val (80/20, stratified for classification).
Save val features to disk alongside existing train/test features.
"""

import os
import json
import torch
import numpy as np
from sklearn.model_selection import train_test_split

FEATURES_DIR = "/data/jehc223/NIPS2026/extraction/features"
VAL_RATIO = 0.2
SEED = 42

DATASETS = {
    "geometry_of_truth_cities": ("train", False),
    "easy2hard_amc": ("train", True),  # regression
    "metatool_task1": ("train", False),
    "retrievalqa": ("train", False),
}


def load_split(dataset, split):
    split_dir = os.path.join(FEATURES_DIR, dataset, split)
    data = {}
    for name in ["input_last_token_hidden", "input_mean_pool_hidden",
                 "input_per_head_activation", "input_attn_stats",
                 "input_attn_value_norms",
                 "gen_last_token_hidden", "gen_mean_pool_hidden",
                 "gen_per_token_hidden_last_layer", "gen_attn_stats_last",
                 "gen_step_boundary_hidden"]:
        path = os.path.join(split_dir, f"{name}.pt")
        if os.path.exists(path):
            data[name] = torch.load(path, map_location="cpu")
    for name in ["input_logit_stats", "gen_logit_stats_last"]:
        path = os.path.join(split_dir, f"{name}.json")
        if os.path.exists(path):
            with open(path) as f:
                data[name] = json.load(f)
    with open(os.path.join(split_dir, "meta.json")) as f:
        data["meta"] = json.load(f)
    return data


def save_split(data, indices, dataset, split_name):
    split_dir = os.path.join(FEATURES_DIR, dataset, split_name)
    os.makedirs(split_dir, exist_ok=True)
    idx = np.array(indices)

    # Tensor fields: slice by indices
    for name in ["input_last_token_hidden", "input_mean_pool_hidden",
                 "input_per_head_activation", "input_attn_stats",
                 "input_attn_value_norms",
                 "gen_last_token_hidden", "gen_mean_pool_hidden",
                 "gen_per_token_hidden_last_layer", "gen_attn_stats_last"]:
        if name in data:
            torch.save(data[name][idx], os.path.join(split_dir, f"{name}.pt"))

    # gen_step_boundary_hidden: list of variable-size tensors
    if "gen_step_boundary_hidden" in data:
        orig = data["gen_step_boundary_hidden"]
        if isinstance(orig, list):
            torch.save([orig[i] for i in idx], os.path.join(split_dir, "gen_step_boundary_hidden.pt"))
        else:
            torch.save(orig, os.path.join(split_dir, "gen_step_boundary_hidden.pt"))

    # JSON fields: slice by indices
    for name in ["input_logit_stats", "gen_logit_stats_last"]:
        if name in data:
            sliced = [data[name][i] for i in idx]
            with open(os.path.join(split_dir, f"{name}.json"), "w") as f:
                json.dump(sliced, f)

    # Meta: slice all list fields
    meta = data["meta"]
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
    with open(os.path.join(split_dir, "meta.json"), "w") as f:
        json.dump(new_meta, f, ensure_ascii=False)

    print(f"  Saved {split_dir}: {len(idx)} samples")


def main():
    for dataset, (train_split, is_regression) in DATASETS.items():
        print(f"\n{dataset}:")
        data = load_split(dataset, train_split)
        labels = np.array(data["meta"]["labels"])
        n = len(labels)

        if is_regression:
            # Regression: random split (no stratification)
            train_idx, val_idx = train_test_split(
                np.arange(n), test_size=VAL_RATIO, random_state=SEED)
        else:
            # Classification: stratified split
            train_idx, val_idx = train_test_split(
                np.arange(n), test_size=VAL_RATIO, random_state=SEED,
                stratify=labels)

        print(f"  Original train: {n}")
        print(f"  New train_sub: {len(train_idx)}, val: {len(val_idx)}")

        save_split(data, train_idx, dataset, "train_sub")
        save_split(data, val_idx, dataset, "val_split")

    print("\nDone!")


if __name__ == "__main__":
    main()
