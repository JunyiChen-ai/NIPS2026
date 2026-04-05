"""
Set up E2H AMC 3-class and 5-class feature directories.

These datasets use the same math problem text as the original easy2hard_amc
regression dataset. Since the model sees identical text, internal states are
identical — only the labels differ. So we:
  1. Symlink all .pt and .json feature files from easy2hard_amc/
  2. Create new meta.json with class_label instead of rating

Original splits:  train (1000), eval (2975), train_sub (800), val_split (200)
New label source: e2h_amc_3class/ and e2h_amc_5class/ JSONL files
"""

import os
import json
import numpy as np
from sklearn.model_selection import train_test_split

FEATURES_DIR = "/data/jehc223/NIPS2026/extraction/features"
DATASETS_DIR = "/data/jehc223/NIPS2026/datasets/reasoning_difficulty"
SEED = 42


def load_class_labels(variant):
    """Load class_label from e2h_amc_{variant} JSONL, keyed by problem text prefix."""
    labels = {}
    base = os.path.join(DATASETS_DIR, f"e2h_amc_{variant}")
    for split in ["train", "eval"]:
        path = os.path.join(base, f"{split}.jsonl")
        with open(path) as f:
            for line in f:
                d = json.loads(line)
                # Use problem text as key (guaranteed unique within each split)
                labels[d["problem"]] = int(d["class_label"])
    return labels


def setup_variant(variant):
    """Create e2h_amc_{variant}/ with symlinks + new meta.json for each split."""
    print(f"\nSetting up e2h_amc_{variant}")
    class_labels = load_class_labels(variant)
    src_base = os.path.join(FEATURES_DIR, "easy2hard_amc")
    dst_base = os.path.join(FEATURES_DIR, f"e2h_amc_{variant}")

    # Process train and eval splits (which have extracted features)
    for split in ["train", "eval"]:
        src_dir = os.path.join(src_base, split)
        dst_dir = os.path.join(dst_base, split)
        if not os.path.exists(src_dir):
            print(f"  Skipping {split} (source not found)")
            continue

        os.makedirs(dst_dir, exist_ok=True)

        # Symlink all .pt and .json files (except meta.json)
        for fname in os.listdir(src_dir):
            if fname == "meta.json":
                continue
            src_path = os.path.join(src_dir, fname)
            dst_path = os.path.join(dst_dir, fname)
            if os.path.exists(dst_path):
                os.remove(dst_path)
            os.symlink(src_path, dst_path)

        # Create new meta.json with class labels
        with open(os.path.join(src_dir, "meta.json")) as f:
            meta = json.load(f)

        # Map each sample's text to its class_label
        new_labels = []
        for text in meta["texts"]:
            if text not in class_labels:
                raise ValueError(f"Text not found in {variant} labels: {text[:80]}...")
            new_labels.append(class_labels[text])

        meta["labels"] = new_labels
        meta["dataset"] = f"e2h_amc_{variant}"
        with open(os.path.join(dst_dir, "meta.json"), "w") as f:
            json.dump(meta, f, ensure_ascii=False)

        from collections import Counter
        print(f"  {split}: {len(new_labels)} samples, labels: {dict(Counter(new_labels))}")

    # Create train_sub and val_split (80/20 stratified from train)
    train_dir = os.path.join(dst_base, "train")
    if os.path.exists(train_dir):
        with open(os.path.join(train_dir, "meta.json")) as f:
            train_meta = json.load(f)

        labels = np.array(train_meta["labels"])
        n = len(labels)
        idx_train, idx_val = train_test_split(
            np.arange(n), test_size=0.2, stratify=labels, random_state=SEED
        )

        # Reuse the split logic from create_val_split.py
        import torch
        src_train_dir = os.path.join(src_base, "train")

        for split_name, indices in [("train_sub", idx_train), ("val_split", idx_val)]:
            dst_split_dir = os.path.join(dst_base, split_name)
            os.makedirs(dst_split_dir, exist_ok=True)
            idx = sorted(indices)

            # For .pt files: we need to slice from the source train .pt files
            # But source train_sub/val_split already exist with the SAME indices
            # (same SEED=42 and same stratify approach)... actually NO.
            # The original create_val_split.py stratified on regression labels (rating),
            # but we stratify on class_label. Different stratification → different indices.
            # So we must slice from the full train .pt files.

            for fname in os.listdir(src_train_dir):
                if not fname.endswith(".pt"):
                    continue
                src_tensor = torch.load(os.path.join(src_train_dir, fname), map_location="cpu")
                if isinstance(src_tensor, torch.Tensor) and src_tensor.shape[0] == n:
                    torch.save(src_tensor[idx], os.path.join(dst_split_dir, fname))
                elif isinstance(src_tensor, list) and len(src_tensor) == n:
                    torch.save([src_tensor[i] for i in idx], os.path.join(dst_split_dir, fname))
                else:
                    # Copy as-is (shouldn't happen but safe fallback)
                    torch.save(src_tensor, os.path.join(dst_split_dir, fname))

            # JSON feature files
            for fname in ["input_logit_stats.json", "gen_logit_stats_last.json"]:
                src_path = os.path.join(src_train_dir, fname)
                if os.path.exists(src_path):
                    with open(src_path) as f:
                        data = json.load(f)
                    sliced = [data[i] for i in idx]
                    with open(os.path.join(dst_split_dir, fname), "w") as f:
                        json.dump(sliced, f)

            # Meta
            new_meta = {
                "model": train_meta["model"],
                "dataset": f"e2h_amc_{variant}",
                "split": split_name,
                "n_samples": len(idx),
                "labels": [train_meta["labels"][i] for i in idx],
                "texts": [train_meta["texts"][i] for i in idx],
                "gen_texts": [train_meta["gen_texts"][i] for i in idx],
                "input_seq_lens": [train_meta["input_seq_lens"][i] for i in idx],
                "gen_lens": [train_meta["gen_lens"][i] for i in idx],
                "gen_step_boundary_indices": [train_meta["gen_step_boundary_indices"][i] for i in idx],
            }
            with open(os.path.join(dst_split_dir, "meta.json"), "w") as f:
                json.dump(new_meta, f, ensure_ascii=False)

            from collections import Counter
            print(f"  {split_name}: {len(idx)} samples, labels: {dict(Counter(new_meta['labels']))}")


def main():
    setup_variant("3class")
    setup_variant("5class")
    print("\nDone!")


if __name__ == "__main__":
    main()
