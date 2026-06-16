"""
Generate stratified 60/20/20 splits per (model, dataset) using correctness labels.

Output: reproduce/correctness_labels/{model}/{dataset}/split_indices.json
  Format: {"train": [...], "val": [...], "test": [...]}

Stratify on the binary correctness label so train/val/test have ~equal pos rate.
seed=42 fixed for reproducibility.
"""
import json
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split

LABELS_ROOT = Path("/home/junyi/NIPS2026/reproduce/correctness_labels")
MODELS = ["qwen2.5-7b", "llama3.1-8b", "mistral-7b-v0.3"]
DATASETS = ["gsm8k", "math", "mmlu", "commonsenseqa", "belebele", "theoremqa"]
SEED = 42


def make_split(labels: list[int]) -> dict:
    """Stratified 60/20/20 split. Returns {'train', 'val', 'test'}: list[int] indices."""
    n = len(labels)
    idx = np.arange(n)
    y = np.array(labels)

    # Edge: if any class has too few samples for stratify, fall back to non-strat
    n_pos = int(y.sum()); n_neg = n - n_pos
    use_strat = n_pos >= 5 and n_neg >= 5
    strat = y if use_strat else None

    # First split off 20% test
    rest_idx, test_idx = train_test_split(
        idx, test_size=0.20, random_state=SEED, stratify=strat
    )
    # From remaining 80%, take 25% as val (= 20% of full) and 75% as train (= 60% of full)
    rest_y = y[rest_idx] if use_strat else None
    train_idx, val_idx = train_test_split(
        rest_idx, test_size=0.25, random_state=SEED, stratify=rest_y
    )

    return {
        "train": sorted(train_idx.tolist()),
        "val": sorted(val_idx.tolist()),
        "test": sorted(test_idx.tolist()),
    }


def main():
    rows = []
    for m in MODELS:
        for d in DATASETS:
            labels_path = LABELS_ROOT / m / d / "labels.json"
            if not labels_path.exists():
                print(f"  skip {m}/{d}: labels.json missing")
                continue
            with open(labels_path) as f:
                lab_data = json.load(f)
            labels = lab_data["labels"]

            split = make_split(labels)

            # Verify
            n_total = len(labels)
            assert len(set(split["train"]) | set(split["val"]) | set(split["test"])) == n_total
            assert not (set(split["train"]) & set(split["val"]))
            assert not (set(split["train"]) & set(split["test"]))
            assert not (set(split["val"]) & set(split["test"]))

            y = np.array(labels)
            tr_pos = y[split["train"]].mean()
            va_pos = y[split["val"]].mean()
            te_pos = y[split["test"]].mean()

            out_path = LABELS_ROOT / m / d / "split_indices.json"
            with open(out_path, "w") as f:
                json.dump(split, f)

            rows.append((m, d, n_total,
                         len(split["train"]), len(split["val"]), len(split["test"]),
                         tr_pos, va_pos, te_pos))
            print(f"  {m:18s} {d:14s} N={n_total:5d}  "
                  f"tr={len(split['train']):5d}({tr_pos:.3f})  "
                  f"va={len(split['val']):5d}({va_pos:.3f})  "
                  f"te={len(split['test']):5d}({te_pos:.3f})")

    # Verify pos-rate consistency: max-min across splits should be < 0.02 (2pp)
    print("\n=== Verification: max/min pos rate diff across splits ===")
    bad = 0
    for r in rows:
        m, d, _, _, _, _, tr, va, te = r
        diff = max(tr, va, te) - min(tr, va, te)
        flag = "OK" if diff < 0.02 else "WARN"
        print(f"  {flag}  {m}/{d}: pos-rate spread = {diff:.4f}")
        if diff >= 0.02: bad += 1
    print(f"\n{bad}/{len(rows)} have spread >= 2pp (expected 0 for stratified)")


if __name__ == "__main__":
    main()
