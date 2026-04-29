#!/usr/bin/env python
"""Validate one (model_tag, dataset) extraction output across ALL split subdirs.

Usage: python validate_extraction.py <model_tag> <dataset>
Exits 0 with "OK ..." on success, 1 with "FAIL: ..." on any problem.

Walks extraction/features/<tag>/<dataset>/*/ and validates every subdir that
contains a meta.json. Phase 2 datasets typically have a single "all" split
(later sliced by split_features.py); Phase 1 datasets have multiple splits
(train/val/eval/test/...).
"""
import sys
import os
import json

REPO = "/data/jehc223/NIPS2026"

REQUIRED_TENSOR_FILES = [
    "input_last_token_hidden.pt",
    "input_mean_pool_hidden.pt",
    "input_per_head_activation.pt",
    "input_attn_stats.pt",
    "input_attn_value_norms.pt",
    "input_logit_stats.json",
    "gen_last_token_hidden.pt",
    "gen_mean_pool_hidden.pt",
    "gen_per_token_hidden_last_layer.pt",
    "gen_logit_stats_last.json",
    "gen_attn_stats_last.pt",
    "gen_step_boundary_hidden.pt",
]


def validate_split(split_dir, n_expected):
    """Returns (problems_list, n_actual, empty_gen, gold_preview)."""
    problems = []
    meta_path = os.path.join(split_dir, "meta.json")

    if not os.path.exists(meta_path):
        return [f"meta.json missing at {meta_path}"], 0, 0, ""

    try:
        with open(meta_path) as f:
            meta = json.load(f)
    except Exception as e:
        return [f"meta.json parse error: {e}"], 0, 0, ""

    n_actual = len(meta.get("gen_texts", []))
    if n_expected > 0 and n_actual != n_expected:
        problems.append(f"gen_texts={n_actual} != expected {n_expected}")

    for k in ("texts", "labels", "gen_texts", "gold_answers",
              "input_seq_lens", "gen_lens"):
        v = meta.get(k)
        if v is None:
            problems.append(f"meta.{k} missing")
        elif isinstance(v, list) and len(v) != n_actual:
            problems.append(f"meta.{k} len={len(v)} != n={n_actual}")

    gen_texts = meta.get("gen_texts", [])
    empty_gen = sum(1 for t in gen_texts if not (t or "").strip())
    if empty_gen > max(3, int(0.05 * n_actual)):
        problems.append(f"empty_gen_texts={empty_gen}/{n_actual} (>5%)")

    gold_answers = meta.get("gold_answers", [])
    empty_gold = sum(1 for t in gold_answers
                     if not (str(t) if t is not None else "").strip())
    if gold_answers and empty_gold > 0:
        problems.append(f"empty_gold_answers={empty_gold}/{len(gold_answers)}")

    missing_files = [f for f in REQUIRED_TENSOR_FILES
                     if not os.path.exists(os.path.join(split_dir, f))]
    if missing_files:
        problems.append(f"missing_tensor_files={missing_files}")

    gold_preview = f" gold[0..2]={gold_answers[:3]}" if gold_answers else ""
    return problems, n_actual, empty_gen, gold_preview


def main():
    if len(sys.argv) != 3:
        print("FAIL: usage validate_extraction.py <tag> <dataset>")
        return 1
    tag, ds = sys.argv[1], sys.argv[2]
    ds_root = os.path.join(REPO, "extraction/features", tag, ds)

    if not os.path.isdir(ds_root):
        print(f"FAIL: dataset dir missing at {ds_root}")
        return 1

    # Find every split subdir that has a meta.json. Phase 1 datasets have
    # multiple (train/val/eval/test/...); Phase 2 have a single "all".
    splits = sorted(
        d for d in os.listdir(ds_root)
        if os.path.isdir(os.path.join(ds_root, d))
        and os.path.exists(os.path.join(ds_root, d, "meta.json"))
    )

    if not splits:
        print(f"FAIL: no split dirs with meta.json under {ds_root}")
        return 1

    # Only Phase 2 datasets have datasets_prepared/<ds>/all.jsonl. Use it as
    # the row-count check for the "all" split when present; skip otherwise.
    prepared = os.path.join(REPO, "datasets_prepared", ds, "all.jsonl")
    n_prepared = -1
    if os.path.exists(prepared):
        with open(prepared) as f:
            n_prepared = sum(1 for _ in f)

    all_problems = []
    summary_parts = []
    for split in splits:
        split_dir = os.path.join(ds_root, split)
        n_expected = n_prepared if (split == "all") else -1
        problems, n_actual, empty_gen, gold_preview = validate_split(
            split_dir, n_expected)
        if problems:
            all_problems.append(f"[{split}] " + " | ".join(problems))
        summary_parts.append(f"{split}=n{n_actual},empty{empty_gen}{gold_preview}")

    if all_problems:
        print("FAIL: " + " ;; ".join(all_problems))
        return 1

    print(f"OK splits={len(splits)} {' '.join(summary_parts)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
