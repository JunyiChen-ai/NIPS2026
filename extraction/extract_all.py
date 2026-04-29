"""
Unified feature extraction for ALL datasets × ANY model.

Replaces the two separate entry points (extract_features.py for Phase 1,
extract_features_new.py for Phase 2) with a single CLI-driven script.

Usage:
  python extract_all.py --model Qwen/Qwen2.5-7B-Instruct --output_dir features/qwen2.5-7b
  python extract_all.py --model meta-llama/Llama-3.1-8B-Instruct --output_dir features/llama3.1-8b
  python extract_all.py --model Qwen/Qwen2.5-7B-Instruct --datasets geometry_of_truth_cities,easy2hard_amc

The FeatureExtractor class (in extract_features.py) is model-agnostic.
This script only handles dataset loading and orchestration.
"""

import os
import sys
import json
import csv
import argparse

sys.path.insert(0, os.path.dirname(__file__))
from extract_features import (
    FeatureExtractor, save_split_features, is_split_done,
)

DATASETS_BASE = "/data/jehc223/NIPS2026/datasets"
PREPARED_DIR = "/data/jehc223/NIPS2026/datasets_prepared"

# ============================================================
# Phase 1 dataset loaders (from extract_features.py)
# ============================================================
def load_geometry_of_truth_cities():
    samples = []
    base = os.path.join(DATASETS_BASE, "knowledge_factual/geometry_of_truth/cities")
    for split in ["train", "val"]:
        path = os.path.join(base, f"{split}.csv")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            for row in csv.DictReader(f):
                label = int(row["label"])
                samples.append({"text": row["statement"], "label": label,
                                "gold_answer": "True" if label == 1 else "False",
                                "split": split})
    return samples


def load_easy2hard_amc():
    samples = []
    base = os.path.join(DATASETS_BASE, "reasoning_difficulty/easy2hard_bench/e2h_amc")
    for split in ["train", "eval"]:
        path = os.path.join(base, f"{split}.jsonl")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            for line in f:
                d = json.loads(line)
                text = d.get("problem", "")
                difficulty = d.get("rating", None)
                answer = d.get("answer", None)
                if text and difficulty is not None:
                    samples.append({"text": text, "label": float(difficulty),
                                    "gold_answer": str(answer) if answer is not None else None,
                                    "split": split})
    return samples


def load_metatool_task1():
    samples = []
    base = os.path.join(DATASETS_BASE, "tool_use_routing/metatool_task1")
    for split in ["train", "test"]:
        path = os.path.join(base, f"{split}.jsonl")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            for line in f:
                d = json.loads(line)
                text = d.get("query", "")
                label = 1 if d.get("label") == "positive" else 0
                if text:
                    samples.append({"text": text, "label": label,
                                    "gold_answer": "yes" if label == 1 else "no",
                                    "split": split})
    return samples


# ============================================================
# Phase 2 dataset loaders (from extract_features_new.py)
# Each loads from datasets_prepared/{name}/all.jsonl as a single "all" split.
# ============================================================

# Derived gold_answer for the 4 Phase 2 olds whose prepared JSONLs predate
# the gold_answer convention. Each dataset maps its int label → the natural-
# language string the model is asked to emit after "Final answer:". The
# 6 new QA datasets (gsm8k/math/...) carry gold_answer in the JSONL itself
# and are not in this map.
PHASE2_LABEL_TO_GOLD = {
    "common_claim_3class": {0: "True", 1: "False", 2: "Neither"},
    "when2call_3class":    {0: "A",    1: "B",     2: "C"},
    "fava":                {0: "no",   1: "yes"},
    "ragtruth":            {0: "no",   1: "yes"},
}


def load_prepared_dataset(dataset_name):
    """Load all.jsonl for a prepared dataset (Phase 2)."""
    path = os.path.join(PREPARED_DIR, dataset_name, "all.jsonl")
    label_to_gold = PHASE2_LABEL_TO_GOLD.get(dataset_name)
    samples = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            gold = d.get("gold_answer")
            if gold is None and label_to_gold is not None:
                gold = label_to_gold.get(int(d["label"]))
            samples.append({
                "text": d["text"],
                "label": d["label"],
                "label_multi": d.get("label_multi"),
                "gold_answer": gold,
                "split": "all",
            })
    return samples


# ============================================================
# Registry: dataset_name → loader function
# ============================================================
# Phase 1: pre-split datasets (loader returns samples with split field)
# NOTE: retrievalqa intentionally dropped from the new-setting re-extraction.
# Its legacy raw-mode features remain in B2 under features_legacy_512_no_chat.
PHASE1_DATASETS = {
    "geometry_of_truth_cities": load_geometry_of_truth_cities,
    "easy2hard_amc": load_easy2hard_amc,
    "metatool_task1": load_metatool_task1,
}

# Phase 2: single-pass datasets (all → later split by split_features.py)
PHASE2_DATASETS = [
    "common_claim_3class",
    "when2call_3class",
    "fava",
    "ragtruth",
    "gsm8k",
    "math",
    "commonsenseqa",
    "theoremqa",
    "mmlu",
    "belebele",
]

ALL_DATASET_NAMES = list(PHASE1_DATASETS.keys()) + PHASE2_DATASETS


# Per-dataset chat-template instructions.
# Every in-scope dataset has a non-empty entry here. extract_features.py:358-365
# applies tokenizer.apply_chat_template iff the instruction is truthy, so this
# dict is what guarantees uniform chat-template wrapping across all datasets.
# Run-time assert in main() fails fast if any selected dataset is missing.

# Original 6 new QA: numeric / multiple-choice answer.
NEW_QA_INSTRUCTION = (
    "Think step by step to answer the question. "
    "At the end, write a line 'Final answer: <value>' where <value> is the answer "
    "(for multiple-choice questions, write only the letter or number of the correct choice)."
)

DATASET_INSTRUCTIONS = {
    # 6 new QA datasets (numeric / multiple-choice)
    "gsm8k":         NEW_QA_INSTRUCTION,
    "math":          NEW_QA_INSTRUCTION,
    "theoremqa":     NEW_QA_INSTRUCTION,
    "commonsenseqa": NEW_QA_INSTRUCTION,
    "mmlu":          NEW_QA_INSTRUCTION,
    "belebele":      NEW_QA_INSTRUCTION,

    # easy2hard_amc: same shape as gsm8k/math (numeric/expression answer).
    "easy2hard_amc": NEW_QA_INSTRUCTION,

    # geometry_of_truth_cities: statement T/F.
    "geometry_of_truth_cities": (
        "Think step by step to decide whether the following statement is true or false. "
        "At the end, write a line 'Final answer: True' if the statement is true, "
        "'Final answer: False' otherwise."
    ),

    # common_claim_3class: claim T/F/Neither.
    "common_claim_3class": (
        "Think step by step to decide whether the following claim is true, false, or neither. "
        "At the end, write a line 'Final answer: True' if the claim is true, "
        "'Final answer: False' if it is false, "
        "'Final answer: Neither' if it is neither true nor false."
    ),

    # metatool_task1: tool-routing yes/no.
    "metatool_task1": (
        "Think step by step to decide whether answering the following user query "
        "requires calling an external tool. "
        "At the end, write a line 'Final answer: yes' if a tool is required, "
        "'Final answer: no' otherwise."
    ),

    # when2call_3class: 3-way multi-choice. The dataset's text already contains
    # the user query and the available tools list; this instruction adds the
    # 3 lettered choices and asks for a letter.
    "when2call_3class": (
        "Think step by step to decide which of the following actions best fits the user query. "
        "Choices:\n"
        "(A) Call one of the available tools.\n"
        "(B) Decline because the query cannot be answered with the available tools.\n"
        "(C) Ask the user for clarification or more information.\n"
        "At the end, write a line 'Final answer: <letter>' where <letter> is A, B, or C."
    ),

    # fava: hallucination judging (text already contains the per-sample task).
    "fava": (
        "Think step by step to decide whether the passage contains any hallucination. "
        "At the end, write 'Final answer: yes' if the passage contains hallucination, "
        "'Final answer: no' otherwise."
    ),

    # ragtruth: hallucination judging (text already contains the per-sample task).
    "ragtruth": (
        "Think step by step to decide whether the response contains any hallucination. "
        "At the end, write 'Final answer: yes' if the response contains hallucination, "
        "'Final answer: no' otherwise."
    ),
}


def run_extraction(extractor, samples, output_dir, dataset_name, split_name,
                   batch_size, model_name):
    """Extract features for one (dataset, split) group."""
    import torch
    from tqdm import tqdm

    if is_split_done(output_dir, dataset_name, split_name):
        print(f"\nSkipping {dataset_name}/{split_name} (already done)")
        return

    current_batch_size = batch_size
    print(f"\nProcessing {dataset_name}/{split_name}: {len(samples)} samples "
          f"(batch_size={current_batch_size})")

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
    pbar = tqdm(total=len(samples), desc=f"{dataset_name}/{split_name}")
    while sample_idx < len(samples):
        batch_end = min(sample_idx + current_batch_size, len(samples))
        batch_samples = samples[sample_idx:batch_end]
        batch_texts = [s["text"] for s in batch_samples]

        instruction = DATASET_INSTRUCTIONS.get(dataset_name)

        try:
            batch_features = extractor.extract_batch(batch_texts, instruction=instruction)
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

        # Early-validation dump: after the FIRST batch of each split, print
        # one sample (text + gold + gen + gen_len) so an external monitor can
        # spot-check format within minutes of a job starting, instead of
        # waiting hours for the whole split. Marker is grep-friendly.
        if sample_idx == 0 and len(batch_features) > 0:
            s0, f0 = batch_samples[0], batch_features[0]
            print(f"\n=== EARLY_SAMPLE_DUMP {dataset_name}/{split_name} ===", flush=True)
            print(f"  TEXT  : {repr(s0['text'][:180])}", flush=True)
            print(f"  LABEL : {s0['label']!r}", flush=True)
            print(f"  GOLD  : {s0.get('gold_answer')!r}", flush=True)
            print(f"  GENLEN: {f0['gen_len']}", flush=True)
            print(f"  GEN   : {repr(f0['gen_text'][:600])}", flush=True)
            print(f"  GENTAIL: {repr(f0['gen_text'][-250:])}", flush=True)
            print(f"=== END_EARLY_SAMPLE_DUMP ===", flush=True)

        pbar.update(len(batch_samples))
        sample_idx = batch_end

    pbar.close()
    save_split_features(results, output_dir, dataset_name, split_name, model_name)

    # For Phase 2 datasets: save labels_multi to meta.json if present
    labels_multi = [s.get("label_multi") for s in samples]
    if any(m is not None for m in labels_multi):
        meta_path = os.path.join(output_dir, dataset_name, split_name, "meta.json")
        with open(meta_path) as f:
            meta = json.load(f)
        meta["labels_multi"] = labels_multi
        with open(meta_path, "w") as f:
            json.dump(meta, f, ensure_ascii=False)
        print(f"  Added labels_multi to meta.json")

    # For QA-style Phase 2 datasets: save gold_answers to meta.json if present
    gold_answers = [s.get("gold_answer") for s in samples]
    if any(g is not None for g in gold_answers):
        meta_path = os.path.join(output_dir, dataset_name, split_name, "meta.json")
        with open(meta_path) as f:
            meta = json.load(f)
        meta["gold_answers"] = gold_answers
        with open(meta_path, "w") as f:
            json.dump(meta, f, ensure_ascii=False)
        print(f"  Added gold_answers to meta.json")


def main():
    parser = argparse.ArgumentParser(
        description="Unified feature extraction for all datasets × any model."
    )
    parser.add_argument("--model", type=str, required=True,
                        help="HuggingFace model name (e.g. Qwen/Qwen2.5-7B-Instruct)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for extracted features")
    parser.add_argument("--datasets", type=str, default="all",
                        help="Comma-separated dataset names, or 'all' (default: all)")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Initial batch size (auto-reduces on OOM)")
    parser.add_argument("--max_new_tokens", type=int, default=512,
                        help="Max new tokens for generation")
    args = parser.parse_args()

    # Parse dataset list
    if args.datasets == "all":
        datasets_to_run = ALL_DATASET_NAMES
    else:
        datasets_to_run = [d.strip() for d in args.datasets.split(",")]
        for d in datasets_to_run:
            if d not in ALL_DATASET_NAMES:
                parser.error(f"Unknown dataset: {d}. "
                             f"Available: {', '.join(ALL_DATASET_NAMES)}")

    # Chat-template invariant: every selected dataset MUST have an instruction
    # in DATASET_INSTRUCTIONS. extract_features.py only applies the chat
    # template when instruction is truthy, so a missing entry would silently
    # fall back to raw continuation mode.
    missing_instr = [d for d in datasets_to_run if not DATASET_INSTRUCTIONS.get(d)]
    if missing_instr:
        parser.error(f"Datasets missing from DATASET_INSTRUCTIONS: {missing_instr}. "
                     f"Add an entry or remove from --datasets.")

    # Override max_new_tokens in extract_features module
    import extract_features as ef
    ef.MAX_NEW_TOKENS = args.max_new_tokens

    print(f"Model: {args.model}")
    print(f"Output: {args.output_dir}")
    print(f"Datasets: {datasets_to_run}")

    extractor = FeatureExtractor(args.model)

    for ds_name in datasets_to_run:
        if ds_name in PHASE1_DATASETS:
            # Phase 1: pre-split datasets
            loader = PHASE1_DATASETS[ds_name]
            all_samples = loader()
            print(f"\nLoaded {ds_name}: {len(all_samples)} samples")

            # Group by split
            groups = {}
            for s in all_samples:
                groups.setdefault(s["split"], []).append(s)

            for split_name, split_samples in groups.items():
                run_extraction(extractor, split_samples, args.output_dir,
                               ds_name, split_name, args.batch_size, args.model)
        else:
            # Phase 2: single "all" pass
            all_samples = load_prepared_dataset(ds_name)
            print(f"\nLoaded {ds_name}: {len(all_samples)} samples")
            run_extraction(extractor, all_samples, args.output_dir,
                           ds_name, "all", args.batch_size, args.model)

    print("\nAll done!")


if __name__ == "__main__":
    main()
