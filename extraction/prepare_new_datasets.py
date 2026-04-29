"""
Prepare new datasets for feature extraction.
For each dataset, outputs:
  - datasets_prepared/{name}/all.jsonl — all samples in one file
  - datasets_prepared/{name}/split_indices.json — {"train": [...], "val": [...], "test": [...]}

Extraction runs on all.jsonl once. Splits are applied afterward via index slicing.

Each JSONL line: {"text": "...", "label": int, "label_multi": [int,...] (optional)}
"""

import os
import re
import json
import csv
import random
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split

SEED = 42
BASE = "/data/jehc223/NIPS2026"
OUT = os.path.join(BASE, "datasets_prepared")

random.seed(SEED)
np.random.seed(SEED)


# ============================================================
# FAVA helpers
# ============================================================
FAVA_TYPO_MAP = {
    "entity": "entity", "relation": "relation", "relational_error": "relation",
    "relative": "relation", "contradictory": "contradictory",
    "contradiction": "contradictory", "contradictary": "contradictory",
    "contraditory": "contradictory", "contradicatory": "contradictory",
    "contradiciary": "contradictory", "contrdictory": "contradictory",
    "contrast": "contradictory", "unverifiable": "unverifiable",
    "unvalidatable": "unverifiable", "invented": "invented",
    "inverted": "invented", "subjective": "subjective",
    "subective": "subjective", "subj": "subjective",
}
FAVA_TYPES = ["entity", "relation", "contradictory", "unverifiable", "invented", "subjective"]
FAVA_STRUCTURAL_TAGS = {
    "mark", "delete", "b", "ref", "insert", "span", "entire", "Delete",
    "input", "nowiki", "br", "r", "s", "a", "add", "marker", "strong",
    "deletion",
}


def clean_fava_completion(text):
    """Reconstruct the original hallucinated passage (what was actually generated).
    <entity><mark>Romania</mark><delete>Brazil</delete></entity> → Brazil
    (keep <delete> content = original hallucinated text, remove <mark> = corrections)
    <unverifiable>...</unverifiable> → ... (text kept, tag removed)
    """
    text = re.sub(r"<mark>.*?</mark>", "", text, flags=re.DOTALL)
    text = re.sub(r"<delete>(.*?)</delete>", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"</?[a-zA-Z_][^>]*>", "", text)
    text = re.sub(r"  +", " ", text).strip()
    return text


def extract_fava_labels(completion):
    """Extract 6-dim multi-label from FAVA completion tags (with typo correction)."""
    found = set()
    for m in re.finditer(r"<(\w+)>", completion):
        tag = m.group(1)
        if tag in FAVA_STRUCTURAL_TAGS:
            continue
        canonical = FAVA_TYPO_MAP.get(tag)
        if canonical:
            found.add(canonical)
    return [1 if t in found else 0 for t in FAVA_TYPES]


# ============================================================
# Save utility: all.jsonl + split_indices.json
# ============================================================
def save_dataset(name, samples, idx_train, idx_val, idx_test):
    """Save all samples to all.jsonl and split indices to split_indices.json."""
    out_dir = os.path.join(OUT, name)
    os.makedirs(out_dir, exist_ok=True)

    # Save all.jsonl
    path = os.path.join(out_dir, "all.jsonl")
    with open(path, "w") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    # Save split indices (as Python lists of ints)
    splits = {
        "train": sorted(int(i) for i in idx_train),
        "val": sorted(int(i) for i in idx_val),
        "test": sorted(int(i) for i in idx_test),
    }
    with open(os.path.join(out_dir, "split_indices.json"), "w") as f:
        json.dump(splits, f)

    for split_name, indices in splits.items():
        labels = Counter(samples[i]["label"] for i in indices)
        print(f"  {split_name}: {len(indices)} samples, labels: {dict(labels)}")


# ============================================================
# Dataset preparation functions
# ============================================================
def prepare_common_claim():
    """common_claim 3-class: True/False/Neither, subsample to 5K."""
    name = "common_claim_3class"
    print(f"\n{'='*60}\nPreparing {name}")
    src = os.path.join(BASE, "baseline/geometry-of-truth/datasets/common_claim.csv")

    label_map = {"True": 0, "False": 1, "Neither": 2}
    samples = []
    with open(src) as f:
        for row in csv.DictReader(f):
            text = row["examples"]
            label = label_map[row["label"]]
            samples.append({"text": text, "label": label})

    print(f"  Raw: {len(samples)} samples, labels: {Counter(s['label'] for s in samples)}")

    # Subsample to 5000, stratified
    labels = [s["label"] for s in samples]
    indices = np.arange(len(samples))
    sub_idx, _ = train_test_split(indices, train_size=5000, stratify=labels, random_state=SEED)
    samples = [samples[i] for i in sorted(sub_idx)]
    print(f"  Subsampled: {len(samples)}, labels: {Counter(s['label'] for s in samples)}")

    # 70/10/20 stratified split
    labels = [s["label"] for s in samples]
    idx_all = np.arange(len(samples))
    idx_trainval, idx_test = train_test_split(idx_all, test_size=0.2, stratify=labels, random_state=SEED)
    labels_trainval = [labels[i] for i in idx_trainval]
    idx_train, idx_val = train_test_split(idx_trainval, test_size=0.125, stratify=labels_trainval, random_state=SEED)

    save_dataset(name, samples, idx_train, idx_val, idx_test)


def prepare_when2call():
    """When2Call 3-class: tool_call/cannot_answer/request_for_info."""
    name = "when2call_3class"
    print(f"\n{'='*60}\nPreparing {name}")
    src = os.path.join(BASE, "datasets/tool_use_routing/when2call/mcq.jsonl")

    label_map = {"tool_call": 0, "cannot_answer": 1, "request_for_info": 2}
    samples = []
    with open(src) as f:
        for line in f:
            d = json.loads(line)
            question = d["question"]
            tools = d["tools"]
            tool_lines = []
            for t_str in tools:
                t = json.loads(t_str) if isinstance(t_str, str) else t_str
                tool_lines.append(f"- {t['name']}: {t['description']}")
            if tool_lines:
                text = f"Query: {question}\nAvailable tools:\n" + "\n".join(tool_lines)
            else:
                text = f"Query: {question}"
            label = label_map[d["correct_answer"]]
            samples.append({"text": text, "label": label})

    print(f"  Raw: {len(samples)} samples, labels: {Counter(s['label'] for s in samples)}")

    # Use all (≤5K). 70/10/20 stratified split
    labels = [s["label"] for s in samples]
    idx_all = np.arange(len(samples))
    idx_trainval, idx_test = train_test_split(idx_all, test_size=0.2, stratify=labels, random_state=SEED)
    labels_trainval = [labels[i] for i in idx_trainval]
    idx_train, idx_val = train_test_split(idx_trainval, test_size=0.125, stratify=labels_trainval, random_state=SEED)

    save_dataset(name, samples, idx_train, idx_val, idx_test)


def prepare_fava():
    """FAVA: binary + 6-label multi-label hallucination detection.
    Subsample to 5K: keep all 635 clean + sample 4365 hallucinated.
    """
    name = "fava"
    print(f"\n{'='*60}\nPreparing {name}")
    src = os.path.join(BASE, "datasets/hallucination/fava/train.jsonl")

    all_samples = []
    with open(src) as f:
        for line in f:
            d = json.loads(line)
            cleaned = clean_fava_completion(d["completion"])
            multi_label = extract_fava_labels(d["completion"])
            binary_label = 1 if any(multi_label) else 0

            text = (
                "Given the following reference text and a generated passage, "
                "determine whether the passage contains any hallucination "
                "(fabricated, incorrect, or unverifiable information).\n\n"
                f"Reference: {d['prompt']}\n\n"
                f"Passage: {cleaned}"
            )
            all_samples.append({
                "text": text,
                "label": binary_label,
                "label_multi": multi_label,
            })

    labels = [s["label"] for s in all_samples]
    print(f"  Raw: {len(all_samples)} samples, binary: {Counter(labels)}")

    # Subsample: keep ALL clean (label=0), sample 4365 hallucinated (label=1)
    clean_idx = [i for i, s in enumerate(all_samples) if s["label"] == 0]
    halluc_idx = [i for i, s in enumerate(all_samples) if s["label"] == 1]
    n_halluc_sample = 5000 - len(clean_idx)
    halluc_sampled = sorted(random.sample(halluc_idx, n_halluc_sample))
    sub_idx = sorted(clean_idx + halluc_sampled)
    samples = [all_samples[i] for i in sub_idx]
    print(f"  Subsampled: {len(samples)}, binary: {Counter(s['label'] for s in samples)}")

    # 70/10/20 stratified on binary label
    labels = [s["label"] for s in samples]
    idx_all = np.arange(len(samples))
    idx_trainval, idx_test = train_test_split(idx_all, test_size=0.2, stratify=labels, random_state=SEED)
    labels_trainval = [labels[i] for i in idx_trainval]
    idx_train, idx_val = train_test_split(idx_trainval, test_size=0.125, stratify=labels_trainval, random_state=SEED)

    save_dataset(name, samples, idx_train, idx_val, idx_test)


def prepare_ragtruth():
    """RAGTruth: binary + 2-label multi-label hallucination detection.
    Subsample to ~5K: 4000 from train (→ train+val), 1000 from test.
    """
    name = "ragtruth"
    print(f"\n{'='*60}\nPreparing {name}")

    def load_ragtruth_file(path):
        samples = []
        with open(path) as f:
            for line in f:
                d = json.loads(line)
                ec = d["hallucination_labels_processed"].get("evident_conflict", 0)
                bi = d["hallucination_labels_processed"].get("baseless_info", 0)
                binary_label = 1 if (ec + bi) > 0 else 0
                multi_label = [1 if ec > 0 else 0, 1 if bi > 0 else 0]

                text = (
                    "Given the following task, source material, and a generated response, "
                    "determine whether the response contains any hallucination.\n\n"
                    f"Task: {d['query']}\n\n"
                    f"Source material: {d['context']}\n\n"
                    f"Generated response: {d['output']}"
                )
                samples.append({
                    "text": text,
                    "label": binary_label,
                    "label_multi": multi_label,
                })
        return samples

    train_all = load_ragtruth_file(os.path.join(BASE, "datasets/hallucination/ragtruth/train.jsonl"))
    test_all = load_ragtruth_file(os.path.join(BASE, "datasets/hallucination/ragtruth/test.jsonl"))
    print(f"  Raw train: {len(train_all)}, binary: {Counter(s['label'] for s in train_all)}")
    print(f"  Raw test: {len(test_all)}, binary: {Counter(s['label'] for s in test_all)}")

    # Subsample train to 4000, stratified on binary label
    train_labels = [s["label"] for s in train_all]
    train_idx = np.arange(len(train_all))
    sub_train_idx, _ = train_test_split(train_idx, train_size=4000, stratify=train_labels, random_state=SEED)
    train_sub = [train_all[i] for i in sorted(sub_train_idx)]

    # Subsample test to 1000, stratified on binary label
    test_labels = [s["label"] for s in test_all]
    test_idx = np.arange(len(test_all))
    sub_test_idx, _ = train_test_split(test_idx, train_size=1000, stratify=test_labels, random_state=SEED)
    test_sub = [test_all[i] for i in sorted(sub_test_idx)]

    # Combine into a single sample list: [train_sub samples..., test_sub samples...]
    # Index into this combined list for splits
    all_samples = train_sub + test_sub  # 4000 + 1000 = 5000

    # Split the train portion (first 4000) into train (3200) + val (800)
    train_portion_labels = [s["label"] for s in train_sub]
    idx_trainval = np.arange(len(train_sub))
    idx_train, idx_val = train_test_split(idx_trainval, test_size=0.2, stratify=train_portion_labels, random_state=SEED)

    # Test indices point to the second portion
    idx_test = np.arange(len(train_sub), len(all_samples))

    save_dataset(name, all_samples, idx_train, idx_val, idx_test)


# ============================================================
# New QA datasets: {gsm8k, math, commonsenseqa, theoremqa, mmlu, belebele}
# Each pulls from the test set of its source, stores gold_answer, 70/10/20 split.
# ============================================================
def _random_split_70_10_20(n):
    """Return (idx_train, idx_val, idx_test) as np arrays — no stratification."""
    idx_all = np.arange(n)
    idx_trainval, idx_test = train_test_split(idx_all, test_size=0.2, random_state=SEED)
    idx_train, idx_val = train_test_split(idx_trainval, test_size=0.125, random_state=SEED)
    return idx_train, idx_val, idx_test


def _stratified_split_70_10_20(n, labels):
    idx_all = np.arange(n)
    idx_trainval, idx_test = train_test_split(
        idx_all, test_size=0.2, stratify=labels, random_state=SEED)
    labels_tv = [labels[i] for i in idx_trainval]
    idx_train, idx_val = train_test_split(
        idx_trainval, test_size=0.125, stratify=labels_tv, random_state=SEED)
    return idx_train, idx_val, idx_test


def prepare_gsm8k():
    """GSM8K test set (6140). text=question, label=0 (placeholder),
    gold_answer=final numeric answer after '####'."""
    name = "gsm8k"
    print(f"\n{'='*60}\nPreparing {name}")
    src = os.path.join(BASE, "datasets/reasoning_math/gsm8k/test.csv")

    samples = []
    with open(src) as f:
        for row in csv.DictReader(f):
            q = row["question"]
            ans = row["answer"]
            if "####" in ans:
                gold = ans.split("####")[-1].strip().replace(",", "")
            else:
                gold = ans.strip()
            samples.append({"text": q, "label": 0, "gold_answer": gold})

    print(f"  Loaded: {len(samples)} samples")
    idx_train, idx_val, idx_test = _random_split_70_10_20(len(samples))
    save_dataset(name, samples, idx_train, idx_val, idx_test)


def prepare_math():
    """MATH test (5000). text=en, label=0, gold_answer=answer string (can be fraction/expr)."""
    name = "math"
    print(f"\n{'='*60}\nPreparing {name}")
    src = os.path.join(BASE, "datasets/reasoning_math/chain_of_embedding/math/test.jsonl")

    samples = []
    with open(src) as f:
        for line in f:
            d = json.loads(line)
            samples.append({"text": d["en"], "label": 0, "gold_answer": str(d["answer"])})

    print(f"  Loaded: {len(samples)} samples")
    idx_train, idx_val, idx_test = _random_split_70_10_20(len(samples))
    save_dataset(name, samples, idx_train, idx_val, idx_test)


def prepare_commonsenseqa():
    """CommonsenseQA test (1221). text=en (question + choices),
    label=int 0..4 from letter A-E, gold_answer=letter."""
    name = "commonsenseqa"
    print(f"\n{'='*60}\nPreparing {name}")
    src = os.path.join(BASE, "datasets/reasoning_math/chain_of_embedding/commonsenseqa/test.jsonl")

    samples = []
    with open(src) as f:
        for line in f:
            d = json.loads(line)
            letter = str(d["answer"]).strip().upper()
            label = ord(letter) - ord("A") if letter and letter[0] in "ABCDE" else 0
            samples.append({"text": d["en"], "label": int(label), "gold_answer": letter})

    print(f"  Loaded: {len(samples)} samples, labels: {Counter(s['label'] for s in samples)}")
    labels = [s["label"] for s in samples]
    idx_train, idx_val, idx_test = _stratified_split_70_10_20(len(samples), labels)
    save_dataset(name, samples, idx_train, idx_val, idx_test)


def prepare_theoremqa():
    """TheoremQA test (800). text=en, label=0 (answers are heterogeneous: float/list/str),
    gold_answer=str(answer)."""
    name = "theoremqa"
    print(f"\n{'='*60}\nPreparing {name}")
    src = os.path.join(BASE, "datasets/reasoning_math/chain_of_embedding/theoremqa/test.jsonl")

    samples = []
    with open(src) as f:
        for line in f:
            d = json.loads(line)
            samples.append({"text": d["en"], "label": 0, "gold_answer": str(d["answer"])})

    print(f"  Loaded: {len(samples)} samples")
    idx_train, idx_val, idx_test = _random_split_70_10_20(len(samples))
    save_dataset(name, samples, idx_train, idx_val, idx_test)


def prepare_mmlu():
    """MMLU: download cais/mmlu 'all' test (14042) via HuggingFace, apply the local
    indices file (datasets/knowledge_factual/mmlu_indices/test.jsonl, 3511 indices)
    to curate the pool, then 70/10/20 stratified on choice idx.

    Format text to match CSQA/Belebele style: "Question: ...\\nChoices:\\n(A) ...\\n..."
    """
    name = "mmlu"
    print(f"\n{'='*60}\nPreparing {name}")
    idx_path = os.path.join(BASE, "datasets/knowledge_factual/mmlu_indices/test.jsonl")
    with open(idx_path) as f:
        indices = [int(line.strip()) for line in f if line.strip()]
    print(f"  Loaded {len(indices)} local indices (max={max(indices)})")

    from datasets import load_dataset
    print("  Downloading cais/mmlu 'all' test via HuggingFace…")
    ds = load_dataset("cais/mmlu", "all", split="test")
    print(f"  Full HF test: {len(ds)} rows")
    if max(indices) >= len(ds):
        raise RuntimeError(
            f"Index {max(indices)} out of range for HF mmlu test size {len(ds)}")

    letters = "ABCDE"
    samples = []
    for idx in indices:
        row = ds[int(idx)]
        q = row["question"]
        choices = row["choices"]
        ans = int(row["answer"])
        choices_block = "\n".join(f"({letters[i]}) {c}" for i, c in enumerate(choices))
        text = f"Question: {q}\nChoices:\n{choices_block}"
        samples.append({
            "text": text,
            "label": ans,
            "gold_answer": letters[ans],
        })

    print(f"  Prepared: {len(samples)} samples, labels: {Counter(s['label'] for s in samples)}")
    labels = [s["label"] for s in samples]
    idx_train, idx_val, idx_test = _stratified_split_70_10_20(len(samples), labels)
    save_dataset(name, samples, idx_train, idx_val, idx_test)


def prepare_belebele():
    """Belebele test (900). text=en, label=int 0..3 from '1'..'4' (1-indexed→0-indexed),
    gold_answer=raw string from source."""
    name = "belebele"
    print(f"\n{'='*60}\nPreparing {name}")
    src = os.path.join(BASE, "datasets/multilingual/chain_of_embedding/belebele/test.jsonl")

    samples = []
    with open(src) as f:
        for line in f:
            d = json.loads(line)
            raw = str(d["answer"]).strip()
            label = int(raw) - 1 if raw.isdigit() else 0  # 1-indexed → 0-indexed
            samples.append({"text": d["en"], "label": label, "gold_answer": raw})

    print(f"  Loaded: {len(samples)} samples, labels: {Counter(s['label'] for s in samples)}")
    labels = [s["label"] for s in samples]
    idx_train, idx_val, idx_test = _stratified_split_70_10_20(len(samples), labels)
    save_dataset(name, samples, idx_train, idx_val, idx_test)


def main():
    os.makedirs(OUT, exist_ok=True)
    import sys
    available = {
        "common_claim": prepare_common_claim,
        "when2call": prepare_when2call,
        "fava": prepare_fava,
        "ragtruth": prepare_ragtruth,
        "gsm8k": prepare_gsm8k,
        "math": prepare_math,
        "commonsenseqa": prepare_commonsenseqa,
        "theoremqa": prepare_theoremqa,
        "mmlu": prepare_mmlu,
        "belebele": prepare_belebele,
    }
    # CLI: `python prepare_new_datasets.py gsm8k,math,...` — comma list or empty=all
    args = sys.argv[1:]
    if args:
        requested = [n.strip() for n in args[0].split(",")]
        for n in requested:
            if n not in available:
                raise ValueError(f"Unknown dataset: {n}. Available: {list(available)}")
        to_run = requested
    else:
        to_run = list(available)
    for n in to_run:
        available[n]()
    print(f"\nAll done ({len(to_run)} datasets). Output at: {OUT}")


if __name__ == "__main__":
    main()
