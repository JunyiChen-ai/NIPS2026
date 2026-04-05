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


def main():
    os.makedirs(OUT, exist_ok=True)
    prepare_common_claim()
    prepare_when2call()
    prepare_fava()
    prepare_ragtruth()
    print(f"\nAll done. Output at: {OUT}")


if __name__ == "__main__":
    main()
