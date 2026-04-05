"""
Save processed feature vectors for each method on each dataset.

For each (dataset, method, split), saves the feature matrix that goes into
the classifier — i.e. the post-processed representation BEFORE training.

Output structure:
  processed_features/{dataset}/{method}/{split}.pt
    - For probe methods: tensor (N, feature_dim)
    - For scorer methods: tensor (N,) — scalar score per sample

Uses the same layer/head/range selection as run_new_datasets.py (val-based).
Saves train, val, and test features so they can be used for analysis.
"""

import os
import json
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize

import sys
sys.path.insert(0, os.path.dirname(__file__))
from methods import (
    compute_lid, llm_check_score, compute_coe_scores,
    seakr_energy_score,
)

FEATURES_DIR = "/data/jehc223/NIPS2026/extraction/features"
OUTPUT_DIR = "/data/jehc223/NIPS2026/reproduce/processed_features"
HIDDEN_DIM = 3584

# Same dataset configs as run_new_datasets.py
DATASETS = {
    "e2h_amc_3class":      ("train_sub", "val_split", "eval",  3, "multiclass"),
    "e2h_amc_5class":      ("train_sub", "val_split", "eval",  5, "multiclass"),
    "common_claim_3class":  ("train",     "val",       "test",  3, "multiclass"),
    "when2call_3class":     ("train",     "val",       "test",  3, "multiclass"),
    "fava_binary":          ("train",     "val",       "test",  2, "binary"),
    "ragtruth_binary":      ("train",     "val",       "test",  2, "binary"),
}

FEATURE_ALIAS = {
    "fava_binary": "fava",
    "ragtruth_binary": "ragtruth",
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
        meta = json.load(f)
    data["labels"] = meta["labels"]
    return data


def save_feat(dataset, method, split_name, tensor):
    out_dir = os.path.join(OUTPUT_DIR, dataset, method)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{split_name}.pt")
    if isinstance(tensor, np.ndarray):
        tensor = torch.from_numpy(tensor)
    torch.save(tensor, path)


def save_meta(dataset, method, meta_dict):
    out_dir = os.path.join(OUTPUT_DIR, dataset, method)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta_dict, f, indent=2)


def select_best_layer(train, val, n_classes):
    """Select best layer via val AUROC using LogisticRegression."""
    n_layers = train["input_last_token_hidden"].shape[1]
    tr_labels = np.array(train["labels"])
    va_labels = np.array(val["labels"])
    best_auroc, best_layer = -1.0, 0
    for layer in range(n_layers):
        tr_acts = train["input_last_token_hidden"][:, layer, :].float().numpy()
        va_acts = val["input_last_token_hidden"][:, layer, :].float().numpy()
        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(tr_acts, tr_labels)
        va_probs = clf.predict_proba(va_acts)
        if n_classes == 2:
            auroc = roc_auc_score(va_labels, va_probs[:, 1])
        else:
            va_bin = label_binarize(va_labels, classes=list(range(n_classes)))
            auroc = roc_auc_score(va_bin, va_probs, average="macro", multi_class="ovr")
        if auroc > best_auroc:
            best_auroc = auroc
            best_layer = layer
    return best_layer


# ============================================================
# Per-method feature extractors
# ============================================================

def process_lr_probe(train, val, test, n_classes, dataset):
    """LR Probe: raw hidden state at best layer (no centering — matches run_new_datasets.py)."""
    best_layer = select_best_layer(train, val, n_classes)
    for split_name, data in [("train", train), ("val", val), ("test", test)]:
        acts = data["input_last_token_hidden"][:, best_layer, :].float()
        save_feat(dataset, "lr_probe", split_name, acts)
    save_meta(dataset, "lr_probe", {"best_layer": best_layer, "shape": "N x hidden_dim", "desc": "raw hidden state at best layer"})
    print(f"    lr_probe: layer={best_layer}")


def process_mm_probe(train, val, test, n_classes, dataset):
    """MM Probe: centered hidden state at best layer (binary only, direction-based)."""
    if n_classes > 2:
        print(f"    mm_probe: skipped (multi-class)")
        return
    best_layer = select_best_layer(train, val, n_classes)
    mean = train["input_last_token_hidden"][:, best_layer, :].float().mean(dim=0)
    for split_name, data in [("train", train), ("val", val), ("test", test)]:
        acts = data["input_last_token_hidden"][:, best_layer, :].float()
        centered = acts - mean
        save_feat(dataset, "mm_probe", split_name, centered)
    save_meta(dataset, "mm_probe", {"best_layer": best_layer, "shape": "N x hidden_dim", "desc": "centered hidden state at best layer (MM Probe uses mass-mean direction on this)"})
    print(f"    mm_probe: layer={best_layer}")


def process_pca_lr(train, val, test, n_classes, dataset):
    """PCA+LR: PCA-reduced + standardized hidden state at best layer."""
    n_layers = train["input_last_token_hidden"].shape[1]
    tr_labels = np.array(train["labels"])
    va_labels = np.array(val["labels"])
    best_auroc, best_layer = -1.0, 0
    for layer in range(n_layers):
        tr_acts = train["input_last_token_hidden"][:, layer, :].float()
        va_acts = val["input_last_token_hidden"][:, layer, :].float()
        mean = tr_acts.mean(dim=0)
        tr_c, va_c = tr_acts - mean, va_acts - mean
        U, S, Vh = torch.linalg.svd(tr_c, full_matrices=False)
        n_comp = min(50, tr_c.shape[0], tr_c.shape[1])
        tr_pca = (tr_c @ Vh.T[:, :n_comp]).numpy()
        va_pca = (va_c @ Vh.T[:, :n_comp]).numpy()
        sc = StandardScaler()
        tr_pca = sc.fit_transform(tr_pca)
        va_pca = sc.transform(va_pca)
        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(tr_pca, tr_labels)
        va_probs = clf.predict_proba(va_pca)
        if n_classes == 2:
            auroc = roc_auc_score(va_labels, va_probs[:, 1])
        else:
            va_bin = label_binarize(va_labels, classes=list(range(n_classes)))
            auroc = roc_auc_score(va_bin, va_probs, average="macro", multi_class="ovr")
        if auroc > best_auroc:
            best_auroc = auroc
            best_layer = layer

    # Save with best layer
    tr_acts = train["input_last_token_hidden"][:, best_layer, :].float()
    mean = tr_acts.mean(dim=0)
    U, S, Vh = torch.linalg.svd(tr_acts - mean, full_matrices=False)
    n_comp = min(50, tr_acts.shape[0], tr_acts.shape[1])
    sc = StandardScaler()
    tr_pca = sc.fit_transform(((tr_acts - mean) @ Vh.T[:, :n_comp]).numpy())
    save_feat(dataset, "pca_lr", "train", tr_pca)
    for split_name, data in [("val", val), ("test", test)]:
        acts = data["input_last_token_hidden"][:, best_layer, :].float()
        pca = sc.transform(((acts - mean) @ Vh.T[:, :n_comp]).numpy())
        save_feat(dataset, "pca_lr", split_name, pca)
    save_meta(dataset, "pca_lr", {"best_layer": best_layer, "n_components": n_comp, "shape": "N x 50", "desc": "PCA-reduced + standardized hidden state"})
    print(f"    pca_lr: layer={best_layer}, n_comp={n_comp}")


def process_iti(train, val, test, n_classes, dataset):
    """ITI: per-head activation at best (layer, head)."""
    tr_acts = train["input_per_head_activation"]
    va_acts = val["input_per_head_activation"]
    te_acts = test["input_per_head_activation"]
    tr_labels = np.array(train["labels"])
    va_labels = np.array(val["labels"])
    n_layers, n_heads = tr_acts.shape[1], tr_acts.shape[2]
    best_val, best_li, best_hi = -1.0, 0, 0
    for li in range(n_layers):
        for hi in range(n_heads):
            clf = LogisticRegression(max_iter=1000, random_state=42)
            clf.fit(tr_acts[:, li, hi, :].numpy(), tr_labels)
            va_probs = clf.predict_proba(va_acts[:, li, hi, :].numpy())
            if n_classes == 2:
                auroc = roc_auc_score(va_labels, va_probs[:, 1])
            else:
                va_bin = label_binarize(va_labels, classes=list(range(n_classes)))
                auroc = roc_auc_score(va_bin, va_probs, average="macro", multi_class="ovr")
            if auroc > best_val:
                best_val = auroc
                best_li, best_hi = li, hi
    for split_name, acts in [("train", tr_acts), ("val", va_acts), ("test", te_acts)]:
        save_feat(dataset, "iti", split_name, acts[:, best_li, best_hi, :])
    save_meta(dataset, "iti", {"best_layer": best_li, "best_head": best_hi, "shape": "N x head_dim", "desc": "per-head activation at best (layer, head)"})
    print(f"    iti: layer={best_li}, head={best_hi}")


def process_kb_mlp(train, val, test, n_classes, dataset):
    """KB MLP: hidden state at mid layer."""
    n_layers = train["input_last_token_hidden"].shape[1]
    mid_layer = n_layers // 2
    for split_name, data in [("train", train), ("val", val), ("test", test)]:
        acts = data["input_last_token_hidden"][:, mid_layer, :].float()
        save_feat(dataset, "kb_mlp", split_name, acts)
    save_meta(dataset, "kb_mlp", {"layer": mid_layer, "shape": "N x hidden_dim", "desc": "hidden state at mid layer (MLP input)"})
    print(f"    kb_mlp: layer={mid_layer}")


def process_lid(train, val, test, n_classes, dataset):
    """LID: scalar intrinsic dimensionality score per sample."""
    if n_classes > 2:
        print(f"    lid: skipped (multi-class)")
        return
    tr_labels = np.array(train["labels"])
    va_labels = np.array(val["labels"])
    n_layers = train["input_last_token_hidden"].shape[1]
    best_val_metric, best_layer = -1.0, 0
    for layer in range(n_layers):
        tr_acts = train["input_last_token_hidden"][:, layer, :]
        va_acts = val["input_last_token_hidden"][:, layer, :]
        ref_acts = tr_acts[tr_labels == 1]
        k = len(ref_acts) - 1
        if k < 2:
            continue
        lids = compute_lid(ref_acts, va_acts, k=k, hidden_dim=HIDDEN_DIM)
        auroc_pos = roc_auc_score(va_labels, lids)
        auroc_neg = roc_auc_score(va_labels, -lids)
        m = max(auroc_pos, auroc_neg)
        if m > best_val_metric:
            best_val_metric = m
            best_layer = layer
    ref_acts = train["input_last_token_hidden"][:, best_layer, :][tr_labels == 1]
    k = len(ref_acts) - 1
    for split_name, data in [("train", train), ("val", val), ("test", test)]:
        acts = data["input_last_token_hidden"][:, best_layer, :]
        scores = compute_lid(ref_acts, acts, k=k, hidden_dim=HIDDEN_DIM)
        save_feat(dataset, "lid", split_name, torch.tensor(scores, dtype=torch.float32))
    save_meta(dataset, "lid", {"best_layer": best_layer, "shape": "N", "desc": "LID score (scalar per sample)"})
    print(f"    lid: layer={best_layer}")


def process_attn_satisfies(train, val, test, n_classes, dataset):
    """Attn Satisfies: max-pooled attention value norms, flattened + standardized."""
    sc = StandardScaler()
    tr_feat = train["input_attn_value_norms"].float().max(dim=-1).values.reshape(len(train["labels"]), -1).numpy()
    tr_feat = sc.fit_transform(tr_feat)
    save_feat(dataset, "attn_satisfies", "train", tr_feat)
    for split_name, data in [("val", val), ("test", test)]:
        feat = data["input_attn_value_norms"].float().max(dim=-1).values.reshape(len(data["labels"]), -1).numpy()
        feat = sc.transform(feat)
        save_feat(dataset, "attn_satisfies", split_name, feat)
    save_meta(dataset, "attn_satisfies", {"shape": "N x (n_layers * n_heads)", "desc": "max-pooled attn value norms, standardized"})
    print(f"    attn_satisfies: done")


def process_llm_check(train, val, test, n_classes, dataset):
    """LLM-Check: scalar attention-based score per sample."""
    if n_classes > 2:
        print(f"    llm_check: skipped (multi-class)")
        return
    va_labels = np.array(val["labels"])
    n_layers = val["input_attn_stats"].shape[1]
    best_val, best_layer = -1.0, 0
    for layer in range(n_layers):
        scores = llm_check_score(val["input_attn_stats"], layer_num=layer)
        auroc_pos = roc_auc_score(va_labels, scores)
        auroc_neg = roc_auc_score(va_labels, -scores)
        m = max(auroc_pos, auroc_neg)
        if m > best_val:
            best_val = m
            best_layer = layer
    for split_name, data in [("train", train), ("val", val), ("test", test)]:
        scores = llm_check_score(data["input_attn_stats"], layer_num=best_layer)
        save_feat(dataset, "llm_check", split_name, torch.tensor(scores, dtype=torch.float32))
    save_meta(dataset, "llm_check", {"best_layer": best_layer, "shape": "N", "desc": "LLM-Check attention score"})
    print(f"    llm_check: layer={best_layer}")


def process_sep(train, val, test, n_classes, dataset):
    """SEP: flattened gen hidden states at best layer range."""
    n_layers = train["gen_last_token_hidden"].shape[1]
    tr_labels = np.array(train["labels"])
    va_labels = np.array(val["labels"])
    best_auroc, best_range = -1.0, (0, 1)
    for start in range(n_layers):
        for end in range(start + 1, min(start + 6, n_layers + 1)):
            X_tr = train["gen_last_token_hidden"][:, start:end, :].float().reshape(len(tr_labels), -1).numpy()
            X_va = val["gen_last_token_hidden"][:, start:end, :].float().reshape(len(va_labels), -1).numpy()
            clf = LogisticRegression(max_iter=1000, random_state=42)
            clf.fit(X_tr, tr_labels)
            va_probs = clf.predict_proba(X_va)
            if n_classes == 2:
                auroc = roc_auc_score(va_labels, va_probs[:, 1])
            else:
                va_bin = label_binarize(va_labels, classes=list(range(n_classes)))
                auroc = roc_auc_score(va_bin, va_probs, average="macro", multi_class="ovr")
            if auroc > best_auroc:
                best_auroc = auroc
                best_range = (start, end)
    for split_name, data in [("train", train), ("val", val), ("test", test)]:
        feat = data["gen_last_token_hidden"][:, best_range[0]:best_range[1], :].float().reshape(len(data["labels"]), -1)
        save_feat(dataset, "sep", split_name, feat)
    save_meta(dataset, "sep", {"best_range": list(best_range), "shape": f"N x ({best_range[1]-best_range[0]} * hidden_dim)", "desc": "flattened gen hidden at best layer range"})
    print(f"    sep: range={best_range}")


def process_coe(train, val, test, n_classes, dataset):
    """CoE: geometric scores (multiple variants, scalar per sample)."""
    if n_classes > 2:
        print(f"    coe: skipped (multi-class)")
        return
    for split_name, data in [("train", train), ("val", val), ("test", test)]:
        scores = compute_coe_scores(data["gen_mean_pool_hidden"])
        # Save each variant under coe/ directory
        for variant, vals in scores.items():
            save_feat(dataset, "coe", f"{split_name}_{variant}", torch.tensor(vals, dtype=torch.float32))
    save_meta(dataset, "coe", {"variants": list(scores.keys()), "shape": "N per variant", "desc": "CoE geometric scores", "file_pattern": "{split}_{variant}.pt"})
    print(f"    coe: {len(scores)} variants")


def process_seakr(train, val, test, n_classes, dataset):
    """SeaKR: energy score (scalar per sample)."""
    if n_classes > 2:
        print(f"    seakr: skipped (multi-class)")
        return
    for split_name, data in [("train", train), ("val", val), ("test", test)]:
        scores = seakr_energy_score(data["gen_logit_stats_last"])
        save_feat(dataset, "seakr", split_name, torch.tensor(scores, dtype=torch.float32))
    save_meta(dataset, "seakr", {"shape": "N", "desc": "SeaKR energy score"})
    print(f"    seakr: done")


def process_step(train, val, test, n_classes, dataset):
    """STEP: gen hidden state at last decoder layer."""
    for split_name, data in [("train", train), ("val", val), ("test", test)]:
        acts = data["gen_last_token_hidden"][:, -2, :].float()
        save_feat(dataset, "step", split_name, acts)
    save_meta(dataset, "step", {"layer": "last decoder (-2)", "shape": "N x hidden_dim", "desc": "gen hidden state at last decoder layer"})
    print(f"    step: done")


PROCESSORS = {
    "lr_probe": process_lr_probe,
    "mm_probe": process_mm_probe,
    "pca_lr": process_pca_lr,
    "iti": process_iti,
    "kb_mlp": process_kb_mlp,
    "lid": process_lid,
    "attn_satisfies": process_attn_satisfies,
    "llm_check": process_llm_check,
    "sep": process_sep,
    "coe": process_coe,
    "seakr": process_seakr,
    "step": process_step,
}


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for dataset, (train_split, val_split, test_split, n_classes, task_type) in DATASETS.items():
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset} ({task_type}, {n_classes} classes)")

        feat_name = FEATURE_ALIAS.get(dataset, dataset)
        train = load_split(feat_name, train_split)
        val = load_split(feat_name, val_split)
        test = load_split(feat_name, test_split)
        print(f"  Train: {len(train['labels'])}, Val: {len(val['labels'])}, Test: {len(test['labels'])}")

        for method_name, processor in PROCESSORS.items():
            try:
                processor(train, val, test, n_classes, dataset)
            except Exception as e:
                print(f"    {method_name}: ERROR {e}")

    print(f"\nAll done! Output at: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
