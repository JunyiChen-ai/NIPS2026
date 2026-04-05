"""
Run all baseline methods on new datasets.
Handles binary, multi-class, and multi-label classification.

Binary datasets: same pipeline as run_all.py
Multi-class: supervised probes adapted to multi-class, unsupervised methods skipped
Multi-label: per-label binary evaluation, aggregated metrics

Results saved to results/all_results_v3.json
"""

import os
import sys
import json
import traceback
import torch
import numpy as np
from sklearn.metrics import (roc_auc_score, accuracy_score, f1_score,
                             log_loss)
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.linear_model import LogisticRegression, Ridge
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(__file__))
from methods import (
    LRProbe, MMProbe, pca_lr_probe, iti_directions,
    KBNet, compute_lid, attention_satisfies_probe,
    llm_check_score, sep_probe, compute_coe_scores,
    seakr_energy_score, STEPScorer,
)

FEATURES_DIR = "/data/jehc223/NIPS2026/extraction/features"
RESULTS_DIR = "/data/jehc223/NIPS2026/reproduce/results"
HIDDEN_DIM = 3584
os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# Data loading
# ============================================================
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
    data["texts"] = meta["texts"]
    data["gen_lens"] = meta["gen_lens"]
    data["labels_multi"] = meta.get("labels_multi")
    return data


# ============================================================
# Evaluation helpers
# ============================================================
def eval_binary(y_true, probs, preds=None):
    y_true = np.array(y_true)
    probs = np.array(probs)
    preds = (probs > 0.5).astype(int) if preds is None else np.array(preds)
    return {"auroc": roc_auc_score(y_true, probs),
            "accuracy": accuracy_score(y_true, preds),
            "f1": f1_score(y_true, preds, zero_division=0)}


def eval_multiclass(y_true, probs_matrix, preds=None):
    """Evaluate multi-class classification.
    probs_matrix: (N, n_classes) probability matrix
    """
    y_true = np.array(y_true)
    probs_matrix = np.array(probs_matrix)
    n_classes = probs_matrix.shape[1]
    preds = probs_matrix.argmax(axis=1) if preds is None else np.array(preds)

    # AUROC: one-vs-rest, macro averaged
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))
    try:
        auroc = roc_auc_score(y_bin, probs_matrix, average="macro", multi_class="ovr")
    except ValueError:
        auroc = float("nan")  # missing class in split

    return {"auroc": auroc,
            "accuracy": accuracy_score(y_true, preds),
            "f1_macro": f1_score(y_true, preds, average="macro", zero_division=0),
            "f1_weighted": f1_score(y_true, preds, average="weighted", zero_division=0)}


def eval_scoring_with_val(val_labels, val_scores, test_labels, test_scores):
    """Direction + threshold on val, evaluate on test (binary only)."""
    val_labels, val_scores = np.array(val_labels), np.array(val_scores)
    test_labels, test_scores = np.array(test_labels), np.array(test_scores)

    auroc_pos = roc_auc_score(val_labels, val_scores)
    auroc_neg = roc_auc_score(val_labels, -val_scores)
    flip = auroc_neg > auroc_pos
    oriented_val = -val_scores if flip else val_scores
    oriented_test = -test_scores if flip else test_scores

    thresholds = np.unique(oriented_val)
    if len(thresholds) > 200:
        thresholds = np.percentile(oriented_val, np.linspace(0, 100, 200))
    best_f1, best_thr = 0, float(np.median(oriented_val))
    for thr in thresholds:
        preds = (oriented_val >= thr).astype(int)
        f = f1_score(val_labels, preds, zero_division=0)
        if f > best_f1:
            best_f1 = f
            best_thr = float(thr)

    preds = (oriented_test >= best_thr).astype(int)
    return {"auroc": roc_auc_score(test_labels, oriented_test),
            "accuracy": accuracy_score(test_labels, preds),
            "f1": f1_score(test_labels, preds, zero_division=0),
            "threshold": best_thr, "flipped": flip}


# ============================================================
# Layer selection (multi-class aware)
# ============================================================
def select_layer_multiclass(train, val, test, n_classes):
    """Select best layer on val using multi-class LogisticRegression."""
    n_layers = train["input_last_token_hidden"].shape[1]
    tr_labels = np.array(train["labels"])
    va_labels = np.array(val["labels"])
    te_labels = np.array(test["labels"])

    best_val_auroc = -1.0
    best_layer = 0

    for layer in range(n_layers):
        tr_acts = train["input_last_token_hidden"][:, layer, :].float().numpy()
        va_acts = val["input_last_token_hidden"][:, layer, :].float().numpy()
        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(tr_acts, tr_labels)
        va_probs = clf.predict_proba(va_acts)
        va_bin = label_binarize(va_labels, classes=list(range(n_classes)))
        auroc = roc_auc_score(va_bin, va_probs, average="macro", multi_class="ovr")
        if auroc > best_val_auroc:
            best_val_auroc = auroc
            best_layer = layer

    # Final eval on test
    tr_acts = train["input_last_token_hidden"][:, best_layer, :].float().numpy()
    te_acts = test["input_last_token_hidden"][:, best_layer, :].float().numpy()
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(tr_acts, tr_labels)
    te_probs = clf.predict_proba(te_acts)
    return {"best_layer": best_layer, "test_results": eval_multiclass(te_labels, te_probs)}


def select_layer_binary(train, val, test):
    """Select best layer on val using binary LogisticRegression (same as original)."""
    n_layers = train["input_last_token_hidden"].shape[1]
    tr_labels = np.array(train["labels"])
    va_labels = np.array(val["labels"])
    te_labels = np.array(test["labels"])

    best_val_auroc = -1.0
    best_layer = 0

    for layer in range(n_layers):
        tr_acts = train["input_last_token_hidden"][:, layer, :].float().numpy()
        va_acts = val["input_last_token_hidden"][:, layer, :].float().numpy()
        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(tr_acts, tr_labels)
        va_probs = clf.predict_proba(va_acts)[:, 1]
        auroc = roc_auc_score(va_labels, va_probs)
        if auroc > best_val_auroc:
            best_val_auroc = auroc
            best_layer = layer

    tr_acts = train["input_last_token_hidden"][:, best_layer, :].float().numpy()
    te_acts = test["input_last_token_hidden"][:, best_layer, :].float().numpy()
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(tr_acts, tr_labels)
    te_probs = clf.predict_proba(te_acts)[:, 1]
    return {"best_layer": best_layer, "test_results": eval_binary(te_labels, te_probs)}


# ============================================================
# Method runners
# ============================================================

# --- Supervised methods (work for both binary and multi-class) ---

def run_lr_probe(train, val, test, n_classes):
    if n_classes == 2:
        return select_layer_binary(train, val, test)
    return select_layer_multiclass(train, val, test, n_classes)


def run_mm_probe(train, val, test, n_classes):
    if n_classes > 2:
        return {"skipped": "MM Probe is binary-only (direction-based)"}
    # Original binary MM Probe
    from run_all import run_mm_probe as _run_mm
    return _run_mm(train, val, test, False)


def run_pca_lr(train, val, test, n_classes):
    """PCA + LogisticRegression. Multi-class via sklearn."""
    n_layers = train["input_last_token_hidden"].shape[1]
    tr_labels = np.array(train["labels"])
    va_labels = np.array(val["labels"])
    te_labels = np.array(test["labels"])

    best_val_auroc = -1.0
    best_layer = 0

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
        if auroc > best_val_auroc:
            best_val_auroc = auroc
            best_layer = layer

    # Final eval
    tr_acts = train["input_last_token_hidden"][:, best_layer, :].float()
    te_acts = test["input_last_token_hidden"][:, best_layer, :].float()
    mean = tr_acts.mean(dim=0)
    tr_c, te_c = tr_acts - mean, te_acts - mean
    U, S, Vh = torch.linalg.svd(tr_c, full_matrices=False)
    n_comp = min(50, tr_c.shape[0], tr_c.shape[1])
    tr_pca = (tr_c @ Vh.T[:, :n_comp]).numpy()
    te_pca = (te_c @ Vh.T[:, :n_comp]).numpy()
    sc = StandardScaler()
    tr_pca = sc.fit_transform(tr_pca)
    te_pca = sc.transform(te_pca)
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(tr_pca, tr_labels)
    te_probs = clf.predict_proba(te_pca)
    if n_classes == 2:
        return {"best_layer": best_layer, "test_results": eval_binary(te_labels, te_probs[:, 1])}
    return {"best_layer": best_layer, "test_results": eval_multiclass(te_labels, te_probs)}


def run_iti(train, val, test, n_classes):
    """Per-head LogisticRegression, select best head on val."""
    tr_acts = train["input_per_head_activation"]
    va_acts = val["input_per_head_activation"]
    te_acts = test["input_per_head_activation"]
    tr_labels = np.array(train["labels"])
    va_labels = np.array(val["labels"])
    te_labels = np.array(test["labels"])
    n_layers, n_heads = tr_acts.shape[1], tr_acts.shape[2]

    best_val = -1.0
    best_li, best_hi = 0, 0
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

    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(tr_acts[:, best_li, best_hi, :].numpy(), tr_labels)
    te_probs = clf.predict_proba(te_acts[:, best_li, best_hi, :].numpy())
    if n_classes == 2:
        result = eval_binary(te_labels, te_probs[:, 1])
    else:
        result = eval_multiclass(te_labels, te_probs)
    return {"best_layer": int(best_li), "best_head": int(best_hi), "test_results": result}


def run_kb_mlp(train, val, test, n_classes):
    """KB MLP: mid-layer hidden states, MLP classifier."""
    n_layers = train["input_last_token_hidden"].shape[1]
    mid_layer = n_layers // 2
    tr_acts = train["input_last_token_hidden"][:, mid_layer, :]
    va_acts = val["input_last_token_hidden"][:, mid_layer, :]
    te_acts = test["input_last_token_hidden"][:, mid_layer, :]
    tr_labels = torch.tensor(train["labels"])
    va_labels = torch.tensor(val["labels"])
    te_labels = torch.tensor(test["labels"])

    if n_classes == 2:
        probs, preds = KBNet.train_and_eval(tr_acts, tr_labels, te_acts, te_labels,
                                             val_acts=va_acts, val_labels=va_labels)
        return {"layer": mid_layer, "test_results": eval_binary(te_labels.numpy(), probs, preds)}
    else:
        # Multi-class KBNet: change output dim
        d_in = tr_acts.shape[-1]
        model = KBNet(d_in)
        # Replace final layer for n_classes
        model.net[-1] = torch.nn.Linear(32, n_classes)
        opt = torch.optim.Adam(model.parameters(), lr=5e-5)
        criterion = torch.nn.CrossEntropyLoss()

        best_state, best_val_auroc = None, -1.0
        for epoch in range(30):
            model.train()
            perm = torch.randperm(len(tr_acts))
            for i in range(0, len(tr_acts), 16):
                idx = perm[i:i+16]
                opt.zero_grad()
                out = model(tr_acts[idx].float())
                loss = criterion(out, tr_labels[idx].long())
                loss.backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                va_logits = model(va_acts.float())
                va_probs = torch.softmax(va_logits, dim=-1).numpy()
            va_bin = label_binarize(va_labels.numpy(), classes=list(range(n_classes)))
            auroc = roc_auc_score(va_bin, va_probs, average="macro", multi_class="ovr")
            if auroc > best_val_auroc:
                best_val_auroc = auroc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            te_logits = model(te_acts.float())
            te_probs = torch.softmax(te_logits, dim=-1).numpy()
        return {"layer": mid_layer, "test_results": eval_multiclass(te_labels.numpy(), te_probs)}


def run_lid(train, val, test, n_classes):
    if n_classes > 2:
        return {"skipped": "LID is unsupervised, cannot do multi-class"}
    from run_all import run_lid as _run_lid
    return _run_lid(train, val, test, False)


def run_attn_satisfies(train, val, test, n_classes):
    if n_classes == 2:
        tr_labels = torch.tensor(train["labels"])
        te_labels = torch.tensor(test["labels"])
        probs, preds = attention_satisfies_probe(
            train["input_attn_value_norms"], tr_labels,
            test["input_attn_value_norms"], te_labels)
        return {"test_results": eval_binary(te_labels.numpy(), probs, preds)}
    else:
        # Multi-class: flatten attn value norms, use sklearn LR
        tr_feat = train["input_attn_value_norms"].float().max(dim=-1).values
        tr_feat = tr_feat.reshape(len(train["labels"]), -1).numpy()
        te_feat = test["input_attn_value_norms"].float().max(dim=-1).values
        te_feat = te_feat.reshape(len(test["labels"]), -1).numpy()
        sc = StandardScaler()
        tr_feat = sc.fit_transform(tr_feat)
        te_feat = sc.transform(te_feat)
        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(tr_feat, np.array(train["labels"]))
        te_probs = clf.predict_proba(te_feat)
        return {"test_results": eval_multiclass(np.array(test["labels"]), te_probs)}


def run_llm_check(train, val, test, n_classes):
    if n_classes > 2:
        return {"skipped": "LLM-Check is unsupervised, cannot do multi-class"}
    from run_all import run_llm_check as _run
    return _run(train, val, test, False)


def run_sep(train, val, test, n_classes):
    """SEP: layer-range selection on gen_last_token_hidden."""
    n_layers = train["gen_last_token_hidden"].shape[1]
    tr_labels = np.array(train["labels"])
    va_labels = np.array(val["labels"])
    te_labels = np.array(test["labels"])

    best_val_auroc = -1.0
    best_range = (0, 1)
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
            if auroc > best_val_auroc:
                best_val_auroc = auroc
                best_range = (start, end)

    X_tr = train["gen_last_token_hidden"][:, best_range[0]:best_range[1], :].float().reshape(len(tr_labels), -1).numpy()
    X_te = test["gen_last_token_hidden"][:, best_range[0]:best_range[1], :].float().reshape(len(te_labels), -1).numpy()
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_tr, tr_labels)
    te_probs = clf.predict_proba(X_te)
    if n_classes == 2:
        result = eval_binary(te_labels, te_probs[:, 1])
    else:
        result = eval_multiclass(te_labels, te_probs)
    return {"best_range": list(best_range), "test_results": result}


def run_coe(train, val, test, n_classes):
    if n_classes > 2:
        return {"skipped": "CoE is unsupervised, cannot do multi-class"}
    va_labels = np.array(val["labels"])
    te_labels = np.array(test["labels"])
    va_scores = compute_coe_scores(val["gen_mean_pool_hidden"])
    te_scores = compute_coe_scores(test["gen_mean_pool_hidden"])
    results = {}
    for k in te_scores:
        results[k] = eval_scoring_with_val(va_labels, va_scores[k], te_labels, te_scores[k])
    return results


def run_seakr(train, val, test, n_classes):
    if n_classes > 2:
        return {"skipped": "SeaKR is unsupervised, cannot do multi-class"}
    va_labels = np.array(val["labels"])
    te_labels = np.array(test["labels"])
    va_scores = seakr_energy_score(val["gen_logit_stats_last"])
    te_scores = seakr_energy_score(test["gen_logit_stats_last"])
    return {"test_results": eval_scoring_with_val(va_labels, va_scores, te_labels, te_scores)}


def run_step(train, val, test, n_classes):
    tr_labels = torch.tensor(train["labels"])
    va_labels = torch.tensor(val["labels"])
    te_labels = torch.tensor(test["labels"])
    tr_acts = train["gen_last_token_hidden"][:, -2, :]
    va_acts = val["gen_last_token_hidden"][:, -2, :]
    te_acts = test["gen_last_token_hidden"][:, -2, :]

    if n_classes == 2:
        probs, preds = STEPScorer.train_and_eval(
            tr_acts, tr_labels, te_acts, te_labels,
            val_acts=va_acts, val_labels=va_labels)
        return {"test_results": eval_binary(te_labels.numpy(), probs, preds)}
    else:
        # Multi-class STEP: use sklearn LR on last decoder layer
        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(tr_acts.float().numpy(), tr_labels.numpy())
        te_probs = clf.predict_proba(te_acts.float().numpy())
        return {"test_results": eval_multiclass(te_labels.numpy(), te_probs)}


# ============================================================
# Multi-label runner
# ============================================================
def run_multilabel(train, val, test, method_fn, label_names):
    """Run a binary method independently per label, aggregate results."""
    labels_multi_tr = np.array(train["labels_multi"])
    labels_multi_va = np.array(val["labels_multi"])
    labels_multi_te = np.array(test["labels_multi"])
    n_labels = labels_multi_tr.shape[1]

    per_label = {}
    for li in range(n_labels):
        lname = label_names[li] if li < len(label_names) else f"label_{li}"
        # Create per-label data views
        tr_copy = dict(train)
        va_copy = dict(val)
        te_copy = dict(test)
        tr_copy["labels"] = labels_multi_tr[:, li].tolist()
        va_copy["labels"] = labels_multi_va[:, li].tolist()
        te_copy["labels"] = labels_multi_te[:, li].tolist()

        result = method_fn(tr_copy, va_copy, te_copy, 2)
        per_label[lname] = result

    # Aggregate: macro average of per-label AUROC/F1/accuracy
    aurocs, accs, f1s = [], [], []
    for lname, r in per_label.items():
        # Handle both {"test_results": {...}} and CoE-style {variant: {...}} formats
        tr = r.get("test_results", None)
        if tr is None:
            # Try first nested dict with "auroc" (e.g. CoE score variants)
            for v in r.values():
                if isinstance(v, dict) and "auroc" in v:
                    tr = v
                    break
        if tr and "auroc" in tr:
            aurocs.append(tr["auroc"])
            accs.append(tr.get("accuracy", float("nan")))
            f1s.append(tr.get("f1", float("nan")))
    agg = {}
    if aurocs:
        agg = {"macro_auroc": np.mean(aurocs), "macro_accuracy": np.mean(accs),
               "macro_f1": np.mean(f1s)}

    return {"per_label": per_label, "aggregated": agg}


# ============================================================
# Dataset configs
# ============================================================
DATASETS = {
    # (train_split, val_split, test_split, n_classes, task_type)
    # task_type: "binary", "multiclass", "multilabel"

    # Multi-class
    "e2h_amc_3class":      ("train_sub", "val_split", "eval",  3, "multiclass"),
    "e2h_amc_5class":      ("train_sub", "val_split", "eval",  5, "multiclass"),
    "common_claim_3class":  ("train",     "val",       "test",  3, "multiclass"),
    "when2call_3class":     ("train",     "val",       "test",  3, "multiclass"),

    # Binary
    "fava_binary":          ("train",     "val",       "test",  2, "binary"),
    "ragtruth_binary":      ("train",     "val",       "test",  2, "binary"),

    # Multi-label (features shared with binary variant)
    "fava_multilabel":      ("train",     "val",       "test",  2, "multilabel"),
    "ragtruth_multilabel":  ("train",     "val",       "test",  2, "multilabel"),
}

# Multi-label datasets share features with their binary counterpart
FEATURE_ALIAS = {
    "fava_binary": "fava",
    "fava_multilabel": "fava",
    "ragtruth_binary": "ragtruth",
    "ragtruth_multilabel": "ragtruth",
}

MULTILABEL_NAMES = {
    "fava_multilabel": ["entity", "relation", "contradictory", "unverifiable", "invented", "subjective"],
    "ragtruth_multilabel": ["evident_conflict", "baseless_info"],
}

METHODS = {
    "lr_probe": run_lr_probe,
    "mm_probe": run_mm_probe,
    "pca_lr": run_pca_lr,
    "iti": run_iti,
    "kb_mlp": run_kb_mlp,
    "lid": run_lid,
    "attn_satisfies": run_attn_satisfies,
    "llm_check": run_llm_check,
    "sep": run_sep,
    "coe": run_coe,
    "seakr": run_seakr,
    "step": run_step,
}


# ============================================================
# Main
# ============================================================
def load_existing_results():
    p = os.path.join(RESULTS_DIR, "all_results_v3.json")
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return {}


def save_results(r):
    with open(os.path.join(RESULTS_DIR, "all_results_v3.json"), "w") as f:
        json.dump(r, f, indent=2, default=str)


def main():
    all_results = load_existing_results()

    for dataset, (train_split, val_split, test_split, n_classes, task_type) in DATASETS.items():
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset} ({task_type}, {n_classes} classes)")
        print(f"{'='*60}")

        if dataset not in all_results:
            all_results[dataset] = {}
        ds_results = all_results[dataset]

        methods_to_run = {k: v for k, v in METHODS.items()
                         if k not in ds_results or "error" in ds_results.get(k, {})}
        if not methods_to_run:
            print("  All methods done, skipping.")
            continue

        # Load features (use alias for fava/ragtruth variants)
        feat_name = FEATURE_ALIAS.get(dataset, dataset)
        train = load_split(feat_name, train_split)
        val = load_split(feat_name, val_split)
        test = load_split(feat_name, test_split)

        # For binary variants of fava/ragtruth, labels are already binary in meta.json
        # For multilabel, we need labels_multi
        print(f"  Train: {len(train['labels'])}, Val: {len(val['labels'])}, Test: {len(test['labels'])}")
        print(f"  Methods to run: {list(methods_to_run.keys())}")

        for name, fn in methods_to_run.items():
            print(f"\n  --- {name} ---", flush=True)
            try:
                if task_type == "multilabel":
                    label_names = MULTILABEL_NAMES.get(dataset, [])
                    result = run_multilabel(train, val, test, fn, label_names)
                else:
                    result = fn(train, val, test, n_classes)

                ds_results[name] = result
                if isinstance(result, dict) and "skipped" in result:
                    print(f"    Skipped: {result['skipped']}")
                elif "test_results" in result:
                    tr = result["test_results"]
                    info = ", ".join(f"{k}={result[k]}" for k in ["best_layer", "best_head", "best_range", "layer"] if k in result)
                    print(f"    {info}: {tr}")
                elif "aggregated" in result:
                    print(f"    Aggregated: {result['aggregated']}")
                else:
                    for k, v in result.items():
                        if isinstance(v, dict) and len(str(v)) < 200:
                            print(f"    {k}: {v}")
            except Exception as e:
                print(f"    ERROR: {e}")
                traceback.print_exc()
                ds_results[name] = {"error": str(e)}
            save_results(all_results)

    print(f"\nAll done! Results at {RESULTS_DIR}/all_results_v3.json")


if __name__ == "__main__":
    main()
