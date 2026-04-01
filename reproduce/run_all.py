"""
Run all reproducible baseline methods on all 4 datasets.
Uses train_sub for training, val_split for model selection (layer/epoch),
and test for final evaluation. No test leakage.

Each method's layer selection follows the original repo:
- Fixed layer: use as-is, no selection needed
- Val-based selection: select on val_split, evaluate on test
- No layer selection: use all layers or fixed computation
"""

import os
import json
import traceback
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from scipy.stats import spearmanr

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
    return data


# ============================================================
# Evaluation helpers
# ============================================================
def eval_cls(y_true, probs, preds=None):
    y_true = np.array(y_true)
    probs = np.array(probs)
    preds = (probs > 0.5).astype(int) if preds is None else np.array(preds)
    return {"auroc": roc_auc_score(y_true, probs),
            "accuracy": accuracy_score(y_true, preds),
            "f1": f1_score(y_true, preds, zero_division=0)}


def eval_reg(y_true, scores):
    y_true = np.array(y_true, dtype=np.float64)
    scores = np.array(scores, dtype=np.float64)
    corr, pval = spearmanr(y_true, scores)
    return {"spearman_r": corr, "spearman_p": pval,
            "mse": float(mean_squared_error(y_true, scores))}


def eval_scoring(y_true, scores):
    y_true = np.array(y_true)
    scores = np.array(scores)
    auroc = roc_auc_score(y_true, scores)
    auroc_flip = roc_auc_score(y_true, -scores)
    return {"auroc": max(auroc, auroc_flip), "auroc_raw": auroc, "auroc_flipped": auroc_flip}


# ============================================================
# Layer selection on val, evaluate on test
# ============================================================
def select_layer_on_val(train_data, val_data, test_data, train_eval_fn, is_regression):
    """Select best layer using val set, then evaluate on test with that layer."""
    n_layers = train_data["input_last_token_hidden"].shape[1]
    train_labels = torch.tensor(train_data["labels"])
    val_labels = torch.tensor(val_data["labels"])
    test_labels = torch.tensor(test_data["labels"])

    best_val_metric = -1.0
    best_layer = 0

    for layer in range(n_layers):
        tr_acts = train_data["input_last_token_hidden"][:, layer, :]
        va_acts = val_data["input_last_token_hidden"][:, layer, :]
        val_result = train_eval_fn(tr_acts, train_labels, va_acts, val_labels)
        metric = abs(val_result.get("spearman_r", val_result.get("auroc", 0)))
        if metric > best_val_metric:
            best_val_metric = metric
            best_layer = layer

    # Final eval on test with best layer
    tr_acts = train_data["input_last_token_hidden"][:, best_layer, :]
    te_acts = test_data["input_last_token_hidden"][:, best_layer, :]
    test_result = train_eval_fn(tr_acts, train_labels, te_acts, test_labels)
    return {"best_layer": best_layer, "val_metric": best_val_metric, "test_results": test_result}


# ============================================================
# Method runners (faithful to original repos)
# ============================================================

# --- 1. LR Probe (GoT) — original: fixed layer, we use val to select ---
def run_lr_probe(train, val, test, is_reg):
    def fn(tr_acts, tr_labels, ev_acts, ev_labels):
        mean = tr_acts.float().mean(dim=0)
        tr_c, ev_c = tr_acts.float() - mean, ev_acts.float() - mean
        if is_reg:
            clf = Ridge(alpha=1.0)
            clf.fit(tr_c.numpy(), tr_labels.numpy())
            return eval_reg(ev_labels.numpy(), clf.predict(ev_c.numpy()))
        else:
            probe = LRProbe.from_data(tr_c, tr_labels.float(), epochs=1000)
            with torch.no_grad():
                probs = probe(ev_c).numpy()
            return eval_cls(ev_labels.numpy(), probs)
    return select_layer_on_val(train, val, test, fn, is_reg)


# --- 2. MM Probe (GoT) — original: fixed layer, we use val to select ---
def run_mm_probe(train, val, test, is_reg):
    if is_reg:
        # MM Probe direction can be used for regression via correlation
        def fn(tr_acts, tr_labels, ev_acts, ev_labels):
            mean = tr_acts.float().mean(dim=0)
            tr_c, ev_c = tr_acts.float() - mean, ev_acts.float() - mean
            # Use median split to get pos/neg for direction, then project for regression
            median = tr_labels.float().median()
            binary = (tr_labels.float() >= median).long()
            probe = MMProbe.from_data(tr_c, binary)
            with torch.no_grad():
                scores = probe(ev_c).numpy()
            return eval_reg(ev_labels.numpy(), scores)
        return select_layer_on_val(train, val, test, fn, is_reg)
    def fn(tr_acts, tr_labels, ev_acts, ev_labels):
        mean = tr_acts.float().mean(dim=0)
        tr_c, ev_c = tr_acts.float() - mean, ev_acts.float() - mean
        probe = MMProbe.from_data(tr_c, tr_labels)
        with torch.no_grad():
            probs = probe(ev_c).numpy()
        return eval_cls(ev_labels.numpy(), probs)
    return select_layer_on_val(train, val, test, fn, is_reg)


# --- 3. PCA+LR (No Answer Needed) — original: 5-fold CV, we use val ---
def run_pca_lr(train, val, test, is_reg):
    def fn(tr_acts, tr_labels, ev_acts, ev_labels):
        if is_reg:
            mean = tr_acts.float().mean(dim=0)
            tr_c, ev_c = tr_acts.float() - mean, ev_acts.float() - mean
            U, S, Vh = torch.linalg.svd(tr_c, full_matrices=False)
            n_comp = min(50, tr_acts.shape[0], tr_acts.shape[1])
            tr_pca = (tr_c @ Vh.T[:, :n_comp]).numpy()
            ev_pca = (ev_c @ Vh.T[:, :n_comp]).numpy()
            sc = StandardScaler()
            tr_pca = sc.fit_transform(tr_pca)
            ev_pca = sc.transform(ev_pca)
            clf = Ridge(alpha=1.0)
            clf.fit(tr_pca, tr_labels.numpy())
            return eval_reg(ev_labels.numpy(), clf.predict(ev_pca))
        else:
            probs, preds = pca_lr_probe(tr_acts, tr_labels, ev_acts, ev_labels)
            return eval_cls(ev_labels.numpy(), probs, preds)
    return select_layer_on_val(train, val, test, fn, is_reg)


# --- 4. ITI — original: val split selects heads, we use val ---
def run_iti(train, val, test, is_reg):
    if is_reg:
        # Per-head Ridge regression, select best head on val
        from sklearn.linear_model import LogisticRegression
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
                clf = Ridge(alpha=1.0)
                clf.fit(tr_acts[:, li, hi, :].numpy(), tr_labels)
                preds = clf.predict(va_acts[:, li, hi, :].numpy())
                m = abs(spearmanr(va_labels, preds)[0])
                if m > best_val:
                    best_val = m
                    best_li, best_hi = li, hi
        clf = Ridge(alpha=1.0)
        clf.fit(tr_acts[:, best_li, best_hi, :].numpy(), tr_labels)
        preds = clf.predict(te_acts[:, best_li, best_hi, :].numpy())
        return {"best_layer": best_li, "best_head": best_hi,
                "test_results": eval_reg(te_labels, preds)}
    tr_acts = train["input_per_head_activation"]
    va_acts = val["input_per_head_activation"]
    te_acts = test["input_per_head_activation"]
    tr_labels = torch.tensor(train["labels"])
    va_labels = torch.tensor(val["labels"])
    te_labels = torch.tensor(test["labels"])

    # Select best head on val
    n_layers, n_heads = tr_acts.shape[1], tr_acts.shape[2]
    val_results = iti_directions(tr_acts, tr_labels, va_acts, va_labels)
    best_idx = np.unravel_index(val_results[:, :, 0].argmax(), val_results[:, :, 0].shape)

    # Evaluate best head on test
    from sklearn.linear_model import LogisticRegression
    li, hi = best_idx
    clf = LogisticRegression(random_state=0, max_iter=1000)
    clf.fit(tr_acts[:, li, hi, :].numpy(), tr_labels.numpy())
    probs = clf.predict_proba(te_acts[:, li, hi, :].numpy())[:, 1]
    test_metrics = eval_cls(te_labels.numpy(), probs)

    return {"best_layer": int(li), "best_head": int(hi),
            "val_auroc": float(val_results[li, hi, 0]),
            "test_results": test_metrics}


# --- 5. KB MLP — original: fixed mid layer + dev selects epoch ---
def run_kb_mlp(train, val, test, is_reg):
    if is_reg:
        # Use mid layer + Ridge for regression
        n_layers = train["input_last_token_hidden"].shape[1]
        mid_layer = n_layers // 2
        tr_acts = train["input_last_token_hidden"][:, mid_layer, :].float().numpy()
        te_acts = test["input_last_token_hidden"][:, mid_layer, :].float().numpy()
        tr_labels = np.array(train["labels"])
        te_labels = np.array(test["labels"])
        clf = Ridge(alpha=1.0)
        clf.fit(tr_acts, tr_labels)
        return {"layer": mid_layer, "test_results": eval_reg(te_labels, clf.predict(te_acts))}
    n_layers = train["input_last_token_hidden"].shape[1]
    mid_layer = n_layers // 2

    tr_acts = train["input_last_token_hidden"][:, mid_layer, :]
    va_acts = val["input_last_token_hidden"][:, mid_layer, :]
    te_acts = test["input_last_token_hidden"][:, mid_layer, :]
    tr_labels = torch.tensor(train["labels"])
    va_labels = torch.tensor(val["labels"])
    te_labels = torch.tensor(test["labels"])
    # Train with val for epoch selection, evaluate on test
    probs, preds = KBNet.train_and_eval(tr_acts, tr_labels, te_acts, te_labels,
                                         val_acts=va_acts, val_labels=va_labels)
    return {"layer": mid_layer, "test_results": eval_cls(te_labels.numpy(), probs, preds)}


# --- 6. LID — original: test selects layer (leaky), we use val ---
def run_lid(train, val, test, is_reg):
    tr_labels = np.array(train["labels"])
    va_labels = np.array(val["labels"])
    te_labels = np.array(test["labels"])
    n_layers = train["input_last_token_hidden"].shape[1]

    best_val_metric = -1.0
    best_layer = 0

    for layer in range(n_layers):
        tr_acts = train["input_last_token_hidden"][:, layer, :]
        va_acts = val["input_last_token_hidden"][:, layer, :]

        if is_reg:
            ref_acts = tr_acts
        else:
            ref_acts = tr_acts[tr_labels == 1]

        k = len(ref_acts) - 1
        if k < 2:
            continue

        lids = compute_lid(ref_acts, va_acts, k=k, hidden_dim=HIDDEN_DIM)

        if is_reg:
            m = abs(eval_reg(va_labels, lids)["spearman_r"])
        else:
            m = eval_scoring(va_labels, -lids)["auroc"]

        if m > best_val_metric:
            best_val_metric = m
            best_layer = layer

    # Final eval on test
    tr_acts = train["input_last_token_hidden"][:, best_layer, :]
    te_acts = test["input_last_token_hidden"][:, best_layer, :]
    if is_reg:
        ref_acts = tr_acts
    else:
        ref_acts = tr_acts[tr_labels == 1]
    k = len(ref_acts) - 1
    lids = compute_lid(ref_acts, te_acts, k=k, hidden_dim=HIDDEN_DIM)
    if is_reg:
        test_result = eval_reg(te_labels, lids)
    else:
        test_result = eval_scoring(te_labels, -lids)
    return {"best_layer": best_layer, "test_results": test_result}


# --- 7. Attn Satisfies — original: all layers, no selection ---
def run_attn_satisfies(train, val, test, is_reg):
    if is_reg:
        # Use max-over-positions flattened features + Ridge
        tr_feat = train["input_attn_value_norms"].float().max(dim=-1).values.reshape(len(train["labels"]), -1).numpy()
        te_feat = test["input_attn_value_norms"].float().max(dim=-1).values.reshape(len(test["labels"]), -1).numpy()
        sc = StandardScaler()
        tr_feat = sc.fit_transform(tr_feat)
        te_feat = sc.transform(te_feat)
        clf = Ridge(alpha=1.0)
        clf.fit(tr_feat, np.array(train["labels"]))
        return {"test_results": eval_reg(np.array(test["labels"]), clf.predict(te_feat))}
    tr_labels = torch.tensor(train["labels"])
    te_labels = torch.tensor(test["labels"])
    probs, preds = attention_satisfies_probe(
        train["input_attn_value_norms"], tr_labels,
        test["input_attn_value_norms"], te_labels)
    return {"test_results": eval_cls(te_labels.numpy(), probs, preds)}


# --- 8. LLM-Check — original: all layers scored, we use val to select ---
def run_llm_check(train, val, test, is_reg):
    va_labels = np.array(val["labels"])
    te_labels = np.array(test["labels"])
    n_layers = val["input_attn_stats"].shape[1]

    best_val_metric = -1.0
    best_layer = 0

    for layer in range(n_layers):
        scores = llm_check_score(val["input_attn_stats"], layer_num=layer)
        if is_reg:
            m = abs(eval_reg(va_labels, scores)["spearman_r"])
        else:
            m = eval_scoring(va_labels, scores)["auroc"]
        if m > best_val_metric:
            best_val_metric = m
            best_layer = layer

    scores = llm_check_score(test["input_attn_stats"], layer_num=best_layer)
    if is_reg:
        test_result = eval_reg(te_labels, scores)
    else:
        test_result = eval_scoring(te_labels, scores)
    return {"best_layer": best_layer, "test_results": test_result}


# --- 9. SEP — original: test selects range (leaky), we use val ---
def run_sep(train, val, test, is_reg):
    tr_labels = torch.tensor(train["labels"])
    va_labels = torch.tensor(val["labels"])
    te_labels = torch.tensor(test["labels"])

    if is_reg:
        n_layers = train["gen_last_token_hidden"].shape[1]
        best_val = -1.0
        best_layer = 0
        for layer in range(n_layers):
            X_tr = train["gen_last_token_hidden"][:, layer, :].float().numpy()
            X_va = val["gen_last_token_hidden"][:, layer, :].float().numpy()
            clf = Ridge(alpha=1.0)
            clf.fit(X_tr, tr_labels.numpy())
            m = abs(eval_reg(va_labels.numpy(), clf.predict(X_va))["spearman_r"])
            if m > best_val:
                best_val = m
                best_layer = layer
        X_tr = train["gen_last_token_hidden"][:, best_layer, :].float().numpy()
        X_te = test["gen_last_token_hidden"][:, best_layer, :].float().numpy()
        clf = Ridge(alpha=1.0)
        clf.fit(X_tr, tr_labels.numpy())
        return {"best_layer": best_layer,
                "test_results": eval_reg(te_labels.numpy(), clf.predict(X_te))}
    else:
        # Select range on val (train on train_sub, eval on val, pick best range)
        from sklearn.linear_model import LogisticRegression
        n_layers = train["gen_last_token_hidden"].shape[1]
        best_val_auroc = 0
        best_range = (0, 1)
        for start in range(n_layers):
            for end in range(start + 1, min(start + 6, n_layers + 1)):
                X_tr = train["gen_last_token_hidden"][:, start:end, :].float().reshape(len(tr_labels), -1).numpy()
                X_va = val["gen_last_token_hidden"][:, start:end, :].float().reshape(len(va_labels), -1).numpy()
                clf = LogisticRegression(max_iter=1000)
                clf.fit(X_tr, tr_labels.numpy())
                va_probs = clf.predict_proba(X_va)[:, 1]
                va_auroc = roc_auc_score(va_labels.numpy(), va_probs)
                if va_auroc > best_val_auroc:
                    best_val_auroc = va_auroc
                    best_range = (start, end)
        # Final eval on test
        X_tr = train["gen_last_token_hidden"][:, best_range[0]:best_range[1], :].float().reshape(len(tr_labels), -1).numpy()
        X_te = test["gen_last_token_hidden"][:, best_range[0]:best_range[1], :].float().reshape(len(te_labels), -1).numpy()
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_tr, tr_labels.numpy())
        probs = clf.predict_proba(X_te)[:, 1]
        return {"best_range": list(best_range),
                "test_results": eval_cls(te_labels.numpy(), probs)}


# --- 10. CoE — original: all layers, no training, no selection ---
def run_coe(train, val, test, is_reg):
    te_labels = np.array(test["labels"])
    scores = compute_coe_scores(test["gen_mean_pool_hidden"])
    results = {}
    for k, v in scores.items():
        results[k] = eval_reg(te_labels, v) if is_reg else eval_scoring(te_labels, v)
    return results


# --- 11. SeaKR — original: fixed layer 15, no selection ---
def run_seakr(train, val, test, is_reg):
    te_labels = np.array(test["labels"])
    scores = seakr_energy_score(test["gen_logit_stats_last"])
    return {"test_results": eval_reg(te_labels, scores) if is_reg else eval_scoring(te_labels, scores)}


# --- 12. STEP — original: fixed last layer + val-based early stopping ---
def run_step(train, val, test, is_reg):
    if is_reg:
        # Use last decoder layer + Ridge for regression
        tr_acts = train["gen_last_token_hidden"][:, -2, :].float().numpy()
        te_acts = test["gen_last_token_hidden"][:, -2, :].float().numpy()
        clf = Ridge(alpha=1.0)
        clf.fit(tr_acts, np.array(train["labels"]))
        return {"test_results": eval_reg(np.array(test["labels"]), clf.predict(te_acts))}
    tr_labels = torch.tensor(train["labels"])
    va_labels = torch.tensor(val["labels"])
    te_labels = torch.tensor(test["labels"])
    tr_acts = train["gen_last_token_hidden"][:, -2, :]  # last decoder layer
    va_acts = val["gen_last_token_hidden"][:, -2, :]
    te_acts = test["gen_last_token_hidden"][:, -2, :]
    probs, preds = STEPScorer.train_and_eval(
        tr_acts, tr_labels, te_acts, te_labels,
        val_acts=va_acts, val_labels=va_labels)
    return {"test_results": eval_cls(te_labels.numpy(), probs, preds)}


# ============================================================
# Main
# ============================================================
DATASETS = {
    "geometry_of_truth_cities": ("train_sub", "val_split", "val", False),
    "easy2hard_amc": ("train_sub", "val_split", "eval", True),
    "metatool_task1": ("train_sub", "val_split", "test", False),
    "retrievalqa": ("train_sub", "val_split", "test", False),
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


def load_existing_results():
    p = os.path.join(RESULTS_DIR, "all_results_v2.json")
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return {}


def save_results(r):
    with open(os.path.join(RESULTS_DIR, "all_results_v2.json"), "w") as f:
        json.dump(r, f, indent=2, default=str)


def main():
    all_results = load_existing_results()

    for dataset, (train_split, val_split, test_split, is_reg) in DATASETS.items():
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset} ({'regression' if is_reg else 'classification'})")
        print(f"{'='*60}")

        if dataset not in all_results:
            all_results[dataset] = {}
        ds_results = all_results[dataset]

        methods_to_run = {k: v for k, v in METHODS.items()
                         if k not in ds_results or "error" in ds_results.get(k, {})}
        if not methods_to_run:
            print("  All methods done, skipping.")
            continue

        train = load_split(dataset, train_split)
        val = load_split(dataset, val_split)
        test = load_split(dataset, test_split)
        print(f"  Train: {len(train['labels'])}, Val: {len(val['labels'])}, Test: {len(test['labels'])}")
        print(f"  Methods to run: {list(methods_to_run.keys())}")

        for name, fn in methods_to_run.items():
            print(f"\n  --- {name} ---", flush=True)
            try:
                result = fn(train, val, test, is_reg)
                ds_results[name] = result
                if "skipped" in result:
                    print(f"    Skipped: {result['skipped']}")
                elif "test_results" in result:
                    tr = result["test_results"]
                    info = ", ".join(f"{k}={result[k]}" for k in ["best_layer", "best_head", "best_range", "layer"] if k in result)
                    print(f"    {info}: {tr}")
                elif "coe_mag" in result:
                    for k, v in result.items():
                        if isinstance(v, dict):
                            print(f"    {k}: {v}")
                else:
                    print(f"    {result}")
            except Exception as e:
                print(f"    ERROR: {e}")
                traceback.print_exc()
                ds_results[name] = {"error": str(e)}
            save_results(all_results)

    print(f"\nAll done! Results at {RESULTS_DIR}/all_results_v2.json")


if __name__ == "__main__":
    main()
