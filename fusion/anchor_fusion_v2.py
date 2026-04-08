"""
Anchor Fusion v2: Score-level fusion WITHOUT PCA.
Each probe trains its own LR on its own full-dimensional features.
We only fuse the probability outputs — no feature-level operations.

Key fix from v1: PCA was degrading probe performance, especially on when2call.
"""

import os, json, time, warnings
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from scipy.optimize import minimize
from scipy.special import softmax

warnings.filterwarnings("ignore")

PROCESSED_DIR = "/home/junyi/NIPS2026/reproduce/processed_features"
EXTRACTION_DIR = "/home/junyi/NIPS2026/extraction/features"

MULTICLASS_METHODS = ["lr_probe", "pca_lr", "iti", "kb_mlp", "attn_satisfies", "sep", "step"]

DATASETS = {
    "common_claim_3class": {"n_classes": 3, "splits": {"train": "train", "val": "val", "test": "test"}},
    "e2h_amc_3class":      {"n_classes": 3, "splits": {"train": "train_sub", "val": "val_split", "test": "eval"}},
    "e2h_amc_5class":      {"n_classes": 5, "splits": {"train": "train_sub", "val": "val_split", "test": "eval"}},
    "when2call_3class":     {"n_classes": 3, "splits": {"train": "train", "val": "val", "test": "test"}},
}

BASELINES = {
    "common_claim_3class": ("pca_lr", 0.7576),
    "e2h_amc_3class":      ("pca_lr", 0.8934),
    "e2h_amc_5class":      ("kb_mlp", 0.8752),
    "when2call_3class":     ("lr_probe", 0.8741),
}

N_FOLDS = 5


def load_labels(dataset, split):
    raw_split = DATASETS[dataset]["splits"][split]
    with open(os.path.join(EXTRACTION_DIR, dataset, raw_split, "meta.json")) as f:
        return np.array(json.load(f)["labels"])


def load_raw_features(dataset, methods):
    """Load features WITHOUT PCA. Standardize only."""
    data = {}
    for split in ["train", "val", "test"]:
        data[split] = {"labels": load_labels(dataset, split), "feats": {}}

    scalers = {}
    for method in methods:
        train_path = os.path.join(PROCESSED_DIR, dataset, method, "train.pt")
        if not os.path.exists(train_path):
            continue
        tr = torch.load(train_path, map_location="cpu").float().numpy()
        va = torch.load(os.path.join(PROCESSED_DIR, dataset, method, "val.pt"), map_location="cpu").float().numpy()
        te = torch.load(os.path.join(PROCESSED_DIR, dataset, method, "test.pt"), map_location="cpu").float().numpy()
        if tr.ndim == 1:
            tr, va, te = tr.reshape(-1, 1), va.reshape(-1, 1), te.reshape(-1, 1)
        sc = StandardScaler()
        tr = sc.fit_transform(tr)
        va = sc.transform(va)
        te = sc.transform(te)
        scalers[method] = sc
        data["train"]["feats"][method] = tr
        data["val"]["feats"][method] = va
        data["test"]["feats"][method] = te
    return data


def compute_auroc(y_true, y_prob, n_classes):
    if n_classes == 2:
        return roc_auc_score(y_true, y_prob[:, 1])
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))
    return roc_auc_score(y_bin, y_prob, average="macro", multi_class="ovr")


def eval_metrics(y_true, y_prob, n_classes):
    y_pred = y_prob.argmax(axis=1)
    return {
        "auroc": compute_auroc(y_true, y_prob, n_classes),
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
    }


def get_cv_probs_nopca(data, n_classes, n_folds=N_FOLDS):
    """CV probs from full-dim features. Each probe trains its own LR."""
    methods = list(data["train"]["feats"].keys())
    all_labels = np.concatenate([data["train"]["labels"], data["val"]["labels"]])
    all_feats = {m: np.vstack([data["train"]["feats"][m], data["val"]["feats"][m]]) for m in methods}
    n_total = len(all_labels)
    n_test = len(data["test"]["labels"])

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    oof_probs = {m: np.zeros((n_total, n_classes)) for m in methods}
    test_probs = {m: np.zeros((n_test, n_classes)) for m in methods}
    fold_aurocs = {m: [] for m in methods}

    for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(all_feats[methods[0]], all_labels)):
        for m in methods:
            clf = LogisticRegression(max_iter=3000, random_state=42, C=1.0, solver="lbfgs")
            clf.fit(all_feats[m][tr_idx], all_labels[tr_idx])
            oof_probs[m][va_idx] = clf.predict_proba(all_feats[m][va_idx])
            test_probs[m] += clf.predict_proba(data["test"]["feats"][m]) / n_folds
            fold_aurocs[m].append(compute_auroc(all_labels[va_idx], oof_probs[m][va_idx], n_classes))

    return methods, all_labels, oof_probs, test_probs, fold_aurocs


# ============================================================
# Method B: LCB-weighted average (best from v1, now without PCA)
# ============================================================
def lcb_weighted_avg(data, n_classes, temps=[0.02, 0.05, 0.1, 0.2, 0.5]):
    methods, all_labels, oof_probs, test_probs, fold_aurocs = get_cv_probs_nopca(data, n_classes)
    lcbs = {m: np.mean(fold_aurocs[m]) - 1.0 * np.std(fold_aurocs[m]) for m in methods}

    best_auroc, best_temp = -1, 0.1
    for temp in temps:
        lcb_arr = np.array([lcbs[m] for m in methods])
        weights = softmax((lcb_arr - lcb_arr.max()) / temp)
        blended = sum(weights[i] * oof_probs[m] for i, m in enumerate(methods))
        auroc = compute_auroc(all_labels, blended, n_classes)
        if auroc > best_auroc:
            best_auroc = auroc
            best_temp = temp

    lcb_arr = np.array([lcbs[m] for m in methods])
    weights = softmax((lcb_arr - lcb_arr.max()) / best_temp)
    test_blend = sum(weights[i] * test_probs[m] for i, m in enumerate(methods))

    r = eval_metrics(data["test"]["labels"], test_blend, n_classes)
    r["temp"] = best_temp
    r["weights"] = {m: float(w) for m, w in zip(methods, weights)}
    r["lcbs"] = {m: float(v) for m, v in lcbs.items()}
    return r


# ============================================================
# Method E: CV Stacking with anchor shrinkage (no PCA)
# ============================================================
def cv_stacking_anchor_shrink(data, n_classes):
    methods, all_labels, oof_probs, test_probs, fold_aurocs = get_cv_probs_nopca(data, n_classes)

    # Anchor = best LCB
    lcbs = {m: np.mean(fold_aurocs[m]) - 1.0 * np.std(fold_aurocs[m]) for m in methods}
    anchor = max(lcbs, key=lcbs.get)
    n_total = len(all_labels)

    oof_meta = np.hstack([oof_probs[m] for m in methods])
    test_meta = np.hstack([test_probs[m] for m in methods])

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    # Find best C
    best_auroc, best_C = -1, 1.0
    for C in [0.01, 0.1, 1.0, 10.0]:
        inner_oof = np.zeros((n_total, n_classes))
        for _, (tr_i, va_i) in enumerate(skf.split(oof_meta, all_labels)):
            clf = LogisticRegression(max_iter=2000, random_state=42, C=C)
            clf.fit(oof_meta[tr_i], all_labels[tr_i])
            inner_oof[va_i] = clf.predict_proba(oof_meta[va_i])
        auroc = compute_auroc(all_labels, inner_oof, n_classes)
        if auroc > best_auroc:
            best_auroc = auroc
            best_C = C

    # Get stacking OOF and test preds
    stacking_oof = np.zeros((n_total, n_classes))
    for _, (tr_i, va_i) in enumerate(skf.split(oof_meta, all_labels)):
        clf = LogisticRegression(max_iter=2000, random_state=42, C=best_C)
        clf.fit(oof_meta[tr_i], all_labels[tr_i])
        stacking_oof[va_i] = clf.predict_proba(oof_meta[va_i])

    clf_final = LogisticRegression(max_iter=2000, random_state=42, C=best_C)
    clf_final.fit(oof_meta, all_labels)
    stacking_test = clf_final.predict_proba(test_meta)

    # Find best shrink
    best_auroc_final, best_shrink = -1, 0
    for shrink in np.arange(0.0, 0.55, 0.05):
        blended = (1 - shrink) * stacking_oof + shrink * oof_probs[anchor]
        auroc = compute_auroc(all_labels, blended, n_classes)
        if auroc > best_auroc_final:
            best_auroc_final = auroc
            best_shrink = shrink

    test_final = (1 - best_shrink) * stacking_test + best_shrink * test_probs[anchor]
    r = eval_metrics(data["test"]["labels"], test_final, n_classes)
    r["anchor"] = anchor
    r["shrink"] = float(best_shrink)
    r["stacking_C"] = best_C
    return r


# ============================================================
# Method F: Per-probe individual LR test AUROC (sanity check)
# ============================================================
def per_probe_lr(data, n_classes):
    """Train LR on train+val (full dim), evaluate on test. For comparison."""
    methods = list(data["train"]["feats"].keys())
    all_labels = np.concatenate([data["train"]["labels"], data["val"]["labels"]])
    results = {}
    for m in methods:
        X = np.vstack([data["train"]["feats"][m], data["val"]["feats"][m]])
        clf = LogisticRegression(max_iter=3000, random_state=42, C=1.0)
        clf.fit(X, all_labels)
        te_prob = clf.predict_proba(data["test"]["feats"][m])
        auroc = compute_auroc(data["test"]["labels"], te_prob, n_classes)
        results[m] = auroc
    return results


def main():
    print("=" * 70)
    print("ANCHOR FUSION v2 — No PCA, Full-Dimensional Score-Level Fusion")
    print("=" * 70)

    all_results = {}
    fusion_methods = [
        ("B: lcb_weighted_avg", lcb_weighted_avg),
        ("E: cv_stack_anchor_shrink", cv_stacking_anchor_shrink),
    ]

    for dataset, info in DATASETS.items():
        n_classes = info["n_classes"]
        baseline_method, baseline_auroc = BASELINES[dataset]

        print(f"\n{'='*60}")
        print(f"Dataset: {dataset} ({n_classes}-class)")
        print(f"Baseline: {baseline_method} AUROC={baseline_auroc:.4f}")
        print(f"{'='*60}")

        t0 = time.time()
        data = load_raw_features(dataset, MULTICLASS_METHODS)
        methods_loaded = list(data["train"]["feats"].keys())
        dims = {m: data["train"]["feats"][m].shape[1] for m in methods_loaded}
        print(f"Loaded {len(methods_loaded)} methods (no PCA): {dims}  ({time.time()-t0:.1f}s)")

        # Sanity: per-probe LR AUROC
        probe_aurocs = per_probe_lr(data, n_classes)
        print(f"Per-probe LR (train+val → test):")
        for m in sorted(probe_aurocs, key=probe_aurocs.get, reverse=True):
            marker = "*" if probe_aurocs[m] >= baseline_auroc else " "
            print(f"  {marker} {m:20s} AUROC={probe_aurocs[m]:.4f}")

        ds_results = {"per_probe": probe_aurocs}
        for name, fn in fusion_methods:
            t1 = time.time()
            try:
                r = fn(data, n_classes)
                delta = r["auroc"] - baseline_auroc
                status = ">>>" if delta > 0 else "   "
                extra = ""
                if "weights" in r:
                    top_w = sorted(r["weights"].items(), key=lambda x: -x[1])[:3]
                    extra += f" top_w={[(m,f'{w:.2f}') for m,w in top_w]}"
                if "anchor" in r:
                    extra += f" anchor={r['anchor']}"
                if "shrink" in r:
                    extra += f" shrink={r['shrink']}"
                print(f"  {status} {name:30s}  AUROC={r['auroc']:.4f} ({delta:+.4f})  "
                      f"Acc={r['accuracy']:.4f}  F1={r['f1_macro']:.4f}  [{time.time()-t1:.1f}s]{extra}")
                ds_results[name] = r
            except Exception as e:
                print(f"      {name:30s}  ERROR: {e}")
                import traceback; traceback.print_exc()

        all_results[dataset] = ds_results

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    ds_short = {"common_claim_3class": "cc_3c", "e2h_amc_3class": "e2h_3c",
                "e2h_amc_5class": "e2h_5c", "when2call_3class": "w2c_3c"}
    print(f"{'Method':32s}", end="")
    for ds in DATASETS:
        print(f"  {ds_short[ds]:>8s}", end="")
    print(f"  {'#wins':>6s}")

    print(f"{'[best single probe]':32s}", end="")
    for ds in DATASETS:
        _, bl = BASELINES[ds]
        print(f"  {bl:8.4f}", end="")
    print()

    for method_name, _ in fusion_methods:
        print(f"{method_name:32s}", end="")
        wins = 0
        for ds in DATASETS:
            if ds in all_results and method_name in all_results[ds]:
                auroc = all_results[ds][method_name]["auroc"]
                _, bl = BASELINES[ds]
                if auroc > bl:
                    wins += 1
                print(f"  {auroc:8.4f}", end="")
            else:
                print(f"  {'N/A':>8s}", end="")
        print(f"  {wins:>6d}/4")

    os.makedirs("/home/junyi/NIPS2026/fusion/results", exist_ok=True)
    with open("/home/junyi/NIPS2026/fusion/results/anchor_v2_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved to fusion/results/anchor_v2_results.json")


if __name__ == "__main__":
    main()
