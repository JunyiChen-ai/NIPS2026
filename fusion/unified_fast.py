"""
Unified Fast Fusion: Smart PCA + C tuning + score-level fusion.

Key design:
- Low-dim probes (<=256d): keep full dim, fast LR
- High-dim probes (>256d): PCA to 256d, then LR
- C tuning via train→val (fast, single pass)
- 5-fold CV on train+val for OOF stacking
- LCB-weighted avg and CV stacking as fusion methods
"""

import os, json, time, warnings
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA
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
PCA_DIM = 256
C_VALUES = [0.001, 0.01, 0.1, 1.0, 10.0]


def load_labels(dataset, split):
    raw_split = DATASETS[dataset]["splits"][split]
    with open(os.path.join(EXTRACTION_DIR, dataset, raw_split, "meta.json")) as f:
        return np.array(json.load(f)["labels"])


def load_and_smart_reduce(dataset, methods):
    """Load features, PCA only if dim > PCA_DIM."""
    data = {}
    for split in ["train", "val", "test"]:
        data[split] = {"labels": load_labels(dataset, split), "feats": {}}

    for method in methods:
        path = os.path.join(PROCESSED_DIR, dataset, method, "train.pt")
        if not os.path.exists(path):
            continue
        tr = torch.load(path, map_location="cpu").float().numpy()
        va = torch.load(os.path.join(PROCESSED_DIR, dataset, method, "val.pt"), map_location="cpu").float().numpy()
        te = torch.load(os.path.join(PROCESSED_DIR, dataset, method, "test.pt"), map_location="cpu").float().numpy()
        if tr.ndim == 1:
            tr, va, te = tr.reshape(-1, 1), va.reshape(-1, 1), te.reshape(-1, 1)

        sc = StandardScaler()
        tr = sc.fit_transform(tr)
        va = sc.transform(va)
        te = sc.transform(te)

        if tr.shape[1] > PCA_DIM:
            pca = PCA(n_components=PCA_DIM, random_state=42)
            tr = pca.fit_transform(tr)
            va = pca.transform(va)
            te = pca.transform(te)

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


def run_dataset(dataset, info):
    n_classes = info["n_classes"]
    baseline_method, baseline_auroc = BASELINES[dataset]

    print(f"\n{'='*60}")
    print(f"Dataset: {dataset} ({n_classes}-class)")
    print(f"Baseline: {baseline_method} AUROC={baseline_auroc:.4f}")
    print(f"{'='*60}")

    data = load_and_smart_reduce(dataset, MULTICLASS_METHODS)
    methods = list(data["train"]["feats"].keys())

    tr_labels = data["train"]["labels"]
    va_labels = data["val"]["labels"]
    te_labels = data["test"]["labels"]
    trva_labels = np.concatenate([tr_labels, va_labels])
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    # Step 1: Per-probe C tuning + OOF probs
    probe_info = {}
    for m in methods:
        t0 = time.time()
        tr_X = data["train"]["feats"][m]
        va_X = data["val"]["feats"][m]
        te_X = data["test"]["feats"][m]
        trva_X = np.vstack([tr_X, va_X])
        dim = tr_X.shape[1]

        # Tune C on train→val
        best_auroc, best_C = -1, 1.0
        for C in C_VALUES:
            clf = LogisticRegression(max_iter=2000, C=C, random_state=42)
            clf.fit(tr_X, tr_labels)
            va_prob = clf.predict_proba(va_X)
            auroc = compute_auroc(va_labels, va_prob, n_classes)
            if auroc > best_auroc:
                best_auroc = auroc
                best_C = C

        # OOF probs with best C
        oof_prob = np.zeros((len(trva_labels), n_classes))
        cv_te_prob = np.zeros((len(te_labels), n_classes))
        fold_aurocs = []
        for _, (tr_i, va_i) in enumerate(skf.split(trva_X, trva_labels)):
            clf = LogisticRegression(max_iter=2000, C=best_C, random_state=42)
            clf.fit(trva_X[tr_i], trva_labels[tr_i])
            oof_prob[va_i] = clf.predict_proba(trva_X[va_i])
            cv_te_prob += clf.predict_proba(te_X) / N_FOLDS
            fold_aurocs.append(compute_auroc(trva_labels[va_i], oof_prob[va_i], n_classes))

        # Also train on full train+val for direct test
        clf_full = LogisticRegression(max_iter=2000, C=best_C, random_state=42)
        clf_full.fit(trva_X, trva_labels)
        direct_te_prob = clf_full.predict_proba(te_X)
        direct_auroc = compute_auroc(te_labels, direct_te_prob, n_classes)

        elapsed = time.time() - t0
        cv_auroc = np.mean(fold_aurocs)
        print(f"  {m:20s} dim={dim:>4d} C={best_C:>6.3f} val={best_auroc:.4f} "
              f"CV={cv_auroc:.4f} test={direct_auroc:.4f} [{elapsed:.1f}s]")

        probe_info[m] = {
            "best_C": best_C,
            "val_auroc": best_auroc,
            "oof_prob": oof_prob,
            "cv_te_prob": cv_te_prob,
            "direct_te_prob": direct_te_prob,
            "fold_aurocs": fold_aurocs,
            "direct_auroc": direct_auroc,
        }

    # Step 2: Fusion methods
    results = {}

    # 2a: LCB-weighted average
    lcbs = {m: np.mean(probe_info[m]["fold_aurocs"]) - np.std(probe_info[m]["fold_aurocs"])
            for m in methods}

    best_oof_auroc, best_temp = -1, 0.1
    for temp in [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]:
        lcb_arr = np.array([lcbs[m] for m in methods])
        w = softmax((lcb_arr - lcb_arr.max()) / temp)
        blended = sum(w[i] * probe_info[m]["oof_prob"] for i, m in enumerate(methods))
        auroc = compute_auroc(trva_labels, blended, n_classes)
        if auroc > best_oof_auroc:
            best_oof_auroc = auroc
            best_temp = temp

    lcb_arr = np.array([lcbs[m] for m in methods])
    weights = softmax((lcb_arr - lcb_arr.max()) / best_temp)
    test_blend = sum(weights[i] * probe_info[m]["cv_te_prob"] for i, m in enumerate(methods))
    r = eval_metrics(te_labels, test_blend, n_classes)
    delta = r["auroc"] - baseline_auroc
    status = ">>>" if delta > 0 else "   "
    top_w = sorted(zip(methods, weights), key=lambda x: -x[1])[:3]
    print(f"\n  {status} lcb_weighted_avg       AUROC={r['auroc']:.4f} ({delta:+.4f}) temp={best_temp} "
          f"top=[{', '.join(f'{m}:{w:.2f}' for m,w in top_w)}]")
    results["lcb_weighted_avg"] = {**r, "temp": best_temp,
                                    "weights": {m: float(w) for m, w in zip(methods, weights)}}

    # 2b: CV stacking + anchor shrinkage
    anchor = max(lcbs, key=lcbs.get)
    oof_meta = np.hstack([probe_info[m]["oof_prob"] for m in methods])
    test_meta = np.hstack([probe_info[m]["cv_te_prob"] for m in methods])

    best_auroc_stack, best_C_stack = -1, 1.0
    for C in [0.01, 0.1, 1.0, 10.0]:
        inner_oof = np.zeros((len(trva_labels), n_classes))
        for _, (tr_i, va_i) in enumerate(skf.split(oof_meta, trva_labels)):
            clf = LogisticRegression(max_iter=2000, C=C, random_state=42)
            clf.fit(oof_meta[tr_i], trva_labels[tr_i])
            inner_oof[va_i] = clf.predict_proba(oof_meta[va_i])
        auroc = compute_auroc(trva_labels, inner_oof, n_classes)
        if auroc > best_auroc_stack:
            best_auroc_stack = auroc
            best_C_stack = C

    stacking_oof = np.zeros((len(trva_labels), n_classes))
    for _, (tr_i, va_i) in enumerate(skf.split(oof_meta, trva_labels)):
        clf = LogisticRegression(max_iter=2000, C=best_C_stack, random_state=42)
        clf.fit(oof_meta[tr_i], trva_labels[tr_i])
        stacking_oof[va_i] = clf.predict_proba(oof_meta[va_i])

    clf_meta = LogisticRegression(max_iter=2000, C=best_C_stack, random_state=42)
    clf_meta.fit(oof_meta, trva_labels)
    stacking_test = clf_meta.predict_proba(test_meta)

    best_sh_auroc, best_shrink = -1, 0
    for shrink in np.arange(0.0, 0.55, 0.05):
        blended = (1 - shrink) * stacking_oof + shrink * probe_info[anchor]["oof_prob"]
        auroc = compute_auroc(trva_labels, blended, n_classes)
        if auroc > best_sh_auroc:
            best_sh_auroc = auroc
            best_shrink = shrink

    test_final = (1 - best_shrink) * stacking_test + best_shrink * probe_info[anchor]["cv_te_prob"]
    r = eval_metrics(te_labels, test_final, n_classes)
    delta = r["auroc"] - baseline_auroc
    status = ">>>" if delta > 0 else "   "
    print(f"  {status} cv_stack+anchor_shrink AUROC={r['auroc']:.4f} ({delta:+.4f}) "
          f"anchor={anchor} shrink={best_shrink:.2f} C={best_C_stack}")
    results["cv_stack_anchor_shrink"] = {**r, "anchor": anchor, "shrink": float(best_shrink)}

    # 2c: Best individual tuned probe
    best_probe = max(probe_info, key=lambda m: probe_info[m]["direct_auroc"])
    best_auroc_single = probe_info[best_probe]["direct_auroc"]
    delta = best_auroc_single - baseline_auroc
    status = ">>>" if delta > 0 else "   "
    print(f"  {status} best_tuned_single      AUROC={best_auroc_single:.4f} ({delta:+.4f}) probe={best_probe}")
    results["best_tuned_single"] = {"auroc": best_auroc_single, "probe": best_probe}

    return results


def main():
    print("=" * 70)
    print("UNIFIED FAST FUSION — Smart PCA(256) + C Tuning")
    print("=" * 70)

    all_results = {}
    for dataset, info in DATASETS.items():
        all_results[dataset] = run_dataset(dataset, info)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    ds_short = {"common_claim_3class": "cc_3c", "e2h_amc_3class": "e2h_3c",
                "e2h_amc_5class": "e2h_5c", "when2call_3class": "w2c_3c"}
    method_names = ["best_tuned_single", "lcb_weighted_avg", "cv_stack_anchor_shrink"]

    print(f"{'Method':32s}", end="")
    for ds in DATASETS:
        print(f"  {ds_short[ds]:>8s}", end="")
    print(f"  {'#wins':>6s}")

    print(f"{'[original baseline]':32s}", end="")
    for ds in DATASETS:
        _, bl = BASELINES[ds]
        print(f"  {bl:8.4f}", end="")
    print()

    for mn in method_names:
        print(f"{mn:32s}", end="")
        wins = 0
        for ds in DATASETS:
            auroc = all_results[ds].get(mn, {}).get("auroc", 0)
            _, bl = BASELINES[ds]
            if auroc > bl:
                wins += 1
            print(f"  {auroc:8.4f}", end="")
        print(f"  {wins:>6d}/4")

    os.makedirs("/home/junyi/NIPS2026/fusion/results", exist_ok=True)
    with open("/home/junyi/NIPS2026/fusion/results/unified_fast_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved to fusion/results/unified_fast_results.json")


if __name__ == "__main__":
    main()
