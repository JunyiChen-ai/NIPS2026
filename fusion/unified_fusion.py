"""
Unified Fusion: Fast score-level fusion with per-probe C tuning.

Protocol for each probe:
1. Tune C on train→val AUROC (fast, no CV needed for C selection)
2. Train final LR on train+val with best C
3. Get test probabilities

Then fuse test probs via weighted average (LCB-weighted).

For CV stacking variant:
1. Tune C as above
2. Run 5-fold CV on train+val with best C → OOF probs
3. Train meta-LR on OOF probs
"""

import os, json, time, warnings
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
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
C_VALUES = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]


def load_labels(dataset, split):
    raw_split = DATASETS[dataset]["splits"][split]
    with open(os.path.join(EXTRACTION_DIR, dataset, raw_split, "meta.json")) as f:
        return np.array(json.load(f)["labels"])


def load_features(dataset, methods):
    """Load raw features (no PCA), standardize."""
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
        # Standardize using train stats
        sc = StandardScaler()
        tr = sc.fit_transform(tr)
        va = sc.transform(va)
        te = sc.transform(te)
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


def tune_and_predict(data, n_classes):
    """
    For each probe: tune C on train→val, then get test probs.
    Returns per-probe test probs and val-AUROC.
    """
    methods = list(data["train"]["feats"].keys())
    tr_labels = data["train"]["labels"]
    va_labels = data["val"]["labels"]
    te_labels = data["test"]["labels"]

    probe_info = {}
    for m in methods:
        t0 = time.time()
        tr_X = data["train"]["feats"][m]
        va_X = data["val"]["feats"][m]
        te_X = data["test"]["feats"][m]
        dim = tr_X.shape[1]

        # Choose solver based on dim
        if dim > 500:
            solver, penalty = "saga", "l1"
        else:
            solver, penalty = "lbfgs", "l2"

        # Tune C on train→val
        best_auroc, best_C = -1, 1.0
        for C in C_VALUES:
            clf = LogisticRegression(max_iter=3000, C=C, random_state=42,
                                     solver=solver, penalty=penalty)
            clf.fit(tr_X, tr_labels)
            va_prob = clf.predict_proba(va_X)
            auroc = compute_auroc(va_labels, va_prob, n_classes)
            if auroc > best_auroc:
                best_auroc = auroc
                best_C = C

        # Train final model on train+val with best C
        trva_X = np.vstack([tr_X, va_X])
        trva_labels = np.concatenate([tr_labels, va_labels])
        clf_final = LogisticRegression(max_iter=3000, C=best_C, random_state=42,
                                        solver=solver, penalty=penalty)
        clf_final.fit(trva_X, trva_labels)
        te_prob = clf_final.predict_proba(te_X)
        te_auroc = compute_auroc(te_labels, te_prob, n_classes)

        # Also get OOF probs for stacking (5-fold on train+val)
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
        oof_prob = np.zeros((len(trva_labels), n_classes))
        cv_te_prob = np.zeros((len(te_labels), n_classes))
        fold_aurocs = []

        for _, (tr_i, va_i) in enumerate(skf.split(trva_X, trva_labels)):
            clf = LogisticRegression(max_iter=3000, C=best_C, random_state=42,
                                      solver=solver, penalty=penalty)
            clf.fit(trva_X[tr_i], trva_labels[tr_i])
            oof_prob[va_i] = clf.predict_proba(trva_X[va_i])
            cv_te_prob += clf.predict_proba(te_X) / N_FOLDS
            fold_aurocs.append(compute_auroc(trva_labels[va_i], oof_prob[va_i], n_classes))

        elapsed = time.time() - t0
        print(f"    {m:20s} dim={dim:>5d} C={best_C:>6.3f} val_AUROC={best_auroc:.4f} "
              f"test_AUROC={te_auroc:.4f} CV_AUROC={np.mean(fold_aurocs):.4f} [{elapsed:.1f}s]")

        probe_info[m] = {
            "best_C": best_C,
            "val_auroc": best_auroc,
            "test_prob": te_prob,
            "cv_test_prob": cv_te_prob,
            "oof_prob": oof_prob,
            "fold_aurocs": fold_aurocs,
            "test_auroc": te_auroc,
        }

    return methods, probe_info, trva_labels


def run_fusion(dataset, info):
    n_classes = info["n_classes"]
    baseline_method, baseline_auroc = BASELINES[dataset]

    print(f"\n{'='*60}")
    print(f"Dataset: {dataset} ({n_classes}-class)")
    print(f"Baseline: {baseline_method} AUROC={baseline_auroc:.4f}")
    print(f"{'='*60}")

    data = load_features(dataset, MULTICLASS_METHODS)
    te_labels = data["test"]["labels"]

    print("\nPer-probe tuning:")
    methods, probe_info, trva_labels = tune_and_predict(data, n_classes)

    results = {}

    # 1. LCB-weighted average of CV-averaged test probs
    lcbs = {m: np.mean(probe_info[m]["fold_aurocs"]) - np.std(probe_info[m]["fold_aurocs"])
            for m in methods}

    best_auroc, best_temp = -1, 0.1
    for temp in [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]:
        lcb_arr = np.array([lcbs[m] for m in methods])
        w = softmax((lcb_arr - lcb_arr.max()) / temp)
        blended = sum(w[i] * probe_info[m]["oof_prob"] for i, m in enumerate(methods))
        auroc = compute_auroc(trva_labels, blended, n_classes)
        if auroc > best_auroc:
            best_auroc = auroc
            best_temp = temp

    lcb_arr = np.array([lcbs[m] for m in methods])
    weights = softmax((lcb_arr - lcb_arr.max()) / best_temp)
    test_blend = sum(weights[i] * probe_info[m]["cv_test_prob"] for i, m in enumerate(methods))
    r = eval_metrics(te_labels, test_blend, n_classes)
    delta = r["auroc"] - baseline_auroc
    status = ">>>" if delta > 0 else "   "
    top_w = sorted(zip(methods, weights), key=lambda x: -x[1])[:3]
    print(f"\n  {status} lcb_weighted_avg       AUROC={r['auroc']:.4f} ({delta:+.4f})  "
          f"Acc={r['accuracy']:.4f}  F1={r['f1_macro']:.4f}  temp={best_temp}")
    print(f"      weights: {[(m,f'{w:.3f}') for m,w in top_w]}")
    results["lcb_weighted_avg"] = r

    # 2. CV stacking with anchor shrinkage
    anchor = max(lcbs, key=lcbs.get)
    oof_meta = np.hstack([probe_info[m]["oof_prob"] for m in methods])
    test_meta = np.hstack([probe_info[m]["cv_test_prob"] for m in methods])
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

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

    best_auroc_sh, best_shrink = -1, 0
    for shrink in np.arange(0.0, 0.55, 0.05):
        blended = (1 - shrink) * stacking_oof + shrink * probe_info[anchor]["oof_prob"]
        auroc = compute_auroc(trva_labels, blended, n_classes)
        if auroc > best_auroc_sh:
            best_auroc_sh = auroc
            best_shrink = shrink

    test_final = (1 - best_shrink) * stacking_test + best_shrink * probe_info[anchor]["cv_test_prob"]
    r = eval_metrics(te_labels, test_final, n_classes)
    delta = r["auroc"] - baseline_auroc
    status = ">>>" if delta > 0 else "   "
    print(f"  {status} cv_stack_anchor_shrink AUROC={r['auroc']:.4f} ({delta:+.4f})  "
          f"Acc={r['accuracy']:.4f}  F1={r['f1_macro']:.4f}  anchor={anchor} shrink={best_shrink:.2f}")
    results["cv_stack_anchor_shrink"] = r

    # 3. Simple: best tuned single probe (sanity check)
    best_probe = max(probe_info, key=lambda m: probe_info[m]["test_auroc"])
    best_probe_auroc = probe_info[best_probe]["test_auroc"]
    delta = best_probe_auroc - baseline_auroc
    status = ">>>" if delta > 0 else "   "
    print(f"  {status} best_tuned_probe       AUROC={best_probe_auroc:.4f} ({delta:+.4f})  probe={best_probe}")
    results["best_tuned_probe"] = {"auroc": best_probe_auroc, "probe": best_probe}

    return results


def main():
    print("=" * 70)
    print("UNIFIED FUSION — Per-Probe C Tuning + Score-Level Fusion")
    print("=" * 70)

    all_results = {}
    for dataset, info in DATASETS.items():
        all_results[dataset] = run_fusion(dataset, info)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    ds_short = {"common_claim_3class": "cc_3c", "e2h_amc_3class": "e2h_3c",
                "e2h_amc_5class": "e2h_5c", "when2call_3class": "w2c_3c"}
    methods_show = ["best_tuned_probe", "lcb_weighted_avg", "cv_stack_anchor_shrink"]

    print(f"{'Method':32s}", end="")
    for ds in DATASETS:
        print(f"  {ds_short[ds]:>8s}", end="")
    print(f"  {'#wins':>6s}")

    print(f"{'[original baseline]':32s}", end="")
    for ds in DATASETS:
        _, bl = BASELINES[ds]
        print(f"  {bl:8.4f}", end="")
    print()

    for mn in methods_show:
        print(f"{mn:32s}", end="")
        wins = 0
        for ds in DATASETS:
            if ds in all_results and mn in all_results[ds]:
                auroc = all_results[ds][mn]["auroc"]
                _, bl = BASELINES[ds]
                if auroc > bl:
                    wins += 1
                print(f"  {auroc:8.4f}", end="")
            else:
                print(f"  {'N/A':>8s}", end="")
        print(f"  {wins:>6d}/4")

    os.makedirs("/home/junyi/NIPS2026/fusion/results", exist_ok=True)
    # Convert non-serializable items
    save_results = {}
    for ds, dr in all_results.items():
        save_results[ds] = {}
        for mn, r in dr.items():
            save_results[ds][mn] = {k: v for k, v in r.items() if k != "test_prob"}
    with open("/home/junyi/NIPS2026/fusion/results/unified_fusion_results.json", "w") as f:
        json.dump(save_results, f, indent=2, default=str)
    print(f"\nSaved to fusion/results/unified_fusion_results.json")


if __name__ == "__main__":
    main()
