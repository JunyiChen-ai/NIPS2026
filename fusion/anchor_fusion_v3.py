"""
Anchor Fusion v3: Score-level fusion with per-probe C tuning.

Key insight: Don't PCA features. Instead, tune regularization (C) per probe
via inner CV to handle the dimensionality variance (50d to 17920d).
Use solver='saga' with L1 for high-dim probes to speed up.
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


def load_labels(dataset, split):
    raw_split = DATASETS[dataset]["splits"][split]
    with open(os.path.join(EXTRACTION_DIR, dataset, raw_split, "meta.json")) as f:
        return np.array(json.load(f)["labels"])


def load_raw_features(dataset, methods):
    data = {}
    for split in ["train", "val", "test"]:
        data[split] = {"labels": load_labels(dataset, split), "feats": {}}
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


def get_optimal_lr(X, y, n_classes, C_values=[0.001, 0.01, 0.1, 1.0, 10.0], n_folds=3):
    """Find best C for LR via quick 3-fold CV."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    dim = X.shape[1]
    # For high-dim: use saga+l1 for speed; for low-dim: lbfgs+l2
    if dim > 500:
        solver, penalty = "saga", "l1"
    else:
        solver, penalty = "lbfgs", "l2"

    best_auroc, best_C = -1, 1.0
    for C in C_values:
        oof = np.zeros((len(y), n_classes))
        for tr_i, va_i in skf.split(X, y):
            clf = LogisticRegression(max_iter=2000, C=C, random_state=42,
                                     solver=solver, penalty=penalty)
            clf.fit(X[tr_i], y[tr_i])
            oof[va_i] = clf.predict_proba(X[va_i])
        auroc = compute_auroc(y, oof, n_classes)
        if auroc > best_auroc:
            best_auroc = auroc
            best_C = C
    return best_C, solver, penalty


def get_cv_probs_tuned(data, n_classes, n_folds=N_FOLDS):
    """CV probs with per-probe C tuning."""
    methods = list(data["train"]["feats"].keys())
    all_labels = np.concatenate([data["train"]["labels"], data["val"]["labels"]])
    all_feats = {m: np.vstack([data["train"]["feats"][m], data["val"]["feats"][m]]) for m in methods}
    n_total = len(all_labels)
    n_test = len(data["test"]["labels"])

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    oof_probs = {m: np.zeros((n_total, n_classes)) for m in methods}
    test_probs = {m: np.zeros((n_test, n_classes)) for m in methods}
    fold_aurocs = {m: [] for m in methods}
    best_Cs = {}

    for m in methods:
        t0 = time.time()
        X = all_feats[m]
        dim = X.shape[1]

        # Tune C per probe
        best_C, solver, penalty = get_optimal_lr(X, all_labels, n_classes)
        best_Cs[m] = best_C

        # Generate OOF probs with tuned C
        for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(X, all_labels)):
            clf = LogisticRegression(max_iter=3000, C=best_C, random_state=42,
                                     solver=solver, penalty=penalty)
            clf.fit(X[tr_idx], all_labels[tr_idx])
            oof_probs[m][va_idx] = clf.predict_proba(X[va_idx])
            test_probs[m] += clf.predict_proba(data["test"]["feats"][m]) / n_folds
            fold_aurocs[m].append(compute_auroc(all_labels[va_idx], oof_probs[m][va_idx], n_classes))

        mean_auroc = np.mean(fold_aurocs[m])
        print(f"    {m:20s} dim={dim:>5d} C={best_C:>6.3f} solver={solver:>5s} "
              f"CV_AUROC={mean_auroc:.4f}  [{time.time()-t0:.1f}s]")

    return methods, all_labels, oof_probs, test_probs, fold_aurocs, best_Cs


def lcb_weighted_avg(methods, all_labels, oof_probs, test_probs, fold_aurocs, n_classes, test_labels):
    lcbs = {m: np.mean(fold_aurocs[m]) - 1.0 * np.std(fold_aurocs[m]) for m in methods}
    best_auroc, best_temp = -1, 0.1
    for temp in [0.02, 0.05, 0.1, 0.2, 0.5]:
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
    r = eval_metrics(test_labels, test_blend, n_classes)
    r["temp"] = best_temp
    r["weights"] = {m: float(w) for m, w in zip(methods, weights)}
    return r


def cv_stacking_shrink(methods, all_labels, oof_probs, test_probs, fold_aurocs, n_classes, test_labels):
    lcbs = {m: np.mean(fold_aurocs[m]) - 1.0 * np.std(fold_aurocs[m]) for m in methods}
    anchor = max(lcbs, key=lcbs.get)
    n_total = len(all_labels)

    oof_meta = np.hstack([oof_probs[m] for m in methods])
    test_meta = np.hstack([test_probs[m] for m in methods])

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

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

    stacking_oof = np.zeros((n_total, n_classes))
    for _, (tr_i, va_i) in enumerate(skf.split(oof_meta, all_labels)):
        clf = LogisticRegression(max_iter=2000, random_state=42, C=best_C)
        clf.fit(oof_meta[tr_i], all_labels[tr_i])
        stacking_oof[va_i] = clf.predict_proba(oof_meta[va_i])

    clf_final = LogisticRegression(max_iter=2000, random_state=42, C=best_C)
    clf_final.fit(oof_meta, all_labels)
    stacking_test = clf_final.predict_proba(test_meta)

    best_auroc_final, best_shrink = -1, 0
    for shrink in np.arange(0.0, 0.55, 0.05):
        blended = (1 - shrink) * stacking_oof + shrink * oof_probs[anchor]
        auroc = compute_auroc(all_labels, blended, n_classes)
        if auroc > best_auroc_final:
            best_auroc_final = auroc
            best_shrink = shrink

    test_final = (1 - best_shrink) * stacking_test + best_shrink * test_probs[anchor]
    r = eval_metrics(test_labels, test_final, n_classes)
    r["anchor"] = anchor
    r["shrink"] = float(best_shrink)
    r["stacking_C"] = best_C
    return r


def main():
    print("=" * 70)
    print("ANCHOR FUSION v3 — Per-Probe C Tuning, No PCA")
    print("=" * 70)

    all_results = {}

    for dataset, info in DATASETS.items():
        n_classes = info["n_classes"]
        baseline_method, baseline_auroc = BASELINES[dataset]

        print(f"\n{'='*60}")
        print(f"Dataset: {dataset} ({n_classes}-class)")
        print(f"Baseline: {baseline_method} AUROC={baseline_auroc:.4f}")
        print(f"{'='*60}")

        data = load_raw_features(dataset, MULTICLASS_METHODS)

        print(f"\nPer-probe CV with C tuning:")
        methods, all_labels, oof_probs, test_probs, fold_aurocs, best_Cs = \
            get_cv_probs_tuned(data, n_classes)

        test_labels = data["test"]["labels"]

        # Individual probe test AUROC (from CV-averaged test probs)
        print(f"\nIndividual probe test AUROC (from CV-averaged predictions):")
        for m in sorted(methods, key=lambda m: -compute_auroc(test_labels, test_probs[m], n_classes)):
            a = compute_auroc(test_labels, test_probs[m], n_classes)
            marker = "*" if a >= baseline_auroc else " "
            print(f"  {marker} {m:20s} AUROC={a:.4f}  (C={best_Cs[m]})")

        ds_results = {}

        # Method B: LCB weighted avg
        r = lcb_weighted_avg(methods, all_labels, oof_probs, test_probs, fold_aurocs, n_classes, test_labels)
        delta = r["auroc"] - baseline_auroc
        status = ">>>" if delta > 0 else "   "
        top_w = sorted(r["weights"].items(), key=lambda x: -x[1])[:3]
        print(f"\n  {status} lcb_weighted_avg       AUROC={r['auroc']:.4f} ({delta:+.4f})  "
              f"Acc={r['accuracy']:.4f}  F1={r['f1_macro']:.4f}  top_w={[(m,f'{w:.2f}') for m,w in top_w]}")
        ds_results["lcb_weighted_avg"] = r

        # Method E: CV stacking + anchor shrink
        r = cv_stacking_shrink(methods, all_labels, oof_probs, test_probs, fold_aurocs, n_classes, test_labels)
        delta = r["auroc"] - baseline_auroc
        status = ">>>" if delta > 0 else "   "
        print(f"  {status} cv_stack_anchor_shrink AUROC={r['auroc']:.4f} ({delta:+.4f})  "
              f"Acc={r['accuracy']:.4f}  F1={r['f1_macro']:.4f}  anchor={r['anchor']} shrink={r['shrink']}")
        ds_results["cv_stack_anchor_shrink"] = r

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

    for mn in ["lcb_weighted_avg", "cv_stack_anchor_shrink"]:
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
    with open("/home/junyi/NIPS2026/fusion/results/anchor_v3_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved to fusion/results/anchor_v3_results.json")


if __name__ == "__main__":
    main()
