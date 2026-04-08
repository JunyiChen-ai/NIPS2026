"""
Pilot v2: Cross-validated stacking + proper feature fusion.
Key fix: Use K-fold CV on train to generate stacking features, not val.
"""

import os, json, time, warnings
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore", category=Warning)

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


def load_labels(dataset, split):
    raw_split = DATASETS[dataset]["splits"][split]
    meta_path = os.path.join(EXTRACTION_DIR, dataset, raw_split, "meta.json")
    with open(meta_path) as f:
        return np.array(json.load(f)["labels"])


def load_and_reduce(dataset, methods, proj_dim=128):
    """Load train/val/test, combine train+val, PCA reduce."""
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

        # Standardize on train
        sc = StandardScaler()
        tr = sc.fit_transform(tr)
        va = sc.transform(va)
        te = sc.transform(te)

        # PCA if dim > proj_dim
        if tr.shape[1] > proj_dim:
            pca = PCA(n_components=proj_dim, random_state=42)
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


def eval_metrics(y_true, y_prob, y_pred, n_classes):
    return {
        "auroc": compute_auroc(y_true, y_prob, n_classes),
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
    }


# ============================================================
# Method 1: CV Stacking (cross-validated out-of-fold probs)
# ============================================================
def cv_stacking(data, n_classes, n_folds=5, C_values=[0.01, 0.1, 1.0, 10.0]):
    """
    For each probe: 5-fold CV on train+val to get OOF probs.
    Meta-classifier trained on OOF probs.
    Hyperparameter C selected via inner CV on the stacking features.
    """
    methods = list(data["train"]["feats"].keys())

    # Combine train + val
    all_labels = np.concatenate([data["train"]["labels"], data["val"]["labels"]])
    all_feats = {}
    for m in methods:
        all_feats[m] = np.vstack([data["train"]["feats"][m], data["val"]["feats"][m]])
    test_feats = {m: data["test"]["feats"][m] for m in methods}
    n_total = len(all_labels)

    # Step 1: Generate OOF probs via CV
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    oof_probs = {m: np.zeros((n_total, n_classes)) for m in methods}
    test_probs = {m: np.zeros((n_total if False else data["test"]["labels"].shape[0], n_classes)) for m in methods}
    test_probs = {m: np.zeros((data["test"]["labels"].shape[0], n_classes)) for m in methods}

    for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(all_feats[methods[0]], all_labels)):
        for m in methods:
            clf = LogisticRegression(max_iter=2000, random_state=42, C=1.0, solver="lbfgs")
            clf.fit(all_feats[m][tr_idx], all_labels[tr_idx])
            oof_probs[m][va_idx] = clf.predict_proba(all_feats[m][va_idx])
            test_probs[m] += clf.predict_proba(test_feats[m]) / n_folds

    # Step 2: Stack OOF probs
    oof_meta = np.hstack([oof_probs[m] for m in methods])  # (N, n_methods * n_classes)
    test_meta = np.hstack([test_probs[m] for m in methods])

    # Step 3: Select C via inner CV
    best_auroc, best_C = -1, 1.0
    for C in C_values:
        inner_oof = np.zeros((n_total, n_classes))
        for _, (tr_i, va_i) in enumerate(skf.split(oof_meta, all_labels)):
            clf = LogisticRegression(max_iter=2000, random_state=42, C=C)
            clf.fit(oof_meta[tr_i], all_labels[tr_i])
            inner_oof[va_i] = clf.predict_proba(oof_meta[va_i])
        auroc = compute_auroc(all_labels, inner_oof, n_classes)
        if auroc > best_auroc:
            best_auroc = auroc
            best_C = C

    # Step 4: Final meta-classifier
    meta_clf = LogisticRegression(max_iter=2000, random_state=42, C=best_C)
    meta_clf.fit(oof_meta, all_labels)

    y_prob = meta_clf.predict_proba(test_meta)
    y_pred = meta_clf.predict(test_meta)
    r = eval_metrics(data["test"]["labels"], y_prob, y_pred, n_classes)
    r["best_C"] = best_C
    return r


# ============================================================
# Method 2: CV Stacking + Feature Concat → LR
# ============================================================
def cv_stacking_plus_feats(data, n_classes, n_folds=5):
    """OOF probs + projected features → LR."""
    methods = list(data["train"]["feats"].keys())
    all_labels = np.concatenate([data["train"]["labels"], data["val"]["labels"]])
    all_feats = {}
    for m in methods:
        all_feats[m] = np.vstack([data["train"]["feats"][m], data["val"]["feats"][m]])
    test_feats = {m: data["test"]["feats"][m] for m in methods}
    n_total = len(all_labels)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    oof_probs = {m: np.zeros((n_total, n_classes)) for m in methods}
    test_probs = {m: np.zeros((data["test"]["labels"].shape[0], n_classes)) for m in methods}

    for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(all_feats[methods[0]], all_labels)):
        for m in methods:
            clf = LogisticRegression(max_iter=2000, random_state=42, C=1.0)
            clf.fit(all_feats[m][tr_idx], all_labels[tr_idx])
            oof_probs[m][va_idx] = clf.predict_proba(all_feats[m][va_idx])
            test_probs[m] += clf.predict_proba(test_feats[m]) / n_folds

    # Combine probs + raw features
    all_concat = np.hstack([all_feats[m] for m in methods])
    test_concat = np.hstack([test_feats[m] for m in methods])

    oof_meta = np.hstack([np.hstack([oof_probs[m] for m in methods]), all_concat])
    test_meta = np.hstack([np.hstack([test_probs[m] for m in methods]), test_concat])

    # LR with regularization tuning
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

    clf = LogisticRegression(max_iter=2000, random_state=42, C=best_C)
    clf.fit(oof_meta, all_labels)

    y_prob = clf.predict_proba(test_meta)
    y_pred = clf.predict(test_meta)
    r = eval_metrics(data["test"]["labels"], y_prob, y_pred, n_classes)
    r["best_C"] = best_C
    return r


# ============================================================
# Method 3: Concat features (train+val) → LR with CV tuning
# ============================================================
def concat_lr_cv(data, n_classes):
    """Simple concat + LR but with train+val and CV tuning."""
    methods = list(data["train"]["feats"].keys())
    all_labels = np.concatenate([data["train"]["labels"], data["val"]["labels"]])
    all_concat = np.hstack([
        np.vstack([data["train"]["feats"][m], data["val"]["feats"][m]])
        for m in methods
    ])
    test_concat = np.hstack([data["test"]["feats"][m] for m in methods])
    n_total = len(all_labels)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    best_auroc, best_C = -1, 1.0
    for C in [0.001, 0.01, 0.1, 1.0, 10.0]:
        oof = np.zeros((n_total, n_classes))
        for _, (tr_i, va_i) in enumerate(skf.split(all_concat, all_labels)):
            clf = LogisticRegression(max_iter=2000, random_state=42, C=C)
            clf.fit(all_concat[tr_i], all_labels[tr_i])
            oof[va_i] = clf.predict_proba(all_concat[va_i])
        auroc = compute_auroc(all_labels, oof, n_classes)
        if auroc > best_auroc:
            best_auroc = auroc
            best_C = C

    clf = LogisticRegression(max_iter=2000, random_state=42, C=best_C)
    clf.fit(all_concat, all_labels)

    y_prob = clf.predict_proba(test_concat)
    y_pred = clf.predict(test_concat)
    r = eval_metrics(data["test"]["labels"], y_prob, y_pred, n_classes)
    r["best_C"] = best_C
    return r


# ============================================================
# Method 4: Concat features → MLP with train+val
# ============================================================
def concat_mlp_cv(data, n_classes):
    methods = list(data["train"]["feats"].keys())
    all_labels = np.concatenate([data["train"]["labels"], data["val"]["labels"]])
    all_concat = np.hstack([
        np.vstack([data["train"]["feats"][m], data["val"]["feats"][m]])
        for m in methods
    ])
    test_concat = np.hstack([data["test"]["feats"][m] for m in methods])

    clf = MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=500, random_state=42,
                        early_stopping=True, validation_fraction=0.1,
                        alpha=1e-3, learning_rate_init=1e-3)
    clf.fit(all_concat, all_labels)

    y_prob = clf.predict_proba(test_concat)
    y_pred = clf.predict(test_concat)
    return eval_metrics(data["test"]["labels"], y_prob, y_pred, n_classes)


# ============================================================
# Method 5: Average Probs (simple ensemble, no meta-learning)
# ============================================================
def avg_probs(data, n_classes):
    """Train individual classifiers on train+val, average their test probs."""
    methods = list(data["train"]["feats"].keys())
    all_labels = np.concatenate([data["train"]["labels"], data["val"]["labels"]])
    test_labels = data["test"]["labels"]

    avg_prob = np.zeros((len(test_labels), n_classes))
    for m in methods:
        X = np.vstack([data["train"]["feats"][m], data["val"]["feats"][m]])
        X_te = data["test"]["feats"][m]
        clf = LogisticRegression(max_iter=2000, random_state=42, C=1.0)
        clf.fit(X, all_labels)
        avg_prob += clf.predict_proba(X_te)
    avg_prob /= len(methods)

    y_pred = avg_prob.argmax(axis=1)
    return eval_metrics(test_labels, avg_prob, y_pred, n_classes)


# ============================================================
# Method 6: Weighted Average (weights tuned on CV)
# ============================================================
def weighted_avg_probs(data, n_classes, n_folds=5):
    """Like avg_probs but with learned weights via CV."""
    methods = list(data["train"]["feats"].keys())
    all_labels = np.concatenate([data["train"]["labels"], data["val"]["labels"]])
    test_labels = data["test"]["labels"]
    n_total = len(all_labels)
    n_methods = len(methods)

    # CV to get per-method OOF AUROC → use as weights
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    method_aurocs = {}
    test_probs_all = {}

    for m in methods:
        X = np.vstack([data["train"]["feats"][m], data["val"]["feats"][m]])
        X_te = data["test"]["feats"][m]
        oof = np.zeros((n_total, n_classes))
        te_prob = np.zeros((len(test_labels), n_classes))

        for _, (tr_i, va_i) in enumerate(skf.split(X, all_labels)):
            clf = LogisticRegression(max_iter=2000, random_state=42, C=1.0)
            clf.fit(X[tr_i], all_labels[tr_i])
            oof[va_i] = clf.predict_proba(X[va_i])
            te_prob += clf.predict_proba(X_te) / n_folds

        method_aurocs[m] = compute_auroc(all_labels, oof, n_classes)
        test_probs_all[m] = te_prob

    # Softmax weights from AUROC
    auroc_arr = np.array([method_aurocs[m] for m in methods])
    # Temperature-scaled softmax
    temp = 0.1
    weights = np.exp((auroc_arr - auroc_arr.max()) / temp)
    weights /= weights.sum()

    avg_prob = np.zeros((len(test_labels), n_classes))
    for i, m in enumerate(methods):
        avg_prob += weights[i] * test_probs_all[m]

    y_pred = avg_prob.argmax(axis=1)
    r = eval_metrics(test_labels, avg_prob, y_pred, n_classes)
    r["weights"] = {m: float(w) for m, w in zip(methods, weights)}
    return r


def main():
    print("=" * 70)
    print("PILOT v2: CV Stacking + Proper Feature Fusion")
    print("=" * 70)

    all_results = {}
    fusion_methods = [
        ("avg_probs", avg_probs),
        ("weighted_avg_probs", weighted_avg_probs),
        ("cv_stacking", cv_stacking),
        ("concat_lr_cv", concat_lr_cv),
        ("concat_mlp_cv", concat_mlp_cv),
        ("cv_stacking+feats", cv_stacking_plus_feats),
    ]

    for dataset, info in DATASETS.items():
        n_classes = info["n_classes"]
        baseline_method, baseline_auroc = BASELINES[dataset]

        print(f"\n{'='*60}")
        print(f"Dataset: {dataset} ({n_classes}-class)")
        print(f"Baseline: {baseline_method} AUROC={baseline_auroc:.4f}")
        print(f"{'='*60}")

        t0 = time.time()
        data = load_and_reduce(dataset, MULTICLASS_METHODS)
        methods_loaded = list(data["train"]["feats"].keys())
        n_tr = len(data["train"]["labels"])
        n_va = len(data["val"]["labels"])
        n_te = len(data["test"]["labels"])
        print(f"Loaded {len(methods_loaded)} methods, train={n_tr} val={n_va} test={n_te}  ({time.time()-t0:.1f}s)")

        ds_results = {}
        for name, fn in fusion_methods:
            t1 = time.time()
            try:
                r = fn(data, n_classes)
                delta = r["auroc"] - baseline_auroc
                status = ">>>" if delta > 0 else "   "
                print(f"  {status} {name:30s}  AUROC={r['auroc']:.4f} ({delta:+.4f})  "
                      f"Acc={r['accuracy']:.4f}  F1={r['f1_macro']:.4f}  [{time.time()-t1:.1f}s]")
                ds_results[name] = r
            except Exception as e:
                print(f"      {name:30s}  ERROR: {e}")
                import traceback; traceback.print_exc()

        all_results[dataset] = ds_results

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Fusion AUROC vs Best Single Probe")
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
    with open("/home/junyi/NIPS2026/fusion/results/pilot_v2_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved to fusion/results/pilot_v2_results.json")


if __name__ == "__main__":
    main()
