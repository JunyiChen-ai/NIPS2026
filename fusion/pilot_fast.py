"""
Fast pilot fusion experiments on hard datasets.
All features are PCA-reduced to max 128d before any LR/MLP fitting.
"""

import os, json, time
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA

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

PROJ_DIM = 128  # Max feature dim after PCA


def load_labels(dataset, split):
    raw_split = DATASETS[dataset]["splits"][split]
    meta_path = os.path.join(EXTRACTION_DIR, dataset, raw_split, "meta.json")
    with open(meta_path) as f:
        return np.array(json.load(f)["labels"])


def load_and_reduce(dataset, methods, proj_dim=PROJ_DIM):
    """Load train/val/test features for all methods, PCA-reduce to proj_dim."""
    data = {}
    for split in ["train", "val", "test"]:
        data[split] = {"labels": load_labels(dataset, split), "feats": {}}

    # For each method: load, standardize, PCA reduce
    projectors = {}
    for method in methods:
        train_path = os.path.join(PROCESSED_DIR, dataset, method, "train.pt")
        if not os.path.exists(train_path):
            continue

        tr = torch.load(train_path, map_location="cpu").float().numpy()
        va = torch.load(os.path.join(PROCESSED_DIR, dataset, method, "val.pt"), map_location="cpu").float().numpy()
        te = torch.load(os.path.join(PROCESSED_DIR, dataset, method, "test.pt"), map_location="cpu").float().numpy()

        if tr.ndim == 1:
            tr, va, te = tr.reshape(-1, 1), va.reshape(-1, 1), te.reshape(-1, 1)

        # Standardize
        sc = StandardScaler()
        tr = sc.fit_transform(tr)
        va = sc.transform(va)
        te = sc.transform(te)

        # PCA if dim > proj_dim
        d = tr.shape[1]
        if d > proj_dim:
            pca = PCA(n_components=proj_dim, random_state=42)
            tr = pca.fit_transform(tr)
            va = pca.transform(va)
            te = pca.transform(te)
            projectors[method] = {"scaler": sc, "pca": pca}
        else:
            projectors[method] = {"scaler": sc, "pca": None}

        data["train"]["feats"][method] = tr
        data["val"]["feats"][method] = va
        data["test"]["feats"][method] = te

    return data, projectors


def compute_auroc(y_true, y_prob, n_classes):
    if n_classes == 2:
        return roc_auc_score(y_true, y_prob[:, 1])
    else:
        y_bin = label_binarize(y_true, classes=list(range(n_classes)))
        return roc_auc_score(y_bin, y_prob, average="macro", multi_class="ovr")


def eval_metrics(y_true, y_prob, y_pred, n_classes):
    return {
        "auroc": compute_auroc(y_true, y_prob, n_classes),
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
    }


# ============================================================
# Method 1: Calibrated Stacking
# ============================================================
def calibrated_stacking(data, n_classes):
    """Per-probe LR probs on train → stack → meta-LR on val → eval on test."""
    methods = list(data["train"]["feats"].keys())
    val_probs, test_probs = [], []

    for method in methods:
        clf = LogisticRegression(max_iter=1000, random_state=42, C=1.0, solver="lbfgs")
        clf.fit(data["train"]["feats"][method], data["train"]["labels"])
        val_probs.append(clf.predict_proba(data["val"]["feats"][method]))
        test_probs.append(clf.predict_proba(data["test"]["feats"][method]))

    val_meta = np.hstack(val_probs)
    test_meta = np.hstack(test_probs)

    meta_clf = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
    meta_clf.fit(val_meta, data["val"]["labels"])

    y_prob = meta_clf.predict_proba(test_meta)
    y_pred = meta_clf.predict(test_meta)
    return eval_metrics(data["test"]["labels"], y_prob, y_pred, n_classes)


# ============================================================
# Method 2: Concat + LR (train+val combined)
# ============================================================
def concat_lr(data, n_classes):
    methods = list(data["train"]["feats"].keys())
    X_tr = np.hstack([data["train"]["feats"][m] for m in methods])
    X_va = np.hstack([data["val"]["feats"][m] for m in methods])
    X_te = np.hstack([data["test"]["feats"][m] for m in methods])

    X_trva = np.vstack([X_tr, X_va])
    y_trva = np.concatenate([data["train"]["labels"], data["val"]["labels"]])

    clf = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
    clf.fit(X_trva, y_trva)

    y_prob = clf.predict_proba(X_te)
    y_pred = clf.predict(X_te)
    return eval_metrics(data["test"]["labels"], y_prob, y_pred, n_classes)


# ============================================================
# Method 3: Concat + MLP
# ============================================================
def concat_mlp(data, n_classes):
    methods = list(data["train"]["feats"].keys())
    X_tr = np.hstack([data["train"]["feats"][m] for m in methods])
    X_va = np.hstack([data["val"]["feats"][m] for m in methods])
    X_te = np.hstack([data["test"]["feats"][m] for m in methods])

    # Grid search on val
    best_auroc, best_cfg = -1, None
    for hidden in [(256, 128), (512, 256), (128, 64)]:
        for alpha in [1e-3, 1e-4]:
            clf = MLPClassifier(hidden_layer_sizes=hidden, max_iter=500, random_state=42,
                                early_stopping=True, validation_fraction=0.15,
                                alpha=alpha, learning_rate_init=1e-3)
            clf.fit(X_tr, data["train"]["labels"])
            val_prob = clf.predict_proba(X_va)
            auroc = compute_auroc(data["val"]["labels"], val_prob, n_classes)
            if auroc > best_auroc:
                best_auroc = auroc
                best_cfg = (hidden, alpha)

    # Retrain on train+val
    X_trva = np.vstack([X_tr, X_va])
    y_trva = np.concatenate([data["train"]["labels"], data["val"]["labels"]])
    clf = MLPClassifier(hidden_layer_sizes=best_cfg[0], max_iter=500, random_state=42,
                        early_stopping=True, validation_fraction=0.1,
                        alpha=best_cfg[1], learning_rate_init=1e-3)
    clf.fit(X_trva, y_trva)

    y_prob = clf.predict_proba(X_te)
    y_pred = clf.predict(X_te)
    r = eval_metrics(data["test"]["labels"], y_prob, y_pred, n_classes)
    r["best_cfg"] = str(best_cfg)
    return r


# ============================================================
# Method 4: Hybrid (probs + features) → LR
# ============================================================
def hybrid_stack_concat(data, n_classes):
    methods = list(data["train"]["feats"].keys())

    # Score-level: per-probe probs
    val_probs, test_probs = [], []
    for method in methods:
        clf = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
        clf.fit(data["train"]["feats"][method], data["train"]["labels"])
        val_probs.append(clf.predict_proba(data["val"]["feats"][method]))
        test_probs.append(clf.predict_proba(data["test"]["feats"][method]))

    # Feature-level: concatenated reduced features
    val_feats = np.hstack([data["val"]["feats"][m] for m in methods])
    test_feats = np.hstack([data["test"]["feats"][m] for m in methods])

    # Hybrid: probs + features
    val_meta = np.hstack([np.hstack(val_probs), val_feats])
    test_meta = np.hstack([np.hstack(test_probs), test_feats])

    meta_clf = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
    meta_clf.fit(val_meta, data["val"]["labels"])

    y_prob = meta_clf.predict_proba(test_meta)
    y_pred = meta_clf.predict(test_meta)
    return eval_metrics(data["test"]["labels"], y_prob, y_pred, n_classes)


# ============================================================
# Method 5: Hybrid → MLP
# ============================================================
def hybrid_stack_mlp(data, n_classes):
    methods = list(data["train"]["feats"].keys())

    # Train-level probs (leave-one-out would be ideal, but use val-based for speed)
    train_probs, val_probs, test_probs = [], [], []
    for method in methods:
        clf = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
        clf.fit(data["train"]["feats"][method], data["train"]["labels"])
        train_probs.append(clf.predict_proba(data["train"]["feats"][method]))
        val_probs.append(clf.predict_proba(data["val"]["feats"][method]))
        test_probs.append(clf.predict_proba(data["test"]["feats"][method]))

    # Build hybrid features for all splits
    tr_feats = np.hstack([data["train"]["feats"][m] for m in methods])
    va_feats = np.hstack([data["val"]["feats"][m] for m in methods])
    te_feats = np.hstack([data["test"]["feats"][m] for m in methods])

    X_tr = np.hstack([np.hstack(train_probs), tr_feats])
    X_va = np.hstack([np.hstack(val_probs), va_feats])
    X_te = np.hstack([np.hstack(test_probs), te_feats])

    X_trva = np.vstack([X_tr, X_va])
    y_trva = np.concatenate([data["train"]["labels"], data["val"]["labels"]])

    clf = MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=500, random_state=42,
                        early_stopping=True, validation_fraction=0.1,
                        alpha=1e-3, learning_rate_init=1e-3)
    clf.fit(X_trva, y_trva)

    y_prob = clf.predict_proba(X_te)
    y_pred = clf.predict(X_te)
    return eval_metrics(data["test"]["labels"], y_prob, y_pred, n_classes)


def main():
    print("=" * 70)
    print("FAST PILOT FUSION — Hard Datasets (PCA-reduced to 128d)")
    print("=" * 70)

    all_results = {}
    fusion_methods = [
        ("calibrated_stacking", calibrated_stacking),
        ("concat_lr", concat_lr),
        ("concat_mlp", concat_mlp),
        ("hybrid_stack_concat_lr", hybrid_stack_concat),
        ("hybrid_stack_mlp", hybrid_stack_mlp),
    ]

    for dataset, info in DATASETS.items():
        n_classes = info["n_classes"]
        baseline_method, baseline_auroc = BASELINES[dataset]

        print(f"\n{'='*60}")
        print(f"Dataset: {dataset} ({n_classes}-class)")
        print(f"Baseline to beat: {baseline_method} AUROC={baseline_auroc:.4f}")
        print(f"{'='*60}")

        t0 = time.time()
        data, _ = load_and_reduce(dataset, MULTICLASS_METHODS)
        methods_loaded = list(data["train"]["feats"].keys())
        print(f"Loaded {len(methods_loaded)} methods: {methods_loaded}  ({time.time()-t0:.1f}s)")

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

        all_results[dataset] = ds_results

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY: Fusion AUROC vs Best Single Probe")
    print("=" * 70)
    print(f"{'Method':32s}", end="")
    for ds in DATASETS:
        short = ds.replace("_3class", "_3c").replace("_5class", "_5c").replace("common_claim", "cc").replace("when2call", "w2c").replace("e2h_amc", "e2h")
        print(f"  {short:>10s}", end="")
    print(f"  {'#wins':>6s}")

    # Print baseline row
    print(f"{'[best single probe]':32s}", end="")
    for ds in DATASETS:
        _, bl = BASELINES[ds]
        print(f"  {bl:10.4f}", end="")
    print()

    for method_name in [m[0] for m in fusion_methods]:
        print(f"{method_name:32s}", end="")
        wins = 0
        for ds in DATASETS:
            if ds in all_results and method_name in all_results[ds]:
                auroc = all_results[ds][method_name]["auroc"]
                _, bl = BASELINES[ds]
                delta = auroc - bl
                if delta > 0:
                    wins += 1
                print(f"  {auroc:10.4f}", end="")
            else:
                print(f"  {'N/A':>10s}", end="")
        print(f"  {wins:>6d}/4")

    # Save
    os.makedirs("/home/junyi/NIPS2026/fusion/results", exist_ok=True)
    with open("/home/junyi/NIPS2026/fusion/results/pilot_fast_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to fusion/results/pilot_fast_results.json")


if __name__ == "__main__":
    main()
