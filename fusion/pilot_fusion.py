"""
Pilot fusion experiments on hard datasets.
Tests multiple fusion strategies over pre-extracted probe features.

Datasets (multi-class, 7 methods each):
  - common_claim_3class (hardest, best AUROC=0.758)
  - e2h_amc_3class (best AUROC=0.893)
  - e2h_amc_5class (best AUROC=0.875)
  - when2call_3class (best AUROC=0.874)

Available probe methods for multi-class: lr_probe, pca_lr, iti, kb_mlp, attn_satisfies, sep, step
"""

import os
import json
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.neural_network import MLPClassifier
from scipy.stats import spearmanr

PROCESSED_DIR = "/home/junyi/NIPS2026/reproduce/processed_features"

# Methods available for multi-class datasets
MULTICLASS_METHODS = ["lr_probe", "pca_lr", "iti", "kb_mlp", "attn_satisfies", "sep", "step"]

DATASETS = {
    "common_claim_3class": {"n_classes": 3, "task": "multiclass"},
    "e2h_amc_3class":      {"n_classes": 3, "task": "multiclass"},
    "e2h_amc_5class":      {"n_classes": 5, "task": "multiclass"},
    "when2call_3class":     {"n_classes": 3, "task": "multiclass"},
}

# Baselines to beat (best single probe AUROC on test)
BASELINES = {
    "common_claim_3class": ("pca_lr", 0.7576),
    "e2h_amc_3class":      ("pca_lr", 0.8934),
    "e2h_amc_5class":      ("kb_mlp", 0.8752),
    "when2call_3class":     ("lr_probe", 0.8741),
}


def load_features(dataset, methods, split="train"):
    """Load processed features for all methods on a dataset split."""
    features = {}
    labels = None
    for method in methods:
        path = os.path.join(PROCESSED_DIR, dataset, method, f"{split}.pt")
        if not os.path.exists(path):
            continue
        feat = torch.load(path, map_location="cpu")
        if isinstance(feat, np.ndarray):
            feat = torch.from_numpy(feat)
        features[method] = feat.float()

        # Load labels from meta or from the extraction dir
        if labels is None:
            meta_path = os.path.join(PROCESSED_DIR, dataset, method, "meta.json")

    # Load labels from the feature extraction directory
    EXTRACTION_DIR = "/home/junyi/NIPS2026/extraction/features"
    # Map normalized split names to extraction dir split names
    SPLIT_MAP = {
        "e2h_amc_3class":  {"train": "train_sub", "val": "val_split", "test": "eval"},
        "e2h_amc_5class":  {"train": "train_sub", "val": "val_split", "test": "eval"},
        "retrievalqa":     {"train": "train_sub", "val": "val_split", "test": "test"},
        "easy2hard_amc":   {"train": "train_sub", "val": "val_split", "test": "eval"},
    }
    raw_split = SPLIT_MAP.get(dataset, {}).get(split, split)
    feat_dir = os.path.join(EXTRACTION_DIR, dataset)
    meta_path = os.path.join(feat_dir, raw_split, "meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        labels = np.array(meta["labels"])

    return features, labels


def compute_auroc(y_true, y_prob, n_classes):
    """Compute AUROC, handling binary and multi-class."""
    if n_classes == 2:
        return roc_auc_score(y_true, y_prob[:, 1])
    else:
        y_bin = label_binarize(y_true, classes=list(range(n_classes)))
        return roc_auc_score(y_bin, y_prob, average="macro", multi_class="ovr")


def project_to_common_dim(features, dim=128):
    """Project each method's features to a common dimensionality."""
    projections = {}
    scalers = {}
    for method, feat in features.items():
        if feat.dim() == 1:
            # Scalar features — expand
            feat = feat.unsqueeze(-1)
        n, d = feat.shape
        # Standardize first
        sc = StandardScaler()
        feat_np = sc.fit_transform(feat.numpy())
        scalers[method] = sc

        if d > dim:
            # PCA reduction
            U, S, Vh = np.linalg.svd(feat_np, full_matrices=False)
            projections[method] = {"type": "pca", "Vh": Vh[:dim], "scaler": sc}
            feat_np = feat_np @ Vh[:dim].T
        elif d < dim:
            # Pad with zeros
            projections[method] = {"type": "pad", "orig_dim": d, "scaler": sc}
            feat_np = np.hstack([feat_np, np.zeros((n, dim - d))])
        else:
            projections[method] = {"type": "none", "scaler": sc}

    return projections


def apply_projection(feat, proj, dim=128):
    """Apply a saved projection to new data."""
    if feat.dim() == 1:
        feat = feat.unsqueeze(-1)
    feat_np = proj["scaler"].transform(feat.numpy())
    if proj["type"] == "pca":
        return feat_np @ proj["Vh"].T
    elif proj["type"] == "pad":
        d = proj["orig_dim"]
        return np.hstack([feat_np, np.zeros((feat_np.shape[0], dim - d))])
    else:
        return feat_np


# ============================================================
# Method 1: Calibrated Stacking (score-level fusion)
# ============================================================
def calibrated_stacking(dataset, n_classes):
    """
    Train individual LogisticRegression per probe on train,
    collect their class probabilities, then train a meta-LR on val probs.
    Evaluate on test.
    """
    methods = MULTICLASS_METHODS

    train_feats, train_labels = load_features(dataset, methods, "train")
    val_feats, val_labels = load_features(dataset, methods, "val")
    test_feats, test_labels = load_features(dataset, methods, "test")

    if train_labels is None:
        print(f"  [calibrated_stacking] Cannot load labels for {dataset}")
        return None

    # Step 1: Train per-probe classifiers on train, predict probs on val and test
    val_probs_list = []
    test_probs_list = []
    method_names = []

    for method in methods:
        if method not in train_feats or method not in val_feats or method not in test_feats:
            continue

        tr = train_feats[method].numpy()
        va = val_feats[method].numpy()
        te = test_feats[method].numpy()

        if tr.ndim == 1:
            tr = tr.reshape(-1, 1)
            va = va.reshape(-1, 1)
            te = te.reshape(-1, 1)

        # Standardize
        sc = StandardScaler()
        tr = sc.fit_transform(tr)
        va = sc.transform(va)
        te = sc.transform(te)

        clf = LogisticRegression(max_iter=2000, random_state=42, C=1.0)
        clf.fit(tr, train_labels)

        val_probs_list.append(clf.predict_proba(va))
        test_probs_list.append(clf.predict_proba(te))
        method_names.append(method)

    # Step 2: Stack val probs as meta-features
    val_meta = np.hstack(val_probs_list)  # (N_val, n_methods * n_classes)
    test_meta = np.hstack(test_probs_list)

    # Step 3: Train meta-classifier on val
    meta_clf = LogisticRegression(max_iter=2000, random_state=42, C=1.0)
    meta_clf.fit(val_meta, val_labels)

    # Step 4: Evaluate on test
    test_pred_probs = meta_clf.predict_proba(test_meta)
    auroc = compute_auroc(test_labels, test_pred_probs, n_classes)
    acc = accuracy_score(test_labels, meta_clf.predict(test_meta))
    f1 = f1_score(test_labels, meta_clf.predict(test_meta), average="macro")

    return {"auroc": auroc, "accuracy": acc, "f1_macro": f1, "methods_used": method_names}


# ============================================================
# Method 2: Feature-level concatenation + LR
# ============================================================
def concat_lr(dataset, n_classes, proj_dim=128):
    """
    Project each probe's features to common dim, concatenate, train LR.
    """
    methods = MULTICLASS_METHODS

    train_feats, train_labels = load_features(dataset, methods, "train")
    val_feats, val_labels = load_features(dataset, methods, "val")
    test_feats, test_labels = load_features(dataset, methods, "test")

    if train_labels is None:
        return None

    # Project each method to common dim
    available = [m for m in methods if m in train_feats and m in val_feats and m in test_feats]

    projections = {}
    train_projected = []
    for method in available:
        feat = train_feats[method]
        if feat.dim() == 1:
            feat = feat.unsqueeze(-1)
        n, d = feat.shape
        sc = StandardScaler()
        feat_np = sc.fit_transform(feat.numpy())

        if d > proj_dim:
            U, S, Vh = np.linalg.svd(feat_np, full_matrices=False)
            projections[method] = {"Vh": Vh[:proj_dim], "scaler": sc}
            feat_np = feat_np @ Vh[:proj_dim].T
        else:
            projections[method] = {"Vh": None, "scaler": sc}

        train_projected.append(feat_np)

    X_train = np.hstack(train_projected)

    # Project val and test
    def project_split(feats):
        parts = []
        for method in available:
            feat = feats[method]
            if feat.dim() == 1:
                feat = feat.unsqueeze(-1)
            feat_np = projections[method]["scaler"].transform(feat.numpy())
            if projections[method]["Vh"] is not None:
                feat_np = feat_np @ projections[method]["Vh"].T
            parts.append(feat_np)
        return np.hstack(parts)

    X_val = project_split(val_feats)
    X_test = project_split(test_feats)

    # Train on train+val combined (use val for early stopping is not needed for LR)
    X_trainval = np.vstack([X_train, X_val])
    y_trainval = np.concatenate([train_labels, val_labels])

    clf = LogisticRegression(max_iter=3000, random_state=42, C=1.0)
    clf.fit(X_trainval, y_trainval)

    test_probs = clf.predict_proba(X_test)
    auroc = compute_auroc(test_labels, test_probs, n_classes)
    acc = accuracy_score(test_labels, clf.predict(X_test))
    f1 = f1_score(test_labels, clf.predict(X_test), average="macro")

    return {"auroc": auroc, "accuracy": acc, "f1_macro": f1, "methods_used": available}


# ============================================================
# Method 3: Feature concat + MLP
# ============================================================
def concat_mlp(dataset, n_classes, proj_dim=128):
    """
    Project each probe's features to common dim, concatenate, train MLP.
    """
    methods = MULTICLASS_METHODS

    train_feats, train_labels = load_features(dataset, methods, "train")
    val_feats, val_labels = load_features(dataset, methods, "val")
    test_feats, test_labels = load_features(dataset, methods, "test")

    if train_labels is None:
        return None

    available = [m for m in methods if m in train_feats and m in val_feats and m in test_feats]

    projections = {}
    train_projected = []
    for method in available:
        feat = train_feats[method]
        if feat.dim() == 1:
            feat = feat.unsqueeze(-1)
        n, d = feat.shape
        sc = StandardScaler()
        feat_np = sc.fit_transform(feat.numpy())

        if d > proj_dim:
            U, S, Vh = np.linalg.svd(feat_np, full_matrices=False)
            projections[method] = {"Vh": Vh[:proj_dim], "scaler": sc}
            feat_np = feat_np @ Vh[:proj_dim].T
        else:
            projections[method] = {"Vh": None, "scaler": sc}

        train_projected.append(feat_np)

    X_train = np.hstack(train_projected)

    def project_split(feats):
        parts = []
        for method in available:
            feat = feats[method]
            if feat.dim() == 1:
                feat = feat.unsqueeze(-1)
            feat_np = projections[method]["scaler"].transform(feat.numpy())
            if projections[method]["Vh"] is not None:
                feat_np = feat_np @ projections[method]["Vh"].T
            parts.append(feat_np)
        return np.hstack(parts)

    X_val = project_split(val_feats)
    X_test = project_split(test_feats)

    # Use val for early stopping
    best_auroc = -1
    best_params = None
    for hidden in [(256, 128), (512, 256), (128, 64)]:
        for alpha in [1e-3, 1e-4]:
            clf = MLPClassifier(
                hidden_layer_sizes=hidden,
                max_iter=500,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.15,
                alpha=alpha,
                learning_rate_init=1e-3,
            )
            clf.fit(X_train, train_labels)
            val_probs = clf.predict_proba(X_val)
            auroc = compute_auroc(val_labels, val_probs, n_classes)
            if auroc > best_auroc:
                best_auroc = auroc
                best_params = (hidden, alpha)

    # Retrain with best params on train+val
    X_trainval = np.vstack([X_train, X_val])
    y_trainval = np.concatenate([train_labels, val_labels])

    clf = MLPClassifier(
        hidden_layer_sizes=best_params[0],
        max_iter=500,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        alpha=best_params[1],
        learning_rate_init=1e-3,
    )
    clf.fit(X_trainval, y_trainval)

    test_probs = clf.predict_proba(X_test)
    auroc = compute_auroc(test_labels, test_probs, n_classes)
    acc = accuracy_score(test_labels, clf.predict(X_test))
    f1 = f1_score(test_labels, clf.predict(X_test), average="macro")

    return {"auroc": auroc, "accuracy": acc, "f1_macro": f1,
            "methods_used": available, "best_params": str(best_params)}


# ============================================================
# Method 4: Stacking + Feature Concat (hybrid)
# ============================================================
def hybrid_stacking_concat(dataset, n_classes, proj_dim=64):
    """
    Combine score-level (per-probe class probs) and feature-level (projected features).
    Meta-features = [probe1_probs, ..., probeK_probs, proj_feat1, ..., proj_featK]
    """
    methods = MULTICLASS_METHODS

    train_feats, train_labels = load_features(dataset, methods, "train")
    val_feats, val_labels = load_features(dataset, methods, "val")
    test_feats, test_labels = load_features(dataset, methods, "test")

    if train_labels is None:
        return None

    available = [m for m in methods if m in train_feats and m in val_feats and m in test_feats]

    # Score-level: per-probe LR probs (train on train, predict val/test)
    val_probs_all = []
    test_probs_all = []

    # Feature-level: projected features
    projections = {}
    train_proj_all = []

    for method in available:
        tr = train_feats[method]
        va = val_feats[method]
        te = test_feats[method]

        if tr.dim() == 1:
            tr = tr.unsqueeze(-1)
            va = va.unsqueeze(-1)
            te = te.unsqueeze(-1)

        # Standardize
        sc = StandardScaler()
        tr_np = sc.fit_transform(tr.numpy())
        va_np = sc.transform(va.numpy())
        te_np = sc.transform(te.numpy())

        # PCA if needed
        d = tr_np.shape[1]
        if d > proj_dim:
            U, S, Vh = np.linalg.svd(tr_np, full_matrices=False)
            proj_matrix = Vh[:proj_dim]
            tr_proj = tr_np @ proj_matrix.T
            va_proj = va_np @ proj_matrix.T
            te_proj = te_np @ proj_matrix.T
            projections[method] = {"Vh": proj_matrix, "scaler": sc}
        else:
            tr_proj = tr_np
            va_proj = va_np
            te_proj = te_np
            projections[method] = {"Vh": None, "scaler": sc}

        train_proj_all.append(tr_proj)

        # Score-level: train LR on train
        clf = LogisticRegression(max_iter=2000, random_state=42)
        clf.fit(tr_np, train_labels)
        val_probs_all.append(clf.predict_proba(va_np))
        test_probs_all.append(clf.predict_proba(te_np))

    # Build val/test projected features
    def get_proj_feats(feats):
        parts = []
        for method in available:
            feat = feats[method]
            if feat.dim() == 1:
                feat = feat.unsqueeze(-1)
            feat_np = projections[method]["scaler"].transform(feat.numpy())
            if projections[method]["Vh"] is not None:
                feat_np = feat_np @ projections[method]["Vh"].T
            parts.append(feat_np)
        return np.hstack(parts)

    val_proj = get_proj_feats(val_feats)
    test_proj = get_proj_feats(test_feats)

    # Hybrid meta-features: probs + projected feats
    val_meta = np.hstack([np.hstack(val_probs_all), val_proj])
    test_meta = np.hstack([np.hstack(test_probs_all), test_proj])

    # Meta-classifier on val
    meta_clf = LogisticRegression(max_iter=3000, random_state=42, C=1.0)
    meta_clf.fit(val_meta, val_labels)

    test_pred_probs = meta_clf.predict_proba(test_meta)
    auroc = compute_auroc(test_labels, test_pred_probs, n_classes)
    acc = accuracy_score(test_labels, meta_clf.predict(test_meta))
    f1 = f1_score(test_labels, meta_clf.predict(test_meta), average="macro")

    return {"auroc": auroc, "accuracy": acc, "f1_macro": f1, "methods_used": available}


def main():
    print("=" * 70)
    print("PILOT FUSION EXPERIMENTS — Hard Datasets")
    print("=" * 70)

    results = {}

    for dataset, info in DATASETS.items():
        n_classes = info["n_classes"]
        baseline_method, baseline_auroc = BASELINES[dataset]

        print(f"\n{'='*60}")
        print(f"Dataset: {dataset} ({n_classes}-class)")
        print(f"Baseline to beat: {baseline_method} AUROC={baseline_auroc:.4f}")
        print(f"{'='*60}")

        ds_results = {}

        # Method 1: Calibrated Stacking
        print("\n[1] Calibrated Stacking...")
        r = calibrated_stacking(dataset, n_classes)
        if r:
            ds_results["calibrated_stacking"] = r
            delta = r["auroc"] - baseline_auroc
            status = "BEAT" if delta > 0 else "MISS"
            print(f"    AUROC={r['auroc']:.4f}  Acc={r['accuracy']:.4f}  F1={r['f1_macro']:.4f}  [{status} by {delta:+.4f}]")

        # Method 2: Concat + LR
        print("\n[2] Feature Concat + LR...")
        r = concat_lr(dataset, n_classes)
        if r:
            ds_results["concat_lr"] = r
            delta = r["auroc"] - baseline_auroc
            status = "BEAT" if delta > 0 else "MISS"
            print(f"    AUROC={r['auroc']:.4f}  Acc={r['accuracy']:.4f}  F1={r['f1_macro']:.4f}  [{status} by {delta:+.4f}]")

        # Method 3: Concat + MLP
        print("\n[3] Feature Concat + MLP...")
        r = concat_mlp(dataset, n_classes)
        if r:
            ds_results["concat_mlp"] = r
            delta = r["auroc"] - baseline_auroc
            status = "BEAT" if delta > 0 else "MISS"
            print(f"    AUROC={r['auroc']:.4f}  Acc={r['accuracy']:.4f}  F1={r['f1_macro']:.4f}  [{status} by {delta:+.4f}]")

        # Method 4: Hybrid Stacking + Concat
        print("\n[4] Hybrid Stacking + Feature Concat...")
        r = hybrid_stacking_concat(dataset, n_classes)
        if r:
            ds_results["hybrid_stacking_concat"] = r
            delta = r["auroc"] - baseline_auroc
            status = "BEAT" if delta > 0 else "MISS"
            print(f"    AUROC={r['auroc']:.4f}  Acc={r['accuracy']:.4f}  F1={r['f1_macro']:.4f}  [{status} by {delta:+.4f}]")

        results[dataset] = ds_results

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for dataset in DATASETS:
        baseline_method, baseline_auroc = BASELINES[dataset]
        print(f"\n{dataset} (baseline: {baseline_method}={baseline_auroc:.4f}):")
        if dataset in results:
            for method, r in results[dataset].items():
                delta = r["auroc"] - baseline_auroc
                marker = ">>>" if delta > 0 else "   "
                print(f"  {marker} {method:30s}  AUROC={r['auroc']:.4f} ({delta:+.4f})  Acc={r['accuracy']:.4f}  F1={r['f1_macro']:.4f}")

    # Save results
    os.makedirs("/home/junyi/NIPS2026/fusion/results", exist_ok=True)
    with open("/home/junyi/NIPS2026/fusion/results/pilot_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to fusion/results/pilot_results.json")


if __name__ == "__main__":
    main()
