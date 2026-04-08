"""
Layerwise Probe-Bank Stacking with Depth Trajectory Fusion.

Stage 1: Train LR probe per (source, layer) → get OOF logits
Stage 2: Compress layer logits into depth trajectory features (RBF windows, mean, max, std)
Stage 3: Meta-LR on trajectory features
Stage 4: Anchor blending for robustness
"""

import os, json, time, warnings, gc
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from scipy.special import softmax

warnings.filterwarnings("ignore")

EXTRACTION_DIR = "/home/junyi/NIPS2026/extraction/features"

DATASETS = {
    "common_claim_3class": {"n_classes": 3, "splits": {"train": "train", "val": "val", "test": "test"}},
    "e2h_amc_3class":      {"n_classes": 3, "splits": {"train": "train_sub", "val": "val_split", "test": "eval"}},
    "e2h_amc_5class":      {"n_classes": 5, "splits": {"train": "train_sub", "val": "val_split", "test": "eval"}},
    "when2call_3class":     {"n_classes": 3, "splits": {"train": "train", "val": "val", "test": "test"}},
}

BASELINES = {
    "common_claim_3class": 0.7576,
    "e2h_amc_3class": 0.8934,
    "e2h_amc_5class": 0.8752,
    "when2call_3class": 0.8741,
}

N_FOLDS = 5
C_GRID = [1e-3, 1e-2, 1e-1, 1.0]


def load_labels(dataset, split):
    with open(os.path.join(EXTRACTION_DIR, dataset, split, "meta.json")) as f:
        return np.array(json.load(f)["labels"])


def load_raw_source(dataset, split, source_name):
    """Load a raw feature tensor."""
    path = os.path.join(EXTRACTION_DIR, dataset, split, f"{source_name}.pt")
    if not os.path.exists(path):
        return None
    t = torch.load(path, map_location="cpu")
    if isinstance(t, torch.Tensor):
        return t.float().numpy()
    return None


def compute_auroc(y_true, y_prob, n_classes):
    if n_classes == 2:
        return roc_auc_score(y_true, y_prob[:, 1])
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))
    return roc_auc_score(y_bin, y_prob, average="macro", multi_class="ovr")


def gaussian_basis(n_layers, n_basis=5, sigma_scale=0.3):
    """Generate Gaussian RBF basis functions over layer indices."""
    centers = np.linspace(0, n_layers - 1, n_basis)
    sigma = sigma_scale * n_layers / n_basis
    basis = np.zeros((n_layers, n_basis))
    for i, c in enumerate(centers):
        basis[:, i] = np.exp(-0.5 * ((np.arange(n_layers) - c) / sigma) ** 2)
    # Normalize each basis to sum to 1
    basis /= basis.sum(axis=0, keepdims=True) + 1e-10
    return basis  # (n_layers, n_basis)


def train_layer_probe(X_train, y_train, X_test, n_classes, C_grid=C_GRID):
    """Train LR on one layer's features, with C selection via train split."""
    sc = StandardScaler()
    X_tr = sc.fit_transform(X_train)
    X_te = sc.transform(X_test)

    # Quick C selection: use last 20% of train as holdout
    n = len(y_train)
    split_idx = int(n * 0.8)
    X_tr_sub, X_val_sub = X_tr[:split_idx], X_tr[split_idx:]
    y_tr_sub, y_val_sub = y_train[:split_idx], y_train[split_idx:]

    best_auroc, best_C = -1, 0.01
    for C in C_grid:
        clf = LogisticRegression(max_iter=2000, C=C, random_state=42, solver="lbfgs")
        clf.fit(X_tr_sub, y_tr_sub)
        val_prob = clf.predict_proba(X_val_sub)
        auroc = compute_auroc(y_val_sub, val_prob, n_classes)
        if auroc > best_auroc:
            best_auroc = auroc
            best_C = C

    # Refit on full train
    clf = LogisticRegression(max_iter=2000, C=best_C, random_state=42, solver="lbfgs")
    clf.fit(X_tr, y_train)
    return clf.predict_proba(X_te), best_C, best_auroc


def run_layerwise_probes(dataset, info):
    """Stage 1: Train per-layer probes, collect logits."""
    n_classes = info["n_classes"]
    splits = info["splits"]

    tr_labels = load_labels(dataset, splits["train"])
    va_labels = load_labels(dataset, splits["val"])
    te_labels = load_labels(dataset, splits["test"])
    trva_labels = np.concatenate([tr_labels, va_labels])

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    # Sources to process: (name, how to extract per-layer features)
    # Each source yields a dict of {layer_idx: (train_feats, val_feats, test_feats)}
    sources_config = [
        ("input_hidden", "input_last_token_hidden"),   # (N, 30, 3584)
        ("gen_hidden", "gen_last_token_hidden"),         # (N, 30, 3584)
        ("head_act", "input_per_head_activation"),       # (N, 28, 28, 128)
        ("attn_stats", "input_attn_stats"),              # (N, 28, 28, 3)
    ]

    all_layer_logits = {}  # {source_layer: {"oof": (N_trva, K), "test": (N_te, K)}}
    layer_aurocs = {}

    for source_name, raw_name in sources_config:
        t0 = time.time()
        tr_raw = load_raw_source(dataset, splits["train"], raw_name)
        va_raw = load_raw_source(dataset, splits["val"], raw_name)
        te_raw = load_raw_source(dataset, splits["test"], raw_name)

        if tr_raw is None:
            print(f"  {source_name}: not found, skipping")
            continue

        # Determine layer structure
        if raw_name in ["input_last_token_hidden", "gen_last_token_hidden"]:
            # (N, n_layers, hidden_dim)
            n_layers = tr_raw.shape[1]
            def get_layer(data, l):
                return data[:, l, :]
        elif raw_name == "input_per_head_activation":
            # (N, n_layers, n_heads, head_dim) → flatten heads per layer
            n_layers = tr_raw.shape[1]
            def get_layer(data, l):
                return data[:, l, :, :].reshape(data.shape[0], -1)
        elif raw_name == "input_attn_stats":
            # (N, n_layers, n_heads, 3) → flatten per layer
            n_layers = tr_raw.shape[1]
            def get_layer(data, l):
                return data[:, l, :, :].reshape(data.shape[0], -1)
        else:
            continue

        # Sample layers: every 2nd for >20 layers, every 3rd for head_act (expensive)
        if raw_name == "input_per_head_activation":
            layer_indices = list(range(0, n_layers, 4))  # every 4th (28 heads × 128d = 3584d)
        elif n_layers > 20:
            layer_indices = list(range(0, n_layers, 2))
        else:
            layer_indices = list(range(n_layers))

        trva_raw = np.concatenate([tr_raw, va_raw], axis=0)

        source_layer_count = 0
        for l in layer_indices:
            key = f"{source_name}_L{l}"

            # Get layer features
            X_trva = get_layer(trva_raw, l)
            X_te = get_layer(te_raw, l)

            # Standardize + PCA if needed
            sc = StandardScaler()
            X_trva_s = sc.fit_transform(X_trva)
            X_te_s = sc.transform(X_te)

            # PCA to 256d if high-dim
            pca_dim = 256
            if X_trva_s.shape[1] > pca_dim:
                from sklearn.decomposition import PCA
                pca = PCA(n_components=pca_dim, random_state=42)
                X_trva_s = pca.fit_transform(X_trva_s)
                X_te_s = pca.transform(X_te_s)

            # C selection via holdout within train portion
            n = len(tr_labels)
            split_idx = int(n * 0.8)
            # Reconstruct train-only features through same pipeline
            tr_layer = get_layer(tr_raw, l)
            tr_layer_s = sc.transform(tr_layer)
            if tr_layer_s.shape[1] > pca_dim:
                tr_layer_s = pca.transform(tr_layer_s)
            X_tr_sub = tr_layer_s[:split_idx]
            X_val_sub = tr_layer_s[split_idx:n]
            y_tr_sub = tr_labels[:split_idx]
            y_val_sub = tr_labels[split_idx:]

            best_C_auroc, best_C = -1, 0.01
            for C in C_GRID:
                clf = LogisticRegression(max_iter=2000, C=C, random_state=42)
                clf.fit(X_tr_sub, y_tr_sub)
                vp = clf.predict_proba(X_val_sub)
                a = compute_auroc(y_val_sub, vp, n_classes)
                if a > best_C_auroc:
                    best_C_auroc = a
                    best_C = C

            # OOF logits
            oof_logits = np.zeros((len(trva_labels), n_classes))
            te_logits = np.zeros((len(te_labels), n_classes))
            fold_aurocs = []

            for _, (tr_i, va_i) in enumerate(skf.split(X_trva_s, trva_labels)):
                clf = LogisticRegression(max_iter=2000, C=best_C, random_state=42)
                clf.fit(X_trva_s[tr_i], trva_labels[tr_i])
                oof_logits[va_i] = clf.predict_proba(X_trva_s[va_i])
                te_logits += clf.predict_proba(X_te_s) / N_FOLDS
                fold_aurocs.append(compute_auroc(trva_labels[va_i], oof_logits[va_i], n_classes))

            mean_auroc = np.mean(fold_aurocs)
            all_layer_logits[key] = {"oof": oof_logits, "test": te_logits}
            layer_aurocs[key] = mean_auroc
            source_layer_count += 1

        elapsed = time.time() - t0
        print(f"  {source_name:15s}: {source_layer_count} layers, dim={get_layer(tr_raw, 0).shape[1]}, "
              f"[{elapsed:.1f}s]")

        # Free memory
        del tr_raw, va_raw, te_raw, trva_raw
        gc.collect()

    return all_layer_logits, layer_aurocs, trva_labels, te_labels


def build_trajectory_features(layer_logits, source_prefix, n_classes, n_basis=5):
    """Stage 2: Convert per-layer logits to depth trajectory features."""
    # Find all layers for this source
    layers = sorted([k for k in layer_logits if k.startswith(source_prefix)],
                    key=lambda k: int(k.split("_L")[1]))
    if not layers:
        return None, None
    n_layers = len(layers)

    # Stack logits: (N, n_layers, K)
    oof_stack = np.stack([layer_logits[k]["oof"] for k in layers], axis=1)
    te_stack = np.stack([layer_logits[k]["test"] for k in layers], axis=1)

    # Gaussian RBF basis
    basis = gaussian_basis(n_layers, n_basis=min(n_basis, n_layers))  # (n_layers, n_basis)

    features_oof = []
    features_te = []

    for c in range(n_classes):
        # Logit trajectory for class c: (N, n_layers)
        traj_oof = oof_stack[:, :, c]
        traj_te = te_stack[:, :, c]

        # RBF features: (N, n_basis)
        rbf_oof = traj_oof @ basis
        rbf_te = traj_te @ basis
        features_oof.append(rbf_oof)
        features_te.append(rbf_te)

        # Summary stats
        features_oof.append(traj_oof.mean(axis=1, keepdims=True))
        features_te.append(traj_te.mean(axis=1, keepdims=True))

        features_oof.append(traj_oof.max(axis=1, keepdims=True))
        features_te.append(traj_te.max(axis=1, keepdims=True))

        features_oof.append(traj_oof.std(axis=1, keepdims=True))
        features_te.append(traj_te.std(axis=1, keepdims=True))

        # Argmax depth (normalized)
        features_oof.append(traj_oof.argmax(axis=1, keepdims=True).astype(float) / n_layers)
        features_te.append(traj_te.argmax(axis=1, keepdims=True).astype(float) / n_layers)

    return np.hstack(features_oof), np.hstack(features_te)


def run_dataset(dataset, info):
    n_classes = info["n_classes"]
    baseline = BASELINES[dataset]

    print(f"\n{'='*60}")
    print(f"Dataset: {dataset} ({n_classes}-class, baseline={baseline:.4f})")
    print(f"{'='*60}")

    # Stage 1: Layerwise probes
    print("\nStage 1: Layerwise probe bank...")
    layer_logits, layer_aurocs, trva_labels, te_labels = run_layerwise_probes(dataset, info)

    # Print top 10 layers by AUROC
    top_layers = sorted(layer_aurocs.items(), key=lambda x: -x[1])[:10]
    print(f"\nTop 10 layers by CV AUROC:")
    for k, a in top_layers:
        print(f"  {k:25s}  AUROC={a:.4f}")

    # Stage 2: Trajectory features
    print(f"\nStage 2: Depth trajectory features...")
    sources = ["input_hidden", "gen_hidden", "head_act", "attn_stats"]
    traj_parts_oof = []
    traj_parts_te = []
    for src in sources:
        feat_oof, feat_te = build_trajectory_features(layer_logits, src, n_classes)
        if feat_oof is not None:
            traj_parts_oof.append(feat_oof)
            traj_parts_te.append(feat_te)
            print(f"  {src:15s}: {feat_oof.shape[1]} features")

    # Also add the raw OOF probs from top-K individual layers
    top_k_layers = [k for k, _ in sorted(layer_aurocs.items(), key=lambda x: -x[1])[:10]]
    for k in top_k_layers:
        traj_parts_oof.append(layer_logits[k]["oof"])
        traj_parts_te.append(layer_logits[k]["test"])

    meta_oof = np.hstack(traj_parts_oof)
    meta_te = np.hstack(traj_parts_te)
    print(f"\nTotal meta-features: {meta_oof.shape[1]}")

    # Stage 3: Meta-classifier
    print(f"\nStage 3: Meta-classifier...")
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    # Standardize
    sc = StandardScaler()
    meta_oof_s = sc.fit_transform(meta_oof)
    meta_te_s = sc.transform(meta_te)

    # C selection
    best_auroc, best_C = -1, 0.1
    for C in [0.001, 0.01, 0.1, 1.0, 10.0]:
        inner_oof = np.zeros((len(trva_labels), n_classes))
        for _, (tr_i, va_i) in enumerate(skf.split(meta_oof_s, trva_labels)):
            clf = LogisticRegression(max_iter=2000, C=C, random_state=42)
            clf.fit(meta_oof_s[tr_i], trva_labels[tr_i])
            inner_oof[va_i] = clf.predict_proba(meta_oof_s[va_i])
        auroc = compute_auroc(trva_labels, inner_oof, n_classes)
        if auroc > best_auroc:
            best_auroc = auroc
            best_C = C

    print(f"  Best meta C={best_C}, OOF AUROC={best_auroc:.4f}")

    # Final meta-classifier
    clf_meta = LogisticRegression(max_iter=2000, C=best_C, random_state=42)
    clf_meta.fit(meta_oof_s, trva_labels)
    meta_te_prob = clf_meta.predict_proba(meta_te_s)

    # Stage 4: Anchor blending
    # Best single layer as anchor
    best_layer_key = top_layers[0][0]
    anchor_oof = layer_logits[best_layer_key]["oof"]
    anchor_te = layer_logits[best_layer_key]["test"]

    # OOF stacking predictions for blending
    stacking_oof = np.zeros((len(trva_labels), n_classes))
    for _, (tr_i, va_i) in enumerate(skf.split(meta_oof_s, trva_labels)):
        clf = LogisticRegression(max_iter=2000, C=best_C, random_state=42)
        clf.fit(meta_oof_s[tr_i], trva_labels[tr_i])
        stacking_oof[va_i] = clf.predict_proba(meta_oof_s[va_i])

    best_blend_auroc, best_alpha = -1, 1.0
    for alpha in np.arange(0.0, 1.05, 0.1):
        blended = alpha * stacking_oof + (1 - alpha) * anchor_oof
        auroc = compute_auroc(trva_labels, blended, n_classes)
        if auroc > best_blend_auroc:
            best_blend_auroc = auroc
            best_alpha = alpha

    test_final = best_alpha * meta_te_prob + (1 - best_alpha) * anchor_te
    test_auroc = compute_auroc(te_labels, test_final, n_classes)
    test_acc = accuracy_score(te_labels, test_final.argmax(axis=1))
    test_f1 = f1_score(te_labels, test_final.argmax(axis=1), average="macro")

    delta = test_auroc - baseline
    print(f"\n  Meta-only test AUROC:  {compute_auroc(te_labels, meta_te_prob, n_classes):.4f}")
    print(f"  Anchor-only test AUROC: {compute_auroc(te_labels, anchor_te, n_classes):.4f}")
    print(f"  Blended (α={best_alpha:.1f}) test AUROC: {test_auroc:.4f} ({delta:+.4f})")
    print(f"  Acc={test_acc:.4f}  F1={test_f1:.4f}")
    target = "+3%" if delta >= 0.03 else "NOT +3%"
    print(f"  [{target}]")

    return {
        "auroc": test_auroc,
        "accuracy": test_acc,
        "f1_macro": test_f1,
        "delta": delta,
        "meta_C": best_C,
        "blend_alpha": best_alpha,
        "n_layer_probes": len(layer_logits),
        "n_meta_features": meta_oof.shape[1],
    }


def main():
    print("=" * 70)
    print("LAYERWISE PROBE-BANK STACKING")
    print("=" * 70)

    all_results = {}
    for dataset, info in DATASETS.items():
        all_results[dataset] = run_dataset(dataset, info)
        gc.collect()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    all_pass = True
    for ds in DATASETS:
        bl = BASELINES[ds]
        r = all_results[ds]
        delta = r["delta"]
        status = "PASS" if delta >= 0.03 else "FAIL"
        if delta < 0.03:
            all_pass = False
        print(f"  {ds:25s}  baseline={bl:.4f}  ours={r['auroc']:.4f}  "
              f"delta={delta:+.4f}  [{status}]  (α={r['blend_alpha']:.1f}, {r['n_layer_probes']} probes)")

    print(f"\n{'ALL TARGETS MET' if all_pass else 'Some targets not met'}")

    os.makedirs("/home/junyi/NIPS2026/fusion/results", exist_ok=True)
    with open("/home/junyi/NIPS2026/fusion/results/layerwise_fusion_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)


if __name__ == "__main__":
    main()
