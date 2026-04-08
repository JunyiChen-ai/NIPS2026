"""
Layerwise Probe-Bank v2: All layers + direct logits + old score features.

Changes from v1:
- All layers (not every 2nd)
- PCA(512) for hidden/gen, PCA(256) for others
- Direct per-layer OOF logits as meta-features (not just trajectory summaries)
- Append old 7-probe OOF logits to meta-features
- Expanded C grid
"""

import os, json, time, warnings, gc, sys
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

EXTRACTION_DIR = "/home/junyi/NIPS2026/extraction/features"
PROCESSED_DIR = "/home/junyi/NIPS2026/reproduce/processed_features"

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

OLD_PROBE_METHODS = ["lr_probe", "pca_lr", "iti", "kb_mlp", "attn_satisfies", "sep", "step"]
N_FOLDS = 5
C_GRID_LAYER = [1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
C_GRID_META = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]


def load_labels(dataset, split):
    with open(os.path.join(EXTRACTION_DIR, dataset, split, "meta.json")) as f:
        return np.array(json.load(f)["labels"])


def compute_auroc(y_true, y_prob, n_classes):
    if n_classes == 2:
        return roc_auc_score(y_true, y_prob[:, 1])
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))
    return roc_auc_score(y_bin, y_prob, average="macro", multi_class="ovr")


def process_layer_bank(dataset, info):
    """Stage 1: Per-layer LR probes → OOF logits and test logits."""
    n_classes = info["n_classes"]
    splits = info["splits"]

    tr_labels = load_labels(dataset, splits["train"])
    va_labels = load_labels(dataset, splits["val"])
    te_labels = load_labels(dataset, splits["test"])
    trva_labels = np.concatenate([tr_labels, va_labels])
    n_trva = len(trva_labels)
    n_te = len(te_labels)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    sources = [
        ("input_hidden", "input_last_token_hidden", 512),   # PCA to 512
        ("gen_hidden", "gen_last_token_hidden", 512),
        ("head_act", "input_per_head_activation", 256),
        ("attn_stats", "input_attn_stats", None),  # 84d, no PCA
        ("attn_vnorms", "input_attn_value_norms", 256),
    ]

    all_oof_logits = []  # list of (n_trva, n_classes) arrays
    all_te_logits = []
    layer_names = []

    for source_name, raw_name, pca_dim in sources:
        t0 = time.time()
        tr_path = os.path.join(EXTRACTION_DIR, dataset, splits["train"], f"{raw_name}.pt")
        va_path = os.path.join(EXTRACTION_DIR, dataset, splits["val"], f"{raw_name}.pt")
        te_path = os.path.join(EXTRACTION_DIR, dataset, splits["test"], f"{raw_name}.pt")

        if not os.path.exists(tr_path):
            print(f"  {source_name}: not found")
            continue

        tr_raw = torch.load(tr_path, map_location="cpu")
        va_raw = torch.load(va_path, map_location="cpu")
        te_raw = torch.load(te_path, map_location="cpu")

        if not isinstance(tr_raw, torch.Tensor):
            print(f"  {source_name}: not tensor, skipping")
            continue

        tr_raw = tr_raw.float().numpy()
        va_raw = va_raw.float().numpy()
        te_raw = te_raw.float().numpy()
        trva_raw = np.concatenate([tr_raw, va_raw], axis=0)

        # Determine layer extraction
        if raw_name in ["input_last_token_hidden", "gen_last_token_hidden"]:
            n_layers = tr_raw.shape[1]
            get_layer = lambda data, l: data[:, l, :]
        elif raw_name == "input_per_head_activation":
            n_layers = tr_raw.shape[1]
            get_layer = lambda data, l: data[:, l, :, :].reshape(data.shape[0], -1)
        elif raw_name == "input_attn_stats":
            n_layers = tr_raw.shape[1]
            get_layer = lambda data, l: data[:, l, :, :].reshape(data.shape[0], -1)
        elif raw_name == "input_attn_value_norms":
            n_layers = tr_raw.shape[1]
            get_layer = lambda data, l: data[:, l, :, :].max(axis=-1)  # max pool over prompt_len
        else:
            continue

        # ALL layers (except head_act: every 2nd)
        if raw_name == "input_per_head_activation":
            layer_indices = list(range(0, n_layers, 2))
        else:
            layer_indices = list(range(n_layers))

        n_processed = 0
        for l in layer_indices:
            X_trva = get_layer(trva_raw, l)
            X_te = get_layer(te_raw, l)

            if X_trva.ndim == 1:
                X_trva = X_trva.reshape(-1, 1)
                X_te = X_te.reshape(-1, 1)

            # Standardize
            sc = StandardScaler()
            X_trva_s = sc.fit_transform(X_trva)
            X_te_s = sc.transform(X_te)

            # PCA if needed
            if pca_dim and X_trva_s.shape[1] > pca_dim:
                pca = PCA(n_components=pca_dim, random_state=42)
                X_trva_s = pca.fit_transform(X_trva_s)
                X_te_s = pca.transform(X_te_s)

            # Quick C selection via 80/20 split of train portion
            n_tr = len(tr_labels)
            sp = int(n_tr * 0.8)
            # Apply same pipeline to train-only
            X_tr_only = sc.transform(get_layer(tr_raw, l) if get_layer(tr_raw, l).ndim > 1 else get_layer(tr_raw, l).reshape(-1, 1))
            if pca_dim and X_tr_only.shape[1] > pca_dim:
                X_tr_only = pca.transform(X_tr_only)

            best_C_auroc, best_C = -1, 0.01
            for C in C_GRID_LAYER:
                clf = LogisticRegression(max_iter=2000, C=C, random_state=42)
                clf.fit(X_tr_only[:sp], tr_labels[:sp])
                vp = clf.predict_proba(X_tr_only[sp:n_tr])
                a = compute_auroc(tr_labels[sp:], vp, n_classes)
                if a > best_C_auroc:
                    best_C_auroc = a
                    best_C = C

            # OOF + test logits
            oof = np.zeros((n_trva, n_classes))
            te_avg = np.zeros((n_te, n_classes))
            for _, (tr_i, va_i) in enumerate(skf.split(X_trva_s, trva_labels)):
                clf = LogisticRegression(max_iter=2000, C=best_C, random_state=42)
                clf.fit(X_trva_s[tr_i], trva_labels[tr_i])
                oof[va_i] = clf.predict_proba(X_trva_s[va_i])
                te_avg += clf.predict_proba(X_te_s) / N_FOLDS

            all_oof_logits.append(oof)
            all_te_logits.append(te_avg)
            layer_names.append(f"{source_name}_L{l}")
            n_processed += 1

        elapsed = time.time() - t0
        print(f"  {source_name:15s}: {n_processed} layers [{elapsed:.1f}s]")

        del tr_raw, va_raw, te_raw, trva_raw
        gc.collect()

    return all_oof_logits, all_te_logits, layer_names, trva_labels, te_labels


def get_old_probe_oof(dataset, info):
    """Get OOF probs from old 7 probe methods (PCA(256) + C-tuned LR)."""
    n_classes = info["n_classes"]
    splits = info["splits"]

    tr_labels = load_labels(dataset, splits["train"])
    va_labels = load_labels(dataset, splits["val"])
    trva_labels = np.concatenate([tr_labels, va_labels])
    te_labels = load_labels(dataset, splits["test"])
    n_trva = len(trva_labels)
    n_te = len(te_labels)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    C_GRID_PROBE = [0.001, 0.01, 0.1, 1.0, 10.0]

    oof_list = []
    te_list = []
    probe_names = []

    for method in OLD_PROBE_METHODS:
        path = os.path.join(PROCESSED_DIR, dataset, method, "train.pt")
        if not os.path.exists(path):
            continue

        tr = torch.load(path, map_location="cpu").float().numpy()
        va = torch.load(os.path.join(PROCESSED_DIR, dataset, method, "val.pt"), map_location="cpu").float().numpy()
        te = torch.load(os.path.join(PROCESSED_DIR, dataset, method, "test.pt"), map_location="cpu").float().numpy()
        if tr.ndim == 1:
            tr, va, te = tr.reshape(-1, 1), va.reshape(-1, 1), te.reshape(-1, 1)

        trva = np.vstack([tr, va])
        sc = StandardScaler()
        trva_s = sc.fit_transform(trva)
        te_s = sc.transform(te)

        if trva_s.shape[1] > 256:
            pca = PCA(n_components=256, random_state=42)
            trva_s = pca.fit_transform(trva_s)
            te_s = pca.transform(te_s)

        # C selection
        n_tr = len(tr_labels)
        sp = int(n_tr * 0.8)
        tr_only_s = sc.transform(tr)
        if tr.shape[1] > 256:
            tr_only_s = pca.transform(tr_only_s)

        best_C_a, best_C = -1, 1.0
        for C in C_GRID_PROBE:
            clf = LogisticRegression(max_iter=2000, C=C, random_state=42)
            clf.fit(tr_only_s[:sp], tr_labels[:sp])
            vp = clf.predict_proba(tr_only_s[sp:n_tr])
            a = compute_auroc(tr_labels[sp:], vp, n_classes)
            if a > best_C_a:
                best_C_a = a
                best_C = C

        oof = np.zeros((n_trva, n_classes))
        te_avg = np.zeros((n_te, n_classes))
        for _, (tr_i, va_i) in enumerate(skf.split(trva_s, trva_labels)):
            clf = LogisticRegression(max_iter=2000, C=best_C, random_state=42)
            clf.fit(trva_s[tr_i], trva_labels[tr_i])
            oof[va_i] = clf.predict_proba(trva_s[va_i])
            te_avg += clf.predict_proba(te_s) / N_FOLDS

        oof_list.append(oof)
        te_list.append(te_avg)
        probe_names.append(f"probe_{method}")

    return oof_list, te_list, probe_names


def run_dataset(dataset, info):
    n_classes = info["n_classes"]
    baseline = BASELINES[dataset]

    print(f"\n{'='*60}")
    print(f"Dataset: {dataset} ({n_classes}-class, baseline={baseline:.4f})")
    print(f"{'='*60}")

    # Stage 1: Layer bank
    print("\nStage 1: Layer bank (all layers, PCA512/256)...")
    layer_oof, layer_te, layer_names, trva_labels, te_labels = process_layer_bank(dataset, info)
    print(f"  Total layer probes: {len(layer_names)}")

    # Stage 1b: Old probe features
    print("\nStage 1b: Old probe methods...")
    probe_oof, probe_te, probe_names = get_old_probe_oof(dataset, info)
    print(f"  Probe methods: {len(probe_names)}")

    # Stage 2: Direct logits as meta-features
    print("\nStage 2: Building meta-features...")

    # Combine: layer logits + probe logits (all direct, no trajectory compression)
    all_oof = layer_oof + probe_oof
    all_te = layer_te + probe_te
    all_names = layer_names + probe_names

    meta_oof = np.hstack(all_oof)  # (n_trva, n_probes * n_classes)
    meta_te = np.hstack(all_te)
    print(f"  Meta-features: {meta_oof.shape[1]} ({len(all_names)} probes × {n_classes} classes)")

    # Stage 3: Meta-classifier
    print("\nStage 3: Meta-classifier...")
    sc = StandardScaler()
    meta_oof_s = sc.fit_transform(meta_oof)
    meta_te_s = sc.transform(meta_te)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    best_auroc, best_C = -1, 0.01
    for C in C_GRID_META:
        inner_oof = np.zeros((len(trva_labels), n_classes))
        for _, (tr_i, va_i) in enumerate(skf.split(meta_oof_s, trva_labels)):
            clf = LogisticRegression(max_iter=3000, C=C, random_state=42)
            clf.fit(meta_oof_s[tr_i], trva_labels[tr_i])
            inner_oof[va_i] = clf.predict_proba(meta_oof_s[va_i])
        auroc = compute_auroc(trva_labels, inner_oof, n_classes)
        if auroc > best_auroc:
            best_auroc = auroc
            best_C = C

    print(f"  Best meta C={best_C}, inner AUROC={best_auroc:.4f}")

    clf_meta = LogisticRegression(max_iter=3000, C=best_C, random_state=42)
    clf_meta.fit(meta_oof_s, trva_labels)
    meta_te_prob = clf_meta.predict_proba(meta_te_s)

    test_auroc = compute_auroc(te_labels, meta_te_prob, n_classes)
    test_acc = accuracy_score(te_labels, meta_te_prob.argmax(axis=1))
    test_f1 = f1_score(te_labels, meta_te_prob.argmax(axis=1), average="macro")

    delta = test_auroc - baseline
    status = "PASS" if delta >= 0.03 else "FAIL"
    print(f"\n  Test: AUROC={test_auroc:.4f} ({delta:+.4f})  Acc={test_acc:.4f}  F1={test_f1:.4f}")
    print(f"  [{status}]")

    return {
        "auroc": test_auroc,
        "accuracy": test_acc,
        "f1_macro": test_f1,
        "delta": delta,
        "meta_C": best_C,
        "n_layer_probes": len(layer_names),
        "n_old_probes": len(probe_names),
        "n_meta_features": meta_oof.shape[1],
    }


def main():
    print("=" * 70)
    print("LAYERWISE PROBE-BANK v2 — All Layers + Direct Logits + Old Probes")
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
              f"delta={delta:+.4f}  [{status}]  ({r['n_layer_probes']}+{r['n_old_probes']} probes, "
              f"{r['n_meta_features']} feats)")

    if all_pass:
        print(f"\nALL TARGETS MET (+3%)")
    else:
        print(f"\nSome targets not met")

    os.makedirs("/home/junyi/NIPS2026/fusion/results", exist_ok=True)
    with open("/home/junyi/NIPS2026/fusion/results/layerwise_v2_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)


if __name__ == "__main__":
    main()
