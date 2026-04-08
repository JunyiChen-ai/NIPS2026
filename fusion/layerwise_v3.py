"""
Layerwise v3: Combine v1 trajectory features + v2 direct logits + old probes.
Uses subsampled layers (v1 speed) but includes both trajectory AND direct logits.
"""

import os, json, time, warnings, gc
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
C_GRID = [1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
C_GRID_META = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 1e-1, 1.0]


def load_labels(dataset, split):
    with open(os.path.join(EXTRACTION_DIR, dataset, split, "meta.json")) as f:
        return np.array(json.load(f)["labels"])

def compute_auroc(y_true, y_prob, n_classes):
    if n_classes == 2:
        return roc_auc_score(y_true, y_prob[:, 1])
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))
    return roc_auc_score(y_bin, y_prob, average="macro", multi_class="ovr")

def gaussian_basis(n_layers, n_basis=5, sigma_scale=0.3):
    centers = np.linspace(0, n_layers - 1, n_basis)
    sigma = sigma_scale * n_layers / n_basis
    basis = np.zeros((n_layers, n_basis))
    for i, c in enumerate(centers):
        basis[:, i] = np.exp(-0.5 * ((np.arange(n_layers) - c) / sigma) ** 2)
    basis /= basis.sum(axis=0, keepdims=True) + 1e-10
    return basis


def run_dataset(dataset, info):
    n_classes = info["n_classes"]
    splits = info["splits"]
    baseline = BASELINES[dataset]

    print(f"\n{'='*60}")
    print(f"Dataset: {dataset} ({n_classes}-class, baseline={baseline:.4f})")
    print(f"{'='*60}")

    tr_labels = load_labels(dataset, splits["train"])
    va_labels = load_labels(dataset, splits["val"])
    te_labels = load_labels(dataset, splits["test"])
    trva_labels = np.concatenate([tr_labels, va_labels])
    n_trva, n_te = len(trva_labels), len(te_labels)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    sources = [
        ("input_hidden", "input_last_token_hidden", 512),
        ("gen_hidden", "gen_last_token_hidden", 512),
        ("head_act", "input_per_head_activation", 256),
        ("attn_stats", "input_attn_stats", None),
        ("attn_vnorms", "input_attn_value_norms", 256),
    ]

    # Collect per-layer OOF logits for both direct meta-features and trajectory features
    source_layer_oof = {}  # {source: {layer: oof_logits}}
    source_layer_te = {}
    all_direct_oof = []
    all_direct_te = []
    all_names = []

    for source_name, raw_name, pca_dim in sources:
        t0 = time.time()
        tr_path = os.path.join(EXTRACTION_DIR, dataset, splits["train"], f"{raw_name}.pt")
        if not os.path.exists(tr_path):
            continue

        tr_raw = torch.load(tr_path, map_location="cpu").float().numpy()
        va_raw = torch.load(os.path.join(EXTRACTION_DIR, dataset, splits["val"], f"{raw_name}.pt"), map_location="cpu").float().numpy()
        te_raw = torch.load(os.path.join(EXTRACTION_DIR, dataset, splits["test"], f"{raw_name}.pt"), map_location="cpu").float().numpy()
        trva_raw = np.concatenate([tr_raw, va_raw], axis=0)

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
            get_layer = lambda data, l: data[:, l, :, :].max(axis=-1)
        else:
            continue

        # Layer sampling: every 2nd for hidden (30→15), every 4th for head_act (28→7), all for stats
        if raw_name == "input_per_head_activation":
            layer_indices = list(range(0, n_layers, 4))
        elif n_layers > 20:
            layer_indices = list(range(0, n_layers, 2))
        else:
            layer_indices = list(range(n_layers))

        source_layer_oof[source_name] = {}
        source_layer_te[source_name] = {}

        for l in layer_indices:
            X_trva = get_layer(trva_raw, l)
            X_te = get_layer(te_raw, l)
            if X_trva.ndim == 1:
                X_trva, X_te = X_trva.reshape(-1, 1), X_te.reshape(-1, 1)

            sc = StandardScaler()
            X_trva_s = sc.fit_transform(X_trva)
            X_te_s = sc.transform(X_te)

            if pca_dim and X_trva_s.shape[1] > pca_dim:
                pca = PCA(n_components=pca_dim, random_state=42)
                X_trva_s = pca.fit_transform(X_trva_s)
                X_te_s = pca.transform(X_te_s)

            # C selection
            n_tr = len(tr_labels)
            sp = int(n_tr * 0.8)
            tr_only = sc.transform(get_layer(tr_raw, l) if get_layer(tr_raw, l).ndim > 1 else get_layer(tr_raw, l).reshape(-1,1))
            if pca_dim and tr_only.shape[1] > pca_dim:
                tr_only = pca.transform(tr_only)

            best_a, best_C = -1, 0.01
            for C in C_GRID:
                clf = LogisticRegression(max_iter=2000, C=C, random_state=42)
                clf.fit(tr_only[:sp], tr_labels[:sp])
                vp = clf.predict_proba(tr_only[sp:n_tr])
                a = compute_auroc(tr_labels[sp:], vp, n_classes)
                if a > best_a:
                    best_a = a
                    best_C = C

            oof = np.zeros((n_trva, n_classes))
            te_avg = np.zeros((n_te, n_classes))
            for _, (tr_i, va_i) in enumerate(skf.split(X_trva_s, trva_labels)):
                clf = LogisticRegression(max_iter=2000, C=best_C, random_state=42)
                clf.fit(X_trva_s[tr_i], trva_labels[tr_i])
                oof[va_i] = clf.predict_proba(X_trva_s[va_i])
                te_avg += clf.predict_proba(X_te_s) / N_FOLDS

            source_layer_oof[source_name][l] = oof
            source_layer_te[source_name][l] = te_avg
            all_direct_oof.append(oof)
            all_direct_te.append(te_avg)
            all_names.append(f"{source_name}_L{l}")

        print(f"  {source_name:15s}: {len(layer_indices)} layers [{time.time()-t0:.1f}s]")
        del tr_raw, va_raw, te_raw, trva_raw
        gc.collect()

    # Build trajectory features from source_layer_oof/te
    print("\n  Building trajectory features...")
    traj_oof_parts = []
    traj_te_parts = []
    for source_name in source_layer_oof:
        layers = sorted(source_layer_oof[source_name].keys())
        n_layers = len(layers)
        if n_layers < 3:
            continue

        oof_stack = np.stack([source_layer_oof[source_name][l] for l in layers], axis=1)  # (N, L, K)
        te_stack = np.stack([source_layer_te[source_name][l] for l in layers], axis=1)

        basis = gaussian_basis(n_layers, n_basis=min(7, n_layers))

        for c in range(n_classes):
            traj_oof = oof_stack[:, :, c]
            traj_te = te_stack[:, :, c]

            traj_oof_parts.append(traj_oof @ basis)
            traj_te_parts.append(traj_te @ basis)

            for fn in [np.mean, np.max, np.std]:
                traj_oof_parts.append(fn(traj_oof, axis=1, keepdims=True))
                traj_te_parts.append(fn(traj_te, axis=1, keepdims=True))

            traj_oof_parts.append(traj_oof.argmax(axis=1, keepdims=True).astype(float) / n_layers)
            traj_te_parts.append(traj_te.argmax(axis=1, keepdims=True).astype(float) / n_layers)

    # Old probe features
    print("  Loading old probe features...")
    probe_oof_parts = []
    probe_te_parts = []
    for method in OLD_PROBE_METHODS:
        path = os.path.join(PROCESSED_DIR, dataset, method, "train.pt")
        if not os.path.exists(path):
            continue
        tr = torch.load(path, map_location="cpu").float().numpy()
        va = torch.load(os.path.join(PROCESSED_DIR, dataset, method, "val.pt"), map_location="cpu").float().numpy()
        te = torch.load(os.path.join(PROCESSED_DIR, dataset, method, "test.pt"), map_location="cpu").float().numpy()
        if tr.ndim == 1: tr, va, te = tr.reshape(-1,1), va.reshape(-1,1), te.reshape(-1,1)
        trva = np.vstack([tr, va])
        sc = StandardScaler()
        trva_s = sc.fit_transform(trva)
        te_s = sc.transform(te)
        if trva_s.shape[1] > 256:
            p = PCA(n_components=256, random_state=42)
            trva_s = p.fit_transform(trva_s)
            te_s = p.transform(te_s)

        n_tr = len(tr_labels)
        sp = int(n_tr * 0.8)
        tr_only = sc.transform(tr)
        if tr.shape[1] > 256:
            tr_only = p.transform(tr_only)
        best_a, best_C = -1, 1.0
        for C in [0.001, 0.01, 0.1, 1.0, 10.0]:
            clf = LogisticRegression(max_iter=2000, C=C, random_state=42)
            clf.fit(tr_only[:sp], tr_labels[:sp])
            vp = clf.predict_proba(tr_only[sp:n_tr])
            a = compute_auroc(tr_labels[sp:], vp, n_classes)
            if a > best_a: best_a, best_C = a, C

        oof = np.zeros((n_trva, n_classes))
        te_avg = np.zeros((n_te, n_classes))
        for _, (tr_i, va_i) in enumerate(skf.split(trva_s, trva_labels)):
            clf = LogisticRegression(max_iter=2000, C=best_C, random_state=42)
            clf.fit(trva_s[tr_i], trva_labels[tr_i])
            oof[va_i] = clf.predict_proba(trva_s[va_i])
            te_avg += clf.predict_proba(te_s) / N_FOLDS
        probe_oof_parts.append(oof)
        probe_te_parts.append(te_avg)

    # Combine ALL features: direct logits + trajectory + old probes
    meta_oof = np.hstack(all_direct_oof + traj_oof_parts + probe_oof_parts)
    meta_te = np.hstack(all_direct_te + traj_te_parts + probe_te_parts)
    print(f"  Total meta-features: {meta_oof.shape[1]} "
          f"(direct={len(all_direct_oof)*n_classes}, "
          f"traj={sum(p.shape[1] for p in traj_oof_parts)}, "
          f"probes={len(probe_oof_parts)*n_classes})")

    # Meta-classifier
    print("\n  Meta-classifier...")
    sc_meta = StandardScaler()
    meta_oof_s = sc_meta.fit_transform(meta_oof)
    meta_te_s = sc_meta.transform(meta_te)

    best_auroc, best_C = -1, 0.01
    for C in C_GRID_META:
        inner_oof = np.zeros((n_trva, n_classes))
        for _, (tr_i, va_i) in enumerate(skf.split(meta_oof_s, trva_labels)):
            clf = LogisticRegression(max_iter=3000, C=C, random_state=42)
            clf.fit(meta_oof_s[tr_i], trva_labels[tr_i])
            inner_oof[va_i] = clf.predict_proba(meta_oof_s[va_i])
        auroc = compute_auroc(trva_labels, inner_oof, n_classes)
        if auroc > best_auroc:
            best_auroc = auroc
            best_C = C

    print(f"  Best meta C={best_C}, inner AUROC={best_auroc:.4f}")

    clf_final = LogisticRegression(max_iter=3000, C=best_C, random_state=42)
    clf_final.fit(meta_oof_s, trva_labels)
    te_prob = clf_final.predict_proba(meta_te_s)

    test_auroc = compute_auroc(te_labels, te_prob, n_classes)
    test_acc = accuracy_score(te_labels, te_prob.argmax(axis=1))
    test_f1 = f1_score(te_labels, te_prob.argmax(axis=1), average="macro")
    delta = test_auroc - baseline

    print(f"\n  Test: AUROC={test_auroc:.4f} ({delta:+.4f})  Acc={test_acc:.4f}  F1={test_f1:.4f}")
    print(f"  [{'PASS' if delta >= 0.03 else 'FAIL'}]")

    return {"auroc": test_auroc, "accuracy": test_acc, "f1_macro": test_f1, "delta": delta,
            "meta_C": best_C, "n_meta_features": meta_oof.shape[1]}


def main():
    print("=" * 70)
    print("LAYERWISE v3 — Direct Logits + Trajectory + Old Probes Combined")
    print("=" * 70)

    all_results = {}
    for dataset, info in DATASETS.items():
        all_results[dataset] = run_dataset(dataset, info)
        gc.collect()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    all_pass = True
    for ds in DATASETS:
        bl = BASELINES[ds]
        r = all_results[ds]
        status = "PASS" if r["delta"] >= 0.03 else "FAIL"
        if r["delta"] < 0.03: all_pass = False
        print(f"  {ds:25s}  baseline={bl:.4f}  ours={r['auroc']:.4f}  delta={r['delta']:+.4f}  [{status}]")

    if all_pass:
        print("\nALL TARGETS MET (+3%)")
    else:
        print("\nSome targets not met")

    os.makedirs("/home/junyi/NIPS2026/fusion/results", exist_ok=True)
    with open("/home/junyi/NIPS2026/fusion/results/layerwise_v3_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

if __name__ == "__main__":
    main()
