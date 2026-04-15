"""
Baseline-Only Fusion v2: Richer feature extraction from baseline processed features.

Key improvements over v1:
1. Multi-granularity PCA: For high-dim features, create probes at multiple PCA levels
   (32, 64, 128, 256) — each captures different signal scales
2. Feature-level fusion path: PCA-reduce each method's features to common dim,
   concatenate, then meta-LR (feature-level, not just score-level)
3. Combined: both OOF probability stacking AND feature concatenation
4. Better C grid search with cross-validation instead of holdout split
"""

import os, json, time, warnings, gc
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from scipy import stats

warnings.filterwarnings("ignore")

PROCESSED_DIR = "/home/junyi/NIPS2026/reproduce/processed_features"
EXTRACTION_DIR = "/home/junyi/NIPS2026/extraction/features"
RESULTS_DIR = "/home/junyi/NIPS2026/fusion/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

FOCUS_DATASETS = {
    "common_claim_3class": {
        "n_classes": 3, "ext": "common_claim_3class",
        "splits": {"train": "train", "val": "val", "test": "test"},
        "best_single": 0.7576, "best_method": "PCA+LR",
    },
    "e2h_amc_3class": {
        "n_classes": 3, "ext": "e2h_amc_3class",
        "splits": {"train": "train_sub", "val": "val_split", "test": "eval"},
        "best_single": 0.8934, "best_method": "PCA+LR",
    },
    "e2h_amc_5class": {
        "n_classes": 5, "ext": "e2h_amc_5class",
        "splits": {"train": "train_sub", "val": "val_split", "test": "eval"},
        "best_single": 0.8752, "best_method": "KB MLP",
    },
    "when2call_3class": {
        "n_classes": 3, "ext": "when2call_3class",
        "splits": {"train": "train", "val": "val", "test": "test"},
        "best_single": 0.8741, "best_method": "LR Probe",
    },
    "ragtruth_binary": {
        "n_classes": 2, "ext": "ragtruth",
        "splits": {"train": "train", "val": "val", "test": "test"},
        "best_single": 0.8808, "best_method": "ITI",
    },
}

MULTICLASS_METHODS = ["lr_probe", "pca_lr", "iti", "kb_mlp", "attn_satisfies", "sep", "step"]
BINARY_METHODS = MULTICLASS_METHODS + ["mm_probe", "lid", "llm_check", "seakr"]

N_FOLDS = 5
C_GRID = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0, 10.0]


def load_labels(ext, split):
    with open(os.path.join(EXTRACTION_DIR, ext, split, "meta.json")) as f:
        return np.array(json.load(f)["labels"])


def compute_auroc(y, p, nc):
    if nc == 2:
        return roc_auc_score(y, p[:, 1])
    yb = label_binarize(y, classes=list(range(nc)))
    return roc_auc_score(yb, p, average="macro", multi_class="ovr")


def bootstrap_ci(y, p, nc, n_boot=2000):
    n = len(y)
    rng = np.random.RandomState(42)
    scores = []
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        try:
            scores.append(compute_auroc(y[idx], p[idx], nc))
        except:
            pass
    scores = sorted(scores)
    return scores[int(0.025 * len(scores))], scores[int(0.975 * len(scores))]


def load_method_features(ds_name, method):
    base = os.path.join(PROCESSED_DIR, ds_name, method)
    result = {}
    for split in ["train", "val", "test"]:
        path = os.path.join(base, f"{split}.pt")
        if not os.path.exists(path):
            return None
        t = torch.load(path, map_location="cpu").float().numpy()
        if t.ndim == 1:
            t = t.reshape(-1, 1)
        result[split] = t
    return result


def oof_probe(X_trva, trva_labels, X_te, nc, skf, C_grid=C_GRID):
    """Train OOF LR probe with C selection via nested CV."""
    n_trva = len(trva_labels)
    # Nested CV for C selection
    best_au, best_C = -1, 1.0
    for C in C_grid:
        inner = np.zeros((n_trva, nc))
        for _, (ti, vi) in enumerate(skf.split(X_trva, trva_labels)):
            clf = LogisticRegression(max_iter=2000, C=C, random_state=42)
            clf.fit(X_trva[ti], trva_labels[ti])
            inner[vi] = clf.predict_proba(X_trva[vi])
        try:
            au = compute_auroc(trva_labels, inner, nc)
        except:
            au = 0.5
        if au > best_au:
            best_au, best_C = au, C

    # Final OOF + test
    oof = np.zeros((n_trva, nc))
    ta = np.zeros((len(X_te), nc))
    for _, (ti, vi) in enumerate(skf.split(X_trva, trva_labels)):
        clf = LogisticRegression(max_iter=2000, C=best_C, random_state=42)
        clf.fit(X_trva[ti], trva_labels[ti])
        oof[vi] = clf.predict_proba(X_trva[vi])
        ta += clf.predict_proba(X_te) / N_FOLDS

    return oof, ta, best_C, best_au


def run_dataset(ds_name, info):
    nc = info["n_classes"]
    sp = info["splits"]
    ext = info["ext"]
    methods = BINARY_METHODS if nc == 2 else MULTICLASS_METHODS

    tr_labels = load_labels(ext, sp["train"])
    va_labels = load_labels(ext, sp["val"])
    te_labels = load_labels(ext, sp["test"])
    trva_labels = np.concatenate([tr_labels, va_labels])
    n_tr, n_trva, n_te = len(tr_labels), len(trva_labels), len(te_labels)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    all_oof = []
    all_te = []
    all_names = []

    # === Strategy 1: Multi-granularity OOF probes ===
    # For each method, create probes at multiple PCA levels
    PCA_LEVELS = [16, 32, 64, 128, 256]

    for method in methods:
        feats = load_method_features(ds_name, method)
        if feats is None:
            continue

        tr, va, te = feats["train"], feats["val"], feats["test"]
        trva = np.vstack([tr, va])
        orig_dim = trva.shape[1]

        sc = StandardScaler()
        trvas = sc.fit_transform(trva)
        tes = sc.transform(te)

        if orig_dim <= 2:
            # Scalar features — just one probe
            oof, ta, best_C, cv_au = oof_probe(trvas, trva_labels, tes, nc, skf)
            all_oof.append(oof)
            all_te.append(ta)
            all_names.append(f"{method}_raw")
            print(f"    {method:20s}: dim={orig_dim:>6d} → 1 probe, C={best_C:.4f}, cv={cv_au:.4f}")
        else:
            # Multi-granularity: probe at multiple PCA levels + full dim if small enough
            probes_added = 0
            for pca_dim in PCA_LEVELS:
                if pca_dim >= orig_dim:
                    continue
                actual_dim = min(pca_dim, trvas.shape[0] - 1)
                pca = PCA(n_components=actual_dim, random_state=42)
                Xp = pca.fit_transform(trvas)
                Xtp = pca.transform(tes)

                oof, ta, best_C, cv_au = oof_probe(Xp, trva_labels, Xtp, nc, skf)
                all_oof.append(oof)
                all_te.append(ta)
                all_names.append(f"{method}_pca{pca_dim}")
                probes_added += 1

            # Also probe at full dim if <= 256
            if orig_dim <= 256:
                oof, ta, best_C, cv_au = oof_probe(trvas, trva_labels, tes, nc, skf)
                all_oof.append(oof)
                all_te.append(ta)
                all_names.append(f"{method}_full")
                probes_added += 1

            print(f"    {method:20s}: dim={orig_dim:>6d} → {probes_added} probes")

    # === Strategy 2: Feature-level concatenation ===
    # PCA each method to a common dim, concatenate, single LR
    CONCAT_DIM = 32  # per method
    feat_parts_trva = []
    feat_parts_te = []
    feat_names = []

    for method in methods:
        feats = load_method_features(ds_name, method)
        if feats is None:
            continue

        tr, va, te = feats["train"], feats["val"], feats["test"]
        trva = np.vstack([tr, va])
        orig_dim = trva.shape[1]

        sc = StandardScaler()
        trvas = sc.fit_transform(trva)
        tes_s = sc.transform(te)

        if orig_dim > CONCAT_DIM:
            actual = min(CONCAT_DIM, trvas.shape[0] - 1)
            pca = PCA(n_components=actual, random_state=42)
            trvas = pca.fit_transform(trvas)
            tes_s = pca.transform(tes_s)

        feat_parts_trva.append(trvas)
        feat_parts_te.append(tes_s)
        feat_names.append(method)

    if feat_parts_trva:
        X_concat_trva = np.hstack(feat_parts_trva)
        X_concat_te = np.hstack(feat_parts_te)

        sc2 = StandardScaler()
        X_concat_trva = sc2.fit_transform(X_concat_trva)
        X_concat_te = sc2.transform(X_concat_te)

        oof, ta, best_C, cv_au = oof_probe(X_concat_trva, trva_labels, X_concat_te, nc, skf)
        all_oof.append(oof)
        all_te.append(ta)
        all_names.append("feat_concat_32")
        print(f"    {'feat_concat_32':20s}: dim={X_concat_trva.shape[1]:>6d} → 1 probe, cv={cv_au:.4f}")

    # Also try larger concat dim
    for cdim in [64, 128]:
        feat_parts_trva2 = []
        feat_parts_te2 = []
        for method in methods:
            feats = load_method_features(ds_name, method)
            if feats is None:
                continue
            tr, va, te = feats["train"], feats["val"], feats["test"]
            trva = np.vstack([tr, va])
            sc = StandardScaler()
            trvas = sc.fit_transform(trva)
            tes_s = sc.transform(te)
            if trvas.shape[1] > cdim:
                actual = min(cdim, trvas.shape[0] - 1)
                pca = PCA(n_components=actual, random_state=42)
                trvas = pca.fit_transform(trvas)
                tes_s = pca.transform(tes_s)
            feat_parts_trva2.append(trvas)
            feat_parts_te2.append(tes_s)

        if feat_parts_trva2:
            Xc = np.hstack(feat_parts_trva2)
            Xct = np.hstack(feat_parts_te2)
            sc3 = StandardScaler()
            Xc = sc3.fit_transform(Xc)
            Xct = sc3.transform(Xct)
            oof, ta, best_C, cv_au = oof_probe(Xc, trva_labels, Xct, nc, skf)
            all_oof.append(oof)
            all_te.append(ta)
            all_names.append(f"feat_concat_{cdim}")
            print(f"    {'feat_concat_'+str(cdim):20s}: dim={Xc.shape[1]:>6d} → 1 probe, cv={cv_au:.4f}")

    if not all_oof:
        return None

    # === META-CLASSIFIER ===
    meta_oof = np.hstack(all_oof)
    meta_te = np.hstack(all_te)
    n_meta = meta_oof.shape[1]
    print(f"    Total meta-features: {n_meta} from {len(all_names)} sources")

    sc_m = StandardScaler()
    mo = sc_m.fit_transform(meta_oof)
    mt = sc_m.transform(meta_te)

    # Meta-LR with nested CV
    meta_C_grid = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0]
    best_au, best_C = -1, 0.01
    for C in meta_C_grid:
        inner = np.zeros((n_trva, nc))
        for _, (ti, vi) in enumerate(skf.split(mo, trva_labels)):
            clf = LogisticRegression(max_iter=3000, C=C, random_state=42)
            clf.fit(mo[ti], trva_labels[ti])
            inner[vi] = clf.predict_proba(mo[vi])
        try:
            au = compute_auroc(trva_labels, inner, nc)
        except:
            au = 0.5
        if au > best_au:
            best_au, best_C = au, C

    clf_f = LogisticRegression(max_iter=3000, C=best_C, random_state=42)
    clf_f.fit(mo, trva_labels)
    te_prob = clf_f.predict_proba(mt)

    auroc = compute_auroc(te_labels, te_prob, nc)
    ci_lo, ci_hi = bootstrap_ci(te_labels, te_prob, nc)
    baseline = info["best_single"]
    delta = auroc - baseline

    print(f"    Meta C={best_C:.5f}, meta-cv AUROC={best_au:.4f}")
    print(f"    TEST AUROC = {auroc:.4f}  (baseline {baseline:.4f}, delta {delta:+.4f} = {delta*100:+.2f}%)")
    print(f"    95% CI: [{ci_lo:.4f}, {ci_hi:.4f}]")

    return {
        "dataset": ds_name,
        "n_classes": nc,
        "n_sources": len(all_names),
        "source_names": all_names,
        "n_meta_features": n_meta,
        "meta_C": best_C,
        "meta_cv_auroc": round(best_au, 4),
        "test_auroc": round(auroc, 4),
        "baseline_auroc": baseline,
        "delta": round(delta, 4),
        "delta_pct": f"{delta*100:+.2f}%",
        "ci_95": [round(ci_lo, 4), round(ci_hi, 4)],
    }


def main():
    print("=" * 60)
    print("BASELINE-ONLY FUSION v2")
    print("Multi-granularity PCA + feature concatenation")
    print("=" * 60)

    results = {}
    for ds_name, info in FOCUS_DATASETS.items():
        print(f"\n{'='*60}")
        print(f"Dataset: {ds_name} (n_classes={info['n_classes']}, best={info['best_single']:.4f})")
        print(f"{'='*60}")
        r = run_dataset(ds_name, info)
        if r:
            results[ds_name] = r

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY: Baseline-Only Fusion v2")
    print(f"{'='*60}")
    print(f"{'Dataset':25s} {'Best Single':>12s} {'Fusion':>8s} {'Delta':>8s} {'CI':>20s}")
    print("-" * 75)

    wins, losses = 0, 0
    deltas = []
    for ds_name, r in results.items():
        d = r["delta"]
        deltas.append(d)
        status = "✅" if d > 0 else "❌"
        if d > 0: wins += 1
        else: losses += 1
        print(f"{ds_name:25s} {r['baseline_auroc']:12.4f} {r['test_auroc']:8.4f} {r['delta_pct']:>8s} [{r['ci_95'][0]:.4f}, {r['ci_95'][1]:.4f}] {status}")

    if len(deltas) > 2:
        stat, pval = stats.wilcoxon(deltas, alternative='greater')
        print(f"\nWin/Loss: {wins}/{losses}")
        print(f"Wilcoxon: stat={stat}, p={pval:.4f}")

    out_path = os.path.join(RESULTS_DIR, "baseline_only_v2_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")
    return results


if __name__ == "__main__":
    main()
