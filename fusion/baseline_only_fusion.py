"""
Baseline-Only Fusion: uses ONLY the 7/12 baseline method processed features.
No raw LLM internal states (hidden states, attention, logit stats).

For each baseline method's processed feature vector:
  StandardScaler → PCA(dim) → C-tuned LR → OOF probabilities
Then stack all OOF probabilities into a meta-classifier.

This is the "can we improve by just combining existing probes?" experiment.
"""

import os, json, time, warnings, gc, sys
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression, RidgeClassifier
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

# Focus datasets (hard, unsaturated)
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

# Also run on other datasets for completeness
ALL_DATASETS = {
    **FOCUS_DATASETS,
    "fava_binary": {
        "n_classes": 2, "ext": "fava",
        "splits": {"train": "train", "val": "val", "test": "test"},
        "best_single": 0.9856, "best_method": "ITI",
    },
}

MULTICLASS_METHODS = ["lr_probe", "pca_lr", "iti", "kb_mlp", "attn_satisfies", "sep", "step"]
BINARY_METHODS = MULTICLASS_METHODS + ["mm_probe", "lid", "llm_check", "seakr", "coe"]

N_FOLDS = 5
C_GRID = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0, 10.0]
PCA_DIMS = [32, 64, 128, 256]  # Try multiple PCA dims per method


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
    """Load train/val/test features for a method. Returns None if not found."""
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


def run_dataset(ds_name, info, config=None):
    """Run baseline-only fusion on one dataset.

    config options:
      pca_dim: int or 'auto' — PCA dimension for each method (default: auto)
      meta_pca: int or None — PCA on meta-features before meta-LR
      multi_pca: bool — try multiple PCA dims per method, pick best
      use_raw_features: bool — also concatenate raw features (not just OOF probs)
      feature_selection: str — 'all', 'top_k'
    """
    if config is None:
        config = {}

    nc = info["n_classes"]
    sp = info["splits"]
    ext = info["ext"]

    methods = BINARY_METHODS if nc == 2 else MULTICLASS_METHODS

    tr_labels = load_labels(ext, sp["train"])
    va_labels = load_labels(ext, sp["val"])
    te_labels = load_labels(ext, sp["test"])
    trva_labels = np.concatenate([tr_labels, va_labels])
    n_tr = len(tr_labels)
    n_trva = len(trva_labels)
    n_te = len(te_labels)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    all_oof = []
    all_te = []
    all_names = []
    per_method_auroc = {}

    for method in methods:
        feats = load_method_features(ds_name, method)
        if feats is None:
            continue

        # Handle CoE which has multiple variants
        if method == "coe":
            # CoE saves as {split}_{variant}.pt
            coe_dir = os.path.join(PROCESSED_DIR, ds_name, "coe")
            variants = []
            for f in os.listdir(coe_dir):
                if f.startswith("train_") and f.endswith(".pt"):
                    v = f[6:-3]  # strip train_ and .pt
                    variants.append(v)

            for variant in sorted(variants):
                tr = torch.load(os.path.join(coe_dir, f"train_{variant}.pt"), map_location="cpu").float().numpy().reshape(-1, 1)
                va = torch.load(os.path.join(coe_dir, f"val_{variant}.pt"), map_location="cpu").float().numpy().reshape(-1, 1)
                te = torch.load(os.path.join(coe_dir, f"test_{variant}.pt"), map_location="cpu").float().numpy().reshape(-1, 1)
                trva = np.vstack([tr, va])

                sc = StandardScaler()
                trvas = sc.fit_transform(trva)
                tes = sc.transform(te)

                # Simple C tuning
                sp_cut = int(n_tr * 0.8)
                best_a, best_C = -1, 1.0
                for C in C_GRID:
                    clf = LogisticRegression(max_iter=2000, C=C, random_state=42)
                    clf.fit(trvas[:sp_cut], tr_labels[:sp_cut])
                    vp = clf.predict_proba(trvas[sp_cut:n_tr])
                    try:
                        a = compute_auroc(tr_labels[sp_cut:], vp, nc)
                    except:
                        a = 0.5
                    if a > best_a:
                        best_a, best_C = a, C

                oof = np.zeros((n_trva, nc))
                ta = np.zeros((n_te, nc))
                for _, (ti, vi) in enumerate(skf.split(trvas, trva_labels)):
                    clf = LogisticRegression(max_iter=2000, C=best_C, random_state=42)
                    clf.fit(trvas[ti], trva_labels[ti])
                    oof[vi] = clf.predict_proba(trvas[vi])
                    ta += clf.predict_proba(tes) / N_FOLDS

                all_oof.append(oof)
                all_te.append(ta)
                all_names.append(f"coe_{variant}")
            continue

        tr, va, te = feats["train"], feats["val"], feats["test"]
        trva = np.vstack([tr, va])

        # StandardScaler
        sc = StandardScaler()
        trvas = sc.fit_transform(trva)
        tes = sc.transform(te)

        # PCA if high-dim
        pca_dim = config.get("pca_dim", "auto")
        pca_obj = None
        if pca_dim == "auto":
            # Auto: reduce if dim > 256
            if trvas.shape[1] > 256:
                actual_dim = min(256, trvas.shape[0] - 1)
                pca_obj = PCA(n_components=actual_dim, random_state=42)
                trvas = pca_obj.fit_transform(trvas)
                tes = pca_obj.transform(tes)
        elif pca_dim is not None and trvas.shape[1] > pca_dim:
            actual_dim = min(pca_dim, trvas.shape[0] - 1)
            pca_obj = PCA(n_components=actual_dim, random_state=42)
            trvas = pca_obj.fit_transform(trvas)
            tes = pca_obj.transform(tes)

        # C selection on holdout
        sp_cut = int(n_tr * 0.8)
        best_a, best_C = -1, 1.0
        for C in C_GRID:
            clf = LogisticRegression(max_iter=2000, C=C, random_state=42)
            clf.fit(trvas[:sp_cut], tr_labels[:sp_cut])
            vp = clf.predict_proba(trvas[sp_cut:n_tr])
            try:
                a = compute_auroc(tr_labels[sp_cut:], vp, nc)
            except:
                a = 0.5
            if a > best_a:
                best_a, best_C = a, C

        # OOF cross-validation
        oof = np.zeros((n_trva, nc))
        ta = np.zeros((n_te, nc))
        for _, (ti, vi) in enumerate(skf.split(trvas, trva_labels)):
            clf = LogisticRegression(max_iter=2000, C=best_C, random_state=42)
            clf.fit(trvas[ti], trva_labels[ti])
            oof[vi] = clf.predict_proba(trvas[vi])
            ta += clf.predict_proba(tes) / N_FOLDS

        # Per-method test AUROC
        method_auroc = compute_auroc(te_labels, ta, nc)
        per_method_auroc[method] = method_auroc

        all_oof.append(oof)
        all_te.append(ta)
        all_names.append(method)
        print(f"    {method:20s}: dim={trva.shape[1]:>6d} → PCA={trvas.shape[1]:>4d}, C={best_C:.4f}, test AUROC={method_auroc:.4f}")

    if not all_oof:
        return None

    # === META-CLASSIFIER: Stack OOF probabilities ===
    meta_oof = np.hstack(all_oof)
    meta_te = np.hstack(all_te)
    n_meta = meta_oof.shape[1]
    print(f"    Meta-features: {n_meta} ({len(all_names)} methods × {nc} classes)")

    # Optional: PCA on meta-features
    meta_pca = config.get("meta_pca", None)
    sc_m = StandardScaler()
    mo = sc_m.fit_transform(meta_oof)
    mt = sc_m.transform(meta_te)

    if meta_pca and mo.shape[1] > meta_pca:
        pca_meta = PCA(n_components=meta_pca, random_state=42)
        mo = pca_meta.fit_transform(mo)
        mt = pca_meta.transform(mt)

    # Meta-LR with nested CV for C selection
    meta_C_grid = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0, 10.0]
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

    # Final meta-classifier
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
        "n_methods": len(all_names),
        "methods_used": all_names,
        "n_meta_features": n_meta,
        "meta_C": best_C,
        "meta_cv_auroc": round(best_au, 4),
        "test_auroc": round(auroc, 4),
        "baseline_auroc": baseline,
        "delta": round(delta, 4),
        "delta_pct": f"{delta*100:+.2f}%",
        "ci_95": [round(ci_lo, 4), round(ci_hi, 4)],
        "per_method_auroc": {k: round(v, 4) for k, v in per_method_auroc.items()},
    }


def run_all(datasets=None, config=None):
    if datasets is None:
        datasets = FOCUS_DATASETS

    results = {}
    for ds_name, info in datasets.items():
        print(f"\n{'='*60}")
        print(f"Dataset: {ds_name} (n_classes={info['n_classes']}, best={info['best_single']:.4f})")
        print(f"{'='*60}")

        r = run_dataset(ds_name, info, config)
        if r:
            results[ds_name] = r

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY: Baseline-Only Fusion")
    print(f"{'='*60}")
    print(f"{'Dataset':25s} {'Best Single':>12s} {'Fusion':>8s} {'Delta':>8s} {'CI':>20s}")
    print("-" * 75)

    wins, losses = 0, 0
    deltas = []
    for ds_name, r in results.items():
        d = r["delta"]
        deltas.append(d)
        status = "✅" if d > 0 else "❌"
        if d > 0:
            wins += 1
        else:
            losses += 1
        print(f"{ds_name:25s} {r['baseline_auroc']:12.4f} {r['test_auroc']:8.4f} {r['delta_pct']:>8s} [{r['ci_95'][0]:.4f}, {r['ci_95'][1]:.4f}] {status}")

    if len(deltas) > 2:
        stat, pval = stats.wilcoxon(deltas, alternative='greater')
        print(f"\nWin/Loss: {wins}/{losses}")
        print(f"Wilcoxon signed-rank (one-sided): stat={stat}, p={pval:.4f}")

    # Save results
    out_path = os.path.join(RESULTS_DIR, "baseline_only_fusion_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    return results


if __name__ == "__main__":
    print("=" * 60)
    print("BASELINE-ONLY FUSION v1")
    print("Input: ONLY baseline method processed features")
    print("No raw LLM internal states")
    print("=" * 60)

    results = run_all(FOCUS_DATASETS)
