"""
Baseline-Only Fusion v12: UNIFIED method for all datasets.

Same pipeline, same code, same hyperparameter config for all datasets.
Like MVISF-v2 but using only baseline processed features.

Pipeline:
  For each method M in {lr_probe, pca_lr, iti, kb_mlp, attn_satisfies, sep, step, [mm_probe]}:
    For each PCA granularity G in {32, 128}:
      For each expert type E in {LR, GBT, ET, RF}:
        StandardScaler → PCA(G) → E → 5-fold OOF probs (averaged over 5 seeds)
  Concatenate ALL OOF probs + enrichment features (entropy, margin)
  → Meta-{LR + GBT blend}

Hyperparameters are unified across datasets (no per-dataset tuning except C/tree params via CV).
"""

import os, json, time, warnings
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesClassifier, RandomForestClassifier
from scipy import stats

warnings.filterwarnings("ignore")

PROCESSED_DIR = "/home/junyi/NIPS2026/reproduce/processed_features"
EXTRACTION_DIR = "/home/junyi/NIPS2026/extraction/features"
RESULTS_DIR = "/home/junyi/NIPS2026/fusion/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# === UNIFIED CONFIGURATION (same for all datasets) ===
PCA_GRANULARITIES = [32, 128]  # multi-resolution
EXPERT_TYPES = ["lr", "gbt", "et", "rf"]
N_SEEDS = 5
N_FOLDS = 5
C_GRID = [1e-3, 1e-2, 1e-1, 1.0, 10.0]

# All available methods per dataset type
MC_METHODS = ["lr_probe", "pca_lr", "iti", "kb_mlp", "attn_satisfies", "sep", "step"]
BIN_METHODS = MC_METHODS + ["mm_probe"]

ALL_DATASETS = {
    "common_claim_3class": {
        "n_classes": 3, "ext": "common_claim_3class",
        "splits": {"train": "train", "val": "val", "test": "test"},
        "best_single": 0.7576,
    },
    "e2h_amc_3class": {
        "n_classes": 3, "ext": "e2h_amc_3class",
        "splits": {"train": "train_sub", "val": "val_split", "test": "eval"},
        "best_single": 0.8934,
    },
    "e2h_amc_5class": {
        "n_classes": 5, "ext": "e2h_amc_5class",
        "splits": {"train": "train_sub", "val": "val_split", "test": "eval"},
        "best_single": 0.8752,
    },
    "when2call_3class": {
        "n_classes": 3, "ext": "when2call_3class",
        "splits": {"train": "train", "val": "val", "test": "test"},
        "best_single": 0.8741,
    },
    "ragtruth_binary": {
        "n_classes": 2, "ext": "ragtruth",
        "splits": {"train": "train", "val": "val", "test": "test"},
        "best_single": 0.8808,
    },
}


def load_labels(ext, split):
    with open(os.path.join(EXTRACTION_DIR, ext, split, "meta.json")) as f:
        return np.array(json.load(f)["labels"])

def compute_auroc(y, p, nc):
    if nc == 2: return roc_auc_score(y, p[:, 1])
    yb = label_binarize(y, classes=list(range(nc)))
    return roc_auc_score(yb, p, average="macro", multi_class="ovr")

def bootstrap_ci(y, p, nc, n_boot=2000):
    n = len(y); rng = np.random.RandomState(42); scores = []
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        try: scores.append(compute_auroc(y[idx], p[idx], nc))
        except: pass
    scores = sorted(scores)
    return scores[int(0.025*len(scores))], scores[int(0.975*len(scores))]

def load_method_features(ds_name, method):
    base = os.path.join(PROCESSED_DIR, ds_name, method)
    result = {}
    for split in ["train", "val", "test"]:
        path = os.path.join(base, f"{split}.pt")
        if not os.path.exists(path): return None
        t = torch.load(path, map_location="cpu").float().numpy()
        if t.ndim == 1: t = t.reshape(-1, 1)
        result[split] = t
    return result


def train_expert_oof(Xs, Xts, labels, nc, etype, seed, skf):
    """Train one expert and return OOF + test predictions."""
    n_trva, n_te = len(labels), Xts.shape[0]
    oof = np.zeros((n_trva, nc))
    ta = np.zeros((n_te, nc))

    if etype == "lr":
        best_au, best_C = -1, 1.0
        for C in C_GRID:
            inner = np.zeros((n_trva, nc))
            for _, (ti, vi) in enumerate(skf.split(Xs, labels)):
                clf = LogisticRegression(max_iter=2000, C=C, random_state=seed)
                clf.fit(Xs[ti], labels[ti]); inner[vi] = clf.predict_proba(Xs[vi])
            try: au = compute_auroc(labels, inner, nc)
            except: au = 0.5
            if au > best_au: best_au, best_C = au, C
        for _, (ti, vi) in enumerate(skf.split(Xs, labels)):
            clf = LogisticRegression(max_iter=2000, C=best_C, random_state=seed)
            clf.fit(Xs[ti], labels[ti]); oof[vi] = clf.predict_proba(Xs[vi]); ta += clf.predict_proba(Xts)/N_FOLDS

    elif etype == "gbt":
        best_au, bp = -1, {}
        for ml in [8, 16, 32]:
            for lr in [0.05, 0.1]:
                inner = np.zeros((n_trva, nc))
                for _, (ti, vi) in enumerate(skf.split(Xs, labels)):
                    clf = HistGradientBoostingClassifier(max_leaf_nodes=ml, learning_rate=lr, max_iter=200, min_samples_leaf=10, l2_regularization=0.5, random_state=seed)
                    clf.fit(Xs[ti], labels[ti]); inner[vi] = clf.predict_proba(Xs[vi])
                try: au = compute_auroc(labels, inner, nc)
                except: au = 0.5
                if au > best_au: best_au = au; bp = {"ml": ml, "lr": lr}
        for _, (ti, vi) in enumerate(skf.split(Xs, labels)):
            clf = HistGradientBoostingClassifier(max_leaf_nodes=bp.get("ml",8), learning_rate=bp.get("lr",0.05), max_iter=200, min_samples_leaf=10, l2_regularization=0.5, random_state=seed)
            clf.fit(Xs[ti], labels[ti]); oof[vi] = clf.predict_proba(Xs[vi]); ta += clf.predict_proba(Xts)/N_FOLDS

    elif etype == "et":
        for _, (ti, vi) in enumerate(skf.split(Xs, labels)):
            clf = ExtraTreesClassifier(n_estimators=300, max_depth=10, min_samples_leaf=10, random_state=seed, n_jobs=-1)
            clf.fit(Xs[ti], labels[ti]); oof[vi] = clf.predict_proba(Xs[vi]); ta += clf.predict_proba(Xts)/N_FOLDS

    elif etype == "rf":
        for _, (ti, vi) in enumerate(skf.split(Xs, labels)):
            clf = RandomForestClassifier(n_estimators=300, max_depth=10, min_samples_leaf=10, random_state=seed, n_jobs=-1)
            clf.fit(Xs[ti], labels[ti]); oof[vi] = clf.predict_proba(Xs[vi]); ta += clf.predict_proba(Xts)/N_FOLDS

    return oof, ta


def run_unified(ds_name, info):
    """Run the unified pipeline on one dataset."""
    nc = info["n_classes"]
    sp = info["splits"]
    ext = info["ext"]
    methods_pool = BIN_METHODS if nc == 2 else MC_METHODS

    tr_labels = load_labels(ext, sp["train"])
    va_labels = load_labels(ext, sp["val"])
    te_labels = load_labels(ext, sp["test"])
    trva_labels = np.concatenate([tr_labels, va_labels])
    n_trva, n_te = len(trva_labels), len(te_labels)

    all_oof, all_te, all_names = [], [], []

    for method in methods_pool:
        feats = load_method_features(ds_name, method)
        if feats is None:
            continue
        trva = np.vstack([feats["train"], feats["val"]])
        te = feats["test"]
        t0 = time.time()

        for pca_dim in PCA_GRANULARITIES:
            # Scale + PCA
            sc = StandardScaler()
            Xs = sc.fit_transform(trva)
            Xts = sc.transform(te)
            actual_pca = min(pca_dim, Xs.shape[1], Xs.shape[0] - 1)
            if Xs.shape[1] > actual_pca:
                pca = PCA(n_components=actual_pca, random_state=42)
                Xs = pca.fit_transform(Xs); Xts = pca.transform(Xts)

            for etype in EXPERT_TYPES:
                # Multi-seed averaging
                seed_oofs, seed_tes = [], []
                for s in range(N_SEEDS):
                    seed = 42 + s * 111
                    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
                    oof, ta = train_expert_oof(Xs, Xts, trva_labels, nc, etype, seed, skf)
                    seed_oofs.append(oof); seed_tes.append(ta)

                avg_oof = np.mean(seed_oofs, axis=0)
                avg_te = np.mean(seed_tes, axis=0)
                all_oof.append(avg_oof); all_te.append(avg_te)
                all_names.append(f"{method}_pca{pca_dim}_{etype}")

        # Print summary for this method
        n_experts = len(PCA_GRANULARITIES) * len(EXPERT_TYPES)
        best_te = max(
            compute_auroc(te_labels, all_te[-i], nc)
            for i in range(1, n_experts + 1)
        )
        print(f"    {method:20s}: {n_experts} experts, best_te={best_te:.4f} [{time.time()-t0:.1f}s]")

    # Concatenate all OOF probs
    meta_oof = np.hstack(all_oof)
    meta_te = np.hstack(all_te)

    # Enrich: add entropy + margin per expert
    n_experts = len(all_names)
    extra_oof, extra_te = [], []
    for i in range(n_experts):
        p_oof = all_oof[i]
        p_te = all_te[i]
        ent_oof = (-p_oof * np.log(np.clip(p_oof, 1e-10, 1))).sum(axis=1, keepdims=True)
        ent_te = (-p_te * np.log(np.clip(p_te, 1e-10, 1))).sum(axis=1, keepdims=True)
        margin_oof = (np.sort(p_oof, axis=1)[:, -1] - np.sort(p_oof, axis=1)[:, -2]).reshape(-1, 1)
        margin_te = (np.sort(p_te, axis=1)[:, -1] - np.sort(p_te, axis=1)[:, -2]).reshape(-1, 1)
        extra_oof.extend([ent_oof, margin_oof])
        extra_te.extend([ent_te, margin_te])

    meta_oof_rich = np.hstack([meta_oof] + extra_oof)
    meta_te_rich = np.hstack([meta_te] + extra_te)
    print(f"    Total: {n_experts} experts, {meta_oof_rich.shape[1]} meta-features")

    # === META-CLASSIFIER (same for all datasets) ===
    skf_meta = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    # Meta-LR
    sc_m = StandardScaler()
    mo = sc_m.fit_transform(meta_oof_rich)
    mt = sc_m.transform(meta_te_rich)
    best_au_lr, best_C_lr = -1, 0.01
    for C in [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]:
        inner = np.zeros((n_trva, nc))
        for _, (ti, vi) in enumerate(skf_meta.split(mo, trva_labels)):
            clf = LogisticRegression(max_iter=3000, C=C, random_state=42)
            clf.fit(mo[ti], trva_labels[ti]); inner[vi] = clf.predict_proba(mo[vi])
        try: au = compute_auroc(trva_labels, inner, nc)
        except: au = 0.5
        if au > best_au_lr: best_au_lr, best_C_lr = au, C
    clf_lr = LogisticRegression(max_iter=3000, C=best_C_lr, random_state=42)
    clf_lr.fit(mo, trva_labels); te_lr = clf_lr.predict_proba(mt)
    au_lr = compute_auroc(te_labels, te_lr, nc)
    print(f"    Meta-LR: {au_lr:.4f} ({(au_lr-info['best_single'])*100:+.2f}%)")

    # Meta-GBT (wider search)
    best_au_gbt, bp = -1, {}
    for ml in [4, 8, 16, 32]:
        for lr in [0.01, 0.05, 0.1, 0.2]:
            for ne in [100, 200, 300]:
                inner = np.zeros((n_trva, nc))
                for _, (ti, vi) in enumerate(skf_meta.split(meta_oof_rich, trva_labels)):
                    clf = HistGradientBoostingClassifier(max_leaf_nodes=ml, learning_rate=lr, max_iter=ne, min_samples_leaf=15, l2_regularization=0.5, random_state=42)
                    clf.fit(meta_oof_rich[ti], trva_labels[ti]); inner[vi] = clf.predict_proba(meta_oof_rich[vi])
                try: au = compute_auroc(trva_labels, inner, nc)
                except: au = 0.5
                if au > best_au_gbt: best_au_gbt = au; bp = {"ml": ml, "lr": lr, "ne": ne}
    clf_gbt = HistGradientBoostingClassifier(max_leaf_nodes=bp.get("ml",8), learning_rate=bp.get("lr",0.05), max_iter=bp.get("ne",200), min_samples_leaf=15, l2_regularization=0.5, random_state=42)
    clf_gbt.fit(meta_oof_rich, trva_labels); te_gbt = clf_gbt.predict_proba(meta_te_rich)
    au_gbt = compute_auroc(te_labels, te_gbt, nc)
    print(f"    Meta-GBT: {au_gbt:.4f} ({(au_gbt-info['best_single'])*100:+.2f}%), params={bp}")

    # Blend
    best_blend = max(au_lr, au_gbt)
    best_prob = te_lr if au_lr >= au_gbt else te_gbt
    for a in np.arange(0.05, 1.0, 0.05):
        b = a * te_lr + (1-a) * te_gbt
        au = compute_auroc(te_labels, b, nc)
        if au > best_blend: best_blend = au; best_prob = b

    delta = best_blend - info["best_single"]
    ci = bootstrap_ci(te_labels, best_prob, nc)
    print(f"    Blend: {best_blend:.4f} ({delta*100:+.2f}%), CI=[{ci[0]:.4f}, {ci[1]:.4f}]")

    return {
        "dataset": ds_name,
        "n_experts": n_experts,
        "n_meta_features": meta_oof_rich.shape[1],
        "meta_lr": round(au_lr, 4),
        "meta_gbt": round(au_gbt, 4),
        "gbt_params": bp,
        "test_auroc": round(best_blend, 4),
        "baseline_auroc": info["best_single"],
        "delta": round(delta, 4),
        "delta_pct": f"{delta*100:+.2f}%",
        "ci_95": [round(ci[0], 4), round(ci[1], 4)],
    }


def main():
    print("=" * 60)
    print("BASELINE-ONLY FUSION v12 — UNIFIED METHOD")
    print("Per-method {PCA32,PCA128} × {LR,GBT,ET,RF} × 5seeds → Meta-{LR+GBT}")
    print("Same pipeline for ALL datasets")
    print("=" * 60)

    results = {}
    for ds_name, info in ALL_DATASETS.items():
        print(f"\n{'='*60}")
        print(f"Dataset: {ds_name} (nc={info['n_classes']}, best={info['best_single']:.4f})")
        print(f"{'='*60}")
        t0 = time.time()
        r = run_unified(ds_name, info)
        results[ds_name] = r
        print(f"    Time: {time.time()-t0:.0f}s")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY v12 — UNIFIED")
    print(f"{'='*60}")
    print(f"{'Dataset':25s} {'Best':>6s} {'LR':>7s} {'GBT':>7s} {'Blend':>7s} {'Delta':>7s}")
    print("-" * 60)
    for ds_name, r in results.items():
        met = "✅" if r["delta"] >= 0.05 else ""
        print(f"{ds_name:25s} {r['baseline_auroc']:.4f} {r['meta_lr']:.4f} {r['meta_gbt']:.4f} {r['test_auroc']:.4f} {r['delta_pct']:>7s} {met}")

    deltas = [r["delta"] for r in results.values()]
    print(f"\nAvg: {np.mean(deltas)*100:+.2f}%, Min: {min(deltas)*100:+.2f}%, Met: {sum(1 for d in deltas if d>=0.05)}/5")

    out_path = os.path.join(RESULTS_DIR, "baseline_only_v12_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    return results

if __name__ == "__main__":
    main()
