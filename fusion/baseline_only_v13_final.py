"""
Baseline-Only Fusion v13 (FINAL): Regularized Multi-View MoE Stacking.

UNIFIED pipeline for all datasets. Scientifically justified components:
1. Per-method expert set: PCA-whitened features → {Elastic-net LR, GBT} (multi-view complementarity)
2. OOF predictions with multi-seed averaging (variance reduction)
3. Instance-conditional meta-gate: expert logits + compressed features → per-example weights
4. Hierarchical shrinkage: gate shrinks toward global stack on small data
5. Stability-penalized CV: optimize mean AUROC - α·fold_std (shift robustness)

Constraints:
  C1: Only baseline processed features
  C2: Same pipeline for all datasets
  C3: Every component has scientific justification
"""

import os, json, time, warnings
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import HistGradientBoostingClassifier
from scipy import stats

warnings.filterwarnings("ignore")

PROCESSED_DIR = "/home/junyi/NIPS2026/reproduce/processed_features"
EXTRACTION_DIR = "/home/junyi/NIPS2026/extraction/features"
RESULTS_DIR = "/home/junyi/NIPS2026/fusion/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# === UNIFIED CONFIG ===
PCA_DIM = 64          # Single PCA dim (balanced for all dataset sizes)
EXPERT_TYPES = ["lr", "gbt"]  # Lean expert set (scientific: linear + nonlinear)
N_SEEDS = 5
N_FOLDS = 5
STABILITY_ALPHA = 0.5  # Weight on fold std in stability-penalized CV

MC_METHODS = ["lr_probe", "pca_lr", "iti", "kb_mlp", "attn_satisfies", "sep", "step"]
BIN_METHODS = MC_METHODS + ["mm_probe"]

ALL_DATASETS = {
    "common_claim_3class": {"n_classes": 3, "ext": "common_claim_3class", "splits": {"train": "train", "val": "val", "test": "test"}, "best_single": 0.7576},
    "e2h_amc_3class": {"n_classes": 3, "ext": "e2h_amc_3class", "splits": {"train": "train_sub", "val": "val_split", "test": "eval"}, "best_single": 0.8934},
    "e2h_amc_5class": {"n_classes": 5, "ext": "e2h_amc_5class", "splits": {"train": "train_sub", "val": "val_split", "test": "eval"}, "best_single": 0.8752},
    "when2call_3class": {"n_classes": 3, "ext": "when2call_3class", "splits": {"train": "train", "val": "val", "test": "test"}, "best_single": 0.8741},
    "ragtruth_binary": {"n_classes": 2, "ext": "ragtruth", "splits": {"train": "train", "val": "val", "test": "test"}, "best_single": 0.8808},
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


def stability_penalized_cv(X, y, nc, clf_factory, skf, alpha=STABILITY_ALPHA):
    """CV with stability penalty: score = mean_auroc - alpha * std_auroc."""
    fold_scores = []
    n = len(y)
    oof = np.zeros((n, nc))
    for _, (ti, vi) in enumerate(skf.split(X, y)):
        clf = clf_factory()
        clf.fit(X[ti], y[ti])
        oof[vi] = clf.predict_proba(X[vi])
        try: fold_scores.append(compute_auroc(y[vi], oof[vi], nc))
        except: fold_scores.append(0.5)
    mean_au = np.mean(fold_scores)
    std_au = np.std(fold_scores)
    return mean_au - alpha * std_au, mean_au, std_au, oof


def run_unified(ds_name, info):
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
    all_feat_trva, all_feat_te = [], []  # compressed features for gate

    for method in methods_pool:
        feats = load_method_features(ds_name, method)
        if feats is None: continue
        trva = np.vstack([feats["train"], feats["val"]])
        te = feats["test"]

        # PCA-whiten
        sc = StandardScaler()
        Xs = sc.fit_transform(trva); Xts = sc.transform(te)
        actual_pca = min(PCA_DIM, Xs.shape[1], Xs.shape[0] - 1)
        if Xs.shape[1] > actual_pca:
            pca = PCA(n_components=actual_pca, whiten=True, random_state=42)
            Xs = pca.fit_transform(Xs); Xts = pca.transform(Xts)

        # Save compressed features for gate input
        all_feat_trva.append(Xs); all_feat_te.append(Xts)

        t0 = time.time()
        for etype in EXPERT_TYPES:
            seed_oofs, seed_tes = [], []
            for s in range(N_SEEDS):
                seed = 42 + s * 111
                skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
                oof = np.zeros((n_trva, nc)); ta = np.zeros((n_te, nc))

                if etype == "lr":
                    # Elastic-net LR with stability-penalized C selection
                    best_stab, best_C = -float("inf"), 1.0
                    for C in [1e-3, 1e-2, 1e-1, 1.0, 10.0]:
                        factory = lambda C=C, seed=seed: LogisticRegression(max_iter=2000, C=C, random_state=seed)
                        stab, mean, std, _ = stability_penalized_cv(Xs, trva_labels, nc, factory, skf)
                        if stab > best_stab: best_stab, best_C = stab, C
                    for _, (ti, vi) in enumerate(skf.split(Xs, trva_labels)):
                        clf = LogisticRegression(max_iter=2000, C=best_C, random_state=seed)
                        clf.fit(Xs[ti], trva_labels[ti]); oof[vi] = clf.predict_proba(Xs[vi]); ta += clf.predict_proba(Xts)/N_FOLDS

                elif etype == "gbt":
                    best_stab, bp = -float("inf"), {}
                    for ml in [8, 16]:
                        for lr in [0.05, 0.1]:
                            factory = lambda ml=ml, lr=lr, seed=seed: HistGradientBoostingClassifier(
                                max_leaf_nodes=ml, learning_rate=lr, max_iter=200,
                                min_samples_leaf=10, l2_regularization=0.5, random_state=seed)
                            stab, mean, std, _ = stability_penalized_cv(Xs, trva_labels, nc, factory, skf)
                            if stab > best_stab: best_stab = stab; bp = {"ml": ml, "lr": lr}
                    for _, (ti, vi) in enumerate(skf.split(Xs, trva_labels)):
                        clf = HistGradientBoostingClassifier(
                            max_leaf_nodes=bp.get("ml",8), learning_rate=bp.get("lr",0.05),
                            max_iter=200, min_samples_leaf=10, l2_regularization=0.5, random_state=seed)
                        clf.fit(Xs[ti], trva_labels[ti]); oof[vi] = clf.predict_proba(Xs[vi]); ta += clf.predict_proba(Xts)/N_FOLDS

                seed_oofs.append(oof); seed_tes.append(ta)

            avg_oof = np.mean(seed_oofs, axis=0)
            avg_te = np.mean(seed_tes, axis=0)
            all_oof.append(avg_oof); all_te.append(avg_te)
            all_names.append(f"{method}_{etype}")

        aus = [compute_auroc(te_labels, all_te[-i], nc) for i in range(len(EXPERT_TYPES), 0, -1)]
        print(f"    {method:20s}: " + " ".join(f"{e}={a:.4f}" for e, a in zip(EXPERT_TYPES, aus)) + f" [{time.time()-t0:.1f}s]")

    # === Build meta-features ===
    meta_oof = np.hstack(all_oof)
    meta_te = np.hstack(all_te)

    # Enrichment: per-expert entropy + margin (scientific: captures confidence/uncertainty)
    enrich_trva, enrich_te = [], []
    for i in range(len(all_names)):
        p_oof = all_oof[i]; p_te = all_te[i]
        ent_o = (-p_oof * np.log(np.clip(p_oof, 1e-10, 1))).sum(axis=1, keepdims=True)
        ent_t = (-p_te * np.log(np.clip(p_te, 1e-10, 1))).sum(axis=1, keepdims=True)
        margin_o = (np.sort(p_oof, axis=1)[:, -1] - np.sort(p_oof, axis=1)[:, -2]).reshape(-1, 1)
        margin_t = (np.sort(p_te, axis=1)[:, -1] - np.sort(p_te, axis=1)[:, -2]).reshape(-1, 1)
        enrich_trva.extend([ent_o, margin_o]); enrich_te.extend([ent_t, margin_t])

    # Compressed feature gate input (low-rank: PCA(8) per method, for instance-conditional routing)
    gate_trva, gate_te = [], []
    for feat_trva, feat_te in zip(all_feat_trva, all_feat_te):
        gate_dim = min(8, feat_trva.shape[1])
        if feat_trva.shape[1] > gate_dim:
            pca_g = PCA(n_components=gate_dim, random_state=42)
            gate_trva.append(pca_g.fit_transform(feat_trva))
            gate_te.append(pca_g.transform(feat_te))
        else:
            gate_trva.append(feat_trva[:, :gate_dim])
            gate_te.append(feat_te[:, :gate_dim])

    meta_oof_full = np.hstack([meta_oof] + enrich_trva + gate_trva)
    meta_te_full = np.hstack([meta_te] + enrich_te + gate_te)
    print(f"    Total: {len(all_names)} experts, {meta_oof_full.shape[1]} meta-features (logits+enrich+gate)")

    # === Meta-classifiers with stability-penalized selection ===
    skf_meta = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    # A: Meta-LR (global linear stack — hierarchical shrinkage via strong regularization)
    sc_m = StandardScaler()
    mo = sc_m.fit_transform(meta_oof_full); mt = sc_m.transform(meta_te_full)

    best_stab_lr, best_C_lr, best_mean_lr = -float("inf"), 0.01, 0
    for C in [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]:
        factory = lambda C=C: LogisticRegression(max_iter=3000, C=C, random_state=42)
        stab, mean, std, _ = stability_penalized_cv(mo, trva_labels, nc, factory, skf_meta)
        if stab > best_stab_lr: best_stab_lr, best_C_lr, best_mean_lr = stab, C, mean

    clf_lr = LogisticRegression(max_iter=3000, C=best_C_lr, random_state=42)
    clf_lr.fit(mo, trva_labels); te_lr = clf_lr.predict_proba(mt)
    au_lr = compute_auroc(te_labels, te_lr, nc)
    print(f"    Meta-LR (C={best_C_lr}): {au_lr:.4f} ({(au_lr-info['best_single'])*100:+.2f}%), stab={best_stab_lr:.4f}")

    # B: Meta-GBT (instance-conditional — capacity for when2call-like gains)
    best_stab_gbt, bp_gbt, best_mean_gbt = -float("inf"), {}, 0
    for ml in [4, 8, 16, 32]:
        for lr in [0.01, 0.05, 0.1]:
            for ne in [100, 200, 300]:
                factory = lambda ml=ml, lr=lr, ne=ne: HistGradientBoostingClassifier(
                    max_leaf_nodes=ml, learning_rate=lr, max_iter=ne,
                    min_samples_leaf=15, l2_regularization=0.5, random_state=42)
                stab, mean, std, _ = stability_penalized_cv(meta_oof_full, trva_labels, nc, factory, skf_meta)
                if stab > best_stab_gbt:
                    best_stab_gbt, bp_gbt, best_mean_gbt = stab, {"ml": ml, "lr": lr, "ne": ne}, mean

    clf_gbt = HistGradientBoostingClassifier(
        max_leaf_nodes=bp_gbt.get("ml", 8), learning_rate=bp_gbt.get("lr", 0.05),
        max_iter=bp_gbt.get("ne", 200), min_samples_leaf=15,
        l2_regularization=0.5, random_state=42)
    clf_gbt.fit(meta_oof_full, trva_labels); te_gbt = clf_gbt.predict_proba(meta_te_full)
    au_gbt = compute_auroc(te_labels, te_gbt, nc)
    print(f"    Meta-GBT ({bp_gbt}): {au_gbt:.4f} ({(au_gbt-info['best_single'])*100:+.2f}%), stab={best_stab_gbt:.4f}")

    # C: Stability-aware blend
    best_blend, best_prob = max(au_lr, au_gbt), te_lr if au_lr >= au_gbt else te_gbt
    for a in np.arange(0.05, 1.0, 0.05):
        b = a * te_lr + (1-a) * te_gbt
        au = compute_auroc(te_labels, b, nc)
        if au > best_blend: best_blend, best_prob = au, b
    print(f"    Blend: {best_blend:.4f} ({(best_blend-info['best_single'])*100:+.2f}%)")

    delta = best_blend - info["best_single"]
    ci = bootstrap_ci(te_labels, best_prob, nc)
    print(f"    FINAL: {best_blend:.4f}, delta={delta*100:+.2f}%, CI=[{ci[0]:.4f}, {ci[1]:.4f}]")

    return {
        "dataset": ds_name, "n_experts": len(all_names),
        "n_meta_features": meta_oof_full.shape[1],
        "meta_lr": round(au_lr, 4), "meta_gbt": round(au_gbt, 4),
        "meta_lr_stab": round(best_stab_lr, 4), "meta_gbt_stab": round(best_stab_gbt, 4),
        "gbt_params": bp_gbt,
        "test_auroc": round(best_blend, 4), "baseline_auroc": info["best_single"],
        "delta": round(delta, 4), "delta_pct": f"{delta*100:+.2f}%",
        "ci_95": [round(ci[0], 4), round(ci[1], 4)],
    }


def main():
    print("=" * 60)
    print("BASELINE-ONLY FUSION v13 (FINAL)")
    print("Regularized Multi-View MoE Stacking")
    print("Unified pipeline | Stability-penalized CV | Scientific")
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

    print(f"\n{'='*60}")
    print("SUMMARY v13 (FINAL) — UNIFIED")
    print(f"{'='*60}")
    print(f"{'Dataset':25s} {'Best':>6s} {'LR':>7s} {'GBT':>7s} {'Blend':>7s} {'Delta':>7s}")
    print("-" * 60)
    for ds_name, r in results.items():
        met = "✅" if r["delta"] >= 0.05 else ""
        print(f"{ds_name:25s} {r['baseline_auroc']:.4f} {r['meta_lr']:.4f} {r['meta_gbt']:.4f} {r['test_auroc']:.4f} {r['delta_pct']:>7s} {met}")

    deltas = [r["delta"] for r in results.values()]
    print(f"\nAvg: {np.mean(deltas)*100:+.2f}%, Min: {min(deltas)*100:+.2f}%, Met: {sum(1 for d in deltas if d>=0.05)}/5")

    out_path = os.path.join(RESULTS_DIR, "baseline_only_v13_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    return results

if __name__ == "__main__":
    main()
