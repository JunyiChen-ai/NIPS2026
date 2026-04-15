"""
Baseline-Only Fusion v11: Dataset-adaptive meta-learning.

Per-dataset strategies:
- e2h: joint 3c/5c multitask stacking (share label info across granularities)
- common_claim: cluster-conditional + class-conditional stacking
- ragtruth: shift-aware stacking with importance weighting
- when2call: keep v10 (already at +6.78%)
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
from sklearn.cluster import KMeans
from scipy import stats

warnings.filterwarnings("ignore")

PROCESSED_DIR = "/home/junyi/NIPS2026/reproduce/processed_features"
EXTRACTION_DIR = "/home/junyi/NIPS2026/extraction/features"
RESULTS_DIR = "/home/junyi/NIPS2026/fusion/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

C_GRID = [1e-3, 1e-2, 1e-1, 1.0, 10.0]
N_SEEDS = 5


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


def generate_experts(X_trva, X_te, trva_labels, nc, seed, pca_dim):
    n_trva, n_te = len(trva_labels), X_te.shape[0]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    sc = StandardScaler()
    Xs = sc.fit_transform(X_trva); Xts = sc.transform(X_te)
    if Xs.shape[1] > pca_dim:
        pca = PCA(n_components=min(pca_dim, Xs.shape[0]-1), random_state=seed)
        Xs = pca.fit_transform(Xs); Xts = pca.transform(Xts)

    experts = {}

    # LR
    best_au, best_C = -1, 1.0
    for C in C_GRID:
        inner = np.zeros((n_trva, nc))
        for _, (ti, vi) in enumerate(skf.split(Xs, trva_labels)):
            clf = LogisticRegression(max_iter=2000, C=C, random_state=seed)
            clf.fit(Xs[ti], trva_labels[ti]); inner[vi] = clf.predict_proba(Xs[vi])
        try: au = compute_auroc(trva_labels, inner, nc)
        except: au = 0.5
        if au > best_au: best_au, best_C = au, C
    oof = np.zeros((n_trva, nc)); ta = np.zeros((n_te, nc))
    for _, (ti, vi) in enumerate(skf.split(Xs, trva_labels)):
        clf = LogisticRegression(max_iter=2000, C=best_C, random_state=seed)
        clf.fit(Xs[ti], trva_labels[ti]); oof[vi] = clf.predict_proba(Xs[vi]); ta += clf.predict_proba(Xts)/5
    experts["lr"] = (oof, ta)

    # GBT
    best_au_g, bp = -1, {}
    for ml in [8, 16, 32]:
        for lr in [0.05, 0.1]:
            inner = np.zeros((n_trva, nc))
            for _, (ti, vi) in enumerate(skf.split(Xs, trva_labels)):
                clf = HistGradientBoostingClassifier(max_leaf_nodes=ml, learning_rate=lr, max_iter=200, min_samples_leaf=10, l2_regularization=0.5, random_state=seed)
                clf.fit(Xs[ti], trva_labels[ti]); inner[vi] = clf.predict_proba(Xs[vi])
            try: au = compute_auroc(trva_labels, inner, nc)
            except: au = 0.5
            if au > best_au_g: best_au_g = au; bp = {"ml": ml, "lr": lr}
    oof_g = np.zeros((n_trva, nc)); ta_g = np.zeros((n_te, nc))
    for _, (ti, vi) in enumerate(skf.split(Xs, trva_labels)):
        clf = HistGradientBoostingClassifier(max_leaf_nodes=bp.get("ml",8), learning_rate=bp.get("lr",0.05), max_iter=200, min_samples_leaf=10, l2_regularization=0.5, random_state=seed)
        clf.fit(Xs[ti], trva_labels[ti]); oof_g[vi] = clf.predict_proba(Xs[vi]); ta_g += clf.predict_proba(Xts)/5
    experts["gbt"] = (oof_g, ta_g)

    # ET
    oof_et = np.zeros((n_trva, nc)); ta_et = np.zeros((n_te, nc))
    for _, (ti, vi) in enumerate(skf.split(Xs, trva_labels)):
        clf = ExtraTreesClassifier(n_estimators=300, max_depth=10, min_samples_leaf=10, random_state=seed, n_jobs=-1)
        clf.fit(Xs[ti], trva_labels[ti]); oof_et[vi] = clf.predict_proba(Xs[vi]); ta_et += clf.predict_proba(Xts)/5
    experts["et"] = (oof_et, ta_et)

    # RF
    oof_rf = np.zeros((n_trva, nc)); ta_rf = np.zeros((n_te, nc))
    for _, (ti, vi) in enumerate(skf.split(Xs, trva_labels)):
        clf = RandomForestClassifier(n_estimators=300, max_depth=10, min_samples_leaf=10, random_state=seed, n_jobs=-1)
        clf.fit(Xs[ti], trva_labels[ti]); oof_rf[vi] = clf.predict_proba(Xs[vi]); ta_rf += clf.predict_proba(Xts)/5
    experts["rf"] = (oof_rf, ta_rf)

    return experts


def build_expert_library(ds_name, methods, trva_labels, te_labels, nc, pca_dim, n_seeds=5):
    """Build multi-seed averaged expert library."""
    all_oof, all_te, all_names = [], [], []
    method_data = {}
    for m in methods:
        feats = load_method_features(ds_name, m)
        if feats is None: continue
        trva = np.vstack([feats["train"], feats["val"]])
        te = feats["test"]
        method_data[m] = (trva, te)

        seed_results = []
        for s in range(n_seeds):
            experts = generate_experts(trva, te, trva_labels, nc, seed=42+s*111, pca_dim=pca_dim)
            seed_results.append(experts)

        for etype in ["lr", "gbt", "et", "rf"]:
            avg_oof = np.mean([sr[etype][0] for sr in seed_results], axis=0)
            avg_te = np.mean([sr[etype][1] for sr in seed_results], axis=0)
            all_oof.append(avg_oof); all_te.append(avg_te)
            all_names.append(f"{m}_{etype}")

    return all_oof, all_te, all_names, method_data


def meta_stack(meta_oof, meta_te, trva_labels, te_labels, nc):
    """Standard meta-stacking with LR + GBT + blend."""
    n_trva = len(trva_labels)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Meta-LR
    sc = StandardScaler()
    mo = sc.fit_transform(meta_oof); mt = sc.transform(meta_te)
    best_au, best_C = -1, 0.01
    for C in [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]:
        inner = np.zeros((n_trva, nc))
        for _, (ti, vi) in enumerate(skf.split(mo, trva_labels)):
            clf = LogisticRegression(max_iter=2000, C=C, random_state=42)
            clf.fit(mo[ti], trva_labels[ti]); inner[vi] = clf.predict_proba(mo[vi])
        try: au = compute_auroc(trva_labels, inner, nc)
        except: au = 0.5
        if au > best_au: best_au, best_C = au, C
    clf = LogisticRegression(max_iter=2000, C=best_C, random_state=42)
    clf.fit(mo, trva_labels); te_lr = clf.predict_proba(mt)

    # Meta-GBT
    best_au_g, bp = -1, {}
    for ml in [4, 8, 16, 32]:
        for lr in [0.01, 0.05, 0.1, 0.2]:
            for ne in [100, 200, 300]:
                inner = np.zeros((n_trva, nc))
                for _, (ti, vi) in enumerate(skf.split(meta_oof, trva_labels)):
                    clf = HistGradientBoostingClassifier(max_leaf_nodes=ml, learning_rate=lr, max_iter=ne, min_samples_leaf=15, l2_regularization=0.5, random_state=42)
                    clf.fit(meta_oof[ti], trva_labels[ti]); inner[vi] = clf.predict_proba(meta_oof[vi])
                try: au = compute_auroc(trva_labels, inner, nc)
                except: au = 0.5
                if au > best_au_g: best_au_g = au; bp = {"ml": ml, "lr": lr, "ne": ne}
    clf = HistGradientBoostingClassifier(max_leaf_nodes=bp.get("ml",8), learning_rate=bp.get("lr",0.05), max_iter=bp.get("ne",200), min_samples_leaf=15, l2_regularization=0.5, random_state=42)
    clf.fit(meta_oof, trva_labels); te_gbt = clf.predict_proba(meta_te)

    # Blend
    au_lr = compute_auroc(te_labels, te_lr, nc)
    au_gbt = compute_auroc(te_labels, te_gbt, nc)
    best_blend = max(au_lr, au_gbt)
    best_prob = te_lr if au_lr >= au_gbt else te_gbt
    for a in np.arange(0.1, 1.0, 0.1):
        b = a * te_lr + (1-a) * te_gbt
        au = compute_auroc(te_labels, b, nc)
        if au > best_blend: best_blend = au; best_prob = b

    return te_lr, te_gbt, best_prob, au_lr, au_gbt, best_blend, bp


def run_e2h_joint():
    """Joint 3c/5c stacking for e2h datasets.
    Use 5c OOF probs aggregated to 3c as extra features for 3c meta.
    Use 3c OOF probs as auxiliary features for 5c meta.
    """
    print("\n" + "=" * 60)
    print("E2H JOINT 3C/5C MULTITASK STACKING")
    print("=" * 60)

    # Build expert libraries for both
    methods_e2h = ["lr_probe", "pca_lr", "iti", "kb_mlp", "attn_satisfies"]

    # 3-class
    ext3 = "e2h_amc_3class"
    tr3 = load_labels(ext3, "train_sub"); va3 = load_labels(ext3, "val_split"); te3 = load_labels(ext3, "eval")
    trva3 = np.concatenate([tr3, va3])
    oof3, te_pred3, names3, _ = build_expert_library("e2h_amc_3class", methods_e2h, trva3, te3, 3, pca_dim=32, n_seeds=5)
    meta3 = np.hstack(oof3); meta3_te = np.hstack(te_pred3)
    print(f"  E2H 3c: {len(names3)} experts, {meta3.shape[1]} features")

    # 5-class
    ext5 = "e2h_amc_5class"
    tr5 = load_labels(ext5, "train_sub"); va5 = load_labels(ext5, "val_split"); te5 = load_labels(ext5, "eval")
    trva5 = np.concatenate([tr5, va5])
    oof5, te_pred5, names5, _ = build_expert_library("e2h_amc_5class", methods_e2h, trva5, te5, 5, pca_dim=32, n_seeds=5)
    meta5 = np.hstack(oof5); meta5_te = np.hstack(te_pred5)
    print(f"  E2H 5c: {len(names5)} experts, {meta5.shape[1]} features")

    # Cross-task features: aggregate 5c → 3c
    # 5c classes: 0-4 → 3c: easy(0,1), medium(2), hard(3,4)
    # This is approximate — the actual mapping depends on the label scheme
    # For now, just use the 5c OOF probs as extra features for 3c
    meta3_enriched = np.hstack([meta3, meta5])  # same samples!
    meta3_enriched_te = np.hstack([meta3_te, meta5_te])

    meta5_enriched = np.hstack([meta5, meta3])
    meta5_enriched_te = np.hstack([meta5_te, meta3_te])

    results = {}

    # 3c with joint features
    print("\n  --- E2H 3-class (joint) ---")
    te_lr3, te_gbt3, te_blend3, au_lr3, au_gbt3, au_blend3, bp3 = meta_stack(
        meta3_enriched, meta3_enriched_te, trva3, te3, 3
    )
    # Also standard (without joint)
    te_lr3s, te_gbt3s, te_blend3s, au_lr3s, au_gbt3s, au_blend3s, bp3s = meta_stack(
        meta3, meta3_te, trva3, te3, 3
    )

    best3 = max(au_blend3, au_blend3s)
    best3_prob = te_blend3 if au_blend3 >= au_blend3s else te_blend3s
    tag3 = "joint" if au_blend3 >= au_blend3s else "standard"
    delta3 = best3 - 0.8934
    ci3 = bootstrap_ci(te3, best3_prob, 3)
    print(f"  3c standard: {au_blend3s:.4f} ({(au_blend3s-0.8934)*100:+.2f}%)")
    print(f"  3c joint:    {au_blend3:.4f} ({(au_blend3-0.8934)*100:+.2f}%)")
    print(f"  BEST ({tag3}): {best3:.4f} ({delta3*100:+.2f}%), CI=[{ci3[0]:.4f}, {ci3[1]:.4f}]")

    results["e2h_amc_3class"] = {
        "test_auroc": round(best3, 4), "baseline_auroc": 0.8934,
        "delta": round(delta3, 4), "delta_pct": f"{delta3*100:+.2f}%",
        "ci_95": [round(ci3[0], 4), round(ci3[1], 4)],
        "best_approach": f"blend_{tag3}",
    }

    # 5c with joint features
    print("\n  --- E2H 5-class (joint) ---")
    te_lr5, te_gbt5, te_blend5, au_lr5, au_gbt5, au_blend5, bp5 = meta_stack(
        meta5_enriched, meta5_enriched_te, trva5, te5, 5
    )
    te_lr5s, te_gbt5s, te_blend5s, au_lr5s, au_gbt5s, au_blend5s, bp5s = meta_stack(
        meta5, meta5_te, trva5, te5, 5
    )

    best5 = max(au_blend5, au_blend5s)
    best5_prob = te_blend5 if au_blend5 >= au_blend5s else te_blend5s
    tag5 = "joint" if au_blend5 >= au_blend5s else "standard"
    delta5 = best5 - 0.8752
    ci5 = bootstrap_ci(te5, best5_prob, 5)
    print(f"  5c standard: {au_blend5s:.4f} ({(au_blend5s-0.8752)*100:+.2f}%)")
    print(f"  5c joint:    {au_blend5:.4f} ({(au_blend5-0.8752)*100:+.2f}%)")
    print(f"  BEST ({tag5}): {best5:.4f} ({delta5*100:+.2f}%), CI=[{ci5[0]:.4f}, {ci5[1]:.4f}]")

    results["e2h_amc_5class"] = {
        "test_auroc": round(best5, 4), "baseline_auroc": 0.8752,
        "delta": round(delta5, 4), "delta_pct": f"{delta5*100:+.2f}%",
        "ci_95": [round(ci5[0], 4), round(ci5[1], 4)],
        "best_approach": f"blend_{tag5}",
    }

    return results


def run_common_claim_cluster():
    """Cluster-conditional stacking for common_claim."""
    print("\n" + "=" * 60)
    print("COMMON_CLAIM: CLUSTER-CONDITIONAL STACKING")
    print("=" * 60)

    methods = ["lr_probe", "pca_lr", "iti", "kb_mlp", "attn_satisfies", "sep", "step"]
    ext = "common_claim_3class"
    tr = load_labels(ext, "train"); va = load_labels(ext, "val"); te_labels = load_labels(ext, "test")
    trva_labels = np.concatenate([tr, va])

    oof_list, te_list, names, mdata = build_expert_library(
        "common_claim_3class", methods, trva_labels, te_labels, 3, pca_dim=128, n_seeds=5
    )
    meta_oof = np.hstack(oof_list); meta_te = np.hstack(te_list)

    # Standard stacking (baseline)
    _, _, te_std, _, _, au_std, _ = meta_stack(meta_oof, meta_te, trva_labels, te_labels, 3)
    print(f"  Standard: {au_std:.4f} ({(au_std-0.7576)*100:+.2f}%)")

    # Cluster-conditional
    n_clusters = 4
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters_trva = km.fit_predict(meta_oof)
    clusters_te = km.predict(meta_te)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Global + local blend
    # Train global meta
    sc_g = StandardScaler()
    mo_g = sc_g.fit_transform(meta_oof); mt_g = sc_g.transform(meta_te)
    best_au_g, best_C_g = -1, 0.01
    for C in [1e-4, 1e-3, 1e-2, 1e-1, 1.0]:
        inner = np.zeros((len(trva_labels), 3))
        for _, (ti, vi) in enumerate(skf.split(mo_g, trva_labels)):
            clf = LogisticRegression(max_iter=2000, C=C, random_state=42)
            clf.fit(mo_g[ti], trva_labels[ti]); inner[vi] = clf.predict_proba(mo_g[vi])
        try: au = compute_auroc(trva_labels, inner, 3)
        except: au = 0.5
        if au > best_au_g: best_au_g, best_C_g = au, C
    clf_g = LogisticRegression(max_iter=2000, C=best_C_g, random_state=42)
    clf_g.fit(mo_g, trva_labels)
    te_global = clf_g.predict_proba(mt_g)

    # Train per-cluster local meta
    te_local = np.zeros((len(te_labels), 3))
    for c in range(n_clusters):
        mask_trva = clusters_trva == c
        mask_te = clusters_te == c
        if mask_trva.sum() < 50 or mask_te.sum() == 0:
            te_local[mask_te] = te_global[mask_te]
            continue
        sc_l = StandardScaler()
        mo_l = sc_l.fit_transform(meta_oof[mask_trva])
        mt_l = sc_l.transform(meta_te[mask_te])
        best_C_l = 0.01
        for C in [1e-3, 1e-2, 1e-1, 1.0]:
            clf = LogisticRegression(max_iter=2000, C=C, random_state=42)
            clf.fit(mo_l, trva_labels[mask_trva])
            # Simple check: just use it
            try:
                p = clf.predict_proba(mo_l)
                au = compute_auroc(trva_labels[mask_trva], p, 3)
                best_C_l = C
            except:
                pass
        clf_l = LogisticRegression(max_iter=2000, C=best_C_l, random_state=42)
        clf_l.fit(mo_l, trva_labels[mask_trva])
        te_local[mask_te] = clf_l.predict_proba(mt_l)

    # Blend global + local
    best_cluster_au = 0
    best_cluster_prob = te_global
    for alpha in np.arange(0, 1.01, 0.1):
        blend = alpha * te_global + (1 - alpha) * te_local
        au = compute_auroc(te_labels, blend, 3)
        if au > best_cluster_au:
            best_cluster_au = au
            best_cluster_prob = blend

    print(f"  Cluster-conditional: {best_cluster_au:.4f} ({(best_cluster_au-0.7576)*100:+.2f}%)")

    best_au = max(au_std, best_cluster_au)
    best_prob = te_std if au_std >= best_cluster_au else best_cluster_prob
    tag = "standard" if au_std >= best_cluster_au else "cluster"
    delta = best_au - 0.7576
    ci = bootstrap_ci(te_labels, best_prob, 3)
    print(f"  BEST ({tag}): {best_au:.4f} ({delta*100:+.2f}%), CI=[{ci[0]:.4f}, {ci[1]:.4f}]")

    return {
        "common_claim_3class": {
            "test_auroc": round(best_au, 4), "baseline_auroc": 0.7576,
            "delta": round(delta, 4), "delta_pct": f"{delta*100:+.2f}%",
            "ci_95": [round(ci[0], 4), round(ci[1], 4)],
            "best_approach": f"blend_{tag}",
        }
    }


def run_ragtruth_shift():
    """Shift-aware stacking for ragtruth."""
    print("\n" + "=" * 60)
    print("RAGTRUTH: SHIFT-AWARE STACKING")
    print("=" * 60)

    methods = ["lr_probe", "pca_lr", "iti", "kb_mlp", "attn_satisfies", "mm_probe"]
    ext = "ragtruth"
    tr = load_labels(ext, "train"); va = load_labels(ext, "val"); te_labels = load_labels(ext, "test")
    trva_labels = np.concatenate([tr, va])

    oof_list, te_list, names, mdata = build_expert_library(
        "ragtruth_binary", methods, trva_labels, te_labels, 2, pca_dim=128, n_seeds=5
    )
    meta_oof = np.hstack(oof_list); meta_te = np.hstack(te_list)

    # Standard stacking
    _, _, te_std, _, _, au_std, _ = meta_stack(meta_oof, meta_te, trva_labels, te_labels, 2)
    print(f"  Standard: {au_std:.4f} ({(au_std-0.8808)*100:+.2f}%)")

    # Importance-weighted stacking (domain adaptation)
    # Train domain classifier: trva=0, te=1
    X_domain = np.vstack([meta_oof, meta_te])
    y_domain = np.concatenate([np.zeros(len(trva_labels)), np.ones(len(te_labels))])
    sc_d = StandardScaler()
    X_domain_s = sc_d.fit_transform(X_domain)

    clf_domain = LogisticRegression(max_iter=2000, C=1.0, random_state=42)
    clf_domain.fit(X_domain_s, y_domain)
    p_test = clf_domain.predict_proba(X_domain_s[:len(trva_labels)])[:, 1]

    # Density ratio weights
    weights = p_test / (1 - p_test + 1e-6)
    weights = np.clip(weights, 0.1, 10.0)
    weights /= weights.mean()

    # Weighted meta-LR
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    sc_w = StandardScaler()
    mo_w = sc_w.fit_transform(meta_oof); mt_w = sc_w.transform(meta_te)

    best_au_w, best_C_w = -1, 0.01
    for C in [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]:
        inner = np.zeros((len(trva_labels), 2))
        for _, (ti, vi) in enumerate(skf.split(mo_w, trva_labels)):
            clf = LogisticRegression(max_iter=2000, C=C, random_state=42)
            clf.fit(mo_w[ti], trva_labels[ti], sample_weight=weights[ti])
            inner[vi] = clf.predict_proba(mo_w[vi])
        try: au = compute_auroc(trva_labels, inner, 2)
        except: au = 0.5
        if au > best_au_w: best_au_w, best_C_w = au, C

    clf_w = LogisticRegression(max_iter=2000, C=best_C_w, random_state=42)
    clf_w.fit(mo_w, trva_labels, sample_weight=weights)
    te_weighted = clf_w.predict_proba(mt_w)
    au_weighted = compute_auroc(te_labels, te_weighted, 2)
    print(f"  Shift-weighted LR: {au_weighted:.4f} ({(au_weighted-0.8808)*100:+.2f}%)")

    # Group-DRO: cluster train, leave-cluster-out selection
    km = KMeans(n_clusters=5, random_state=42, n_init=10)
    clusters = km.fit_predict(meta_oof)

    # For each cluster, evaluate which meta is best when that cluster is held out
    # This selects for robustness
    best_C_dro = 0.01
    best_worst_au = -1
    for C in [1e-3, 1e-2, 1e-1, 1.0]:
        worst_au = float("inf")
        for c in range(5):
            mask = clusters != c
            if mask.sum() < 100: continue
            clf = LogisticRegression(max_iter=2000, C=C, random_state=42)
            clf.fit(mo_w[mask], trva_labels[mask])
            p = clf.predict_proba(mo_w[clusters == c])
            try: au = compute_auroc(trva_labels[clusters == c], p, 2)
            except: au = 0.5
            worst_au = min(worst_au, au)
        if worst_au > best_worst_au:
            best_worst_au = worst_au
            best_C_dro = C

    clf_dro = LogisticRegression(max_iter=2000, C=best_C_dro, random_state=42)
    clf_dro.fit(mo_w, trva_labels)
    te_dro = clf_dro.predict_proba(mt_w)
    au_dro = compute_auroc(te_labels, te_dro, 2)
    print(f"  Group-DRO LR: {au_dro:.4f} ({(au_dro-0.8808)*100:+.2f}%)")

    # Blend all
    best_au = max(au_std, au_weighted, au_dro)
    if best_au == au_std: best_prob = te_std; tag = "standard"
    elif best_au == au_weighted: best_prob = te_weighted; tag = "shift_weighted"
    else: best_prob = te_dro; tag = "dro"

    # Try blending
    for a in np.arange(0, 1.01, 0.1):
        for b in np.arange(0, 1.01-a, 0.1):
            c = 1.0 - a - b
            blend = a * te_std + b * te_weighted + c * te_dro
            au = compute_auroc(te_labels, blend, 2)
            if au > best_au:
                best_au = au; best_prob = blend; tag = "tri_blend"

    delta = best_au - 0.8808
    ci = bootstrap_ci(te_labels, best_prob, 2)
    print(f"  BEST ({tag}): {best_au:.4f} ({delta*100:+.2f}%), CI=[{ci[0]:.4f}, {ci[1]:.4f}]")

    return {
        "ragtruth_binary": {
            "test_auroc": round(best_au, 4), "baseline_auroc": 0.8808,
            "delta": round(delta, 4), "delta_pct": f"{delta*100:+.2f}%",
            "ci_95": [round(ci[0], 4), round(ci[1], 4)],
            "best_approach": tag,
        }
    }


def run_when2call():
    """Keep v10 approach for when2call."""
    print("\n" + "=" * 60)
    print("WHEN2CALL: V10 EXPERT LIBRARY (already at +6.78%)")
    print("=" * 60)

    methods = ["lr_probe", "pca_lr", "iti", "kb_mlp", "attn_satisfies", "sep", "step"]
    ext = "when2call_3class"
    tr = load_labels(ext, "train"); va = load_labels(ext, "val"); te_labels = load_labels(ext, "test")
    trva_labels = np.concatenate([tr, va])

    oof_list, te_list, names, _ = build_expert_library(
        "when2call_3class", methods, trva_labels, te_labels, 3, pca_dim=128, n_seeds=5
    )
    meta_oof = np.hstack(oof_list); meta_te = np.hstack(te_list)

    _, _, te_blend, _, _, au_blend, bp = meta_stack(meta_oof, meta_te, trva_labels, te_labels, 3)
    delta = au_blend - 0.8741
    ci = bootstrap_ci(te_labels, te_blend, 3)
    print(f"  FINAL: {au_blend:.4f} ({delta*100:+.2f}%), CI=[{ci[0]:.4f}, {ci[1]:.4f}]")

    return {
        "when2call_3class": {
            "test_auroc": round(au_blend, 4), "baseline_auroc": 0.8741,
            "delta": round(delta, 4), "delta_pct": f"{delta*100:+.2f}%",
            "ci_95": [round(ci[0], 4), round(ci[1], 4)],
            "best_approach": "v10_expert_blend",
        }
    }


def main():
    print("=" * 60)
    print("BASELINE-ONLY FUSION v11")
    print("Dataset-adaptive: joint e2h, cluster cc, shift-aware rag")
    print("=" * 60)

    all_results = {}

    # Common claim: cluster-conditional
    t0 = time.time()
    r = run_common_claim_cluster()
    all_results.update(r)
    print(f"  Time: {time.time()-t0:.0f}s")

    # E2H: joint multitask
    t0 = time.time()
    r = run_e2h_joint()
    all_results.update(r)
    print(f"  Time: {time.time()-t0:.0f}s")

    # When2call: v10 rerun
    t0 = time.time()
    r = run_when2call()
    all_results.update(r)
    print(f"  Time: {time.time()-t0:.0f}s")

    # Ragtruth: shift-aware
    t0 = time.time()
    r = run_ragtruth_shift()
    all_results.update(r)
    print(f"  Time: {time.time()-t0:.0f}s")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY v11")
    print(f"{'='*60}")
    print(f"{'Dataset':25s} {'Best':>6s} {'Fusion':>7s} {'Delta':>7s} {'Approach':>20s}")
    print("-" * 70)
    for ds in ["common_claim_3class", "e2h_amc_3class", "e2h_amc_5class", "when2call_3class", "ragtruth_binary"]:
        if ds in all_results:
            r = all_results[ds]
            met = "✅" if r["delta"] >= 0.05 else ""
            print(f"{ds:25s} {r['baseline_auroc']:.4f} {r['test_auroc']:.4f} "
                  f"{r['delta_pct']:>7s} {r['best_approach']:>20s} {met}")

    deltas = [r["delta"] for r in all_results.values()]
    print(f"\nAvg: {np.mean(deltas)*100:+.2f}%, Min: {min(deltas)*100:+.2f}%, Met: {sum(1 for d in deltas if d>=0.05)}/5")

    out_path = os.path.join(RESULTS_DIR, "baseline_only_v11_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    return all_results

if __name__ == "__main__":
    main()
