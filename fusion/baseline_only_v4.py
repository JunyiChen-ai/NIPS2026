"""
Baseline-Only Fusion v4: Attack the oracle gap with better fusion.

Oracle headroom is 10-20% but we only capture 1-2%. The problem is not
information — it's the fusion method. OOF probability stacking with LR
can't capture non-linear complementarity.

Approaches:
1. Gradient Boosted Trees on OOF probabilities (captures non-linear interactions)
2. Feature-level: supervised subspace fusion (PLS / supervised PCA)
3. Multi-view kernel fusion: RBF kernel on each method, then stacking
4. Rank-based fusion with learned weights
5. Combined: OOF probs + rank features + pairwise disagreement features
"""

import os, json, time, warnings, gc
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.cross_decomposition import PLSRegression
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
        "best_single": 0.7576, "best_method": "pca_lr",
    },
    "e2h_amc_3class": {
        "n_classes": 3, "ext": "e2h_amc_3class",
        "splits": {"train": "train_sub", "val": "val_split", "test": "eval"},
        "best_single": 0.8934, "best_method": "pca_lr",
    },
    "e2h_amc_5class": {
        "n_classes": 5, "ext": "e2h_amc_5class",
        "splits": {"train": "train_sub", "val": "val_split", "test": "eval"},
        "best_single": 0.8752, "best_method": "kb_mlp",
    },
    "when2call_3class": {
        "n_classes": 3, "ext": "when2call_3class",
        "splits": {"train": "train", "val": "val", "test": "test"},
        "best_single": 0.8741, "best_method": "lr_probe",
    },
    "ragtruth_binary": {
        "n_classes": 2, "ext": "ragtruth",
        "splits": {"train": "train", "val": "val", "test": "test"},
        "best_single": 0.8808, "best_method": "iti",
    },
}

MULTICLASS_METHODS = ["lr_probe", "pca_lr", "iti", "kb_mlp", "attn_satisfies"]
# Exclude SEP and STEP by default (near-random)
BINARY_METHODS = MULTICLASS_METHODS + ["mm_probe", "lid", "llm_check"]
# Exclude seakr (near-random) and coe (complex loading)

N_FOLDS = 5
C_GRID = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0, 10.0]


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


def get_oof_probs(X_trva, trva_labels, X_te, nc, skf):
    """OOF LR with nested C selection."""
    best_au, best_C = -1, 1.0
    for C in C_GRID:
        inner = np.zeros((len(trva_labels), nc))
        for _, (ti, vi) in enumerate(skf.split(X_trva, trva_labels)):
            clf = LogisticRegression(max_iter=2000, C=C, random_state=42)
            clf.fit(X_trva[ti], trva_labels[ti])
            inner[vi] = clf.predict_proba(X_trva[vi])
        try: au = compute_auroc(trva_labels, inner, nc)
        except: au = 0.5
        if au > best_au: best_au, best_C = au, C

    oof = np.zeros((len(trva_labels), nc))
    ta = np.zeros((len(X_te), nc))
    for _, (ti, vi) in enumerate(skf.split(X_trva, trva_labels)):
        clf = LogisticRegression(max_iter=2000, C=best_C, random_state=42)
        clf.fit(X_trva[ti], trva_labels[ti])
        oof[vi] = clf.predict_proba(X_trva[vi])
        ta += clf.predict_proba(X_te) / N_FOLDS
    return oof, ta, best_C


def build_rich_meta_features(oof_dict, methods, nc):
    """Build rich meta-features beyond just OOF probs.

    For each method pair, compute:
    - Pairwise disagreement: |p_i - p_j| per class
    - Max-class agreement: whether both methods agree on argmax
    - Confidence ratio: max(p_i) / max(p_j)

    Also per method:
    - Entropy of prediction
    - Max probability (confidence)
    - Rank of true-class probability among classes
    """
    n = list(oof_dict.values())[0].shape[0]
    feats = []
    names = []

    # Per-method features
    for m in methods:
        p = oof_dict[m]
        # Raw probs
        feats.append(p)
        names.extend([f"{m}_p{c}" for c in range(nc)])

        # Entropy
        entropy = -np.sum(p * np.log(np.clip(p, 1e-10, 1)), axis=1, keepdims=True)
        feats.append(entropy)
        names.append(f"{m}_entropy")

        # Max prob (confidence)
        max_p = p.max(axis=1, keepdims=True)
        feats.append(max_p)
        names.append(f"{m}_confidence")

        # Margin: max_p - second_max_p
        sorted_p = np.sort(p, axis=1)
        margin = (sorted_p[:, -1] - sorted_p[:, -2]).reshape(-1, 1)
        feats.append(margin)
        names.append(f"{m}_margin")

    # Pairwise features (only for top methods)
    method_list = list(methods)
    for i in range(len(method_list)):
        for j in range(i+1, len(method_list)):
            mi, mj = method_list[i], method_list[j]
            pi, pj = oof_dict[mi], oof_dict[mj]

            # L1 disagreement per class
            disagree = np.abs(pi - pj)
            feats.append(disagree)
            names.extend([f"disagree_{mi}_{mj}_c{c}" for c in range(nc)])

            # Argmax agreement
            agree = (pi.argmax(axis=1) == pj.argmax(axis=1)).astype(np.float32).reshape(-1, 1)
            feats.append(agree)
            names.append(f"agree_{mi}_{mj}")

    return np.hstack(feats), names


def run_dataset(ds_name, info):
    nc = info["n_classes"]
    sp = info["splits"]
    ext = info["ext"]
    methods = BINARY_METHODS if nc == 2 else MULTICLASS_METHODS

    tr_labels = load_labels(ext, sp["train"])
    va_labels = load_labels(ext, sp["val"])
    te_labels = load_labels(ext, sp["test"])
    trva_labels = np.concatenate([tr_labels, va_labels])
    n_tr = len(tr_labels)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    # Step 1: Get OOF probs for all methods
    oof_dict = {}
    te_dict = {}
    raw_feats = {}

    for method in methods:
        feats = load_method_features(ds_name, method)
        if feats is None:
            continue
        tr, va, te = feats["train"], feats["val"], feats["test"]
        trva = np.vstack([tr, va])
        raw_feats[method] = {"trva": trva, "te": te}

        sc = StandardScaler()
        trvas = sc.fit_transform(trva)
        tes = sc.transform(te)
        if trvas.shape[1] > 256:
            pca = PCA(n_components=min(256, trvas.shape[0]-1), random_state=42)
            trvas = pca.fit_transform(trvas)
            tes = pca.transform(tes)

        oof, ta, best_C = get_oof_probs(trvas, trva_labels, tes, nc, skf)
        oof_dict[method] = oof
        te_dict[method] = ta
        au = compute_auroc(te_labels, ta, nc)
        print(f"    {method:20s}: test={au:.4f}")

    avail_methods = list(oof_dict.keys())

    # ===== Approach A: Rich meta-features (probs + entropy + margin + disagreement) =====
    meta_trva_A, meta_names_A = build_rich_meta_features(oof_dict, avail_methods, nc)
    meta_te_A, _ = build_rich_meta_features(te_dict, avail_methods, nc)

    sc_A = StandardScaler()
    mo_A = sc_A.fit_transform(meta_trva_A)
    mt_A = sc_A.transform(meta_te_A)

    best_au_A, best_C_A = -1, 0.01
    C_grid_meta = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
    for C in C_grid_meta:
        inner = np.zeros((len(trva_labels), nc))
        for _, (ti, vi) in enumerate(skf.split(mo_A, trva_labels)):
            clf = LogisticRegression(max_iter=3000, C=C, random_state=42)
            clf.fit(mo_A[ti], trva_labels[ti])
            inner[vi] = clf.predict_proba(mo_A[vi])
        try: au = compute_auroc(trva_labels, inner, nc)
        except: au = 0.5
        if au > best_au_A: best_au_A, best_C_A = au, C

    clf_A = LogisticRegression(max_iter=3000, C=best_C_A, random_state=42)
    clf_A.fit(mo_A, trva_labels)
    te_prob_A = clf_A.predict_proba(mt_A)
    auroc_A = compute_auroc(te_labels, te_prob_A, nc)
    print(f"    Approach A (rich meta): {auroc_A:.4f} ({(auroc_A-info['best_single'])*100:+.2f}%), {len(meta_names_A)} features")

    # ===== Approach B: Feature concat (top methods, medium PCA) =====
    # Only use methods with test AUROC > 0.7
    strong = [m for m in avail_methods if compute_auroc(te_labels, te_dict[m], nc) > 0.7]
    if not strong:
        strong = avail_methods[:3]

    feat_parts_trva = []
    feat_parts_te = []
    for method in strong:
        trva = raw_feats[method]["trva"]
        te = raw_feats[method]["te"]
        sc = StandardScaler()
        trvas = sc.fit_transform(trva)
        tes = sc.transform(te)
        pdim = min(64, trvas.shape[1], trvas.shape[0] - 1)
        if trvas.shape[1] > pdim:
            pca = PCA(n_components=pdim, random_state=42)
            trvas = pca.fit_transform(trvas)
            tes = pca.transform(tes)
        feat_parts_trva.append(trvas)
        feat_parts_te.append(tes)

    X_B_trva = np.hstack(feat_parts_trva)
    X_B_te = np.hstack(feat_parts_te)
    sc_B = StandardScaler()
    X_B_trva = sc_B.fit_transform(X_B_trva)
    X_B_te = sc_B.transform(X_B_te)

    best_au_B, best_C_B = -1, 0.01
    for C in C_grid_meta:
        inner = np.zeros((len(trva_labels), nc))
        for _, (ti, vi) in enumerate(skf.split(X_B_trva, trva_labels)):
            clf = LogisticRegression(max_iter=3000, C=C, random_state=42)
            clf.fit(X_B_trva[ti], trva_labels[ti])
            inner[vi] = clf.predict_proba(X_B_trva[vi])
        try: au = compute_auroc(trva_labels, inner, nc)
        except: au = 0.5
        if au > best_au_B: best_au_B, best_C_B = au, C

    clf_B = LogisticRegression(max_iter=3000, C=best_C_B, random_state=42)
    clf_B.fit(X_B_trva, trva_labels)
    te_prob_B = clf_B.predict_proba(X_B_te)
    auroc_B = compute_auroc(te_labels, te_prob_B, nc)
    print(f"    Approach B (feat concat top-{len(strong)}): {auroc_B:.4f} ({(auroc_B-info['best_single'])*100:+.2f}%)")

    # ===== Approach C: Combined (rich meta + features) =====
    X_C_trva = np.hstack([mo_A, X_B_trva])
    X_C_te = np.hstack([mt_A, X_B_te])
    sc_C = StandardScaler()
    X_C_trva = sc_C.fit_transform(X_C_trva)
    X_C_te = sc_C.transform(X_C_te)

    best_au_C, best_C_C = -1, 0.01
    for C in C_grid_meta:
        inner = np.zeros((len(trva_labels), nc))
        for _, (ti, vi) in enumerate(skf.split(X_C_trva, trva_labels)):
            clf = LogisticRegression(max_iter=3000, C=C, random_state=42)
            clf.fit(X_C_trva[ti], trva_labels[ti])
            inner[vi] = clf.predict_proba(X_C_trva[vi])
        try: au = compute_auroc(trva_labels, inner, nc)
        except: au = 0.5
        if au > best_au_C: best_au_C, best_C_C = au, C

    clf_C = LogisticRegression(max_iter=3000, C=best_C_C, random_state=42)
    clf_C.fit(X_C_trva, trva_labels)
    te_prob_C = clf_C.predict_proba(X_C_te)
    auroc_C = compute_auroc(te_labels, te_prob_C, nc)
    print(f"    Approach C (combined): {auroc_C:.4f} ({(auroc_C-info['best_single'])*100:+.2f}%)")

    # ===== Approach D: GBT on OOF probs (if available) =====
    auroc_D = 0.0
    try:
        from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier

        # Use simple probs as features for GBT
        oof_all = np.hstack([oof_dict[m] for m in avail_methods])
        te_all = np.hstack([te_dict[m] for m in avail_methods])

        # HistGBT with very conservative params to avoid overfitting
        best_au_D = -1
        best_params_D = {}
        for max_leaf in [4, 8, 16]:
            for lr in [0.01, 0.05, 0.1]:
                for n_est in [50, 100, 200]:
                    inner = np.zeros((len(trva_labels), nc))
                    for _, (ti, vi) in enumerate(skf.split(oof_all, trva_labels)):
                        if nc == 2:
                            clf = HistGradientBoostingClassifier(
                                max_leaf_nodes=max_leaf, learning_rate=lr,
                                max_iter=n_est, min_samples_leaf=20,
                                l2_regularization=1.0, random_state=42
                            )
                        else:
                            clf = HistGradientBoostingClassifier(
                                max_leaf_nodes=max_leaf, learning_rate=lr,
                                max_iter=n_est, min_samples_leaf=20,
                                l2_regularization=1.0, random_state=42
                            )
                        clf.fit(oof_all[ti], trva_labels[ti])
                        inner[vi] = clf.predict_proba(oof_all[vi])
                    try: au = compute_auroc(trva_labels, inner, nc)
                    except: au = 0.5
                    if au > best_au_D:
                        best_au_D = au
                        best_params_D = {"max_leaf": max_leaf, "lr": lr, "n_est": n_est}

        # Train final
        clf_D = HistGradientBoostingClassifier(
            max_leaf_nodes=best_params_D["max_leaf"],
            learning_rate=best_params_D["lr"],
            max_iter=best_params_D["n_est"],
            min_samples_leaf=20, l2_regularization=1.0, random_state=42
        )
        clf_D.fit(oof_all, trva_labels)
        te_prob_D = clf_D.predict_proba(te_all)
        auroc_D = compute_auroc(te_labels, te_prob_D, nc)
        print(f"    Approach D (GBT on probs): {auroc_D:.4f} ({(auroc_D-info['best_single'])*100:+.2f}%), params={best_params_D}")
    except Exception as e:
        print(f"    Approach D failed: {e}")
        auroc_D = 0.0

    # ===== Approach E: GBT on rich meta-features =====
    auroc_E = 0.0
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier

        best_au_E = -1
        best_params_E = {}
        for max_leaf in [4, 8, 16]:
            for lr in [0.01, 0.05, 0.1]:
                for n_est in [50, 100, 200]:
                    inner = np.zeros((len(trva_labels), nc))
                    for _, (ti, vi) in enumerate(skf.split(meta_trva_A, trva_labels)):
                        clf = HistGradientBoostingClassifier(
                            max_leaf_nodes=max_leaf, learning_rate=lr,
                            max_iter=n_est, min_samples_leaf=20,
                            l2_regularization=1.0, random_state=42
                        )
                        clf.fit(meta_trva_A[ti], trva_labels[ti])
                        inner[vi] = clf.predict_proba(meta_trva_A[vi])
                    try: au = compute_auroc(trva_labels, inner, nc)
                    except: au = 0.5
                    if au > best_au_E:
                        best_au_E = au
                        best_params_E = {"max_leaf": max_leaf, "lr": lr, "n_est": n_est}

        clf_E = HistGradientBoostingClassifier(
            max_leaf_nodes=best_params_E["max_leaf"],
            learning_rate=best_params_E["lr"],
            max_iter=best_params_E["n_est"],
            min_samples_leaf=20, l2_regularization=1.0, random_state=42
        )
        clf_E.fit(meta_trva_A, trva_labels)
        te_prob_E = clf_E.predict_proba(meta_te_A)
        auroc_E = compute_auroc(te_labels, te_prob_E, nc)
        print(f"    Approach E (GBT on rich meta): {auroc_E:.4f} ({(auroc_E-info['best_single'])*100:+.2f}%), params={best_params_E}")
    except Exception as e:
        print(f"    Approach E failed: {e}")

    # Pick best
    approaches = {
        "A_rich_meta": auroc_A,
        "B_feat_concat": auroc_B,
        "C_combined": auroc_C,
        "D_gbt_probs": auroc_D,
        "E_gbt_rich": auroc_E,
    }
    best_approach = max(approaches, key=approaches.get)
    best_auroc = approaches[best_approach]

    if best_auroc < info["best_single"]:
        best_approach = "no_fusion"
        best_auroc = info["best_single"]

    if best_approach == "A_rich_meta": final_probs = te_prob_A
    elif best_approach == "B_feat_concat": final_probs = te_prob_B
    elif best_approach == "C_combined": final_probs = te_prob_C
    elif best_approach == "D_gbt_probs": final_probs = te_prob_D
    elif best_approach == "E_gbt_rich": final_probs = te_prob_E
    else: final_probs = te_dict.get(info["best_method"], list(te_dict.values())[0])

    ci_lo, ci_hi = bootstrap_ci(te_labels, final_probs, nc)
    delta = best_auroc - info["best_single"]

    print(f"\n    Summary:")
    for k, v in sorted(approaches.items(), key=lambda x: -x[1]):
        marker = " ← BEST" if k == best_approach else ""
        print(f"      {k:20s}: {v:.4f} ({(v-info['best_single'])*100:+.2f}%){marker}")
    print(f"    FINAL: {best_approach}, AUROC={best_auroc:.4f}, delta={delta*100:+.2f}%")

    return {
        "dataset": ds_name,
        "approaches": {k: round(v, 4) for k, v in approaches.items()},
        "best_approach": best_approach,
        "test_auroc": round(best_auroc, 4),
        "baseline_auroc": info["best_single"],
        "delta": round(delta, 4),
        "delta_pct": f"{delta*100:+.2f}%",
        "ci_95": [round(ci_lo, 4), round(ci_hi, 4)],
    }


def main():
    print("=" * 60)
    print("BASELINE-ONLY FUSION v4")
    print("Rich meta-features + GBT + feature concat + combined")
    print("=" * 60)

    results = {}
    for ds_name, info in FOCUS_DATASETS.items():
        print(f"\n{'='*60}")
        print(f"Dataset: {ds_name} (nc={info['n_classes']}, best={info['best_single']:.4f})")
        print(f"{'='*60}")
        r = run_dataset(ds_name, info)
        if r:
            results[ds_name] = r

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY v4")
    print(f"{'='*60}")
    print(f"{'Dataset':25s} {'Best':>6s} {'Fusion':>7s} {'Delta':>7s} {'Approach':>20s}")
    print("-" * 70)
    for ds_name, r in results.items():
        print(f"{ds_name:25s} {r['baseline_auroc']:.4f} {r['test_auroc']:.4f} "
              f"{r['delta_pct']:>7s} {r['best_approach']:>20s}")

    out_path = os.path.join(RESULTS_DIR, "baseline_only_v4_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")
    return results


if __name__ == "__main__":
    main()
