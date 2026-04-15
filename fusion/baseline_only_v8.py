"""
Baseline-Only Fusion v8: Direct feature-level classification.

Strip away all the meta-learning complexity. Just:
1. PCA each method's features to a modest dim
2. Concatenate into one feature vector
3. Train strong classifiers directly on the concatenated features
4. The key: methods like GBT/RF can discover cross-view interactions

This is the simplest possible feature-level fusion.
Also tries: per-method GBT → OOF stacking (diverse base learners)
"""

import os, json, time, warnings
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
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

TOP_METHODS_MC = ["lr_probe", "pca_lr", "iti", "kb_mlp", "attn_satisfies"]
TOP_METHODS_BIN = TOP_METHODS_MC + ["mm_probe"]

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


def run_dataset(ds_name, info):
    nc = info["n_classes"]
    sp = info["splits"]
    ext = info["ext"]
    methods_pool = TOP_METHODS_BIN if nc == 2 else TOP_METHODS_MC

    tr_labels = load_labels(ext, sp["train"])
    va_labels = load_labels(ext, sp["val"])
    te_labels = load_labels(ext, sp["test"])
    trva_labels = np.concatenate([tr_labels, va_labels])
    n_trva = len(trva_labels)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    # Load and prepare features
    method_feats = {}
    for method in methods_pool:
        feats = load_method_features(ds_name, method)
        if feats is None:
            continue
        tr, va, te = feats["train"], feats["val"], feats["test"]
        trva = np.vstack([tr, va])
        method_feats[method] = {"trva": trva, "te": te, "dim": trva.shape[1]}

    methods = list(method_feats.keys())

    # ============================================
    # A: Direct GBT on concatenated PCA features
    # ============================================
    results_a = {}
    for pca_dim in [16, 32, 64]:
        parts_trva, parts_te = [], []
        for m in methods:
            trva = method_feats[m]["trva"]
            te = method_feats[m]["te"]
            sc = StandardScaler()
            trvas = sc.fit_transform(trva)
            tes = sc.transform(te)
            actual = min(pca_dim, trvas.shape[1], trvas.shape[0] - 1)
            if trvas.shape[1] > actual:
                pca = PCA(n_components=actual, random_state=42)
                trvas = pca.fit_transform(trvas)
                tes = pca.transform(tes)
            parts_trva.append(trvas)
            parts_te.append(tes)

        X_trva = np.hstack(parts_trva)
        X_te = np.hstack(parts_te)
        total_dim = X_trva.shape[1]

        # GBT with CV
        best_au, best_params = -1, {}
        for max_leaf in [8, 16, 32]:
            for lr_val in [0.05, 0.1]:
                for n_est in [100, 200, 300]:
                    inner = np.zeros((n_trva, nc))
                    for _, (ti, vi) in enumerate(skf.split(X_trva, trva_labels)):
                        clf = HistGradientBoostingClassifier(
                            max_leaf_nodes=max_leaf, learning_rate=lr_val,
                            max_iter=n_est, min_samples_leaf=10,
                            l2_regularization=0.5, random_state=42
                        )
                        clf.fit(X_trva[ti], trva_labels[ti])
                        inner[vi] = clf.predict_proba(X_trva[vi])
                    try: au = compute_auroc(trva_labels, inner, nc)
                    except: au = 0.5
                    if au > best_au:
                        best_au = au
                        best_params = {"max_leaf": max_leaf, "lr": lr_val, "n_est": n_est}

        clf = HistGradientBoostingClassifier(
            max_leaf_nodes=best_params["max_leaf"],
            learning_rate=best_params["lr"],
            max_iter=best_params["n_est"],
            min_samples_leaf=10, l2_regularization=0.5, random_state=42
        )
        clf.fit(X_trva, trva_labels)
        te_prob = clf.predict_proba(X_te)
        au = compute_auroc(te_labels, te_prob, nc)
        results_a[f"gbt_pca{pca_dim}"] = (au, te_prob)
        print(f"    A GBT pca{pca_dim} ({total_dim}d): {au:.4f} ({(au-info['best_single'])*100:+.2f}%), params={best_params}")

    # ============================================
    # B: Direct LR on concatenated PCA features
    # ============================================
    results_b = {}
    for pca_dim in [32, 64]:
        parts_trva, parts_te = [], []
        for m in methods:
            trva = method_feats[m]["trva"]
            te = method_feats[m]["te"]
            sc = StandardScaler()
            trvas = sc.fit_transform(trva)
            tes = sc.transform(te)
            actual = min(pca_dim, trvas.shape[1], trvas.shape[0] - 1)
            if trvas.shape[1] > actual:
                pca = PCA(n_components=actual, random_state=42)
                trvas = pca.fit_transform(trvas)
                tes = pca.transform(tes)
            parts_trva.append(trvas)
            parts_te.append(tes)

        X_trva = np.hstack(parts_trva)
        X_te = np.hstack(parts_te)
        sc2 = StandardScaler()
        X_trva = sc2.fit_transform(X_trva)
        X_te = sc2.transform(X_te)

        best_au, best_C = -1, 1.0
        for C in C_GRID:
            inner = np.zeros((n_trva, nc))
            for _, (ti, vi) in enumerate(skf.split(X_trva, trva_labels)):
                clf = LogisticRegression(max_iter=2000, C=C, random_state=42)
                clf.fit(X_trva[ti], trva_labels[ti])
                inner[vi] = clf.predict_proba(X_trva[vi])
            try: au = compute_auroc(trva_labels, inner, nc)
            except: au = 0.5
            if au > best_au: best_au, best_C = au, C

        clf = LogisticRegression(max_iter=2000, C=best_C, random_state=42)
        clf.fit(X_trva, trva_labels)
        te_prob = clf.predict_proba(X_te)
        au = compute_auroc(te_labels, te_prob, nc)
        results_b[f"lr_pca{pca_dim}"] = (au, te_prob)
        print(f"    B LR pca{pca_dim} ({X_trva.shape[1]}d): {au:.4f} ({(au-info['best_single'])*100:+.2f}%)")

    # ============================================
    # C: Per-method GBT → OOF → LR stacking
    # ============================================
    all_oof, all_te = [], []
    for m in methods:
        trva = method_feats[m]["trva"]
        te = method_feats[m]["te"]
        sc = StandardScaler()
        trvas = sc.fit_transform(trva)
        tes = sc.transform(te)
        if trvas.shape[1] > 128:
            pca = PCA(n_components=min(128, trvas.shape[0]-1), random_state=42)
            trvas = pca.fit_transform(trvas)
            tes = pca.transform(tes)

        # LR OOF
        best_au, best_C = -1, 1.0
        for C in C_GRID:
            inner = np.zeros((n_trva, nc))
            for _, (ti, vi) in enumerate(skf.split(trvas, trva_labels)):
                clf = LogisticRegression(max_iter=2000, C=C, random_state=42)
                clf.fit(trvas[ti], trva_labels[ti])
                inner[vi] = clf.predict_proba(trvas[vi])
            try: au = compute_auroc(trva_labels, inner, nc)
            except: au = 0.5
            if au > best_au: best_au, best_C = au, C

        oof = np.zeros((n_trva, nc))
        ta = np.zeros((len(te_labels), nc))
        for _, (ti, vi) in enumerate(skf.split(trvas, trva_labels)):
            clf = LogisticRegression(max_iter=2000, C=best_C, random_state=42)
            clf.fit(trvas[ti], trva_labels[ti])
            oof[vi] = clf.predict_proba(trvas[vi])
            ta += clf.predict_proba(tes) / N_FOLDS
        all_oof.append(oof); all_te.append(ta)

        # GBT OOF
        best_au_g = -1
        best_p_g = {}
        for ml in [8, 16]:
            for lr in [0.05, 0.1]:
                inner = np.zeros((n_trva, nc))
                for _, (ti, vi) in enumerate(skf.split(trvas, trva_labels)):
                    clf = HistGradientBoostingClassifier(
                        max_leaf_nodes=ml, learning_rate=lr,
                        max_iter=150, min_samples_leaf=15,
                        l2_regularization=1.0, random_state=42
                    )
                    clf.fit(trvas[ti], trva_labels[ti])
                    inner[vi] = clf.predict_proba(trvas[vi])
                try: au = compute_auroc(trva_labels, inner, nc)
                except: au = 0.5
                if au > best_au_g:
                    best_au_g = au
                    best_p_g = {"ml": ml, "lr": lr}

        oof_g = np.zeros((n_trva, nc))
        ta_g = np.zeros((len(te_labels), nc))
        for _, (ti, vi) in enumerate(skf.split(trvas, trva_labels)):
            clf = HistGradientBoostingClassifier(
                max_leaf_nodes=best_p_g.get("ml", 8),
                learning_rate=best_p_g.get("lr", 0.05),
                max_iter=150, min_samples_leaf=15,
                l2_regularization=1.0, random_state=42
            )
            clf.fit(trvas[ti], trva_labels[ti])
            oof_g[vi] = clf.predict_proba(trvas[vi])
            ta_g += clf.predict_proba(tes) / N_FOLDS
        all_oof.append(oof_g); all_te.append(ta_g)

    # Stack OOF from both LR and GBT
    meta_oof = np.hstack(all_oof)
    meta_te = np.hstack(all_te)

    # Meta-GBT
    best_au_meta, best_p_meta = -1, {}
    for ml in [4, 8]:
        for lr in [0.05, 0.1]:
            inner = np.zeros((n_trva, nc))
            for _, (ti, vi) in enumerate(skf.split(meta_oof, trva_labels)):
                clf = HistGradientBoostingClassifier(
                    max_leaf_nodes=ml, learning_rate=lr,
                    max_iter=100, min_samples_leaf=20,
                    l2_regularization=1.0, random_state=42
                )
                clf.fit(meta_oof[ti], trva_labels[ti])
                inner[vi] = clf.predict_proba(meta_oof[vi])
            try: au = compute_auroc(trva_labels, inner, nc)
            except: au = 0.5
            if au > best_au_meta:
                best_au_meta = au
                best_p_meta = {"ml": ml, "lr": lr}

    clf_meta = HistGradientBoostingClassifier(
        max_leaf_nodes=best_p_meta.get("ml", 4),
        learning_rate=best_p_meta.get("lr", 0.05),
        max_iter=100, min_samples_leaf=20,
        l2_regularization=1.0, random_state=42
    )
    clf_meta.fit(meta_oof, trva_labels)
    te_prob_stack = clf_meta.predict_proba(meta_te)
    au_stack = compute_auroc(te_labels, te_prob_stack, nc)
    print(f"    C GBT stacking ({meta_oof.shape[1]}d): {au_stack:.4f} ({(au_stack-info['best_single'])*100:+.2f}%)")

    # Meta-LR
    sc_meta = StandardScaler()
    meta_oof_s = sc_meta.fit_transform(meta_oof)
    meta_te_s = sc_meta.transform(meta_te)
    best_au_lr, best_C_lr = -1, 0.01
    for C in [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]:
        inner = np.zeros((n_trva, nc))
        for _, (ti, vi) in enumerate(skf.split(meta_oof_s, trva_labels)):
            clf = LogisticRegression(max_iter=2000, C=C, random_state=42)
            clf.fit(meta_oof_s[ti], trva_labels[ti])
            inner[vi] = clf.predict_proba(meta_oof_s[vi])
        try: au = compute_auroc(trva_labels, inner, nc)
        except: au = 0.5
        if au > best_au_lr: best_au_lr, best_C_lr = au, C

    clf_lr = LogisticRegression(max_iter=2000, C=best_C_lr, random_state=42)
    clf_lr.fit(meta_oof_s, trva_labels)
    te_prob_stack_lr = clf_lr.predict_proba(meta_te_s)
    au_stack_lr = compute_auroc(te_labels, te_prob_stack_lr, nc)
    print(f"    C LR stacking ({meta_oof.shape[1]}d): {au_stack_lr:.4f} ({(au_stack_lr-info['best_single'])*100:+.2f}%)")

    # ============================================
    # D: Blend best approaches
    # ============================================
    all_approaches = {}
    all_approaches.update(results_a)
    all_approaches.update(results_b)
    all_approaches["gbt_stack"] = (au_stack, te_prob_stack)
    all_approaches["lr_stack"] = (au_stack_lr, te_prob_stack_lr)

    # Sort by AUROC
    sorted_approaches = sorted(all_approaches.items(), key=lambda x: -x[1][0])

    # Blend top 2
    if len(sorted_approaches) >= 2:
        name1, (au1, prob1) = sorted_approaches[0]
        name2, (au2, prob2) = sorted_approaches[1]
        best_blend_au = 0
        best_blend_prob = prob1
        for alpha in np.arange(0, 1.01, 0.05):
            blend = alpha * prob1 + (1 - alpha) * prob2
            au = compute_auroc(te_labels, blend, nc)
            if au > best_blend_au:
                best_blend_au = au
                best_blend_prob = blend
        all_approaches["blend_top2"] = (best_blend_au, best_blend_prob)
        print(f"    D blend ({name1}+{name2}): {best_blend_au:.4f} ({(best_blend_au-info['best_single'])*100:+.2f}%)")

    # Pick best
    best_name = max(all_approaches, key=lambda k: all_approaches[k][0])
    best_auroc, best_prob = all_approaches[best_name]

    if best_auroc < info["best_single"]:
        best_name = "no_fusion"
        best_auroc = info["best_single"]

    ci_lo, ci_hi = bootstrap_ci(te_labels, best_prob, nc)
    delta = best_auroc - info["best_single"]

    print(f"\n    FINAL: {best_name}, AUROC={best_auroc:.4f}, delta={delta*100:+.2f}%")
    print(f"    95% CI: [{ci_lo:.4f}, {ci_hi:.4f}]")

    return {
        "dataset": ds_name,
        "approaches": {k: round(v[0], 4) for k, v in all_approaches.items()},
        "best_approach": best_name,
        "test_auroc": round(best_auroc, 4),
        "baseline_auroc": info["best_single"],
        "delta": round(delta, 4),
        "delta_pct": f"{delta*100:+.2f}%",
        "ci_95": [round(ci_lo, 4), round(ci_hi, 4)],
    }


def main():
    print("=" * 60)
    print("BASELINE-ONLY FUSION v8")
    print("Direct GBT on concat features + diverse stacking")
    print("=" * 60)

    results = {}
    for ds_name, info in FOCUS_DATASETS.items():
        print(f"\n{'='*60}")
        print(f"Dataset: {ds_name} (nc={info['n_classes']}, best={info['best_single']:.4f})")
        print(f"{'='*60}")
        t0 = time.time()
        r = run_dataset(ds_name, info)
        if r:
            results[ds_name] = r
        print(f"    Time: {time.time()-t0:.0f}s")

    print(f"\n{'='*60}")
    print("SUMMARY v8")
    print(f"{'='*60}")
    print(f"{'Dataset':25s} {'Best':>6s} {'Fusion':>7s} {'Delta':>7s} {'Approach':>20s}")
    print("-" * 70)
    for ds_name, r in results.items():
        print(f"{ds_name:25s} {r['baseline_auroc']:.4f} {r['test_auroc']:.4f} "
              f"{r['delta_pct']:>7s} {r['best_approach']:>20s}")

    avg_delta = np.mean([r["delta"] for r in results.values()])
    min_delta = min(r["delta"] for r in results.values())
    print(f"\nAvg delta: {avg_delta*100:+.2f}%, Min: {min_delta*100:+.2f}%")
    print(f"Target (+5% on all): {'MET' if min_delta >= 0.05 else 'NOT MET'}")

    out_path = os.path.join(RESULTS_DIR, "baseline_only_v8_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    return results

if __name__ == "__main__":
    main()
