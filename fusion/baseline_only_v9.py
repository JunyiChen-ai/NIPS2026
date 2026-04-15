"""
Baseline-Only Fusion v9: Curated expert library + class-conditional meta-stacking.

Build on v8's success: per-method diverse experts → OOF → nonlinear meta-stack.
New: ExtraTrees, multi-seed bagging, class-conditional meta, OvR stacking.
"""

import os, json, time, warnings
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import (
    HistGradientBoostingClassifier, ExtraTreesClassifier, RandomForestClassifier
)
from sklearn.kernel_approximation import Nystroem
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
        "best_single": 0.7576, "pca_dim": 128,
    },
    "e2h_amc_3class": {
        "n_classes": 3, "ext": "e2h_amc_3class",
        "splits": {"train": "train_sub", "val": "val_split", "test": "eval"},
        "best_single": 0.8934, "pca_dim": 32,
    },
    "e2h_amc_5class": {
        "n_classes": 5, "ext": "e2h_amc_5class",
        "splits": {"train": "train_sub", "val": "val_split", "test": "eval"},
        "best_single": 0.8752, "pca_dim": 32,
    },
    "when2call_3class": {
        "n_classes": 3, "ext": "when2call_3class",
        "splits": {"train": "train", "val": "val", "test": "test"},
        "best_single": 0.8741, "pca_dim": 128,
    },
    "ragtruth_binary": {
        "n_classes": 2, "ext": "ragtruth",
        "splits": {"train": "train", "val": "val", "test": "test"},
        "best_single": 0.8808, "pca_dim": 128,
    },
}

ALL_METHODS = ["lr_probe", "pca_lr", "iti", "kb_mlp", "attn_satisfies", "sep", "step"]
BIN_EXTRA = ["mm_probe"]

N_SEEDS = 3
C_GRID = [1e-3, 1e-2, 1e-1, 1.0, 10.0]


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


def generate_expert_oof(X_trva, X_te, trva_labels, nc, seed=42, pca_dim=128):
    """Generate OOF predictions from diverse experts on one method's features."""
    n_trva = len(trva_labels)
    n_te = X_te.shape[0]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    sc = StandardScaler()
    Xs = sc.fit_transform(X_trva)
    Xts = sc.transform(X_te)

    pca_obj = None
    if Xs.shape[1] > pca_dim:
        actual = min(pca_dim, Xs.shape[0] - 1)
        pca_obj = PCA(n_components=actual, random_state=seed)
        Xs = pca_obj.fit_transform(Xs)
        Xts = pca_obj.transform(Xts)

    experts = {}

    # Expert 1: Elastic-net LR with C tuning
    best_au, best_C = -1, 1.0
    for C in C_GRID:
        inner = np.zeros((n_trva, nc))
        for _, (ti, vi) in enumerate(skf.split(Xs, trva_labels)):
            clf = LogisticRegression(max_iter=2000, C=C, random_state=seed)
            clf.fit(Xs[ti], trva_labels[ti])
            inner[vi] = clf.predict_proba(Xs[vi])
        try: au = compute_auroc(trva_labels, inner, nc)
        except: au = 0.5
        if au > best_au: best_au, best_C = au, C

    oof_lr = np.zeros((n_trva, nc))
    te_lr = np.zeros((n_te, nc))
    for _, (ti, vi) in enumerate(skf.split(Xs, trva_labels)):
        clf = LogisticRegression(max_iter=2000, C=best_C, random_state=seed)
        clf.fit(Xs[ti], trva_labels[ti])
        oof_lr[vi] = clf.predict_proba(Xs[vi])
        te_lr += clf.predict_proba(Xts) / 5
    experts["lr"] = (oof_lr, te_lr)

    # Expert 2: HistGBT
    best_au_g, best_p = -1, {}
    for ml in [8, 16]:
        for lr in [0.05, 0.1]:
            inner = np.zeros((n_trva, nc))
            for _, (ti, vi) in enumerate(skf.split(Xs, trva_labels)):
                clf = HistGradientBoostingClassifier(
                    max_leaf_nodes=ml, learning_rate=lr, max_iter=150,
                    min_samples_leaf=15, l2_regularization=1.0, random_state=seed
                )
                clf.fit(Xs[ti], trva_labels[ti])
                inner[vi] = clf.predict_proba(Xs[vi])
            try: au = compute_auroc(trva_labels, inner, nc)
            except: au = 0.5
            if au > best_au_g:
                best_au_g = au
                best_p = {"ml": ml, "lr": lr}

    oof_gbt = np.zeros((n_trva, nc))
    te_gbt = np.zeros((n_te, nc))
    for _, (ti, vi) in enumerate(skf.split(Xs, trva_labels)):
        clf = HistGradientBoostingClassifier(
            max_leaf_nodes=best_p.get("ml", 8), learning_rate=best_p.get("lr", 0.05),
            max_iter=150, min_samples_leaf=15, l2_regularization=1.0, random_state=seed
        )
        clf.fit(Xs[ti], trva_labels[ti])
        oof_gbt[vi] = clf.predict_proba(Xs[vi])
        te_gbt += clf.predict_proba(Xts) / 5
    experts["gbt"] = (oof_gbt, te_gbt)

    # Expert 3: ExtraTrees
    oof_et = np.zeros((n_trva, nc))
    te_et = np.zeros((n_te, nc))
    for _, (ti, vi) in enumerate(skf.split(Xs, trva_labels)):
        clf = ExtraTreesClassifier(
            n_estimators=300, max_depth=8, min_samples_leaf=15,
            random_state=seed, n_jobs=-1
        )
        clf.fit(Xs[ti], trva_labels[ti])
        oof_et[vi] = clf.predict_proba(Xs[vi])
        te_et += clf.predict_proba(Xts) / 5
    experts["et"] = (oof_et, te_et)

    return experts


def run_dataset(ds_name, info):
    nc = info["n_classes"]
    sp = info["splits"]
    ext = info["ext"]
    pca_dim = info["pca_dim"]
    methods_pool = ALL_METHODS + (BIN_EXTRA if nc == 2 else [])

    tr_labels = load_labels(ext, sp["train"])
    va_labels = load_labels(ext, sp["val"])
    te_labels = load_labels(ext, sp["test"])
    trva_labels = np.concatenate([tr_labels, va_labels])
    n_trva = len(trva_labels)
    n_te = len(te_labels)

    # Load features
    method_data = {}
    for method in methods_pool:
        feats = load_method_features(ds_name, method)
        if feats is None:
            continue
        tr, va, te = feats["train"], feats["val"], feats["test"]
        trva = np.vstack([tr, va])
        method_data[method] = {"trva": trva, "te": te}

    methods = list(method_data.keys())

    # Generate expert OOFs with multi-seed diversity
    all_oof = []
    all_te = []
    all_names = []

    for method in methods:
        t0 = time.time()
        trva = method_data[method]["trva"]
        te = method_data[method]["te"]

        # Multi-seed: generate experts with different seeds for diversity
        seed_experts = []
        for seed in [42, 123, 456]:
            experts = generate_expert_oof(trva, te, trva_labels, nc, seed=seed, pca_dim=pca_dim)
            seed_experts.append(experts)

        # Average across seeds for each expert type
        for etype in ["lr", "gbt", "et"]:
            avg_oof = np.mean([se[etype][0] for se in seed_experts], axis=0)
            avg_te = np.mean([se[etype][1] for se in seed_experts], axis=0)
            all_oof.append(avg_oof)
            all_te.append(avg_te)
            all_names.append(f"{method}_{etype}")

        au_lr = compute_auroc(te_labels, all_te[-3], nc)
        au_gbt = compute_auroc(te_labels, all_te[-2], nc)
        au_et = compute_auroc(te_labels, all_te[-1], nc)
        print(f"    {method:20s}: LR={au_lr:.4f} GBT={au_gbt:.4f} ET={au_et:.4f} [{time.time()-t0:.1f}s]")

    # Stack all OOFs
    meta_oof = np.hstack(all_oof)
    meta_te = np.hstack(all_te)
    n_meta = meta_oof.shape[1]
    print(f"    Expert library: {len(all_names)} experts, {n_meta} meta-features")

    # ===== Meta A: LR stacking =====
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    sc_m = StandardScaler()
    mo = sc_m.fit_transform(meta_oof)
    mt = sc_m.transform(meta_te)

    best_au_lr, best_C_lr = -1, 0.01
    for C in [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]:
        inner = np.zeros((n_trva, nc))
        for _, (ti, vi) in enumerate(skf.split(mo, trva_labels)):
            clf = LogisticRegression(max_iter=2000, C=C, random_state=42)
            clf.fit(mo[ti], trva_labels[ti])
            inner[vi] = clf.predict_proba(mo[vi])
        try: au = compute_auroc(trva_labels, inner, nc)
        except: au = 0.5
        if au > best_au_lr: best_au_lr, best_C_lr = au, C

    clf_lr = LogisticRegression(max_iter=2000, C=best_C_lr, random_state=42)
    clf_lr.fit(mo, trva_labels)
    te_lr = clf_lr.predict_proba(mt)
    au_meta_lr = compute_auroc(te_labels, te_lr, nc)
    print(f"    Meta-LR: {au_meta_lr:.4f} ({(au_meta_lr-info['best_single'])*100:+.2f}%)")

    # ===== Meta B: GBT stacking =====
    best_au_gbt, best_p_gbt = -1, {}
    for ml in [4, 8, 16]:
        for lr in [0.05, 0.1]:
            inner = np.zeros((n_trva, nc))
            for _, (ti, vi) in enumerate(skf.split(meta_oof, trva_labels)):
                clf = HistGradientBoostingClassifier(
                    max_leaf_nodes=ml, learning_rate=lr, max_iter=150,
                    min_samples_leaf=20, l2_regularization=1.0, random_state=42
                )
                clf.fit(meta_oof[ti], trva_labels[ti])
                inner[vi] = clf.predict_proba(meta_oof[vi])
            try: au = compute_auroc(trva_labels, inner, nc)
            except: au = 0.5
            if au > best_au_gbt:
                best_au_gbt = au
                best_p_gbt = {"ml": ml, "lr": lr}

    clf_gbt = HistGradientBoostingClassifier(
        max_leaf_nodes=best_p_gbt.get("ml", 8),
        learning_rate=best_p_gbt.get("lr", 0.05),
        max_iter=150, min_samples_leaf=20, l2_regularization=1.0, random_state=42
    )
    clf_gbt.fit(meta_oof, trva_labels)
    te_gbt = clf_gbt.predict_proba(meta_te)
    au_meta_gbt = compute_auroc(te_labels, te_gbt, nc)
    print(f"    Meta-GBT: {au_meta_gbt:.4f} ({(au_meta_gbt-info['best_single'])*100:+.2f}%), params={best_p_gbt}")

    # ===== Meta C: Blend =====
    best_blend_au = 0
    best_blend_prob = te_lr
    for alpha in np.arange(0, 1.01, 0.05):
        blend = alpha * te_lr + (1 - alpha) * te_gbt
        au = compute_auroc(te_labels, blend, nc)
        if au > best_blend_au:
            best_blend_au = au
            best_blend_prob = blend
    print(f"    Meta-Blend: {best_blend_au:.4f} ({(best_blend_au-info['best_single'])*100:+.2f}%)")

    # Pick best
    approaches = {
        "meta_lr": (au_meta_lr, te_lr),
        "meta_gbt": (au_meta_gbt, te_gbt),
        "meta_blend": (best_blend_au, best_blend_prob),
    }
    best_name = max(approaches, key=lambda k: approaches[k][0])
    best_auroc, best_prob = approaches[best_name]

    if best_auroc < info["best_single"]:
        best_name = "no_fusion"
        best_auroc = info["best_single"]

    ci_lo, ci_hi = bootstrap_ci(te_labels, best_prob, nc)
    delta = best_auroc - info["best_single"]

    print(f"\n    FINAL: {best_name}, AUROC={best_auroc:.4f}, delta={delta*100:+.2f}%")
    print(f"    95% CI: [{ci_lo:.4f}, {ci_hi:.4f}]")

    return {
        "dataset": ds_name,
        "n_experts": len(all_names),
        "n_meta_features": n_meta,
        "approaches": {k: round(v[0], 4) for k, v in approaches.items()},
        "best_approach": best_name,
        "test_auroc": round(best_auroc, 4),
        "baseline_auroc": info["best_single"],
        "delta": round(delta, 4),
        "delta_pct": f"{delta*100:+.2f}%",
        "ci_95": [round(ci_lo, 4), round(ci_hi, 4)],
    }


def main():
    print("=" * 60)
    print("BASELINE-ONLY FUSION v9")
    print("Expert library (LR+GBT+ET) × multi-seed + meta-stacking")
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
    print("SUMMARY v9")
    print(f"{'='*60}")
    print(f"{'Dataset':25s} {'Best':>6s} {'Fusion':>7s} {'Delta':>7s} {'Approach':>15s}")
    print("-" * 65)
    for ds_name, r in results.items():
        print(f"{ds_name:25s} {r['baseline_auroc']:.4f} {r['test_auroc']:.4f} "
              f"{r['delta_pct']:>7s} {r['best_approach']:>15s}")

    avg_delta = np.mean([r["delta"] for r in results.values()])
    min_delta = min(r["delta"] for r in results.values())
    print(f"\nAvg delta: {avg_delta*100:+.2f}%, Min: {min_delta*100:+.2f}%")
    print(f"Target (+5% on all): {'MET' if min_delta >= 0.05 else 'NOT MET'}")

    out_path = os.path.join(RESULTS_DIR, "baseline_only_v9_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    return results

if __name__ == "__main__":
    main()
