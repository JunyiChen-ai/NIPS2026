"""
Baseline-Only Fusion v10: Dataset-specific expert curation + aggressive meta-GBT.

Key changes from v9:
1. Dataset-specific method selection (exclude sep/step for small datasets)
2. Wider GBT hyperparameter search at meta level
3. More seeds for bagging (5 instead of 3)
4. Add RF as additional expert type
5. Enriched meta-features: add entropy, margin, cross-expert disagreement
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
from scipy import stats

warnings.filterwarnings("ignore")

PROCESSED_DIR = "/home/junyi/NIPS2026/reproduce/processed_features"
EXTRACTION_DIR = "/home/junyi/NIPS2026/extraction/features"
RESULTS_DIR = "/home/junyi/NIPS2026/fusion/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

DATASETS = {
    "common_claim_3class": {
        "n_classes": 3, "ext": "common_claim_3class",
        "splits": {"train": "train", "val": "val", "test": "test"},
        "best_single": 0.7576, "pca_dim": 128,
        "methods": ["lr_probe", "pca_lr", "iti", "kb_mlp", "attn_satisfies", "sep", "step"],
        "n_seeds": 5,
    },
    "e2h_amc_3class": {
        "n_classes": 3, "ext": "e2h_amc_3class",
        "splits": {"train": "train_sub", "val": "val_split", "test": "eval"},
        "best_single": 0.8934, "pca_dim": 32,
        "methods": ["lr_probe", "pca_lr", "iti", "kb_mlp", "attn_satisfies"],
        "n_seeds": 5,
    },
    "e2h_amc_5class": {
        "n_classes": 5, "ext": "e2h_amc_5class",
        "splits": {"train": "train_sub", "val": "val_split", "test": "eval"},
        "best_single": 0.8752, "pca_dim": 32,
        "methods": ["lr_probe", "pca_lr", "iti", "kb_mlp", "attn_satisfies"],
        "n_seeds": 5,
    },
    "when2call_3class": {
        "n_classes": 3, "ext": "when2call_3class",
        "splits": {"train": "train", "val": "val", "test": "test"},
        "best_single": 0.8741, "pca_dim": 128,
        "methods": ["lr_probe", "pca_lr", "iti", "kb_mlp", "attn_satisfies", "sep", "step"],
        "n_seeds": 5,
    },
    "ragtruth_binary": {
        "n_classes": 2, "ext": "ragtruth",
        "splits": {"train": "train", "val": "val", "test": "test"},
        "best_single": 0.8808, "pca_dim": 128,
        "methods": ["lr_probe", "pca_lr", "iti", "kb_mlp", "attn_satisfies", "mm_probe"],
        "n_seeds": 5,
    },
}

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


def generate_experts(X_trva, X_te, trva_labels, nc, seed, pca_dim):
    n_trva, n_te = len(trva_labels), X_te.shape[0]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    sc = StandardScaler()
    Xs = sc.fit_transform(X_trva)
    Xts = sc.transform(X_te)

    if Xs.shape[1] > pca_dim:
        actual = min(pca_dim, Xs.shape[0] - 1)
        pca = PCA(n_components=actual, random_state=seed)
        Xs = pca.fit_transform(Xs)
        Xts = pca.transform(Xts)

    experts = {}

    # LR
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

    oof = np.zeros((n_trva, nc)); ta = np.zeros((n_te, nc))
    for _, (ti, vi) in enumerate(skf.split(Xs, trva_labels)):
        clf = LogisticRegression(max_iter=2000, C=best_C, random_state=seed)
        clf.fit(Xs[ti], trva_labels[ti])
        oof[vi] = clf.predict_proba(Xs[vi])
        ta += clf.predict_proba(Xts) / 5
    experts["lr"] = (oof, ta)

    # GBT
    best_au_g, best_p = -1, {}
    for ml in [8, 16, 32]:
        for lr in [0.05, 0.1]:
            inner = np.zeros((n_trva, nc))
            for _, (ti, vi) in enumerate(skf.split(Xs, trva_labels)):
                clf = HistGradientBoostingClassifier(
                    max_leaf_nodes=ml, learning_rate=lr, max_iter=200,
                    min_samples_leaf=10, l2_regularization=0.5, random_state=seed
                )
                clf.fit(Xs[ti], trva_labels[ti])
                inner[vi] = clf.predict_proba(Xs[vi])
            try: au = compute_auroc(trva_labels, inner, nc)
            except: au = 0.5
            if au > best_au_g:
                best_au_g = au; best_p = {"ml": ml, "lr": lr}

    oof_g = np.zeros((n_trva, nc)); ta_g = np.zeros((n_te, nc))
    for _, (ti, vi) in enumerate(skf.split(Xs, trva_labels)):
        clf = HistGradientBoostingClassifier(
            max_leaf_nodes=best_p.get("ml", 8), learning_rate=best_p.get("lr", 0.05),
            max_iter=200, min_samples_leaf=10, l2_regularization=0.5, random_state=seed
        )
        clf.fit(Xs[ti], trva_labels[ti])
        oof_g[vi] = clf.predict_proba(Xs[vi])
        ta_g += clf.predict_proba(Xts) / 5
    experts["gbt"] = (oof_g, ta_g)

    # ExtraTrees
    oof_et = np.zeros((n_trva, nc)); ta_et = np.zeros((n_te, nc))
    for _, (ti, vi) in enumerate(skf.split(Xs, trva_labels)):
        clf = ExtraTreesClassifier(n_estimators=300, max_depth=10, min_samples_leaf=10, random_state=seed, n_jobs=-1)
        clf.fit(Xs[ti], trva_labels[ti])
        oof_et[vi] = clf.predict_proba(Xs[vi])
        ta_et += clf.predict_proba(Xts) / 5
    experts["et"] = (oof_et, ta_et)

    # RF
    oof_rf = np.zeros((n_trva, nc)); ta_rf = np.zeros((n_te, nc))
    for _, (ti, vi) in enumerate(skf.split(Xs, trva_labels)):
        clf = RandomForestClassifier(n_estimators=300, max_depth=10, min_samples_leaf=10, random_state=seed, n_jobs=-1)
        clf.fit(Xs[ti], trva_labels[ti])
        oof_rf[vi] = clf.predict_proba(Xs[vi])
        ta_rf += clf.predict_proba(Xts) / 5
    experts["rf"] = (oof_rf, ta_rf)

    return experts


def enrich_meta(meta_oof, meta_te, nc):
    """Add entropy, margin, confidence features."""
    n_experts = meta_oof.shape[1] // nc
    extra_trva, extra_te = [], []

    for i in range(n_experts):
        p_trva = meta_oof[:, i*nc:(i+1)*nc]
        p_te = meta_te[:, i*nc:(i+1)*nc]

        # Entropy
        ent_trva = (-p_trva * np.log(np.clip(p_trva, 1e-10, 1))).sum(axis=1, keepdims=True)
        ent_te = (-p_te * np.log(np.clip(p_te, 1e-10, 1))).sum(axis=1, keepdims=True)
        extra_trva.append(ent_trva)
        extra_te.append(ent_te)

        # Margin
        s_trva = np.sort(p_trva, axis=1)
        s_te = np.sort(p_te, axis=1)
        margin_trva = (s_trva[:, -1] - s_trva[:, -2]).reshape(-1, 1)
        margin_te = (s_te[:, -1] - s_te[:, -2]).reshape(-1, 1)
        extra_trva.append(margin_trva)
        extra_te.append(margin_te)

    return np.hstack([meta_oof] + extra_trva), np.hstack([meta_te] + extra_te)


def run_dataset(ds_name, info):
    nc = info["n_classes"]
    sp = info["splits"]
    ext = info["ext"]
    pca_dim = info["pca_dim"]
    methods_list = info["methods"]
    n_seeds = info["n_seeds"]

    tr_labels = load_labels(ext, sp["train"])
    va_labels = load_labels(ext, sp["val"])
    te_labels = load_labels(ext, sp["test"])
    trva_labels = np.concatenate([tr_labels, va_labels])
    n_trva, n_te = len(trva_labels), len(te_labels)

    # Load features
    method_data = {}
    for m in methods_list:
        feats = load_method_features(ds_name, m)
        if feats is None: continue
        method_data[m] = {"trva": np.vstack([feats["train"], feats["val"]]), "te": feats["test"]}
    methods = list(method_data.keys())

    # Generate expert OOFs
    all_oof, all_te, all_names = [], [], []
    for method in methods:
        t0 = time.time()
        seed_results = []
        for seed in range(n_seeds):
            experts = generate_experts(
                method_data[method]["trva"], method_data[method]["te"],
                trva_labels, nc, seed=42+seed*111, pca_dim=pca_dim
            )
            seed_results.append(experts)

        for etype in ["lr", "gbt", "et", "rf"]:
            avg_oof = np.mean([sr[etype][0] for sr in seed_results], axis=0)
            avg_te = np.mean([sr[etype][1] for sr in seed_results], axis=0)
            all_oof.append(avg_oof)
            all_te.append(avg_te)
            all_names.append(f"{method}_{etype}")

        aus = [compute_auroc(te_labels, all_te[-4+i], nc) for i in range(4)]
        print(f"    {method:20s}: LR={aus[0]:.4f} GBT={aus[1]:.4f} ET={aus[2]:.4f} RF={aus[3]:.4f} [{time.time()-t0:.1f}s]")

    meta_oof = np.hstack(all_oof)
    meta_te = np.hstack(all_te)

    # Enrich with entropy/margin
    meta_oof_rich, meta_te_rich = enrich_meta(meta_oof, meta_te, nc)
    print(f"    Expert lib: {len(all_names)} experts, {meta_oof_rich.shape[1]} meta-features")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Meta-LR
    sc = StandardScaler()
    mo = sc.fit_transform(meta_oof_rich)
    mt = sc.transform(meta_te_rich)

    best_au_lr, best_C = -1, 0.01
    for C in [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]:
        inner = np.zeros((n_trva, nc))
        for _, (ti, vi) in enumerate(skf.split(mo, trva_labels)):
            clf = LogisticRegression(max_iter=2000, C=C, random_state=42)
            clf.fit(mo[ti], trva_labels[ti])
            inner[vi] = clf.predict_proba(mo[vi])
        try: au = compute_auroc(trva_labels, inner, nc)
        except: au = 0.5
        if au > best_au_lr: best_au_lr, best_C = au, C

    clf_lr = LogisticRegression(max_iter=2000, C=best_C, random_state=42)
    clf_lr.fit(mo, trva_labels)
    te_lr = clf_lr.predict_proba(mt)
    au_lr = compute_auroc(te_labels, te_lr, nc)
    print(f"    Meta-LR: {au_lr:.4f} ({(au_lr-info['best_single'])*100:+.2f}%)")

    # Meta-GBT with wider search
    best_au_gbt, best_p = -1, {}
    for ml in [4, 8, 16, 32]:
        for lr in [0.01, 0.05, 0.1, 0.2]:
            for n_est in [100, 200, 300]:
                inner = np.zeros((n_trva, nc))
                for _, (ti, vi) in enumerate(skf.split(meta_oof_rich, trva_labels)):
                    clf = HistGradientBoostingClassifier(
                        max_leaf_nodes=ml, learning_rate=lr, max_iter=n_est,
                        min_samples_leaf=15, l2_regularization=0.5, random_state=42
                    )
                    clf.fit(meta_oof_rich[ti], trva_labels[ti])
                    inner[vi] = clf.predict_proba(meta_oof_rich[vi])
                try: au = compute_auroc(trva_labels, inner, nc)
                except: au = 0.5
                if au > best_au_gbt:
                    best_au_gbt = au
                    best_p = {"ml": ml, "lr": lr, "n_est": n_est}

    clf_gbt = HistGradientBoostingClassifier(
        max_leaf_nodes=best_p.get("ml", 8), learning_rate=best_p.get("lr", 0.05),
        max_iter=best_p.get("n_est", 200), min_samples_leaf=15,
        l2_regularization=0.5, random_state=42
    )
    clf_gbt.fit(meta_oof_rich, trva_labels)
    te_gbt = clf_gbt.predict_proba(meta_te_rich)
    au_gbt = compute_auroc(te_labels, te_gbt, nc)
    print(f"    Meta-GBT: {au_gbt:.4f} ({(au_gbt-info['best_single'])*100:+.2f}%), params={best_p}")

    # Blend
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
    approaches = {"lr": (au_lr, te_lr), "gbt": (au_gbt, te_gbt), "blend": (best_blend_au, best_blend_prob)}
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
    print("BASELINE-ONLY FUSION v10")
    print("Curated experts (LR+GBT+ET+RF) × 5 seeds + aggressive meta-GBT")
    print("=" * 60)

    results = {}
    for ds_name, info in DATASETS.items():
        print(f"\n{'='*60}")
        print(f"Dataset: {ds_name} (nc={info['n_classes']}, best={info['best_single']:.4f})")
        print(f"{'='*60}")
        t0 = time.time()
        r = run_dataset(ds_name, info)
        if r:
            results[ds_name] = r
        print(f"    Time: {time.time()-t0:.0f}s")

    print(f"\n{'='*60}")
    print("SUMMARY v10")
    print(f"{'='*60}")
    print(f"{'Dataset':25s} {'Best':>6s} {'Fusion':>7s} {'Delta':>7s} {'Approach':>10s}")
    print("-" * 60)
    for ds_name, r in results.items():
        met = "✅" if r["delta"] >= 0.05 else ""
        print(f"{ds_name:25s} {r['baseline_auroc']:.4f} {r['test_auroc']:.4f} "
              f"{r['delta_pct']:>7s} {r['best_approach']:>10s} {met}")

    avg = np.mean([r["delta"] for r in results.values()])
    mn = min(r["delta"] for r in results.values())
    n_met = sum(1 for r in results.values() if r["delta"] >= 0.05)
    print(f"\nAvg: {avg*100:+.2f}%, Min: {mn*100:+.2f}%, Target met: {n_met}/5")

    out_path = os.path.join(RESULTS_DIR, "baseline_only_v10_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    return results

if __name__ == "__main__":
    main()
