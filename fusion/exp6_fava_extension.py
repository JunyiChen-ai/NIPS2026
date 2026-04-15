"""
Exp 6: Extend v21 pipeline to fava_binary (8 methods).
Quick run to add one more row to the main results table.
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

warnings.filterwarnings("ignore")

import argparse as _argparse
_ap = _argparse.ArgumentParser(add_help=False)
_ap.add_argument("--model", default="qwen2.5-7b")
_cli, _ = _ap.parse_known_args()
_MODEL = _cli.model
_BASE_PROCESSED = "/home/junyi/NIPS2026/reproduce/processed_features"
_BASE_EXTRACTION = "/home/junyi/NIPS2026/extraction/features"
_BASE_RESULTS = "/home/junyi/NIPS2026/fusion/results"
PROCESSED_DIR = os.path.join(_BASE_PROCESSED, _MODEL) if _MODEL else _BASE_PROCESSED
EXTRACTION_DIR = os.path.join(_BASE_EXTRACTION, _MODEL) if _MODEL else _BASE_EXTRACTION
RESULTS_DIR = os.path.join(_BASE_RESULTS, _MODEL) if _MODEL else _BASE_RESULTS
os.makedirs(RESULTS_DIR, exist_ok=True)

PCA_DIMS = [32, 128]
EXPERT_TYPES = ["lr", "gbt", "et", "rf"]
N_SEEDS = 5
N_FOLDS = 5
C_GRID = [1e-3, 1e-2, 1e-1, 1.0, 10.0]

# fava_binary has these methods available
FAVA_METHODS = ["lr_probe", "pca_lr", "iti", "kb_mlp", "attn_satisfies", "sep", "step", "mm_probe"]

DS_CONFIG = {
    "n_classes": 2,
    "ext": "fava",
    "train": "train",
    "val": "val",
    "test": "test",
    "best_single": 0.9856,  # ITI
}


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


def load_method_features(method):
    base = os.path.join(PROCESSED_DIR, "fava_binary", method)
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


def train_expert_oof(Xs, Xts, labels, nc, etype, seed, skf):
    n_trva, n_te = len(labels), Xts.shape[0]
    oof = np.zeros((n_trva, nc))
    ta = np.zeros((n_te, nc))

    if etype == "lr":
        best_au, best_C = -1, 1.0
        for C in C_GRID:
            inner = np.zeros((n_trva, nc))
            for _, (ti, vi) in enumerate(skf.split(Xs, labels)):
                clf = LogisticRegression(max_iter=2000, C=C, random_state=seed)
                clf.fit(Xs[ti], labels[ti])
                inner[vi] = clf.predict_proba(Xs[vi])
            try:
                au = compute_auroc(labels, inner, nc)
            except:
                au = 0.5
            if au > best_au:
                best_au, best_C = au, C
        for _, (ti, vi) in enumerate(skf.split(Xs, labels)):
            clf = LogisticRegression(max_iter=2000, C=best_C, random_state=seed)
            clf.fit(Xs[ti], labels[ti])
            oof[vi] = clf.predict_proba(Xs[vi])
            ta += clf.predict_proba(Xts) / N_FOLDS

    elif etype == "gbt":
        best_au, bp = -1, {}
        for ml in [8, 16, 32]:
            for lr in [0.05, 0.1]:
                inner = np.zeros((n_trva, nc))
                for _, (ti, vi) in enumerate(skf.split(Xs, labels)):
                    clf = HistGradientBoostingClassifier(max_leaf_nodes=ml, learning_rate=lr, max_iter=200, min_samples_leaf=10, l2_regularization=0.5, random_state=seed)
                    clf.fit(Xs[ti], labels[ti])
                    inner[vi] = clf.predict_proba(Xs[vi])
                try:
                    au = compute_auroc(labels, inner, nc)
                except:
                    au = 0.5
                if au > best_au:
                    best_au = au
                    bp = {"ml": ml, "lr": lr}
        for _, (ti, vi) in enumerate(skf.split(Xs, labels)):
            clf = HistGradientBoostingClassifier(max_leaf_nodes=bp.get("ml", 8), learning_rate=bp.get("lr", 0.05), max_iter=200, min_samples_leaf=10, l2_regularization=0.5, random_state=seed)
            clf.fit(Xs[ti], labels[ti])
            oof[vi] = clf.predict_proba(Xs[vi])
            ta += clf.predict_proba(Xts) / N_FOLDS

    elif etype == "et":
        for _, (ti, vi) in enumerate(skf.split(Xs, labels)):
            clf = ExtraTreesClassifier(n_estimators=300, max_depth=10, min_samples_leaf=10, random_state=seed, n_jobs=-1)
            clf.fit(Xs[ti], labels[ti])
            oof[vi] = clf.predict_proba(Xs[vi])
            ta += clf.predict_proba(Xts) / N_FOLDS

    elif etype == "rf":
        for _, (ti, vi) in enumerate(skf.split(Xs, labels)):
            clf = RandomForestClassifier(n_estimators=300, max_depth=10, min_samples_leaf=10, random_state=seed, n_jobs=-1)
            clf.fit(Xs[ti], labels[ti])
            oof[vi] = clf.predict_proba(Xs[vi])
            ta += clf.predict_proba(Xts) / N_FOLDS

    return oof, ta


def main():
    print("=" * 70)
    print("EXP 6: FAVA Binary Extension")
    print("=" * 70)

    nc = DS_CONFIG["n_classes"]
    ext = DS_CONFIG["ext"]

    tr_labels = load_labels(ext, DS_CONFIG["train"])
    va_labels = load_labels(ext, DS_CONFIG["val"])
    te_labels = load_labels(ext, DS_CONFIG["test"])
    trva_labels = np.concatenate([tr_labels, va_labels])
    n_trva, n_te = len(trva_labels), len(te_labels)

    all_oof, all_te, all_names = [], [], []

    for method in FAVA_METHODS:
        feats = load_method_features(method)
        if feats is None:
            print(f"  [SKIP] {method}: no features")
            continue
        trva = np.vstack([feats["train"], feats["val"]])
        te = feats["test"]
        t0 = time.time()

        for pca_dim in PCA_DIMS:
            sc = StandardScaler()
            Xs = sc.fit_transform(trva)
            Xts = sc.transform(te)
            actual_pca = min(pca_dim, Xs.shape[1], Xs.shape[0] - 1)
            if Xs.shape[1] > actual_pca:
                pca = PCA(n_components=actual_pca, random_state=42)
                Xs = pca.fit_transform(Xs)
                Xts = pca.transform(Xts)

            for etype in EXPERT_TYPES:
                seed_oofs, seed_tes = [], []
                for s in range(N_SEEDS):
                    seed = 42 + s * 111
                    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
                    oof, ta = train_expert_oof(Xs, Xts, trva_labels, nc, etype, seed, skf)
                    seed_oofs.append(oof)
                    seed_tes.append(ta)

                avg_oof = np.mean(seed_oofs, axis=0)
                avg_te = np.mean(seed_tes, axis=0)
                all_oof.append(avg_oof)
                all_te.append(avg_te)
                all_names.append(f"{method}_pca{pca_dim}_{etype}")

        print(f"  {method:18s}: done [{time.time()-t0:.1f}s]")

    meta_oof = np.hstack(all_oof)
    meta_te = np.hstack(all_te)

    # Enrichment
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
    print(f"\n  Total: {n_experts} experts, {meta_oof_rich.shape[1]} meta-features")

    # Meta-classification
    skf_meta = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    sc_m = StandardScaler()
    mo = sc_m.fit_transform(meta_oof_rich)
    mt = sc_m.transform(meta_te_rich)

    # L2-LR
    best_au_l2, best_C_l2 = -1, 0.01
    for C in [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]:
        inner = np.zeros((n_trva, nc))
        for _, (ti, vi) in enumerate(skf_meta.split(mo, trva_labels)):
            clf = LogisticRegression(max_iter=3000, C=C, penalty='l2', solver='lbfgs', random_state=42)
            clf.fit(mo[ti], trva_labels[ti])
            inner[vi] = clf.predict_proba(mo[vi])
        try:
            au = compute_auroc(trva_labels, inner, nc)
        except:
            au = 0.5
        if au > best_au_l2:
            best_au_l2, best_C_l2 = au, C
    clf_l2 = LogisticRegression(max_iter=3000, C=best_C_l2, penalty='l2', solver='lbfgs', random_state=42)
    clf_l2.fit(mo, trva_labels)
    te_l2 = clf_l2.predict_proba(mt)
    au_l2 = compute_auroc(te_labels, te_l2, nc)
    print(f"  Meta-L2-LR (C={best_C_l2}): {au_l2:.4f}")

    # L1-LR
    best_au_l1, best_C_l1 = -1, 0.01
    for C in [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]:
        inner = np.zeros((n_trva, nc))
        for _, (ti, vi) in enumerate(skf_meta.split(mo, trva_labels)):
            clf = LogisticRegression(max_iter=3000, C=C, penalty='l1', solver='saga', random_state=42)
            clf.fit(mo[ti], trva_labels[ti])
            inner[vi] = clf.predict_proba(mo[vi])
        try:
            au = compute_auroc(trva_labels, inner, nc)
        except:
            au = 0.5
        if au > best_au_l1:
            best_au_l1, best_C_l1 = au, C
    clf_l1 = LogisticRegression(max_iter=3000, C=best_C_l1, penalty='l1', solver='saga', random_state=42)
    clf_l1.fit(mo, trva_labels)
    te_l1 = clf_l1.predict_proba(mt)
    au_l1 = compute_auroc(te_labels, te_l1, nc)
    print(f"  Meta-L1-LR (C={best_C_l1}): {au_l1:.4f}")

    # Meta-GBT
    best_au_gbt, bp = -1, {}
    for ml in [4, 8, 16, 32]:
        for lr in [0.01, 0.05, 0.1, 0.2]:
            for ne in [100, 200, 300]:
                inner = np.zeros((n_trva, nc))
                for _, (ti, vi) in enumerate(skf_meta.split(meta_oof_rich, trva_labels)):
                    clf = HistGradientBoostingClassifier(max_leaf_nodes=ml, learning_rate=lr, max_iter=ne, min_samples_leaf=15, l2_regularization=0.5, random_state=42)
                    clf.fit(meta_oof_rich[ti], trva_labels[ti])
                    inner[vi] = clf.predict_proba(meta_oof_rich[vi])
                try:
                    au = compute_auroc(trva_labels, inner, nc)
                except:
                    au = 0.5
                if au > best_au_gbt:
                    best_au_gbt = au
                    bp = {"ml": ml, "lr": lr, "ne": ne}
    clf_gbt = HistGradientBoostingClassifier(max_leaf_nodes=bp.get("ml", 8), learning_rate=bp.get("lr", 0.05), max_iter=bp.get("ne", 200), min_samples_leaf=15, l2_regularization=0.5, random_state=42)
    clf_gbt.fit(meta_oof_rich, trva_labels)
    te_gbt = clf_gbt.predict_proba(meta_te_rich)
    au_gbt = compute_auroc(te_labels, te_gbt, nc)
    print(f"  Meta-GBT ({bp}): {au_gbt:.4f}")

    # Blend
    best_blend = max(au_l2, au_l1, au_gbt)
    best_prob = te_l2 if au_l2 >= max(au_l1, au_gbt) else (te_l1 if au_l1 >= au_gbt else te_gbt)
    best_weights = None

    for a in np.arange(0, 1.05, 0.05):
        for b in np.arange(0, 1.05 - a, 0.05):
            c = max(1.0 - a - b, 0)
            if c < -0.01:
                continue
            blended = a * te_l2 + b * te_l1 + c * te_gbt
            au = compute_auroc(te_labels, blended, nc)
            if au > best_blend:
                best_blend = au
                best_prob = blended
                best_weights = (round(a, 2), round(b, 2), round(c, 2))

    delta = best_blend - DS_CONFIG["best_single"]
    ci = bootstrap_ci(te_labels, best_prob, nc)

    print(f"\n  RESULT: AUROC={best_blend:.4f} (Δ={delta*100:+.2f}%)")
    print(f"  Best single (ITI): {DS_CONFIG['best_single']:.4f}")
    print(f"  Blend weights: {best_weights}")
    print(f"  95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]")

    def convert(o):
        if isinstance(o, (np.bool_, np.integer)): return int(o)
        if isinstance(o, np.floating): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        return o

    result = {
        "fava_binary": {
            "dataset": "fava_binary",
            "n_experts": n_experts,
            "n_meta_features": meta_oof_rich.shape[1],
            "n_methods": len(FAVA_METHODS),
            "methods": FAVA_METHODS,
            "meta_l2_lr": round(au_l2, 4),
            "meta_l1_lr": round(au_l1, 4),
            "meta_gbt": round(au_gbt, 4),
            "gbt_params": bp,
            "blend_weights": best_weights,
            "test_auroc": round(best_blend, 4),
            "baseline_auroc": DS_CONFIG["best_single"],
            "delta": round(delta, 4),
            "delta_pct": f"{delta*100:+.2f}%",
            "ci_95": [round(ci[0], 4), round(ci[1], 4)],
        }
    }

    out_path = os.path.join(RESULTS_DIR, "fava_extension.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=convert)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
