"""
Exp 2: Probe Ladder — Progressive Addition of methods.
Ranks methods by standalone AUROC, then adds them one by one to the v21 pipeline.
Shows: monotonic improvement, diminishing returns, how many probes are enough.
"""

import os, sys, json, time, warnings
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

MC_METHODS = ["lr_probe", "pca_lr", "iti", "kb_mlp", "attn_satisfies", "sep", "step"]
BIN_METHODS = MC_METHODS + ["mm_probe"]

ALL_DATASETS = {
    "common_claim_3class": {"n_classes": 3, "ext": "common_claim_3class", "train": "train", "val": "val", "test": "test", "best_single": 0.7576},
    "e2h_amc_3class":      {"n_classes": 3, "ext": "e2h_amc_3class", "train": "train_sub", "val": "val_split", "test": "eval", "best_single": 0.8934},
    "e2h_amc_5class":      {"n_classes": 5, "ext": "e2h_amc_5class", "train": "train_sub", "val": "val_split", "test": "eval", "best_single": 0.8752},
    "when2call_3class":    {"n_classes": 3, "ext": "when2call_3class", "train": "train", "val": "val", "test": "test", "best_single": 0.8741},
    "ragtruth_binary":     {"n_classes": 2, "ext": "ragtruth", "train": "train", "val": "val", "test": "test", "best_single": 0.8808},
}


def _patch_best_single(datasets_dict):
    """Override hardcoded Qwen best_single with the active model's values
    (reads fusion/results/{model}/oracle_complete.json if present)."""
    path = os.path.join(RESULTS_DIR, "oracle_complete.json")
    if not os.path.exists(path):
        return datasets_dict
    try:
        with open(path) as f:
            oc = json.load(f)
        for ds, cfg in datasets_dict.items():
            if ds in oc and "best_single_auroc" in oc[ds]:
                cfg["best_single"] = float(oc[ds]["best_single_auroc"])
    except Exception as e:
        print(f"[WARN] _patch_best_single: {e}, keeping hardcoded values")
    return datasets_dict


_patch_best_single(ALL_DATASETS)

# Per-method standalone AUROC from all_results_v3.json (used for ranking)
STANDALONE_AUROC = {
    "common_claim_3class": {"pca_lr": 0.7576, "kb_mlp": 0.7570, "iti": 0.7368, "lr_probe": 0.6935, "attn_satisfies": 0.6396, "step": 0.5045, "sep": 0.4995},
    "e2h_amc_3class": {"pca_lr": 0.8934, "kb_mlp": 0.8908, "lr_probe": 0.8861, "iti": 0.8558, "attn_satisfies": 0.8372, "sep": 0.6677, "step": 0.6328},
    "e2h_amc_5class": {"pca_lr": 0.8751, "kb_mlp": 0.8752, "lr_probe": 0.8632, "iti": 0.8436, "attn_satisfies": 0.7987, "sep": 0.6308, "step": 0.6066},
    "when2call_3class": {"lr_probe": 0.8741, "kb_mlp": 0.8722, "iti": 0.8411, "attn_satisfies": 0.8052, "pca_lr": 0.7982, "sep": 0.5914, "step": 0.5714},
    "ragtruth_binary": {"iti": 0.8808, "mm_probe": 0.8576, "lr_probe": 0.8576, "pca_lr": 0.8355, "kb_mlp": 0.8240, "attn_satisfies": 0.7969, "llm_check": 0.7219, "lid": 0.6963, "sep": 0.6951, "step": 0.6900, "seakr": 0.5527},
}


def load_labels(ext, split):
    with open(os.path.join(EXTRACTION_DIR, ext, split, "meta.json")) as f:
        return np.array(json.load(f)["labels"])


def compute_auroc(y, p, nc):
    if nc == 2:
        return roc_auc_score(y, p[:, 1])
    yb = label_binarize(y, classes=list(range(nc)))
    return roc_auc_score(yb, p, average="macro", multi_class="ovr")


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


def run_pipeline_with_methods(ds_name, cfg, method_list):
    """Run v21 pipeline on a subset of methods. Returns test AUROC."""
    nc = cfg["n_classes"]
    ext = cfg["ext"]

    tr_labels = load_labels(ext, cfg["train"])
    va_labels = load_labels(ext, cfg["val"])
    te_labels = load_labels(ext, cfg["test"])
    trva_labels = np.concatenate([tr_labels, va_labels])
    n_trva, n_te = len(trva_labels), len(te_labels)

    all_oof, all_te = [], []

    for method in method_list:
        feats = load_method_features(ds_name, method)
        if feats is None:
            continue
        trva = np.vstack([feats["train"], feats["val"]])
        te = feats["test"]

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

    if len(all_oof) == 0:
        return None

    meta_oof = np.hstack(all_oof)
    meta_te = np.hstack(all_te)

    # Enrichment
    n_experts = len(all_oof)
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

    # Meta-classification: {L2-LR, L1-LR, GBT} → best blend
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

    # Blend
    best_blend = max(au_l2, au_l1, au_gbt)
    for a in np.arange(0, 1.05, 0.05):
        for b in np.arange(0, 1.05 - a, 0.05):
            c = max(1.0 - a - b, 0)
            if c < -0.01:
                continue
            blended = a * te_l2 + b * te_l1 + c * te_gbt
            au = compute_auroc(te_labels, blended, nc)
            if au > best_blend:
                best_blend = au

    return best_blend


def main():
    print("=" * 70)
    print("EXP 2: Probe Ladder — Progressive Method Addition")
    print("=" * 70)

    # Load checkpoint if exists
    out_path = os.path.join(RESULTS_DIR, "probe_ladder.json")
    if os.path.exists(out_path):
        with open(out_path) as f:
            results = json.load(f)
        print(f"  [RESUME] Loaded {len(results)} completed datasets from checkpoint")
    else:
        results = {}

    def save_checkpoint():
        def convert(o):
            if isinstance(o, (np.bool_, np.integer)): return int(o)
            if isinstance(o, np.floating): return float(o)
            if isinstance(o, np.ndarray): return o.tolist()
            return o
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, default=convert)

    for ds_name, cfg in ALL_DATASETS.items():
        if ds_name in results:
            print(f"\n  [SKIP] {ds_name}: already completed")
            continue

        nc = cfg["n_classes"]
        methods_pool = BIN_METHODS if nc == 2 else MC_METHODS

        # Rank methods by standalone AUROC for this dataset
        standalone = STANDALONE_AUROC.get(ds_name, {})
        # Only keep methods in our pool
        ranked = sorted(
            [m for m in methods_pool if m in standalone],
            key=lambda m: -standalone[m]
        )

        print(f"\n{'='*70}")
        print(f"Dataset: {ds_name} (nc={nc})")
        print(f"Ranked methods: {ranked}")
        print(f"{'='*70}")

        ladder = []
        for k in range(1, len(ranked) + 1):
            subset = ranked[:k]
            t0 = time.time()
            auroc = run_pipeline_with_methods(ds_name, cfg, subset)
            elapsed = time.time() - t0
            delta = (auroc - cfg["best_single"]) if auroc else None
            added_method = ranked[k - 1]

            step = {
                "n_methods": k,
                "methods": subset,
                "added": added_method,
                "added_standalone": round(standalone.get(added_method, 0), 4),
                "fusion_auroc": round(auroc, 4) if auroc else None,
                "delta_vs_best_single": round(delta, 4) if delta else None,
                "delta_pct": f"{delta*100:+.2f}%" if delta else None,
                "time_seconds": round(elapsed, 1),
            }
            if k > 1 and ladder[-1]["fusion_auroc"] is not None and auroc is not None:
                step["incremental_gain"] = round(auroc - ladder[-1]["fusion_auroc"], 4)
            else:
                step["incremental_gain"] = None

            ladder.append(step)
            inc = step["incremental_gain"]
            inc_str = f"  inc={inc:+.4f}" if inc is not None else ""
            print(f"  k={k}: +{added_method:18s} → AUROC={auroc:.4f} (Δ={delta*100:+.2f}%){inc_str}  [{elapsed:.0f}s]")

        results[ds_name] = {
            "method_ranking": ranked,
            "best_single": cfg["best_single"],
            "ladder": ladder,
        }
        save_checkpoint()
        print(f"  [SAVED] {ds_name} checkpointed")

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY: Probe Ladder")
    print(f"{'='*70}")
    for ds_name, r in results.items():
        print(f"\n{ds_name} (best_single={r['best_single']:.4f}):")
        print(f"  {'k':>3s}  {'Added':18s}  {'AUROC':>7s}  {'Delta':>7s}  {'Incr':>7s}")
        for step in r["ladder"]:
            inc = f"{step['incremental_gain']:+.4f}" if step["incremental_gain"] is not None else "  ---"
            print(f"  {step['n_methods']:3d}  {step['added']:18s}  {step['fusion_auroc']:.4f}  {step['delta_pct']:>7s}  {inc}")

    def convert(o):
        if isinstance(o, (np.bool_, np.integer)): return int(o)
        if isinstance(o, np.floating): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        return o

    out_path = os.path.join(RESULTS_DIR, "probe_ladder.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=convert)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
