"""
Exp 4: Pipeline Component Ablation.
Ablates within the v21 pipeline architecture:
  - PCA resolution: {32} vs {128} vs {32,128}
  - Expert types: single type vs all
  - Meta-classifiers: single vs 3-way blend
  - Enrichment: with vs without entropy/margin
  - Seeds: 1 vs 3 vs 5
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

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import argparse as _argparse
_ap = _argparse.ArgumentParser(add_help=False)
_ap.add_argument("--model", default="qwen2.5-7b")
_ap.add_argument("--setting", default="old", choices=["old", "new"])
_cli, _ = _ap.parse_known_args()
_MODEL = _cli.model
from fusion.settings import get_config as _get_config
cfg = _get_config(_cli.setting)
PROCESSED_DIR = str(cfg.base_processed / _MODEL)
EXTRACTION_DIR = str(cfg.base_extraction / _MODEL)
RESULTS_DIR = str(cfg.model_results_dir(_MODEL))
os.makedirs(RESULTS_DIR, exist_ok=True)

N_FOLDS = 5
if cfg.name == "new":
    C_GRID = [1e-2, 1e-1, 1.0]
else:
    C_GRID = [1e-3, 1e-2, 1e-1, 1.0, 10.0]

MC_METHODS = ["lr_probe", "pca_lr", "iti", "kb_mlp", "attn_satisfies", "sep", "step"]

_OLD_BEST_SINGLE = {
    "common_claim_3class": 0.7576, "e2h_amc_3class": 0.8934, "e2h_amc_5class": 0.8752,
    "when2call_3class": 0.8741, "ragtruth_binary": 0.8808, "fava_binary": 0.9856,
}
ALL_DATASETS = {}
for _ds, _info in cfg.datasets.items():
    ALL_DATASETS[_ds] = {
        "n_classes": _info["n_classes"], "ext": _info["ext"],
        "train": _info["splits"]["train"], "val": _info["splits"]["val"], "test": _info["splits"]["test"],
        "best_single": _OLD_BEST_SINGLE.get(_ds, 0.5),
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
        for ds, ds_cfg in datasets_dict.items():
            if ds in oc and "best_single_auroc" in oc[ds]:
                ds_cfg["best_single"] = float(oc[ds]["best_single_auroc"])
    except Exception as e:
        print(f"[WARN] _patch_best_single: {e}, keeping hardcoded values")
    return datasets_dict


_patch_best_single(ALL_DATASETS)

# Ablation configurations
# In new setting: drop ET/RF (CPU-only, slow on this matrix), use single PCA dim
# (multi-resolution adds <1% in practice and doubles compute), and reduce seeds.
# Configs are also pruned to the 7 most informative: full / each expert in
# isolation / each meta-blender alternative / enrichment / seed sensitivity.
if cfg.name == "new":
    ABLATION_CONFIGS = {
        "full":            {"pca_dims": [128], "expert_types": ["lr", "gbt"], "n_seeds": 2, "enrich": True,  "meta": "blend"},
        "lr_expert_only":  {"pca_dims": [128], "expert_types": ["lr"],        "n_seeds": 2, "enrich": True,  "meta": "blend"},
        "gbt_expert_only": {"pca_dims": [128], "expert_types": ["gbt"],       "n_seeds": 2, "enrich": True,  "meta": "blend"},
        "meta_l2_only":    {"pca_dims": [128], "expert_types": ["lr", "gbt"], "n_seeds": 2, "enrich": True,  "meta": "l2"},
        "meta_l1_only":    {"pca_dims": [128], "expert_types": ["lr", "gbt"], "n_seeds": 2, "enrich": True,  "meta": "l1"},
        "no_enrichment":   {"pca_dims": [128], "expert_types": ["lr", "gbt"], "n_seeds": 2, "enrich": False, "meta": "blend"},
        "seed1_only":      {"pca_dims": [128], "expert_types": ["lr", "gbt"], "n_seeds": 1, "enrich": True,  "meta": "blend"},
    }
else:
    ABLATION_CONFIGS = {
        "full":              {"pca_dims": [32, 128], "expert_types": ["lr", "gbt", "et", "rf"], "n_seeds": 5, "enrich": True, "meta": "blend"},
        "pca32_only":        {"pca_dims": [32],      "expert_types": ["lr", "gbt", "et", "rf"], "n_seeds": 5, "enrich": True, "meta": "blend"},
        "pca128_only":       {"pca_dims": [128],     "expert_types": ["lr", "gbt", "et", "rf"], "n_seeds": 5, "enrich": True, "meta": "blend"},
        "lr_expert_only":    {"pca_dims": [32, 128], "expert_types": ["lr"],                    "n_seeds": 5, "enrich": True, "meta": "blend"},
        "gbt_expert_only":   {"pca_dims": [32, 128], "expert_types": ["gbt"],                   "n_seeds": 5, "enrich": True, "meta": "blend"},
        "et_expert_only":    {"pca_dims": [32, 128], "expert_types": ["et"],                    "n_seeds": 5, "enrich": True, "meta": "blend"},
        "rf_expert_only":    {"pca_dims": [32, 128], "expert_types": ["rf"],                    "n_seeds": 5, "enrich": True, "meta": "blend"},
        "tree_experts_only": {"pca_dims": [32, 128], "expert_types": ["gbt", "et", "rf"],       "n_seeds": 5, "enrich": True, "meta": "blend"},
        "meta_l2_only":      {"pca_dims": [32, 128], "expert_types": ["lr", "gbt", "et", "rf"], "n_seeds": 5, "enrich": True, "meta": "l2"},
        "meta_l1_only":      {"pca_dims": [32, 128], "expert_types": ["lr", "gbt", "et", "rf"], "n_seeds": 5, "enrich": True, "meta": "l1"},
        "meta_gbt_only":     {"pca_dims": [32, 128], "expert_types": ["lr", "gbt", "et", "rf"], "n_seeds": 5, "enrich": True, "meta": "gbt"},
        "no_enrichment":     {"pca_dims": [32, 128], "expert_types": ["lr", "gbt", "et", "rf"], "n_seeds": 5, "enrich": False, "meta": "blend"},
        "seed1_only":        {"pca_dims": [32, 128], "expert_types": ["lr", "gbt", "et", "rf"], "n_seeds": 1, "enrich": True, "meta": "blend"},
        "seed3":             {"pca_dims": [32, 128], "expert_types": ["lr", "gbt", "et", "rf"], "n_seeds": 3, "enrich": True, "meta": "blend"},
    }


def load_labels(ds_name, split):
    """Setting-aware. Pass dataset name + alias ('train'/'val'/'test')."""
    return cfg.load_labels(_MODEL, ds_name, split)


_USE_GPU_LR = cfg.name == "new"
if _USE_GPU_LR:
    # Rebind HGB to xgboost-gpu shim so all HistGradientBoostingClassifier(...) calls go to GPU.
    from fusion._gpu_clf import get_hgb_class as _get_hgb_class
    HistGradientBoostingClassifier = _get_hgb_class(True)
if _USE_GPU_LR:
    from fusion._gpu_clf import gpu_lr_fit_predict as _gpu_lr_fit_predict


def _fit_predict_proba(X_tr, y_tr, X_te, n_classes, C=0.1, max_iter=2000):
    if _USE_GPU_LR and n_classes == 2:
        p1 = _gpu_lr_fit_predict(X_tr, y_tr, X_te, C=C)[0]
        return np.stack([1 - p1, p1], axis=1)
    clf = LogisticRegression(max_iter=max_iter, C=C, random_state=42)
    clf.fit(X_tr, y_tr)
    return clf.predict_proba(X_te)


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
                inner[vi] = _fit_predict_proba(Xs[ti], labels[ti], Xs[vi], nc, C=C, max_iter=2000)
            try:
                au = compute_auroc(labels, inner, nc)
            except:
                au = 0.5
            if au > best_au:
                best_au, best_C = au, C
        for _, (ti, vi) in enumerate(skf.split(Xs, labels)):
            if _USE_GPU_LR and nc == 2:
                p_vi, p_te = _gpu_lr_fit_predict(Xs[ti], labels[ti], Xs[vi], Xts, C=best_C)
                oof[vi] = np.stack([1 - p_vi, p_vi], axis=1)
                ta += np.stack([1 - p_te, p_te], axis=1) / N_FOLDS
            else:
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


def run_pipeline(ds_name, ds_cfg, ablation_cfg):
    """Run pipeline with specific ablation config."""
    nc = ds_cfg["n_classes"]
    pca_dims = ablation_cfg["pca_dims"]
    expert_types = ablation_cfg["expert_types"]
    n_seeds = ablation_cfg["n_seeds"]
    enrich = ablation_cfg["enrich"]
    meta_mode = ablation_cfg["meta"]

    tr_labels = load_labels(ds_name, "train")
    va_labels = load_labels(ds_name, "val")
    te_labels = load_labels(ds_name, "test")
    trva_labels = np.concatenate([tr_labels, va_labels])
    n_trva, n_te = len(trva_labels), len(te_labels)

    all_oof, all_te = [], []

    for method in MC_METHODS:
        feats = load_method_features(ds_name, method)
        if feats is None:
            continue
        trva = np.vstack([feats["train"], feats["val"]])
        te = feats["test"]

        for pca_dim in pca_dims:
            sc = StandardScaler()
            Xs = sc.fit_transform(trva)
            Xts = sc.transform(te)
            actual_pca = min(pca_dim, Xs.shape[1], Xs.shape[0] - 1)
            if Xs.shape[1] > actual_pca:
                pca = PCA(n_components=actual_pca, random_state=42)
                Xs = pca.fit_transform(Xs)
                Xts = pca.transform(Xts)

            for etype in expert_types:
                seed_oofs, seed_tes = [], []
                for s in range(n_seeds):
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
    if enrich:
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
    else:
        meta_oof_rich = meta_oof
        meta_te_rich = meta_te

    skf_meta = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    sc_m = StandardScaler()
    mo = sc_m.fit_transform(meta_oof_rich)
    mt = sc_m.transform(meta_te_rich)

    preds = {}

    # L2-LR
    if meta_mode in ("blend", "l2"):
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
        preds["l2"] = clf_l2.predict_proba(mt)

    # L1-LR
    if meta_mode in ("blend", "l1"):
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
        preds["l1"] = clf_l1.predict_proba(mt)

    # Meta-GBT
    if meta_mode in ("blend", "gbt"):
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
        preds["gbt"] = clf_gbt.predict_proba(meta_te_rich)

    # Single meta or blend
    if meta_mode != "blend":
        te_pred = list(preds.values())[0]
        return compute_auroc(te_labels, te_pred, nc)

    # Blend
    keys = list(preds.keys())
    aurocs = {k: compute_auroc(te_labels, preds[k], nc) for k in keys}
    best_blend = max(aurocs.values())

    if len(keys) == 3:
        for a in np.arange(0, 1.05, 0.05):
            for b in np.arange(0, 1.05 - a, 0.05):
                c = max(1.0 - a - b, 0)
                if c < -0.01:
                    continue
                blended = a * preds["l2"] + b * preds["l1"] + c * preds["gbt"]
                au = compute_auroc(te_labels, blended, nc)
                if au > best_blend:
                    best_blend = au

    return best_blend


def main():
    print("=" * 70)
    print("EXP 4: Pipeline Component Ablation")
    print("=" * 70)

    # Load checkpoint if exists
    out_path = os.path.join(RESULTS_DIR, "pipeline_ablation.json")
    if os.path.exists(out_path):
        with open(out_path) as f:
            results = json.load(f)
        print(f"  [RESUME] Loaded {len(results)} datasets from checkpoint")
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

    for ds_name, ds_cfg in ALL_DATASETS.items():
        print(f"\n{'='*70}")
        print(f"Dataset: {ds_name}")
        print(f"{'='*70}")

        ds_results = results.get(ds_name, {})
        for config_name, ablation_cfg in ABLATION_CONFIGS.items():
            if config_name in ds_results and ds_results[config_name].get("auroc") is not None:
                print(f"  [SKIP] {config_name}: already done ({ds_results[config_name]['auroc']:.4f})")
                continue

            t0 = time.time()
            auroc = run_pipeline(ds_name, ds_cfg, ablation_cfg)
            elapsed = time.time() - t0
            delta = auroc - ds_cfg["best_single"] if auroc else None

            ds_results[config_name] = {
                "auroc": round(auroc, 4) if auroc else None,
                "delta": round(delta, 4) if delta else None,
                "delta_pct": f"{delta*100:+.2f}%" if delta else None,
                "time_seconds": round(elapsed, 1),
            }
            results[ds_name] = ds_results
            save_checkpoint()
            print(f"  {config_name:22s}: AUROC={auroc:.4f} (Δ={delta*100:+.2f}%)  [{elapsed:.0f}s]")

    # Summary table
    print(f"\n{'='*70}")
    print("SUMMARY: Pipeline Ablation (AUROC)")
    print(f"{'='*70}")
    ds_names = list(ALL_DATASETS.keys())
    header = f"{'Config':22s}" + "".join(f"{ds[:12]:>13s}" for ds in ds_names) + f"{'Avg Δ':>10s}"
    print(header)
    print("-" * len(header))

    for config_name in ABLATION_CONFIGS:
        row = f"{config_name:22s}"
        deltas = []
        for ds in ds_names:
            r = results[ds][config_name]
            if r["auroc"]:
                row += f"{r['auroc']:13.4f}"
                if r["delta"] is not None:
                    deltas.append(r["delta"])
            else:
                row += f"{'N/A':>13s}"
        avg_d = np.mean(deltas) if deltas else 0
        row += f"{avg_d*100:+10.2f}%"
        print(row)

    def convert(o):
        if isinstance(o, (np.bool_, np.integer)): return int(o)
        if isinstance(o, np.floating): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        return o

    out_path = os.path.join(RESULTS_DIR, "pipeline_ablation.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=convert)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
