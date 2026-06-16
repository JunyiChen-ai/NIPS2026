"""Quick ablation: LR + GBT experts only (1 linear + 1 nonlinear)."""
import os, sys, json, time, warnings, argparse
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import HistGradientBoostingClassifier
from fusion.settings import get_config as _get_config

warnings.filterwarnings("ignore")

_ap = argparse.ArgumentParser(add_help=False)
_ap.add_argument("--model", default="qwen2.5-7b")
_ap.add_argument("--setting", default="old", choices=["old", "new"])
_cli, _ = _ap.parse_known_args()
MODEL = _cli.model
cfg = _get_config(_cli.setting)
PROCESSED_DIR = str(cfg.base_processed / MODEL)
EXTRACTION_DIR = str(cfg.base_extraction / MODEL)

PCA_DIM = 128
EXPERT_TYPES = ["lr", "gbt"]  # only 2
N_SEEDS = 1
N_FOLDS = 5
C_GRID = [1e-2, 1e-1, 1.0]

MC_METHODS = ["lr_probe", "pca_lr", "iti", "kb_mlp", "attn_satisfies", "sep", "step"]

ALL_DATASETS = {}
for _ds, _info in cfg.datasets.items():
    ALL_DATASETS[_ds] = {
        "n_classes": _info["n_classes"], "ext": _info["ext"],
        "train": _info["splits"]["train"], "val": _info["splits"]["val"], "test": _info["splits"]["test"],
    }

# Load best_single from oracle (per-model)
oracle_path = str(cfg.model_results_dir(MODEL) / "oracle_complete.json")
if os.path.exists(oracle_path):
    with open(oracle_path) as f:
        oracle = json.load(f)
    BEST_SINGLE = {ds: oracle[ds]["best_single_auroc"]
                   for ds in ALL_DATASETS if ds in oracle and "best_single_auroc" in oracle[ds]}
else:
    BEST_SINGLE = {}
    print(f"[WARN] oracle_complete.json not found at {oracle_path}; BEST_SINGLE empty")


_USE_GPU_LR = cfg.name == "new"
if _USE_GPU_LR:
    # Rebind HGB to xgboost-gpu shim so all HistGradientBoostingClassifier(...) calls go to GPU.
    from fusion._gpu_clf import get_hgb_class as _get_hgb_class
    HistGradientBoostingClassifier = _get_hgb_class(True)
if _USE_GPU_LR:
    from fusion._gpu_clf import gpu_lr_fit_predict as _gpu_lr_fit_predict


def load_labels(ds_name, split):
    return cfg.load_labels(MODEL, ds_name, split)

def compute_auroc(y, p, nc):
    if nc == 2: return roc_auc_score(y, p[:, 1])
    yb = label_binarize(y, classes=list(range(nc)))
    return roc_auc_score(yb, p, average="macro", multi_class="ovr")

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
    n_trva, n_te = len(labels), Xts.shape[0]
    oof = np.zeros((n_trva, nc))
    ta = np.zeros((n_te, nc))
    if etype == "lr":
        best_au, best_C = -1, 1.0
        for C in C_GRID:
            inner = np.zeros((n_trva, nc))
            for _, (ti, vi) in enumerate(skf.split(Xs, labels)):
                if _USE_GPU_LR and nc == 2:
                    p_vi = _gpu_lr_fit_predict(Xs[ti], labels[ti], Xs[vi], C=C)[0]
                    inner[vi] = np.stack([1 - p_vi, p_vi], axis=1)
                else:
                    clf = LogisticRegression(max_iter=2000, C=C, random_state=seed)
                    clf.fit(Xs[ti], labels[ti]); inner[vi] = clf.predict_proba(Xs[vi])
            try: au = compute_auroc(labels, inner, nc)
            except: au = 0.5
            if au > best_au: best_au, best_C = au, C
        for _, (ti, vi) in enumerate(skf.split(Xs, labels)):
            if _USE_GPU_LR and nc == 2:
                p_vi, p_te = _gpu_lr_fit_predict(Xs[ti], labels[ti], Xs[vi], Xts, C=best_C)
                oof[vi] = np.stack([1 - p_vi, p_vi], axis=1)
                ta += np.stack([1 - p_te, p_te], axis=1) / N_FOLDS
            else:
                clf = LogisticRegression(max_iter=2000, C=best_C, random_state=seed)
                clf.fit(Xs[ti], labels[ti]); oof[vi] = clf.predict_proba(Xs[vi]); ta += clf.predict_proba(Xts)/N_FOLDS
    elif etype == "gbt":
        for _, (ti, vi) in enumerate(skf.split(Xs, labels)):
            clf = HistGradientBoostingClassifier(max_leaf_nodes=16, learning_rate=0.1, max_iter=200, min_samples_leaf=10, l2_regularization=0.5, random_state=seed)
            clf.fit(Xs[ti], labels[ti]); oof[vi] = clf.predict_proba(Xs[vi]); ta += clf.predict_proba(Xts)/N_FOLDS
    return oof, ta

def run(ds_name, info):
    nc = info["n_classes"]
    tr_labels = load_labels(ds_name, "train")
    va_labels = load_labels(ds_name, "val")
    te_labels = load_labels(ds_name, "test")
    trva_labels = np.concatenate([tr_labels, va_labels])

    all_oof, all_te = [], []
    for method in MC_METHODS:
        feats = load_method_features(ds_name, method)
        if feats is None: continue
        trva = np.vstack([feats["train"], feats["val"]])
        te = feats["test"]
        sc = StandardScaler()
        Xs = sc.fit_transform(trva); Xts = sc.transform(te)
        actual_pca = min(PCA_DIM, Xs.shape[1], Xs.shape[0] - 1)
        if Xs.shape[1] > actual_pca:
            pca = PCA(n_components=actual_pca, random_state=42)
            Xs = pca.fit_transform(Xs); Xts = pca.transform(Xts)
        for etype in EXPERT_TYPES:
            seed_oofs, seed_tes = [], []
            for s in range(N_SEEDS):
                seed = 42 + s * 111
                skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
                oof, ta = train_expert_oof(Xs, Xts, trva_labels, nc, etype, seed, skf)
                seed_oofs.append(oof); seed_tes.append(ta)
            all_oof.append(np.mean(seed_oofs, axis=0))
            all_te.append(np.mean(seed_tes, axis=0))

    meta_oof = np.hstack(all_oof); meta_te = np.hstack(all_te)
    n_trva = len(trva_labels)

    # Meta: single L2-LR
    sc_m = StandardScaler()
    mo = sc_m.fit_transform(meta_oof); mt = sc_m.transform(meta_te)
    skf_meta = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    best_au_l2, best_C_l2 = -1, 0.01
    for C in [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]:
        inner = np.zeros((n_trva, nc))
        for _, (ti, vi) in enumerate(skf_meta.split(mo, trva_labels)):
            clf = LogisticRegression(max_iter=3000, C=C, penalty='l2', solver='lbfgs', random_state=42)
            clf.fit(mo[ti], trva_labels[ti]); inner[vi] = clf.predict_proba(mo[vi])
        try: au = compute_auroc(trva_labels, inner, nc)
        except: au = 0.5
        if au > best_au_l2: best_au_l2, best_C_l2 = au, C
    clf_l2 = LogisticRegression(max_iter=3000, C=best_C_l2, penalty='l2', solver='lbfgs', random_state=42)
    clf_l2.fit(mo, trva_labels)
    best_blend = compute_auroc(te_labels, clf_l2.predict_proba(mt), nc)

    bs = BEST_SINGLE[ds_name]
    delta = best_blend - bs
    return {"auroc": round(best_blend, 4), "best_single": round(bs, 4), "delta": round(delta, 4), "delta_pct": f"{delta*100:+.2f}%"}

print("LR + GBT only (1 linear + 1 nonlinear)")
print("=" * 60)
results = {}
for ds_name, info in ALL_DATASETS.items():
    t0 = time.time()
    r = run(ds_name, info)
    results[ds_name] = r
    print(f"{ds_name:25s} best={r['best_single']:.4f} fusion={r['auroc']:.4f} delta={r['delta_pct']} [{time.time()-t0:.0f}s]")

deltas = [r["delta"]*100 for r in results.values()]
print(f"\nAvg delta: {np.mean(deltas):+.2f}%")
