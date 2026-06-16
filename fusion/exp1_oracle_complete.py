"""
Exp 1: Per-Example Oracle — Complete Coverage (all 5 target datasets + fava).
Fills the gap: existing oracle only covers 4 datasets, missing e2h_3c & e2h_5c.
"""

import os, json, warnings
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

import sys as _sys
_sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1]))
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

MC_METHODS = ["lr_probe", "pca_lr", "iti", "kb_mlp", "attn_satisfies", "sep", "step"]
BIN_EXTRA = ["mm_probe", "lid", "llm_check", "seakr"]

ALL_DATASETS = cfg.datasets


def load_labels(ds_name, split):
    """Setting-aware label loader. Caller passes dataset name (key in cfg.datasets)
    and a split alias ('train'/'val'/'test'). The cfg figures out the per-setting
    physical layout (old: meta.json from ext/split_dir; new: correctness_labels)."""
    return cfg.load_labels(_MODEL, ds_name, split)


# In new setting (all binary), use GPU LBFGS for ~100x speedup over sklearn LR.
# In old setting (3/5-class), keep sklearn for multinomial support.
_USE_GPU_LR = cfg.name == "new"
if _USE_GPU_LR:
    from fusion._gpu_clf import gpu_lr_fit_predict as _gpu_lr_fit_predict


def _fit_predict_proba(X_tr, y_tr, X_te, n_classes, C=0.1, max_iter=2000):
    """Returns (n_te, n_classes) probabilities. Uses GPU LBFGS for binary in new
    setting; otherwise sklearn LR."""
    if _USE_GPU_LR and n_classes == 2:
        # GPU path returns positive-class probabilities; assemble (n,2) matrix.
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


def main():
    print("=" * 70)
    print("EXP 1: Per-Example Oracle — Complete Coverage")
    print("=" * 70)

    results = {}

    for ds_name, ds_cfg in ALL_DATASETS.items():
        nc = ds_cfg["n_classes"]
        probe_dir = os.path.join(PROCESSED_DIR, ds_name)
        if not os.path.exists(probe_dir):
            print(f"\n[SKIP] {ds_name}: no processed features")
            continue

        tr_labels = load_labels(ds_name, "train")
        va_labels = load_labels(ds_name, "val")
        te_labels = load_labels(ds_name, "test")
        trva_labels = np.concatenate([tr_labels, va_labels])

        methods = MC_METHODS if nc > 2 else MC_METHODS + BIN_EXTRA

        all_probe_preds = {}
        for method in methods:
            tr_path = os.path.join(probe_dir, method, "train.pt")
            if not os.path.exists(tr_path):
                continue
            tr = torch.load(tr_path, map_location="cpu").float().numpy()
            va = torch.load(os.path.join(probe_dir, method, "val.pt"), map_location="cpu").float().numpy()
            te = torch.load(os.path.join(probe_dir, method, "test.pt"), map_location="cpu").float().numpy()
            if tr.ndim == 1:
                tr, va, te = tr.reshape(-1, 1), va.reshape(-1, 1), te.reshape(-1, 1)

            trva = np.vstack([tr, va])
            sc = StandardScaler()
            trva_s = sc.fit_transform(trva)
            te_s = sc.transform(te)
            if trva_s.shape[1] > 512:
                pca = PCA(n_components=256, random_state=42)
                trva_s = pca.fit_transform(trva_s)
                te_s = pca.transform(te_s)

            try:
                all_probe_preds[method] = _fit_predict_proba(
                    trva_s, trva_labels, te_s, nc, C=0.1, max_iter=2000)
            except Exception as e:
                print(f"  [WARN] {method} failed: {e}")

        if len(all_probe_preds) < 2:
            print(f"\n[SKIP] {ds_name}: only {len(all_probe_preds)} probes")
            continue

        # Per-probe AUROC
        probe_aurocs = {}
        for m, preds in all_probe_preds.items():
            try:
                probe_aurocs[m] = compute_auroc(te_labels, preds, nc)
            except:
                pass

        # Oracle: for each sample, pick probe giving highest prob to correct class
        n_te = len(te_labels)
        oracle_probs = np.zeros((n_te, nc))
        oracle_correct = 0
        for i in range(n_te):
            true_class = int(te_labels[i])
            best_prob = -1
            best_preds = None
            for m in all_probe_preds:
                p = all_probe_preds[m][i, true_class]
                if p > best_prob:
                    best_prob = p
                    best_preds = all_probe_preds[m][i]
            oracle_probs[i] = best_preds
            if np.argmax(best_preds) == true_class:
                oracle_correct += 1

        oracle_auroc = compute_auroc(te_labels, oracle_probs, nc)
        oracle_acc = oracle_correct / n_te
        best_single = max(probe_aurocs.values())
        best_method = max(probe_aurocs, key=probe_aurocs.get)
        headroom = oracle_auroc - best_single

        results[ds_name] = {
            "oracle_auroc": float(oracle_auroc),
            "oracle_accuracy": float(oracle_acc),
            "best_single_auroc": float(best_single),
            "best_single_method": best_method,
            "headroom": float(headroom),
            "n_probes": len(all_probe_preds),
            "per_probe_auroc": {k: round(float(v), 4) for k, v in sorted(probe_aurocs.items(), key=lambda x: -x[1])},
        }

        print(f"\n{ds_name} ({len(all_probe_preds)} probes):")
        print(f"  Best single: {best_single:.4f} ({best_method})")
        print(f"  Oracle AUROC: {oracle_auroc:.4f}  Acc: {oracle_acc:.3f}")
        print(f"  Headroom: {headroom:+.4f} ({headroom*100:+.1f}%)")

    # Summary table
    print(f"\n{'='*70}")
    print(f"{'Dataset':25s} {'Best':>7s} {'Oracle':>7s} {'Headroom':>9s} {'Probes':>6s}")
    print("-" * 60)
    for ds, r in results.items():
        print(f"{ds:25s} {r['best_single_auroc']:.4f}  {r['oracle_auroc']:.4f}  {r['headroom']*100:+6.1f}%   {r['n_probes']}")

    out_path = os.path.join(RESULTS_DIR, "oracle_complete.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
