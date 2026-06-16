"""Simple fusion baselines: score averaging + LR on concatenated probe predictions.

Runs across all 3 models (Qwen, Llama, Mistral) and saves per-model JSON to
`fusion/results/{model}/simple_fusion_baselines.json`.

Schema per dataset:
    { "common_claim_3class": {"best_single": 0.7576, "score_avg": 0.7651,
                              "lr_on_preds": 0.7735}, ... }
"""
import os, json, time, warnings, argparse
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
from fusion.settings import get_config as _get_config

warnings.filterwarnings("ignore")

# Setting parsed at the bottom; default to old.
_ap = argparse.ArgumentParser(add_help=False)
_ap.add_argument("--setting", default="old", choices=["old", "new"])
_ap.add_argument("--models", nargs="+", default=["qwen2.5-7b", "llama3.1-8b", "mistral-7b-v0.3"])
_cli, _ = _ap.parse_known_args()
cfg = _get_config(_cli.setting)
MODELS = _cli.models

MC_METHODS = ["lr_probe", "pca_lr", "iti", "kb_mlp", "attn_satisfies", "sep", "step"]

ALL_DATASETS = {}
for _ds, _info in cfg.datasets.items():
    ALL_DATASETS[_ds] = {
        "n_classes": _info["n_classes"], "ext": _info["ext"],
        "train": _info["splits"]["train"], "val": _info["splits"]["val"], "test": _info["splits"]["test"],
    }


_USE_GPU_LR = cfg.name == "new"
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


def load_labels(model, ds_name, split_alias):
    return cfg.load_labels(model, ds_name, split_alias)


def load_method_features(processed_dir, ds_name, method):
    base = os.path.join(processed_dir, ds_name, method)
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


def run(ds_name, info, processed_dir, extraction_dir, best_single, model):
    nc = info["n_classes"]
    tr_labels = load_labels(model, ds_name, "train")
    va_labels = load_labels(model, ds_name, "val")
    te_labels = load_labels(model, ds_name, "test")
    trva_labels = np.concatenate([tr_labels, va_labels])
    n_trva = len(trva_labels)

    # Get per-method OOF predictions using simple LR (no expert library)
    all_oof, all_te = [], []
    for method in MC_METHODS:
        feats = load_method_features(processed_dir, ds_name, method)
        if feats is None:
            continue
        trva = np.vstack([feats["train"], feats["val"]])
        te = feats["test"]
        sc = StandardScaler(); Xs = sc.fit_transform(trva); Xts = sc.transform(te)
        actual_pca = min(128, Xs.shape[1], Xs.shape[0] - 1)
        if Xs.shape[1] > actual_pca:
            pca = PCA(n_components=actual_pca, random_state=42)
            Xs = pca.fit_transform(Xs); Xts = pca.transform(Xts)

        # Simple LR with CV-tuned C
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        best_au, best_C = -1, 1.0
        for C in [1e-2, 1e-1, 1.0, 10.0]:
            inner = np.zeros((n_trva, nc))
            for _, (ti, vi) in enumerate(skf.split(Xs, trva_labels)):
                inner[vi] = _fit_predict_proba(Xs[ti], trva_labels[ti], Xs[vi], nc, C=C, max_iter=2000)
            try:
                au = compute_auroc(trva_labels, inner, nc)
            except Exception:
                au = 0.5
            if au > best_au:
                best_au, best_C = au, C

        oof = np.zeros((n_trva, nc)); ta = np.zeros((len(te_labels), nc))
        for _, (ti, vi) in enumerate(skf.split(Xs, trva_labels)):
            if _USE_GPU_LR and nc == 2:
                p_vi, p_te = _gpu_lr_fit_predict(Xs[ti], trva_labels[ti], Xs[vi], Xts, C=best_C)
                oof[vi] = np.stack([1 - p_vi, p_vi], axis=1)
                ta += np.stack([1 - p_te, p_te], axis=1) / 5
            else:
                clf = LogisticRegression(max_iter=2000, C=best_C, random_state=42)
                clf.fit(Xs[ti], trva_labels[ti])
                oof[vi] = clf.predict_proba(Xs[vi])
                ta += clf.predict_proba(Xts) / 5
        all_oof.append(oof); all_te.append(ta)

    bs = best_single

    # Baseline 1: Score averaging
    avg_te = np.mean(all_te, axis=0)
    auroc_avg = compute_auroc(te_labels, avg_te, nc)

    # Baseline 2: LR on concatenated probe predictions
    meta_oof = np.hstack(all_oof); meta_te = np.hstack(all_te)
    sc_m = StandardScaler(); mo = sc_m.fit_transform(meta_oof); mt = sc_m.transform(meta_te)
    skf_meta = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    best_au_lr, best_C_lr = -1, 0.01
    for C in [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]:
        inner = np.zeros((n_trva, nc))
        for _, (ti, vi) in enumerate(skf_meta.split(mo, trva_labels)):
            clf = LogisticRegression(max_iter=3000, C=C, penalty='l2', solver='lbfgs', random_state=42)
            clf.fit(mo[ti], trva_labels[ti]); inner[vi] = clf.predict_proba(mo[vi])
        try:
            au = compute_auroc(trva_labels, inner, nc)
        except Exception:
            au = 0.5
        if au > best_au_lr:
            best_au_lr, best_C_lr = au, C
    clf_lr = LogisticRegression(max_iter=3000, C=best_C_lr, penalty='l2', solver='lbfgs', random_state=42)
    clf_lr.fit(mo, trva_labels)
    auroc_lr = compute_auroc(te_labels, clf_lr.predict_proba(mt), nc)

    return {
        "best_single": round(bs, 4),
        "score_avg": round(auroc_avg, 4),
        "score_avg_delta": f"{(auroc_avg - bs) * 100:+.2f}%",
        "lr_on_preds": round(auroc_lr, 4),
        "lr_on_preds_delta": f"{(auroc_lr - bs) * 100:+.2f}%",
    }


def main():
    for model in MODELS:
        processed_dir = str(cfg.base_processed / model)
        extraction_dir = str(cfg.base_extraction / model)
        oracle_path = str(cfg.model_results_dir(model) / "oracle_complete.json")
        out_path = str(cfg.model_results_dir(model) / "simple_fusion_baselines.json")
        os.makedirs(str(cfg.model_results_dir(model)), exist_ok=True)

        if not os.path.exists(oracle_path):
            print(f"[SKIP] {model}: oracle_complete.json missing at {oracle_path} — run exp1 first.")
            continue
        with open(oracle_path) as f:
            oracle = json.load(f)
        best_single = {ds: oracle[ds]["best_single_auroc"]
                       for ds in ALL_DATASETS if ds in oracle and "best_single_auroc" in oracle[ds]}

        print(f"\n===== {model} ({cfg.name}) =====")
        print("Simple Fusion Baselines")
        print("=" * 70)

        out = {}
        for ds_name, info in ALL_DATASETS.items():
            if ds_name not in best_single:
                print(f"[SKIP] {ds_name}: no best_single in oracle file")
                continue
            t0 = time.time()
            r = run(ds_name, info, processed_dir, extraction_dir, best_single[ds_name], model)
            out[ds_name] = r
            print(
                f"{ds_name:25s} best={r['best_single']:.4f} | "
                f"avg={r['score_avg']:.4f}({r['score_avg_delta']}) | "
                f"LR={r['lr_on_preds']:.4f}({r['lr_on_preds_delta']}) "
                f"[{time.time() - t0:.0f}s]"
            )

        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
