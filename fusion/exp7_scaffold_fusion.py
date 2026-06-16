"""Exp 7: Plug-and-play scaffold fusion.

Goal
----
This experiment tests a paper-facing method idea rather than another ad-hoc
classifier tweak:

  1. Each internal-state method plugs in by exposing train/val/test processed
     features. No method-specific model code is needed in this script.
  2. A generic adapter maps each method's feature to calibrated probabilities.
  3. The scaffold composes methods by validation-estimated reliability, with
     family-aware and timing-aware aggregators.
  4. The final scaffold variant is selected on validation only and evaluated on
     held-out test.

This is intentionally lighter than v21: it does not train a large expert
library or retrain the LLM. It asks whether a scalable scaffold can get robust
gains from heterogeneous probes with minimal per-method work.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, label_binarize

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fusion.settings import get_config


METHOD_TIMING = {
    "lr_probe": "input_side",
    "mm_probe": "input_side",
    "pca_lr": "input_side",
    "iti": "input_side",
    "attn_satisfies": "input_side",
    "kb_mlp": "generation_side",
    "lid": "generation_side",
    "llm_check": "generation_side",
    "sep": "generation_side",
    "coe": "generation_side",
    "seakr": "generation_side",
    "step": "generation_side",
}

METHOD_FAMILY = {
    "lr_probe": "residual_hidden",
    "mm_probe": "residual_hidden",
    "pca_lr": "residual_hidden",
    "kb_mlp": "residual_hidden",
    "iti": "attention_head",
    "attn_satisfies": "attention_flow",
    "lid": "geometry_uncertainty",
    "llm_check": "mixed_uncertainty",
    "sep": "semantic_uncertainty",
    "coe": "trajectory_geometry",
    "seakr": "sample_consistency",
    "step": "step_trajectory",
}

MC_METHODS = ["lr_probe", "pca_lr", "iti", "kb_mlp", "attn_satisfies", "sep", "step"]
BIN_EXTRA = ["mm_probe", "lid", "llm_check", "coe", "seakr"]


def compute_auroc(y, p, n_classes):
    y = np.asarray(y)
    p = np.asarray(p)
    if n_classes == 2:
        return float(roc_auc_score(y, p[:, 1]))
    yb = label_binarize(y, classes=list(range(n_classes)))
    return float(roc_auc_score(yb, p, average="macro", multi_class="ovr"))


def load_features(base, dataset, method):
    out = {}
    for split in ["train", "val", "test"]:
        path = base / dataset / method / f"{split}.pt"
        if not path.exists():
            return None
        x = torch.load(path, map_location="cpu", weights_only=False).float().numpy()
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        out[split] = x
    return out


def fit_adapter_predict(x_train, y_train, x_val, x_test, n_classes, max_dim, c_value):
    """Generic method adapter: feature vector -> probability vector.

    This is deliberately simple and uniform across methods. High-dimensional
    features are compressed after standardization; scalar scores pass through a
    logistic adapter. This keeps plug-in cost low for future methods.
    """
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_val = sc.transform(x_val)
    x_test = sc.transform(x_test)

    dim = min(max_dim, x_train.shape[1], max(1, x_train.shape[0] - 1))
    if x_train.shape[1] > dim:
        pca = PCA(n_components=dim, random_state=42)
        x_train = pca.fit_transform(x_train)
        x_val = pca.transform(x_val)
        x_test = pca.transform(x_test)

    clf = LogisticRegression(max_iter=2000, C=c_value, random_state=42)
    clf.fit(x_train, y_train)
    return clf.predict_proba(x_val), clf.predict_proba(x_test)


def fit_adapter_oof_predict(
    x_train, y_train, x_val, x_test, n_classes, max_dim, c_value, n_folds=5
):
    """Cross-fit generic adapters on train and average val/test predictions.

    This gives the meta scaffold training data that was not produced by models
    trained on the same examples, avoiding the validation-overfit problem of a
    meta learner trained directly on val predictions.
    """
    y_train = np.asarray(y_train)
    oof = np.zeros((len(y_train), n_classes), dtype=np.float64)
    val_avg = np.zeros((len(x_val), n_classes), dtype=np.float64)
    test_avg = np.zeros((len(x_test), n_classes), dtype=np.float64)
    n_splits = min(n_folds, int(np.bincount(y_train).min())) if n_classes == 2 else n_folds
    n_splits = max(2, min(n_folds, n_splits))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    for train_idx, held_idx in skf.split(x_train, y_train):
        pv_held, _ = fit_adapter_predict(
            x_train[train_idx], y_train[train_idx], x_train[held_idx], x_train[held_idx],
            n_classes, max_dim, c_value,
        )
        pv, pt = fit_adapter_predict(
            x_train[train_idx], y_train[train_idx], x_val, x_test,
            n_classes, max_dim, c_value,
        )
        oof[held_idx] = pv_held
        val_avg += pv / n_splits
        test_avg += pt / n_splits
    return oof, val_avg, test_avg


def reliability_weight(scores, floor=0.5, temperature=12.0):
    """Convert validation AUROCs into non-negative reliability weights."""
    vals = np.asarray([max(0.0, s - floor) for s in scores], dtype=np.float64)
    if vals.sum() <= 1e-12:
        return np.ones(len(scores), dtype=np.float64) / max(1, len(scores))
    logits = temperature * vals
    logits -= logits.max()
    w = np.exp(logits)
    return w / w.sum()


def weighted_average(preds, scores):
    names = list(preds)
    weights = reliability_weight([scores[n] for n in names])
    arr = np.zeros_like(preds[names[0]])
    for w, n in zip(weights, names):
        arr += w * preds[n]
    return arr, {n: float(w) for n, w in zip(names, weights)}


def group_weighted(preds, scores, group_map):
    """Select/weight methods inside groups, then weight groups."""
    groups = {}
    for name in preds:
        groups.setdefault(group_map.get(name, "unknown"), []).append(name)

    group_preds = {}
    group_scores = {}
    group_details = {}
    for group, names in groups.items():
        sub_preds = {n: preds[n] for n in names}
        sub_scores = {n: scores[n] for n in names}
        gp, gw = weighted_average(sub_preds, sub_scores)
        group_preds[group] = gp
        group_scores[group] = max(scores[n] for n in names)
        group_details[group] = gw

    final, group_weights = weighted_average(group_preds, group_scores)
    return final, {"group_weights": group_weights, "within_group_weights": group_details}


def meta_features(preds):
    names = list(preds)
    parts = []
    for n in names:
        p = preds[n]
        parts.append(p)
        ent = (-p * np.log(np.clip(p, 1e-12, 1.0))).sum(axis=1, keepdims=True)
        margin = np.sort(p, axis=1)[:, -1:] - np.sort(p, axis=1)[:, -2:-1]
        parts.extend([ent, margin])
    return np.hstack(parts), names


def fit_meta_lr(val_preds, y_val, test_preds, n_classes):
    x_val, names = meta_features(val_preds)
    x_test, _ = meta_features(test_preds)
    sc = StandardScaler()
    x_val = sc.fit_transform(x_val)
    x_test = sc.transform(x_test)
    clf = LogisticRegression(max_iter=2000, C=0.1, random_state=42)
    clf.fit(x_val, y_val)
    return clf.predict_proba(x_val), clf.predict_proba(x_test), names


def fit_oof_meta_lr(train_oof_preds, y_train, val_preds, y_val, test_preds, n_classes):
    x_train, names = meta_features(train_oof_preds)
    x_val, _ = meta_features(val_preds)
    x_test, _ = meta_features(test_preds)
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_val = sc.transform(x_val)
    x_test = sc.transform(x_test)

    best_auc, best_c = -1.0, 0.1
    for c in [1e-3, 1e-2, 1e-1, 1.0]:
        clf = LogisticRegression(max_iter=2000, C=c, random_state=42)
        clf.fit(x_train, y_train)
        try:
            auc = compute_auroc(y_val, clf.predict_proba(x_val), n_classes)
        except Exception:
            auc = 0.5
        if auc > best_auc:
            best_auc, best_c = auc, c

    clf = LogisticRegression(max_iter=2000, C=best_c, random_state=42)
    clf.fit(x_train, y_train)
    return clf.predict_proba(x_val), clf.predict_proba(x_test), names, best_c


def select_by_validation(candidates, y_val, y_test, n_classes):
    ranked = []
    for name, item in candidates.items():
        val_auc = compute_auroc(y_val, item["val_pred"], n_classes)
        test_auc = compute_auroc(y_test, item["test_pred"], n_classes)
        ranked.append((name, val_auc, test_auc))
    ranked.sort(key=lambda x: (-x[1], x[0]))
    selected = ranked[0]
    return {
        "selected_variant": selected[0],
        "selected_val_auroc": selected[1],
        "selected_test_auroc": selected[2],
        "variant_ranking": [
            {"variant": n, "val_auroc": float(v), "test_auroc": float(t)}
            for n, v, t in ranked
        ],
    }


def select_by_validation_with_margin(candidates, y_val, y_test, n_classes, margin):
    """Validation-only selection with optional no-regression guard.

    If the validation-best candidate is an aggregate, it must beat the
    validation-selected single method by at least `margin`; otherwise the
    scaffold falls back to the single method. With margin=0 this reduces to
    ordinary validation selection.
    """
    selected = select_by_validation(candidates, y_val, y_test, n_classes)
    if margin <= 0:
        selected["selection_margin"] = float(margin)
        selected["margin_guard_triggered"] = False
        return selected

    ranking = selected["variant_ranking"]
    single = next((r for r in ranking if r["variant"].startswith("single:")), None)
    best = ranking[0]
    if single is None or best["variant"].startswith("single:"):
        selected["selection_margin"] = float(margin)
        selected["margin_guard_triggered"] = False
        return selected

    if best["val_auroc"] - single["val_auroc"] >= margin:
        selected["selection_margin"] = float(margin)
        selected["margin_guard_triggered"] = False
        return selected

    guarded = {
        "selected_variant": single["variant"],
        "selected_val_auroc": single["val_auroc"],
        "selected_test_auroc": single["test_auroc"],
        "variant_ranking": ranking,
        "selection_margin": float(margin),
        "margin_guard_triggered": True,
        "guarded_best_variant": best["variant"],
        "guarded_best_val_auroc": best["val_auroc"],
        "guarded_best_test_auroc": best["test_auroc"],
    }
    return guarded


def run_dataset(
    cfg, model, dataset, ds_cfg, max_dim, c_value,
    enable_oof_meta=False, selection_margin=0.0,
):
    base = cfg.base_processed / model
    n_classes = ds_cfg["n_classes"]
    methods = MC_METHODS if n_classes > 2 else MC_METHODS + BIN_EXTRA

    try:
        y_train = cfg.load_labels(model, dataset, "train")
        y_val = cfg.load_labels(model, dataset, "val")
        y_test = cfg.load_labels(model, dataset, "test")
    except FileNotFoundError as e:
        return {"status": "skipped", "reason": f"missing labels/features: {e}"}

    val_preds = {}
    test_preds = {}
    train_oof_preds = {}
    method_rows = {}

    for method in methods:
        feats = load_features(base, dataset, method)
        if feats is None:
            continue
        try:
            pv, pt = fit_adapter_predict(
                feats["train"], y_train, feats["val"], feats["test"],
                n_classes=n_classes, max_dim=max_dim, c_value=c_value,
            )
            if enable_oof_meta:
                oof, _, _ = fit_adapter_oof_predict(
                    feats["train"], y_train, feats["val"], feats["test"],
                    n_classes=n_classes, max_dim=max_dim, c_value=c_value,
                )
                train_oof_preds[method] = oof
            val_auc = compute_auroc(y_val, pv, n_classes)
            test_auc = compute_auroc(y_test, pt, n_classes)
        except Exception as e:
            method_rows[method] = {"status": "failed", "error": str(e)}
            continue
        val_preds[method] = pv
        test_preds[method] = pt
        method_rows[method] = {
            "status": "ok",
            "val_auroc": float(val_auc),
            "test_auroc": float(test_auc),
            "timing": METHOD_TIMING.get(method, "unknown"),
            "family": METHOD_FAMILY.get(method, "unknown"),
        }

    if len(test_preds) < 2:
        return {"status": "skipped", "reason": f"only {len(test_preds)} usable methods"}

    val_scores = {m: method_rows[m]["val_auroc"] for m in test_preds}

    candidates = {}
    # 1. Best validation-selected single method.
    best_val_method = max(val_scores, key=val_scores.get)
    candidates[f"single:{best_val_method}"] = {
        "val_pred": val_preds[best_val_method],
        "test_pred": test_preds[best_val_method],
        "details": {"method": best_val_method},
    }

    # 2. Reliability-weighted all-method composition.
    v_all, w_all = weighted_average(val_preds, val_scores)
    t_all, _ = weighted_average(test_preds, val_scores)
    candidates["weighted_all_methods"] = {
        "val_pred": v_all,
        "test_pred": t_all,
        "details": {"weights": w_all},
    }

    # 3. Family-aware composition.
    v_family, d_family = group_weighted(val_preds, val_scores, METHOD_FAMILY)
    t_family, _ = group_weighted(test_preds, val_scores, METHOD_FAMILY)
    candidates["family_aware_scaffold"] = {
        "val_pred": v_family,
        "test_pred": t_family,
        "details": d_family,
    }

    # 4. Timing-aware composition: input-side vs generation-side.
    v_timing, d_timing = group_weighted(val_preds, val_scores, METHOD_TIMING)
    t_timing, _ = group_weighted(test_preds, val_scores, METHOD_TIMING)
    candidates["timing_aware_scaffold"] = {
        "val_pred": v_timing,
        "test_pred": t_timing,
        "details": d_timing,
    }

    # 5. Validation-trained meta adapter on method probabilities only.
    try:
        mv, mt, meta_names = fit_meta_lr(val_preds, y_val, test_preds, n_classes)
        candidates["meta_lr_scaffold"] = {
            "val_pred": mv,
            "test_pred": mt,
            "details": {"meta_input_methods": meta_names},
        }
    except Exception as e:
        candidates["meta_lr_scaffold_failed"] = {
            "val_pred": val_preds[best_val_method],
            "test_pred": test_preds[best_val_method],
            "details": {"error": str(e)},
        }

    if enable_oof_meta and len(train_oof_preds) == len(test_preds):
        try:
            ov, ot, oof_meta_names, best_c = fit_oof_meta_lr(
                train_oof_preds, y_train, val_preds, y_val, test_preds, n_classes
            )
            candidates["oof_meta_scaffold"] = {
                "val_pred": ov,
                "test_pred": ot,
                "details": {"meta_input_methods": oof_meta_names, "C": best_c},
            }
        except Exception as e:
            candidates["oof_meta_scaffold_failed"] = {
                "val_pred": val_preds[best_val_method],
                "test_pred": test_preds[best_val_method],
                "details": {"error": str(e)},
            }

    # Default scaffold policy: select only among low-capacity, plug-and-play
    # aggregators. A validation-trained meta-LR is still recorded as a diagnostic
    # candidate, but it is not allowed to drive the default selection because the
    # smoke test showed it can overfit validation and hurt held-out test.
    stable_candidates = {
        k: v for k, v in candidates.items()
        if k in {
            f"single:{best_val_method}",
            "weighted_all_methods",
            "family_aware_scaffold",
            "timing_aware_scaffold",
            "oof_meta_scaffold",
        }
    }
    selection = select_by_validation_with_margin(
        stable_candidates, y_val, y_test, n_classes, selection_margin
    )
    diagnostic_selection_all = select_by_validation(candidates, y_val, y_test, n_classes)
    best_test_method = max(
        (m for m in method_rows if method_rows[m].get("status") == "ok"),
        key=lambda m: method_rows[m]["test_auroc"],
    )
    best_val_single_test = method_rows[best_val_method]["test_auroc"]
    best_test_single = method_rows[best_test_method]["test_auroc"]

    return {
        "status": "done",
        "n_methods": len(test_preds),
        "method_results": method_rows,
        "best_val_single_method": best_val_method,
        "best_val_single_test_auroc": float(best_val_single_test),
        "best_test_single_method": best_test_method,
        "best_test_single_auroc_oracle": float(best_test_single),
        "selection": selection,
        "diagnostic_selection_all_candidates": diagnostic_selection_all,
        "selected_delta_vs_val_single": float(selection["selected_test_auroc"] - best_val_single_test),
        "selected_delta_vs_test_oracle_single": float(selection["selected_test_auroc"] - best_test_single),
        "candidate_details": {
            name: item.get("details", {}) for name, item in candidates.items()
        },
    }


def write_markdown(path, model, setting, results):
    lines = [f"# Exp7 Scaffold Fusion: {model} ({setting})", ""]
    lines.append("| Dataset | Selected | Test AUROC | Best val-single test | Delta | Test-oracle single |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for ds, r in results.items():
        if r.get("status") != "done":
            lines.append(f"| {ds} | skipped | | | | |")
            continue
        sel = r["selection"]
        lines.append(
            f"| {ds} | {sel['selected_variant']} | {sel['selected_test_auroc']:.4f} | "
            f"{r['best_val_single_test_auroc']:.4f} | {r['selected_delta_vs_val_single']:+.4f} | "
            f"{r['best_test_single_auroc_oracle']:.4f} |"
        )
    path.write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5-7b")
    ap.add_argument("--setting", default="old", choices=["old", "new"])
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--max-dim", type=int, default=128)
    ap.add_argument("--C", type=float, default=0.1)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--out-prefix", default="scaffold_fusion")
    ap.add_argument("--enable-oof-meta", action="store_true")
    ap.add_argument("--selection-margin", type=float, default=0.0)
    args = ap.parse_args()

    cfg = get_config(args.setting)
    out_dir = cfg.model_results_dir(args.model)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / f"{args.out_prefix}.json"
    out_md = out_dir / f"{args.out_prefix}.md"

    selected_datasets = args.datasets or list(cfg.datasets)

    print("=" * 70)
    print(f"EXP 7: Plug-and-play Scaffold Fusion | model={args.model} setting={args.setting}")
    print("=" * 70)

    results = {}
    if out_json.exists():
        try:
            results = json.loads(out_json.read_text())
            print(f"[CHECKPOINT] loaded {len(results)} existing datasets from {out_json}")
        except Exception:
            results = {}

    for ds in selected_datasets:
        if ds not in cfg.datasets:
            print(f"[SKIP] {ds}: not in setting config")
            continue
        if not args.force and ds in results and results[ds].get("status") == "done":
            print(f"[SKIP] {ds}: already done")
            continue
        t0 = time.time()
        r = run_dataset(
            cfg, args.model, ds, cfg.datasets[ds],
            max_dim=args.max_dim, c_value=args.C,
            enable_oof_meta=args.enable_oof_meta,
            selection_margin=args.selection_margin,
        )
        results[ds] = r
        if r.get("status") == "done":
            sel = r["selection"]
            print(
                f"{ds:25s} selected={sel['selected_variant']:26s} "
                f"test={sel['selected_test_auroc']:.4f} "
                f"delta_vs_val_single={r['selected_delta_vs_val_single']:+.4f} "
                f"oracle_single={r['best_test_single_auroc_oracle']:.4f} "
                f"[{time.time() - t0:.1f}s]"
            )
        else:
            print(f"{ds:25s} skipped: {r.get('reason')}")
        out_json.write_text(json.dumps(results, indent=2))
        write_markdown(out_md, args.model, args.setting, results)

    print(f"Saved {out_json}")
    print(f"Saved {out_md}")


if __name__ == "__main__":
    main()
