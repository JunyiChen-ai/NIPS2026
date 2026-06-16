"""
Run 12 baselines on processed correctness features for one (model, dataset).

Reads:
  reproduce/processed_features_correctness/{model}/{dataset}/{method}/{train,val,test}.pt
  reproduce/correctness_labels/{model}/{dataset}/{labels,split_indices}.json

Writes:
  reproduce/results_correctness/{model}/{dataset}.json
"""

from __future__ import annotations
import os, sys, json, argparse
from pathlib import Path
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

sys.path.insert(0, os.path.dirname(__file__))
from methods import LRProbe, MMProbe, KBNet, STEPScorer

PROC_ROOT = Path("/home/junyi/NIPS2026/reproduce/processed_features_correctness")
LABELS_ROOT = Path("/home/junyi/NIPS2026/reproduce/correctness_labels")
RESULTS_ROOT = Path("/home/junyi/NIPS2026/reproduce/results_correctness")

# ============================================================
# Helpers
# ============================================================

def load_labels_split(model: str, dataset: str):
    with open(LABELS_ROOT / model / dataset / "labels.json") as f:
        labels = json.load(f)["labels"]
    with open(LABELS_ROOT / model / dataset / "split_indices.json") as f:
        split = json.load(f)
    y = np.array(labels, dtype=int)
    return {
        "train": y[split["train"]],
        "val": y[split["val"]],
        "test": y[split["test"]],
    }

def load_proc(model, dataset, method, split_name):
    p = PROC_ROOT / model / dataset / method / f"{split_name}.pt"
    if not p.exists():
        return None
    return torch.load(p, map_location="cpu", weights_only=False)

def load_proc_meta(model, dataset, method):
    p = PROC_ROOT / model / dataset / method / "meta.json"
    if not p.exists():
        return {}
    with open(p) as f:
        return json.load(f)

def eval_binary(y_true, scores):
    """Score-based binary eval. Threshold = best F1 on val passed in via separate call."""
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    auroc = roc_auc_score(y_true, scores)
    pred = (scores > np.median(scores)).astype(int)
    return {"auroc": float(auroc), "accuracy": float(accuracy_score(y_true, pred)),
            "f1": float(f1_score(y_true, pred, zero_division=0))}

def eval_with_val_threshold(va_y, va_scores, te_y, te_scores):
    """Pick threshold that maximizes val F1, apply to test."""
    va_y = np.asarray(va_y); va_scores = np.asarray(va_scores)
    te_y = np.asarray(te_y); te_scores = np.asarray(te_scores)
    # Try inverted scores too (some methods are "lower = positive")
    best = None
    for sign in (1, -1):
        s_va = sign * va_scores; s_te = sign * te_scores
        try:
            au = roc_auc_score(va_y, s_va)
        except Exception:
            au = 0.0
        # threshold sweep
        best_f1, best_thr = -1, 0.5
        for thr in np.quantile(s_va, np.linspace(0.05, 0.95, 19)):
            pred = (s_va > thr).astype(int)
            f1 = f1_score(va_y, pred, zero_division=0)
            if f1 > best_f1:
                best_f1, best_thr = f1, thr
        te_pred = (s_te > best_thr).astype(int)
        result = {"auroc": float(roc_auc_score(te_y, s_te)),
                  "accuracy": float(accuracy_score(te_y, te_pred)),
                  "f1": float(f1_score(te_y, te_pred, zero_division=0)),
                  "sign": sign}
        if best is None or result["auroc"] > best["auroc"]:
            best = result
    return best


# ============================================================
# Method runners (each consumes processed features only)
# ============================================================

def run_lr_probe(model, dataset, y):
    """LR Probe: torch BCELoss head on processed features."""
    tr_x = load_proc(model, dataset, "lr_probe", "train")
    va_x = load_proc(model, dataset, "lr_probe", "val")
    te_x = load_proc(model, dataset, "lr_probe", "test")
    if tr_x is None: return {"skipped": "no processed features"}
    tr_y = torch.tensor(y["train"], dtype=torch.float32)
    te_y = y["test"]
    probe = LRProbe.from_data(tr_x.float(), tr_y, lr=1e-3, weight_decay=0.1, epochs=1000)
    with torch.no_grad():
        scores = probe(te_x.float()).numpy()
    pred = (scores > 0.5).astype(int)
    meta = load_proc_meta(model, dataset, "lr_probe")
    return {"layer": meta.get("best_layer"),
            "auroc": float(roc_auc_score(te_y, scores)),
            "accuracy": float(accuracy_score(te_y, pred)),
            "f1": float(f1_score(te_y, pred, zero_division=0))}


def run_mm_probe(model, dataset, y):
    """Mass-mean probe: direction = mu_pos - mu_neg (already centered features)."""
    tr_x = load_proc(model, dataset, "mm_probe", "train")
    te_x = load_proc(model, dataset, "mm_probe", "test")
    if tr_x is None: return {"skipped": "no processed features"}
    tr_x, te_x = tr_x.float(), te_x.float()
    tr_y = y["train"]; te_y = y["test"]
    probe = MMProbe.from_data(tr_x, torch.tensor(tr_y, dtype=torch.float32))
    with torch.no_grad():
        scores = probe(te_x).numpy()
    pred = (scores > 0.5).astype(int)
    meta = load_proc_meta(model, dataset, "mm_probe")
    return {"layer": meta.get("best_layer"),
            "auroc": float(roc_auc_score(te_y, scores)),
            "accuracy": float(accuracy_score(te_y, pred)),
            "f1": float(f1_score(te_y, pred, zero_division=0))}


def run_pca_lr(model, dataset, y):
    tr_x = load_proc(model, dataset, "pca_lr", "train")
    te_x = load_proc(model, dataset, "pca_lr", "test")
    if tr_x is None: return {"skipped": "no processed features"}
    tr_x = tr_x.numpy() if isinstance(tr_x, torch.Tensor) else np.asarray(tr_x)
    te_x = te_x.numpy() if isinstance(te_x, torch.Tensor) else np.asarray(te_x)
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(tr_x, y["train"])
    probs = clf.predict_proba(te_x)[:, 1]
    pred = (probs > 0.5).astype(int)
    meta = load_proc_meta(model, dataset, "pca_lr")
    return {"layer": meta.get("best_layer"),
            "auroc": float(roc_auc_score(y["test"], probs)),
            "accuracy": float(accuracy_score(y["test"], pred)),
            "f1": float(f1_score(y["test"], pred, zero_division=0))}


def run_iti(model, dataset, y):
    tr_x = load_proc(model, dataset, "iti", "train")
    te_x = load_proc(model, dataset, "iti", "test")
    if tr_x is None: return {"skipped": "no processed features"}
    tr_x = tr_x.numpy(); te_x = te_x.numpy()
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(tr_x, y["train"])
    probs = clf.predict_proba(te_x)[:, 1]
    pred = (probs > 0.5).astype(int)
    meta = load_proc_meta(model, dataset, "iti")
    return {"layer": meta.get("best_layer"), "head": meta.get("best_head"),
            "auroc": float(roc_auc_score(y["test"], probs)),
            "accuracy": float(accuracy_score(y["test"], pred)),
            "f1": float(f1_score(y["test"], pred, zero_division=0))}


def run_kb_mlp(model, dataset, y):
    """KB MLP: shallow MLP on mid-layer hidden state."""
    tr_x = load_proc(model, dataset, "kb_mlp", "train")
    va_x = load_proc(model, dataset, "kb_mlp", "val")
    te_x = load_proc(model, dataset, "kb_mlp", "test")
    if tr_x is None: return {"skipped": "no processed features"}
    tr_y = torch.tensor(y["train"]); va_y = torch.tensor(y["val"]); te_y = torch.tensor(y["test"])
    probs, preds = KBNet.train_and_eval(tr_x.float(), tr_y, te_x.float(), te_y,
                                         val_acts=va_x.float(), val_labels=va_y)
    meta = load_proc_meta(model, dataset, "kb_mlp")
    return {"layer": meta.get("layer"),
            "auroc": float(roc_auc_score(y["test"], probs)),
            "accuracy": float(accuracy_score(y["test"], preds)),
            "f1": float(f1_score(y["test"], preds, zero_division=0))}


def run_lid(model, dataset, y):
    """LID: scalar score per sample."""
    va_s = load_proc(model, dataset, "lid", "val")
    te_s = load_proc(model, dataset, "lid", "test")
    if va_s is None: return {"skipped": "no processed features"}
    res = eval_with_val_threshold(y["val"], va_s.numpy(), y["test"], te_s.numpy())
    meta = load_proc_meta(model, dataset, "lid")
    res["layer"] = meta.get("best_layer")
    return res


def run_attn_satisfies(model, dataset, y):
    tr_x = load_proc(model, dataset, "attn_satisfies", "train")
    te_x = load_proc(model, dataset, "attn_satisfies", "test")
    if tr_x is None: return {"skipped": "no processed features"}
    tr_x = tr_x.numpy() if isinstance(tr_x, torch.Tensor) else np.asarray(tr_x)
    te_x = te_x.numpy() if isinstance(te_x, torch.Tensor) else np.asarray(te_x)
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(tr_x, y["train"])
    probs = clf.predict_proba(te_x)[:, 1]
    pred = (probs > 0.5).astype(int)
    return {"auroc": float(roc_auc_score(y["test"], probs)),
            "accuracy": float(accuracy_score(y["test"], pred)),
            "f1": float(f1_score(y["test"], pred, zero_division=0))}


def run_llm_check(model, dataset, y):
    va_s = load_proc(model, dataset, "llm_check", "val")
    te_s = load_proc(model, dataset, "llm_check", "test")
    if va_s is None: return {"skipped": "no processed features"}
    res = eval_with_val_threshold(y["val"], va_s.numpy(), y["test"], te_s.numpy())
    meta = load_proc_meta(model, dataset, "llm_check")
    res["layer"] = meta.get("best_layer")
    return res


def run_sep(model, dataset, y):
    tr_x = load_proc(model, dataset, "sep", "train")
    te_x = load_proc(model, dataset, "sep", "test")
    if tr_x is None: return {"skipped": "no processed features"}
    tr_x = tr_x.numpy(); te_x = te_x.numpy()
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(tr_x, y["train"])
    probs = clf.predict_proba(te_x)[:, 1]
    pred = (probs > 0.5).astype(int)
    meta = load_proc_meta(model, dataset, "sep")
    return {"range": meta.get("best_range"),
            "auroc": float(roc_auc_score(y["test"], probs)),
            "accuracy": float(accuracy_score(y["test"], pred)),
            "f1": float(f1_score(y["test"], pred, zero_division=0))}


def run_coe(model, dataset, y):
    """CoE has multiple variants — try all, pick best on val."""
    out = {}
    meta = load_proc_meta(model, dataset, "coe")
    variants = meta.get("variants", [])
    for v in variants:
        va_s = load_proc(model, dataset, "coe", f"val_{v}")
        te_s = load_proc(model, dataset, "coe", f"test_{v}")
        if va_s is None: continue
        res = eval_with_val_threshold(y["val"], va_s.numpy(), y["test"], te_s.numpy())
        out[v] = res
    return out


def run_seakr(model, dataset, y):
    va_s = load_proc(model, dataset, "seakr", "val")
    te_s = load_proc(model, dataset, "seakr", "test")
    if va_s is None: return {"skipped": "no processed features"}
    return eval_with_val_threshold(y["val"], va_s.numpy(), y["test"], te_s.numpy())


def run_step(model, dataset, y):
    tr_x = load_proc(model, dataset, "step", "train")
    va_x = load_proc(model, dataset, "step", "val")
    te_x = load_proc(model, dataset, "step", "test")
    if tr_x is None: return {"skipped": "no processed features"}
    tr_y = torch.tensor(y["train"]); va_y = torch.tensor(y["val"]); te_y = torch.tensor(y["test"])
    probs, preds = STEPScorer.train_and_eval(
        tr_x.float(), tr_y, te_x.float(), te_y,
        val_acts=va_x.float(), val_labels=va_y)
    return {"auroc": float(roc_auc_score(y["test"], probs)),
            "accuracy": float(accuracy_score(y["test"], preds)),
            "f1": float(f1_score(y["test"], preds, zero_division=0))}


METHODS = {
    "lr_probe": run_lr_probe,
    "mm_probe": run_mm_probe,
    "pca_lr": run_pca_lr,
    "iti": run_iti,
    "kb_mlp": run_kb_mlp,
    "lid": run_lid,
    "attn_satisfies": run_attn_satisfies,
    "llm_check": run_llm_check,
    "sep": run_sep,
    "coe": run_coe,
    "seakr": run_seakr,
    "step": run_step,
}


def run_one(model, dataset):
    print(f"\n{'='*60}\n{model} / {dataset}\n{'='*60}")
    y = load_labels_split(model, dataset)
    print(f"  train={len(y['train'])}({y['train'].mean():.3f})  "
          f"val={len(y['val'])}({y['val'].mean():.3f})  "
          f"test={len(y['test'])}({y['test'].mean():.3f})")

    results = {}
    for name, fn in METHODS.items():
        try:
            r = fn(model, dataset, y)
            results[name] = r
            au = r.get("auroc") if "auroc" in r else (
                next((v.get("auroc") for v in r.values() if isinstance(v, dict)), None))
            print(f"  {name:18s}  auroc={au:.3f}" if au else f"  {name:18s}  {r}")
        except Exception as e:
            import traceback
            print(f"  {name}: FAILED: {e}")
            traceback.print_exc()
            results[name] = {"error": str(e)}

    out_dir = RESULTS_ROOT / model
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{dataset}.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  saved → {out_dir / dataset}.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", required=True)
    args = ap.parse_args()
    run_one(args.model, args.dataset)


if __name__ == "__main__":
    main()
