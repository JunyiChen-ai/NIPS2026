"""
Per-(model, dataset) processed feature generation for the 6 CoE-style datasets.

Reads:
  /home/junyi/b2-nips/extraction/features/{model}/{dataset}/all/*.{pt,json}  (B2 mount)
  reproduce/correctness_labels/{model}/{dataset}/labels.json
  reproduce/correctness_labels/{model}/{dataset}/split_indices.json

Writes:
  reproduce/processed_features_correctness/{model}/{dataset}/{method}/{split}.pt

Methods (binary classification with correctness label):
  lr_probe, mm_probe, pca_lr, iti, kb_mlp, lid, attn_satisfies, llm_check, sep, coe, seakr, step
"""

from __future__ import annotations
import os, sys, json, time, argparse
from pathlib import Path
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(__file__))
from methods import compute_lid, llm_check_score, compute_coe_scores, seakr_energy_score


# ============================================================
# Fast LR for layer/range selection — GPU LBFGS, much faster than sklearn
# ============================================================
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def fast_lr_auroc(X_tr, y_tr, X_va, y_va, l2=1e-2, max_iter=50):
    """Fast logistic regression on GPU using LBFGS. Returns val AUROC."""
    Xt = torch.as_tensor(X_tr, dtype=torch.float32, device=_DEVICE)
    yt = torch.as_tensor(y_tr, dtype=torch.float32, device=_DEVICE)
    Xv = torch.as_tensor(X_va, dtype=torch.float32, device=_DEVICE)
    D = Xt.shape[1]
    W = torch.zeros(D, requires_grad=True, device=_DEVICE)
    b = torch.zeros(1, requires_grad=True, device=_DEVICE)
    opt = torch.optim.LBFGS([W, b], lr=1.0, max_iter=max_iter, tolerance_grad=1e-6, history_size=10)
    def closure():
        opt.zero_grad()
        logits = Xt @ W + b
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, yt)
        loss = loss + l2 * (W * W).sum()
        loss.backward()
        return loss
    opt.step(closure)
    with torch.no_grad():
        logits = Xv @ W + b
        scores = torch.sigmoid(logits).cpu().numpy()
    del Xt, Xv, yt, W, b
    if _DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    return roc_auc_score(y_va, scores)

FEATURES_ROOT = Path("/home/junyi/b2-nips/extraction/features")
LABELS_ROOT = Path("/home/junyi/NIPS2026/reproduce/correctness_labels")
OUTPUT_ROOT = Path("/home/junyi/NIPS2026/reproduce/processed_features_correctness")
HIDDEN_DIM = 3584


# ============================================================
# Loaders (each tensor file is read directly from B2 mount)
# ============================================================

def load_tensor(model: str, dataset: str, name: str):
    path = FEATURES_ROOT / model / dataset / "all" / f"{name}.pt"
    return torch.load(path, map_location="cpu", weights_only=False)

def load_json(model: str, dataset: str, name: str):
    path = FEATURES_ROOT / model / dataset / "all" / f"{name}.json"
    with open(path) as f:
        return json.load(f)

def load_labels_split(model: str, dataset: str):
    with open(LABELS_ROOT / model / dataset / "labels.json") as f:
        labels = json.load(f)["labels"]
    with open(LABELS_ROOT / model / dataset / "split_indices.json") as f:
        split = json.load(f)
    return np.array(labels, dtype=int), split


def slice_split(tensor, indices):
    """Slice tensor (or list-of-things-zero-indexed) by integer index list."""
    idx = torch.as_tensor(indices, dtype=torch.long)
    return tensor[idx]


# ============================================================
# I/O helpers
# ============================================================

def save_feat(model, dataset, method, split_name, tensor):
    out_dir = OUTPUT_ROOT / model / dataset / method
    out_dir.mkdir(parents=True, exist_ok=True)
    if isinstance(tensor, np.ndarray):
        tensor = torch.from_numpy(tensor)
    torch.save(tensor, out_dir / f"{split_name}.pt")

def save_meta(model, dataset, method, meta_dict):
    out_dir = OUTPUT_ROOT / model / dataset / method
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta_dict, f, indent=2)

def is_method_done(model, dataset, method, splits=("train", "val", "test")):
    out_dir = OUTPUT_ROOT / model / dataset / method
    if not out_dir.exists():
        return False
    # CoE writes split_variant.pt, others write split.pt
    if method == "coe":
        # Just check if any train_*.pt exists
        return any(p.name.startswith("train_") for p in out_dir.iterdir())
    return all((out_dir / f"{s}.pt").exists() for s in splits)


# ============================================================
# Methods
# ============================================================

def select_best_layer_binary(tr_x, tr_y, va_x, va_y):
    """tr_x, va_x: (N, n_layers, hidden_dim). Return best layer index. GPU LBFGS for speed."""
    n_layers = tr_x.shape[1]
    best, best_layer = -1.0, 0
    for layer in range(n_layers):
        Xtr = tr_x[:, layer, :].float().numpy()
        Xva = va_x[:, layer, :].float().numpy()
        au = fast_lr_auroc(Xtr, tr_y, Xva, va_y)
        if au > best:
            best, best_layer = au, layer
    return best_layer, best


def process_lr_probe(model, dataset, data, labels, split):
    tr_idx, va_idx = split["train"], split["val"]
    h = data["input_last_token_hidden"]  # (N, n_layers, D)
    tr_x = h[tr_idx]; va_x = h[va_idx]
    tr_y = labels[tr_idx]; va_y = labels[va_idx]
    layer, va_auc = select_best_layer_binary(tr_x, tr_y, va_x, va_y)
    for s in ("train", "val", "test"):
        idx = split[s]
        feat = h[idx][:, layer, :].float()
        save_feat(model, dataset, "lr_probe", s, feat)
    save_meta(model, dataset, "lr_probe", {"best_layer": layer, "val_auroc": float(va_auc),
                                            "shape": "N x hidden_dim"})
    print(f"    lr_probe: layer={layer}  val_auroc={va_auc:.3f}")


def process_mm_probe(model, dataset, data, labels, split):
    tr_idx, va_idx = split["train"], split["val"]
    h = data["input_last_token_hidden"]
    tr_x = h[tr_idx]; va_x = h[va_idx]
    tr_y = labels[tr_idx]; va_y = labels[va_idx]
    layer, _ = select_best_layer_binary(tr_x, tr_y, va_x, va_y)
    mean = h[tr_idx][:, layer, :].float().mean(dim=0)
    for s in ("train", "val", "test"):
        idx = split[s]
        feat = h[idx][:, layer, :].float() - mean
        save_feat(model, dataset, "mm_probe", s, feat)
    save_meta(model, dataset, "mm_probe", {"best_layer": layer, "shape": "N x hidden_dim",
                                            "desc": "centered hidden state at best layer"})
    print(f"    mm_probe: layer={layer}")


def process_pca_lr(model, dataset, data, labels, split):
    tr_idx, va_idx = split["train"], split["val"]
    h = data["input_last_token_hidden"]
    tr_y = labels[tr_idx]; va_y = labels[va_idx]
    n_layers = h.shape[1]
    best_auroc, best_layer = -1.0, 0
    for layer in range(n_layers):
        tr_acts = h[tr_idx][:, layer, :].float()
        va_acts = h[va_idx][:, layer, :].float()
        mean = tr_acts.mean(dim=0)
        tr_c, va_c = tr_acts - mean, va_acts - mean
        U, S, Vh = torch.linalg.svd(tr_c, full_matrices=False)
        n_comp = min(50, tr_c.shape[0], tr_c.shape[1])
        tr_pca = (tr_c @ Vh.T[:, :n_comp]).numpy()
        va_pca = (va_c @ Vh.T[:, :n_comp]).numpy()
        sc = StandardScaler()
        tr_pca = sc.fit_transform(tr_pca)
        va_pca = sc.transform(va_pca)
        # PCA features are small (50 dim), sklearn LR is fast — keep
        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(tr_pca, tr_y)
        auroc = roc_auc_score(va_y, clf.predict_proba(va_pca)[:, 1])
        if auroc > best_auroc:
            best_auroc, best_layer = auroc, layer
    tr_acts = h[tr_idx][:, best_layer, :].float()
    mean = tr_acts.mean(dim=0)
    U, S, Vh = torch.linalg.svd(tr_acts - mean, full_matrices=False)
    n_comp = min(50, tr_acts.shape[0], tr_acts.shape[1])
    sc = StandardScaler()
    sc.fit(((tr_acts - mean) @ Vh.T[:, :n_comp]).numpy())
    for s in ("train", "val", "test"):
        idx = split[s]
        acts = h[idx][:, best_layer, :].float()
        pca = sc.transform(((acts - mean) @ Vh.T[:, :n_comp]).numpy())
        save_feat(model, dataset, "pca_lr", s, pca)
    save_meta(model, dataset, "pca_lr", {"best_layer": best_layer, "n_components": n_comp,
                                          "shape": "N x 50"})
    print(f"    pca_lr: layer={best_layer}, n_comp={n_comp}")


def process_iti(model, dataset, data, labels, split):
    """ITI: scan all (layer, head) on GPU. Each fit is small (head_dim=128)."""
    tr_idx, va_idx = split["train"], split["val"]
    a = data["input_per_head_activation"]  # (N, n_layers, n_heads, head_dim)
    tr_y = labels[tr_idx]; va_y = labels[va_idx]
    n_layers, n_heads = a.shape[1], a.shape[2]
    best_val, best_li, best_hi = -1.0, 0, 0
    for li in range(n_layers):
        for hi in range(n_heads):
            Xtr = a[tr_idx][:, li, hi, :].numpy()
            Xva = a[va_idx][:, li, hi, :].numpy()
            au = fast_lr_auroc(Xtr, tr_y, Xva, va_y, max_iter=30)
            if au > best_val:
                best_val, best_li, best_hi = au, li, hi
    for s in ("train", "val", "test"):
        idx = split[s]
        save_feat(model, dataset, "iti", s, a[idx][:, best_li, best_hi, :])
    save_meta(model, dataset, "iti", {"best_layer": best_li, "best_head": best_hi,
                                       "val_auroc": float(best_val), "shape": "N x head_dim"})
    print(f"    iti: layer={best_li}, head={best_hi}, val_auroc={best_val:.3f}")


def process_kb_mlp(model, dataset, data, labels, split):
    h = data["input_last_token_hidden"]
    n_layers = h.shape[1]
    mid = n_layers // 2
    for s in ("train", "val", "test"):
        idx = split[s]
        save_feat(model, dataset, "kb_mlp", s, h[idx][:, mid, :].float())
    save_meta(model, dataset, "kb_mlp", {"layer": mid, "shape": "N x hidden_dim"})
    print(f"    kb_mlp: layer={mid}")


def process_lid(model, dataset, data, labels, split):
    h = data["input_last_token_hidden"]
    tr_idx, va_idx = split["train"], split["val"]
    tr_y = labels[tr_idx]; va_y = labels[va_idx]
    n_layers = h.shape[1]
    best_metric, best_layer = -1.0, 0
    for layer in range(n_layers):
        tr_acts = h[tr_idx][:, layer, :]
        va_acts = h[va_idx][:, layer, :]
        ref = tr_acts[torch.tensor(tr_y) == 1]
        k = len(ref) - 1
        if k < 2:
            continue
        try:
            lids = compute_lid(ref, va_acts, k=k, hidden_dim=HIDDEN_DIM)
        except Exception as e:
            print(f"      LID layer {layer} failed: {e}")
            continue
        ap = roc_auc_score(va_y, lids); an = roc_auc_score(va_y, -lids)
        m = max(ap, an)
        if m > best_metric:
            best_metric, best_layer = m, layer
    ref = h[tr_idx][:, best_layer, :][torch.tensor(tr_y) == 1]
    k = len(ref) - 1
    for s in ("train", "val", "test"):
        idx = split[s]
        acts = h[idx][:, best_layer, :]
        scores = compute_lid(ref, acts, k=k, hidden_dim=HIDDEN_DIM)
        save_feat(model, dataset, "lid", s, torch.tensor(scores, dtype=torch.float32))
    save_meta(model, dataset, "lid", {"best_layer": best_layer, "val_auroc": float(best_metric),
                                       "shape": "N"})
    print(f"    lid: layer={best_layer}, val_auroc={best_metric:.3f}")


def process_attn_satisfies(model, dataset, data, labels, split):
    """Memory-efficient: amax over seq dim before float promotion.
    Avoids materializing full fp32 tensor (25+ GB for math)."""
    avn = data["input_attn_value_norms"]  # (N, n_layers, n_heads, seq_len) fp16
    pooled = avn.amax(dim=-1).float()  # (N, n_layers, n_heads) — small
    tr_idx = split["train"]
    n_tr = len(tr_idx)
    tr_feat = pooled[tr_idx].reshape(n_tr, -1).numpy()
    sc = StandardScaler()
    tr_feat = sc.fit_transform(tr_feat)
    save_feat(model, dataset, "attn_satisfies", "train", tr_feat)
    for s in ("val", "test"):
        idx = split[s]
        feat = pooled[idx].reshape(len(idx), -1).numpy()
        save_feat(model, dataset, "attn_satisfies", s, sc.transform(feat))
    save_meta(model, dataset, "attn_satisfies", {"shape": "N x (n_layers*n_heads)"})
    print(f"    attn_satisfies: done")


def process_llm_check(model, dataset, data, labels, split):
    attn = data["input_attn_stats"]  # (N, n_layers, n_heads, 3)
    va_idx = split["val"]
    va_y = labels[va_idx]
    n_layers = attn.shape[1]
    best_val, best_layer = -1.0, 0
    for layer in range(n_layers):
        scores = llm_check_score(attn[va_idx], layer_num=layer)
        ap = roc_auc_score(va_y, scores); an = roc_auc_score(va_y, -scores)
        m = max(ap, an)
        if m > best_val:
            best_val, best_layer = m, layer
    for s in ("train", "val", "test"):
        idx = split[s]
        scores = llm_check_score(attn[idx], layer_num=best_layer)
        save_feat(model, dataset, "llm_check", s, torch.tensor(scores, dtype=torch.float32))
    save_meta(model, dataset, "llm_check", {"best_layer": best_layer, "val_auroc": float(best_val),
                                              "shape": "N"})
    print(f"    llm_check: layer={best_layer}, val_auroc={best_val:.3f}")


def process_sep(model, dataset, data, labels, split):
    """SEP layer-range selection. Use GPU LBFGS for fast scan, sklearn LR not needed
    since processed features are saved (re-trained downstream)."""
    h = data["gen_last_token_hidden"]
    tr_idx, va_idx = split["train"], split["val"]
    tr_y = labels[tr_idx]; va_y = labels[va_idx]
    n_layers = h.shape[1]

    # Single-layer scan (fast, 30 GPU LR fits)
    layer_aurocs = {}
    for layer in range(n_layers):
        Xtr = h[tr_idx][:, layer, :].float().numpy()
        Xva = h[va_idx][:, layer, :].float().numpy()
        layer_aurocs[layer] = fast_lr_auroc(Xtr, tr_y, Xva, va_y)

    # Range expansion around top-3 layers (~25 GPU LR fits)
    best_layers = sorted(layer_aurocs, key=lambda x: -layer_aurocs[x])[:3]
    best_auroc = max(layer_aurocs.values())
    best_layer = max(layer_aurocs, key=lambda k: layer_aurocs[k])
    best_range = (best_layer, best_layer + 1)
    for center in best_layers:
        for start in range(max(0, center-2), min(n_layers, center+3)):
            for end in range(start + 2, min(start + 6, n_layers + 1)):
                Xtr = h[tr_idx][:, start:end, :].float().reshape(len(tr_idx), -1).numpy()
                Xva = h[va_idx][:, start:end, :].float().reshape(len(va_idx), -1).numpy()
                au = fast_lr_auroc(Xtr, tr_y, Xva, va_y)
                if au > best_auroc:
                    best_auroc, best_range = au, (start, end)

    for s in ("train", "val", "test"):
        idx = split[s]
        feat = h[idx][:, best_range[0]:best_range[1], :].float().reshape(len(idx), -1)
        save_feat(model, dataset, "sep", s, feat)
    save_meta(model, dataset, "sep", {"best_range": list(best_range), "val_auroc": float(best_auroc),
                                       "shape": f"N x ({best_range[1]-best_range[0]} * D)"})
    print(f"    sep: range={best_range}, val_auroc={best_auroc:.3f}")


def process_coe(model, dataset, data, labels, split):
    h = data["gen_mean_pool_hidden"]  # (N, n_layers, D)
    for s in ("train", "val", "test"):
        idx = split[s]
        scores = compute_coe_scores(h[idx])
        for variant, vals in scores.items():
            save_feat(model, dataset, "coe", f"{s}_{variant}",
                      torch.tensor(vals, dtype=torch.float32))
    save_meta(model, dataset, "coe", {"variants": list(scores.keys()), "shape": "N per variant"})
    print(f"    coe: {len(scores)} variants")


def process_seakr(model, dataset, data, labels, split):
    logits = data["gen_logit_stats_last"]
    for s in ("train", "val", "test"):
        idx = split[s]
        sub = [logits[i] for i in idx]
        scores = seakr_energy_score(sub)
        save_feat(model, dataset, "seakr", s, torch.tensor(scores, dtype=torch.float32))
    save_meta(model, dataset, "seakr", {"shape": "N"})
    print(f"    seakr: done")


def process_step(model, dataset, data, labels, split):
    h = data["gen_last_token_hidden"]
    for s in ("train", "val", "test"):
        idx = split[s]
        save_feat(model, dataset, "step", s, h[idx][:, -2, :].float())
    save_meta(model, dataset, "step", {"layer": "last decoder (-2)", "shape": "N x D"})
    print(f"    step: done")


# ============================================================
# Driver
# ============================================================

# Each method maps to which raw feature(s) it needs (so we can lazy-load)
METHOD_DEPS = {
    "lr_probe": ["input_last_token_hidden"],
    "mm_probe": ["input_last_token_hidden"],
    "pca_lr": ["input_last_token_hidden"],
    "iti": ["input_per_head_activation"],
    "kb_mlp": ["input_last_token_hidden"],
    "lid": ["input_last_token_hidden"],
    "attn_satisfies": ["input_attn_value_norms"],
    "llm_check": ["input_attn_stats"],
    "sep": ["gen_last_token_hidden"],
    "coe": ["gen_mean_pool_hidden"],
    "seakr": ["gen_logit_stats_last"],
    "step": ["gen_last_token_hidden"],
}

PROCESSORS = {
    "lr_probe": process_lr_probe,
    "mm_probe": process_mm_probe,
    "pca_lr": process_pca_lr,
    "iti": process_iti,
    "kb_mlp": process_kb_mlp,
    "lid": process_lid,
    "attn_satisfies": process_attn_satisfies,
    "llm_check": process_llm_check,
    "sep": process_sep,
    "coe": process_coe,
    "seakr": process_seakr,
    "step": process_step,
}


def run_one(model, dataset, methods=None, force=False):
    """Per-method on-demand loading with eager release.
    Avoids holding all 7 raw feature tensors in RAM simultaneously (OOM on math/mmlu).
    """
    import gc
    if methods is None:
        methods = list(PROCESSORS.keys())
    print(f"\n{'='*60}\n{model} / {dataset}\n{'='*60}")

    labels, split = load_labels_split(model, dataset)
    print(f"  N_total={len(labels)}  pos_rate={labels.mean():.3f}")
    print(f"  splits: train={len(split['train'])} val={len(split['val'])} test={len(split['test'])}")

    # Plan: order methods so that methods sharing tensors run consecutively.
    # Group by feature dep so we load each tensor at most once.
    DEP_ORDER = [
        "input_last_token_hidden",
        "input_per_head_activation",
        "input_attn_stats",
        "input_attn_value_norms",  # large — release ASAP after attn_satisfies
        "gen_last_token_hidden",
        "gen_mean_pool_hidden",
        "gen_logit_stats_last",
    ]
    DEP_TO_METHODS = {dep: [] for dep in DEP_ORDER}
    for m in methods:
        if force or not is_method_done(model, dataset, m):
            for dep in METHOD_DEPS[m]:
                if m not in DEP_TO_METHODS[dep]:
                    DEP_TO_METHODS[dep].append(m)

    data = {}
    done_methods = set()
    for dep in DEP_ORDER:
        ms = DEP_TO_METHODS[dep]
        ms = [m for m in ms if m not in done_methods]
        if not ms:
            continue
        # Load this dep
        t0 = time.time()
        if dep in ("input_logit_stats", "gen_logit_stats_last"):
            data[dep] = load_json(model, dataset, dep)
        else:
            data[dep] = load_tensor(model, dataset, dep)
        shape_str = getattr(data[dep], 'shape', f'list[{len(data[dep])}]')
        print(f"  loaded {dep}: shape={shape_str}  ({time.time()-t0:.1f}s)")

        # Run all methods that need this dep (and nothing else still pending)
        for m in ms:
            try:
                tm = time.time()
                PROCESSORS[m](model, dataset, data, labels, split)
                print(f"      ({time.time()-tm:.1f}s)")
                done_methods.add(m)
            except Exception as e:
                import traceback
                print(f"    {m}: FAILED: {e}")
                traceback.print_exc()
                done_methods.add(m)  # mark so we don't retry

        # Release this dep if no remaining method needs it
        still_needed = False
        for m in methods:
            if m in done_methods:
                continue
            if dep in METHOD_DEPS[m]:
                still_needed = True
                break
        if not still_needed and dep in data:
            del data[dep]
            gc.collect()
            print(f"  released {dep}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--methods", nargs="+", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    run_one(args.model, args.dataset, args.methods, args.force)


if __name__ == "__main__":
    main()
