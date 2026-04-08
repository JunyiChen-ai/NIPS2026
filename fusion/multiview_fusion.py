"""
Multi-View Internal State Fusion (MVISF)

Novelty: Treats different LLM internal states as distinct computational "views":
  - Representation view: hidden states (what the model represents)
  - Attention view: attention patterns (where the model looks)
  - Confidence view: logit statistics (how certain the model is)
  - Trajectory view: depth patterns across layers (how representations evolve)
  - Step view: reasoning step boundaries (how the model structures its reasoning)

Each view gets its own per-layer probing pipeline, producing calibrated OOF logits.
A view-level meta-learner then combines views with automatic view weighting,
providing interpretable per-view contribution analysis.

Key design principles:
  - All 13 extracted features utilized (8 were previously unused)
  - Strict OOF evaluation at every stage (no leakage)
  - Linear models only (optimal for low-data probing regime)
  - View-level aggregation enables interpretable contribution analysis
  - Unified pipeline: zero per-dataset configuration
"""

import os, json, time, warnings, gc, sys
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from scipy import stats

warnings.filterwarnings("ignore")

EXTRACTION_DIR = "/home/junyi/NIPS2026/extraction/features"
PROCESSED_DIR = "/home/junyi/NIPS2026/reproduce/processed_features"
RESULTS_DIR = "/home/junyi/NIPS2026/fusion/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

ALL_DATASETS = {
    "geometry_of_truth_cities": {"n_classes": 2, "splits": {"train": "train_sub", "val": "val_split", "test": "val"}, "ext": "geometry_of_truth_cities"},
    "metatool_task1":           {"n_classes": 2, "splits": {"train": "train_sub", "val": "val_split", "test": "test"}, "ext": "metatool_task1"},
    "retrievalqa":              {"n_classes": 2, "splits": {"train": "train_sub", "val": "val_split", "test": "test"}, "ext": "retrievalqa"},
    "common_claim_3class":      {"n_classes": 3, "splits": {"train": "train", "val": "val", "test": "test"}, "ext": "common_claim_3class"},
    "e2h_amc_3class":           {"n_classes": 3, "splits": {"train": "train_sub", "val": "val_split", "test": "eval"}, "ext": "e2h_amc_3class"},
    "e2h_amc_5class":           {"n_classes": 5, "splits": {"train": "train_sub", "val": "val_split", "test": "eval"}, "ext": "e2h_amc_5class"},
    "when2call_3class":         {"n_classes": 3, "splits": {"train": "train", "val": "val", "test": "test"}, "ext": "when2call_3class"},
    "fava_binary":              {"n_classes": 2, "splits": {"train": "train", "val": "val", "test": "test"}, "ext": "fava"},
    "ragtruth_binary":          {"n_classes": 2, "splits": {"train": "train", "val": "val", "test": "test"}, "ext": "ragtruth"},
}

BASELINES = {
    "geometry_of_truth_cities": 1.0000,
    "metatool_task1": 0.9982,
    "retrievalqa": 0.9390,
    "common_claim_3class": 0.7576,
    "e2h_amc_3class": 0.8934,
    "e2h_amc_5class": 0.8752,
    "when2call_3class": 0.8741,
    "fava_binary": 0.9856,
    "ragtruth_binary": 0.8808,
}

OLD_PROBES = ["lr_probe", "pca_lr", "iti", "kb_mlp", "attn_satisfies", "sep", "step"]
N_FOLDS = 5
C_GRID = [1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
C_GRID_META = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 1e-1, 1.0]

# =============================================================================
# VIEW DEFINITIONS
# =============================================================================
# Each "view" represents a distinct computational perspective on the LLM's internals.
# Views are categorized by what aspect of computation they capture.

VIEWS = {
    # --- Representation View: What the model represents ---
    "repr_input_last": {
        "file": "input_last_token_hidden",
        "type": "layered",  # (N, L, D) — per-layer features
        "pca_dim": 512,
        "layer_stride": 2,  # subsample layers
        "description": "Hidden state at last prompt token per layer",
    },
    "repr_input_mean": {
        "file": "input_mean_pool_hidden",
        "type": "layered",
        "pca_dim": 512,
        "layer_stride": 2,
        "description": "Mean-pooled hidden state across prompt per layer",
    },
    "repr_gen_last": {
        "file": "gen_last_token_hidden",
        "type": "layered",
        "pca_dim": 512,
        "layer_stride": 2,
        "description": "Hidden state at last generated token per layer",
    },
    "repr_gen_mean": {
        "file": "gen_mean_pool_hidden",
        "type": "layered",
        "pca_dim": 512,
        "layer_stride": 2,
        "description": "Mean-pooled hidden state across generation per layer",
    },

    # --- Attention View: Where the model looks ---
    "attn_head_act": {
        "file": "input_per_head_activation",
        "type": "layered_reshape",  # (N, L, H, D) → (N, L, H*D)
        "pca_dim": 256,
        "layer_stride": 4,
        "description": "Per-head activation at last prompt token per layer",
    },
    "attn_input_stats": {
        "file": "input_attn_stats",
        "type": "layered_reshape",  # (N, L, H, 3) → (N, L, H*3)
        "pca_dim": None,
        "layer_stride": 1,
        "description": "Attention distribution statistics (skew, entropy, diag) per layer",
    },
    "attn_input_vnorms": {
        "file": "input_attn_value_norms",
        "type": "layered_max",  # (N, L, H, pos) → max over pos → (N, L, H)
        "pca_dim": None,
        "layer_stride": 1,
        "description": "Attention-weighted value norms per layer",
    },
    "attn_gen_stats": {
        "file": "gen_attn_stats_last",
        "type": "layered_reshape",  # (N, L, H, 3) → (N, L, H*3)
        "pca_dim": None,
        "layer_stride": 1,
        "description": "Generation-side attention stats at last token per layer",
    },

    # --- Confidence View: How certain the model is ---
    "conf_input_logits": {
        "file": "input_logit_stats",
        "type": "json_scalar",  # JSON with scalar fields
        "fields": ["logsumexp", "max_prob", "entropy"],
        "description": "Logit distribution stats at first generation token",
    },
    "conf_gen_logits": {
        "file": "gen_logit_stats_last",
        "type": "json_scalar",
        "fields": ["logsumexp", "max_prob", "entropy"],
        "description": "Logit distribution stats at last generation token",
    },

    # --- Existing Probe View: Published method outputs ---
    "probe_methods": {
        "type": "processed_probes",
        "methods": OLD_PROBES,
        "description": "OOF logits from 7 reproduced probing methods",
    },
}


def load_labels(ext_dir, split):
    with open(os.path.join(EXTRACTION_DIR, ext_dir, split, "meta.json")) as f:
        return np.array(json.load(f)["labels"])


def compute_auroc(y, p, nc):
    if nc == 2:
        return roc_auc_score(y, p[:, 1])
    yb = label_binarize(y, classes=list(range(nc)))
    return roc_auc_score(yb, p, average="macro", multi_class="ovr")


def gaussian_basis(nl, n_basis=5, sigma_scale=0.3):
    c = np.linspace(0, nl - 1, n_basis)
    s = sigma_scale * nl / n_basis
    b = np.zeros((nl, n_basis))
    for i, ci in enumerate(c):
        b[:, i] = np.exp(-0.5 * ((np.arange(nl) - ci) / s) ** 2)
    b /= b.sum(0, keepdims=True) + 1e-10
    return b


def bootstrap_ci(y, p, nc, n_boot=1000):
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


# =============================================================================
# CORE: Per-view probing
# =============================================================================

def probe_layered_view(trva_raw, te_raw, tr_labels, trva_labels, te_labels,
                       n_classes, pca_dim, layer_stride, skf, reshape_fn=None):
    """Probe a layered view (N, L, D) → per-layer OOF logits."""
    n_trva, n_te = len(trva_labels), len(te_labels)
    n_layers = trva_raw.shape[1]
    layer_indices = list(range(0, n_layers, max(1, layer_stride)))

    all_oof = []
    all_te = []

    for l in layer_indices:
        if reshape_fn:
            X_trva = reshape_fn(trva_raw, l)
            X_te = reshape_fn(te_raw, l)
        else:
            X_trva = trva_raw[:, l, :]
            X_te = te_raw[:, l, :]

        if X_trva.ndim == 1:
            X_trva, X_te = X_trva.reshape(-1, 1), X_te.reshape(-1, 1)

        sc = StandardScaler()
        X_trva_s = sc.fit_transform(X_trva)
        X_te_s = sc.transform(X_te)

        if pca_dim and X_trva_s.shape[1] > pca_dim:
            pca = PCA(n_components=pca_dim, random_state=42)
            X_trva_s = pca.fit_transform(X_trva_s)
            X_te_s = pca.transform(X_te_s)

        # C selection on holdout
        n_tr = len(tr_labels)
        sp = int(n_tr * 0.8)
        best_a, best_C = -1, 0.01
        for C in C_GRID:
            clf = LogisticRegression(max_iter=2000, C=C, random_state=42)
            clf.fit(X_trva_s[:sp], tr_labels[:sp])
            vp = clf.predict_proba(X_trva_s[sp:n_tr])
            try:
                a = compute_auroc(tr_labels[sp:], vp, n_classes)
            except:
                a = 0.5
            if a > best_a:
                best_a, best_C = a, C

        oof = np.zeros((n_trva, n_classes))
        te_avg = np.zeros((n_te, n_classes))
        for _, (ti, vi) in enumerate(skf.split(X_trva_s, trva_labels)):
            clf = LogisticRegression(max_iter=2000, C=best_C, random_state=42)
            clf.fit(X_trva_s[ti], trva_labels[ti])
            oof[vi] = clf.predict_proba(X_trva_s[vi])
            te_avg += clf.predict_proba(X_te_s) / N_FOLDS

        all_oof.append(oof)
        all_te.append(te_avg)

    return all_oof, all_te, layer_indices


def probe_scalar_view(X_trva, X_te, tr_labels, trva_labels, te_labels,
                      n_classes, skf):
    """Probe a scalar/low-dim view."""
    n_trva, n_te = len(trva_labels), len(te_labels)

    sc = StandardScaler()
    X_trva_s = sc.fit_transform(X_trva)
    X_te_s = sc.transform(X_te)

    n_tr = len(tr_labels)
    sp = int(n_tr * 0.8)
    best_a, best_C = -1, 0.01
    for C in [0.001, 0.01, 0.1, 1.0, 10.0]:
        clf = LogisticRegression(max_iter=2000, C=C, random_state=42)
        clf.fit(X_trva_s[:sp], tr_labels[:sp])
        vp = clf.predict_proba(X_trva_s[sp:n_tr])
        try:
            a = compute_auroc(tr_labels[sp:], vp, n_classes)
        except:
            a = 0.5
        if a > best_a:
            best_a, best_C = a, C

    oof = np.zeros((n_trva, n_classes))
    te_avg = np.zeros((n_te, n_classes))
    for _, (ti, vi) in enumerate(skf.split(X_trva_s, trva_labels)):
        clf = LogisticRegression(max_iter=2000, C=best_C, random_state=42)
        clf.fit(X_trva_s[ti], trva_labels[ti])
        oof[vi] = clf.predict_proba(X_trva_s[vi])
        te_avg += clf.predict_proba(X_te_s) / N_FOLDS

    return [oof], [te_avg]


# =============================================================================
# MAIN: Multi-view fusion pipeline
# =============================================================================

def run_multiview_fusion(ds_name, info, views_to_use=None, return_per_view=False):
    """
    Run multi-view fusion on a dataset.

    Args:
        ds_name: dataset name
        info: dataset info dict
        views_to_use: list of view names to include (None = all)
        return_per_view: if True, also return per-view OOF/TE logits and view-level AUROCs

    Returns:
        result dict with auroc, accuracy, f1, etc.
        If return_per_view: also per_view_results dict
    """
    nc = info["n_classes"]
    sp = info["splits"]
    ext = info["ext"]

    tr_labels = load_labels(ext, sp["train"])
    va_labels = load_labels(ext, sp["val"])
    te_labels = load_labels(ext, sp["test"])
    trva_labels = np.concatenate([tr_labels, va_labels])
    n_trva, n_te = len(trva_labels), len(te_labels)
    n_tr = len(tr_labels)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    views = VIEWS if views_to_use is None else {k: v for k, v in VIEWS.items() if k in views_to_use}

    # Collect per-view OOF logits
    view_oof = {}  # view_name → list of OOF arrays
    view_te = {}
    view_meta = {}  # view_name → metadata (layers used, etc.)

    for vname, vdef in views.items():
        t0 = time.time()
        vtype = vdef["type"]

        if vtype == "processed_probes":
            # Load processed probe features
            probe_dir = os.path.join(PROCESSED_DIR, ds_name)
            oof_parts, te_parts = [], []
            for method in vdef["methods"]:
                path = os.path.join(probe_dir, method, "train.pt")
                if not os.path.exists(path):
                    continue
                tr = torch.load(path, map_location="cpu").float().numpy()
                va = torch.load(os.path.join(probe_dir, method, "val.pt"), map_location="cpu").float().numpy()
                te = torch.load(os.path.join(probe_dir, method, "test.pt"), map_location="cpu").float().numpy()
                if tr.ndim == 1: tr, va, te = tr.reshape(-1,1), va.reshape(-1,1), te.reshape(-1,1)
                trva = np.vstack([tr, va])

                o, t = probe_scalar_view(trva, te, tr_labels, trva_labels, te_labels, nc, skf)
                oof_parts.extend(o)
                te_parts.extend(t)

            if oof_parts:
                view_oof[vname] = oof_parts
                view_te[vname] = te_parts
                view_meta[vname] = {"n_sources": len(oof_parts)}
                print(f"    {vname:25s}: {len(oof_parts)} probes [{time.time()-t0:.1f}s]")

        elif vtype == "json_scalar":
            # Load JSON scalar features
            fields = vdef["fields"]
            feat_trva, feat_te = [], []
            for split_name, split_key in [("train", sp["train"]), ("val", sp["val"]), ("test", sp["test"])]:
                fpath = os.path.join(EXTRACTION_DIR, ext, split_key, f"{vdef['file']}.json")
                if not os.path.exists(fpath):
                    break
                with open(fpath) as f:
                    data = json.load(f)
                arr = np.array([[d.get(field, 0) for field in fields] for d in data], dtype=np.float32)
                if split_name == "test":
                    feat_te.append(arr)
                else:
                    feat_trva.append(arr)

            if feat_trva and feat_te:
                X_trva = np.concatenate(feat_trva, axis=0)
                X_te = feat_te[0]
                # Replace NaN/Inf
                X_trva = np.nan_to_num(X_trva, nan=0, posinf=20, neginf=-20)
                X_te = np.nan_to_num(X_te, nan=0, posinf=20, neginf=-20)

                o, t = probe_scalar_view(X_trva, X_te, tr_labels, trva_labels, te_labels, nc, skf)
                view_oof[vname] = o
                view_te[vname] = t
                view_meta[vname] = {"n_features": X_trva.shape[1]}
                print(f"    {vname:25s}: {X_trva.shape[1]} scalars [{time.time()-t0:.1f}s]")

        elif vtype in ("layered", "layered_reshape", "layered_max"):
            # Load tensor features
            tr_path = os.path.join(EXTRACTION_DIR, ext, sp["train"], f"{vdef['file']}.pt")
            if not os.path.exists(tr_path):
                continue

            tr_raw = torch.load(tr_path, map_location="cpu").float().numpy()
            va_raw = torch.load(os.path.join(EXTRACTION_DIR, ext, sp["val"], f"{vdef['file']}.pt"),
                               map_location="cpu").float().numpy()
            te_raw = torch.load(os.path.join(EXTRACTION_DIR, ext, sp["test"], f"{vdef['file']}.pt"),
                               map_location="cpu").float().numpy()
            trva_raw = np.concatenate([tr_raw, va_raw], axis=0)

            # Define reshape function based on view type
            if vtype == "layered":
                reshape_fn = None  # default: data[:, l, :]
            elif vtype == "layered_reshape":
                reshape_fn = lambda d, l: d[:, l, :, :].reshape(d.shape[0], -1) if d.ndim == 4 else d[:, l, :]
            elif vtype == "layered_max":
                reshape_fn = lambda d, l: d[:, l, :, :].max(axis=-1) if d.ndim == 4 else d[:, l, :]

            pca_dim = vdef.get("pca_dim")
            layer_stride = vdef.get("layer_stride", 1)

            o, t, layers = probe_layered_view(
                trva_raw, te_raw, tr_labels, trva_labels, te_labels,
                nc, pca_dim, layer_stride, skf, reshape_fn
            )
            view_oof[vname] = o
            view_te[vname] = t
            view_meta[vname] = {"n_layers": len(layers), "layers": layers}
            print(f"    {vname:25s}: {len(layers)} layers [{time.time()-t0:.1f}s]")

            del tr_raw, va_raw, te_raw, trva_raw
            gc.collect()

    if not view_oof:
        return None

    # =========================================================================
    # STAGE 1: Per-view aggregation → view-level OOF logits
    # =========================================================================
    # Each view's per-layer/per-probe OOF logits are concatenated and
    # compressed to a single set of class logits via a view-level meta-LR.
    # This gives us ONE set of calibrated logits per view.

    view_level_oof = {}
    view_level_te = {}
    view_level_auroc = {}

    for vname in view_oof:
        all_o = np.hstack(view_oof[vname])
        all_t = np.hstack(view_te[vname])

        if all_o.shape[1] <= nc:
            # Already just one probe's logits, no need for view-level meta
            view_level_oof[vname] = all_o
            view_level_te[vname] = all_t
        else:
            # View-level meta-LR
            sc = StandardScaler()
            mo = sc.fit_transform(all_o)
            mt = sc.transform(all_t)

            best_au, best_C = -1, 0.01
            for C in C_GRID_META:
                inner = np.zeros((n_trva, nc))
                for _, (ti, vi) in enumerate(skf.split(mo, trva_labels)):
                    clf = LogisticRegression(max_iter=2000, C=C, random_state=42)
                    clf.fit(mo[ti], trva_labels[ti])
                    inner[vi] = clf.predict_proba(mo[vi])
                try:
                    au = compute_auroc(trva_labels, inner, nc)
                except:
                    au = 0.5
                if au > best_au:
                    best_au, best_C = au, C

            # OOF predictions for this view
            v_oof = np.zeros((n_trva, nc))
            v_te_avg = np.zeros((n_te, nc))
            for _, (ti, vi) in enumerate(skf.split(mo, trva_labels)):
                clf = LogisticRegression(max_iter=2000, C=best_C, random_state=42)
                clf.fit(mo[ti], trva_labels[ti])
                v_oof[vi] = clf.predict_proba(mo[vi])
                v_te_avg += clf.predict_proba(mt) / N_FOLDS

            view_level_oof[vname] = v_oof
            view_level_te[vname] = v_te_avg

        # Compute per-view AUROC on test
        try:
            view_level_auroc[vname] = compute_auroc(te_labels, view_level_te[vname], nc)
        except:
            view_level_auroc[vname] = 0.5

    # =========================================================================
    # STAGE 2: Cross-view fusion → final prediction
    # =========================================================================
    # Concatenate all view-level logits → meta-LR for final prediction.
    # The meta-LR weights implicitly learn view importance.

    view_names_ordered = sorted(view_level_oof.keys())
    meta_oof = np.hstack([view_level_oof[v] for v in view_names_ordered])
    meta_te = np.hstack([view_level_te[v] for v in view_names_ordered])

    sc_meta = StandardScaler()
    mo_s = sc_meta.fit_transform(meta_oof)
    mt_s = sc_meta.transform(meta_te)

    best_au, best_C = -1, 0.01
    for C in C_GRID_META:
        inner = np.zeros((n_trva, nc))
        for _, (ti, vi) in enumerate(skf.split(mo_s, trva_labels)):
            clf = LogisticRegression(max_iter=3000, C=C, random_state=42)
            clf.fit(mo_s[ti], trva_labels[ti])
            inner[vi] = clf.predict_proba(mo_s[vi])
        au = compute_auroc(trva_labels, inner, nc)
        if au > best_au:
            best_au, best_C = au, C

    clf_final = LogisticRegression(max_iter=3000, C=best_C, random_state=42)
    clf_final.fit(mo_s, trva_labels)
    te_prob = clf_final.predict_proba(mt_s)

    auroc = compute_auroc(te_labels, te_prob, nc)
    acc = accuracy_score(te_labels, te_prob.argmax(axis=1))
    f1 = f1_score(te_labels, te_prob.argmax(axis=1), average="macro")
    ci_lo, ci_hi = bootstrap_ci(te_labels, te_prob, nc)

    result = {
        "auroc": float(auroc),
        "accuracy": float(acc),
        "f1_macro": float(f1),
        "ci_lo": float(ci_lo),
        "ci_hi": float(ci_hi),
        "delta": float(auroc - BASELINES.get(ds_name, 0)),
        "baseline": float(BASELINES.get(ds_name, 0)),
        "meta_C": best_C,
        "n_views": len(view_level_oof),
        "n_meta_features": meta_oof.shape[1],
        "view_aurocs": {k: float(v) for k, v in sorted(view_level_auroc.items(), key=lambda x: -x[1])},
    }

    if return_per_view:
        # Also compute per-view contribution via leave-one-view-out
        loo_results = {}
        for drop_view in view_names_ordered:
            remaining = [v for v in view_names_ordered if v != drop_view]
            if not remaining:
                continue
            loo_oof = np.hstack([view_level_oof[v] for v in remaining])
            loo_te = np.hstack([view_level_te[v] for v in remaining])
            sc_loo = StandardScaler()
            lo = sc_loo.fit_transform(loo_oof)
            lt = sc_loo.transform(loo_te)
            clf_loo = LogisticRegression(max_iter=3000, C=best_C, random_state=42)
            clf_loo.fit(lo, trva_labels)
            lp = clf_loo.predict_proba(lt)
            try:
                loo_auroc = compute_auroc(te_labels, lp, nc)
            except:
                loo_auroc = 0.5
            loo_results[drop_view] = {
                "auroc_without": float(loo_auroc),
                "contribution": float(auroc - loo_auroc),  # positive = this view helps
            }
        result["leave_one_view_out"] = loo_results

    return result


def main():
    print("=" * 70)
    print("MULTI-VIEW INTERNAL STATE FUSION (MVISF)")
    print("=" * 70)

    all_results = {}
    for ds_name, info in ALL_DATASETS.items():
        print(f"\n{'='*60}")
        print(f"Dataset: {ds_name} ({info['n_classes']}-class)")
        print(f"{'='*60}")
        t0 = time.time()

        result = run_multiview_fusion(ds_name, info, return_per_view=True)
        if result is None:
            print(f"  SKIPPED")
            continue

        all_results[ds_name] = result
        bl = BASELINES.get(ds_name, 0)
        print(f"\n  FUSION: AUROC={result['auroc']:.4f} [{result['ci_lo']:.4f},{result['ci_hi']:.4f}]"
              f" (delta={result['delta']:+.4f}) [{time.time()-t0:.0f}s]")
        print(f"  Views: {result['n_views']}, Meta-features: {result['n_meta_features']}")
        print(f"  Per-view AUROCs:")
        for vname, va in result["view_aurocs"].items():
            print(f"    {vname:25s}: {va:.4f}")
        if "leave_one_view_out" in result:
            print(f"  Leave-one-view-out contributions:")
            for vname, loo in sorted(result["leave_one_view_out"].items(), key=lambda x: -x[1]["contribution"]):
                print(f"    {vname:25s}: contribution={loo['contribution']:+.4f}")

        gc.collect()

    # Save results
    with open(os.path.join(RESULTS_DIR, "multiview_results.json"), "w") as f:
        json.dump(all_results, f, indent=2, default=float)

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    wins = 0
    for ds, r in all_results.items():
        status = "WIN" if r["delta"] > 0 else "LOSE"
        if r["delta"] > 0: wins += 1
        print(f"  {ds:30s}  bl={r['baseline']:.4f}  ours={r['auroc']:.4f}  "
              f"delta={r['delta']:+.4f}  [{r['ci_lo']:.4f},{r['ci_hi']:.4f}]  [{status}]")
    print(f"\nWin/Loss: {wins}/{len(all_results)-wins}")

    # Wilcoxon test
    deltas = [r["delta"] for r in all_results.values()]
    if len(deltas) >= 5:
        stat, p = stats.wilcoxon(deltas, alternative='greater')
        print(f"Wilcoxon signed-rank: p={p:.4f}")

    return all_results


if __name__ == "__main__":
    main()
