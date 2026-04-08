"""
Multi-View Fusion v2: Direct multi-view stacking without view-level bottleneck.

Key change from v1: Instead of compressing each view to a single set of class logits
(view-level meta-LR), we pass ALL per-layer logits from ALL views directly to
the final meta-classifier. This avoids the information bottleneck while still
utilizing all 11 views.

Also adds: per-view and cross-view analysis.
"""

import os, json, time, warnings, gc
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
    "geometry_of_truth_cities": 1.0000, "metatool_task1": 0.9982,
    "retrievalqa": 0.9390, "common_claim_3class": 0.7576,
    "e2h_amc_3class": 0.8934, "e2h_amc_5class": 0.8752,
    "when2call_3class": 0.8741, "fava_binary": 0.9856,
    "ragtruth_binary": 0.8808,
}

OLD_PROBES = ["lr_probe", "pca_lr", "iti", "kb_mlp", "attn_satisfies", "sep", "step"]
N_FOLDS = 5
C_GRID = [1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
C_GRID_META = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 1e-1, 1.0]

# Views: each produces per-layer OOF logits that go DIRECTLY to meta-classifier
RAW_VIEWS = [
    # (view_name, file_name, pca_dim, layer_stride, reshape_type)
    # Representation view
    ("repr_input_last", "input_last_token_hidden", 512, 2, "default"),
    ("repr_input_mean", "input_mean_pool_hidden", 512, 2, "default"),
    ("repr_gen_last", "gen_last_token_hidden", 512, 2, "default"),
    ("repr_gen_mean", "gen_mean_pool_hidden", 512, 2, "default"),
    # Attention view
    ("attn_head_act", "input_per_head_activation", 256, 4, "reshape"),
    ("attn_input_stats", "input_attn_stats", None, 1, "reshape"),
    ("attn_input_vnorms", "input_attn_value_norms", None, 1, "max"),
    ("attn_gen_stats", "gen_attn_stats_last", None, 1, "reshape"),
]

SCALAR_VIEWS = [
    # (view_name, file_name, fields)
    ("conf_input", "input_logit_stats", ["logsumexp", "max_prob", "entropy"]),
    ("conf_gen", "gen_logit_stats_last", ["logsumexp", "max_prob", "entropy"]),
]


def load_labels(ext, split):
    with open(os.path.join(EXTRACTION_DIR, ext, split, "meta.json")) as f:
        return np.array(json.load(f)["labels"])

def compute_auroc(y, p, nc):
    if nc == 2: return roc_auc_score(y, p[:, 1])
    yb = label_binarize(y, classes=list(range(nc)))
    return roc_auc_score(yb, p, average="macro", multi_class="ovr")

def bootstrap_ci(y, p, nc, n_boot=1000):
    n = len(y); rng = np.random.RandomState(42)
    scores = []
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        try: scores.append(compute_auroc(y[idx], p[idx], nc))
        except: pass
    scores = sorted(scores)
    return scores[int(0.025*len(scores))], scores[int(0.975*len(scores))]


def run_dataset(ds_name, info):
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

    all_oof = []  # all per-layer/per-probe OOF logits
    all_te = []
    all_names = []  # track which view each set belongs to
    view_ranges = {}  # view_name → (start_idx, end_idx) in all_oof

    # === RAW VIEWS ===
    for vname, fname, pca_dim, stride, rtype in RAW_VIEWS:
        t0 = time.time()
        tr_path = os.path.join(EXTRACTION_DIR, ext, sp["train"], f"{fname}.pt")
        if not os.path.exists(tr_path):
            continue

        tr_raw = torch.load(tr_path, map_location="cpu").float().numpy()
        va_raw = torch.load(os.path.join(EXTRACTION_DIR, ext, sp["val"], f"{fname}.pt"), map_location="cpu").float().numpy()
        te_raw = torch.load(os.path.join(EXTRACTION_DIR, ext, sp["test"], f"{fname}.pt"), map_location="cpu").float().numpy()
        trva_raw = np.concatenate([tr_raw, va_raw])

        nl = tr_raw.shape[1]
        if rtype == "reshape":
            gl = lambda d, l: d[:, l, :, :].reshape(d.shape[0], -1) if d.ndim == 4 else d[:, l, :]
        elif rtype == "max":
            gl = lambda d, l: d[:, l, :, :].max(axis=-1) if d.ndim == 4 else d[:, l, :]
        else:
            gl = lambda d, l: d[:, l, :]

        layers = list(range(0, nl, max(1, stride)))
        start_idx = len(all_oof)

        for l in layers:
            X_trva = gl(trva_raw, l)
            X_te = gl(te_raw, l)
            if X_trva.ndim == 1: X_trva, X_te = X_trva.reshape(-1,1), X_te.reshape(-1,1)

            sc = StandardScaler()
            Xs = sc.fit_transform(X_trva); Xts = sc.transform(X_te)
            if pca_dim and Xs.shape[1] > pca_dim:
                pca = PCA(n_components=pca_dim, random_state=42)
                Xs = pca.fit_transform(Xs); Xts = pca.transform(Xts)

            # C selection
            sp_cut = int(n_tr * 0.8)
            best_a, best_C = -1, 0.01
            for C in C_GRID:
                clf = LogisticRegression(max_iter=2000, C=C, random_state=42)
                clf.fit(Xs[:sp_cut], tr_labels[:sp_cut])
                vp = clf.predict_proba(Xs[sp_cut:n_tr])
                try: a = compute_auroc(tr_labels[sp_cut:], vp, nc)
                except: a = 0.5
                if a > best_a: best_a, best_C = a, C

            oof = np.zeros((n_trva, nc)); ta = np.zeros((n_te, nc))
            for _, (ti, vi) in enumerate(skf.split(Xs, trva_labels)):
                clf = LogisticRegression(max_iter=2000, C=best_C, random_state=42)
                clf.fit(Xs[ti], trva_labels[ti])
                oof[vi] = clf.predict_proba(Xs[vi])
                ta += clf.predict_proba(Xts) / N_FOLDS

            all_oof.append(oof); all_te.append(ta)
            all_names.append(f"{vname}_L{l}")

        view_ranges[vname] = (start_idx, len(all_oof))
        print(f"    {vname:25s}: {len(layers)} layers [{time.time()-t0:.1f}s]")
        del tr_raw, va_raw, te_raw, trva_raw; gc.collect()

    # === SCALAR VIEWS (confidence) ===
    for vname, fname, fields in SCALAR_VIEWS:
        feat_parts = []
        for split_key in [sp["train"], sp["val"], sp["test"]]:
            fpath = os.path.join(EXTRACTION_DIR, ext, split_key, f"{fname}.json")
            if not os.path.exists(fpath): break
            with open(fpath) as f:
                data = json.load(f)
            arr = np.array([[d.get(fi, 0) for fi in fields] for d in data], dtype=np.float32)
            feat_parts.append(arr)

        if len(feat_parts) == 3:
            X_trva = np.nan_to_num(np.concatenate(feat_parts[:2]), nan=0, posinf=20, neginf=-20)
            X_te = np.nan_to_num(feat_parts[2], nan=0, posinf=20, neginf=-20)

            sc = StandardScaler()
            Xs = sc.fit_transform(X_trva); Xts = sc.transform(X_te)

            sp_cut = int(n_tr * 0.8)
            best_a, best_C = -1, 0.01
            for C in [0.001, 0.01, 0.1, 1.0, 10.0]:
                clf = LogisticRegression(max_iter=2000, C=C, random_state=42)
                clf.fit(Xs[:sp_cut], tr_labels[:sp_cut])
                vp = clf.predict_proba(Xs[sp_cut:n_tr])
                try: a = compute_auroc(tr_labels[sp_cut:], vp, nc)
                except: a = 0.5
                if a > best_a: best_a, best_C = a, C

            start_idx = len(all_oof)
            oof = np.zeros((n_trva, nc)); ta = np.zeros((n_te, nc))
            for _, (ti, vi) in enumerate(skf.split(Xs, trva_labels)):
                clf = LogisticRegression(max_iter=2000, C=best_C, random_state=42)
                clf.fit(Xs[ti], trva_labels[ti])
                oof[vi] = clf.predict_proba(Xs[vi])
                ta += clf.predict_proba(Xts) / N_FOLDS
            all_oof.append(oof); all_te.append(ta)
            all_names.append(vname)
            view_ranges[vname] = (start_idx, len(all_oof))
            print(f"    {vname:25s}: {len(fields)} scalars [0.0s]")

    # === PROBE VIEW ===
    probe_dir = os.path.join(PROCESSED_DIR, ds_name)
    if os.path.exists(probe_dir):
        t0 = time.time()
        start_idx = len(all_oof)
        for method in OLD_PROBES:
            path = os.path.join(probe_dir, method, "train.pt")
            if not os.path.exists(path): continue
            tr = torch.load(path, map_location="cpu").float().numpy()
            va = torch.load(os.path.join(probe_dir, method, "val.pt"), map_location="cpu").float().numpy()
            te = torch.load(os.path.join(probe_dir, method, "test.pt"), map_location="cpu").float().numpy()
            if tr.ndim == 1: tr, va, te = tr.reshape(-1,1), va.reshape(-1,1), te.reshape(-1,1)
            trva = np.vstack([tr, va])
            sc = StandardScaler(); trvas = sc.fit_transform(trva); tes = sc.transform(te)
            if trvas.shape[1] > 256:
                p = PCA(n_components=256, random_state=42)
                trvas = p.fit_transform(trvas); tes = p.transform(tes)

            sp_cut = int(n_tr * 0.8)
            tr_only = sc.transform(tr)
            if tr.shape[1] > 256: tr_only = p.transform(tr_only)
            best_a, best_C = -1, 1.0
            for C in [0.001, 0.01, 0.1, 1.0, 10.0]:
                clf = LogisticRegression(max_iter=2000, C=C, random_state=42)
                clf.fit(tr_only[:sp_cut], tr_labels[:sp_cut])
                vp = clf.predict_proba(tr_only[sp_cut:n_tr])
                try: a = compute_auroc(tr_labels[sp_cut:], vp, nc)
                except: a = 0.5
                if a > best_a: best_a, best_C = a, C

            oof = np.zeros((n_trva, nc)); ta = np.zeros((n_te, nc))
            for _, (ti, vi) in enumerate(skf.split(trvas, trva_labels)):
                clf = LogisticRegression(max_iter=2000, C=best_C, random_state=42)
                clf.fit(trvas[ti], trva_labels[ti])
                oof[vi] = clf.predict_proba(trvas[vi])
                ta += clf.predict_proba(tes) / N_FOLDS
            all_oof.append(oof); all_te.append(ta)
            all_names.append(f"probe_{method}")

        view_ranges["probe_methods"] = (start_idx, len(all_oof))
        print(f"    {'probe_methods':25s}: {len(all_oof)-start_idx} probes [{time.time()-t0:.1f}s]")

    if not all_oof:
        return None

    # === META-CLASSIFIER ===
    meta_oof = np.hstack(all_oof)
    meta_te = np.hstack(all_te)
    print(f"    Total meta-features: {meta_oof.shape[1]}")

    sc_m = StandardScaler()
    mo = sc_m.fit_transform(meta_oof); mt = sc_m.transform(meta_te)

    best_au, best_C = -1, 0.01
    for C in C_GRID_META:
        inner = np.zeros((n_trva, nc))
        for _, (ti, vi) in enumerate(skf.split(mo, trva_labels)):
            clf = LogisticRegression(max_iter=3000, C=C, random_state=42)
            clf.fit(mo[ti], trva_labels[ti])
            inner[vi] = clf.predict_proba(mo[vi])
        au = compute_auroc(trva_labels, inner, nc)
        if au > best_au: best_au, best_C = au, C

    clf_f = LogisticRegression(max_iter=3000, C=best_C, random_state=42)
    clf_f.fit(mo, trva_labels)
    te_prob = clf_f.predict_proba(mt)

    auroc = compute_auroc(te_labels, te_prob, nc)
    ci_lo, ci_hi = bootstrap_ci(te_labels, te_prob, nc)
    acc = accuracy_score(te_labels, te_prob.argmax(1))
    f1 = f1_score(te_labels, te_prob.argmax(1), average="macro")
    bl = BASELINES.get(ds_name, 0)

    # Leave-one-view-out
    loo = {}
    for vname, (si, ei) in view_ranges.items():
        keep_idx = list(range(si)) + list(range(ei, len(all_oof)))
        if not keep_idx: continue
        loo_oof = np.hstack([all_oof[i] for i in keep_idx])
        loo_te = np.hstack([all_te[i] for i in keep_idx])
        sc_l = StandardScaler()
        lo = sc_l.fit_transform(loo_oof); lt = sc_l.transform(loo_te)
        clf_l = LogisticRegression(max_iter=3000, C=best_C, random_state=42)
        clf_l.fit(lo, trva_labels)
        lp = clf_l.predict_proba(lt)
        try: la = compute_auroc(te_labels, lp, nc)
        except: la = 0.5
        loo[vname] = {"auroc_without": float(la), "contribution": float(auroc - la)}

    return {
        "auroc": float(auroc), "ci_lo": float(ci_lo), "ci_hi": float(ci_hi),
        "accuracy": float(acc), "f1_macro": float(f1),
        "delta": float(auroc - bl), "baseline": float(bl),
        "meta_C": best_C, "n_meta_features": meta_oof.shape[1],
        "n_views": len(view_ranges),
        "leave_one_view_out": loo,
    }


def main():
    print("=" * 70)
    print("MULTI-VIEW FUSION v2 — Direct Stacking (no view-level bottleneck)")
    print("=" * 70)

    results = {}
    for ds_name, info in ALL_DATASETS.items():
        print(f"\n{'='*60}\n{ds_name} ({info['n_classes']}-class)\n{'='*60}")
        t0 = time.time()
        r = run_dataset(ds_name, info)
        if r is None: print("  SKIPPED"); continue
        results[ds_name] = r
        print(f"\n  AUROC={r['auroc']:.4f} [{r['ci_lo']:.4f},{r['ci_hi']:.4f}] "
              f"(delta={r['delta']:+.4f}) [{time.time()-t0:.0f}s]")
        print(f"  LOO contributions:")
        for v, l in sorted(r["leave_one_view_out"].items(), key=lambda x: -x[1]["contribution"]):
            print(f"    {v:25s}: {l['contribution']:+.4f}")
        gc.collect()

    with open(os.path.join(RESULTS_DIR, "multiview_v2_results.json"), "w") as f:
        json.dump(results, f, indent=2, default=float)

    print(f"\n{'='*70}\nSUMMARY\n{'='*70}")
    wins = 0
    for ds, r in results.items():
        s = "WIN" if r["delta"] > 0 else "LOSE"
        if r["delta"] > 0: wins += 1
        print(f"  {ds:30s}  bl={r['baseline']:.4f}  ours={r['auroc']:.4f}  "
              f"delta={r['delta']:+.4f}  [{r['ci_lo']:.4f},{r['ci_hi']:.4f}]  [{s}]")
    print(f"\nWin/Loss: {wins}/{len(results)-wins}")
    deltas = [r["delta"] for r in results.values()]
    if len(deltas) >= 5:
        _, p = stats.wilcoxon(deltas, alternative='greater')
        print(f"Wilcoxon p={p:.4f}")


if __name__ == "__main__":
    main()
