"""
Minimal ablation: only use already-computed OOF logits from the comprehensive results.
For fast ablation, compare:
  full = already have results
  probes_only = quick (no raw features)
  We cache per-layer results from a single full run, then ablate components.
"""
import os, json, time, warnings, gc
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

EXTRACTION_DIR = "/home/junyi/NIPS2026/extraction/features"
PROCESSED_DIR = "/home/junyi/NIPS2026/reproduce/processed_features"
RESULTS_DIR = "/home/junyi/NIPS2026/fusion/results"

DATASETS = {
    "common_claim_3class": {"n_classes": 3, "splits": {"train": "train", "val": "val", "test": "test"}},
    "e2h_amc_3class": {"n_classes": 3, "splits": {"train": "train_sub", "val": "val_split", "test": "eval"}},
    "e2h_amc_5class": {"n_classes": 5, "splits": {"train": "train_sub", "val": "val_split", "test": "eval"}},
    "when2call_3class": {"n_classes": 3, "splits": {"train": "train", "val": "val", "test": "test"}},
}

SOURCES = [
    ("input_hidden", "input_last_token_hidden", 512),
    ("gen_hidden", "gen_last_token_hidden", 512),
    ("head_act", "input_per_head_activation", 256),
    ("attn_stats", "input_attn_stats", None),
    ("attn_vnorms", "input_attn_value_norms", 256),
]
OLD_PROBES = ["lr_probe", "pca_lr", "iti", "kb_mlp", "attn_satisfies", "sep", "step"]
N_FOLDS = 5
C_GRID = [1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
C_GRID_META = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 1e-1, 1.0]


def load_labels(ds, split):
    with open(os.path.join(EXTRACTION_DIR, ds, split, "meta.json")) as f:
        return np.array(json.load(f)["labels"])


def compute_auroc(y, p, nc):
    if nc == 2: return roc_auc_score(y, p[:, 1])
    yb = label_binarize(y, classes=list(range(nc)))
    return roc_auc_score(yb, p, average="macro", multi_class="ovr")


def gaussian_basis(nl, nb=5, n_basis=None, ss=0.3):
    if n_basis is not None: nb = n_basis
    c = np.linspace(0, nl-1, nb)
    s = ss * nl / nb
    b = np.zeros((nl, nb))
    for i, ci in enumerate(c):
        b[:, i] = np.exp(-0.5*((np.arange(nl)-ci)/s)**2)
    b /= b.sum(0, keepdims=True)+1e-10
    return b


def build_layer_cache(ds_name, info):
    """Build per-layer OOF logits cache (expensive, done once)."""
    nc = info["n_classes"]
    sp = info["splits"]
    tr_labels = load_labels(ds_name, sp["train"])
    va_labels = load_labels(ds_name, sp["val"])
    te_labels = load_labels(ds_name, sp["test"])
    trva_labels = np.concatenate([tr_labels, va_labels])
    n_trva, n_te = len(trva_labels), len(te_labels)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    cache = {"source_layer_oof": {}, "source_layer_te": {},
             "trva_labels": trva_labels, "te_labels": te_labels,
             "tr_labels": tr_labels, "n_classes": nc}

    for sname, raw_name, pca_dim in SOURCES:
        tr_path = os.path.join(EXTRACTION_DIR, ds_name, sp["train"], f"{raw_name}.pt")
        if not os.path.exists(tr_path): continue

        tr_raw = torch.load(tr_path, map_location="cpu").float().numpy()
        va_raw = torch.load(os.path.join(EXTRACTION_DIR, ds_name, sp["val"], f"{raw_name}.pt"), map_location="cpu").float().numpy()
        te_raw = torch.load(os.path.join(EXTRACTION_DIR, ds_name, sp["test"], f"{raw_name}.pt"), map_location="cpu").float().numpy()
        trva_raw = np.concatenate([tr_raw, va_raw])

        if raw_name in ["input_last_token_hidden", "gen_last_token_hidden"]:
            nl = tr_raw.shape[1]; gl = lambda d, l: d[:, l, :]
        elif raw_name == "input_per_head_activation":
            nl = tr_raw.shape[1]; gl = lambda d, l: d[:, l, :, :].reshape(d.shape[0], -1)
        elif raw_name == "input_attn_stats":
            nl = tr_raw.shape[1]; gl = lambda d, l: d[:, l, :, :].reshape(d.shape[0], -1)
        elif raw_name == "input_attn_value_norms":
            nl = tr_raw.shape[1]; gl = lambda d, l: d[:, l, :, :].max(axis=-1)
        else: del tr_raw, va_raw, te_raw, trva_raw; continue

        if raw_name == "input_per_head_activation":
            layers = list(range(0, nl, 4))
        elif nl > 20:
            layers = list(range(0, nl, 2))
        else:
            layers = list(range(nl))

        cache["source_layer_oof"][sname] = {}
        cache["source_layer_te"][sname] = {}

        for l in layers:
            X_trva = gl(trva_raw, l)
            X_te = gl(te_raw, l)
            if X_trva.ndim == 1: X_trva, X_te = X_trva.reshape(-1,1), X_te.reshape(-1,1)
            sc = StandardScaler()
            X_trva_s = sc.fit_transform(X_trva)
            X_te_s = sc.transform(X_te)
            if pca_dim and X_trva_s.shape[1] > pca_dim:
                pca = PCA(n_components=pca_dim, random_state=42)
                X_trva_s = pca.fit_transform(X_trva_s)
                X_te_s = pca.transform(X_te_s)

            n_tr = len(tr_labels); sp_cut = int(n_tr*0.8)
            ld = gl(tr_raw, l)
            if ld.ndim == 1: ld = ld.reshape(-1,1)
            to = sc.transform(ld)
            if pca_dim and to.shape[1] > pca_dim: to = pca.transform(to)

            best_a, best_C = -1, 0.01
            for C in C_GRID:
                clf = LogisticRegression(max_iter=2000, C=C, random_state=42)
                clf.fit(to[:sp_cut], tr_labels[:sp_cut])
                vp = clf.predict_proba(to[sp_cut:n_tr])
                a = compute_auroc(tr_labels[sp_cut:], vp, nc)
                if a > best_a: best_a, best_C = a, C

            oof = np.zeros((n_trva, nc))
            te_avg = np.zeros((n_te, nc))
            for _, (ti, vi) in enumerate(skf.split(X_trva_s, trva_labels)):
                clf = LogisticRegression(max_iter=2000, C=best_C, random_state=42)
                clf.fit(X_trva_s[ti], trva_labels[ti])
                oof[vi] = clf.predict_proba(X_trva_s[vi])
                te_avg += clf.predict_proba(X_te_s) / N_FOLDS

            cache["source_layer_oof"][sname][l] = oof
            cache["source_layer_te"][sname][l] = te_avg

        print(f"  {sname}: {len(layers)} layers cached")
        del tr_raw, va_raw, te_raw, trva_raw; gc.collect()

    # Probe logits
    cache["probe_oof"] = {}
    cache["probe_te"] = {}
    for method in OLD_PROBES:
        path = os.path.join(PROCESSED_DIR, ds_name, method, "train.pt")
        if not os.path.exists(path): continue
        tr = torch.load(path, map_location="cpu").float().numpy()
        va = torch.load(os.path.join(PROCESSED_DIR, ds_name, method, "val.pt"), map_location="cpu").float().numpy()
        te = torch.load(os.path.join(PROCESSED_DIR, ds_name, method, "test.pt"), map_location="cpu").float().numpy()
        if tr.ndim == 1: tr, va, te = tr.reshape(-1,1), va.reshape(-1,1), te.reshape(-1,1)
        trva = np.vstack([tr, va])
        sc = StandardScaler(); trva_s = sc.fit_transform(trva); te_s = sc.transform(te)
        if trva_s.shape[1] > 256:
            p = PCA(n_components=256, random_state=42)
            trva_s = p.fit_transform(trva_s); te_s = p.transform(te_s)

        n_tr = len(tr_labels); sp_cut = int(n_tr*0.8)
        tr_only = sc.transform(tr)
        if tr.shape[1] > 256: tr_only = p.transform(tr_only)
        best_a, best_C = -1, 1.0
        for C in [0.001, 0.01, 0.1, 1.0, 10.0]:
            clf = LogisticRegression(max_iter=2000, C=C, random_state=42)
            clf.fit(tr_only[:sp_cut], tr_labels[:sp_cut])
            vp = clf.predict_proba(tr_only[sp_cut:n_tr])
            a = compute_auroc(tr_labels[sp_cut:], vp, nc)
            if a > best_a: best_a, best_C = a, C

        oof = np.zeros((n_trva, nc)); te_agg = np.zeros((n_te, nc))
        for _, (ti, vi) in enumerate(skf.split(trva_s, trva_labels)):
            clf = LogisticRegression(max_iter=2000, C=best_C, random_state=42)
            clf.fit(trva_s[ti], trva_labels[ti])
            oof[vi] = clf.predict_proba(trva_s[vi])
            te_agg += clf.predict_proba(te_s) / N_FOLDS
        cache["probe_oof"][method] = oof
        cache["probe_te"][method] = te_agg

    print(f"  {len(cache['probe_oof'])} probes cached")
    return cache


def assemble_and_evaluate(cache, source_names=None, include_traj=True,
                          include_direct=True, include_probes=True):
    """Assemble meta-features from cache and evaluate. Fast (seconds)."""
    nc = cache["n_classes"]
    trva_labels = cache["trva_labels"]
    te_labels = cache["te_labels"]
    n_trva, n_te = len(trva_labels), len(te_labels)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    all_oof, all_te = [], []

    # Direct per-layer logits
    if include_direct:
        for sname in cache["source_layer_oof"]:
            if source_names and sname not in source_names: continue
            for l in sorted(cache["source_layer_oof"][sname]):
                all_oof.append(cache["source_layer_oof"][sname][l])
                all_te.append(cache["source_layer_te"][sname][l])

    # Trajectory
    if include_traj:
        for sname in cache["source_layer_oof"]:
            if source_names and sname not in source_names: continue
            layers = sorted(cache["source_layer_oof"][sname].keys())
            if len(layers) < 3: continue
            oof_stack = np.stack([cache["source_layer_oof"][sname][l] for l in layers], axis=1)
            te_stack = np.stack([cache["source_layer_te"][sname][l] for l in layers], axis=1)
            basis = gaussian_basis(len(layers), n_basis=min(7, len(layers)))
            for c in range(nc):
                to, tt = oof_stack[:, :, c], te_stack[:, :, c]
                all_oof.append(to @ basis); all_te.append(tt @ basis)
                for fn in [np.mean, np.max, np.std]:
                    all_oof.append(fn(to, axis=1, keepdims=True))
                    all_te.append(fn(tt, axis=1, keepdims=True))
                all_oof.append(to.argmax(1, keepdims=True).astype(float)/len(layers))
                all_te.append(tt.argmax(1, keepdims=True).astype(float)/len(layers))

    # Probes
    if include_probes:
        for method in cache["probe_oof"]:
            all_oof.append(cache["probe_oof"][method])
            all_te.append(cache["probe_te"][method])

    if not all_oof:
        return None

    meta_oof = np.hstack(all_oof)
    meta_te = np.hstack(all_te)

    sc = StandardScaler()
    mo = sc.fit_transform(meta_oof)
    mt = sc.transform(meta_te)

    best_au, best_C = -1, 0.01
    for C in C_GRID_META:
        inner = np.zeros((n_trva, nc))
        for _, (ti, vi) in enumerate(skf.split(mo, trva_labels)):
            clf = LogisticRegression(max_iter=3000, C=C, random_state=42)
            clf.fit(mo[ti], trva_labels[ti])
            inner[vi] = clf.predict_proba(mo[vi])
        au = compute_auroc(trva_labels, inner, nc)
        if au > best_au: best_au, best_C = au, C

    clf = LogisticRegression(max_iter=3000, C=best_C, random_state=42)
    clf.fit(mo, trva_labels)
    te_prob = clf.predict_proba(mt)
    auroc = compute_auroc(te_labels, te_prob, nc)
    return {"auroc": float(auroc), "n_features": meta_oof.shape[1], "meta_C": best_C}


def main():
    ablation_configs = {
        "full":          dict(source_names=None, include_traj=True, include_direct=True, include_probes=True),
        "no_trajectory": dict(source_names=None, include_traj=False, include_direct=True, include_probes=True),
        "no_probes":     dict(source_names=None, include_traj=True, include_direct=True, include_probes=False),
        "no_direct":     dict(source_names=None, include_traj=True, include_direct=False, include_probes=True),
        "probes_only":   dict(source_names=None, include_traj=False, include_direct=False, include_probes=True),
        "traj_only":     dict(source_names=None, include_traj=True, include_direct=False, include_probes=False),
        "direct_only":   dict(source_names=None, include_traj=False, include_direct=True, include_probes=False),
        # Drop one source
        "drop_input_hidden": dict(source_names=["gen_hidden","head_act","attn_stats","attn_vnorms"], include_traj=True, include_direct=True, include_probes=True),
        "drop_gen_hidden":   dict(source_names=["input_hidden","head_act","attn_stats","attn_vnorms"], include_traj=True, include_direct=True, include_probes=True),
        "drop_head_act":     dict(source_names=["input_hidden","gen_hidden","attn_stats","attn_vnorms"], include_traj=True, include_direct=True, include_probes=True),
        "drop_attn":         dict(source_names=["input_hidden","gen_hidden","head_act"], include_traj=True, include_direct=True, include_probes=True),
        # Single source
        "input_hidden_only": dict(source_names=["input_hidden"], include_traj=True, include_direct=True, include_probes=False),
        "gen_hidden_only":   dict(source_names=["gen_hidden"], include_traj=True, include_direct=True, include_probes=False),
    }

    results = {}
    for ds_name, info in DATASETS.items():
        print(f"\n{'='*60}\nCaching {ds_name}...\n{'='*60}")
        t0 = time.time()
        cache = build_layer_cache(ds_name, info)
        print(f"Cache built in {time.time()-t0:.0f}s")

        results[ds_name] = {}
        for cfg_name, cfg in ablation_configs.items():
            t1 = time.time()
            r = assemble_and_evaluate(cache, **cfg)
            if r:
                results[ds_name][cfg_name] = r
                print(f"  {cfg_name:20s}: AUROC={r['auroc']:.4f}  n={r['n_features']}  [{time.time()-t1:.1f}s]")
            else:
                results[ds_name][cfg_name] = {"auroc": None}
                print(f"  {cfg_name:20s}: SKIP")

        # Seed stability — re-run full with different seeds using same cache structure
        # (Not possible with cached data since OOF depends on seed. Skip for now.)

        del cache; gc.collect()

    with open(os.path.join(RESULTS_DIR, "ablation_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Summary table
    print(f"\n{'='*60}\nABLATION SUMMARY\n{'='*60}")
    header = f"{'Config':20s}"
    for ds in DATASETS: header += f"  {ds[:12]:>12s}"
    print(header)
    print("-"*len(header))
    for cfg_name in ablation_configs:
        row = f"{cfg_name:20s}"
        for ds in DATASETS:
            r = results[ds].get(cfg_name, {})
            a = r.get("auroc")
            row += f"  {a:.4f}" if a else f"  {'N/A':>12s}"
            row += "       "
        print(row)

    return results


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\nTotal: {time.time()-t0:.0f}s ({(time.time()-t0)/60:.1f}min)")
