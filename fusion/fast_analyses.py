"""
Fast analyses addressing reviewer concerns.
Streamlined: fewer bootstrap iterations, key ablations only, minimal I/O.
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

DATASETS = {
    "common_claim_3class": {"n_classes": 3, "splits": {"train": "train", "val": "val", "test": "test"}, "ext": "common_claim_3class", "has_probes": True},
    "e2h_amc_3class": {"n_classes": 3, "splits": {"train": "train_sub", "val": "val_split", "test": "eval"}, "ext": "e2h_amc_3class", "has_probes": True},
    "e2h_amc_5class": {"n_classes": 5, "splits": {"train": "train_sub", "val": "val_split", "test": "eval"}, "ext": "e2h_amc_5class", "has_probes": True},
    "when2call_3class": {"n_classes": 3, "splits": {"train": "train", "val": "val", "test": "test"}, "ext": "when2call_3class", "has_probes": True},
}

OLD_PROBES = ["lr_probe", "pca_lr", "iti", "kb_mlp", "attn_satisfies", "sep", "step"]
N_FOLDS = 5
C_GRID = [1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
C_GRID_META = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 1e-1, 1.0]

SOURCES = [
    ("input_hidden", "input_last_token_hidden", 512),
    ("gen_hidden", "gen_last_token_hidden", 512),
    ("head_act", "input_per_head_activation", 256),
    ("attn_stats", "input_attn_stats", None),
    ("attn_vnorms", "input_attn_value_norms", 256),
]


def load_labels(ext_dir, split):
    with open(os.path.join(EXTRACTION_DIR, ext_dir, split, "meta.json")) as f:
        return np.array(json.load(f)["labels"])


def compute_auroc(y_true, y_prob, n_classes):
    if n_classes == 2:
        return roc_auc_score(y_true, y_prob[:, 1])
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))
    return roc_auc_score(y_bin, y_prob, average="macro", multi_class="ovr")


def gaussian_basis(n_layers, n_basis=5, sigma_scale=0.3):
    centers = np.linspace(0, n_layers - 1, n_basis)
    sigma = sigma_scale * n_layers / n_basis
    basis = np.zeros((n_layers, n_basis))
    for i, c in enumerate(centers):
        basis[:, i] = np.exp(-0.5 * ((np.arange(n_layers) - c) / sigma) ** 2)
    basis /= basis.sum(axis=0, keepdims=True) + 1e-10
    return basis


def bootstrap_ci(y_true, y_prob, n_classes, n_boot=500):
    n = len(y_true)
    rng = np.random.RandomState(42)
    scores = []
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        try:
            scores.append(compute_auroc(y_true[idx], y_prob[idx], n_classes))
        except:
            pass
    scores = sorted(scores)
    return scores[int(0.025 * len(scores))], scores[int(0.975 * len(scores))]


def run_fusion(ds_name, info, source_names=None, include_traj=True, include_probes=True,
               include_direct=True, layer_mode="subsample", seed=42):
    """Core fusion. Returns (te_prob, te_labels, n_meta_features)."""
    n_classes = info["n_classes"]
    sp = info["splits"]
    ext = info["ext"]

    tr_labels = load_labels(ext, sp["train"])
    va_labels = load_labels(ext, sp["val"])
    te_labels = load_labels(ext, sp["test"])
    trva_labels = np.concatenate([tr_labels, va_labels])
    n_trva, n_te = len(trva_labels), len(te_labels)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)

    sources = SOURCES if source_names is None else [s for s in SOURCES if s[0] in source_names]

    source_layer_oof = {}
    source_layer_te = {}
    all_direct_oof = []
    all_direct_te = []

    for sname, raw_name, pca_dim in sources:
        tr_path = os.path.join(EXTRACTION_DIR, ext, sp["train"], f"{raw_name}.pt")
        if not os.path.exists(tr_path):
            continue

        tr_raw = torch.load(tr_path, map_location="cpu").float().numpy()
        va_raw = torch.load(os.path.join(EXTRACTION_DIR, ext, sp["val"], f"{raw_name}.pt"), map_location="cpu").float().numpy()
        te_raw = torch.load(os.path.join(EXTRACTION_DIR, ext, sp["test"], f"{raw_name}.pt"), map_location="cpu").float().numpy()
        trva_raw = np.concatenate([tr_raw, va_raw])

        if raw_name in ["input_last_token_hidden", "gen_last_token_hidden"]:
            n_layers = tr_raw.shape[1]
            get_layer = lambda d, l: d[:, l, :]
        elif raw_name == "input_per_head_activation":
            n_layers = tr_raw.shape[1]
            get_layer = lambda d, l: d[:, l, :, :].reshape(d.shape[0], -1)
        elif raw_name == "input_attn_stats":
            n_layers = tr_raw.shape[1]
            get_layer = lambda d, l: d[:, l, :, :].reshape(d.shape[0], -1)
        elif raw_name == "input_attn_value_norms":
            n_layers = tr_raw.shape[1]
            get_layer = lambda d, l: d[:, l, :, :].max(axis=-1)
        else:
            del tr_raw, va_raw, te_raw, trva_raw; continue

        if layer_mode == "best_only":
            layer_indices = [n_layers // 2, n_layers - 1]
        elif layer_mode == "all":
            layer_indices = list(range(n_layers))
        else:
            if raw_name == "input_per_head_activation":
                layer_indices = list(range(0, n_layers, 4))
            elif n_layers > 20:
                layer_indices = list(range(0, n_layers, 2))
            else:
                layer_indices = list(range(n_layers))

        source_layer_oof[sname] = {}
        source_layer_te[sname] = {}

        for l in layer_indices:
            X_trva = get_layer(trva_raw, l)
            X_te = get_layer(te_raw, l)
            if X_trva.ndim == 1: X_trva, X_te = X_trva.reshape(-1, 1), X_te.reshape(-1, 1)

            sc = StandardScaler()
            X_trva_s = sc.fit_transform(X_trva)
            X_te_s = sc.transform(X_te)

            if pca_dim and X_trva_s.shape[1] > pca_dim:
                pca = PCA(n_components=pca_dim, random_state=42)
                X_trva_s = pca.fit_transform(X_trva_s)
                X_te_s = pca.transform(X_te_s)

            # C selection
            n_tr = len(tr_labels)
            sp_cut = int(n_tr * 0.8)
            ld = get_layer(tr_raw, l)
            if ld.ndim == 1: ld = ld.reshape(-1, 1)
            tr_only = sc.transform(ld)
            if pca_dim and tr_only.shape[1] > pca_dim:
                tr_only = pca.transform(tr_only)

            best_a, best_C = -1, 0.01
            for C in C_GRID:
                clf = LogisticRegression(max_iter=2000, C=C, random_state=42)
                clf.fit(tr_only[:sp_cut], tr_labels[:sp_cut])
                vp = clf.predict_proba(tr_only[sp_cut:n_tr])
                a = compute_auroc(tr_labels[sp_cut:], vp, n_classes)
                if a > best_a: best_a, best_C = a, C

            oof = np.zeros((n_trva, n_classes))
            te_avg = np.zeros((n_te, n_classes))
            for _, (tr_i, va_i) in enumerate(skf.split(X_trva_s, trva_labels)):
                clf = LogisticRegression(max_iter=2000, C=best_C, random_state=42)
                clf.fit(X_trva_s[tr_i], trva_labels[tr_i])
                oof[va_i] = clf.predict_proba(X_trva_s[va_i])
                te_avg += clf.predict_proba(X_te_s) / N_FOLDS

            source_layer_oof[sname][l] = oof
            source_layer_te[sname][l] = te_avg
            if include_direct:
                all_direct_oof.append(oof)
                all_direct_te.append(te_avg)

        del tr_raw, va_raw, te_raw, trva_raw
        gc.collect()

    # Trajectory
    traj_oof, traj_te = [], []
    if include_traj:
        for sname in source_layer_oof:
            layers = sorted(source_layer_oof[sname].keys())
            if len(layers) < 3: continue
            oof_stack = np.stack([source_layer_oof[sname][l] for l in layers], axis=1)
            te_stack = np.stack([source_layer_te[sname][l] for l in layers], axis=1)
            basis = gaussian_basis(len(layers), n_basis=min(7, len(layers)))
            for c in range(n_classes):
                to, tt = oof_stack[:, :, c], te_stack[:, :, c]
                traj_oof.append(to @ basis)
                traj_te.append(tt @ basis)
                for fn in [np.mean, np.max, np.std]:
                    traj_oof.append(fn(to, axis=1, keepdims=True))
                    traj_te.append(fn(tt, axis=1, keepdims=True))
                traj_oof.append(to.argmax(axis=1, keepdims=True).astype(float) / len(layers))
                traj_te.append(tt.argmax(axis=1, keepdims=True).astype(float) / len(layers))

    # Old probes
    probe_oof, probe_te = [], []
    if include_probes and info.get("has_probes"):
        for method in OLD_PROBES:
            path = os.path.join(PROCESSED_DIR, ds_name, method, "train.pt")
            if not os.path.exists(path): continue
            tr = torch.load(path, map_location="cpu").float().numpy()
            va = torch.load(os.path.join(PROCESSED_DIR, ds_name, method, "val.pt"), map_location="cpu").float().numpy()
            te = torch.load(os.path.join(PROCESSED_DIR, ds_name, method, "test.pt"), map_location="cpu").float().numpy()
            if tr.ndim == 1: tr, va, te = tr.reshape(-1,1), va.reshape(-1,1), te.reshape(-1,1)
            trva = np.vstack([tr, va])
            sc = StandardScaler()
            trva_s = sc.fit_transform(trva)
            te_s = sc.transform(te)
            if trva_s.shape[1] > 256:
                p = PCA(n_components=256, random_state=42)
                trva_s = p.fit_transform(trva_s)
                te_s = p.transform(te_s)

            n_tr = len(tr_labels)
            sp_cut = int(n_tr * 0.8)
            tr_only_s = sc.transform(tr)
            if tr.shape[1] > 256:
                tr_only_s = p.transform(tr_only_s)
            best_a, best_C = -1, 1.0
            for C in [0.001, 0.01, 0.1, 1.0, 10.0]:
                clf = LogisticRegression(max_iter=2000, C=C, random_state=42)
                clf.fit(tr_only_s[:sp_cut], tr_labels[:sp_cut])
                vp = clf.predict_proba(tr_only_s[sp_cut:n_tr])
                a = compute_auroc(tr_labels[sp_cut:], vp, n_classes)
                if a > best_a: best_a, best_C = a, C

            oof = np.zeros((n_trva, n_classes))
            te_agg = np.zeros((n_te, n_classes))
            for _, (tr_i, va_i) in enumerate(skf.split(trva_s, trva_labels)):
                clf = LogisticRegression(max_iter=2000, C=best_C, random_state=42)
                clf.fit(trva_s[tr_i], trva_labels[tr_i])
                oof[va_i] = clf.predict_proba(trva_s[va_i])
                te_agg += clf.predict_proba(te_s) / N_FOLDS
            probe_oof.append(oof)
            probe_te.append(te_agg)

    parts_oof = all_direct_oof + traj_oof + probe_oof
    parts_te = all_direct_te + traj_te + probe_te
    if not parts_oof:
        return None, te_labels, 0

    meta_oof = np.hstack(parts_oof)
    meta_te = np.hstack(parts_te)

    # Meta
    sc_m = StandardScaler()
    mo_s = sc_m.fit_transform(meta_oof)
    mt_s = sc_m.transform(meta_te)

    best_au, best_C = -1, 0.01
    for C in C_GRID_META:
        inner_oof = np.zeros((n_trva, n_classes))
        for _, (tr_i, va_i) in enumerate(skf.split(mo_s, trva_labels)):
            clf = LogisticRegression(max_iter=3000, C=C, random_state=42)
            clf.fit(mo_s[tr_i], trva_labels[tr_i])
            inner_oof[va_i] = clf.predict_proba(mo_s[va_i])
        au = compute_auroc(trva_labels, inner_oof, n_classes)
        if au > best_au: best_au, best_C = au, C

    clf_f = LogisticRegression(max_iter=3000, C=best_C, random_state=42)
    clf_f.fit(mo_s, trva_labels)
    te_prob = clf_f.predict_proba(mt_s)

    return te_prob, te_labels, meta_oof.shape[1]


def run_ablations():
    """Key ablation configs only."""
    configs = {
        "full":            dict(source_names=None, include_traj=True,  include_probes=True,  include_direct=True,  layer_mode="subsample"),
        "no_trajectory":   dict(source_names=None, include_traj=False, include_probes=True,  include_direct=True,  layer_mode="subsample"),
        "no_probes":       dict(source_names=None, include_traj=True,  include_probes=False, include_direct=True,  layer_mode="subsample"),
        "no_direct":       dict(source_names=None, include_traj=True,  include_probes=True,  include_direct=False, layer_mode="subsample"),
        "best_layer":      dict(source_names=None, include_traj=False, include_probes=True,  include_direct=True,  layer_mode="best_only"),
        "probes_only":     dict(source_names=None, include_traj=False, include_probes=True,  include_direct=False, layer_mode="subsample"),
        "traj_only":       dict(source_names=None, include_traj=True,  include_probes=False, include_direct=False, layer_mode="subsample"),
        "drop_input_hid":  dict(source_names=["gen_hidden","head_act","attn_stats","attn_vnorms"], include_traj=True, include_probes=True, include_direct=True, layer_mode="subsample"),
        "drop_gen_hid":    dict(source_names=["input_hidden","head_act","attn_stats","attn_vnorms"], include_traj=True, include_probes=True, include_direct=True, layer_mode="subsample"),
        "drop_attn":       dict(source_names=["input_hidden","gen_hidden","head_act"], include_traj=True, include_probes=True, include_direct=True, layer_mode="subsample"),
        "input_hid_only":  dict(source_names=["input_hidden"], include_traj=True, include_probes=False, include_direct=True, layer_mode="subsample"),
    }

    results = {}
    for ds_name, info in DATASETS.items():
        results[ds_name] = {}
        print(f"\n{'='*50}\n{ds_name}\n{'='*50}")
        for cfg_name, cfg in configs.items():
            t0 = time.time()
            try:
                te_prob, te_labels, n_feat = run_fusion(ds_name, info, **cfg)
                if te_prob is None:
                    results[ds_name][cfg_name] = {"auroc": None}
                    print(f"  {cfg_name:20s}: SKIP")
                    continue
                auroc = compute_auroc(te_labels, te_prob, info["n_classes"])
                results[ds_name][cfg_name] = {"auroc": float(auroc), "n_features": n_feat}
                print(f"  {cfg_name:20s}: AUROC={auroc:.4f}  n_feat={n_feat}  [{time.time()-t0:.0f}s]")
            except Exception as e:
                results[ds_name][cfg_name] = {"auroc": None, "error": str(e)[:100]}
                print(f"  {cfg_name:20s}: ERROR {str(e)[:60]}")
            gc.collect()

    with open(os.path.join(RESULTS_DIR, "ablation_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    return results


def run_seed_stability():
    """Multi-seed test on 2 datasets."""
    seeds = [42, 123, 456, 789, 1024]
    results = {}
    for ds_name in ["common_claim_3class", "when2call_3class"]:
        info = DATASETS[ds_name]
        aurocs = []
        print(f"\n{ds_name}:")
        for seed in seeds:
            te_prob, te_labels, _ = run_fusion(ds_name, info, seed=seed)
            if te_prob is None: continue
            auroc = compute_auroc(te_labels, te_prob, info["n_classes"])
            aurocs.append(float(auroc))
            print(f"  seed={seed}: {auroc:.4f}")
            gc.collect()
        results[ds_name] = {
            "seeds": seeds, "aurocs": aurocs,
            "mean": float(np.mean(aurocs)), "std": float(np.std(aurocs))
        }
        print(f"  → {np.mean(aurocs):.4f} ± {np.std(aurocs):.4f}")

    with open(os.path.join(RESULTS_DIR, "seed_stability.json"), "w") as f:
        json.dump(results, f, indent=2)
    return results


def run_cheap_fusion():
    """Cost-performance analysis."""
    configs = [
        ("probes_only",          dict(source_names=None, include_traj=False, include_probes=True, include_direct=False, layer_mode="subsample"), 7.3),
        ("input_hid_best_layer", dict(source_names=["input_hidden"], include_traj=False, include_probes=False, include_direct=True, layer_mode="best_only"), 30),
        ("input_hid_subsample",  dict(source_names=["input_hidden"], include_traj=True, include_probes=False, include_direct=True, layer_mode="subsample"), 30),
        ("top2+probes",          dict(source_names=["input_hidden","gen_hidden"], include_traj=True, include_probes=True, include_direct=True, layer_mode="subsample"), 67),
        ("full",                 dict(source_names=None, include_traj=True, include_probes=True, include_direct=True, layer_mode="subsample"), 392),
    ]

    results = {}
    for ds_name in ["common_claim_3class", "when2call_3class"]:
        info = DATASETS[ds_name]
        results[ds_name] = {}
        print(f"\n{ds_name}:")
        for name, cfg, disk_gb in configs:
            t0 = time.time()
            te_prob, te_labels, n_feat = run_fusion(ds_name, info, **cfg)
            if te_prob is None:
                results[ds_name][name] = {"auroc": None}
                continue
            auroc = compute_auroc(te_labels, te_prob, info["n_classes"])
            results[ds_name][name] = {"auroc": float(auroc), "disk_gb": disk_gb, "n_features": n_feat}
            print(f"  {name:25s}: AUROC={auroc:.4f}  disk={disk_gb}GB  [{time.time()-t0:.0f}s]")
            gc.collect()

    with open(os.path.join(RESULTS_DIR, "cheap_fusion_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    return results


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    t0 = time.time()

    if which in ("ablation", "all"):
        run_ablations()
    if which in ("seed", "all"):
        run_seed_stability()
    if which in ("cheap", "all"):
        run_cheap_fusion()

    print(f"\nTotal: {time.time()-t0:.0f}s ({(time.time()-t0)/60:.1f}min)")
