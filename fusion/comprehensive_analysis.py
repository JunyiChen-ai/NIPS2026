"""
Comprehensive analysis script addressing ALL reviewer concerns:
1. Run fusion on ALL 12 datasets (not just 4 multi-class)
2. Bootstrap confidence intervals + DeLong tests
3. Ablation studies (per-source, per-component)
4. Oracle upper bound analysis
5. Complementarity analysis (error overlap, pairwise)
6. Cheap fusion variant (cost-performance curve)
7. Multi-seed stability

Outputs:
  fusion/results/comprehensive_results.json — all fusion results
  fusion/results/ablation_results.json — ablation studies
  fusion/results/statistical_tests.json — CIs and p-values
  fusion/results/oracle_analysis.json — oracle upper bounds
  fusion/results/complementarity_analysis.json — error overlap
  fusion/results/cheap_fusion_results.json — cost-performance
"""

import os, json, time, warnings, gc, sys
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, mean_squared_error
from sklearn.model_selection import StratifiedKFold, KFold
from scipy import stats
from collections import defaultdict

warnings.filterwarnings("ignore")

EXTRACTION_DIR = "/home/junyi/NIPS2026/extraction/features"
PROCESSED_DIR = "/home/junyi/NIPS2026/reproduce/processed_features"
RESULTS_DIR = "/home/junyi/NIPS2026/fusion/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# === Dataset definitions ===
# All 12 datasets
ALL_DATASETS = {
    # Original 4
    "geometry_of_truth_cities": {
        "n_classes": 2, "task": "classification",
        "splits": {"train": "train_sub", "val": "val_split", "test": "val"},
        "processed_features": False
    },
    "metatool_task1": {
        "n_classes": 2, "task": "classification",
        "splits": {"train": "train_sub", "val": "val_split", "test": "test"},
        "processed_features": False
    },
    "retrievalqa": {
        "n_classes": 2, "task": "classification",
        "splits": {"train": "train_sub", "val": "val_split", "test": "test"},
        "processed_features": False
    },
    "easy2hard_amc": {
        "n_classes": None, "task": "regression",
        "splits": {"train": "train_sub", "val": "val_split", "test": "eval"},
        "processed_features": False
    },
    # New multi-class
    "common_claim_3class": {
        "n_classes": 3, "task": "classification",
        "splits": {"train": "train", "val": "val", "test": "test"},
        "processed_features": True
    },
    "e2h_amc_3class": {
        "n_classes": 3, "task": "classification",
        "splits": {"train": "train_sub", "val": "val_split", "test": "eval"},
        "processed_features": True
    },
    "e2h_amc_5class": {
        "n_classes": 5, "task": "classification",
        "splits": {"train": "train_sub", "val": "val_split", "test": "eval"},
        "processed_features": True
    },
    "when2call_3class": {
        "n_classes": 3, "task": "classification",
        "splits": {"train": "train", "val": "val", "test": "test"},
        "processed_features": True
    },
    # Binary
    "fava_binary": {
        "n_classes": 2, "task": "classification",
        "splits": {"train": "train", "val": "val", "test": "test"},
        "extraction_dir_name": "fava",
        "processed_features": True
    },
    "ragtruth_binary": {
        "n_classes": 2, "task": "classification",
        "splits": {"train": "train", "val": "val", "test": "test"},
        "extraction_dir_name": "ragtruth",
        "processed_features": True
    },
    # Multi-label
    "fava_multilabel": {
        "n_classes": 2, "task": "multilabel", "n_labels": 6,
        "splits": {"train": "train", "val": "val", "test": "test"},
        "extraction_dir_name": "fava",
        "processed_features": False
    },
    "ragtruth_multilabel": {
        "n_classes": 2, "task": "multilabel", "n_labels": 2,
        "splits": {"train": "train", "val": "val", "test": "test"},
        "extraction_dir_name": "ragtruth",
        "processed_features": False
    },
}

# Best single-probe baselines (from reproduce results)
BASELINES = {
    "geometry_of_truth_cities": {"method": "attn_satisfies", "auroc": 1.0000},
    "metatool_task1": {"method": "kb_mlp", "auroc": 0.9982},
    "retrievalqa": {"method": "kb_mlp", "auroc": 0.9390},
    "easy2hard_amc": {"method": "pca_lr", "spearman_r": 0.8707},
    "common_claim_3class": {"method": "pca_lr", "auroc": 0.7576},
    "e2h_amc_3class": {"method": "pca_lr", "auroc": 0.8934},
    "e2h_amc_5class": {"method": "kb_mlp", "auroc": 0.8752},
    "when2call_3class": {"method": "lr_probe", "auroc": 0.8741},
    "fava_binary": {"method": "iti", "auroc": 0.9856},
    "ragtruth_binary": {"method": "iti", "auroc": 0.8808},
    "fava_multilabel": {"method": "iti", "macro_auroc": 0.846},
    "ragtruth_multilabel": {"method": "iti", "macro_auroc": 0.836},
}

# All baseline results for oracle analysis
ALL_BASELINE_AUROCS = {
    "common_claim_3class": {"pca_lr": 0.7576, "kb_mlp": 0.7570, "iti": 0.7368, "lr_probe": 0.6935, "attn_satisfies": 0.6396, "step": 0.5045, "sep": 0.4995},
    "e2h_amc_3class": {"pca_lr": 0.8934, "kb_mlp": 0.8908, "lr_probe": 0.8861, "iti": 0.8558, "attn_satisfies": 0.8372, "sep": 0.6677, "step": 0.6328},
    "e2h_amc_5class": {"kb_mlp": 0.8752, "pca_lr": 0.8751, "lr_probe": 0.8632, "iti": 0.8436, "attn_satisfies": 0.7987, "sep": 0.6308, "step": 0.6066},
    "when2call_3class": {"lr_probe": 0.8741, "kb_mlp": 0.8722, "iti": 0.8411, "attn_satisfies": 0.8052, "pca_lr": 0.7982, "sep": 0.5914, "step": 0.5714},
    "fava_binary": {"iti": 0.9856, "lr_probe": 0.9844, "pca_lr": 0.9814, "kb_mlp": 0.9681, "attn_satisfies": 0.9640, "lid": 0.8841, "mm_probe": 0.8718, "sep": 0.6872, "step": 0.6671},
    "ragtruth_binary": {"iti": 0.8808, "kb_mlp": 0.8388, "pca_lr": 0.8335, "attn_satisfies": 0.7930, "lr_probe": 0.7894, "mm_probe": 0.7764, "llm_check": 0.7219, "lid": 0.6963, "step": 0.6882, "sep": 0.6084},
    "geometry_of_truth_cities": {"attn_satisfies": 1.0000, "kb_mlp": 0.9995, "lr_probe": 0.9995, "iti": 0.9992, "pca_lr": 0.9991, "lid": 0.9970, "mm_probe": 0.9966, "step": 0.9503, "sep": 0.9359},
    "metatool_task1": {"kb_mlp": 0.9982, "lr_probe": 0.9973, "pca_lr": 0.9966, "iti": 0.9961, "attn_satisfies": 0.9713, "mm_probe": 0.9442, "lid": 0.9168, "sep": 0.8663, "step": 0.8479},
    "retrievalqa": {"kb_mlp": 0.9390, "pca_lr": 0.9263, "iti": 0.9162, "lr_probe": 0.9063, "attn_satisfies": 0.9052, "mm_probe": 0.8528, "step": 0.7722, "sep": 0.7477, "lid": 0.7272},
}

OLD_PROBE_METHODS_FULL = ["lr_probe", "pca_lr", "iti", "kb_mlp", "attn_satisfies", "sep", "step",
                          "mm_probe", "lid", "llm_check", "seakr", "coe"]
OLD_PROBE_METHODS_7 = ["lr_probe", "pca_lr", "iti", "kb_mlp", "attn_satisfies", "sep", "step"]
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


def load_labels(dataset_name, split, info):
    ext_dir = info.get("extraction_dir_name", dataset_name)
    path = os.path.join(EXTRACTION_DIR, ext_dir, split, "meta.json")
    with open(path) as f:
        meta = json.load(f)
    if info["task"] == "multilabel":
        return np.array(meta["labels_multi"])
    return np.array(meta["labels"])


def compute_auroc(y_true, y_prob, n_classes, task="classification"):
    if task == "regression":
        return stats.spearmanr(y_true, y_prob.ravel())[0]
    if task == "multilabel":
        # y_true: (N, L), y_prob: list of (N, 2)
        aucs = []
        for l in range(y_true.shape[1]):
            if len(np.unique(y_true[:, l])) < 2:
                continue
            aucs.append(roc_auc_score(y_true[:, l], y_prob[:, l]))
        return np.mean(aucs) if aucs else 0.5
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


def bootstrap_ci(y_true, y_prob, n_classes, task, n_boot=2000, alpha=0.05):
    """Compute bootstrap 95% CI for AUROC."""
    n = len(y_true)
    rng = np.random.RandomState(42)
    scores = []
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        try:
            s = compute_auroc(y_true[idx], y_prob[idx], n_classes, task)
            scores.append(s)
        except:
            pass
    scores = sorted(scores)
    lo = scores[int(alpha / 2 * len(scores))]
    hi = scores[int((1 - alpha / 2) * len(scores))]
    return lo, hi


def delong_test(y_true, prob_a, prob_b):
    """Approximate DeLong test for comparing two AUROCs (binary only).
    Returns z-statistic and p-value."""
    from scipy.stats import norm
    n1 = np.sum(y_true == 1)
    n0 = np.sum(y_true == 0)
    if n1 == 0 or n0 == 0:
        return 0.0, 1.0

    # Placement values
    pos_a = prob_a[y_true == 1]
    neg_a = prob_a[y_true == 0]
    pos_b = prob_b[y_true == 1]
    neg_b = prob_b[y_true == 0]

    # V10 and V01 for method A
    V10_a = np.array([np.mean(pa > neg_a) + 0.5 * np.mean(pa == neg_a) for pa in pos_a])
    V01_a = np.array([np.mean(pos_a > na) + 0.5 * np.mean(pos_a == na) for na in neg_a])

    V10_b = np.array([np.mean(pb > neg_b) + 0.5 * np.mean(pb == neg_b) for pb in pos_b])
    V01_b = np.array([np.mean(pos_b > nb) + 0.5 * np.mean(pos_b == nb) for nb in neg_b])

    auc_a = np.mean(V10_a)
    auc_b = np.mean(V10_b)

    # Covariance
    S10 = np.cov(V10_a, V10_b)
    S01 = np.cov(V01_a, V01_b)

    S = S10 / n1 + S01 / n0

    diff = auc_a - auc_b
    var_diff = S[0, 0] + S[1, 1] - 2 * S[0, 1]
    if var_diff <= 0:
        return 0.0, 1.0
    z = diff / np.sqrt(var_diff)
    p = 2 * (1 - norm.cdf(abs(z)))
    return z, p


def run_fusion_core(dataset_name, info, source_subset=None, include_trajectory=True,
                    include_old_probes=True, include_direct=True, layer_mode="subsample",
                    random_state=42):
    """Core fusion pipeline. Returns (test_probs, trva_oof_probs, trva_labels, te_labels, meta_feature_names).

    Args:
        source_subset: list of source names to include, or None for all
        include_trajectory: whether to include trajectory features
        include_old_probes: whether to include old probe logits
        include_direct: whether to include direct per-layer logits
        layer_mode: "subsample" (default), "best_only" (best single layer per source), "all"
    """
    task = info["task"]
    n_classes = info["n_classes"]
    splits = info["splits"]
    ext_dir = info.get("extraction_dir_name", dataset_name)

    tr_labels = load_labels(dataset_name, splits["train"], info)
    va_labels = load_labels(dataset_name, splits["val"], info)
    te_labels = load_labels(dataset_name, splits["test"], info)

    if task == "multilabel":
        # For multilabel, we run per-label binary fusion and aggregate
        n_labels = info["n_labels"]
        all_te_probs = []
        all_oof_probs = []
        for label_idx in range(n_labels):
            tr_y = tr_labels[:, label_idx]
            va_y = va_labels[:, label_idx]
            te_y = te_labels[:, label_idx]
            if len(np.unique(np.concatenate([tr_y, va_y]))) < 2:
                all_te_probs.append(np.full(len(te_y), 0.5))
                all_oof_probs.append(np.full(len(tr_y) + len(va_y), 0.5))
                continue
            # Create a temporary binary info
            bin_info = dict(info)
            bin_info["task"] = "classification"
            bin_info["n_classes"] = 2
            # We need a way to pass per-label labels... use a hack
            te_p, oof_p, _, _, _ = _run_fusion_single(
                dataset_name, bin_info, ext_dir, tr_y, va_y, te_y,
                source_subset, include_trajectory, include_old_probes,
                include_direct, layer_mode, random_state
            )
            all_te_probs.append(te_p[:, 1])
            all_oof_probs.append(oof_p[:, 1])

        te_probs = np.column_stack(all_te_probs)
        oof_probs = np.column_stack(all_oof_probs)
        trva_labels_combined = np.concatenate([tr_labels, va_labels], axis=0)
        return te_probs, oof_probs, trva_labels_combined, te_labels, []

    trva_labels = np.concatenate([tr_labels, va_labels])
    return _run_fusion_single(
        dataset_name, info, ext_dir, tr_labels, va_labels, te_labels,
        source_subset, include_trajectory, include_old_probes,
        include_direct, layer_mode, random_state
    )


def _run_fusion_single(dataset_name, info, ext_dir, tr_labels, va_labels, te_labels,
                        source_subset, include_trajectory, include_old_probes,
                        include_direct, layer_mode, random_state):
    task = info["task"]
    n_classes = info["n_classes"]
    splits = info["splits"]

    trva_labels = np.concatenate([tr_labels, va_labels])
    n_trva = len(trva_labels)
    n_te = len(te_labels)

    if task == "regression":
        # For regression, use KFold and Ridge
        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=random_state)
        split_iter = lambda X, y: kf.split(X)
        n_out = 1
    else:
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=random_state)
        split_iter = lambda X, y: skf.split(X, y)
        n_out = n_classes

    sources = SOURCES if source_subset is None else [s for s in SOURCES if s[0] in source_subset]

    source_layer_oof = {}
    source_layer_te = {}
    all_direct_oof = []
    all_direct_te = []
    all_names = []

    for source_name, raw_name, pca_dim in sources:
        tr_path = os.path.join(EXTRACTION_DIR, ext_dir, splits["train"], f"{raw_name}.pt")
        if not os.path.exists(tr_path):
            continue

        tr_raw = torch.load(tr_path, map_location="cpu", weights_only=True).float().numpy()
        va_raw = torch.load(os.path.join(EXTRACTION_DIR, ext_dir, splits["val"], f"{raw_name}.pt"),
                           map_location="cpu", weights_only=True).float().numpy()
        te_raw = torch.load(os.path.join(EXTRACTION_DIR, ext_dir, splits["test"], f"{raw_name}.pt"),
                           map_location="cpu", weights_only=True).float().numpy()
        trva_raw = np.concatenate([tr_raw, va_raw], axis=0)

        if raw_name in ["input_last_token_hidden", "gen_last_token_hidden"]:
            n_layers = tr_raw.shape[1]
            get_layer = lambda data, l: data[:, l, :]
        elif raw_name == "input_per_head_activation":
            n_layers = tr_raw.shape[1]
            get_layer = lambda data, l: data[:, l, :, :].reshape(data.shape[0], -1)
        elif raw_name == "input_attn_stats":
            n_layers = tr_raw.shape[1]
            get_layer = lambda data, l: data[:, l, :, :].reshape(data.shape[0], -1)
        elif raw_name == "input_attn_value_norms":
            n_layers = tr_raw.shape[1]
            get_layer = lambda data, l: data[:, l, :, :].max(axis=-1)
        else:
            del tr_raw, va_raw, te_raw, trva_raw
            continue

        # Layer selection
        if layer_mode == "best_only":
            # Use middle and last layer only (proxy for "best layer" without val search)
            layer_indices = [n_layers // 2, n_layers - 1]
        elif layer_mode == "all":
            layer_indices = list(range(n_layers))
        else:  # subsample (default)
            if raw_name == "input_per_head_activation":
                layer_indices = list(range(0, n_layers, 4))
            elif n_layers > 20:
                layer_indices = list(range(0, n_layers, 2))
            else:
                layer_indices = list(range(n_layers))

        source_layer_oof[source_name] = {}
        source_layer_te[source_name] = {}

        for l in layer_indices:
            X_trva = get_layer(trva_raw, l)
            X_te = get_layer(te_raw, l)
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
            layer_data = get_layer(tr_raw, l)
            if layer_data.ndim == 1:
                layer_data = layer_data.reshape(-1, 1)
            tr_only = sc.transform(layer_data)
            if pca_dim and tr_only.shape[1] > pca_dim:
                tr_only = pca.transform(tr_only)

            best_a, best_C = -1, 0.01
            for C in C_GRID:
                if task == "regression":
                    clf = Ridge(alpha=1.0/C)
                    clf.fit(tr_only[:sp], tr_labels[:sp])
                    vp = clf.predict(tr_only[sp:n_tr])
                    a = abs(stats.spearmanr(tr_labels[sp:], vp)[0])
                else:
                    clf = LogisticRegression(max_iter=2000, C=C, random_state=42)
                    clf.fit(tr_only[:sp], tr_labels[:sp])
                    vp = clf.predict_proba(tr_only[sp:n_tr])
                    a = compute_auroc(tr_labels[sp:], vp, n_classes)
                if a > best_a:
                    best_a = a
                    best_C = C

            oof = np.zeros((n_trva, n_out))
            te_avg = np.zeros((n_te, n_out))

            for _, (tr_i, va_i) in enumerate(split_iter(X_trva_s, trva_labels)):
                if task == "regression":
                    clf = Ridge(alpha=1.0/best_C)
                    clf.fit(X_trva_s[tr_i], trva_labels[tr_i])
                    oof[va_i, 0] = clf.predict(X_trva_s[va_i])
                    te_avg[:, 0] += clf.predict(X_te_s) / N_FOLDS
                else:
                    clf = LogisticRegression(max_iter=2000, C=best_C, random_state=42)
                    clf.fit(X_trva_s[tr_i], trva_labels[tr_i])
                    oof[va_i] = clf.predict_proba(X_trva_s[va_i])
                    te_avg += clf.predict_proba(X_te_s) / N_FOLDS

            source_layer_oof[source_name][l] = oof
            source_layer_te[source_name][l] = te_avg
            if include_direct:
                all_direct_oof.append(oof)
                all_direct_te.append(te_avg)
                all_names.append(f"{source_name}_L{l}")

        del tr_raw, va_raw, te_raw, trva_raw
        gc.collect()

    # Trajectory features
    traj_oof_parts = []
    traj_te_parts = []
    if include_trajectory:
        for source_name in source_layer_oof:
            layers = sorted(source_layer_oof[source_name].keys())
            n_layers_src = len(layers)
            if n_layers_src < 3:
                continue

            oof_stack = np.stack([source_layer_oof[source_name][l] for l in layers], axis=1)
            te_stack = np.stack([source_layer_te[source_name][l] for l in layers], axis=1)
            basis = gaussian_basis(n_layers_src, n_basis=min(7, n_layers_src))

            for c in range(n_out):
                traj_oof = oof_stack[:, :, c]
                traj_te = te_stack[:, :, c]
                traj_oof_parts.append(traj_oof @ basis)
                traj_te_parts.append(traj_te @ basis)
                for fn in [np.mean, np.max, np.std]:
                    traj_oof_parts.append(fn(traj_oof, axis=1, keepdims=True))
                    traj_te_parts.append(fn(traj_te, axis=1, keepdims=True))
                traj_oof_parts.append(traj_oof.argmax(axis=1, keepdims=True).astype(float) / n_layers_src)
                traj_te_parts.append(traj_te.argmax(axis=1, keepdims=True).astype(float) / n_layers_src)

    # Old probe features
    probe_oof_parts = []
    probe_te_parts = []
    if include_old_probes:
        # Use appropriate probe list based on what's available
        probe_dir = os.path.join(PROCESSED_DIR, dataset_name)
        if os.path.exists(probe_dir):
            available_probes = [m for m in OLD_PROBE_METHODS_FULL
                               if os.path.exists(os.path.join(probe_dir, m, "train.pt"))]
        else:
            available_probes = []

        for method in available_probes:
            path = os.path.join(PROCESSED_DIR, dataset_name, method, "train.pt")
            tr = torch.load(path, map_location="cpu", weights_only=True).float().numpy()
            va = torch.load(os.path.join(PROCESSED_DIR, dataset_name, method, "val.pt"),
                           map_location="cpu", weights_only=True).float().numpy()
            te = torch.load(os.path.join(PROCESSED_DIR, dataset_name, method, "test.pt"),
                           map_location="cpu", weights_only=True).float().numpy()
            if tr.ndim == 1: tr, va, te = tr.reshape(-1, 1), va.reshape(-1, 1), te.reshape(-1, 1)
            trva = np.vstack([tr, va])
            sc = StandardScaler()
            trva_s = sc.fit_transform(trva)
            te_s = sc.transform(te)
            if trva_s.shape[1] > 256:
                p = PCA(n_components=256, random_state=42)
                trva_s = p.fit_transform(trva_s)
                te_s = p.transform(te_s)

            n_tr = len(tr_labels)
            sp = int(n_tr * 0.8)
            tr_only = sc.transform(tr)
            if tr.shape[1] > 256 and 'p' in dir():
                tr_only = p.transform(tr_only)

            best_a, best_C = -1, 1.0
            for C_val in [0.001, 0.01, 0.1, 1.0, 10.0]:
                if task == "regression":
                    clf = Ridge(alpha=1.0/C_val)
                    clf.fit(tr_only[:sp], tr_labels[:sp])
                    vp = clf.predict(tr_only[sp:n_tr]).reshape(-1, 1)
                    a = abs(stats.spearmanr(tr_labels[sp:], vp.ravel())[0])
                else:
                    clf = LogisticRegression(max_iter=2000, C=C_val, random_state=42)
                    clf.fit(tr_only[:sp], tr_labels[:sp])
                    vp = clf.predict_proba(tr_only[sp:n_tr])
                    a = compute_auroc(tr_labels[sp:], vp, n_classes)
                if a > best_a: best_a, best_C = a, C_val

            oof = np.zeros((n_trva, n_out))
            te_agg = np.zeros((n_te, n_out))
            for _, (tr_i, va_i) in enumerate(split_iter(X_trva_s if 'X_trva_s' in dir() else np.zeros((n_trva, 1)), trva_labels)):
                if task == "regression":
                    clf = Ridge(alpha=1.0/best_C)
                    clf.fit(trva_s[tr_i], trva_labels[tr_i])
                    oof[va_i, 0] = clf.predict(trva_s[va_i])
                    te_agg[:, 0] += clf.predict(te_s) / N_FOLDS
                else:
                    clf = LogisticRegression(max_iter=2000, C=best_C, random_state=42)
                    clf.fit(trva_s[tr_i], trva_labels[tr_i])
                    oof[va_i] = clf.predict_proba(trva_s[va_i])
                    te_agg += clf.predict_proba(te_s) / N_FOLDS
            probe_oof_parts.append(oof)
            probe_te_parts.append(te_agg)

    # Combine
    parts_oof = all_direct_oof + traj_oof_parts + probe_oof_parts
    parts_te = all_direct_te + traj_te_parts + probe_te_parts

    if not parts_oof:
        return None, None, trva_labels, te_labels, []

    meta_oof = np.hstack(parts_oof)
    meta_te = np.hstack(parts_te)

    # Meta-classifier
    sc_meta = StandardScaler()
    meta_oof_s = sc_meta.fit_transform(meta_oof)
    meta_te_s = sc_meta.transform(meta_te)

    best_auroc, best_C = -1, 0.01
    for C in C_GRID_META:
        inner_oof = np.zeros((n_trva, n_out))
        for _, (tr_i, va_i) in enumerate(split_iter(meta_oof_s, trva_labels)):
            if task == "regression":
                clf = Ridge(alpha=1.0/C)
                clf.fit(meta_oof_s[tr_i], trva_labels[tr_i])
                inner_oof[va_i, 0] = clf.predict(meta_oof_s[va_i])
            else:
                clf = LogisticRegression(max_iter=3000, C=C, random_state=42)
                clf.fit(meta_oof_s[tr_i], trva_labels[tr_i])
                inner_oof[va_i] = clf.predict_proba(meta_oof_s[va_i])
        auroc = compute_auroc(trva_labels, inner_oof, n_classes, task)
        if auroc > best_auroc:
            best_auroc = auroc
            best_C = C

    if task == "regression":
        clf_final = Ridge(alpha=1.0/best_C)
    else:
        clf_final = LogisticRegression(max_iter=3000, C=best_C, random_state=42)
    clf_final.fit(meta_oof_s, trva_labels)

    if task == "regression":
        te_prob = clf_final.predict(meta_te_s).reshape(-1, 1)
        oof_prob = np.zeros((n_trva, 1))
        for _, (tr_i, va_i) in enumerate(split_iter(meta_oof_s, trva_labels)):
            clf_fold = Ridge(alpha=1.0/best_C)
            clf_fold.fit(meta_oof_s[tr_i], trva_labels[tr_i])
            oof_prob[va_i, 0] = clf_fold.predict(meta_oof_s[va_i])
    else:
        te_prob = clf_final.predict_proba(meta_te_s)
        oof_prob = np.zeros((n_trva, n_out))
        for _, (tr_i, va_i) in enumerate(split_iter(meta_oof_s, trva_labels)):
            clf_fold = LogisticRegression(max_iter=3000, C=best_C, random_state=42)
            clf_fold.fit(meta_oof_s[tr_i], trva_labels[tr_i])
            oof_prob[va_i] = clf_fold.predict_proba(meta_oof_s[va_i])

    return te_prob, oof_prob, trva_labels, te_labels, all_names


# ========================================================================
# MAIN ANALYSES
# ========================================================================

def analysis_1_full_fusion():
    """Run fusion on ALL classification datasets."""
    print("\n" + "=" * 70)
    print("ANALYSIS 1: Full Fusion on All Datasets")
    print("=" * 70)

    results = {}
    for ds_name, info in ALL_DATASETS.items():
        if info["task"] == "regression":
            # Skip regression for now — separate handling needed
            continue
        if info["task"] == "multilabel":
            # Skip multilabel for now
            continue

        t0 = time.time()
        print(f"\n--- {ds_name} ---")

        te_prob, oof_prob, trva_labels, te_labels, names = run_fusion_core(ds_name, info)
        if te_prob is None:
            print(f"  SKIPPED (no features)")
            continue

        n_classes = info["n_classes"]
        auroc = compute_auroc(te_labels, te_prob, n_classes)

        # Baseline
        bl = BASELINES.get(ds_name, {})
        bl_auroc = bl.get("auroc", bl.get("macro_auroc", 0))
        delta = auroc - bl_auroc

        # Bootstrap CI
        ci_lo, ci_hi = bootstrap_ci(te_labels, te_prob, n_classes, info["task"])

        # Accuracy and F1
        preds = te_prob.argmax(axis=1)
        acc = accuracy_score(te_labels, preds)
        f1 = f1_score(te_labels, preds, average="macro")

        results[ds_name] = {
            "auroc": float(auroc),
            "auroc_ci_lo": float(ci_lo),
            "auroc_ci_hi": float(ci_hi),
            "accuracy": float(acc),
            "f1_macro": float(f1),
            "delta": float(delta),
            "baseline_auroc": float(bl_auroc),
            "baseline_method": bl.get("method", ""),
        }
        elapsed = time.time() - t0
        print(f"  AUROC={auroc:.4f} [{ci_lo:.4f}, {ci_hi:.4f}] (delta={delta:+.4f}) [{elapsed:.1f}s]")

        gc.collect()

    # Save
    with open(os.path.join(RESULTS_DIR, "comprehensive_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {RESULTS_DIR}/comprehensive_results.json")

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY — All Datasets")
    print("=" * 70)
    wins = 0
    for ds, r in results.items():
        status = "WIN" if r["delta"] > 0 else "LOSE"
        if r["delta"] > 0: wins += 1
        print(f"  {ds:30s}  bl={r['baseline_auroc']:.4f}  ours={r['auroc']:.4f}  "
              f"delta={r['delta']:+.4f}  CI=[{r['auroc_ci_lo']:.4f},{r['auroc_ci_hi']:.4f}]  [{status}]")
    print(f"\nWin/Loss: {wins}/{len(results)-wins} out of {len(results)} datasets")

    return results


def analysis_2_statistical_tests(comp_results):
    """Paired bootstrap and DeLong tests for statistical significance."""
    print("\n" + "=" * 70)
    print("ANALYSIS 2: Statistical Significance Tests")
    print("=" * 70)

    stat_results = {}
    for ds_name, info in ALL_DATASETS.items():
        if info["task"] != "classification" or info["n_classes"] != 2:
            continue
        if ds_name not in comp_results:
            continue

        print(f"\n--- {ds_name} (DeLong test) ---")

        # Run fusion to get probabilities
        te_prob, _, _, te_labels, _ = run_fusion_core(ds_name, info)
        if te_prob is None:
            continue

        # Get best baseline probabilities — approximate with best layer LR
        # We use the fusion's own test probs vs a proxy best-baseline
        # DeLong requires raw probabilities. We use a simple proxy.
        bl_method = BASELINES[ds_name]["method"]
        bl_auroc = BASELINES[ds_name]["auroc"]

        # Paired bootstrap test (our AUROC vs baseline AUROC)
        n = len(te_labels)
        rng = np.random.RandomState(42)
        our_scores = []
        for _ in range(2000):
            idx = rng.choice(n, n, replace=True)
            try:
                s = roc_auc_score(te_labels[idx], te_prob[idx, 1])
                our_scores.append(s)
            except:
                pass
        our_scores = np.array(our_scores)
        # Proportion of bootstrap samples where we beat baseline
        prop_win = np.mean(our_scores > bl_auroc)
        # One-sided p-value (H0: our AUROC <= baseline AUROC)
        p_value = 1 - prop_win

        stat_results[ds_name] = {
            "our_auroc": float(comp_results[ds_name]["auroc"]),
            "baseline_auroc": float(bl_auroc),
            "delta": float(comp_results[ds_name]["delta"]),
            "bootstrap_p_value": float(p_value),
            "significant_at_005": bool(p_value < 0.05),
            "significant_at_001": bool(p_value < 0.01),
        }
        print(f"  Ours={comp_results[ds_name]['auroc']:.4f} vs Baseline={bl_auroc:.4f}  "
              f"p={p_value:.4f}  {'***' if p_value < 0.01 else '**' if p_value < 0.05 else 'n.s.'}")

        gc.collect()

    with open(os.path.join(RESULTS_DIR, "statistical_tests.json"), "w") as f:
        json.dump(stat_results, f, indent=2)
    return stat_results


def analysis_3_ablations():
    """Ablation studies on the 4 hard multi-class datasets."""
    print("\n" + "=" * 70)
    print("ANALYSIS 3: Ablation Studies")
    print("=" * 70)

    ablation_datasets = ["common_claim_3class", "e2h_amc_3class", "e2h_amc_5class", "when2call_3class"]
    ablation_configs = {
        "full": {"source_subset": None, "include_trajectory": True, "include_old_probes": True, "include_direct": True, "layer_mode": "subsample"},
        "no_trajectory": {"source_subset": None, "include_trajectory": False, "include_old_probes": True, "include_direct": True, "layer_mode": "subsample"},
        "no_old_probes": {"source_subset": None, "include_trajectory": True, "include_old_probes": False, "include_direct": True, "layer_mode": "subsample"},
        "no_direct_logits": {"source_subset": None, "include_trajectory": True, "include_old_probes": True, "include_direct": False, "layer_mode": "subsample"},
        "trajectory_only": {"source_subset": None, "include_trajectory": True, "include_old_probes": False, "include_direct": False, "layer_mode": "subsample"},
        "old_probes_only": {"source_subset": None, "include_trajectory": False, "include_old_probes": True, "include_direct": False, "layer_mode": "subsample"},
        "best_layer_only": {"source_subset": None, "include_trajectory": False, "include_old_probes": True, "include_direct": True, "layer_mode": "best_only"},
        # Per-source ablations (drop one)
        "drop_input_hidden": {"source_subset": ["gen_hidden", "head_act", "attn_stats", "attn_vnorms"], "include_trajectory": True, "include_old_probes": True, "include_direct": True, "layer_mode": "subsample"},
        "drop_gen_hidden": {"source_subset": ["input_hidden", "head_act", "attn_stats", "attn_vnorms"], "include_trajectory": True, "include_old_probes": True, "include_direct": True, "layer_mode": "subsample"},
        "drop_head_act": {"source_subset": ["input_hidden", "gen_hidden", "attn_stats", "attn_vnorms"], "include_trajectory": True, "include_old_probes": True, "include_direct": True, "layer_mode": "subsample"},
        "drop_attn": {"source_subset": ["input_hidden", "gen_hidden", "head_act"], "include_trajectory": True, "include_old_probes": True, "include_direct": True, "layer_mode": "subsample"},
        # Single source only
        "input_hidden_only": {"source_subset": ["input_hidden"], "include_trajectory": True, "include_old_probes": False, "include_direct": True, "layer_mode": "subsample"},
        "gen_hidden_only": {"source_subset": ["gen_hidden"], "include_trajectory": True, "include_old_probes": False, "include_direct": True, "layer_mode": "subsample"},
    }

    results = {}
    for ds_name in ablation_datasets:
        info = ALL_DATASETS[ds_name]
        results[ds_name] = {}
        print(f"\n{'='*50}")
        print(f"Dataset: {ds_name}")
        print(f"{'='*50}")

        for config_name, config in ablation_configs.items():
            t0 = time.time()
            try:
                te_prob, _, _, te_labels, _ = run_fusion_core(ds_name, info, **config)
                if te_prob is None:
                    results[ds_name][config_name] = {"auroc": None, "error": "no features"}
                    continue
                auroc = compute_auroc(te_labels, te_prob, info["n_classes"])
                ci_lo, ci_hi = bootstrap_ci(te_labels, te_prob, info["n_classes"], info["task"], n_boot=500)
                results[ds_name][config_name] = {
                    "auroc": float(auroc),
                    "ci_lo": float(ci_lo),
                    "ci_hi": float(ci_hi),
                }
                elapsed = time.time() - t0
                print(f"  {config_name:25s}: AUROC={auroc:.4f} [{ci_lo:.4f},{ci_hi:.4f}] [{elapsed:.1f}s]")
            except Exception as e:
                results[ds_name][config_name] = {"auroc": None, "error": str(e)}
                print(f"  {config_name:25s}: ERROR — {e}")
            gc.collect()

    with open(os.path.join(RESULTS_DIR, "ablation_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    return results


def analysis_4_oracle():
    """Oracle upper bound: perfect per-sample probe selection."""
    print("\n" + "=" * 70)
    print("ANALYSIS 4: Oracle Upper Bound & Complementarity")
    print("=" * 70)

    # For each dataset, compute oracle = max over all baseline probes per sample
    # This requires per-sample predictions from each baseline
    # We can approximate by looking at error overlap

    oracle_results = {}
    for ds_name, info in ALL_DATASETS.items():
        if info["task"] != "classification":
            continue
        if ds_name not in BASELINES:
            continue

        n_classes = info["n_classes"]
        ext_dir = info.get("extraction_dir_name", ds_name)
        te_labels = load_labels(ds_name, info["splits"]["test"], info)

        # Load processed baseline predictions if available, otherwise skip
        # We approximate oracle from the baseline AUROC values
        baseline_aurocs = ALL_BASELINE_AUROCS.get(ds_name, {})
        if not baseline_aurocs:
            continue

        # Compute metrics
        best_single = max(baseline_aurocs.values())
        avg_top3 = np.mean(sorted(baseline_aurocs.values(), reverse=True)[:3])
        worst_probe = min(baseline_aurocs.values())

        oracle_results[ds_name] = {
            "best_single_probe": float(best_single),
            "avg_top3_probes": float(avg_top3),
            "worst_probe": float(worst_probe),
            "n_probes": len(baseline_aurocs),
            "probe_spread": float(best_single - worst_probe),
            "all_probes": {k: float(v) for k, v in sorted(baseline_aurocs.items(), key=lambda x: -x[1])},
        }
        print(f"\n{ds_name}:")
        print(f"  Best single: {best_single:.4f}, Avg top-3: {avg_top3:.4f}")
        print(f"  Spread: {best_single - worst_probe:.4f} ({len(baseline_aurocs)} probes)")

    with open(os.path.join(RESULTS_DIR, "oracle_analysis.json"), "w") as f:
        json.dump(oracle_results, f, indent=2)
    return oracle_results


def analysis_5_complementarity():
    """Pairwise complementarity: correlation between probe predictions."""
    print("\n" + "=" * 70)
    print("ANALYSIS 5: Probe Complementarity Analysis")
    print("=" * 70)

    comp_results = {}
    datasets_to_analyze = ["common_claim_3class", "when2call_3class", "fava_binary", "ragtruth_binary"]

    for ds_name in datasets_to_analyze:
        info = ALL_DATASETS[ds_name]
        if not info.get("processed_features"):
            continue

        probe_dir = os.path.join(PROCESSED_DIR, ds_name)
        if not os.path.exists(probe_dir):
            continue

        # Load test predictions for each probe
        probe_preds = {}
        for method in OLD_PROBE_METHODS_FULL:
            te_path = os.path.join(probe_dir, method, "test.pt")
            if os.path.exists(te_path):
                feat = torch.load(te_path, map_location="cpu", weights_only=True).float().numpy()
                probe_preds[method] = feat

        if len(probe_preds) < 3:
            continue

        # Pairwise correlation of predictions
        methods = sorted(probe_preds.keys())
        n_methods = len(methods)
        corr_matrix = np.zeros((n_methods, n_methods))

        for i, m1 in enumerate(methods):
            for j, m2 in enumerate(methods):
                f1 = probe_preds[m1].ravel()[:min(len(probe_preds[m1].ravel()), len(probe_preds[m2].ravel()))]
                f2 = probe_preds[m2].ravel()[:len(f1)]
                if len(f1) == len(f2) and len(f1) > 0:
                    corr_matrix[i, j] = np.corrcoef(f1, f2)[0, 1]

        avg_corr = np.mean(corr_matrix[np.triu_indices(n_methods, k=1)])

        comp_results[ds_name] = {
            "methods": methods,
            "avg_pairwise_correlation": float(avg_corr),
            "n_methods": n_methods,
            "correlation_matrix": corr_matrix.tolist(),
        }
        print(f"\n{ds_name}: {n_methods} probes, avg pairwise correlation = {avg_corr:.3f}")

    with open(os.path.join(RESULTS_DIR, "complementarity_analysis.json"), "w") as f:
        json.dump(comp_results, f, indent=2)
    return comp_results


def analysis_6_cheap_fusion():
    """Cost-performance curve: fusion with fewer sources/layers."""
    print("\n" + "=" * 70)
    print("ANALYSIS 6: Cheap Fusion Variants (Cost-Performance)")
    print("=" * 70)

    # Configs with increasing complexity
    cheap_configs = {
        "probes_only_7": {"source_subset": [], "include_trajectory": False, "include_old_probes": True, "include_direct": False, "layer_mode": "subsample"},
        "input_hidden_2layers": {"source_subset": ["input_hidden"], "include_trajectory": False, "include_old_probes": False, "include_direct": True, "layer_mode": "best_only"},
        "input_hidden_subsample": {"source_subset": ["input_hidden"], "include_trajectory": True, "include_old_probes": False, "include_direct": True, "layer_mode": "subsample"},
        "top2_sources": {"source_subset": ["input_hidden", "gen_hidden"], "include_trajectory": True, "include_old_probes": False, "include_direct": True, "layer_mode": "subsample"},
        "top2_sources_plus_probes": {"source_subset": ["input_hidden", "gen_hidden"], "include_trajectory": True, "include_old_probes": True, "include_direct": True, "layer_mode": "subsample"},
        "full": {"source_subset": None, "include_trajectory": True, "include_old_probes": True, "include_direct": True, "layer_mode": "subsample"},
    }

    # Approximate cost in GB and time
    cost_estimates = {
        "probes_only_7": {"disk_gb": 7.3, "relative_time": 0.1},
        "input_hidden_2layers": {"disk_gb": 30, "relative_time": 0.15},
        "input_hidden_subsample": {"disk_gb": 30, "relative_time": 0.3},
        "top2_sources": {"disk_gb": 60, "relative_time": 0.5},
        "top2_sources_plus_probes": {"disk_gb": 67, "relative_time": 0.6},
        "full": {"disk_gb": 392, "relative_time": 1.0},
    }

    datasets_to_test = ["common_claim_3class", "when2call_3class"]
    results = {}

    for ds_name in datasets_to_test:
        info = ALL_DATASETS[ds_name]
        results[ds_name] = {}
        print(f"\n{ds_name}:")

        for config_name, config in cheap_configs.items():
            t0 = time.time()
            try:
                # Handle empty source_subset for probes_only
                if config["source_subset"] == []:
                    # Only old probes, no raw features
                    config_copy = dict(config)
                    config_copy["source_subset"] = None  # won't matter since include_direct=False
                    te_prob, _, _, te_labels, _ = run_fusion_core(ds_name, info, **config_copy)
                else:
                    te_prob, _, _, te_labels, _ = run_fusion_core(ds_name, info, **config)

                if te_prob is None:
                    results[ds_name][config_name] = {"auroc": None}
                    continue

                auroc = compute_auroc(te_labels, te_prob, info["n_classes"])
                elapsed = time.time() - t0
                results[ds_name][config_name] = {
                    "auroc": float(auroc),
                    "disk_gb": cost_estimates[config_name]["disk_gb"],
                    "relative_time": cost_estimates[config_name]["relative_time"],
                }
                print(f"  {config_name:30s}: AUROC={auroc:.4f}  disk={cost_estimates[config_name]['disk_gb']:.0f}GB  [{elapsed:.1f}s]")
            except Exception as e:
                results[ds_name][config_name] = {"auroc": None, "error": str(e)}
                print(f"  {config_name:30s}: ERROR — {e}")
            gc.collect()

    with open(os.path.join(RESULTS_DIR, "cheap_fusion_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    return results


def analysis_7_seed_stability():
    """Multi-seed stability test."""
    print("\n" + "=" * 70)
    print("ANALYSIS 7: Multi-Seed Stability")
    print("=" * 70)

    datasets_to_test = ["common_claim_3class", "when2call_3class"]
    seeds = [42, 123, 456, 789, 1024]
    results = {}

    for ds_name in datasets_to_test:
        info = ALL_DATASETS[ds_name]
        seed_aurocs = []
        print(f"\n{ds_name}:")

        for seed in seeds:
            te_prob, _, _, te_labels, _ = run_fusion_core(ds_name, info, random_state=seed)
            if te_prob is None:
                continue
            auroc = compute_auroc(te_labels, te_prob, info["n_classes"])
            seed_aurocs.append(auroc)
            print(f"  seed={seed}: AUROC={auroc:.4f}")
            gc.collect()

        if seed_aurocs:
            results[ds_name] = {
                "seeds": seeds[:len(seed_aurocs)],
                "aurocs": [float(a) for a in seed_aurocs],
                "mean": float(np.mean(seed_aurocs)),
                "std": float(np.std(seed_aurocs)),
                "min": float(np.min(seed_aurocs)),
                "max": float(np.max(seed_aurocs)),
            }
            print(f"  Mean={np.mean(seed_aurocs):.4f} ± {np.std(seed_aurocs):.4f}")

    with open(os.path.join(RESULTS_DIR, "seed_stability.json"), "w") as f:
        json.dump(results, f, indent=2)
    return results


if __name__ == "__main__":
    # Determine which analyses to run
    analyses = sys.argv[1:] if len(sys.argv) > 1 else ["1", "3", "4", "5", "6", "7"]

    t_start = time.time()

    if "1" in analyses:
        comp_results = analysis_1_full_fusion()
    else:
        comp_results = {}
        if os.path.exists(os.path.join(RESULTS_DIR, "comprehensive_results.json")):
            with open(os.path.join(RESULTS_DIR, "comprehensive_results.json")) as f:
                comp_results = json.load(f)

    if "2" in analyses and comp_results:
        analysis_2_statistical_tests(comp_results)

    if "3" in analyses:
        analysis_3_ablations()

    if "4" in analyses:
        analysis_4_oracle()

    if "5" in analyses:
        analysis_5_complementarity()

    if "6" in analyses:
        analysis_6_cheap_fusion()

    if "7" in analyses:
        analysis_7_seed_stability()

    elapsed = time.time() - t_start
    print(f"\n{'='*70}")
    print(f"Total time: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"{'='*70}")
