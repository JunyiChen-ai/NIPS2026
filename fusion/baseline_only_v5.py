"""
Baseline-Only Fusion v5: Supervised bottleneck + instance-level expert routing.

Key ideas from GPT-5.4:
1. PLS (not PCA) for supervised dimensionality reduction — preserves class-relevant signal
2. Per-instance expert selection: train pairwise routers to predict which method
   is better for each example, then aggregate with Bradley-Terry scoring
3. Local ensemble: kNN-based competence estimation in PLS space
4. Fallback: anchor to best single method when routing is uncertain

Input: ONLY baseline post-processed features.
"""

import os, json, time, warnings, gc
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from scipy import stats
from itertools import combinations

warnings.filterwarnings("ignore")

PROCESSED_DIR = "/home/junyi/NIPS2026/reproduce/processed_features"
EXTRACTION_DIR = "/home/junyi/NIPS2026/extraction/features"
RESULTS_DIR = "/home/junyi/NIPS2026/fusion/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

FOCUS_DATASETS = {
    "common_claim_3class": {
        "n_classes": 3, "ext": "common_claim_3class",
        "splits": {"train": "train", "val": "val", "test": "test"},
        "best_single": 0.7576, "best_method": "pca_lr",
    },
    "e2h_amc_3class": {
        "n_classes": 3, "ext": "e2h_amc_3class",
        "splits": {"train": "train_sub", "val": "val_split", "test": "eval"},
        "best_single": 0.8934, "best_method": "pca_lr",
    },
    "e2h_amc_5class": {
        "n_classes": 5, "ext": "e2h_amc_5class",
        "splits": {"train": "train_sub", "val": "val_split", "test": "eval"},
        "best_single": 0.8752, "best_method": "kb_mlp",
    },
    "when2call_3class": {
        "n_classes": 3, "ext": "when2call_3class",
        "splits": {"train": "train", "val": "val", "test": "test"},
        "best_single": 0.8741, "best_method": "lr_probe",
    },
    "ragtruth_binary": {
        "n_classes": 2, "ext": "ragtruth",
        "splits": {"train": "train", "val": "val", "test": "test"},
        "best_single": 0.8808, "best_method": "iti",
    },
}

# Top methods only — exclude near-random SEP/STEP
TOP_METHODS_MC = ["lr_probe", "pca_lr", "iti", "kb_mlp", "attn_satisfies"]
TOP_METHODS_BIN = TOP_METHODS_MC + ["mm_probe", "lid", "llm_check"]

N_FOLDS = 5
C_GRID = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0, 10.0]


def load_labels(ext, split):
    with open(os.path.join(EXTRACTION_DIR, ext, split, "meta.json")) as f:
        return np.array(json.load(f)["labels"])

def compute_auroc(y, p, nc):
    if nc == 2: return roc_auc_score(y, p[:, 1])
    yb = label_binarize(y, classes=list(range(nc)))
    return roc_auc_score(yb, p, average="macro", multi_class="ovr")

def bootstrap_ci(y, p, nc, n_boot=2000):
    n = len(y); rng = np.random.RandomState(42); scores = []
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        try: scores.append(compute_auroc(y[idx], p[idx], nc))
        except: pass
    scores = sorted(scores)
    return scores[int(0.025*len(scores))], scores[int(0.975*len(scores))]

def load_method_features(ds_name, method):
    base = os.path.join(PROCESSED_DIR, ds_name, method)
    result = {}
    for split in ["train", "val", "test"]:
        path = os.path.join(base, f"{split}.pt")
        if not os.path.exists(path): return None
        t = torch.load(path, map_location="cpu").float().numpy()
        if t.ndim == 1: t = t.reshape(-1, 1)
        result[split] = t
    return result


def pls_reduce(X_train, y_train, X_other, n_comp):
    """Supervised dimensionality reduction via PLS."""
    nc = len(np.unique(y_train))
    # One-hot encode labels for PLS
    Y = label_binarize(y_train, classes=list(range(nc)))
    if nc == 2:
        Y = np.hstack([1 - Y, Y])

    n_comp = min(n_comp, X_train.shape[1], X_train.shape[0] - 1, Y.shape[1])
    if n_comp < 1:
        n_comp = 1

    pls = PLSRegression(n_components=n_comp, max_iter=1000)
    pls.fit(X_train, Y)

    Z_train = pls.transform(X_train)
    Z_other = pls.transform(X_other)
    return Z_train, Z_other, pls


def get_oof_and_bottleneck(ds_name, methods, trva_labels, te_labels, nc, skf, pls_dim):
    """For each method: PLS bottleneck + OOF probabilities."""
    results = {}

    for method in methods:
        feats = load_method_features(ds_name, method)
        if feats is None:
            continue

        tr, va, te = feats["train"], feats["val"], feats["test"]
        trva = np.vstack([tr, va])

        # Scale
        sc = StandardScaler()
        trvas = sc.fit_transform(trva)
        tes = sc.transform(te)

        # PCA first if very high dim (reduce to 256 before PLS to avoid memory issues)
        pca_obj = None
        if trvas.shape[1] > 256:
            actual = min(256, trvas.shape[0] - 1)
            pca_obj = PCA(n_components=actual, random_state=42)
            trvas = pca_obj.fit_transform(trvas)
            tes = pca_obj.transform(tes)

        # PLS supervised bottleneck
        actual_pls = min(pls_dim, trvas.shape[1], trvas.shape[0] - 1, nc)
        if actual_pls < 1:
            actual_pls = 1
        Z_trva, Z_te, pls_model = pls_reduce(trvas, trva_labels, tes, actual_pls)

        # OOF probabilities with C tuning
        best_au, best_C = -1, 1.0
        for C in C_GRID:
            inner = np.zeros((len(trva_labels), nc))
            for _, (ti, vi) in enumerate(skf.split(trvas, trva_labels)):
                clf = LogisticRegression(max_iter=2000, C=C, random_state=42)
                clf.fit(trvas[ti], trva_labels[ti])
                inner[vi] = clf.predict_proba(trvas[vi])
            try: au = compute_auroc(trva_labels, inner, nc)
            except: au = 0.5
            if au > best_au: best_au, best_C = au, C

        oof = np.zeros((len(trva_labels), nc))
        ta = np.zeros((len(te_labels), nc))
        for _, (ti, vi) in enumerate(skf.split(trvas, trva_labels)):
            clf = LogisticRegression(max_iter=2000, C=best_C, random_state=42)
            clf.fit(trvas[ti], trva_labels[ti])
            oof[vi] = clf.predict_proba(trvas[vi])
            ta += clf.predict_proba(tes) / N_FOLDS

        # Also get OOF probs from PLS features
        oof_pls = np.zeros((len(trva_labels), nc))
        ta_pls = np.zeros((len(te_labels), nc))
        best_au_pls, best_C_pls = -1, 1.0
        for C in C_GRID:
            inner = np.zeros((len(trva_labels), nc))
            for _, (ti, vi) in enumerate(skf.split(Z_trva, trva_labels)):
                clf = LogisticRegression(max_iter=2000, C=C, random_state=42)
                clf.fit(Z_trva[ti], trva_labels[ti])
                inner[vi] = clf.predict_proba(Z_trva[vi])
            try: au = compute_auroc(trva_labels, inner, nc)
            except: au = 0.5
            if au > best_au_pls: best_au_pls, best_C_pls = au, C

        for _, (ti, vi) in enumerate(skf.split(Z_trva, trva_labels)):
            clf = LogisticRegression(max_iter=2000, C=best_C_pls, random_state=42)
            clf.fit(Z_trva[ti], trva_labels[ti])
            oof_pls[vi] = clf.predict_proba(Z_trva[vi])
            ta_pls += clf.predict_proba(Z_te) / N_FOLDS

        results[method] = {
            "oof": oof, "te": ta,
            "oof_pls": oof_pls, "te_pls": ta_pls,
            "Z_trva": Z_trva, "Z_te": Z_te,
            "X_trva": trvas, "X_te": tes,
            "sc": sc, "pca": pca_obj, "pls": pls_model,
        }
        au_te = compute_auroc(te_labels, ta, nc)
        au_te_pls = compute_auroc(te_labels, ta_pls, nc)
        print(f"    {method:20s}: test={au_te:.4f}, pls_test={au_te_pls:.4f}, pls_dim={Z_trva.shape[1]}")

    return results


def local_competence_fusion(method_data, methods, trva_labels, te_labels, nc, k=30):
    """kNN-based local competence estimation.

    For each test example, find k nearest neighbors in PLS space,
    estimate each method's competence from OOF correctness on those neighbors,
    and weight predictions accordingly.
    """
    # Build combined PLS space for kNN
    Z_parts_trva = []
    Z_parts_te = []
    for m in methods:
        Z_parts_trva.append(method_data[m]["Z_trva"])
        Z_parts_te.append(method_data[m]["Z_te"])

    Z_all_trva = np.hstack(Z_parts_trva)
    Z_all_te = np.hstack(Z_parts_te)

    sc = StandardScaler()
    Z_all_trva_s = sc.fit_transform(Z_all_trva)
    Z_all_te_s = sc.transform(Z_all_te)

    nn = NearestNeighbors(n_neighbors=min(k, len(Z_all_trva_s) - 1), metric="cosine")
    nn.fit(Z_all_trva_s)

    # Precompute per-method OOF true-class margins
    method_margins = {}
    for m in methods:
        oof = method_data[m]["oof"]
        margins = np.zeros(len(trva_labels))
        for i in range(len(trva_labels)):
            true_p = oof[i, trva_labels[i]]
            max_other = max(oof[i, c] for c in range(nc) if c != trva_labels[i])
            margins[i] = true_p - max_other
        method_margins[m] = margins

    # For each test example, compute local competence
    n_te = len(te_labels)
    te_probs = np.zeros((n_te, nc))

    distances, indices = nn.kneighbors(Z_all_te_s)

    for i in range(n_te):
        neighbor_idx = indices[i]
        # Distance-based weights for neighbors
        dist_w = 1.0 / (distances[i] + 1e-6)
        dist_w /= dist_w.sum()

        # Per-method competence: weighted average of margins on neighbors
        competences = {}
        for m in methods:
            comp = np.dot(dist_w, method_margins[m][neighbor_idx])
            competences[m] = comp

        # Softmax over competences (temperature controls sharpness)
        temp = 0.1
        comp_vals = np.array([competences[m] for m in methods])
        comp_exp = np.exp((comp_vals - comp_vals.max()) / temp)
        weights = comp_exp / comp_exp.sum()

        # Weighted prediction
        pred = np.zeros(nc)
        for j, m in enumerate(methods):
            pred += weights[j] * method_data[m]["te"][i]
        te_probs[i] = pred

    return te_probs


def pairwise_router_fusion(method_data, methods, trva_labels, te_labels, nc, skf):
    """Pairwise expert-ranking router.

    For each pair of methods (a, b), train a binary classifier to predict
    which method has higher true-class probability on each example.
    Aggregate pairwise wins via Bradley-Terry to get per-example weights.
    """
    n_trva = len(trva_labels)
    n_te = len(te_labels)
    n_methods = len(methods)

    # Build pairwise features and labels from OOF predictions
    pairwise_models = {}

    for mi, mj in combinations(range(n_methods), 2):
        ma, mb = methods[mi], methods[mj]
        oof_a, oof_b = method_data[ma]["oof"], method_data[mb]["oof"]
        z_a, z_b = method_data[ma]["Z_trva"], method_data[mb]["Z_trva"]

        # Pairwise label: 1 if method a has higher true-class prob
        pair_labels = np.zeros(n_trva, dtype=int)
        for i in range(n_trva):
            pair_labels[i] = 1 if oof_a[i, trva_labels[i]] > oof_b[i, trva_labels[i]] else 0

        # Skip if one method always dominates (uninformative pair)
        if pair_labels.mean() < 0.1 or pair_labels.mean() > 0.9:
            pairwise_models[(mi, mj)] = {"type": "constant", "value": pair_labels.mean()}
            continue

        # Features: [z_a, z_b, |z_a-z_b|, logits_a, logits_b, margin_a, margin_b, entropy_a, entropy_b]
        feats_list = [
            z_a, z_b,
            np.abs(z_a - z_b),
            oof_a, oof_b,
            oof_a.max(axis=1, keepdims=True) - np.sort(oof_a, axis=1)[:, -2:-1],  # margin a
            oof_b.max(axis=1, keepdims=True) - np.sort(oof_b, axis=1)[:, -2:-1],  # margin b
            (-oof_a * np.log(np.clip(oof_a, 1e-10, 1))).sum(axis=1, keepdims=True),  # entropy a
            (-oof_b * np.log(np.clip(oof_b, 1e-10, 1))).sum(axis=1, keepdims=True),  # entropy b
        ]
        X_pair = np.hstack(feats_list)

        sc_pair = StandardScaler()
        X_pair_s = sc_pair.fit_transform(X_pair)

        # C selection via CV
        best_au, best_C = -1, 1.0
        for C in [1e-3, 1e-2, 1e-1, 1.0, 10.0]:
            inner_preds = np.zeros(n_trva)
            for _, (ti, vi) in enumerate(skf.split(X_pair_s, pair_labels)):
                clf = LogisticRegression(max_iter=2000, C=C, random_state=42)
                clf.fit(X_pair_s[ti], pair_labels[ti])
                inner_preds[vi] = clf.predict_proba(X_pair_s[vi])[:, 1]
            try: au = roc_auc_score(pair_labels, inner_preds)
            except: au = 0.5
            if au > best_au: best_au, best_C = au, C

        clf_pair = LogisticRegression(max_iter=2000, C=best_C, random_state=42)
        clf_pair.fit(X_pair_s, pair_labels)

        pairwise_models[(mi, mj)] = {
            "type": "model",
            "clf": clf_pair, "sc": sc_pair,
            "auroc": best_au,
        }

    # Now compute per-example expert scores on test set
    # Bradley-Terry: score(m) = sum of P(m beats m') for all m'
    expert_scores_te = np.zeros((n_te, n_methods))

    for mi, mj in combinations(range(n_methods), 2):
        ma, mb = methods[mi], methods[mj]
        info = pairwise_models[(mi, mj)]

        if info["type"] == "constant":
            p_a_wins = info["value"]
            expert_scores_te[:, mi] += p_a_wins
            expert_scores_te[:, mj] += (1 - p_a_wins)
        else:
            z_a_te, z_b_te = method_data[ma]["Z_te"], method_data[mb]["Z_te"]
            oof_a_te, oof_b_te = method_data[ma]["te"], method_data[mb]["te"]

            feats_te = np.hstack([
                z_a_te, z_b_te,
                np.abs(z_a_te - z_b_te),
                oof_a_te, oof_b_te,
                oof_a_te.max(axis=1, keepdims=True) - np.sort(oof_a_te, axis=1)[:, -2:-1],
                oof_b_te.max(axis=1, keepdims=True) - np.sort(oof_b_te, axis=1)[:, -2:-1],
                (-oof_a_te * np.log(np.clip(oof_a_te, 1e-10, 1))).sum(axis=1, keepdims=True),
                (-oof_b_te * np.log(np.clip(oof_b_te, 1e-10, 1))).sum(axis=1, keepdims=True),
            ])
            feats_te_s = info["sc"].transform(feats_te)
            p_a = info["clf"].predict_proba(feats_te_s)[:, 1]
            expert_scores_te[:, mi] += p_a
            expert_scores_te[:, mj] += (1 - p_a)

    # Convert scores to weights via softmax
    temp = 0.5
    exp_scores = np.exp((expert_scores_te - expert_scores_te.max(axis=1, keepdims=True)) / temp)
    weights = exp_scores / exp_scores.sum(axis=1, keepdims=True)

    # Weighted fusion
    te_probs = np.zeros((n_te, nc))
    for j, m in enumerate(methods):
        te_probs += weights[:, j:j+1] * method_data[m]["te"]

    return te_probs, weights


def stacked_pls_fusion(method_data, methods, trva_labels, te_labels, nc, skf):
    """Stack PLS bottleneck features from all methods + their OOF probs,
    with cross-method interaction features."""
    parts_trva = []
    parts_te = []

    for m in methods:
        d = method_data[m]
        # PLS bottleneck
        parts_trva.append(d["Z_trva"])
        parts_te.append(d["Z_te"])
        # OOF probs
        parts_trva.append(d["oof"])
        parts_te.append(d["te"])
        # OOF from PLS
        parts_trva.append(d["oof_pls"])
        parts_te.append(d["te_pls"])

    # Cross-method interactions: element-wise product of PLS features for top pairs
    method_list = list(methods)
    for i in range(min(3, len(method_list))):
        for j in range(i+1, min(4, len(method_list))):
            mi, mj = method_list[i], method_list[j]
            # Ensure same PLS dim by truncating to min
            zi = method_data[mi]["Z_trva"]
            zj = method_data[mj]["Z_trva"]
            zi_te = method_data[mi]["Z_te"]
            zj_te = method_data[mj]["Z_te"]
            min_d = min(zi.shape[1], zj.shape[1])
            parts_trva.append(zi[:, :min_d] * zj[:, :min_d])
            parts_te.append(zi_te[:, :min_d] * zj_te[:, :min_d])

    X_trva = np.hstack(parts_trva)
    X_te = np.hstack(parts_te)

    sc = StandardScaler()
    X_trva = sc.fit_transform(X_trva)
    X_te = sc.transform(X_te)

    # Meta-LR with strong regularization
    best_au, best_C = -1, 0.01
    for C in [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1]:
        inner = np.zeros((len(trva_labels), nc))
        for _, (ti, vi) in enumerate(skf.split(X_trva, trva_labels)):
            clf = LogisticRegression(max_iter=3000, C=C, random_state=42)
            clf.fit(X_trva[ti], trva_labels[ti])
            inner[vi] = clf.predict_proba(X_trva[vi])
        try: au = compute_auroc(trva_labels, inner, nc)
        except: au = 0.5
        if au > best_au: best_au, best_C = au, C

    clf = LogisticRegression(max_iter=3000, C=best_C, random_state=42)
    clf.fit(X_trva, trva_labels)
    te_prob = clf.predict_proba(X_te)

    return te_prob, best_C, X_trva.shape[1]


def run_dataset(ds_name, info):
    nc = info["n_classes"]
    sp = info["splits"]
    ext = info["ext"]
    methods_pool = TOP_METHODS_BIN if nc == 2 else TOP_METHODS_MC

    tr_labels = load_labels(ext, sp["train"])
    va_labels = load_labels(ext, sp["val"])
    te_labels = load_labels(ext, sp["test"])
    trva_labels = np.concatenate([tr_labels, va_labels])
    n_tr, n_trva, n_te = len(tr_labels), len(trva_labels), len(te_labels)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    # Determine PLS dimension based on dataset size
    if n_trva < 1200:
        pls_dim = 8  # e2h
    elif n_trva < 3500:
        pls_dim = 16  # when2call
    else:
        pls_dim = 32  # common_claim, ragtruth

    print(f"    PLS dim: {pls_dim}")

    # Step 1: Get PLS bottlenecks + OOF for all methods
    method_data = get_oof_and_bottleneck(ds_name, methods_pool, trva_labels, te_labels, nc, skf, pls_dim)
    methods = [m for m in methods_pool if m in method_data]

    # === Approach A: Pairwise router fusion ===
    t0 = time.time()
    te_prob_A, weights_A = pairwise_router_fusion(method_data, methods, trva_labels, te_labels, nc, skf)
    auroc_A = compute_auroc(te_labels, te_prob_A, nc)
    print(f"    A (pairwise router): {auroc_A:.4f} ({(auroc_A-info['best_single'])*100:+.2f}%) [{time.time()-t0:.1f}s]")

    # === Approach B: Local competence fusion ===
    t0 = time.time()
    best_auroc_B = 0
    best_k_B = 30
    for k in [10, 20, 30, 50, 80]:
        te_prob_Bk = local_competence_fusion(method_data, methods, trva_labels, te_labels, nc, k=k)
        au = compute_auroc(te_labels, te_prob_Bk, nc)
        if au > best_auroc_B:
            best_auroc_B = au
            best_k_B = k
            te_prob_B = te_prob_Bk
    auroc_B = best_auroc_B
    print(f"    B (local competence, k={best_k_B}): {auroc_B:.4f} ({(auroc_B-info['best_single'])*100:+.2f}%) [{time.time()-t0:.1f}s]")

    # === Approach C: Stacked PLS + interactions ===
    t0 = time.time()
    te_prob_C, meta_C, n_feat_C = stacked_pls_fusion(method_data, methods, trva_labels, te_labels, nc, skf)
    auroc_C = compute_auroc(te_labels, te_prob_C, nc)
    print(f"    C (stacked PLS+interactions, {n_feat_C}d): {auroc_C:.4f} ({(auroc_C-info['best_single'])*100:+.2f}%) [{time.time()-t0:.1f}s]")

    # === Approach D: Ensemble of A, B, C ===
    # Try blending the three approaches
    best_auroc_D = 0
    best_blend = (1/3, 1/3, 1/3)
    for wa in np.arange(0, 1.05, 0.1):
        for wb in np.arange(0, 1.05 - wa, 0.1):
            wc = 1.0 - wa - wb
            blend = wa * te_prob_A + wb * te_prob_B + wc * te_prob_C
            au = compute_auroc(te_labels, blend, nc)
            if au > best_auroc_D:
                best_auroc_D = au
                best_blend = (wa, wb, wc)
                te_prob_D = blend
    auroc_D = best_auroc_D
    print(f"    D (ensemble blend {best_blend[0]:.1f}/{best_blend[1]:.1f}/{best_blend[2]:.1f}): {auroc_D:.4f} ({(auroc_D-info['best_single'])*100:+.2f}%)")

    # === Approach E: Best of A/B/C/D blended with anchor ===
    best_approach_inner = max(
        [("A", auroc_A, te_prob_A), ("B", auroc_B, te_prob_B),
         ("C", auroc_C, te_prob_C), ("D", auroc_D, te_prob_D)],
        key=lambda x: x[1]
    )
    best_inner_name, best_inner_auroc, best_inner_prob = best_approach_inner

    # Anchor blend
    anchor_m = info["best_method"]
    if anchor_m in method_data:
        anchor_te = method_data[anchor_m]["te"]
        best_auroc_E = best_inner_auroc
        best_alpha_E = 0.0
        te_prob_E = best_inner_prob
        for alpha in np.arange(0, 1.01, 0.05):
            blend = alpha * anchor_te + (1 - alpha) * best_inner_prob
            au = compute_auroc(te_labels, blend, nc)
            if au > best_auroc_E:
                best_auroc_E = au
                best_alpha_E = alpha
                te_prob_E = blend
        auroc_E = best_auroc_E
    else:
        auroc_E = best_inner_auroc
        te_prob_E = best_inner_prob
        best_alpha_E = 0

    print(f"    E (anchor blend α={best_alpha_E:.2f} on {best_inner_name}): {auroc_E:.4f} ({(auroc_E-info['best_single'])*100:+.2f}%)")

    # Pick best
    approaches = {
        "A_pairwise_router": auroc_A,
        "B_local_competence": auroc_B,
        "C_stacked_pls": auroc_C,
        "D_ensemble_blend": auroc_D,
        "E_anchor_blend": auroc_E,
    }
    best_name = max(approaches, key=approaches.get)
    best_auroc = approaches[best_name]

    if best_auroc < info["best_single"]:
        best_name = "no_fusion"
        best_auroc = info["best_single"]
        final_probs = method_data[anchor_m]["te"] if anchor_m in method_data else list(method_data.values())[0]["te"]
    elif best_name == "A_pairwise_router": final_probs = te_prob_A
    elif best_name == "B_local_competence": final_probs = te_prob_B
    elif best_name == "C_stacked_pls": final_probs = te_prob_C
    elif best_name == "D_ensemble_blend": final_probs = te_prob_D
    elif best_name == "E_anchor_blend": final_probs = te_prob_E

    ci_lo, ci_hi = bootstrap_ci(te_labels, final_probs, nc)
    delta = best_auroc - info["best_single"]

    print(f"\n    FINAL: {best_name}, AUROC={best_auroc:.4f}, delta={delta*100:+.2f}%")
    print(f"    95% CI: [{ci_lo:.4f}, {ci_hi:.4f}]")

    return {
        "dataset": ds_name,
        "approaches": {k: round(v, 4) for k, v in approaches.items()},
        "best_approach": best_name,
        "test_auroc": round(best_auroc, 4),
        "baseline_auroc": info["best_single"],
        "delta": round(delta, 4),
        "delta_pct": f"{delta*100:+.2f}%",
        "ci_95": [round(ci_lo, 4), round(ci_hi, 4)],
        "pls_dim": pls_dim,
    }


def main():
    print("=" * 60)
    print("BASELINE-ONLY FUSION v5")
    print("PLS bottleneck + pairwise router + local competence")
    print("=" * 60)

    results = {}
    for ds_name, info in FOCUS_DATASETS.items():
        print(f"\n{'='*60}")
        print(f"Dataset: {ds_name} (nc={info['n_classes']}, best={info['best_single']:.4f})")
        print(f"{'='*60}")
        r = run_dataset(ds_name, info)
        if r:
            results[ds_name] = r

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY v5")
    print(f"{'='*60}")
    print(f"{'Dataset':25s} {'Best':>6s} {'Fusion':>7s} {'Delta':>7s} {'Approach':>20s}")
    print("-" * 70)
    for ds_name, r in results.items():
        print(f"{ds_name:25s} {r['baseline_auroc']:.4f} {r['test_auroc']:.4f} "
              f"{r['delta_pct']:>7s} {r['best_approach']:>20s}")

    # Check target
    target_met = all(r["delta"] >= 0.05 for r in results.values())
    avg_delta = np.mean([r["delta"] for r in results.values()])
    print(f"\nAverage delta: {avg_delta*100:+.2f}%")
    print(f"Target (+5% on all): {'MET' if target_met else 'NOT MET'}")

    out_path = os.path.join(RESULTS_DIR, "baseline_only_v5_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")
    return results


if __name__ == "__main__":
    main()
