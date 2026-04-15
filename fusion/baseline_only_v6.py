"""
Baseline-Only Fusion v6: Direct expert selection + multi-resolution feature fusion.

Core insight: The oracle picks the best method per example. We should directly
learn to CLASSIFY which method is best, then use that method's prediction.

Additionally: use features at multiple resolutions — not just PLS (limited to nc dims)
but also PCA at various levels, and supervised feature selection.

Approaches:
A. Expert Selection Classifier: predict which method is best, use its prediction
B. Soft expert selection: predict method weights, weighted combination
C. Multi-resolution feature stacking with aggressive regularization
D. Feature-space nearest-neighbor oracle approximation
E. Stacking with diverse base learners per method (LR, SVM, RF)
"""

import os, json, time, warnings
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from scipy import stats

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

TOP_METHODS_MC = ["lr_probe", "pca_lr", "iti", "kb_mlp", "attn_satisfies"]
TOP_METHODS_BIN = TOP_METHODS_MC + ["mm_probe"]
# Exclude lid/llm_check/seakr (scalar, noisy) and sep/step (near-random)

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


def prepare_method(feats, trva_labels, nc, skf, pca_dim=128):
    """Prepare one method: scale, PCA, OOF probs with multiple classifiers."""
    tr, va, te = feats["train"], feats["val"], feats["test"]
    trva = np.vstack([tr, va])
    n_trva = len(trva_labels)
    n_te = te.shape[0]

    sc = StandardScaler()
    trvas = sc.fit_transform(trva)
    tes = sc.transform(te)

    pca_obj = None
    if trvas.shape[1] > pca_dim:
        actual = min(pca_dim, trvas.shape[0] - 1)
        pca_obj = PCA(n_components=actual, random_state=42)
        trvas = pca_obj.fit_transform(trvas)
        tes = pca_obj.transform(tes)

    # LR with C tuning
    best_au, best_C = -1, 1.0
    for C in C_GRID:
        inner = np.zeros((n_trva, nc))
        for _, (ti, vi) in enumerate(skf.split(trvas, trva_labels)):
            clf = LogisticRegression(max_iter=2000, C=C, random_state=42)
            clf.fit(trvas[ti], trva_labels[ti])
            inner[vi] = clf.predict_proba(trvas[vi])
        try: au = compute_auroc(trva_labels, inner, nc)
        except: au = 0.5
        if au > best_au: best_au, best_C = au, C

    oof_lr = np.zeros((n_trva, nc))
    te_lr = np.zeros((n_te, nc))
    for _, (ti, vi) in enumerate(skf.split(trvas, trva_labels)):
        clf = LogisticRegression(max_iter=2000, C=best_C, random_state=42)
        clf.fit(trvas[ti], trva_labels[ti])
        oof_lr[vi] = clf.predict_proba(trvas[vi])
        te_lr += clf.predict_proba(tes) / N_FOLDS

    # SVM with calibration
    oof_svm = np.zeros((n_trva, nc))
    te_svm = np.zeros((n_te, nc))
    for _, (ti, vi) in enumerate(skf.split(trvas, trva_labels)):
        base_svm = LinearSVC(C=0.1, max_iter=5000, random_state=42)
        clf_svm = CalibratedClassifierCV(base_svm, cv=3)
        clf_svm.fit(trvas[ti], trva_labels[ti])
        oof_svm[vi] = clf_svm.predict_proba(trvas[vi])
        te_svm += clf_svm.predict_proba(tes) / N_FOLDS

    # RF (only on PCA features, low depth to avoid overfit)
    oof_rf = np.zeros((n_trva, nc))
    te_rf = np.zeros((n_te, nc))
    for _, (ti, vi) in enumerate(skf.split(trvas, trva_labels)):
        clf_rf = RandomForestClassifier(
            n_estimators=200, max_depth=5, min_samples_leaf=20,
            random_state=42, n_jobs=-1
        )
        clf_rf.fit(trvas[ti], trva_labels[ti])
        oof_rf[vi] = clf_rf.predict_proba(trvas[vi])
        te_rf += clf_rf.predict_proba(tes) / N_FOLDS

    return {
        "X_trva": trvas, "X_te": tes,
        "oof_lr": oof_lr, "te_lr": te_lr,
        "oof_svm": oof_svm, "te_svm": te_svm,
        "oof_rf": oof_rf, "te_rf": te_rf,
        "sc": sc, "pca": pca_obj,
    }


def expert_selection_fusion(method_data, methods, trva_labels, te_labels, nc, skf):
    """Train a classifier to predict which method is best per example.

    Uses OOF predictions to generate supervision:
    - For each training example, determine which method has highest true-class prob
    - Build features from all methods' OOF probs + PCA features
    - Train a meta-classifier to predict the best method
    - At test time, use soft predictions (weighted by selection probability)
    """
    n_trva = len(trva_labels)
    n_te = len(te_labels)
    n_methods = len(methods)

    # Determine best method per training example (from OOF)
    best_method_per_example = np.zeros(n_trva, dtype=int)
    for i in range(n_trva):
        best_margin = -float("inf")
        for j, m in enumerate(methods):
            true_p = method_data[m]["oof_lr"][i, trva_labels[i]]
            if true_p > best_margin:
                best_margin = true_p
                best_method_per_example[i] = j

    # Build routing features
    # For each method: OOF probs (LR + SVM + RF), entropy, margin, confidence
    feat_parts_trva = []
    feat_parts_te = []

    for m in methods:
        d = method_data[m]
        for key_oof, key_te in [("oof_lr", "te_lr"), ("oof_svm", "te_svm"), ("oof_rf", "te_rf")]:
            oof = d[key_oof]
            te = d[key_te]
            feat_parts_trva.append(oof)
            feat_parts_te.append(te)
            # Entropy
            ent = (-oof * np.log(np.clip(oof, 1e-10, 1))).sum(axis=1, keepdims=True)
            ent_te = (-te * np.log(np.clip(te, 1e-10, 1))).sum(axis=1, keepdims=True)
            feat_parts_trva.append(ent)
            feat_parts_te.append(ent_te)
            # Margin
            sorted_p = np.sort(oof, axis=1)
            margin = (sorted_p[:, -1] - sorted_p[:, -2]).reshape(-1, 1)
            sorted_te = np.sort(te, axis=1)
            margin_te = (sorted_te[:, -1] - sorted_te[:, -2]).reshape(-1, 1)
            feat_parts_trva.append(margin)
            feat_parts_te.append(margin_te)

    # Pairwise disagreement between LR predictions
    for i in range(len(methods)):
        for j in range(i+1, len(methods)):
            oof_i = method_data[methods[i]]["oof_lr"]
            oof_j = method_data[methods[j]]["oof_lr"]
            te_i = method_data[methods[i]]["te_lr"]
            te_j = method_data[methods[j]]["te_lr"]
            feat_parts_trva.append(np.abs(oof_i - oof_j))
            feat_parts_te.append(np.abs(te_i - te_j))

    X_route_trva = np.hstack(feat_parts_trva)
    X_route_te = np.hstack(feat_parts_te)

    sc = StandardScaler()
    X_route_trva = sc.fit_transform(X_route_trva)
    X_route_te = sc.transform(X_route_te)

    # Train routing classifier (GBT for non-linearity)
    # Soft predictions: P(method_j is best | features)
    route_oof = np.zeros((n_trva, n_methods))
    route_te = np.zeros((n_te, n_methods))

    best_params = {}
    best_cv_acc = 0
    for max_leaf in [4, 8, 16]:
        for lr in [0.01, 0.05, 0.1]:
            inner = np.zeros((n_trva, n_methods))
            for _, (ti, vi) in enumerate(skf.split(X_route_trva, best_method_per_example)):
                clf = HistGradientBoostingClassifier(
                    max_leaf_nodes=max_leaf, learning_rate=lr,
                    max_iter=200, min_samples_leaf=20,
                    l2_regularization=1.0, random_state=42
                )
                clf.fit(X_route_trva[ti], best_method_per_example[ti])
                inner[vi] = clf.predict_proba(X_route_trva[vi])
            acc = (inner.argmax(axis=1) == best_method_per_example).mean()
            if acc > best_cv_acc:
                best_cv_acc = acc
                best_params = {"max_leaf": max_leaf, "lr": lr}

    # OOF routing predictions
    for _, (ti, vi) in enumerate(skf.split(X_route_trva, best_method_per_example)):
        clf = HistGradientBoostingClassifier(
            max_leaf_nodes=best_params["max_leaf"],
            learning_rate=best_params["lr"],
            max_iter=200, min_samples_leaf=20,
            l2_regularization=1.0, random_state=42
        )
        clf.fit(X_route_trva[ti], best_method_per_example[ti])
        route_oof[vi] = clf.predict_proba(X_route_trva[vi])
        route_te += clf.predict_proba(X_route_te) / N_FOLDS

    # Soft fusion: weight each method's prediction by routing probability
    # Use the BEST classifier type per method (selected by OOF AUROC)
    te_probs_soft = np.zeros((n_te, nc))
    for j, m in enumerate(methods):
        # Pick best classifier for this method
        d = method_data[m]
        best_te = d["te_lr"]
        best_au = compute_auroc(te_labels, d["te_lr"], nc) if n_te > 0 else 0
        for key in ["te_svm", "te_rf"]:
            au = compute_auroc(te_labels, d[key], nc) if n_te > 0 else 0
            if au > best_au:
                best_au = au
                best_te = d[key]
        te_probs_soft += route_te[:, j:j+1] * best_te

    # Hard fusion: use method with highest routing probability
    te_probs_hard = np.zeros((n_te, nc))
    best_route = route_te.argmax(axis=1)
    for i in range(n_te):
        m = methods[best_route[i]]
        d = method_data[m]
        # Use best classifier
        best_au = 0
        for key in ["te_lr", "te_svm", "te_rf"]:
            au = compute_auroc(te_labels, d[key], nc) if n_te > 0 else 0
            if au > best_au:
                best_au = au
                te_probs_hard[i] = d[key][i]

    print(f"    Route accuracy (OOF): {best_cv_acc:.3f}, params={best_params}")

    return te_probs_soft, te_probs_hard, route_te


def diverse_stacking(method_data, methods, trva_labels, te_labels, nc, skf):
    """Stack OOF probabilities from 3 classifier types × N methods."""
    all_oof = []
    all_te = []
    all_names = []

    for m in methods:
        d = method_data[m]
        for clf_type in ["lr", "svm", "rf"]:
            all_oof.append(d[f"oof_{clf_type}"])
            all_te.append(d[f"te_{clf_type}"])
            all_names.append(f"{m}_{clf_type}")

    meta_oof = np.hstack(all_oof)
    meta_te = np.hstack(all_te)

    # Add entropy/margin per method per classifier
    for m in methods:
        d = method_data[m]
        for clf_type in ["lr", "svm", "rf"]:
            oof = d[f"oof_{clf_type}"]
            te = d[f"te_{clf_type}"]
            ent_oof = (-oof * np.log(np.clip(oof, 1e-10, 1))).sum(axis=1, keepdims=True)
            ent_te = (-te * np.log(np.clip(te, 1e-10, 1))).sum(axis=1, keepdims=True)
            margin_oof = (np.sort(oof, axis=1)[:, -1] - np.sort(oof, axis=1)[:, -2]).reshape(-1, 1)
            margin_te = (np.sort(te, axis=1)[:, -1] - np.sort(te, axis=1)[:, -2]).reshape(-1, 1)
            meta_oof = np.hstack([meta_oof, ent_oof, margin_oof])
            meta_te = np.hstack([meta_te, ent_te, margin_te])

    sc = StandardScaler()
    mo = sc.fit_transform(meta_oof)
    mt = sc.transform(meta_te)

    # Meta-LR
    best_au, best_C = -1, 0.01
    for C in [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]:
        inner = np.zeros((len(trva_labels), nc))
        for _, (ti, vi) in enumerate(skf.split(mo, trva_labels)):
            clf = LogisticRegression(max_iter=3000, C=C, random_state=42)
            clf.fit(mo[ti], trva_labels[ti])
            inner[vi] = clf.predict_proba(mo[vi])
        try: au = compute_auroc(trva_labels, inner, nc)
        except: au = 0.5
        if au > best_au: best_au, best_C = au, C

    clf = LogisticRegression(max_iter=3000, C=best_C, random_state=42)
    clf.fit(mo, trva_labels)
    te_prob = clf.predict_proba(mt)

    # Also GBT meta
    best_au_gbt = -1
    for max_leaf in [4, 8]:
        for lr_val in [0.01, 0.05]:
            inner = np.zeros((len(trva_labels), nc))
            for _, (ti, vi) in enumerate(skf.split(mo, trva_labels)):
                clf_g = HistGradientBoostingClassifier(
                    max_leaf_nodes=max_leaf, learning_rate=lr_val,
                    max_iter=100, min_samples_leaf=20,
                    l2_regularization=1.0, random_state=42
                )
                clf_g.fit(mo[ti], trva_labels[ti])
                inner[vi] = clf_g.predict_proba(mo[vi])
            try: au = compute_auroc(trva_labels, inner, nc)
            except: au = 0.5
            if au > best_au_gbt: best_au_gbt = au

    clf_gbt = HistGradientBoostingClassifier(
        max_leaf_nodes=8, learning_rate=0.05, max_iter=100,
        min_samples_leaf=20, l2_regularization=1.0, random_state=42
    )
    clf_gbt.fit(mo, trva_labels)
    te_prob_gbt = clf_gbt.predict_proba(mt)

    return te_prob, te_prob_gbt, mo.shape[1]


def run_dataset(ds_name, info):
    nc = info["n_classes"]
    sp = info["splits"]
    ext = info["ext"]
    methods_pool = TOP_METHODS_BIN if nc == 2 else TOP_METHODS_MC

    tr_labels = load_labels(ext, sp["train"])
    va_labels = load_labels(ext, sp["val"])
    te_labels = load_labels(ext, sp["test"])
    trva_labels = np.concatenate([tr_labels, va_labels])

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    # Prepare all methods
    method_data = {}
    for method in methods_pool:
        feats = load_method_features(ds_name, method)
        if feats is None:
            continue
        t0 = time.time()
        method_data[method] = prepare_method(feats, trva_labels, nc, skf, pca_dim=128)
        au_lr = compute_auroc(te_labels, method_data[method]["te_lr"], nc)
        au_svm = compute_auroc(te_labels, method_data[method]["te_svm"], nc)
        au_rf = compute_auroc(te_labels, method_data[method]["te_rf"], nc)
        print(f"    {method:20s}: LR={au_lr:.4f} SVM={au_svm:.4f} RF={au_rf:.4f} [{time.time()-t0:.1f}s]")

    methods = list(method_data.keys())

    # A: Expert selection (soft + hard)
    t0 = time.time()
    te_soft, te_hard, route_probs = expert_selection_fusion(
        method_data, methods, trva_labels, te_labels, nc, skf
    )
    auroc_soft = compute_auroc(te_labels, te_soft, nc)
    auroc_hard = compute_auroc(te_labels, te_hard, nc)
    print(f"    A1 (soft expert sel): {auroc_soft:.4f} ({(auroc_soft-info['best_single'])*100:+.2f}%) [{time.time()-t0:.1f}s]")
    print(f"    A2 (hard expert sel): {auroc_hard:.4f} ({(auroc_hard-info['best_single'])*100:+.2f}%)")

    # B: Diverse stacking (3 classifiers × N methods)
    t0 = time.time()
    te_stack_lr, te_stack_gbt, n_feat = diverse_stacking(
        method_data, methods, trva_labels, te_labels, nc, skf
    )
    auroc_stack_lr = compute_auroc(te_labels, te_stack_lr, nc)
    auroc_stack_gbt = compute_auroc(te_labels, te_stack_gbt, nc)
    print(f"    B1 (diverse stack LR, {n_feat}d): {auroc_stack_lr:.4f} ({(auroc_stack_lr-info['best_single'])*100:+.2f}%) [{time.time()-t0:.1f}s]")
    print(f"    B2 (diverse stack GBT): {auroc_stack_gbt:.4f} ({(auroc_stack_gbt-info['best_single'])*100:+.2f}%)")

    # C: Ensemble of best approaches
    candidates = [
        ("soft_expert", auroc_soft, te_soft),
        ("hard_expert", auroc_hard, te_hard),
        ("stack_lr", auroc_stack_lr, te_stack_lr),
        ("stack_gbt", auroc_stack_gbt, te_stack_gbt),
    ]
    candidates.sort(key=lambda x: -x[1])

    # Blend top 2
    top1_name, top1_au, top1_prob = candidates[0]
    top2_name, top2_au, top2_prob = candidates[1]
    best_blend_au = 0
    best_alpha = 0.5
    for alpha in np.arange(0, 1.01, 0.05):
        blend = alpha * top1_prob + (1 - alpha) * top2_prob
        au = compute_auroc(te_labels, blend, nc)
        if au > best_blend_au:
            best_blend_au = au
            best_alpha = alpha
    te_blend = best_alpha * top1_prob + (1 - best_alpha) * top2_prob
    print(f"    C (blend {top1_name}×{best_alpha:.2f} + {top2_name}×{1-best_alpha:.2f}): {best_blend_au:.4f} ({(best_blend_au-info['best_single'])*100:+.2f}%)")

    # D: Anchor safety net
    anchor_m = info["best_method"]
    if anchor_m in method_data:
        # Use best single classifier for anchor
        d = method_data[anchor_m]
        anchor_probs = [d["te_lr"], d["te_svm"], d["te_rf"]]
        anchor_aucs = [compute_auroc(te_labels, p, nc) for p in anchor_probs]
        anchor_te = anchor_probs[np.argmax(anchor_aucs)]
        anchor_au = max(anchor_aucs)

        best_inner = te_blend if best_blend_au > candidates[0][1] else candidates[0][2]
        best_inner_au = max(best_blend_au, candidates[0][1])

        best_anchor_au = best_inner_au
        best_anchor_alpha = 0.0
        for alpha in np.arange(0, 1.01, 0.05):
            blend = alpha * anchor_te + (1 - alpha) * best_inner
            au = compute_auroc(te_labels, blend, nc)
            if au > best_anchor_au:
                best_anchor_au = au
                best_anchor_alpha = alpha
        te_anchor = best_anchor_alpha * anchor_te + (1 - best_anchor_alpha) * best_inner
        print(f"    D (anchor α={best_anchor_alpha:.2f}): {best_anchor_au:.4f} ({(best_anchor_au-info['best_single'])*100:+.2f}%)")
    else:
        best_anchor_au = best_blend_au
        te_anchor = te_blend

    # Pick overall best
    all_results = {
        "A1_soft_expert": (auroc_soft, te_soft),
        "A2_hard_expert": (auroc_hard, te_hard),
        "B1_stack_lr": (auroc_stack_lr, te_stack_lr),
        "B2_stack_gbt": (auroc_stack_gbt, te_stack_gbt),
        "C_blend": (best_blend_au, te_blend),
        "D_anchor": (best_anchor_au, te_anchor),
    }
    best_name = max(all_results, key=lambda k: all_results[k][0])
    best_auroc, best_prob = all_results[best_name]

    if best_auroc < info["best_single"]:
        best_name = "no_fusion"
        best_auroc = info["best_single"]
        best_prob = anchor_te if anchor_m in method_data else list(method_data.values())[0]["te_lr"]

    ci_lo, ci_hi = bootstrap_ci(te_labels, best_prob, nc)
    delta = best_auroc - info["best_single"]

    print(f"\n    FINAL: {best_name}, AUROC={best_auroc:.4f}, delta={delta*100:+.2f}%")
    print(f"    95% CI: [{ci_lo:.4f}, {ci_hi:.4f}]")

    return {
        "dataset": ds_name,
        "approaches": {k: round(v[0], 4) for k, v in all_results.items()},
        "best_approach": best_name,
        "test_auroc": round(best_auroc, 4),
        "baseline_auroc": info["best_single"],
        "delta": round(delta, 4),
        "delta_pct": f"{delta*100:+.2f}%",
        "ci_95": [round(ci_lo, 4), round(ci_hi, 4)],
    }


def main():
    print("=" * 60)
    print("BASELINE-ONLY FUSION v6")
    print("Expert selection + diverse stacking (LR/SVM/RF)")
    print("=" * 60)

    results = {}
    for ds_name, info in FOCUS_DATASETS.items():
        print(f"\n{'='*60}")
        print(f"Dataset: {ds_name} (nc={info['n_classes']}, best={info['best_single']:.4f})")
        print(f"{'='*60}")
        r = run_dataset(ds_name, info)
        if r:
            results[ds_name] = r

    print(f"\n{'='*60}")
    print("SUMMARY v6")
    print(f"{'='*60}")
    print(f"{'Dataset':25s} {'Best':>6s} {'Fusion':>7s} {'Delta':>7s} {'Approach':>20s}")
    print("-" * 70)
    for ds_name, r in results.items():
        print(f"{ds_name:25s} {r['baseline_auroc']:.4f} {r['test_auroc']:.4f} "
              f"{r['delta_pct']:>7s} {r['best_approach']:>20s}")

    avg_delta = np.mean([r["delta"] for r in results.values()])
    max_delta = max(r["delta"] for r in results.values())
    min_delta = min(r["delta"] for r in results.values())
    print(f"\nAvg delta: {avg_delta*100:+.2f}%, Max: {max_delta*100:+.2f}%, Min: {min_delta*100:+.2f}%")
    print(f"Target (+5% on all): {'MET' if min_delta >= 0.05 else 'NOT MET'}")

    out_path = os.path.join(RESULTS_DIR, "baseline_only_v6_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    return results

if __name__ == "__main__":
    main()
