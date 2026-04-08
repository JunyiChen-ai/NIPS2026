"""
Anchor-Residual Fusion: Conservative score-level fusion with anchor fallback.

Strategy:
1. Pick anchor probe via lower-confidence-bound CV score
2. Convex ensemble over probe probs with fold-DRO
3. Conservative gate: blend anchor and ensemble based on disagreement features
4. ProbeDrop regularization

All experiments in one script, building up from simplest to full model.
"""

import os, json, time, warnings
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, log_loss
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from scipy.optimize import minimize
from scipy.special import softmax

warnings.filterwarnings("ignore")

PROCESSED_DIR = "/home/junyi/NIPS2026/reproduce/processed_features"
EXTRACTION_DIR = "/home/junyi/NIPS2026/extraction/features"

MULTICLASS_METHODS = ["lr_probe", "pca_lr", "iti", "kb_mlp", "attn_satisfies", "sep", "step"]

DATASETS = {
    "common_claim_3class": {"n_classes": 3, "splits": {"train": "train", "val": "val", "test": "test"}},
    "e2h_amc_3class":      {"n_classes": 3, "splits": {"train": "train_sub", "val": "val_split", "test": "eval"}},
    "e2h_amc_5class":      {"n_classes": 5, "splits": {"train": "train_sub", "val": "val_split", "test": "eval"}},
    "when2call_3class":     {"n_classes": 3, "splits": {"train": "train", "val": "val", "test": "test"}},
}

BASELINES = {
    "common_claim_3class": ("pca_lr", 0.7576),
    "e2h_amc_3class":      ("pca_lr", 0.8934),
    "e2h_amc_5class":      ("kb_mlp", 0.8752),
    "when2call_3class":     ("lr_probe", 0.8741),
}

N_FOLDS = 5


def load_labels(dataset, split):
    raw_split = DATASETS[dataset]["splits"][split]
    meta_path = os.path.join(EXTRACTION_DIR, dataset, raw_split, "meta.json")
    with open(meta_path) as f:
        return np.array(json.load(f)["labels"])


def load_and_reduce(dataset, methods, proj_dim=128):
    data = {}
    for split in ["train", "val", "test"]:
        data[split] = {"labels": load_labels(dataset, split), "feats": {}}

    for method in methods:
        train_path = os.path.join(PROCESSED_DIR, dataset, method, "train.pt")
        if not os.path.exists(train_path):
            continue
        tr = torch.load(train_path, map_location="cpu").float().numpy()
        va = torch.load(os.path.join(PROCESSED_DIR, dataset, method, "val.pt"), map_location="cpu").float().numpy()
        te = torch.load(os.path.join(PROCESSED_DIR, dataset, method, "test.pt"), map_location="cpu").float().numpy()
        if tr.ndim == 1:
            tr, va, te = tr.reshape(-1, 1), va.reshape(-1, 1), te.reshape(-1, 1)
        sc = StandardScaler()
        tr = sc.fit_transform(tr)
        va = sc.transform(va)
        te = sc.transform(te)
        if tr.shape[1] > proj_dim:
            pca = PCA(n_components=proj_dim, random_state=42)
            tr = pca.fit_transform(tr)
            va = pca.transform(va)
            te = pca.transform(te)
        data["train"]["feats"][method] = tr
        data["val"]["feats"][method] = va
        data["test"]["feats"][method] = te
    return data


def compute_auroc(y_true, y_prob, n_classes):
    if n_classes == 2:
        return roc_auc_score(y_true, y_prob[:, 1])
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))
    return roc_auc_score(y_bin, y_prob, average="macro", multi_class="ovr")


def eval_metrics(y_true, y_prob, n_classes):
    y_pred = y_prob.argmax(axis=1)
    return {
        "auroc": compute_auroc(y_true, y_prob, n_classes),
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
    }


def get_cv_probs(data, n_classes, n_folds=N_FOLDS):
    """Generate OOF probs and test probs for each method via CV."""
    methods = list(data["train"]["feats"].keys())
    all_labels = np.concatenate([data["train"]["labels"], data["val"]["labels"]])
    all_feats = {m: np.vstack([data["train"]["feats"][m], data["val"]["feats"][m]]) for m in methods}
    n_total = len(all_labels)
    n_test = len(data["test"]["labels"])

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    oof_probs = {m: np.zeros((n_total, n_classes)) for m in methods}
    test_probs = {m: np.zeros((n_test, n_classes)) for m in methods}
    fold_aurocs = {m: [] for m in methods}

    for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(all_feats[methods[0]], all_labels)):
        for m in methods:
            clf = LogisticRegression(max_iter=2000, random_state=42, C=1.0)
            clf.fit(all_feats[m][tr_idx], all_labels[tr_idx])
            oof_probs[m][va_idx] = clf.predict_proba(all_feats[m][va_idx])
            test_probs[m] += clf.predict_proba(data["test"]["feats"][m]) / n_folds
            # Per-fold AUROC
            fold_auroc = compute_auroc(all_labels[va_idx], oof_probs[m][va_idx], n_classes)
            fold_aurocs[m].append(fold_auroc)

    return methods, all_labels, oof_probs, test_probs, fold_aurocs


def select_anchor(fold_aurocs, lam=1.0):
    """Select anchor probe with best lower confidence bound."""
    lcbs = {}
    for m, aurocs in fold_aurocs.items():
        lcbs[m] = np.mean(aurocs) - lam * np.std(aurocs)
    anchor = max(lcbs, key=lcbs.get)
    return anchor, lcbs


# ============================================================
# Method A: Fixed alpha anchor blend
# p = (1-α) * p_anchor + α * p_ensemble
# ============================================================
def anchor_blend_fixed(data, n_classes, alphas=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5]):
    methods, all_labels, oof_probs, test_probs, fold_aurocs = get_cv_probs(data, n_classes)
    anchor, lcbs = select_anchor(fold_aurocs)
    n_total = len(all_labels)

    # Simple average ensemble (excluding anchor)
    non_anchor = [m for m in methods if m != anchor]
    oof_ens = np.mean([oof_probs[m] for m in non_anchor], axis=0)
    test_ens = np.mean([test_probs[m] for m in non_anchor], axis=0)

    # Try different alphas, pick best on OOF
    best_auroc, best_alpha = -1, 0
    for alpha in alphas:
        blended = (1 - alpha) * oof_probs[anchor] + alpha * oof_ens
        auroc = compute_auroc(all_labels, blended, n_classes)
        if auroc > best_auroc:
            best_auroc = auroc
            best_alpha = alpha

    # Apply to test
    test_blend = (1 - best_alpha) * test_probs[anchor] + best_alpha * test_ens
    r = eval_metrics(data["test"]["labels"], test_blend, n_classes)
    r["anchor"] = anchor
    r["alpha"] = best_alpha
    return r


# ============================================================
# Method B: Weighted average with LCB-softmax weights
# ============================================================
def lcb_weighted_avg(data, n_classes, temps=[0.05, 0.1, 0.2, 0.5]):
    methods, all_labels, oof_probs, test_probs, fold_aurocs = get_cv_probs(data, n_classes)

    # LCB scores
    lcbs = {m: np.mean(fold_aurocs[m]) - 1.0 * np.std(fold_aurocs[m]) for m in methods}

    best_auroc, best_temp = -1, 0.1
    for temp in temps:
        lcb_arr = np.array([lcbs[m] for m in methods])
        weights = softmax((lcb_arr - lcb_arr.max()) / temp)
        blended = sum(weights[i] * oof_probs[m] for i, m in enumerate(methods))
        auroc = compute_auroc(all_labels, blended, n_classes)
        if auroc > best_auroc:
            best_auroc = auroc
            best_temp = temp

    lcb_arr = np.array([lcbs[m] for m in methods])
    weights = softmax((lcb_arr - lcb_arr.max()) / best_temp)
    test_blend = sum(weights[i] * test_probs[m] for i, m in enumerate(methods))

    r = eval_metrics(data["test"]["labels"], test_blend, n_classes)
    r["temp"] = best_temp
    r["weights"] = {m: float(w) for m, w in zip(methods, weights)}
    return r


# ============================================================
# Method C: Fold-DRO simplex weights (worst-fold optimization)
# ============================================================
def fold_dro_simplex(data, n_classes):
    methods, all_labels, oof_probs, test_probs, fold_aurocs = get_cv_probs(data, n_classes)
    n_methods = len(methods)
    n_total = len(all_labels)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    fold_indices = list(skf.split(np.zeros(n_total), all_labels))

    def neg_worst_fold_auroc(w_raw):
        # Project to simplex via softmax
        w = softmax(w_raw)
        fold_aurocs_ens = []
        for _, va_idx in fold_indices:
            blended = sum(w[i] * oof_probs[m][va_idx] for i, m in enumerate(methods))
            auroc = compute_auroc(all_labels[va_idx], blended, n_classes)
            fold_aurocs_ens.append(auroc)
        return -min(fold_aurocs_ens)  # Maximize worst fold

    # Initialize with uniform
    w0 = np.zeros(n_methods)
    result = minimize(neg_worst_fold_auroc, w0, method="Nelder-Mead",
                      options={"maxiter": 500, "xatol": 1e-4})
    w_opt = softmax(result.x)

    test_blend = sum(w_opt[i] * test_probs[m] for i, m in enumerate(methods))
    r = eval_metrics(data["test"]["labels"], test_blend, n_classes)
    r["weights"] = {m: float(w) for m, w in zip(methods, w_opt)}
    r["worst_fold_auroc"] = -result.fun
    return r


# ============================================================
# Method D: Anchor + fold-DRO ensemble with conservative gate
# ============================================================
def anchor_dro_gate(data, n_classes):
    methods, all_labels, oof_probs, test_probs, fold_aurocs = get_cv_probs(data, n_classes)
    anchor, lcbs = select_anchor(fold_aurocs)
    n_total = len(all_labels)
    n_test = len(data["test"]["labels"])

    # Step 1: Build DRO ensemble (same as Method C)
    n_methods = len(methods)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    fold_indices = list(skf.split(np.zeros(n_total), all_labels))

    def neg_worst_fold_auroc(w_raw):
        w = softmax(w_raw)
        fold_aurocs_ens = []
        for _, va_idx in fold_indices:
            blended = sum(w[i] * oof_probs[m][va_idx] for i, m in enumerate(methods))
            auroc = compute_auroc(all_labels[va_idx], blended, n_classes)
            fold_aurocs_ens.append(auroc)
        return -min(fold_aurocs_ens)

    w0 = np.zeros(n_methods)
    result = minimize(neg_worst_fold_auroc, w0, method="Nelder-Mead",
                      options={"maxiter": 500})
    w_opt = softmax(result.x)

    oof_ens = sum(w_opt[i] * oof_probs[m] for i, m in enumerate(methods))
    test_ens = sum(w_opt[i] * test_probs[m] for i, m in enumerate(methods))

    # Step 2: Build disagreement features for the gate
    def build_gate_features(probs_dict, ens_probs, anchor_name, n):
        feats = []
        anchor_p = probs_dict[anchor_name]

        # Anchor entropy and margin
        anchor_ent = -np.sum(anchor_p * np.log(anchor_p + 1e-10), axis=1)
        anchor_margin = np.sort(anchor_p, axis=1)[:, -1] - np.sort(anchor_p, axis=1)[:, -2]

        # Ensemble entropy and margin
        ens_ent = -np.sum(ens_probs * np.log(ens_probs + 1e-10), axis=1)
        ens_margin = np.sort(ens_probs, axis=1)[:, -1] - np.sort(ens_probs, axis=1)[:, -2]

        # Agreement: how many probes agree with anchor top class
        anchor_top = anchor_p.argmax(axis=1)
        n_agree = np.zeros(n)
        for m in probs_dict:
            n_agree += (probs_dict[m].argmax(axis=1) == anchor_top).astype(float)
        n_agree /= len(probs_dict)

        # Mean pairwise disagreement (KL)
        all_probs = [probs_dict[m] for m in probs_dict]
        mean_kl = np.zeros(n)
        count = 0
        for i in range(len(all_probs)):
            for j in range(i+1, len(all_probs)):
                kl = np.sum(all_probs[i] * np.log((all_probs[i] + 1e-10) / (all_probs[j] + 1e-10)), axis=1)
                mean_kl += np.abs(kl)
                count += 1
        if count > 0:
            mean_kl /= count

        # Variance of top-class prob across probes
        top_class_probs = np.stack([probs_dict[m][np.arange(n), anchor_top] for m in probs_dict], axis=1)
        var_top = np.var(top_class_probs, axis=1)

        gate_feats = np.column_stack([
            anchor_ent, anchor_margin, ens_ent, ens_margin,
            n_agree, mean_kl, var_top,
        ])
        return gate_feats

    oof_gate_feats = build_gate_features(oof_probs, oof_ens, anchor, n_total)
    test_gate_feats = build_gate_features(test_probs, test_ens, anchor, n_test)

    # Step 3: Train gate — target is 1 when ensemble beats anchor
    # For each OOF sample, check if ensemble gives higher prob to correct class
    oof_anchor = oof_probs[anchor]
    oof_correct_anchor = oof_anchor[np.arange(n_total), all_labels]
    oof_correct_ens = oof_ens[np.arange(n_total), all_labels]
    gate_target = (oof_correct_ens > oof_correct_anchor + 0.01).astype(float)

    # Train conservative gate (biased toward 0 = anchor)
    gate_sc = StandardScaler()
    oof_gf = gate_sc.fit_transform(oof_gate_feats)
    test_gf = gate_sc.transform(test_gate_feats)

    # Try different regularizations
    best_auroc, best_gate_C = -1, 0.01
    for gate_C in [0.001, 0.01, 0.1, 1.0]:
        gate_clf = LogisticRegression(max_iter=2000, C=gate_C, random_state=42)
        gate_clf.fit(oof_gf, gate_target)
        g = gate_clf.predict_proba(oof_gf)[:, 1]
        # Cap gate at 0.5
        g = np.minimum(g, 0.5)
        blended = (1 - g)[:, None] * oof_anchor + g[:, None] * oof_ens
        auroc = compute_auroc(all_labels, blended, n_classes)
        if auroc > best_auroc:
            best_auroc = auroc
            best_gate_C = gate_C

    gate_clf = LogisticRegression(max_iter=2000, C=best_gate_C, random_state=42)
    gate_clf.fit(oof_gf, gate_target)

    # Apply to test
    g_test = gate_clf.predict_proba(test_gf)[:, 1]
    g_test = np.minimum(g_test, 0.5)
    test_blend = (1 - g_test)[:, None] * test_probs[anchor] + g_test[:, None] * test_ens

    r = eval_metrics(data["test"]["labels"], test_blend, n_classes)
    r["anchor"] = anchor
    r["gate_C"] = best_gate_C
    r["gate_mean"] = float(g_test.mean())
    r["dro_weights"] = {m: float(w) for m, w in zip(methods, w_opt)}
    return r


# ============================================================
# Method E: CV Stacking with anchor shrinkage
# ============================================================
def cv_stacking_anchor_shrink(data, n_classes):
    """CV stacking meta-LR, then shrink toward anchor."""
    methods, all_labels, oof_probs, test_probs, fold_aurocs = get_cv_probs(data, n_classes)
    anchor, lcbs = select_anchor(fold_aurocs)
    n_total = len(all_labels)

    # Stack OOF probs
    oof_meta = np.hstack([oof_probs[m] for m in methods])
    test_meta = np.hstack([test_probs[m] for m in methods])

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    # Find best C via CV
    best_auroc, best_C = -1, 1.0
    for C in [0.01, 0.1, 1.0, 10.0]:
        inner_oof = np.zeros((n_total, n_classes))
        for _, (tr_i, va_i) in enumerate(skf.split(oof_meta, all_labels)):
            clf = LogisticRegression(max_iter=2000, random_state=42, C=C)
            clf.fit(oof_meta[tr_i], all_labels[tr_i])
            inner_oof[va_i] = clf.predict_proba(oof_meta[va_i])
        auroc = compute_auroc(all_labels, inner_oof, n_classes)
        if auroc > best_auroc:
            best_auroc = auroc
            best_C = C

    clf = LogisticRegression(max_iter=2000, random_state=42, C=best_C)
    clf.fit(oof_meta, all_labels)

    stacking_test = clf.predict_proba(test_meta)

    # Shrink toward anchor
    best_auroc_final, best_shrink = -1, 0
    for shrink in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
        stacking_oof = np.zeros((n_total, n_classes))
        for _, (tr_i, va_i) in enumerate(skf.split(oof_meta, all_labels)):
            c = LogisticRegression(max_iter=2000, random_state=42, C=best_C)
            c.fit(oof_meta[tr_i], all_labels[tr_i])
            stacking_oof[va_i] = c.predict_proba(oof_meta[va_i])

        blended = (1 - shrink) * stacking_oof + shrink * oof_probs[anchor]
        auroc = compute_auroc(all_labels, blended, n_classes)
        if auroc > best_auroc_final:
            best_auroc_final = auroc
            best_shrink = shrink

    test_final = (1 - best_shrink) * stacking_test + best_shrink * test_probs[anchor]
    r = eval_metrics(data["test"]["labels"], test_final, n_classes)
    r["anchor"] = anchor
    r["shrink"] = best_shrink
    r["stacking_C"] = best_C
    return r


def main():
    print("=" * 70)
    print("ANCHOR FUSION — Conservative Score-Level Fusion")
    print("=" * 70)

    all_results = {}
    fusion_methods = [
        ("A: anchor_blend_fixed", anchor_blend_fixed),
        ("B: lcb_weighted_avg", lcb_weighted_avg),
        ("C: fold_dro_simplex", fold_dro_simplex),
        ("D: anchor_dro_gate", anchor_dro_gate),
        ("E: cv_stack_anchor_shrink", cv_stacking_anchor_shrink),
    ]

    for dataset, info in DATASETS.items():
        n_classes = info["n_classes"]
        baseline_method, baseline_auroc = BASELINES[dataset]

        print(f"\n{'='*60}")
        print(f"Dataset: {dataset} ({n_classes}-class)")
        print(f"Baseline: {baseline_method} AUROC={baseline_auroc:.4f}")
        print(f"{'='*60}")

        t0 = time.time()
        data = load_and_reduce(dataset, MULTICLASS_METHODS)
        print(f"Loaded in {time.time()-t0:.1f}s")

        ds_results = {}
        for name, fn in fusion_methods:
            t1 = time.time()
            try:
                r = fn(data, n_classes)
                delta = r["auroc"] - baseline_auroc
                status = ">>>" if delta > 0 else "   "
                extra = ""
                if "anchor" in r:
                    extra += f" anchor={r['anchor']}"
                if "alpha" in r:
                    extra += f" α={r['alpha']}"
                if "weights" in r:
                    top_w = sorted(r["weights"].items(), key=lambda x: -x[1])[:3]
                    extra += f" top_w={[(m,f'{w:.2f}') for m,w in top_w]}"
                if "gate_mean" in r:
                    extra += f" gate_mean={r['gate_mean']:.3f}"
                if "shrink" in r:
                    extra += f" shrink={r['shrink']}"
                print(f"  {status} {name:30s}  AUROC={r['auroc']:.4f} ({delta:+.4f})  "
                      f"Acc={r['accuracy']:.4f}  F1={r['f1_macro']:.4f}  [{time.time()-t1:.1f}s]{extra}")
                ds_results[name] = r
            except Exception as e:
                print(f"      {name:30s}  ERROR: {e}")
                import traceback; traceback.print_exc()

        all_results[dataset] = ds_results

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    ds_short = {"common_claim_3class": "cc_3c", "e2h_amc_3class": "e2h_3c",
                "e2h_amc_5class": "e2h_5c", "when2call_3class": "w2c_3c"}
    print(f"{'Method':32s}", end="")
    for ds in DATASETS:
        print(f"  {ds_short[ds]:>8s}", end="")
    print(f"  {'#wins':>6s}")

    print(f"{'[best single probe]':32s}", end="")
    for ds in DATASETS:
        _, bl = BASELINES[ds]
        print(f"  {bl:8.4f}", end="")
    print()

    for method_name, _ in fusion_methods:
        print(f"{method_name:32s}", end="")
        wins = 0
        for ds in DATASETS:
            if ds in all_results and method_name in all_results[ds]:
                auroc = all_results[ds][method_name]["auroc"]
                _, bl = BASELINES[ds]
                if auroc > bl:
                    wins += 1
                print(f"  {auroc:8.4f}", end="")
            else:
                print(f"  {'N/A':>8s}", end="")
        print(f"  {wins:>6d}/4")

    os.makedirs("/home/junyi/NIPS2026/fusion/results", exist_ok=True)
    with open("/home/junyi/NIPS2026/fusion/results/anchor_fusion_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved to fusion/results/anchor_fusion_results.json")


if __name__ == "__main__":
    main()
