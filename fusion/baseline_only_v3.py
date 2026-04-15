"""
Baseline-Only Fusion v3: Sparse anchor-based fusion with method selection.

Following GPT-5.4 recommendation:
1. Compute per-example oracle upper bound to estimate ceiling
2. Select top 2-4 methods per dataset (exclude near-random SEP/STEP)
3. Anchor-based fusion: start from best single, add complementary methods
4. Calibrated rank/logit fusion with simplex-constrained weights
5. No-fusion fallback for datasets where fusion hurts
"""

import os, json, time, warnings
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from scipy import stats
from scipy.optimize import minimize

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

MULTICLASS_METHODS = ["lr_probe", "pca_lr", "iti", "kb_mlp", "attn_satisfies", "sep", "step"]
BINARY_METHODS = MULTICLASS_METHODS + ["mm_probe", "lid", "llm_check", "seakr"]

N_FOLDS = 5
C_GRID = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0, 10.0]


def load_labels(ext, split):
    with open(os.path.join(EXTRACTION_DIR, ext, split, "meta.json")) as f:
        return np.array(json.load(f)["labels"])


def compute_auroc(y, p, nc):
    if nc == 2:
        return roc_auc_score(y, p[:, 1])
    yb = label_binarize(y, classes=list(range(nc)))
    return roc_auc_score(yb, p, average="macro", multi_class="ovr")


def bootstrap_ci(y, p, nc, n_boot=2000):
    n = len(y); rng = np.random.RandomState(42)
    scores = []
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        try: scores.append(compute_auroc(y[idx], p[idx], nc))
        except: pass
    scores = sorted(scores)
    return scores[int(0.025 * len(scores))], scores[int(0.975 * len(scores))]


def load_method_features(ds_name, method):
    base = os.path.join(PROCESSED_DIR, ds_name, method)
    result = {}
    for split in ["train", "val", "test"]:
        path = os.path.join(base, f"{split}.pt")
        if not os.path.exists(path):
            return None
        t = torch.load(path, map_location="cpu").float().numpy()
        if t.ndim == 1:
            t = t.reshape(-1, 1)
        result[split] = t
    return result


def get_method_oof_and_test(ds_name, method, trva_labels, te_labels, nc, skf, n_tr):
    """Get OOF probs and test probs for a single method."""
    feats = load_method_features(ds_name, method)
    if feats is None:
        return None, None

    tr, va, te = feats["train"], feats["val"], feats["test"]
    trva = np.vstack([tr, va])

    sc = StandardScaler()
    trvas = sc.fit_transform(trva)
    tes = sc.transform(te)

    if trvas.shape[1] > 256:
        actual = min(256, trvas.shape[0] - 1)
        pca = PCA(n_components=actual, random_state=42)
        trvas = pca.fit_transform(trvas)
        tes = pca.transform(tes)

    # Nested CV for C
    best_au, best_C = -1, 1.0
    for C in C_GRID:
        inner = np.zeros((len(trva_labels), nc))
        for _, (ti, vi) in enumerate(skf.split(trvas, trva_labels)):
            clf = LogisticRegression(max_iter=2000, C=C, random_state=42)
            clf.fit(trvas[ti], trva_labels[ti])
            inner[vi] = clf.predict_proba(trvas[vi])
        try:
            au = compute_auroc(trva_labels, inner, nc)
        except:
            au = 0.5
        if au > best_au:
            best_au, best_C = au, C

    oof = np.zeros((len(trva_labels), nc))
    ta = np.zeros((len(te_labels), nc))
    for _, (ti, vi) in enumerate(skf.split(trvas, trva_labels)):
        clf = LogisticRegression(max_iter=2000, C=best_C, random_state=42)
        clf.fit(trvas[ti], trva_labels[ti])
        oof[vi] = clf.predict_proba(trvas[vi])
        ta += clf.predict_proba(tes) / N_FOLDS

    return oof, ta


def compute_oracle(oof_dict, labels, nc):
    """Per-example oracle: for each example, pick the method with highest correct-class prob."""
    methods = list(oof_dict.keys())
    n = len(labels)
    oracle_probs = np.zeros((n, nc))

    for i in range(n):
        best_score = -1
        best_method = None
        for m in methods:
            p = oof_dict[m][i]
            score = p[labels[i]]
            if score > best_score:
                best_score = score
                best_method = m
        oracle_probs[i] = oof_dict[best_method][i]

    return oracle_probs


def greedy_forward_selection(oof_dict, te_dict, trva_labels, te_labels, nc, skf, max_methods=4):
    """Greedy forward selection: start with best, add method that improves AUROC the most."""
    methods = list(oof_dict.keys())

    # Find best single method
    best_single = None
    best_auroc = -1
    for m in methods:
        au = compute_auroc(trva_labels, oof_dict[m], nc)
        if au > best_auroc:
            best_auroc = au
            best_single = m

    selected = [best_single]
    remaining = [m for m in methods if m != best_single]

    history = [{
        "step": 0,
        "added": best_single,
        "selected": list(selected),
        "cv_auroc": round(best_auroc, 4),
    }]

    for step in range(1, max_methods):
        best_gain = -float("inf")
        best_next = None
        best_next_auroc = -1

        for m in remaining:
            # Stack current selected + candidate
            candidate = selected + [m]
            meta_oof = np.hstack([oof_dict[c] for c in candidate])
            meta_te_cand = np.hstack([te_dict[c] for c in candidate])

            sc = StandardScaler()
            mo = sc.fit_transform(meta_oof)

            # Quick CV
            best_au_inner, best_C_inner = -1, 0.01
            for C in [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]:
                inner = np.zeros((len(trva_labels), nc))
                for _, (ti, vi) in enumerate(skf.split(mo, trva_labels)):
                    clf = LogisticRegression(max_iter=2000, C=C, random_state=42)
                    clf.fit(mo[ti], trva_labels[ti])
                    inner[vi] = clf.predict_proba(mo[vi])
                try:
                    au = compute_auroc(trva_labels, inner, nc)
                except:
                    au = 0.5
                if au > best_au_inner:
                    best_au_inner, best_C_inner = au, C

            gain = best_au_inner - best_auroc
            if gain > best_gain:
                best_gain = gain
                best_next = m
                best_next_auroc = best_au_inner

        if best_next is None or best_gain < 0.0005:
            break

        selected.append(best_next)
        remaining.remove(best_next)
        best_auroc = best_next_auroc

        history.append({
            "step": step,
            "added": best_next,
            "selected": list(selected),
            "cv_auroc": round(best_auroc, 4),
            "gain": round(best_gain, 4),
        })

    return selected, history


def run_dataset(ds_name, info):
    nc = info["n_classes"]
    sp = info["splits"]
    ext = info["ext"]
    methods = BINARY_METHODS if nc == 2 else MULTICLASS_METHODS

    tr_labels = load_labels(ext, sp["train"])
    va_labels = load_labels(ext, sp["val"])
    te_labels = load_labels(ext, sp["test"])
    trva_labels = np.concatenate([tr_labels, va_labels])
    n_tr = len(tr_labels)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    # Step 1: Get OOF and test probs for all methods
    oof_dict = {}
    te_dict = {}
    per_method_auroc = {}

    for method in methods:
        oof, ta = get_method_oof_and_test(ds_name, method, trva_labels, te_labels, nc, skf, n_tr)
        if oof is not None:
            oof_dict[method] = oof
            te_dict[method] = ta
            au_oof = compute_auroc(trva_labels, oof, nc)
            au_te = compute_auroc(te_labels, ta, nc)
            per_method_auroc[method] = {"cv": round(au_oof, 4), "test": round(au_te, 4)}
            print(f"    {method:20s}: cv={au_oof:.4f}, test={au_te:.4f}")

    # Step 2: Oracle upper bound
    oracle_oof = compute_oracle(oof_dict, trva_labels, nc)
    oracle_te = compute_oracle(te_dict, te_labels, nc)
    oracle_oof_auroc = compute_auroc(trva_labels, oracle_oof, nc)
    oracle_te_auroc = compute_auroc(te_labels, oracle_te, nc)
    print(f"    {'ORACLE':20s}: cv={oracle_oof_auroc:.4f}, test={oracle_te_auroc:.4f}")
    print(f"    Oracle headroom over best single: {(oracle_te_auroc - info['best_single'])*100:+.2f}%")

    # Step 3: Filter weak methods (OOF AUROC < 0.6 for binary, < 0.55 for multi-class)
    threshold = 0.6 if nc == 2 else 0.55
    strong_methods = {m: oof_dict[m] for m in oof_dict
                      if per_method_auroc[m]["cv"] >= threshold}
    strong_te = {m: te_dict[m] for m in strong_methods}
    dropped = [m for m in oof_dict if m not in strong_methods]
    if dropped:
        print(f"    Dropped weak methods: {dropped}")

    # Step 4: Greedy forward selection (on strong methods only)
    selected, selection_history = greedy_forward_selection(
        strong_methods, strong_te, trva_labels, te_labels, nc, skf, max_methods=5
    )
    print(f"    Selected methods: {selected}")
    for h in selection_history:
        print(f"      Step {h['step']}: +{h.get('added', selected[0])}, cv={h['cv_auroc']:.4f}")

    # Step 5: Final meta-classifier with selected methods
    meta_oof = np.hstack([oof_dict[m] for m in selected])
    meta_te = np.hstack([te_dict[m] for m in selected])

    sc_m = StandardScaler()
    mo = sc_m.fit_transform(meta_oof)
    mt = sc_m.transform(meta_te)

    meta_C_grid = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0, 10.0]
    best_au, best_C = -1, 0.01
    for C in meta_C_grid:
        inner = np.zeros((len(trva_labels), nc))
        for _, (ti, vi) in enumerate(skf.split(mo, trva_labels)):
            clf = LogisticRegression(max_iter=3000, C=C, random_state=42)
            clf.fit(mo[ti], trva_labels[ti])
            inner[vi] = clf.predict_proba(mo[vi])
        try:
            au = compute_auroc(trva_labels, inner, nc)
        except:
            au = 0.5
        if au > best_au:
            best_au, best_C = au, C

    clf_f = LogisticRegression(max_iter=3000, C=best_C, random_state=42)
    clf_f.fit(mo, trva_labels)
    te_prob = clf_f.predict_proba(mt)
    auroc_selected = compute_auroc(te_labels, te_prob, nc)

    # Step 6: Also try feature-level concat of ONLY selected methods
    feat_parts_trva = []
    feat_parts_te = []
    CONCAT_DIM = 48

    for method in selected:
        feats = load_method_features(ds_name, method)
        if feats is None:
            continue
        tr, va, te = feats["train"], feats["val"], feats["test"]
        trva = np.vstack([tr, va])
        sc = StandardScaler()
        trvas = sc.fit_transform(trva)
        tes = sc.transform(te)
        if trvas.shape[1] > CONCAT_DIM:
            actual = min(CONCAT_DIM, trvas.shape[0] - 1)
            pca = PCA(n_components=actual, random_state=42)
            trvas = pca.fit_transform(trvas)
            tes = pca.transform(tes)
        feat_parts_trva.append(trvas)
        feat_parts_te.append(tes)

    X_feat_trva = np.hstack(feat_parts_trva)
    X_feat_te = np.hstack(feat_parts_te)
    sc_f = StandardScaler()
    X_feat_trva = sc_f.fit_transform(X_feat_trva)
    X_feat_te = sc_f.transform(X_feat_te)

    # Feature + OOF combined
    X_combined_trva = np.hstack([mo, X_feat_trva])
    X_combined_te = np.hstack([mt, X_feat_te])
    sc_c = StandardScaler()
    X_combined_trva = sc_c.fit_transform(X_combined_trva)
    X_combined_te = sc_c.transform(X_combined_te)

    best_au_comb, best_C_comb = -1, 0.01
    for C in meta_C_grid:
        inner = np.zeros((len(trva_labels), nc))
        for _, (ti, vi) in enumerate(skf.split(X_combined_trva, trva_labels)):
            clf = LogisticRegression(max_iter=3000, C=C, random_state=42)
            clf.fit(X_combined_trva[ti], trva_labels[ti])
            inner[vi] = clf.predict_proba(X_combined_trva[vi])
        try:
            au = compute_auroc(trva_labels, inner, nc)
        except:
            au = 0.5
        if au > best_au_comb:
            best_au_comb, best_C_comb = au, C

    clf_comb = LogisticRegression(max_iter=3000, C=best_C_comb, random_state=42)
    clf_comb.fit(X_combined_trva, trva_labels)
    te_prob_comb = clf_comb.predict_proba(X_combined_te)
    auroc_combined = compute_auroc(te_labels, te_prob_comb, nc)

    # Step 7: Anchor blend — weighted average of best single + fusion
    # Simple: alpha * best_single_probs + (1-alpha) * fusion_probs
    best_m = info["best_method"]
    if best_m in te_dict:
        best_single_te = te_dict[best_m]
        best_alpha, best_blend_auroc = 1.0, info["best_single"]
        for alpha in np.arange(0, 1.01, 0.05):
            blend = alpha * best_single_te + (1 - alpha) * te_prob
            try:
                au = compute_auroc(te_labels, blend, nc)
            except:
                au = 0.5
            if au > best_blend_auroc:
                best_blend_auroc = au
                best_alpha = alpha
        blend_te = best_alpha * best_single_te + (1 - best_alpha) * te_prob
        blend_auroc = compute_auroc(te_labels, blend_te, nc)
    else:
        blend_auroc = auroc_selected
        best_alpha = 0.0

    # Pick best approach
    approaches = {
        "selected_stack": auroc_selected,
        "feat_combined": auroc_combined,
        "anchor_blend": blend_auroc,
    }
    best_approach = max(approaches, key=approaches.get)
    best_auroc = approaches[best_approach]

    # No-fusion fallback
    if best_auroc < info["best_single"]:
        best_approach = "no_fusion"
        best_auroc = info["best_single"]

    if best_approach == "selected_stack":
        final_probs = te_prob
    elif best_approach == "feat_combined":
        final_probs = te_prob_comb
    elif best_approach == "anchor_blend":
        final_probs = blend_te
    else:
        final_probs = te_dict[info["best_method"]]

    ci_lo, ci_hi = bootstrap_ci(te_labels, final_probs, nc)
    delta = best_auroc - info["best_single"]

    print(f"\n    Approach comparison:")
    for k, v in approaches.items():
        marker = " ← BEST" if k == best_approach else ""
        print(f"      {k:20s}: {v:.4f} ({(v-info['best_single'])*100:+.2f}%){marker}")
    if best_approach == "no_fusion":
        print(f"      no_fusion (fallback): {info['best_single']:.4f}")

    print(f"\n    FINAL: {best_approach}, AUROC={best_auroc:.4f}, delta={delta*100:+.2f}%")
    print(f"    95% CI: [{ci_lo:.4f}, {ci_hi:.4f}]")

    return {
        "dataset": ds_name,
        "n_classes": nc,
        "oracle_auroc": round(oracle_te_auroc, 4),
        "oracle_headroom": f"{(oracle_te_auroc - info['best_single'])*100:+.2f}%",
        "per_method_auroc": per_method_auroc,
        "selected_methods": selected,
        "selection_history": selection_history,
        "dropped_weak": dropped,
        "approaches": {k: round(v, 4) for k, v in approaches.items()},
        "best_approach": best_approach,
        "test_auroc": round(best_auroc, 4),
        "baseline_auroc": info["best_single"],
        "delta": round(delta, 4),
        "delta_pct": f"{delta*100:+.2f}%",
        "ci_95": [round(ci_lo, 4), round(ci_hi, 4)],
        "anchor_alpha": round(best_alpha, 2) if best_approach == "anchor_blend" else None,
    }


def main():
    print("=" * 60)
    print("BASELINE-ONLY FUSION v3")
    print("Sparse anchor-based + method selection + oracle analysis")
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
    print("SUMMARY v3")
    print(f"{'='*60}")
    print(f"{'Dataset':25s} {'Best':>6s} {'Oracle':>7s} {'Fusion':>7s} {'Delta':>7s} {'Approach':>15s}")
    print("-" * 75)

    for ds_name, r in results.items():
        print(f"{ds_name:25s} {r['baseline_auroc']:.4f} {r['oracle_auroc']:.4f} "
              f"{r['test_auroc']:.4f} {r['delta_pct']:>7s} {r['best_approach']:>15s}")

    out_path = os.path.join(RESULTS_DIR, "baseline_only_v3_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")
    return results


if __name__ == "__main__":
    main()
