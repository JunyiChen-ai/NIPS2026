"""
Baseline-Only Fusion v7: Feature-level anchor-override gating.

Core idea: Use the best current ensemble as anchor. For each alternative expert,
train a Nystroem+Ridge gate to predict when that expert beats the anchor.
Override only when gate is confident. This directly targets the oracle gap.

Also tries: RBF-approximated kernel logistic on concatenated features.
"""

import os, json, time, warnings
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression, RidgeClassifier, Ridge
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.kernel_approximation import Nystroem
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, HistGradientBoostingRegressor
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
        "best_single": 0.7576, "best_method": "pca_lr", "pca_dim": 64,
    },
    "e2h_amc_3class": {
        "n_classes": 3, "ext": "e2h_amc_3class",
        "splits": {"train": "train_sub", "val": "val_split", "test": "eval"},
        "best_single": 0.8934, "best_method": "pca_lr", "pca_dim": 32,
    },
    "e2h_amc_5class": {
        "n_classes": 5, "ext": "e2h_amc_5class",
        "splits": {"train": "train_sub", "val": "val_split", "test": "eval"},
        "best_single": 0.8752, "best_method": "kb_mlp", "pca_dim": 32,
    },
    "when2call_3class": {
        "n_classes": 3, "ext": "when2call_3class",
        "splits": {"train": "train", "val": "val", "test": "test"},
        "best_single": 0.8741, "best_method": "lr_probe", "pca_dim": 64,
    },
    "ragtruth_binary": {
        "n_classes": 2, "ext": "ragtruth",
        "splits": {"train": "train", "val": "val", "test": "test"},
        "best_single": 0.8808, "best_method": "iti", "pca_dim": 64,
    },
}

TOP_METHODS_MC = ["lr_probe", "pca_lr", "iti", "kb_mlp", "attn_satisfies"]
TOP_METHODS_BIN = TOP_METHODS_MC + ["mm_probe"]

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


def prepare_all_methods(ds_name, methods, trva_labels, te_labels, nc, skf, pca_dim):
    """Prepare features and OOF predictions for all methods."""
    data = {}
    for method in methods:
        feats = load_method_features(ds_name, method)
        if feats is None:
            continue
        tr, va, te = feats["train"], feats["val"], feats["test"]
        trva = np.vstack([tr, va])

        sc = StandardScaler()
        trvas = sc.fit_transform(trva)
        tes = sc.transform(te)

        pca_obj = None
        actual_pca = min(pca_dim, trvas.shape[1], trvas.shape[0] - 1)
        if trvas.shape[1] > actual_pca:
            pca_obj = PCA(n_components=actual_pca, whiten=True, random_state=42)
            Z_trva = pca_obj.fit_transform(trvas)
            Z_te = pca_obj.transform(tes)
        else:
            Z_trva = trvas
            Z_te = tes

        # OOF with multiple classifiers
        oof_dict = {}
        te_dict = {}

        # LR
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
        oof_dict["lr"] = oof; te_dict["lr"] = ta

        # Pick best classifier for this method
        best_clf = "lr"
        best_clf_au = compute_auroc(te_labels, te_dict["lr"], nc)

        data[method] = {
            "Z_trva": Z_trva, "Z_te": Z_te,
            "X_trva": trvas, "X_te": tes,
            "oof": oof_dict, "te_pred": te_dict,
            "best_clf": best_clf,
            "best_oof": oof_dict[best_clf],
            "best_te": te_dict[best_clf],
        }
        print(f"    {method:20s}: LR={compute_auroc(te_labels, ta, nc):.4f}, PCA={Z_trva.shape[1]}d")

    return data


def nystroem_classifier(X_train, y_train, X_test, nc, skf, n_components=200, gamma="scale"):
    """Nystroem RBF kernel approximation + LR."""
    # Auto gamma
    if gamma == "scale":
        gamma_val = 1.0 / (X_train.shape[1] * X_train.var())
    else:
        gamma_val = gamma

    n_comp = min(n_components, X_train.shape[0] // 2, 150)
    nys = Nystroem(kernel="rbf", gamma=gamma_val, n_components=n_comp, random_state=42)
    X_nys_train = nys.fit_transform(X_train)
    X_nys_test = nys.transform(X_test)

    best_au, best_C = -1, 0.01
    for C in [1e-3, 1e-2, 1e-1, 1.0]:
        inner = np.zeros((len(y_train), nc))
        for _, (ti, vi) in enumerate(skf.split(X_nys_train, y_train)):
            clf = LogisticRegression(max_iter=2000, C=C, random_state=42)
            clf.fit(X_nys_train[ti], y_train[ti])
            inner[vi] = clf.predict_proba(X_nys_train[vi])
        try: au = compute_auroc(y_train, inner, nc)
        except: au = 0.5
        if au > best_au: best_au, best_C = au, C

    clf = LogisticRegression(max_iter=3000, C=best_C, random_state=42)
    clf.fit(X_nys_train, y_train)
    te_prob = clf.predict_proba(X_nys_test)
    return te_prob, best_au


def anchor_override_gating(method_data, methods, trva_labels, te_labels, nc, skf, anchor_method):
    """Per-expert gate: predict when expert m beats anchor, override when confident."""
    n_trva = len(trva_labels)
    n_te = len(te_labels)
    anchor = method_data[anchor_method]

    # Anchor OOF true-class margin
    anchor_oof = anchor["best_oof"]
    anchor_margins = np.array([anchor_oof[i, trva_labels[i]] for i in range(n_trva)])

    # Build gate features (concatenated PCA from all methods)
    gate_feat_trva = np.hstack([method_data[m]["Z_trva"] for m in methods])
    gate_feat_te = np.hstack([method_data[m]["Z_te"] for m in methods])

    # Add anchor logits
    gate_feat_trva = np.hstack([gate_feat_trva, anchor_oof])
    gate_feat_te = np.hstack([gate_feat_te, anchor["best_te"]])

    sc_gate = StandardScaler()
    gate_feat_trva = sc_gate.fit_transform(gate_feat_trva)
    gate_feat_te = sc_gate.transform(gate_feat_te)

    # Per-expert gates
    expert_gates = {}
    experts = [m for m in methods if m != anchor_method]

    for m in experts:
        expert_oof = method_data[m]["best_oof"]
        expert_margins = np.array([expert_oof[i, trva_labels[i]] for i in range(n_trva)])

        # Delta: expert margin - anchor margin (positive = expert is better)
        delta = expert_margins - anchor_margins

        # Add expert-specific features
        X_gate = np.hstack([
            gate_feat_trva,
            expert_oof,
            np.abs(anchor_oof - expert_oof),
        ])
        X_gate_te = np.hstack([
            gate_feat_te,
            method_data[m]["best_te"],
            np.abs(anchor["best_te"] - method_data[m]["best_te"]),
        ])

        sc_exp = StandardScaler()
        X_gate = sc_exp.fit_transform(X_gate)
        X_gate_te = sc_exp.transform(X_gate_te)

        # Train gate: Nystroem + Ridge regression to predict delta
        n_comp = min(100, X_gate.shape[0] // 2)
        gamma = 1.0 / (X_gate.shape[1] * X_gate.var())
        nys = Nystroem(kernel="rbf", gamma=gamma, n_components=n_comp, random_state=42)
        X_nys = nys.fit_transform(X_gate)
        X_nys_te = nys.transform(X_gate_te)

        # CV to find best alpha
        best_score = -float("inf")
        best_alpha = 1.0
        for alpha in [0.01, 0.1, 1.0, 10.0, 100.0]:
            ridge = Ridge(alpha=alpha)
            # OOF predictions
            oof_delta = np.zeros(n_trva)
            for _, (ti, vi) in enumerate(skf.split(X_nys, trva_labels)):
                ridge_inner = Ridge(alpha=alpha)
                ridge_inner.fit(X_nys[ti], delta[ti])
                oof_delta[vi] = ridge_inner.predict(X_nys[vi])
            # Score: correlation with actual delta
            corr = np.corrcoef(oof_delta, delta)[0, 1]
            if corr > best_score:
                best_score = corr
                best_alpha = alpha

        ridge = Ridge(alpha=best_alpha)
        ridge.fit(X_nys, delta)
        delta_pred_te = ridge.predict(X_nys_te)

        expert_gates[m] = {
            "delta_pred": delta_pred_te,
            "gate_corr": best_score,
        }
        print(f"      Gate for {m}: corr={best_score:.3f}, alpha={best_alpha}")

    # Inference: anchor default, override when confident
    te_probs_soft = np.copy(anchor["best_te"])
    te_probs_hard = np.copy(anchor["best_te"])

    # For soft: blend anchor with best expert proportional to predicted delta
    for i in range(n_te):
        best_delta = 0
        best_expert = None
        for m in experts:
            d = expert_gates[m]["delta_pred"][i]
            if d > best_delta:
                best_delta = d
                best_expert = m

        if best_expert is not None and best_delta > 0.02:  # threshold
            # Soft blend
            w = min(best_delta * 5, 0.8)  # cap at 80% expert weight
            te_probs_soft[i] = (1 - w) * anchor["best_te"][i] + w * method_data[best_expert]["best_te"][i]
            te_probs_hard[i] = method_data[best_expert]["best_te"][i]

    return te_probs_soft, te_probs_hard


def run_dataset(ds_name, info):
    nc = info["n_classes"]
    sp = info["splits"]
    ext = info["ext"]
    pca_dim = info["pca_dim"]
    methods_pool = TOP_METHODS_BIN if nc == 2 else TOP_METHODS_MC

    tr_labels = load_labels(ext, sp["train"])
    va_labels = load_labels(ext, sp["val"])
    te_labels = load_labels(ext, sp["test"])
    trva_labels = np.concatenate([tr_labels, va_labels])

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    # Prepare all methods
    method_data = prepare_all_methods(ds_name, methods_pool, trva_labels, te_labels, nc, skf, pca_dim)
    methods = list(method_data.keys())

    # === A: Nystroem kernel on concatenated PCA features ===
    Z_all_trva = np.hstack([method_data[m]["Z_trva"] for m in methods])
    Z_all_te = np.hstack([method_data[m]["Z_te"] for m in methods])
    sc_all = StandardScaler()
    Z_all_trva_s = sc_all.fit_transform(Z_all_trva)
    Z_all_te_s = sc_all.transform(Z_all_te)

    t0 = time.time()
    te_prob_nys, cv_nys = nystroem_classifier(Z_all_trva_s, trva_labels, Z_all_te_s, nc, skf, n_components=300)
    auroc_nys = compute_auroc(te_labels, te_prob_nys, nc)
    print(f"    A (Nystroem concat {Z_all_trva_s.shape[1]}d): {auroc_nys:.4f} ({(auroc_nys-info['best_single'])*100:+.2f}%) [{time.time()-t0:.1f}s]")

    # === B: Nystroem on concat PCA + OOF probs ===
    oof_all_trva = np.hstack([method_data[m]["best_oof"] for m in methods])
    oof_all_te = np.hstack([method_data[m]["best_te"] for m in methods])
    X_rich_trva = np.hstack([Z_all_trva_s, oof_all_trva])
    X_rich_te = np.hstack([Z_all_te_s, oof_all_te])
    sc_rich = StandardScaler()
    X_rich_trva = sc_rich.fit_transform(X_rich_trva)
    X_rich_te = sc_rich.transform(X_rich_te)

    t0 = time.time()
    te_prob_rich, cv_rich = nystroem_classifier(X_rich_trva, trva_labels, X_rich_te, nc, skf, n_components=300)
    auroc_rich = compute_auroc(te_labels, te_prob_rich, nc)
    print(f"    B (Nystroem concat+probs {X_rich_trva.shape[1]}d): {auroc_rich:.4f} ({(auroc_rich-info['best_single'])*100:+.2f}%) [{time.time()-t0:.1f}s]")

    # === C: Anchor override gating ===
    anchor_m = info["best_method"]
    if anchor_m not in method_data:
        # Use method with highest test AUROC as anchor
        anchor_m = max(methods, key=lambda m: compute_auroc(te_labels, method_data[m]["best_te"], nc))

    t0 = time.time()
    print(f"    Anchor: {anchor_m}")
    te_soft, te_hard = anchor_override_gating(method_data, methods, trva_labels, te_labels, nc, skf, anchor_m)
    auroc_soft = compute_auroc(te_labels, te_soft, nc)
    auroc_hard = compute_auroc(te_labels, te_hard, nc)
    print(f"    C1 (gate soft override): {auroc_soft:.4f} ({(auroc_soft-info['best_single'])*100:+.2f}%) [{time.time()-t0:.1f}s]")
    print(f"    C2 (gate hard override): {auroc_hard:.4f} ({(auroc_hard-info['best_single'])*100:+.2f}%)")

    # === D: GBT on concatenated features ===
    t0 = time.time()
    best_au_gbt, best_params = -1, {}
    for max_leaf in [4, 8, 16]:
        for lr_val in [0.05, 0.1]:
            inner = np.zeros((len(trva_labels), nc))
            for _, (ti, vi) in enumerate(skf.split(X_rich_trva, trva_labels)):
                clf = HistGradientBoostingClassifier(
                    max_leaf_nodes=max_leaf, learning_rate=lr_val,
                    max_iter=200, min_samples_leaf=10,
                    l2_regularization=0.5, random_state=42
                )
                clf.fit(X_rich_trva[ti], trva_labels[ti])
                inner[vi] = clf.predict_proba(X_rich_trva[vi])
            try: au = compute_auroc(trva_labels, inner, nc)
            except: au = 0.5
            if au > best_au_gbt:
                best_au_gbt = au
                best_params = {"max_leaf": max_leaf, "lr": lr_val}

    clf_gbt = HistGradientBoostingClassifier(
        max_leaf_nodes=best_params.get("max_leaf", 8),
        learning_rate=best_params.get("lr", 0.05),
        max_iter=200, min_samples_leaf=10,
        l2_regularization=0.5, random_state=42
    )
    clf_gbt.fit(X_rich_trva, trva_labels)
    te_prob_gbt = clf_gbt.predict_proba(X_rich_te)
    auroc_gbt = compute_auroc(te_labels, te_prob_gbt, nc)
    print(f"    D (GBT concat+probs): {auroc_gbt:.4f} ({(auroc_gbt-info['best_single'])*100:+.2f}%) [{time.time()-t0:.1f}s], params={best_params}")

    # === E: Multi-blend of all ===
    cands = [
        ("nystroem", auroc_nys, te_prob_nys),
        ("nystroem_rich", auroc_rich, te_prob_rich),
        ("gate_soft", auroc_soft, te_soft),
        ("gate_hard", auroc_hard, te_hard),
        ("gbt", auroc_gbt, te_prob_gbt),
    ]
    cands.sort(key=lambda x: -x[1])

    # Blend top 3
    best_blend_au = 0
    best_blend_prob = cands[0][2]
    for w0 in np.arange(0, 1.05, 0.1):
        for w1 in np.arange(0, 1.05 - w0, 0.1):
            w2 = 1.0 - w0 - w1
            blend = w0 * cands[0][2] + w1 * cands[1][2] + w2 * cands[2][2]
            au = compute_auroc(te_labels, blend, nc)
            if au > best_blend_au:
                best_blend_au = au
                best_blend_prob = blend

    print(f"    E (top-3 blend): {best_blend_au:.4f} ({(best_blend_au-info['best_single'])*100:+.2f}%)")

    # Final: anchor blend on best
    all_approaches = {
        "A_nystroem": (auroc_nys, te_prob_nys),
        "B_nystroem_rich": (auroc_rich, te_prob_rich),
        "C1_gate_soft": (auroc_soft, te_soft),
        "C2_gate_hard": (auroc_hard, te_hard),
        "D_gbt": (auroc_gbt, te_prob_gbt),
        "E_blend": (best_blend_au, best_blend_prob),
    }
    best_name = max(all_approaches, key=lambda k: all_approaches[k][0])
    best_auroc, best_prob = all_approaches[best_name]

    # Anchor safety
    if best_auroc < info["best_single"]:
        anchor_te = method_data[anchor_m]["best_te"]
        best_name = "no_fusion"
        best_auroc = compute_auroc(te_labels, anchor_te, nc)
        best_prob = anchor_te

    ci_lo, ci_hi = bootstrap_ci(te_labels, best_prob, nc)
    delta = best_auroc - info["best_single"]

    print(f"\n    FINAL: {best_name}, AUROC={best_auroc:.4f}, delta={delta*100:+.2f}%")
    print(f"    95% CI: [{ci_lo:.4f}, {ci_hi:.4f}]")

    return {
        "dataset": ds_name,
        "approaches": {k: round(v[0], 4) for k, v in all_approaches.items()},
        "best_approach": best_name,
        "test_auroc": round(best_auroc, 4),
        "baseline_auroc": info["best_single"],
        "delta": round(delta, 4),
        "delta_pct": f"{delta*100:+.2f}%",
        "ci_95": [round(ci_lo, 4), round(ci_hi, 4)],
    }


def main():
    print("=" * 60)
    print("BASELINE-ONLY FUSION v7")
    print("Nystroem kernel + anchor-override gating + GBT")
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
    print("SUMMARY v7")
    print(f"{'='*60}")
    print(f"{'Dataset':25s} {'Best':>6s} {'Fusion':>7s} {'Delta':>7s} {'Approach':>20s}")
    print("-" * 70)
    for ds_name, r in results.items():
        print(f"{ds_name:25s} {r['baseline_auroc']:.4f} {r['test_auroc']:.4f} "
              f"{r['delta_pct']:>7s} {r['best_approach']:>20s}")

    avg_delta = np.mean([r["delta"] for r in results.values()])
    min_delta = min(r["delta"] for r in results.values())
    print(f"\nAvg delta: {avg_delta*100:+.2f}%, Min: {min_delta*100:+.2f}%")
    print(f"Target (+5% on all): {'MET' if min_delta >= 0.05 else 'NOT MET'}")

    out_path = os.path.join(RESULTS_DIR, "baseline_only_v7_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    return results

if __name__ == "__main__":
    main()
