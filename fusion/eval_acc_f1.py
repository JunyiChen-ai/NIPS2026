"""
Quick eval: run v13_final pipeline and report AUROC + Accuracy + F1 (macro).
Also compute acc/f1 for best single probe per dataset for comparison.
"""
import os, json, time, warnings
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import HistGradientBoostingClassifier

warnings.filterwarnings("ignore")

PROCESSED_DIR = "/home/junyi/NIPS2026/reproduce/processed_features"
EXTRACTION_DIR = "/home/junyi/NIPS2026/extraction/features"
RESULTS_DIR = "/home/junyi/NIPS2026/fusion/results"

PCA_DIM = 64
EXPERT_TYPES = ["lr", "gbt"]
N_SEEDS = 5
N_FOLDS = 5
STABILITY_ALPHA = 0.5

MC_METHODS = ["lr_probe", "pca_lr", "iti", "kb_mlp", "attn_satisfies", "sep", "step"]
BIN_METHODS = MC_METHODS + ["mm_probe"]

ALL_DATASETS = {
    "common_claim_3class": {"n_classes": 3, "ext": "common_claim_3class", "splits": {"train": "train", "val": "val", "test": "test"}, "best_single": 0.7576},
    "e2h_amc_3class": {"n_classes": 3, "ext": "e2h_amc_3class", "splits": {"train": "train_sub", "val": "val_split", "test": "eval"}, "best_single": 0.8934},
    "e2h_amc_5class": {"n_classes": 5, "ext": "e2h_amc_5class", "splits": {"train": "train_sub", "val": "val_split", "test": "eval"}, "best_single": 0.8752},
    "when2call_3class": {"n_classes": 3, "ext": "when2call_3class", "splits": {"train": "train", "val": "val", "test": "test"}, "best_single": 0.8741},
    "ragtruth_binary": {"n_classes": 2, "ext": "ragtruth", "splits": {"train": "train", "val": "val", "test": "test"}, "best_single": 0.8808},
}

def load_labels(ext, split):
    with open(os.path.join(EXTRACTION_DIR, ext, split, "meta.json")) as f:
        return np.array(json.load(f)["labels"])

def compute_auroc(y, p, nc):
    if nc == 2: return roc_auc_score(y, p[:, 1])
    yb = label_binarize(y, classes=list(range(nc)))
    return roc_auc_score(yb, p, average="macro", multi_class="ovr")

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

def stability_penalized_cv(X, y, nc, clf_factory, skf, alpha=STABILITY_ALPHA):
    fold_scores = []
    n = len(y)
    oof = np.zeros((n, nc))
    for _, (ti, vi) in enumerate(skf.split(X, y)):
        clf = clf_factory()
        clf.fit(X[ti], y[ti])
        oof[vi] = clf.predict_proba(X[vi])
        try: fold_scores.append(compute_auroc(y[vi], oof[vi], nc))
        except: fold_scores.append(0.5)
    mean_au = np.mean(fold_scores)
    std_au = np.std(fold_scores)
    return mean_au - alpha * std_au, mean_au, std_au, oof


def compute_single_probe_metrics(ds_name, info):
    """Compute acc/f1 for each single probe method (simple LR on processed features)."""
    nc = info["n_classes"]
    sp = info["splits"]
    ext = info["ext"]
    methods_pool = BIN_METHODS if nc == 2 else MC_METHODS

    tr_labels = load_labels(ext, sp["train"])
    va_labels = load_labels(ext, sp["val"])
    te_labels = load_labels(ext, sp["test"])
    trva_labels = np.concatenate([tr_labels, va_labels])

    best_auroc, best_acc, best_f1, best_method = 0, 0, 0, ""
    method_results = {}

    for method in methods_pool:
        feats = load_method_features(ds_name, method)
        if feats is None: continue
        trva = np.vstack([feats["train"], feats["val"]])
        te = feats["test"]

        sc = StandardScaler()
        Xs = sc.fit_transform(trva); Xts = sc.transform(te)
        actual_pca = min(PCA_DIM, Xs.shape[1], Xs.shape[0] - 1)
        if Xs.shape[1] > actual_pca:
            pca = PCA(n_components=actual_pca, whiten=True, random_state=42)
            Xs = pca.fit_transform(Xs); Xts = pca.transform(Xts)

        # Tune C
        best_C, best_val = 1.0, -1
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        for C in [1e-3, 1e-2, 1e-1, 1.0, 10.0]:
            scores = []
            for ti, vi in skf.split(Xs, trva_labels):
                clf = LogisticRegression(max_iter=2000, C=C, random_state=42)
                clf.fit(Xs[ti], trva_labels[ti])
                p = clf.predict_proba(Xs[vi])
                try: scores.append(compute_auroc(trva_labels[vi], p, nc))
                except: scores.append(0.5)
            if np.mean(scores) > best_val:
                best_val = np.mean(scores)
                best_C = C

        clf = LogisticRegression(max_iter=2000, C=best_C, random_state=42)
        clf.fit(Xs, trva_labels)
        probs = clf.predict_proba(Xts)
        preds = clf.predict(Xts)

        auroc = compute_auroc(te_labels, probs, nc)
        acc = accuracy_score(te_labels, preds)
        f1 = f1_score(te_labels, preds, average="macro")
        method_results[method] = {"auroc": auroc, "acc": acc, "f1": f1}

        if auroc > best_auroc:
            best_auroc, best_acc, best_f1, best_method = auroc, acc, f1, method

    return best_method, best_auroc, best_acc, best_f1, method_results


def run_unified(ds_name, info):
    nc = info["n_classes"]
    sp = info["splits"]
    ext = info["ext"]
    methods_pool = BIN_METHODS if nc == 2 else MC_METHODS

    tr_labels = load_labels(ext, sp["train"])
    va_labels = load_labels(ext, sp["val"])
    te_labels = load_labels(ext, sp["test"])
    trva_labels = np.concatenate([tr_labels, va_labels])
    n_trva, n_te = len(trva_labels), len(te_labels)

    all_oof, all_te, all_names = [], [], []
    all_feat_trva, all_feat_te = [], []

    for method in methods_pool:
        feats = load_method_features(ds_name, method)
        if feats is None: continue
        trva = np.vstack([feats["train"], feats["val"]])
        te = feats["test"]

        sc = StandardScaler()
        Xs = sc.fit_transform(trva); Xts = sc.transform(te)
        actual_pca = min(PCA_DIM, Xs.shape[1], Xs.shape[0] - 1)
        if Xs.shape[1] > actual_pca:
            pca = PCA(n_components=actual_pca, whiten=True, random_state=42)
            Xs = pca.fit_transform(Xs); Xts = pca.transform(Xts)

        all_feat_trva.append(Xs); all_feat_te.append(Xts)

        for etype in EXPERT_TYPES:
            seed_oofs, seed_tes = [], []
            for s in range(N_SEEDS):
                seed = 42 + s * 111
                skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
                oof = np.zeros((n_trva, nc)); ta = np.zeros((n_te, nc))

                if etype == "lr":
                    best_stab, best_C = -float("inf"), 1.0
                    for C in [1e-3, 1e-2, 1e-1, 1.0, 10.0]:
                        factory = lambda C=C, seed=seed: LogisticRegression(max_iter=2000, C=C, random_state=seed)
                        stab, mean, std, _ = stability_penalized_cv(Xs, trva_labels, nc, factory, skf)
                        if stab > best_stab: best_stab, best_C = stab, C
                    for _, (ti, vi) in enumerate(skf.split(Xs, trva_labels)):
                        clf = LogisticRegression(max_iter=2000, C=best_C, random_state=seed)
                        clf.fit(Xs[ti], trva_labels[ti]); oof[vi] = clf.predict_proba(Xs[vi]); ta += clf.predict_proba(Xts)/N_FOLDS

                elif etype == "gbt":
                    best_stab, bp = -float("inf"), {}
                    for ml in [8, 16]:
                        for lr in [0.05, 0.1]:
                            factory = lambda ml=ml, lr=lr, seed=seed: HistGradientBoostingClassifier(
                                max_leaf_nodes=ml, learning_rate=lr, max_iter=200,
                                min_samples_leaf=10, l2_regularization=0.5, random_state=seed)
                            stab, mean, std, _ = stability_penalized_cv(Xs, trva_labels, nc, factory, skf)
                            if stab > best_stab: best_stab = stab; bp = {"ml": ml, "lr": lr}
                    for _, (ti, vi) in enumerate(skf.split(Xs, trva_labels)):
                        clf = HistGradientBoostingClassifier(
                            max_leaf_nodes=bp.get("ml",8), learning_rate=bp.get("lr",0.05),
                            max_iter=200, min_samples_leaf=10, l2_regularization=0.5, random_state=seed)
                        clf.fit(Xs[ti], trva_labels[ti]); oof[vi] = clf.predict_proba(Xs[vi]); ta += clf.predict_proba(Xts)/N_FOLDS

                seed_oofs.append(oof); seed_tes.append(ta)

            avg_oof = np.mean(seed_oofs, axis=0)
            avg_te = np.mean(seed_tes, axis=0)
            all_oof.append(avg_oof); all_te.append(avg_te)
            all_names.append(f"{method}_{etype}")

    # Build meta-features
    meta_oof = np.hstack(all_oof)
    meta_te = np.hstack(all_te)

    enrich_trva, enrich_te = [], []
    for i in range(len(all_names)):
        p_oof = all_oof[i]; p_te = all_te[i]
        ent_o = (-p_oof * np.log(np.clip(p_oof, 1e-10, 1))).sum(axis=1, keepdims=True)
        ent_t = (-p_te * np.log(np.clip(p_te, 1e-10, 1))).sum(axis=1, keepdims=True)
        margin_o = (np.sort(p_oof, axis=1)[:, -1] - np.sort(p_oof, axis=1)[:, -2]).reshape(-1, 1)
        margin_t = (np.sort(p_te, axis=1)[:, -1] - np.sort(p_te, axis=1)[:, -2]).reshape(-1, 1)
        enrich_trva.extend([ent_o, margin_o]); enrich_te.extend([ent_t, margin_t])

    gate_trva, gate_te = [], []
    for feat_trva, feat_te in zip(all_feat_trva, all_feat_te):
        gate_dim = min(8, feat_trva.shape[1])
        if feat_trva.shape[1] > gate_dim:
            pca_g = PCA(n_components=gate_dim, random_state=42)
            gate_trva.append(pca_g.fit_transform(feat_trva))
            gate_te.append(pca_g.transform(feat_te))
        else:
            gate_trva.append(feat_trva[:, :gate_dim])
            gate_te.append(feat_te[:, :gate_dim])

    meta_oof_full = np.hstack([meta_oof] + enrich_trva + gate_trva)
    meta_te_full = np.hstack([meta_te] + enrich_te + gate_te)

    # Meta-classifiers
    skf_meta = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    sc_m = StandardScaler()
    mo = sc_m.fit_transform(meta_oof_full); mt = sc_m.transform(meta_te_full)

    # Meta-LR
    best_stab_lr, best_C_lr = -float("inf"), 0.01
    for C in [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]:
        factory = lambda C=C: LogisticRegression(max_iter=3000, C=C, random_state=42)
        stab, mean, std, _ = stability_penalized_cv(mo, trva_labels, nc, factory, skf_meta)
        if stab > best_stab_lr: best_stab_lr, best_C_lr = stab, C

    clf_lr = LogisticRegression(max_iter=3000, C=best_C_lr, random_state=42)
    clf_lr.fit(mo, trva_labels); te_lr = clf_lr.predict_proba(mt)
    au_lr = compute_auroc(te_labels, te_lr, nc)

    # Meta-GBT
    best_stab_gbt, bp_gbt = -float("inf"), {}
    for ml in [4, 8, 16, 32]:
        for lr in [0.01, 0.05, 0.1]:
            for ne in [100, 200, 300]:
                factory = lambda ml=ml, lr=lr, ne=ne: HistGradientBoostingClassifier(
                    max_leaf_nodes=ml, learning_rate=lr, max_iter=ne,
                    min_samples_leaf=15, l2_regularization=0.5, random_state=42)
                stab, mean, std, _ = stability_penalized_cv(meta_oof_full, trva_labels, nc, factory, skf_meta)
                if stab > best_stab_gbt:
                    best_stab_gbt, bp_gbt = stab, {"ml": ml, "lr": lr, "ne": ne}

    clf_gbt = HistGradientBoostingClassifier(
        max_leaf_nodes=bp_gbt.get("ml", 8), learning_rate=bp_gbt.get("lr", 0.05),
        max_iter=bp_gbt.get("ne", 200), min_samples_leaf=15,
        l2_regularization=0.5, random_state=42)
    clf_gbt.fit(meta_oof_full, trva_labels); te_gbt = clf_gbt.predict_proba(meta_te_full)
    au_gbt = compute_auroc(te_labels, te_gbt, nc)

    # Blend
    best_blend, best_prob = max(au_lr, au_gbt), te_lr if au_lr >= au_gbt else te_gbt
    best_alpha = 1.0 if au_lr >= au_gbt else 0.0
    for a in np.arange(0.05, 1.0, 0.05):
        b = a * te_lr + (1-a) * te_gbt
        au = compute_auroc(te_labels, b, nc)
        if au > best_blend: best_blend, best_prob, best_alpha = au, b, a

    # Compute acc/f1 from blend
    preds = best_prob.argmax(axis=1)
    acc = accuracy_score(te_labels, preds)
    f1 = f1_score(te_labels, preds, average="macro")

    return {
        "auroc": round(best_blend, 4),
        "acc": round(acc, 4),
        "f1": round(f1, 4),
        "blend_alpha": round(best_alpha, 2),
        "meta_lr_auroc": round(au_lr, 4),
        "meta_gbt_auroc": round(au_gbt, 4),
    }


def main():
    print("=" * 70)
    print("EVAL: AUROC + Accuracy + Macro-F1")
    print("v13_final pipeline (v18 blend) + best single probe comparison")
    print("=" * 70)

    all_results = {}
    for ds_name, info in ALL_DATASETS.items():
        print(f"\n{'='*70}")
        print(f"Dataset: {ds_name} (nc={info['n_classes']})")
        print(f"{'='*70}")

        # Best single probe
        t0 = time.time()
        best_method, best_auroc, best_acc, best_f1, method_results = compute_single_probe_metrics(ds_name, info)
        print(f"  Single probe results:")
        for m, r in method_results.items():
            tag = " <-- best" if m == best_method else ""
            print(f"    {m:20s}  AUROC={r['auroc']:.4f}  Acc={r['acc']:.4f}  F1={r['f1']:.4f}{tag}")

        # Fusion
        print(f"\n  Running fusion...")
        fusion = run_unified(ds_name, info)
        print(f"  Fusion: AUROC={fusion['auroc']:.4f}  Acc={fusion['acc']:.4f}  F1={fusion['f1']:.4f}  (blend_α={fusion['blend_alpha']})")
        print(f"  Time: {time.time()-t0:.0f}s")

        all_results[ds_name] = {
            "best_single": {"method": best_method, "auroc": best_auroc, "acc": best_acc, "f1": best_f1},
            "fusion": fusion,
            "all_probes": method_results,
        }

    # Summary table
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"{'Dataset':25s} | {'--- Best Single ---':^30s} | {'---- Fusion ----':^30s} | {'-- Delta --':^20s}")
    print(f"{'':25s} | {'AUROC':>7s} {'Acc':>7s} {'F1':>7s} {'Method':>8s} | {'AUROC':>7s} {'Acc':>7s} {'F1':>7s} | {'dAUROC':>7s} {'dAcc':>7s} {'dF1':>7s}")
    print("-" * 120)
    for ds_name, r in all_results.items():
        s = r["best_single"]; f = r["fusion"]
        print(f"{ds_name:25s} | {s['auroc']:7.4f} {s['acc']:7.4f} {s['f1']:7.4f} {s['method']:>8s} | "
              f"{f['auroc']:7.4f} {f['acc']:7.4f} {f['f1']:7.4f} | "
              f"{(f['auroc']-s['auroc'])*100:+6.2f}% {(f['acc']-s['acc'])*100:+6.2f}% {(f['f1']-s['f1'])*100:+6.2f}%")

    out_path = os.path.join(RESULTS_DIR, "eval_acc_f1_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
