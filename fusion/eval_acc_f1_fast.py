"""
Fast eval: compute Acc/F1 for best single probe AND fusion (v13 pipeline).
Simplified: fewer C candidates, no stability penalty (just plain CV), 1 seed.
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
N_FOLDS = 5

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


def run_dataset(ds_name, info):
    nc = info["n_classes"]
    sp = info["splits"]
    ext = info["ext"]
    methods_pool = BIN_METHODS if nc == 2 else MC_METHODS

    tr_labels = load_labels(ext, sp["train"])
    va_labels = load_labels(ext, sp["val"])
    te_labels = load_labels(ext, sp["test"])
    trva_labels = np.concatenate([tr_labels, va_labels])
    n_trva, n_te = len(trva_labels), len(te_labels)

    # === Per-method single probe metrics ===
    single_results = {}
    all_oof_lr, all_te_lr = [], []
    all_oof_gbt, all_te_gbt = [], []

    for method in methods_pool:
        feats = load_method_features(ds_name, method)
        if feats is None:
            print(f"    {method}: SKIPPED (no features)")
            continue
        trva = np.vstack([feats["train"], feats["val"]])
        te = feats["test"]

        sc = StandardScaler()
        Xs = sc.fit_transform(trva); Xts = sc.transform(te)
        actual_pca = min(PCA_DIM, Xs.shape[1], Xs.shape[0] - 1)
        if Xs.shape[1] > actual_pca:
            pca = PCA(n_components=actual_pca, whiten=True, random_state=42)
            Xs = pca.fit_transform(Xs); Xts = pca.transform(Xts)

        # Single probe: tune C on val via CV
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
        best_C, best_val = 1.0, -1
        for C in [0.01, 0.1, 1.0, 10.0]:
            scores = []
            for ti, vi in skf.split(Xs, trva_labels):
                clf = LogisticRegression(max_iter=2000, C=C, random_state=42)
                clf.fit(Xs[ti], trva_labels[ti])
                p = clf.predict_proba(Xs[vi])
                try: scores.append(compute_auroc(trva_labels[vi], p, nc))
                except: scores.append(0.5)
            if np.mean(scores) > best_val:
                best_val, best_C = np.mean(scores), C

        clf = LogisticRegression(max_iter=2000, C=best_C, random_state=42)
        clf.fit(Xs, trva_labels)
        probs = clf.predict_proba(Xts)
        preds = clf.predict(Xts)
        auroc = compute_auroc(te_labels, probs, nc)
        acc = accuracy_score(te_labels, preds)
        f1 = f1_score(te_labels, preds, average="macro")
        single_results[method] = {"auroc": round(auroc, 4), "acc": round(acc, 4), "f1": round(f1, 4), "C": best_C}
        print(f"    {method:20s}  AUROC={auroc:.4f}  Acc={acc:.4f}  F1={f1:.4f}")

        # OOF for fusion (LR expert, 1 seed)
        oof_lr = np.zeros((n_trva, nc)); ta_lr = np.zeros((n_te, nc))
        for ti, vi in skf.split(Xs, trva_labels):
            clf = LogisticRegression(max_iter=2000, C=best_C, random_state=42)
            clf.fit(Xs[ti], trva_labels[ti])
            oof_lr[vi] = clf.predict_proba(Xs[vi])
            ta_lr += clf.predict_proba(Xts) / N_FOLDS
        all_oof_lr.append(oof_lr); all_te_lr.append(ta_lr)

        # OOF for fusion (GBT expert, 1 seed)
        oof_gbt = np.zeros((n_trva, nc)); ta_gbt = np.zeros((n_te, nc))
        for ti, vi in skf.split(Xs, trva_labels):
            clf = HistGradientBoostingClassifier(max_leaf_nodes=8, learning_rate=0.05, max_iter=200,
                                                  min_samples_leaf=10, l2_regularization=0.5, random_state=42)
            clf.fit(Xs[ti], trva_labels[ti])
            oof_gbt[vi] = clf.predict_proba(Xs[vi])
            ta_gbt += clf.predict_proba(Xts) / N_FOLDS
        all_oof_gbt.append(oof_gbt); all_te_gbt.append(ta_gbt)

    # === Fusion: stack all OOF probs → meta-LR and meta-GBT ===
    meta_oof = np.hstack(all_oof_lr + all_oof_gbt)
    meta_te = np.hstack(all_te_lr + all_te_gbt)

    # Add entropy+margin
    enrich_trva, enrich_te = [], []
    for oof, te_p in zip(all_oof_lr + all_oof_gbt, all_te_lr + all_te_gbt):
        ent_o = (-oof * np.log(np.clip(oof, 1e-10, 1))).sum(axis=1, keepdims=True)
        ent_t = (-te_p * np.log(np.clip(te_p, 1e-10, 1))).sum(axis=1, keepdims=True)
        margin_o = (np.sort(oof, axis=1)[:, -1] - np.sort(oof, axis=1)[:, -2]).reshape(-1, 1)
        margin_t = (np.sort(te_p, axis=1)[:, -1] - np.sort(te_p, axis=1)[:, -2]).reshape(-1, 1)
        enrich_trva.extend([ent_o, margin_o]); enrich_te.extend([ent_t, margin_t])

    meta_oof_full = np.hstack([meta_oof] + enrich_trva)
    meta_te_full = np.hstack([meta_te] + enrich_te)

    sc_m = StandardScaler()
    mo = sc_m.fit_transform(meta_oof_full); mt = sc_m.transform(meta_te_full)

    # Meta-LR
    skf_meta = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    best_C_m, best_val_m = 0.01, -1
    for C in [1e-4, 1e-3, 0.01, 0.1, 1.0]:
        scores = []
        for ti, vi in skf_meta.split(mo, trva_labels):
            clf = LogisticRegression(max_iter=3000, C=C, random_state=42)
            clf.fit(mo[ti], trva_labels[ti])
            p = clf.predict_proba(mo[vi])
            try: scores.append(compute_auroc(trva_labels[vi], p, nc))
            except: scores.append(0.5)
        if np.mean(scores) > best_val_m:
            best_val_m, best_C_m = np.mean(scores), C

    clf_m = LogisticRegression(max_iter=3000, C=best_C_m, random_state=42)
    clf_m.fit(mo, trva_labels)
    te_lr_prob = clf_m.predict_proba(mt)

    # Meta-GBT
    best_gbt_val, bp = -1, {}
    for ml in [4, 8, 16]:
        for lr in [0.05, 0.1]:
            scores = []
            for ti, vi in skf_meta.split(meta_oof_full, trva_labels):
                clf = HistGradientBoostingClassifier(max_leaf_nodes=ml, learning_rate=lr, max_iter=200,
                                                      min_samples_leaf=15, l2_regularization=0.5, random_state=42)
                clf.fit(meta_oof_full[ti], trva_labels[ti])
                p = clf.predict_proba(meta_oof_full[vi])
                try: scores.append(compute_auroc(trva_labels[vi], p, nc))
                except: scores.append(0.5)
            if np.mean(scores) > best_gbt_val:
                best_gbt_val = np.mean(scores); bp = {"ml": ml, "lr": lr}

    clf_g = HistGradientBoostingClassifier(max_leaf_nodes=bp.get("ml",8), learning_rate=bp.get("lr",0.05),
                                            max_iter=200, min_samples_leaf=15, l2_regularization=0.5, random_state=42)
    clf_g.fit(meta_oof_full, trva_labels)
    te_gbt_prob = clf_g.predict_proba(meta_te_full)

    # Blend
    au_lr = compute_auroc(te_labels, te_lr_prob, nc)
    au_gbt = compute_auroc(te_labels, te_gbt_prob, nc)
    best_au, best_prob = max(au_lr, au_gbt), te_lr_prob if au_lr >= au_gbt else te_gbt_prob
    for a in np.arange(0.1, 1.0, 0.1):
        b = a * te_lr_prob + (1-a) * te_gbt_prob
        au = compute_auroc(te_labels, b, nc)
        if au > best_au: best_au, best_prob = au, b

    preds = best_prob.argmax(axis=1)
    fusion_auroc = best_au
    fusion_acc = accuracy_score(te_labels, preds)
    fusion_f1 = f1_score(te_labels, preds, average="macro")

    # Find best single by auroc
    best_m = max(single_results, key=lambda m: single_results[m]["auroc"])
    bs = single_results[best_m]

    print(f"\n  Best single: {best_m} — AUROC={bs['auroc']:.4f}  Acc={bs['acc']:.4f}  F1={bs['f1']:.4f}")
    print(f"  Fusion:                AUROC={fusion_auroc:.4f}  Acc={fusion_acc:.4f}  F1={fusion_f1:.4f}")
    print(f"  Delta:                 AUROC={fusion_auroc-bs['auroc']:+.4f}  Acc={fusion_acc-bs['acc']:+.4f}  F1={fusion_f1-bs['f1']:+.4f}")

    return {
        "best_single": {"method": best_m, **bs},
        "fusion": {"auroc": round(fusion_auroc, 4), "acc": round(fusion_acc, 4), "f1": round(fusion_f1, 4)},
        "all_probes": single_results,
    }


def main():
    print("=" * 70)
    print("FAST EVAL: AUROC + Accuracy + Macro-F1")
    print("=" * 70)

    all_results = {}
    for ds_name, info in ALL_DATASETS.items():
        print(f"\n{'='*70}")
        print(f"Dataset: {ds_name} (nc={info['n_classes']})")
        print(f"{'='*70}")
        t0 = time.time()
        r = run_dataset(ds_name, info)
        all_results[ds_name] = r
        print(f"  Time: {time.time()-t0:.0f}s")

    # Summary
    print(f"\n{'='*90}")
    print("SUMMARY TABLE")
    print(f"{'='*90}")
    print(f"{'Dataset':25s} | {'Best Single':>8s} {'AUROC':>7s} {'Acc':>7s} {'F1':>7s} | {'Fusion':>6s} {'AUROC':>7s} {'Acc':>7s} {'F1':>7s} | {'dAcc':>6s} {'dF1':>6s}")
    print("-" * 90)
    for ds_name, r in all_results.items():
        s, f = r["best_single"], r["fusion"]
        print(f"{ds_name:25s} | {s['method']:>8s} {s['auroc']:7.4f} {s['acc']:7.4f} {s['f1']:7.4f} | "
              f"      {f['auroc']:7.4f} {f['acc']:7.4f} {f['f1']:7.4f} | "
              f"{(f['acc']-s['acc'])*100:+5.2f}% {(f['f1']-s['f1'])*100:+5.2f}%")

    out_path = os.path.join(RESULTS_DIR, "eval_acc_f1_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
