"""Quick ablation: LR + GBT experts only (1 linear + 1 nonlinear)."""
import os, sys, json, time, warnings
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import HistGradientBoostingClassifier

warnings.filterwarnings("ignore")

MODEL = "qwen2.5-7b"
PROCESSED_DIR = f"/home/junyi/NIPS2026/reproduce/processed_features/{MODEL}"
EXTRACTION_DIR = f"/home/junyi/NIPS2026/extraction/features/{MODEL}"

PCA_DIM = 128
EXPERT_TYPES = ["lr", "gbt"]  # only 2
N_SEEDS = 1
N_FOLDS = 5
C_GRID = [1e-2, 1e-1, 1.0]

MC_METHODS = ["lr_probe", "pca_lr", "iti", "kb_mlp", "attn_satisfies", "sep", "step"]

ALL_DATASETS = {
    "common_claim_3class": {"n_classes": 3, "ext": "common_claim_3class", "train": "train", "val": "val", "test": "test"},
    "e2h_amc_3class": {"n_classes": 3, "ext": "e2h_amc_3class", "train": "train_sub", "val": "val_split", "test": "eval"},
    "e2h_amc_5class": {"n_classes": 5, "ext": "e2h_amc_5class", "train": "train_sub", "val": "val_split", "test": "eval"},
    "when2call_3class": {"n_classes": 3, "ext": "when2call_3class", "train": "train", "val": "val", "test": "test"},
    "ragtruth_binary": {"n_classes": 2, "ext": "ragtruth", "train": "train", "val": "val", "test": "test"},
}

# Load best_single from oracle
oracle_path = f"/home/junyi/NIPS2026/fusion/results/{MODEL}/oracle_complete.json"
with open(oracle_path) as f:
    oracle = json.load(f)
BEST_SINGLE = {ds: oracle[ds]["best_single_auroc"] for ds in ALL_DATASETS}

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

def train_expert_oof(Xs, Xts, labels, nc, etype, seed, skf):
    n_trva, n_te = len(labels), Xts.shape[0]
    oof = np.zeros((n_trva, nc))
    ta = np.zeros((n_te, nc))
    if etype == "lr":
        best_au, best_C = -1, 1.0
        for C in C_GRID:
            inner = np.zeros((n_trva, nc))
            for _, (ti, vi) in enumerate(skf.split(Xs, labels)):
                clf = LogisticRegression(max_iter=2000, C=C, random_state=seed)
                clf.fit(Xs[ti], labels[ti]); inner[vi] = clf.predict_proba(Xs[vi])
            try: au = compute_auroc(labels, inner, nc)
            except: au = 0.5
            if au > best_au: best_au, best_C = au, C
        for _, (ti, vi) in enumerate(skf.split(Xs, labels)):
            clf = LogisticRegression(max_iter=2000, C=best_C, random_state=seed)
            clf.fit(Xs[ti], labels[ti]); oof[vi] = clf.predict_proba(Xs[vi]); ta += clf.predict_proba(Xts)/N_FOLDS
    elif etype == "gbt":
        for _, (ti, vi) in enumerate(skf.split(Xs, labels)):
            clf = HistGradientBoostingClassifier(max_leaf_nodes=16, learning_rate=0.1, max_iter=200, min_samples_leaf=10, l2_regularization=0.5, random_state=seed)
            clf.fit(Xs[ti], labels[ti]); oof[vi] = clf.predict_proba(Xs[vi]); ta += clf.predict_proba(Xts)/N_FOLDS
    return oof, ta

def run(ds_name, info):
    nc = info["n_classes"]
    ext = info["ext"]
    tr_labels = load_labels(ext, info["train"])
    va_labels = load_labels(ext, info["val"])
    te_labels = load_labels(ext, info["test"])
    trva_labels = np.concatenate([tr_labels, va_labels])

    all_oof, all_te = [], []
    for method in MC_METHODS:
        feats = load_method_features(ds_name, method)
        if feats is None: continue
        trva = np.vstack([feats["train"], feats["val"]])
        te = feats["test"]
        sc = StandardScaler()
        Xs = sc.fit_transform(trva); Xts = sc.transform(te)
        actual_pca = min(PCA_DIM, Xs.shape[1], Xs.shape[0] - 1)
        if Xs.shape[1] > actual_pca:
            pca = PCA(n_components=actual_pca, random_state=42)
            Xs = pca.fit_transform(Xs); Xts = pca.transform(Xts)
        for etype in EXPERT_TYPES:
            seed_oofs, seed_tes = [], []
            for s in range(N_SEEDS):
                seed = 42 + s * 111
                skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
                oof, ta = train_expert_oof(Xs, Xts, trva_labels, nc, etype, seed, skf)
                seed_oofs.append(oof); seed_tes.append(ta)
            all_oof.append(np.mean(seed_oofs, axis=0))
            all_te.append(np.mean(seed_tes, axis=0))

    meta_oof = np.hstack(all_oof); meta_te = np.hstack(all_te)
    n_trva = len(trva_labels)

    # Meta: single L2-LR
    sc_m = StandardScaler()
    mo = sc_m.fit_transform(meta_oof); mt = sc_m.transform(meta_te)
    skf_meta = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    best_au_l2, best_C_l2 = -1, 0.01
    for C in [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]:
        inner = np.zeros((n_trva, nc))
        for _, (ti, vi) in enumerate(skf_meta.split(mo, trva_labels)):
            clf = LogisticRegression(max_iter=3000, C=C, penalty='l2', solver='lbfgs', random_state=42)
            clf.fit(mo[ti], trva_labels[ti]); inner[vi] = clf.predict_proba(mo[vi])
        try: au = compute_auroc(trva_labels, inner, nc)
        except: au = 0.5
        if au > best_au_l2: best_au_l2, best_C_l2 = au, C
    clf_l2 = LogisticRegression(max_iter=3000, C=best_C_l2, penalty='l2', solver='lbfgs', random_state=42)
    clf_l2.fit(mo, trva_labels)
    best_blend = compute_auroc(te_labels, clf_l2.predict_proba(mt), nc)

    bs = BEST_SINGLE[ds_name]
    delta = best_blend - bs
    return {"auroc": round(best_blend, 4), "best_single": round(bs, 4), "delta": round(delta, 4), "delta_pct": f"{delta*100:+.2f}%"}

print("LR + GBT only (1 linear + 1 nonlinear)")
print("=" * 60)
results = {}
for ds_name, info in ALL_DATASETS.items():
    t0 = time.time()
    r = run(ds_name, info)
    results[ds_name] = r
    print(f"{ds_name:25s} best={r['best_single']:.4f} fusion={r['auroc']:.4f} delta={r['delta_pct']} [{time.time()-t0:.0f}s]")

deltas = [r["delta"]*100 for r in results.values()]
print(f"\nAvg delta: {np.mean(deltas):+.2f}%")
