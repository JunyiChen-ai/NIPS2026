"""
Exp 5: Probe Error Correlation & Clustering.
Computes pairwise prediction correlation + error agreement between probing methods.
Produces correlation matrices and hierarchical clustering data.
"""

import os, json, warnings
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, cohen_kappa_score
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

import argparse as _argparse
_ap = _argparse.ArgumentParser(add_help=False)
_ap.add_argument("--model", default="qwen2.5-7b")
_cli, _ = _ap.parse_known_args()
_MODEL = _cli.model
_BASE_PROCESSED = "/home/junyi/NIPS2026/reproduce/processed_features"
_BASE_EXTRACTION = "/home/junyi/NIPS2026/extraction/features"
_BASE_RESULTS = "/home/junyi/NIPS2026/fusion/results"
PROCESSED_DIR = os.path.join(_BASE_PROCESSED, _MODEL) if _MODEL else _BASE_PROCESSED
EXTRACTION_DIR = os.path.join(_BASE_EXTRACTION, _MODEL) if _MODEL else _BASE_EXTRACTION
RESULTS_DIR = os.path.join(_BASE_RESULTS, _MODEL) if _MODEL else _BASE_RESULTS
os.makedirs(RESULTS_DIR, exist_ok=True)

MC_METHODS = ["lr_probe", "pca_lr", "iti", "kb_mlp", "attn_satisfies", "sep", "step"]

ALL_DATASETS = {
    "common_claim_3class": {"n_classes": 3, "ext": "common_claim_3class", "train": "train", "val": "val", "test": "test"},
    "e2h_amc_3class":      {"n_classes": 3, "ext": "e2h_amc_3class", "train": "train_sub", "val": "val_split", "test": "eval"},
    "e2h_amc_5class":      {"n_classes": 5, "ext": "e2h_amc_5class", "train": "train_sub", "val": "val_split", "test": "eval"},
    "when2call_3class":    {"n_classes": 3, "ext": "when2call_3class", "train": "train", "val": "val", "test": "test"},
    "ragtruth_binary":     {"n_classes": 2, "ext": "ragtruth", "train": "train", "val": "val", "test": "test"},
}


def load_labels(ext, split):
    with open(os.path.join(EXTRACTION_DIR, ext, split, "meta.json")) as f:
        return np.array(json.load(f)["labels"])


def compute_auroc(y, p, nc):
    if nc == 2:
        return roc_auc_score(y, p[:, 1])
    yb = label_binarize(y, classes=list(range(nc)))
    return roc_auc_score(yb, p, average="macro", multi_class="ovr")


def main():
    print("=" * 70)
    print("EXP 5: Probe Error Correlation & Clustering")
    print("=" * 70)

    results = {}

    for ds_name, cfg in ALL_DATASETS.items():
        nc = cfg["n_classes"]
        ext = cfg["ext"]
        probe_dir = os.path.join(PROCESSED_DIR, ds_name)

        tr_labels = load_labels(ext, cfg["train"])
        va_labels = load_labels(ext, cfg["val"])
        te_labels = load_labels(ext, cfg["test"])
        trva_labels = np.concatenate([tr_labels, va_labels])

        # Train each method and get test predictions
        preds = {}
        for method in MC_METHODS:
            tr_path = os.path.join(probe_dir, method, "train.pt")
            if not os.path.exists(tr_path):
                continue
            tr = torch.load(tr_path, map_location="cpu").float().numpy()
            va = torch.load(os.path.join(probe_dir, method, "val.pt"), map_location="cpu").float().numpy()
            te = torch.load(os.path.join(probe_dir, method, "test.pt"), map_location="cpu").float().numpy()
            if tr.ndim == 1:
                tr, va, te = tr.reshape(-1, 1), va.reshape(-1, 1), te.reshape(-1, 1)

            trva = np.vstack([tr, va])
            sc = StandardScaler()
            trva_s = sc.fit_transform(trva)
            te_s = sc.transform(te)
            if trva_s.shape[1] > 512:
                pca = PCA(n_components=256, random_state=42)
                trva_s = pca.fit_transform(trva_s)
                te_s = pca.transform(te_s)

            try:
                clf = LogisticRegression(max_iter=2000, C=0.1, random_state=42)
                clf.fit(trva_s, trva_labels)
                preds[method] = clf.predict_proba(te_s)
            except:
                pass

        methods = list(preds.keys())
        n_methods = len(methods)
        n_te = len(te_labels)
        print(f"\n{ds_name}: {n_methods} methods, {n_te} test samples")

        # Per-method predictions and correctness
        pred_labels = {}
        correct = {}
        aurocs = {}
        for m in methods:
            pred_labels[m] = np.argmax(preds[m], axis=1)
            correct[m] = (pred_labels[m] == te_labels).astype(int)
            try:
                aurocs[m] = round(compute_auroc(te_labels, preds[m], nc), 4)
            except:
                aurocs[m] = 0.5

        # 1) Spearman rank correlation of predicted probabilities
        # For multi-class: use predicted prob of correct class for each sample
        prob_correct = {}
        for m in methods:
            prob_correct[m] = np.array([preds[m][i, int(te_labels[i])] for i in range(n_te)])

        spearman_matrix = np.ones((n_methods, n_methods))
        for i in range(n_methods):
            for j in range(i + 1, n_methods):
                rho, _ = spearmanr(prob_correct[methods[i]], prob_correct[methods[j]])
                spearman_matrix[i, j] = rho
                spearman_matrix[j, i] = rho

        # 2) Cohen's kappa on predicted labels
        kappa_matrix = np.ones((n_methods, n_methods))
        for i in range(n_methods):
            for j in range(i + 1, n_methods):
                k = cohen_kappa_score(pred_labels[methods[i]], pred_labels[methods[j]])
                kappa_matrix[i, j] = k
                kappa_matrix[j, i] = k

        # 3) Jaccard similarity on error sets
        jaccard_matrix = np.ones((n_methods, n_methods))
        for i in range(n_methods):
            for j in range(i + 1, n_methods):
                err_i = set(np.where(correct[methods[i]] == 0)[0])
                err_j = set(np.where(correct[methods[j]] == 0)[0])
                if len(err_i | err_j) == 0:
                    jacc = 1.0
                else:
                    jacc = len(err_i & err_j) / len(err_i | err_j)
                jaccard_matrix[i, j] = jacc
                jaccard_matrix[j, i] = jacc

        # 4) Hierarchical clustering on (1 - spearman) distance
        dist = np.clip(1 - spearman_matrix, 0, 2)
        np.fill_diagonal(dist, 0)
        condensed = squareform(dist)
        Z = linkage(condensed, method='ward')

        # Print correlation summary
        print(f"  Spearman correlation (prob of correct class):")
        header = "  " + " " * 18 + "".join(f"{m[:8]:>9s}" for m in methods)
        print(header)
        for i, m in enumerate(methods):
            row = f"  {m:18s}" + "".join(f"{spearman_matrix[i, j]:9.3f}" for j in range(n_methods))
            print(row)

        print(f"\n  Error Jaccard similarity:")
        for i, m in enumerate(methods):
            row = f"  {m:18s}" + "".join(f"{jaccard_matrix[i, j]:9.3f}" for j in range(n_methods))
            print(row)

        # Store results
        results[ds_name] = {
            "methods": methods,
            "aurocs": aurocs,
            "spearman_matrix": spearman_matrix.tolist(),
            "kappa_matrix": kappa_matrix.tolist(),
            "jaccard_matrix": jaccard_matrix.tolist(),
            "linkage": Z.tolist(),
            "n_test": n_te,
        }

    # Global average across datasets
    print(f"\n{'='*70}")
    print("GLOBAL AVERAGE Spearman Correlation")
    print("=" * 70)
    all_methods = MC_METHODS
    n = len(all_methods)
    avg_spearman = np.zeros((n, n))
    count = 0
    for ds, r in results.items():
        if r["methods"] == all_methods:
            avg_spearman += np.array(r["spearman_matrix"])
            count += 1
    if count > 0:
        avg_spearman /= count
        header = "  " + " " * 18 + "".join(f"{m[:8]:>9s}" for m in all_methods)
        print(header)
        for i, m in enumerate(all_methods):
            row = f"  {m:18s}" + "".join(f"{avg_spearman[i, j]:9.3f}" for j in range(n))
            print(row)

        # Global clustering
        dist = np.clip(1 - avg_spearman, 0, 2)
        np.fill_diagonal(dist, 0)
        condensed = squareform(dist)
        Z_global = linkage(condensed, method='ward')
        results["global_average"] = {
            "methods": all_methods,
            "avg_spearman_matrix": avg_spearman.tolist(),
            "linkage": Z_global.tolist(),
            "n_datasets": count,
        }

    out_path = os.path.join(RESULTS_DIR, "probe_clustering.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
