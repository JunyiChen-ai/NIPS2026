"""Figure 3b: t-SNE / PCA projection of probe prediction vectors.

Each method becomes a single high-dim vector by concatenating its predicted
probability-of-correct-class across all test samples in all 5 datasets.
We then project the 7 methods to 2D so cluster structure is visible spatially.
t-SNE with 7 points is unstable, so we additionally show PCA(2) which is
deterministic and often clearer for so few points.
"""
import json
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))
import warnings
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from paper_plot_style import setup_style, METHOD_LABELS

warnings.filterwarnings("ignore")
setup_style()

PROCESSED_DIR = "/home/junyi/NIPS2026/reproduce/processed_features"
EXTRACTION_DIR = "/home/junyi/NIPS2026/extraction/features"
OUT = "/home/junyi/NIPS2026/figures/fig3b_tsne"

METHODS = ["lr_probe", "pca_lr", "iti", "kb_mlp", "attn_satisfies", "sep", "step"]

# Semantic cluster colors
CLUSTERS = {
    "lr_probe":       ("hidden",     "#4C72B0"),
    "pca_lr":         ("hidden",     "#4C72B0"),
    "kb_mlp":         ("hidden",     "#4C72B0"),
    "iti":            ("attention",  "#DD8452"),
    "attn_satisfies": ("attention",  "#DD8452"),
    "sep":            ("generation", "#55A868"),
    "step":           ("generation", "#55A868"),
}

DATASETS = {
    "common_claim_3class": {"ext": "common_claim_3class", "train": "train", "val": "val", "test": "test"},
    "e2h_amc_3class":      {"ext": "e2h_amc_3class", "train": "train_sub", "val": "val_split", "test": "eval"},
    "e2h_amc_5class":      {"ext": "e2h_amc_5class", "train": "train_sub", "val": "val_split", "test": "eval"},
    "when2call_3class":    {"ext": "when2call_3class", "train": "train", "val": "val", "test": "test"},
    "ragtruth_binary":     {"ext": "ragtruth", "train": "train", "val": "val", "test": "test"},
}


def load_labels(ext, split):
    with open(os.path.join(EXTRACTION_DIR, ext, split, "meta.json")) as f:
        return np.array(json.load(f)["labels"])


def load_features(ds, method):
    base = os.path.join(PROCESSED_DIR, ds, method)
    out = {}
    for split in ["train", "val", "test"]:
        path = os.path.join(base, f"{split}.pt")
        if not os.path.exists(path):
            return None
        t = torch.load(path, map_location="cpu").float().numpy()
        if t.ndim == 1:
            t = t.reshape(-1, 1)
        out[split] = t
    return out


def get_prob_correct_vector(method):
    """Concatenate prob-of-correct-class across 5 datasets for this method."""
    chunks = []
    for ds, cfg in DATASETS.items():
        feats = load_features(ds, method)
        if feats is None:
            return None
        ext = cfg["ext"]
        tr_l = load_labels(ext, cfg["train"])
        va_l = load_labels(ext, cfg["val"])
        te_l = load_labels(ext, cfg["test"])
        trva = np.vstack([feats["train"], feats["val"]])
        trva_l = np.concatenate([tr_l, va_l])
        sc = StandardScaler()
        trva_s = sc.fit_transform(trva)
        te_s = sc.transform(feats["test"])
        if trva_s.shape[1] > 512:
            pca = PCA(n_components=256, random_state=42)
            trva_s = pca.fit_transform(trva_s)
            te_s = pca.transform(te_s)
        clf = LogisticRegression(max_iter=2000, C=0.1, random_state=42)
        clf.fit(trva_s, trva_l)
        probs = clf.predict_proba(te_s)  # (n_test, n_classes)
        # Pick prob of correct class for each sample
        n = len(te_l)
        prob_correct = probs[np.arange(n), te_l.astype(int)]
        chunks.append(prob_correct)
    return np.concatenate(chunks)


print("Building per-method prediction vectors...")
vecs = {}
for m in METHODS:
    v = get_prob_correct_vector(m)
    if v is None:
        print(f"  [SKIP] {m}: missing features")
        continue
    vecs[m] = v
    print(f"  {m:18s}: {v.shape}")

methods = list(vecs.keys())
X = np.stack([vecs[m] for m in methods], axis=0)  # (7, total_n_test)
print(f"Stacked X shape: {X.shape}")

# Standardize columns so all samples contribute equally
X_z = (X - X.mean(axis=0, keepdims=True)) / (X.std(axis=0, keepdims=True) + 1e-9)

# === PCA(2) ===
pca2 = PCA(n_components=2, random_state=42)
X_pca = pca2.fit_transform(X_z)
print(f"PCA explained variance: {pca2.explained_variance_ratio_}")

# === t-SNE(2) ===
# perplexity must be < n_samples
tsne = TSNE(n_components=2, perplexity=2, random_state=42,
            init='pca', learning_rate='auto', n_iter=2000)
X_tsne = tsne.fit_transform(X_z)

# === Plot side by side ===
fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))

for ax, X2, title in [(axes[0], X_pca, "PCA"),
                       (axes[1], X_tsne, "t-SNE")]:
    for i, m in enumerate(methods):
        color = CLUSTERS[m][1]
        ax.scatter(X2[i, 0], X2[i, 1], s=200, color=color,
                   edgecolor='white', linewidth=1.4, zorder=3)
        # Offset label so it doesn't sit on the dot
        ax.annotate(METHOD_LABELS[m], xy=(X2[i, 0], X2[i, 1]),
                    xytext=(8, 6), textcoords='offset points',
                    fontsize=12, color=color, fontweight='bold')
    ax.set_xlabel(f"{title} 1")
    ax.set_ylabel(f"{title} 2")
    # Pad a bit so labels don't get cropped
    xpad = 0.15 * (X2[:, 0].max() - X2[:, 0].min() + 1e-6)
    ypad = 0.18 * (X2[:, 1].max() - X2[:, 1].min() + 1e-6)
    ax.set_xlim(X2[:, 0].min() - xpad, X2[:, 0].max() + xpad)
    ax.set_ylim(X2[:, 1].min() - ypad, X2[:, 1].max() + ypad)

# Shared legend
import matplotlib.patches as mpatches
legend_handles = [
    mpatches.Patch(color="#4C72B0", label="hidden state"),
    mpatches.Patch(color="#DD8452", label="attention"),
    mpatches.Patch(color="#55A868", label="generation"),
]
fig.legend(handles=legend_handles, loc='lower center',
           ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02),
           fontsize=12)

plt.tight_layout()
plt.subplots_adjust(bottom=0.18)
plt.savefig(f"{OUT}.pdf")
plt.savefig(f"{OUT}.png")
print(f"Saved {OUT}.pdf and .png")
