"""Baseline AUROC heatmap: per-method AUROC on each dataset, 3 models side by side."""
import json
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from paper_plot_style import setup_style, DATASET_LABELS, METHOD_LABELS, CMAP_SEQ

setup_style()

BASE = "/home/junyi/NIPS2026/fusion/results"
MODELS = [
    ("Qwen-2.5-7B", "qwen2.5-7b"),
    ("Llama-3.1-8B", "llama3.1-8b"),
    ("Mistral-7B", "mistral-7b-v0.3"),
]
OUT = "/home/junyi/NIPS2026/figures/fig_baseline_heatmap"

METHODS = ["lr_probe", "pca_lr", "iti", "kb_mlp", "attn_satisfies", "sep", "step"]
DATASETS = ["common_claim_3class", "e2h_amc_3class", "e2h_amc_5class",
            "when2call_3class", "ragtruth_binary"]

all_vals = []
all_matrices = []
for _, subdir in MODELS:
    path = os.path.join(BASE, subdir, "oracle_complete.json")
    with open(path) as f:
        data = json.load(f)
    M = np.zeros((len(METHODS), len(DATASETS)))
    for j, ds in enumerate(DATASETS):
        ppa = data[ds]["per_probe_auroc"]
        for i, m in enumerate(METHODS):
            M[i, j] = ppa[m]
            all_vals.append(ppa[m])
    all_matrices.append(M)

vmin = max(0.0, min(all_vals) - 0.02)
vmax = min(1.0, max(all_vals) + 0.02)

# Physical width matched to LaTeX \linewidth (~5.5 in). Slight oversize
# (6.8 in) keeps text crisp under tiny downscaling.
fig = plt.figure(figsize=(6.8, 2.35))
gs = GridSpec(
    1, 4,
    width_ratios=[1, 1, 1, 0.035],
    wspace=0.10,
    left=0.085, right=0.955, top=0.88, bottom=0.28,
)
axes = [fig.add_subplot(gs[0, k]) for k in range(3)]
cax = fig.add_subplot(gs[0, 3])

for ax_idx, (model_label, M) in enumerate(zip([m[0] for m in MODELS], all_matrices)):
    ax = axes[ax_idx]
    im = ax.imshow(M, cmap=CMAP_SEQ, vmin=vmin, vmax=vmax, aspect='auto')

    for j in range(len(DATASETS)):
        col_vals = M[:, j]
        best_idx = int(np.argmax(col_vals))
        for i in range(len(METHODS)):
            val = M[i, j]
            norm_val = (val - vmin) / (vmax - vmin + 1e-9)
            text_color = 'white' if norm_val > 0.62 else '#1a1a1a'
            weight = 'bold' if i == best_idx else 'regular'
            ax.text(j, i, f"{val:.3f}", ha='center', va='center',
                    color=text_color, fontsize=7, fontweight=weight)

    ax.set_xticks(range(len(DATASETS)))
    ax.set_xticklabels([DATASET_LABELS[d] for d in DATASETS],
                       rotation=32, ha='right', fontsize=8)
    ax.set_title(model_label, fontsize=10, pad=4)

    if ax_idx == 0:
        ax.set_yticks(range(len(METHODS)))
        ax.set_yticklabels([METHOD_LABELS[m] for m in METHODS], fontsize=8.5)
    else:
        ax.set_yticks([])

    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(top=False, bottom=False, left=False, right=False)

cbar = fig.colorbar(im, cax=cax)
cbar.set_label("AUROC", fontsize=9, labelpad=4)
cbar.ax.tick_params(labelsize=8)
cbar.outline.set_visible(False)

fig.savefig(f"{OUT}.pdf")
fig.savefig(f"{OUT}.png")
print(f"Saved {OUT}.pdf and .png")
