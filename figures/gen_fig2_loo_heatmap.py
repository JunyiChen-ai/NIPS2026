"""Figure 2: Leave-one-method-out contribution heatmap (7 methods × 5 datasets)."""
import json
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from paper_plot_style import setup_style, DATASET_LABELS, METHOD_LABELS

setup_style()

DATA = "/home/junyi/NIPS2026/fusion/results/leave_one_method_out.json"
OUT = "/home/junyi/NIPS2026/figures/fig2_loo_heatmap"

with open(DATA) as f:
    data = json.load(f)

METHODS = ["lr_probe", "pca_lr", "iti", "kb_mlp", "attn_satisfies", "sep", "step"]
DATASETS = ["common_claim_3class", "e2h_amc_3class", "e2h_amc_5class",
            "when2call_3class", "ragtruth_binary"]

# Build matrix: rows = methods, cols = datasets, values = contribution in %
M = np.zeros((len(METHODS), len(DATASETS)))
for j, ds in enumerate(DATASETS):
    abl = data[ds]["ablations"]
    for i, m in enumerate(METHODS):
        M[i, j] = abl[m]["contribution"] * 100  # to %

fig, ax = plt.subplots(figsize=(6.5, 4.6))

vmax = max(abs(M.min()), abs(M.max()))
norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

im = ax.imshow(M, cmap='RdBu_r', norm=norm, aspect='auto')

# Annotate cells
for i in range(len(METHODS)):
    for j in range(len(DATASETS)):
        val = M[i, j]
        # Pick text color based on cell brightness
        text_color = 'white' if abs(val) > vmax * 0.55 else 'black'
        ax.text(j, i, f"{val:+.2f}", ha='center', va='center',
                color=text_color, fontsize=11)

ax.set_xticks(range(len(DATASETS)))
ax.set_xticklabels([DATASET_LABELS[d] for d in DATASETS], rotation=20, ha='right')
ax.set_yticks(range(len(METHODS)))
ax.set_yticklabels([METHOD_LABELS[m] for m in METHODS])

# Colorbar
cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Contribution (%)", fontsize=13)

# Hide spines
for s in ax.spines.values():
    s.set_visible(False)
ax.tick_params(top=False, bottom=False, left=False, right=False)

plt.savefig(f"{OUT}.pdf")
plt.savefig(f"{OUT}.png")
print(f"Saved {OUT}.pdf and .png")
