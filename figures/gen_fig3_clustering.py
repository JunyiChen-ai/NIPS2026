"""Figure 3: Probe clustering — correlation heatmap + dendrogram (side by side).

Three semantic clusters annotated by computational source:
  hidden-state: LR Probe, PCA+LR, KB MLP
  attention:    ITI, AttnSat
  generation:   SEP, STEP
"""
import json
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.cluster.hierarchy import dendrogram
from paper_plot_style import setup_style, METHOD_LABELS

setup_style()

DATA = "/home/junyi/NIPS2026/fusion/results/probe_clustering.json"
OUT = "/home/junyi/NIPS2026/figures/fig3_clustering"

# Semantic cluster assignments
CLUSTERS = {
    "lr_probe":       ("hidden",     "#4C72B0"),  # blue
    "pca_lr":         ("hidden",     "#4C72B0"),
    "kb_mlp":         ("hidden",     "#4C72B0"),
    "iti":            ("attention",  "#DD8452"),  # orange
    "attn_satisfies": ("attention",  "#DD8452"),
    "sep":            ("generation", "#55A868"),  # green
    "step":           ("generation", "#55A868"),
}

from scipy.cluster.hierarchy import linkage as scipy_linkage
from scipy.spatial.distance import squareform

with open(DATA) as f:
    data = json.load(f)

g = data["global_average"]
orig_methods = g["methods"]
M_orig = np.array(g["avg_spearman_matrix"])

# Reorder so methods of the same family are adjacent in the heatmap
ORDER = ["lr_probe", "pca_lr", "kb_mlp",      # hidden
         "iti", "attn_satisfies",              # attention
         "sep", "step"]                        # generation
perm = [orig_methods.index(m) for m in ORDER]
methods = ORDER
M = M_orig[np.ix_(perm, perm)]

# Recompute linkage with optimal_ordering so leaves are visually grouped
dist_mat = np.clip(1 - M, 0, 2)
np.fill_diagonal(dist_mat, 0)
condensed = squareform(dist_mat, checks=False)
linkage_arr = scipy_linkage(condensed, method='ward', optimal_ordering=True)

labels = [METHOD_LABELS[m] for m in methods]

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8),
                         gridspec_kw={'width_ratios': [1.05, 1.0]})

# === Left: correlation heatmap ===
ax1 = axes[0]
im = ax1.imshow(M, cmap='YlGnBu', vmin=0.25, vmax=1.0, aspect='auto')

for i in range(len(methods)):
    for j in range(len(methods)):
        val = M[i, j]
        text_color = 'white' if val > 0.7 else 'black'
        ax1.text(j, i, f"{val:.2f}", ha='center', va='center',
                 color=text_color, fontsize=10)

ax1.set_xticks(range(len(methods)))
ax1.set_xticklabels(labels, rotation=35, ha='right')
ax1.set_yticks(range(len(methods)))
ax1.set_yticklabels(labels)

# Color tick labels by cluster
for tick, m in zip(ax1.get_xticklabels(), methods):
    tick.set_color(CLUSTERS[m][1])
    tick.set_fontweight('bold')
for tick, m in zip(ax1.get_yticklabels(), methods):
    tick.set_color(CLUSTERS[m][1])
    tick.set_fontweight('bold')

# Draw cluster bounding boxes (now all clusters are contiguous after reorder)
def add_box(ax, i0, i1, color):
    rect = mpatches.Rectangle(
        (i0 - 0.5, i0 - 0.5), i1 - i0 + 1, i1 - i0 + 1,
        fill=False, edgecolor=color, linewidth=2.8,
    )
    ax.add_patch(rect)

add_box(ax1, 0, 2, "#4C72B0")  # hidden: LR Probe, PCA+LR, KB MLP
add_box(ax1, 3, 4, "#DD8452")  # attention: ITI, AttnSat
add_box(ax1, 5, 6, "#55A868")  # generation: SEP, STEP

cbar = plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
cbar.set_label("Spearman ρ", fontsize=12)

for s in ax1.spines.values():
    s.set_visible(False)
ax1.tick_params(top=False, bottom=False, left=False, right=False)

# === Right: dendrogram with cut line and cluster colors ===
ax2 = axes[1]

# Custom link colors based on cluster — color the leaves manually after
ddata = dendrogram(
    linkage_arr,
    labels=labels,
    ax=ax2,
    color_threshold=0.65,
    above_threshold_color='#8C8C8C',
)
ax2.set_ylabel("Distance (1 − ρ)")

# Cut line at 0.65
ax2.axhline(y=0.65, color='#C44E52', linestyle='--', linewidth=1.6, alpha=0.85)
ax2.text(0.02, 0.66, "cut", color='#C44E52', fontsize=11,
         ha='left', va='bottom', transform=ax2.get_yaxis_transform())

ax2.tick_params(axis='x', rotation=35)
for tick in ax2.get_xticklabels():
    tick.set_ha('right')
    # Color by cluster
    label_to_method = {METHOD_LABELS[m]: m for m in methods}
    m = label_to_method.get(tick.get_text())
    if m is not None:
        tick.set_color(CLUSTERS[m][1])
        tick.set_fontweight('bold')

ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

# Legend for cluster colors
legend_handles = [
    mpatches.Patch(color="#4C72B0", label="hidden state"),
    mpatches.Patch(color="#DD8452", label="attention"),
    mpatches.Patch(color="#55A868", label="generation"),
]
fig.legend(handles=legend_handles, loc='lower center',
           ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.04),
           fontsize=12)

plt.tight_layout()
plt.subplots_adjust(bottom=0.22)
plt.savefig(f"{OUT}.pdf")
plt.savefig(f"{OUT}.png")
print(f"Saved {OUT}.pdf and .png")
