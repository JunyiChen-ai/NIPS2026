"""Spearman correlation heatmap + dendrogram, 3 models (2 rows x 3 cols)."""
import json
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from scipy.cluster.hierarchy import dendrogram, linkage as scipy_linkage
from scipy.spatial.distance import squareform
from paper_plot_style import setup_style, METHOD_LABELS, FAMILY_COLORS, CMAP_CORR

setup_style()

BASE = "/home/junyi/NIPS2026/fusion/results"
MODELS = [
    ("Qwen-2.5-7B", "qwen2.5-7b"),
    ("Llama-3.1-8B", "llama3.1-8b"),
    ("Mistral-7B", "mistral-7b-v0.3"),
]
OUT = "/home/junyi/NIPS2026/figures/fig3_clustering"

FAMILY_OF = {
    "lr_probe": "hidden", "pca_lr": "hidden", "kb_mlp": "hidden",
    "iti": "attention", "attn_satisfies": "attention",
    "sep": "generation", "step": "generation",
}
ORDER = ["lr_probe", "pca_lr", "kb_mlp", "iti", "attn_satisfies", "sep", "step"]

fig = plt.figure(figsize=(6.8, 5.0))
gs = GridSpec(
    3, 4,
    width_ratios=[1, 1, 1, 0.035],
    height_ratios=[1.0, 0.80, 0.07],
    hspace=1.05, wspace=0.12,
    left=0.095, right=0.945, top=0.945, bottom=0.05,
)

heatmap_axes = [fig.add_subplot(gs[0, k]) for k in range(3)]
dendro_axes = [fig.add_subplot(gs[1, 0])]
for k in range(1, 3):
    dendro_axes.append(fig.add_subplot(gs[1, k], sharey=dendro_axes[0]))
cax = fig.add_subplot(gs[0, 3])
legend_ax = fig.add_subplot(gs[2, :])
legend_ax.axis('off')

for col, (model_label, subdir) in enumerate(MODELS):
    data_path = os.path.join(BASE, subdir, "probe_clustering.json")
    with open(data_path) as f:
        data = json.load(f)

    g = data["global_average"]
    orig_methods = g["methods"]
    M_orig = np.array(g["avg_spearman_matrix"])
    perm = [orig_methods.index(m) for m in ORDER]
    methods = ORDER
    M = M_orig[np.ix_(perm, perm)]

    dist_mat = np.clip(1 - M, 0, 2)
    np.fill_diagonal(dist_mat, 0)
    condensed = squareform(dist_mat, checks=False)
    linkage_arr = scipy_linkage(condensed, method='ward', optimal_ordering=True)
    labels = [METHOD_LABELS[m] for m in methods]

    ax1 = heatmap_axes[col]
    im = ax1.imshow(M, cmap=CMAP_CORR, vmin=0.15, vmax=1.0, aspect='auto')
    for i in range(len(methods)):
        for j in range(len(methods)):
            val = M[i, j]
            text_color = 'white' if val > 0.70 else '#1a1a1a'
            ax1.text(j, i, f"{val:.2f}", ha='center', va='center',
                     color=text_color, fontsize=6)

    ax1.set_xticks(range(len(methods)))
    ax1.set_xticklabels(labels, rotation=40, ha='right', fontsize=7.5)
    ax1.set_title(model_label, fontsize=10, pad=4)

    if col == 0:
        ax1.set_yticks(range(len(methods)))
        ax1.set_yticklabels(labels, fontsize=7.5)
    else:
        ax1.set_yticks([])

    for tick, m in zip(ax1.get_xticklabels(), methods):
        tick.set_color(FAMILY_COLORS[FAMILY_OF[m]])
        tick.set_fontweight('bold')
    if col == 0:
        for tick, m in zip(ax1.get_yticklabels(), methods):
            tick.set_color(FAMILY_COLORS[FAMILY_OF[m]])
            tick.set_fontweight('bold')

    def add_box(ax, i0, i1, color):
        rect = mpatches.Rectangle(
            (i0 - 0.5, i0 - 0.5), i1 - i0 + 1, i1 - i0 + 1,
            fill=False, edgecolor=color, linewidth=1.4,
        )
        ax.add_patch(rect)
    add_box(ax1, 0, 2, FAMILY_COLORS['hidden'])
    add_box(ax1, 3, 4, FAMILY_COLORS['attention'])
    add_box(ax1, 5, 6, FAMILY_COLORS['generation'])

    for s in ax1.spines.values():
        s.set_visible(False)
    ax1.tick_params(top=False, bottom=False, left=False, right=False)

    ax2 = dendro_axes[col]
    # Start with all links gray, then recolor each leaf's pedestal (the
    # vertical segment rising directly from the x-axis label) in its
    # family color. This makes the family assignment of every probe
    # visible even when Ward merges blend families early.
    n_leaves = len(methods)
    leaf_family = [FAMILY_OF[m] for m in methods]
    NEUTRAL = '#8C8C8C'

    ddata = dendrogram(
        linkage_arr, labels=labels, ax=ax2,
        link_color_func=lambda _: NEUTRAL,
    )

    # ddata['icoord'][i] / ddata['dcoord'][i] describe each inverted-U link:
    # four (x, y) points forming left-up, top, right-down. A leaf pedestal
    # is the segment where the bottom y equals 0.
    leaf_x_to_idx = {
        int((pos + 1) / 10 - 0.5): idx for idx, pos in enumerate(ddata['leaves'])
    }
    # Actual x positions from scipy are 5, 15, 25, ... so map those directly.
    leaf_positions = {5 + 10 * k: ddata['leaves'][k] for k in range(n_leaves)}
    for icoord, dcoord in zip(ddata['icoord'], ddata['dcoord']):
        # Left leg: (icoord[0], 0) -> (icoord[0], dcoord[1])
        if dcoord[0] == 0 and int(icoord[0]) in leaf_positions:
            leaf_idx = leaf_positions[int(icoord[0])]
            color = FAMILY_COLORS[leaf_family[leaf_idx]]
            ax2.plot([icoord[0], icoord[0]], [0, dcoord[1]],
                     color=color, linewidth=1.6, solid_capstyle='round')
        # Right leg: (icoord[3], 0) -> (icoord[3], dcoord[2])
        if dcoord[3] == 0 and int(icoord[3]) in leaf_positions:
            leaf_idx = leaf_positions[int(icoord[3])]
            color = FAMILY_COLORS[leaf_family[leaf_idx]]
            ax2.plot([icoord[3], icoord[3]], [0, dcoord[2]],
                     color=color, linewidth=1.6, solid_capstyle='round')
    if col == 0:
        ax2.set_ylabel(r"Dist. $(1{-}\rho)$", fontsize=9)
        ax2.tick_params(axis='y', labelsize=8)
    else:
        ax2.tick_params(axis='y', labelleft=False, left=False)
    ax2.axhline(y=0.65, color='#B8384E', linestyle='--',
                linewidth=0.9, alpha=0.8)

    ax2.tick_params(axis='x', rotation=40, labelsize=7.5)
    label_to_method = {METHOD_LABELS[m]: m for m in methods}
    for tick in ax2.get_xticklabels():
        tick.set_ha('right')
        m = label_to_method.get(tick.get_text())
        if m is not None:
            tick.set_color(FAMILY_COLORS[FAMILY_OF[m]])
            tick.set_fontweight('bold')

    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    if col > 0:
        ax2.spines['left'].set_visible(False)

cbar = fig.colorbar(im, cax=cax)
cbar.set_label(r"$\rho$", fontsize=9, labelpad=3)
cbar.ax.tick_params(labelsize=8)
cbar.outline.set_visible(False)

legend_handles = [
    mpatches.Patch(color=FAMILY_COLORS['hidden'],     label='Hidden state'),
    mpatches.Patch(color=FAMILY_COLORS['attention'],  label='Attention'),
    mpatches.Patch(color=FAMILY_COLORS['generation'], label='Generation'),
]
legend_ax.legend(
    handles=legend_handles, loc='center', ncol=3,
    frameon=False, fontsize=8.5, columnspacing=1.8, handlelength=1.4,
)

fig.savefig(f"{OUT}.pdf")
fig.savefig(f"{OUT}.png")
print(f"Saved {OUT}.pdf and .png")
