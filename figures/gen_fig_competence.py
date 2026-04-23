"""Redundancy vs. marginal-gain scatter plot, 3 models side by side.

For every probe-ladder addition event (n_methods >= 2) we compute:
    x = mean Spearman rho between the newly-added probe and the probes
        already in the ensemble (using the global-average matrix from
        probe_clustering.json).
    y = incremental_gain (AUROC) * 100   (pp AUROC)

Per-model Spearman correlations plus the pooled 90-event correlation are
printed to stdout. Events whose added probe is not in the 7-method clustering
matrix (e.g. the binary-only `mm_probe` added on RAGTruth) are dropped so the
per-model event count is 30 = 5 datasets x 6 steps.
"""
import json
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from scipy.stats import spearmanr

from paper_plot_style import (
    setup_style,
    FAMILY_COLORS,
    DATASET_COLORS,
    DATASET_LABELS,
    METHOD_LABELS,
)

setup_style()

BASE = "/home/junyi/NIPS2026/fusion/results"
MODELS = [
    ("qwen2.5-7b", "Qwen-2.5-7B"),
    ("llama3.1-8b", "Llama-3.1-8B"),
    ("mistral-7b-v0.3", "Mistral-7B"),
]
OUT = "/home/junyi/NIPS2026/figures/fig_competence"

DATASETS = [
    "common_claim_3class",
    "e2h_amc_3class",
    "e2h_amc_5class",
    "when2call_3class",
    "ragtruth_binary",
]

FAMILY_OF = {
    "lr_probe": "hidden", "pca_lr": "hidden", "kb_mlp": "hidden",
    "iti": "attention", "attn_satisfies": "attention",
    "sep": "generation", "step": "generation",
}
FAMILY_MARKER = {
    "hidden": "o",
    "attention": "s",
    "generation": "^",
}


def collect_events(model):
    with open(f"{BASE}/{model}/probe_clustering.json") as f:
        clust = json.load(f)
    with open(f"{BASE}/{model}/probe_ladder.json") as f:
        ladder = json.load(f)

    g = clust["global_average"]
    methods_in_mat = set(g["methods"])
    M = np.array(g["avg_spearman_matrix"])
    idx = {m: i for i, m in enumerate(g["methods"])}

    events = []
    for ds in DATASETS:
        info = ladder[ds]
        for entry in info["ladder"]:
            gain = entry["incremental_gain"]
            if gain is None:
                continue
            added = entry["added"]
            if added not in methods_in_mat:
                continue
            prior = [m for m in entry["methods"]
                     if m != added and m in methods_in_mat]
            if not prior:
                continue
            rs = [M[idx[added], idx[m]] for m in prior]
            events.append({
                "dataset": ds,
                "added": added,
                "family": FAMILY_OF[added],
                "x": float(np.mean(rs)),
                "y": float(gain) * 100.0,
                "k": entry["n_methods"],
            })
    return events


# -------------------- figure --------------------
fig = plt.figure(figsize=(6.8, 2.7))
gs = GridSpec(
    1, 3, wspace=0.10,
    left=0.085, right=0.985, top=0.88, bottom=0.36,
)
axes = [fig.add_subplot(gs[0, k]) for k in range(3)]
axes[1].sharey(axes[0])
axes[2].sharey(axes[0])

per_model_events = {}
for ax, (model_key, model_label) in zip(axes, MODELS):
    events = collect_events(model_key)
    per_model_events[model_key] = events

    xs = np.array([e["x"] for e in events])
    ys = np.array([e["y"] for e in events])

    for e in events:
        ax.scatter(
            e["x"], e["y"],
            color=DATASET_COLORS[e["dataset"]],
            marker=FAMILY_MARKER[e["family"]],
            s=30,
            edgecolors="white",
            linewidths=0.7,
            alpha=0.9,
            zorder=3,
        )

    # Linear trendline (gray dashed)
    if len(xs) > 1:
        m, b = np.polyfit(xs, ys, 1)
        xline = np.linspace(xs.min(), xs.max(), 50)
        ax.plot(xline, m * xline + b,
                color="#8C8C8C", linestyle="--", linewidth=1.0,
                alpha=0.9, zorder=2)

    rho, p = spearmanr(xs, ys)
    ax.text(
        0.04, 0.95,
        rf"$\rho={rho:.2f},\ p={p:.3f}$",
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=8,
        bbox=dict(facecolor="white", edgecolor="none",
                  alpha=0.75, pad=1.2),
    )

    ax.axhline(0, color="#BBBBBB", linewidth=0.7, linestyle=":", zorder=1)
    ax.set_title(model_label, fontsize=10, pad=4)
    ax.tick_params(axis="both", labelsize=8)
    ax.grid(True, axis="both", linestyle="--", linewidth=0.4,
            color="#CCCCCC", alpha=0.5)
    ax.set_axisbelow(True)

axes[0].set_ylabel("Marginal gain (pp AUROC)", fontsize=9)
for ax in axes[1:]:
    plt.setp(ax.get_yticklabels(), visible=False)
fig.supxlabel(
    r"Redundancy (mean Spearman $\rho$ with prior ensemble)",
    fontsize=9, y=0.17,
)

# Legend: dataset colors + family markers
dataset_handles = [
    Line2D([0], [0], marker="o", color="w",
           markerfacecolor=DATASET_COLORS[d], markeredgecolor="white",
           markeredgewidth=0.7, markersize=6, label=DATASET_LABELS[d])
    for d in DATASETS
]
family_handles = [
    Line2D([0], [0], marker=FAMILY_MARKER[f], color="#444444",
           linestyle="None", markersize=6, label=f.capitalize())
    for f in ["hidden", "attention", "generation"]
]

leg1 = fig.legend(
    handles=dataset_handles,
    loc="lower center",
    ncol=5,
    bbox_to_anchor=(0.5, -0.03),
    frameon=False,
    fontsize=7.5,
    handletextpad=0.3,
    columnspacing=1.2,
)
fig.add_artist(leg1)
fig.legend(
    handles=family_handles,
    loc="lower center",
    ncol=3,
    bbox_to_anchor=(0.5, -0.12),
    frameon=False,
    fontsize=7.5,
    handletextpad=0.3,
    columnspacing=1.6,
)

fig.savefig(f"{OUT}.pdf")
fig.savefig(f"{OUT}.png")

# -------------------- stdout numeric summary --------------------
print("\n== Redundancy vs. marginal-gain Spearman correlations ==")
for model_key, _ in MODELS:
    events = per_model_events[model_key]
    xs = [e["x"] for e in events]
    ys = [e["y"] for e in events]
    rho, p = spearmanr(xs, ys)
    print(f"  {model_key:18s}  n={len(events):2d}  rho={rho:+.4f}  p={p:.4g}")

pooled = [e for ev in per_model_events.values() for e in ev]
xs = [e["x"] for e in pooled]
ys = [e["y"] for e in pooled]
rho, p = spearmanr(xs, ys)
print(f"  {'POOLED':18s}  n={len(pooled):2d}  rho={rho:+.4f}  p={p:.4g}")
print(f"\nSaved {OUT}.pdf and .png")
