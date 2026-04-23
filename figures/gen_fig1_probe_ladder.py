"""Probe ladder: fusion AUROC vs number of probes, 3 models side by side."""
import json
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from paper_plot_style import setup_style, DATASET_COLORS, DATASET_LABELS, MARKERS

setup_style()

BASE = "/home/junyi/NIPS2026/fusion/results"
MODELS = [
    ("Qwen-2.5-7B", "qwen2.5-7b"),
    ("Llama-3.1-8B", "llama3.1-8b"),
    ("Mistral-7B", "mistral-7b-v0.3"),
]
OUT = "/home/junyi/NIPS2026/figures/fig1_probe_ladder"

DATASETS = ["common_claim_3class", "e2h_amc_3class", "e2h_amc_5class",
            "when2call_3class", "ragtruth_binary"]

fig = plt.figure(figsize=(6.8, 2.4))
gs = GridSpec(
    1, 3, wspace=0.10,
    left=0.085, right=0.985, top=0.88, bottom=0.30,
)
axes = [fig.add_subplot(gs[0, k]) for k in range(3)]
axes[1].sharey(axes[0])
axes[2].sharey(axes[0])

handles, labels = [], []
for ax_idx, (model_label, subdir) in enumerate(MODELS):
    ax = axes[ax_idx]
    data_path = os.path.join(BASE, subdir, "probe_ladder.json")
    with open(data_path) as f:
        data = json.load(f)

    for i, ds in enumerate(DATASETS):
        ladder = data[ds]["ladder"]
        ks = [step["n_methods"] for step in ladder]
        aurocs = [step["fusion_auroc"] for step in ladder]
        best_single = data[ds]["best_single"]
        color = DATASET_COLORS[ds]

        line, = ax.plot(
            ks, aurocs,
            marker=MARKERS[i % len(MARKERS)],
            color=color, markersize=4.2,
            linewidth=1.4, markeredgecolor='white', markeredgewidth=0.6,
        )
        ax.axhline(y=best_single, color=color, linestyle=':',
                   linewidth=0.9, alpha=0.5)
        if ax_idx == 0:
            handles.append(line)
            labels.append(DATASET_LABELS[ds])

    ax.set_xlabel("Probes ($k$)", fontsize=9)
    ax.set_title(model_label, fontsize=10, pad=4)
    ax.set_xticks(range(1, 8))
    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.grid(True, axis='y', linestyle='--', linewidth=0.5,
            color='#BBBBBB', alpha=0.5)
    ax.set_axisbelow(True)

    if ax_idx == 0:
        ax.set_ylabel("Test AUROC", fontsize=9)
    else:
        plt.setp(ax.get_yticklabels(), visible=False)

fig.legend(
    handles, labels,
    loc='lower center', ncol=len(DATASETS),
    frameon=False, bbox_to_anchor=(0.5, -0.02),
    fontsize=8.5, columnspacing=1.2, handlelength=1.5,
)

fig.savefig(f"{OUT}.pdf")
fig.savefig(f"{OUT}.png")
print(f"Saved {OUT}.pdf and .png")
