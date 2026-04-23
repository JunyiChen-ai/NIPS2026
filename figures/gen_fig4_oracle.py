"""Best Single vs Stacking Fusion vs Oracle bar chart, 3 models side by side."""
import json
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from paper_plot_style import setup_style, DATASET_LABELS

setup_style()

BASE = "/home/junyi/NIPS2026/fusion/results"
MODELS = [
    ("qwen2.5-7b", "Qwen-2.5-7B", "qwen2.5-7b"),
    ("llama3.1-8b", "Llama-3.1-8B", "llama3.1-8b"),
    ("mistral-7b-v0.3", "Mistral-7B", "mistral-7b-v0.3"),
]
OUT = "/home/junyi/NIPS2026/figures/fig4_oracle"

DATASETS = ["common_claim_3class", "e2h_amc_3class", "e2h_amc_5class",
            "when2call_3class", "ragtruth_binary"]

BAR_COLORS = {
    'best_single': '#9CB4D9',
    'fusion':      '#E07A3E',
    'oracle':      '#3A8A5A',
}

fig = plt.figure(figsize=(6.8, 2.5))
gs = GridSpec(
    1, 3, wspace=0.10,
    left=0.085, right=0.985, top=0.86, bottom=0.30,
)
axes = [fig.add_subplot(gs[0, k]) for k in range(3)]
axes[1].sharey(axes[0])
axes[2].sharey(axes[0])

for ax_idx, (_, model_label, subdir) in enumerate(MODELS):
    ax = axes[ax_idx]
    oracle_path = os.path.join(BASE, subdir, "oracle_complete.json")
    simple_path = os.path.join(BASE, subdir, "simple_fusion_baselines.json")

    with open(oracle_path) as f:
        oracle = json.load(f)
    with open(simple_path) as f:
        simple = json.load(f)

    best_single_vals, fusion_vals, oracle_vals = [], [], []
    for ds in DATASETS:
        best_single_vals.append(oracle[ds]["best_single_auroc"])
        # Simple LR stack on concatenated per-probe predictions
        fusion_vals.append(simple[ds]["lr_on_preds"])
        oracle_vals.append(oracle[ds]["oracle_auroc"])

    x = np.arange(len(DATASETS))
    width = 0.27

    ax.bar(x - width, best_single_vals, width,
           label="Best single", color=BAR_COLORS['best_single'],
           edgecolor='white', linewidth=0.5)
    ax.bar(x, fusion_vals, width,
           label="Stacking fusion", color=BAR_COLORS['fusion'],
           edgecolor='white', linewidth=0.5)
    ax.bar(x + width, oracle_vals, width,
           label="Oracle", color=BAR_COLORS['oracle'],
           edgecolor='white', linewidth=0.5)

    for i, (bs, orc) in enumerate(zip(best_single_vals, oracle_vals)):
        head_pct = (orc - bs) * 100
        ax.text(i + width, orc + 0.010, f"+{head_pct:.1f}",
                ha='center', va='bottom',
                fontsize=7.5, color=BAR_COLORS['oracle'], fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_LABELS[d] for d in DATASETS],
                       rotation=28, ha='right', fontsize=8)
    ax.set_title(model_label, fontsize=10, pad=4)
    ax.set_ylim(0.68, 1.10)
    ax.tick_params(axis='y', labelsize=8)
    ax.grid(True, axis='y', linestyle='--', linewidth=0.5,
            color='#BBBBBB', alpha=0.5)
    ax.set_axisbelow(True)
    if ax_idx == 0:
        ax.set_ylabel("Test AUROC", fontsize=9)
    else:
        plt.setp(ax.get_yticklabels(), visible=False)

handles = [
    plt.Rectangle((0, 0), 1, 1, color=BAR_COLORS['best_single']),
    plt.Rectangle((0, 0), 1, 1, color=BAR_COLORS['fusion']),
    plt.Rectangle((0, 0), 1, 1, color=BAR_COLORS['oracle']),
]
labels_txt = ["Best single", "Stacking fusion", "Oracle"]
fig.legend(
    handles, labels_txt,
    loc='lower center', ncol=3, frameon=False, fontsize=8.5,
    bbox_to_anchor=(0.5, -0.02), columnspacing=2.0, handlelength=1.4,
)

fig.savefig(f"{OUT}.pdf")
fig.savefig(f"{OUT}.png")
print(f"Saved {OUT}.pdf and .png")
