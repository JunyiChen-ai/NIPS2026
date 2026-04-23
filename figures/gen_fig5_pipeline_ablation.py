"""Pipeline ablation: average delta across 5 datasets, sorted."""
import json
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import matplotlib.pyplot as plt
from paper_plot_style import setup_style

setup_style()

DATA = "/home/junyi/NIPS2026/fusion/results/qwen2.5-7b/pipeline_ablation.json"
OUT = "/home/junyi/NIPS2026/figures/fig5_pipeline_ablation"

with open(DATA) as f:
    data = json.load(f)

LABELS = {
    "full":              "Full (Ours)",
    "pca128_only":       "PCA(128) only",
    "no_enrichment":     "No Enrichment",
    "seed3":             "3 Seeds",
    "seed1_only":        "1 Seed",
    "meta_gbt_only":     "Meta-GBT only",
    "tree_experts_only": "Tree Experts only",
    "meta_l2_only":      "Meta-L2 only",
    "meta_l1_only":      "Meta-L1 only",
    "gbt_expert_only":   "GBT Expert only",
    "pca32_only":        "PCA(32) only",
    "lr_expert_only":    "LR Expert only",
    "rf_expert_only":    "RF Expert only",
    "et_expert_only":    "ET Expert only",
}

configs = list(LABELS.keys())
avg_deltas = {}
for cfg in configs:
    deltas = []
    for ds, cfgs in data.items():
        if cfg in cfgs and cfgs[cfg]["delta"] is not None:
            deltas.append(cfgs[cfg]["delta"])
    avg_deltas[cfg] = np.mean(deltas) * 100

sorted_cfgs = sorted(configs, key=lambda c: -avg_deltas[c])
names = [LABELS[c] for c in sorted_cfgs]
vals = [avg_deltas[c] for c in sorted_cfgs]

highlight = '#E07A3E'
neutral = '#6B90C6'
colors_bar = [highlight if c == "full" else neutral for c in sorted_cfgs]

fig, ax = plt.subplots(figsize=(5.2, 3.4))
y = np.arange(len(names))[::-1]
ax.barh(y, vals, color=colors_bar, edgecolor='white', linewidth=0.7, height=0.72)

xmax = max(vals) * 1.22
for i, v in enumerate(vals):
    ax.text(v + xmax * 0.012, y[i], f"+{v:.2f}",
            va='center', ha='left', fontsize=8,
            color='#1a1a1a')

ax.set_yticks(y)
ax.set_yticklabels(names, fontsize=8.5)
ax.set_xlabel(r"Avg. $\Delta$ AUROC vs. best single (%)", fontsize=9)
ax.tick_params(axis='x', labelsize=8)
ax.set_xlim(0, xmax)

full_val = avg_deltas["full"]
ax.axvline(x=full_val, color=highlight, linestyle=':',
           linewidth=1.0, alpha=0.7)

ax.grid(True, axis='x', linestyle='--', linewidth=0.5,
        color='#BBBBBB', alpha=0.5)
ax.set_axisbelow(True)

plt.tight_layout(pad=0.2)
plt.savefig(f"{OUT}.pdf")
plt.savefig(f"{OUT}.png")
print(f"Saved {OUT}.pdf and .png")
