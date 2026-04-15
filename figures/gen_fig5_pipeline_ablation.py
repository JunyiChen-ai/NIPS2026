"""Figure 5: Pipeline ablation — average delta across 5 datasets, sorted."""
import json
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import matplotlib.pyplot as plt
from paper_plot_style import setup_style, COLORS

setup_style()

DATA = "/home/junyi/NIPS2026/fusion/results/pipeline_ablation.json"
OUT = "/home/junyi/NIPS2026/figures/fig5_pipeline_ablation"

with open(DATA) as f:
    data = json.load(f)

# Pretty config labels
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

# Compute avg delta per config
configs = list(LABELS.keys())
avg_deltas = {}
for cfg in configs:
    deltas = []
    for ds, cfgs in data.items():
        if cfg in cfgs and cfgs[cfg]["delta"] is not None:
            deltas.append(cfgs[cfg]["delta"])
    avg_deltas[cfg] = np.mean(deltas) * 100  # to %

# Sort by delta descending
sorted_cfgs = sorted(configs, key=lambda c: -avg_deltas[c])

names = [LABELS[c] for c in sorted_cfgs]
vals = [avg_deltas[c] for c in sorted_cfgs]
colors_bar = [COLORS[3] if c == "full" else COLORS[0] for c in sorted_cfgs]

fig, ax = plt.subplots(figsize=(6.8, 5.2))
y = np.arange(len(names))[::-1]  # top = best
bars = ax.barh(y, vals, color=colors_bar, edgecolor='white', linewidth=0.6, height=0.7)

# Annotate values
for i, v in enumerate(vals):
    ax.text(v + 0.04, y[i], f"+{v:.2f}", va='center', ha='left', fontsize=11)

ax.set_yticks(y)
ax.set_yticklabels(names)
ax.set_xlabel("Avg. Δ AUROC vs Best Single (%)")
ax.set_xlim(0, max(vals) * 1.18)

# Reference line at full
full_val = avg_deltas["full"]
ax.axvline(x=full_val, color=COLORS[3], linestyle=':', linewidth=1.2, alpha=0.6)

plt.savefig(f"{OUT}.pdf")
plt.savefig(f"{OUT}.png")
print(f"Saved {OUT}.pdf and .png")
