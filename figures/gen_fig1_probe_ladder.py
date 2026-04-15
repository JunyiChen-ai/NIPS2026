"""Figure 1: Probe ladder — fusion AUROC vs number of probes."""
import json
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))
import matplotlib.pyplot as plt
from paper_plot_style import setup_style, DATASET_COLORS, DATASET_LABELS, MARKERS

setup_style()

DATA = "/home/junyi/NIPS2026/fusion/results/probe_ladder.json"
OUT = "/home/junyi/NIPS2026/figures/fig1_probe_ladder"

with open(DATA) as f:
    data = json.load(f)

# Order datasets for legend
DATASETS = ["common_claim_3class", "e2h_amc_3class", "e2h_amc_5class",
            "when2call_3class", "ragtruth_binary"]

fig, ax = plt.subplots(figsize=(6.0, 4.2))

for i, ds in enumerate(DATASETS):
    ladder = data[ds]["ladder"]
    ks = [step["n_methods"] for step in ladder]
    aurocs = [step["fusion_auroc"] for step in ladder]
    best_single = data[ds]["best_single"]
    color = DATASET_COLORS[ds]

    # Fusion curve
    ax.plot(ks, aurocs, marker=MARKERS[i % len(MARKERS)],
            color=color, label=DATASET_LABELS[ds],
            markersize=7, linewidth=2.0)

    # Best-single horizontal line (dashed)
    ax.axhline(y=best_single, color=color, linestyle=':', linewidth=1.2, alpha=0.7)

ax.set_xlabel("Number of Probing Methods (k)")
ax.set_ylabel("Test AUROC")
ax.set_xticks(range(1, 9))
ax.legend(loc='center right', frameon=False, ncol=1)

plt.savefig(f"{OUT}.pdf")
plt.savefig(f"{OUT}.png")
print(f"Saved {OUT}.pdf and .png")
