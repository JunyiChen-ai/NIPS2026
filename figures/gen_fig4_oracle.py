"""Figure 4: Best Single → v21 Fusion → Oracle on each dataset (grouped bars)."""
import json
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import matplotlib.pyplot as plt
from paper_plot_style import setup_style, COLORS, DATASET_LABELS

setup_style()

ORACLE_FILE = "/home/junyi/NIPS2026/fusion/results/oracle_complete.json"
V21_FILE = "/home/junyi/NIPS2026/fusion/results/baseline_only_v21_winning_results.json"
FAVA_FILE = "/home/junyi/NIPS2026/fusion/results/fava_extension.json"
OUT = "/home/junyi/NIPS2026/figures/fig4_oracle"

with open(ORACLE_FILE) as f:
    oracle = json.load(f)
with open(V21_FILE) as f:
    v21 = json.load(f)
with open(FAVA_FILE) as f:
    fava = json.load(f)

DATASETS = ["common_claim_3class", "e2h_amc_3class", "e2h_amc_5class",
            "when2call_3class", "ragtruth_binary", "fava_binary"]

# Collect: best_single (from v3 results), fusion (v21 or fava), oracle
# Use oracle's best_single for consistency
best_single_vals = []
fusion_vals = []
oracle_vals = []
for ds in DATASETS:
    bs = oracle[ds]["best_single_auroc"]
    if ds == "fava_binary":
        fu = fava["fava_binary"]["test_auroc"]
    else:
        fu = v21[ds]["test_auroc"]
    orc = oracle[ds]["oracle_auroc"]
    best_single_vals.append(bs)
    fusion_vals.append(fu)
    oracle_vals.append(orc)

x = np.arange(len(DATASETS))
width = 0.27

fig, ax = plt.subplots(figsize=(8.5, 4.4))

bars1 = ax.bar(x - width, best_single_vals, width, label="Best Single Probe",
               color=COLORS[0], edgecolor='white', linewidth=0.6)
bars2 = ax.bar(x, fusion_vals, width, label="Our Fusion (v21)",
               color=COLORS[1], edgecolor='white', linewidth=0.6)
bars3 = ax.bar(x + width, oracle_vals, width, label="Per-Example Oracle",
               color=COLORS[2], edgecolor='white', linewidth=0.6)

# Annotate fusion delta
for i, (bs, fu, orc) in enumerate(zip(best_single_vals, fusion_vals, oracle_vals)):
    delta_pct = (fu - bs) * 100
    ax.text(i, fu + 0.01, f"+{delta_pct:.1f}", ha='center', va='bottom',
            fontsize=9, color=COLORS[1])
    head_pct = (orc - bs) * 100
    ax.text(i + width, orc + 0.01, f"+{head_pct:.1f}", ha='center', va='bottom',
            fontsize=9, color=COLORS[2])

ax.set_xticks(x)
ax.set_xticklabels([DATASET_LABELS[d] for d in DATASETS], rotation=15, ha='right')
ax.set_ylabel("Test AUROC")
ax.set_ylim(0.7, 1.07)
ax.legend(loc='upper left', frameon=False, ncol=3, bbox_to_anchor=(0.0, 1.12))

plt.savefig(f"{OUT}.pdf")
plt.savefig(f"{OUT}.png")
print(f"Saved {OUT}.pdf and .png")
