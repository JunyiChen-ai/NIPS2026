"""Hero figure (Figure 1 in the paper): Paradigm diagram.

Top panel: existing paradigm — each probe in isolation, with task-specific
check/cross marks showing fragmentation.
Bottom panel: our paradigm — all probes feed into a unified fusion framework
as plug-in experts, with consistent improvement across tasks and a dashed
"New probe?" box indicating extensibility.
"""
import os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.lines import Line2D

from paper_plot_style import setup_style

setup_style()

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PDF = os.path.join(HERE, "fig0_hero.pdf")
OUT_PNG = os.path.join(HERE, "fig0_hero.png")

# Signal family colours.
C_HIDDEN = '#4C72B0'
C_ATTN   = '#DD8452'
C_GEN    = '#55A868'
C_FRAME  = '#444444'
C_GOOD   = '#1f7a3b'
C_BAD    = '#b5393f'

fig, (ax_top, ax_bot) = plt.subplots(
    2, 1, figsize=(11.0, 6.8), gridspec_kw={'height_ratios': [1.0, 1.0]}
)
for ax in (ax_top, ax_bot):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5.2)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

# ---------------- Top: existing paradigm ----------------
ax_top.set_title(
    "Existing paradigm: each probe in isolation, task-specific wins and losses",
    fontsize=13, loc='left', pad=6
)

# Probe boxes
probes_top = [
    ("LR Probe",  C_HIDDEN, 0.8),
    ("ITI",       C_ATTN,   4.0),
    ("SEP",       C_GEN,    7.2),
    ("AttnSat",   C_ATTN,  10.4),
]
for name, color, x in probes_top:
    box = FancyBboxPatch(
        (x, 3.4), 1.4, 0.8, boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.4, edgecolor=color, facecolor='white'
    )
    ax_top.add_patch(box)
    ax_top.text(x + 0.7, 3.8, name, ha='center', va='center',
                fontsize=11, color=color, weight='bold')

# Task labels on the right per probe (check/cross)
# Each tuple: (probe_x, [(task, hit)])
task_hits = [
    (0.8,  [("factuality", True),  ("hallucination", False), ("routing", False)]),
    (4.0,  [("hallucination", True), ("routing", False), ("difficulty", False)]),
    (7.2,  [("uncertainty", True), ("difficulty", False), ("factuality", False)]),
    (10.4, [("routing", True), ("factuality", False), ("hallucination", False)]),
]
for x_probe, hits in task_hits:
    for i, (task, ok) in enumerate(hits):
        y = 2.9 - 0.55 * i
        sym = "yes" if ok else "no "
        col = C_GOOD if ok else C_BAD
        ax_top.text(x_probe + 0.7, y, f"{sym}  {task}", ha='center', va='center',
                    fontsize=9.5, color=col)

ax_top.text(6.0, 0.35,
            "Different probes, different internal signals, no single winner",
            ha='center', va='center', fontsize=11, style='italic', color=C_FRAME)

# ---------------- Bottom: our paradigm ----------------
ax_bot.set_title(
    "Our paradigm: heterogeneous probes as plug-in experts in a unified fusion framework",
    fontsize=13, loc='left', pad=6
)

# Left column of probes
probes_bot = [
    ("LR Probe",  C_HIDDEN, 0.6, 4.3),
    ("PCA+LR",    C_HIDDEN, 0.6, 3.4),
    ("KB MLP",    C_HIDDEN, 0.6, 2.5),
    ("ITI",       C_ATTN,   0.6, 1.6),
    ("AttnSat",   C_ATTN,   0.6, 0.7),
]
for name, color, x, y in probes_bot:
    box = FancyBboxPatch(
        (x, y), 1.4, 0.6, boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.3, edgecolor=color, facecolor='white'
    )
    ax_bot.add_patch(box)
    ax_bot.text(x + 0.7, y + 0.3, name, ha='center', va='center',
                fontsize=10.5, color=color, weight='bold')

# Extensibility box
ext = FancyBboxPatch(
    (0.6, -0.15), 1.4, 0.55, boxstyle="round,pad=0.02,rounding_size=0.08",
    linewidth=1.3, edgecolor=C_FRAME, facecolor='white', linestyle='--'
)
ax_bot.add_patch(ext)
ax_bot.text(1.3, 0.12, "new probe?", ha='center', va='center',
            fontsize=10, color=C_FRAME, style='italic')

# Fusion block
fus = FancyBboxPatch(
    (4.0, 1.2), 3.8, 2.8, boxstyle="round,pad=0.05,rounding_size=0.12",
    linewidth=1.5, edgecolor=C_FRAME, facecolor='#f4f4f8'
)
ax_bot.add_patch(fus)
ax_bot.text(5.9, 3.5, "Extensible fusion", ha='center', va='center',
            fontsize=12, weight='bold', color=C_FRAME)
ax_bot.text(5.9, 2.9, "per-probe experts", ha='center', va='center',
            fontsize=10.5, color=C_FRAME)
ax_bot.text(5.9, 2.45, "{LR, GBT, ET, RF}", ha='center', va='center',
            fontsize=9.5, color=C_FRAME, family='monospace')
ax_bot.text(5.9, 1.95, "meta-blend", ha='center', va='center',
            fontsize=10.5, color=C_FRAME)
ax_bot.text(5.9, 1.5, "{L2-LR, L1-LR, GBT}", ha='center', va='center',
            fontsize=9.5, color=C_FRAME, family='monospace')

# Arrows from probes to fusion
for name, color, x, y in probes_bot:
    arr = FancyArrowPatch(
        (x + 1.45, y + 0.3), (4.0, 2.6),
        arrowstyle='-|>', mutation_scale=10,
        color=color, linewidth=1.2, alpha=0.85
    )
    ax_bot.add_patch(arr)
arr = FancyArrowPatch(
    (2.05, 0.12), (4.0, 2.5),
    arrowstyle='-|>', mutation_scale=10,
    color=C_FRAME, linewidth=1.1, linestyle='--', alpha=0.85
)
ax_bot.add_patch(arr)

# Right: task outcomes with deltas
tasks_right = [
    ("CommonClaim",  "+2.41%", 4.3),
    ("E2H-AMC",      "+1.61%", 3.4),
    ("When2Call",    "+6.51%", 2.5),
    ("RAGTruth",     "+1.22%", 1.6),
    ("FAVA",         "+0.24%", 0.7),
]
for task, delta, y in tasks_right:
    box = FancyBboxPatch(
        (9.6, y), 1.9, 0.6, boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.3, edgecolor=C_GOOD, facecolor='white'
    )
    ax_bot.add_patch(box)
    ax_bot.text(10.15, y + 0.3, task, ha='center', va='center',
                fontsize=10, color=C_FRAME, weight='bold')
    ax_bot.text(11.2, y + 0.3, delta, ha='center', va='center',
                fontsize=10, color=C_GOOD, weight='bold')
    arr = FancyArrowPatch(
        (7.85, 2.6), (9.55, y + 0.3),
        arrowstyle='-|>', mutation_scale=10,
        color=C_GOOD, linewidth=1.1, alpha=0.7
    )
    ax_bot.add_patch(arr)

ax_bot.text(6.0, -0.5,
            "All probes contribute; fusion improves over the best single probe on every task",
            ha='center', va='center', fontsize=11, style='italic', color=C_FRAME)

# Legend for signal families (top-right of top panel)
legend_handles = [
    Line2D([0], [0], marker='s', color='w', markerfacecolor=C_HIDDEN, markersize=10, label='hidden state'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor=C_ATTN,   markersize=10, label='attention'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor=C_GEN,    markersize=10, label='generation'),
]
ax_top.legend(handles=legend_handles, loc='lower right',
              frameon=False, fontsize=10, ncol=3, bbox_to_anchor=(1.0, -0.05))

plt.tight_layout()
fig.savefig(OUT_PDF)
fig.savefig(OUT_PNG, dpi=200)
print(f"wrote {OUT_PDF}")
print(f"wrote {OUT_PNG}")
