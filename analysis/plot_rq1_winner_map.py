"""Paper-style RQ1 winner map.

One figure with three model-level panels. Each panel shows the winning probing
method for every dataset under the two target settings:

  - Knowledge Boundary / dataset-label setting
  - Output Correctness setting

The figure intentionally shows only winners, not the full AUROC matrix, because
the RQ1 presentation should emphasize winner instability rather than distract
with every losing score.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "analysis/figures"

OLD_SUMMARY = ROOT / "fusion/results/cross_model_summary.json"
NEW_SUMMARY = ROOT / "fusion/results_correctness/cross_model_summary.json"

MODELS = ["qwen2.5-7b", "llama3.1-8b", "mistral-7b-v0.3"]
MODEL_LABEL = {
    "qwen2.5-7b": "Qwen2.5-7B",
    "llama3.1-8b": "Llama3.1-8B",
    "mistral-7b-v0.3": "Mistral-7B",
}

OLD_DATASETS = [
    ("common_claim_3class", "Claim"),
    ("e2h_amc_3class", "E2H-3"),
    ("e2h_amc_5class", "E2H-5"),
    ("when2call_3class", "W2C"),
    ("ragtruth_binary", "RAG"),
    ("fava_binary", "FAVA"),
]

NEW_DATASETS = [
    ("gsm8k", "GSM8K"),
    ("math", "MATH"),
    ("mmlu", "MMLU"),
    ("commonsenseqa", "CSQA"),
    ("belebele", "Belebele"),
    ("theoremqa", "TQA"),
    ("fava", "FAVA"),
    ("ragtruth", "RAG"),
    ("common_claim_3class", "Claim"),
    ("when2call_3class", "W2C"),
]

METHOD_ABBR = {
    "lr_probe": "LR",
    "mm_probe": "MM",
    "pca_lr": "PCA",
    "iti": "ITI",
    "attn_satisfies": "Attn",
    "kb_mlp": "KB",
    "lid": "LID",
    "llm_check": "Check",
    "seakr": "SeaKR",
    "coe": "CoE",
    "sep": "SEP",
    "step": "STEP",
}

METHOD_COLOR = {
    "lr_probe": "#4E79A7",
    "mm_probe": "#76B7B2",
    "pca_lr": "#9C755F",
    "iti": "#59A14F",
    "attn_satisfies": "#8CD17D",
    "kb_mlp": "#F28E2B",
    "lid": "#BAB0AC",
    "llm_check": "#79706E",
    "seakr": "#B07AA1",
    "coe": "#D37295",
    "sep": "#E15759",
    "step": "#FF9DA7",
}


def load_winners(path: Path) -> dict:
    data = json.loads(path.read_text())
    out = {}
    for model, model_data in data.items():
        out[model] = {}
        for dataset, row in model_data.get("oracle_baseline", {}).items():
            out[model][dataset] = row.get("best_single_method")
    return out


def draw_cell(ax, x, y, method):
    color = METHOD_COLOR.get(method, "#D9D9D9")
    rect = Rectangle((x, y), 0.92, 0.72, facecolor=color, edgecolor="white", linewidth=1.2)
    ax.add_patch(rect)
    ax.text(
        x + 0.46,
        y + 0.36,
        METHOD_ABBR.get(method, method or ""),
        ha="center",
        va="center",
        fontsize=8.5,
        fontweight="bold",
        color="white" if method not in {"step", "attn_satisfies", "lid"} else "#222222",
    )


def make_figure():
    old = load_winners(OLD_SUMMARY)
    new = load_winners(NEW_SUMMARY)

    gap = 1.0
    old_x = list(range(len(OLD_DATASETS)))
    new_start = len(OLD_DATASETS) + gap
    new_x = [new_start + i for i in range(len(NEW_DATASETS))]
    total_w = len(OLD_DATASETS) + gap + len(NEW_DATASETS)

    fig, axes = plt.subplots(
        nrows=3,
        ncols=1,
        figsize=(13.2, 3.7),
        sharex=True,
        constrained_layout=False,
    )

    for ax, model in zip(axes, MODELS):
        ax.set_xlim(-0.2, total_w + 0.1)
        ax.set_ylim(-0.22, 1.12)
        ax.set_yticks([])
        ax.tick_params(axis="y", length=0)
        ax.tick_params(axis="x", length=0)

        # Background bands for the two blocks.
        ax.axvspan(-0.08, len(OLD_DATASETS) - 0.08, facecolor="#F5F7FA", zorder=0)
        ax.axvspan(new_start - 0.08, new_start + len(NEW_DATASETS) - 0.08, facecolor="#FAF7F2", zorder=0)
        ax.axvline(len(OLD_DATASETS) + gap / 2 - 0.08, color="#BDBDBD", linewidth=0.8)

        for x, (dataset, _) in zip(old_x, OLD_DATASETS):
            method = old.get(model, {}).get(dataset)
            if method:
                draw_cell(ax, x, 0.12, method)
        for x, (dataset, _) in zip(new_x, NEW_DATASETS):
            method = new.get(model, {}).get(dataset)
            if method:
                draw_cell(ax, x, 0.12, method)

        ax.text(
            -0.15,
            0.95,
            MODEL_LABEL.get(model, model),
            ha="left",
            va="center",
            fontsize=11,
            fontweight="bold",
        )
        if ax is axes[0]:
            ax.text(
                len(OLD_DATASETS) / 2 - 0.08,
                0.95,
                "Knowledge Boundary setting",
                ha="center",
                va="center",
                fontsize=9.5,
                color="#4A4A4A",
            )
            ax.text(
                new_start + len(NEW_DATASETS) / 2 - 0.08,
                0.95,
                "Output Correctness setting",
                ha="center",
                va="center",
                fontsize=9.5,
                color="#4A4A4A",
            )

        for spine in ax.spines.values():
            spine.set_visible(False)

    xticks = old_x + new_x
    xticklabels = [label for _, label in OLD_DATASETS] + [label for _, label in NEW_DATASETS]
    axes[-1].set_xticks([x + 0.46 for x in xticks])
    axes[-1].set_xticklabels(xticklabels, rotation=0, ha="center", fontsize=8.5)

    # Legend contains only methods that are winners in these two settings.
    winners = []
    for model in MODELS:
        winners.extend([old.get(model, {}).get(d) for d, _ in OLD_DATASETS])
        winners.extend([new.get(model, {}).get(d) for d, _ in NEW_DATASETS])
    winners = [w for w in METHOD_ABBR if w in set(winners)]
    handles = [
        Rectangle((0, 0), 1, 1, facecolor=METHOD_COLOR[m], edgecolor="none")
        for m in winners
    ]
    labels = [METHOD_ABBR[m] for m in winners]
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(labels),
        frameon=False,
        bbox_to_anchor=(0.5, 0.015),
        fontsize=8.5,
        columnspacing=1.0,
        handlelength=1.0,
    )
    fig.suptitle(
        "RQ1: Winning probing method varies across datasets, even within the same setting",
        y=0.985,
        fontsize=12.5,
        fontweight="bold",
    )
    fig.subplots_adjust(left=0.055, right=0.99, top=0.86, bottom=0.24, hspace=0.20)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    png = OUT_DIR / "rq1_winner_map_by_model_compact.png"
    pdf = OUT_DIR / "rq1_winner_map_by_model_compact.pdf"
    fig.savefig(png, dpi=240, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def main():
    png, pdf = make_figure()
    print(f"Wrote {png}")
    print(f"Wrote {pdf}")


if __name__ == "__main__":
    main()
