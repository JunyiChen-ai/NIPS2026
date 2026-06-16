"""Main-text RQ1 AUROC figure."""

from __future__ import annotations

import json
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "analysis/figures"

KB_SUMMARY = ROOT / "fusion/results/cross_model_summary.json"
OC_SUMMARY = ROOT / "fusion/results_correctness/cross_model_summary.json"

MODELS = ["qwen2.5-7b", "llama3.1-8b", "mistral-7b-v0.3"]
MODEL_LABEL = {
    "qwen2.5-7b": "Qwen2.5-7B",
    "llama3.1-8b": "Llama3.1-8B",
    "mistral-7b-v0.3": "Mistral-7B",
}

METHODS = [
    ("lr_probe", "LR Probe"),
    ("pca_lr", "PCA-LR"),
    ("iti", "ITI"),
    ("kb_mlp", "KB-MLP"),
    ("attn_satisfies", "Attention"),
    ("sep", "SEP"),
    ("step", "STEP"),
]

KB_DATASETS = [
    ("common_claim_3class", "CommonClaim"),
    ("when2call_3class", "When2Call"),
    ("ragtruth_binary", "RAGTruth"),
    ("fava_binary", "FAVA"),
    ("belebele", "Belebele"),
]

OC_DATASETS = [
    ("gsm8k", "GSM8K"),
    ("math", "MATH"),
    ("belebele", "Belebele"),
    ("theoremqa", "TheoremQA"),
    ("ragtruth", "RAGTruth"),
    ("when2call_3class", "When2Call"),
    ("common_claim_3class", "CommonClaim"),
    ("fava", "FAVA"),
]


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def matrix_for(
    model: str, kb: dict, oc: dict
) -> tuple[np.ndarray, list[str], list[str], list[str | None]]:
    cols = [("KB", d, label) for d, label in KB_DATASETS] + [
        ("OC", d, label) for d, label in OC_DATASETS
    ]
    mat = np.full((len(METHODS), len(cols)), np.nan)
    winners: list[str | None] = []

    for col, (setting, dataset, _) in enumerate(cols):
        source = kb if setting == "KB" else oc
        result = source.get(model, {}).get("oracle_baseline", {}).get(dataset, {})
        per_probe = result.get("per_probe_auroc") or {}
        winners.append(result.get("best_single_method"))
        for row, (method, _) in enumerate(METHODS):
            if method in per_probe:
                mat[row, col] = float(per_probe[method])
    return mat, [setting for setting, _, _ in cols], [label for _, _, label in cols], winners


def value_color(value: float) -> str:
    if np.isnan(value):
        return "#777777"
    return "white" if value >= 0.78 else "#202020"


def write_values_csv(kb: dict, oc: dict) -> None:
    out_path = OUT_DIR / "rq1_main_values.csv"
    rows = []
    for model in MODELS:
        mat, settings, datasets, winners = matrix_for(model, kb, oc)
        for col, (setting, dataset, winner) in enumerate(zip(settings, datasets, winners)):
            for row, (method, method_label) in enumerate(METHODS):
                value = mat[row, col]
                rows.append({
                    "model": MODEL_LABEL[model],
                    "setting": "Knowledge Boundary" if setting == "KB" else "Output Correctness",
                    "dataset": dataset,
                    "method": method,
                    "method_label": method_label,
                    "auroc": "" if np.isnan(value) else f"{value:.4f}",
                    "is_winner": str(method == winner).lower(),
                })
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {out_path}")


def main() -> None:
    kb = load(KB_SUMMARY)
    oc = load(OC_SUMMARY)

    fig = plt.figure(figsize=(7.3, 5.55))
    gs = fig.add_gridspec(
        nrows=3,
        ncols=2,
        width_ratios=[1, 0.016],
        left=0.122,
        right=0.965,
        top=0.925,
        bottom=0.175,
        hspace=0.22,
        wspace=0.035,
    )
    axes = [fig.add_subplot(gs[i, 0]) for i in range(3)]
    cax = fig.add_subplot(gs[:, 1])

    cmap = plt.get_cmap("YlGnBu").copy()
    cmap.set_bad("#F1F1F1")
    vmin, vmax = 0.45, 1.00
    im = None

    method_names = [label for _, label in METHODS]
    split = len(KB_DATASETS)
    n_cols = split + len(OC_DATASETS)

    for idx, (ax, model) in enumerate(zip(axes, MODELS)):
        mat, _, dataset_labels, winners = matrix_for(model, kb, oc)
        im = ax.imshow(np.ma.masked_invalid(mat), cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")

        ax.set_yticks(np.arange(len(METHODS)))
        ax.set_yticklabels(method_names, fontsize=7.2)
        ax.set_xticks(np.arange(n_cols))
        ax.set_xlim(-0.5, n_cols - 0.5)
        ax.set_ylim(len(METHODS) - 0.5, -0.5)
        ax.text(-0.48, -0.88, MODEL_LABEL[model], transform=ax.transData,
                ha="left", va="bottom", fontsize=7.6, fontweight="bold", clip_on=False)

        ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(METHODS), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.7)
        ax.tick_params(which="minor", bottom=False, left=False)
        ax.tick_params(axis="both", length=0)
        ax.axvline(split - 0.5, color="#262626", linewidth=1.0)

        if idx == 0:
            trans = ax.get_xaxis_transform()
            ax.text((split - 1) / 2, 1.14, "Knowledge Boundary", transform=trans,
                    ha="center", va="bottom", fontsize=7.4)
            ax.text(split + (len(OC_DATASETS) - 1) / 2, 1.14, "Output Correctness",
                    transform=trans, ha="center", va="bottom", fontsize=7.4)

        for col, winner in enumerate(winners):
            methods = [m for m, _ in METHODS]
            if winner in methods:
                row = methods.index(winner)
                ax.add_patch(Rectangle((col - 0.5, row - 0.5), 1, 1, fill=False,
                                       edgecolor="#111111", linewidth=1.2))

        for row in range(mat.shape[0]):
            for col in range(mat.shape[1]):
                value = mat[row, col]
                text = "-" if np.isnan(value) else f"{value:.2f}"
                weight = "bold" if METHODS[row][0] == winners[col] else "normal"
                ax.text(col, row, text, ha="center", va="center",
                        fontsize=5.9, color=value_color(value), fontweight=weight)

        if idx < len(axes) - 1:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xticklabels(dataset_labels, rotation=38, ha="right", fontsize=7.0)

        for spine in ax.spines.values():
            spine.set_visible(False)

    assert im is not None
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("AUROC", fontsize=7.2)
    cbar.ax.tick_params(labelsize=6.5, length=2)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_values_csv(kb, oc)
    png = OUT_DIR / "rq1_main.png"
    pdf = OUT_DIR / "rq1_main.pdf"
    fig.savefig(png, dpi=450)
    fig.savefig(pdf)
    plt.close(fig)
    print(f"Wrote {png}")
    print(f"Wrote {pdf}")


if __name__ == "__main__":
    main()
