"""Paper-style RQ1 raw AUROC panels.

This figure is intended as the main evidence plot for RQ1. It preserves the
raw single-probe AUROC values and uses a thin outline only to mark the winner
within each model/dataset column.
"""

from __future__ import annotations

import json
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
    "qwen2.5-7b": "(a) Qwen2.5-7B",
    "llama3.1-8b": "(b) Llama3.1-8B",
    "mistral-7b-v0.3": "(c) Mistral-7B",
}

MODEL_NAME = {
    "qwen2.5-7b": "Qwen2.5-7B",
    "llama3.1-8b": "Llama3.1-8B",
    "mistral-7b-v0.3": "Mistral-7B",
}

KB_DATASETS = [
    ("common_claim_3class", "Claim"),
    ("e2h_amc_3class", "E2H-3"),
    ("e2h_amc_5class", "E2H-5"),
    ("when2call_3class", "W2C"),
    ("ragtruth_binary", "RAG"),
    ("fava_binary", "FAVA"),
]

OC_DATASETS = [
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

METHOD_ORDER = [
    "lr_probe",
    "mm_probe",
    "pca_lr",
    "kb_mlp",
    "iti",
    "attn_satisfies",
    "lid",
    "llm_check",
    "seakr",
    "coe",
    "sep",
    "step",
]

METHOD_LABEL = {
    "lr_probe": "LR",
    "mm_probe": "MM",
    "pca_lr": "PCA-LR",
    "kb_mlp": "KB-MLP",
    "iti": "ITI",
    "attn_satisfies": "AttnSat",
    "lid": "LID",
    "llm_check": "LLM-Check",
    "seakr": "SeaKR",
    "coe": "CoE",
    "sep": "SEP",
    "step": "STEP",
}


def load_summary(path: Path) -> dict:
    return json.loads(path.read_text())


def collect_methods(kb: dict, oc: dict) -> list[str]:
    seen = set()
    for source in [kb, oc]:
        for model_data in source.values():
            for row in model_data.get("oracle_baseline", {}).values():
                seen.update((row.get("per_probe_auroc") or {}).keys())
    return [m for m in METHOD_ORDER if m in seen] + sorted(seen - set(METHOD_ORDER))


def build_matrix(model: str, methods: list[str], kb: dict, oc: dict) -> tuple[np.ndarray, list[str], list[str]]:
    datasets = [("KB", k, label) for k, label in KB_DATASETS] + [
        ("OC", k, label) for k, label in OC_DATASETS
    ]
    mat = np.full((len(methods), len(datasets)), np.nan, dtype=float)
    winners = []
    for col, (setting, dataset, _) in enumerate(datasets):
        source = kb if setting == "KB" else oc
        row = source.get(model, {}).get("oracle_baseline", {}).get(dataset, {})
        per_probe = row.get("per_probe_auroc") or {}
        winners.append(row.get("best_single_method"))
        for row_idx, method in enumerate(methods):
            if method in per_probe:
                mat[row_idx, col] = float(per_probe[method])
    return mat, [label for _, _, label in datasets], winners


def text_color(value: float) -> str:
    if np.isnan(value):
        return "#9A9A9A"
    return "white" if value >= 0.78 else "#1F1F1F"


def write_csv(kb: dict, oc: dict, methods: list[str]) -> Path:
    rows = ["model,setting,dataset,method,auroc,is_winner"]
    for model in MODELS:
        for setting, source, datasets in [
            ("Knowledge Boundary", kb, KB_DATASETS),
            ("Output Correctness", oc, OC_DATASETS),
        ]:
            for dataset, dataset_label in datasets:
                row = source.get(model, {}).get("oracle_baseline", {}).get(dataset, {})
                winner = row.get("best_single_method")
                per_probe = row.get("per_probe_auroc") or {}
                for method in methods:
                    value = per_probe.get(method)
                    if value is None:
                        continue
                    rows.append(
                        ",".join(
                            [
                                MODEL_NAME.get(model, model),
                                setting,
                                dataset_label,
                                METHOD_LABEL.get(method, method),
                                f"{float(value):.6f}",
                                str(method == winner),
                            ]
                        )
                    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "rq1_raw_auroc_values.csv"
    out.write_text("\n".join(rows) + "\n")
    return out


def plot() -> tuple[Path, Path, Path]:
    kb = load_summary(KB_SUMMARY)
    oc = load_summary(OC_SUMMARY)
    methods = collect_methods(kb, oc)
    method_labels = [METHOD_LABEL.get(m, m) for m in methods]
    csv = write_csv(kb, oc, methods)

    fig = plt.figure(figsize=(7.15, 6.25))
    gs = fig.add_gridspec(
        nrows=3,
        ncols=2,
        width_ratios=[1.0, 0.018],
        height_ratios=[1.0, 1.0, 1.0],
        left=0.105,
        right=0.955,
        top=0.955,
        bottom=0.105,
        hspace=0.28,
        wspace=0.035,
    )
    axes = [fig.add_subplot(gs[i, 0]) for i in range(3)]
    cax = fig.add_subplot(gs[:, 1])

    cmap = plt.get_cmap("YlGnBu").copy()
    cmap.set_bad("#F3F3F3")
    vmin, vmax = 0.45, 1.00

    last_im = None
    for ax, model in zip(axes, MODELS):
        mat, dataset_labels, winners = build_matrix(model, methods, kb, oc)
        masked = np.ma.masked_invalid(mat)
        last_im = ax.imshow(masked, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")

        n_methods, n_datasets = mat.shape
        ax.set_yticks(np.arange(n_methods))
        ax.set_yticklabels(method_labels, fontsize=7.2)
        ax.set_xticks(np.arange(n_datasets))
        ax.set_xlim(-0.5, n_datasets - 0.5)
        ax.set_ylim(n_methods - 0.5, -0.5)
        ax.set_title(MODEL_LABEL.get(model, model), loc="left", fontsize=7.6, fontweight="bold", pad=2.0)

        ax.set_xticks(np.arange(-0.5, n_datasets, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, n_methods, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.55)
        ax.tick_params(which="minor", bottom=False, left=False)
        ax.tick_params(axis="both", length=0)

        # Separate the two target settings without turning them into separate figures.
        ax.axvline(len(KB_DATASETS) - 0.5, color="#222222", linewidth=0.9)
        for col, winner in enumerate(winners):
            if winner in methods:
                row_idx = methods.index(winner)
                ax.add_patch(
                    Rectangle(
                        (col - 0.5, row_idx - 0.5),
                        1,
                        1,
                        fill=False,
                        edgecolor="#111111",
                        linewidth=1.35,
                    )
                )

        for row_idx in range(n_methods):
            for col in range(n_datasets):
                value = mat[row_idx, col]
                if np.isnan(value):
                    label = "-"
                    color = "#B0B0B0"
                    weight = "normal"
                else:
                    label = f"{value:.2f}"
                    color = text_color(value)
                    weight = "bold" if methods[row_idx] == winners[col] else "normal"
                ax.text(
                    col,
                    row_idx,
                    label,
                    ha="center",
                    va="center",
                    fontsize=4.8,
                    color=color,
                    fontweight=weight,
                )

        for spine in ax.spines.values():
            spine.set_visible(False)

    axes[-1].set_xticklabels(dataset_labels, rotation=35, ha="right", fontsize=6.6)
    for ax in axes[:-1]:
        ax.tick_params(labelbottom=False)

    cbar = fig.colorbar(last_im, cax=cax)
    cbar.set_label("AUROC", fontsize=8)
    cbar.ax.tick_params(labelsize=6.8, length=2)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    png = OUT_DIR / "rq1_raw_auroc_by_model_paper.png"
    pdf = OUT_DIR / "rq1_raw_auroc_by_model_paper.pdf"
    fig.savefig(png, dpi=450)
    fig.savefig(pdf)
    plt.close(fig)
    return png, pdf, csv


def main() -> None:
    png, pdf, csv = plot()
    print(f"Wrote {png}")
    print(f"Wrote {pdf}")
    print(f"Wrote {csv}")


if __name__ == "__main__":
    main()
