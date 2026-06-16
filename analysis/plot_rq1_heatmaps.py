"""Plot RQ1 method-fragmentation heatmaps.

Rows are model/dataset pairs. Columns are probing methods. Cell values are
AUROC deltas relative to the row-best method, so 0 means best for that row and
negative values indicate how far a method falls behind the best available
single probe.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "analysis/figures"

METHOD_ORDER = [
    "lr_probe", "mm_probe", "pca_lr", "iti", "attn_satisfies",
    "kb_mlp", "lid", "llm_check", "seakr", "coe", "sep", "step",
]

METHOD_LABELS = {
    "lr_probe": "LR",
    "mm_probe": "MM",
    "pca_lr": "PCA-LR",
    "iti": "ITI",
    "attn_satisfies": "AttnSat",
    "kb_mlp": "KB-MLP",
    "lid": "LID",
    "llm_check": "LLM-Check",
    "seakr": "SeaKR",
    "coe": "CoE",
    "sep": "SEP",
    "step": "STEP",
}

MODEL_SHORT = {
    "qwen2.5-7b": "Qwen",
    "llama3.1-8b": "Llama",
    "mistral-7b-v0.3": "Mistral",
}

SETTING_PATH = {
    "old": ROOT / "fusion/results/cross_model_summary.json",
    "new": ROOT / "fusion/results_correctness/cross_model_summary.json",
}

SETTING_TITLE = {
    "old": "Dataset-label / latent-style setting",
    "new": "Generative answer correctness setting",
}

DATASET_ORDER = {
    "old": [
        "common_claim_3class",
        "e2h_amc_3class",
        "e2h_amc_5class",
        "when2call_3class",
        "ragtruth_binary",
        "fava_binary",
    ],
    "new": [
        "gsm8k",
        "math",
        "mmlu",
        "commonsenseqa",
        "belebele",
        "theoremqa",
        "fava",
        "ragtruth",
        "common_claim_3class",
        "when2call_3class",
    ],
}


def load_rows(setting: str):
    data = json.loads(SETTING_PATH[setting].read_text())
    rows = []
    for model, model_data in data.items():
        oracle = model_data.get("oracle_baseline", {})
        for dataset in DATASET_ORDER[setting]:
            if dataset not in oracle:
                continue
            per_probe = oracle[dataset].get("per_probe_auroc") or {}
            if not per_probe:
                continue
            best = max(float(v) for v in per_probe.values())
            row = {
                "model": model,
                "dataset": dataset,
                "row_label": f"{MODEL_SHORT.get(model, model)} | {dataset}",
                "best_method": oracle[dataset].get("best_single_method"),
                "best_auroc": best,
            }
            for m in METHOD_ORDER:
                row[m] = np.nan if m not in per_probe else float(per_probe[m]) - best
            rows.append(row)
    return rows


def make_plot(setting: str, clip_min: float = -0.20):
    rows = load_rows(setting)
    if not rows:
        raise RuntimeError(f"No rows for setting={setting}")
    df = pd.DataFrame(rows)
    methods = [m for m in METHOD_ORDER if m in df and df[m].notna().any()]
    mat = df[methods].copy()
    mat.index = df["row_label"]

    # Annotation: star for row-best, otherwise signed delta in percentage points.
    ann = mat.copy().astype(object)
    for i, row in enumerate(rows):
        for m in methods:
            v = row.get(m)
            if pd.isna(v):
                ann.iloc[i, methods.index(m)] = ""
            elif m == row["best_method"] or abs(v) < 5e-5:
                ann.iloc[i, methods.index(m)] = "★"
            else:
                ann.iloc[i, methods.index(m)] = f"{100*v:.0f}"

    plot_mat = mat.clip(lower=clip_min, upper=0.0)
    n_rows = len(plot_mat)
    n_cols = len(methods)
    fig_w = max(9.5, 0.72 * n_cols + 3.8)
    fig_h = max(4.8, 0.34 * n_rows + 1.6)

    sns.set_theme(style="white", font_scale=0.85)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    cmap = sns.color_palette("rocket_r", as_cmap=True)
    sns.heatmap(
        plot_mat,
        ax=ax,
        cmap=cmap,
        vmin=clip_min,
        vmax=0,
        linewidths=0.35,
        linecolor="#f0f0f0",
        annot=ann,
        fmt="",
        cbar_kws={"label": "AUROC delta to row-best"},
        mask=plot_mat.isna(),
    )
    ax.set_title(f"RQ1: Method fragmentation — {SETTING_TITLE[setting]}", pad=14)
    ax.set_xlabel("Probing method")
    ax.set_ylabel("Model | Dataset")
    ax.set_xticklabels([METHOD_LABELS.get(m, m) for m in methods], rotation=35, ha="right")
    ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    png = OUT_DIR / f"rq1_delta_heatmap_{setting}.png"
    pdf = OUT_DIR / f"rq1_delta_heatmap_{setting}.pdf"
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def write_winner_counts(setting: str):
    rows = load_rows(setting)
    counts = {}
    for r in rows:
        counts[r["best_method"]] = counts.get(r["best_method"], 0) + 1
    out = {
        "setting": setting,
        "n_rows": len(rows),
        "winner_counts": dict(sorted(counts.items(), key=lambda x: (-x[1], x[0]))),
        "rows": [
            {
                "model": r["model"],
                "dataset": r["dataset"],
                "best_method": r["best_method"],
                "best_auroc": r["best_auroc"],
            }
            for r in rows
        ],
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"rq1_winner_counts_{setting}.json"
    path.write_text(json.dumps(out, indent=2))
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--settings", nargs="+", default=["old", "new"], choices=["old", "new"])
    args = ap.parse_args()
    for setting in args.settings:
        png, pdf = make_plot(setting)
        counts = write_winner_counts(setting)
        print(f"{setting}: wrote {png}")
        print(f"{setting}: wrote {pdf}")
        print(f"{setting}: wrote {counts}")


if __name__ == "__main__":
    main()
