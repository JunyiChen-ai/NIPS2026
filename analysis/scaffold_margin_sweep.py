"""Validation-margin sweep for scaffold deployment policies.

Uses existing exp7 candidate rankings. No model retraining.
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
MODELS = ["qwen2.5-7b", "llama3.1-8b", "mistral-7b-v0.3"]
MARGINS = [0, 0.005, 0.01, 0.015, 0.02, 0.03, 0.05]


def rows_for(setting):
    base = ROOT / ("fusion/results_correctness" if setting == "new" else "fusion/results")
    rows = []
    for model in MODELS:
        path = base / model / "scaffold_fusion_dim64.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        for dataset, r in data.items():
            if r.get("status") != "done":
                continue
            ranking = r["selection"]["variant_ranking"]
            single = [x for x in ranking if x["variant"].startswith("single:")][0]
            best = ranking[0]
            rows.append((model, dataset, single, best, r))
    return rows


def summarize(setting):
    out = []
    for margin in MARGINS:
        deltas = []
        oracle_deltas = []
        choices = {}
        for model, dataset, single, best, r in rows_for(setting):
            chosen = best
            if (
                not best["variant"].startswith("single:")
                and best["val_auroc"] - single["val_auroc"] < margin
            ):
                chosen = single
            deltas.append(chosen["test_auroc"] - single["test_auroc"])
            oracle_deltas.append(chosen["test_auroc"] - r["best_test_single_auroc_oracle"])
            choices[chosen["variant"]] = choices.get(chosen["variant"], 0) + 1
        if not deltas:
            continue
        out.append({
            "setting": setting,
            "margin": margin,
            "n": len(deltas),
            "mean_delta_vs_val_single": mean(deltas),
            "win_rate_vs_val_single": sum(x > 0 for x in deltas) / len(deltas),
            "min_delta_vs_val_single": min(deltas),
            "mean_delta_vs_test_oracle_single": mean(oracle_deltas),
            "choices": dict(sorted(choices.items())),
        })
    return out


def write_md(path, payload):
    lines = ["# Scaffold Margin Sweep", ""]
    lines.append("| Setting | Margin | N | Mean Δ vs val-single | Win rate | Min Δ | Mean Δ vs test-oracle single | Choices |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---|")
    for r in payload:
        lines.append(
            f"| {r['setting']} | {r['margin']:.3f} | {r['n']} | "
            f"{r['mean_delta_vs_val_single']:+.4f} | {r['win_rate_vs_val_single']*100:.1f}% | "
            f"{r['min_delta_vs_val_single']:+.4f} | {r['mean_delta_vs_test_oracle_single']:+.4f} | "
            f"`{r['choices']}` |"
        )
    path.write_text("\n".join(lines))


def main():
    payload = summarize("new") + summarize("old")
    out_dir = ROOT / "analysis/overnight_runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "scaffold_margin_sweep.json").write_text(json.dumps(payload, indent=2))
    write_md(out_dir / "scaffold_margin_sweep.md", payload)
    print(f"Wrote {out_dir / 'scaffold_margin_sweep.json'}")
    print(f"Wrote {out_dir / 'scaffold_margin_sweep.md'}")


if __name__ == "__main__":
    main()
