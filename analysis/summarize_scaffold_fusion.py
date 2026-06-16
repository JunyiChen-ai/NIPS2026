"""Summarize Exp7 scaffold-fusion results across models/settings."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median

ROOT = Path(__file__).resolve().parents[1]
MODELS = ["qwen2.5-7b", "llama3.1-8b", "mistral-7b-v0.3"]


def result_root(setting: str) -> Path:
    return ROOT / ("fusion/results_correctness" if setting == "new" else "fusion/results")


def load_json(path: Path):
    with path.open() as f:
        return json.load(f)


def collect(setting: str, result_name: str = "scaffold_fusion"):
    rows = []
    base = result_root(setting)
    for model in MODELS:
        path = base / model / f"{result_name}.json"
        if not path.exists():
            continue
        data = load_json(path)
        for ds, r in data.items():
            if r.get("status") != "done":
                continue
            sel = r["selection"]
            diag = r.get("diagnostic_selection_all_candidates", {})
            rows.append({
                "setting": setting,
                "model": model,
                "dataset": ds,
                "selected": sel["selected_variant"],
                "selected_test": sel["selected_test_auroc"],
                "best_val_single_method": r["best_val_single_method"],
                "best_val_single_test": r["best_val_single_test_auroc"],
                "best_test_single_method": r["best_test_single_method"],
                "best_test_single": r["best_test_single_auroc_oracle"],
                "delta_vs_val_single": r["selected_delta_vs_val_single"],
                "delta_vs_test_oracle_single": r["selected_delta_vs_test_oracle_single"],
                "all_candidate_selected": diag.get("selected_variant"),
                "all_candidate_test": diag.get("selected_test_auroc"),
                "meta_overfit_gap": (
                    None if not diag else diag.get("selected_test_auroc", 0) - sel["selected_test_auroc"]
                ),
            })
    return rows


def stats(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return {"n": 0}
    return {
        "n": len(vals),
        "mean": mean(vals),
        "median": median(vals),
        "min": min(vals),
        "max": max(vals),
        "positive_rate": sum(v > 0 for v in vals) / len(vals),
    }


def write_md(path: Path, payload: dict):
    lines = ["# Scaffold Fusion Summary", ""]
    for setting, s in payload["summary"].items():
        lines.append(f"## {setting}")
        lines.append("")
        lines.append(f"- N datasets: `{s['n']}`")
        lines.append(f"- Mean delta vs val-selected single: `{s['delta_vs_val_single']['mean']:+.4f}`")
        lines.append(f"- Median delta vs val-selected single: `{s['delta_vs_val_single']['median']:+.4f}`")
        lines.append(f"- Win rate vs val-selected single: `{s['delta_vs_val_single']['positive_rate']*100:.1f}%`")
        lines.append(f"- Mean delta vs test-oracle single: `{s['delta_vs_test_oracle_single']['mean']:+.4f}`")
        lines.append(f"- Selected variants: `{s['selected_counts']}`")
        lines.append("")

    lines.append("## Rows")
    lines.append("")
    lines.append("| Setting | Model | Dataset | Selected | Test | Val-single | Δ val-single | Test-oracle single | Δ oracle |")
    lines.append("|---|---|---|---|---:|---:|---:|---:|---:|")
    for r in payload["rows"]:
        lines.append(
            f"| {r['setting']} | {r['model']} | {r['dataset']} | {r['selected']} | "
            f"{r['selected_test']:.4f} | {r['best_val_single_test']:.4f} | "
            f"{r['delta_vs_val_single']:+.4f} | {r['best_test_single']:.4f} | "
            f"{r['delta_vs_test_oracle_single']:+.4f} |"
        )

    lines.append("")
    lines.append("## Failure Cases")
    lines.append("")
    lines.append("| Setting | Model | Dataset | Selected | Δ val-single | Δ oracle |")
    lines.append("|---|---|---|---|---:|---:|")
    failures = [r for r in payload["rows"] if r["delta_vs_val_single"] < 0]
    failures.sort(key=lambda r: r["delta_vs_val_single"])
    for r in failures:
        lines.append(
            f"| {r['setting']} | {r['model']} | {r['dataset']} | {r['selected']} | "
            f"{r['delta_vs_val_single']:+.4f} | {r['delta_vs_test_oracle_single']:+.4f} |"
        )

    path.write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--settings", nargs="+", default=["new"])
    ap.add_argument("--result-name", default="scaffold_fusion")
    args = ap.parse_args()

    all_rows = []
    summary = {}
    for setting in args.settings:
        rows = collect(setting, result_name=args.result_name)
        all_rows.extend(rows)
        summary[setting] = {
            "n": len(rows),
            "delta_vs_val_single": stats([r["delta_vs_val_single"] for r in rows]),
            "delta_vs_test_oracle_single": stats([r["delta_vs_test_oracle_single"] for r in rows]),
            "selected_counts": dict(Counter(r["selected"] for r in rows)),
            "best_val_single_counts": dict(Counter(r["best_val_single_method"] for r in rows)),
            "best_test_single_counts": dict(Counter(r["best_test_single_method"] for r in rows)),
            "by_model": {},
        }
        by_model = defaultdict(list)
        for r in rows:
            by_model[r["model"]].append(r)
        for model, rs in by_model.items():
            summary[setting]["by_model"][model] = {
                "n": len(rs),
                "delta_vs_val_single": stats([r["delta_vs_val_single"] for r in rs]),
                "selected_counts": dict(Counter(r["selected"] for r in rs)),
            }

    payload = {"summary": summary, "rows": all_rows}
    out_dir = ROOT / "analysis/overnight_runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{args.result_name}_summary.json"
    md_path = out_dir / f"{args.result_name}_summary.md"
    json_path.write_text(json.dumps(payload, indent=2))
    write_md(md_path, payload)
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
