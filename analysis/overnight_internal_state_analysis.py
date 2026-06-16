"""Overnight analysis for internal-state probing experiments.

This script is intentionally read-only with respect to the primary experiment
artifacts. It consumes existing reproduce/fusion JSON files and writes a
timestamped analysis bundle under analysis/overnight_runs/.

Hypotheses checked:
  H1. Dataset-label / latent-task settings favor input-side representations.
  H2. LLM-response correctness settings favor generation-side representations.
  H3. Fusion headroom persists even when one family dominates best-single wins.
  H4. Post-generation dominance is not uniform across datasets/models.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean, median


ROOT = Path(__file__).resolve().parents[1]
OLD_SUMMARY = ROOT / "fusion/results/cross_model_summary.json"
NEW_SUMMARY = ROOT / "fusion/results_correctness/cross_model_summary.json"
NEW_NATIVE = ROOT / "reproduce/results_correctness/_summary.json"
OUT_ROOT = ROOT / "analysis/overnight_runs"

MODELS = ["qwen2.5-7b", "llama3.1-8b", "mistral-7b-v0.3"]

TIMING = {
    # Input/prompt-side or pre-generation representations.
    "lr_probe": "input_side",
    "mm_probe": "input_side",
    "pca_lr": "input_side",
    "iti": "input_side",
    "attn_satisfies": "input_side",
    # Generation/response-side representations or generated trajectory scores.
    "kb_mlp": "generation_side",
    "lid": "generation_side",
    "llm_check": "generation_side",
    "sep": "generation_side",
    "coe": "generation_side",
    "seakr": "generation_side",
    "step": "generation_side",
}

SIGNAL_FAMILY = {
    "lr_probe": "residual_hidden",
    "mm_probe": "residual_hidden",
    "pca_lr": "residual_hidden",
    "kb_mlp": "residual_hidden",
    "iti": "attention_head",
    "attn_satisfies": "attention_flow",
    "lid": "geometry_uncertainty",
    "llm_check": "mixed_uncertainty",
    "sep": "semantic_uncertainty",
    "coe": "trajectory_geometry",
    "seakr": "sample_consistency",
    "step": "step_trajectory",
}

DATASET_DOMAIN = {
    "common_claim_3class": "factual_claim",
    "e2h_amc_3class": "math_difficulty",
    "e2h_amc_5class": "math_difficulty",
    "when2call_3class": "tool_routing",
    "ragtruth_binary": "rag_hallucination",
    "fava_binary": "factual_hallucination",
    "gsm8k": "math_reasoning",
    "math": "math_reasoning",
    "mmlu": "knowledge_qa",
    "commonsenseqa": "commonsense_qa",
    "belebele": "reading_comprehension",
    "theoremqa": "math_theory_qa",
    "fava": "factual_hallucination",
    "ragtruth": "rag_hallucination",
}


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open() as f:
        return json.load(f)


def safe_float(x):
    if x is None:
        return None
    try:
        v = float(x)
    except Exception:
        return None
    if math.isnan(v):
        return None
    return v


def collect_best_single(summary: dict, setting: str) -> list[dict]:
    rows = []
    for model, model_data in summary.items():
        oracle = model_data.get("oracle_baseline", {})
        for dataset, r in oracle.items():
            method = r.get("best_single_method")
            auroc = safe_float(r.get("best_single_auroc"))
            if not method or auroc is None:
                continue
            per_probe = {k: safe_float(v) for k, v in (r.get("per_probe_auroc") or {}).items()}
            per_probe = {k: v for k, v in per_probe.items() if v is not None}
            best_timing = TIMING.get(method, "unknown")
            rows.append({
                "setting": setting,
                "model": model,
                "dataset": dataset,
                "domain": DATASET_DOMAIN.get(dataset, "unknown"),
                "best_method": method,
                "best_auroc": auroc,
                "best_timing": best_timing,
                "best_family": SIGNAL_FAMILY.get(method, "unknown"),
                "per_probe_auroc": per_probe,
            })
    return rows


def collect_headroom(summary: dict, setting: str) -> list[dict]:
    rows = []
    for model, model_data in summary.items():
        oracle = model_data.get("oracle_baseline", {})
        v21 = model_data.get("v21_fusion", {})
        ladder = model_data.get("ladder", {})
        for dataset, r in oracle.items():
            best = safe_float(r.get("best_single_auroc"))
            oracle_auroc = safe_float(r.get("oracle_auroc"))
            headroom = safe_float(r.get("headroom"))
            fusion = safe_float((v21.get(dataset) or {}).get("test_auroc"))
            fusion_delta = None if fusion is None or best is None else fusion - best
            final_ladder = safe_float((ladder.get(dataset) or {}).get("final_fusion"))
            rows.append({
                "setting": setting,
                "model": model,
                "dataset": dataset,
                "best_single": best,
                "oracle_auroc": oracle_auroc,
                "oracle_headroom": headroom,
                "v21_fusion": fusion,
                "v21_delta": fusion_delta,
                "ladder_final": final_ladder,
            })
    return rows


def gap_by_group(rows: list[dict]) -> list[dict]:
    """For each row, compute best generation-side vs input-side AUROC gap."""
    out = []
    for row in rows:
        grouped = defaultdict(list)
        for method, score in row["per_probe_auroc"].items():
            grouped[TIMING.get(method, "unknown")].append((method, score))
        if not grouped.get("input_side") or not grouped.get("generation_side"):
            continue
        best_input = max(grouped["input_side"], key=lambda x: x[1])
        best_gen = max(grouped["generation_side"], key=lambda x: x[1])
        out.append({
            "setting": row["setting"],
            "model": row["model"],
            "dataset": row["dataset"],
            "domain": row["domain"],
            "best_input_method": best_input[0],
            "best_input_auroc": best_input[1],
            "best_generation_method": best_gen[0],
            "best_generation_auroc": best_gen[1],
            "generation_minus_input": best_gen[1] - best_input[1],
        })
    return out


def summarize_counts(rows: list[dict], key: str) -> dict:
    c = Counter(r[key] for r in rows)
    return dict(sorted(c.items(), key=lambda x: (-x[1], x[0])))


def summarize_numeric(rows: list[dict], key: str) -> dict:
    vals = [safe_float(r.get(key)) for r in rows]
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


def domain_table(rows: list[dict]) -> dict:
    out = {}
    by_domain = defaultdict(list)
    for r in rows:
        by_domain[r["domain"]].append(r)
    for domain, rs in sorted(by_domain.items()):
        out[domain] = {
            "n": len(rs),
            "best_timing_counts": summarize_counts(rs, "best_timing"),
            "best_method_counts": summarize_counts(rs, "best_method"),
        }
    return out


def native_correctness_rows(native: dict) -> list[dict]:
    rows = []
    for key, r in native.items():
        if r.get("status") != "done":
            continue
        model, dataset = key.split("/", 1)
        aurocs = {k: safe_float(v) for k, v in r.get("aurocs", {}).items()}
        aurocs = {k: v for k, v in aurocs.items() if v is not None}
        if not aurocs:
            continue
        best_method, best_score = max(aurocs.items(), key=lambda x: x[1])
        rows.append({
            "setting": "correctness_native",
            "model": model,
            "dataset": dataset,
            "domain": DATASET_DOMAIN.get(dataset, "unknown"),
            "best_method": best_method,
            "best_auroc": best_score,
            "best_timing": TIMING.get(best_method, "unknown"),
            "best_family": SIGNAL_FAMILY.get(best_method, "unknown"),
            "per_probe_auroc": aurocs,
        })
    return rows


def fmt_pct(x: float | None) -> str:
    if x is None:
        return "NA"
    return f"{x * 100:.1f}%"


def fmt_num(x: float | None) -> str:
    if x is None:
        return "NA"
    return f"{x:.4f}"


def write_markdown(path: Path, payload: dict):
    lines = []
    lines.append("# Overnight Internal-State Probing Analysis")
    lines.append("")
    lines.append(f"Generated: `{payload['timestamp']}`")
    lines.append("")
    lines.append("## Hypotheses")
    lines.append("")
    for h in payload["hypotheses"]:
        lines.append(f"- **{h['id']}**: {h['text']}")
    lines.append("")

    lines.append("## Best-Single Winner Counts")
    lines.append("")
    lines.append("| Setting | N | Timing counts | Method counts | Family counts |")
    lines.append("|---|---:|---|---|---|")
    for setting, s in payload["best_single_summary"].items():
        lines.append(
            f"| {setting} | {s['n']} | `{s['timing_counts']}` | "
            f"`{s['method_counts']}` | `{s['family_counts']}` |"
        )
    lines.append("")

    lines.append("## Generation vs Input Gap")
    lines.append("")
    lines.append("| Setting | N | Mean gen-input | Median | Positive rate | Min | Max |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for setting, s in payload["gap_summary"].items():
        lines.append(
            f"| {setting} | {s.get('n', 0)} | {fmt_num(s.get('mean'))} | "
            f"{fmt_num(s.get('median'))} | {fmt_pct(s.get('positive_rate'))} | "
            f"{fmt_num(s.get('min'))} | {fmt_num(s.get('max'))} |"
        )
    lines.append("")

    lines.append("## Fusion / Oracle Headroom")
    lines.append("")
    lines.append("| Setting | N | Mean oracle headroom | Median oracle headroom | Positive rate | Mean v21 delta |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for setting, s in payload["headroom_summary"].items():
        lines.append(
            f"| {setting} | {s['oracle_headroom'].get('n', 0)} | "
            f"{fmt_num(s['oracle_headroom'].get('mean'))} | "
            f"{fmt_num(s['oracle_headroom'].get('median'))} | "
            f"{fmt_pct(s['oracle_headroom'].get('positive_rate'))} | "
            f"{fmt_num(s['v21_delta'].get('mean'))} |"
        )
    lines.append("")

    lines.append("## Domain-Level Winner Pattern")
    lines.append("")
    for setting, table in payload["domain_tables"].items():
        lines.append(f"### {setting}")
        lines.append("")
        lines.append("| Domain | N | Timing counts | Method counts |")
        lines.append("|---|---:|---|---|")
        for domain, r in table.items():
            lines.append(
                f"| {domain} | {r['n']} | `{r['best_timing_counts']}` | "
                f"`{r['best_method_counts']}` |"
            )
        lines.append("")

    lines.append("## Top Exceptions")
    lines.append("")
    for setting, rows in payload["top_exceptions"].items():
        lines.append(f"### {setting}")
        lines.append("")
        if not rows:
            lines.append("No rows.")
            lines.append("")
            continue
        lines.append("| Model | Dataset | Domain | Best input | Best generation | Gen-input |")
        lines.append("|---|---|---|---|---|---:|")
        for r in rows:
            lines.append(
                f"| {r['model']} | {r['dataset']} | {r['domain']} | "
                f"{r['best_input_method']} ({r['best_input_auroc']:.4f}) | "
                f"{r['best_generation_method']} ({r['best_generation_auroc']:.4f}) | "
                f"{r['generation_minus_input']:+.4f} |"
            )
        lines.append("")

    path.write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = args.tag or timestamp
    out_dir = OUT_ROOT / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    old_summary = load_json(OLD_SUMMARY)
    new_summary = load_json(NEW_SUMMARY)
    native = load_json(NEW_NATIVE)

    old_rows = collect_best_single(old_summary, "dataset_label")
    new_rows = collect_best_single(new_summary, "response_correctness_fusion")
    native_rows = native_correctness_rows(native)

    gap_rows = {
        "dataset_label": gap_by_group(old_rows),
        "response_correctness_fusion": gap_by_group(new_rows),
        "response_correctness_native": gap_by_group(native_rows),
    }

    headroom_rows = {
        "dataset_label": collect_headroom(old_summary, "dataset_label"),
        "response_correctness_fusion": collect_headroom(new_summary, "response_correctness_fusion"),
    }

    best_single_summary = {}
    for setting, rows in [
        ("dataset_label", old_rows),
        ("response_correctness_fusion", new_rows),
        ("response_correctness_native", native_rows),
    ]:
        best_single_summary[setting] = {
            "n": len(rows),
            "timing_counts": summarize_counts(rows, "best_timing"),
            "method_counts": summarize_counts(rows, "best_method"),
            "family_counts": summarize_counts(rows, "best_family"),
        }

    gap_summary = {k: summarize_numeric(v, "generation_minus_input") for k, v in gap_rows.items()}

    headroom_summary = {}
    for setting, rows in headroom_rows.items():
        headroom_summary[setting] = {
            "oracle_headroom": summarize_numeric(rows, "oracle_headroom"),
            "v21_delta": summarize_numeric(rows, "v21_delta"),
        }

    # Exceptions: for dataset-label, generation beats input most strongly; for
    # correctness, input beats generation most strongly.
    top_exceptions = {}
    old_gap_sorted = sorted(gap_rows["dataset_label"], key=lambda r: -r["generation_minus_input"])
    top_exceptions["dataset_label_generation_wins"] = old_gap_sorted[:10]
    new_gap_sorted = sorted(gap_rows["response_correctness_fusion"], key=lambda r: r["generation_minus_input"])
    top_exceptions["correctness_input_wins_fusion"] = new_gap_sorted[:10]
    native_gap_sorted = sorted(gap_rows["response_correctness_native"], key=lambda r: r["generation_minus_input"])
    top_exceptions["correctness_input_wins_native"] = native_gap_sorted[:10]

    payload = {
        "timestamp": timestamp,
        "run_id": run_id,
        "inputs": {
            "old_summary": str(OLD_SUMMARY.relative_to(ROOT)),
            "new_summary": str(NEW_SUMMARY.relative_to(ROOT)),
            "new_native": str(NEW_NATIVE.relative_to(ROOT)),
        },
        "hypotheses": [
            {
                "id": "H1",
                "text": "Dataset-label / latent-task settings should favor input-side probes.",
            },
            {
                "id": "H2",
                "text": "LLM-response correctness settings should favor generation-side probes.",
            },
            {
                "id": "H3",
                "text": "Fusion/oracle headroom should remain positive even under best-single dominance.",
            },
            {
                "id": "H4",
                "text": "Post-generation dominance should have dataset/model exceptions worth cherry-picking.",
            },
        ],
        "best_single_summary": best_single_summary,
        "gap_summary": gap_summary,
        "headroom_summary": headroom_summary,
        "domain_tables": {
            "dataset_label": domain_table(old_rows),
            "response_correctness_fusion": domain_table(new_rows),
            "response_correctness_native": domain_table(native_rows),
        },
        "top_exceptions": top_exceptions,
        "rows": {
            "dataset_label_best_single": old_rows,
            "response_correctness_fusion_best_single": new_rows,
            "response_correctness_native_best_single": native_rows,
            "dataset_label_gap": gap_rows["dataset_label"],
            "response_correctness_fusion_gap": gap_rows["response_correctness_fusion"],
            "response_correctness_native_gap": gap_rows["response_correctness_native"],
            "dataset_label_headroom": headroom_rows["dataset_label"],
            "response_correctness_fusion_headroom": headroom_rows["response_correctness_fusion"],
        },
    }

    json_path = out_dir / "analysis.json"
    md_path = out_dir / "analysis.md"
    json_path.write_text(json.dumps(payload, indent=2))
    write_markdown(md_path, payload)

    latest_json = OUT_ROOT / "latest_analysis.json"
    latest_md = OUT_ROOT / "latest_analysis.md"
    latest_json.write_text(json.dumps(payload, indent=2))
    write_markdown(latest_md, payload)

    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    print(f"Wrote {latest_json}")
    print(f"Wrote {latest_md}")


if __name__ == "__main__":
    main()
