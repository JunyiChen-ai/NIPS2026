"""
Cross-model aggregation: walk fusion/results/{model}/ for each model and emit
a single JSON with normalized metrics, plus markdown tables that answer RQ1/2/3.

Usage:
    python fusion/aggregate_cross_model.py
"""

import os, json, argparse
from collections import OrderedDict

BASE_RESULTS = "/home/junyi/NIPS2026/fusion/results"
DEFAULT_MODELS = ["qwen2.5-7b", "llama3.1-8b", "mistral-7b-v0.3"]
DATASETS = ["common_claim_3class", "e2h_amc_3class", "e2h_amc_5class",
            "when2call_3class", "ragtruth_binary", "fava_binary"]

EXP_FILES = {
    "oracle_baseline":     "oracle_complete.json",
    "oracle_with_raw":     "oracle_with_raw.json",
    "v21_fusion":          "baseline_only_v21_winning_results.json",
    "ladder":              "probe_ladder.json",
    "leave_one_out":       "leave_one_method_out.json",
    "pipeline_ablation":   "pipeline_ablation.json",
    "probe_clustering":    "probe_clustering.json",
    "fava_extension":      "fava_extension.json",
}


def load_if_exists(path):
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        print(f"  [WARN] failed to load {path}: {e}")
        return None


def normalize_one_model(model):
    """Read all 8 JSONs for one model. Return a dict keyed by experiment → dataset → metrics."""
    model_dir = os.path.join(BASE_RESULTS, model)
    out = {}

    # oracle (baseline only)
    d = load_if_exists(os.path.join(model_dir, EXP_FILES["oracle_baseline"]))
    if d:
        out["oracle_baseline"] = {
            ds: {
                "oracle_auroc":      r.get("oracle_auroc"),
                "best_single_auroc": r.get("best_single_auroc"),
                "best_single_method":r.get("best_single_method"),
                "headroom":          r.get("headroom"),
                "per_probe_auroc":   r.get("per_probe_auroc"),
            }
            for ds, r in d.items()
        }

    # oracle (with raw) — from exp1b
    d = load_if_exists(os.path.join(model_dir, EXP_FILES["oracle_with_raw"]))
    if d:
        out["oracle_with_raw"] = {
            ds: {
                "oracle_auroc_baseline_only": r.get("oracle_auroc_baseline_only"),
                "oracle_auroc_with_raw":      r.get("oracle_auroc_with_raw"),
                "raw_headroom_delta":         r.get("raw_headroom_delta"),
                "raw_view_win_rate":          r.get("raw_view_win_rate"),
                "per_raw_view_auroc":         r.get("per_raw_view_auroc"),
                "best_single_method":         r.get("best_single_method"),
            }
            for ds, r in d.items()
        }

    # v21 fusion
    d = load_if_exists(os.path.join(model_dir, EXP_FILES["v21_fusion"]))
    if d:
        out["v21_fusion"] = {
            ds: {
                "test_auroc":     r.get("test_auroc"),
                "baseline_auroc": r.get("baseline_auroc"),
                "delta":          r.get("delta"),
                "delta_pct":      r.get("delta_pct"),
                "ci_95":          r.get("ci_95"),
            }
            for ds, r in d.items() if isinstance(r, dict)
        }

    # fava extension (adds to v21_fusion)
    d = load_if_exists(os.path.join(model_dir, EXP_FILES["fava_extension"]))
    if d:
        out.setdefault("v21_fusion", {})
        for ds, r in d.items():
            if isinstance(r, dict):
                out["v21_fusion"][ds] = {
                    "test_auroc":     r.get("test_auroc"),
                    "baseline_auroc": r.get("baseline_auroc"),
                    "delta":          r.get("delta"),
                    "delta_pct":      r.get("delta_pct"),
                    "n_methods":      r.get("n_methods"),
                }

    # probe ladder
    d = load_if_exists(os.path.join(model_dir, EXP_FILES["ladder"]))
    if d:
        out["ladder"] = {}
        for ds, r in d.items():
            ladder = r.get("ladder", [])
            final_auroc = ladder[-1]["fusion_auroc"] if ladder else None
            first_above_best = None
            best_single = r.get("best_single")
            if best_single and ladder:
                for step in ladder:
                    if step["fusion_auroc"] > best_single:
                        first_above_best = step["n_methods"]
                        break
            out["ladder"][ds] = {
                "method_ranking": r.get("method_ranking"),
                "best_single":    best_single,
                "final_fusion":   final_auroc,
                "first_k_above_best": first_above_best,
                "n_steps": len(ladder),
                "ladder_curve": [(s["n_methods"], s["fusion_auroc"]) for s in ladder],
            }

    # leave-one-out
    d = load_if_exists(os.path.join(model_dir, EXP_FILES["leave_one_out"]))
    if d:
        out["leave_one_out"] = {}
        for ds, r in d.items():
            # flatten ablation rankings: method → contribution
            contribs = {}
            abl = r.get("ablations", {})
            for m, info in abl.items():
                contribs[m] = info.get("contribution")
            ranking = sorted(contribs.items(), key=lambda x: -(x[1] if x[1] is not None else -999))
            out["leave_one_out"][ds] = {
                "full_fusion_auroc": r.get("full_fusion_auroc"),
                "best_single":       r.get("best_single"),
                "contributions":     contribs,
                "top_contributor":   ranking[0][0] if ranking else None,
                "ranking":           [m for m, _ in ranking],
            }

    # pipeline ablation
    d = load_if_exists(os.path.join(model_dir, EXP_FILES["pipeline_ablation"]))
    if d:
        out["pipeline_ablation"] = {}
        for ds, r in d.items():
            configs = {cfg: info["auroc"] for cfg, info in r.items() if isinstance(info, dict) and "auroc" in info}
            ranked = sorted(configs.items(), key=lambda x: -x[1])
            out["pipeline_ablation"][ds] = {
                "configs": configs,
                "best_config": ranked[0][0] if ranked else None,
                "full_auroc": configs.get("full"),
            }

    # probe clustering
    d = load_if_exists(os.path.join(model_dir, EXP_FILES["probe_clustering"]))
    if d:
        out["probe_clustering"] = {}
        for ds, r in d.items():
            out["probe_clustering"][ds] = {
                "methods":         r.get("methods"),
                "spearman_matrix": r.get("spearman_matrix"),
                "linkage":         r.get("linkage"),
                "n_test":          r.get("n_test"),
            }

    return out


def build_summary(models):
    summary = OrderedDict()
    for model in models:
        model_dir = os.path.join(BASE_RESULTS, model)
        if not os.path.isdir(model_dir):
            print(f"[INFO] no results dir for {model}, skipping")
            continue
        print(f"[INFO] aggregating {model}")
        summary[model] = normalize_one_model(model)
    return summary


def markdown_rq1(summary, models, datasets):
    """Best-single-probe identity and AUROC per (model, dataset)."""
    lines = ["### RQ1 — Best single probe per (model, dataset)", ""]
    header = "| Dataset | " + " | ".join(models) + " |"
    lines.append(header)
    lines.append("|" + "|".join(["---"] * (len(models) + 1)) + "|")
    for ds in datasets:
        row = [ds]
        for m in models:
            oc = summary.get(m, {}).get("oracle_baseline", {}).get(ds)
            if oc:
                row.append(f"{oc.get('best_single_method','?')} ({oc.get('best_single_auroc',0):.4f})")
            else:
                row.append("—")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    return "\n".join(lines)


def markdown_rq2(summary, models, datasets):
    lines = ["### RQ2 — Fusion gains & oracle headroom", ""]
    lines.append("**v21 fusion vs best single (Δ AUROC):**")
    lines.append("")
    lines.append("| Dataset | " + " | ".join(models) + " |")
    lines.append("|" + "|".join(["---"] * (len(models) + 1)) + "|")
    for ds in datasets:
        row = [ds]
        for m in models:
            v = summary.get(m, {}).get("v21_fusion", {}).get(ds)
            if v:
                row.append(f"{v.get('test_auroc',0):.4f} (Δ{v.get('delta_pct','?')})")
            else:
                row.append("—")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    lines.append("**Oracle headroom — baseline-only vs with-raw (AUROC):**")
    lines.append("")
    lines.append("| Dataset | Model | Best single | Oracle (BL) | Oracle (+raw) | Δraw | Rawwin% |")
    lines.append("|---|---|---|---|---|---|---|")
    for ds in datasets:
        for m in models:
            r = summary.get(m, {}).get("oracle_with_raw", {}).get(ds)
            if not r:
                continue
            bl = r.get("oracle_auroc_baseline_only")
            wr = r.get("oracle_auroc_with_raw")
            delta = r.get("raw_headroom_delta")
            win = r.get("raw_view_win_rate")
            best_method = r.get("best_single_method", "?")
            lines.append(f"| {ds} | {m} | {best_method} | "
                         f"{bl:.4f} | {wr:.4f} | +{(delta or 0)*100:.2f}pp | {(win or 0)*100:.0f}% |")
    lines.append("")
    return "\n".join(lines)


def markdown_rq3(summary, models, datasets):
    lines = ["### RQ3 — Method contribution & pipeline ablation", ""]
    lines.append("**Top LOO contributor per (model, dataset):**")
    lines.append("")
    lines.append("| Dataset | " + " | ".join(models) + " |")
    lines.append("|" + "|".join(["---"] * (len(models) + 1)) + "|")
    for ds in datasets:
        row = [ds]
        for m in models:
            loo = summary.get(m, {}).get("leave_one_out", {}).get(ds)
            if loo:
                top = loo.get("top_contributor", "?")
                contrib = loo.get("contributions", {}).get(top, 0)
                row.append(f"{top} (+{contrib*100:.2f}pp)")
            else:
                row.append("—")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    lines.append("**Pipeline ablation — best config per (model, dataset):**")
    lines.append("")
    lines.append("| Dataset | " + " | ".join(models) + " |")
    lines.append("|" + "|".join(["---"] * (len(models) + 1)) + "|")
    for ds in datasets:
        row = [ds]
        for m in models:
            pa = summary.get(m, {}).get("pipeline_ablation", {}).get(ds)
            if pa:
                row.append(f"{pa.get('best_config','?')} ({pa.get('full_auroc',0):.4f})")
            else:
                row.append("—")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--out-json", default=os.path.join(BASE_RESULTS, "cross_model_summary.json"))
    ap.add_argument("--out-md",   default=os.path.join(BASE_RESULTS, "cross_model_summary.md"))
    args = ap.parse_args()

    summary = build_summary(args.models)

    # Coverage report — which (model, experiment) cells are missing
    print("\n=== Coverage ===")
    for m in args.models:
        if m not in summary:
            print(f"  {m}: MISSING (no results dir)")
            continue
        for exp in ["oracle_baseline","oracle_with_raw","v21_fusion","ladder",
                    "leave_one_out","pipeline_ablation","probe_clustering"]:
            have = sum(1 for ds in args.datasets if ds in summary[m].get(exp, {}))
            print(f"  {m}/{exp}: {have}/{len(args.datasets)} datasets")

    with open(args.out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved JSON → {args.out_json}")

    md = [
        "# Cross-Model Aggregation",
        f"models: {args.models}",
        f"datasets: {args.datasets}",
        "",
        markdown_rq1(summary, args.models, args.datasets),
        markdown_rq2(summary, args.models, args.datasets),
        markdown_rq3(summary, args.models, args.datasets),
    ]
    with open(args.out_md, "w") as f:
        f.write("\n".join(md))
    print(f"Saved MD   → {args.out_md}")


if __name__ == "__main__":
    main()
