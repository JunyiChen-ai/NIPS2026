"""
Post-hoc fix for llama exp2/3/4 JSONs.

The original exp2/exp3/exp4 scripts hardcoded Qwen's `best_single` and (in
exp3) Qwen's full v21 fusion AUROC as numerical constants. When run on
Llama these scripts computed the base AUROCs correctly from real Llama
features but then subtracted the wrong (Qwen) baseline when building
`delta`, `delta_pct`, `contribution`, and `full_fusion_auroc` fields.

This script walks fusion/results/{model}/probe_ladder.json,
leave_one_method_out.json and pipeline_ablation.json and rewrites the
derived fields using the correct, model-specific baselines loaded from
oracle_complete.json (best_single) and baseline_only_v21_winning_results.json
(full v21 fusion).

Underlying AUROCs are left untouched — this only recomputes deltas.

Usage:
    python fusion/patch_llama_deltas.py --model llama3.1-8b
"""

import os, json, argparse, shutil

BASE_RESULTS = "/home/junyi/NIPS2026/fusion/results"


def load_baselines(model_dir):
    best_single = {}
    oc_path = os.path.join(model_dir, "oracle_complete.json")
    if os.path.exists(oc_path):
        with open(oc_path) as f:
            oc = json.load(f)
        for ds, r in oc.items():
            if "best_single_auroc" in r:
                best_single[ds] = float(r["best_single_auroc"])

    full_v21 = {}
    v21_path = os.path.join(model_dir, "baseline_only_v21_winning_results.json")
    if os.path.exists(v21_path):
        with open(v21_path) as f:
            v21 = json.load(f)
        for ds, r in v21.items():
            if isinstance(r, dict) and "test_auroc" in r:
                full_v21[ds] = float(r["test_auroc"])

    return best_single, full_v21


def patch_probe_ladder(path, best_single):
    if not os.path.exists(path):
        print(f"  [SKIP] {path} not found")
        return 0
    with open(path) as f:
        data = json.load(f)
    n_changed = 0
    for ds, r in data.items():
        if ds not in best_single:
            continue
        bs = best_single[ds]
        old_bs = r.get("best_single")
        r["best_single"] = bs
        for step in r.get("ladder", []):
            auroc = step.get("fusion_auroc")
            if auroc is None:
                continue
            delta = auroc - bs
            step["delta_vs_best_single"] = round(delta, 4)
            step["delta_pct"] = f"{delta*100:+.2f}%"
            n_changed += 1
        print(f"  probe_ladder.{ds}: best_single {old_bs} → {bs} ({n_changed} ladder steps updated)")
    shutil.copy2(path, path + ".bak")
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return n_changed


def patch_leave_one_out(path, best_single, full_v21):
    if not os.path.exists(path):
        print(f"  [SKIP] {path} not found")
        return 0
    with open(path) as f:
        data = json.load(f)
    n_changed = 0
    for ds, r in data.items():
        new_full = full_v21.get(ds)
        new_bs = best_single.get(ds)
        if new_full is None or new_bs is None:
            continue
        old_full = r.get("full_fusion_auroc")
        r["full_fusion_auroc"] = new_full
        r["best_single"] = new_bs
        # contribution = full - without_method
        for m, info in r.get("ablations", {}).items():
            without = info.get("auroc_without")
            if without is None:
                continue
            contrib = new_full - without
            info["contribution"] = round(contrib, 4)
            info["contribution_pct"] = f"{contrib*100:+.3f}%"
            info["full_auroc"] = new_full
            n_changed += 1
        # re-rank by new contributions
        ranking = sorted(
            r.get("ablations", {}).items(),
            key=lambda x: -(x[1].get("contribution") if x[1].get("contribution") is not None else -999),
        )
        r["contribution_ranking"] = [m for m, _ in ranking]
        print(f"  leave_one_method_out.{ds}: full {old_full} → {new_full}, best_single → {new_bs}")
    shutil.copy2(path, path + ".bak")
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return n_changed


def patch_pipeline_ablation(path, best_single):
    if not os.path.exists(path):
        print(f"  [SKIP] {path} not found")
        return 0
    with open(path) as f:
        data = json.load(f)
    n_changed = 0
    for ds, ds_r in data.items():
        if ds not in best_single:
            continue
        bs = best_single[ds]
        for cfg_name, info in ds_r.items():
            if not isinstance(info, dict):
                continue
            auroc = info.get("auroc")
            if auroc is None:
                continue
            delta = auroc - bs
            info["delta"] = round(delta, 4)
            info["delta_pct"] = f"{delta*100:+.2f}%"
            n_changed += 1
        print(f"  pipeline_ablation.{ds}: best_single → {bs}  ({n_changed} configs updated so far)")
    shutil.copy2(path, path + ".bak")
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return n_changed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    args = ap.parse_args()

    model_dir = os.path.join(BASE_RESULTS, args.model)
    if not os.path.isdir(model_dir):
        print(f"[ERROR] {model_dir} not found")
        return

    best_single, full_v21 = load_baselines(model_dir)
    print(f"=== Patching {args.model} ===")
    print(f"best_single (from oracle_complete): {best_single}")
    print(f"full_v21    (from v21 results):    {full_v21}")
    print()

    print("patching probe_ladder.json...")
    patch_probe_ladder(os.path.join(model_dir, "probe_ladder.json"), best_single)
    print()

    print("patching leave_one_method_out.json...")
    patch_leave_one_out(os.path.join(model_dir, "leave_one_method_out.json"), best_single, full_v21)
    print()

    print("patching pipeline_ablation.json...")
    patch_pipeline_ablation(os.path.join(model_dir, "pipeline_ablation.json"), best_single)
    print()

    print("Done. Original files saved with .bak suffix.")


if __name__ == "__main__":
    main()
