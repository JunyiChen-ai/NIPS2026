"""
Batch driver: run save_processed_features_correctness + run_baselines_correctness
for all 18 (model, dataset) combos.

Each combo:
  1. Generate processed features (skip if already done)
  2. Run baselines (skip if results.json exists)

Logs to: reproduce/logs_correctness/{model}_{dataset}.log
Aggregate summary: reproduce/results_correctness/_summary.json
"""

import os, sys, json, time, subprocess
from pathlib import Path

ROOT = Path("/home/junyi/NIPS2026/reproduce")
LOGS_DIR = ROOT / "logs_correctness"
RESULTS_ROOT = ROOT / "results_correctness"
PROC_ROOT = ROOT / "processed_features_correctness"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

PYTHON = "/home/junyi/miniconda3/envs/REAL/bin/python3"

MODELS = ["qwen2.5-7b", "llama3.1-8b", "mistral-7b-v0.3"]
DATASETS = ["gsm8k", "math", "mmlu", "commonsenseqa", "belebele", "theoremqa",
            "fava", "ragtruth", "common_claim_3class", "when2call_3class"]

ALL_METHODS = ["lr_probe", "mm_probe", "pca_lr", "iti", "kb_mlp", "lid",
               "attn_satisfies", "llm_check", "sep", "coe", "seakr", "step"]


def is_proc_done(model, dataset):
    """All 12 methods have output."""
    base = PROC_ROOT / model / dataset
    if not base.exists():
        return False
    for m in ALL_METHODS:
        sub = base / m
        if not sub.exists():
            return False
        if m == "coe":
            if not any(p.name.startswith("train_") for p in sub.iterdir()):
                return False
        else:
            if not (sub / "train.pt").exists():
                return False
    return True


def is_baseline_done(model, dataset):
    p = RESULTS_ROOT / model / f"{dataset}.json"
    return p.exists()


def run_step(cmd, log_path, timeout=14400):
    """Returns (rc, elapsed_s). On timeout, returns (-9, timeout)."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(log_path, "ab") as f:
        f.write(f"\n\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} :: {' '.join(cmd)} ===\n".encode())
        try:
            rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, timeout=timeout).returncode
        except subprocess.TimeoutExpired:
            f.write(f"\n[TIMEOUT after {timeout}s]\n".encode())
            return -9, time.time() - t0
    return rc, time.time() - t0


def check_disk(min_gb=20):
    import shutil
    free = shutil.disk_usage("/home/junyi").free / (1024**3)
    return free >= min_gb, free


def main():
    summary = {}
    t_start = time.time()
    for m in MODELS:
        for d in DATASETS:
            t0 = time.time()
            log_path = LOGS_DIR / f"{m}_{d}.log"
            print(f"\n{'#'*70}\n# {m} / {d}\n{'#'*70}")

            ok, free = check_disk()
            if not ok:
                print(f"  ABORT: free disk {free:.1f} GB < 20 GB threshold")
                summary[f"{m}/{d}"] = {"status": "disk_low", "free_gb": free}
                break

            # Step 1: processed features
            if is_proc_done(m, d):
                print(f"  proc: skipped (already done)")
            else:
                print(f"  proc: running...")
                # 4-hour cap per dataset (mmlu/math have 28x28 ITI = slowest)
                rc, secs = run_step([PYTHON, "-u", str(ROOT / "save_processed_features_correctness.py"),
                              "--model", m, "--dataset", d], log_path, timeout=14400)
                print(f"  proc: rc={rc} in {secs:.0f}s")
                if rc != 0:
                    print(f"  proc: FAILED — see {log_path}")
                    summary[f"{m}/{d}"] = {"status": "proc_failed", "rc": rc, "elapsed_s": secs}
                    continue

            # Step 2: baselines
            if is_baseline_done(m, d):
                print(f"  baselines: skipped (already done)")
            else:
                print(f"  baselines: running...")
                rc, secs = run_step([PYTHON, "-u", str(ROOT / "run_baselines_correctness.py"),
                              "--model", m, "--dataset", d], log_path, timeout=3600)
                print(f"  baselines: rc={rc} in {secs:.0f}s")
                if rc != 0:
                    print(f"  baselines: FAILED — see {log_path}")
                    summary[f"{m}/{d}"] = {"status": "baselines_failed", "rc": rc, "elapsed_s": secs}
                    continue

            # Load results for summary
            with open(RESULTS_ROOT / m / f"{d}.json") as f:
                results = json.load(f)
            aurocs = {}
            for method, res in results.items():
                if isinstance(res, dict):
                    if "auroc" in res:
                        aurocs[method] = res["auroc"]
                    elif method == "coe":
                        aurocs[method] = max((v.get("auroc", 0) for v in res.values()
                                              if isinstance(v, dict)), default=None)
            summary[f"{m}/{d}"] = {"status": "done", "elapsed_s": time.time() - t0,
                                    "aurocs": aurocs}
            print(f"  done in {time.time()-t0:.0f}s")
            print(f"  AUROCs: " + ", ".join(f"{k}={v:.3f}" for k, v in aurocs.items() if v))

    print(f"\n{'#'*70}\n# Total elapsed: {time.time()-t_start:.0f}s\n{'#'*70}")
    with open(RESULTS_ROOT / "_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    # Print summary table
    print("\n=== Summary table (test AUROC) ===")
    print(f"{'model':18s} {'dataset':14s} " + " ".join(f"{m:>6s}" for m in ALL_METHODS))
    for k, v in summary.items():
        if v.get("status") == "done":
            au = v["aurocs"]
            row = " ".join(f"{au.get(m,0):>6.3f}" if au.get(m) else "  N/A " for m in ALL_METHODS)
            print(f"{k:33s} {row}")


if __name__ == "__main__":
    main()
