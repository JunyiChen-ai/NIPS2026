# Mistral (mistral-7b-v0.3) Runbook

Step-by-step guide to run the NIPS2026 probing-fusion research pipeline for `mistral-7b-v0.3` on a fresh machine. Qwen and Llama results are already produced on the original host; this runbook handles the third model so results can be merged during final aggregation (Part E).

---

## 0. Prerequisites

| Requirement | Notes |
|---|---|
| **Python env with**: `torch`, `scikit-learn >= 1.6`, `scipy`, `numpy`, `matplotlib` | Original host uses `/home/junyi/miniconda3/envs/WWW/bin/python`. Any env with the same package set works. |
| **rclone configured with `b2:` remote** pointing at `junyi-data` bucket | Test with `rclone lsd b2:junyi-data/NIPS2026/` — should list `extraction`, `reproduce`, `baseline`, `datasets`. |
| **Free disk**: ≥ 560 GB during the run (raw extraction ~525 GB + processed ~9 GB + slack). After raw is deleted post-experiment the footprint drops to ~9 GB. |
| **Free RAM**: ≥ 32 GB during `exp1b_oracle_with_raw.py`. Other scripts are light. |
| **Wall time budget**: ~1.5 days end-to-end. Fast experiments (exp1/1b/5/6) finish in <1 h. v21 ≈ 2–3 h. Slow ones (exp2/3/4) ≈ 10–20 h each on CPU. |

---

## 1. Clone the repo and point it at the right paths

```bash
git clone <NIPS2026 repo url> NIPS2026
cd NIPS2026
# Optional: edit BASE_PROCESSED / BASE_EXTRACTION / BASE_RESULTS at the top of each fusion/exp*.py
# if this machine uses a different install prefix. By default they point at
# /home/junyi/NIPS2026/... — change to the new absolute path if needed.
grep -n "_BASE_PROCESSED" fusion/exp1_oracle_complete.py
```

If this machine is not `/home/junyi/NIPS2026`, do a bulk replace:

```bash
NEW_ROOT=/your/path/NIPS2026   # adjust
sed -i "s|/home/junyi/NIPS2026|$NEW_ROOT|g" \
    fusion/exp1_oracle_complete.py \
    fusion/exp1b_oracle_with_raw.py \
    fusion/exp2_probe_ladder.py \
    fusion/exp3_leave_one_out.py \
    fusion/exp4_pipeline_ablation.py \
    fusion/exp5_probe_clustering.py \
    fusion/exp6_fava_extension.py \
    fusion/baseline_only_v21_winning.py \
    fusion/aggregate_cross_model.py \
    fusion/run-experiments.sh
```

---

## 2. Download Mistral data from B2

Both trees come from the same bucket. Download the **raw extraction features** (~525 GB) and the **pre-computed processed features** (~9 GB) in parallel.

```bash
mkdir -p extraction/features/mistral-7b-v0.3
mkdir -p reproduce/processed_features/mistral-7b-v0.3

# Raw extraction (525 GB, network-bound, ~60–90 min on a fast link)
nohup rclone sync \
    b2:junyi-data/NIPS2026/extraction/features/mistral-7b-v0.3 \
    extraction/features/mistral-7b-v0.3 \
    --transfers 16 --checkers 32 --progress --stats 60s \
    > /tmp/rclone_mistral_raw.log 2>&1 &

# Processed features (9 GB, finishes in minutes)
nohup rclone sync \
    b2:junyi-data/NIPS2026/reproduce/processed_features/mistral-7b-v0.3 \
    reproduce/processed_features/mistral-7b-v0.3 \
    --transfers 16 --checkers 32 --progress --stats 30s \
    > /tmp/rclone_mistral_proc.log 2>&1 &
```

**Verify download is complete** before starting experiments:

```bash
# Expected file counts from b2 (verified on original host):
#   raw:        416 files total across 10 datasets × (train/val/test/all) splits
#   processed:  226 files total across 6 datasets × (up to 12 probes × 3 .pt + meta)
find extraction/features/mistral-7b-v0.3 -type f | wc -l            # want 416
find reproduce/processed_features/mistral-7b-v0.3 -type f | wc -l   # want 226
```

If either count is short, resume with the same `rclone sync` command — it is idempotent.

---

## 3. Run the eight experiments

Preferred: use the bundled sequential runner. It runs fast experiments first (so early partial results are available for cross-model aggregation) and long ones last.

```bash
nohup bash fusion/run-experiments.sh mistral-7b-v0.3 \
    > fusion/results/mistral_run.out 2>&1 &
tail -f fusion/results/mistral_run.out
```

Every experiment writes one JSON under `fusion/results/mistral-7b-v0.3/` plus a log under `fusion/results/mistral-7b-v0.3/logs/`. The runner prints one line on start and one on completion of each step.

### Experiment → RQ mapping

| # | Script | Command | Wall time | Output JSON | RQ(s) answered | What it tells you |
|---|---|---|---|---|---|---|
| **1** | `exp1_oracle_complete.py` | `python fusion/exp1_oracle_complete.py --model mistral-7b-v0.3` | ~30 s | `oracle_complete.json` | RQ1, RQ2 | Per-example oracle upper bound using only the 7 (or 11 on binary) baseline probes. Headroom column tells you how much room is left above the best single probe. |
| **1b** | `exp1b_oracle_with_raw.py` | `python fusion/exp1b_oracle_with_raw.py --model mistral-7b-v0.3` | ~15–25 min | `oracle_with_raw.json` | RQ2 (+ raw vs processed) | Same oracle, but also adds 11 raw LLM views (hidden, attention, logit-stats, step-boundary) as candidates. Reports `raw_headroom_delta` and `raw_view_win_rate`. **Must run while raw features are on disk.** |
| **5** | `exp5_probe_clustering.py` | `python fusion/exp5_probe_clustering.py --model mistral-7b-v0.3` | ~2–5 min | `probe_clustering.json` | RQ3 | Pairwise Spearman / Cohen kappa / Jaccard between probe predictions, plus Ward linkage. Used to argue probes partition into families. |
| **6** | `exp6_fava_extension.py` | `python fusion/exp6_fava_extension.py --model mistral-7b-v0.3` | ~15–20 min | `fava_extension.json` | RQ2 | v21 fusion on fava_binary (the 6th, easier dataset). Adds one more row to the main results table. |
| **v21** | `baseline_only_v21_winning.py` | `python fusion/baseline_only_v21_winning.py --model mistral-7b-v0.3` | ~2–3 h | `baseline_only_v21_winning_results.json` | RQ2 | Main fusion method result: 7 probes × {PCA 32, 128} × {LR, GBT, ET, RF} × 5 seeds → meta-blend. Reports `test_auroc`, `baseline_auroc`, `delta_pct` per dataset. |
| **2** | `exp2_probe_ladder.py` | `python fusion/exp2_probe_ladder.py --model mistral-7b-v0.3` | ~8–12 h | `probe_ladder.json` | RQ2 | Ranks methods by standalone AUROC, then progressively adds them to the v21 pipeline (k=1…7). Produces the "saturation" curve: how many probes are enough? |
| **3** | `exp3_leave_one_out.py` | `python fusion/exp3_leave_one_out.py --model mistral-7b-v0.3` | ~12–18 h | `leave_one_method_out.json` | RQ3 | For each method, removes it and re-runs full v21. `contribution = full - without` → which probe is most load-bearing on which task. |
| **4** | `exp4_pipeline_ablation.py` | `python fusion/exp4_pipeline_ablation.py --model mistral-7b-v0.3` | ~12–20 h | `pipeline_ablation.json` | RQ3 | 14 config ablations of the v21 pipeline (PCA resolution, expert types, meta blend, enrichment, seed count). Tests which architectural choices are load-bearing. |

### Restart semantics

`exp2`, `exp3`, `exp4` have checkpoint/resume built in — if the process is killed mid-run, rerunning the same command picks up from the last completed dataset. `exp1`, `exp1b`, `v21`, `exp5`, `exp6` are fast enough to restart from scratch.

### Fixed-order dependency

`exp2_probe_ladder.py`, `exp3_leave_one_out.py`, and `exp4_pipeline_ablation.py` each read the best-single baseline from `fusion/results/{model}/oracle_complete.json` at startup, and `exp3` additionally reads the full v21 fusion AUROC from `fusion/results/{model}/baseline_only_v21_winning_results.json`. This is why the runner runs `exp1` and `v21` **before** the three slow experiments — the deltas will be computed against the *correct model-specific baselines*, not stale hardcoded values. If you manually run `exp2/3/4` standalone, confirm those two JSONs exist first.

If you skip that ordering and later discover wrong delta / contribution numbers in `probe_ladder.json` / `leave_one_method_out.json` / `pipeline_ablation.json`, run the post-hoc fixer:

```bash
python fusion/patch_llama_deltas.py --model mistral-7b-v0.3
```

The script only rewrites derived fields (delta, delta_pct, contribution, full_fusion_auroc, best_single) and leaves the underlying AUROCs unchanged, so it's safe to run repeatedly.

### Ordering constraint

**`exp1b` must run before the raw extraction features are deleted.** The runner already enforces this by putting `exp1b` second. If you choose to run individual scripts manually, keep raw features on disk until you have `fusion/results/mistral-7b-v0.3/oracle_with_raw.json`.

---

## 4. Verify completion and clean up raw data

After the runner reports "Done", confirm all 8 JSONs were produced:

```bash
ls -la fusion/results/mistral-7b-v0.3/*.json
# Expected 8 files:
#   baseline_only_v21_winning_results.json
#   fava_extension.json
#   leave_one_method_out.json
#   oracle_complete.json
#   oracle_with_raw.json
#   pipeline_ablation.json
#   probe_clustering.json
#   probe_ladder.json
```

All 8 present → raw extraction features can be deleted to reclaim ~525 GB:

```bash
rm -rf extraction/features/mistral-7b-v0.3
```

**Keep `reproduce/processed_features/mistral-7b-v0.3/` (~9 GB)** — cheap insurance in case something needs to be rerun later. No other experiment (including the cross-model aggregator) needs raw features.

---

## 5. Send mistral JSONs back to the original host

Two options:

**A. Upload to B2 under a results subfolder:**

```bash
rclone copy fusion/results/mistral-7b-v0.3 b2:junyi-data/NIPS2026/fusion_results/mistral-7b-v0.3
```

On the original host, pull them into place:

```bash
rclone copy b2:junyi-data/NIPS2026/fusion_results/mistral-7b-v0.3 \
    /home/junyi/NIPS2026/fusion/results/mistral-7b-v0.3
```

**B. rsync / scp** the `fusion/results/mistral-7b-v0.3/` directory to the original host directly.

---

## 6. Optional — run the cross-model aggregator here

If you prefer to finalize from this machine, first sync Qwen and Llama JSONs down from the original host (scp/rsync), then:

```bash
python fusion/aggregate_cross_model.py \
    --models qwen2.5-7b llama3.1-8b mistral-7b-v0.3
```

Outputs:

- `fusion/results/cross_model_summary.json` — nested `{model → experiment → dataset → metrics}`
- `fusion/results/cross_model_summary.md` — markdown tables:
  - **RQ1**: best-single probe identity × model (shows when/how probe choice shifts across models)
  - **RQ2**: two separate tables —
    - *Realizable gain*: v21 fusion vs best-single-baseline (deployable headroom)
    - *Oracle headroom*: baseline-only vs with-raw oracle (theoretical upper bound, split into what baseline probes already exhausted vs what raw adds)
  - **RQ3**: LOO top contributor × model, pipeline ablation best config × model

The aggregator is tolerant of missing cells — it prints a coverage report and warns if any (model, experiment) is absent but still emits everything it can.

---

## 7. Rough time budget on a fresh machine

| Phase | Wall time |
|---|---|
| rclone raw+processed | 1–2 h (network-bound) |
| exp1 | 30 s |
| exp1b (raw-feature oracle) | 15–25 min |
| exp5 (probe clustering) | 2–5 min |
| exp6 (fava fusion) | 15–20 min |
| v21 | 2–3 h |
| exp2 probe ladder | 8–12 h |
| exp3 leave-one-out | 12–18 h |
| exp4 pipeline ablation | 12–20 h |
| cleanup + upload | 5 min |
| **Total** | **~35–55 h** |

Parallelizing exp2/3/4 is safe if you have spare CPU — they are independent. On a 24-core machine like the original host they saturate ~12 cores each, so 2× in parallel is fine; 3× would oversubscribe.

---

## Troubleshooting

- **OOM in exp1b on the two big raw views** (`input_attn_value_norms`, `gen_per_token_hidden_last_layer`): the loader pools per-head first and falls back to `IncrementalPCA`. If it still OOMs, reduce `PCA(256)` in `fit_probe_on_features` to `PCA(128)`.
- **Missing `meta.json` for a split**: the raw extraction tree must contain `meta.json` per split (it stores labels). Redownload just that split:
  ```bash
  rclone sync b2:junyi-data/NIPS2026/extraction/features/mistral-7b-v0.3/<dataset>/<split> \
      extraction/features/mistral-7b-v0.3/<dataset>/<split>
  ```
- **exp2/3/4 appear stuck**: these print one line per drop/step. Tail the per-experiment log:
  ```bash
  tail -f fusion/results/mistral-7b-v0.3/logs/exp3.log
  ```
- **Probe counts differ**: binary datasets (ragtruth, fava) have up to 12 probes (7 multiclass + 4 binary-only); multiclass (common_claim, e2h_3c, e2h_5c, when2call) have 7. This is expected.
