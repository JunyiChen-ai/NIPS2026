# Repository Map

LLM internal-state probing for hallucination / answer-correctness detection
(NeurIPS 2026). This file is the navigation guide: what each file is for, and
whether it is **current method**, **diagnostic evidence**, **infra**, or
**archived**. Keep it in sync when the structure changes.

## Two settings (the main source of apparent complexity)

The repo runs the *same* probe + fusion machinery under two label definitions:

| Setting | Label = | Pipeline | Results |
|---|---|---|---|
| **old** — dataset-label | dataset-provided annotation (hallucination/factuality/difficulty tags) | old `reproduce/` + `processed_features/` | `fusion/results/` |
| **new** — answer-correctness | whether the model's *own generated answer* was correct | `reproduce/*_correctness*` + `processed_features_correctness/` | `fusion/results_correctness/`, `reproduce/results_correctness/` |

The **new (answer-correctness) setting + the `exp7` scaffold are the current
direction.** The old setting + `exp1..exp6` + `v21` remain as the RQ1/RQ2/RQ3
diagnostic evidence behind the paper.

## Top-level layout

```
extraction/        feature extraction from LLMs (upstream stage, runs on the GPU host)
reproduce/         the 12 baseline probes + feature processing (old + new pipelines)
fusion/            fusion experiments (exp1-7) + methods (v21, scaffold) + infra
analysis/          paper figures, tables, and the scaffold-method diagnosis
archive/           dead/superseded code, kept for provenance (see archive/README.md)
research-wiki/     persistent knowledge base (papers / ideas / experiments / claims)

# gitignored, tracked or stored separately:
paper/ slides/ figures/        # paper artifacts (separate tracking)
datasets/ datasets_prepared/   # raw + prepared data
baseline/                      # cloned upstream baseline repos
*/processed_features*/ *.pt    # extracted features (stored on B2)
```

## `extraction/` — feature extraction

Runs on the extraction GPU host (paths hardcode `/data/jehc223/...`); this stage
is **complete** for all 3 models. Two-pass extraction (prompt/generate/replay)
with attention hooks; captures hidden states, attention stats, and logits.

| File | Role | Notes |
|---|---|---|
| `extract_all.py` | **current** entrypoint | Unified CLI; replaced the two old Phase-1/Phase-2 scripts. |
| `extract_features.py` | infra (library) | `FeatureExtractor`, `save_split_features`; imported by `extract_all.py`. (Its own bottom-of-file `main()` is the legacy Phase-1 runner.) |
| `prepare_new_datasets.py` | infra | Builds `datasets_prepared/{name}/`. |
| `setup_e2h_multiclass.py` | infra | Builds the e2h_amc 3/5-class feature dirs (symlink + relabel). |
| `split_features.py` | infra | Slices `features/{dataset}/all/` into train/val/test. |

## `reproduce/` — baseline probes + feature processing

`methods.py` is the core: the 12 internal-state probes (lr_probe, mm_probe,
pca_lr, iti, kb_mlp, lid, attn_satisfies, llm_check [input-side]; sep, coe,
seakr, step [generation-side]). Two parallel pipelines build features from it.

| File | Role | Notes |
|---|---|---|
| `methods.py` | **infra (core)** | The 12 probes; imported by every runner/feature-saver. |
| **OLD pipeline (diagnostic)** | | regenerates old-setting `processed_features/` for exp1-6/v21 |
| `run_all.py` | diagnostic | Old-setting runner (4 original datasets); imported by `run_new_datasets.py`. |
| `run_new_datasets.py` | diagnostic | Old-setting baseline runner; basis for `save_processed_features.py`. |
| `save_processed_features.py` | diagnostic | Writes the old-setting `processed_features/` tree. |
| **NEW pipeline (current)** | | answer-correctness setting feeding exp7 |
| `grade_correctness.py` | **current** | Rule-based per-dataset grader → `correctness_labels/{model}/{ds}/labels.json` (no judge model). |
| `grade_correctness_test.py` | current | Regression test for the grader. |
| `make_correctness_splits.py` | **current** | Stratified 60/20/20 → `split_indices.json`. |
| `save_processed_features_correctness.py` | **current** | Writes `processed_features_correctness/`. |
| `run_baselines_correctness.py` | **current** | Runs the 12 probes on correctness features. |
| `run_all_correctness.py` | **current** | Batch driver over all (model, dataset) cells. |
| `save_processed_features_haloscope.py` | **current** | HaloScope plug-in probe (newest; Qwen done, Llama/Mistral pending). |
| `save_processed_features_kb_belebele.py` | current | KB-probe features for belebele. |

## `fusion/` — fusion experiments + methods

| File | Role | Notes |
|---|---|---|
| `__init__.py`, `settings.py`, `_gpu_clf.py` | **infra** | Package marker; old/new config registry (`get_config`); GPU classifier helpers. Imported by ~every script here. |
| `exp7_scaffold_fusion.py` | **current method** | Plug-and-play scaffold: per-method `StandardScaler→PCA(64)→LR` adapter + reliability-weighted family/timing-aware composition + val-only selection. The current paper-facing method. |
| `baseline_only_v21_winning.py` | method / diagnostic | v21 Multi-View Expert-Library Stacking; the RQ2 "realizable gain". Still read by `exp3` + `aggregate_cross_model.py`. |
| `exp1_oracle_complete.py` | diagnostic | RQ1/RQ2 per-example oracle; `oracle_complete.json` is read by exp2/3/4. |
| `exp1b_oracle_with_raw.py` | diagnostic | RQ2 raw-vs-processed oracle headroom (old setting). |
| `exp2_probe_ladder.py` | diagnostic | RQ2 probe-count saturation curve. |
| `exp3_leave_one_out.py` | diagnostic | RQ3 leave-one-method-out contribution. |
| `exp4_pipeline_ablation.py` | diagnostic | RQ3 pipeline-component ablation. |
| `exp5_probe_clustering.py` | diagnostic | RQ3 probe-family clustering. |
| `exp6_fava_extension.py` | diagnostic | RQ2 v21 on fava_binary (extra main-table row). |
| `ablation_lr_gbt_only.py`, `ablation_simple_baselines.py` | ablation | Standalone; NOT in the canonical runner. Kept (may back ablation claims). |
| `aggregate_cross_model.py` | infra | Reads every experiment JSON → `cross_model_summary.{json,md}`. |
| `patch_llama_deltas.py` | infra (utility) | Idempotent post-hoc delta fixer documented in the runbook. |
| `run-experiments.sh` | **infra (runner)** | Canonical sequential runner for exp1-6 + v21; takes `old|new` setting. **Note: does NOT run exp7** (scaffold runs are launched separately). |

## `analysis/` — figures, tables, diagnosis (all current)

`summarize_scaffold_fusion.py`, `export_scaffold_tables.py`,
`scaffold_margin_sweep.py`, `overnight_internal_state_analysis.py` (scaffold /
correctness analysis); `plot_rq1_*.py` (4 RQ1 figure scripts, dual-setting).
`overnight_runs/` holds the experiment log, the `scaffold_method_proposal.md`,
summaries, and the LaTeX tables.

## Key docs (root)

| Doc | Status |
|---|---|
| `LITERATURE.md` | current — the ~21 baselines + unification attempts + fusion machinery |
| `EXTRACTION_REFERENCE.md` | current — feature layout, splits, B2 paths |
| `IDEA_REPORT.md` | foundational — landscape/novelty/method evolution (its "Current Status" predates the scaffold+correctness pivot; references a `TARGET_LOOP.md` that is no longer present) |
| `PAPER_PLAN.md` | gitignored — paper outline |
| `EXPERIMENTS_RESULTS.md` | **stale** — scoped to the v21 diagnostic on the OLD setting; predates scaffold + correctness. Kept as the RQ1/2/3 narrative; would need a correctness-setting update before reuse. |
| `MISTRAL_RUNBOOK.md` | task done (Mistral complete) but still the canonical exp→RQ mapping + reproduction guide |

## Archived

`archive/` — see `archive/README.md`. This round: `debug_shapes.py`,
`extract_features_new.py`, `neural_fusion.py` (failed approach),
`create_val_split.py`.
