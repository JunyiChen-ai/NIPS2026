# Overnight Experiment Log

## Objective

Develop and test a paper-facing method from the diagnosis that internal-state
methods are fragmented by signal source, timing, and target semantics. The new
method should work as a plug-and-play scaffold: new probes expose processed
features, and the scaffold handles generic calibration, family/timing-aware
composition, and validation-only selection without retraining the LLM.

## Method Implemented

`fusion/exp7_scaffold_fusion.py`

- Input contract: each method contributes `train.pt`, `val.pt`, `test.pt`.
- Generic adapter: `StandardScaler -> PCA(max_dim) -> LogisticRegression`.
- Scaffold candidates:
  - validation-selected single method,
  - reliability-weighted all-method average,
  - family-aware scaffold,
  - timing-aware scaffold,
  - diagnostic val-trained meta-LR,
  - optional OOF-meta scaffold.
- Default selection: validation-only selection among low-capacity plug-and-play
  candidates. The val-trained meta-LR is recorded as a diagnostic candidate
  because it can overfit validation.

## Main Commands

```bash
/home/junyi/miniconda3/envs/REAL/bin/python -u fusion/exp7_scaffold_fusion.py \
  --model qwen2.5-7b --setting new --max-dim 64 --out-prefix scaffold_fusion_dim64 --force

/home/junyi/miniconda3/envs/REAL/bin/python -u fusion/exp7_scaffold_fusion.py \
  --model llama3.1-8b --setting new --max-dim 64 --out-prefix scaffold_fusion_dim64 --force

/home/junyi/miniconda3/envs/REAL/bin/python -u fusion/exp7_scaffold_fusion.py \
  --model mistral-7b-v0.3 --setting new --max-dim 64 --out-prefix scaffold_fusion_dim64 --force

/home/junyi/miniconda3/envs/REAL/bin/python -u fusion/exp7_scaffold_fusion.py \
  --model qwen2.5-7b --setting old --max-dim 64 --out-prefix scaffold_fusion_dim64 --force

/home/junyi/miniconda3/envs/REAL/bin/python -u fusion/exp7_scaffold_fusion.py \
  --model llama3.1-8b --setting old --max-dim 64 --out-prefix scaffold_fusion_dim64 --force

/home/junyi/miniconda3/envs/REAL/bin/python -u fusion/exp7_scaffold_fusion.py \
  --model mistral-7b-v0.3 --setting old --max-dim 64 --out-prefix scaffold_fusion_dim64 --force

/home/junyi/miniconda3/envs/REAL/bin/python analysis/summarize_scaffold_fusion.py \
  --settings new old --result-name scaffold_fusion_dim64

/home/junyi/miniconda3/envs/REAL/bin/python analysis/scaffold_margin_sweep.py
```

## Primary Evidence

Artifacts:

- `analysis/overnight_runs/scaffold_fusion_dim64_summary.md`
- `analysis/overnight_runs/scaffold_fusion_dim64_summary.json`
- `analysis/overnight_runs/scaffold_margin_sweep.md`
- `analysis/overnight_runs/scaffold_margin_sweep.json`
- `fusion/results_correctness/*/scaffold_fusion_dim64.json`
- `fusion/results/*/scaffold_fusion_dim64.json`

Correctness / Generative Answer Correctness setting:

- 30 model-dataset points.
- Mean delta vs validation-selected single method: `+0.0310`.
- Median delta: `+0.0174`.
- Win rate: `80.0%`.
- Mean delta vs test-oracle single: approximately `0.0000`.
- Selected variants:
  - family-aware scaffold: 15,
  - weighted all-methods: 9,
  - timing-aware scaffold: 3,
  - fallback single method: 3.

Dataset-label / latent-style setting:

- 17 completed model-dataset points. Mistral/FAVA old setting is skipped because
  the old extraction label file is missing.
- Mean delta vs validation-selected single method: `+0.0113`.
- Median delta: `+0.0108`.
- Win rate: `82.4%`.
- Mean delta vs test-oracle single: `+0.0073`.

Capacity diagnosis:

- `max_dim=128`: mean delta `+0.0224`, win rate `66.7%`.
- `max_dim=64`: mean delta `+0.0310`, win rate `80.0%`.
- `max_dim=32`: mean delta `+0.0276`, win rate `76.7%`.
- Interpretation: lower-capacity adapters are more stable for plug-and-play
  generalization; `max_dim=64` is the best current default.

Deployment policy diagnosis:

- Aggressive policy (`margin=0`) gives largest mean lift in new setting:
  `+0.0310`, win rate `80.0%`, min delta `-0.0276`.
- Conservative policy (`margin=0.02`) still gives positive mean lift:
  `+0.0237`, min delta `0.0000`.
- Interpretation: the scaffold can support two modes:
  - aggressive mode for best average performance,
  - conservative mode for no-regression deployment relative to val-selected single.

## Failure Diagnosis

Main remaining failure:

- `mistral-7b-v0.3 / theoremqa / new`.
- `weighted_all_methods`: val AUROC `0.9020`, test AUROC `0.6852`.
- `single:lr_probe`: val AUROC `0.8868`, test AUROC `0.7128`.
- Best test single in this run is `kb_mlp` at `0.7325`, but validation ranks it
  lower.
- Diagnosis: this is a validation-test ranking shift, not simply a missing
  feature or an obvious model-capacity issue.

OOF-meta check:

- `fusion/results_correctness/mistral-7b-v0.3/scaffold_fusion_dim64_oofmeta.json`.
- OOF-meta did not repair this case: val AUROC `0.8525`, test AUROC `0.6622`.
- The val-trained diagnostic meta-LR reaches test AUROC `0.7444`, but it is not
  used by default because on other cases it can overfit validation.

## Paper-Framing Takeaway

The scaffold result directly supports the core paper claim:

1. Diagnosis: single methods are fragmented; method family and timing matter.
2. Method: a generic scaffold can accept heterogeneous probes without rewriting
   method-specific training logic.
3. Generality: the same scaffold works in both dataset-label and response-
   correctness settings.
4. Practicality: no LLM retraining is required; only lightweight adapters are
   trained on already-extracted processed features.
5. Scalability: adding a new method only requires adding processed features to
   the method directory; scaffold composition and selection are shared.
