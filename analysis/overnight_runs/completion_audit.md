# Completion Audit: Scaffold-Oriented Core Method

## Restated Objective

The paper should land on a rethinking-style contribution: empirical diagnosis
of fragmented internal-state probing methods, followed by a new generalizable
plug-and-play scaffold. The method should be scalable, support adding new
methods directly, avoid unnecessary retraining from scratch, and outperform
single methods across settings.

## Prompt-to-Artifact Checklist

| Requirement | Evidence | Status |
|---|---|---|
| Use experimental observations to develop a new method | `analysis/overnight_internal_state_analysis.py`; `analysis/overnight_runs/latest_analysis.md`; `analysis/scaffold_method_proposal.md` links diagnosis to method design | Covered |
| New method behaves like a scalable scaffold | `fusion/exp7_scaffold_fusion.py` defines a generic processed-feature plug-in contract and shared adapters/composers | Covered |
| Supports plug-and-play new methods | `fusion/exp7_scaffold_fusion.py` consumes `{method}/train.pt`, `{method}/val.pt`, `{method}/test.pt` with no method-specific training branch | Covered |
| Avoids retraining LLM from scratch | `analysis/scaffold_method_proposal.md` documents lightweight adapters only; `fusion/exp7_scaffold_fusion.py` uses processed features and sklearn adapters, no LLM forward/retraining | Covered |
| Works across different target semantics/settings | Exp7 run on new correctness setting and old dataset-label setting; artifacts in `fusion/results_correctness/*/scaffold_fusion_dim64.json` and `fusion/results/*/scaffold_fusion_dim64.json` | Covered |
| Better than a single method on average | `analysis/overnight_runs/scaffold_fusion_dim64_summary.md`: new +3.10pp / 80.0% win rate; old +1.13pp / 82.4% win rate vs validation-selected single | Covered |
| Has conservative/no-regression mode | `fusion/exp7_scaffold_fusion.py --selection-margin`; actual runs in `fusion/results_correctness/*/scaffold_fusion_dim64_margin002.json` and `fusion/results/*/scaffold_fusion_dim64_margin002.json`; summary in `analysis/overnight_runs/scaffold_fusion_dim64_margin002_summary.md` | Covered |
| Bad results are diagnosed rather than discarded | `analysis/overnight_runs/experiment_log.md` documents Mistral-TheoremQA failure; `scaffold_fusion_dim64_oofmeta.json` tests OOF-meta and records non-fix | Covered |
| Results are recorded and reproducible | `analysis/overnight_runs/experiment_log.md` records commands; summaries and LaTeX tables exported under `analysis/overnight_runs/` | Covered |
| Verification run exists | `py_compile` passed for new scripts; `reproduce/grade_correctness_test.py` direct execution passed; checked no active background experiment process | Covered |

## Evidence Inspected

- Method script:
  - `fusion/exp7_scaffold_fusion.py`
- Analysis and proposal:
  - `analysis/scaffold_method_proposal.md`
  - `analysis/overnight_runs/experiment_log.md`
  - `analysis/overnight_runs/scaffold_fusion_dim64_summary.md`
  - `analysis/overnight_runs/scaffold_fusion_dim64_margin002_summary.md`
  - `analysis/overnight_runs/scaffold_margin_sweep.md`
  - `analysis/overnight_runs/scaffold_tables.tex`
- Primary outputs:
  - `fusion/results_correctness/qwen2.5-7b/scaffold_fusion_dim64.json`
  - `fusion/results_correctness/llama3.1-8b/scaffold_fusion_dim64.json`
  - `fusion/results_correctness/mistral-7b-v0.3/scaffold_fusion_dim64.json`
  - `fusion/results/qwen2.5-7b/scaffold_fusion_dim64.json`
  - `fusion/results/llama3.1-8b/scaffold_fusion_dim64.json`
  - `fusion/results/mistral-7b-v0.3/scaffold_fusion_dim64.json`

## Residual Risks / Gaps

- Old setting has 17 completed model-dataset points, not 18, because
  `mistral-7b-v0.3/fava/train/meta.json` is missing in the old extraction
  layout. The script now records this as skipped instead of crashing.
- Conservative mode guarantees no regression relative to validation-selected
  single in the actual run, but not relative to test-oracle single. This is
  expected because test-oracle selection is unavailable at deployment time.
- The method is not yet written into `paper/sections/*.tex`; this audit covers
  method development, experiment artifacts, and paper-ready notes/tables.

## Conclusion

The requested scaffold-oriented method stage is achieved: there is an
implemented plug-and-play scaffold, cross-setting experimental evidence,
diagnosis of failures, conservative deployment mode, and paper-ready records.
