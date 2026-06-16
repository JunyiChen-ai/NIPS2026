# archive/

Dead or superseded code, kept for provenance — **not deleted**. Nothing here is
imported or invoked by the live pipeline (verified by grep across `*.py`/`*.sh`/
`*.md` before archiving). Git history is preserved (files were `git mv`-ed, so
`git log --follow` still works).

> ⚠️ Archived scripts may reference absolute paths or sibling modules that have
> since moved; they are snapshots for reference, not guaranteed to run as-is from
> here. To resurrect one, move it back to its original directory first.

## Contents

| Archived path | Original path | Why archived |
|---|---|---|
| `extraction/debug_shapes.py` | `extraction/debug_shapes.py` | Throwaway scratch script (`Debug: print the exact shapes of generate() outputs`). Zero references anywhere in the repo. Used once while building the extractor. |
| `extraction/extract_features_new.py` | `extraction/extract_features_new.py` | Phase-2 extractor, explicitly **superseded** by `extraction/extract_all.py`, whose docstring states it "replaces the two separate entry points (`extract_features.py` for Phase 1, `extract_features_new.py` for Phase 2) with a single CLI-driven script". No live importer. (Note: the *library* `extraction/extract_features.py` stays live — `extract_all.py` imports `FeatureExtractor` from it.) |
| `fusion/neural_fusion.py` | `fusion/neural_fusion.py` | **Failed approach** (Hierarchical Multi-Layer Adapter Fusion, ~493K params): −2% to −9% AUROC, overfit on the 800–3500-sample datasets. Kept as a documented negative result. Imported by nothing; absent from `run-experiments.sh` and `MISTRAL_RUNBOOK.md`. The lightweight, sklearn-only `fusion/exp7_scaffold_fusion.py` (current) is the working successor. |
| `reproduce/create_val_split.py` | `reproduce/create_val_split.py` | Old-setting one-off that built validation splits under the old `/data/jehc223` layout. Superseded by `reproduce/make_correctness_splits.py` (new setting) and `extraction/setup_e2h_multiclass.py` (which copied its stratification logic — see the comments there). No importer. |

## What was deliberately **kept** (looks old, but load-bearing)

These were considered for archiving and **kept** because they still back the
paper's diagnostic evidence and/or are import/JSON dependencies:

- `fusion/baseline_only_v21_winning.py` (v21) — `exp3` reads its results JSON to
  compute leave-one-out contributions, and `aggregate_cross_model.py` maps it
  into the RQ2 cross-model table.
- `fusion/exp1..exp6` — the RQ1/RQ2/RQ3 diagnostic experiments in the canonical
  `run-experiments.sh`.
- Old `reproduce/` pipeline (`run_all.py`, `run_new_datasets.py`,
  `save_processed_features.py`) — the only way to regenerate the old-setting
  `processed_features/` tree that `exp1..exp6`/v21 consume; `run_new_datasets.py`
  also imports `run_all.py`.

See `../STRUCTURE.md` for the full repo map.
