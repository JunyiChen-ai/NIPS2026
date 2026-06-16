---
type: experiment
node_id: exp:correctness_track
title: "Answer-correctness setting (grading + baselines, 30 cells)"
status: done
added: 2026-06-16T00:00:00Z
---

# Answer-correctness track

## Question
Does the fragmentation story (and the scaffold) hold when the label is whether the
model's own generated answer was correct, not a dataset-provided annotation?

## Setup
`reproduce/grade_correctness.py` (rule-based per-dataset grader, no judge model) →
`correctness_labels/`; `make_correctness_splits.py` (stratified 60/20/20) →
`save_processed_features_correctness.py` → `run_baselines_correctness.py`. 3 models
× 10 datasets (gsm8k, math, theoremqa, mmlu, commonsenseqa, belebele + fava,
ragtruth, common_claim_3class, when2call_3class) = 30 cells.

## Key results
Generation-side probes (sep, step) dominate this setting ([[claim:C7]]); the
scaffold lifts +3.10pp over val-single ([[exp:exp7_scaffold]], [[claim:C8]]). All
30 cells graded/split/featured/baselined.

## Artifacts
`reproduce/correctness_labels/`, `reproduce/results_correctness/`,
`fusion/results_correctness/`, `processed_features_correctness/` (B2).

## Connections
[AUTO-GENERATED from graph/edges.jsonl — do not edit manually]
