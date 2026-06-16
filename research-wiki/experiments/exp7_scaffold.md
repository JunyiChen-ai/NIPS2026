---
type: experiment
node_id: exp:exp7_scaffold
title: "exp7 — plug-and-play scaffold fusion (current method)"
status: done
added: 2026-06-16T00:00:00Z
---

# exp7 — scaffold fusion

## Question
Can one low-capacity plug-and-play scaffold beat the val-selected single probe
across BOTH target semantics, with a no-regression mode?

## Setup
`fusion/exp7_scaffold_fusion.py`. Adapter `StandardScaler→PCA(d)→LR`; candidates =
val-single, weighted-all, family-aware, timing-aware (+ diagnostic meta-LR /
opt-in OOF-meta); val-only selection; margin gate. Run on both settings × 3 models
× dim {32,64,128}; margin sweep. NOTE: not in `run-experiments.sh` — driven by
`analysis/overnight_runs/experiment_log.md`.

## Key results
dim64: answer-correctness +3.10pp / 80% win (N=30, ≈ test-oracle); dataset-label
+1.13pp / 82.4% win (beats oracle +0.73pp). dim64 sweet spot ([[claim:C9]]);
margin002 = no-regression ([[claim:C10]]). Supports [[claim:C3]], [[claim:C8]].
Failure: mistral/theoremqa ([[claim:C12]]).

## Artifacts
`fusion/results_correctness/*/scaffold_fusion_dim64*.json`,
`fusion/results/*/scaffold_fusion_dim64*.json`,
`analysis/overnight_runs/scaffold_fusion_dim64_summary.md`,
`analysis/scaffold_method_proposal.md`.

## Connections
[AUTO-GENERATED from graph/edges.jsonl — do not edit manually]
