---
type: experiment
node_id: exp:v21_fusion
title: "v21 — Multi-View Expert-Library Stacking main result"
status: done
added: 2026-06-16T00:00:00Z
---

# v21 fusion result

## Question
Does the expert-library stacking method ([[idea:001]]) beat the best single probe
across models?

## Setup
`fusion/baseline_only_v21_winning.py` (runner step 'v21'), all 3 models, 5 core
datasets + fava + belebele. Baseline-feature-only (C1–C3 constraints).

## Key results
All datasets positive: Δ +0.24% to +6.51% (when2call), avg ≈ +2.16–2.54%. Biggest
wins when2call and belebele. Supports [[claim:C3]]. This is the RQ2 realizable-
gain table; read by [[exp:exp3_loo]] and `aggregate_cross_model.py`.

## Artifacts
`fusion/results/*/baseline_only_v21_winning_results.json`; `cross_model_summary`.

## Connections
[AUTO-GENERATED from graph/edges.jsonl — do not edit manually]
