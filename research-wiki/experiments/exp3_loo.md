---
type: experiment
node_id: exp:exp3_loo
title: "exp3 — leave-one-method-out contribution"
status: done
added: 2026-06-16T00:00:00Z
---

# exp3 — leave-one-method-out

## Question
What does each probe contribute to the fusion (full − leave-one-out)?

## Setup
`fusion/exp3_leave_one_out.py` (runner step 'exp3'); reads the v21 results JSON
([[exp:v21_fusion]]) to compute per-method deltas.

## Key results
Contribution is task-dependent ([[claim:C5]]): iti +3.48pp on ragtruth vs +0.03pp
on e2h_amc_3class; weak probes (sep/step) ≈ −0.04pp (harmless, not harmful).

## Artifacts
`fusion/results/*/leave_one_method_out.json`; `analysis/figures/fig2_loo_heatmap`.

## Connections
[AUTO-GENERATED from graph/edges.jsonl — do not edit manually]
