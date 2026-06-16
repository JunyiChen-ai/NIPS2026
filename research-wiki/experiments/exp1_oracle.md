---
type: experiment
node_id: exp:exp1_oracle
title: "exp1 — per-example oracle headroom"
status: done
added: 2026-06-16T00:00:00Z
---

# exp1 — per-example oracle

## Question
If you could pick the right probe per example, how much would you beat the best
single probe?

## Setup
`fusion/exp1_oracle_complete.py` (canonical runner step 'exp1'). Produces
`oracle_complete.json` (best_single + oracle), consumed by exp2/3/4.

## Key results
Oracle headroom 10–21% per dataset, 100% positive rate across settings
([[claim:C2]]); confirms probes err on different examples ([[claim:C1]]).

## Artifacts
`fusion/results/*/oracle_complete.json`.

## Connections
[AUTO-GENERATED from graph/edges.jsonl — do not edit manually]
