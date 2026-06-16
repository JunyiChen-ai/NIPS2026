---
type: experiment
node_id: exp:exp2_probe_ladder
title: "exp2 — probe ladder (saturation curve)"
status: done
added: 2026-06-16T00:00:00Z
---

# exp2 — probe ladder

## Question
How many probes are enough? Where do fusion gains saturate?

## Setup
`fusion/exp2_probe_ladder.py` (runner step 'exp2'): add probes one at a time,
ranked by standalone AUROC.

## Key results
Gains peak at **k=4–5** probes; k=1 already beats the best single on 4/5 datasets.
Supports the "a few complementary experts suffice" design of [[idea:002]].

## Artifacts
`fusion/results/*/probe_ladder.json`; `analysis/figures/fig1_probe_ladder`.

## Connections
[AUTO-GENERATED from graph/edges.jsonl — do not edit manually]
