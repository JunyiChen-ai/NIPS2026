---
type: experiment
node_id: exp:exp1b_oracle_raw
title: "exp1b — raw-vs-processed oracle headroom"
status: done
added: 2026-06-16T00:00:00Z
---

# exp1b — oracle with raw features

## Question
Do raw LLM hidden states add per-example oracle headroom beyond the post-processed
baseline features?

## Setup
`fusion/exp1b_oracle_with_raw.py` (runner step 'exp1b', old setting only; must run
while raw features are on disk).

## Key results
Raw views win 43–95% of per-sample oracle competitions; "+raw" oracle reaches
≈0.99–1.00 on several datasets. Supports [[claim:C2]]. (Note: the paper's C1/C2/C3
framing constrains the method to baseline-feature-only — C1 constraint.)

## Artifacts
`fusion/results/*/oracle_with_raw.json`; cross-model "+raw" table in
`cross_model_summary.md`.

## Connections
[AUTO-GENERATED from graph/edges.jsonl — do not edit manually]
