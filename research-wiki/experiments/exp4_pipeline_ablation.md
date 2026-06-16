---
type: experiment
node_id: exp:exp4_pipeline_ablation
title: "exp4 — pipeline component ablation"
status: done
added: 2026-06-16T00:00:00Z
---

# exp4 — pipeline ablation

## Question
Which design choices in the v21 stacking pipeline are load-bearing?

## Setup
`fusion/exp4_pipeline_ablation.py` (runner step 'exp4'): ablate PCA resolution,
expert types, meta-blend, enrichment, seeds (≈14 configs).

## Key results
Load-bearing: expert-type diversity (−0.8 to −2.5% when removed) and the meta-
blend (−0.5 to −1%). Droppable with no loss (≈40% compute saved): PCA(32),
enrichment, seeds 4–5. Informs the lean capacity of [[idea:002]].

## Artifacts
`fusion/results/*/pipeline_ablation.json`;
`analysis/figures/fig5_pipeline_ablation`. (fava_binary / belebele rows are "—".)

## Connections
[AUTO-GENERATED from graph/edges.jsonl — do not edit manually]
