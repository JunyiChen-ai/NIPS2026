---
type: experiment
node_id: exp:exp5_clustering
title: "exp5 — probe error-correlation clustering"
status: done
added: 2026-06-16T00:00:00Z
---

# exp5 — probe clustering

## Question
Do probes group by computational signal type?

## Setup
`fusion/exp5_probe_clustering.py` (runner step 'exp5'): error-correlation matrix +
hierarchical (Ward) clustering + t-SNE.

## Key results
~2 algorithmic clusters (5 input-side residual/attention probes vs sep/step
generation pair); errors near-orthogonal across families ([[claim:C4]]). Grounds
the family-aware composition of [[idea:002]].

## Artifacts
`fusion/results/*/probe_clustering.json`; `analysis/figures/fig3_clustering`,
`fig3b_tsne`.

## Connections
[AUTO-GENERATED from graph/edges.jsonl — do not edit manually]
