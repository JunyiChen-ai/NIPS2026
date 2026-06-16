---
type: idea
node_id: idea:004
title: "Neural hierarchical fusion (493K params)"
stage: failed
outcome: negative
added: 2026-06-16T00:00:00Z
---

# Neural hierarchical fusion (493K params)

## Summary
A learned hierarchical multi-layer adapter network fusing per-source raw features
(code archived at `archive/fusion/neural_fusion.py`).

## Result / Outcome
**−2% to −9% AUROC** vs the best single probe across datasets.

## Failure notes (anti-repeat memory)
493K parameters massively overfit the 800–3500-sample datasets. This is the
single strongest evidence that **neural fusion does not work at this data scale;
linear stacking is optimal** ([[claim:C6]]). Kept as a documented negative
result. **Do not reintroduce a neural fusion head** unless dataset sizes grow by
an order of magnitude.

## Connections
[AUTO-GENERATED from graph/edges.jsonl — do not edit manually]
