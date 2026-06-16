---
type: idea
node_id: idea:001
title: "Multi-View Expert-Library Stacking (v21)"
stage: succeeded
outcome: positive
added: 2026-06-16T00:00:00Z
---

# Multi-View Expert-Library Stacking (v21)

## Summary
Treat each probe as an independent plug-in expert. Per method:
`StandardScaler → PCA({32,128}) → {LR, GBT, ExtraTrees, RF} → 5-fold OOF × 5 seeds`,
then entropy/margin enrichment, then a `{L2-LR, L1-LR, GBT}` meta-blend. Pure
sklearn, no neural components, no LLM retraining. Won a 19-iteration
target-driven loop over 15+ architectures.

## Based on
[[paper:unifact2025]] (hybrid strongest), fusion machinery [[paper:han2024_fusemoe]],
[[paper:yun2024_flex_moe]], [[paper:hemker2024_healnet]].

## Target gaps
[[gap:G1]] (fuse heterogeneous sources), [[gap:G3]].

## Result / Outcome
All 5 core datasets positive vs best single: +0.96% to +6.51% (when2call),
avg ≈ +2.54%, Wilcoxon p = 0.0098. The diagnostic "winning" method written into
the (Apr 22) paper draft. Still load-bearing: [[exp:exp3_loo]] reads its results.

## Notes
Superseded as the *headline* method by the lighter [[idea:002]] scaffold, but
retained as the RQ2 realizable-gain result. The heavier capacity is unnecessary
under the plug-and-play framing.

## Connections
[AUTO-GENERATED from graph/edges.jsonl — do not edit manually]
