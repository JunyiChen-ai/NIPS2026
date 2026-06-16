---
type: idea
node_id: idea:009
title: "Layerwise v3 (5 raw sources)"
stage: partial
outcome: partial
added: 2026-06-16T00:00:00Z
---

# Layerwise v3 (5 raw sources)

## Summary
Initial winning fusion using only 5 of the 13 extracted raw feature types
(input/gen last-token hidden, per-head activation, attn stats, attn value norms).

## Result / Outcome
**+1.9% to +3.2%** on 4 multi-class datasets — a real but partial win.

## Failure notes / lesson
Superseded by MVISF-v2 ([[idea:001]]), which added the 5 previously unused feature
types (notably **mean-pooled prompt hidden states**, [[claim:C11]]) and removed the
view bottleneck. Lesson: do not prune feature sources early — the unused
mean-pool signal turned out to be the strongest complementary view (+2.9% on
when2call routing).

## Connections
[AUTO-GENERATED from graph/edges.jsonl — do not edit manually]
