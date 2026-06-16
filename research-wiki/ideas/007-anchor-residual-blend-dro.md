---
type: idea
node_id: idea:007
title: "Anchor-residual blend / DRO"
stage: partial
outcome: partial
added: 2026-06-16T00:00:00Z
---

# Anchor-residual blend / DRO

## Summary
Anchor on the best single probe and learn a residual correction; distributionally-
robust variant to hedge worst-case datasets.

## Result / Outcome
**Same ceiling as score-level stacking (+0.3% to +1.8%).** No improvement over the
simpler expert-library stack.

## Failure notes (anti-repeat memory)
Anchoring on a single probe caps the gain near that probe's ceiling; the
complementarity headroom (10–21%, [[claim:C2]]) needs genuine multi-expert
combination, not a residual on one anchor. DRO added robustness machinery without
beating plain val-selection. **Prefer simple reliability-weighted composition over
robust-optimization wrappers at this scale.**

## Connections
[AUTO-GENERATED from graph/edges.jsonl — do not edit manually]
