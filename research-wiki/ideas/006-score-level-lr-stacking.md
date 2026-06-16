---
type: idea
node_id: idea:006
title: "Score-level LR stacking"
stage: partial
outcome: partial
added: 2026-06-16T00:00:00Z
---

# Score-level LR stacking

## Summary
Stack the scalar output scores of each probe with a single logistic regression.

## Result / Outcome
**Weak positive: +0.3% to +1.8%.** Hit a probability-compression ceiling.

## Failure notes (anti-repeat memory)
Collapsing each probe to one scalar before stacking throws away the per-class /
per-layer structure that the expert library exploits. The fix was to keep
per-method OOF probability **vectors** (and per-layer logits), which raised gains
to +2.5%+ in [[idea:001]]. **Stack calibrated probability vectors, not scalar
scores.**

## Connections
[AUTO-GENERATED from graph/edges.jsonl — do not edit manually]
