---
type: idea
node_id: idea:008
title: "Multi-View v1 (view bottleneck)"
stage: failed
outcome: partial
added: 2026-06-16T00:00:00Z
---

# Multi-View v1 (view bottleneck)

## Summary
Organize features into views, but pass each view through a low-dim bottleneck
before the meta-combiner.

## Result / Outcome
**Mixed** — the view-level bottleneck helped some datasets and hurt others.

## Failure notes (anti-repeat memory)
Forcing every view through a fixed bottleneck destroys signal on datasets where a
view is high-rank. Removing the bottleneck (sending all per-layer OOF logits
straight to the meta-LR) is exactly what made MVISF-v2 / [[idea:001]] win.
**Avoid premature dimensionality bottlenecks between experts and the meta-layer.**

## Connections
[AUTO-GENERATED from graph/edges.jsonl — do not edit manually]
