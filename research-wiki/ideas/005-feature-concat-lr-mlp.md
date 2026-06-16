---
type: idea
node_id: idea:005
title: "Feature concatenation + LR/MLP"
stage: failed
outcome: negative
added: 2026-06-16T00:00:00Z
---

# Feature concatenation + LR/MLP

## Summary
Concatenate all probes' raw feature vectors (scalar → 17920-dim) and train one
LR or MLP on the stacked vector.

## Result / Outcome
**All negative.** Eliminated by the curse of dimensionality: a few thousand
samples cannot support a classifier over a concatenated feature space that mixes
17920-dim SEP features with scalar scores.

## Failure notes (anti-repeat memory)
Early feature-level fusion is the wrong altitude for extreme-heterogeneity probe
fusion. **Fuse at the prediction level (per-expert calibrated probabilities), not
the feature level.** This motivated the per-expert design of [[idea:001]] and the
per-method adapter of [[idea:002]].

## Connections
[AUTO-GENERATED from graph/edges.jsonl — do not edit manually]
