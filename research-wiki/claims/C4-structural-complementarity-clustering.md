---
type: claim
node_id: claim:C4
statement: "Probe complementarity is structural: probes cluster by signal type with near-orthogonal errors across groups."
status: supported
added: 2026-06-16T00:00:00Z
---

# C4 — Structural complementarity (RQ3)

## Statement
The differences between probes are driven by computational design, not random
noise: errors correlate within a signal family and are near-orthogonal across
families.

## Evidence
Hierarchical (Ward) clustering of probe error correlations yields ~2 algorithmic
clusters (5 input-side residual/attention probes vs the sep/step generation
pair); the semantic "3-family" view (residual / attention / generation) overlays
on this. See `fusion/results/.../probe_clustering.json`, `analysis/figures/`.

## Status rationale
Supported by [[exp:exp5_clustering]]. Justifies treating each probe as an
independent expert rather than merging features ([[idea:005]] failed).

## Connections
[AUTO-GENERATED from graph/edges.jsonl — do not edit manually]
