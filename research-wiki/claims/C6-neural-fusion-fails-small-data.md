---
type: claim
node_id: claim:C6
statement: "Neural fusion fails on small (800-3500-sample) probing datasets; linear stacking is optimal."
status: supported
added: 2026-06-16T00:00:00Z
---

# C6 — Neural fusion fails at this data scale

## Statement
High-capacity neural fusion overfits probing-scale datasets; sklearn linear
stacking is the right capacity.

## Evidence
Neural hierarchical fusion (493K params): −2% to −9% AUROC ([[idea:004]]). MoE
routing abandoned as too complex for the sample sizes ([[idea:003]]). Feature
concat + MLP all negative ([[idea:005]]). The scaffold's capacity sweep confirms
lower capacity (PCA dim 64 < 128) generalizes better ([[claim:C9]]).

## Status rationale
Supported by the archived `archive/fusion/neural_fusion.py` run and the idea
post-mortems. A standing design constraint for all future fusion work here.

## Connections
[AUTO-GENERATED from graph/edges.jsonl — do not edit manually]
