---
type: claim
node_id: claim:C9
statement: "Lower-capacity adapters (PCA dim 64) generalize better than higher capacity (128) for the scaffold."
status: supported
added: 2026-06-16T00:00:00Z
---

# C9 — Capacity sweet spot is dim 64

## Statement
Adapter capacity trades off; dim 64 maximizes mean gain and win rate, and dim 128
adds nothing.

## Evidence
Answer-correctness capacity sweep (vs val-single): dim32 +2.76pp / 76.7% win;
**dim64 +3.10pp / 80.0%**; dim128 +2.24pp / 66.7%. Lower capacity is also more
stable on the worst cases (dim32 has a far worse mistral/theoremqa outlier).

## Status rationale
Supported by [[exp:exp7_scaffold]] capacity sweep. Consistent with [[claim:C6]]
(low capacity wins at this data scale).

## Connections
[AUTO-GENERATED from graph/edges.jsonl — do not edit manually]
