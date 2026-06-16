---
type: claim
node_id: claim:C10
statement: "The scaffold supports a conservative no-regression deployment mode via a validation margin gate."
status: supported
added: 2026-06-16T00:00:00Z
---

# C10 — Conservative no-regression mode

## Statement
With a validation margin `gamma`, the scaffold only adopts an aggregate candidate
if it beats the val-best single by `gamma` on validation, else falls back —
guaranteeing no regression vs the val-selected single.

## Evidence
margin=0 (aggressive): +3.10pp mean, but min delta −2.76pp. margin=0.02
(conservative): +2.37pp mean, **min delta 0.00pp** (no regression vs val-single).
Raising margin monotonically trades mean gain for tail safety. Artifact:
`scaffold_fusion_dim64_margin002` summaries.

## Status rationale
Supported by [[exp:exp7_scaffold]] margin sweep. Note: no-regression holds vs the
*validation-selected* single, not vs the (unavailable-at-deploy) test-oracle.

## Connections
[AUTO-GENERATED from graph/edges.jsonl — do not edit manually]
