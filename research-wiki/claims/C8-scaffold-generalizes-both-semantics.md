---
type: claim
node_id: claim:C8
statement: "A single low-capacity plug-and-play scaffold generalizes across both target semantics (dataset-label AND answer-correctness)."
status: supported
added: 2026-06-16T00:00:00Z
---

# C8 — Scaffold generalizes across target semantics

## Statement
The same scaffold, with no task-specific tuning, beats the validation-selected
single probe in both the dataset-label and the answer-correctness settings.

## Evidence
exp7 dim64: answer-correctness +3.10pp / 80% win (N=30); dataset-label +1.13pp /
82.4% win (N=17). Selected variants split across family-aware (15), weighted-all
(9), timing-aware (3), single fallback (3) — i.e. the composition that wins is
itself task-dependent, which the val-only selector handles.

## Status rationale
Supported by [[exp:exp7_scaffold]]. This is the central paper claim of the current
direction.

## Connections
[AUTO-GENERATED from graph/edges.jsonl — do not edit manually]
