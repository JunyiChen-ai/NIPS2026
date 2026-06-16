---
type: claim
node_id: claim:C7
statement: "Dataset-label settings favor input-side probes; answer-correctness settings favor generation-side probes."
status: supported
added: 2026-06-16T00:00:00Z
---

# C7 — Timing depends on target semantics (H1/H2)

## Statement
Observation timing that wins depends on what is being predicted: input-side for
dataset labels, generation-side for response correctness.

## Evidence
Generation−input AUROC gap: dataset-label mean **−0.022** (only 8.3% positive),
answer-correctness mean **+0.080** (83.3% positive) / native +0.089 (86.7%).
Best-single timing counts: dataset-label = 11 input / 1 gen; correctness = 25 gen
/ 5 input. See `analysis/overnight_runs/latest_analysis.md`.

## Status rationale
Supported by [[exp:correctness_track]] diagnosis. This is the core fragmentation
([[gap:G4]]) the scaffold ([[idea:002]]) is designed to absorb.

## Connections
[AUTO-GENERATED from graph/edges.jsonl — do not edit manually]
