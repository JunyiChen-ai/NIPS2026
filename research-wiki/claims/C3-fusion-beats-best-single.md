---
type: claim
node_id: claim:C3
statement: "Fusing multiple probes beats the best single probe across tasks and models."
status: supported
added: 2026-06-16T00:00:00Z
---

# C3 — Fusion beats best single (RQ2)

## Statement
A learned combination of probes outperforms the best individual probe, realizing
part of the oracle headroom.

## Evidence
v21 stacking: all 5 core datasets positive, +0.24% to +6.51% (avg ≈ +2.54%,
Wilcoxon p = 0.0098). Scaffold (exp7): +3.10pp / 80% win (answer-correctness,
N=30) and +1.13pp / 82.4% win (dataset-label, N=17) vs validation-selected single.

## Status rationale
Supported by [[exp:v21_fusion]] and [[exp:exp7_scaffold]]. Realized gain is well
below oracle headroom ([[claim:C2]]) — the gap is future work.

## Connections
[AUTO-GENERATED from graph/edges.jsonl — do not edit manually]
