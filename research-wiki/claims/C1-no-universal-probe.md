---
type: claim
node_id: claim:C1
statement: "No single internal-state probe generalizes across tasks and models."
status: supported
added: 2026-06-16T00:00:00Z
---

# C1 — No universal probe (RQ1)

## Statement
The best-performing probe changes across datasets and across model families; no
single probe is dominant everywhere.

## Evidence
Cross-model best-single map: pca_lr wins on difficulty (e2h_amc up to 0.8937),
iti on ragtruth (0.88), lr_probe/kb_mlp on when2call, sep/pca_lr on belebele —
and the winner shifts again under the answer-correctness setting (sep/step
dominate generation-side). See `fusion/results/cross_model_summary.md`,
`analysis/overnight_runs/latest_analysis.md`.

## Status rationale
Supported by [[exp:baseline_repro]] and [[exp:exp1_oracle]] across 3 models × 7
(old) / 10 (correctness) datasets.

## Connections
[AUTO-GENERATED from graph/edges.jsonl — do not edit manually]
