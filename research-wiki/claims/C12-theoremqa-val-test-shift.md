---
type: claim
node_id: claim:C12
statement: "The scaffold's main remaining failure (mistral / theoremqa) is a validation-test ranking shift, not a missing feature."
status: reported
added: 2026-06-16T00:00:00Z
---

# C12 — TheoremQA failure is a val-test ranking shift

## Statement
The worst scaffold case is caused by validation rankings not transferring to test,
not by a missing probe or model-capacity limit.

## Evidence
mistral-7b-v0.3 / theoremqa (answer-correctness, dim64): weighted_all_methods val
0.9020 → test 0.6852; lr_probe single val 0.8868 → test 0.7128; the best test
single (kb_mlp 0.7325) is ranked lower on val. OOF-meta did NOT repair it
(val 0.8525 → test 0.6622); the val-trained diagnostic meta-LR reaches test 0.7444
but overfits val elsewhere, so it is not deployed.

## Status rationale
Reported from `analysis/overnight_runs/experiment_log.md`. Open problem: a
val-test-robust selector for low-N reasoning datasets.

## Connections
[AUTO-GENERATED from graph/edges.jsonl — do not edit manually]
