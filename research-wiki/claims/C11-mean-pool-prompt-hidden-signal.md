---
type: claim
node_id: claim:C11
statement: "Mean-pooled prompt hidden states are a strong, underexplored complementary signal."
status: reported
added: 2026-06-16T00:00:00Z
---

# C11 — Mean-pooled prompt hidden states

## Statement
Averaging the prompt's hidden states (vs reading only the last token) supplies a
complementary signal that probing literature underuses.

## Evidence
In MVISF-v2 ([[idea:001]] / [[idea:009]]), adding mean-pooled prompt hidden states
gave +2.9% over last-token on the when2call routing task and was the single
biggest gain source moving from layerwise-v3 to MVISF-v2.

## Status rationale
Reported from the old dataset-label MVISF-v2 analysis (IDEA_REPORT.md). Not yet
re-verified under the current scaffold + answer-correctness setting — a good
targeted follow-up. Status: reported (pending re-confirmation).

## Connections
[AUTO-GENERATED from graph/edges.jsonl — do not edit manually]
