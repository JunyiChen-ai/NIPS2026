---
type: claim
node_id: claim:C2
statement: "Probes are per-example complementary: a per-example oracle exposes 10-21% AUROC headroom over the best single probe."
status: supported
added: 2026-06-16T00:00:00Z
---

# C2 — Per-example complementarity headroom (RQ2)

## Statement
Different probes are correct on different examples; an oracle that picks the right
probe per example beats the best single probe by a wide margin.

## Evidence
Oracle headroom: dataset-label mean +11.6% (100% positive rate), answer-
correctness mean +18.8% (100% positive). Per-dataset old setting: common_claim
+20.8%, when2call +12.6%, ragtruth +11.9%, fava +1.5% (saturated). Raw-feature
oracle (exp1b): raw views win 43–95% of per-sample oracle competitions.

## Status rationale
Supported by [[exp:exp1_oracle]] and [[exp:exp1b_oracle_raw]]. This headroom is
the motivation for fusion ([[claim:C3]]).

## Connections
[AUTO-GENERATED from graph/edges.jsonl — do not edit manually]
