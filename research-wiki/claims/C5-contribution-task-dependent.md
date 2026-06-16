---
type: claim
node_id: claim:C5
statement: "Each probe's contribution to the fusion is task-dependent."
status: supported
added: 2026-06-16T00:00:00Z
---

# C5 — Task-dependent contribution (RQ3)

## Statement
A probe that is load-bearing on one task is near-redundant on another; no probe is
uniformly important.

## Evidence
Leave-one-method-out deltas: iti contributes +3.48pp on ragtruth but +0.03pp on
e2h_amc_3class; attn_satisfies is the top contributor on when2call (+0.80pp);
weak probes (sep/step) sit near −0.04pp (harmless). See
`fusion/results/.../leave_one_method_out.json`.

## Status rationale
Supported by [[exp:exp3_loo]]. Directly motivates the reliability-weighted,
family/timing-aware composition of [[idea:002]].

## Connections
[AUTO-GENERATED from graph/edges.jsonl — do not edit manually]
