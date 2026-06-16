---
type: idea
node_id: idea:002
title: "Plug-and-play scaffold fusion (exp7)"
stage: succeeded
outcome: positive
added: 2026-06-16T00:00:00Z
---

# Plug-and-play scaffold fusion (exp7)

## Summary
The **current paper-facing method**. A low-capacity scaffold that absorbs
heterogeneous probes via a uniform adapter `StandardScaler → PCA(64) → LR`, then
composes calibrated per-method probabilities three ways — reliability-weighted
all-methods, family-aware, timing-aware — and selects among them (plus the
val-best single) by **validation AUROC only**. A new probe plugs in by writing
`{train,val,test}.pt`; no method-specific training logic, no LLM retraining.
Optional conservative `margin` mode gives no-regression deployment.

## Based on
Diagnosis that fragmentation runs along signal-source, timing, and target-
semantics ([[gap:G4]]); differentiates from [[paper:sriramanan2024_llm_check]]
(no learned fusion), [[paper:ghadiri2025_gnosis]] (2-source, backbone-bound),
[[paper:halunet2025]] (output-side only).

## Target gaps
[[gap:G1]], [[gap:G3]], [[gap:G4]].

## Result / Outcome
vs validation-selected single probe: **new (answer-correctness) +3.10pp / 80% win
(N=30); old (dataset-label) +1.13pp / 82.4% win (N=17)**; matches the test-oracle
single on the new setting and beats it (+0.73pp) on the old. dim64 is the capacity
sweet spot. Generalizes across BOTH target semantics → [[claim:C8]].

## Open follow-ups
HaloScope and HaMI-style adaptive token position are natural next plug-ins
([[gap:G2]]). Remaining failure: mistral/theoremqa val-test ranking shift
([[claim:C12]]).

## Connections
[AUTO-GENERATED from graph/edges.jsonl — do not edit manually]
