---
type: idea
node_id: idea:003
title: "ProbeCoalition (MoE + router + disagreement)"
stage: failed
outcome: negative
added: 2026-06-16T00:00:00Z
---

# ProbeCoalition (MoE + router + disagreement)

## Summary
A learned mixture-of-experts router over probes, with a disagreement signal, in
the spirit of FuseMoE/Flex-MoE.

## Result / Outcome
Not implemented to completion — judged **too complex for the available sample
sizes (800–3500)**; the neural variants it depended on already failed.

## Failure notes (anti-repeat memory)
Learned routing/gating needs more data than probing datasets provide. At this
sample scale, any high-capacity gate overfits. **Do not re-attempt neural MoE
routing over probes without first securing 10k+ labeled examples per cell.**
The working alternative is reliability-weighted composition with val-only
selection ([[idea:002]]).

## Connections
[AUTO-GENERATED from graph/edges.jsonl — do not edit manually]
