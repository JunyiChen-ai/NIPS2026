---
type: paper
node_id: paper:wehner2025_repe_survey
title: "Representation Engineering: A Survey"
authors: ["Jan Wehner"]
year: 2025
venue: "arXiv (2025-02)"
external_ids:
  arxiv: "2502.17601"
  doi: null
  s2: null
tags: ["survey", "representation-engineering"]
added: 2026-06-16T00:00:00Z
---

# Representation Engineering: A Survey

## One-line thesis
A normalising survey that frames representation engineering — reading and steering model internal representations — as a coherent research area spanning probing and intervention.

## Problem / Gap
Work on probing internal states and on steering them had grown without a shared vocabulary or map. The survey organises probe-and-steer methods rather than benchmarking them.

## Method
Read the representation-engineering literature; group methods by what they read from internal states and how they intervene; output a structured taxonomy covering probing and steering; no new benchmark is run.

## Key Results
Provides a taxonomy and survey rather than empirical results; no numbers to record.

## Assumptions
Probing and steering share enough structure to be surveyed together; internal representations carry recoverable, steerable concepts.

## Limitations / Failure Modes
Surveys lag the fast-moving literature; no head-to-head method comparison; coverage choices reflect the authors' framing.

## Reusable Ingredients
Confirms the gap our work targets: no prior work fuses multiple probing methods. We cite it as the canonical map of the probe-and-steer space.

## Open Questions
Where does multi-probe fusion sit in this taxonomy? Does combining read-out methods change the steering story?

## Connections
[AUTO-GENERATED from graph/edges.jsonl — do not edit manually]

## Relevance to This Project
This survey is our evidence that no prior work fuses multiple probing methods, motivating the multi-view expert-library stacking idea; it normalises the probe + steering space we draw experts from.
