---
type: paper
node_id: paper:niu2025_hami
title: "HaMI: Adaptive Token Selection for Hallucination Detection via Multiple-Instance Learning"
authors: ["Niu"]
year: 2025
venue: "NeurIPS 2025"
external_ids:
  arxiv: "2504.07863"
  doi: null
  s2: null
tags: ["token-position", "MIL", "learned-position"]
added: 2026-06-16T00:00:00Z
---

# HaMI: Adaptive Token Selection for Hallucination Detection via Multiple-Instance Learning

## One-line thesis
HaMI casts hallucination detection as multiple-instance learning so the model learns which token positions to probe rather than fixing them.

## Problem / Gap
Fixed probe positions (prompt-last, gen-last) can miss the informative tokens. HaMI learns the best position instead of hand-picking it.

## Method
Read-what: token-level internal representations treated as instances in a bag. Compute-what: a multiple-instance learning framework that selects and weights the most informative token positions. Output-what: a sample-level hallucination prediction with learned position selection.

## Key Results
Not extracted here; the contribution of interest is learned adaptive token selection under MIL.

## Assumptions
The informative positions vary per sample and can be learned via MIL bag-level supervision.

## Limitations / Failure Modes
Adds a learned selection stage; gains depend on having token-level features stored. Focuses on position selection within a source, not cross-source fusion.

## Reusable Ingredients
Learns the best probe position (axis 2). A natural future plug-in for our scaffold: an adaptive position selector feeding any internal-state expert.

## Open Questions
Can HaMI's MIL position selector be shared across heterogeneous experts, or must each source learn its own positions?

## Connections
[AUTO-GENERATED from graph/edges.jsonl — do not edit manually]

## Relevance to This Project
HaMI learns the best probe position (axis 2) via multiple-instance learning. It is a future plug-in for our scaffold: instead of fixing query-token position, we can let an adaptive selector choose it per expert, complementing our cross-source fusion.
