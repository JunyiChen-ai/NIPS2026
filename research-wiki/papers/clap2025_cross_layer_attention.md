---
type: paper
node_id: paper:clap2025_cross_layer_attention
title: "Cross-Layer Attention Probing (CLAP) for Hallucination Detection"
authors: ["(preprint)"]
year: 2025
venue: "arXiv preprint (2025)"
external_ids:
  arxiv: "2509.09700"
  doi: null
  s2: null
tags: ["attention", "cross-layer", "single-source"]
added: 2026-06-16T00:00:00Z
---

# Cross-Layer Attention Probing (CLAP) for Hallucination Detection

## One-line thesis
CLAP probes attention across layers and tokens for hallucination detection, but draws from a single signal source.

## Problem / Gap
Attention probes often read one layer or one token. CLAP widens the read to span layers and tokens, yet stays within the attention signal source alone.

## Method
Read-what: attention patterns spanning multiple layers and multiple tokens. Compute-what: a cross-layer / cross-token probe maps them to a detection score. Output-what: a hallucination signal.

## Key Results
Not extracted here; the relevant point is that breadth across layers and tokens stays within attention.

## Assumptions
Attention across layers and tokens carries sufficient detection signal on its own.

## Limitations / Failure Modes
Single signal source (attention). No residual, logit, SAE, or multi-sample fusion. Cross-layer breadth does not equal cross-source breadth.

## Reusable Ingredients
Maps to an attention expert in our library, with a cross-layer aggregation idea we can borrow. We extend it from one source to a fused multi-source library.

## Open Questions
Does CLAP's cross-layer attention representation complement a residual trajectory expert, or do they encode overlapping signal?

## Connections
[AUTO-GENERATED from graph/edges.jsonl — do not edit manually]

## Relevance to This Project
A cross-layer and cross-token attention probe, but still a single signal source — a differentiation target. Our fusion treats attention as one expert among many across signal sources rather than the whole detector.
