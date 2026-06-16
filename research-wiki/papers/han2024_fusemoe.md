---
type: paper
node_id: paper:han2024_fusemoe
title: "FuseMoE: Mixture-of-Experts Transformers for Fleximodal Fusion"
authors: ["Han"]
year: 2024
venue: "NeurIPS 2024"
external_ids:
  arxiv: null
  doi: null
  s2: null
tags: ["multimodal", "moe", "fusion-machinery"]
added: 2026-06-16T00:00:00Z
---

# FuseMoE: Mixture-of-Experts Transformers for Fleximodal Fusion

## One-line thesis
A sparse mixture-of-experts transformer fuses an arbitrary, possibly incomplete set of modalities by routing each modality through specialised experts under a Laplace-distribution gating scheme.

## Problem / Gap
Real-world multimodal data are heterogeneous and often partially missing. Dense fusion architectures assume every modality is present and scale poorly as the number of modalities grows.

## Method
Read per-modality token embeddings; route each through a sparse MoE layer whose gating network selects a small subset of experts; the gate uses a Laplace-based formulation argued to give better convergence than softmax gating; output a fused representation robust to missing or irregular modalities.

## Key Results
Reported gains on clinical/time-series multimodal benchmarks with missing modalities; exact numbers not recorded in our notes.

## Assumptions
Each modality maps to a token stream; sparse routing suffices to specialise experts; gating can absorb modality dropout.

## Limitations / Failure Modes
Sparse routing can leave experts under-trained; load-balancing instability; benefits concentrated on heterogeneous/missing-modality regimes.

## Reusable Ingredients
The sparse-MoE layer plus Laplace gating is the fusion machinery we borrow for combining heterogeneous probe "experts" (residual / attention / logit signals treated as modalities).

## Open Questions
Does Laplace gating help when experts are linear probes rather than transformer blocks? How does routing behave with only a dozen experts?

## Claims

## Connections
[AUTO-GENERATED from graph/edges.jsonl — do not edit manually]

## Relevance to This Project
Provides the sparse-MoE + Laplace-gating machinery we adapt to fuse heterogeneous internal-state probe experts, where each probe signal source plays the role of a modality.
