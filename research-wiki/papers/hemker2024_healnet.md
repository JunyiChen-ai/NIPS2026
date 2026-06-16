---
type: paper
node_id: paper:hemker2024_healnet
title: "HEALNet: Multimodal Fusion for Heterogeneous Biomedical Data"
authors: ["Hemker"]
year: 2024
venue: "NeurIPS 2024"
external_ids:
  arxiv: null
  doi: null
  s2: null
tags: ["multimodal", "cross-attention", "fusion-machinery"]
added: 2026-06-16T00:00:00Z
---

# HEALNet: Multimodal Fusion for Heterogeneous Biomedical Data

## One-line thesis
A hybrid early-fusion network uses a shared latent bottleneck with cross-attention to integrate structurally heterogeneous biomedical modalities while preserving modality-specific structure and tolerating missing inputs.

## Problem / Gap
Biomedical modalities (e.g. omics tables and whole-slide images) differ in structure, so naive concatenation loses signal and intermediate fusion discards modality-specific geometry. Missing modalities are common.

## Method
Read each modality into a shared latent array via cross-attention (Perceiver-style iterative attention); the shared latent both captures cross-modal interactions and retains modality-specific information; output a fused representation usable when some modalities are absent at inference.

## Key Results
Reported strong survival-prediction performance on multimodal cancer datasets and robustness to missing modalities; exact numbers not recorded in our notes.

## Assumptions
A shared latent bottleneck can absorb heterogeneous structure; cross-attention is enough to align modalities of differing dimensionality.

## Limitations / Failure Modes
Modality collapse — one dominant modality can crowd out weaker ones in the shared latent; cross-attention cost grows with latent size.

## Reusable Ingredients
The hybrid early-fusion-with-cross-attention pattern is fusion machinery we borrow; modality collapse is the explicit failure mode we cite as a risk when one probe signal dominates the fusion.

## Open Questions
How to detect and prevent collapse when experts are probes of unequal signal strength? Does a shared latent help over a simple learned gate for low-dimensional probe outputs?

## Connections
[AUTO-GENERATED from graph/edges.jsonl — do not edit manually]

## Relevance to This Project
Supplies the cross-attention early-fusion machinery and names the modality-collapse risk we must guard against when fusing heterogeneous internal-state probes.
