---
type: paper
node_id: paper:yun2024_flex_moe
title: "Flex-MoE: Modeling Arbitrary Modality Combination via the Flexible Mixture-of-Experts"
authors: ["Yun"]
year: 2024
venue: "NeurIPS 2024 (Spotlight)"
external_ids:
  arxiv: null
  doi: null
  s2: null
tags: ["multimodal", "moe", "fusion-machinery"]
added: 2026-06-16T00:00:00Z
---

# Flex-MoE: Modeling Arbitrary Modality Combination via the Flexible Mixture-of-Experts

## One-line thesis
A flexible mixture-of-experts handles any subset of available modalities by combining a generalised router with modality-specialised routing, so the model degrades gracefully when modalities are absent.

## Problem / Gap
Most multimodal models train on a fixed full set of modalities and break when only an arbitrary subset is present at inference. The combinatorial space of modality combinations is rarely modelled explicitly.

## Method
Read per-modality embeddings; a dual routing scheme pairs a generalised (shared) router that handles any combination with modality-specific routing that injects specialised knowledge; missing modalities are imputed or bypassed; output a fused embedding for the present combination.

## Key Results
Reported state-of-the-art on multimodal benchmarks (e.g. Alzheimer's / ADNI-style data) across many missing-modality combinations; exact numbers not recorded in our notes.

## Assumptions
A shared router can cover unseen combinations; modality-specific experts add value beyond the shared path.

## Limitations / Failure Modes
Imputation of missing modalities can inject noise; scaling to very many modalities stresses the combinatorial router.

## Reusable Ingredients
The dual-router design (shared + modality-specific) is the machinery we borrow for arbitrary probe combinations: any subset of our 12 probes can be routed without retraining a fixed fusion head.

## Open Questions
Does the shared/specific split help when "modalities" are probe outputs that already share a backbone? How small can the specialised experts be?

## Connections
[AUTO-GENERATED from graph/edges.jsonl — do not edit manually]

## Relevance to This Project
Gives a concrete dual-router recipe for fusing arbitrary combinations of probe experts, directly supporting our goal of a single fusion weight over a flexible subset of the 12 probes.
