---
type: paper
node_id: paper:du2024_haloscope
title: "HaloScope: Harnessing Unlabeled LLM Generations for Hallucination Detection"
authors: ["Xuefeng Du"]
year: 2024
venue: "NeurIPS 2024 (Spotlight)"
external_ids:
  arxiv: "2409.17504"
  doi: null
  s2: null
tags: ["residual", "unsupervised", "membership"]
added: 2026-06-16T00:00:00Z
---

# HaloScope: Harnessing Unlabeled LLM Generations for Hallucination Detection

## One-line thesis
Unlabeled model generations can be automatically split into likely-truthful and likely-hallucinated groups, yielding a membership-style estimator that trains a hallucination detector without human labels.

## Problem / Gap
Supervised hallucination detectors need labeled truth/hallucination data that is costly to collect. HaloScope asks whether unlabeled generations alone can supervise a detector.

## Method
Read hidden-state representations of unlabeled generations. Estimate a truthful subspace from the representation distribution and use it to assign each unlabeled sample a membership score (likely truthful vs likely hallucinated). Use these inferred memberships as pseudo-labels to train a binary detector over the representations.

## Key Results
Reported that an unlabeled-data detector approaches supervised performance. No specific numbers recorded here.

## Assumptions
Truthful and hallucinated generations separate in representation space well enough for an unsupervised membership estimate to act as a label.

## Limitations / Failure Modes
Depends on the quality of the unsupervised split; a poor subspace estimate corrupts the pseudo-labels.

## Reusable Ingredients
Maps to our `haloscope` probe — the newest probe being plugged into our scaffold (Qwen extraction done; Llama and Mistral pending).

## Open Questions
How stable is the truthful-subspace estimate across model families, and does it transfer to our other datasets?

## Connections
[AUTO-GENERATED from graph/edges.jsonl — do not edit manually]

## Relevance to This Project
HaloScope is the latest expert we are integrating into the fusion scaffold, a residual unsupervised-membership source that tests how cleanly a brand-new probe drops into our expert library.
