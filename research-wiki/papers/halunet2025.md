---
type: paper
node_id: paper:halunet2025
title: "HaluNet: Learned Fusion of Output-side Hallucination Signals"
authors: ["(preprint)"]
year: 2025
venue: "arXiv preprint (2025-12)"
external_ids:
  arxiv: "2512.24562"
  doi: null
  s2: null
tags: ["fusion", "output-side", "learned"]
added: 2026-06-16T00:00:00Z
---

# HaluNet: Learned Fusion of Output-side Hallucination Signals

## One-line thesis
HaluNet learns to fuse three output-side signals — semantic embedding, log-probability, and entropy — into one hallucination detector.

## Problem / Gap
Single output-side cues each miss part of the picture. HaluNet shows learned fusion of several output-side signals beats any one alone, but never reaches into the model's internal state.

## Method
Read-what: semantic embedding of the response, token log-probabilities, and predictive entropy. Compute-what: a learned fusion module combines the three signal paths. Output-what: a hallucination score on QA outputs.

## Key Results
Not extracted here; the contribution of interest is the learned multi-signal fusion design, not a specific number.

## Assumptions
Output-side signals (embedding, logprob, entropy) suffice for the target QA setting; access to internal activations is not required.

## Limitations / Failure Modes
All three signals are output-side. It never touches attention, SAE latents, or cross-layer trajectories. Evaluation is QA-only.

## Reusable Ingredients
Maps to learned fusion in our scaffold, but with an all-output-side signal set and QA-only scope; it is a differentiation target. Our fusion adds internal-state experts (residual, attention, MLP, SAE, multi-sample) on top of output-side cues.

## Open Questions
Does adding internal-state experts to HaluNet's three output-side signals yield orthogonal gains, or do the signals overlap?

## Connections
[AUTO-GENERATED from graph/edges.jsonl — do not edit manually]

## Relevance to This Project
HaluNet is learned fusion, but every signal is output-side (embedding + logprob + entropy) and it is evaluated on QA only. It is a direct differentiation target: our work fuses internal-state signal sources that HaluNet leaves untouched and tests beyond QA.
