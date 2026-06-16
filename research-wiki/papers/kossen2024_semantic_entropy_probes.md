---
type: paper
node_id: paper:kossen2024_semantic_entropy_probes
title: "Semantic Entropy Probes: Robust and Cheap Hallucination Detection in LLMs"
authors: ["Jannik Kossen"]
year: 2024
venue: "NeurIPS 2024 SafeGenAI Workshop"
external_ids:
  arxiv: "2406.15927"
  doi: null
  s2: null
tags: ["residual", "generation-side", "semantic-entropy"]
added: 2026-06-16T00:00:00Z
---

# Semantic Entropy Probes: Robust and Cheap Hallucination Detection in LLMs

## One-line thesis
A linear probe on a single generation's hidden state approximates semantic entropy, normally requiring many samples, giving cheap hallucination detection.

## Problem / Gap
Semantic entropy detects hallucination well but needs repeated sampling and clustering, which is expensive. The paper seeks a one-forward-pass surrogate.

## Method
Read the hidden state at the last generated token. Fit a logistic regression to predict the (high/low) semantic-entropy label derived from multi-sample reference computations. Output an entropy-proxy uncertainty score from a single generation.

## Key Results
The single-pass probe recovers much of multi-sample semantic entropy's hallucination-detection signal at a fraction of the cost. (Qualitative; no numbers asserted here.)

## Assumptions
Semantic entropy is linearly readable from one generation's hidden state; training labels from full multi-sample semantic entropy are available.

## Limitations / Failure Modes
A proxy can diverge from true semantic entropy when generations are atypical. Probe quality depends on the label source it was distilled from.

## Reusable Ingredients
Maps to our `sep` probe over `gen_last_token_hidden`.

## Open Questions
How well does the proxy hold under distribution shift versus recomputing entropy? Which layer best captures it?

## Connections
[AUTO-GENERATED from graph/edges.jsonl — do not edit manually]

## Relevance to This Project
`gen_last_token_hidden` is sufficient. In our fusion it is the "post-generation uncertainty expert."
