---
type: paper
node_id: paper:marks2024_geometry_of_truth
title: "The Geometry of Truth: Emergent Linear Structure in LLM Representations of True/False Datasets"
authors: ["Samuel Marks", "Max Tegmark"]
year: 2024
venue: "COLM 2024 (Spotlight)"
external_ids:
  arxiv: "2310.06824"
  doi: null
  s2: null
tags: ["residual", "linear-probe", "truthfulness"]
added: 2026-06-16T00:00:00Z
---

# The Geometry of Truth: Emergent Linear Structure in LLM Representations of True/False Datasets

## One-line thesis
The truth value of a simple factual statement is encoded as a linear direction in an LLM's mid-layer residual stream, recoverable with a single linear probe.

## Problem / Gap
Whether LLMs represent the truth of declarative statements in a structured, accessible way was unclear. The paper tests whether true/false is linearly separable in hidden activations.

## Method
Read the mid-layer residual activation at the last token of a true/false statement. Compute a truth direction either by mass-mean (difference of class means) or by logistic regression on these activations. Output a scalar truth score by projecting onto that direction.

## Key Results
Linear probes separate true from false statements with high accuracy, and the recovered directions generalize across related true/false datasets. (Qualitative; no specific numbers asserted here.)

## Assumptions
Statements are short, declarative, and have a clean true/false label; a single mid layer carries the signal; truth is approximately linearly encoded.

## Limitations / Failure Modes
Curated simple statements may not transfer to open generation or multi-hop claims. Probe directions can entangle topic or surface features rather than truth itself.

## Reusable Ingredients
Maps directly to our `lr_probe` and `mm_probe` (mass-mean) experts over `input_last_token_hidden`. The canonical residual-LR expert in the fusion library.

## Open Questions
Does the truth direction survive paraphrase, negation, and longer generations? How stable is it across model families?

## Claims

## Connections
[AUTO-GENERATED from graph/edges.jsonl — do not edit manually]

## Relevance to This Project
`input_last_token_hidden` fully covers this signal; a plain LR suffices. In our fusion it is the representative "residual LR expert."
