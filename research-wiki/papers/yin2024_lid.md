---
type: paper
node_id: paper:yin2024_lid
title: "Characterizing Truthfulness in LLM Generations with Local Intrinsic Dimension"
authors: ["Fan Yin"]
year: 2024
venue: "ICML 2024"
external_ids:
  arxiv: "2402.18048"
  doi: null
  s2: null
tags: ["residual", "geometry", "unsupervised"]
added: 2026-06-16T00:00:00Z
---

# Characterizing Truthfulness in LLM Generations with Local Intrinsic Dimension

## One-line thesis
The local intrinsic dimension of a generation's hidden-state manifold is an unsupervised signal of truthfulness.

## Problem / Gap
Most truthfulness probes are supervised and linear. The paper looks for a label-free geometric property of activations that tracks whether a generation is truthful.

## Method
Read the hidden states across all layers at the last generated token. Estimate the local intrinsic dimension via an MLE estimator over neighborhoods in activation space. Output a scalar truthfulness score from that dimension estimate.

## Key Results
Local intrinsic dimension correlates with truthfulness and competes with supervised baselines without using labels. (Qualitative; no numbers asserted here.)

## Assumptions
Truthful and untruthful generations occupy manifolds of differing local dimension; neighborhood estimation is stable at the chosen token/layers.

## Limitations / Failure Modes
Intrinsic-dimension estimates are sensitive to neighborhood size and sample count. Being unsupervised, it cannot be tuned to a specific task label.

## Reusable Ingredients
Maps to our `lid` probe: a nonlinear, manifold-geometry aggregator over `gen_last_token_hidden` (all layers).

## Open Questions
How does estimator choice affect stability? Can the dimension signal be combined with linear probes additively?

## Connections
[AUTO-GENERATED from graph/edges.jsonl — do not edit manually]

## Relevance to This Project
`gen_last_token_hidden` is sufficient. In our fusion it is the "manifold-geometry expert," a nonlinear aggregator distinct from the linear probes.
