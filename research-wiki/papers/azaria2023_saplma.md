---
type: paper
node_id: paper:azaria2023_saplma
title: "The Internal State of an LLM Knows When It's Lying (SAPLMA)"
authors: ["Amos Azaria", "Tom Mitchell"]
year: 2023
venue: "Findings of EMNLP 2023"
external_ids:
  arxiv: "2304.13734"
  doi: null
  s2: null
tags: ["residual", "mlp-probe", "correctness"]
added: 2026-06-16T00:00:00Z
---

# The Internal State of an LLM Knows When It's Lying (SAPLMA)

## One-line thesis
A small classifier on an LLM's mid-layer hidden state predicts whether a statement is true before the model generates anything.

## Problem / Gap
LLMs assert false statements confidently. The paper asks whether the model's own internal activations already separate true from false, giving a cheap pre-generation truthfulness signal.

## Method
Read the hidden state at the last prompt token from a middle layer. Reduce dimensionality (PCA) and fit a classifier (logistic regression / small MLP) to the true/false label. Output a correctness probability for the statement.

## Key Results
The internal-state classifier predicts statement truthfulness well above chance across topic-specific datasets, supporting the "internal state knows" claim. (Qualitative; no numbers asserted here.)

## Assumptions
A single mid layer is informative; the last prompt token aggregates enough context; truth is approximately linearly/PCA-recoverable.

## Limitations / Failure Modes
Trained per topic, transfer across domains is uneven. The probe captures the model's belief, not external truth, so confidently-wrong beliefs fool it.

## Reusable Ingredients
Maps to our `pca_lr` expert: PCA followed by LR on mid-layer `input_last_token_hidden`.

## Open Questions
How much does PCA help versus full-dimensional LR? Does the signal hold for generated answers rather than given statements?

## Connections
[AUTO-GENERATED from graph/edges.jsonl — do not edit manually]

## Relevance to This Project
`input_last_token_hidden` is sufficient. In our fusion it is the "pre-generation correctness expert."
