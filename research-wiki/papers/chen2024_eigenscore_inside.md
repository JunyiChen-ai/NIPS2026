---
type: paper
node_id: paper:chen2024_eigenscore_inside
title: "INSIDE: LLMs' Internal States Retain the Power of Hallucination Detection (EigenScore)"
authors: ["Chao Chen"]
year: 2024
venue: "ICLR 2024"
external_ids:
  arxiv: "2402.03744"
  doi: null
  s2: null
tags: ["residual", "multi-sample", "eigen", "unsupervised"]
added: 2026-06-16T00:00:00Z
---

# INSIDE: LLMs' Internal States Retain the Power of Hallucination Detection (EigenScore)

## One-line thesis
The eigen-spectrum of the covariance across multiple sampled responses' internal states gives an unsupervised hallucination score.

## Problem / Gap
Output-space consistency checks miss signal held in representations. The paper argues internal states across resamples expose semantic divergence that surface text hides.

## Method
For one prompt, sample roughly ten generations. Read each generation's middle-layer hidden state. Form the covariance over these response embeddings and compute its eigenvalues; the EigenScore from this spectrum measures internal consistency, with high divergence indicating hallucination. No labels are used.

## Key Results
Reported strong unsupervised detection from the response-covariance eigen-spectrum. No specific numbers recorded here.

## Assumptions
Multiple samples are available per prompt, and semantic divergence shows up as spread in the hidden-state covariance.

## Limitations / Failure Modes
Requires multi-sample generation; with a single sample the covariance and its spectrum degenerate, collapsing the score.

## Reusable Ingredients
NOT fully reproduced. We sampled once, so the EigenScore degrades; faithful reproduction needs nine additional samples per prompt.

## Open Questions
Is there a 1-sample surrogate that recovers part of the multi-sample eigen signal?

## Connections
[AUTO-GENERATED from graph/edges.jsonl — do not edit manually]

## Relevance to This Project
EigenScore marks the multi-sample frontier of our taxonomy. It is a residual eigen expert we cannot yet plug in at full fidelity, flagging multi-sample extraction as a concrete gap in our feature store.
