---
type: paper
node_id: paper:wang2025_chain_of_embedding
title: "Latent Space Chain-of-Embedding Enables Output-free LLM Self-Evaluation"
authors: ["Yiming Wang"]
year: 2025
venue: "ICLR 2025"
external_ids:
  arxiv: "2410.13640"
  doi: null
  s2: null
tags: ["trajectory", "generation-side", "unsupervised"]
added: 2026-06-16T00:00:00Z
---

# Latent Space Chain-of-Embedding Enables Output-free LLM Self-Evaluation

## One-line thesis
The layer-wise trajectory of a token's hidden state encodes self-evaluation signal that needs no output text or labels.

## Problem / Gap
Self-evaluation usually relies on output tokens or sampling. The paper asks whether the per-layer evolution of a single hidden state ("chain of embedding") alone reveals answer quality.

## Method
Read the full-layer hidden states at a fixed token position, treated as a trajectory. Compute per-layer magnitude and angle statistics, summarized into a few scalar features. Output an output-free self-evaluation score from those statistics.

## Key Results
The trajectory statistics support output-free self-evaluation competitive with output-based methods. (Qualitative; no numbers asserted here.)

## Assumptions
A fixed token position captures the relevant computation; magnitude/angle dynamics across layers track correctness; original work targets binary AUROC.

## Limitations / Failure Modes
Hand-designed trajectory statistics may miss higher-order structure. Binary framing limits multi-class use without extension.

## Reusable Ingredients
Maps to our `coe` probe over `gen_mean_pool_hidden`. The original yields binary AUROC; we can extend its four scalars into a multi-class LR.

## Open Questions
Which trajectory statistics matter most? Does a learned aggregator beat the hand-designed magnitude/angle features?

## Connections
[AUTO-GENERATED from graph/edges.jsonl — do not edit manually]

## Relevance to This Project
`gen_mean_pool_hidden` is sufficient. In our fusion it is the "trajectory expert"; since the original only does binary AUROC, we can extend it to multi-class LR.
