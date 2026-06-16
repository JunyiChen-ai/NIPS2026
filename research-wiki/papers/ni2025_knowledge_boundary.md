---
type: paper
node_id: paper:ni2025_knowledge_boundary
title: "LLM Knowledge-Boundary Perception"
authors: ["Ni"]
year: 2025
venue: "ACL 2025 Main"
external_ids:
  arxiv: "2502.11677"
  doi: null
  s2: null
tags: ["residual", "mlp-probe", "knowledge-boundary"]
added: 2026-06-16T00:00:00Z
---

# LLM Knowledge-Boundary Perception

## One-line thesis
An MLP over generation-side hidden states predicts whether a question falls inside or outside the model's knowledge boundary.

## Problem / Gap
LLMs answer beyond what they reliably know. The paper probes whether internal states perceive the boundary of the model's knowledge, with token position and layer treated as design choices.

## Method
Read the hidden state of generated tokens at a selectable position (first / last / average / min-probability) and layer (mid / last / all). Feed it to an MLP. Output a knowledge-boundary score indicating whether the model knows the answer.

## Key Results
The MLP probe detects knowledge-boundary cases across position/layer configurations, with the best settings outperforming naive baselines. (Qualitative; no numbers asserted here.)

## Assumptions
Knowledge-boundary status is recoverable from generation-side hidden states; the chosen token position and layer carry the signal.

## Limitations / Failure Modes
Position/layer choice strongly affects performance, requiring tuning. The boundary label may be noisy and dataset-dependent.

## Reusable Ingredients
Maps to our `kb_mlp` probe over `gen_last_token_hidden` plus `gen_mean_pool_hidden`. The current project's "KB MLP" baseline.

## Open Questions
Which position/layer combination generalizes across tasks? Does the MLP beat a linear probe on the same features?

## Connections
[AUTO-GENERATED from graph/edges.jsonl — do not edit manually]

## Relevance to This Project
`gen_last_token_hidden` and `gen_mean_pool_hidden` are sufficient; this is our "KB MLP" baseline, and its position/layer axis directly motivates treating query-token position as a first-class fusion variable.
