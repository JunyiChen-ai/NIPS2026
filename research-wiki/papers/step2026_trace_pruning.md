---
type: paper
node_id: paper:step2026_trace_pruning
title: "STEP: Step-level Trace Evaluation and Pruning"
authors: ["(preprint)"]
year: 2026
venue: "arXiv preprint"
external_ids:
  arxiv: "2601.09093"
  doi: null
  s2: null
tags: ["trajectory", "step-level", "generation-side"]
added: 2026-06-16T00:00:00Z
---

# STEP: Step-level Trace Evaluation and Pruning

## One-line thesis
Scoring the hidden state at each chain-of-thought step boundary lets a lightweight probe prune low-quality reasoning traces online.

## Problem / Gap
Chain-of-thought generation can drift into low-quality reasoning. The paper asks whether per-step internal states can flag bad steps early enough to prune them during decoding.

## Method
Read the hidden state at each `\n\n` step boundary during CoT generation. Pass it through a lightweight MLP scorer to rate step quality. Use the scores online to prune low-quality traces before they consume more compute.

## Key Results
Reported online pruning of weak reasoning traces from step-level scores. No specific numbers recorded here.

## Assumptions
Step boundaries are detectable from delimiter tokens, and per-step hidden states carry quality-relevant signal.

## Limitations / Failure Modes
Granularity is step-level, not sample-level; using it for sample-level classification needs an aggregation step.

## Reusable Ingredients
Maps to our `step` probe. Our `gen_step_boundary_hidden` covers the read, but we must aggregate step-level scores up to a sample-level decision.

## Open Questions
What aggregation over step scores best preserves the sample-level signal, and does step granularity add information beyond gen-last probes?

## Connections
[AUTO-GENERATED from graph/edges.jsonl — do not edit manually]

## Relevance to This Project
STEP is the generation-side trajectory expert in our library, the one source that reads intermediate reasoning steps rather than a single fixed token, broadening the query-token-position axis of our fusion.
