---
type: paper
node_id: paper:yao2025_seakr
title: "SeaKR: Self-aware Knowledge Retrieval for Adaptive Retrieval-Augmented Generation"
authors: ["Zijun Yao"]
year: 2025
venue: "ACL 2025 (Oral)"
external_ids:
  arxiv: "2406.19215"
  doi: null
  s2: null
tags: ["multi-sample", "eigen", "routing"]
added: 2026-06-16T00:00:00Z
---

# SeaKR: Self-aware Knowledge Retrieval for Adaptive Retrieval-Augmented Generation

## One-line thesis
A model's own internal uncertainty, read from multi-sample FFN activations, decides when to retrieve external knowledge and how to rerank and reason over it.

## Problem / Gap
Retrieval-augmented generation retrieves indiscriminately, wasting calls when the model already knows the answer. SeaKR asks the model to gate retrieval on its self-aware uncertainty.

## Method
Sample multiple generations and read their FFN internal activations. Compute an eigen-score capturing internal consistency. Threshold the score to trigger RAG retrieval, then use it again to rerank retrieved snippets and select a reasoning strategy. The internal signal acts as a routing gate rather than a standalone detector.

## Key Results
Reported gains on knowledge-intensive QA by retrieving adaptively. No specific numbers recorded here.

## Assumptions
Multiple samples per query are available, and FFN-activation eigen-scores track when external knowledge is needed.

## Limitations / Failure Modes
Multi-sampling is costly; with one sample the eigen-score degrades to a SeaKR-inspired surrogate rather than the original signal.

## Reusable Ingredients
Maps to our `seakr` probe, degraded to a 1-sample energy score. Faithful reproduction needs roughly 10x sampling to recover the eigen口径.

## Open Questions
Can the 1-sample energy surrogate retain useful routing signal, or is multi-sampling essential here too?

## Connections
[AUTO-GENERATED from graph/edges.jsonl — do not edit manually]

## Relevance to This Project
SeaKR contributes the routing/decision-application expert and a second multi-sample eigen source. In our 1-sample setting it degrades, marking it alongside EigenScore as a multi-sample gap in our extraction.
