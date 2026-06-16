---
type: paper
node_id: paper:hallucination_survey2025
title: "A Survey of Hallucination Detection in Large Language Models"
authors: ["(survey)"]
year: 2025
venue: "arXiv (2025-10)"
external_ids:
  arxiv: "2510.06265"
  doi: null
  s2: null
tags: ["survey", "hallucination-detection"]
added: 2026-06-16T00:00:00Z
---

# A Survey of Hallucination Detection in Large Language Models

## One-line thesis
A survey that taxonomises LLM hallucination detection into five method families, giving a shared map of how internal and external signals are used to flag unfaithful generations.

## Problem / Gap
Hallucination-detection methods had proliferated across retrieval, uncertainty, and internal-state signals without a unifying organisation, making method comparison hard.

## Method
Read the hallucination-detection literature; sort methods by their signal into five families — retrieval-based, uncertainty-based, embedding-based, learning-based, and self-consistency-based; output a taxonomy and discussion; no new benchmark.

## Key Results
A five-family taxonomy and qualitative discussion; no empirical numbers to record.

## Assumptions
Detection methods cleanly fall into the five families; the families are roughly exhaustive of current practice.

## Limitations / Failure Modes
Boundaries between families blur for hybrid methods; surveys date quickly; no controlled comparison across families.

## Reusable Ingredients
Its "embedding-based" family corresponds to our residual + attention probes (LITERATURE.md §2.1 + §2.2); the taxonomy lets us position fusion as spanning multiple families at once.

## Open Questions
Does a fusion across all five families beat the best single family? Where do SAE and multi-sample probes land in this scheme?

## Connections
[AUTO-GENERATED from graph/edges.jsonl — do not edit manually]

## Relevance to This Project
It taxonomises detection into five families, and our residual + attention probes map to its embedding-based family — the survey frames where our fused detector sits and what families it should reach across.
