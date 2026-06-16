---
type: paper
node_id: paper:unifact2025
title: "UniFact: Towards Unification of Hallucination Detection and Fact Verification"
authors: ["(preprint)"]
year: 2025
venue: "arXiv preprint (2025-12)"
external_ids:
  arxiv: "2512.02772"
  doi: null
  s2: null
tags: ["benchmark", "unification", "hybrid"]
added: 2026-06-16T00:00:00Z
---

# UniFact: Towards Unification of Hallucination Detection and Fact Verification

## One-line thesis
UniFact is an evaluation framework uniting hallucination detection and fact verification, and its headline finding is that no single paradigm dominates while hybrid approaches stay strongest.

## Problem / Gap
Hallucination detection (HD) and fact verification (FV) are studied separately with incompatible protocols. UniFact builds a shared evaluation surface to compare paradigms head to head.

## Method
Read-what: detectors and verifiers spanning retrieval-, uncertainty-, embedding-, and learning-based paradigms. Compute-what: evaluate them under a unified protocol across the HD-FV union. Output-what: comparative performance showing where each paradigm wins and that hybrids consistently lead.

## Key Results
Qualitative conclusion reported as authoritative: no paradigm dominates across settings; hybrid combinations are consistently strongest. No specific numbers invented here.

## Assumptions
HD and FV are close enough to share an evaluation framework; the paradigm taxonomy covers the relevant detector families.

## Limitations / Failure Modes
A benchmark, not a model — it diagnoses the landscape rather than proposing a fused detector.

## Reusable Ingredients
Provides the empirical justification for building a fusion model: if no single paradigm wins and hybrid is strongest, a learned cross-source fusion is the natural next step.

## Open Questions
Which paradigm pairs are most complementary, and does the hybrid advantage hold across model families?

## Connections
[AUTO-GENERATED from graph/edges.jsonl — do not edit manually]

## Relevance to This Project
UniFact is the direct empirical justification for our thesis: it is not a model but an evaluation framework whose conclusion is that no paradigm dominates and hybrid is always strongest. That is the experience base motivating our multi-view expert-library stacking.
