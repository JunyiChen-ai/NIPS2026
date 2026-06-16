---
type: paper
node_id: paper:ghadiri2025_gnosis
title: "Gnosis: A Trained Stop-Head for Hallucination Detection"
authors: ["Ghadiri", "Niu"]
year: 2025
venue: "arXiv preprint (2025-12)"
external_ids:
  arxiv: "2512.20578"
  doi: null
  s2: null
tags: ["fusion", "attention", "residual", "learned", "backbone-bound"]
added: 2026-06-16T00:00:00Z
---

# Gnosis: A Trained Stop-Head for Hallucination Detection

## One-line thesis
Gnosis adds a trained `_should_stop` head onto a frozen backbone that fuses raw attention, last-layer hidden state, and token probabilities into a single correctness score.

## Problem / Gap
Most detectors read one signal source. Gnosis tries learned fusion across two internal sources, but stays tied to a single backbone and a single label.

## Method
Read-what: full raw attention maps, last-layer hidden state, and per-token probabilities from a frozen backbone. Compute-what: pass attention through an FFT plus CNN stack, combine with the hidden state and probabilities in a small learned head. Output-what: a scalar correctness / stop signal.

## Key Results
Not extracted here; treat as a learned-fusion prior rather than a benchmark anchor.

## Assumptions
Raw attention tensors are available at inference; the head can be co-trained with (or attached to) the backbone it serves.

## Limitations / Failure Modes
Only two signal sources (attention + residual). Head is backbone-bound, so it does not transfer across model families. Supervision is correctness-only. Raw attention maps are large and were not stored in our extraction.

## Reusable Ingredients
Maps to the closest LEARNED-fusion prior work for our scaffold and is our differentiation target: we span more signal sources, are not backbone-bound, and treat query-token position as a first-class axis.

## Open Questions
Can the FFT-plus-CNN attention encoder be replicated from stored scalar attention statistics, or does it strictly require raw maps?

## Claims

## Connections
[AUTO-GENERATED from graph/edges.jsonl — do not edit manually]

## Relevance to This Project
Gnosis is learned fusion, but only 2 sources, bound to the backbone, and labeled with correctness only; reproducing it would require raw attention maps (too large to store) and retraining the head on Qwen2.5-7B. It marks the prior boundary our multi-view expert-library stacking aims to push past.
