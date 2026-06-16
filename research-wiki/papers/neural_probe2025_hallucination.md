---
type: paper
node_id: paper:neural_probe2025_hallucination
title: "Neural Probe-Based Hallucination Detection across Tasks"
authors: ["(preprint)"]
year: 2025
venue: "arXiv preprint (2025-12)"
external_ids:
  arxiv: "2512.20949"
  doi: null
  s2: null
tags: ["residual", "neural-probe", "cross-task"]
added: 2026-06-16T00:00:00Z
---

# Neural Probe-Based Hallucination Detection across Tasks

## One-line thesis
A neural probe over residual representations detects hallucinations and transfers across tasks, but reads from a single signal source.

## Problem / Gap
Many probes are tuned to one task. This work shows a neural probe on internal representations can generalize across tasks, yet it still draws from one signal source.

## Method
Read-what: residual / hidden-state representations from the backbone. Compute-what: a neural probe maps them to a hallucination score. Output-what: a cross-task detection signal.

## Key Results
Not extracted here; the relevant point is cross-task transfer from a single-source neural probe.

## Assumptions
A residual-only probe carries enough signal to transfer across tasks.

## Limitations / Failure Modes
Single signal source (residual). No attention, SAE, logit, or multi-sample fusion. Cross-task breadth does not imply cross-source breadth.

## Reusable Ingredients
Maps to a single residual expert in our library. Its cross-task framing supports our cross-task evaluation goal; we extend it from one source to a fused multi-source library.

## Open Questions
Would fusing attention and logit experts onto this residual probe improve cross-task robustness?

## Connections
[AUTO-GENERATED from graph/edges.jsonl — do not edit manually]

## Relevance to This Project
A cross-task neural probe, but built on a single signal source — a differentiation target. Our scaffold keeps the cross-task ambition while fusing heterogeneous experts across signal sources rather than relying on residual alone.
