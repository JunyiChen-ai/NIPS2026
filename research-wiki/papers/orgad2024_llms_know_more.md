---
type: paper
node_id: paper:orgad2024_llms_know_more
title: "LLMs Know More Than They Show: On the Intrinsic Representation of LLM Hallucinations"
authors: ["Hadas Orgad"]
year: 2024
venue: "arXiv preprint (ICLR 2025)"
external_ids:
  arxiv: "2410.02707"
  doi: null
  s2: null
tags: ["token-position", "exact-answer-tokens", "probing"]
added: 2026-06-16T00:00:00Z
---

# LLMs Know More Than They Show: On the Intrinsic Representation of LLM Hallucinations

## One-line thesis
Truthfulness signal concentrates on the exact answer tokens, so probing the wrong token position discards the signal.

## Problem / Gap
Probes usually read at prompt-last or generation-last positions. This work shows that choice can miss where the truthfulness signal actually lives.

## Method
Read-what: internal representations at specific token positions, focusing on the exact answer tokens. Compute-what: probe these positions for truthfulness. Output-what: evidence that signal localizes to answer-specific tokens and that position selection matters.

## Key Results
Qualitative finding reported as authoritative: truthfulness encoding concentrates on exact-answer tokens; mislocating the probe loses signal. No specific numbers invented here.

## Assumptions
There exist identifiable exact-answer token positions where intrinsic truthfulness signal is strongest.

## Limitations / Failure Modes
Requires locating answer-specific tokens, which is itself nontrivial outside clean QA. Focuses on where to read more than on fusing sources.

## Reusable Ingredients
Motivates query-token position as a first-class axis (axis 2 in our taxonomy: prompt-last / gen-last / answer-specific). Justifies treating position as a tunable variable in our scaffold.

## Open Questions
Can answer-specific token positions be located reliably enough to use as a fusion expert input at scale?

## Connections
[AUTO-GENERATED from graph/edges.jsonl — do not edit manually]

## Relevance to This Project
Truthfulness signal concentrates on exact-answer tokens, so the wrong position loses it. This motivates making query-token position (axis 2) a first-class variable in our fusion design rather than fixing it at prompt-last or gen-last.
