---
type: paper
node_id: paper:yuksekgonul2024_sat_probe
title: "Attention Satisfies: A Constraint-Satisfaction Lens on Factual Errors of LLMs (SAT Probe)"
authors: ["Mert Yuksekgonul"]
year: 2024
venue: "ICLR 2024"
external_ids:
  arxiv: "2309.15098"
  doi: null
  s2: null
tags: ["attention", "constraint", "linear-probe"]
added: 2026-06-16T00:00:00Z
---

# Attention Satisfies: A Constraint-Satisfaction Lens on Factual Errors of LLMs (SAT Probe)

## One-line thesis
Factual queries can be cast as constraint-satisfaction problems, and how attention flows from the prompt's last token back to constraint tokens predicts whether the model will satisfy each constraint.

## Problem / Gap
Factual errors are hard to anticipate from text alone. The paper asks whether the model's own attention pattern over the constraint span already signals an impending error before generation.

## Method
Read the per-head attention from the prompt's last token back over the constraint span. Compute, per head, the attention-weighted value-output norm, roughly the magnitude pushed through the head toward the constrained tokens. Flatten these per-head quantities into a feature vector and fit a linear probe (logistic regression) to predict whether the constraint is satisfied.

## Key Results
Reported that an attention-flow signal predicts constraint satisfaction and tracks factual correctness across several model families. No specific numbers recorded here.

## Assumptions
A factual query decomposes into identifiable constraints with locatable token spans; attention toward those spans carries the relevant signal.

## Limitations / Failure Modes
Needs reliable constraint-span localization; weakens when constraints are implicit or span boundaries are ambiguous.

## Reusable Ingredients
Maps to our `attn_satisfies` probe. Our `input_attn_value_norms` covers most of the signal; we lack the W_o output projection and explicit constraint-span localization.

## Open Questions
How robust is span localization across tasks, and can the projection step be approximated from stored statistics?

## Claims

## Connections
[AUTO-GENERATED from graph/edges.jsonl — do not edit manually]

## Relevance to This Project
This is the attention-flow expert in our fusion library: a residual-orthogonal signal source that complements hidden-state probes and supplies the attention view our stacking design wants to combine.
