---
type: paper
node_id: paper:li2023_iti
title: "Inference-Time Intervention: Eliciting Truthful Answers from a Language Model (ITI)"
authors: ["Kenneth Li"]
year: 2023
venue: "NeurIPS 2023 (Spotlight)"
external_ids:
  arxiv: "2306.03341"
  doi: null
  s2: null
tags: ["attention-head", "linear-probe", "steering"]
added: 2026-06-16T00:00:00Z
---

# Inference-Time Intervention: Eliciting Truthful Answers from a Language Model (ITI)

## One-line thesis
A small set of attention heads carries a linear truthfulness direction, and shifting activations along that direction at inference time makes the model answer more truthfully.

## Problem / Gap
Models often know the truthful answer internally yet still produce false statements. The paper seeks a lightweight, training-free way to elicit the truthful behavior already present in representations.

## Method
Read per-head attention output activations. Fit a linear probe (logistic regression) per head to find a truthfulness direction, ranking heads by how well they separate true from false. At inference, add the truthful direction back into the activations of the selected heads to steer generation. For detection, the per-head probe scores are the signal; the steering step is a separate use.

## Key Results
Reported improved truthfulness on TruthfulQA with minimal intervention. No specific numbers recorded here.

## Assumptions
Truthfulness is approximately linearly encoded in a sparse subset of attention heads.

## Limitations / Failure Modes
Steering strength and head selection are sensitive hyperparameters; the truth direction is dataset-dependent.

## Reusable Ingredients
Maps to our `iti` probe. Our `input_per_head_activation` matches the read口径; a plain LR over per-head outputs gives the per-head truth expert.

## Open Questions
How transferable is the head set across tasks, and does the detection signal survive when steering is removed?

## Connections
[AUTO-GENERATED from graph/edges.jsonl — do not edit manually]

## Relevance to This Project
ITI supplies the per-head truth expert for our fusion library. We reuse only its detection side: linear probes over per-head attention outputs, one of the attention-signal experts in the stack.
