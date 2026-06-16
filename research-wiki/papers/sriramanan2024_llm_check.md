---
type: paper
node_id: paper:sriramanan2024_llm_check
title: "LLM-Check: Investigating Detection of Hallucinations in Large Language Models"
authors: ["Gaurang Sriramanan"]
year: 2024
venue: "NeurIPS 2024"
external_ids:
  arxiv: null
  doi: null
  s2: null
tags: ["multi-source", "attention", "residual", "logit", "unsupervised"]
added: 2026-06-16T00:00:00Z
---

# LLM-Check: Investigating Detection of Hallucinations in Large Language Models

## One-line thesis
Three independent internal signals — attention eigen-spectra, residual SVD, and logit-based perplexity/entropy — each detect hallucinations without any learned fusion across them.

## Problem / Gap
Hallucination detection often relies on a single signal or external sampling. The paper surveys complementary internal signals and asks how far each goes on its own.

## Method
Run three parallel readouts. (1) Attention: eigen-spectrum of the attention kernel. (2) Residual: centered SVD of hidden states. (3) Logit: perplexity plus logit entropy over tokens. Each path produces its own score independently; the paper deliberately does not learn a fusion that combines the three.

## Key Results
Reported competitive unsupervised detection from each source. No specific numbers recorded here.

## Assumptions
Each signal source independently carries hallucination-relevant structure; no labels are needed.

## Limitations / Failure Modes
No learned combination means the breadth of sources is not converted into joint discriminative power; perplexity needs full-sequence logits.

## Reusable Ingredients
Maps to our `llm_check` reference. We partially reproduce all three: `input_attn_stats[...,2]` (diag_logmean), a generation last-layer SVD over `gen_last_token_hidden`, and `gen_logit_stats_last.entropy`. PPL degrades because we store only last-token logits.

## Open Questions
How much accuracy is left on the table by not learning a fusion over the three sources?

## Connections
[AUTO-GENERATED from graph/edges.jsonl — do not edit manually]

## Relevance to This Project
LLM-Check is the key unification attempt to beat: it has breadth across attention, residual, and logit sources but uses no learned fusion. Our stacking design targets exactly that gap.
