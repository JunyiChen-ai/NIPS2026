---
type: experiment
node_id: exp:baseline_repro
title: "Faithful reproduction of 12 internal-state probes under unified extraction"
status: done
added: 2026-06-16T00:00:00Z
---

# Baseline reproduction

## Question
Under one extraction + split + metric protocol, how does each of the 12 probes
perform across models and datasets?

## Setup
12 probes (`reproduce/methods.py`): lr_probe, mm_probe, pca_lr, iti, kb_mlp, lid,
attn_satisfies, llm_check (input-side); sep, coe, seakr, step (generation-side).
3 models (qwen2.5-7b, llama3.1-8b, mistral-7b-v0.3). Two settings: old dataset-
label (7 datasets) and new answer-correctness (10 datasets, 30 cells). Val-based
layer/head/range/threshold selection; no test leakage (fixed original LID/SEP
leaks). Metrics: AUROC / accuracy / F1.

## Key results
No probe dominates ([[claim:C1]]). Winner shifts by task, model, and setting.

## Artifacts
`reproduce/results/`, `reproduce/results_correctness/_summary.json` (30 cells
done), `fusion/results/cross_model_summary.md`.

## Connections
[AUTO-GENERATED from graph/edges.jsonl — do not edit manually]
