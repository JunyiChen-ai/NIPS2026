# NIPS2026 Internal States Study — Progress Summary

## Project Goal
Study how LLM internal states (hidden states, attention, logits) can be used to predict model behavior: whether it needs thinking, retrieval, tool use, or whether its outputs are correct/hallucinated.

## What's Done

### 1. Literature Survey (24 papers)
- Analyzed 24 papers from user's Excel on LLM internal states
- Cloned 19 repos to `/data/jehc223/NIPS2026/baseline/`
- Documented each paper's method, internal states used, datasets, and post-processing
- Key docs: `extraction/post_processing_doc.md`, `extraction_plan.md`

### 2. Dataset Organization
- **4 primary datasets** selected and split (train_sub/val_split/test):
  - Geometry of Truth / Cities — binary classification (true/false statements), 956/240/300
  - Easy2Hard-Bench / AMC — regression (difficulty [0,1]), 800/200/2975
  - MetaTool Task1 — binary classification (needs tool?), 665/167/208
  - RetrievalQA — binary classification (needs retrieval?), 1782/446/557
- **Additional datasets** organized:
  - E2H AMC 3-class and 5-class variants (difficulty bins)
  - FAVA (hallucination, 30K, text-level annotation, no test split)
  - RAGTruth (hallucination, 15K train / 2.7K test, span-level annotation, 4 types)
  - When2Call (3-class tool routing, 3.6K test)
  - common_claim (3-class: True/False/Neither, 20K)
- All at `/data/jehc223/NIPS2026/datasets/`

### 3. Feature Extraction (Completed, Extended 2026-04-03)
- **Model**: Qwen2.5-7B-Instruct, bfloat16, 2×A100, device_map='auto'
- **Method**: 2-pass (generate with prefill hooks → replay forward with hooks)
- **Monkey-patched** self_attn.forward for attention weights without upstream retention
- **Phase 1**: 9,296 samples across 4 datasets, 103 GB
- **Phase 2**: 18,652 samples across 4 new datasets, 139 GB (common_claim, when2call, fava, ragtruth)
- **E2H 3/5-class**: reused Phase 1 features (same text, symlinked .pt, new labels)
- **Total**: 27,948 samples, 8 datasets, **392 GB** features
- **13 features per sample** — see `extraction/post_processing_doc.md` for full spec
- All Codex-reviewed: features verified semantically correct
- Stored at `/data/jehc223/NIPS2026/extraction/features/`

### 4. Baseline Reproduction (Completed)
- **12 methods** reproduced faithful to original repos (verified by Codex):
  1. LR Probe (Geometry of Truth, COLM 2024)
  2. MM Probe (Geometry of Truth, COLM 2024)
  3. PCA+LR (No Answer Needed, arXiv 2025)
  4. ITI per-head probe (Inference-Time Intervention, NeurIPS 2023)
  5. KB MLP (Knowledge Boundary Perception, ACL 2025)
  6. LID (LID Hallucination Detection, arXiv 2024)
  7. Attention Satisfies probe (ICLR 2024)
  8. LLM-Check scoring (NeurIPS 2024)
  9. SEP layer-range probe (Semantic Entropy Probes, arXiv 2024)
  10. CoE geometric scores (Chain of Embedding, ICLR 2025)
  11. SeaKR energy score (ACL 2025)
  12. STEP MLP scorer (ICLR 2025)

- **Evaluation protocol** (no test leakage):
  - Layer/head/range selection on val_split
  - Training on train_sub
  - Final evaluation on test only
  - Unsupervised scorers: val-based direction + max-F1 threshold selection
  - Classification: AUROC, Accuracy, F1
  - Regression: Spearman r, MSE

- Results at `/data/jehc223/NIPS2026/reproduce/results/all_results_v2.json`
- Code at `/data/jehc223/NIPS2026/reproduce/methods.py` and `run_all.py`

### 5. Key Results

**Classification (AUROC):**

| Method | GoT Cities | MetaTool | RetrievalQA |
|--------|-----------|----------|-------------|
| KB MLP | 0.999 | **0.998** | **0.939** |
| LR Probe | 0.999 | 0.997 | 0.906 |
| Attn Satisfies | **1.000** | 0.971 | 0.905 |
| PCA+LR | 0.999 | 0.997 | 0.926 |
| ITI | 0.999 | 0.996 | 0.916 |
| MM Probe | 0.997 | 0.944 | 0.853 |
| LID | 0.997 | 0.917 | 0.727 |
| STEP | 0.950 | 0.848 | 0.772 |
| SEP | 0.936 | 0.866 | 0.748 |
| SeaKR | 0.693 | 0.490 | 0.548 |
| LLM-Check | 0.662 | 0.589 | 0.590 |
| CoE (best) | 0.645 | 0.690 | 0.577 |

**Regression — E2H AMC (|Spearman r|):**

| Method | Spearman r |
|--------|-----------|
| PCA+LR | 0.871 |
| ITI | 0.826 |
| LR Probe | 0.810 |
| KB MLP | 0.807 |
| MM Probe | 0.793 |
| LID | 0.723 |
| LLM-Check | 0.485 |
| SEP | 0.446 |
| Attn Satisfies | 0.426 |
| STEP | 0.345 |
| CoE | 0.191 |
| SeaKR | 0.179 |

### 6. Key Observations
- **Hidden state probing** (LR Probe, PCA+LR, KB MLP, ITI) dominates across all tasks
- **Attention-based** (Attn Satisfies) achieves perfect AUROC on GoT Cities
- **Unsupervised methods** (CoE, SeaKR, LLM-Check) are near random on most tasks
- **Generation-based** methods (STEP, SEP) fall between probing and unsupervised
- RetrievalQA is the hardest dataset — all methods drop significantly

### 7. New Datasets — Feature Extraction & Baseline Reproduction (Completed 2026-04-03)

**8 new dataset variants** processed end-to-end:

#### Data Preparation
- Standardized all datasets to `all.jsonl` + `split_indices.json` format
- FAVA: XML tag cleaning (keep `<delete>` = hallucinated text, strip `<mark>` = corrections), typo-corrected 6 hallucination types
- RAGTruth: binary + 2-label from `hallucination_labels_processed`
- FAVA/RAGTruth text includes task prompt ("determine whether...contains hallucination") + reference/context
- Other datasets: raw text only (no task prompt)
- Scripts: `extraction/prepare_new_datasets.py`

#### Feature Extraction
- **Model**: Qwen2.5-7B-Instruct, bfloat16, 2×A100, device_map='auto'
- **Method**: Same 2-pass extraction as original 4 datasets
- **Samples**: 18,652 across 4 datasets (common_claim 5K, when2call 3.6K, fava 5K, ragtruth 5K)
- **Time**: ~9.5 hours total
- **Disk**: 139 GB new features
- E2H 3/5-class: reused existing easy2hard_amc features (same text), symlinked .pt + new meta.json with class_label
- Pipeline: extract `all` once → split by index (no redundant extraction)
- Scripts: `extraction/extract_features_new.py`, `extraction/split_features.py`, `extraction/setup_e2h_multiclass.py`

#### Baseline Reproduction
- **12 methods** × **8 datasets** = 96 runs
- Multi-class (3/5-class): supervised probes adapted to multi-class LogisticRegression; unsupervised methods (CoE, SeaKR, LLM-Check, LID, MM Probe) skipped (output scalar, cannot classify multi-class)
- Multi-label: per-label binary evaluation, macro-averaged metrics
- Binary: same pipeline as original 4 datasets
- Scripts: `reproduce/run_new_datasets.py`
- Processed feature vectors saved: `reproduce/save_processed_features.py` → `reproduce/processed_features/`

#### Results — Multi-class Classification (AUROC)

| Method | E2H 3-class | E2H 5-class | common_claim | When2Call |
|--------|-------------|-------------|--------------|----------|
| PCA+LR | **0.893** | **0.875** | **0.758** | 0.798 |
| KB MLP | 0.891 | **0.875** | 0.757 | 0.872 |
| LR Probe | 0.886 | 0.863 | 0.694 | **0.874** |
| ITI | 0.856 | 0.844 | 0.737 | 0.841 |
| Attn Satisfies | 0.837 | 0.799 | 0.640 | 0.805 |
| SEP | 0.668 | 0.631 | 0.500 | 0.591 |
| STEP | 0.633 | 0.607 | 0.505 | 0.571 |

(MM Probe, LID, LLM-Check, CoE, SeaKR: skipped for multi-class)

#### Results — Binary Classification (AUROC)

| Method | FAVA binary | RAGTruth binary |
|--------|-------------|-----------------|
| ITI | **0.986** | **0.881** |
| LR Probe | 0.984 | 0.789 |
| PCA+LR | 0.981 | 0.834 |
| KB MLP | 0.968 | 0.839 |
| Attn Satisfies | 0.964 | 0.793 |
| LID | 0.884 | 0.696 |
| MM Probe | 0.872 | 0.776 |
| SEP | 0.687 | 0.608 |
| STEP | 0.667 | 0.688 |
| CoE (best) | 0.631 | 0.591 |
| LLM-Check | 0.619 | 0.722 |
| SeaKR | 0.470 | 0.553 |

#### Results — Multi-label (Macro AUROC)

| Method | FAVA 6-label | RAGTruth 2-label |
|--------|-------------|-----------------|
| ITI | **0.846** | **0.836** |
| KB MLP | 0.809 | 0.822 |
| LR Probe | 0.810 | 0.796 |
| PCA+LR | 0.802 | 0.802 |
| Attn Satisfies | 0.767 | 0.778 |
| MM Probe | 0.703 | 0.759 |
| LLM-Check | 0.571 | 0.717 |
| LID | 0.645 | 0.698 |
| STEP | 0.546 | 0.674 |
| SEP | 0.565 | 0.643 |
| CoE | 0.503 | 0.579 |
| SeaKR | 0.499 | 0.545 |

### 8. Key Observations (New Datasets)
- **Hidden state probing dominance confirmed** on harder datasets: ITI, LR Probe, PCA+LR, KB MLP consistently top
- **ITI emerges as strongest** on binary/multi-label (was not top on original datasets), suggesting per-head probing better captures fine-grained signals
- **Multi-class difficulty scales as expected**: 3-class > 5-class performance drop, common_claim is hardest (near random for generation-based methods)
- **FAVA binary is easy** (AUROC > 0.96 for probes) — model internal states clearly encode hallucination presence
- **RAGTruth is harder** — long context + generation makes probing more challenging
- **Generation-side methods (SEP, STEP) still underperform input-side** — consistent with original findings
- **Unsupervised methods (CoE, SeaKR) remain near-random** — internal states fundamentally outperform output-based signals

## What's Not Done / Next Steps

### Research Directions
- Gap identified: no dataset simultaneously labels retrieval + thinking + tool use needs
- Hidden state probing is strong → can we design a unified probe across tasks?
- Generation-side features (STEP, SEP) underperform input-side → why?
- Unsupervised methods fail → are internal states fundamentally better than output-based signals?
- ITI's strong multi-label performance → per-head specialization for different error types?

## File Structure
```
/data/jehc223/NIPS2026/
├── baseline/                    # 19 cloned repos
├── datasets/                    # Raw organized datasets with splits
│   ├── knowledge_factual/
│   ├── reasoning_difficulty/    # E2H AMC + 3class + 5class variants
│   ├── tool_use_routing/        # MetaTool, When2Call
│   ├── retrieval_routing/       # RetrievalQA, Self-RAG
│   ├── multilingual/
│   └── hallucination/           # FAVA, RAGTruth
├── datasets_prepared/           # Standardized JSONL for new datasets
│   ├── common_claim_3class/     # all.jsonl + split_indices.json
│   ├── when2call_3class/
│   ├── fava/
│   └── ragtruth/
├── extraction/
│   ├── extract_features.py      # Original 2-pass extraction (Codex-reviewed)
│   ├── extract_features_new.py  # New dataset extraction (Codex-reviewed)
│   ├── split_features.py        # Index-based split of all/ → train/val/test
│   ├── setup_e2h_multiclass.py  # E2H 3/5-class symlink setup
│   ├── prepare_new_datasets.py  # Data prep: subsample, split, label extraction
│   ├── features/                # 392 GB extracted features (original + new)
│   ├── post_processing_doc.md
│   └── hf_cache/
├── reproduce/
│   ├── methods.py               # 12 method implementations (Codex-reviewed)
│   ├── run_all.py               # Original runner (4 datasets)
│   ├── run_new_datasets.py      # New runner (8 datasets, multi-class/label)
│   ├── save_processed_features.py # Save processed feature vectors
│   ├── create_val_split.py
│   ├── results/                 # all_results_v2.json, all_results_v3.json
│   └── processed_features/      # 7.3 GB per-method processed features
├── extraction_plan.md
├── selected_datasets.md
├── NEW_DATASETS_PIPELINE.md     # Pipeline I/O specification
└── PROGRESS.md                  # This file
```

## Disk Usage
- Features (original 4 datasets): ~103 GB
- Features (new 4 datasets): ~139 GB
- Features (E2H 3/5-class symlinks): ~0 GB (symlinked)
- Processed features: ~7.3 GB
- Model cache: ~15 GB
- Baseline repos: ~4 GB
- Total NIPS2026: ~630 GB
- Quota: 290G soft limit / 3000G hard limit
