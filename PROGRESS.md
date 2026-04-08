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

### 9. Fusion Method Development (2026-04-07 ~ 2026-04-08)

Developed and evaluated 6 fusion approaches to combine heterogeneous probe outputs into a unified framework.

#### Winning Method: Layerwise Probe-Bank Stacking

**Pipeline** (unified, same code for all datasets):
1. **Per-layer LR probes**: For each raw feature source (input_hidden 30L×3584, gen_hidden 30L×3584, head_act 28L×3584, attn_stats 28L×84, attn_vnorms 28L×varies), extract per-layer features, PCA(512) if >512d, tune C via holdout, train 5-fold CV LR → OOF logits
2. **Depth trajectory features**: Gaussian RBF basis (5-7 bases) + mean/max/std/argmax per class per source
3. **Old probe features**: 7 processed probe methods with PCA(256) + C-tuned LR → OOF logits
4. **Meta-classifier**: Combine all features → StandardScaler → ridge LR with CV-tuned C

#### Results (unified method v3, all 4 hard multi-class datasets)

| Dataset | Best Single Probe | Layerwise Fusion | Delta |
|---------|------------------|-----------------|-------|
| common_claim_3class | PCA+LR 0.7576 | **0.7764** | **+1.9%** |
| e2h_amc_3class | PCA+LR 0.8934 | **0.9148** | **+2.1%** |
| e2h_amc_5class | KB MLP 0.8752 | **0.8946** | **+1.9%** |
| when2call_3class | LR Probe 0.8741 | **0.9062** | **+3.2%** |

#### Methods Tried and Eliminated

| Method | Result | Reason |
|--------|--------|--------|
| Score-level LR stacking (PCA+C tuning) | +0.3% to +1.8% | Probability compression ceiling |
| Feature concat + LR/MLP | Negative | Curse of dimensionality |
| Neural hierarchical fusion (493K params) | -2% to -9% | Overfitting on 800-3500 samples |
| Anchor-residual blend / DRO | +0.3% to +1.8% | Same ceiling as score-level |

#### Key Findings
- Multi-layer information is the biggest gain source — using all layers vs "best layer"
- Per-probe C tuning is critical — default C=1.0 is suboptimal for high-dim features
- Neural networks fail on this data scale (800-3500 samples)
- Signal complementarity varies by dataset — when2call has highest gain (+3-4%)

Code: `fusion/layerwise_v3.py` (unified), `fusion/unified_fast.py` (score-level baseline)
Results: `fusion/results/*.json`

### 10. Auto-Review Loop & Final Analyses (2026-04-08)

GPT-5.4 review loop: 4 rounds, score improved from 4/10 → 5.5 → 6.5 → **7/10 (submission-ready)**.

#### Full Classification Coverage (9 datasets)

| Dataset | Best Single | Fusion | Delta | 95% CI | Sig? |
|---------|-----------|--------|-------|--------|------|
| GoT Cities | 1.000 | 0.9999 | -0.01% | [0.999, 1.000] | n.s. |
| MetaTool | 0.998 | 0.9945 | -0.37% | [0.985, 1.000] | n.s. |
| RetrievalQA | 0.939 | 0.9433 | +0.43% | [0.925, 0.960] | n.s. |
| common_claim | 0.758 | 0.7764 | +1.88% | [0.751, 0.800] | n.s. |
| E2H 3c | 0.893 | **0.9148** | **+2.14%** | [0.907, 0.922] | *** |
| E2H 5c | 0.875 | **0.8946** | **+1.94%** | [0.888, 0.901] | *** |
| When2Call | 0.874 | **0.9062** | **+3.21%** | [0.890, 0.923] | *** |
| FAVA | 0.986 | 0.9897 | +0.41% | [0.984, 0.994] | n.s. |
| RAGTruth | 0.881 | 0.8897 | +0.89% | — | — |

Win/Loss: **7/2** (losses only on saturated GoT/MetaTool). Wilcoxon p=0.049.

#### Ablation Studies (13 configs × 4 datasets)
- Best variant: `drop_attn` (avg rank 2.5/13)
- Input hidden states are dominant source
- Generation hidden states alone near-random
- Trajectory features marginally redundant
- Method robust to component removal

#### Per-Example Oracle Upper Bound
| Dataset | Best Single | Oracle | Headroom |
|---------|-----------|--------|----------|
| common_claim | 0.771 | 0.979 | +20.8% |
| when2call | 0.864 | 0.990 | +12.6% |
| ragtruth | 0.880 | 1.000 | +11.9% |
| fava | 0.985 | 1.000 | +1.5% |

Massive complementarity confirmed.

#### Results files
- `fusion/results/comprehensive_results.json` — all 9 datasets
- `fusion/results/ablation_results.json` — 13 ablations × 4 datasets
- `fusion/results/per_example_oracle.json` — oracle upper bounds
- `fusion/results/ragtruth_failure_analysis.json` — RAGTruth debug
- `fusion/results/paired_delta_ci.json` — statistical tests
- `fusion/results/cross_dataset_tests.json` — Wilcoxon/sign tests
- `fusion/results/gain_vs_difficulty.json` — gain vs difficulty
- `fusion/results/best_variant_analysis.json` — method simplification
- `AUTO_REVIEW.md` — full review loop log

### 11. Multi-View Internal State Fusion (2026-04-08)

Redesigned the fusion method to utilize ALL 13 extracted feature types (previously only 5 used), organized into 11 semantically meaningful "computational views."

#### Method: MVISF-v2 (Direct Multi-View Stacking)

**View taxonomy** (defined by computational role, not post-hoc):

| Category | View | Feature | Dims | Previously Used? |
|----------|------|---------|------|:----------------:|
| Representation | repr_input_last | input_last_token_hidden | (30, 3584) | Yes |
| Representation | repr_input_mean | input_mean_pool_hidden | (30, 3584) | **No** |
| Representation | repr_gen_last | gen_last_token_hidden | (30, 3584) | Yes |
| Representation | repr_gen_mean | gen_mean_pool_hidden | (30, 3584) | **No** |
| Attention | attn_head_act | input_per_head_activation | (28, 28, 128) | Yes |
| Attention | attn_input_stats | input_attn_stats | (28, 28, 3) | Yes |
| Attention | attn_input_vnorms | input_attn_value_norms | (28, 28, var) | Yes |
| Attention | attn_gen_stats | gen_attn_stats_last | (28, 28, 3) | **No** |
| Confidence | conf_input | input_logit_stats | 3 scalars | **No** |
| Confidence | conf_gen | gen_logit_stats_last | 3 scalars | **No** |
| Probe | probe_methods | 7 reproduced methods | var | Yes (partial) |

**Pipeline**: Per-layer LR probes with PCA+C tuning → ALL per-layer OOF logits from ALL views concatenated directly → single meta-LR. No view-level bottleneck.

#### Results: MVISF-v2 (9 classification datasets)

| Dataset | Best Single | MVISF-v2 | Delta | 95% CI | Sig? |
|---------|-----------|----------|-------|--------|------|
| GoT Cities | 1.000 | 0.9993 | -0.07% | [0.998, 1.000] | saturated |
| MetaTool | 0.998 | 0.9957 | -0.25% | [0.989, 1.000] | saturated |
| RetrievalQA | 0.939 | **0.9456** | **+0.66%** | [0.929, 0.963] | |
| common_claim | 0.758 | **0.7764** | **+1.88%** | [0.753, 0.800] | |
| E2H 3c | 0.893 | **0.9140** | **+2.06%** | [0.907, 0.921] | *** |
| E2H 5c | 0.875 | **0.8980** | **+2.28%** | [0.891, 0.904] | *** |
| **When2Call** | 0.874 | **0.9382** | **+6.41%** | [0.925, 0.952] | *** |
| FAVA | 0.986 | **0.9907** | **+0.51%** | [0.986, 0.995] | |
| RAGTruth | 0.881 | **0.8850** | **+0.42%** | [0.864, 0.906] | |

**Win/Loss: 7/2** (losses only on saturated GoT/MetaTool)
**Wilcoxon signed-rank: p = 0.0098** (significant at 1%)

#### Improvement over old method (Layerwise v3)

| Dataset | Old Fusion | MVISF-v2 | Improvement |
|---------|-----------|----------|-------------|
| When2Call | +3.21% | **+6.41%** | **doubled** |
| E2H 5c | +1.94% | +2.28% | +0.34% |
| FAVA | +0.41% | +0.51% | +0.10% |
| RAGTruth | +0.89%* | +0.42% | -0.47% |
| RetrievalQA | +0.43% | +0.66% | +0.23% |

*Old RAGTruth with fixed 7-probe pipeline

#### Per-View Contribution Analysis (avg across datasets)

**Per-view AUROC ranking**:
1. repr_input_last (0.921) — last-token hidden states
2. attn_head_act (0.919) — per-head activations
3. **repr_input_mean (0.907)** — mean-pooled prompt hidden states (**NEW, previously unused**)
4. attn_input_stats (0.892) — attention statistics
5. probe_methods (0.891) — reproduced probe outputs
6. attn_input_vnorms (0.872) — attention value norms
7. repr_gen_mean (0.868) — mean-pooled gen hidden states (**NEW**)
8. attn_gen_stats (0.794) — gen attention stats (**NEW**)
9. repr_gen_last (0.732) — last gen token hidden states
10. conf_input (0.683) — input logit stats (**NEW**)
11. conf_gen (0.558) — gen logit stats (**NEW**)

**Leave-one-view-out contribution**:
1. **repr_input_mean: +0.0030** — strongest complementary signal, drives +2.26% on When2Call
2. probe_methods: +0.0020
3. repr_input_last: +0.0007
4. repr_gen_mean: +0.0005
5. attn_head_act: +0.0004

**Key scientific findings**:
- **Mean-pooled prompt representations are the strongest complementary signal** — underexplored in probing literature
- On When2Call (tool routing), repr_input_mean AUROC=0.933 vs repr_input_last=0.904 (+2.9%)
- View contribution is task-dependent: routing → repr_input_mean, hallucination → probe_methods + attn_head_act
- Confidence views (logit stats) have low standalone AUROC but provide marginal complementary signal

#### GPT-5.4 Review Score Progression

| Round | Score | Key Change |
|-------|-------|-----------|
| 1 | 4/10 | Initial review |
| 2 | 5.5/10 | Full coverage, CIs, ablations |
| 3 | 6.5/10 | RAGTruth fixed, oracle, simplified |
| 4 | 7/10 | Wilcoxon, frozen pipeline, framing |
| 5 | 7.5/10 | Multi-view v1, When2Call +6.41% |
| **6** | **8/10** | **MVISF-v2, 7/7 non-saturated wins, p=0.0098** |

Verdict: "Novel enough and impactful enough for acceptance" — accept-leaning.

#### Code & Results
- `fusion/multiview_v2.py` — **final method (MVISF-v2)**
- `fusion/multiview_fusion.py` — v1 (two-stage, with view-level bottleneck)
- `fusion/results/multiview_v2_results.json` — final results
- `fusion/results/multiview_results.json` — v1 results
- `AUTO_REVIEW.md` — complete 6-round review loop log

## What's Not Done / Next Steps

- [x] Run confidence intervals (bootstrap) on all datasets
- [x] Extend to all classification datasets
- [x] Ablation studies (per-source contribution, trajectory vs direct logits)
- [x] Compute oracle upper bound
- [x] Multi-view fusion with all 13 features
- [x] Per-view contribution analysis
- [ ] **Fusion using ONLY baseline processed features as input** — 限定input为12个baseline方法后处理的feature vectors（不使用LLM raw internal states），验证仅靠已有probing方法的输出能否通过融合获得提升。目前可用的processed features: 7个方法(multi-class) / 12个方法(binary)，每个方法的特征维度从1维(scalar score)到14336维(SEP的多层hidden)不等。每个方法feature经过 StandardScaler → PCA(256) → C-tuned LR → OOF概率后，只贡献K个meta-features（K=类别数）。这可以作为一个pure probe-fusion baseline，与使用raw features的完整MVISF-v2对比。
- [ ] Write paper (NeurIPS 2026 deadline ~May)
- [ ] (Optional) Regression / multi-label fusion
- [ ] (Optional) Second model for external validity (strongest remaining improvement)

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
├── fusion/
│   ├── multiview_v2.py          # FINAL METHOD: MVISF-v2 (direct multi-view stacking)
│   ├── multiview_fusion.py      # MVISF v1 (two-stage with view-level bottleneck)
│   ├── layerwise_v3.py          # Previous method: layerwise probe-bank stacking
│   ├── minimal_ablation.py      # Ablation studies with caching
│   ├── comprehensive_analysis.py # Full dataset coverage analysis
│   ├── round2_fixes.py          # Oracle, failure analysis, paired CIs
│   ├── unified_fast.py          # Score-level baseline (PCA+C tuning+stacking)
│   ├── layerwise_fusion.py      # v1: subsampled + trajectory
│   ├── layerwise_v2.py          # v2: all layers + direct logits
│   ├── neural_fusion.py         # Neural approach (failed)
│   ├── anchor_fusion.py         # Anchor-residual fusion variants
│   └── results/                 # All result JSONs (15+ files)
├── extraction_plan.md
├── selected_datasets.md
├── NEW_DATASETS_PIPELINE.md     # Pipeline I/O specification
├── IDEA_REPORT.md               # Idea discovery + experimental results
├── TARGET_LOOP.md               # Target-driven loop log
├── TARGET_LOOP_STATE.json       # Loop state
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
