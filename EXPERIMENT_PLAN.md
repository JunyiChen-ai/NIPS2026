# Experiment Plan: Baseline Probing Feature Fusion

**Scope**: Post-processed features from 12 baseline probing methods ONLY. No raw LLM internal states.
**Created**: 2026-04-10
**Method under study**: Multi-View Expert-Library Stacking (v21)
**Code**: `fusion/baseline_only_v21_winning.py`

---

## Paper Narrative

1. Internal signals are diverse — 12+ methods extract different signals (hidden states, attention, logits, geometry)
2. No single probe generalizes — heatmap shows fragmented winners across 9 datasets
3. Fusion helps — oracle headroom 12-21%, our method +0.9~6.5%
4. Fusion principles — task-feature matching emerges naturally; contribution varies by task; robust to component removal

## Research Questions

- **RQ1**: Can a single probe generalize across tasks? (No — heatmap evidence)
- **RQ2**: Does fusing multiple probes improve over the best single? (Yes — how much? diminishing returns?)
- **RQ3**: What are the fusion principles? (Task-feature matching; complementarity; which components work and why)

## Available Data

### Datasets (5 hard unsaturated + 1 extension)

| Dataset | Type | N_train | N_test | N_methods | Best Single AUROC |
|---------|------|---------|--------|-----------|-------------------|
| common_claim_3class | 3-class | 3000 | 2000 | 7 | 0.7576 (PCA+LR) |
| e2h_amc_3class | 3-class | 800 | 2975 | 7 | 0.8934 (PCA+LR) |
| e2h_amc_5class | 5-class | 800 | 2975 | 7 | 0.8752 (KB MLP) |
| when2call_3class | 3-class | 2700 | 3654 | 7 | 0.8741 (LR Probe) |
| ragtruth_binary | binary | 3750 | 2500 | 8 | 0.8808 (ITI) |
| fava_binary | binary | 3750 | 2500 | 8+ | 0.9856 (ITI) |

### 7 Core Methods (multi-class)

| Method | Signal Type | Dim | Paper |
|--------|------------|-----|-------|
| lr_probe | Hidden state @ best layer | 3584 | Geometry of Truth, COLM 2024 |
| pca_lr | PCA-reduced hidden | 50 | No Answer Needed, arXiv 2025 |
| iti | Per-head activation | 128 | ITI, NeurIPS 2023 |
| kb_mlp | Hidden state @ mid layer | 3584 | KB Perception, ACL 2025 |
| attn_satisfies | Attn value norms | 784 | ICLR 2024 |
| sep | Gen hidden layer range | 17920 | SE Probes, arXiv 2024 |
| step | Gen hidden last decoder | 3584 | ICLR 2025 |

### Additional Binary-Only Methods

| Method | Signal Type | Dim |
|--------|------------|-----|
| mm_probe | Centered hidden state | 3584 |
| lid | LID score | scalar |
| llm_check | Attention diagonal | scalar |
| seakr | Energy score | scalar |
| coe | Geometric scores | 4 |

### Current Fusion Results (v21 baseline)

| Dataset | Best Single | Fusion | Delta |
|---------|-----------|--------|-------|
| common_claim | 0.7576 | 0.7817 | +2.41% |
| e2h_3c | 0.8934 | 0.9030 | +0.96% |
| e2h_5c | 0.8752 | 0.8913 | +1.61% |
| when2call | 0.8741 | 0.9392 | +6.51% |
| ragtruth | 0.8808 | 0.8930 | +1.22% |

### Existing Oracle Analysis (4 datasets)

| Dataset | Best Single | Oracle | Headroom |
|---------|-----------|--------|----------|
| common_claim | 0.771 | 0.979 | +20.8% |
| when2call | 0.864 | 0.990 | +12.6% |
| fava | 0.985 | 1.000 | +1.5% |
| ragtruth | 0.880 | 1.000 | +11.9% |

Missing: e2h_3c, e2h_5c.

---

## Experiments

### Exp 1: Per-Example Oracle — Complete Coverage [DONE]
**Supports**: RQ2 (fusion is theoretically justified)
**Gap**: Oracle only covers 4/5 target datasets (missing e2h_3c, e2h_5c)
**Method**:
- For each test sample, pick the probe giving the correct class highest probability
- Compute "oracle AUROC" as upper bound of any selection-based fusion
- Use 7 methods for multi-class, 8+ for binary

**Output**: `fusion/results/oracle_complete.json`
- Table: `best_single → oracle → headroom` for all 5 target datasets + fava
- Key number: headroom on e2h_3c and e2h_5c (expect 10-15%)

**Priority**: P2 (low effort, fills gap)
**Effort**: ~30 min

---

### Exp 2: Probe Ladder (Progressive Addition) [DONE]
**Supports**: RQ2 (is improvement monotonic? diminishing returns? how many probes are enough?)
**Method**:
1. Rank 7 methods by standalone AUROC on each dataset (dataset-specific ranking)
2. Progressively add methods: best-1 → best-2 → ... → all-7
3. At each step, run the v21 pipeline: `PCA({32,128}) × {LR,GBT,ET,RF} × 5seeds → entropy/margin → {L2,L1,GBT} blend`
4. Report test AUROC at each step

**Output**: `fusion/results/probe_ladder.json`
- Per-dataset AUROC curve (x = n_probes, y = AUROC)
- Incremental delta table: what does adding each method contribute?
- Key finding: rapid gain 1→3, diminishing 4→7, never worse than best single

**Variants**:
- (a) Ranked by standalone AUROC (greedy by quality)
- (b) Ranked by complementarity to current set (greedy by marginal gain)

**Priority**: P0 (core RQ2 evidence, most compelling visual)
**Effort**: ~2-3 hours (7 steps × 5 datasets × full pipeline)

---

### Exp 3: Leave-One-Method-Out (Method Contribution) [DONE]
**Supports**: RQ3 (which methods matter? is contribution task-dependent?)
**Method**:
1. For each of the 7 methods: remove it, run full v21 pipeline on remaining 6
2. Compute contribution = `full_fusion_AUROC - leave_one_out_AUROC`
3. Positive contribution = method helps; negative = method hurts (noise)

**Output**: `fusion/results/leave_one_method_out.json`
- 7×5 contribution matrix
- Heatmap showing task-method affinity
- Key findings:
  - Different methods contribute most on different tasks (task-feature matching)
  - Low-AUROC methods (SEP, STEP) contribute near-zero → can safely remove → fusion is robust
  - LR Probe dominant on routing tasks, ITI on hallucination, PCA+LR on difficulty

**Priority**: P0 (core RQ3 evidence, directly answers "which probes matter")
**Effort**: ~2-3 hours (7 ablations × 5 datasets)

---

### Exp 4: Expert-Type Ablation (Pipeline Component Analysis) [DONE]
**Supports**: RQ3 (why does this specific fusion pipeline work? which design choice matters?)
**Method**: Ablate within the v21 pipeline architecture:

| Config | Description |
|--------|-------------|
| `full` | Baseline: PCA({32,128}) × {LR,GBT,ET,RF} × 5seeds → {L2,L1,GBT} blend |
| `pca32_only` | PCA(32) only |
| `pca128_only` | PCA(128) only |
| `lr_expert_only` | LR experts only (no trees) |
| `gbt_expert_only` | GBT experts only |
| `et_expert_only` | ET experts only |
| `rf_expert_only` | RF experts only |
| `tree_experts_only` | {GBT,ET,RF} only (no LR) |
| `meta_l2_only` | L2-LR meta-classifier only (no blend) |
| `meta_l1_only` | L1-LR meta-classifier only |
| `meta_gbt_only` | GBT meta-classifier only |
| `no_enrichment` | Remove entropy/margin features |
| `seed1_only` | 1 seed instead of 5 |
| `seed3` | 3 seeds instead of 5 |

**Output**: `fusion/results/pipeline_ablation.json`
- 14 configs × 5 datasets table
- Key questions answered:
  - Multi-resolution PCA vs single: how much does {32,128} beat {128}?
  - Tree vs linear experts: are trees essential?
  - Meta-blend vs single meta-learner: is blending necessary?
  - Entropy/margin enrichment: load-bearing or cosmetic?
  - Seed count: is 5 seeds worth the compute?

**Priority**: P1 (reviewer defense: "why these design choices?")
**Effort**: ~3-4 hours (14 configs × 5 datasets)

---

### Exp 5: Probe Error Correlation & Clustering [DONE]
**Supports**: RQ3 (probes using similar signals cluster; "nearby = same phenomenon, different computation")
**Method**:
1. For each method on each dataset, generate per-sample predicted probabilities (from OOF or test)
2. Compute pairwise metrics:
   - Rank correlation of predicted probabilities (Spearman)
   - Cohen's kappa on predicted labels (after threshold)
   - Jaccard similarity on error sets (which samples both get wrong)
3. Hierarchical clustering (Ward linkage) on correlation matrix → dendrogram
4. Average across datasets for a global clustering view

**Output**: `fusion/results/probe_clustering.json`
- Pairwise correlation matrix (7×7 per dataset)
- Dendrogram (global + per-dataset)
- Expected clusters:
  - Hidden-state cluster: {lr_probe, pca_lr, kb_mlp} (all use hidden representations)
  - Generation cluster: {sep, step} (both use generation-side features)
  - Attention/head cluster: {iti, attn_satisfies} (attention-based)
- Key finding: high within-cluster correlation, low between-cluster → fusion works because it combines orthogonal computational perspectives

**Priority**: P1 (interpretability, supports "different computational perspectives" narrative)
**Effort**: ~1 hour

---

### Exp 6: Extended Coverage — fava_binary [DONE]
**Supports**: Generality — pipeline works on binary hallucination with more methods
**Gap**: v21 only runs 5 datasets; fava_binary has 8 methods, not included
**Method**: Run v21 pipeline on fava_binary with all 8 available methods (7 core + mm_probe)

**Output**: Add row to `fusion/results/baseline_only_v21_winning_results.json` or separate file
- AUROC, delta vs best single (ITI 0.9856)
- Note: fava is "easy" (AUROC > 0.98), so delta will be small but should be non-negative

**Priority**: P2 (low effort, +1 dataset in paper table)
**Effort**: ~20 min

---

## Execution Plan

### Phase 1: P0 Experiments (Exp 2 + Exp 3) — parallel
Both are independent and can run simultaneously.

```
# Terminal 1: Probe Ladder
python fusion/exp2_probe_ladder.py

# Terminal 2: Leave-One-Method-Out
python fusion/exp3_leave_one_out.py
```

Estimated: ~3 hours wall-clock (bottleneck: GBT hyperparameter search)

### Phase 2: P1 Experiments (Exp 4 + Exp 5) — parallel
```
# Terminal 1: Pipeline Ablation
python fusion/exp4_pipeline_ablation.py

# Terminal 2: Probe Clustering
python fusion/exp5_probe_clustering.py
```

Estimated: ~4 hours wall-clock

### Phase 3: P2 Experiments (Exp 1 + Exp 6) — parallel, quick
```
python fusion/exp1_oracle_complete.py
python fusion/exp6_fava_extension.py
```

Estimated: ~30 min wall-clock

### Total: ~7-8 hours compute, ~5-6 hours wall-clock (with parallelism)

---

## Expected Paper Figures & Tables

| Figure/Table | Source Experiment | Content |
|---|---|---|
| **Table 1**: Single-probe heatmap (12×9) | Existing | AUROC of each method on each dataset |
| **Table 2**: Fusion vs best single (6 datasets) | Existing + Exp 6 | Main result table |
| **Figure 1**: Probe ladder curves | Exp 2 | AUROC vs n_probes, per dataset |
| **Figure 2**: Method contribution heatmap | Exp 3 | 7×5 leave-one-out delta matrix |
| **Figure 3**: Probe error dendrogram | Exp 5 | Hierarchical clustering of 7 methods |
| **Table 3**: Pipeline ablation | Exp 4 | 14 configs × 5 datasets |
| **Table 4**: Oracle upper bound | Exp 1 | best_single → oracle → headroom |

---

## Out of Scope (Deferred)

| Item | Reason |
|------|--------|
| Domain-specific datasets (code/math/logic) | Needs new feature extraction + GPU |
| Second model (non-Qwen) | Needs GPU, strongest remaining improvement for rebuttal |
| Raw-feature fusion (MVISF) | Already done, this plan is baseline-only scope |
| Neural fusion approaches | Falsified — sample size too small (800-3500) |
| Cross-dataset transfer | Train on one dataset, evaluate on another — interesting but secondary |
