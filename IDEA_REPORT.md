# Research Idea Report

**Direction**: Unified feature fusion framework for LLM internal state probing
**Generated**: 2026-04-07, **Updated**: 2026-04-10
**Ideas evaluated**: 12 generated → 6 implemented → winning method: Multi-View Expert-Library Stacking (baseline-only, v21)

## Landscape Summary

The probing literature is fragmented: 12+ methods each extract different features from LLM internal states (hidden states, attention patterns, logit statistics, geometric scores) and each evaluates on its own preferred tasks. No existing work fuses outputs from multiple probing methods.

The multimodal fusion literature offers mature machinery: FuseMoE (NeurIPS 2024) uses sparse MoE with Laplace gating for heterogeneous modalities; Flex-MoE (NeurIPS 2024 Spotlight) handles arbitrary modality combinations with dual-routing; HEALNet (NeurIPS 2024) uses hybrid early-fusion with cross-attention. The modality collapse problem (ICML 2025) warns that one dominant source can drown others — directly relevant since our top probes (KB MLP, LR Probe) dominate on most tasks.

The gap is clear: nobody has applied multimodal fusion machinery to fuse heterogeneous probing method outputs for LLM internal states. This is technically novel and practically motivated.

## Recommended Idea: ProbeCoalition — Robust Coalition Fusion of Heterogeneous Probe Views

### Architecture: Calibrate → Disagree → Select → Fuse → Residualize

Rather than proposing 5 separate ideas, GPT-5.4 and our analysis converged on a **unified composite system** that combines the best elements:

#### Core Design

```
Input: 12 probe feature vectors x_1, ..., x_12 (dims: scalar to 17920)
                     │
                     ▼
    ┌─────────────────────────────┐
    │  Per-Probe Adapters         │  x_i → h_i ∈ R^128 (dense projection)
    │  + Scalar Head              │  x_i → s_i ∈ R (calibrated score)
    │  + Probe-ID Embedding       │  probe type/family embedding
    └─────────────────────────────┘
                     │
                     ▼
    ┌─────────────────────────────┐
    │  Pairwise Disagreement      │  |p_i - p_j|, JS div, rank conflict
    │  Features                   │  → attention bias in router
    └─────────────────────────────┘
                     │
                     ▼
    ┌─────────────────────────────┐
    │  Sparse Coalition Router    │  Set-transformer with disagreement bias
    │  (top-k=3-4, max-weight    │  → gate weights g_1, ..., g_12
    │   cap per probe)            │  Trained with ProbeDrop + load balancing
    └─────────────────────────────┘
                     │
                     ▼
    ┌─────────────────────────────┐
    │  Coalition Aggregator       │  Attention pooling over selected probes
    │  → learned representation z │
    └─────────────────────────────┘
                     │
                     ▼
    ┌─────────────────────────────┐
    │  Rank-Calibrated Baseline   │  Monotonic fusion of calibrated scores
    │  y_base                     │  → strong non-learned baseline path
    └─────────────────────────────┘
                     │
                     ▼
    ┌─────────────────────────────┐
    │  Residual Prediction        │  y = y_base + δ(z, y_base)
    │                             │  Learned path only corrects baseline
    └─────────────────────────────┘
```

#### Key Components

1. **Rank-Calibrated Scalar Path** (from Idea #7): Each probe produces a calibrated scalar score; a monotonic fusion combines them. This is the robust baseline that the learned path must beat.

2. **Sparse Coalition Router** (from Ideas #1, #9): A set-transformer router selects top-k probes per example. Max-weight cap prevents any single probe from dominating. This addresses the core modality collapse risk.

3. **Disagreement-Biased Attention** (from Idea #3): Pairwise disagreement features (score difference, JS divergence, rank conflict) are injected as attention bias into the router, NOT a separate GNN. This captures when probes disagree — the most informative signal for fusion.

4. **ProbeDrop Regularization** (from Idea #6): During training, randomly and adversarially drop the strongest probe. Consistency loss between masked/full predictions prevents over-reliance on any single probe.

5. **Residual Architecture** (from Idea #2): The learned fusion only needs to improve upon the calibrated baseline, not replace it entirely. This ensures the model is at least as good as calibrated stacking.

#### Training

- Task-specific output heads (binary/multi-class/regression)
- Loss: task loss + load-balancing reg + ProbeDrop consistency loss
- Optional: DRO over within-dataset groups (high-disagreement, anchor-failure slices)
- Train per-dataset (not multi-task, to avoid complicating the story)

### Hypothesis

Probe utility is input-dependent and complementary: different probes capture different computational perspectives on the same input. A sparse coalition with disagreement-aware routing will outperform any single probe because it can dynamically recruit the right experts for each example, while anti-collapse mechanisms prevent degenerating to single-probe behavior.

### Success Criteria

- **Primary**: Match or exceed the best single probe on every dataset (AUROC/Spearman)
- **Secondary**: Mean rank #1 across all datasets, positive on #datasets won
- **Diagnostic**: Show diverse per-example coalitions (not static weighting)

### Novelty Assessment

- **No existing work fuses multiple probing methods** — confirmed by RepE Survey (2025), PING (2025)
- **Fusion machinery** (MoE, cross-attention, anti-collapse) is from multimodal literature — novel application to probing
- **Closest work**: PING (single probe type), FuseMoE (different domain), Flex-MoE (different domain)
- **Differentiation**: We fuse COMPUTATIONAL PERSPECTIVES, not data modalities; extreme heterogeneity (scalar to 17920-dim); disagreement-biased routing is new

### Risk Assessment

- **Risk**: LOW-MEDIUM
- **Main risk**: Rank-calibrated baseline is so strong that the learned path adds marginal improvement
- **Mitigation**: Residual architecture ensures we never lose to the baseline; ProbeDrop ensures we use complementary probes
- **Reviewer concern**: "This is just calibrated stacking with extra steps"
- **Counter**: Show that (a) coalition composition varies per example, (b) improvement concentrates on high-disagreement examples, (c) ablation shows each component matters

### Estimated Effort

- Rank-calibrated baseline: 1-2 days
- Full ProbeCoalition model: 3-4 days
- Ablation studies: 2-3 days
- Paper writing: 1-2 weeks
- **Total**: ~4 weeks (fits NeurIPS deadline)

## Pilot Experiments Planned

### Pilot 1: Rank-Calibrated Stacking Baseline
- **Dataset**: fava_binary + ragtruth_binary (all 12 probes available)
- **Method**: Each probe → scalar score → isotonic calibration → weighted average (learned weights)
- **Time**: ~30 min
- **Success**: Beat best single probe AUROC

### Pilot 2: Simple Concatenation + MLP
- **Dataset**: fava_binary + ragtruth_binary
- **Method**: Project each probe to 128d → concatenate → 2-layer MLP
- **Time**: ~30 min
- **Success**: Beat Pilot 1 (shows feature-level fusion > score-level)

### Pilot 3: ProbeCoalition (simplified)
- **Dataset**: fava_binary + ragtruth_binary
- **Method**: Adapters + sparse router (top-3) + ProbeDrop + residual over calibrated baseline
- **Time**: ~1-2 hours
- **Success**: Beat Pilot 2 and show diverse coalitions

## Eliminated Ideas

| Idea | Reason |
|------|--------|
| Consensus-Specialist Decomposition (#4) | Too complex for 5-week timeline, MEDIUM-HIGH risk |
| Agreement-Conditioned Distillation (#5) | Adds complexity without clear advantage over ProbeDrop |
| Neighbor-of-Experts (#10) | kNN retrieval per probe is slow at inference; complex implementation |
| Latent Reliability Factor Model (#11) | Too theoretical for tight timeline; HIGH risk |
| Task-Conditioned Hypernetwork (#12) | Needs many tasks for meta-learning; only 12 datasets |
| Full GNN Disagreement Graph (#3 as standalone) | Not justified over set-attention; kept as attention bias instead |

## Feature Inventory

Available processed features (per sample):

| Method | Dim | Type | Available Datasets |
|--------|-----|------|--------------------|
| lr_probe | 3584 | hidden state at best layer | 6 new (need to generate for original 4) |
| mm_probe | 3584 | centered hidden state | binary only (fava, ragtruth) |
| pca_lr | 50 | PCA-reduced hidden | all 6 new |
| iti | 128 | per-head activation | all 6 new |
| kb_mlp | 3584 | hidden state at mid layer | all 6 new |
| attn_satisfies | 784 | attention value norms | all 6 new |
| sep | 17920 | gen hidden layer range | all 6 new |
| step | 3584 | gen hidden last decoder | all 6 new |
| lid | scalar | LID score | binary only |
| llm_check | scalar | attention diagonal score | binary only |
| seakr | scalar | energy score | binary only |
| coe | 4 scalars | geometric scores (4 variants) | binary only |

**Note**: Original 4 datasets (GoT Cities, E2H AMC, MetaTool, RetrievalQA) need processed features generated. Multi-class datasets have 7 methods (unsupervised scorers skipped).

## Experimental Results (2026-04-08)

### Evolution: Layerwise v3 → Multi-View Fusion v2

**Layerwise v3** (initial winning method, 5 raw sources):
- Used only 5 of 13 extracted features (input_last_token_hidden, gen_last_token_hidden, input_per_head_activation, input_attn_stats, input_attn_value_norms)
- Results: +1.9% to +3.2% on 4 multi-class datasets

**Multi-View v2 (MVISF-v2)** (final method, ALL 13 features organized into 11 views):
- Added 5 previously unused feature types: input_mean_pool_hidden, gen_mean_pool_hidden, gen_attn_stats_last, input_logit_stats, gen_logit_stats_last
- Removed view-level bottleneck: all per-layer OOF logits go directly to meta-LR
- Results: dramatically improved, especially on When2Call (+6.41%)

### Final Results (MVISF-v2, 9 classification datasets)

| Dataset | Best Single Probe | MVISF-v2 | Delta |
|---------|------------------|----------|-------|
| GoT Cities | attn_sat 1.000 | 0.9993 | -0.07% (saturated) |
| MetaTool | kb_mlp 0.998 | 0.9957 | -0.25% (saturated) |
| RetrievalQA | kb_mlp 0.939 | **0.9456** | **+0.66%** |
| common_claim_3class | PCA+LR 0.758 | **0.7764** | **+1.88%** |
| e2h_amc_3class | PCA+LR 0.893 | **0.9140** | **+2.06%** |
| e2h_amc_5class | KB MLP 0.875 | **0.8980** | **+2.28%** |
| when2call_3class | LR Probe 0.874 | **0.9382** | **+6.41%** |
| fava_binary | iti 0.986 | **0.9907** | **+0.51%** |
| ragtruth_binary | iti 0.881 | **0.8850** | **+0.42%** |

Win/Loss: **7/2** (losses only on saturated datasets) | Wilcoxon p = **0.0098**

### Methods Tried and Eliminated

| Method | Best Result | Why Eliminated |
|--------|-----------|---------------|
| Score-level LR stacking | +0.3% to +1.8% | Probability compression ceiling |
| Feature concat + LR/MLP | All negative | Curse of dimensionality |
| Neural hierarchical fusion (493K) | -2% to -9% | Overfitting on 800-3500 samples |
| Anchor-residual blend / DRO | +0.3% to +1.8% | Same ceiling as score-level |
| Layerwise v3 (5 sources) | +1.9% to +3.2% | Superseded by MVISF-v2 |
| Multi-view v1 (view bottleneck) | Mixed | Bottleneck hurts some datasets |

### Key Findings

1. **Mean-pooled prompt hidden states are the strongest complementary signal** — underexplored in probing literature; +2.9% over last-token on When2Call routing task
2. **Multi-layer information is the key gain source** — using all layers instead of "best layer"
3. **Neural networks fail** — 800-3500 samples too small; linear stacking is optimal
4. **View contribution is task-dependent** — routing: mean-pool; hallucination: probes+attention; difficulty: hidden+heads
5. **Per-example oracle headroom is 12-21%** — probes make genuinely different errors

### Baseline-Feature-Only Fusion (2026-04-08 ~ 2026-04-10)

Constrained to ONLY baseline method post-processed features (C1: no raw LLM states, C2: unified method, C3: scientific soundness).

**Winning Method: Multi-View Expert-Library Stacking (v21)**

Pipeline: Per-method `StandardScaler → PCA({32,128}) → {LR, GBT, ET, RF} → 5-fold OOF × 5 seeds` → entropy/margin enrichment → `{L2-LR, L1-LR, GBT}` blend.

| Dataset | Best Single | Fusion | Delta |
|---------|-----------|--------|-------|
| common_claim | 0.7576 | **0.7817** | **+2.41%** |
| e2h_3c | 0.8934 | **0.9030** | **+0.96%** |
| e2h_5c | 0.8752 | **0.8913** | **+1.61%** |
| when2call | 0.8741 | **0.9392** | **+6.51%** |
| ragtruth | 0.8808 | **0.8930** | **+1.22%** |

All 5 datasets positive. Avg +2.54%. Code: `fusion/baseline_only_v21_winning.py`

19-iteration target-driven loop (GPT-5.4 supervised) explored 15+ architectures. +5% on ALL datasets falsified under C1-C3. See `TARGET_LOOP.md`.

## Next Steps

- [x] Run confidence intervals (bootstrap) on all datasets
- [x] Compute oracle upper bound
- [x] Extend to all classification datasets (9 datasets)
- [x] Ablation studies (13 configs × 4 datasets)
- [x] Multi-view fusion with all 13 features
- [x] **Baseline-feature-only fusion** — Multi-View Expert-Library Stacking, all 5 datasets positive, avg +2.54%
- [ ] Write paper (NeurIPS 2026 deadline ~May)
- [ ] Second model for external validity (needs GPU)

## References

- FuseMoE (Han et al., NeurIPS 2024): Sparse MoE with Laplace gating
- Flex-MoE (Yun et al., NeurIPS 2024 Spotlight): Dual-router for arbitrary modality combinations
- HEALNet (Hemker et al., NeurIPS 2024): Hybrid early-fusion with cross-attention
- Modality Collapse (Chen et al., ICML 2025): Cross-modal distillation to prevent collapse
- PING (2025): Unified probing framework (single method)
- RepE Survey (Wehner et al., arXiv 2025): Representation engineering survey
- Patchscopes (Ghandeharioun et al., ICML 2024): Unified framework for inspecting LLM hidden states
