# Research Idea Report

**Direction**: Unified feature fusion framework for LLM internal state probing
**Generated**: 2026-04-07, **Updated**: 2026-04-16
**Ideas evaluated**: 12 generated → 6 implemented → winning method: Multi-View Expert-Library Stacking (baseline-only, v21)

## Landscape Summary

The probing literature is fragmented: 12+ methods each extract different features from LLM internal states (hidden states, attention patterns, logit statistics, geometric scores) and each evaluates on its own preferred tasks. No existing work fuses outputs from multiple probing methods.

The multimodal fusion literature offers mature machinery: FuseMoE (NeurIPS 2024) uses sparse MoE with Laplace gating for heterogeneous modalities; Flex-MoE (NeurIPS 2024 Spotlight) handles arbitrary modality combinations with dual-routing; HEALNet (NeurIPS 2024) uses hybrid early-fusion with cross-attention. The modality collapse problem (ICML 2025) warns that one dominant source can drown others — directly relevant since our top probes (KB MLP, LR Probe) dominate on most tasks.

The gap is clear: nobody has applied multimodal fusion machinery to fuse heterogeneous probing method outputs for LLM internal states. This is technically novel and practically motivated.

## Novelty Assessment

- **No existing work fuses multiple probing methods** — confirmed by RepE Survey (2025), PING (2025)
- **Closest work**: PING (single probe type), FuseMoE (different domain), Flex-MoE (different domain)
- **Differentiation**: We fuse COMPUTATIONAL PERSPECTIVES, not data modalities; extreme heterogeneity (scalar to 17920-dim)

## Methods Explored and Eliminated

| Approach | Best Result | Eliminated Because |
|---|---|---|
| ProbeCoalition (MoE + router + disagreement) | Not implemented | Too complex for sample sizes (800–3500); neural methods failed |
| Score-level LR stacking | +0.3% to +1.8% | Probability compression ceiling |
| Feature concat + LR/MLP | All negative | Curse of dimensionality |
| Neural hierarchical fusion (493K params) | -2% to -9% | Overfitting on small datasets |
| Anchor-residual blend / DRO | +0.3% to +1.8% | Same ceiling as score-level |
| Multi-View v1 (view bottleneck) | Mixed | Bottleneck hurts some datasets |

→ Winning method: **Multi-View Expert-Library Stacking (v21)**, pure sklearn, no neural components. See `TARGET_LOOP.md` for 19-iteration evolution.

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

## Current Status (2026-04-16)

- [x] All 6 experiments (exp1–6) complete on Qwen
- [x] Raw-feature oracle (exp1b) — raw views win 43–95% of per-sample oracle competitions
- [x] Cross-model replication on Llama (7/8 done, exp4 running)
- [ ] Mistral replication (MISTRAL_RUNBOOK.md ready)
- [ ] Paper writing (NeurIPS 2026 position track deadline: May 6)

## References

- FuseMoE (Han et al., NeurIPS 2024): Sparse MoE with Laplace gating
- Flex-MoE (Yun et al., NeurIPS 2024 Spotlight): Dual-router for arbitrary modality combinations
- HEALNet (Hemker et al., NeurIPS 2024): Hybrid early-fusion with cross-attention
- Modality Collapse (Chen et al., ICML 2025): Cross-modal distillation to prevent collapse
- PING (2025): Unified probing framework (single method)
- RepE Survey (Wehner et al., arXiv 2025): Representation engineering survey
- Patchscopes (Ghandeharioun et al., ICML 2024): Unified framework for inspecting LLM hidden states
