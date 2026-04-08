# Target-Driven Research Loop

## Target
- Metric: AUROC (classification)
- Condition: Unified method >= best single probe on ALL hard datasets, ideally +3%
- Datasets & baselines to beat:
  - common_claim_3class: PCA+LR AUROC=0.7576
  - e2h_amc_3class: PCA+LR AUROC=0.8934
  - e2h_amc_5class: KB MLP AUROC=0.8752
  - when2call_3class: LR Probe AUROC=0.8741
- Method constraints: Must be ONE unified framework applied to all datasets

## Iteration 1 — Score-Level LR Fusion (PCA+C tuning)

- **Approach**: Per-probe PCA(128) → LR → OOF probs → weighted avg / CV stacking
- **Results**:
  - cc_3c: 0.775 (+1.7%) ✅
  - e2h_3c: 0.903 (+0.9%) ✅
  - e2h_5c: 0.884 (+0.9%) ✅
  - w2c_3c: 0.852 (-2.2%) ❌
- **Diagnosis**: PCA(128) degrades high-dim probes on when2call. Score-level fusion ceiling ~+2%.
- **Decision**: REFINE — fix PCA dim, tune C per probe

## Iteration 2 — Score-Level Fusion v2 (PCA(256) + C tuning)

- **Approach**: PCA(256), per-probe C tuning via train→val, CV stacking + anchor shrinkage
- **Results** (unified method `cv_stack_anchor_shrink`):
  - cc_3c: 0.776 (+1.8%) ✅
  - e2h_3c: 0.907 (+1.3%) ✅
  - e2h_5c: 0.891 (+1.6%) ✅
  - w2c_3c: 0.877 (+0.3%) ✅
- **Status**: ALL 4 datasets beaten. But gains only +0.3% to +1.8%.
- **Decision**: PIVOT — need +3%, score-level fusion insufficient

## Iteration 3 — Neural Hierarchical Fusion

- **Approach**: Per-source adapters (LayerAttention on 30×3584 hidden states) → concat → MLP. 493K params.
- **Results**: ALL NEGATIVE (-1.8% to -8.7%). Overfitting on small data.
- **Decision**: PIVOT — neural networks don't work here

## Iteration 4 — Layerwise Probe-Bank v1 (subsampled + trajectory)

- **Approach**: Per-layer LR probes (every 2nd layer, PCA(256)) → Gaussian RBF trajectory features + summary stats → meta-LR + anchor blending
- **Results**:
  - cc_3c: 0.777 (+1.9%) ✅
  - e2h_3c: 0.913 (+1.9%) ✅
  - e2h_5c: 0.898 (+2.2%) ✅
  - w2c_3c: 0.898 (+2.4%) ✅
- **Key insight**: Multi-layer information breaks the +2% ceiling on when2call. Trajectory features capture depth structure.
- **Decision**: REFINE — try all layers, richer features

## Iteration 5 — Layerwise v2 (all layers + direct logits + old probes)

- **Approach**: All 30 layers, PCA(512), direct per-layer OOF logits as meta-features, plus old 7-probe OOF logits
- **Results**:
  - cc_3c: 0.775 (+1.7%)
  - e2h_3c: 0.915 (+2.2%) ✅ (best for this dataset)
  - e2h_5c: 0.889 (+1.4%)
  - w2c_3c: 0.915 (+4.1%) ✅ (best for this dataset)
- **Key insight**: All layers help when2call dramatically (+4.1%) but too many meta-features hurt cc_3c/e2h_5c.
- **Decision**: REFINE — combine trajectory + direct logits

## Iteration 6 — Layerwise v3 (combined: direct + trajectory + old probes)

- **Approach**: Subsampled layers, PCA(512), combined trajectory features + direct logits + old probe logits → meta-LR
- **Results** (unified method):
  - cc_3c: 0.776 (+1.9%)
  - e2h_3c: 0.915 (+2.1%)
  - e2h_5c: 0.895 (+1.9%)
  - w2c_3c: 0.906 (+3.2%) ✅
- **Status**: when2call passes +3%. Other 3 datasets stuck at ~+2%.

## GPT-5.4 Assessment (after Iteration 6)

**+3% on ALL 4 datasets is likely unrealistic with current features.**

Three very different fusion regimes (score-level, layerwise stacking, neural) all converge to the same ~+2% ceiling on cc_3c, e2h_3c, e2h_5c. The bottleneck is complementary signal in the features, not model capacity. when2call breaking through to +4.1% shows the method works when complementarity exists.

Recommendation: Move to paper mode. The story is strong — layerwise probe-bank stacking is a novel, effective, unified method that consistently improves over all baselines.

## Iteration 7 — Multi-View Fusion v1 (two-stage with view-level bottleneck)

- **Approach**: Reorganize ALL 13 extracted features into 11 computational views (representation×4, attention×4, confidence×2, probe×1). Two-stage: within-view meta-LR → cross-view meta-LR.
- **Results**: When2Call +6.41% (massive improvement), but mixed on other datasets due to view-level information bottleneck.
- **Decision**: REFINE — remove bottleneck

## Iteration 8 — Multi-View Fusion v2 (direct stacking, FINAL METHOD)

- **Approach**: Same 11 views, but ALL per-layer OOF logits go directly to a single meta-LR. No intermediate view-level compression.
- **Results**:
  - All 7 non-saturated datasets improved (7/7 wins)
  - Wilcoxon p=0.0098
  - When2Call +6.41%, E2H 5c +2.28%, common_claim +1.88%

## Best Results Summary (MVISF-v2 — Final Method)

| Dataset | Best Single Probe | MVISF-v2 | Delta |
|---------|------------------|----------|-------|
| GoT Cities | attn_satisfies 1.000 | 0.9993 | -0.07% (saturated) |
| MetaTool | kb_mlp 0.998 | 0.9957 | -0.25% (saturated) |
| RetrievalQA | kb_mlp 0.939 | **0.9456** | **+0.66%** |
| common_claim_3class | PCA+LR 0.758 | **0.7764** | **+1.88%** |
| e2h_amc_3class | PCA+LR 0.893 | **0.9140** | **+2.06%** |
| e2h_amc_5class | KB MLP 0.875 | **0.8980** | **+2.28%** |
| when2call_3class | LR Probe 0.874 | **0.9382** | **+6.41%** |
| fava_binary | iti 0.986 | **0.9907** | **+0.51%** |
| ragtruth_binary | iti 0.881 | **0.8850** | **+0.42%** |

Win/Loss: **7/2** | Wilcoxon p = **0.0098**

## Final Method: Multi-View Internal State Fusion v2 (MVISF-v2)

### Pipeline (unified, zero per-dataset configuration)

**Input**: 10 raw feature sources (from LLM hooks) + 7 baseline method OOF outputs = 11 views

**Stage 1 — Per-layer probing within each raw view**:
For each of the 10 raw feature sources, at each sampled layer:
1. Extract layer features
2. StandardScaler → PCA (512d for hidden states, 256d for head_act, none for small features)
3. C hyperparameter tuning on holdout
4. 5-fold cross-validated Logistic Regression → OOF class probabilities

Layer sampling: stride 2 for hidden states (30→15 layers), stride 4 for head_act (28→7), stride 1 for attention stats (28 layers, low-dim)

For 7 baseline methods: StandardScaler → PCA(256) if needed → C-tuned LR → OOF probabilities

**Stage 2 — Cross-view meta-classification**:
Concatenate ALL OOF probabilities from ALL views (300-800 meta-features) → StandardScaler → Ridge Logistic Regression with CV-tuned C → final prediction

### Code
- `fusion/multiview_v2.py` — **final method (MVISF-v2)**
- `fusion/multiview_fusion.py` — v1 (two-stage with bottleneck)
- `fusion/layerwise_v3.py` — previous method
- `fusion/results/multiview_v2_results.json` — final results

## All Methods Tried and Eliminated

| Method | Result | Why Eliminated |
|--------|--------|---------------|
| Score-level LR stacking | +0.3% to +1.8% | Probability compression ceiling |
| Feature concat + LR/MLP | All negative | Curse of dimensionality |
| Neural hierarchical fusion (493K) | -2% to -9% | Overfitting on 800-3500 samples |
| Anchor-residual blend / DRO | +0.3% to +1.8% | Same ceiling as score-level |
| Layerwise v3 (5 sources only) | +1.9% to +3.2% | Superseded by MVISF-v2 (more views, better results) |
| Multi-view v1 (view-level bottleneck) | Mixed | Bottleneck hurts some datasets |
