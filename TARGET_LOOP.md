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

## Best Results Summary (Unified Method = v3)

| Dataset | Best Single Probe | Our Fusion (v3) | Delta |
|---------|------------------|----------------|-------|
| common_claim_3class | PCA+LR 0.7576 | **0.7764** | **+1.9%** |
| e2h_amc_3class | PCA+LR 0.8934 | **0.9148** | **+2.1%** |
| e2h_amc_5class | KB MLP 0.8752 | **0.8946** | **+1.9%** |
| when2call_3class | LR Probe 0.8741 | **0.9062** | **+3.2%** |

## Winning Method: Layerwise Probe-Bank Stacking (v3)

### Pipeline (unified, same code for all datasets)
1. **Per-layer LR probes**: For each raw feature source (input_hidden, gen_hidden, head_act, attn_stats, attn_vnorms), extract per-layer features, PCA(512) if >512d, tune C via holdout, train 5-fold CV LR → OOF logits
2. **Depth trajectory features**: Gaussian RBF basis + mean/max/std/argmax per class per source
3. **Old probe features**: 7 processed probe methods with PCA(256) + C-tuned LR → OOF logits
4. **Meta-classifier**: Combine all (direct logits + trajectory + probe logits) → StandardScaler → ridge LR with CV-tuned C

### Code
- `fusion/layerwise_v3.py` — unified method
- `fusion/unified_fast.py` — score-level baseline
- `fusion/results/` — all result JSONs

## Methods Tried and Eliminated

| Method | Result | Why Eliminated |
|--------|--------|---------------|
| Feature concat + LR/MLP | All negative | Curse of dimensionality, overfitting |
| Neural hierarchical fusion | All negative (-2% to -9%) | Overfitting on 800-3500 samples |
| Score-level LR stacking | +0.3% to +1.8% | Ceiling too low, probability compression |
| Anchor-residual blend | +0.3% to +1.8% | Same ceiling as score-level |
| Fold-DRO simplex weights | +0.4% to +1.1% | DRO doesn't help when signal is absent |
| Full 3584d LR (no PCA) | Too slow (25 min/probe) | Impractical for iterative development |
