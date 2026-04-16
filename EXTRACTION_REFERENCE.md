# Extraction & Dataset Reference

Compact reference for the feature extraction pipeline. All stages completed (Apr 2026).

## Models

| Model | Layers | Heads | KV Heads | Hidden | Extraction stored on |
|---|---|---|---|---|---|
| Qwen2.5-7B-Instruct | 28+2 (30 total) | 28 | 4 | 3584 | local + b2 |
| Llama-3.1-8B | 32+2 (34 total) | 32 | 8 | 4096 | b2 (+ local when running) |
| Mistral-7B-v0.3 | 32+2 (34 total) | 32 | 8 | 4096 | b2 |

Extraction method: 2-pass per sample (generate with prefill hooks → replay with hooks). All hidden states stored via explicit hooks. Attention stats computed in hooks (not full attn matrices).

## Extracted Features (13 files per split)

### Prompt-side
| File | Shape (Qwen example) | Notes |
|---|---|---|
| `input_last_token_hidden.pt` | (N, 30, 3584) fp16 | All-layer hidden at last prompt token |
| `input_mean_pool_hidden.pt` | (N, 30, 3584) fp16 | Mean across prompt tokens per layer |
| `input_per_head_activation.pt` | (N, 28, 28, 128) fp16 | o_proj input at last token, per-head |
| `input_attn_stats.pt` | (N, 28, 28, 3) fp32 | skewness / entropy / diag_logmean |
| `input_attn_value_norms.pt` | (N, 28, 28, seq_len) fp16 | ‖attn×value‖ L2 per position (variable last dim) |
| `input_logit_stats.json` | 5 scalars/sample | logsumexp, max_prob, entropy, top5 |

### Generation-side
| File | Shape (Qwen example) | Notes |
|---|---|---|
| `gen_last_token_hidden.pt` | (N, 30, 3584) fp16 | All-layer hidden at last gen token |
| `gen_mean_pool_hidden.pt` | (N, 30, 3584) fp16 | Mean across gen tokens per layer |
| `gen_per_token_hidden_last_layer.pt` | (N, 512, 3584) fp16 | Last decoder layer per gen token, zero-padded |
| `gen_step_boundary_hidden.pt` | list[N] of list of (30, D) | All-layer hidden at \\n\\n boundaries (variable length) |
| `gen_attn_stats_last.pt` | (N, 28, 28, 3) fp32 | Same 3 stats for last gen token |
| `gen_logit_stats_last.json` | 5 scalars/sample | Same stats at last gen token |
| `meta.json` | — | labels, texts, model, split info |

## Datasets (6 target + 4 original)

### Target datasets (used in fusion experiments)
| Dataset | Type | Train | Val | Test | Split convention |
|---|---|---|---|---|---|
| common_claim_3class | 3-class | 3500 | 500 | 1000 | train/val/test |
| e2h_amc_3class | 3-class | 800 | 200 | 2975 | train_sub/val_split/eval |
| e2h_amc_5class | 5-class | 800 | 200 | 2975 | train_sub/val_split/eval |
| when2call_3class | 3-class | 2555 | 366 | 731 | train/val/test |
| ragtruth_binary | binary | 3200 | 800 | 1000 | train/val/test |
| fava_binary | binary | 3500 | 500 | 1000 | train/val/test |

E2H 3-class and 5-class share .pt features (same text, only labels differ in meta.json). Symlinked from `easy2hard_amc`.

### Original datasets (saturated — not used in fusion)
| Dataset | Type | Train | Test |
|---|---|---|---|
| geometry_of_truth_cities | binary | 1196 | 300 |
| easy2hard_amc | regression | 1000 | 2975 |
| metatool_task1 | binary | 832 | 208 |
| retrievalqa | binary | 2228 | 557 |

## Baseline Probes (12 methods)

| Method | Input feature | Available on |
|---|---|---|
| lr_probe | input_last_token_hidden (best layer) | all 6 target |
| pca_lr | input_last_token_hidden → PCA(50) | all 6 target |
| iti | input_per_head_activation | all 6 target |
| kb_mlp | input_last_token_hidden (mid layer) | all 6 target |
| attn_satisfies | input_attn_value_norms | all 6 target |
| sep | gen hidden (layer range) | all 6 target |
| step | gen hidden (last decoder) | all 6 target |
| mm_probe | input_last_token_hidden (centered) | binary only |
| lid | KNN-based scoring | binary only |
| llm_check | input_attn_stats (diag_logmean) | binary only |
| seakr | gen_logit_stats (logsumexp) | binary only |
| coe | gen_mean_pool_hidden (4 geometric) | binary only |

Processed outputs at `reproduce/processed_features/{model}/{dataset}/{method}/{train,val,test}.pt`

## B2 Remote Layout

```
b2:junyi-data/NIPS2026/
├── extraction/features/{qwen2.5-7b,llama3.1-8b,mistral-7b-v0.3}/{dataset}/{split}/
└── reproduce/processed_features/{qwen2.5-7b,llama3.1-8b,mistral-7b-v0.3}/{dataset}/{method}/
```

Download: `rclone sync b2:junyi-data/NIPS2026/extraction/features/{model} extraction/features/{model}`

## Reproducibility Notes

- 11/15 baseline methods fully reproducible from extracted features
- Not reproducible: DoLa, ROME, MEMIT (online/editing methods), Self-Routing RAG (no code)
- Partially reproducible: INSIDE/EigenScore (needs 10× sampling), ICR Probe (needs all-layer per-token), Gnosis (needs raw attn matrices), SAE Entities (needs pre-trained SAE)
