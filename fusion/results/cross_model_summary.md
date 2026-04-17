# Cross-Model Aggregation
models: ['qwen2.5-7b', 'llama3.1-8b']
datasets: ['common_claim_3class', 'e2h_amc_3class', 'e2h_amc_5class', 'when2call_3class', 'ragtruth_binary', 'fava_binary']

### RQ1 — Best single probe per (model, dataset)

| Dataset | qwen2.5-7b | llama3.1-8b |
|---|---|---|
| common_claim_3class | lr_probe (0.7712) | pca_lr (0.7621) |
| e2h_amc_3class | pca_lr (0.8937) | pca_lr (0.8785) |
| e2h_amc_5class | pca_lr (0.8760) | pca_lr (0.8576) |
| when2call_3class | lr_probe (0.8640) | kb_mlp (0.8824) |
| ragtruth_binary | iti (0.8804) | iti (0.8862) |
| fava_binary | iti (0.9846) | pca_lr (0.9926) |

### RQ2 — Fusion gains & oracle headroom

**v21 fusion vs best single (Δ AUROC):**

| Dataset | qwen2.5-7b | llama3.1-8b |
|---|---|---|
| common_claim_3class | 0.7817 (Δ+2.41%) | 0.7796 (Δ+1.75%) |
| e2h_amc_3class | 0.9030 (Δ+0.96%) | 0.8928 (Δ+1.43%) |
| e2h_amc_5class | 0.8913 (Δ+1.61%) | 0.8757 (Δ+1.81%) |
| when2call_3class | 0.9392 (Δ+6.51%) | 0.9255 (Δ+4.31%) |
| ragtruth_binary | 0.8930 (Δ+1.22%) | 0.9005 (Δ+1.43%) |
| fava_binary | 0.9880 (Δ+0.24%) | 0.9955 (Δ+0.99%) |

**Oracle headroom — baseline-only vs with-raw (AUROC):**

| Dataset | Model | Best single | Oracle (BL) | Oracle (+raw) | Δraw | Rawwin% |
|---|---|---|---|---|---|---|
| common_claim_3class | qwen2.5-7b | lr_probe | 0.9792 | 0.9957 | +1.65pp | 58% |
| common_claim_3class | llama3.1-8b | pca_lr | 0.9777 | 0.9956 | +1.79pp | 59% |
| e2h_amc_3class | qwen2.5-7b | pca_lr | 0.9986 | 1.0000 | +0.14pp | 92% |
| e2h_amc_3class | llama3.1-8b | pca_lr | 0.9977 | 0.9999 | +0.22pp | 91% |
| e2h_amc_5class | qwen2.5-7b | pca_lr | 0.9941 | 0.9997 | +0.56pp | 86% |
| e2h_amc_5class | llama3.1-8b | pca_lr | 0.9930 | 0.9995 | +0.65pp | 85% |
| when2call_3class | qwen2.5-7b | lr_probe | 0.9902 | 0.9995 | +0.94pp | 78% |
| when2call_3class | llama3.1-8b | kb_mlp | 0.9923 | 0.9996 | +0.73pp | 73% |
| ragtruth_binary | qwen2.5-7b | iti | 0.9997 | 0.9999 | +0.02pp | 43% |
| ragtruth_binary | llama3.1-8b | iti | 0.9996 | 0.9999 | +0.03pp | 52% |
| fava_binary | qwen2.5-7b | iti | 1.0000 | 1.0000 | +0.00pp | 92% |
| fava_binary | llama3.1-8b | pca_lr | 1.0000 | 1.0000 | +0.00pp | 96% |

### RQ3 — Method contribution & pipeline ablation

**Top LOO contributor per (model, dataset):**

| Dataset | qwen2.5-7b | llama3.1-8b |
|---|---|---|
| common_claim_3class | lr_probe (+0.81pp) | pca_lr (+0.58pp) |
| e2h_amc_3class | attn_satisfies (+0.15pp) | iti (+0.50pp) |
| e2h_amc_5class | lr_probe (+0.65pp) | iti (+0.51pp) |
| when2call_3class | iti (+0.87pp) | attn_satisfies (+0.80pp) |
| ragtruth_binary | iti (+3.48pp) | iti (+3.16pp) |
| fava_binary | — | — |

**Pipeline ablation — best config per (model, dataset):**

| Dataset | qwen2.5-7b | llama3.1-8b |
|---|---|---|
| common_claim_3class | pca128_only (0.7817) | full (0.7796) |
| e2h_amc_3class | pca128_only (0.9030) | seed1_only (0.8928) |
| e2h_amc_5class | full (0.8913) | lr_expert_only (0.8757) |
| when2call_3class | seed3 (0.9392) | pca128_only (0.9255) |
| ragtruth_binary | full (0.8926) | no_enrichment (0.9009) |
| fava_binary | — | — |
