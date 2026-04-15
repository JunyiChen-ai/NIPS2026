# Cross-Model Aggregation
models: ['qwen2.5-7b']
datasets: ['common_claim_3class', 'e2h_amc_3class', 'e2h_amc_5class', 'when2call_3class', 'ragtruth_binary', 'fava_binary']

### RQ1 — Best single probe per (model, dataset)

| Dataset | qwen2.5-7b |
|---|---|
| common_claim_3class | lr_probe (0.7712) |
| e2h_amc_3class | pca_lr (0.8937) |
| e2h_amc_5class | pca_lr (0.8760) |
| when2call_3class | lr_probe (0.8640) |
| ragtruth_binary | iti (0.8804) |
| fava_binary | iti (0.9846) |

### RQ2 — Fusion gains & oracle headroom

**v21 fusion vs best single (Δ AUROC):**

| Dataset | qwen2.5-7b |
|---|---|
| common_claim_3class | 0.7817 (Δ+2.41%) |
| e2h_amc_3class | 0.9030 (Δ+0.96%) |
| e2h_amc_5class | 0.8913 (Δ+1.61%) |
| when2call_3class | 0.9392 (Δ+6.51%) |
| ragtruth_binary | 0.8930 (Δ+1.22%) |
| fava_binary | 0.9880 (Δ+0.24%) |

**Oracle headroom — baseline-only vs with-raw (AUROC):**

| Dataset | Model | Best single | Oracle (BL) | Oracle (+raw) | Δraw | Rawwin% |
|---|---|---|---|---|---|---|
| common_claim_3class | qwen2.5-7b | lr_probe | 0.9792 | 0.9957 | +1.65pp | 58% |
| e2h_amc_3class | qwen2.5-7b | pca_lr | 0.9986 | 1.0000 | +0.14pp | 92% |
| e2h_amc_5class | qwen2.5-7b | pca_lr | 0.9941 | 0.9997 | +0.56pp | 86% |
| when2call_3class | qwen2.5-7b | lr_probe | 0.9902 | 0.9995 | +0.94pp | 78% |
| ragtruth_binary | qwen2.5-7b | iti | 0.9997 | 0.9999 | +0.02pp | 43% |
| fava_binary | qwen2.5-7b | iti | 1.0000 | 1.0000 | +0.00pp | 92% |

### RQ3 — Method contribution & pipeline ablation

**Top LOO contributor per (model, dataset):**

| Dataset | qwen2.5-7b |
|---|---|
| common_claim_3class | lr_probe (+0.81pp) |
| e2h_amc_3class | attn_satisfies (+0.15pp) |
| e2h_amc_5class | lr_probe (+0.65pp) |
| when2call_3class | iti (+0.87pp) |
| ragtruth_binary | iti (+3.48pp) |
| fava_binary | — |

**Pipeline ablation — best config per (model, dataset):**

| Dataset | qwen2.5-7b |
|---|---|
| common_claim_3class | pca128_only (0.7817) |
| e2h_amc_3class | pca128_only (0.9030) |
| e2h_amc_5class | full (0.8913) |
| when2call_3class | seed3 (0.9392) |
| ragtruth_binary | full (0.8926) |
| fava_binary | — |
