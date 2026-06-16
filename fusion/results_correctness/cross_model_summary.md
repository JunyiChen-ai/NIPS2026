# Cross-Model Aggregation
models: ['qwen2.5-7b', 'llama3.1-8b', 'mistral-7b-v0.3']
datasets: ['gsm8k', 'math', 'mmlu', 'commonsenseqa', 'belebele', 'theoremqa', 'fava', 'ragtruth', 'common_claim_3class', 'when2call_3class']

### RQ1 — Best single probe per (model, dataset)

| Dataset | qwen2.5-7b | llama3.1-8b | mistral-7b-v0.3 |
|---|---|---|---|
| gsm8k | sep (0.8640) | seakr (0.7614) | sep (0.7542) |
| math | sep (0.8853) | sep (0.9058) | step (0.8159) |
| mmlu | sep (0.7568) | sep (0.7664) | sep (0.7402) |
| commonsenseqa | sep (0.8366) | step (0.7270) | iti (0.7277) |
| belebele | step (0.8364) | seakr (0.8549) | step (0.6856) |
| theoremqa | lr_probe (0.8505) | attn_satisfies (0.7502) | pca_lr (0.7337) |
| fava | sep (0.9487) | sep (0.9764) | sep (0.9512) |
| ragtruth | sep (0.7341) | sep (0.8432) | iti (0.8522) |
| common_claim_3class | sep (0.8222) | sep (0.8577) | sep (0.8258) |
| when2call_3class | sep (0.7687) | sep (0.7867) | kb_mlp (0.7188) |

### RQ2 — Fusion gains & oracle headroom

**v21 fusion vs best single (Δ AUROC):**

| Dataset | qwen2.5-7b | llama3.1-8b | mistral-7b-v0.3 |
|---|---|---|---|
| gsm8k | 0.8848 (Δ+2.08%) | 0.8618 (Δ+10.04%) | 0.7645 (Δ+1.41%) |
| math | 0.8925 (Δ+0.72%) | 0.9090 (Δ+0.32%) | 0.8207 (Δ+0.77%) |
| mmlu | 0.7784 (Δ+2.16%) | 0.7846 (Δ+1.82%) | 0.7686 (Δ+2.87%) |
| commonsenseqa | 0.8374 (Δ+0.08%) | 0.7354 (Δ+0.84%) | 0.7375 (Δ+1.38%) |
| belebele | 0.8522 (Δ+1.58%) | 0.9026 (Δ+4.76%) | 0.7047 (Δ+3.60%) |
| theoremqa | 0.7793 (Δ-7.11%) | 0.7607 (Δ+1.05%) | 0.7489 (Δ+2.08%) |
| fava | 0.9592 (Δ+28.65%) | 0.9844 (Δ+0.80%) | 0.9691 (Δ+2.87%) |
| ragtruth | 0.7730 (Δ+27.30%) | 0.8578 (Δ+1.46%) | 0.8261 (Δ-0.64%) |
| common_claim_3class | 0.8302 (Δ+7.26%) | 0.8584 (Δ+0.07%) | 0.8389 (Δ+2.25%) |
| when2call_3class | 0.7970 (Δ-7.71%) | 0.8147 (Δ+2.80%) | 0.7532 (Δ+3.55%) |

**Oracle headroom — baseline-only vs with-raw (AUROC):**

| Dataset | Model | Best single | Oracle (BL) | Oracle (+raw) | Δraw | Rawwin% |
|---|---|---|---|---|---|---|

### RQ3 — Method contribution & pipeline ablation

**Top LOO contributor per (model, dataset):**

| Dataset | qwen2.5-7b | llama3.1-8b | mistral-7b-v0.3 |
|---|---|---|---|
| gsm8k | sep (+0.42pp) | sep (+2.94pp) | sep (+-0.14pp) |
| math | sep (+1.19pp) | sep (+0.89pp) | step (+1.70pp) |
| mmlu | sep (+-0.42pp) | sep (+2.05pp) | sep (+1.25pp) |
| commonsenseqa | sep (+9.02pp) | step (+3.31pp) | attn_satisfies (+-2.03pp) |
| belebele | sep (+4.53pp) | sep (+13.54pp) | iti (+2.16pp) |
| theoremqa | pca_lr (+5.47pp) | kb_mlp (+1.58pp) | iti (+2.25pp) |
| fava | sep (+-27.62pp) | sep (+0.62pp) | step (+4.50pp) |
| ragtruth | sep (+-21.70pp) | sep (+2.15pp) | kb_mlp (+1.98pp) |
| common_claim_3class | sep (+-3.37pp) | sep (+1.70pp) | step (+2.00pp) |
| when2call_3class | sep (+15.50pp) | sep (+1.11pp) | kb_mlp (+2.83pp) |

**Pipeline ablation — best config per (model, dataset):**

| Dataset | qwen2.5-7b | llama3.1-8b | mistral-7b-v0.3 |
|---|---|---|---|
| gsm8k | no_enrichment (0.8866) | pca32_only (0.8612) | lr_expert_only (0.7625) |
| math | full (0.8909) | seed1_only (0.9084) | gbt_expert_only (0.8220) |
| mmlu | seed1_only (0.7785) | full (0.7815) | gbt_expert_only (0.7709) |
| commonsenseqa | lr_expert_only (0.8376) | lr_expert_only (0.7350) | lr_expert_only (0.7377) |
| belebele | gbt_expert_only (0.8519) | no_enrichment (0.8860) | full (0.7060) |
| theoremqa | lr_expert_only (0.7793) | gbt_expert_only (0.7539) | seed1_only (0.7607) |
| fava | seed1_only (0.9586) | no_enrichment (0.9839) | full (0.9689) |
| ragtruth | gbt_expert_only (0.7746) | full (0.8550) | no_enrichment (0.8261) |
| common_claim_3class | no_enrichment (0.8303) | no_enrichment (0.8579) | seed1_only (0.8389) |
| when2call_3class | full (0.7993) | seed1_only (0.8119) | no_enrichment (0.7522) |
