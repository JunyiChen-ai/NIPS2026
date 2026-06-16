# Exp7 Scaffold Fusion: mistral-7b-v0.3 (new)

| Dataset | Selected | Test AUROC | Best val-single test | Delta | Test-oracle single |
|---|---|---:|---:|---:|---:|
| gsm8k | family_aware_scaffold | 0.7682 | 0.7656 | +0.0025 | 0.7656 |
| math | single:sep | 0.7965 | 0.7965 | +0.0000 | 0.8047 |
| mmlu | family_aware_scaffold | 0.7530 | 0.7319 | +0.0211 | 0.7466 |
| commonsenseqa | family_aware_scaffold | 0.7554 | 0.6704 | +0.0850 | 0.7375 |
| belebele | timing_aware_scaffold | 0.6604 | 0.6087 | +0.0518 | 0.6647 |
| theoremqa | single:sep | 0.5580 | 0.5580 | +0.0000 | 0.7461 |
| fava | timing_aware_scaffold | 0.9597 | 0.9053 | +0.0543 | 0.9354 |
| ragtruth | single:iti | 0.8450 | 0.8450 | +0.0000 | 0.8450 |
| common_claim_3class | family_aware_scaffold | 0.8361 | 0.8233 | +0.0128 | 0.8233 |
| when2call_3class | weighted_all_methods | 0.7230 | 0.7100 | +0.0130 | 0.7247 |