# Exp7 Scaffold Fusion: qwen2.5-7b (new)

| Dataset | Selected | Test AUROC | Best val-single test | Delta | Test-oracle single |
|---|---|---:|---:|---:|---:|
| gsm8k | family_aware_scaffold | 0.8486 | 0.8490 | -0.0004 | 0.8490 |
| math | timing_aware_scaffold | 0.8811 | 0.8835 | -0.0024 | 0.8835 |
| mmlu | family_aware_scaffold | 0.7623 | 0.7388 | +0.0234 | 0.7388 |
| commonsenseqa | family_aware_scaffold | 0.8019 | 0.8324 | -0.0304 | 0.8324 |
| belebele | weighted_all_methods | 0.7418 | 0.6204 | +0.1214 | 0.7767 |
| theoremqa | timing_aware_scaffold | 0.7819 | 0.7236 | +0.0583 | 0.7563 |
| fava | weighted_all_methods | 0.9419 | 0.9270 | +0.0148 | 0.9270 |
| ragtruth | family_aware_scaffold | 0.7366 | 0.7229 | +0.0137 | 0.7229 |
| common_claim_3class | single:sep | 0.8218 | 0.8218 | +0.0000 | 0.8218 |
| when2call_3class | family_aware_scaffold | 0.7710 | 0.7619 | +0.0091 | 0.7619 |