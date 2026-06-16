# Exp7 Scaffold Fusion: qwen2.5-7b (new)

| Dataset | Selected | Test AUROC | Best val-single test | Delta | Test-oracle single |
|---|---|---:|---:|---:|---:|
| gsm8k | family_aware_scaffold | 0.8506 | 0.8391 | +0.0115 | 0.8396 |
| math | family_aware_scaffold | 0.8757 | 0.8772 | -0.0015 | 0.8772 |
| mmlu | timing_aware_scaffold | 0.7522 | 0.7144 | +0.0378 | 0.7471 |
| commonsenseqa | family_aware_scaffold | 0.7905 | 0.6360 | +0.1545 | 0.8127 |
| belebele | weighted_all_methods | 0.7853 | 0.6163 | +0.1691 | 0.7881 |
| theoremqa | family_aware_scaffold | 0.7925 | 0.7642 | +0.0283 | 0.7747 |
| fava | weighted_all_methods | 0.9374 | 0.9248 | +0.0126 | 0.9248 |
| ragtruth | family_aware_scaffold | 0.6997 | 0.6850 | +0.0147 | 0.6850 |
| common_claim_3class | single:sep | 0.8184 | 0.8184 | +0.0000 | 0.8184 |
| when2call_3class | family_aware_scaffold | 0.7626 | 0.7452 | +0.0175 | 0.7452 |