# Exp7 Scaffold Fusion: mistral-7b-v0.3 (new)

| Dataset | Selected | Test AUROC | Best val-single test | Delta | Test-oracle single |
|---|---|---:|---:|---:|---:|
| gsm8k | family_aware_scaffold | 0.7785 | 0.7767 | +0.0018 | 0.7767 |
| math | single:sep | 0.8108 | 0.8108 | +0.0000 | 0.8179 |
| mmlu | family_aware_scaffold | 0.7485 | 0.7256 | +0.0229 | 0.7458 |
| commonsenseqa | timing_aware_scaffold | 0.7213 | 0.6704 | +0.0509 | 0.7487 |
| belebele | weighted_all_methods | 0.6993 | 0.6853 | +0.0140 | 0.6867 |
| theoremqa | weighted_all_methods | 0.6852 | 0.7128 | -0.0276 | 0.7325 |
| fava | timing_aware_scaffold | 0.9476 | 0.9053 | +0.0423 | 0.9099 |
| ragtruth | single:iti | 0.8388 | 0.8388 | +0.0000 | 0.8388 |
| common_claim_3class | family_aware_scaffold | 0.8364 | 0.8191 | +0.0172 | 0.8260 |
| when2call_3class | weighted_all_methods | 0.7102 | 0.6757 | +0.0345 | 0.7323 |