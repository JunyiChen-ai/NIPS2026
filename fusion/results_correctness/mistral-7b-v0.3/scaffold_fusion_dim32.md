# Exp7 Scaffold Fusion: mistral-7b-v0.3 (new)

| Dataset | Selected | Test AUROC | Best val-single test | Delta | Test-oracle single |
|---|---|---:|---:|---:|---:|
| gsm8k | family_aware_scaffold | 0.7886 | 0.7708 | +0.0178 | 0.7774 |
| math | family_aware_scaffold | 0.8100 | 0.8132 | -0.0032 | 0.8132 |
| mmlu | family_aware_scaffold | 0.7522 | 0.7295 | +0.0226 | 0.7443 |
| commonsenseqa | timing_aware_scaffold | 0.7041 | 0.6828 | +0.0213 | 0.7112 |
| belebele | timing_aware_scaffold | 0.7031 | 0.7202 | -0.0171 | 0.7296 |
| theoremqa | weighted_all_methods | 0.6436 | 0.5760 | +0.0676 | 0.6627 |
| fava | timing_aware_scaffold | 0.9339 | 0.8808 | +0.0531 | 0.8895 |
| ragtruth | single:iti | 0.8332 | 0.8332 | +0.0000 | 0.8332 |
| common_claim_3class | family_aware_scaffold | 0.8359 | 0.8278 | +0.0081 | 0.8318 |
| when2call_3class | weighted_all_methods | 0.6977 | 0.6697 | +0.0280 | 0.7192 |