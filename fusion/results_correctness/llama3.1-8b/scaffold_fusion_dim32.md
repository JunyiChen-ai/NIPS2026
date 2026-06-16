# Exp7 Scaffold Fusion: llama3.1-8b (new)

| Dataset | Selected | Test AUROC | Best val-single test | Delta | Test-oracle single |
|---|---|---:|---:|---:|---:|
| gsm8k | family_aware_scaffold | 0.8452 | 0.8059 | +0.0393 | 0.8059 |
| math | family_aware_scaffold | 0.8872 | 0.8848 | +0.0024 | 0.8857 |
| mmlu | timing_aware_scaffold | 0.7590 | 0.7408 | +0.0182 | 0.7490 |
| commonsenseqa | family_aware_scaffold | 0.7191 | 0.7379 | -0.0188 | 0.7379 |
| belebele | timing_aware_scaffold | 0.8676 | 0.8549 | +0.0126 | 0.8549 |
| theoremqa | weighted_all_methods | 0.7520 | 0.6752 | +0.0769 | 0.7668 |
| fava | weighted_all_methods | 0.9752 | 0.9670 | +0.0082 | 0.9670 |
| ragtruth | single:sep | 0.8074 | 0.8074 | +0.0000 | 0.8074 |
| common_claim_3class | single:sep | 0.8476 | 0.8476 | +0.0000 | 0.8476 |
| when2call_3class | family_aware_scaffold | 0.7937 | 0.7771 | +0.0167 | 0.7771 |