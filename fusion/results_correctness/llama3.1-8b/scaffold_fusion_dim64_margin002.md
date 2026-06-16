# Exp7 Scaffold Fusion: llama3.1-8b (new)

| Dataset | Selected | Test AUROC | Best val-single test | Delta | Test-oracle single |
|---|---|---:|---:|---:|---:|
| gsm8k | weighted_all_methods | 0.8206 | 0.7653 | +0.0554 | 0.7907 |
| math | single:sep | 0.8930 | 0.8930 | +0.0000 | 0.8937 |
| mmlu | family_aware_scaffold | 0.7615 | 0.7434 | +0.0182 | 0.7459 |
| commonsenseqa | family_aware_scaffold | 0.6927 | 0.5753 | +0.1174 | 0.7562 |
| belebele | family_aware_scaffold | 0.8838 | 0.8549 | +0.0289 | 0.8549 |
| theoremqa | single:pca_lr | 0.6749 | 0.6749 | +0.0000 | 0.7702 |
| fava | single:sep | 0.9754 | 0.9754 | +0.0000 | 0.9754 |
| ragtruth | single:step | 0.8048 | 0.8048 | +0.0000 | 0.8151 |
| common_claim_3class | single:sep | 0.8471 | 0.8471 | +0.0000 | 0.8471 |
| when2call_3class | family_aware_scaffold | 0.7979 | 0.7568 | +0.0410 | 0.7847 |