# Exp7 Scaffold Fusion: qwen2.5-7b (old)

| Dataset | Selected | Test AUROC | Best val-single test | Delta | Test-oracle single |
|---|---|---:|---:|---:|---:|
| common_claim_3class | single:pca_lr | 0.7577 | 0.7577 | +0.0000 | 0.7649 |
| e2h_amc_3class | single:pca_lr | 0.8911 | 0.8911 | +0.0000 | 0.8911 |
| e2h_amc_5class | single:lr_probe | 0.8702 | 0.8702 | +0.0000 | 0.8702 |
| when2call_3class | family_aware_scaffold | 0.8211 | 0.8149 | +0.0062 | 0.8158 |
| ragtruth_binary | single:iti | 0.8675 | 0.8675 | +0.0000 | 0.8675 |
| fava_binary | single:lr_probe | 0.9828 | 0.9828 | +0.0000 | 0.9829 |
| belebele | weighted_all_methods | 0.9689 | 0.9337 | +0.0351 | 0.9368 |