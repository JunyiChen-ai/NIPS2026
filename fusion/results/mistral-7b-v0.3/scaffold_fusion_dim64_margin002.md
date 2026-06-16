# Exp7 Scaffold Fusion: mistral-7b-v0.3 (old)

| Dataset | Selected | Test AUROC | Best val-single test | Delta | Test-oracle single |
|---|---|---:|---:|---:|---:|
| common_claim_3class | single:lr_probe | 0.7575 | 0.7575 | +0.0000 | 0.7650 |
| e2h_amc_3class | single:attn_satisfies | 0.8593 | 0.8593 | +0.0000 | 0.8772 |
| e2h_amc_5class | single:pca_lr | 0.8578 | 0.8578 | +0.0000 | 0.8578 |
| when2call_3class | family_aware_scaffold | 0.8239 | 0.7883 | +0.0357 | 0.8079 |
| ragtruth_binary | single:iti | 0.8570 | 0.8570 | +0.0000 | 0.8570 |
| fava_binary | skipped | | | | |
| belebele | family_aware_scaffold | 0.9084 | 0.8657 | +0.0427 | 0.8682 |