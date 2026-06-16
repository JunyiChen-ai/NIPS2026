# Exp7 Scaffold Fusion: llama3.1-8b (new)

| Dataset | Selected | Test AUROC | Best val-single test | Delta | Test-oracle single |
|---|---|---:|---:|---:|---:|
| gsm8k | weighted_all_methods | 0.8238 | 0.7708 | +0.0530 | 0.7772 |
| math | family_aware_scaffold | 0.8974 | 0.9031 | -0.0057 | 0.9031 |
| mmlu | family_aware_scaffold | 0.7575 | 0.7339 | +0.0236 | 0.7339 |
| commonsenseqa | family_aware_scaffold | 0.7043 | 0.7178 | -0.0135 | 0.7178 |
| belebele | family_aware_scaffold | 0.8654 | 0.7629 | +0.1025 | 0.8549 |
| theoremqa | weighted_all_methods | 0.7126 | 0.6749 | +0.0376 | 0.7607 |
| fava | weighted_all_methods | 0.9825 | 0.9758 | +0.0067 | 0.9758 |
| ragtruth | weighted_all_methods | 0.8318 | 0.8295 | +0.0023 | 0.8295 |
| common_claim_3class | single:sep | 0.8545 | 0.8545 | +0.0000 | 0.8545 |
| when2call_3class | family_aware_scaffold | 0.8043 | 0.7885 | +0.0158 | 0.7885 |