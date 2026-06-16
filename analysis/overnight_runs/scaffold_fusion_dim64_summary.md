# Scaffold Fusion Summary

## new

- N datasets: `30`
- Mean delta vs val-selected single: `+0.0310`
- Median delta vs val-selected single: `+0.0174`
- Win rate vs val-selected single: `80.0%`
- Mean delta vs test-oracle single: `-0.0000`
- Selected variants: `{'family_aware_scaffold': 15, 'timing_aware_scaffold': 3, 'weighted_all_methods': 9, 'single:sep': 2, 'single:iti': 1}`

## old

- N datasets: `17`
- Mean delta vs val-selected single: `+0.0113`
- Median delta vs val-selected single: `+0.0108`
- Win rate vs val-selected single: `82.4%`
- Mean delta vs test-oracle single: `+0.0073`
- Selected variants: `{'weighted_all_methods': 8, 'family_aware_scaffold': 5, 'single:iti': 3, 'timing_aware_scaffold': 1}`

## Rows

| Setting | Model | Dataset | Selected | Test | Val-single | Δ val-single | Test-oracle single | Δ oracle |
|---|---|---|---|---:|---:|---:|---:|---:|
| new | qwen2.5-7b | gsm8k | family_aware_scaffold | 0.8506 | 0.8391 | +0.0115 | 0.8396 | +0.0109 |
| new | qwen2.5-7b | math | family_aware_scaffold | 0.8757 | 0.8772 | -0.0015 | 0.8772 | -0.0015 |
| new | qwen2.5-7b | mmlu | timing_aware_scaffold | 0.7522 | 0.7144 | +0.0378 | 0.7471 | +0.0052 |
| new | qwen2.5-7b | commonsenseqa | family_aware_scaffold | 0.7905 | 0.6360 | +0.1545 | 0.8127 | -0.0222 |
| new | qwen2.5-7b | belebele | weighted_all_methods | 0.7853 | 0.6163 | +0.1691 | 0.7881 | -0.0027 |
| new | qwen2.5-7b | theoremqa | family_aware_scaffold | 0.7925 | 0.7642 | +0.0283 | 0.7747 | +0.0178 |
| new | qwen2.5-7b | fava | weighted_all_methods | 0.9374 | 0.9248 | +0.0126 | 0.9248 | +0.0126 |
| new | qwen2.5-7b | ragtruth | family_aware_scaffold | 0.6997 | 0.6850 | +0.0147 | 0.6850 | +0.0147 |
| new | qwen2.5-7b | common_claim_3class | single:sep | 0.8184 | 0.8184 | +0.0000 | 0.8184 | +0.0000 |
| new | qwen2.5-7b | when2call_3class | family_aware_scaffold | 0.7626 | 0.7452 | +0.0175 | 0.7452 | +0.0175 |
| new | llama3.1-8b | gsm8k | weighted_all_methods | 0.8206 | 0.7653 | +0.0554 | 0.7907 | +0.0299 |
| new | llama3.1-8b | math | family_aware_scaffold | 0.8911 | 0.8930 | -0.0019 | 0.8937 | -0.0026 |
| new | llama3.1-8b | mmlu | family_aware_scaffold | 0.7615 | 0.7434 | +0.0182 | 0.7459 | +0.0156 |
| new | llama3.1-8b | commonsenseqa | family_aware_scaffold | 0.6927 | 0.5753 | +0.1174 | 0.7562 | -0.0635 |
| new | llama3.1-8b | belebele | family_aware_scaffold | 0.8838 | 0.8549 | +0.0289 | 0.8549 | +0.0289 |
| new | llama3.1-8b | theoremqa | weighted_all_methods | 0.7252 | 0.6749 | +0.0503 | 0.7702 | -0.0450 |
| new | llama3.1-8b | fava | weighted_all_methods | 0.9797 | 0.9754 | +0.0043 | 0.9754 | +0.0043 |
| new | llama3.1-8b | ragtruth | weighted_all_methods | 0.8161 | 0.8048 | +0.0112 | 0.8151 | +0.0010 |
| new | llama3.1-8b | common_claim_3class | family_aware_scaffold | 0.8508 | 0.8471 | +0.0037 | 0.8471 | +0.0037 |
| new | llama3.1-8b | when2call_3class | family_aware_scaffold | 0.7979 | 0.7568 | +0.0410 | 0.7847 | +0.0132 |
| new | mistral-7b-v0.3 | gsm8k | family_aware_scaffold | 0.7785 | 0.7767 | +0.0018 | 0.7767 | +0.0018 |
| new | mistral-7b-v0.3 | math | single:sep | 0.8108 | 0.8108 | +0.0000 | 0.8179 | -0.0071 |
| new | mistral-7b-v0.3 | mmlu | family_aware_scaffold | 0.7485 | 0.7256 | +0.0229 | 0.7458 | +0.0027 |
| new | mistral-7b-v0.3 | commonsenseqa | timing_aware_scaffold | 0.7213 | 0.6704 | +0.0509 | 0.7487 | -0.0274 |
| new | mistral-7b-v0.3 | belebele | weighted_all_methods | 0.6993 | 0.6853 | +0.0140 | 0.6867 | +0.0127 |
| new | mistral-7b-v0.3 | theoremqa | weighted_all_methods | 0.6852 | 0.7128 | -0.0276 | 0.7325 | -0.0473 |
| new | mistral-7b-v0.3 | fava | timing_aware_scaffold | 0.9476 | 0.9053 | +0.0423 | 0.9099 | +0.0377 |
| new | mistral-7b-v0.3 | ragtruth | single:iti | 0.8388 | 0.8388 | +0.0000 | 0.8388 | +0.0000 |
| new | mistral-7b-v0.3 | common_claim_3class | family_aware_scaffold | 0.8364 | 0.8191 | +0.0172 | 0.8260 | +0.0104 |
| new | mistral-7b-v0.3 | when2call_3class | weighted_all_methods | 0.7102 | 0.6757 | +0.0345 | 0.7323 | -0.0221 |
| old | qwen2.5-7b | common_claim_3class | weighted_all_methods | 0.7685 | 0.7577 | +0.0108 | 0.7649 | +0.0036 |
| old | qwen2.5-7b | e2h_amc_3class | weighted_all_methods | 0.9037 | 0.8911 | +0.0126 | 0.8911 | +0.0126 |
| old | qwen2.5-7b | e2h_amc_5class | weighted_all_methods | 0.8824 | 0.8702 | +0.0121 | 0.8702 | +0.0121 |
| old | qwen2.5-7b | when2call_3class | family_aware_scaffold | 0.8211 | 0.8149 | +0.0062 | 0.8158 | +0.0053 |
| old | qwen2.5-7b | ragtruth_binary | single:iti | 0.8675 | 0.8675 | +0.0000 | 0.8675 | +0.0000 |
| old | qwen2.5-7b | fava_binary | weighted_all_methods | 0.9878 | 0.9828 | +0.0050 | 0.9829 | +0.0048 |
| old | llama3.1-8b | common_claim_3class | weighted_all_methods | 0.7699 | 0.7589 | +0.0110 | 0.7603 | +0.0097 |
| old | llama3.1-8b | e2h_amc_3class | weighted_all_methods | 0.8933 | 0.8692 | +0.0241 | 0.8795 | +0.0138 |
| old | llama3.1-8b | e2h_amc_5class | family_aware_scaffold | 0.8663 | 0.8503 | +0.0159 | 0.8503 | +0.0159 |
| old | llama3.1-8b | when2call_3class | family_aware_scaffold | 0.8417 | 0.8318 | +0.0099 | 0.8318 | +0.0099 |
| old | llama3.1-8b | ragtruth_binary | single:iti | 0.8763 | 0.8763 | +0.0000 | 0.8763 | +0.0000 |
| old | llama3.1-8b | fava_binary | timing_aware_scaffold | 0.9951 | 0.9906 | +0.0045 | 0.9930 | +0.0021 |
| old | mistral-7b-v0.3 | common_claim_3class | weighted_all_methods | 0.7647 | 0.7575 | +0.0072 | 0.7650 | -0.0003 |
| old | mistral-7b-v0.3 | e2h_amc_3class | family_aware_scaffold | 0.8848 | 0.8593 | +0.0255 | 0.8772 | +0.0076 |
| old | mistral-7b-v0.3 | e2h_amc_5class | weighted_all_methods | 0.8695 | 0.8578 | +0.0116 | 0.8578 | +0.0116 |
| old | mistral-7b-v0.3 | when2call_3class | family_aware_scaffold | 0.8239 | 0.7883 | +0.0357 | 0.8079 | +0.0161 |
| old | mistral-7b-v0.3 | ragtruth_binary | single:iti | 0.8570 | 0.8570 | +0.0000 | 0.8570 | +0.0000 |

## Failure Cases

| Setting | Model | Dataset | Selected | Δ val-single | Δ oracle |
|---|---|---|---|---:|---:|
| new | mistral-7b-v0.3 | theoremqa | weighted_all_methods | -0.0276 | -0.0473 |
| new | llama3.1-8b | math | family_aware_scaffold | -0.0019 | -0.0026 |
| new | qwen2.5-7b | math | family_aware_scaffold | -0.0015 | -0.0015 |