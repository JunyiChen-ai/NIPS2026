# Scaffold Fusion Summary

## new

- N datasets: `30`
- Mean delta vs val-selected single: `+0.0276`
- Median delta vs val-selected single: `+0.0171`
- Win rate vs val-selected single: `76.7%`
- Mean delta vs test-oracle single: `+0.0006`
- Selected variants: `{'family_aware_scaffold': 13, 'timing_aware_scaffold': 7, 'weighted_all_methods': 7, 'single:sep': 2, 'single:iti': 1}`

## Rows

| Setting | Model | Dataset | Selected | Test | Val-single | Δ val-single | Test-oracle single | Δ oracle |
|---|---|---|---|---:|---:|---:|---:|---:|
| new | qwen2.5-7b | gsm8k | family_aware_scaffold | 0.8411 | 0.8122 | +0.0289 | 0.8697 | -0.0286 |
| new | qwen2.5-7b | math | timing_aware_scaffold | 0.8678 | 0.8643 | +0.0034 | 0.8643 | +0.0034 |
| new | qwen2.5-7b | mmlu | weighted_all_methods | 0.7446 | 0.7252 | +0.0194 | 0.7375 | +0.0071 |
| new | qwen2.5-7b | commonsenseqa | family_aware_scaffold | 0.7762 | 0.6528 | +0.1234 | 0.8145 | -0.0383 |
| new | qwen2.5-7b | belebele | weighted_all_methods | 0.7668 | 0.5710 | +0.1958 | 0.7990 | -0.0322 |
| new | qwen2.5-7b | theoremqa | family_aware_scaffold | 0.7991 | 0.7816 | +0.0175 | 0.7908 | +0.0082 |
| new | qwen2.5-7b | fava | weighted_all_methods | 0.9320 | 0.9156 | +0.0164 | 0.9172 | +0.0148 |
| new | qwen2.5-7b | ragtruth | family_aware_scaffold | 0.6771 | 0.6186 | +0.0585 | 0.6505 | +0.0267 |
| new | qwen2.5-7b | common_claim_3class | timing_aware_scaffold | 0.7921 | 0.7949 | -0.0028 | 0.7949 | -0.0028 |
| new | qwen2.5-7b | when2call_3class | family_aware_scaffold | 0.7557 | 0.7410 | +0.0147 | 0.7410 | +0.0147 |
| new | llama3.1-8b | gsm8k | family_aware_scaffold | 0.8452 | 0.8059 | +0.0393 | 0.8059 | +0.0393 |
| new | llama3.1-8b | math | family_aware_scaffold | 0.8872 | 0.8848 | +0.0024 | 0.8857 | +0.0015 |
| new | llama3.1-8b | mmlu | timing_aware_scaffold | 0.7590 | 0.7408 | +0.0182 | 0.7490 | +0.0100 |
| new | llama3.1-8b | commonsenseqa | family_aware_scaffold | 0.7191 | 0.7379 | -0.0188 | 0.7379 | -0.0188 |
| new | llama3.1-8b | belebele | timing_aware_scaffold | 0.8676 | 0.8549 | +0.0126 | 0.8549 | +0.0126 |
| new | llama3.1-8b | theoremqa | weighted_all_methods | 0.7520 | 0.6752 | +0.0769 | 0.7668 | -0.0147 |
| new | llama3.1-8b | fava | weighted_all_methods | 0.9752 | 0.9670 | +0.0082 | 0.9670 | +0.0082 |
| new | llama3.1-8b | ragtruth | single:sep | 0.8074 | 0.8074 | +0.0000 | 0.8074 | +0.0000 |
| new | llama3.1-8b | common_claim_3class | single:sep | 0.8476 | 0.8476 | +0.0000 | 0.8476 | +0.0000 |
| new | llama3.1-8b | when2call_3class | family_aware_scaffold | 0.7937 | 0.7771 | +0.0167 | 0.7771 | +0.0167 |
| new | mistral-7b-v0.3 | gsm8k | family_aware_scaffold | 0.7886 | 0.7708 | +0.0178 | 0.7774 | +0.0112 |
| new | mistral-7b-v0.3 | math | family_aware_scaffold | 0.8100 | 0.8132 | -0.0032 | 0.8132 | -0.0032 |
| new | mistral-7b-v0.3 | mmlu | family_aware_scaffold | 0.7522 | 0.7295 | +0.0226 | 0.7443 | +0.0079 |
| new | mistral-7b-v0.3 | commonsenseqa | timing_aware_scaffold | 0.7041 | 0.6828 | +0.0213 | 0.7112 | -0.0071 |
| new | mistral-7b-v0.3 | belebele | timing_aware_scaffold | 0.7031 | 0.7202 | -0.0171 | 0.7296 | -0.0264 |
| new | mistral-7b-v0.3 | theoremqa | weighted_all_methods | 0.6436 | 0.5760 | +0.0676 | 0.6627 | -0.0191 |
| new | mistral-7b-v0.3 | fava | timing_aware_scaffold | 0.9339 | 0.8808 | +0.0531 | 0.8895 | +0.0444 |
| new | mistral-7b-v0.3 | ragtruth | single:iti | 0.8332 | 0.8332 | +0.0000 | 0.8332 | +0.0000 |
| new | mistral-7b-v0.3 | common_claim_3class | family_aware_scaffold | 0.8359 | 0.8278 | +0.0081 | 0.8318 | +0.0041 |
| new | mistral-7b-v0.3 | when2call_3class | weighted_all_methods | 0.6977 | 0.6697 | +0.0280 | 0.7192 | -0.0215 |

## Failure Cases

| Setting | Model | Dataset | Selected | Δ val-single | Δ oracle |
|---|---|---|---|---:|---:|
| new | llama3.1-8b | commonsenseqa | family_aware_scaffold | -0.0188 | -0.0188 |
| new | mistral-7b-v0.3 | belebele | timing_aware_scaffold | -0.0171 | -0.0264 |
| new | mistral-7b-v0.3 | math | family_aware_scaffold | -0.0032 | -0.0032 |
| new | qwen2.5-7b | common_claim_3class | timing_aware_scaffold | -0.0028 | -0.0028 |