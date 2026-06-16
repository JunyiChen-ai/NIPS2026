# Scaffold Fusion Summary

## new

- N datasets: `30`
- Mean delta vs val-selected single: `+0.0224`
- Median delta vs val-selected single: `+0.0129`
- Win rate vs val-selected single: `66.7%`
- Mean delta vs test-oracle single: `-0.0027`
- Selected variants: `{'family_aware_scaffold': 14, 'timing_aware_scaffold': 4, 'weighted_all_methods': 7, 'single:sep': 4, 'single:iti': 1}`

## Rows

| Setting | Model | Dataset | Selected | Test | Val-single | Δ val-single | Test-oracle single | Δ oracle |
|---|---|---|---|---:|---:|---:|---:|---:|
| new | qwen2.5-7b | gsm8k | family_aware_scaffold | 0.8486 | 0.8490 | -0.0004 | 0.8490 | -0.0004 |
| new | qwen2.5-7b | math | timing_aware_scaffold | 0.8811 | 0.8835 | -0.0024 | 0.8835 | -0.0024 |
| new | qwen2.5-7b | mmlu | family_aware_scaffold | 0.7623 | 0.7388 | +0.0234 | 0.7388 | +0.0234 |
| new | qwen2.5-7b | commonsenseqa | family_aware_scaffold | 0.8019 | 0.8324 | -0.0304 | 0.8324 | -0.0304 |
| new | qwen2.5-7b | belebele | weighted_all_methods | 0.7418 | 0.6204 | +0.1214 | 0.7767 | -0.0350 |
| new | qwen2.5-7b | theoremqa | timing_aware_scaffold | 0.7819 | 0.7236 | +0.0583 | 0.7563 | +0.0257 |
| new | qwen2.5-7b | fava | weighted_all_methods | 0.9419 | 0.9270 | +0.0148 | 0.9270 | +0.0148 |
| new | qwen2.5-7b | ragtruth | family_aware_scaffold | 0.7366 | 0.7229 | +0.0137 | 0.7229 | +0.0137 |
| new | qwen2.5-7b | common_claim_3class | single:sep | 0.8218 | 0.8218 | +0.0000 | 0.8218 | +0.0000 |
| new | qwen2.5-7b | when2call_3class | family_aware_scaffold | 0.7710 | 0.7619 | +0.0091 | 0.7619 | +0.0091 |
| new | llama3.1-8b | gsm8k | weighted_all_methods | 0.8238 | 0.7708 | +0.0530 | 0.7772 | +0.0465 |
| new | llama3.1-8b | math | family_aware_scaffold | 0.8974 | 0.9031 | -0.0057 | 0.9031 | -0.0057 |
| new | llama3.1-8b | mmlu | family_aware_scaffold | 0.7575 | 0.7339 | +0.0236 | 0.7339 | +0.0236 |
| new | llama3.1-8b | commonsenseqa | family_aware_scaffold | 0.7043 | 0.7178 | -0.0135 | 0.7178 | -0.0135 |
| new | llama3.1-8b | belebele | family_aware_scaffold | 0.8654 | 0.7629 | +0.1025 | 0.8549 | +0.0105 |
| new | llama3.1-8b | theoremqa | weighted_all_methods | 0.7126 | 0.6749 | +0.0376 | 0.7607 | -0.0482 |
| new | llama3.1-8b | fava | weighted_all_methods | 0.9825 | 0.9758 | +0.0067 | 0.9758 | +0.0067 |
| new | llama3.1-8b | ragtruth | weighted_all_methods | 0.8318 | 0.8295 | +0.0023 | 0.8295 | +0.0023 |
| new | llama3.1-8b | common_claim_3class | single:sep | 0.8545 | 0.8545 | +0.0000 | 0.8545 | +0.0000 |
| new | llama3.1-8b | when2call_3class | family_aware_scaffold | 0.8043 | 0.7885 | +0.0158 | 0.7885 | +0.0158 |
| new | mistral-7b-v0.3 | gsm8k | family_aware_scaffold | 0.7682 | 0.7656 | +0.0025 | 0.7656 | +0.0025 |
| new | mistral-7b-v0.3 | math | single:sep | 0.7965 | 0.7965 | +0.0000 | 0.8047 | -0.0082 |
| new | mistral-7b-v0.3 | mmlu | family_aware_scaffold | 0.7530 | 0.7319 | +0.0211 | 0.7466 | +0.0065 |
| new | mistral-7b-v0.3 | commonsenseqa | family_aware_scaffold | 0.7554 | 0.6704 | +0.0850 | 0.7375 | +0.0179 |
| new | mistral-7b-v0.3 | belebele | timing_aware_scaffold | 0.6604 | 0.6087 | +0.0518 | 0.6647 | -0.0042 |
| new | mistral-7b-v0.3 | theoremqa | single:sep | 0.5580 | 0.5580 | +0.0000 | 0.7461 | -0.1881 |
| new | mistral-7b-v0.3 | fava | timing_aware_scaffold | 0.9597 | 0.9053 | +0.0543 | 0.9354 | +0.0243 |
| new | mistral-7b-v0.3 | ragtruth | single:iti | 0.8450 | 0.8450 | +0.0000 | 0.8450 | +0.0000 |
| new | mistral-7b-v0.3 | common_claim_3class | family_aware_scaffold | 0.8361 | 0.8233 | +0.0128 | 0.8233 | +0.0128 |
| new | mistral-7b-v0.3 | when2call_3class | weighted_all_methods | 0.7230 | 0.7100 | +0.0130 | 0.7247 | -0.0018 |

## Failure Cases

| Setting | Model | Dataset | Selected | Δ val-single | Δ oracle |
|---|---|---|---|---:|---:|
| new | qwen2.5-7b | commonsenseqa | family_aware_scaffold | -0.0304 | -0.0304 |
| new | llama3.1-8b | commonsenseqa | family_aware_scaffold | -0.0135 | -0.0135 |
| new | llama3.1-8b | math | family_aware_scaffold | -0.0057 | -0.0057 |
| new | qwen2.5-7b | math | timing_aware_scaffold | -0.0024 | -0.0024 |
| new | qwen2.5-7b | gsm8k | family_aware_scaffold | -0.0004 | -0.0004 |