# Scaffold Margin Sweep

| Setting | Margin | N | Mean Δ vs val-single | Win rate | Min Δ | Mean Δ vs test-oracle single | Choices |
|---|---:|---:|---:|---:|---:|---:|---|
| new | 0.000 | 30 | +0.0310 | 80.0% | -0.0276 | -0.0000 | `{'family_aware_scaffold': 15, 'single:iti': 1, 'single:sep': 2, 'timing_aware_scaffold': 3, 'weighted_all_methods': 9}` |
| new | 0.005 | 30 | +0.0301 | 70.0% | -0.0276 | -0.0009 | `{'family_aware_scaffold': 13, 'single:iti': 1, 'single:sep': 5, 'single:step': 1, 'timing_aware_scaffold': 3, 'weighted_all_methods': 7}` |
| new | 0.010 | 30 | +0.0295 | 63.3% | -0.0276 | -0.0015 | `{'family_aware_scaffold': 10, 'single:iti': 1, 'single:sep': 8, 'single:step': 1, 'timing_aware_scaffold': 3, 'weighted_all_methods': 7}` |
| new | 0.015 | 30 | +0.0265 | 50.0% | -0.0276 | -0.0045 | `{'family_aware_scaffold': 8, 'single:iti': 1, 'single:pca_lr': 1, 'single:sep': 11, 'single:step': 1, 'timing_aware_scaffold': 3, 'weighted_all_methods': 5}` |
| new | 0.020 | 30 | +0.0237 | 36.7% | +0.0000 | -0.0073 | `{'family_aware_scaffold': 7, 'single:iti': 1, 'single:lr_probe': 2, 'single:pca_lr': 2, 'single:sep': 13, 'single:step': 1, 'timing_aware_scaffold': 2, 'weighted_all_methods': 2}` |
| new | 0.030 | 30 | +0.0198 | 23.3% | +0.0000 | -0.0112 | `{'family_aware_scaffold': 4, 'single:iti': 1, 'single:lr_probe': 2, 'single:pca_lr': 3, 'single:sep': 15, 'single:step': 2, 'timing_aware_scaffold': 1, 'weighted_all_methods': 2}` |
| new | 0.050 | 30 | +0.0136 | 13.3% | +0.0000 | -0.0174 | `{'family_aware_scaffold': 2, 'single:iti': 1, 'single:kb_mlp': 1, 'single:lr_probe': 2, 'single:pca_lr': 4, 'single:sep': 16, 'single:step': 2, 'weighted_all_methods': 2}` |
| old | 0.000 | 17 | +0.0113 | 82.4% | +0.0000 | +0.0073 | `{'family_aware_scaffold': 5, 'single:iti': 3, 'timing_aware_scaffold': 1, 'weighted_all_methods': 8}` |
| old | 0.005 | 17 | +0.0103 | 64.7% | +0.0000 | +0.0064 | `{'family_aware_scaffold': 5, 'single:iti': 4, 'single:lr_probe': 2, 'weighted_all_methods': 6}` |
| old | 0.010 | 17 | +0.0084 | 47.1% | +0.0000 | +0.0044 | `{'family_aware_scaffold': 4, 'single:iti': 5, 'single:lr_probe': 3, 'single:pca_lr': 1, 'weighted_all_methods': 4}` |
| old | 0.015 | 17 | +0.0063 | 29.4% | +0.0000 | +0.0024 | `{'family_aware_scaffold': 4, 'single:iti': 5, 'single:lr_probe': 3, 'single:pca_lr': 4, 'weighted_all_methods': 1}` |
| old | 0.020 | 17 | +0.0034 | 17.6% | +0.0000 | -0.0006 | `{'family_aware_scaffold': 3, 'single:attn_satisfies': 1, 'single:iti': 5, 'single:kb_mlp': 1, 'single:lr_probe': 3, 'single:pca_lr': 4}` |
| old | 0.030 | 17 | +0.0000 | 0.0% | +0.0000 | -0.0040 | `{'single:attn_satisfies': 1, 'single:iti': 7, 'single:kb_mlp': 1, 'single:lr_probe': 3, 'single:pca_lr': 5}` |
| old | 0.050 | 17 | +0.0000 | 0.0% | +0.0000 | -0.0040 | `{'single:attn_satisfies': 1, 'single:iti': 7, 'single:kb_mlp': 1, 'single:lr_probe': 3, 'single:pca_lr': 5}` |