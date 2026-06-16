# Method Proposal: Plug-and-Play Internal-State Probe Scaffold

## Motivation

Existing internal-state methods are fragmented along three dimensions:

1. **Signal source**: residual hidden states, attention heads, generated-token
   trajectories, uncertainty scores, semantic consistency, etc.
2. **Observation timing**: input-side / pre-generation vs generation-side /
   post-generation.
3. **Target semantics**: latent knowledge recovery, knowledge boundary
   perception, and generative answer verification.

The empirical diagnosis shows that no single method family dominates across
both target semantics. Dataset-label settings favor input-side methods, while
response-correctness settings often favor generation-side methods. A new method
should therefore not be another fixed probe, but a scaffold that can absorb
heterogeneous probes and decide how to combine them.

## Plug-In Contract

Each method `m` only needs to expose processed features:

```text
processed_features/{model}/{dataset}/{method}/train.pt
processed_features/{model}/{dataset}/{method}/val.pt
processed_features/{model}/{dataset}/{method}/test.pt
```

No method-specific training loop is required inside the scaffold. If a new
method is added later, it only needs to implement its own feature extraction and
save the three tensors.

## Generic Adapter

For each plugged-in method, the scaffold trains the same lightweight adapter:

```text
z_m = Adapter_m(phi_m(x))
Adapter_m = StandardScaler -> PCA(d) -> LogisticRegression
```

where `phi_m(x)` is the processed feature from method `m`, and `z_m` is a
calibrated probability vector over the task labels.

The default dimension is `d=64`, based on the capacity sweep:

| max_dim | New mean delta | New win rate |
|---:|---:|---:|
| 32 | +2.76pp | 76.7% |
| 64 | +3.10pp | 80.0% |
| 128 | +2.24pp | 66.7% |

Interpretation: the scaffold should stay low-capacity to remain generalizable.

## Scaffold Composition

Let `s_m` be method `m`'s validation AUROC. The scaffold computes reliability
weights:

```text
w_m = softmax(tau * max(0, s_m - 0.5))
```

and tests three low-capacity compositions:

### 1. Weighted All-Method Composition

```text
p_all = sum_m w_m z_m
```

### 2. Family-Aware Composition

First group methods by signal family, e.g. residual, attention, semantic
uncertainty, trajectory. Within each family, compute a weighted average; then
weight family-level predictions:

```text
p_g = sum_{m in family g} w_{m|g} z_m
p_family = sum_g w_g p_g
```

### 3. Timing-Aware Composition

Group methods by observation timing:

```text
input-side:      h(x)
generation-side: h(x, y_hat)
```

Then compose predictions in the same two-level way:

```text
p_timing = w_input p_input + w_generation p_generation
```

## Selection Policy

The scaffold uses validation-only selection among:

```text
best validation-selected single method
weighted all-method composition
family-aware composition
timing-aware composition
```

This avoids using test labels for method selection.

Two deployment modes are supported:

### Aggressive Mode

Select the validation-best candidate.

Empirical result:

- New correctness setting: mean delta `+3.10pp`, win rate `80.0%`.
- Old dataset-label setting: mean delta `+1.13pp`, win rate `82.4%`.

### Conservative Mode

Only use an aggregate candidate if it beats the best validation-selected single
method by at least a margin `gamma` on validation. Otherwise fall back to the
single method.

For `gamma=0.02`:

- New correctness setting: mean delta `+2.37pp`.
- Minimum delta vs validation-selected single: `0.00pp`.
- Actual run artifact: `analysis/overnight_runs/scaffold_fusion_dim64_margin002_summary.md`.

This gives a no-regression mode for deployment or ablation tables.

## Diagnostic Meta-Learners

The scaffold also records meta-LR variants for diagnosis:

1. **Validation-trained meta-LR** can sometimes discover useful cross-method
   interactions, but it can overfit validation.
2. **OOF-meta** avoids training on the same examples whose predictions feed the
   meta-learner, but did not repair the current hardest failure case
   (`mistral-7b-v0.3 / theoremqa`).

Therefore meta-LR is kept as diagnostic evidence, not the default deployed
scaffold.

## Empirical Evidence

Primary artifacts:

- `fusion/exp7_scaffold_fusion.py`
- `analysis/overnight_runs/scaffold_fusion_dim64_summary.md`
- `analysis/overnight_runs/scaffold_margin_sweep.md`
- `analysis/overnight_runs/experiment_log.md`

Main results:

| Setting | N | Mean delta vs val-single | Median delta | Win rate |
|---|---:|---:|---:|---:|
| Response correctness | 30 | +3.10pp | +1.74pp | 80.0% |
| Dataset label / latent-style | 17 | +1.13pp | +1.08pp | 82.4% |

This supports the claim that the scaffold is not tied to a single task semantic.
It works across both the old dataset-label setting and the new generative
answer correctness setting.

## Community Value

This method contributes a reusable scaffold rather than another isolated probe:

1. **Scalable**: new methods can be added by writing processed features.
2. **Plug-and-play**: no method-specific retraining logic is needed in the
   scaffold.
3. **Lightweight**: only small adapters are trained; the LLM is not retrained.
4. **Generalizable**: the same scaffold applies across target semantics,
   datasets, and models.
5. **Diagnostic**: family/timing-aware variants reveal which signal groups are
   useful or harmful for each task.
