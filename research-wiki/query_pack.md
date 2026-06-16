# Query Pack (compressed landscape for ideation)

_Auto-style summary, manually built 2026-06-16. Budget ≤ 8000 chars._

## Project direction
Rethink LLM internal-state probing. Empirical diagnosis: across 12 reproduced
probes, 3 open LLMs (qwen2.5-7b, llama3.1-8b, mistral-7b-v0.3), and two label
settings (dataset-label and answer-correctness), **no single probe generalizes**;
a per-example oracle exposes 10-21% headroom; complementarity is **structural**
(probes cluster by signal family with near-orthogonal cross-family errors). The
contribution is not another probe but an **extensible plug-and-play scaffold**:
each probe is an independent expert exposing `{train,val,test}.pt`; a uniform
adapter (`StandardScaler→PCA(64)→LR`) calibrates it; reliability-weighted
family/timing-aware composition is selected by validation only. Pure sklearn, no
LLM retraining. Two instantiations exist: v21 expert-library stacking (heavier,
in the Apr-22 paper draft) and the current lighter exp7 scaffold.

## Top gaps
- **G1** No fusion covers all signal sources (residual/attention/MLP/logit/SAE/
  multi-sample); prior unifiers do ≤3 sources or no learned fusion. *addressed*.
- **G4** Fragmentation by target-semantics × timing: dataset-label favors
  input-side probes, answer-correctness favors generation-side. *addressed*.
- **G3** No single cross-model/cross-task frozen-weight fusion (UniFact shows
  hybrid wins, gives no method). *partial*.
- **G2** Query-token position is not a first-class fusion variable; position
  matters as much as signal (Orgad, HaMI). *unresolved — strong next direction*.
- **G5** No unified cross-task/cross-model probe benchmark. *addressed by our
  reproduction*.

## Failed ideas (DO NOT REPEAT)
- **Neural hierarchical fusion (493K params)**: −2% to −9%, overfit 800-3500
  samples. Neural fusion does not work at this data scale.
- **ProbeCoalition (MoE router + disagreement)**: abandoned — learned gating too
  data-hungry for probing-scale datasets.
- **Feature concat + LR/MLP**: all negative (curse of dimensionality mixing
  17920-dim SEP with scalar scores). Fuse at prediction level, not feature level.
- **Score-level LR stacking** / **anchor-residual / DRO**: weak (+0.3-1.8%),
  probability-compression ceiling / anchor caps gain. Keep per-method probability
  vectors, not scalars; avoid robust-opt wrappers at this scale.
- **Multi-View v1 bottleneck**: view bottleneck hurts high-rank datasets. No
  premature bottleneck between experts and meta-layer.
- **Layerwise v3 (5 sources)**: superseded by adding the 5 unused feature types,
  esp. mean-pooled prompt hidden (biggest single gain).

## Paper clusters
- **Reproduced probes (13)** by signal source: residual/hidden (geometry-of-truth,
  saplma, sep, lid, chain-of-embedding, knowledge-boundary, step, eigenscore,
  haloscope); attention (sat-probe, iti); multi-source (llm-check); multi-sample
  routing (seakr).
- **Unification attempts (5)**: llm-check (3 sources, no learned fusion), gnosis
  (2 sources, backbone-bound), halunet (output-side only), neural-probe / clap
  (single source). UniFact (benchmark) = "no paradigm dominates" — our direct
  justification.
- **Token-position line (2)**: orgad "LLMs know more than they show", HaMI
  (learned token selection) → motivates G2.
- **Fusion machinery (3)**: FuseMoE, Flex-MoE, HEALNet (multimodal MoE / cross-
  attention; modality-collapse is the risk we cite).
- **Surveys (2)**: RepE survey, hallucination-detection survey.

## Top papers
unifact2025 (thesis justification), sriramanan2024_llm_check / ghadiri2025_gnosis
/ halunet2025 (closest unifiers to beat), du2024_haloscope (newest plug-in),
orgad2024_llms_know_more + niu2025_hami (position axis), han2024_fusemoe +
yun2024_flex_moe (machinery), marks2024_geometry_of_truth + li2023_iti
(canonical probes).

## Active chains (limitation → opportunity)
- Realized fusion gain (+1-3pp) ≪ oracle headroom (10-21%) → better per-example
  routing without overfitting (the open prize).
- Scaffold's worst case = val-test ranking shift on low-N reasoning (mistral/
  theoremqa) → val-test-robust selector.
- Position axis (G2) untouched → plug a HaMI-style adaptive-position probe into
  the scaffold.
- Mean-pool prompt-hidden signal (C11) only verified in old setting → re-verify
  under answer-correctness.

## Open unknowns
Can one frozen cross-model weight match per-model val-selection (G3)? Does adding
SAE / multi-sample sources (currently degraded) lift the scaffold? Is the
correctness grader's regex labeling noise material on free-form generations?
