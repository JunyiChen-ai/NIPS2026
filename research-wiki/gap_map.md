# Field Gap Map

Stable IDs (never renumber). Status: `unresolved` | `partially-addressed` | `addressed-by-us`.

## G1 — No fusion covers all signal sources
No prior method fuses the full set of internal-state signal sources (residual /
attention / MLP / logit / SAE / multi-sample). The closest unification attempts
each fall short: LLM-Check reads 3 sources but does **no learned fusion**; Gnosis
learns a fusion but over only 2 sources and is backbone-bound; HaluNet learns a
fusion but over **output-side signals only**. 
- **Status**: addressed-by-us (the scaffold fuses heterogeneous probes via a generic adapter).
- Linked: [[idea:001]], [[idea:002]], [[paper:sriramanan2024_llm_check]], [[paper:ghadiri2025_gnosis]], [[paper:halunet2025]].

## G2 — Query-token position is not a first-class fusion variable
Most probes read at prompt-last or gen-last. A separate line (Orgad et al., HaMI,
first-hallucination-token) shows the **position** of the probe matters as much as
the **signal**. No fusion treats query-token position (axis 2 of our taxonomy) as
a tunable, fusible dimension.
- **Status**: unresolved (future scaffold plug-in via HaMI-style adaptive selection).
- Linked: [[paper:orgad2024_llms_know_more]], [[paper:niu2025_hami]].

## G3 — No cross-model / cross-task single-weight fusion
UniFact's benchmark shows "no paradigm dominates, hybrid is strongest" but offers
no method. No prior work delivers one fusion that holds across model families and
tasks.
- **Status**: partially-addressed (scaffold validated on 3 models × both settings, but a single frozen cross-model weight is not yet shown).
- Linked: [[paper:unifact2025]], [[idea:002]], [[exp:exp7_scaffold]].

## G4 — Fragmentation by target semantics × observation timing
No single probe family dominates across both **target semantics** (dataset-label
vs answer-correctness) and **observation timing** (input-side vs generation-side).
Dataset-label settings favor input-side probes; answer-correctness settings favor
generation-side probes. A method must therefore be a *scaffold* that absorbs
heterogeneous probes, not another fixed probe.
- **Status**: addressed-by-us (this is the specific diagnostic the scaffold answers).
- Linked: [[idea:002]], [[claim:C7]], [[exp:exp7_scaffold]], [[exp:correctness_track]].

## G5 — No unified cross-task/cross-model probe benchmark
Each probe is evaluated on its own preferred task with its own extraction. There
is no apples-to-apples reproduction of many probes under one extraction, split,
and metric protocol.
- **Status**: addressed-by-us (12 probes reproduced under unified extraction + val-based selection across 3 models, 2 settings).
- Linked: [[exp:baseline_repro]].
