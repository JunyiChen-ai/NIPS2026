# Research Wiki — Index

**Project**: Rethinking LLM internal-state probing — from fragmented signals to
an extensible plug-and-play fusion scaffold (NeurIPS 2026).
**Built**: 2026-06-16 from the current progress snapshot (manual build; ARIS
`research_wiki.py` helper not installed on this host).

Counts: 25 papers · 9 ideas · 10 experiments · 12 claims · 5 gaps · 71 edges.

## Gaps → `gap_map.md`
- **G1** no fusion covers all signal sources — *addressed-by-us*
- **G2** query-token position not a first-class fusion variable — *unresolved*
- **G3** no cross-model/cross-task single-weight fusion — *partially-addressed*
- **G4** fragmentation by target-semantics × timing — *addressed-by-us*
- **G5** no unified cross-task/cross-model probe benchmark — *addressed-by-us*

## Ideas (`ideas/`)
| ID | Idea | Stage |
|---|---|---|
| idea:002 | Plug-and-play scaffold fusion (exp7) — **current method** | succeeded |
| idea:001 | Multi-View Expert-Library Stacking (v21) | succeeded |
| idea:009 | Layerwise v3 (5 raw sources) | partial (superseded) |
| idea:008 | Multi-View v1 (view bottleneck) | failed |
| idea:007 | Anchor-residual blend / DRO | partial |
| idea:006 | Score-level LR stacking | partial |
| idea:005 | Feature concat + LR/MLP | failed |
| idea:004 | Neural hierarchical fusion (493K) | failed |
| idea:003 | ProbeCoalition (MoE + router) | failed |

## Experiments (`experiments/`)
baseline_repro · exp1_oracle · exp1b_oracle_raw · exp2_probe_ladder · exp3_loo ·
exp4_pipeline_ablation · exp5_clustering · v21_fusion · **exp7_scaffold** ·
correctness_track

## Claims (`claims/`)
- Supported: C1 no-universal-probe · C2 oracle-headroom · C3 fusion-beats-single ·
  C4 structural-complementarity · C5 task-dependent-contribution · C6 neural-fusion-fails ·
  C7 input-vs-generation-timing · C8 scaffold-generalizes-both-semantics ·
  C9 lower-capacity-generalizes · C10 conservative-no-regression
- Reported (pending re-confirmation): C11 mean-pool-prompt-signal · C12 theoremqa-val-test-shift

## Papers (`papers/`) — 25
**Reproduced probes**: marks2024_geometry_of_truth, azaria2023_saplma,
kossen2024_semantic_entropy_probes, yin2024_lid, wang2025_chain_of_embedding,
ni2025_knowledge_boundary, yuksekgonul2024_sat_probe, li2023_iti,
sriramanan2024_llm_check, step2026_trace_pruning, chen2024_eigenscore_inside,
yao2025_seakr, du2024_haloscope.
**Unification attempts**: ghadiri2025_gnosis, halunet2025, unifact2025,
neural_probe2025_hallucination, clap2025_cross_layer_attention.
**Token-position line**: orgad2024_llms_know_more, niu2025_hami.
**Fusion machinery**: han2024_fusemoe, yun2024_flex_moe, hemker2024_healnet.
**Surveys**: wehner2025_repe_survey, hallucination_survey2025.

## Notes
`## Connections` sections on each page are static placeholders — the relationship
graph lives in `graph/edges.jsonl`. Install ARIS (`research_wiki.py`) to
auto-generate Connections, `query_pack.md`, and lint reports.
