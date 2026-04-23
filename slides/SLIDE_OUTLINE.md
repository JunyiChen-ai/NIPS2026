# Slide Outline

**Paper**: Rethinking LLM Internal-State Probing: From Fragmented Signals to Extensible Fusion
**Venue**: NeurIPS 2026
**Talk type**: spotlight (~8 min, ~12 slides) — default
**Aspect**: 16:9
**Figures available**: fig0_hero, fig_baseline_heatmap, fig4_oracle, fig3_clustering, fig_competence, fig1_probe_ladder, fig2_loo_heatmap

---

| # | Title | Content | Figure | Time |
|:-:|-------|---------|--------|:----:|
| 1 | Title | Paper title, authors, affiliation, NeurIPS 2026 | — | 15s |
| 2 | Why Probe LLMs? | Hallucination, factuality, calibration are latent inside the model. Probing = read internal signal, train lightweight classifier | — | 50s |
| 3 | From Fragmented Probes to an Extensible Scaffold | Current paradigm: isolated probes, different winner per task. Ideal paradigm: every probe plugs into a shared scaffold. | fig0_hero (full, top + bottom) | 90s |
| 4 | RQ1: Does any probe generalize? | Empirical setup: 7 probes × 5 datasets × 3 LLMs. | fig_baseline_heatmap | 60s |
| 5 | Answer: No single probe dominates | Different winner in almost every cell. Ranking shifts across tasks and models. | (same heatmap) | 60s |
| 6 | RQ2: Is the disagreement exploitable? | Per-example oracle: for each test case pick the probe that gets it right. | fig4_oracle | 60s |
| 7 | Oracle reveals large headroom | Gap between oracle and best single probe >10pp on every dataset. Probes make genuinely different errors. | (same bar chart) | 40s |
| 8 | RQ3: Where does diversity come from? | Pairwise Spearman between probes → cluster by signal family (hidden / attention / generation). Structural, not noise. | fig3_clustering | 60s |
| 9 | But orthogonality ≠ usefulness | Redundancy vs. marginal gain scatter: most orthogonal probes (generation-side) contribute least. Competence matters too. | fig_competence | 50s |
| 10 | Design Targets | Two insights → (i) decision-level fusion; (ii) learned meta-combiner; (iii) plug-in extensibility | — | 40s |
| 11 | The Scaffold: Two Stages | Stage 1: per-probe LR+GBT expert library → OOF probabilities. Stage 2: concat + L2 logistic regression meta-combiner. | — | 60s |
| 12 | Main Results | Ranks first on all 15 dataset-model pairs. Median +1.7%, up to +5.6% AUROC over best single probe. | — | 60s |
| 13 | Ablation | Naive combiners (Pred Avg / LR Stack) lose up to 6.1pp; LR or GBT alone lose pp on heterogeneous tasks. Both load-bearing. | — | 50s |
| 14 | Takeaway | Stop searching for a universal probe. Maintain a scaffold that absorbs each new signal as a plug-in expert. | — | 40s |
| 15 | Thank You / Q&A | QR code for paper + code | — | — |

**Total**: 15 slides, ~13 min of speaking + Q&A buffer.

**Note**: Slide 3 shows the hero figure once (top = current paradigm + bottom = ideal paradigm) to frame the whole talk in one visual. Hero does not reappear.
