# Layer Selection: Paper Description vs Code vs Our Implementation

Comprehensive comparison of how each method selects which layer(s) to use,
across three sources: the published paper, the original code release, and our reproduction.

## Summary Table

| # | Method | Paper | Original Code | Our Reproduction | Match? |
|---|--------|-------|---------------|-----------------|--------|
| 1 | LR Probe (GoT, COLM'24) | Fixed layer from activation patching (e.g. layer 15 for LLaMA-2-13B). "Qualitative results insensitive to choice among early-middle to late-middle layers." | `config.ini` specifies `probe_layer` | Val sweep all layers | **No** — we give more search space |
| 2 | MM Probe (GoT, COLM'24) | Same as LR Probe — patching-guided fixed layer | Same `config.ini` | Val sweep all layers | **No** — same as above |
| 3 | PCA+LR (No Answer Needed, arXiv'25) | "Identify the layer that most effectively discriminates" via 10K TriviaQA, 3-fold CV, sampled every 2-4 layers | Matches paper | Val sweep all layers | **Partial** — same idea, different protocol |
| 4 | ITI (NeurIPS'23) | "Rank all attention heads by probe accuracy on validation set. Take top-K heads." K=48 for LLaMA-7B (out of 1024) | Val-based top-K heads | Val-based best-1 head | **Partial** — we pick 1 head, paper uses 48 |
| 5 | KB MLP (ACL'25) | "Extract representations from intermediate layers (i.e., 16 for Llama2-7B)" — fixed mid layer, citing prior work | Fixed mid layer + dev epoch selection | Fixed `n_layers//2` + val epoch | **Yes** ✓ |
| 6 | LID (ICML'24) | "l = argmax_l Σ LID" — select layer maximizing summed LID **on test set** | Sweep layers on test first 500 samples | Val sweep all layers | **No** — paper is test-leaky, we fixed it |
| 7 | Attn Satisfies (ICLR'24) | All layers all heads jointly, max over positions then flatten | Matches paper | Matches paper | **Yes** ✓ |
| 8 | LLM-Check (NeurIPS'24) | Paper analyzes all 32 layers, finds "performance increases from top to middle layers (8-14), then declines." Code default layer 15 | Fixed `layer_num=15` | Val sweep all layers | **No** — paper studies per-layer, code fixes 15, we sweep |
| 9 | SEP (arXiv'24) | "Select adjacent layers...based on highest mean AUROC in the in-distribution setting" | Selects range **on test** (leaky) | Val sweep, range width ≤ 5 | **No** — paper says in-distribution, code leaks to test, we use val |
| 10 | CoE (ICLR'25) | All layers aggregated — "tracking hidden state evolution across layers" | Matches paper | Matches paper | **Yes** ✓ |
| 11 | SeaKR (ACL'25) | Main method: mid layer (`l=L/2`) hidden state Gram determinant over multiple generations. Energy score is **ablation baseline only** | Fixed layer 15 + per-token energy average | Last-token energy proxy only | **No** — paper's main method is Gram det, both code and we use simplified energy |
| 12 | STEP (ICLR'25) | "Use the last-layer hidden state of step-end token" — fixed last layer | Fixed last layer + val early stopping | Fixed last decoder layer (-2) + val early stopping | **Yes** ✓ |

## Detailed Notes

### Methods where we deviate (need attention)

**LR/MM Probe (GoT)**: The paper's fixed layer is justified by activation patching on the specific model.
Since we use Qwen2.5-7B (different from their LLaMA-2), patching results don't transfer.
Options: (a) run patching on Qwen2.5-7B to find the right layer, (b) keep val sweep.
→ **Action: run patching on Qwen2.5-7B.**

**ITI**: Paper uses top-48 heads for joint intervention. We only classify with the single best head.
This is a fundamental difference — ITI's power comes from multi-head intervention, not single-head probing.
→ Note: for our study (probing, not intervention), single-head is appropriate.

**LLM-Check**: Paper finds optimal layers are 8-14. Their code defaults to 15 (slightly off from their own finding). Our val sweep will find the best layer automatically.
→ Our approach is more principled than the original code.

**SeaKR**: We and the original code both use a simplified energy score. The paper's actual method requires multiple sampled generations per query, which we don't have.
→ Known limitation, documented.

### Methods where we fixed test leakage

**LID**: Paper selects layer by maximizing summed LID on test. This is information leakage.
**SEP**: Original code selects layer range on test AUROC. Paper claims "in-distribution" but code leaks.
→ Our val-based selection is strictly fairer.

### Methods that match perfectly

**KB MLP, Attn Satisfies, CoE, STEP**: Our implementation matches both paper and code.

## References

- Geometry of Truth: https://arxiv.org/abs/2310.06824
- ITI: https://arxiv.org/abs/2306.03341
- No Answer Needed: https://arxiv.org/abs/2509.10625
- KB MLP: https://arxiv.org/abs/2502.11677
- LID: https://arxiv.org/abs/2402.18048
- Attn Satisfies: https://arxiv.org/abs/2310.13650
- LLM-Check: NeurIPS 2024 proceedings
- SEP: https://arxiv.org/abs/2406.15927
- CoE: ICLR 2025 proceedings
- SeaKR: https://arxiv.org/abs/2406.19215
- STEP: https://arxiv.org/abs/2601.09093
