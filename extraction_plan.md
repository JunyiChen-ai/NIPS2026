# Offline Feature Extraction Plan (FINAL — completed)

## Model
- Qwen/Qwen2.5-7B-Instruct (28 layers, 28 heads, 4 KV heads, hidden_dim=3584, head_dim=128)
- 2 x A100 80GB, bfloat16, `device_map="auto"`, `attn_implementation="eager"`

## Datasets (9,296 samples total)
1. Geometry of Truth / Cities — 1,196 train + 300 val (binary classification: true/false)
2. Easy2Hard-Bench / AMC — 1,000 train + 2,975 eval (regression: difficulty [0,1])
3. MetaTool Task1 — 832 train + 208 test (binary classification: needs tool?)
4. RetrievalQA — 2,228 train + 557 test (binary classification: needs retrieval?)

## Extraction method
- **2-pass per sample**: generate() with prefill hooks → replay model(prompt+gen) with hooks
- **self_attn monkey-patched**: wraps Accelerate's forward to force `output_attentions=True` locally; upper layers keep `output_attentions=False` so no attention retention/OOM
- **All hidden states via explicit hooks**: embed_tokens, each decoder layer, final norm (30 entries total)
- **Attention stats computed in hooks**: only last-token row + full diagonal stored, not full attention matrix
- **Adaptive batch size**: starts at 256, halves on OOM, raises if batch=1 still OOMs
- **Checkpoint**: skips completed splits (checks meta.json)

## Extracted fields per sample

### Prompt-side (from generate() prefill)
| Field | Shape | Stored dtype | Semantic |
|-------|-------|-------------|----------|
| input_last_token_hidden | (30, 3584) | fp16 | All-layer hidden at last prompt token |
| input_mean_pool_hidden | (30, 3584) | fp16 | Mean across non-padded prompt tokens, per layer |
| input_per_head_activation | (28, 28, 128) | fp16 | o_proj input at last prompt token, per-head |
| input_logit_stats | 5 scalars | float (JSON) | logsumexp, max_prob, entropy, top5 of prompt-final logits |
| input_attn_stats | (28, 28, 3) | fp32 | [0] skewness, [1] entropy of last-prompt-token attn row; [2] diag_logmean over full prompt square (LLM-Check style) |
| input_attn_value_norms | (28, 28, prompt_len) | fp16 | ||attn_weight × value|| L2 norm per position, GQA-mapped |

### Generation-side (from replay forward)
| Field | Shape | Stored dtype | Semantic |
|-------|-------|-------------|----------|
| gen_text | string | JSON | Decoded generation (EOS excluded) |
| gen_last_token_hidden | (30, 3584) | fp16 | All-layer hidden at last content token |
| gen_mean_pool_hidden | (30, 3584) | fp16 | Mean across generated content tokens, per layer |
| gen_per_token_hidden_last_layer | (max_gen_len, 3584) | fp16 | Last decoder layer (pre-norm) at each gen token, zero-padded |
| gen_logit_stats_last | 5 scalars | float (JSON) | Stats of logits at last content token position |
| gen_attn_stats_last | (28, 28, 3) | fp32 | Same 3 stats as input_attn_stats but for last gen token over full sequence |
| gen_step_boundary_hidden | list of (30, 3584) | fp16 | All-layer hidden at \n\n boundaries |

### Storage decisions
- **All-layer per-token hidden NOT stored**: (n_gen, 30, 3584) ≈ 107 MB/sample → ~1 TB total. Only last-layer per-token stored instead (~3.6 MB/sample).
- **Raw attention matrices NOT stored**: (n_layers, n_heads, seq, seq) ≈ 1.6 GB/sample. Only stats (3 scalars/head/layer) + value norms stored.
- **INSIDE 10x sampling NOT done**: Would need 10 separate generations per sample. Can be done separately if needed.

## Baseline reproducibility

### Fully reproducible (11)
| Method | Features used | Task |
|--------|--------------|------|
| Geometry of Truth (linear probe) | input_last_token_hidden | Classification |
| Geometry of Truth (mass-mean probe) | input_last_token_hidden | Classification |
| ITI (direction intervention) | input_per_head_activation | Direction analysis |
| Knowledge Boundary (MLP) | input_last_token_hidden + input_logit_stats | Classification |
| LID | input_last_token_hidden | KNN-based scoring |
| Attention Satisfies | input_attn_value_norms | Classification |
| LLM-Check | input_attn_stats (diag_logmean) | Scoring |
| No Answer Needed (PCA + LR) | input_last_token_hidden | Classification |
| SEP | gen_last_token_hidden | Classification |
| CoE | gen_mean_pool_hidden | Unsupervised scoring |
| SeaKR | gen_logit_stats_last (logsumexp) | Scoring |

### Partially reproducible (4)
| Method | What's missing | Why |
|--------|---------------|-----|
| INSIDE/EigenScore | 10 sampled generations | Only 1 generation extracted |
| ICR Probe | All-layer per-gen-token hidden | ~107 MB/sample, stored last-layer only |
| Gnosis | Raw attention matrices | ~1.6 GB/sample |
| SAE Entities | Pre-trained SAE model | Not available |

### Not reproducible (4)
| Method | Reason |
|--------|--------|
| DoLa | Online contrastive decoding |
| ROME | Causal tracing + parameter editing |
| MEMIT | Causal tracing + parameter editing |
| Self-Routing RAG | Code not released |

## STEP also partially reproducible
STEP trains an MLP on step-boundary hidden states. We extracted `gen_step_boundary_hidden`, but STEP's original training uses many reasoning traces per problem. Our single greedy generation may produce few or no step boundaries for some datasets. Listed as "fully reproducible" above because the method can be applied to our data, but results may differ from original paper.

## Actual storage
- Total: 68 GB across 8 splits
- Quota: 308G / 290G soft limit (in grace period)
