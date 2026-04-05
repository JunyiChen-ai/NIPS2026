# Post-Processing Documentation for ALL Baselines

How each baseline paper processes raw extracted features before feeding into their classifier/scorer.
This documents what transformations need to happen AFTER our offline extraction.

---

## Category A: Offline-compatible (input forward pass only, no generation)

### 1. Geometry of Truth (COLM 2024)
- **Input**: `input_last_token_hidden[layer]` → shape `[N, hidden_dim]`
- **Centering**: `acts = acts - acts.mean(dim=0)` (mandatory)
- **Scaling**: `acts = acts / acts.std(dim=0)` (optional, flag `scale=True`)
- **PCA**: Optional, eigendecomposition on centered data, select top-k components
- **Classifier**: Linear logistic regression (LRProbe) or mass-mean probe (MMProbe)
- **Code**: `geometry-of-truth/utils.py:61-63`, `probes.py:3-71`
- **Extraction field**: `input_last_token_hidden` ✅

### 2. ITI / Honest LLaMA (NeurIPS 2023)
- **Input**: `input_per_head_activation[layer, head]` → shape `[N, head_dim]`
- **Direction**: `direction = mean(acts[true]) - mean(acts[false])` per head
- **Normalize**: `direction = direction / ||direction||`
- **Scaling**: `alpha * std(acts @ direction.T) * direction` for intervention
- **Classifier**: Not a classifier — uses direction for inference-time intervention
- **Code**: `honest_llama/utils.py:778-779`, `validation/validate_2fold.py:171-175`
- **Extraction field**: `input_per_head_activation` ✅

### 3. Knowledge Boundary Perception (ACL 2025)
- **Input**: `input_last_token_hidden` in multiple modes:
  - `"last"`: final layer hidden state → `[N, hidden_dim]`
  - `"mid"`: middle layer (e.g., layer 14 for 28-layer) → `[N, hidden_dim]`
  - `"all"`: average across all layers → `[N, hidden_dim]`
- **Additional**: logit stats (top-k tracking across layers), attention weights
- **Classifier**: 4-layer MLP binary classifier (input → hidden → output confidence)
- **Calibration**: C³ method — reformulate question, compare confidence consistency
- **Code**: `LLM-Knowledge-Boundary-Perception-via-Internal-States/utils/llm.py:87-102`
- **Extraction fields**: `input_last_token_hidden` + `input_logit_stats` ✅

### 4. LID Hallucination Detection
- **Input**: `input_last_token_hidden[layer]` per layer → shape `[N, hidden_dim]`
- **No centering/scaling**
- **KNN**: Compute k-nearest neighbors distances using train set as reference
- **LID**: `LID = -1 / mean(log(distances / max_distance))`
- **Output**: Scalar per sample per layer
- **Code**: `lid-hallucinationdetection/src/lids.py:25-40`
- **Extraction field**: `input_last_token_hidden` ✅

### 5. Attention Satisfies / Mechanistic Error Probe (ICLR 2024)
- **Input**: `attention_weights` × `value_vectors` → per-head per-token contribution
- **Computation**: `token_contribs = attn_weights[:, pos, :] * (o_proj @ value_states)`
- **Feature**: `L2_norm(token_contribs)` per head → shape `[n_layers, n_heads, seq_len]`
- **Classifier**: Logistic regression on contribution norms
- **Code**: `mechanistic-error-probe/model_lib/attention_tools.py:56-90`
- **Extraction field**: `input_attn_value_norms` ✅

### 6. LLM-Check (NeurIPS 2024)
- **Input**: Attention diagonal statistics across all tokens
- **Computation**: `sum(log(diag(attn_matrix[i1:i2, i1:i2])).mean())` across heads per layer
- **Feature**: Scalar score per sample
- **Code**: `LLM_Check_Hallucination_Detection/common_utils.py:319-352`
- **Extraction fields**: `input_attn_stats` (diag_logmean component) ✅
- **Note**: diag_logmean uses the full non-padded attention matrix diagonal, consistent with LLM-Check's original implementation

### 7. No Answer Needed / Correctness Model Internals
- **Input**: `input_last_token_hidden[layer]` → shape `[N, hidden_dim]`
- **Centering**: Yes (via PCA mean subtraction)
- **PCA**: Mandatory — SVD on centered data, project to top-k components
- **Scaling**: Optional StandardScaler after PCA
- **Classifier**: Logistic regression on PCA-reduced features
- **Code**: `correctness-model-internals/src/classifying/activations_handler.py:300-311`
- **Extraction field**: `input_last_token_hidden` ✅

### 8. SAE Entities / Do I Know This Entity (ICLR 2025)
- **Input**: `input_last_token_hidden` at entity token or last token → shape `[N, n_layers, hidden_dim]`
- **SAE**: Feed hidden states into pre-trained Sparse Autoencoder to get sparse features
- **Feature**: SAE activation directions as entity recognition indicators
- **Causal**: Can steer model by amplifying/suppressing SAE directions
- **Code**: `sae_entities/utils/activation_cache.py:34-80`
- **Extraction field**: `input_last_token_hidden` ✅ (SAE applied downstream)

---

## Category B: Needs generation (replay forward for generation-side features)

### 9. INSIDE / EigenScore (ICLR 2024)
- **Input**: `gen_last_token_hidden` from middle layer, across **10 sampled generations**
- **Covariance**: Compute covariance matrix across samples + alpha regularization (α=1e-3)
- **Metric**: `EigenScore = mean(log10(singular_values(CovMatrix)))`
- **Output**: Scalar per question
- **Code**: `eigenscore/func/metric.py:154-230`
- **Extraction fields**: `gen_last_token_hidden` (partial — only 1 generation, not 10)
- **Note**: Full reproduction needs 10 sampled generations per question (separate runs)

### 10. Semantic Entropy Probes (SEP)
- **Input**: `gen_last_token_hidden[-1]` (final layer) → shape `[N, hidden_dim]`
- **Classifier**: Logistic regression on raw embeddings
- **Code**: `semantic-entropy-probes/.../huggingface_models.py:339-341`
- **Extraction field**: `gen_last_token_hidden` ✅

### 11. ICR Probe (ACL 2025)
- **Input**: Per-output-token hidden states across all layers + attention weights
- **Computation**: Skewness/entropy for induction head selection → JS divergence
- **Code**: `ICR_Probe/src/icr_score.py:129-267`
- **Extraction fields**: `gen_per_token_hidden_last_layer` + `gen_attn_stats_last` (partial)

### 12. Chain of Embedding (ICLR 2025)
- **Input**: `gen_mean_pool_hidden` all layers → shape `[n_layers+2, hidden_dim]`
- **CoE-Mag**: `||hs[i+1] - hs[i]||₂ / ||hs[-1] - hs[0]||₂` per layer pair
- **CoE-Ang**: Angular differences between consecutive layers
- **Output**: 4 scalar scores (Mag, Ang, R-score, C-score)
- **Code**: `Chain-of-Embedding/score.py:48-105`
- **Extraction field**: `gen_mean_pool_hidden` ✅

### 13. STEP (MLP Scorer)
- **Input**: Hidden states at reasoning step boundaries during generation
- **Modes**: `"last"` / `"layer_mean"` / `"concat"` / `"certain_layer"`
- **Classifier**: 2-layer MLP (hidden_dim → 512 → ReLU → 1)
- **Code**: `STEP/STEP/train_scorer/train_scorer.py:166-200`
- **Extraction fields**: `gen_step_boundary_hidden` + `gen_per_token_hidden_last_layer` ✅

### 14. Gnosis / Can LLMs Predict Own Failures
- **Input**: Full sequence (prompt+response) hidden states + all raw attention maps
- **Attention processing**: FFT spectral analysis (13 features/map) + CNN on raw matrices
- **Classifier**: Stop head with set-transformer pooling → sigmoid
- **Code**: `Gnosis/src/demo.py:127-147`
- **Note**: Raw attention maps NOT stored (too large). Partial coverage via `gen_attn_stats_last`.

### 15. SeaKR (ACL 2025)
- **Input**: Hidden states + logits at EOS token during generation
- **Metric**: `energy_score = logsumexp(logits)`, eigen_score from embedding covariance
- **Code**: `SeaKR/vllm_uncertainty/vllm/model_executor/layers/sampler.py:67,133-139`
- **Extraction fields**: `gen_last_token_hidden` + `gen_logit_stats_last` ✅

---

## Category C: Online / Special methods (cannot pre-extract)

### 16. DoLa (ICLR 2024)
- **Method**: Contrastive decoding — compares logits from different layers at each generation step
- **Why online**: Modifies token-by-token generation process

### 17. ROME (NeurIPS 2022)
- **Method**: Causal tracing to locate factual knowledge + rank-one MLP editing
- **Why special**: Needs causal tracing (corrupted vs clean forward passes) + parameter editing

### 18. MEMIT (ICLR 2023)
- **Method**: Multi-layer batch knowledge editing via least-squares constraints
- **Why special**: Same as ROME — causal analysis + parameter editing

### 19. Self-Routing RAG (arXiv 2025)
- **Status**: Code not released

---

## Extracted Feature Definitions (Semantic Specification)

Model: Qwen2.5-7B-Instruct (28 layers, 28 heads, 4 KV heads, hidden_dim=3584, head_dim=128)
Loaded in bfloat16, stored as float16 (hidden states) / float32 (attention stats) / float64 (logit stats in JSON).

### Prompt-side features (from generate() prefill hooks)

| # | Field | Shape | Semantic Definition | Verified |
|---|-------|-------|---------------------|----------|
| 1 | `input_last_token_hidden` | (30, 3584) | Hidden states at last prompt token. 30 = embed + 28 layers + norm. | ✅ |
| 2 | `input_mean_pool_hidden` | (30, 3584) | Mean of hidden states across all non-padded prompt tokens, per layer. | ✅ |
| 3 | `input_per_head_activation` | (28, 28, 128) | o_proj input at last prompt token, reshaped per-head. | ✅ |
| 4 | `input_logit_stats` | dict (5 scalars) | Stats of logits at last prompt token (predicting first gen token): logsumexp, max_prob, entropy, top5. | ✅ |
| 5 | `input_attn_stats` | (28, 28, 3) | Per-head: [0] skewness of last-prompt-token attention row over prompt, [1] entropy of same row, [2] mean log of attention diagonal over ALL non-padded prompt tokens (not just last token row — matches LLM-Check). | ✅ |
| 6 | `input_attn_value_norms` | (28, 28, prompt_len) | Per-position: L2 norm of (attention_weight × value_vector) at last prompt token. GQA: kv_head = attn_head // 7. | ✅ |

### Generation-side features (from replay forward hooks)

| # | Field | Shape | Semantic Definition | Verified |
|---|-------|-------|---------------------|----------|
| 7 | `gen_text` | string | Decoded generated text, EOS excluded, skip_special_tokens=True. | ✅ |
| 8 | `gen_last_token_hidden` | (30, 3584) | Hidden states at last generated content token (EOS excluded). | ✅ |
| 9 | `gen_mean_pool_hidden` | (30, 3584) | Mean of hidden states across generated content tokens only, per layer. | ✅ |
| 10 | `gen_per_token_hidden_last_layer` | (gen_len, 3584) | Last decoder layer output (pre-norm) at each generated token. | ✅ |
| 11 | `gen_logit_stats_last` | dict (5 scalars) | Stats of logits at last content token position. If generation stopped by EOS, these are the logits that selected EOS. If stopped by max_new_tokens, these are the would-be-next-token logits at truncation. | ✅ |
| 12 | `gen_attn_stats_last` | (28, 28, 3) | Per-head: [0] skewness of last-gen-token attention row over full sequence, [1] entropy of same row, [2] mean log of attention diagonal over ALL tokens in prompt+gen (not just last token row — matches LLM-Check). | ✅ |
| 13 | `gen_step_boundary_hidden` | list of (30, 3584) | All-layer hidden states at tokens where incrementally decoded text ends with `\n\n`. | ✅ |

### Meta fields

| Field | Description |
|-------|-------------|
| `labels` | Original dataset labels (int for classification, float for regression) |
| `texts` | Input texts |
| `gen_texts` | Generated texts |
| `input_seq_lens` | Per-sample prompt lengths (before padding) |
| `gen_lens` | Per-sample generation lengths (EOS excluded) |
| `gen_step_boundary_indices` | Per-sample list of token indices where `\n\n` boundaries occur |

### File structure
```
features/{dataset}/{split}/
├── input_last_token_hidden.pt       (N, 30, 3584) fp16
├── input_mean_pool_hidden.pt        (N, 30, 3584) fp16
├── input_per_head_activation.pt     (N, 28, 28, 128) fp16
├── input_attn_stats.pt              (N, 28, 28, 3) fp32
├── input_attn_value_norms.pt        (N, 28, 28, max_prompt_len) fp16, zero-padded
├── input_logit_stats.json           list of dicts
├── gen_last_token_hidden.pt         (N, 30, 3584) fp16
├── gen_mean_pool_hidden.pt          (N, 30, 3584) fp16
├── gen_per_token_hidden_last_layer.pt  (N, max_gen_len, 3584) fp16, zero-padded
├── gen_logit_stats_last.json        list of dicts
├── gen_attn_stats_last.pt           (N, 28, 28, 3) fp32
├── gen_step_boundary_hidden.pt      list of variable-size tensors
└── meta.json                        labels, texts, gen_texts, seq_lens, gen_lens, boundary_indices
```

### Extraction method
- 2-pass: generate() with prefill hooks → replay model(prompt+gen) with hooks
- self_attn monkey-patched to always return attn_weights (Qwen2 eager mode)
- Attention not retained by upper layers (output_attentions=False at model level)
- Adaptive batch size: starts at 256, halves on OOM
- Checkpoint: skips completed splits (checks meta.json existence)

### Baseline coverage
- **Fully covered (input only)**: Geometry of Truth, ITI, Knowledge Boundary, LID, Attention Satisfies, LLM-Check (partial), No Answer Needed, SAE Entities — 8 baselines
- **Fully covered (input + generation)**: SEP, CoE, STEP, SeaKR — 4 baselines
- **Partially covered**: INSIDE (1 gen, not 10), ICR Probe (stats only), Gnosis (no raw attn)
- **Not covered**: DoLa, ROME, MEMIT (online/special), Self-Routing RAG (no code)
