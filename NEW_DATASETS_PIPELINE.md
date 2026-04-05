# New Datasets Pipeline — I/O Specification

## Overview

8 new dataset variants, 3 stages each: **Data Prep → Feature Extraction → Method Reproduction**.

All datasets subsampled to ~5K total samples. Max 2 GPUs (2×A100-80GB).

---

## Stage 1: Data Preparation (No GPU)

For each dataset, create standardized splits saved as JSONL at:
`/data/jehc223/NIPS2026/datasets_prepared/{dataset_name}/{split}.jsonl`

Each line: `{"text": "...", "label": ..., "label_multi": [...] (optional)}`

### 1a. E2H AMC 3-class

- **Source**: `datasets/reasoning_difficulty/e2h_amc_3class/{train,eval}.jsonl`
- **Text field**: `d["problem"]` (raw math problem, same as original E2H)
- **Label field**: `d["class_label"]` → int {0, 1, 2}
- **Raw size**: 1,000 train + 2,975 eval = 3,975 total
- **Split plan**: ≤5K so use ALL. Stratified 70/10/20 from train → train/val. eval → test.
  - train: 700, val: 100, test: 200 (from original train, stratified)
  - **Wait** — original train is only 1,000 and eval is 2,975. Better approach:
  - train: 800 (from original train), val: 200 (from original train), test: 2,975 (original eval)
  - This follows the same convention as the original E2H regression dataset.
- **Output**: `datasets_prepared/e2h_amc_3class/{train,val,test}.jsonl`

### 1b. E2H AMC 5-class

- **Source**: `datasets/reasoning_difficulty/e2h_amc_5class/{train,eval}.jsonl`
- **Text field**: `d["problem"]`
- **Label field**: `d["class_label"]` → int {0, 1, 2, 3, 4}
- **Raw size**: 1,000 + 2,975 = 3,975
- **Split plan**: Same as 3-class: train 800, val 200 (from original train, stratified), test 2,975 (original eval)
- **Output**: `datasets_prepared/e2h_amc_5class/{train,val,test}.jsonl`

### 1c. common_claim 3-class

- **Source**: `baseline/geometry-of-truth/datasets/common_claim.csv`
- **Text field**: `row["examples"]` (factual statement)
- **Label**: `row["label"]` → map {"True": 0, "False": 1, "Neither": 2}
- **Raw size**: 20,000 (True: 12,063, False: 4,321, Neither: 3,616)
- **Split plan**: Subsample to 5,000 (stratified on label). Then 70/10/20 stratified split.
  - train: 3,500, val: 500, test: 1,000
- **Output**: `datasets_prepared/common_claim_3class/{train,val,test}.jsonl`

### 1d. When2Call 3-class

- **Source**: `datasets/tool_use_routing/when2call/mcq.jsonl`
- **Text field**: `d["question"]` + tool descriptions. Concatenated as:
  ```
  Query: {question}
  Available tools:
  - {tool1_name}: {tool1_description}
  - {tool2_name}: {tool2_description}
  ...
  ```
  (Tools are JSON strings in `d["tools"]`, parse each to get `name` and `description`)
  Samples with 0 tools: just the query.
- **Label**: `d["correct_answer"]` → map {"tool_call": 0, "cannot_answer": 1, "request_for_info": 2}
- **Raw size**: 3,652 (tool_call: 1295, cannot_answer: 1295, request_for_info: 1062)
- **Split plan**: ≤5K so use ALL. 70/10/20 stratified split.
  - train: 2,556, val: 365, test: 731
- **Output**: `datasets_prepared/when2call_3class/{train,val,test}.jsonl`

### 1e. FAVA binary

- **Source**: `datasets/hallucination/fava/train.jsonl`
- **Text field**: Task prompt + cleaned completion. The prompt tells the model to judge hallucination:
  ```
  Given the following reference text and a generated passage, determine whether the passage contains any hallucination (fabricated, incorrect, or unverifiable information).

  Reference: {d["prompt"]}

  Passage: {cleaned_completion}
  ```
  Cleaning: strip `<delete>...</delete>` content, keep `<mark>...</mark>` content, remove all other XML tags.
- **Label (binary)**: has any hallucination tag (entity/relation/contradictory/unverifiable/invented/subjective) → 1, else → 0
- **Raw size**: 30,073 (halluc: 29,438 = 97.9%, clean: 635 = 2.1%)
- **Split plan**: Subsample to 5,000. Keep ALL 635 clean samples + randomly sample 4,365 hallucinated. Then 70/10/20 stratified split.
  - train: 3,500, val: 500, test: 1,000
  - Class imbalance (12.7% clean) is left as-is — handling it is each method's responsibility, not a data prep concern.
- **Output**: `datasets_prepared/fava_binary/{train,val,test}.jsonl`

### 1f. FAVA multi-label

- **Source**: Same as 1e (same samples, same text, same subsample)
- **Text field**: Same as 1e
- **Label**: 6-dim binary vector `[entity, relation, contradictory, unverifiable, invented, subjective]`
  - Each element: 1 if corresponding tag found (after typo correction), 0 otherwise
  - Typo map: contradictary/contraditory/contradicatory/contradiciary/contrdictory/contrast → contradictory, unvalidatable → unverifiable, inverted → invented, subective/subj → subjective, relational_error/relative → relation
- **Split plan**: Same samples as 1e (shared features)
- **Output**: `datasets_prepared/fava_multilabel/{train,val,test}.jsonl`
  - (Or: same files as fava_binary but with additional `label_multi` field)

### 1g. RAGTruth binary

- **Source**: `datasets/hallucination/ragtruth/{train,test}.jsonl`
- **Text field**: Task prompt + generated output. The prompt tells model to judge hallucination:
  ```
  Given the following task, source material, and a generated response, determine whether the response contains any hallucination.

  Task: {d["query"]}

  Source material: {d["context"]}

  Generated response: {d["output"]}
  ```
- **Label (binary)**: `(d["hallucination_labels_processed"]["evident_conflict"] + d["hallucination_labels_processed"]["baseless_info"]) > 0` → 1, else → 0
- **Raw size**: 15,090 train (halluc: 6,721, clean: 8,369) + 2,700 test (halluc: 943, clean: 1,757)
- **Split plan**: Subsample to ~5,000 total. From train: stratified sample 4,000. Split into train 3,200 + val 800. From test: stratified sample 1,000.
  - train: 3,200, val: 800, test: 1,000
- **Output**: `datasets_prepared/ragtruth_binary/{train,val,test}.jsonl`

### 1h. RAGTruth multi-label

- **Source**: Same as 1g (same samples, same text, same subsample)
- **Text field**: Same as 1g
- **Label**: 2-dim binary vector `[evident_conflict, baseless_info]`
  - Each element: 1 if count > 0, 0 otherwise
- **Split plan**: Same samples as 1g (shared features)
- **Output**: `datasets_prepared/ragtruth_multilabel/{train,val,test}.jsonl`
  - (Or: same files as ragtruth_binary with additional `label_multi` field)

---

## Stage 2: Feature Extraction (GPU, 2×A100)

**Input**: `datasets_prepared/{dataset_name}/{split}.jsonl` from Stage 1
**Script**: New `extract_features_v2.py` (extends original with new loaders)
**Model**: Qwen2.5-7B-Instruct, bfloat16, 2×A100, device_map='auto'
**Method**: Same 2-pass extraction (generate with prefill hooks → replay with hooks)

### What goes into the model (the `text` field):

| Dataset | Text content | Task prompt? |
|---------|-------------|--------------|
| e2h_amc_3class | raw math problem | No |
| e2h_amc_5class | raw math problem | No |
| common_claim_3class | factual statement | No |
| when2call_3class | query + tool descriptions | No (natural format) |
| fava_binary | reference + cleaned completion | Yes (hallucination judgment) |
| fava_multilabel | same as fava_binary | Yes (same prompt, shared features) |
| ragtruth_binary | task + context + output | Yes (hallucination judgment) |
| ragtruth_multilabel | same as ragtruth_binary | Yes (same prompt, shared features) |

### Extraction grouping (shared features):
- fava_binary and fava_multilabel → extract once as `fava`, labels differ only in meta.json
- ragtruth_binary and ragtruth_multilabel → extract once as `ragtruth`, labels differ only in meta.json

So **6 actual extraction runs**: e2h_amc_3class, e2h_amc_5class, common_claim, when2call, fava, ragtruth.

### Output per dataset/split:

```
features/{dataset_name}/{split}/
├── input_last_token_hidden.pt      # (N, n_layers, hidden_dim) float16
├── input_mean_pool_hidden.pt       # (N, n_layers, hidden_dim) float16
├── input_per_head_activation.pt    # (N, n_layers, n_heads, head_dim) float16
├── input_attn_stats.pt             # (N, n_layers, n_heads, 3) float32
├── input_attn_value_norms.pt       # list of (n_layers, n_heads, seq_len_i) float16
├── gen_last_token_hidden.pt        # (N, n_layers, hidden_dim) float16
├── gen_mean_pool_hidden.pt         # (N, n_layers, hidden_dim) float16
├── gen_per_token_hidden_last_layer.pt  # list of (gen_len_i, hidden_dim) float16
├── gen_attn_stats_last.pt          # (N, n_heads, 3) float32
├── gen_step_boundary_hidden.pt     # list of variable-size tensors
├── input_logit_stats.json          # list of dicts
├── gen_logit_stats_last.json       # list of dicts
└── meta.json                       # {model, dataset, split, n_samples, labels, texts, ...}
```

### Disk estimate:
- E2H 3/5-class: ~3975 × ~47 MB/sample = ~187 GB each (AMC math is long text!)
  - **Problem**: this is ~374 GB for both. Too much.
  - **Mitigation**: E2H 3-class and 5-class are the SAME text. Extract once, symlink .pt files, only differ in meta.json labels.
  - So: extract `e2h_amc_multiclass` once (~187 GB), create meta.json variants for 3-class and 5-class.
  - **Actually**: these are same problems as original `easy2hard_amc` which is already extracted (46.9 GB). The TEXT is identical — only the labels differ. So we can reuse the .pt files completely, just create new meta.json with class_label instead of rating.
  - **Wait — user said prompt changes matter.** But for E2H, the raw math problem IS the prompt, same for regression and classification. The model sees the same text. Only the downstream label is different. So .pt files CAN be reused. Only meta.json needs new labels.
  - **Conclusion**: Symlink existing `easy2hard_amc` .pt files → `e2h_amc_3class/` and `e2h_amc_5class/`, with new meta.json. Cost: ~0 GB.
- common_claim: ~5000 × ~13 MB/sample (short text) = ~65 GB
- when2call: ~3652 × ~15 MB/sample = ~55 GB
- fava: ~5000 × ~15 MB/sample = ~75 GB
- ragtruth: ~5000 × ~20 MB/sample (long context) = ~100 GB
- **Total new extraction: ~295 GB** (excluding E2H which is free via symlink)

### Execution plan:
Since max 2 GPUs and model needs 2×A100:
- Run 1: common_claim + when2call (8,652 samples, ~6-8 hours)
- Run 2: fava + ragtruth (10,000 samples, ~8-10 hours)

---

## Stage 3: Val Split + Method Reproduction (CPU/GPU)

### 3a. Create val splits

**Input**: `features/{dataset}/train/` (extracted features)
**Script**: Extended `create_val_split.py`
**Output**: `features/{dataset}/train_sub/` and `features/{dataset}/val_split/`
**Method**: 80/20 stratified split (for classification) from train features

For multi-label (FAVA, RAGTruth): stratify on the binary label (since multi-label stratification is more complex).

### 3b. Run all 12 methods

**Input**: `features/{dataset}/{train_sub,val_split,test}/`
**Script**: Extended `run_all.py`
**Output**: `reproduce/results/all_results_v3.json`

New dataset configs to add to `run_all.py`:

```python
# (train_split, val_split, test_split, is_regression)
"e2h_amc_3class":       ("train_sub", "val_split", "eval",  False),   # 3-class
"e2h_amc_5class":       ("train_sub", "val_split", "eval",  False),   # 5-class
"common_claim_3class":  ("train_sub", "val_split", "test",  False),   # 3-class
"when2call_3class":     ("train_sub", "val_split", "test",  False),   # 3-class
"fava_binary":          ("train_sub", "val_split", "test",  False),   # binary
"fava_multilabel":      ("train_sub", "val_split", "test",  False),   # 6-label
"ragtruth_binary":      ("train_sub", "val_split", "test",  False),   # binary
"ragtruth_multilabel":  ("train_sub", "val_split", "test",  False),   # 2-label
```

**Multi-label evaluation**: For fava_multilabel and ragtruth_multilabel, each method needs to predict/score each label independently. Metrics: per-label AUROC, macro-averaged AUROC, sample-averaged F1.

---

## Resolved Design Decisions

1. **E2H 3/5-class**: Reuse existing .pt features (same text, same model states). Only create new meta.json with class_label. Saves ~187 GB + GPU hours. ✓
2. **FAVA text**: Option A — `Reference: {prompt}\n\nPassage: {cleaned_completion}` (full context with reference). ✓
3. **RAGTruth text length**: median=752 tokens, p95=1646, max=3141. Only 2.0% exceed MAX_SEQ_LEN=2048. Truncation acceptable. ✓
4. **Evaluation metrics**: Report AUROC + Accuracy + F1 for all datasets (already done by existing code). ✓
5. **FAVA class imbalance**: Not a data prep concern. Keep natural distribution (12.7% clean after subsample). Each method handles imbalance itself. ✓
6. **Multi-label methods**: Supervised probes (LR, PCA+LR, KB MLP, ITI, etc.) train per-label independently. Unsupervised scorers (CoE, SeaKR, LLM-Check) produce a single score — use val-set threshold tuning per-label (same as binary, applied to each label column). ✓
