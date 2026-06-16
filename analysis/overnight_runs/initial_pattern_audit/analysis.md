# Overnight Internal-State Probing Analysis

Generated: `20260510_070745`

## Hypotheses

- **H1**: Dataset-label / latent-task settings should favor input-side probes.
- **H2**: LLM-response correctness settings should favor generation-side probes.
- **H3**: Fusion/oracle headroom should remain positive even under best-single dominance.
- **H4**: Post-generation dominance should have dataset/model exceptions worth cherry-picking.

## Best-Single Winner Counts

| Setting | N | Timing counts | Method counts | Family counts |
|---|---:|---|---|---|
| dataset_label | 12 | `{'input_side': 11, 'generation_side': 1}` | `{'pca_lr': 6, 'iti': 3, 'lr_probe': 2, 'kb_mlp': 1}` | `{'residual_hidden': 9, 'attention_head': 3}` |
| response_correctness_fusion | 30 | `{'generation_side': 25, 'input_side': 5}` | `{'sep': 18, 'step': 4, 'iti': 2, 'seakr': 2, 'attn_satisfies': 1, 'kb_mlp': 1, 'lr_probe': 1, 'pca_lr': 1}` | `{'semantic_uncertainty': 18, 'step_trajectory': 4, 'residual_hidden': 3, 'attention_head': 2, 'sample_consistency': 2, 'attention_flow': 1}` |
| response_correctness_native | 30 | `{'generation_side': 26, 'input_side': 4}` | `{'step': 16, 'sep': 8, 'iti': 2, 'kb_mlp': 2, 'lr_probe': 2}` | `{'step_trajectory': 16, 'semantic_uncertainty': 8, 'residual_hidden': 4, 'attention_head': 2}` |

## Generation vs Input Gap

| Setting | N | Mean gen-input | Median | Positive rate | Min | Max |
|---|---:|---:|---:|---:|---:|---:|
| dataset_label | 12 | -0.0221 | -0.0231 | 8.3% | -0.0564 | 0.0092 |
| response_correctness_fusion | 30 | 0.0798 | 0.0688 | 83.3% | -0.0946 | 0.2760 |
| response_correctness_native | 30 | 0.0888 | 0.0804 | 86.7% | -0.0456 | 0.2597 |

## Fusion / Oracle Headroom

| Setting | N | Mean oracle headroom | Median oracle headroom | Positive rate | Mean v21 delta |
|---|---:|---:|---:|---:|---:|
| dataset_label | 12 | 0.1161 | 0.1186 | 100.0% | 0.0197 |
| response_correctness_fusion | 30 | 0.1884 | 0.1806 | 100.0% | 0.0149 |

## Domain-Level Winner Pattern

### dataset_label

| Domain | N | Timing counts | Method counts |
|---|---:|---|---|
| factual_claim | 2 | `{'input_side': 2}` | `{'lr_probe': 1, 'pca_lr': 1}` |
| factual_hallucination | 2 | `{'input_side': 2}` | `{'iti': 1, 'pca_lr': 1}` |
| math_difficulty | 4 | `{'input_side': 4}` | `{'pca_lr': 4}` |
| rag_hallucination | 2 | `{'input_side': 2}` | `{'iti': 2}` |
| tool_routing | 2 | `{'generation_side': 1, 'input_side': 1}` | `{'kb_mlp': 1, 'lr_probe': 1}` |

### response_correctness_fusion

| Domain | N | Timing counts | Method counts |
|---|---:|---|---|
| commonsense_qa | 3 | `{'generation_side': 2, 'input_side': 1}` | `{'iti': 1, 'sep': 1, 'step': 1}` |
| factual_claim | 3 | `{'generation_side': 3}` | `{'sep': 3}` |
| factual_hallucination | 3 | `{'generation_side': 3}` | `{'sep': 3}` |
| knowledge_qa | 3 | `{'generation_side': 3}` | `{'sep': 3}` |
| math_reasoning | 6 | `{'generation_side': 6}` | `{'sep': 4, 'seakr': 1, 'step': 1}` |
| math_theory_qa | 3 | `{'input_side': 3}` | `{'attn_satisfies': 1, 'lr_probe': 1, 'pca_lr': 1}` |
| rag_hallucination | 3 | `{'generation_side': 2, 'input_side': 1}` | `{'sep': 2, 'iti': 1}` |
| reading_comprehension | 3 | `{'generation_side': 3}` | `{'step': 2, 'seakr': 1}` |
| tool_routing | 3 | `{'generation_side': 3}` | `{'sep': 2, 'kb_mlp': 1}` |

### response_correctness_native

| Domain | N | Timing counts | Method counts |
|---|---:|---|---|
| commonsense_qa | 3 | `{'generation_side': 2, 'input_side': 1}` | `{'sep': 2, 'iti': 1}` |
| factual_claim | 3 | `{'generation_side': 3}` | `{'kb_mlp': 1, 'sep': 1, 'step': 1}` |
| factual_hallucination | 3 | `{'generation_side': 3}` | `{'step': 2, 'sep': 1}` |
| knowledge_qa | 3 | `{'generation_side': 3}` | `{'step': 2, 'kb_mlp': 1}` |
| math_reasoning | 6 | `{'generation_side': 6}` | `{'step': 5, 'sep': 1}` |
| math_theory_qa | 3 | `{'input_side': 2, 'generation_side': 1}` | `{'lr_probe': 2, 'step': 1}` |
| rag_hallucination | 3 | `{'generation_side': 2, 'input_side': 1}` | `{'step': 2, 'iti': 1}` |
| reading_comprehension | 3 | `{'generation_side': 3}` | `{'step': 2, 'sep': 1}` |
| tool_routing | 3 | `{'generation_side': 3}` | `{'sep': 2, 'step': 1}` |

## Top Exceptions

### dataset_label_generation_wins

| Model | Dataset | Domain | Best input | Best generation | Gen-input |
|---|---|---|---|---|---:|
| llama3.1-8b | when2call_3class | tool_routing | lr_probe (0.8732) | kb_mlp (0.8824) | +0.0092 |
| llama3.1-8b | fava_binary | factual_hallucination | pca_lr (0.9926) | kb_mlp (0.9907) | -0.0019 |
| llama3.1-8b | common_claim_3class | factual_claim | pca_lr (0.7621) | kb_mlp (0.7559) | -0.0062 |
| qwen2.5-7b | when2call_3class | tool_routing | lr_probe (0.8640) | kb_mlp (0.8494) | -0.0146 |
| qwen2.5-7b | common_claim_3class | factual_claim | lr_probe (0.7712) | kb_mlp (0.7499) | -0.0213 |
| qwen2.5-7b | fava_binary | factual_hallucination | iti (0.9846) | kb_mlp (0.9632) | -0.0214 |
| llama3.1-8b | ragtruth_binary | rag_hallucination | iti (0.8862) | kb_mlp (0.8615) | -0.0247 |
| llama3.1-8b | e2h_amc_3class | math_difficulty | pca_lr (0.8785) | kb_mlp (0.8488) | -0.0297 |
| qwen2.5-7b | e2h_amc_3class | math_difficulty | pca_lr (0.8937) | kb_mlp (0.8637) | -0.0300 |
| llama3.1-8b | e2h_amc_5class | math_difficulty | pca_lr (0.8576) | kb_mlp (0.8246) | -0.0330 |

### correctness_input_wins_fusion

| Model | Dataset | Domain | Best input | Best generation | Gen-input |
|---|---|---|---|---|---:|
| qwen2.5-7b | theoremqa | math_theory_qa | lr_probe (0.8505) | step (0.7559) | -0.0946 |
| mistral-7b-v0.3 | theoremqa | math_theory_qa | pca_lr (0.7337) | sep (0.7089) | -0.0248 |
| mistral-7b-v0.3 | ragtruth | rag_hallucination | iti (0.8522) | kb_mlp (0.8360) | -0.0162 |
| llama3.1-8b | theoremqa | math_theory_qa | attn_satisfies (0.7502) | step (0.7420) | -0.0082 |
| mistral-7b-v0.3 | commonsenseqa | commonsense_qa | iti (0.7277) | sep (0.7221) | -0.0056 |
| mistral-7b-v0.3 | mmlu | knowledge_qa | attn_satisfies (0.7399) | sep (0.7402) | +0.0003 |
| mistral-7b-v0.3 | common_claim_3class | factual_claim | pca_lr (0.8160) | sep (0.8258) | +0.0098 |
| mistral-7b-v0.3 | math | math_reasoning | mm_probe (0.8008) | step (0.8159) | +0.0151 |
| mistral-7b-v0.3 | when2call_3class | tool_routing | attn_satisfies (0.7024) | kb_mlp (0.7188) | +0.0164 |
| llama3.1-8b | gsm8k | math_reasoning | pca_lr (0.7412) | seakr (0.7614) | +0.0202 |

### correctness_input_wins_native

| Model | Dataset | Domain | Best input | Best generation | Gen-input |
|---|---|---|---|---|---:|
| mistral-7b-v0.3 | theoremqa | math_theory_qa | lr_probe (0.7264) | coe (0.6807) | -0.0456 |
| qwen2.5-7b | theoremqa | math_theory_qa | lr_probe (0.7717) | step (0.7299) | -0.0418 |
| mistral-7b-v0.3 | ragtruth | rag_hallucination | iti (0.8461) | kb_mlp (0.8289) | -0.0172 |
| mistral-7b-v0.3 | commonsenseqa | commonsense_qa | iti (0.7442) | sep (0.7382) | -0.0060 |
| qwen2.5-7b | mmlu | knowledge_qa | pca_lr (0.7224) | step (0.7265) | +0.0041 |
| mistral-7b-v0.3 | common_claim_3class | factual_claim | pca_lr (0.8179) | kb_mlp (0.8256) | +0.0077 |
| mistral-7b-v0.3 | when2call_3class | tool_routing | lr_probe (0.7096) | step (0.7180) | +0.0084 |
| mistral-7b-v0.3 | mmlu | knowledge_qa | pca_lr (0.7304) | kb_mlp (0.7527) | +0.0223 |
| mistral-7b-v0.3 | math | math_reasoning | lr_probe (0.7882) | step (0.8257) | +0.0374 |
| mistral-7b-v0.3 | fava | factual_hallucination | lr_probe (0.9196) | step (0.9628) | +0.0432 |
