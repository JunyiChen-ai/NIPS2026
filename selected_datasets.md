# Selected Datasets for Internal States Study

## 1. Knowledge — Geometry of Truth / Cities
- **Task**: Binary classification — statement true or false
- **Input**: Statement (e.g., "The city of Krasnodar is in Russia.")
- **Output**: 0/1
- **Path**: `/data/jehc223/NIPS2026/datasets/knowledge_factual/geometry_of_truth/cities/`
- **Split**: train=1,196 / val=300 (torch.randperm, seed=42, 80/20)
- **Source**: Geometry of Truth (COLM 2024)
- **Note**: Input is statement, not query. Label is model-agnostic (objective fact).

## 2. Thinking/Difficulty — Easy2Hard-Bench / AMC
- **Task**: Regression — predict question difficulty [0,1]
- **Input**: Math competition question
- **Output**: Continuous difficulty score (human IRT-based)
- **Path**: `/data/jehc223/NIPS2026/datasets/reasoning_difficulty/easy2hard_bench/e2h_amc/`
- **Split**: train=1,000 / eval=2,975 (official split)
- **Source**: Easy2Hard-Bench (NeurIPS 2024)
- **Note**: Human-derived difficulty, model-agnostic. Regression, not binary "think/don't think".

## 3. Tool Use — MetaTool Task1
- **Task**: Binary classification — does the query need external tools?
- **Input**: User query (e.g., "Check trending discussions on Google Trends")
- **Output**: positive (needs tool) / negative (doesn't need tool)
- **Path**: `/data/jehc223/NIPS2026/datasets/tool_use_routing/metatool_task1/`
- **Split**: train=832 (416/416) / test=208 (104/104) (seed=42, stratified 80/20)
- **Source**: MetaTool (ICLR 2024)
- **Note**: Model-agnostic labels. Balanced classes.

## 4. Retrieval — RetrievalQA
- **Task**: Binary classification — does the query need retrieval?
- **Input**: Question (e.g., "What is Edward Corser's occupation?")
- **Output**: param_knowledge_answerable = 0 (needs retrieval) / 1 (doesn't need)
- **Path**: `/data/jehc223/NIPS2026/datasets/retrieval_routing/retrievalqa/`
- **Split**: train=2,228 (1017/1211) / test=557 (254/303) (seed=42, stratified 80/20)
- **Source**: RetrievalQA (ACL 2024 Findings)
- **Note**: GPT-4 ceiling-based labeling. Approximately model-agnostic.

## Summary Table

| Category | Dataset | Task Type | Train | Test | Balanced? |
|----------|---------|-----------|-------|------|-----------|
| Knowledge | Geometry of Truth / Cities | Binary classification | 1,196 | 300 | ~yes |
| Thinking | Easy2Hard-Bench / AMC | Regression [0,1] | 1,000 | 2,975 | N/A |
| Tool Use | MetaTool Task1 | Binary classification | 832 | 208 | yes |
| Retrieval | RetrievalQA | Binary classification | 2,228 | 557 | ~yes |
