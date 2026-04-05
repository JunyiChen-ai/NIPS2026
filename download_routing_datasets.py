"""
Download routing-decision datasets (model-agnostic + ceiling-based).
Organized by category under /data/jehc223/NIPS2026/datasets/
"""
import os
import json
from datasets import load_dataset

BASE = '/data/jehc223/NIPS2026/datasets'

def save_split(dataset, out_dir, split_name, fmt='jsonl'):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f'{split_name}.{fmt}')
    if fmt == 'jsonl':
        with open(path, 'w') as f:
            for row in dataset:
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
    print(f"  {split_name}: {len(dataset)} samples -> {path}")

# ============================================================
# Category 1: Model-Agnostic
# ============================================================

# --- Easy2Hard-Bench (human subsets with train splits) ---
print("=" * 60)
print("Easy2Hard-Bench (human difficulty, model-agnostic)")
print("=" * 60)
e2h_dir = os.path.join(BASE, 'reasoning_difficulty', 'easy2hard_bench')

for config in ['E2H-AMC', 'E2H-Codeforces', 'E2H-Lichess']:
    print(f"\n[{config}]")
    ds = load_dataset('furonghuang-lab/Easy2Hard-Bench', config, trust_remote_code=True)
    out = os.path.join(e2h_dir, config.lower().replace('-', '_'))
    for split_name in ds:
        save_split(ds[split_name], out, split_name)

# Also download eval-only subsets (LLM-based difficulty, ceiling-based)
print("\nEasy2Hard-Bench (LLM-based difficulty, ceiling-based)")
e2h_llm_dir = os.path.join(BASE, 'reasoning_difficulty', 'easy2hard_bench_llm')
for config in ['E2H-GSM8K', 'E2H-ARC', 'E2H-Winogrande']:
    print(f"\n[{config}]")
    ds = load_dataset('furonghuang-lab/Easy2Hard-Bench', config, trust_remote_code=True)
    out = os.path.join(e2h_llm_dir, config.lower().replace('-', '_'))
    for split_name in ds:
        save_split(ds[split_name], out, split_name)

# --- When2Call (tool use routing) ---
print("\n" + "=" * 60)
print("When2Call (tool use necessity, model-agnostic)")
print("=" * 60)
w2c_dir = os.path.join(BASE, 'tool_use_routing', 'when2call')
ds = load_dataset('nvidia/When2Call', trust_remote_code=True)
for split_name in ds:
    save_split(ds[split_name], w2c_dir, split_name)

# --- MetaTool (tool awareness) ---
print("\n" + "=" * 60)
print("MetaTool (tool awareness, model-agnostic)")
print("=" * 60)
mt_dir = os.path.join(BASE, 'tool_use_routing', 'metatool')
try:
    ds = load_dataset('HowieHwong/MetaTool', trust_remote_code=True)
    for split_name in ds:
        save_split(ds[split_name], mt_dir, split_name)
except Exception as e:
    print(f"  MetaTool HF download failed: {e}")
    print("  Will try GitHub download separately")

# ============================================================
# Category 2: Ceiling-based (approximately model-agnostic)
# ============================================================

# --- RetrievalQA ---
print("\n" + "=" * 60)
print("RetrievalQA (retrieval necessity, GPT-4 ceiling)")
print("=" * 60)
rqa_dir = os.path.join(BASE, 'retrieval_routing', 'retrievalqa')
ds = load_dataset('zihanz/RetrievalQA', trust_remote_code=True)
for split_name in ds:
    save_split(ds[split_name], rqa_dir, split_name)

# --- Self-RAG training data ---
print("\n" + "=" * 60)
print("Self-RAG (retrieval reflection tokens, GPT-4 semantic)")
print("=" * 60)
srag_dir = os.path.join(BASE, 'retrieval_routing', 'self_rag')
try:
    ds = load_dataset('selfrag/selfrag_train_data', trust_remote_code=True)
    for split_name in ds:
        save_split(ds[split_name], srag_dir, split_name)
except Exception as e:
    print(f"  Self-RAG download failed: {e}")
    print("  May need manual download from GitHub")

print("\n" + "=" * 60)
print("All downloads complete!")
print("=" * 60)
