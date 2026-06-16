"""Debug: print the exact shapes of generate() outputs."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
CACHE_DIR = "/data/jehc223/NIPS2026/extraction/hf_cache"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.float16, device_map="auto",
    attn_implementation="eager", cache_dir=CACHE_DIR,
)
model.eval()

text = "The city of Paris is in France."
inputs = tokenizer(text, return_tensors="pt")
input_ids = inputs["input_ids"].to(model.device)
attention_mask = inputs["attention_mask"].to(model.device)
prompt_len = input_ids.shape[1]

with torch.no_grad():
    out = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=20,
        do_sample=False,
        temperature=1.0,
        top_p=1.0,
        top_k=0,
        output_hidden_states=True,
        output_attentions=True,
        output_scores=True,
        return_dict_in_generate=True,
    )

gen_ids = out.sequences[0]
n_gen = len(gen_ids) - prompt_len
print(f"prompt_len={prompt_len}, n_gen={n_gen}")
print(f"sequences shape: {out.sequences.shape}")

print(f"\nhidden_states: len={len(out.hidden_states)}")
for i, hs in enumerate(out.hidden_states):
    print(f"  [{i}]: {len(hs)} layers, first layer shape={hs[0].shape}")

print(f"\nattentions: len={len(out.attentions)}")
for i, att in enumerate(out.attentions):
    print(f"  [{i}]: {len(att)} layers, first layer shape={att[0].shape}")

print(f"\nscores: len={len(out.scores)}")
if len(out.scores) > 0:
    print(f"  [0] shape={out.scores[0].shape}")

print(f"\ngenerated text: {tokenizer.decode(gen_ids[prompt_len:])}")
