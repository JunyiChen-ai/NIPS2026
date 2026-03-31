"""
Unified offline feature extraction via single generate() call.
Qwen2.5-7B-Instruct, 2 GPUs, per-field storage.

One generate() call per sample extracts BOTH input and generation features:
- The first generation step's hidden_states contains the full prompt hidden states
- Subsequent steps contain per-token generation hidden states
"""

import os
import json
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from scipy.stats import skew as scipy_skew

# ============================================================
# Config
# ============================================================
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
OUTPUT_DIR = "/data/jehc223/NIPS2026/extraction/features"
DATASETS_BASE = "/data/jehc223/NIPS2026/datasets"
MAX_SEQ_LEN = 2048
MAX_NEW_TOKENS = 512

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# Dataset loaders
# ============================================================
def load_geometry_of_truth_cities():
    import csv
    samples = []
    base = os.path.join(DATASETS_BASE, "knowledge_factual/geometry_of_truth/cities")
    for split in ["train", "val"]:
        path = os.path.join(base, f"{split}.csv")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            for row in csv.DictReader(f):
                samples.append({
                    "text": row["statement"],
                    "label": int(row["label"]),
                    "dataset": "geometry_of_truth_cities",
                    "split": split,
                })
    return samples


def load_easy2hard_amc():
    samples = []
    base = os.path.join(DATASETS_BASE, "reasoning_difficulty/easy2hard_bench/e2h_amc")
    for split in ["train", "eval"]:
        path = os.path.join(base, f"{split}.jsonl")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            for line in f:
                d = json.loads(line)
                text = d.get("problem", "")
                difficulty = d.get("rating", None)
                if text and difficulty is not None:
                    samples.append({
                        "text": text,
                        "label": float(difficulty),
                        "dataset": "easy2hard_amc",
                        "split": split,
                    })
    return samples


def load_metatool_task1():
    samples = []
    base = os.path.join(DATASETS_BASE, "tool_use_routing/metatool_task1")
    for split in ["train", "test"]:
        path = os.path.join(base, f"{split}.jsonl")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            for line in f:
                d = json.loads(line)
                text = d.get("query", "")
                label = 1 if d.get("label") == "positive" else 0
                if text:
                    samples.append({
                        "text": text,
                        "label": label,
                        "dataset": "metatool_task1",
                        "split": split,
                    })
    return samples


def load_retrievalqa():
    samples = []
    base = os.path.join(DATASETS_BASE, "retrieval_routing/retrievalqa")
    for split in ["train", "test"]:
        path = os.path.join(base, f"{split}.jsonl")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            for line in f:
                d = json.loads(line)
                text = d.get("question", "")
                label = d.get("param_knowledge_answerable", None)
                if text and label is not None:
                    samples.append({
                        "text": text,
                        "label": int(label),
                        "dataset": "retrievalqa",
                        "split": split,
                    })
    return samples


# ============================================================
# Helpers
# ============================================================
def compute_attn_stats(attentions, n_layers, n_heads, seq_len):
    """Compute per-head attention stats: skewness, entropy, diag_logmean."""
    attn_stats = torch.zeros(n_layers, n_heads, 3, dtype=torch.float32)
    for li in range(min(n_layers, len(attentions))):
        attn_layer = attentions[li][0].float()  # (n_heads, seq_len, seq_len) — move to float, stays on same device
        for hi in range(n_heads):
            attn_row = attn_layer[hi, -1, :seq_len]
            if attn_row.sum() > 0:
                attn_stats[li, hi, 0] = scipy_skew(attn_row.cpu().numpy())
            ar = attn_row.clamp(min=1e-10)
            attn_stats[li, hi, 1] = -(ar * torch.log(ar)).sum().item()
            diag = torch.diagonal(attn_layer[hi, :seq_len, :seq_len])
            attn_stats[li, hi, 2] = torch.log(diag.clamp(min=1e-10)).mean().item()
    return attn_stats


def compute_logit_stats(logits):
    """Compute scalar stats from logits at a single position."""
    logits = logits.float()
    probs = torch.softmax(logits, dim=-1)
    top5 = torch.topk(logits, 5)
    return {
        "logsumexp": torch.logsumexp(logits, dim=-1).item(),
        "max_prob": probs.max().item(),
        "entropy": -(probs * torch.log(probs + 1e-10)).sum().item(),
        "top5_values": top5.values.cpu().tolist(),
        "top5_indices": top5.indices.cpu().tolist(),
    }


# ============================================================
# Feature extractor
# ============================================================
class FeatureExtractor:
    def __init__(self, model_name):
        print(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            attn_implementation="eager",
        )
        self.model.eval()

        config = self.model.config
        self.n_layers = config.num_hidden_layers
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = getattr(config, "num_key_value_heads", self.n_heads)
        self.hidden_dim = config.hidden_size
        self.head_dim = self.hidden_dim // self.n_heads
        self.kv_group_size = self.n_heads // self.n_kv_heads
        print(f"Model: {self.n_layers}L, {self.n_heads}H, {self.n_kv_heads}KV, "
              f"dim={self.hidden_dim}, head={self.head_dim}")

        # Determine which device the first and last layers are on
        self.collect_device = torch.device("cpu")  # always collect to CPU to avoid cross-device issues

        # Hooks for per-head activation (before o_proj) and value states
        self.head_activations = {}
        self.value_states = {}
        self._register_hooks()

    def _register_hooks(self):
        for idx in range(self.n_layers):
            layer = self.model.model.layers[idx]

            def hook_oproj(module, input, output, i=idx):
                # Only capture on first forward (prefill) — don't overwrite
                if i not in self.head_activations:
                    self.head_activations[i] = input[0].detach().cpu()
            layer.self_attn.o_proj.register_forward_hook(hook_oproj)

            def hook_vproj(module, input, output, i=idx):
                if i not in self.value_states:
                    self.value_states[i] = output.detach().cpu()
            layer.self_attn.v_proj.register_forward_hook(hook_vproj)

    def _clear_hooks(self):
        self.head_activations = {}
        self.value_states = {}

    @torch.no_grad()
    def extract(self, text):
        """Single generate() call extracts both input and generation features."""
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True,
            max_length=MAX_SEQ_LEN, padding=False
        )
        input_ids = inputs["input_ids"].to(self.model.device)
        attention_mask = inputs["attention_mask"].to(self.model.device)
        prompt_len = input_ids.shape[1]

        self._clear_hooks()

        # === Single generate() call ===
        gen_out = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            output_hidden_states=True,
            output_attentions=True,
            output_scores=True,
            return_dict_in_generate=True,
        )

        gen_ids = gen_out.sequences[0]
        gen_token_ids = gen_ids[prompt_len:]
        n_gen = len(gen_token_ids)
        gen_text = self.tokenizer.decode(gen_token_ids, skip_special_tokens=True)

        # ================================================================
        # INPUT FEATURES (from first generation step's hidden states)
        # gen_out.hidden_states[0] is a tuple of (n_layers+1) tensors
        # each has shape (1, prompt_len, hidden_dim) — full prompt states
        # ================================================================
        prompt_hs = gen_out.hidden_states[0]  # tuple of (n_layers+1,)

        # input_last_token_hidden: (n_layers+1, hidden_dim)
        input_last_token_hidden = torch.stack(
            [h[0, -1, :].cpu() for h in prompt_hs]
        ).to(torch.float16)

        # input_mean_pool_hidden: (n_layers+1, hidden_dim)
        input_mean_pool_hidden = torch.stack(
            [h[0, :prompt_len, :].float().mean(dim=0).cpu() for h in prompt_hs]
        ).to(torch.float16)

        # input_logit_stats: from first generation step's scores
        # gen_out.scores[0] = logits at first gen step (= logits after seeing full prompt)
        input_logit_stats = compute_logit_stats(gen_out.scores[0][0]) if len(gen_out.scores) > 0 else {}

        # input_attn_stats: from first generation step's attentions
        # gen_out.attentions[0] = attention at first gen step, tuple of n_layers
        # each (1, n_heads, prompt_len+1, prompt_len+1) — includes first gen token
        if len(gen_out.attentions) > 0:
            input_attn_stats = compute_attn_stats(
                gen_out.attentions[0], self.n_layers, self.n_heads, prompt_len
            )
        else:
            input_attn_stats = torch.zeros(self.n_layers, self.n_heads, 3)

        # ================================================================
        # INPUT: per_head_activation + attn_value_norms
        # Hooks captured during prefill (first forward in generate).
        # Hooks only fire once per layer due to `if i not in self.head_activations` guard.
        # ================================================================

        # per_head_activation: (n_layers, n_heads, head_dim)
        input_per_head_activation = torch.zeros(
            self.n_layers, self.n_heads, self.head_dim, dtype=torch.float16
        )
        for li in range(self.n_layers):
            if li in self.head_activations:
                act = self.head_activations[li][0, -1, :]  # already on CPU from hook
                input_per_head_activation[li] = act.reshape(
                    self.n_heads, self.head_dim
                ).to(torch.float16)

        # attn_value_norms: (n_layers, n_heads, prompt_len)
        # Use attention weights from gen_out.attentions[0] (prefill step)
        # and value states from hooks (also prefill)
        input_attn_value_norms = torch.zeros(
            self.n_layers, self.n_heads, prompt_len, dtype=torch.float16
        )
        prefill_attentions = gen_out.attentions[0] if len(gen_out.attentions) > 0 else None
        if prefill_attentions is not None:
            for li in range(self.n_layers):
                if li not in self.value_states or li >= len(prefill_attentions):
                    continue
                # prefill attention shape: (1, n_heads, prompt_len+1, prompt_len+1)
                # we want the prompt portion: last prompt token attending to all prompt tokens
                attn_layer = prefill_attentions[li][0].cpu().float()
                vs = self.value_states[li][0, :prompt_len, :].reshape(
                    prompt_len, self.n_kv_heads, self.head_dim
                ).permute(1, 0, 2)  # (n_kv_heads, prompt_len, head_dim)
                for hi in range(self.n_heads):
                    kv_idx = hi // self.kv_group_size
                    # Use prompt's last token row of attention, only prompt positions
                    aw = attn_layer[hi, prompt_len - 1, :prompt_len].unsqueeze(-1)
                    weighted_v = aw * vs[kv_idx]
                    input_attn_value_norms[li, hi, :] = torch.norm(
                        weighted_v, dim=-1
                    ).to(torch.float16)

        # ================================================================
        # GENERATION FEATURES
        # ================================================================
        if n_gen == 0:
            gen_last_token_hidden = torch.zeros(self.n_layers + 1, self.hidden_dim, dtype=torch.float16)
            gen_mean_pool_hidden = torch.zeros(self.n_layers + 1, self.hidden_dim, dtype=torch.float16)
            gen_per_token_hidden_last_layer = torch.zeros(0, self.hidden_dim, dtype=torch.float16)
            gen_logit_stats_eos = {}
            gen_attn_stats_last = torch.zeros(self.n_layers, self.n_heads, 3)
            gen_step_boundary_hidden = torch.zeros(0, self.n_layers + 1, self.hidden_dim, dtype=torch.float16)
            step_boundary_indices = []
        else:
            # gen_out.hidden_states: tuple of (n_gen+1) elements
            # [0] = prompt step: (n_layers+1) tensors of (1, prompt_len, dim)
            # [1..n_gen] = gen steps: (n_layers+1) tensors of (1, 1, dim)

            # gen_last_token_hidden: all layers at last generated token
            last_hs = gen_out.hidden_states[-1]
            gen_last_token_hidden = torch.stack(
                [h[0, -1, :].cpu() for h in last_hs]
            ).to(torch.float16)

            # gen_per_token_hidden_last_layer + accumulate for mean pool
            gen_per_token_hidden_last_layer = torch.zeros(n_gen, self.hidden_dim, dtype=torch.float16)
            layer_sums = [torch.zeros(self.hidden_dim, dtype=torch.float32) for _ in range(self.n_layers + 1)]

            for t in range(n_gen):
                step_hs = gen_out.hidden_states[t + 1]  # +1 because [0] is prompt
                gen_per_token_hidden_last_layer[t] = step_hs[-1][0, -1, :].cpu().to(torch.float16)
                for li in range(self.n_layers + 1):
                    layer_sums[li] += step_hs[li][0, -1, :].cpu().float()

            gen_mean_pool_hidden = torch.stack(
                [s / n_gen for s in layer_sums]
            ).to(torch.float16)

            # gen_logit_stats_eos
            gen_logit_stats_eos = compute_logit_stats(gen_out.scores[-1][0])

            # gen_attn_stats_last
            if len(gen_out.attentions) > 0:
                last_attn = gen_out.attentions[-1]
                total_seq = prompt_len + n_gen
                gen_attn_stats_last = compute_attn_stats(
                    last_attn, self.n_layers, self.n_heads, total_seq
                )
            else:
                gen_attn_stats_last = torch.zeros(self.n_layers, self.n_heads, 3)

            # Step boundaries (\n\n) for STEP
            step_boundary_indices = []
            for t in range(n_gen):
                partial = self.tokenizer.decode(gen_token_ids[:t + 1], skip_special_tokens=False)
                if partial.endswith("\n\n"):
                    step_boundary_indices.append(t)

            if step_boundary_indices:
                gen_step_boundary_hidden = torch.stack([
                    torch.stack([
                        gen_out.hidden_states[t + 1][li][0, -1, :].cpu()
                        for li in range(self.n_layers + 1)
                    ]) for t in step_boundary_indices
                ]).to(torch.float16)
            else:
                gen_step_boundary_hidden = torch.zeros(
                    0, self.n_layers + 1, self.hidden_dim, dtype=torch.float16
                )

        del gen_out
        torch.cuda.empty_cache()

        return {
            # Input features
            "input_last_token_hidden": input_last_token_hidden,
            "input_mean_pool_hidden": input_mean_pool_hidden,
            "input_per_head_activation": input_per_head_activation,
            "input_logit_stats": input_logit_stats,
            "input_attn_stats": input_attn_stats,
            "input_attn_value_norms": input_attn_value_norms,
            "input_seq_len": prompt_len,
            # Generation features
            "gen_text": gen_text,
            "gen_last_token_hidden": gen_last_token_hidden,
            "gen_mean_pool_hidden": gen_mean_pool_hidden,
            "gen_per_token_hidden_last_layer": gen_per_token_hidden_last_layer,
            "gen_logit_stats_eos": gen_logit_stats_eos,
            "gen_attn_stats_last": gen_attn_stats_last,
            "gen_step_boundary_hidden": gen_step_boundary_hidden,
            "gen_step_boundary_indices": step_boundary_indices,
            "gen_len": n_gen,
        }


# ============================================================
# Save
# ============================================================
def save_split_features(results, out_dir, dataset, split, model_name):
    split_dir = os.path.join(out_dir, dataset, split)
    os.makedirs(split_dir, exist_ok=True)
    n = len(results["labels"])

    # Fixed-shape tensors
    for field in ["input_last_token_hidden", "input_mean_pool_hidden",
                  "input_per_head_activation", "input_attn_stats",
                  "gen_last_token_hidden", "gen_mean_pool_hidden",
                  "gen_attn_stats_last"]:
        torch.save(torch.stack(results[field]),
                   os.path.join(split_dir, f"{field}.pt"))

    # Variable-length: input_attn_value_norms — pad to max
    max_sl = max(t.shape[-1] for t in results["input_attn_value_norms"])
    padded = []
    for t in results["input_attn_value_norms"]:
        sl = t.shape[-1]
        if sl < max_sl:
            t = torch.cat([t, torch.zeros(*t.shape[:-1], max_sl - sl, dtype=t.dtype)], dim=-1)
        padded.append(t)
    torch.save(torch.stack(padded), os.path.join(split_dir, "input_attn_value_norms.pt"))

    # Variable-length: gen_per_token_hidden_last_layer — pad to max
    max_gl = max(t.shape[0] for t in results["gen_per_token_hidden_last_layer"])
    if max_gl > 0:
        padded_gen = []
        for t in results["gen_per_token_hidden_last_layer"]:
            gl = t.shape[0]
            if gl < max_gl:
                t = torch.cat([t, torch.zeros(max_gl - gl, t.shape[-1], dtype=t.dtype)], dim=0)
            padded_gen.append(t)
        torch.save(torch.stack(padded_gen),
                   os.path.join(split_dir, "gen_per_token_hidden_last_layer.pt"))

    # Ragged: gen_step_boundary_hidden — list of tensors
    torch.save(results["gen_step_boundary_hidden"],
               os.path.join(split_dir, "gen_step_boundary_hidden.pt"))

    # JSON
    with open(os.path.join(split_dir, "input_logit_stats.json"), "w") as f:
        json.dump(results["input_logit_stats"], f)
    with open(os.path.join(split_dir, "gen_logit_stats_eos.json"), "w") as f:
        json.dump(results["gen_logit_stats_eos"], f)

    # Meta
    meta = {
        "model": model_name,
        "dataset": dataset,
        "split": split,
        "n_samples": n,
        "labels": results["labels"],
        "texts": results["texts"],
        "gen_texts": results["gen_texts"],
        "input_seq_lens": results["input_seq_lens"],
        "gen_lens": results["gen_lens"],
        "gen_step_boundary_indices": results["gen_step_boundary_indices"],
    }
    with open(os.path.join(split_dir, "meta.json"), "w") as f:
        json.dump(meta, f, ensure_ascii=False)

    total_mb = sum(
        os.path.getsize(os.path.join(split_dir, f)) / 1024 / 1024
        for f in os.listdir(split_dir)
    )
    print(f"  Saved {split_dir}: {n} samples, {total_mb:.1f} MB")


# ============================================================
# Main
# ============================================================
def main():
    print("Loading datasets...")
    all_samples = []
    for name, loader in [
        ("geometry_of_truth_cities", load_geometry_of_truth_cities),
        ("easy2hard_amc", load_easy2hard_amc),
        ("metatool_task1", load_metatool_task1),
        ("retrievalqa", load_retrievalqa),
    ]:
        samples = loader()
        print(f"  {name}: {len(samples)} samples")
        all_samples.extend(samples)
    print(f"Total samples: {len(all_samples)}")

    extractor = FeatureExtractor(MODEL_NAME)

    groups = {}
    for s in all_samples:
        key = (s["dataset"], s["split"])
        groups.setdefault(key, []).append(s)

    for (dataset, split), samples in groups.items():
        print(f"\nProcessing {dataset}/{split}: {len(samples)} samples")

        results = {k: [] for k in [
            "input_last_token_hidden", "input_mean_pool_hidden",
            "input_per_head_activation", "input_logit_stats",
            "input_attn_stats", "input_attn_value_norms",
            "gen_last_token_hidden", "gen_mean_pool_hidden",
            "gen_per_token_hidden_last_layer", "gen_logit_stats_eos",
            "gen_attn_stats_last", "gen_step_boundary_hidden",
            "gen_step_boundary_indices",
            "labels", "texts", "gen_texts", "input_seq_lens", "gen_lens",
        ]}

        for i, sample in enumerate(tqdm(samples, desc=f"{dataset}/{split}")):
            features = extractor.extract(sample["text"])
            for k in results:
                if k in ("labels",):
                    results[k].append(sample["label"])
                elif k in ("texts",):
                    results[k].append(sample["text"])
                elif k == "gen_texts":
                    results[k].append(features["gen_text"])
                elif k == "input_seq_lens":
                    results[k].append(features["input_seq_len"])
                elif k == "gen_lens":
                    results[k].append(features["gen_len"])
                elif k == "gen_step_boundary_indices":
                    results[k].append(features["gen_step_boundary_indices"])
                else:
                    results[k].append(features[k])

        save_split_features(results, OUTPUT_DIR, dataset, split, MODEL_NAME)

    print("\nAll done!")


if __name__ == "__main__":
    main()
