"""
Offline feature extraction: 3-step approach (A->B->C).
  Step A: model(prompt) — extract all prompt-side features via hooks + output_attentions
  Step B: model.generate(prompt) — get generated token ids only
  Step C: model(prompt+gen) — extract all generation-side features via hooks + output_attentions

Qwen2.5-7B-Instruct, 2 x A100, device_map='auto', attn_implementation='eager'.
All hidden states captured via explicit hooks (embed_tokens, each layer, final norm)
to avoid any ambiguity in outputs.hidden_states indexing.
"""

import os
import json
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

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
                samples.append({"text": row["statement"], "label": int(row["label"]),
                                "dataset": "geometry_of_truth_cities", "split": split})
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
                    samples.append({"text": text, "label": float(difficulty),
                                    "dataset": "easy2hard_amc", "split": split})
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
                    samples.append({"text": text, "label": label,
                                    "dataset": "metatool_task1", "split": split})
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
                    samples.append({"text": text, "label": int(label),
                                    "dataset": "retrievalqa", "split": split})
    return samples


# ============================================================
# Helpers
# ============================================================
def safe_log(x, eps=1e-12):
    return x.clamp_min(eps).log()


def row_skewness(x):
    x = x.float()
    m = x.mean()
    c = x - m
    var = c.pow(2).mean()
    if var.item() <= 1e-20:
        return torch.tensor(0.0, dtype=torch.float32)
    return (c.pow(3).mean() / var.sqrt().pow(3)).cpu()


def row_entropy(p):
    p = p.float().clamp_min(1e-12)
    return (-(p * p.log()).sum()).cpu()


def compute_logit_stats(logits_1d):
    logits = logits_1d.float()
    probs = torch.softmax(logits, dim=-1)
    topk = torch.topk(logits, k=5, dim=-1)
    return {
        "logsumexp": float(torch.logsumexp(logits, dim=-1).item()),
        "max_prob": float(probs.max().item()),
        "entropy": float((-(probs * safe_log(probs))).sum().item()),
        "top5_values": topk.values.cpu().tolist(),
        "top5_indices": topk.indices.cpu().tolist(),
    }


def compute_attn_stats_from_full_matrix(attentions, n_layers, n_heads, row_idx, col_end):
    """Compute attn stats from full attention matrices (NOT KV-cache decode).
    attentions: list of (1, n_heads, seq_len, seq_len) tensors on CPU.
    row_idx: which query position's attention row to analyze.
    col_end: how many key positions to consider.
    """
    out = torch.zeros(n_layers, n_heads, 3, dtype=torch.float32)
    for li in range(n_layers):
        attn_layer = attentions[li][0].float()  # (n_heads, seq_len, seq_len)
        diag = attn_layer[:, torch.arange(col_end), torch.arange(col_end)]  # (n_heads, col_end)
        for hi in range(n_heads):
            row = attn_layer[hi, row_idx, :col_end]
            out[li, hi, 0] = row_skewness(row)
            out[li, hi, 1] = row_entropy(row)
            out[li, hi, 2] = safe_log(diag[hi]).mean().cpu()
    return out


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
            model_name, torch_dtype=torch.float16, device_map="auto",
            attn_implementation="eager",
        )
        self.model.eval()

        cfg = self.model.config
        self.n_layers = cfg.num_hidden_layers
        self.n_heads = cfg.num_attention_heads
        self.n_kv_heads = getattr(cfg, "num_key_value_heads", self.n_heads)
        self.hidden_dim = cfg.hidden_size
        self.head_dim = getattr(cfg, "head_dim", self.hidden_dim // self.n_heads)
        self.kv_group_size = self.n_heads // self.n_kv_heads
        self.input_device = self.model.get_input_embeddings().weight.device

        print(f"Model: {self.n_layers}L, {self.n_heads}H, {self.n_kv_heads}KV, "
              f"dim={self.hidden_dim}, head={self.head_dim}, input_device={self.input_device}")

        self._layer_hidden = {}
        self._embed_hidden = None
        self._norm_hidden = None
        self._o_proj_inputs = {}
        self._v_proj_outputs = {}
        self._hooks = []
        self._register_hooks()

    def _register_hooks(self):
        def hook_embed(module, input, output):
            self._embed_hidden = output.detach().cpu()
        self._hooks.append(self.model.model.embed_tokens.register_forward_hook(hook_embed))

        for idx, layer in enumerate(self.model.model.layers):
            def hook_layer(module, input, output, i=idx):
                self._layer_hidden[i] = output[0].detach().cpu()
            self._hooks.append(layer.register_forward_hook(hook_layer))

            def hook_oproj(module, input, output, i=idx):
                self._o_proj_inputs[i] = input[0].detach().cpu()
            self._hooks.append(layer.self_attn.o_proj.register_forward_hook(hook_oproj))

            def hook_vproj(module, input, output, i=idx):
                self._v_proj_outputs[i] = output.detach().cpu()
            self._hooks.append(layer.self_attn.v_proj.register_forward_hook(hook_vproj))

        def hook_norm(module, input, output):
            self._norm_hidden = output.detach().cpu()
        self._hooks.append(self.model.model.norm.register_forward_hook(hook_norm))

    def _clear_hooks_data(self):
        self._layer_hidden = {}
        self._embed_hidden = None
        self._norm_hidden = None
        self._o_proj_inputs = {}
        self._v_proj_outputs = {}

    def _get_all_hidden_states(self):
        """Return list: [embed, layer_0, ..., layer_{n-1}, norm]. Length = n_layers + 2."""
        states = [self._embed_hidden]
        for i in range(self.n_layers):
            assert i in self._layer_hidden, f"Missing layer {i} hidden state"
            states.append(self._layer_hidden[i])
        assert self._norm_hidden is not None, "Missing norm hidden state"
        states.append(self._norm_hidden)
        return states

    def _to_input_device(self, input_ids, attention_mask):
        return (input_ids.to(self.input_device, non_blocking=True),
                attention_mask.to(self.input_device, non_blocking=True))

    @torch.no_grad()
    def extract(self, text):
        batch = self.tokenizer(text, return_tensors="pt", truncation=True,
                               max_length=MAX_SEQ_LEN, padding=False)
        input_ids, attention_mask = self._to_input_device(batch["input_ids"], batch["attention_mask"])
        prompt_len = int(input_ids.shape[1])

        # ==========================================================
        # Step A: model(prompt) — all prompt-side features
        # ==========================================================
        self._clear_hooks_data()
        out_a = self.model(
            input_ids=input_ids, attention_mask=attention_mask,
            use_cache=False, output_hidden_states=False, output_attentions=True,
        )

        prompt_hidden = self._get_all_hidden_states()  # n_layers+2 tensors

        input_last_token_hidden = torch.stack(
            [hs[0, prompt_len - 1, :] for hs in prompt_hidden]
        ).to(torch.float16)

        input_mean_pool_hidden = torch.stack(
            [hs[0, :prompt_len, :].float().mean(dim=0) for hs in prompt_hidden]
        ).to(torch.float16)

        input_per_head_activation = torch.empty(
            self.n_layers, self.n_heads, self.head_dim, dtype=torch.float16)
        for li in range(self.n_layers):
            last_tok = self._o_proj_inputs[li][0, prompt_len - 1, :]
            input_per_head_activation[li] = last_tok.view(self.n_heads, self.head_dim).to(torch.float16)

        input_logit_stats = compute_logit_stats(out_a.logits[0, prompt_len - 1, :])

        prompt_attentions = [a.detach().cpu() for a in out_a.attentions]
        input_attn_stats = compute_attn_stats_from_full_matrix(
            prompt_attentions, self.n_layers, self.n_heads,
            row_idx=prompt_len - 1, col_end=prompt_len
        )

        input_attn_value_norms = torch.empty(
            self.n_layers, self.n_heads, prompt_len, dtype=torch.float16)
        for li in range(self.n_layers):
            attn = prompt_attentions[li][0].float()
            v = self._v_proj_outputs[li][0, :prompt_len, :]
            v = v.view(prompt_len, self.n_kv_heads, self.head_dim).permute(1, 0, 2).float()
            for hi in range(self.n_heads):
                kv_hi = hi // self.kv_group_size
                aw = attn[hi, prompt_len - 1, :prompt_len].unsqueeze(-1)
                weighted = aw * v[kv_hi]
                input_attn_value_norms[li, hi] = weighted.norm(dim=-1).to(torch.float16)

        del out_a, prompt_attentions
        torch.cuda.empty_cache()

        # ==========================================================
        # Step B: model.generate(prompt) — token ids only
        # ==========================================================
        gen_out = self.model.generate(
            input_ids=input_ids, attention_mask=attention_mask,
            max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
            return_dict_in_generate=True,
        )
        gen_ids = gen_out.sequences[0]
        raw_gen_token_ids = gen_ids[prompt_len:]
        n_gen = len(raw_gen_token_ids)
        gen_text = self.tokenizer.decode(raw_gen_token_ids, skip_special_tokens=True,
                                         clean_up_tokenization_spaces=False)
        del gen_out
        torch.cuda.empty_cache()

        # ==========================================================
        # Step C: model(prompt+gen) — all generation-side features
        # ==========================================================
        n_hs = self.n_layers + 2
        if n_gen == 0:
            return {
                "input_last_token_hidden": input_last_token_hidden,
                "input_mean_pool_hidden": input_mean_pool_hidden,
                "input_per_head_activation": input_per_head_activation,
                "input_logit_stats": input_logit_stats,
                "input_attn_stats": input_attn_stats,
                "input_attn_value_norms": input_attn_value_norms,
                "input_seq_len": prompt_len,
                "gen_text": gen_text,
                "gen_last_token_hidden": torch.zeros(n_hs, self.hidden_dim, dtype=torch.float16),
                "gen_mean_pool_hidden": torch.zeros(n_hs, self.hidden_dim, dtype=torch.float16),
                "gen_per_token_hidden_last_layer": torch.zeros(0, self.hidden_dim, dtype=torch.float16),
                "gen_logit_stats_eos": {},
                "gen_attn_stats_last": torch.zeros(self.n_layers, self.n_heads, 3, dtype=torch.float32),
                "gen_step_boundary_hidden": [],
                "gen_step_boundary_indices": [],
                "gen_len": 0,
            }

        full_ids = gen_ids.unsqueeze(0).to(self.input_device)
        full_mask = torch.ones_like(full_ids, device=self.input_device)
        total_len = prompt_len + n_gen

        self._clear_hooks_data()
        out_c = self.model(
            input_ids=full_ids, attention_mask=full_mask,
            use_cache=False, output_hidden_states=False, output_attentions=True,
        )

        full_hidden = self._get_all_hidden_states()
        full_attentions = [a.detach().cpu() for a in out_c.attentions]

        gen_last_token_hidden = torch.stack(
            [hs[0, total_len - 1, :] for hs in full_hidden]
        ).to(torch.float16)

        gen_mean_pool_hidden = torch.stack(
            [hs[0, prompt_len:total_len, :].float().mean(dim=0) for hs in full_hidden]
        ).to(torch.float16)

        gen_per_token_hidden_last_layer = self._layer_hidden[self.n_layers - 1][
            0, prompt_len:total_len, :
        ].to(torch.float16)

        gen_logit_stats_eos = compute_logit_stats(out_c.logits[0, total_len - 1, :])

        gen_attn_stats_last = compute_attn_stats_from_full_matrix(
            full_attentions, self.n_layers, self.n_heads,
            row_idx=total_len - 1, col_end=total_len
        )

        step_boundary_indices = []
        for t in range(n_gen):
            partial = self.tokenizer.decode(
                raw_gen_token_ids[:t + 1], skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            if partial.endswith("\n\n"):
                step_boundary_indices.append(t)

        gen_step_boundary_hidden = [
            torch.stack([hs[0, prompt_len + t, :] for hs in full_hidden]).to(torch.float16)
            for t in step_boundary_indices
        ]

        del out_c, full_hidden, full_attentions
        torch.cuda.empty_cache()

        return {
            "input_last_token_hidden": input_last_token_hidden,
            "input_mean_pool_hidden": input_mean_pool_hidden,
            "input_per_head_activation": input_per_head_activation,
            "input_logit_stats": input_logit_stats,
            "input_attn_stats": input_attn_stats,
            "input_attn_value_norms": input_attn_value_norms,
            "input_seq_len": prompt_len,
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

    for field in ["input_last_token_hidden", "input_mean_pool_hidden",
                  "input_per_head_activation", "input_attn_stats",
                  "gen_last_token_hidden", "gen_mean_pool_hidden",
                  "gen_attn_stats_last"]:
        torch.save(torch.stack(results[field]), os.path.join(split_dir, f"{field}.pt"))

    max_sl = max(t.shape[-1] for t in results["input_attn_value_norms"])
    padded = []
    for t in results["input_attn_value_norms"]:
        sl = t.shape[-1]
        if sl < max_sl:
            t = torch.cat([t, torch.zeros(*t.shape[:-1], max_sl - sl, dtype=t.dtype)], dim=-1)
        padded.append(t)
    torch.save(torch.stack(padded), os.path.join(split_dir, "input_attn_value_norms.pt"))

    max_gl = max(t.shape[0] for t in results["gen_per_token_hidden_last_layer"])
    if max_gl > 0:
        padded_gen = []
        for t in results["gen_per_token_hidden_last_layer"]:
            gl = t.shape[0]
            if gl < max_gl:
                t = torch.cat([t, torch.zeros(max_gl - gl, t.shape[-1], dtype=t.dtype)], dim=0)
            padded_gen.append(t)
        torch.save(torch.stack(padded_gen), os.path.join(split_dir, "gen_per_token_hidden_last_layer.pt"))

    torch.save(results["gen_step_boundary_hidden"], os.path.join(split_dir, "gen_step_boundary_hidden.pt"))

    with open(os.path.join(split_dir, "input_logit_stats.json"), "w") as f:
        json.dump(results["input_logit_stats"], f)
    with open(os.path.join(split_dir, "gen_logit_stats_eos.json"), "w") as f:
        json.dump(results["gen_logit_stats_eos"], f)

    meta = {
        "model": model_name, "dataset": dataset, "split": split, "n_samples": n,
        "labels": results["labels"], "texts": results["texts"],
        "gen_texts": results["gen_texts"],
        "input_seq_lens": results["input_seq_lens"],
        "gen_lens": results["gen_lens"],
        "gen_step_boundary_indices": results["gen_step_boundary_indices"],
    }
    with open(os.path.join(split_dir, "meta.json"), "w") as f:
        json.dump(meta, f, ensure_ascii=False)

    total_mb = sum(os.path.getsize(os.path.join(split_dir, f)) / 1024 / 1024
                   for f in os.listdir(split_dir))
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
                if k == "labels":
                    results[k].append(sample["label"])
                elif k == "texts":
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
