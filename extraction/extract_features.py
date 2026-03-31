"""
Offline feature extraction: 2-pass design.
  Pass 1: model.generate(prompt) — prefill hooks capture all prompt-side features.
          generate() only used for token ids; no gen_out.hidden_states/attentions used.
  Pass 2: model(prompt+gen_ids) — replay forward with hooks for all generation-side features.
          Attention captured via self_attn hook output[1] (monkey-patched to always compute).

Qwen2.5-7B-Instruct, 2 x A100, device_map='auto', attn_implementation='eager'.
All hidden states from explicit hooks. Attention weights from self_attn hook.
"""

import os
import json
import types
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def patch_self_attn_for_hooks(model):
    """Monkey-patch each self_attn to always compute attn_weights internally,
    without enabling model-level attention collection. This way our hooks on
    self_attn can read output[1] (attn_weights) while generate()/model() with
    output_attentions=False won't retain any attention tensors upstream."""
    for layer in model.model.layers:
        attn = layer.self_attn
        if hasattr(attn, "_orig_forward_for_hook"):
            continue
        attn._orig_forward_for_hook = attn.forward.__func__

        def patched_forward(self, *args, **kwargs):
            kwargs["output_attentions"] = True
            return self._orig_forward_for_hook(self, *args, **kwargs)

        attn.forward = types.MethodType(patched_forward, attn)

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


def row_skewness(row):
    row = row.float()
    mu = row.mean()
    c = row - mu
    var = c.pow(2).mean()
    if float(var) <= 1e-20:
        return 0.0
    return float((c.pow(3).mean() / var.sqrt().pow(3)).item())


def row_entropy(row):
    row = row.float().clamp_min(1e-12)
    return float((-(row * row.log()).sum()).item())


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


# ============================================================
# Feature extractor
# ============================================================
class FeatureExtractor:
    def __init__(self, model_name=MODEL_NAME):
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
        self.n_kv_heads = cfg.num_key_value_heads
        self.hidden_dim = cfg.hidden_size
        self.head_dim = getattr(cfg, "head_dim", self.hidden_dim // self.n_heads)
        self.kv_group_size = self.n_heads // self.n_kv_heads
        self.input_device = self.model.get_input_embeddings().weight.device

        print(f"Model: {self.n_layers}L, {self.n_heads}H, {self.n_kv_heads}KV, "
              f"dim={self.hidden_dim}, head={self.head_dim}, input_device={self.input_device}")

        patch_self_attn_for_hooks(self.model)

        self._hooks = []
        self._clear_all()
        self._register_hooks()

    # ----------------------------------------------------------
    # State management
    # ----------------------------------------------------------
    def _clear_all(self):
        self._mode = None          # "generate" or "replay"
        self._forward_idx = -1     # counts model.model forward calls
        self._phase = None         # "prefill", "decode", or "replay"
        self._prompt_len = 0
        self._full_len = 0

        # Prompt-side buffers (filled during generate prefill)
        self._prompt_embed = None
        self._prompt_layers = {}
        self._prompt_norm = None
        self._prompt_oproj = {}
        self._prompt_vproj = {}
        self._prompt_attn_last_rows = {}   # {layer: (n_heads, prompt_len)}
        self._prompt_attn_diag_logmean = {}  # {layer: (n_heads,)}
        self._prefill_logits = None

        # Replay buffers (filled during full-sequence forward)
        self._replay_embed = None
        self._replay_layers = {}
        self._replay_norm = None
        self._replay_attn_last_rows = {}
        self._replay_attn_diag_logmean = {}

    def _clear_replay_only(self):
        self._forward_idx = -1
        self._phase = None
        self._full_len = 0
        self._replay_embed = None
        self._replay_layers = {}
        self._replay_norm = None
        self._replay_attn_last_rows = {}
        self._replay_attn_diag_logmean = {}

    # ----------------------------------------------------------
    # Hooks
    # ----------------------------------------------------------
    def _register_hooks(self):
        # Pre-hook on model backbone to count forwards and set phase
        def model_pre_hook(module, args, kwargs):
            self._forward_idx += 1
            if self._mode == "generate":
                self._phase = "prefill" if self._forward_idx == 0 else "decode"
            elif self._mode == "replay":
                self._phase = "replay"
            else:
                self._phase = None
        self._hooks.append(
            self.model.model.register_forward_pre_hook(model_pre_hook, with_kwargs=True)
        )

        # Embedding hook
        def embed_hook(module, inputs, output):
            if self._phase == "prefill":
                self._prompt_embed = output.detach().cpu()
            elif self._phase == "replay":
                self._replay_embed = output.detach().cpu()
        self._hooks.append(self.model.model.embed_tokens.register_forward_hook(embed_hook))

        # Final norm hook
        def norm_hook(module, inputs, output):
            if self._phase == "prefill":
                self._prompt_norm = output.detach().cpu()
            elif self._phase == "replay":
                self._replay_norm = output.detach().cpu()
        self._hooks.append(self.model.model.norm.register_forward_hook(norm_hook))

        # LM head hook (for input_logit_stats from prefill)
        def lm_head_hook(module, inputs, output):
            if self._phase == "prefill":
                self._prefill_logits = output[0, -1, :].detach().float().cpu()
        self._hooks.append(self.model.lm_head.register_forward_hook(lm_head_hook))

        # Per-layer hooks
        for li, layer in enumerate(self.model.model.layers):
            # Decoder layer output: hidden states
            def layer_hook(module, inputs, output, idx=li):
                hidden = output if torch.is_tensor(output) else output[0]
                if self._phase == "prefill":
                    self._prompt_layers[idx] = hidden.detach().cpu()
                elif self._phase == "replay":
                    self._replay_layers[idx] = hidden.detach().cpu()
            self._hooks.append(layer.register_forward_hook(layer_hook))

            # o_proj input (before projection, per-head activations) — prefill only
            def oproj_hook(module, inputs, output, idx=li):
                if self._phase == "prefill":
                    self._prompt_oproj[idx] = inputs[0].detach().cpu()
            self._hooks.append(layer.self_attn.o_proj.register_forward_hook(oproj_hook))

            # v_proj output (value states for attn_value_norms) — prefill only
            def vproj_hook(module, inputs, output, idx=li):
                if self._phase == "prefill":
                    self._prompt_vproj[idx] = output.detach().cpu()
            self._hooks.append(layer.self_attn.v_proj.register_forward_hook(vproj_hook))

            # self_attn hook: capture attention weights from output[1]
            # Qwen2 eager attention returns (attn_output, attn_weights, past_kv)
            def attn_hook(module, inputs, output, idx=li):
                attn_weights = output[1]
                if attn_weights is None:
                    return
                if self._phase == "prefill":
                    aw = attn_weights[0].detach().float().cpu()  # (n_heads, prompt_len, prompt_len)
                    p = self._prompt_len
                    self._prompt_attn_last_rows[idx] = aw[:, p - 1, :p].contiguous()
                    diag = aw[:, torch.arange(p), torch.arange(p)]
                    self._prompt_attn_diag_logmean[idx] = safe_log(diag).mean(dim=-1).cpu()
                elif self._phase == "replay":
                    aw = attn_weights[0].detach().float().cpu()  # (n_heads, total_len, total_len)
                    t = self._full_len
                    self._replay_attn_last_rows[idx] = aw[:, t - 1, :t].contiguous()
                    diag = aw[:, torch.arange(t), torch.arange(t)]
                    self._replay_attn_diag_logmean[idx] = safe_log(diag).mean(dim=-1).cpu()
            self._hooks.append(layer.self_attn.register_forward_hook(attn_hook))

    # ----------------------------------------------------------
    # State assembly
    # ----------------------------------------------------------
    def _prompt_states(self):
        """[embed, layer_0, ..., layer_{n-1}, norm] — length n_layers+2."""
        states = [self._prompt_embed]
        for i in range(self.n_layers):
            assert i in self._prompt_layers, f"Missing prompt layer {i}"
            states.append(self._prompt_layers[i])
        assert self._prompt_norm is not None, "Missing prompt norm"
        states.append(self._prompt_norm)
        return states

    def _replay_states(self):
        states = [self._replay_embed]
        for i in range(self.n_layers):
            assert i in self._replay_layers, f"Missing replay layer {i}"
            states.append(self._replay_layers[i])
        assert self._replay_norm is not None, "Missing replay norm"
        states.append(self._replay_norm)
        return states

    def _to_input_device(self, batch):
        return (batch["input_ids"].to(self.input_device, non_blocking=True),
                batch["attention_mask"].to(self.input_device, non_blocking=True))

    # ----------------------------------------------------------
    # Main extraction
    # ----------------------------------------------------------
    @torch.inference_mode()
    def extract(self, text):
        batch = self.tokenizer(text, return_tensors="pt", truncation=True,
                               max_length=MAX_SEQ_LEN, padding=False)
        input_ids, attention_mask = self._to_input_device(batch)
        prompt_len = int(input_ids.shape[1])

        # ==============================================================
        # Pass 1: generate() — prefill hooks capture prompt features,
        #         generate gives us token ids
        # ==============================================================
        self._clear_all()
        self._mode = "generate"
        self._prompt_len = prompt_len

        sequences = self.model.generate(
            input_ids=input_ids, attention_mask=attention_mask,
            max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
            return_dict_in_generate=False,
        )

        full_ids = sequences.to(self.input_device)
        raw_gen_ids = full_ids[0, prompt_len:]
        n_gen = int(raw_gen_ids.shape[0])
        gen_text = self.tokenizer.decode(raw_gen_ids, skip_special_tokens=True,
                                         clean_up_tokenization_spaces=False)

        # --- Extract prompt-side features from hooks ---
        prompt_states = self._prompt_states()

        input_last_token_hidden = torch.stack(
            [hs[0, prompt_len - 1, :] for hs in prompt_states]
        ).to(torch.float16)

        input_mean_pool_hidden = torch.stack(
            [hs[0, :prompt_len, :].float().mean(dim=0) for hs in prompt_states]
        ).to(torch.float16)

        input_per_head_activation = torch.empty(
            self.n_layers, self.n_heads, self.head_dim, dtype=torch.float16)
        for li in range(self.n_layers):
            x = self._prompt_oproj[li][0, prompt_len - 1, :]
            input_per_head_activation[li] = x.view(self.n_heads, self.head_dim).to(torch.float16)

        input_logit_stats = compute_logit_stats(self._prefill_logits)

        input_attn_stats = torch.empty(self.n_layers, self.n_heads, 3, dtype=torch.float32)
        for li in range(self.n_layers):
            for hi in range(self.n_heads):
                row = self._prompt_attn_last_rows[li][hi]
                input_attn_stats[li, hi, 0] = row_skewness(row)
                input_attn_stats[li, hi, 1] = row_entropy(row)
                input_attn_stats[li, hi, 2] = self._prompt_attn_diag_logmean[li][hi]

        input_attn_value_norms = torch.empty(
            self.n_layers, self.n_heads, prompt_len, dtype=torch.float16)
        for li in range(self.n_layers):
            v = self._prompt_vproj[li][0, :prompt_len, :]
            v = v.view(prompt_len, self.n_kv_heads, self.head_dim).permute(1, 0, 2).float()
            for hi in range(self.n_heads):
                kv_hi = hi // self.kv_group_size
                aw = self._prompt_attn_last_rows[li][hi].unsqueeze(-1)
                input_attn_value_norms[li, hi] = (aw * v[kv_hi]).norm(dim=-1).to(torch.float16)

        # --- Early return if no generation ---
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

        # ==============================================================
        # Pass 2: model(prompt+gen_ids) — replay for generation features
        #         output_attentions=False; attn from self_attn hook
        # ==============================================================
        self._clear_replay_only()
        self._mode = "replay"
        self._full_len = int(full_ids.shape[1])

        full_mask = torch.ones_like(full_ids, device=self.input_device)
        replay_out = self.model(
            input_ids=full_ids, attention_mask=full_mask,
            use_cache=False, output_attentions=False, output_hidden_states=False,
        )

        replay_states = self._replay_states()
        total_len = int(full_ids.shape[1])

        gen_last_token_hidden = torch.stack(
            [hs[0, total_len - 1, :] for hs in replay_states]
        ).to(torch.float16)

        gen_mean_pool_hidden = torch.stack(
            [hs[0, prompt_len:total_len, :].float().mean(dim=0) for hs in replay_states]
        ).to(torch.float16)

        gen_per_token_hidden_last_layer = self._replay_layers[self.n_layers - 1][
            0, prompt_len:total_len, :
        ].to(torch.float16)

        gen_logit_stats_eos = compute_logit_stats(replay_out.logits[0, total_len - 1, :])

        gen_attn_stats_last = torch.empty(self.n_layers, self.n_heads, 3, dtype=torch.float32)
        for li in range(self.n_layers):
            for hi in range(self.n_heads):
                row = self._replay_attn_last_rows[li][hi]
                gen_attn_stats_last[li, hi, 0] = row_skewness(row)
                gen_attn_stats_last[li, hi, 1] = row_entropy(row)
                gen_attn_stats_last[li, hi, 2] = self._replay_attn_diag_logmean[li][hi]

        step_boundary_indices = []
        for t in range(n_gen):
            partial = self.tokenizer.decode(raw_gen_ids[:t + 1], skip_special_tokens=False,
                                            clean_up_tokenization_spaces=False)
            if partial.endswith("\n\n"):
                step_boundary_indices.append(t)

        gen_step_boundary_hidden = [
            torch.stack([hs[0, prompt_len + t, :] for hs in replay_states]).to(torch.float16)
            for t in step_boundary_indices
        ]

        self._mode = None

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
