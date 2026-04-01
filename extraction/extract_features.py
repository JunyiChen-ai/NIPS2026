"""
Offline feature extraction: 2-pass design with batching and checkpointing.
  Pass 1: model.generate(batch) — prefill hooks capture prompt-side features.
  Pass 2: model(batch of prompt+gen) — replay forward for generation-side features.

Qwen2.5-7B-Instruct, 2 x A100, device_map='auto', attn_implementation='eager'.
self_attn.forward monkey-patched to always return attn_weights to hooks,
without model-level attention collection (no OOM from retained attention).
"""

import os
import json
import types
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
BATCH_SIZE = 256  # will auto-reduce on OOM

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# Monkey-patch
# ============================================================
def patch_self_attn_for_hooks(model):
    """Wrap each self_attn.forward to force output_attentions=True locally.
    Upper layers (DecoderLayer, Model) keep output_attentions=False so they
    don't collect/retain attention tensors. Preserves Accelerate dispatch."""
    for layer in model.model.layers:
        attn = layer.self_attn
        if hasattr(attn, "_patched_for_hooks"):
            continue
        orig_forward = attn.forward

        def make_wrapper(orig_fn):
            def wrapper(*args, **kwargs):
                kwargs["output_attentions"] = True
                return orig_fn(*args, **kwargs)
            return wrapper

        attn.forward = make_wrapper(orig_forward)
        attn._patched_for_hooks = True


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
            model_name, torch_dtype=torch.bfloat16, device_map="auto",
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
        self.pad_token_id = self.tokenizer.pad_token_id

        print(f"Model: {self.n_layers}L, {self.n_heads}H, {self.n_kv_heads}KV, "
              f"dim={self.hidden_dim}, head={self.head_dim}, input_device={self.input_device}")

        patch_self_attn_for_hooks(self.model)

        self._hooks_handles = []
        self._clear_all()
        self._register_hooks()

    # ----------------------------------------------------------
    # State
    # ----------------------------------------------------------
    def _clear_all(self):
        self._mode = None
        self._forward_idx = -1
        self._phase = None
        self._prompt_lens = []       # set before generate for attn hook
        self._replay_lens = []       # set before replay for attn hook
        # Prefill buffers (batched)
        self._pf_embed = None       # (B, padded_prompt_len, dim)
        self._pf_layers = {}        # {layer: (B, padded_prompt_len, dim)}
        self._pf_norm = None        # (B, padded_prompt_len, dim)
        self._pf_oproj = {}         # {layer: (B, padded_prompt_len, n_heads*head_dim)}
        self._pf_vproj = {}         # {layer: (B, padded_prompt_len, n_kv_heads*head_dim)}
        self._pf_attn_last_rows = {}   # {layer: list of (n_heads, prompt_len_b) per sample}
        self._pf_attn_diag_logmean = {} # {layer: list of (n_heads,) per sample}
        self._pf_logits = None      # (B, vocab) — only last position
        # Replay buffers (batched)
        self._rp_embed = None
        self._rp_layers = {}
        self._rp_norm = None
        self._rp_attn_last_rows = {}   # {layer: list of (n_heads, total_len_b) per sample}
        self._rp_attn_diag_logmean = {} # {layer: list of (n_heads,) per sample}

    def _clear_replay(self):
        self._forward_idx = -1
        self._phase = None
        self._rp_embed = None
        self._rp_layers = {}
        self._rp_norm = None
        self._rp_attn_last_rows = {}
        self._rp_attn_diag_logmean = {}

    # ----------------------------------------------------------
    # Hooks
    # ----------------------------------------------------------
    def _register_hooks(self):
        def model_pre_hook(module, args, kwargs):
            self._forward_idx += 1
            if self._mode == "generate":
                self._phase = "prefill" if self._forward_idx == 0 else "decode"
            elif self._mode == "replay":
                self._phase = "replay"
            else:
                self._phase = None
        self._hooks_handles.append(
            self.model.model.register_forward_pre_hook(model_pre_hook, with_kwargs=True))

        def embed_hook(module, inputs, output):
            if self._phase == "prefill":
                self._pf_embed = output.detach().cpu()
            elif self._phase == "replay":
                self._rp_embed = output.detach().cpu()
        self._hooks_handles.append(
            self.model.model.embed_tokens.register_forward_hook(embed_hook))

        def norm_hook(module, inputs, output):
            if self._phase == "prefill":
                self._pf_norm = output.detach().cpu()
            elif self._phase == "replay":
                self._rp_norm = output.detach().cpu()
        self._hooks_handles.append(
            self.model.model.norm.register_forward_hook(norm_hook))

        def lm_head_hook(module, inputs, output):
            if self._phase == "prefill":
                # Only keep last position logits per sample (B, vocab) to avoid storing full (B, seq, vocab)
                self._pf_logits = output[:, -1, :].detach().float().cpu()
        self._hooks_handles.append(
            self.model.lm_head.register_forward_hook(lm_head_hook))

        for li, layer in enumerate(self.model.model.layers):
            def layer_hook(module, inputs, output, idx=li):
                hidden = output if torch.is_tensor(output) else output[0]
                if self._phase == "prefill":
                    self._pf_layers[idx] = hidden.detach().cpu()
                elif self._phase == "replay":
                    self._rp_layers[idx] = hidden.detach().cpu()
            self._hooks_handles.append(layer.register_forward_hook(layer_hook))

            def oproj_hook(module, inputs, output, idx=li):
                if self._phase == "prefill":
                    self._pf_oproj[idx] = inputs[0].detach().cpu()
            self._hooks_handles.append(
                layer.self_attn.o_proj.register_forward_hook(oproj_hook))

            def vproj_hook(module, inputs, output, idx=li):
                if self._phase == "prefill":
                    self._pf_vproj[idx] = output.detach().cpu()
            self._hooks_handles.append(
                layer.self_attn.v_proj.register_forward_hook(vproj_hook))

            def attn_hook(module, inputs, output, idx=li):
                attn_weights = output[1]  # (B, n_heads, seq, seq)
                if attn_weights is None:
                    return
                aw = attn_weights.detach().float()  # keep on device briefly
                B_aw = aw.shape[0]
                seq = aw.shape[-1]
                if self._phase == "prefill":
                    rows_list = []
                    diag_list = []
                    padded = aw.shape[-1]
                    for b in range(B_aw):
                        pl = self._prompt_lens[b]
                        start = padded - pl
                        last = padded - 1
                        row = aw[b, :, last, start:padded].cpu()  # (n_heads, pl)
                        diag = aw[b, :, torch.arange(start, padded), torch.arange(start, padded)]
                        diag_lm = safe_log(diag).mean(dim=-1).cpu()  # (n_heads,)
                        rows_list.append(row)
                        diag_list.append(diag_lm)
                    self._pf_attn_last_rows[idx] = rows_list
                    self._pf_attn_diag_logmean[idx] = diag_list
                elif self._phase == "replay":
                    rows_list = []
                    diag_list = []
                    for b in range(B_aw):
                        tl = self._replay_lens[b]
                        row = aw[b, :, tl - 1, :tl].cpu()  # (n_heads, tl)
                        diag = aw[b, :, torch.arange(tl), torch.arange(tl)]
                        diag_lm = safe_log(diag).mean(dim=-1).cpu()
                        rows_list.append(row)
                        diag_list.append(diag_lm)
                    self._rp_attn_last_rows[idx] = rows_list
                    self._rp_attn_diag_logmean[idx] = diag_list
                del aw
            self._hooks_handles.append(
                layer.self_attn.register_forward_hook(attn_hook))

    # ----------------------------------------------------------
    # Batched states assembly
    # ----------------------------------------------------------
    def _pf_states(self):
        """[embed, layer_0, ..., layer_{n-1}, norm] each (B, seq, dim)."""
        s = [self._pf_embed]
        for i in range(self.n_layers):
            s.append(self._pf_layers[i])
        s.append(self._pf_norm)
        return s

    def _rp_states(self):
        s = [self._rp_embed]
        for i in range(self.n_layers):
            s.append(self._rp_layers[i])
        s.append(self._rp_norm)
        return s

    # ----------------------------------------------------------
    # Batched extraction
    # ----------------------------------------------------------
    @torch.inference_mode()
    def extract_batch(self, texts):
        """Extract features for a batch of texts."""
        B = len(texts)

        # Tokenize with left-padding
        batch = self.tokenizer(
            texts, return_tensors="pt", truncation=True,
            max_length=MAX_SEQ_LEN, padding=True,
        )
        input_ids = batch["input_ids"].to(self.input_device)
        attention_mask = batch["attention_mask"].to(self.input_device)
        padded_prompt_len = int(input_ids.shape[1])

        # Per-sample prompt lengths from attention mask
        prompt_lens = attention_mask.sum(dim=1).cpu().tolist()

        # ==============================================================
        # Pass 1: generate() — prefill hooks + token ids
        # ==============================================================
        self._clear_all()
        self._mode = "generate"
        self._prompt_lens = [int(pl) for pl in prompt_lens]

        sequences = self.model.generate(
            input_ids=input_ids, attention_mask=attention_mask,
            max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
            return_dict_in_generate=False,
        )

        pf_states = self._pf_states()
        max_total_len = int(sequences.shape[1])

        # Per-sample: extract prompt features + gen token ids
        all_features = []
        all_gen_ids = []
        all_gen_lens = []

        for b in range(B):
            pl = int(prompt_lens[b])
            last_pos = padded_prompt_len - 1
            start_pos = padded_prompt_len - pl
            feat = {}

            # 1. input_last_token_hidden
            feat["input_last_token_hidden"] = torch.stack(
                [hs[b, last_pos, :] for hs in pf_states]
            ).to(torch.float16)

            # 2. input_mean_pool_hidden
            feat["input_mean_pool_hidden"] = torch.stack(
                [hs[b, start_pos:padded_prompt_len, :].float().mean(dim=0) for hs in pf_states]
            ).to(torch.float16)

            # 3. input_per_head_activation
            pha = torch.empty(self.n_layers, self.n_heads, self.head_dim, dtype=torch.float16)
            for li in range(self.n_layers):
                x = self._pf_oproj[li][b, last_pos, :]
                pha[li] = x.view(self.n_heads, self.head_dim).to(torch.float16)
            feat["input_per_head_activation"] = pha

            # 4. input_logit_stats (pf_logits is now (B, vocab))
            feat["input_logit_stats"] = compute_logit_stats(self._pf_logits[b, :])

            # 5. input_attn_stats (from pre-computed last rows + diag)
            ias = torch.empty(self.n_layers, self.n_heads, 3, dtype=torch.float32)
            for li in range(self.n_layers):
                row = self._pf_attn_last_rows[li][b]  # (n_heads, pl)
                dlm = self._pf_attn_diag_logmean[li][b]  # (n_heads,)
                for hi in range(self.n_heads):
                    ias[li, hi, 0] = row_skewness(row[hi])
                    ias[li, hi, 1] = row_entropy(row[hi])
                    ias[li, hi, 2] = float(dlm[hi].item())
            feat["input_attn_stats"] = ias

            # 6. input_attn_value_norms
            iavn = torch.empty(self.n_layers, self.n_heads, pl, dtype=torch.float16)
            for li in range(self.n_layers):
                attn_row = self._pf_attn_last_rows[li][b]  # (n_heads, pl)
                v = self._pf_vproj[li][b, start_pos:padded_prompt_len, :]
                v = v.view(pl, self.n_kv_heads, self.head_dim).permute(1, 0, 2).float()
                for hi in range(self.n_heads):
                    kv_hi = hi // self.kv_group_size
                    aw = attn_row[hi].unsqueeze(-1)  # (pl, 1)
                    iavn[li, hi] = (aw * v[kv_hi]).norm(dim=-1).to(torch.float16)
            feat["input_attn_value_norms"] = iavn
            feat["input_seq_len"] = pl

            # Gen token ids: use generate() output length tracking
            # sequences[b] = [left_pad... prompt... gen_tokens... right_pad...]
            # gen tokens start at padded_prompt_len
            # gen_len = total non-pad length after prompt
            # Fix for EOS==PAD: count from padded_prompt_len to the end of the sequence
            # generate() pads shorter sequences with pad_token_id on the right
            # The actual generated tokens are everything from padded_prompt_len
            # up to (but not including) trailing padding.
            # Since EOS may equal PAD, we use the fact that generate() right-pads
            # AFTER the sequence ends. If a token equals pad_token_id but is followed
            # by a non-pad token, it's a real EOS, not padding.
            gen_part = sequences[b, padded_prompt_len:].cpu()
            gen_len = int(gen_part.shape[0])
            # Strip only TRAILING pad tokens (from the right)
            while gen_len > 0 and int(gen_part[gen_len - 1].item()) == self.pad_token_id:
                gen_len -= 1
            # But if the last real token IS EOS (== pad), we stripped one too many.
            # Check: if gen_len < original and gen_part[gen_len] == pad, was gen_len+1 the EOS?
            # Heuristic: if the original gen_part had ANY non-pad token, the sequence
            # generated at least one token. If gen_len==0 but there were tokens, add back.
            # Safer approach: use sequences length info. generate() with greedy returns
            # all sequences of same length (no per-sequence EOS stopping by default).
            # Actually: Qwen2.5 with do_sample=False still stops at EOS per sequence.
            # The safest way: all tokens up to and including the first EOS are real.
            # Everything after the first EOS is padding.
            gen_part_full = sequences[b, padded_prompt_len:].cpu()
            eos_id = self.tokenizer.eos_token_id
            gen_len = int(gen_part_full.shape[0])
            for t in range(gen_len):
                if int(gen_part_full[t].item()) == eos_id:
                    gen_len = t  # exclude the EOS token
                    break

            raw_gen_ids = gen_part_full[:gen_len]
            gen_text = self.tokenizer.decode(raw_gen_ids, skip_special_tokens=True,
                                             clean_up_tokenization_spaces=False)
            feat["gen_text"] = gen_text
            feat["gen_len"] = gen_len
            all_features.append(feat)
            all_gen_ids.append(raw_gen_ids)
            all_gen_lens.append(gen_len)

        # ==============================================================
        # Pass 2: replay forward on full sequences
        # ==============================================================
        max_gen_len = max(all_gen_lens) if all_gen_lens else 0
        n_hs = self.n_layers + 2

        if max_gen_len == 0:
            # All samples had immediate EOS or no generation.
            # Gen fields stay zeros/empty to avoid semantic confusion.
            for b in range(B):
                all_features[b]["gen_last_token_hidden"] = torch.zeros(n_hs, self.hidden_dim, dtype=torch.float16)
                all_features[b]["gen_mean_pool_hidden"] = torch.zeros(n_hs, self.hidden_dim, dtype=torch.float16)
                all_features[b]["gen_per_token_hidden_last_layer"] = torch.zeros(0, self.hidden_dim, dtype=torch.float16)
                all_features[b]["gen_logit_stats_eos"] = {}
                all_features[b]["gen_attn_stats_last"] = torch.zeros(self.n_layers, self.n_heads, 3, dtype=torch.float32)
                all_features[b]["gen_step_boundary_hidden"] = []
                all_features[b]["gen_step_boundary_indices"] = []
            return all_features

        # Build replay: unpadded_prompt + gen_ids, right-padded
        replay_lens = [int(prompt_lens[b]) + all_gen_lens[b] for b in range(B)]
        max_replay_len = max(replay_lens)

        replay_ids = torch.full((B, max_replay_len), self.pad_token_id, dtype=torch.long)
        replay_mask = torch.zeros(B, max_replay_len, dtype=torch.long)
        for b in range(B):
            pl = int(prompt_lens[b])
            gl = all_gen_lens[b]
            tl = pl + gl
            start_pos = padded_prompt_len - pl
            prompt_tokens = sequences[b, start_pos:padded_prompt_len].cpu()
            gen_tokens = all_gen_ids[b].cpu()
            replay_ids[b, :tl] = torch.cat([prompt_tokens, gen_tokens])
            replay_mask[b, :tl] = 1

        replay_ids = replay_ids.to(self.input_device)
        replay_mask = replay_mask.to(self.input_device)

        self._clear_replay()
        self._mode = "replay"
        self._replay_lens = replay_lens

        replay_out = self.model(
            input_ids=replay_ids, attention_mask=replay_mask,
            use_cache=False, output_attentions=False, output_hidden_states=False,
        )

        rp_states = self._rp_states()

        for b in range(B):
            pl = int(prompt_lens[b])
            gl = all_gen_lens[b]
            tl = pl + gl
            feat = all_features[b]

            if gl == 0:
                # Immediate EOS: gen fields stay zeros/empty
                feat["gen_last_token_hidden"] = torch.zeros(n_hs, self.hidden_dim, dtype=torch.float16)
                feat["gen_mean_pool_hidden"] = torch.zeros(n_hs, self.hidden_dim, dtype=torch.float16)
                feat["gen_per_token_hidden_last_layer"] = torch.zeros(0, self.hidden_dim, dtype=torch.float16)
                feat["gen_logit_stats_eos"] = {}
                feat["gen_attn_stats_last"] = torch.zeros(self.n_layers, self.n_heads, 3, dtype=torch.float32)
                feat["gen_step_boundary_hidden"] = []
                feat["gen_step_boundary_indices"] = []
                continue

            # 8. gen_last_token_hidden
            feat["gen_last_token_hidden"] = torch.stack(
                [hs[b, tl - 1, :] for hs in rp_states]
            ).to(torch.float16)

            # 9. gen_mean_pool_hidden
            feat["gen_mean_pool_hidden"] = torch.stack(
                [hs[b, pl:tl, :].float().mean(dim=0) for hs in rp_states]
            ).to(torch.float16)

            # 10. gen_per_token_hidden_last_layer
            feat["gen_per_token_hidden_last_layer"] = self._rp_layers[self.n_layers - 1][
                b, pl:tl, :
            ].to(torch.float16)

            # 11. gen_logit_stats_eos
            feat["gen_logit_stats_eos"] = compute_logit_stats(replay_out.logits[b, tl - 1, :])

            # 12. gen_attn_stats_last (from pre-computed last rows + diag)
            gas = torch.empty(self.n_layers, self.n_heads, 3, dtype=torch.float32)
            for li in range(self.n_layers):
                row = self._rp_attn_last_rows[li][b]  # (n_heads, tl)
                dlm = self._rp_attn_diag_logmean[li][b]  # (n_heads,)
                for hi in range(self.n_heads):
                    gas[li, hi, 0] = row_skewness(row[hi])
                    gas[li, hi, 1] = row_entropy(row[hi])
                    gas[li, hi, 2] = float(dlm[hi].item())
            feat["gen_attn_stats_last"] = gas

            # 13. gen_step_boundary_hidden
            raw_gen = all_gen_ids[b]
            step_indices = []
            for t in range(gl):
                partial = self.tokenizer.decode(raw_gen[:t + 1], skip_special_tokens=False,
                                                clean_up_tokenization_spaces=False)
                if partial.endswith("\n\n"):
                    step_indices.append(t)
            feat["gen_step_boundary_indices"] = step_indices
            feat["gen_step_boundary_hidden"] = [
                torch.stack([hs[b, pl + t, :] for hs in rp_states]).to(torch.float16)
                for t in step_indices
            ]

        self._mode = None
        del replay_out
        return all_features


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

    # Variable-length: input_attn_value_norms
    max_sl = max(t.shape[-1] for t in results["input_attn_value_norms"])
    padded = []
    for t in results["input_attn_value_norms"]:
        sl = t.shape[-1]
        if sl < max_sl:
            t = torch.cat([t, torch.zeros(*t.shape[:-1], max_sl - sl, dtype=t.dtype)], dim=-1)
        padded.append(t)
    torch.save(torch.stack(padded), os.path.join(split_dir, "input_attn_value_norms.pt"))

    # Variable-length: gen_per_token_hidden_last_layer
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


def is_split_done(out_dir, dataset, split):
    """Checkpoint: check if this split is already extracted."""
    split_dir = os.path.join(out_dir, dataset, split)
    meta_path = os.path.join(split_dir, "meta.json")
    return os.path.exists(meta_path)


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
        if is_split_done(OUTPUT_DIR, dataset, split):
            print(f"\nSkipping {dataset}/{split} (already done)")
            continue

        current_batch_size = BATCH_SIZE
        print(f"\nProcessing {dataset}/{split}: {len(samples)} samples (initial batch_size={current_batch_size})")
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

        sample_idx = 0
        pbar = tqdm(total=len(samples), desc=f"{dataset}/{split}")
        while sample_idx < len(samples):
            batch_end = min(sample_idx + current_batch_size, len(samples))
            batch_samples = samples[sample_idx:batch_end]
            batch_texts = [s["text"] for s in batch_samples]

            try:
                batch_features = extractor.extract_batch(batch_texts)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                extractor._clear_all()
                old_bs = current_batch_size
                if old_bs <= 1:
                    raise RuntimeError(f"OOM even with batch_size=1 at sample {sample_idx}")
                current_batch_size = max(1, old_bs // 2)
                print(f"\n  OOM at batch_size={old_bs}, reducing to {current_batch_size}")
                continue

            for i, (sample, feat) in enumerate(zip(batch_samples, batch_features)):
                results["labels"].append(sample["label"])
                results["texts"].append(sample["text"])
                results["gen_texts"].append(feat["gen_text"])
                results["input_seq_lens"].append(feat["input_seq_len"])
                results["gen_lens"].append(feat["gen_len"])
                results["gen_step_boundary_indices"].append(feat["gen_step_boundary_indices"])
                for k in ["input_last_token_hidden", "input_mean_pool_hidden",
                           "input_per_head_activation", "input_logit_stats",
                           "input_attn_stats", "input_attn_value_norms",
                           "gen_last_token_hidden", "gen_mean_pool_hidden",
                           "gen_per_token_hidden_last_layer", "gen_logit_stats_eos",
                           "gen_attn_stats_last", "gen_step_boundary_hidden"]:
                    results[k].append(feat[k])

            pbar.update(len(batch_samples))
            sample_idx = batch_end

        pbar.close()
        save_split_features(results, OUTPUT_DIR, dataset, split, MODEL_NAME)

    print("\nAll done!")


if __name__ == "__main__":
    main()
