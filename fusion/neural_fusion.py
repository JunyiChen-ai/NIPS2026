"""
Hierarchical Multi-Layer Adapter Fusion.

Architecture:
1. Per-source adapters with learned layer/head attention
2. Auxiliary per-source classification heads
3. Shallow cross-source fusion (concat + LR or 1-layer attention)

Sources (from raw extracted features):
  A. input_last_token_hidden: (N, 30, 3584) — prompt hidden states
  B. input_per_head_activation: (N, 28, 28, 128) — per-head activations
  C. input_attn_value_norms: (N, 28, 28, prompt_len) — attn value norms
  D. input_attn_stats: (N, 28, 28, 3) — attn statistics
  E. gen_last_token_hidden: (N, 30, 3584) — generation hidden states
"""

import os, json, time, warnings, gc
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore")

EXTRACTION_DIR = "/home/junyi/NIPS2026/extraction/features"

DATASETS = {
    "common_claim_3class": {"n_classes": 3, "splits": {"train": "train", "val": "val", "test": "test"}},
    "e2h_amc_3class":      {"n_classes": 3, "splits": {"train": "train_sub", "val": "val_split", "test": "eval"}},
    "e2h_amc_5class":      {"n_classes": 5, "splits": {"train": "train_sub", "val": "val_split", "test": "eval"}},
    "when2call_3class":     {"n_classes": 3, "splits": {"train": "train", "val": "val", "test": "test"}},
}

BASELINES = {
    "common_claim_3class": 0.7576,
    "e2h_amc_3class": 0.8934,
    "e2h_amc_5class": 0.8752,
    "when2call_3class": 0.8741,
}

EMBED_DIM = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class ProbeDataset(Dataset):
    def __init__(self, features_dict, labels):
        self.features = features_dict  # dict of tensor sources
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = {k: v[idx].float() for k, v in self.features.items()}
        return sample, self.labels[idx]


class LayerAttentionAdapter(nn.Module):
    """Adapter for multi-layer hidden states: (L, D) -> (embed_dim,)"""
    def __init__(self, n_layers, hidden_dim, embed_dim=EMBED_DIM):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, embed_dim)
        self.layer_attn = nn.Sequential(
            nn.Linear(embed_dim, 1),
            nn.Softmax(dim=0),
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: (batch, n_layers, hidden_dim)
        h = self.proj(x)  # (batch, n_layers, embed_dim)
        # Layer attention
        attn_weights = self.layer_attn(h)  # (batch, n_layers, 1)
        out = (h * attn_weights).sum(dim=1)  # (batch, embed_dim)
        return self.norm(out)


class HeadAttentionAdapter(nn.Module):
    """Adapter for per-head activations: (n_layers, n_heads, head_dim) -> (embed_dim,)"""
    def __init__(self, n_layers, n_heads, head_dim, embed_dim=EMBED_DIM):
        super().__init__()
        self.head_proj = nn.Linear(head_dim, embed_dim // 2)
        self.head_attn = nn.Sequential(nn.Linear(embed_dim // 2, 1))
        self.layer_attn = nn.Sequential(nn.Linear(embed_dim // 2, 1))
        self.out_proj = nn.Linear(embed_dim // 2, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: (batch, n_layers, n_heads, head_dim)
        B, L, H, D = x.shape
        h = self.head_proj(x)  # (B, L, H, embed_dim//2)
        # Head attention within each layer
        head_w = F.softmax(self.head_attn(h), dim=2)  # (B, L, H, 1)
        layer_h = (h * head_w).sum(dim=2)  # (B, L, embed_dim//2)
        # Layer attention
        layer_w = F.softmax(self.layer_attn(layer_h), dim=1)  # (B, L, 1)
        out = (layer_h * layer_w).sum(dim=1)  # (B, embed_dim//2)
        return self.norm(self.out_proj(out))


class AttnStatsAdapter(nn.Module):
    """Adapter for attention statistics: (n_layers, n_heads, n_stats) -> (embed_dim,)"""
    def __init__(self, n_layers, n_heads, n_stats, embed_dim=EMBED_DIM):
        super().__init__()
        self.proj = nn.Linear(n_stats, embed_dim // 2)
        self.head_attn = nn.Sequential(nn.Linear(embed_dim // 2, 1))
        self.layer_attn = nn.Sequential(nn.Linear(embed_dim // 2, 1))
        self.out_proj = nn.Linear(embed_dim // 2, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, L, H, S = x.shape
        h = self.proj(x)  # (B, L, H, embed_dim//2)
        head_w = F.softmax(self.head_attn(h), dim=2)
        layer_h = (h * head_w).sum(dim=2)  # (B, L, embed_dim//2)
        layer_w = F.softmax(self.layer_attn(layer_h), dim=1)
        out = (layer_h * layer_w).sum(dim=1)
        return self.norm(self.out_proj(out))


class AttnValueNormAdapter(nn.Module):
    """Adapter for attn value norms: (n_layers, n_heads, prompt_len) -> (embed_dim,)
    Max-pool over prompt_len first, then head/layer attention."""
    def __init__(self, n_layers, n_heads, embed_dim=EMBED_DIM):
        super().__init__()
        # After max-pool over prompt_len, each head has a scalar
        self.proj = nn.Linear(1, embed_dim // 2)
        self.head_attn = nn.Sequential(nn.Linear(embed_dim // 2, 1))
        self.layer_attn = nn.Sequential(nn.Linear(embed_dim // 2, 1))
        self.out_proj = nn.Linear(embed_dim // 2, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: (B, n_layers, n_heads, prompt_len)
        x_max = x.max(dim=-1, keepdim=True).values  # (B, L, H, 1)
        h = self.proj(x_max)  # (B, L, H, embed_dim//2)
        head_w = F.softmax(self.head_attn(h), dim=2)
        layer_h = (h * head_w).sum(dim=2)
        layer_w = F.softmax(self.layer_attn(layer_h), dim=1)
        out = (layer_h * layer_w).sum(dim=1)
        return self.norm(self.out_proj(out))


class HierarchicalFusion(nn.Module):
    """Full hierarchical fusion model."""
    def __init__(self, n_classes, n_layers=30, n_attn_layers=28, n_heads=28,
                 hidden_dim=3584, head_dim=128, embed_dim=EMBED_DIM):
        super().__init__()

        # Source A: input hidden states (30 layers × 3584)
        self.input_hidden_adapter = LayerAttentionAdapter(n_layers, hidden_dim, embed_dim)
        self.input_hidden_head = nn.Linear(embed_dim, n_classes)

        # Source B: per-head activations (28 × 28 × 128)
        self.head_act_adapter = HeadAttentionAdapter(n_attn_layers, n_heads, head_dim, embed_dim)
        self.head_act_head = nn.Linear(embed_dim, n_classes)

        # Source C: attn value norms (28 × 28 × prompt_len)
        self.attn_vnorm_adapter = AttnValueNormAdapter(n_attn_layers, n_heads, embed_dim)
        self.attn_vnorm_head = nn.Linear(embed_dim, n_classes)

        # Source D: attn stats (28 × 28 × 3)
        self.attn_stats_adapter = AttnStatsAdapter(n_attn_layers, n_heads, 3, embed_dim)
        self.attn_stats_head = nn.Linear(embed_dim, n_classes)

        # Source E: gen hidden states (30 × 3584)
        self.gen_hidden_adapter = LayerAttentionAdapter(n_layers, hidden_dim, embed_dim)
        self.gen_hidden_head = nn.Linear(embed_dim, n_classes)

        # Fusion: concat 5 embeddings → classifier
        self.fusion_norm = nn.LayerNorm(embed_dim * 5)
        self.fusion_head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(embed_dim * 5, embed_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, n_classes),
        )

        self.n_sources = 5

    def forward(self, batch):
        embeddings = []
        aux_logits = []

        # A: input hidden
        e_a = self.input_hidden_adapter(batch["input_last_token_hidden"])
        embeddings.append(e_a)
        aux_logits.append(self.input_hidden_head(e_a))

        # B: per-head activation
        e_b = self.head_act_adapter(batch["input_per_head_activation"])
        embeddings.append(e_b)
        aux_logits.append(self.head_act_head(e_b))

        # C: attn value norms
        e_c = self.attn_vnorm_adapter(batch["input_attn_value_norms"])
        embeddings.append(e_c)
        aux_logits.append(self.attn_vnorm_head(e_c))

        # D: attn stats
        e_d = self.attn_stats_adapter(batch["input_attn_stats"])
        embeddings.append(e_d)
        aux_logits.append(self.attn_stats_head(e_d))

        # E: gen hidden
        e_e = self.gen_hidden_adapter(batch["gen_last_token_hidden"])
        embeddings.append(e_e)
        aux_logits.append(self.gen_hidden_head(e_e))

        # Fusion
        fused = torch.cat(embeddings, dim=-1)  # (B, embed_dim * 5)
        fused = self.fusion_norm(fused)
        main_logits = self.fusion_head(fused)

        return main_logits, aux_logits


def load_split_data(dataset, split_name):
    """Load raw features for a split."""
    split_dir = os.path.join(EXTRACTION_DIR, dataset, split_name)
    features = {}

    for fname in ["input_last_token_hidden", "input_per_head_activation",
                   "input_attn_value_norms", "input_attn_stats",
                   "gen_last_token_hidden"]:
        path = os.path.join(split_dir, f"{fname}.pt")
        if os.path.exists(path):
            t = torch.load(path, map_location="cpu")
            if isinstance(t, torch.Tensor):
                features[fname] = t.half()  # keep as fp16 to save memory

    with open(os.path.join(split_dir, "meta.json")) as f:
        labels = np.array(json.load(f)["labels"])

    return features, labels


def compute_auroc(y_true, y_prob, n_classes):
    if n_classes == 2:
        return roc_auc_score(y_true, y_prob[:, 1])
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))
    return roc_auc_score(y_bin, y_prob, average="macro", multi_class="ovr")


def train_and_eval(dataset, info):
    n_classes = info["n_classes"]
    splits = info["splits"]
    baseline = BASELINES[dataset]

    print(f"\n{'='*60}")
    print(f"Dataset: {dataset} ({n_classes}-class, baseline={baseline:.4f})")
    print(f"{'='*60}")

    # Load data
    t0 = time.time()
    train_feats, train_labels = load_split_data(dataset, splits["train"])
    val_feats, val_labels = load_split_data(dataset, splits["val"])
    test_feats, test_labels = load_split_data(dataset, splits["test"])
    print(f"Loaded: train={len(train_labels)}, val={len(val_labels)}, test={len(test_labels)}  [{time.time()-t0:.1f}s]")
    print(f"Features: {[(k, v.shape) for k, v in train_feats.items()]}")

    # Create datasets
    train_ds = ProbeDataset(train_feats, train_labels)
    val_ds = ProbeDataset(val_feats, val_labels)
    test_ds = ProbeDataset(test_feats, test_labels)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=0, pin_memory=True)

    # Build model
    model = HierarchicalFusion(n_classes=n_classes).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model params: {n_params:,}")

    # Class weights for imbalanced data
    class_counts = np.bincount(train_labels, minlength=n_classes)
    class_weights = torch.FloatTensor(1.0 / (class_counts + 1)).to(DEVICE)
    class_weights /= class_weights.sum()

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    aux_weight = 0.2

    # Training loop
    best_val_auroc = -1
    best_state = None
    patience = 15
    no_improve = 0

    for epoch in range(100):
        model.train()
        train_loss = 0
        for batch, labels in train_loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            labels = labels.to(DEVICE)

            # Source dropout: randomly zero out one source embedding
            main_logits, aux_logits = model(batch)

            # Main loss
            loss = criterion(main_logits, labels)

            # Auxiliary losses
            for aux_l in aux_logits:
                loss += aux_weight / len(aux_logits) * criterion(aux_l, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()

        # Validation
        model.eval()
        val_probs = []
        val_true = []
        with torch.no_grad():
            for batch, labels in val_loader:
                batch = {k: v.to(DEVICE) for k, v in batch.items()}
                main_logits, _ = model(batch)
                val_probs.append(F.softmax(main_logits, dim=-1).cpu().numpy())
                val_true.append(labels.numpy())

        val_probs = np.concatenate(val_probs)
        val_true = np.concatenate(val_true)
        val_auroc = compute_auroc(val_true, val_probs, n_classes)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}  loss={train_loss/len(train_loader):.4f}  "
                  f"val_AUROC={val_auroc:.4f}  best={best_val_auroc:.4f}")

        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    # Test evaluation
    model.load_state_dict(best_state)
    model.eval()
    test_probs = []
    test_true = []
    with torch.no_grad():
        for batch, labels in test_loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            main_logits, _ = model(batch)
            test_probs.append(F.softmax(main_logits, dim=-1).cpu().numpy())
            test_true.append(labels.numpy())

    test_probs = np.concatenate(test_probs)
    test_true = np.concatenate(test_true)
    test_auroc = compute_auroc(test_true, test_probs, n_classes)
    test_acc = accuracy_score(test_true, test_probs.argmax(axis=1))
    test_f1 = f1_score(test_true, test_probs.argmax(axis=1), average="macro")

    delta = test_auroc - baseline
    status = ">>>" if delta >= 0.03 else (">>>" if delta > 0 else "   ")
    print(f"\n  Test: AUROC={test_auroc:.4f} ({delta:+.4f})  Acc={test_acc:.4f}  F1={test_f1:.4f}")
    print(f"  {'TARGET MET (+3%)' if delta >= 0.03 else 'Not yet +3%'}")

    # Cleanup
    del model, train_ds, val_ds, test_ds
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "auroc": test_auroc,
        "accuracy": test_acc,
        "f1_macro": test_f1,
        "delta": delta,
        "best_val_auroc": best_val_auroc,
        "n_params": n_params,
    }


def main():
    print("=" * 70)
    print("HIERARCHICAL MULTI-LAYER ADAPTER FUSION")
    print(f"Device: {DEVICE}")
    print("=" * 70)

    all_results = {}
    for dataset, info in DATASETS.items():
        all_results[dataset] = train_and_eval(dataset, info)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    all_pass = True
    for ds in DATASETS:
        bl = BASELINES[ds]
        r = all_results[ds]
        delta = r["delta"]
        status = "PASS" if delta >= 0.03 else "FAIL"
        if delta < 0.03:
            all_pass = False
        print(f"  {ds:25s}  baseline={bl:.4f}  ours={r['auroc']:.4f}  delta={delta:+.4f}  [{status}]")

    print(f"\n{'ALL TARGETS MET' if all_pass else 'Some targets not met'}")

    os.makedirs("/home/junyi/NIPS2026/fusion/results", exist_ok=True)
    with open("/home/junyi/NIPS2026/fusion/results/neural_fusion_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)


if __name__ == "__main__":
    main()
