"""
Exp 1b: Per-Example Oracle with RAW LLM features added to candidate pool.

Extends exp1_oracle_complete.py by loading raw internal-state views from
extraction/features/{model}/{ext}/{split}/ and training a per-view classifier,
then mixing those predictions into the same per-example oracle selection.

Run:
    python exp1b_oracle_with_raw.py --model qwen2.5-7b
"""

import os, json, argparse, warnings, gc, time
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

BASE_PROCESSED = "/home/junyi/NIPS2026/reproduce/processed_features"
BASE_EXTRACTION = "/home/junyi/NIPS2026/extraction/features"
BASE_RESULTS = "/home/junyi/NIPS2026/fusion/results"

MC_METHODS = ["lr_probe", "pca_lr", "iti", "kb_mlp", "attn_satisfies", "sep", "step"]
BIN_EXTRA = ["mm_probe", "lid", "llm_check", "seakr"]

ALL_DATASETS = {
    "common_claim_3class": {"n_classes": 3, "ext": "common_claim_3class", "train": "train",     "val": "val",       "test": "test"},
    "e2h_amc_3class":      {"n_classes": 3, "ext": "e2h_amc_3class",      "train": "train_sub", "val": "val_split", "test": "eval"},
    "e2h_amc_5class":      {"n_classes": 5, "ext": "e2h_amc_5class",      "train": "train_sub", "val": "val_split", "test": "eval"},
    "when2call_3class":    {"n_classes": 3, "ext": "when2call_3class",    "train": "train",     "val": "val",       "test": "test"},
    "ragtruth_binary":     {"n_classes": 2, "ext": "ragtruth",            "train": "train",     "val": "val",       "test": "test"},
    "fava_binary":         {"n_classes": 2, "ext": "fava",                "train": "train",     "val": "val",       "test": "test"},
}

# Raw views: (key, filename, category)
# category drives how we flatten each sample.
#   "hidden":   torch tensor [N, L, D]  → flatten to L*D per sample
#   "per_head": torch tensor [N, H1, H2, D] → flatten per-sample
#   "stats":    torch tensor [N, ...]  → flatten (no PCA needed if small)
#   "logit":    json list[N] of dict   → parse to fixed-dim vector
#   "boundary": python list[N] of list of [L,D] tensors → mean-pool into [D]
RAW_VIEWS = [
    ("raw_input_last_tok",     "input_last_token_hidden.pt",       "hidden"),
    ("raw_input_mean_pool",    "input_mean_pool_hidden.pt",        "hidden"),
    ("raw_gen_last_tok",       "gen_last_token_hidden.pt",         "hidden"),
    ("raw_gen_mean_pool",      "gen_mean_pool_hidden.pt",          "hidden"),
    ("raw_input_per_head",     "input_per_head_activation.pt",     "per_head"),
    ("raw_input_attn_valnorms","input_attn_value_norms.pt",        "per_head"),
    ("raw_input_attn_stats",   "input_attn_stats.pt",              "stats"),
    ("raw_gen_attn_stats",     "gen_attn_stats_last.pt",           "stats"),
    ("raw_input_logit_stats",  "input_logit_stats.json",           "logit"),
    ("raw_gen_logit_stats",    "gen_logit_stats_last.json",        "logit"),
    ("raw_gen_step_boundary",  "gen_step_boundary_hidden.pt",      "boundary"),
]


def load_labels(extraction_dir, ext, split):
    with open(os.path.join(extraction_dir, ext, split, "meta.json")) as f:
        return np.array(json.load(f)["labels"])


def compute_auroc(y, p, nc):
    if nc == 2:
        return roc_auc_score(y, p[:, 1])
    yb = label_binarize(y, classes=list(range(nc)))
    return roc_auc_score(yb, p, average="macro", multi_class="ovr")


def _flatten_logit_stats(entries):
    """Turn a list of dicts into a fixed-width numpy array.
       Fields: logsumexp, max_prob, entropy, top5_values (5), top5_indices modulo-bucket (5)."""
    out = np.zeros((len(entries), 13), dtype=np.float32)
    for i, d in enumerate(entries):
        out[i, 0] = d.get("logsumexp", 0.0)
        out[i, 1] = d.get("max_prob", 0.0)
        out[i, 2] = d.get("entropy", 0.0)
        tv = d.get("top5_values", [0.0]*5)[:5]
        out[i, 3:3+len(tv)] = tv
        # Use log(1+idx % 1000) as a coarse hashed signature — purely cheap stat feature.
        ti = d.get("top5_indices", [0]*5)[:5]
        out[i, 8:8+len(ti)] = np.log1p(np.array(ti, dtype=np.float32) % 1000)
    return out


def load_raw_view(extraction_dir, ext, split, filename, category):
    """Return a numpy array [N, D] or None on failure."""
    path = os.path.join(extraction_dir, ext, split, filename)
    if not os.path.exists(path):
        return None

    try:
        if category == "logit":
            with open(path) as f:
                entries = json.load(f)
            return _flatten_logit_stats(entries)

        if category == "boundary":
            obj = torch.load(path, map_location="cpu", weights_only=False)
            # list[N] of list[k] of [L,D] tensor
            pooled = []
            for sample in obj:
                if isinstance(sample, list) and len(sample) > 0:
                    # stack and mean-pool over steps and layers
                    try:
                        steps = [t.float().mean(dim=0) if t.ndim == 2 else t.float().reshape(-1) for t in sample]
                        vec = torch.stack(steps, dim=0).mean(dim=0)
                    except Exception:
                        vec = torch.zeros(3584, dtype=torch.float32)
                elif hasattr(sample, "shape"):
                    vec = sample.float().reshape(-1)
                else:
                    vec = torch.zeros(3584, dtype=torch.float32)
                pooled.append(vec.numpy())
            # pad/truncate to common length
            max_d = max(v.shape[0] for v in pooled)
            arr = np.zeros((len(pooled), max_d), dtype=np.float32)
            for i, v in enumerate(pooled):
                arr[i, :v.shape[0]] = v
            return arr

        # tensor-backed
        t = torch.load(path, map_location="cpu", weights_only=False)
        if not hasattr(t, "shape"):
            return None
        t = t.float()
        if category == "hidden":
            # [N, L, D] → flatten LD
            if t.ndim == 3:
                return t.reshape(t.shape[0], -1).numpy()
            return t.reshape(t.shape[0], -1).numpy()
        if category == "per_head":
            # [N, H1, H2, D] → per-head mean over H1 first to cut size, then flatten
            if t.ndim == 4:
                # reduce along the layer/head axis we choose as H1 (the first non-sample dim)
                t = t.mean(dim=1)  # [N, H2, D]
            return t.reshape(t.shape[0], -1).numpy()
        if category == "stats":
            return t.reshape(t.shape[0], -1).numpy()
    except Exception as e:
        print(f"    [WARN] load_raw_view failed for {path}: {type(e).__name__}: {str(e)[:100]}")
        return None
    return None


def fit_probe_on_features(trva, te, trva_labels, pca_dim=256, max_iter=2000):
    """Standard scaler → (optional PCA) → LogisticRegression.
       Returns predict_proba on the test set, or None on failure."""
    try:
        trva = np.nan_to_num(trva, nan=0.0, posinf=0.0, neginf=0.0)
        te = np.nan_to_num(te, nan=0.0, posinf=0.0, neginf=0.0)
        sc = StandardScaler()
        trva_s = sc.fit_transform(trva)
        te_s = sc.transform(te)
        if trva_s.shape[1] > 512:
            # guard against massive matrices with IncrementalPCA fallback
            try:
                pca = PCA(n_components=pca_dim, random_state=42)
                trva_s = pca.fit_transform(trva_s)
                te_s = pca.transform(te_s)
            except MemoryError:
                pca = IncrementalPCA(n_components=pca_dim, batch_size=256)
                trva_s = pca.fit_transform(trva_s)
                te_s = pca.transform(te_s)
        clf = LogisticRegression(max_iter=max_iter, C=0.1, random_state=42)
        clf.fit(trva_s, trva_labels)
        return clf.predict_proba(te_s)
    except Exception as e:
        print(f"    [WARN] fit_probe_on_features failed: {type(e).__name__}: {str(e)[:100]}")
        return None


def load_baseline_probe(processed_dir, ds_name, method, trva_labels):
    """Replicates exp1_oracle_complete loading for one baseline probe."""
    tr_path = os.path.join(processed_dir, ds_name, method, "train.pt")
    if not os.path.exists(tr_path):
        return None
    try:
        tr = torch.load(tr_path, map_location="cpu").float().numpy()
        va = torch.load(os.path.join(processed_dir, ds_name, method, "val.pt"),  map_location="cpu").float().numpy()
        te = torch.load(os.path.join(processed_dir, ds_name, method, "test.pt"), map_location="cpu").float().numpy()
        if tr.ndim == 1:
            tr, va, te = tr.reshape(-1, 1), va.reshape(-1, 1), te.reshape(-1, 1)
        trva = np.vstack([tr, va])
        return fit_probe_on_features(trva, te, trva_labels)
    except Exception as e:
        print(f"    [WARN] baseline probe {method} failed: {e}")
        return None


def oracle_over_pool(all_preds, te_labels, nc):
    """Per-sample pick the candidate giving highest prob to the correct class."""
    n_te = len(te_labels)
    oracle_probs = np.zeros((n_te, nc))
    winners = []
    correct = 0
    for i in range(n_te):
        true_class = int(te_labels[i])
        best_prob, best_preds, best_key = -1.0, None, None
        for key in all_preds:
            p = all_preds[key][i, true_class]
            if p > best_prob:
                best_prob = p
                best_preds = all_preds[key][i]
                best_key = key
        oracle_probs[i] = best_preds
        winners.append(best_key)
        if np.argmax(best_preds) == true_class:
            correct += 1
    return oracle_probs, correct / n_te, winners


def process_dataset(ds_name, cfg, processed_dir, extraction_dir):
    nc = cfg["n_classes"]
    ext = cfg["ext"]
    probe_dir = os.path.join(processed_dir, ds_name)
    if not os.path.exists(probe_dir):
        print(f"\n[SKIP] {ds_name}: no processed features at {probe_dir}")
        return None

    tr_labels = load_labels(extraction_dir, ext, cfg["train"])
    va_labels = load_labels(extraction_dir, ext, cfg["val"])
    te_labels = load_labels(extraction_dir, ext, cfg["test"])
    trva_labels = np.concatenate([tr_labels, va_labels])

    # --- Baseline probes ---
    methods = MC_METHODS if nc > 2 else MC_METHODS + BIN_EXTRA
    baseline_preds = {}
    for method in methods:
        preds = load_baseline_probe(processed_dir, ds_name, method, trva_labels)
        if preds is not None:
            baseline_preds[method] = preds

    # --- Raw views ---
    raw_preds = {}
    for key, filename, category in RAW_VIEWS:
        t0 = time.time()
        tr_feat = load_raw_view(extraction_dir, ext, cfg["train"], filename, category)
        va_feat = load_raw_view(extraction_dir, ext, cfg["val"],   filename, category)
        te_feat = load_raw_view(extraction_dir, ext, cfg["test"],  filename, category)
        if tr_feat is None or va_feat is None or te_feat is None:
            print(f"    [SKIP] {key}: missing file or load failure")
            continue
        # ensure column widths match across splits
        min_d = min(tr_feat.shape[1], va_feat.shape[1], te_feat.shape[1])
        tr_feat, va_feat, te_feat = tr_feat[:, :min_d], va_feat[:, :min_d], te_feat[:, :min_d]
        trva_feat = np.vstack([tr_feat, va_feat])
        preds = fit_probe_on_features(trva_feat, te_feat, trva_labels)
        if preds is not None:
            raw_preds[key] = preds
        dt = time.time() - t0
        print(f"    {key}: dim={tr_feat.shape[1]:>6}  took {dt:5.1f}s  {'OK' if preds is not None else 'FAIL'}")
        # free memory aggressively on big views
        del tr_feat, va_feat, te_feat, trva_feat
        gc.collect()

    if len(baseline_preds) < 2:
        print(f"[SKIP] {ds_name}: only {len(baseline_preds)} baseline probes loaded")
        return None

    # --- Per-probe AUROC ---
    per_baseline_auroc = {}
    for m, p in baseline_preds.items():
        try: per_baseline_auroc[m] = float(compute_auroc(te_labels, p, nc))
        except: pass
    per_raw_auroc = {}
    for m, p in raw_preds.items():
        try: per_raw_auroc[m] = float(compute_auroc(te_labels, p, nc))
        except: pass

    best_single_auroc = max(per_baseline_auroc.values())
    best_single_method = max(per_baseline_auroc, key=per_baseline_auroc.get)

    # --- Oracle: baseline only (for validation against exp1) ---
    orp_b, acc_b, win_b = oracle_over_pool(baseline_preds, te_labels, nc)
    oracle_baseline_only = float(compute_auroc(te_labels, orp_b, nc))

    # --- Oracle: baseline ∪ raw ---
    combined = {**baseline_preds, **raw_preds}
    orp_c, acc_c, win_c = oracle_over_pool(combined, te_labels, nc)
    oracle_with_raw = float(compute_auroc(te_labels, orp_c, nc))

    raw_win_rate = sum(1 for k in win_c if k.startswith("raw_")) / len(win_c)

    return {
        "n_classes": nc,
        "n_baseline_probes": len(baseline_preds),
        "n_raw_views": len(raw_preds),
        "best_single_auroc": float(best_single_auroc),
        "best_single_method": best_single_method,
        "oracle_auroc_baseline_only": oracle_baseline_only,
        "oracle_accuracy_baseline_only": float(acc_b),
        "oracle_auroc_with_raw": oracle_with_raw,
        "oracle_accuracy_with_raw": float(acc_c),
        "raw_headroom_delta": oracle_with_raw - oracle_baseline_only,
        "raw_view_win_rate": float(raw_win_rate),
        "per_baseline_auroc": {k: round(v, 4) for k, v in sorted(per_baseline_auroc.items(), key=lambda x: -x[1])},
        "per_raw_view_auroc": {k: round(v, 4) for k, v in sorted(per_raw_auroc.items(), key=lambda x: -x[1])},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5-7b",
                    help="Model subdir under extraction/features and processed_features. "
                         "Use '' (empty) to point at the legacy flat layout.")
    args = ap.parse_args()

    if args.model:
        processed_dir  = os.path.join(BASE_PROCESSED,  args.model)
        extraction_dir = os.path.join(BASE_EXTRACTION, args.model)
        results_dir    = os.path.join(BASE_RESULTS,    args.model)
    else:
        processed_dir  = BASE_PROCESSED
        extraction_dir = BASE_EXTRACTION
        results_dir    = BASE_RESULTS

    # Back-compat: if the per-model subdir doesn't exist yet (pre-reorg), fall back to flat.
    os.makedirs(results_dir, exist_ok=True)

    print("=" * 72)
    print(f"EXP 1b: Oracle with Raw LLM Features — model={args.model or '(flat)'}")
    print(f"  processed_dir={processed_dir}")
    print(f"  extraction_dir={extraction_dir}")
    print("=" * 72)

    results = {}
    for ds_name, cfg in ALL_DATASETS.items():
        print(f"\n>>> {ds_name}")
        r = process_dataset(ds_name, cfg, processed_dir, extraction_dir)
        if r is None:
            continue
        results[ds_name] = r
        print(f"  best_single:      {r['best_single_auroc']:.4f} ({r['best_single_method']})")
        print(f"  oracle baseline:  {r['oracle_auroc_baseline_only']:.4f}")
        print(f"  oracle w/ raw:    {r['oracle_auroc_with_raw']:.4f}  (+{r['raw_headroom_delta']*100:+.2f}%)")
        print(f"  raw win rate:     {r['raw_view_win_rate']*100:.1f}%")
        print(f"  n_raw_views used: {r['n_raw_views']}")

    print("\n" + "=" * 72)
    print(f"{'Dataset':24s} {'Best':>7s} {'OracleBL':>9s} {'OracleRaw':>10s} {'Δ':>7s} {'Rawwin':>8s}")
    print("-" * 72)
    for ds, r in results.items():
        print(f"{ds:24s} {r['best_single_auroc']:.4f}  {r['oracle_auroc_baseline_only']:.4f}   "
              f"{r['oracle_auroc_with_raw']:.4f}  {r['raw_headroom_delta']*100:+5.2f}%  {r['raw_view_win_rate']*100:5.1f}%")

    out_path = os.path.join(results_dir, "oracle_with_raw.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
