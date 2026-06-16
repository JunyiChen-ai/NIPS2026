"""Save HaloScope processed features (plug-in contract: train/val/test.pt + meta.json).

HaloScope (Du, Xiao, Li — NeurIPS 2024, arXiv:2409.17504) scores each sample by
its projection onto the top singular subspace of (unlabeled) generation
embeddings. Reference implementation (hal_det_llama.py):

    centered = embed_generated[:, layer, :] - mean
    _, sin_value, V_p = torch.linalg.svd(centered)
    projection = sin_value[:k] * V_p[:k, :].T          # weighted_svd variant
    scores = np.mean(centered @ projection, -1)         # signed mean projection

The original selects (layer, k, sign) by AUROC on a labeled validation split
and collapses the projection to a scalar. Here we instead save the signed
sigma-weighted projection onto each of the top max_k singular directions for
every layer; any of HaloScope's signed scalar scores is a linear function of
these columns, so the scaffold's generic adapter (StandardScaler -> PCA -> LR)
subsumes the original (layer, k, sign) selection. The subspaces are fit on the
train split only and use no labels, matching the unsupervised character of the
method.

Raw input: gen_last_token_hidden.pt (N x n_layers x hidden_dim).
For the new setting the raw tensors live on the B2 remote; pass --download to
fetch each file into the expected local path and delete it afterwards.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fusion.settings import get_config

RAW_FILE = "gen_last_token_hidden.pt"
B2_PREFIX = "b2:junyi-data/NIPS2026/extraction/features"


def haloscope_scores(train_x, ks):
    """Fit per-layer SVD subspaces on train; return scorer(x) -> N x (L*len(ks))."""
    n_layers = train_x.shape[1]
    projections = []  # (layer, k) -> hidden_dim x k weighted projection
    means = []
    for layer in range(n_layers):
        feats = train_x[:, layer, :]
        mu = feats.mean(dim=0)
        centered = feats - mu
        # Randomized SVD: only the top max(ks) right singular vectors are needed.
        q = min(max(ks) + 8, *centered.shape)
        _, s, v = torch.svd_lowrank(centered, q=q)
        means.append(mu)
        projections.append([(s[:k] * v[:, :k]) for k in ks])

    def score(x):
        cols = []
        for layer in range(n_layers):
            centered = x[:, layer, :] - means[layer]
            for proj in projections[layer]:
                cols.append((centered @ proj).mean(dim=1))
        return torch.stack(cols, dim=1)

    return score


def fetch_raw_if_needed(cfg, model, dataset, download):
    """New setting: ensure the 'all' raw tensor exists locally; return cleanup path."""
    if cfg.name != "new":
        return None
    local = cfg.base_extraction / model / dataset / "all" / RAW_FILE
    if local.exists():
        return None
    if not download:
        raise FileNotFoundError(f"{local} missing; rerun with --download")
    remote = f"{B2_PREFIX}/{model}/{dataset}/all/{RAW_FILE}"
    local.parent.mkdir(parents=True, exist_ok=True)
    print(f"    downloading {remote}")
    subprocess.run(
        ["rclone", "copyto", remote, str(local), "--retries", "5"], check=True
    )
    return local


def process_dataset(cfg, model, dataset, ks, download, force):
    out_dir = cfg.base_processed / model / dataset / "haloscope"
    if not force and all((out_dir / f"{s}.pt").exists() for s in ["train", "val", "test"]):
        print(f"  {dataset}: already done")
        return "skipped"

    cleanup = fetch_raw_if_needed(cfg, model, dataset, download)
    try:
        splits = {}
        for split in ["train", "val", "test"]:
            x = cfg.raw_view(model, dataset, split, RAW_FILE).float()
            splits[split] = x
    except FileNotFoundError as e:
        print(f"  {dataset}: missing raw features ({e})")
        return "missing"

    t0 = time.time()
    score = haloscope_scores(splits["train"], ks)
    out_dir.mkdir(parents=True, exist_ok=True)
    for split, x in splits.items():
        torch.save(score(x).to(torch.float32), out_dir / f"{split}.pt")
    meta = {
        "method": "haloscope",
        "source": RAW_FILE,
        "ks": ks,
        "n_layers": int(splits["train"].shape[1]),
        "shape": f"N x (n_layers * {len(ks)})",
        "desc": "signed HaloScope membership scores (sigma-weighted top-k SVD "
                "subspace of train-split gen embeddings) per layer per k",
        "reference": "Du, Xiao, Li. HaloScope. NeurIPS 2024. arXiv:2409.17504",
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"  {dataset}: saved {splits['train'].shape[0]}/{splits['val'].shape[0]}/"
          f"{splits['test'].shape[0]} x {splits['train'].shape[1] * len(ks)} "
          f"[{time.time() - t0:.1f}s]")

    if cleanup is not None:
        cleanup.unlink()
        # Remove now-empty dirs so the b2 mount point stays clean.
        for p in [cleanup.parent, cleanup.parent.parent]:
            if p.is_dir() and not any(p.iterdir()):
                p.rmdir()
    return "done"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--setting", default="old", choices=["old", "new"])
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--ks", nargs="*", type=int, default=[1, 2, 4, 8])
    ap.add_argument("--download", action="store_true",
                    help="new setting: fetch missing raw tensors from B2, delete after")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cfg = get_config(args.setting)
    datasets = args.datasets or list(cfg.datasets)
    print(f"HaloScope features | model={args.model} setting={args.setting} ks={args.ks}")
    for ds in datasets:
        process_dataset(cfg, args.model, ds, args.ks, args.download, args.force)


if __name__ == "__main__":
    main()
