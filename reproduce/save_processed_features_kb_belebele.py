"""Generate Knowledge-Boundary processed features for Belebele.

Target label: the dataset's 4-way gold answer option in raw extraction meta.
Splits: reuse reproduce/correctness_labels/{model}/belebele/split_indices.json
so KB and output-correctness runs use the same examples.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler, label_binarize

sys.path.insert(0, os.path.dirname(__file__))


ROOT = Path("/home/junyi/NIPS2026")
B2_FEATURES = Path("/home/junyi/b2-nips/extraction/features")
LOCAL_EXTRACTION = ROOT / "extraction/features"
OUTPUT_ROOT = ROOT / "reproduce/processed_features"
SPLIT_ROOT = ROOT / "reproduce/correctness_labels"

METHODS = ["lr_probe", "pca_lr", "iti", "kb_mlp", "attn_satisfies", "sep", "step"]
N_CLASSES = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_tensor(model: str, name: str) -> torch.Tensor:
    return torch.load(B2_FEATURES / model / "belebele" / "all" / f"{name}.pt", map_location="cpu", weights_only=False)


def load_json(model: str, name: str):
    with open(B2_FEATURES / model / "belebele" / "all" / f"{name}.json") as f:
        return json.load(f)


def load_meta_labels(model: str) -> np.ndarray:
    with open(B2_FEATURES / model / "belebele" / "all" / "meta.json") as f:
        return np.asarray(json.load(f)["labels"], dtype=int)


def load_split(model: str) -> dict[str, np.ndarray]:
    with open(SPLIT_ROOT / model / "belebele" / "split_indices.json") as f:
        split = json.load(f)
    return {k: np.asarray(v, dtype=np.int64) for k, v in split.items()}


def write_local_label_meta(model: str, labels: np.ndarray, split: dict[str, np.ndarray]) -> None:
    base = LOCAL_EXTRACTION / model / "belebele"
    for s in ["train", "val", "test"]:
        out = base / s
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "meta.json", "w") as f:
            json.dump({"labels": labels[split[s]].astype(int).tolist()}, f)


def save_feat(model: str, method: str, split_name: str, tensor) -> None:
    out = OUTPUT_ROOT / model / "belebele" / method
    out.mkdir(parents=True, exist_ok=True)
    if isinstance(tensor, np.ndarray):
        tensor = torch.from_numpy(tensor)
    torch.save(tensor, out / f"{split_name}.pt")


def save_meta(model: str, method: str, meta: dict) -> None:
    out = OUTPUT_ROOT / model / "belebele" / method
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)


def method_done(model: str, method: str) -> bool:
    out = OUTPUT_ROOT / model / "belebele" / method
    return all((out / f"{s}.pt").exists() for s in ["train", "val", "test"]) and (out / "meta.json").exists()


def auroc_mc(y, probs) -> float:
    yb = label_binarize(y, classes=list(range(N_CLASSES)))
    return roc_auc_score(yb, probs, average="macro", multi_class="ovr")


def fast_mc_auroc(X_tr, y_tr, X_va, y_va, l2=1e-2, max_iter=1000) -> float:
    Xt = torch.as_tensor(X_tr, dtype=torch.float32, device=DEVICE)
    yt = torch.as_tensor(y_tr, dtype=torch.long, device=DEVICE)
    Xv = torch.as_tensor(X_va, dtype=torch.float32, device=DEVICE)
    W = torch.zeros((Xt.shape[1], N_CLASSES), requires_grad=True, device=DEVICE)
    b = torch.zeros(N_CLASSES, requires_grad=True, device=DEVICE)
    opt = torch.optim.LBFGS([W, b], lr=1.0, max_iter=max_iter, tolerance_grad=1e-6, history_size=10)

    def closure():
        opt.zero_grad()
        logits = Xt @ W + b
        loss = torch.nn.functional.cross_entropy(logits, yt) + l2 * (W * W).sum()
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        probs = torch.softmax(Xv @ W + b, dim=-1).cpu().numpy()
    del Xt, yt, Xv, W, b
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    return auroc_mc(y_va, probs)


def select_best_layer(h: torch.Tensor, labels: np.ndarray, split: dict[str, np.ndarray]) -> tuple[int, float]:
    tr_idx, va_idx = split["train"], split["val"]
    tr_y, va_y = labels[tr_idx], labels[va_idx]
    best_layer, best_auc = 0, -1.0
    for layer in range(h.shape[1]):
        Xtr = h[tr_idx][:, layer, :].float().numpy()
        Xva = h[va_idx][:, layer, :].float().numpy()
        auc = fast_mc_auroc(Xtr, tr_y, Xva, va_y)
        if auc > best_auc:
            best_layer, best_auc = layer, auc
    return best_layer, best_auc


def process_lr_probe(model: str, labels: np.ndarray, split: dict[str, np.ndarray]) -> None:
    h = load_tensor(model, "input_last_token_hidden")
    layer, val_auc = select_best_layer(h, labels, split)
    for s in ["train", "val", "test"]:
        save_feat(model, "lr_probe", s, h[split[s]][:, layer, :].float())
    save_meta(model, "lr_probe", {"best_layer": layer, "val_auroc": float(val_auc), "shape": "N x hidden_dim"})


def process_pca_lr(model: str, labels: np.ndarray, split: dict[str, np.ndarray]) -> None:
    h = load_tensor(model, "input_last_token_hidden")
    tr_idx, va_idx = split["train"], split["val"]
    tr_y, va_y = labels[tr_idx], labels[va_idx]
    best_layer, best_auc = 0, -1.0
    for layer in range(h.shape[1]):
        tr = h[tr_idx][:, layer, :].float()
        va = h[va_idx][:, layer, :].float()
        mean = tr.mean(dim=0)
        n_comp = min(50, tr.shape[0], tr.shape[1])
        pca = PCA(n_components=n_comp, random_state=42)
        sc = StandardScaler()
        Xtr = sc.fit_transform(pca.fit_transform((tr - mean).numpy()))
        Xva = sc.transform(pca.transform((va - mean).numpy()))
        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(Xtr, tr_y)
        auc = auroc_mc(va_y, clf.predict_proba(Xva))
        if auc > best_auc:
            best_layer, best_auc = layer, auc
    tr = h[tr_idx][:, best_layer, :].float()
    mean = tr.mean(dim=0)
    n_comp = min(50, tr.shape[0], tr.shape[1])
    pca = PCA(n_components=n_comp, random_state=42)
    sc = StandardScaler()
    Xtr = sc.fit_transform(pca.fit_transform((tr - mean).numpy()))
    save_feat(model, "pca_lr", "train", Xtr)
    for s in ["val", "test"]:
        acts = h[split[s]][:, best_layer, :].float()
        save_feat(model, "pca_lr", s, sc.transform(pca.transform((acts - mean).numpy())))
    save_meta(model, "pca_lr", {"best_layer": best_layer, "val_auroc": float(best_auc), "n_components": n_comp})


def process_iti(model: str, labels: np.ndarray, split: dict[str, np.ndarray]) -> None:
    a = load_tensor(model, "input_per_head_activation")
    tr_idx, va_idx = split["train"], split["val"]
    tr_y, va_y = labels[tr_idx], labels[va_idx]
    best_auc, best_li, best_hi = -1.0, 0, 0
    for li in range(a.shape[1]):
        for hi in range(a.shape[2]):
            Xtr = a[tr_idx][:, li, hi, :].float().numpy()
            Xva = a[va_idx][:, li, hi, :].float().numpy()
            auc = fast_mc_auroc(Xtr, tr_y, Xva, va_y)
            if auc > best_auc:
                best_auc, best_li, best_hi = auc, li, hi
    for s in ["train", "val", "test"]:
        save_feat(model, "iti", s, a[split[s]][:, best_li, best_hi, :].float())
    save_meta(model, "iti", {"best_layer": best_li, "best_head": best_hi, "val_auroc": float(best_auc)})


def process_kb_mlp(model: str, labels: np.ndarray, split: dict[str, np.ndarray]) -> None:
    h = load_tensor(model, "input_last_token_hidden")
    layer = h.shape[1] // 2
    for s in ["train", "val", "test"]:
        save_feat(model, "kb_mlp", s, h[split[s]][:, layer, :].float())
    save_meta(model, "kb_mlp", {"layer": layer, "shape": "N x hidden_dim"})


def process_attn_satisfies(model: str, labels: np.ndarray, split: dict[str, np.ndarray]) -> None:
    a = load_tensor(model, "input_attn_value_norms")
    sc = StandardScaler()
    tr = a[split["train"]].float().max(dim=-1).values.reshape(len(split["train"]), -1).numpy()
    save_feat(model, "attn_satisfies", "train", sc.fit_transform(tr))
    for s in ["val", "test"]:
        feat = a[split[s]].float().max(dim=-1).values.reshape(len(split[s]), -1).numpy()
        save_feat(model, "attn_satisfies", s, sc.transform(feat))
    save_meta(model, "attn_satisfies", {"shape": "N x (n_layers * n_heads)"})


def process_sep(model: str, labels: np.ndarray, split: dict[str, np.ndarray]) -> None:
    h = load_tensor(model, "gen_last_token_hidden")
    tr_idx, va_idx = split["train"], split["val"]
    tr_y, va_y = labels[tr_idx], labels[va_idx]
    best_auc, best_range = -1.0, (0, 1)
    for start in range(h.shape[1]):
        for end in range(start + 1, min(start + 6, h.shape[1] + 1)):
            Xtr = h[tr_idx][:, start:end, :].float().reshape(len(tr_idx), -1).numpy()
            Xva = h[va_idx][:, start:end, :].float().reshape(len(va_idx), -1).numpy()
            clf = LogisticRegression(max_iter=1000, random_state=42)
            clf.fit(Xtr, tr_y)
            auc = auroc_mc(va_y, clf.predict_proba(Xva))
            if auc > best_auc:
                best_auc, best_range = auc, (start, end)
    for s in ["train", "val", "test"]:
        idx = split[s]
        feat = h[idx][:, best_range[0]:best_range[1], :].float().reshape(len(idx), -1)
        save_feat(model, "sep", s, feat)
    save_meta(model, "sep", {"best_range": list(best_range), "val_auroc": float(best_auc)})


def process_step(model: str, labels: np.ndarray, split: dict[str, np.ndarray]) -> None:
    h = load_tensor(model, "gen_last_token_hidden")
    for s in ["train", "val", "test"]:
        save_feat(model, "step", s, h[split[s]][:, -2, :].float())
    save_meta(model, "step", {"layer": "last decoder (-2)", "shape": "N x hidden_dim"})


PROCESSORS = {
    "lr_probe": process_lr_probe,
    "pca_lr": process_pca_lr,
    "iti": process_iti,
    "kb_mlp": process_kb_mlp,
    "attn_satisfies": process_attn_satisfies,
    "sep": process_sep,
    "step": process_step,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["qwen2.5-7b", "llama3.1-8b", "mistral-7b-v0.3"])
    ap.add_argument("--methods", nargs="*", default=METHODS)
    args = ap.parse_args()

    labels = load_meta_labels(args.model)
    split = load_split(args.model)
    write_local_label_meta(args.model, labels, split)
    print(f"{args.model}/belebele: labels={len(labels)} splits={{train:{len(split['train'])}, val:{len(split['val'])}, test:{len(split['test'])}}}")
    for method in args.methods:
        if method_done(args.model, method):
            print(f"  {method}: skip")
            continue
        print(f"  {method}")
        PROCESSORS[method](args.model, labels, split)


if __name__ == "__main__":
    main()
