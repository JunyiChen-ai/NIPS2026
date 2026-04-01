"""
Run all 11 reproducible baseline methods on all 4 datasets.
Loads pre-extracted features, trains/evaluates, reports metrics.
"""

import os
import json
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, mean_squared_error
from scipy.stats import spearmanr

from methods import (
    LRProbe, MMProbe, pca_lr_probe, iti_directions,
    KBNet, compute_lid, attention_satisfies_probe,
    llm_check_score, sep_probe, compute_coe_scores,
    seakr_energy_score, STEPScorer,
)

FEATURES_DIR = "/data/jehc223/NIPS2026/extraction/features"
RESULTS_DIR = "/data/jehc223/NIPS2026/reproduce/results"
os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# Data loading
# ============================================================
def load_split(dataset, split):
    """Load all features for a dataset split."""
    split_dir = os.path.join(FEATURES_DIR, dataset, split)
    data = {}

    # Tensors
    for name in ["input_last_token_hidden", "input_mean_pool_hidden",
                 "input_per_head_activation", "input_attn_stats",
                 "input_attn_value_norms",
                 "gen_last_token_hidden", "gen_mean_pool_hidden",
                 "gen_per_token_hidden_last_layer", "gen_attn_stats_last"]:
        path = os.path.join(split_dir, f"{name}.pt")
        if os.path.exists(path):
            data[name] = torch.load(path, map_location="cpu")

    # JSON
    for name in ["input_logit_stats", "gen_logit_stats_last"]:
        path = os.path.join(split_dir, f"{name}.json")
        if os.path.exists(path):
            with open(path) as f:
                data[name] = json.load(f)

    # Meta
    with open(os.path.join(split_dir, "meta.json")) as f:
        meta = json.load(f)
    data["labels"] = meta["labels"]
    data["texts"] = meta["texts"]
    data["gen_lens"] = meta["gen_lens"]

    return data


def load_dataset(dataset, train_split, test_split):
    train = load_split(dataset, train_split)
    test = load_split(dataset, test_split)
    return train, test


# ============================================================
# Evaluation helpers
# ============================================================
def eval_classification(y_true, probs, preds=None):
    """Evaluate binary classification."""
    y_true = np.array(y_true)
    probs = np.array(probs)
    if preds is None:
        preds = (probs > 0.5).astype(int)
    else:
        preds = np.array(preds)
    metrics = {
        "auroc": roc_auc_score(y_true, probs),
        "accuracy": accuracy_score(y_true, preds),
        "f1": f1_score(y_true, preds, zero_division=0),
    }
    return metrics


def eval_regression(y_true, scores):
    """Evaluate regression / ranking."""
    y_true = np.array(y_true)
    scores = np.array(scores)
    corr, pval = spearmanr(y_true, scores)
    metrics = {
        "spearman_r": corr,
        "spearman_p": pval,
        "mse": mean_squared_error(y_true, scores),
    }
    return metrics


def eval_scoring(y_true, scores):
    """Evaluate unsupervised scoring (no training, just AUROC of score vs label)."""
    y_true = np.array(y_true)
    scores = np.array(scores)
    auroc = roc_auc_score(y_true, scores)
    # Try flipped direction too
    auroc_flip = roc_auc_score(y_true, -scores)
    return {"auroc": max(auroc, auroc_flip), "auroc_raw": auroc, "auroc_flipped": auroc_flip}


# ============================================================
# Per-layer evaluation helper
# ============================================================
def eval_per_layer(train_data, test_data, method_fn, is_regression=False):
    """Run a method across all layers, return best layer results."""
    n_layers_plus = train_data["input_last_token_hidden"].shape[1]  # 30
    train_labels = torch.tensor(train_data["labels"])
    test_labels = torch.tensor(test_data["labels"])

    best_metric = -float('inf')
    best_layer = 0
    best_results = {}
    all_layer_results = {}

    for layer in range(n_layers_plus):
        train_acts = train_data["input_last_token_hidden"][:, layer, :]
        test_acts = test_data["input_last_token_hidden"][:, layer, :]

        result = method_fn(train_acts, train_labels, test_acts, test_labels)
        all_layer_results[layer] = result

        metric_val = result.get("spearman_r", result.get("auroc", 0))
        if metric_val > best_metric:
            best_metric = metric_val
            best_layer = layer
            best_results = result

    return {"best_layer": best_layer, "best_results": best_results,
            "all_layers": all_layer_results}


# ============================================================
# Method runners
# ============================================================
def run_lr_probe(train_data, test_data, is_regression=False):
    """Geometry of Truth — LR Probe, per layer."""
    def method_fn(train_acts, train_labels, test_acts, test_labels):
        # Center (original: utils.py line 61)
        mean = train_acts.float().mean(dim=0)
        train_c = train_acts.float() - mean
        test_c = test_acts.float() - mean

        if is_regression:
            # For regression, use sklearn LR (not binary probe)
            from sklearn.linear_model import Ridge
            clf = Ridge(alpha=1.0)
            clf.fit(train_c.numpy(), train_labels.numpy())
            preds = clf.predict(test_c.numpy())
            return eval_regression(test_labels.numpy(), preds)
        else:
            probe = LRProbe.from_data(train_c, train_labels.float(), epochs=1000)
            with torch.no_grad():
                probs = probe(test_c).numpy()
                preds = probe.pred(test_c).numpy()
            return eval_classification(test_labels.numpy(), probs, preds)

    return eval_per_layer(train_data, test_data, method_fn, is_regression)


def run_mm_probe(train_data, test_data, is_regression=False):
    """Geometry of Truth — Mass-Mean Probe, per layer."""
    if is_regression:
        return {"skipped": "MMProbe is classification-only"}

    def method_fn(train_acts, train_labels, test_acts, test_labels):
        mean = train_acts.float().mean(dim=0)
        train_c = train_acts.float() - mean
        test_c = test_acts.float() - mean

        probe = MMProbe.from_data(train_c, train_labels)
        with torch.no_grad():
            probs = probe(test_c).numpy()
            preds = probe.pred(test_c).numpy()
        return eval_classification(test_labels.numpy(), probs, preds)

    return eval_per_layer(train_data, test_data, method_fn)


def run_pca_lr(train_data, test_data, is_regression=False):
    """No Answer Needed — PCA + LR, per layer."""
    def method_fn(train_acts, train_labels, test_acts, test_labels):
        if is_regression:
            from sklearn.linear_model import Ridge
            mean = train_acts.float().mean(dim=0)
            U, S, Vh = torch.linalg.svd((train_acts.float() - mean), full_matrices=False)
            n_comp = min(50, train_acts.shape[0], train_acts.shape[1])
            train_pca = ((train_acts.float() - mean) @ Vh.T[:, :n_comp]).numpy()
            test_pca = ((test_acts.float() - mean) @ Vh.T[:, :n_comp]).numpy()
            scaler = StandardScaler()
            train_pca = scaler.fit_transform(train_pca)
            test_pca = scaler.transform(test_pca)
            clf = Ridge(alpha=1.0)
            clf.fit(train_pca, train_labels.numpy())
            preds = clf.predict(test_pca)
            return eval_regression(test_labels.numpy(), preds)
        else:
            probs, preds = pca_lr_probe(train_acts, train_labels, test_acts, test_labels)
            return eval_classification(test_labels.numpy(), probs, preds)

    return eval_per_layer(train_data, test_data, method_fn, is_regression)


def run_iti(train_data, test_data, is_regression=False):
    """ITI — Direction per head."""
    if is_regression:
        return {"skipped": "ITI is classification-only"}
    train_acts = train_data["input_per_head_activation"]
    test_acts = test_data["input_per_head_activation"]
    train_labels = torch.tensor(train_data["labels"])
    test_labels = torch.tensor(test_data["labels"])

    results = iti_directions(train_acts, train_labels, test_acts, test_labels)
    best_idx = np.unravel_index(results[:, :, 0].argmax(), results[:, :, 0].shape)
    return {
        "best_layer": int(best_idx[0]),
        "best_head": int(best_idx[1]),
        "best_auroc": float(results[best_idx[0], best_idx[1], 0]),
        "best_accuracy": float(results[best_idx[0], best_idx[1], 1]),
        "all_results_shape": list(results.shape),
    }


def run_kb_mlp(train_data, test_data, is_regression=False):
    """Knowledge Boundary — MLP, best layer."""
    if is_regression:
        return {"skipped": "KBNet is classification-only"}

    train_labels = torch.tensor(train_data["labels"])
    test_labels = torch.tensor(test_data["labels"])
    best_auroc = 0
    best_layer = 0
    best_results = {}

    for layer in range(train_data["input_last_token_hidden"].shape[1]):
        train_acts = train_data["input_last_token_hidden"][:, layer, :]
        test_acts = test_data["input_last_token_hidden"][:, layer, :]
        probs, preds = KBNet.train_and_eval(train_acts, train_labels, test_acts, test_labels)
        metrics = eval_classification(test_labels.numpy(), probs, preds)
        if metrics["auroc"] > best_auroc:
            best_auroc = metrics["auroc"]
            best_layer = layer
            best_results = metrics

    return {"best_layer": best_layer, "best_results": best_results}


def run_lid(train_data, test_data, is_regression=False):
    """LID scoring, per layer."""
    train_labels = np.array(train_data["labels"])
    test_labels = np.array(test_data["labels"])

    best_auroc = 0
    best_layer = 0

    for layer in range(train_data["input_last_token_hidden"].shape[1]):
        train_acts = train_data["input_last_token_hidden"][:, layer, :]
        test_acts = test_data["input_last_token_hidden"][:, layer, :]

        # LID uses correct samples as reference (original: lids.py line 74)
        correct_mask = train_labels == 1
        ref_acts = train_acts[correct_mask]
        k = min(200, len(ref_acts) - 1)
        if k < 2:
            continue

        lids = compute_lid(ref_acts, test_acts, k=k)

        if is_regression:
            metrics = eval_regression(test_labels, lids)
            metric_val = abs(metrics["spearman_r"])
        else:
            metrics = eval_scoring(test_labels, lids)
            metric_val = metrics["auroc"]

        if metric_val > best_auroc:
            best_auroc = metric_val
            best_layer = layer
            best_results = metrics

    return {"best_layer": best_layer, "best_results": best_results}


def run_attn_satisfies(train_data, test_data, is_regression=False):
    """Attention Satisfies — LR on attn_value_norms."""
    if is_regression:
        return {"skipped": "Attention Satisfies probe is classification-only"}
    train_labels = torch.tensor(train_data["labels"])
    test_labels = torch.tensor(test_data["labels"])
    probs, preds = attention_satisfies_probe(
        train_data["input_attn_value_norms"], train_labels,
        test_data["input_attn_value_norms"], test_labels)
    return eval_classification(test_labels.numpy(), probs, preds)


def run_llm_check(train_data, test_data, is_regression=False):
    """LLM-Check — Attention diagonal scoring (unsupervised)."""
    test_labels = np.array(test_data["labels"])
    scores = llm_check_score(test_data["input_attn_stats"])
    if is_regression:
        return eval_regression(test_labels, scores)
    else:
        return eval_scoring(test_labels, scores)


def run_sep(train_data, test_data, is_regression=False):
    """SEP — LR on generation hidden states."""
    train_labels = torch.tensor(train_data["labels"])
    test_labels = torch.tensor(test_data["labels"])

    if is_regression:
        from sklearn.linear_model import Ridge
        best_r = -1
        best_layer = 0
        best_results = {}
        for layer in range(train_data["gen_last_token_hidden"].shape[1]):
            X_train = train_data["gen_last_token_hidden"][:, layer, :].float().numpy()
            X_test = test_data["gen_last_token_hidden"][:, layer, :].float().numpy()
            clf = Ridge(alpha=1.0)
            clf.fit(X_train, train_labels.numpy())
            preds = clf.predict(X_test)
            metrics = eval_regression(test_labels.numpy(), preds)
            if abs(metrics["spearman_r"]) > best_r:
                best_r = abs(metrics["spearman_r"])
                best_layer = layer
                best_results = metrics
        return {"best_layer": best_layer, "best_results": best_results}
    else:
        best_auroc = 0
        best_layer = 0
        best_results = {}
        for layer in range(train_data["gen_last_token_hidden"].shape[1]):
            probs, preds = sep_probe(
                train_data["gen_last_token_hidden"], train_labels,
                test_data["gen_last_token_hidden"], test_labels,
                layer=layer)
            metrics = eval_classification(test_labels.numpy(), probs, preds)
            if metrics["auroc"] > best_auroc:
                best_auroc = metrics["auroc"]
                best_layer = layer
                best_results = metrics
        return {"best_layer": best_layer, "best_results": best_results}


def run_coe(train_data, test_data, is_regression=False):
    """CoE — Unsupervised geometric scores on generation hidden states."""
    test_labels = np.array(test_data["labels"])
    scores = compute_coe_scores(test_data["gen_mean_pool_hidden"])
    results = {}
    for score_name, score_vals in scores.items():
        if is_regression:
            results[score_name] = eval_regression(test_labels, score_vals)
        else:
            results[score_name] = eval_scoring(test_labels, score_vals)
    return results


def run_seakr(train_data, test_data, is_regression=False):
    """SeaKR — Energy score (logsumexp) from generation logits."""
    test_labels = np.array(test_data["labels"])
    scores = seakr_energy_score(test_data["gen_logit_stats_last"])
    if is_regression:
        return eval_regression(test_labels, scores)
    else:
        return eval_scoring(test_labels, scores)


# ============================================================
# Main
# ============================================================
DATASETS = {
    "geometry_of_truth_cities": ("train", "val", False),
    "easy2hard_amc": ("train", "eval", True),   # regression
    "metatool_task1": ("train", "test", False),
    "retrievalqa": ("train", "test", False),
}

METHODS = {
    "lr_probe": run_lr_probe,
    "mm_probe": run_mm_probe,
    "pca_lr": run_pca_lr,
    "iti": run_iti,
    "kb_mlp": run_kb_mlp,
    "lid": run_lid,
    "attn_satisfies": run_attn_satisfies,
    "llm_check": run_llm_check,
    "sep": run_sep,
    "coe": run_coe,
    "seakr": run_seakr,
}


def main():
    all_results = {}

    for dataset, (train_split, test_split, is_regression) in DATASETS.items():
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset} ({'regression' if is_regression else 'classification'})")
        print(f"{'='*60}")

        train_data, test_data = load_dataset(dataset, train_split, test_split)
        print(f"  Train: {len(train_data['labels'])} samples")
        print(f"  Test:  {len(test_data['labels'])} samples")

        dataset_results = {}
        for method_name, method_fn in METHODS.items():
            print(f"\n  --- {method_name} ---")
            result = method_fn(train_data, test_data, is_regression=is_regression)
            dataset_results[method_name] = result

            # Print key metrics
            if isinstance(result, dict):
                if "skipped" in result:
                    print(f"    Skipped: {result['skipped']}")
                elif "best_results" in result:
                    br = result["best_results"]
                    layer_info = f"layer={result.get('best_layer', '?')}"
                    if "best_head" in result:
                        layer_info += f", head={result['best_head']}"
                    print(f"    Best {layer_info}: {br}")
                elif "coe_mag" in result:
                    for k, v in result.items():
                        if isinstance(v, dict):
                            print(f"    {k}: {v}")
                else:
                    print(f"    {result}")

        all_results[dataset] = dataset_results

    # Save
    results_path = os.path.join(RESULTS_DIR, "all_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
