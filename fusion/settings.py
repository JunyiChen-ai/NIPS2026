"""Setting registry: old (Type-1, dataset-level GT) vs new (Type-2, model-correctness).

Centralizes all path / dataset / split / label-loading differences. Every fusion
script imports `get_config(setting)` and uses cfg.* exclusively — no hardcoded
paths, no hardcoded dataset dicts.
"""
from __future__ import annotations
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Old setting (Type-1 dataset-level ground truth)
# Datasets carry their own pre-split features at extraction/features/{model}/
#   {ext}/{train|val|test|train_sub|val_split|eval}/. Labels live in each
#   split's meta.json under "labels".
# ---------------------------------------------------------------------------

OLD_DATASETS = {
    "common_claim_3class": {"n_classes": 3, "ext": "common_claim_3class",
                            "splits": {"train": "train", "val": "val", "test": "test"}},
    "e2h_amc_3class":      {"n_classes": 3, "ext": "e2h_amc_3class",
                            "splits": {"train": "train_sub", "val": "val_split", "test": "eval"}},
    "e2h_amc_5class":      {"n_classes": 5, "ext": "e2h_amc_5class",
                            "splits": {"train": "train_sub", "val": "val_split", "test": "eval"}},
    "when2call_3class":    {"n_classes": 3, "ext": "when2call_3class",
                            "splits": {"train": "train", "val": "val", "test": "test"}},
    "ragtruth_binary":     {"n_classes": 2, "ext": "ragtruth",
                            "splits": {"train": "train", "val": "val", "test": "test"}},
    "fava_binary":         {"n_classes": 2, "ext": "fava",
                            "splits": {"train": "train", "val": "val", "test": "test"}},
    "belebele":            {"n_classes": 4, "ext": "belebele",
                            "splits": {"train": "train", "val": "val", "test": "test"}},
}

# ---------------------------------------------------------------------------
# New setting (Type-2 model-correctness)
# B2 mount has only a single "all" split per (model, dataset). Labels are
# derived per-model by reproduce/grade_correctness.py. Splits live alongside
# labels at reproduce/correctness_labels/{model}/{dataset}/split_indices.json.
# ---------------------------------------------------------------------------

NEW_DATASETS_LIST = [
    "gsm8k", "math", "mmlu", "commonsenseqa", "belebele", "theoremqa",
    "fava", "ragtruth", "common_claim_3class", "when2call_3class",
]
NEW_DATASETS = {
    ds: {"n_classes": 2, "ext": ds, "splits": {"train": "train", "val": "val", "test": "test"}}
    for ds in NEW_DATASETS_LIST
}

# Where new-setting per-model labels live (one labels.json + split_indices.json).
NEW_LABELS_ROOT = Path("/home/junyi/NIPS2026/reproduce/correctness_labels")


@dataclass
class SettingConfig:
    name: str
    base_processed: Path
    base_extraction: Path
    base_results: Path
    datasets: dict
    # In-memory cache: {(model, dataset): {"labels": ndarray, "splits": {split: idx_array}}}
    _cache: dict = field(default_factory=dict)

    # ---- label / split loading ----
    def _load_new_label_data(self, model, ds_name):
        key = (model, ds_name)
        if key in self._cache:
            return self._cache[key]
        with open(NEW_LABELS_ROOT / model / ds_name / "labels.json") as f:
            labels = np.asarray(json.load(f)["labels"])
        with open(NEW_LABELS_ROOT / model / ds_name / "split_indices.json") as f:
            splits = json.load(f)
        splits = {k: np.asarray(v, dtype=np.int64) for k, v in splits.items()}
        self._cache[key] = {"labels": labels, "splits": splits}
        return self._cache[key]

    def load_labels(self, model, ds_name, split):
        """Return labels (ndarray) for a given (model, dataset, split)."""
        if self.name == "old":
            ext = self.datasets[ds_name]["ext"]
            split_dir = self.datasets[ds_name]["splits"][split]
            with open(self.base_extraction / model / ext / split_dir / "meta.json") as f:
                return np.asarray(json.load(f)["labels"])
        else:
            d = self._load_new_label_data(model, ds_name)
            return d["labels"][d["splits"][split]]

    def split_indices(self, model, ds_name, split):
        """For new setting only: return the integer indices into the 'all' tensor."""
        if self.name == "old":
            raise RuntimeError("split_indices is only for new setting (B2 has single 'all' split)")
        return self._load_new_label_data(model, ds_name)["splits"][split]

    # ---- path builders ----
    def processed_pt(self, model, ds_name, method, split) -> Path:
        return self.base_processed / model / ds_name / method / f"{split}.pt"

    def model_results_dir(self, model) -> Path:
        return self.base_results / model

    def raw_view(self, model, ds_name, split, filename):
        """Load a raw-extraction tensor file. For new setting this loads the full
        'all' tensor and indexes by the split. Returns torch.Tensor.
        """
        import torch
        if self.name == "old":
            ext = self.datasets[ds_name]["ext"]
            split_dir = self.datasets[ds_name]["splits"][split]
            return torch.load(
                self.base_extraction / model / ext / split_dir / filename,
                map_location="cpu", weights_only=False,
            )
        # new: load full 'all' tensor then index by split
        full = torch.load(
            self.base_extraction / model / ds_name / "all" / filename,
            map_location="cpu", weights_only=False,
        )
        idx = self.split_indices(model, ds_name, split)
        # Handle both tensor and list-of-tensor types
        if hasattr(full, "shape"):
            return full[idx]
        return [full[i] for i in idx]

    def raw_view_json(self, model, ds_name, split, filename):
        """JSON variant of raw_view: e.g. logit_stats. Returns list (length matches split)."""
        if self.name == "old":
            ext = self.datasets[ds_name]["ext"]
            split_dir = self.datasets[ds_name]["splits"][split]
            with open(self.base_extraction / model / ext / split_dir / filename) as f:
                return json.load(f)
        with open(self.base_extraction / model / ds_name / "all" / filename) as f:
            full = json.load(f)
        idx = self.split_indices(model, ds_name, split)
        return [full[i] for i in idx.tolist()]


OLD = SettingConfig(
    name="old",
    base_processed=Path("/home/junyi/NIPS2026/reproduce/processed_features"),
    base_extraction=Path("/home/junyi/NIPS2026/extraction/features"),
    base_results=Path("/home/junyi/NIPS2026/fusion/results"),
    datasets=OLD_DATASETS,
)

NEW = SettingConfig(
    name="new",
    base_processed=Path("/home/junyi/NIPS2026/reproduce/processed_features_correctness"),
    base_extraction=Path("/home/junyi/b2-nips/extraction/features"),
    base_results=Path("/home/junyi/NIPS2026/fusion/results_correctness"),
    datasets=NEW_DATASETS,
)


def get_config(setting: str) -> SettingConfig:
    s = setting.lower()
    if s == "old":
        return OLD
    if s == "new":
        return NEW
    raise ValueError(f"Unknown setting {setting!r}. Use 'old' or 'new'.")
