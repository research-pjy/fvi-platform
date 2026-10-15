"""
Unified Dataset Loader & In-Memory Tensor Preprocessor for Tabular and Image Modalities.

Supports:
  1. Ephemeral In-RAM Tensor Bundles (`bundle_from_in_memory_arrays`):
     Allows the browser to parse CSVs or resize image folders locally on the
     user's laptop and send only a compact numerical array `(X, y)` to `rama`'s
     RAM—so ZERO dataset files are ever written to `rama`'s hard disk!
  2. Server-side CSV and ImageFolder loading (`load_and_preprocess_dataset`) for
     CLI batch runs.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from fvi.config import DatasetConfig
from fvi.ingestion.schema import IMAGE_EXTENSIONS, inspect_dataset_source


@dataclass
class LoadedDatasetBundle:
    """
    Stores preprocessed NumPy arrays and metadata for Train, Validation, and Test splits.
    """

    modality: str               # "tabular" or "image"
    X_train: np.ndarray         # (N_train, D) for tabular or (N_train, C, H, W) for image
    y_train: np.ndarray         # (N_train,) int64 in [0, num_classes - 1]
    X_val: np.ndarray           # (N_val, ...)
    y_val: np.ndarray           # (N_val,)
    X_test: np.ndarray          # (N_test, ...)
    y_test: np.ndarray          # (N_test,)
    class_names: List[str]
    feature_names: List[str]
    input_dim: int              # D for tabular, or C*H*W for image
    image_shape: Optional[Tuple[int, int, int]] = None  # (C, H, W) for image

    @property
    def num_classes(self) -> int:
        return len(self.class_names)


def _stratified_train_val_test_split(
    y: np.ndarray,
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Splits indices into stratified (train_idx, val_idx, test_idx)."""
    rng = np.random.default_rng(seed)
    train_idx: List[int] = []
    val_idx: List[int] = []
    test_idx: List[int] = []

    for c in np.unique(y):
        c_indices = np.where(y == c)[0]
        rng.shuffle(c_indices)
        n = len(c_indices)
        n_test = max(1, int(round(n * test_fraction)))
        n_val = max(1, int(round(n * val_fraction)))
        if n_test + n_val >= n and n >= 3:
            n_test = 1
            n_val = 1

        test_idx.extend(c_indices[:n_test].tolist())
        val_idx.extend(c_indices[n_test : n_test + n_val].tolist())
        train_idx.extend(c_indices[n_test + n_val :].tolist())

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)
    return (
        np.array(train_idx, dtype=np.int64),
        np.array(val_idx, dtype=np.int64),
        np.array(test_idx, dtype=np.int64),
    )


def bundle_from_in_memory_arrays(
    X_raw: np.ndarray,
    y_raw: np.ndarray,
    modality: str,
    class_names: Sequence[str],
    feature_names: Sequence[str],
    image_shape: Optional[Tuple[int, int, int]] = None,
    val_fraction: float = 0.10,
    test_fraction: float = 0.20,
    seed: int = 42,
) -> LoadedDatasetBundle:
    """
    Constructs a leakage-free `LoadedDatasetBundle` purely in RAM from browser-
    preprocessed arrays `(X_raw, y_raw)` without touching the server's disk.
    """
    y_all = np.asarray(y_raw, dtype=np.int64)
    X_arr = np.asarray(X_raw, dtype=np.float32)

    train_idx, val_idx, test_idx = _stratified_train_val_test_split(
        y_all, val_fraction=val_fraction, test_fraction=test_fraction, seed=seed
    )

    if modality.lower() == "image":
        if image_shape is None:
            if X_arr.ndim == 4:
                image_shape = (int(X_arr.shape[1]), int(X_arr.shape[2]), int(X_arr.shape[3]))
            else:
                side = int(round(np.sqrt(X_arr.shape[1])))
                image_shape = (1, side, side)
        X_arr = X_arr.reshape(-1, *image_shape)
        # If pixels are in [0, 255] or [0, 1], normalize to [-1, 1]
        if X_arr.max() > 1.5:
            X_arr = (X_arr / 255.0 - 0.5) / 0.5
        elif X_arr.min() >= 0.0 and X_arr.max() <= 1.0:
            X_arr = (X_arr - 0.5) / 0.5

        return LoadedDatasetBundle(
            modality="image",
            X_train=X_arr[train_idx],
            y_train=y_all[train_idx],
            X_val=X_arr[val_idx],
            y_val=y_all[val_idx],
            X_test=X_arr[test_idx],
            y_test=y_all[test_idx],
            class_names=list(class_names),
            feature_names=list(feature_names) or [f"pixel_{image_shape}"],
            input_dim=int(np.prod(image_shape)),
            image_shape=image_shape,
        )

    # Tabular modality: fit median imputer & Z-score standardization strictly on X_train
    if X_arr.ndim > 2:
        X_arr = X_arr.reshape(X_arr.shape[0], -1)

    X_train_raw = X_arr[train_idx]
    col_medians = np.nanmedian(X_train_raw, axis=0)
    col_medians = np.where(np.isnan(col_medians), 0.0, col_medians)

    for j in range(X_arr.shape[1]):
        mask = ~np.isfinite(X_arr[:, j])
        if np.any(mask):
            X_arr[mask, j] = col_medians[j]

    mean = X_arr[train_idx].mean(axis=0, keepdims=True)
    std = X_arr[train_idx].std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    X_scaled = ((X_arr - mean) / std).astype(np.float32)

    return LoadedDatasetBundle(
        modality="tabular",
        X_train=X_scaled[train_idx],
        y_train=y_all[train_idx],
        X_val=X_scaled[val_idx],
        y_val=y_all[val_idx],
        X_test=X_scaled[test_idx],
        y_test=y_all[test_idx],
        class_names=list(class_names),
        feature_names=list(feature_names),
        input_dim=int(X_scaled.shape[1]),
        image_shape=None,
    )


def _load_tabular_csv(cfg: DatasetConfig) -> LoadedDatasetBundle:
    path = Path(cfg.data_path)
    schema = inspect_dataset_source(path, target_column=cfg.target_column)
    if not schema.exists:
        raise FileNotFoundError(f"Dataset file not found: {path}")

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        header = [h.strip() for h in next(reader)]
        rows = [r for r in reader if len(r) == len(header)]

    target_col = cfg.target_column if cfg.target_column in header else schema.suggested_target
    target_idx = header.index(target_col)

    if (
        not cfg.feature_columns
        or cfg.feature_columns == ["all"]
        or "all" in cfg.feature_columns
    ):
        selected_features = [c for c in header if c != target_col]
    else:
        selected_features = [c for c in cfg.feature_columns if c in header and c != target_col]
        if not selected_features:
            selected_features = [c for c in header if c != target_col]

    col_type_map = {c.name: c.inferred_type for c in schema.columns}

    raw_targets = [r[target_idx].strip() for r in rows]
    unique_classes = sorted(set(t for t in raw_targets if t != ""))
    class_to_id = {name: idx for idx, name in enumerate(unique_classes)}

    valid_rows = [r for r in rows if r[target_idx].strip() in class_to_id]
    y_all = np.array(
        [class_to_id[r[target_idx].strip()] for r in valid_rows], dtype=np.int64
    )

    feature_blocks: List[np.ndarray] = []
    expanded_feature_names: List[str] = []

    for feat_name in selected_features:
        f_idx = header.index(feat_name)
        f_type = col_type_map.get(feat_name, "numerical")
        col_vals = [r[f_idx].strip() for r in valid_rows]

        if f_type == "numerical":
            arr = np.full(len(valid_rows), np.nan, dtype=np.float32)
            for i, v in enumerate(col_vals):
                try:
                    arr[i] = float(v)
                except ValueError:
                    arr[i] = np.nan
            feature_blocks.append(arr[:, None])
            expanded_feature_names.append(feat_name)
        else:
            cats = sorted(set(v for v in col_vals if v != ""))[:32]
            if not cats:
                continue
            oh = np.zeros((len(valid_rows), len(cats)), dtype=np.float32)
            cat_map = {c_val: j for j, c_val in enumerate(cats)}
            for i, v in enumerate(col_vals):
                if v in cat_map:
                    oh[i, cat_map[v]] = 1.0
            feature_blocks.append(oh)
            for c_val in cats:
                expanded_feature_names.append(f"{feat_name}={c_val}")

    X_raw = np.concatenate(feature_blocks, axis=1).astype(np.float32)
    return bundle_from_in_memory_arrays(
        X_raw=X_raw,
        y_raw=y_all,
        modality="tabular",
        class_names=unique_classes,
        feature_names=expanded_feature_names,
        val_fraction=cfg.val_fraction,
        test_fraction=cfg.test_fraction,
        seed=cfg.seed,
    )


def _load_single_image_tensor(
    img_path: Path, image_size: int, image_channels: int
) -> np.ndarray:
    mode = "L" if image_channels == 1 else "RGB"
    with Image.open(img_path) as im:
        im_conv = im.convert(mode).resize((image_size, image_size), Image.Resampling.BILINEAR)
        arr = np.asarray(im_conv, dtype=np.float32) / 255.0
        if image_channels == 1:
            arr = arr[None, :, :]
        else:
            arr = np.transpose(arr, (2, 0, 1))
        return ((arr - 0.5) / 0.5).astype(np.float32)


def _load_image_dataset(cfg: DatasetConfig) -> LoadedDatasetBundle:
    path = Path(cfg.data_path)
    if not path.exists():
        raise FileNotFoundError(f"Image dataset path not found: {path}")

    image_paths: List[Path] = []
    raw_labels: List[str] = []

    if path.is_dir():
        for sub in sorted(path.iterdir()):
            if sub.is_dir() and not sub.name.startswith("."):
                for p in sorted(sub.iterdir()):
                    if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
                        image_paths.append(p)
                        raw_labels.append(sub.name)
    else:
        schema = inspect_dataset_source(path, target_column=cfg.target_column)
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f)
            header = [h.strip() for h in next(reader)]
            rows = [r for r in reader if len(r) == len(header)]

        target_col = cfg.target_column if cfg.target_column in header else schema.suggested_target
        target_idx = header.index(target_col)

        if cfg.image_path_column and cfg.image_path_column in header:
            img_col = cfg.image_path_column
        elif cfg.feature_columns and cfg.feature_columns[0] in header:
            img_col = cfg.feature_columns[0]
        else:
            img_col = schema.suggested_features[0]
        img_idx = header.index(img_col)

        for r in rows:
            p_cand = Path(r[img_idx].strip())
            if not p_cand.exists():
                p_cand = path.parent / p_cand
            if p_cand.exists():
                image_paths.append(p_cand)
                raw_labels.append(r[target_idx].strip())

    if not image_paths:
        raise ValueError(f"No valid images found in {path}")

    unique_classes = sorted(set(raw_labels))
    class_to_id = {c: i for i, c in enumerate(unique_classes)}
    y_all = np.array([class_to_id[lbl] for lbl in raw_labels], dtype=np.int64)

    X_all = np.stack(
        [
            _load_single_image_tensor(p, cfg.image_size, cfg.image_channels)
            for p in image_paths
        ],
        axis=0,
    )
    img_shape = (cfg.image_channels, cfg.image_size, cfg.image_size)
    return bundle_from_in_memory_arrays(
        X_raw=X_all,
        y_raw=y_all,
        modality="image",
        class_names=unique_classes,
        feature_names=[f"pixel_grid_{cfg.image_channels}x{cfg.image_size}x{cfg.image_size}"],
        image_shape=img_shape,
        val_fraction=cfg.val_fraction,
        test_fraction=cfg.test_fraction,
        seed=cfg.seed,
    )


def load_and_preprocess_dataset(cfg: DatasetConfig) -> LoadedDatasetBundle:
    schema = inspect_dataset_source(cfg.data_path, target_column=cfg.target_column)
    modality = cfg.modality.lower() if cfg.modality in ("tabular", "image") else schema.detected_modality
    if schema.source_format == "image_folder" or schema.detected_modality == "image":
        modality = "image"

    if modality == "image":
        return _load_image_dataset(cfg)
    return _load_tabular_csv(cfg)
