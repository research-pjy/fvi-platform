"""
Unified Dataset Loader & Preprocessor for Tabular and Image Modalities.

Handles:
  1. Tabular CSV datasets:
     - Parses selected `feature_columns` and `target_column`
     - Automatically encodes categorical feature columns (one-hot encoding) and
       imputes missing numerical values using training-split medians
     - Applies Z-score standardization (`StandardScaler`) fit strictly on the
       training split so zero test information leaks into training
  2. Image datasets (both `ImageFolder` directory trees and CSV manifests where
     a feature column contains image file paths):
     - Loads images via Pillow (`L` for 1-channel Grayscale malware byteplots or
       `RGB` for 3-channel natural images)
     - Resizes to `(image_size, image_size)` and normalizes pixel intensities to
       `[-1, 1]` (`(x / 255.0 - 0.5) / 0.5`), returning clean `(N, C, H, W)`
       float32 arrays ready for PyTorch `TensorDataset` / `DataLoader`.
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
    input_dim: int              # D for tabular
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

    # Encode target labels into sorted class names 0..C-1
    raw_targets = [r[target_idx].strip() for r in rows]
    unique_classes = sorted(set(t for t in raw_targets if t != ""))
    class_to_id = {name: idx for idx, name in enumerate(unique_classes)}

    valid_rows = [r for r in rows if r[target_idx].strip() in class_to_id]
    y_all = np.array(
        [class_to_id[r[target_idx].strip()] for r in valid_rows], dtype=np.int64
    )

    # Build numerical and one-hot encoded feature columns
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
            # Categorical one-hot encoding (capped at top 32 categories to avoid explosion)
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

    train_idx, val_idx, test_idx = _stratified_train_val_test_split(
        y_all, cfg.val_fraction, cfg.test_fraction, cfg.seed
    )

    # Fit median imputer & Z-score scaler strictly on X_train
    X_train_raw = X_raw[train_idx]
    col_medians = np.nanmedian(X_train_raw, axis=0)
    col_medians = np.where(np.isnan(col_medians), 0.0, col_medians)

    for j in range(X_raw.shape[1]):
        mask = np.isnan(X_raw[:, j])
        if np.any(mask):
            X_raw[mask, j] = col_medians[j]

    mean = X_raw[train_idx].mean(axis=0, keepdims=True)
    std = X_raw[train_idx].std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    X_scaled = ((X_raw - mean) / std).astype(np.float32)

    return LoadedDatasetBundle(
        modality="tabular",
        X_train=X_scaled[train_idx],
        y_train=y_all[train_idx],
        X_val=X_scaled[val_idx],
        y_val=y_all[val_idx],
        X_test=X_scaled[test_idx],
        y_test=y_all[test_idx],
        class_names=unique_classes,
        feature_names=expanded_feature_names,
        input_dim=int(X_scaled.shape[1]),
        image_shape=None,
    )


def _load_single_image_tensor(
    img_path: Path, image_size: int, image_channels: int
) -> np.ndarray:
    """Loads an image, converts to Grayscale (1ch) or RGB (3ch), resizes, and normalizes to [-1, 1]."""
    mode = "L" if image_channels == 1 else "RGB"
    with Image.open(img_path) as im:
        im_conv = im.convert(mode).resize((image_size, image_size), Image.Resampling.BILINEAR)
        arr = np.asarray(im_conv, dtype=np.float32) / 255.0
        if image_channels == 1:
            arr = arr[None, :, :]  # (1, H, W)
        else:
            arr = np.transpose(arr, (2, 0, 1))  # (3, H, W)
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

    train_idx, val_idx, test_idx = _stratified_train_val_test_split(
        y_all, cfg.val_fraction, cfg.test_fraction, cfg.seed
    )
    img_shape = (cfg.image_channels, cfg.image_size, cfg.image_size)

    return LoadedDatasetBundle(
        modality="image",
        X_train=X_all[train_idx],
        y_train=y_all[train_idx],
        X_val=X_all[val_idx],
        y_val=y_all[val_idx],
        X_test=X_all[test_idx],
        y_test=y_all[test_idx],
        class_names=unique_classes,
        feature_names=[f"pixel_grid_{cfg.image_channels}x{cfg.image_size}x{cfg.image_size}"],
        input_dim=int(np.prod(img_shape)),
        image_shape=img_shape,
    )


def load_and_preprocess_dataset(cfg: DatasetConfig) -> LoadedDatasetBundle:
    """
    Master entry point that inspects `cfg.data_path` and loads either a Tabular
    or Image `LoadedDatasetBundle`.
    """
    schema = inspect_dataset_source(cfg.data_path, target_column=cfg.target_column)
    modality = cfg.modality.lower() if cfg.modality in ("tabular", "image") else schema.detected_modality
    if schema.source_format == "image_folder" or schema.detected_modality == "image":
        modality = "image"

    if modality == "image":
        return _load_image_dataset(cfg)
    return _load_tabular_csv(cfg)
