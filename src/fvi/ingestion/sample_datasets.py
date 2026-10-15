"""
Built-in Benchmark Dataset Generators for Malware Analysis (Tabular & Grayscale Byteplot Images).

Provides realistic out-of-the-box cybersecurity datasets so you and your faculty
advisor can immediately test both Tabular FL and Image FL before plugging in
external CSVs or image directories:

  1. `generate_pe_malware_tabular_csv`:
     Creates `data/pe_malware_features.csv` (2,500 samples, 16 static/behavioral
     PE executable features, and 5 target classes in `malware_family`:
     `Benign`, `Ransomware`, `Trojan`, `Spyware`, `Worm`).
  2. `generate_malware_byteplot_images`:
     Creates `data/malware_byteplots/` (ImageFolder tree) and
     `data/malware_byteplots_manifest.csv` (`image_path`, `malware_family`),
     modeling the classic MalImg grayscale binary visualization where raw
     executable byte streams are reshaped into 2D grayscale texture images.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image


MALWARE_CLASSES = ["Benign", "Ransomware", "Trojan", "Spyware", "Worm"]

PE_FEATURE_NAMES = [
    "byte_entropy_mean",
    "byte_entropy_max",
    "pe_import_count",
    "pe_section_count",
    "virtual_to_raw_ratio",
    "is_packed_flag",
    "suspicious_api_calls",
    "tls_callback_count",
    "resource_entropy",
    "export_table_size",
    "code_section_ratio",
    "debug_stripped_flag",
    "network_socket_imports",
    "registry_write_imports",
    "crypto_constant_hits",
    "section_permission_wx",
]


def generate_pe_malware_tabular_csv(
    csv_path: Path | str = "data/pe_malware_features.csv",
    num_samples: int = 2500,
    seed: int = 42,
) -> Path:
    """
    Generates a realistic multi-class PE malware analysis tabular dataset in CSV format.
    """
    out_path = Path(csv_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    num_classes = len(MALWARE_CLASSES)
    num_features = len(PE_FEATURE_NAMES)

    # Distinct class prototypes in 16-D feature space representing realistic malware signatures
    prototypes = np.array(
        [
            # Benign: moderate entropy, normal imports, low suspicious APIs, not packed
            [4.8, 6.1, 85.0, 5.0, 1.05, 0.05, 1.2, 0.1, 3.2, 12.0, 0.65, 0.20, 2.0, 1.5, 0.5, 0.02],
            # Ransomware: very high entropy, packed, high crypto hits, high suspicious APIs
            [7.4, 7.9, 32.0, 7.5, 2.40, 0.88, 18.5, 1.4, 7.5, 1.0, 0.28, 0.90, 4.5, 9.0, 14.0, 0.82],
            # Trojan: moderate-high entropy, high network socket imports, stealthy sections
            [6.2, 7.3, 64.0, 6.2, 1.55, 0.52, 12.0, 0.8, 5.8, 3.0, 0.48, 0.78, 15.0, 6.5, 4.0, 0.45],
            # Spyware: high registry writes, high network & API calls, moderate packing
            [5.9, 7.0, 72.0, 5.8, 1.35, 0.40, 14.5, 0.5, 5.1, 4.5, 0.52, 0.72, 11.5, 16.0, 2.5, 0.35],
            # Worm: high network socket imports, self-propagating packed stubs, TLS callbacks
            [6.8, 7.6, 45.0, 6.8, 1.95, 0.75, 15.0, 2.1, 6.4, 2.0, 0.36, 0.85, 19.5, 8.0, 5.5, 0.68],
        ],
        dtype=np.float64,
    )
    scales = np.array(
        [0.65, 0.45, 22.0, 1.5, 0.45, 0.35, 4.5, 0.7, 1.1, 4.0, 0.14, 0.35, 4.2, 3.8, 3.2, 0.32],
        dtype=np.float64,
    )

    # Create a linear mixing matrix so classes require non-trivial non-linear boundaries
    mix_mat = np.eye(num_features) + rng.normal(0.0, 0.12, size=(num_features, num_features))

    rows = []
    samples_per_class = num_samples // num_classes
    for c_idx, c_name in enumerate(MALWARE_CLASSES):
        n_c = samples_per_class if c_idx < num_classes - 1 else (num_samples - len(rows))
        z = rng.normal(0.0, 1.0, size=(n_c, num_features))
        feats = prototypes[c_idx] + (z @ mix_mat) * scales

        # Clamp non-negative physical quantities and binary flags
        feats[:, 0] = np.clip(feats[:, 0], 1.0, 8.0)   # byte_entropy_mean in [1, 8]
        feats[:, 1] = np.clip(feats[:, 1], 2.0, 8.0)   # byte_entropy_max in [2, 8]
        feats[:, 2] = np.clip(np.round(feats[:, 2]), 1, 300)
        feats[:, 3] = np.clip(np.round(feats[:, 3]), 1, 16)
        feats[:, 4] = np.clip(feats[:, 4], 0.5, 5.0)
        feats[:, 5] = (feats[:, 5] > 0.5).astype(int)  # binary flag
        feats[:, 6] = np.clip(np.round(feats[:, 6]), 0, 50)
        feats[:, 7] = np.clip(np.round(feats[:, 7]), 0, 8)
        feats[:, 8] = np.clip(feats[:, 8], 0.5, 8.0)
        feats[:, 9] = np.clip(np.round(feats[:, 9]), 0, 100)
        feats[:, 10] = np.clip(feats[:, 10], 0.05, 0.98)
        feats[:, 11] = (feats[:, 11] > 0.5).astype(int)
        feats[:, 12] = np.clip(np.round(feats[:, 12]), 0, 40)
        feats[:, 13] = np.clip(np.round(feats[:, 13]), 0, 40)
        feats[:, 14] = np.clip(np.round(feats[:, 14]), 0, 35)
        feats[:, 15] = (feats[:, 15] > 0.5).astype(int)

        for i in range(n_c):
            row = [f"{feats[i, j]:.4f}" if j in (0, 1, 4, 8, 10) else f"{int(feats[i, j])}" for j in range(num_features)]
            row.append(c_name)
            rows.append(row)

    rng.shuffle(rows)

    header = PE_FEATURE_NAMES + ["malware_family"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    return out_path


def generate_malware_byteplot_images(
    root_dir: Path | str = "data/malware_byteplots",
    manifest_csv_path: Path | str = "data/malware_byteplots_manifest.csv",
    num_samples: int = 600,
    image_size: int = 32,
    seed: int = 42,
) -> Tuple[Path, Path]:
    """
    Generates a synthetic Malware Grayscale Byteplot dataset (inspired by MalImg):
      - Saves 32x32 grayscale `.png` images inside `data/malware_byteplots/<Class>/`
      - Saves a CSV manifest `data/malware_byteplots_manifest.csv` with columns:
        `image_path,malware_family` so the user can test both ImageFolder and
        CSV-Manifest image ingestion!
    """
    root_path = Path(root_dir)
    manifest_path = Path(manifest_csv_path)
    root_path.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    num_classes = len(MALWARE_CLASSES)
    per_class = num_samples // num_classes

    yy, xx = np.meshgrid(
        np.linspace(0, 1, image_size), np.linspace(0, 1, image_size), indexing="ij"
    )

    manifest_rows = []
    for c_idx, c_name in enumerate(MALWARE_CLASSES):
        c_dir = root_path / c_name
        c_dir.mkdir(parents=True, exist_ok=True)

        for idx in range(per_class):
            noise = rng.normal(0.0, 0.18, size=(image_size, image_size))
            phase = rng.uniform(0, 2 * np.pi)

            if c_idx == 0:  # Benign: structured horizontal PE section bands
                pattern = 0.5 + 0.35 * np.sin(2 * np.pi * 3.0 * yy + phase)
            elif c_idx == 1:  # Ransomware: high-entropy dense block in lower 70% + bright header
                pattern = np.where(yy < 0.25, 0.85, 0.35) + rng.normal(0.0, 0.28, size=(image_size, image_size))
            elif c_idx == 2:  # Trojan: diagonal stride bytecode texture
                pattern = 0.5 + 0.35 * np.sin(2 * np.pi * 4.0 * (xx + yy) + phase)
            elif c_idx == 3:  # Spyware: vertical import/resource stripes + center block
                pattern = 0.45 + 0.32 * np.cos(2 * np.pi * 4.0 * xx + phase)
            else:  # Worm: concentric / checkerboard packed stub pattern
                pattern = 0.5 + 0.35 * np.sin(2 * np.pi * 3.0 * xx) * np.cos(2 * np.pi * 3.0 * yy + phase)

            img_arr = np.clip((pattern + noise) * 255.0, 0, 255).astype(np.uint8)
            img_file = c_dir / f"{c_name.lower()}_{idx:04d}.png"
            Image.fromarray(img_arr, mode="L").save(img_file)
            manifest_rows.append([str(img_file), c_name])

    rng.shuffle(manifest_rows)
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "malware_family"])
        writer.writerows(manifest_rows)

    return root_path, manifest_path


def ensure_sample_datasets_exist(base_data_dir: Path | str = "data") -> dict:
    """Ensures both sample malware datasets exist on disk and returns their paths."""
    base = Path(base_data_dir)
    csv_path = base / "pe_malware_features.csv"
    img_dir = base / "malware_byteplots"
    manifest_path = base / "malware_byteplots_manifest.csv"

    if not csv_path.exists():
        generate_pe_malware_tabular_csv(csv_path=csv_path)
    if not manifest_path.exists() or not img_dir.exists():
        generate_malware_byteplot_images(root_dir=img_dir, manifest_csv_path=manifest_path)

    return {
        "tabular_malware_csv": str(csv_path),
        "image_malware_folder": str(img_dir),
        "image_malware_manifest_csv": str(manifest_path),
    }
