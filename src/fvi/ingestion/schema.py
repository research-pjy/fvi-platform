"""
Dataset Schema Inspector for the Federated Viability Index (FVI) Platform.

Automatically inspects a CSV file or an ImageFolder directory to determine:
  - Modality (`tabular` vs `image`)
  - Available columns and their inferred types (`numerical`, `categorical`, `image_path`)
  - Suggested `target_column` and `feature_columns`
  - Class distribution counts and sample preview rows for the Web UI
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".pgm", ".tif", ".tiff", ".webp"}


@dataclass
class ColumnProfile:
    """Metadata profile for a single column in a CSV dataset."""

    name: str
    inferred_type: str  # "numerical", "categorical", or "image_path"
    unique_count: int
    missing_count: int
    sample_values: List[str]


@dataclass
class DatasetSchemaReport:
    """Complete schema inspection report returned to the Web UI or CLI."""

    data_path: str
    exists: bool
    detected_modality: str  # "tabular" or "image"
    source_format: str      # "csv" or "image_folder"
    num_rows: int
    columns: List[ColumnProfile]
    suggested_target: str
    suggested_features: List[str]
    class_distribution: Dict[str, int]
    preview_rows: List[Dict[str, str]]
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _is_float(val: str) -> bool:
    try:
        float(val)
        return True
    except (ValueError, TypeError):
        return False


def _looks_like_image_path(val: str) -> bool:
    s = val.strip().lower()
    return any(s.endswith(ext) for ext in IMAGE_EXTENSIONS)


def inspect_dataset_source(
    data_path: Path | str,
    target_column: Optional[str] = None,
) -> DatasetSchemaReport:
    """
    Inspects either a CSV file (`tabular` or `image` manifest) or a directory
    of class subfolders (`image_folder`) and returns a `DatasetSchemaReport`.
    """
    path = Path(data_path)
    if not path.exists():
        return DatasetSchemaReport(
            data_path=str(data_path),
            exists=False,
            detected_modality="tabular",
            source_format="unknown",
            num_rows=0,
            columns=[],
            suggested_target="",
            suggested_features=[],
            class_distribution={},
            preview_rows=[],
            error_message=f"Path not found: {data_path}",
        )

    # Case 1: Directory -> ImageFolder structure (`root/<class_name>/*.png`)
    if path.is_dir():
        class_dist: Dict[str, int] = {}
        preview_rows: List[Dict[str, str]] = []
        total_images = 0

        for sub in sorted(path.iterdir()):
            if sub.is_dir() and not sub.name.startswith("."):
                imgs = [
                    p
                    for p in sub.iterdir()
                    if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
                ]
                if imgs:
                    class_dist[sub.name] = len(imgs)
                    total_images += len(imgs)
                    for p in imgs[:2]:
                        if len(preview_rows) < 6:
                            preview_rows.append(
                                {"image_path": str(p), "class_label": sub.name}
                            )

        columns = [
            ColumnProfile(
                name="image_path",
                inferred_type="image_path",
                unique_count=total_images,
                missing_count=0,
                sample_values=[r["image_path"] for r in preview_rows[:3]],
            ),
            ColumnProfile(
                name="class_label",
                inferred_type="categorical",
                unique_count=len(class_dist),
                missing_count=0,
                sample_values=list(class_dist.keys())[:5],
            ),
        ]
        return DatasetSchemaReport(
            data_path=str(path),
            exists=True,
            detected_modality="image",
            source_format="image_folder",
            num_rows=total_images,
            columns=columns,
            suggested_target="class_label",
            suggested_features=["image_path"],
            class_distribution=class_dist,
            preview_rows=preview_rows,
        )

    # Case 2: CSV file (either Tabular dataset or Image Manifest CSV)
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        try:
            header = [h.strip() for h in next(reader)]
        except StopIteration:
            return DatasetSchemaReport(
                data_path=str(path),
                exists=True,
                detected_modality="tabular",
                source_format="csv",
                num_rows=0,
                columns=[],
                suggested_target="",
                suggested_features=[],
                class_distribution={},
                preview_rows=[],
                error_message="CSV file is empty.",
            )

        raw_rows = list(reader)

    num_rows = len(raw_rows)
    col_profiles: List[ColumnProfile] = []
    has_image_col = False

    for c_idx, col_name in enumerate(header):
        vals = [r[c_idx].strip() for r in raw_rows if c_idx < len(r)]
        non_empty = [v for v in vals if v != ""]
        missing = num_rows - len(non_empty)
        unique_cnt = len(set(non_empty))
        sample_check = non_empty[:50]

        if sample_check and all(_looks_like_image_path(v) for v in sample_check):
            c_type = "image_path"
            has_image_col = True
        elif sample_check and sum(_is_float(v) for v in sample_check) >= int(0.9 * len(sample_check)):
            c_type = "numerical"
        else:
            c_type = "categorical"

        col_profiles.append(
            ColumnProfile(
                name=col_name,
                inferred_type=c_type,
                unique_count=unique_cnt,
                missing_count=missing,
                sample_values=non_empty[:4],
            )
        )

    # Choose default target column (user-specified or last column)
    if target_column and target_column in header:
        chosen_target = target_column
    else:
        chosen_target = header[-1]

    target_idx = header.index(chosen_target)
    class_dist = {}
    for r in raw_rows:
        if target_idx < len(r):
            lbl = r[target_idx].strip()
            if lbl:
                class_dist[lbl] = class_dist.get(lbl, 0) + 1

    if has_image_col:
        detected_modality = "image"
        suggested_features = [
            c.name for c in col_profiles if c.inferred_type == "image_path" and c.name != chosen_target
        ]
    else:
        detected_modality = "tabular"
        suggested_features = [c.name for c in col_profiles if c.name != chosen_target]

    preview_dicts = []
    for r in raw_rows[:5]:
        preview_dicts.append(
            {header[i]: (r[i] if i < len(r) else "") for i in range(len(header))}
        )

    return DatasetSchemaReport(
        data_path=str(path),
        exists=True,
        detected_modality=detected_modality,
        source_format="csv",
        num_rows=num_rows,
        columns=col_profiles,
        suggested_target=chosen_target,
        suggested_features=suggested_features,
        class_distribution=class_dist,
        preview_rows=preview_dicts,
    )
