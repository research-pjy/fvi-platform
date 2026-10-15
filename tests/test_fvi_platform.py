"""
End-to-End Unit & Integration Test Suite for `fvi-platform` (FVI in [0.000, 1.000] & Zero-Disk In-RAM Mode).

Run locally or on `rama`:
    PYTHONPATH=src python3 -m unittest discover -s tests -v
"""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR / "src") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "src"))

from fvi.config import (
    BenchmarkConfig,
    DatasetConfig,
    FVJobConfig,
    ModelConfig,
    PartitionConfig,
)
from fvi.ingestion.loader import (
    bundle_from_in_memory_arrays,
    load_and_preprocess_dataset,
)
from fvi.ingestion.sample_datasets import (
    generate_malware_byteplot_images,
    generate_pe_malware_tabular_csv,
)
from fvi.ingestion.schema import inspect_dataset_source
from fvi.metrics.fvi_calculator import compute_single_fvi
from fvi.partition.splitter import partition_dataset
from fvi.pipeline import (
    preview_partition_from_bundle,
    run_fvi_benchmark_pipeline,
    run_fvi_on_loaded_bundle,
)


class TestFVIPlatform(unittest.TestCase):
    """Verifies Tabular & Image ingestion, Partitioning, FVI in [0, 1], and Zero-Disk In-RAM execution."""

    def test_pe_malware_tabular_ingestion_and_schema(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "malware.csv"
            generate_pe_malware_tabular_csv(csv_path, num_samples=300, seed=42)

            schema = inspect_dataset_source(csv_path)
            self.assertTrue(schema.exists)
            self.assertEqual(schema.detected_modality, "tabular")
            self.assertEqual(schema.suggested_target, "malware_family")
            self.assertEqual(len(schema.suggested_features), 16)

            bundle = load_and_preprocess_dataset(
                DatasetConfig(data_path=str(csv_path), modality="tabular", target_column="malware_family")
            )
            self.assertEqual(bundle.num_classes, 5)
            self.assertEqual(bundle.input_dim, 16)
            self.assertEqual(len(bundle.X_train) + len(bundle.X_val) + len(bundle.X_test), 300)

    def test_malware_byteplot_image_ingestion_folder_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            img_dir = Path(tmp) / "byteplots"
            manifest_csv = Path(tmp) / "manifest.csv"
            generate_malware_byteplot_images(
                root_dir=img_dir, manifest_csv_path=manifest_csv, num_samples=100, image_size=32
            )

            schema_folder = inspect_dataset_source(img_dir)
            self.assertEqual(schema_folder.detected_modality, "image")
            self.assertEqual(schema_folder.source_format, "image_folder")

            bundle_folder = load_and_preprocess_dataset(
                DatasetConfig(data_path=str(img_dir), modality="image", image_size=32, image_channels=1)
            )
            self.assertEqual(bundle_folder.num_classes, 5)
            self.assertEqual(bundle_folder.image_shape, (1, 32, 32))

    def test_all_four_partitioning_strategies(self) -> None:
        y = np.repeat(np.arange(5), 100)
        for strat in ("iid", "dirichlet", "quantity_skew", "generalist_specialist"):
            part = partition_dataset(
                y,
                PartitionConfig(num_clients=6, strategy=strat, dirichlet_alpha=0.1, specialist_classes=[0, 1]),
                num_classes=5,
            )
            self.assertEqual(part.num_clients, 6)
            self.assertEqual(sum(part.client_sample_totals), 500)
            self.assertTrue(0.0 <= part.mean_normalized_entropy <= 1.0)

    def test_fvi_mathematical_bounds_in_zero_to_one(self) -> None:
        # High viability case: FL (89%) close to Centralized (90%) and well above Isolated (65%)
        cpr, cg, hr, fvi = compute_single_fvi(a_fl=89.0, a_cent=90.0, a_local=65.0, a_iid=90.0)
        self.assertGreaterEqual(fvi, 0.90)
        self.assertLessEqual(fvi, 1.00)
        self.assertTrue(0.0 <= cpr <= 1.0 and 0.0 <= cg <= 1.0 and 0.0 <= hr <= 1.0)

        # Low viability case: FL (55%) no better than Isolated (58%) vs Centralized (90%)
        _, cg_low, _, fvi_low = compute_single_fvi(a_fl=55.0, a_cent=90.0, a_local=58.0, a_iid=88.0)
        self.assertEqual(cg_low, 0.0)
        self.assertLess(fvi_low, 0.50)

    def test_zero_disk_in_memory_pipeline(self) -> None:
        rng = np.random.default_rng(42)
        X = rng.normal(0.0, 1.0, size=(240, 12)).astype(np.float32)
        y = np.repeat(np.arange(4), 60).astype(np.int64)
        # Add class signal
        for c in range(4):
            X[y == c, c * 3 : (c + 1) * 3] += 2.5

        bundle = bundle_from_in_memory_arrays(
            X_raw=X,
            y_raw=y,
            modality="tabular",
            class_names=["Class0", "Class1", "Class2", "Class3"],
            feature_names=[f"f{i}" for i in range(12)],
        )
        job_cfg = FVJobConfig(
            partition=PartitionConfig(num_clients=4, strategy="dirichlet", dirichlet_alpha=0.1),
            model=ModelConfig(arch="auto", hidden_dims=[32, 16]),
            benchmark=BenchmarkConfig(
                algorithms=["fedavg", "fens"],
                centralized_epochs=5,
                isolated_epochs=5,
                fl_global_rounds=5,
                fl_local_epochs=1,
                fens_phase1_epochs=5,
                fens_phase2_rounds=15,
            ),
        )
        prev = preview_partition_from_bundle(bundle, job_cfg)
        self.assertTrue(prev["partition_chart_base64"].startswith("data:image/png;base64,"))

        res = run_fvi_on_loaded_bundle(
            bundle=bundle,
            job_cfg=job_cfg,
            dataset_label="Browser In-Memory Tensor",
            save_artifacts_to_disk=False,
        )
        fvi_val = res["scorecard"]["overall_fvi_score"]
        self.assertTrue(0.0 <= fvi_val <= 1.0)
        self.assertEqual(res["report_path"], "")


if __name__ == "__main__":
    unittest.main(verbosity=2)
