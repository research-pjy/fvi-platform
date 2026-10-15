"""
Data Ingestion & Auto-Schema Detection for Tabular and Image Datasets.
"""

from fvi.ingestion.loader import LoadedDatasetBundle, load_and_preprocess_dataset
from fvi.ingestion.sample_datasets import (
    ensure_sample_datasets_exist,
    generate_malware_byteplot_images,
    generate_pe_malware_tabular_csv,
)
from fvi.ingestion.schema import ColumnProfile, DatasetSchemaReport, inspect_dataset_source

__all__ = [
    "ColumnProfile",
    "DatasetSchemaReport",
    "inspect_dataset_source",
    "LoadedDatasetBundle",
    "load_and_preprocess_dataset",
    "generate_pe_malware_tabular_csv",
    "generate_malware_byteplot_images",
    "ensure_sample_datasets_exist",
]
