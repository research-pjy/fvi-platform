"""
Configuration dataclasses for the Federated Viability Index (FVI) Platform.

Separates dataset/schema configuration, client partitioning parameters,
Mode 1 unified model architecture settings, FL algorithm hyperparameters,
and L40S (`rama`) hardware acceleration settings.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence


@dataclass
class HardwareConfig:
    """Hardware execution parameters for local development and `rama` (L40S)."""

    device: str = "cuda"
    use_bf16: bool = True
    num_workers: int = 2
    pin_memory: bool = True


@dataclass
class DatasetConfig:
    """
    Specifies the dataset source, modality, selected features, and target variable.
    Works uniformly across Tabular CSVs, Image Manifest CSVs, and ImageFolder trees.
    """

    data_path: str = "data/pe_malware_features.csv"
    modality: str = "tabular"  # "tabular" or "image"
    target_column: str = "malware_family"
    feature_columns: Optional[List[str]] = None  # None or ["all"] -> auto-select all non-target columns
    # Image-specific settings (used when modality == "image")
    image_path_column: Optional[str] = "image_path"  # For CSV manifest; None if ImageFolder directory
    image_size: int = 32
    image_channels: int = 1  # 1 for Grayscale (Malware Byteplots), 3 for RGB
    test_fraction: float = 0.20
    val_fraction: float = 0.10
    seed: int = 42


@dataclass
class PartitionConfig:
    """
    Specifies how the training dataset is split across simulated federated clients.
    """

    num_clients: int = 10
    strategy: str = "dirichlet"  # "iid", "dirichlet", "quantity_skew", "generalist_specialist"
    dirichlet_alpha: float = 0.1
    # Used when strategy == "generalist_specialist"
    specialist_classes: List[int] = field(default_factory=lambda: [0, 1])
    fens_train_fraction: float = 0.90  # 90% D_i1 local training, 10% D_i2 aggregator training
    seed: int = 42


@dataclass
class ModelConfig:
    """
    Mode 1 Unified Client Model Configuration.
    All clients train the exact same model backbone so data heterogeneity and
    FL algorithms are benchmarked on a controlled, apples-to-apples basis.
    """

    arch: str = "auto"  # "auto", "tabular_mlp", "simple_cnn", "resnet8_light", "resnet8"
    hidden_dims: List[int] = field(default_factory=lambda: [128, 64])
    dropout: float = 0.15


@dataclass
class BenchmarkConfig:
    """
    Hyperparameters for the Golden Triad benchmark runs:
      1. Centralized Upper Bound
      2. Isolated Local Silos (Lower Bound)
      3. Federated Strategies (FedAvg, FedProx, FedAdam, FENS)
    """

    algorithms: List[str] = field(
        default_factory=lambda: ["fedavg", "fedprox", "fedadam", "fens"]
    )
    # Centralized & Isolated Silo epochs
    centralized_epochs: int = 25
    isolated_epochs: int = 25
    # Iterative FL (FedAvg, FedProx, FedAdam) settings
    fl_global_rounds: int = 25
    fl_local_epochs: int = 2
    batch_size: int = 32
    client_lr: float = 0.01
    server_lr: float = 0.01
    prox_mu: float = 0.05
    # FENS (One-Shot + Phase 2 Aggregator) settings
    fens_phase1_epochs: int = 25
    fens_phase2_rounds: int = 150
    fens_aggregator_type: str = "mlp"  # "mlp" or "per_class_linear"
    fens_hidden_dim: int = 32
    fens_client_lr: float = 0.5
    fens_server_lr: float = 0.005
    fens_quantize_int8: bool = True
    # Optional multi-alpha sweep for exact Heterogeneity Resilience (HR) calculation
    run_iid_reference_for_hr: bool = True
    seed: int = 42


@dataclass
class FVJobConfig:
    """Complete job configuration payload submitted from the Web UI or CLI."""

    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    partition: PartitionConfig = field(default_factory=PartitionConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    output_dir: str = "outputs/latest_run"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
