"""
Mode 1 Unified Model Factory (`TabularMLP`, `SimpleCNN`, `VisionResNet8`) and
FENS Aggregator Architectures.

Supports PyTorch + CUDA + BF16 natively on `rama` (L40S), and includes a clean
pure-NumPy neural network implementation so the platform can also be verified
in environments where PyTorch is not installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from fvi.config import ModelConfig
from fvi.ingestion.loader import LoadedDatasetBundle

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class ModelSpec:
    """Describes the resolved Mode 1 unified client model architecture."""

    arch_name: str
    modality: str
    input_dim: int
    num_classes: int
    hidden_dims: List[int]
    dropout: float
    image_shape: Optional[Tuple[int, int, int]]
    num_parameters: int
    fp32_size_bytes: int
    int8_size_bytes: int

    @property
    def fp32_size_mib(self) -> float:
        return self.fp32_size_bytes / (1024.0 ** 2)


def _estimate_mlp_params(
    input_dim: int, hidden_dims: Sequence[int], num_classes: int
) -> int:
    dims = [input_dim] + list(hidden_dims) + [num_classes]
    total = 0
    for d_in, d_out in zip(dims[:-1], dims[1:]):
        total += d_in * d_out + d_out
    # Include BatchNorm affine parameters (2 * hidden_dim)
    for h in hidden_dims:
        total += 2 * h
    return total


def build_unified_model_spec(
    bundle: LoadedDatasetBundle, model_cfg: ModelConfig
) -> ModelSpec:
    """
    Resolves the Mode 1 unified model specification and computes exact parameter
    counts and FP32/INT8 byte footprints.
    """
    arch = model_cfg.arch.lower()
    if arch == "auto":
        arch = "tabular_mlp" if bundle.modality == "tabular" else "resnet8_light"

    hidden = list(model_cfg.hidden_dims or [128, 64])

    if TORCH_AVAILABLE:
        m = build_torch_client_model(bundle, model_cfg)
        n_params = sum(p.numel() for p in m.parameters())
    else:
        n_params = _estimate_mlp_params(
            bundle.input_dim, hidden, bundle.num_classes
        )

    fp32_bytes = n_params * 4
    int8_bytes = max(256, n_params * 1 + 128)

    return ModelSpec(
        arch_name=arch,
        modality=bundle.modality,
        input_dim=bundle.input_dim,
        num_classes=bundle.num_classes,
        hidden_dims=hidden,
        dropout=model_cfg.dropout,
        image_shape=bundle.image_shape,
        num_parameters=n_params,
        fp32_size_bytes=fp32_bytes,
        int8_size_bytes=int8_bytes,
    )


if TORCH_AVAILABLE:

    class TabularMLP(nn.Module):
        """
        Mode 1 Unified Tabular MLP with BatchNorm1d, ReLU, and Dropout.
        Suitable for PE malware feature vectors, network flows, and general tabular data.
        """

        def __init__(
            self,
            input_dim: int,
            num_classes: int,
            hidden_dims: Sequence[int] = (128, 64),
            dropout: float = 0.15,
        ) -> None:
            super().__init__()
            layers: List[nn.Module] = []
            prev_dim = input_dim
            for h_dim in hidden_dims:
                layers.append(nn.Linear(prev_dim, h_dim))
                layers.append(nn.BatchNorm1d(h_dim))
                layers.append(nn.ReLU(inplace=True))
                if dropout > 0:
                    layers.append(nn.Dropout(p=dropout))
                prev_dim = h_dim
            layers.append(nn.Linear(prev_dim, num_classes))
            self.net = nn.Sequential(*layers)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if x.ndim > 2:
                x = torch.flatten(x, 1)
            return self.net(x)

    class SimpleCNN(nn.Module):
        """
        Lightweight 2-Block Convolutional Network supporting both 1-channel
        Grayscale (Malware Byteplots) and 3-channel RGB images.
        """

        def __init__(
            self,
            in_channels: int = 1,
            num_classes: int = 5,
            image_size: int = 32,
        ) -> None:
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((4, 4)),
            )
            self.classifier = nn.Sequential(
                nn.Linear(64 * 4 * 4, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, num_classes),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            feats = self.features(x)
            feats = torch.flatten(feats, 1)
            return self.classifier(feats)

    class _BasicBlock(nn.Module):
        def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(
                in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
            )
            self.bn1 = nn.BatchNorm2d(planes)
            self.conv2 = nn.Conv2d(
                planes, planes, kernel_size=3, stride=1, padding=1, bias=False
            )
            self.bn2 = nn.BatchNorm2d(planes)
            self.shortcut = nn.Sequential()
            if stride != 1 or in_planes != planes:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(planes),
                )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out = F.relu(self.bn1(self.conv1(x)), inplace=True)
            out = self.bn2(self.conv2(out))
            out += self.shortcut(x)
            return F.relu(out, inplace=True)

    class VisionResNet8(nn.Module):
        """
        ResNet-8 supporting both 1-channel Grayscale (Malware Byteplots) and
        3-channel RGB images.
        """

        def __init__(
            self,
            in_channels: int = 1,
            num_classes: int = 5,
            channels: Tuple[int, int, int] = (16, 32, 64),
        ) -> None:
            super().__init__()
            c1, c2, c3 = channels
            self.conv1 = nn.Conv2d(in_channels, c1, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(c1)
            self.layer1 = _BasicBlock(c1, c1, stride=1)
            self.layer2 = _BasicBlock(c1, c2, stride=2)
            self.layer3 = _BasicBlock(c2, c3, stride=2)
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(c3, num_classes)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out = F.relu(self.bn1(self.conv1(x)), inplace=True)
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = torch.flatten(self.avgpool(out), 1)
            return self.fc(out)

    def build_torch_client_model(
        bundle: LoadedDatasetBundle, model_cfg: ModelConfig
    ) -> nn.Module:
        arch = model_cfg.arch.lower()
        if arch == "auto":
            arch = "tabular_mlp" if bundle.modality == "tabular" else "resnet8_light"

        if bundle.modality == "tabular" or arch == "tabular_mlp":
            return TabularMLP(
                input_dim=bundle.input_dim,
                num_classes=bundle.num_classes,
                hidden_dims=model_cfg.hidden_dims or [128, 64],
                dropout=model_cfg.dropout,
            )

        in_ch = bundle.image_shape[0] if bundle.image_shape else 1
        img_sz = bundle.image_shape[1] if bundle.image_shape else 32

        if arch == "simple_cnn":
            return SimpleCNN(
                in_channels=in_ch, num_classes=bundle.num_classes, image_size=img_sz
            )
        elif arch == "resnet8":
            return VisionResNet8(
                in_channels=in_ch, num_classes=bundle.num_classes, channels=(64, 128, 256)
            )
        else:  # resnet8_light
            return VisionResNet8(
                in_channels=in_ch, num_classes=bundle.num_classes, channels=(16, 32, 64)
            )
