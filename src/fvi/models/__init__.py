"""
Mode 1 Unified Client Models & FENS Aggregators for `fvi-platform`.
"""

from fvi.models.factory import (
    TORCH_AVAILABLE,
    ModelSpec,
    build_unified_model_spec,
)

__all__ = [
    "TORCH_AVAILABLE",
    "ModelSpec",
    "build_unified_model_spec",
]
