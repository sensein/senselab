""".. include:: ./doc.md"""  # noqa: D415

from .api import (
    EmptyVideoError,
    StandardizedVideo,
    VideoStandardizationError,
    standardize_video_encoding,
    target_dims,
)
from .model_specs import MODEL_INPUT_SPECS, ModelInputSpec, get_model_spec

__all__ = [
    "EmptyVideoError",
    "MODEL_INPUT_SPECS",
    "ModelInputSpec",
    "StandardizedVideo",
    "VideoStandardizationError",
    "get_model_spec",
    "standardize_video_encoding",
    "target_dims",
]
