"""Embeddings of a video from a Hugging Face self-supervised video model (for example V-JEPA 2).

The video is resampled to `fps`, cut into consecutive clips of `frames_per_clip` frames (a trailing partial
clip is dropped), and each clip is encoded; the token states of a clip are pooled to one vector. The model must be
given with a resolved 40-hex commit (a branch or tag name is refused), so a run names the exact weights it used.
"""

import inspect
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import torch

from senselab.utils.data_structures import DeviceType, HFModel
from senselab.utils.dependencies import resolve_model

_HEX40 = re.compile(r"[0-9a-f]{40}")
POOLINGS = ("mean",)

Preprocess = Callable[[torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class VideoSSLEmbeddings:
    """Per-clip embeddings of one video and how they were made.

    Attributes:
        embeddings: Tensor of shape `(n_clips, hidden)`.
        provenance: Model id and commit, clip settings, library versions and the device used.
    """

    embeddings: torch.Tensor
    provenance: Dict[str, Any] = field(default_factory=dict)


def _load_model(model: HFModel, device: Optional[DeviceType]) -> Tuple[Any, Preprocess, str]:
    """Load the model at its pinned commit.

    Returns:
        The model in eval mode, a function from one clip `(T, H, W, C)` uint8 to the model's `pixel_values_videos`
        `(1, T, C, H, W)`, and the device name.
    """
    from transformers import AutoModel, AutoVideoProcessor

    from senselab.utils.data_structures.device import _select_device_and_dtype

    name, revision = str(model.path_or_uri), model.revision
    chosen, _ = _select_device_and_dtype(user_preference=device, compatible_devices=[DeviceType.CUDA, DeviceType.CPU])
    sha, _ = resolve_model(name, revision)
    encoder = AutoModel.from_pretrained(name, revision=sha).to(chosen.value).eval()
    processor = AutoVideoProcessor.from_pretrained(name, revision=sha)

    def preprocess(clip: torch.Tensor) -> torch.Tensor:
        return processor(clip.permute(0, 3, 1, 2), return_tensors="pt")["pixel_values_videos"]

    return encoder, preprocess, chosen.value


def _resample_indices(n_frames: int, source_fps: float, fps: float) -> np.ndarray:
    """Frame indices that resample a video from `source_fps` to `fps` (nearest earlier frame)."""
    n_out = int(n_frames / source_fps * fps)
    return np.minimum(np.floor(np.arange(n_out) * source_fps / fps).astype(int), n_frames - 1)


def extract_ssl_embeddings_from_video(
    video: Any,  # noqa: ANN401 - senselab.video.data_structures.Video
    model: HFModel,
    *,
    frames_per_clip: int,
    fps: float,
    pooling: str,
    device: Optional[DeviceType] = None,
) -> VideoSSLEmbeddings:
    """Embed a video clip by clip with a Hugging Face video model.

    Args:
        video: A `Video` (frames `(T, H, W, C)` and a frame rate).
        model: A Hugging Face video model whose `revision` is a 40-hex commit.
        frames_per_clip: Frames in each clip, after resampling.
        fps: Frame rate the clips are sampled at.
        pooling: How a clip's token states become one vector; `mean`.
        device: Device for inference (default: auto-select).

    Returns:
        The `(n_clips, hidden)` embeddings and a provenance record.

    Raises:
        ValueError: If the revision is not a 40-hex commit, the pooling is unknown, a setting is not positive, or
            the resampled video is shorter than one clip.
    """
    if not _HEX40.fullmatch(model.revision):
        raise ValueError(f"model revision must be a resolved 40-hex commit, got {model.revision!r}")
    if pooling not in POOLINGS:
        raise ValueError(f"unknown pooling {pooling!r}; use one of {POOLINGS}")
    if frames_per_clip < 1 or fps <= 0:
        raise ValueError("frames_per_clip and fps must be positive")
    frames = video.frames
    indices = _resample_indices(frames.shape[0], float(video.frame_rate), fps)
    n_clips = len(indices) // frames_per_clip
    if n_clips < 1:
        raise ValueError(f"the video is shorter than one clip of {frames_per_clip} frames at {fps} fps")
    encoder, preprocess, device_name = _load_model(model, device)
    extra = {"skip_predictor": True} if "skip_predictor" in inspect.signature(encoder.forward).parameters else {}
    vectors = []
    with torch.no_grad():
        for c in range(n_clips):
            clip = frames[indices[c * frames_per_clip : (c + 1) * frames_per_clip]]
            pixels = preprocess(clip).to(device_name)
            states = encoder(pixel_values_videos=pixels, **extra).last_hidden_state
            vectors.append(states.mean(dim=1).squeeze(0).cpu())
    import transformers

    provenance = {
        "model": str(model.path_or_uri),
        "revision": model.revision,
        "frames_per_clip": frames_per_clip,
        "fps": fps,
        "pooling": pooling,
        "n_clips": n_clips,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "device": device_name,
    }
    return VideoSSLEmbeddings(embeddings=torch.stack(vectors), provenance=provenance)
