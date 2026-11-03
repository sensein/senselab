# ruff: noqa: E501
"""Declarative input requirements of video, vision and audio models, and the encode targets they imply.

A ``ModelInputSpec`` records what a model's processor expects (fixed or patch-multiple dimensions, frame rate,
frame budget, audio rate and channels, colour format) and where that came from. ``standardize_video_encoding(
video, out, model="qwen3-vl")`` turns the spec into encode targets. Entries marked ``verified=False`` were not
read from a model card or processor config in this repository's session; treat them as starting points.

``requires_specific_dims`` answers "does the model fail or degrade unless the input has exact dimensions?":
``True`` only when the model itself performs no resize (the caller must prepare the size); ``False`` when its
processor resizes or accepts any size (a patch multiple may still be preferred); ``None`` unknown.
Frame sampling (how many frames, at what rate) is done by each model's processor; ``fps`` here is a preferred
rate for pre-resampling, not a requirement, and ``max_frames`` / ``max_duration_s`` are informational.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple


@dataclass(frozen=True)
class ModelInputSpec:
    """What one model wants as input (see module docstring)."""

    name: str
    modality: str  # "video", "image", "audio" or "audio+video"
    requires_specific_dims: Optional[bool]
    source: str
    verified: bool
    fixed_size: Optional[Tuple[int, int]] = None  # (W, H) the model needs when requires_specific_dims is True
    resize_mode: str = "crop"  # how to reach fixed_size: fit | pad | crop | stretch
    processor_size: Optional[Tuple[int, int]] = None  # (W, H) the model's own processor resizes to (informational)
    patch_multiple: Optional[int] = None  # dimensions preferably a multiple of this (patch size x merge size)
    max_side: Optional[int] = None
    fps: Optional[float] = None
    max_frames: Optional[int] = None
    max_duration_s: Optional[float] = None
    color: str = "rgb"
    audio_rate: Optional[int] = None
    audio_channels: Optional[int] = None
    notes: str = ""

    def encode_targets(self) -> Dict[str, Any]:
        """Targets for ``standardize_video_encoding``: resolution, mode, multiple, fps, audio rate and channels."""
        out: Dict[str, Any] = {"fps": self.fps, "audio_rate": self.audio_rate, "audio_channels": self.audio_channels}
        if self.requires_specific_dims and self.fixed_size:
            out.update(resolution=self.fixed_size, resolution_mode=self.resize_mode)
        elif self.max_side:
            out["resolution"] = self.max_side
        if self.patch_multiple:
            out["resolution_multiple"] = self.patch_multiple
        return out


_CACHE = "processor config in the SAILS model cache (read 2026-11-02)"
MODEL_INPUT_SPECS: Dict[str, ModelInputSpec] = {
    s.name: s
    for s in (
        ModelInputSpec(
            name="sam3",
            modality="video",
            requires_specific_dims=False,
            processor_size=(1008, 1008),
            verified=True,
            source=f"facebook/sam3 processor_config.json: image and video processors resize to 1008x1008; {_CACHE}",
            notes="The processor stretches to a square; original aspect is restored by the caller for masks.",
        ),
        ModelInputSpec(
            name="hrnet",
            modality="image",
            requires_specific_dims=False,
            processor_size=(192, 256),
            verified=False,
            source="deep-high-resolution-net.pytorch top-down configs: person crops of 192x256 (W x H) from detector boxes",
            notes="Operates on detector crops, not on the whole frame; any video size is accepted.",
        ),
        ModelInputSpec(
            name="vitpose-plus-huge",
            modality="image",
            requires_specific_dims=False,
            processor_size=(192, 256),
            verified=True,
            source=f"usyd-community/vitpose-plus-huge preprocessor_config.json size 256x192; {_CACHE}",
            notes="Top-down; person crops.",
        ),
        ModelInputSpec(
            name="vjepa2",
            modality="video",
            requires_specific_dims=False,
            processor_size=(256, 256),
            max_frames=64,
            verified=True,
            source=f"facebook/vjepa2-vitg-fpc64-256 video_preprocessor_config.json: resize shortest edge 292, "
            f"centre crop 256x256; fpc64 = 64 frames per clip; {_CACHE}",
            notes="Processor does the resize and crop; pass fixed_size=(256, 256) with mode crop to pre-apply it.",
        ),
        ModelInputSpec(
            name="vjepa2.1",
            modality="video",
            requires_specific_dims=None,
            verified=False,
            source="V-JEPA 2.1 release; not in the SAILS model cache, no processor config read",
            notes="Expected to follow V-JEPA 2 (square crop, 64-frame clips); confirm before relying on it.",
        ),
        ModelInputSpec(
            name="dinov3",
            modality="image",
            requires_specific_dims=False,
            patch_multiple=16,
            verified=False,
            source="DINOv3 ViT patch size 16 (model card); not in the SAILS model cache, no processor config read",
            notes="Accepts any size that is a multiple of the patch size; the HF processor resizes by default.",
        ),
        ModelInputSpec(
            name="qwen3-omni",
            modality="audio+video",
            requires_specific_dims=False,
            patch_multiple=32,
            audio_rate=16000,
            audio_channels=1,
            verified=True,
            source=f"Qwen/Qwen3-Omni-30B-A3B-Instruct preprocessor_config.json: patch_size 16, merge_size 2, sampling_rate "
            f"16000, min/max_pixels 3136/12845056, temporal_patch_size 2; {_CACHE}",
            notes="Mono is assumed (Whisper-style mel front end, feature_size 128); frame rate is chosen by the processor.",
        ),
        ModelInputSpec(
            name="qwen2.5-vl",
            modality="video",
            requires_specific_dims=False,
            patch_multiple=28,
            verified=True,
            source=f"Qwen/Qwen2.5-VL-7B-Instruct preprocessor_config.json: patch_size 14, merge_size 2, "
            f"min/max_pixels 3136/12845056, temporal_patch_size 2; {_CACHE}",
        ),
        ModelInputSpec(
            name="qwen3-vl",
            modality="video",
            requires_specific_dims=False,
            patch_multiple=32,
            verified=True,
            source=f"Qwen/Qwen3-VL-8B-Instruct preprocessor_config.json and video_preprocessor_config.json: patch_size 16, "
            f"merge_size 2, temporal_patch_size 2; {_CACHE}",
        ),
        ModelInputSpec(
            name="internvl3.5",
            modality="video",
            requires_specific_dims=False,
            processor_size=(384, 384),
            patch_multiple=14,
            verified=True,
            source=f"OpenGVLab/InternVL3_5-8B-HF video_preprocessor_config.json size 384x384 (images 448x448 "
            f"tiles), vision patch_size 14, downsample_ratio 0.5; {_CACHE}",
            notes="The processor resizes to a square.",
        ),
        ModelInputSpec(
            name="ovis2.5",
            modality="video",
            requires_specific_dims=None,
            processor_size=(512, 512),
            patch_multiple=16,
            verified=False,
            source=f"ATH-MaaS/Ovis2.5-9B preprocessor_config.json size 512x512, vit patch_size 16; {_CACHE}",
            notes="Native-resolution ViT is stated on the model card (not machine-checked): dims need not be fixed.",
        ),
        ModelInputSpec(
            name="cosmos-reason2",
            modality="video",
            requires_specific_dims=False,
            patch_multiple=32,
            fps=4.0,
            verified=False,
            source=f"nvidia/Cosmos-Reason2-8B preprocessor configs: patch_size 16, merge_size 2 (verified, {_CACHE}); the "
            "4 fps preference is from the Cosmos-Reason model card and is not machine-checked",
        ),
        ModelInputSpec(
            name="whisper",
            modality="audio",
            requires_specific_dims=None,
            audio_rate=16000,
            audio_channels=1,
            verified=True,
            source=f"openai/whisper-large-v3 preprocessor_config.json sampling_rate 16000 (feature_size 128); {_CACHE}",
            notes="The feature extractor takes a mono waveform.",
        ),
        ModelInputSpec(
            name="crisperwhisper",
            modality="audio",
            requires_specific_dims=None,
            audio_rate=16000,
            audio_channels=1,
            verified=True,
            source=f"nyralabs/CrisperWhisper preprocessor_config.json sampling_rate 16000; {_CACHE}",
        ),
        ModelInputSpec(
            name="qwen3-asr",
            modality="audio",
            requires_specific_dims=None,
            audio_rate=16000,
            audio_channels=1,
            verified=False,
            source="Qwen/Qwen3-ASR-1.7B preprocessor_config.json has feature_size 128 but no sampling_rate; 16 kHz mono "
            "assumed from the Whisper-style front end",
        ),
    )
}


def get_model_spec(name: str) -> ModelInputSpec:
    """The registered spec for ``name`` (case-insensitive); ``KeyError`` lists the known names."""
    try:
        return MODEL_INPUT_SPECS[name.lower()]
    except KeyError:
        raise KeyError(f"unknown model {name!r}; known: {sorted(MODEL_INPUT_SPECS)}") from None
