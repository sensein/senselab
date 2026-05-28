"""HuggingFace transformers WavLM speaker-embedding backend.

Mirrors the SpeechBrain backend's contract: takes a batch of mono 16 kHz
``Audio`` objects, returns ``List[torch.Tensor]`` (one 1-D embedding per
input). Default checkpoint is ``microsoft/wavlm-base-plus-sv`` — the only
official WavLM checkpoint published with an X-Vector / SV head
(``microsoft/wavlm-large`` is the headless backbone). The model id is
configurable so a WavLM-Large SV checkpoint can be substituted if one
becomes available (FR-019).

WavLM SSL pretraining on a large, diverse, noise/overlap-aware corpus gives
genuine error decorrelation against the VoxCeleb-supervised SpeechBrain
models (ECAPA, ResNet-TDNN), which is the rationale for including it in the
default three-way consensus (FR-018, see research.md R3).
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

from senselab.audio.data_structures import Audio
from senselab.utils.data_structures import DeviceType, TransformersWavLMModel, _select_device_and_dtype
from senselab.utils.dependencies import retry_on_transient_error

try:
    from transformers import AutoFeatureExtractor, WavLMForXVector

    WAVLM_AVAILABLE = True
except ImportError:
    WAVLM_AVAILABLE = False


class WavLMEmbeddings:
    """A factory for extracting speaker embeddings using transformers WavLM SV models."""

    # Cache loaded (model, feature_extractor) pairs by (path_or_uri, revision, device).
    _models: Dict[str, tuple] = {}  # type: ignore[type-arg]

    @classmethod
    def _get_wavlm_model(
        cls,
        model: TransformersWavLMModel,
        device: Optional[DeviceType] = None,
    ) -> tuple:  # (WavLMForXVector, AutoFeatureExtractor, torch.device)
        """Get or create a WavLM-SV model + matching feature extractor."""
        if not WAVLM_AVAILABLE:
            raise ModuleNotFoundError(
                "`transformers` is not installed or does not provide `WavLMForXVector`. "
                "Please install senselab audio dependencies."
            )

        device_type, _ = _select_device_and_dtype(
            user_preference=device, compatible_devices=[DeviceType.CUDA, DeviceType.CPU]
        )
        torch_device = torch.device(device_type.value)

        key = f"{model.path_or_uri}-{model.revision}-{device_type.value}"
        if key not in cls._models:
            extractor = retry_on_transient_error(
                AutoFeatureExtractor.from_pretrained,
                str(model.path_or_uri),
                revision=model.revision,
            )
            net: WavLMForXVector = retry_on_transient_error(
                WavLMForXVector.from_pretrained,
                str(model.path_or_uri),
                revision=model.revision,
            )
            net = net.to(torch_device).eval()  # type: ignore[arg-type,assignment]
            cls._models[key] = (net, extractor, torch_device)
        return cls._models[key]

    @classmethod
    def extract_wavlm_speaker_embeddings_from_audios(
        cls,
        audios: List[Audio],
        model: Optional[TransformersWavLMModel] = None,
        device: Optional[DeviceType] = None,
    ) -> List[torch.Tensor]:
        """Compute speaker embeddings for a batch of mono 16 kHz audios.

        Matches the SpeechBrain backend's contract:

        - Input order is preserved.
        - Each audio must be mono and 16 kHz (the model card's expected rate);
          we raise on mismatch rather than silently resampling.
        - Returns a list of 1-D ``torch.Tensor`` (typically 512-D for
          ``wavlm-base-plus-sv``).

        Args:
            audios: Mono 16 kHz ``Audio`` objects to embed.
            model: WavLM SV model handle. Defaults to
                ``microsoft/wavlm-base-plus-sv`` @ ``main``.
            device: Optional ``DeviceType`` (CPU or CUDA).

        Returns:
            One 1-D embedding tensor per input.
        """
        if not WAVLM_AVAILABLE:
            raise ModuleNotFoundError(
                "`transformers` is not installed or does not provide `WavLMForXVector`. "
                "Please install senselab audio dependencies."
            )

        if len(audios) == 0:
            return []

        if model is None:
            model = TransformersWavLMModel(path_or_uri="microsoft/wavlm-base-plus-sv", revision="main")

        net, extractor, torch_device = cls._get_wavlm_model(model=model, device=device)

        # ``WavLMForXVector`` expects mono 16 kHz; mirror the SpeechBrain backend's
        # strict validation rather than silently resampling.
        expected_sample_rate = 16000
        for audio in audios:
            if audio.waveform.shape[0] != 1:
                raise ValueError(f"Audio waveform must be mono (1 channel), but got {audio.waveform.shape[0]} channels")
            if audio.sampling_rate != expected_sample_rate:
                raise ValueError(
                    f"Audio sampling rate {audio.sampling_rate} does not match expected {expected_sample_rate}"
                )

        # Pad to common length so we can batch through the feature extractor.
        lengths = torch.tensor([a.waveform.shape[1] for a in audios])
        max_len = int(torch.max(lengths).item())
        padded = [F.pad(a.waveform, (0, max_len - a.waveform.shape[1])) for a in audios]
        # Each entry is (1, samples); squeeze channel for the extractor's expected (B, samples).
        wave_batch = [w.squeeze(0).cpu().numpy() for w in padded]
        # The HF feature extractor returns ``input_values`` and ``attention_mask``.
        inputs = extractor(
            wave_batch,
            sampling_rate=expected_sample_rate,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(torch_device) for k, v in inputs.items()}

        with torch.inference_mode():
            outputs = net(**inputs)

        # ``outputs.embeddings`` is shape (B, dim); split into a list of 1-D tensors.
        embeddings = outputs.embeddings  # (B, D)
        return [embeddings[i].detach().cpu().squeeze() for i in range(embeddings.shape[0])]
