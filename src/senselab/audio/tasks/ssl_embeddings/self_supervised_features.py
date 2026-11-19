"""This module contains functions for extracting features from pre-trained self-supervised models."""

import math
import os
import re
from typing import Dict, List, Optional, Union

import torch
import torch.nn.functional as F
from transformers import AutoFeatureExtractor, AutoModel

from senselab.audio.data_structures import Audio
from senselab.utils.data_structures import (
    DeviceType,
    HFModel,
    SpeechBrainModel,
    _select_device_and_dtype,
    device_run_opt,
)
from senselab.utils.dependencies import (
    resolve_model,
    retry_on_transient_error,
    speechbrain_loading_cwd,
    speechbrain_savedir,
)

try:
    from speechbrain.inference.speaker import EncoderClassifier

    SPEECHBRAIN_AVAILABLE = True
except ModuleNotFoundError:
    SPEECHBRAIN_AVAILABLE = False

_HEX40 = re.compile(r"[0-9a-f]{40}")


def _windowed_encoder_states(
    feat_extractor: AutoFeatureExtractor,
    encoder: torch.nn.Module,
    waveform: torch.Tensor,
    sampling_rate: int,
    device: DeviceType,
) -> torch.Tensor:
    """Encoder hidden states over consecutive windows of the feature extractor's `n_samples` (30 s for Whisper).

    Args:
        feat_extractor: A feature extractor that returns `input_features` for a fixed-length window.
        encoder: The encoder of an encoder-decoder model.
        waveform: Mono samples, shape `(n_samples,)`.
        sampling_rate: Sampling rate of `waveform`.
        device: Device the encoder runs on; features are cast to it and to the encoder's dtype.

    Returns:
        Tensor `(layers + 1, frames, hidden)`. Window k fills frames `[k * F, (k + 1) * F)`, F being the frames of a
        full window; the last window is cut to the samples it holds.

    Raises:
        ValueError: If the waveform has no samples.
    """
    if waveform.shape[-1] == 0:
        raise ValueError("audio has no samples")
    window = int(feat_extractor.n_samples)  # type: ignore[attr-defined]
    dtype = next(encoder.parameters()).dtype
    pieces = []
    with torch.no_grad():
        for start in range(0, waveform.shape[-1], window):
            chunk = waveform[start : start + window]
            features = feat_extractor(  # type: ignore[operator]
                chunk.cpu().numpy(), sampling_rate=sampling_rate, return_tensors="pt"
            ).input_features
            out = encoder(features.to(device.value, dtype), output_hidden_states=True)
            states = torch.cat(out.hidden_states)
            pieces.append(states[:, : math.ceil(chunk.shape[-1] * states.shape[1] / window)])
    return torch.cat(pieces, dim=1)


class SSLEmbeddingsFactory:
    """A factory for managing self-supervised models."""

    _feat_extractor: Dict[str, AutoFeatureExtractor] = {}
    _model: Dict[str, AutoModel] = {}

    @classmethod
    def _get_feature_extractor(
        cls,
        model: HFModel,
        cache_dir: Optional[Union[str, os.PathLike]] = None,
    ) -> AutoFeatureExtractor:
        """Get or create a feature extractor for SSL model.

        Args:
            model (HFModel): The HuggingFace model.
            cache_dir (Optional[str, os.PathLike]): The path to where the model's weights will be saved.

        Returns:
            AutoFeatureExtractor: The feature extractor for the model.
        """
        key = f"{model.path_or_uri}-{model.revision}"
        if key not in cls._feat_extractor:
            # SHA-pin only for the default HF cache: resolve_model stages into the
            # default cache, so a custom cache_dir would miss it and re-download
            # (double download). For the common cache_dir=None path this gives the
            # full 429 fix (download-once + no Hub HEAD); a custom cache_dir keeps the
            # prior ref-based behavior rather than staging into the wrong place.
            revision = model.revision
            if cache_dir is None:
                revision, _ = resolve_model(str(model.path_or_uri), model.revision or "main")
            cls._feat_extractor[key] = AutoFeatureExtractor.from_pretrained(
                model.path_or_uri, revision=revision, cache_dir=cache_dir
            )
        return cls._feat_extractor[key]

    @classmethod
    def _load_model(
        cls,
        model: HFModel,
        device: DeviceType,
        cache_dir: Optional[Union[str, os.PathLike]] = None,
    ) -> AutoModel:
        """Load weights of SSL model.

        Args:
            model (HFModel): The Hugging Face model.
            cache_dir (Optional[os.PathLike, str]): The path to where the model's weights will be saved.
            device (DeviceType): The device to run the model on.

        Returns:
            AutoModel: The SSL model.
        """
        key = f"{model.path_or_uri}-{model.revision}-{device.value}"
        if key not in cls._model:
            # SHA-pin only for the default HF cache (see _get_feature_extractor); a
            # custom cache_dir keeps prior behavior to avoid a double download.
            revision = model.revision
            if cache_dir is None:
                revision, _ = resolve_model(str(model.path_or_uri), model.revision or "main")
            cls._model[key] = AutoModel.from_pretrained(model.path_or_uri, revision=revision, cache_dir=cache_dir).to(
                device.value
            )
        return cls._model[key]

    @classmethod
    def extract_ssl_embeddings(
        cls,
        audios: List[Audio],
        model: HFModel,
        cache_dir: Optional[Union[str, os.PathLike]] = None,
        device: Optional[DeviceType] = None,
    ) -> List[torch.Tensor]:
        """Compute the ssl embeddings of audio signals.

        A model whose feature extractor returns `input_features` (Whisper-style encoder-decoder) runs its encoder only,
        window by window (see `_windowed_encoder_states`), and must be given a 40-hex commit as `revision`.

        Args:
            audios (List[Audio]): A list of Audio objects containing the audio signals and their properties.
            model (HFModel): The model used to compute the embeddings
                (default is "facebook/wav2vec2-base").
            cache_dir (Optional[str, os.PathLike]): The path to where the model's weights will be saved.
            device (Optional[DeviceType]): The device to run the model on (default is None).
                Only CPU and CUDA are supported.

        Returns:
            List[torch.Tensor]: A list of tensors containing the ssl embeddings for each audio file.
        """
        device, _ = _select_device_and_dtype(
            user_preference=device, compatible_devices=[DeviceType.CUDA, DeviceType.CPU]
        )
        # Load feature extractor and model
        feat_extractor = cls._get_feature_extractor(model=model, cache_dir=cache_dir)
        sr = getattr(feat_extractor, "sampling_rate", None)
        if not isinstance(sr, int):
            raise AttributeError(
                "The feature extractor doesn't expose an integer `sampling_rate`. "
                "Please use a model with a known sampling rate."
            )
        sampling_rate: int = sr
        encoder_only = "input_features" in getattr(feat_extractor, "model_input_names", [])
        if encoder_only and not _HEX40.fullmatch(str(model.revision)):
            raise ValueError(f"an encoder-decoder model needs a resolved 40-hex commit, got {model.revision!r}")
        ssl_model = cls._load_model(model=model, cache_dir=cache_dir, device=device)

        # Check that all audio objects have the correct sampling rate
        for audio in audios:
            if audio.waveform.shape[0] != 1:
                raise ValueError(f"Audio waveform must be mono (1 channel), but got {audio.waveform.shape[0]} channels")
            if audio.sampling_rate != sampling_rate:
                raise ValueError(
                    "Audio sampling rate " + str(audio.sampling_rate) + " does not match expected " + str(sampling_rate)
                )
        if encoder_only:
            encoder = ssl_model.get_encoder()  # type: ignore[attr-defined]
            return [
                _windowed_encoder_states(feat_extractor, encoder, audio.waveform[0], sampling_rate, device)
                for audio in audios
            ]
        # Pre-process audio using the SSL mode feature extractor
        preprocessed_audios = [
            feat_extractor(  # type: ignore[operator]
                audio.waveform, sampling_rate=sampling_rate, return_tensors="pt"
            )
            for audio in audios
        ]

        # Extract embeddings (hidden states from all layers) from pre-trained model
        embeddings = [
            ssl_model(  # type: ignore[operator]
                audio.input_values.squeeze(0).to(device.value, ssl_model.dtype),  # type: ignore[attr-defined]
                output_hidden_states=True,
            )
            for audio in preprocessed_audios
        ]
        embeddings = [torch.cat(embedding.hidden_states) for embedding in embeddings]

        return embeddings


class SpeechBrainSSLEmbeddings:
    """Extracts SSL embeddings using SpeechBrain encoder models.

    Reuses the SpeechBrain ``EncoderClassifier`` infrastructure from
    the speaker_embeddings module to produce fixed-dimensional embeddings
    (e.g., ECAPA-TDNN 192-D).
    """

    _models: Dict[str, "EncoderClassifier"] = {}

    @classmethod
    def _get_model(
        cls,
        model: SpeechBrainModel,
        device: Optional[DeviceType] = None,
    ) -> "EncoderClassifier":
        """Get or create a SpeechBrain encoder model.

        Args:
            model: The SpeechBrain model spec.
            device: Device for inference (CPU or CUDA).

        Returns:
            The loaded EncoderClassifier.
        """
        if not SPEECHBRAIN_AVAILABLE:
            raise ModuleNotFoundError(
                "`speechbrain` is not installed. "
                "Please install senselab audio dependencies using `pip install senselab`."
            )

        device, _ = _select_device_and_dtype(
            user_preference=device, compatible_devices=[DeviceType.CUDA, DeviceType.CPU]
        )
        key = f"{model.path_or_uri}-{model.revision}-{device.value}"
        if key not in cls._models:
            # Stage once (download-once via the heartbeat lock) + load from the local
            # snapshot dir so SpeechBrain makes no per-file Hub HEAD (429 source under
            # batch). SpeechBrain has no revision arg -> pin via the snapshot path.
            _, snapshot_path = resolve_model(str(model.path_or_uri), model.revision or "main")
            savedir = speechbrain_savedir(str(model.path_or_uri), model.revision)
            with speechbrain_loading_cwd(savedir):
                cls._models[key] = retry_on_transient_error(
                    EncoderClassifier.from_hparams,
                    source=str(snapshot_path),
                    savedir=str(savedir),
                    run_opts={"device": device_run_opt(device)},
                )
        return cls._models[key]

    @classmethod
    def extract_speechbrain_ssl_embeddings(
        cls,
        audios: List[Audio],
        model: SpeechBrainModel,
        device: Optional[DeviceType] = None,
    ) -> List[torch.Tensor]:
        """Extract embeddings from audios using a SpeechBrain encoder model.

        This uses SpeechBrain's ``EncoderClassifier.encode_batch()``
        which returns fixed-dimensional embeddings (e.g., 192-D for
        ECAPA-TDNN, 256-D for ResNet, 512-D for x-vector).

        Args:
            audios: List of mono Audio objects (16 kHz recommended).
            model: SpeechBrain model spec (e.g.,
                ``SpeechBrainModel(path_or_uri="speechbrain/spkrec-ecapa-voxceleb")``).
            device: Device for inference (CPU or CUDA).

        Returns:
            List of 1-D tensors, one embedding per audio.
        """
        if not SPEECHBRAIN_AVAILABLE:
            raise ModuleNotFoundError(
                "`speechbrain` is not installed. "
                "Please install senselab audio dependencies using `pip install senselab`."
            )

        if len(audios) == 0:
            return []

        classifier = cls._get_model(model=model, device=device)
        expected_sample_rate = 16000

        for audio in audios:
            if audio.waveform.shape[0] != 1:
                raise ValueError(f"Audio waveform must be mono (1 channel), but got {audio.waveform.shape[0]} channels")
            if audio.sampling_rate != expected_sample_rate:
                raise ValueError(
                    "Audio sampling rate "
                    + str(audio.sampling_rate)
                    + " does not match expected "
                    + str(expected_sample_rate)
                )

        # Calculate relative lengths for batch processing
        lengths = torch.tensor([audio.waveform.shape[1] for audio in audios])
        max_len = torch.max(lengths)
        wav_lens = lengths.float() / max_len

        # Stack with zero-padding
        waveforms = torch.stack(
            [F.pad(audio.waveform, (0, int(max_len - audio.waveform.shape[1]))) for audio in audios]
        ).squeeze(dim=1)  # Remove channel dim only, keep batch

        embeddings_batch = classifier.encode_batch(waveforms, wav_lens)
        embeddings = [embedding.squeeze(dim=0) for embedding in embeddings_batch]

        return embeddings
