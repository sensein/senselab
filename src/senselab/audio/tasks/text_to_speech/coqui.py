"""Coqui TTS wrapper for senselab."""

import os
import tempfile
from typing import Any, Dict, List, Optional, Union

import torch

from senselab.audio.data_structures import Audio
from senselab.utils.data_structures import CoquiTTSModel, DeviceType, Language, _select_device_and_dtype

# Try to import Coqui TTS
try:
    from TTS.api import TTS

    TTS_AVAILABLE = True
except ModuleNotFoundError:
    TTS_AVAILABLE = False


class CoquiTTS:
    """Factory for initializing and using Coqui TTS models."""

    _models: Dict[str, "TTS"] = {}

    @classmethod
    def _get_tts_model(cls, model: CoquiTTSModel, device: Optional[DeviceType] = None) -> "TTS":
        """Get or load a Coqui TTS model, caching by model ID and device.

        Args:
            model: CoquiTTSModel specifying Coqui model ID.
            device: DeviceType to run on (CPU or CUDA).

        Raises:
            ModuleNotFoundError: if `coqui-tts` is not installed.
        """
        if not TTS_AVAILABLE:
            raise ModuleNotFoundError(
                "`coqui-tts` is not available. "
                "Please install senselab audio dependencies using `pip install senselab['audio']`."
            )

        # select device
        device, _ = _select_device_and_dtype(
            user_preference=device, compatible_devices=[DeviceType.CUDA, DeviceType.CPU]
        )
        key = f"{model.path_or_uri}-{device.value}"

        if key not in cls._models:
            # agree to TOS if not already set
            if os.environ.get("COQUI_TOS_AGREED") != "1":
                os.environ["COQUI_TOS_AGREED"] = "1"

            tts = TTS(model.path_or_uri).to(device.value)
            cls._models[key] = tts

        return cls._models[key]

    @classmethod
    def synthesize_texts_with_coqui(
        cls,
        texts: List[str],
        targets: Optional[List[Audio]] = None,
        language: Optional[Language] = None,
        model: Optional[CoquiTTSModel] = None,
        device: Optional[DeviceType] = None,
        **tts_kwargs: Dict[str, Any],
    ) -> List[Audio]:
        """Synthesize text to speech using Coqui TTS.

        Args:
            texts: List of input text strings.
            targets: If provided, a list of Audio objects for voice cloning
                     (must supply equal-length list).
            language: Language of input text.
            model: CoquiTTSModel specifying Coqui model ID.
                If None, the default model "tts_models/multilingual/multi-dataset/xtts_v2" is used.
            device: DeviceType to run on (CPU or CUDA).
            tts_kwargs: Additional kwargs passed to Coqui's `tts` call.

        Returns:
            List[Audio]: Synthesized audio objects.
        """
        # default model if none provided
        if model is None:
            model = CoquiTTSModel(path_or_uri="tts_models/multilingual/multi-dataset/xtts_v2")

        # ensure it's a TTS model
        if model._scope != "tts_models":
            all_models = TTS.list_models()
            tts_models = [m for m in all_models if m.startswith("tts_models")]
            raise ValueError(f"Model {model.path_or_uri} is not a TTS model. Available TTS models: {tts_models}")

        # load & cache
        tts = cls._get_tts_model(model, device)
        output_sr = tts.synthesizer.output_sample_rate

        if targets is not None and len(targets) != len(texts):
            raise ValueError(f"Length of targets ({len(targets)}) must match texts ({len(texts)})")

        outputs: List[Audio] = []
        for idx, text in enumerate(texts):
            call_args: Dict[str, Any] = {"text": text}
            if language is not None:
                call_args["language"] = language.alpha_2

            if targets:
                audio = targets[idx]
                tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                tmp_path = tmp.name
                tmp.close()
                audio.save_to_file(tmp_path)
                call_args["speaker_wav"] = tmp_path
            elif getattr(tts, "speakers", None):
                call_args["speaker"] = tts.speakers[0]

            # merge any additional kwargs
            call_args.update(tts_kwargs)

            # run TTS
            wav = tts.tts(**call_args)
            outputs.append(Audio(waveform=wav, sampling_rate=output_sr))
            if targets:
                os.remove(tmp_path)

        return outputs
