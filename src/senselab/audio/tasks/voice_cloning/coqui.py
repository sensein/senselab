"""Voice cloning using Coqui TTS via isolated subprocess venv.

Coqui TTS has conflicting dependencies (pins old torch versions, requires
Python <=3.11). It runs in an isolated subprocess venv managed by uv.
Audio data is serialized as FLAC for efficient lossless transfer.
"""

import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch

from senselab.audio.data_structures import Audio
from senselab.utils.data_structures import CoquiTTSModel, DeviceType
from senselab.utils.subprocess_venv import call_in_venv

# Coqui venv specification
_COQUI_VENV = "coqui"
_COQUI_REQUIREMENTS = ["coqui-tts~=0.27", "torch~=2.8", "torchaudio~=2.8"]
_COQUI_PYTHON = "3.11"


class CoquiVoiceCloner:
    """Voice cloning via Coqui TTS in an isolated subprocess venv."""

    @classmethod
    def clone_voices(
        cls,
        source_audios: List[Audio],
        target_audios: List[Audio],
        model: Optional[CoquiTTSModel] = None,
        device: Optional[DeviceType] = None,
    ) -> List[Audio]:
        """Clone voices from source audios to target audios using Coqui TTS.

        The actual Coqui TTS operations run in an isolated subprocess venv
        with its own Python and torch version. Audio is transferred via
        lossless FLAC files.

        Args:
            source_audios: List of source audio objects.
            target_audios: List of target audio objects.
            model: Coqui TTS model. Default: knnvc voice conversion.
            device: Device preference (CUDA or CPU).
        """
        if model is None:
            model = CoquiTTSModel(path_or_uri="voice_conversion_models/multilingual/multi-dataset/knnvc")

        if len(source_audios) != len(target_audios):
            raise ValueError("Number of source and target audios must be the same.")

        for i, (src, tgt) in enumerate(zip(source_audios, target_audios)):
            if src.waveform.squeeze().dim() != 1 or tgt.waveform.squeeze().dim() != 1:
                raise ValueError(f"[Pair {i}] Only mono audio is supported.")

        # Serialize audio to temp FLAC files
        with tempfile.TemporaryDirectory(prefix="senselab-coqui-") as tmpdir:
            tmp = Path(tmpdir)

            # Write source/target audio files
            source_paths = []
            target_paths = []
            for i, (src, tgt) in enumerate(zip(source_audios, target_audios)):
                src_path = str(tmp / f"source_{i}.flac")
                tgt_path = str(tmp / f"target_{i}.flac")
                _save_audio_flac(src, src_path)
                _save_audio_flac(tgt, tgt_path)
                source_paths.append(src_path)
                target_paths.append(tgt_path)

            # Call coqui in isolated venv
            result = call_in_venv(
                name=_COQUI_VENV,
                requirements=_COQUI_REQUIREMENTS,
                module="senselab.audio.tasks.voice_cloning._coqui_worker",
                function="run_voice_cloning",
                args={
                    "source_paths": source_paths,
                    "target_paths": target_paths,
                    "model_id": str(model.path_or_uri),
                    "device": device.value if device else "cpu",
                    "output_dir": str(tmp),
                },
                python_version=_COQUI_PYTHON,
                timeout=600,
            )

            # Load results
            cloned_audios = []
            output_paths = result.get("output_paths", []) if isinstance(result, dict) else []
            for out_path in output_paths:
                audio = _load_audio_flac(out_path)
                cloned_audios.append(audio)

            return cloned_audios


def _save_audio_flac(audio: Audio, path: str) -> None:
    """Save Audio to FLAC (lossless, compressed)."""
    try:
        from torchcodec.encoders import AudioEncoder

        encoder = AudioEncoder(samples=audio.waveform, sample_rate=audio.sampling_rate)
        encoder.to_file(path)
    except (ImportError, RuntimeError):
        import torchaudio

        torchaudio.save(path, audio.waveform, audio.sampling_rate, format="flac")


def _load_audio_flac(path: str) -> Audio:
    """Load Audio from FLAC."""
    try:
        from torchcodec.decoders import AudioDecoder

        decoder = AudioDecoder(path)
        samples = decoder.get_all_samples()
        return Audio(waveform=samples.data, sampling_rate=samples.sample_rate)
    except (ImportError, RuntimeError):
        import torchaudio

        waveform, sr = torchaudio.load(path)
        return Audio(waveform=waveform, sampling_rate=sr)
