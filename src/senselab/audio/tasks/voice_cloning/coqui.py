"""Voice cloning using Coqui TTS via isolated subprocess venv.

Coqui TTS has conflicting dependencies (pins old torch versions, requires
Python <=3.11). It runs in an isolated subprocess venv managed by uv.
Audio data is serialized as FLAC for efficient lossless transfer.
"""

import tempfile
from pathlib import Path
from typing import List, Optional

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

        with tempfile.TemporaryDirectory(prefix="senselab-coqui-") as tmpdir:
            tmp = Path(tmpdir)

            # Validate and serialize in a single pass
            source_paths = []
            target_paths = []
            for i, (src, tgt) in enumerate(zip(source_audios, target_audios)):
                if src.waveform.squeeze().dim() != 1 or tgt.waveform.squeeze().dim() != 1:
                    raise ValueError(f"[Pair {i}] Only mono audio is supported.")

                src_path = str(tmp / f"source_{i}.flac")
                tgt_path = str(tmp / f"target_{i}.flac")
                src.save_to_file(src_path, format="flac")
                tgt.save_to_file(tgt_path, format="flac")
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
                cloned_audios.append(Audio(filepath=out_path))

            return cloned_audios
