"""Phonetic Posteriorgrams (PPGs) via isolated subprocess venv.

ppgs depends on espnet, snorkel, and lightning which conflict with modern
torch/Python. It runs in an isolated subprocess venv managed by uv.
"""

import tempfile
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

from senselab.audio.data_structures import Audio
from senselab.utils.data_structures import DeviceType, _select_device_and_dtype, logger
from senselab.utils.subprocess_venv import call_in_venv

# PPGs venv specification
_PPGS_VENV = "ppgs"
_PPGS_REQUIREMENTS = [
    "ppgs>=0.0.9,<0.0.10",
    "espnet",
    "snorkel>=0.10.0,<0.11.0",
    "lightning~=2.4",
    "torch~=2.8",
    "torchaudio~=2.8",
]
_PPGS_PYTHON = "3.11"


def extract_ppgs_from_audios(audios: List[Audio], device: Optional[DeviceType] = None) -> List[torch.Tensor]:
    """Extracts phonetic posteriorgrams (PPGs) from every audio.

    The ppgs model runs in an isolated subprocess venv with its own
    Python and dependencies. Audio is transferred via FLAC files.

    Args:
        audios: The audios to extract PPGs from.
        device: Device to use (CUDA or CPU).

    Returns:
        List of PPG tensors, one per input audio.
    """
    device, _ = _select_device_and_dtype(user_preference=device, compatible_devices=[DeviceType.CUDA, DeviceType.CPU])

    if any(audio.waveform.shape[0] != 1 for audio in audios):
        raise ValueError("Only mono audio is supported by ppgs model.")

    with tempfile.TemporaryDirectory(prefix="senselab-ppgs-") as tmpdir:
        tmp = Path(tmpdir)

        # Serialize audios to FLAC
        audio_paths = []
        for i, audio in enumerate(audios):
            path = str(tmp / f"audio_{i}.flac")
            audio.save_to_file(path, format="flac")
            audio_paths.append(path)

        # Call ppgs in isolated venv
        result = call_in_venv(
            name=_PPGS_VENV,
            requirements=_PPGS_REQUIREMENTS,
            module="senselab.audio.tasks.features_extraction._ppg_worker",
            function="run_ppg_extraction",
            args={
                "audio_paths": audio_paths,
                "device": device.value,
                "output_dir": str(tmp),
            },
            python_version=_PPGS_PYTHON,
            timeout=600,
        )

        # Load results
        posteriorgrams = []
        output_paths = result.get("output_paths", []) if isinstance(result, dict) else []
        for out_path in output_paths:
            tensor = torch.from_numpy(np.load(out_path))
            posteriorgrams.append(tensor)

        return posteriorgrams
