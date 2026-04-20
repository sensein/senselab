"""Phonetic Posteriorgrams (PPGs) via isolated subprocess venv.

ppgs depends on espnet, snorkel, and lightning which conflict with modern
torch/Python. It runs in an isolated subprocess venv managed by uv.
"""

import json
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

from senselab.audio.data_structures import Audio
from senselab.utils.data_structures import DeviceType, _select_device_and_dtype, logger
from senselab.utils.subprocess_venv import ensure_venv

# PPGs venv specification
_PPGS_VENV = "ppgs"
_PPGS_REQUIREMENTS = [
    "ppgs>=0.0.9,<0.0.10",
    "espnet",
    "snorkel>=0.10.0,<0.11.0",
    "lightning~=2.4",
    "torch>=2.8,<2.9",
    "torchaudio>=2.8,<2.9",
    "numpy",
    "soundfile",
]
_PPGS_PYTHON = "3.11"

# Worker script — runs inside the isolated venv (no senselab imports)
_WORKER_SCRIPT = r"""
import json
import sys
import traceback
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

args = json.loads(sys.stdin.read())
audio_paths = args["audio_paths"]
device = args["device"]
output_dir = args["output_dir"]

import ppgs

gpu = 0 if device == "cuda" else None

output_paths = []
for i, audio_path in enumerate(audio_paths):
    data, sr = sf.read(audio_path, dtype="float32")
    waveform = torch.from_numpy(data).unsqueeze(0) if data.ndim == 1 else torch.from_numpy(data.T)
    try:
        posteriorgram = ppgs.from_audio(
            torch.unsqueeze(waveform, dim=0),
            ppgs.SAMPLE_RATE,
            gpu=gpu,
        ).cpu()
    except RuntimeError as e:
        print(f"RuntimeError extracting PPGs for audio {i}: {e}", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        posteriorgram = torch.tensor(float("nan"))

    out_path = str(Path(output_dir) / f"ppg_{i}.npy")
    np.save(out_path, posteriorgram.float().numpy())
    output_paths.append(out_path)

print(json.dumps({"output_paths": output_paths}))
"""


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

    venv_dir = ensure_venv(_PPGS_VENV, _PPGS_REQUIREMENTS, python_version=_PPGS_PYTHON)
    python = str(venv_dir / "bin" / "python")

    with tempfile.TemporaryDirectory(prefix="senselab-ppgs-") as tmpdir:
        tmp = Path(tmpdir)

        # Serialize audios to FLAC
        audio_paths = []
        for i, audio in enumerate(audios):
            path = str(tmp / f"audio_{i}.flac")
            audio.save_to_file(path, format="flac")
            audio_paths.append(path)

        # Run worker in isolated venv
        input_json = json.dumps(
            {
                "audio_paths": audio_paths,
                "device": device.value,
                "output_dir": str(tmp),
            }
        )

        result = subprocess.run(
            [python, "-c", _WORKER_SCRIPT],
            input=input_json,
            capture_output=True,
            text=True,
            timeout=600,
        )

        if result.returncode != 0:
            raise RuntimeError(f"PPGs venv failed:\n{result.stderr}")

        # Parse last line only — libraries may print to stdout during init
        output = json.loads(result.stdout.strip().splitlines()[-1])

        # Load results
        posteriorgrams = []
        for out_path in output.get("output_paths", []):
            tensor = torch.from_numpy(np.load(out_path))
            posteriorgrams.append(tensor)

        return posteriorgrams
