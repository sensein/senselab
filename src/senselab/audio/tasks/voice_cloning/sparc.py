"""Voice cloning using SPARC via isolated subprocess venv.

speech-articulatory-coding (SPARC) has dependencies that may conflict
with the main environment. It runs in an isolated subprocess venv.
"""

import json
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional

from senselab.audio.data_structures import Audio
from senselab.utils.data_structures import DeviceType, Language, _select_device_and_dtype
from senselab.utils.subprocess_venv import ensure_venv, parse_subprocess_result, venv_python

# Reuse the same venv spec as features_extraction/sparc.py
_SPARC_VENV = "sparc"
_SPARC_REQUIREMENTS = [
    "speech-articulatory-coding>=0.1",
    "torch>=2.8,<2.9",
    "torchaudio>=2.8,<2.9",
    "numpy",
    "soundfile",
    "librosa",
    "transformers",
    "torchcrepe==0.0.23",
    "penn==0.0.14",
    "huggingface-hub",
]
_SPARC_PYTHON = "3.11"

# Worker script for voice cloning — runs inside the isolated venv
_WORKER_SCRIPT = r"""
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from sparc import load_model

args = json.loads(sys.stdin.read())
source_paths = args["source_paths"]
target_paths = args["target_paths"]
language = args["language"]
device = args["device"]
output_dir = args["output_dir"]

coder = load_model(language, device=device)

output_paths = []
for i, (src_path, tgt_path) in enumerate(zip(source_paths, target_paths)):
    src_data, src_sr = sf.read(src_path, dtype="float32")
    tgt_data, tgt_sr = sf.read(tgt_path, dtype="float32")

    src_wav = src_data.squeeze().astype(np.float32)
    tgt_wav = tgt_data.squeeze().astype(np.float32)

    converted = coder.convert(src_wav=src_wav, trg_wav=tgt_wav)

    # converted may be a tensor or numpy array
    if hasattr(converted, "cpu"):
        converted = converted.cpu().numpy()
    if hasattr(converted, "squeeze"):
        converted = converted.squeeze()

    sr = coder.sr
    out_path = str(Path(output_dir) / f"cloned_{i}.flac")
    sf.write(out_path, converted, sr, format="FLAC")
    output_paths.append(out_path)

print(json.dumps({"output_paths": output_paths, "expected_sr": coder.sr}))
"""


class SparcVoiceCloner:
    """Voice cloning via SPARC in an isolated subprocess venv."""

    @classmethod
    def clone_voices(
        cls,
        source_audios: List[Audio],
        target_audios: List[Audio],
        lang: Optional[Language] = None,
        device: Optional[DeviceType] = None,
    ) -> List[Audio]:
        """Clone voices from source audios to target audios using SPARC.

        The SPARC model runs in an isolated subprocess venv with its own
        Python and dependencies. Audio is transferred via FLAC files.

        Args:
            source_audios: List of source audio objects.
            target_audios: List of target audio objects.
            lang: Language for the SPARC model. None means multi-language.
            device: Device to use (CUDA or CPU).

        Returns:
            List of cloned audio objects.
        """
        device, _ = _select_device_and_dtype(
            user_preference=device, compatible_devices=[DeviceType.CUDA, DeviceType.CPU]
        )

        if len(source_audios) != len(target_audios):
            raise ValueError("Number of source and target audios must be the same.")

        if lang is None:
            used_language = "multi"
        elif lang.name == "english":
            used_language = "en+"
        else:
            raise ValueError(f"Language {lang.name} not supported. Supported: english or None (multi-language).")

        venv_dir = ensure_venv(_SPARC_VENV, _SPARC_REQUIREMENTS, python_version=_SPARC_PYTHON)
        python = venv_python(venv_dir)

        with tempfile.TemporaryDirectory(prefix="senselab-sparc-vc-") as tmpdir:
            tmp = Path(tmpdir)

            # Validate and serialize
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

            # Run worker in isolated venv
            input_json = json.dumps(
                {
                    "source_paths": source_paths,
                    "target_paths": target_paths,
                    "language": used_language,
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

            output = parse_subprocess_result(result, "SPARC")

            # Load results eagerly (tempdir is cleaned up after this block)
            cloned_audios = []
            for out_path in output.get("output_paths", []):
                audio = Audio(filepath=out_path)
                _ = audio.waveform  # force load before tempdir cleanup
                cloned_audios.append(audio)

            return cloned_audios
