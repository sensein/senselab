"""Voice cloning using Coqui TTS via isolated subprocess venv.

Coqui TTS has conflicting dependencies (pins old torch versions, requires
Python <=3.11). It runs in an isolated subprocess venv managed by uv.
Audio data is serialized as FLAC for efficient lossless transfer.
"""

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional

from senselab.audio.data_structures import Audio
from senselab.utils.data_structures import CoquiTTSModel, DeviceType
from senselab.utils.subprocess_venv import _clean_subprocess_env, ensure_venv, parse_subprocess_result, venv_python

# Coqui venv specification
_COQUI_VENV = "coqui"
_COQUI_REQUIREMENTS = [
    "coqui-tts~=0.27",
    "torch>=2.8,<2.9",
    "torchaudio>=2.8,<2.9",
    "transformers>=4.52,<5",  # coqui-tts 0.27 needs >=4.52; isin_mps_friendly removed in 5.0
    "numpy",
    "soundfile",
]
_COQUI_PYTHON = "3.11"

# Worker script — runs inside the isolated venv (no senselab imports)
_WORKER_SCRIPT = r"""
import json
import sys
from pathlib import Path

try:
    import soundfile as sf
    import torch
    from TTS.api import TTS

    args = json.loads(sys.stdin.read())
    source_paths = args["source_paths"]
    target_paths = args["target_paths"]
    model_id = args["model_id"]
    device = args["device"]
    output_dir = args["output_dir"]

    tts = TTS(model_id).to(device=device)
    audio_config = tts.voice_converter.vc_config.audio

    expected_sample_rate = (
        audio_config.input_sample_rate
        if hasattr(audio_config, "input_sample_rate")
        else getattr(audio_config, "sample_rate", None)
    )
    output_sample_rate = (
        audio_config.output_sample_rate
        if hasattr(audio_config, "output_sample_rate")
        else getattr(audio_config, "sample_rate", expected_sample_rate)
    )

    output_paths = []
    for i, (src_path, tgt_path) in enumerate(zip(source_paths, target_paths)):
        src_data, src_sr = sf.read(src_path, dtype="float32")
        tgt_data, tgt_sr = sf.read(tgt_path, dtype="float32")
        src_wav = torch.from_numpy(src_data).unsqueeze(0) if src_data.ndim == 1 else torch.from_numpy(src_data.T)
        tgt_wav = torch.from_numpy(tgt_data).unsqueeze(0) if tgt_data.ndim == 1 else torch.from_numpy(tgt_data.T)

        if expected_sample_rate is not None:
            if src_sr != expected_sample_rate:
                raise ValueError(
                    f"[Pair {i}] Source sample rate {src_sr} != model expected {expected_sample_rate}"
                )
            if tgt_sr != expected_sample_rate:
                raise ValueError(
                    f"[Pair {i}] Target sample rate {tgt_sr} != model expected {expected_sample_rate}"
                )

        converted = tts.voice_conversion(
            source_wav=src_wav.squeeze(),
            target_wav=tgt_wav.squeeze(),
        )

        if not isinstance(converted, torch.Tensor):
            converted = torch.tensor(converted)
        if converted.dim() == 1:
            converted = converted.unsqueeze(0)

        sr = output_sample_rate or src_sr
        out_path = str(Path(output_dir) / f"cloned_{i}.flac")
        sf.write(out_path, converted.squeeze().cpu().numpy(), sr, format="FLAC")
        output_paths.append(out_path)

    print(json.dumps({"output_paths": output_paths}))
except Exception as exc:
    print(json.dumps({"error": {"type": type(exc).__name__, "message": str(exc)}}))
    sys.exit(1)
"""


def list_coqui_models() -> list:
    """List available Coqui TTS models via the isolated subprocess venv."""
    venv_dir = ensure_venv(_COQUI_VENV, _COQUI_REQUIREMENTS, python_version=_COQUI_PYTHON)
    python = venv_python(venv_dir)
    env = _clean_subprocess_env()
    result = subprocess.run(
        [python, "-c", "from TTS.api import TTS; import json; print(json.dumps(list(TTS().list_models())))"],
        capture_output=True,
        text=True,
        timeout=300,  # cold start can be slow (venv creation + TTS model list download)
        env=env,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to list Coqui models:\n{result.stderr}")
    return json.loads(result.stdout.strip().splitlines()[-1])


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

        venv_dir = ensure_venv(_COQUI_VENV, _COQUI_REQUIREMENTS, python_version=_COQUI_PYTHON)
        python = venv_python(venv_dir)

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

            # Run worker in isolated venv
            input_json = json.dumps(
                {
                    "source_paths": source_paths,
                    "target_paths": target_paths,
                    "model_id": str(model.path_or_uri),
                    "device": device.value if device else "cpu",
                    "output_dir": str(tmp),
                }
            )

            # Clear MPLBACKEND to avoid matplotlib_inline errors in subprocess
            env = {k: v for k, v in os.environ.items() if k != "MPLBACKEND"}
            result = subprocess.run(
                [python, "-c", _WORKER_SCRIPT],
                input=input_json,
                capture_output=True,
                text=True,
                timeout=600,
                env=env,
            )

            output = parse_subprocess_result(result, "Coqui")

            # Load results eagerly (tempdir is cleaned up after this block)
            cloned_audios = []
            for out_path in output.get("output_paths", []):
                audio = Audio(filepath=out_path)
                _ = audio.waveform  # force load before tempdir cleanup
                cloned_audios.append(audio)

            return cloned_audios
