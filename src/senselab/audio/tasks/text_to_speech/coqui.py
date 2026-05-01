"""Coqui TTS synthesis via isolated subprocess venv.

Coqui TTS has conflicting dependencies (pins older transformers, requires
Python <=3.11). It runs in an isolated subprocess venv managed by uv.
"""

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from senselab.audio.data_structures import Audio
from senselab.utils.data_structures import CoquiTTSModel, DeviceType, Language, _select_device_and_dtype
from senselab.utils.subprocess_venv import _clean_subprocess_env, ensure_venv, parse_subprocess_result, venv_python

# Reuse the same coqui venv as voice_cloning
_COQUI_VENV = "coqui"
_COQUI_REQUIREMENTS = [
    "coqui-tts~=0.27",
    "torch>=2.8,<2.9",
    "torchaudio>=2.8,<2.9",
    "transformers>=4.52,<5",
    "numpy",
    "soundfile",
]
_COQUI_PYTHON = "3.11"

_WORKER_SCRIPT = r"""
import json
import os
import sys
from pathlib import Path

try:
    os.environ["COQUI_TOS_AGREED"] = "1"
    import soundfile as sf
    import torch
    from TTS.api import TTS

    args = json.loads(sys.stdin.read())
    texts = args["texts"]
    model_id = args["model_id"]
    device = args["device"]
    language = args.get("language")
    target_paths = args.get("target_paths", [])
    output_dir = args["output_dir"]

    tts = TTS(model_id).to(device=device)
    output_sr = tts.synthesizer.output_sample_rate

    output_paths = []
    for idx, text in enumerate(texts):
        call_args = {"text": text}
        if language:
            call_args["language"] = language

        if idx < len(target_paths) and target_paths[idx]:
            call_args["speaker_wav"] = target_paths[idx]
        elif getattr(tts, "speakers", None):
            call_args["speaker"] = tts.speakers[0]

        wav = tts.tts(**call_args)
        if not isinstance(wav, torch.Tensor):
            wav = torch.tensor(wav)
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)

        out_path = str(Path(output_dir) / f"tts_{idx}.flac")
        sf.write(out_path, wav.squeeze().cpu().numpy(), output_sr, format="FLAC")
        output_paths.append(out_path)

    print(json.dumps({"output_paths": output_paths, "sample_rate": output_sr}))
except Exception as exc:
    print(json.dumps({"error": {"type": type(exc).__name__, "message": str(exc)}}))
    sys.exit(1)
"""


class CoquiTTS:
    """Factory for TTS synthesis via Coqui in an isolated subprocess venv."""

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

        The actual Coqui TTS operations run in an isolated subprocess venv
        with its own Python and torch version.

        Args:
            texts: List of input text strings.
            targets: If provided, a list of Audio objects for voice cloning.
            language: Language of input text.
            model: CoquiTTSModel specifying model ID.
            device: DeviceType to run on (CPU or CUDA).
            tts_kwargs: Additional kwargs (currently unused in subprocess mode).

        Returns:
            List[Audio]: Synthesized audio objects.
        """
        if model is None:
            model = CoquiTTSModel(path_or_uri="tts_models/multilingual/multi-dataset/xtts_v2")

        device_type, _ = _select_device_and_dtype(
            user_preference=device, compatible_devices=[DeviceType.CUDA, DeviceType.CPU]
        )

        if targets is not None and len(targets) != len(texts):
            raise ValueError(f"Length of targets ({len(targets)}) must match texts ({len(texts)})")

        venv_dir = ensure_venv(_COQUI_VENV, _COQUI_REQUIREMENTS, python_version=_COQUI_PYTHON)
        python = venv_python(venv_dir)

        with tempfile.TemporaryDirectory(prefix="senselab-coqui-tts-") as tmpdir:
            tmp = Path(tmpdir)

            # Save target audio files if provided
            target_paths: List[str] = []
            if targets:
                for i, audio in enumerate(targets):
                    path = str(tmp / f"target_{i}.flac")
                    audio.save_to_file(path, format="flac")
                    target_paths.append(path)

            input_json = json.dumps(
                {
                    "texts": texts,
                    "model_id": str(model.path_or_uri),
                    "device": device_type.value,
                    "language": language.alpha_2 if language else None,
                    "target_paths": target_paths,
                    "output_dir": str(tmp),
                }
            )

            env = _clean_subprocess_env()
            result = subprocess.run(
                [python, "-c", _WORKER_SCRIPT],
                input=input_json,
                capture_output=True,
                text=True,
                timeout=600,
                env=env,
            )

            output = parse_subprocess_result(result, "Coqui TTS")

            # Load results eagerly (tempdir cleaned up after this block)
            synthesized = []
            for out_path in output.get("output_paths", []):
                audio = Audio(filepath=out_path)
                _ = audio.waveform  # force load before tempdir cleanup
                synthesized.append(audio)

            return synthesized
