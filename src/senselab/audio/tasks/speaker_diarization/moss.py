"""MOSS-Transcribe-Diarize diarization via isolated subprocess venv.

MOSS-Transcribe-Diarize (OpenMOSS-Team, Apache 2.0) is a 0.9B-parameter unified
ASR+diarization model — much lighter than VibeVoice-ASR-HF (7B), and genuinely
locally-runnable (no cloud dependency, unlike Deepgram). It needs
``transformers>=5.6.0,<6.0.0`` (newer than the ``>=5.3`` this repo's core
environment pins for VibeVoice) plus its own helper package
(``moss-transcribe-diarize``, pip-installable straight from its GitHub repo — it
ships a real ``pyproject.toml``) with extra runtime deps (``librosa``, ``numba``,
``av``, and even a bundled FastAPI web app) that don't belong in senselab's core
environment. So this runs in its own isolated venv, same template as
``speech_to_text/canary_qwen.py`` (a pip-installable git package, not a manual
clone-and-sys.path-insert like the USC-SAIL child-adult backend needed) —
isolation here is purely about keeping unrelated heavy dependencies (and a
different transformers pin) out of the main environment, not about a hard
CUDA-only requirement like child_adult.py.

Loading requires ``trust_remote_code=True`` (the model ships custom modeling
code in its HF repo) — this executes inside the isolated venv, same trust
consideration as any other ``trust_remote_code`` model.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional

from senselab.audio.data_structures import Audio
from senselab.utils.data_structures import DeviceType, HFModel, ScriptLine, _select_device_and_dtype
from senselab.utils.subprocess_venv import _clean_subprocess_env, ensure_venv, parse_subprocess_result, venv_python

_MOSS_VENV = "moss-transcribe-diarize"
_MOSS_REQUIREMENTS = [
    "moss-transcribe-diarize @ git+https://github.com/OpenMOSS/MOSS-Transcribe-Diarize.git",
    "torch>=2.8,<2.9",
    "torchaudio>=2.8,<2.9",
]
_MOSS_PYTHON = "3.12"

# Worker script — runs inside the isolated venv.
_WORKER_SCRIPT = r"""
import json
import sys

try:
    args = json.loads(sys.stdin.read())
    audio_paths = args["audio_paths"]
    model_name = args["model_name"]
    device_arg = args["device"]
    max_new_tokens = args["max_new_tokens"]

    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor
    from moss_transcribe_diarize import parse_transcript
    from moss_transcribe_diarize.inference_utils import (
        build_transcription_messages,
        generate_transcription,
        resolve_device,
    )

    device = resolve_device(device_arg)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True, dtype="auto"
    ).to(dtype=dtype).to(device).eval()
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    all_results = []
    for audio_path in audio_paths:
        messages = build_transcription_messages(audio_path)
        result = generate_transcription(
            model, processor, messages,
            max_new_tokens=max_new_tokens, do_sample=False,
            device=device, dtype=dtype,
        )
        segments = parse_transcript(result["text"])
        all_results.append([
            {"speaker": seg.speaker, "start": seg.start, "end": seg.end, "text": seg.text}
            for seg in segments
        ])

    print(json.dumps({"results": all_results}))
except Exception as exc:
    print(json.dumps({"error": {"type": type(exc).__name__, "message": str(exc)}}))
    sys.exit(1)
"""


def diarize_audios_with_moss(
    audios: List[Audio],
    model: Optional[HFModel] = None,
    device: Optional[DeviceType] = None,
    max_new_tokens: int = 4096,
) -> List[List[ScriptLine]]:
    """Diarize audios with **MOSS-Transcribe-Diarize**; returns per-speaker segments per audio.

    MOSS-Transcribe-Diarize is a 0.9B-parameter unified ASR+diarization model
    (``OpenMOSS-Team/MOSS-Transcribe-Diarize``, Apache 2.0). Each audio is
    transcribed in a single pass; the model emits a compact
    ``[start][Sxx]text[end]`` transcript that ``parse_transcript()`` (from the
    ``moss-transcribe-diarize`` package, installed in this backend's isolated
    venv) turns into structured segments.

    Args:
        audios (list[Audio]):
            Audio clips to diarize.
        model (HFModel | None):
            Defaults to ``HFModel(path_or_uri="OpenMOSS-Team/MOSS-Transcribe-Diarize")``.
        device (DeviceType | None):
            Preferred device (e.g., ``DeviceType.CPU``, ``DeviceType.CUDA``). At
            0.9B params this is far lighter than VibeVoice-ASR-HF (7B) and more
            plausible to run on CPU, though a GPU is still faster.
        max_new_tokens (int):
            Generation budget. Defaults to 4096; raise this for longer recordings.

    Returns:
        list[list[ScriptLine]]: One list per input audio; each `ScriptLine` carries
        `speaker` (e.g. `"S01"`), `start`, `end`, and `text`.

    Example:
        >>> from pathlib import Path
        >>> from senselab.audio.data_structures import Audio
        >>> from senselab.utils.data_structures import DeviceType
        >>> a1 = Audio(filepath=Path("sample1.wav").resolve())
        >>> lines = diarize_audios_with_moss([a1], device=DeviceType.CPU)  # doctest: +SKIP
        >>> len(lines) == 1  # doctest: +SKIP
        True
    """
    if model is None:
        model = HFModel(path_or_uri="OpenMOSS-Team/MOSS-Transcribe-Diarize")

    resolved_device, _ = _select_device_and_dtype(
        user_preference=device, compatible_devices=[DeviceType.CUDA, DeviceType.CPU]
    )

    venv_dir = ensure_venv(_MOSS_VENV, _MOSS_REQUIREMENTS, python_version=_MOSS_PYTHON)
    python = venv_python(venv_dir)

    with tempfile.TemporaryDirectory(prefix="senselab-moss-") as tmpdir:
        tmp = Path(tmpdir)
        audio_paths = []
        for i, audio in enumerate(audios):
            path = str(tmp / f"audio_{i}.wav")
            audio.save_to_file(path)
            audio_paths.append(path)

        input_json = json.dumps(
            {
                "audio_paths": audio_paths,
                "model_name": str(model.path_or_uri),
                "device": resolved_device.value,
                "max_new_tokens": max_new_tokens,
            }
        )

        env = _clean_subprocess_env()
        result = subprocess.run(
            [python, "-c", _WORKER_SCRIPT],
            input=input_json,
            capture_output=True,
            text=True,
            timeout=1200,
            env=env,
        )

        output = parse_subprocess_result(result, "MOSS-Transcribe-Diarize")

        results: List[List[ScriptLine]] = []
        for segments in output.get("results", []):
            script_lines = [
                ScriptLine(
                    speaker=str(seg.get("speaker", "")),
                    start=float(seg.get("start", 0.0)),
                    end=float(seg.get("end", 0.0)),
                    text=seg.get("text"),
                )
                for seg in segments
            ]
            results.append(sorted(script_lines, key=lambda x: x.start or 0.0))

        return results
