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

Not wired into ``audio_analysis``
---------------------------------
This MOSS-Transcribe-Diarize backend is reachable through :func:`diarize_audios` and
deliberately **not** through ``scripts/analyze_audio.py --diarization-models``. Two
hazard classes motivate that split: a **role-label** backend, whose ``speaker``
output names a role (e.g. ``CHILD``/``ADULT``/``OVERLAP``) rather than a speaker
identity, would build a per-role centroid blending distinct speakers under one
label and snap ambiguous frames to whichever centroid is nearest; a
**speaker-identity** backend with its own unreconciled labelling scheme would feed
those labels straight into cross-diarizer agreement and embedding clustering
before they are harmonized against the pass-wide cluster IDs those steps key on,
reading as spurious disagreement against every real diarization model. This
backend falls in the second class — it assigns its own per-audio speaker
identities (``Sxx`` tags parsed out of its transcript) with no reconciliation
against the pass-wide cluster IDs, so wiring it into ``--diarization-models`` as-is
would feed unreconciled labels straight into cross-diarizer consensus and
embedding clustering. The guards for both hazard classes live in
``workflows/audio_analysis/{clustering,identity,presence}.py``, which this branch
does not carry. Port those guards from PR #537 before wiring any of the four new
backends into the workflow.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional

from senselab.audio.data_structures import Audio
from senselab.utils.data_structures import DeviceType, HFModel, ScriptLine, _select_device_and_dtype
from senselab.utils.data_structures.logging import logger
from senselab.utils.dependencies import hf_subprocess_env
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
    model_revision = args["model_revision"]
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
        model_name, revision=model_revision, trust_remote_code=True, dtype="auto"
    ).to(dtype=dtype).to(device).eval()
    processor = AutoProcessor.from_pretrained(model_name, revision=model_revision, trust_remote_code=True)

    all_results = []
    for audio_path in audio_paths:
        messages = build_transcription_messages(audio_path)
        result = generate_transcription(
            model, processor, messages,
            max_new_tokens=max_new_tokens, do_sample=False,
            device=device, dtype=dtype,
        )
        segments = parse_transcript(result["text"])
        all_results.append({
            "segments": [
                {"speaker": seg.speaker, "start": seg.start, "end": seg.end, "text": seg.text}
                for seg in segments
            ],
            # generate() only stops before max_new_tokens on EOS; hitting the
            # budget means result["text"] is very likely cut off mid-transcript,
            # a different failure mode than "no speech" that parse_transcript()
            # alone can't signal since it just returns whatever prefix parsed.
            "truncated": result["generated_tokens"] >= max_new_tokens,
        })

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
                "model_revision": model.revision,
                "device": resolved_device.value,
                "max_new_tokens": max_new_tokens,
            }
        )

        # Stage the model once (cross-process, via the heartbeat lock) + run the
        # worker offline so its from_pretrained calls make no per-call Hub version
        # check — the 429 source under parallel batch load.
        env = hf_subprocess_env(str(model.path_or_uri), model.revision, base_env=_clean_subprocess_env())
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
        for i, audio_result in enumerate(output.get("results", [])):
            if audio_result.get("truncated"):
                logger.warning(
                    f"MOSS-Transcribe-Diarize hit max_new_tokens={max_new_tokens} on "
                    f"{audio_paths[i]!r} without generating an end token; the transcript "
                    "is likely truncated. Pass a higher max_new_tokens for longer recordings."
                )
            script_lines = [
                ScriptLine(
                    speaker=str(seg.get("speaker", "")),
                    start=float(seg.get("start", 0.0)),
                    end=float(seg.get("end", 0.0)),
                    text=seg.get("text"),
                )
                for seg in audio_result.get("segments", [])
            ]
            results.append(sorted(script_lines, key=lambda x: x.start or 0.0))

        return results
