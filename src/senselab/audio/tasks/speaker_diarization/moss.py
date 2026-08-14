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

What the ``speaker`` field means here
-------------------------------------
This backend assigns its own per-file speaker labels (``Sxx`` tags parsed out of its transcript). They are labels, not
identities: the same tag in two different files carries no claim of being the same
person, which :func:`~senselab.audio.tasks.speaker_diarization.api.capabilities_for`
reports as ``labels_stable_across_files=False``.

That is true of diarizers generally -- ``SPEAKER_00`` is no more an identity than
``S01`` is. Reconciling labels from different backends into one namespace is a separate
concern with its own utility, and no backend module decides it. This one reports what it
produced and stops there.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.speaker_diarization.capabilities import DiarizationCapabilities
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

CAPABILITIES = DiarizationCapabilities(
    populates_text=True,  # joint ASR+diarization: measured 6/6 segments carried text
    speaker_label_kind="index",  # emits S01/S02 tags parsed from its transcript
    labels_stable_across_files=False,  # not measured; False is the conservative default
    # Seed-17 speaker-ceiling probe: at k=8, predicted counts ranged 6..12 — it overshoots
    # the true count rather than plateauing, so no structural ceiling was observed.
    max_speakers=None,
    max_speakers_evidence="measured: no saturation, emits up to 12 (probe seed-17)",
    honors_speaker_hints=False,
)

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
            # Lets the host tell "genuine silence" (model emitted ~nothing) apart
            # from "parse_transcript() couldn't parse a real transcript" (the
            # realistic shape of an upstream format change) — both would otherwise
            # collapse to the same empty `segments` list.
            "raw_text_len": len((result["text"] or "").strip()),
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
        `speaker` (e.g. `"S01"`), `start`, `end`, and `text`. Empty for an audio that
        was genuinely silent or produced no parseable transcript (logged as a
        warning, not raised) — unless *every* audio in the batch produced a
        non-empty transcript that failed to parse, which raises instead (see
        Raises).

    Raises:
        RuntimeError: If every audio in the batch produced a non-empty transcript
            that ``parse_transcript()`` could not parse into any segments —
            treated as a broken transcript-format contract, not silent no-speech.

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

        # Forward the resolved commit SHA to the worker, never the ref -- it has no senselab
        # install and cannot re-resolve, so a bare ref would load whatever this host's cache
        # resolves it to right now, which can disagree with the rest of a multi-node run.
        # commit_sha is already populated by HFModel's constructor-time resolution; the
        # resolve_revision fallback only matters if that somehow did not happen. Deferred
        # import (not at module top) keeps this monkeypatch-friendly at
        # senselab.utils.model_revision.resolve_revision, matching the rest of the codebase.
        from senselab.utils.model_revision import resolve_revision

        model_name = str(model.path_or_uri)
        revision = model.commit_sha or resolve_revision(model_name, model.revision)

        input_json = json.dumps(
            {
                "audio_paths": audio_paths,
                "model_name": model_name,
                "model_revision": revision,
                "device": resolved_device.value,
                "max_new_tokens": max_new_tokens,
            }
        )

        # Stage the model once (cross-process, via the heartbeat lock) + run the
        # worker offline so its from_pretrained calls make no per-call Hub version
        # check — the 429 source under parallel batch load.
        env = hf_subprocess_env(model_name, revision, base_env=_clean_subprocess_env())
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
        parse_failures = 0
        audio_results = output.get("results", [])
        for i, audio_result in enumerate(audio_results):
            if audio_result.get("truncated"):
                logger.warning(
                    f"MOSS-Transcribe-Diarize hit max_new_tokens={max_new_tokens} on "
                    f"{audio_paths[i]!r} without generating an end token; the transcript "
                    "is likely truncated. Pass a higher max_new_tokens for longer recordings."
                )
            segments = audio_result.get("segments", [])
            if not segments and audio_result.get("raw_text_len", 0) > 0:
                # The model emitted a non-empty transcript, but parse_transcript()
                # produced no segments from it — the realistic shape of an upstream
                # transcript-format change, not genuine silence. Without raw_text_len,
                # this and true silence both collapse to the same empty `segments`.
                parse_failures += 1
                logger.warning(
                    f"MOSS-Transcribe-Diarize emitted a non-empty transcript for "
                    f"{audio_paths[i]!r} that parse_transcript() could not parse into any "
                    "segments; this looks like a broken transcript-format contract rather "
                    "than an absence of speech."
                )
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

        if audio_results and parse_failures == len(audio_results):
            # Every audio in the batch produced a non-empty transcript that failed to
            # parse — far more likely a broken parse_transcript() contract (e.g. an
            # upstream output-format change) than every clip happening to be silent.
            # Raising here keeps that distinguishable from a legitimate empty-segments
            # result, which would otherwise be recorded as a "status-ok" outcome.
            raise RuntimeError(
                f"MOSS-Transcribe-Diarize produced a non-empty transcript that failed to parse "
                f"into any segments for all {len(audio_results)} audio(s) in this batch; this "
                "looks like a broken parse_transcript() contract rather than an absence of "
                "speech. See the preceding warning(s) for the affected audios."
            )

        return results
