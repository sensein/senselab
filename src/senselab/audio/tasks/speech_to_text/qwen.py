"""Alibaba Qwen3-ASR via isolated subprocess venv.

Qwen3-ASR is loaded via Alibaba's ``qwen-asr`` Python wrapper (the
``Qwen3ASRModel`` class) which itself wraps a Hugging Face Transformers
model under the hood. It optionally pairs with the companion
``Qwen3ForcedAligner`` (default companion model:
``Qwen/Qwen3-ForcedAligner-0.6B``) to produce per-word / per-CJK-char
timestamps as part of the same call.

We isolate this in its own venv (``qwen-asr``) — kept separate from the
existing NeMo and Canary-Qwen venvs — because the wrapper pulls a
fairly large dependency tree (gradio, dynet38, nagisa) that we do not
want to leak into the senselab core install.
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

_QWEN_VENV = "qwen-asr"
_QWEN_REQUIREMENTS = [
    # Pin to a known-good release; bump intentionally as Alibaba publishes new
    # versions of the wrapper. 0.0.6 is the first version that exposes both
    # Qwen3ASRModel and Qwen3ForcedAligner via from_pretrained on PyPI.
    "qwen-asr==0.0.6",
]
_QWEN_PYTHON = "3.12"

# Default companion forced-aligner model. Loaded on-demand only when
# return_timestamps=True; the caller-supplied model id is the ASR model
# (Qwen3-ASR-1.7B / Qwen3-ASR-3B / etc.).
_DEFAULT_FORCED_ALIGNER = "Qwen/Qwen3-ForcedAligner-0.6B"

# Worker script — runs inside the isolated venv.
# Uses Qwen3ASRModel.from_pretrained's built-in `forced_aligner` kwarg
# (a string id) so the wrapper handles aligner construction internally.
# The transcribe() call returns a list[ASRTranscription], where each item
# has .text, .language, and (when return_time_stamps=True) .time_stamps
# (a ForcedAlignResult with .items[].text/.start_time/.end_time).
_QWEN_WORKER_SCRIPT = r"""
import json
import sys

try:
    import torch
    from qwen_asr import Qwen3ASRModel

    args = json.loads(sys.stdin.read())
    audio_paths = args["audio_paths"]
    model_name = args["model_name"]
    device = args["device"]
    return_timestamps = bool(args.get("return_timestamps", True))
    aligner_name = args.get("forced_aligner") if return_timestamps else None

    load_kwargs = {}
    if aligner_name:
        load_kwargs["forced_aligner"] = aligner_name

    asr = Qwen3ASRModel.from_pretrained(model_name, **load_kwargs)
    # The wrapper holds inner HF modules on .model / .forced_aligner.model;
    # try to move them onto the requested device when CUDA is available.
    if device == "cuda" and torch.cuda.is_available():
        try:
            asr.model = asr.model.cuda()
            if getattr(asr, "forced_aligner", None) is not None:
                asr.forced_aligner.model = asr.forced_aligner.model.cuda()
        except Exception:
            # If the wrapper internals diverge, fall back to letting the
            # wrapper's own device-handling kick in.
            pass

    results = asr.transcribe(
        audio=audio_paths,
        return_time_stamps=return_timestamps,
    )

    serialized = []
    for r in results:
        item = {"text": r.text, "language": r.language}
        if return_timestamps and r.time_stamps is not None:
            chunks = []
            for span in r.time_stamps:
                chunks.append({
                    "text": span.text,
                    "start": float(span.start_time),
                    "end": float(span.end_time),
                })
            item["chunks"] = chunks
        serialized.append(item)

    print(json.dumps({"results": serialized}))
except Exception as exc:
    import traceback
    err = {
        "type": type(exc).__name__,
        "message": str(exc),
        "traceback": traceback.format_exc(limit=5),
    }
    print(json.dumps({"error": err}))
    sys.exit(1)
"""


class QwenASR:
    """Alibaba Qwen3-ASR transcription via isolated subprocess venv.

    Routed automatically by ``senselab.audio.tasks.speech_to_text.api`` when
    the model id matches the ``Qwen/Qwen3-ASR`` prefix. Returns ScriptLines
    with ``text`` and (when ``return_timestamps=True``) per-word / per-char
    chunks populated from the companion ``Qwen3-ForcedAligner-0.6B``.
    """

    @classmethod
    def transcribe_with_qwen(
        cls,
        audios: List[Audio],
        model: Optional[HFModel] = None,
        device: Optional[DeviceType] = None,
        return_timestamps: bool = True,
        forced_aligner: Optional[str] = None,
    ) -> List[ScriptLine]:
        """Transcribe audios with Qwen3-ASR via the dedicated subprocess venv.

        Args:
            audios: Audio clips to transcribe (mono, 16 kHz expected).
            model: HF model id (default: ``Qwen/Qwen3-ASR-1.7B``).
            device: CPU or CUDA. CUDA strongly recommended.
            return_timestamps: When True, also load the companion forced
                aligner and populate per-span ``chunks`` on each ScriptLine.
            forced_aligner: Override the companion aligner model id.
                Defaults to ``Qwen/Qwen3-ForcedAligner-0.6B`` when
                ``return_timestamps=True``.

        Returns:
            One ``ScriptLine`` per input audio with ``text`` populated. When
            ``return_timestamps=True``, ``chunks`` is a list of word /
            CJK-char-level ``ScriptLine`` entries with ``start``/``end``.
        """
        model_name = model.path_or_uri if model is not None else "Qwen/Qwen3-ASR-1.7B"
        device_type = device or _select_device_and_dtype(compatible_devices=[DeviceType.CUDA, DeviceType.CPU])[0]
        aligner_name = forced_aligner or _DEFAULT_FORCED_ALIGNER

        venv_dir = ensure_venv(_QWEN_VENV, _QWEN_REQUIREMENTS, python_version=_QWEN_PYTHON)
        python = venv_python(venv_dir)

        with tempfile.TemporaryDirectory(prefix="senselab-qwen-asr-") as tmpdir:
            tmp = Path(tmpdir)

            audio_paths: List[str] = []
            for i, audio in enumerate(audios):
                path = str(tmp / f"audio_{i}.wav")
                audio.save_to_file(path)
                audio_paths.append(path)

            input_json = json.dumps(
                {
                    "audio_paths": audio_paths,
                    "model_name": model_name,
                    "device": device_type.value,
                    "return_timestamps": return_timestamps,
                    "forced_aligner": aligner_name if return_timestamps else None,
                }
            )

            env = _clean_subprocess_env()
            result = subprocess.run(
                [python, "-c", _QWEN_WORKER_SCRIPT],
                input=input_json,
                capture_output=True,
                text=True,
                timeout=1800,  # 1.7B-3B ASR + 0.6B aligner load + per-audio decode; allow 30 min.
                env=env,
            )

            output = parse_subprocess_result(result, "Qwen3-ASR")

            results: List[ScriptLine] = []
            for entry in output.get("results", []):
                chunks_raw = entry.get("chunks")
                chunks: Optional[List[ScriptLine]] = None
                line_start: Optional[float] = None
                line_end: Optional[float] = None
                if chunks_raw:
                    chunks = [
                        ScriptLine(
                            text=c["text"],
                            start=float(c["start"]),
                            end=float(c["end"]),
                        )
                        for c in chunks_raw
                    ]
                    line_start = chunks[0].start
                    line_end = chunks[-1].end
                results.append(
                    ScriptLine(
                        text=entry.get("text", ""),
                        start=line_start,
                        end=line_end,
                        chunks=chunks,
                    )
                )

            return results
