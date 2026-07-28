"""CrisperWhisper 2.0 (verbatim, word-timed) via an isolated subprocess venv.

CrisperWhisper 2.0 (nyralabs) is a Whisper-derived model tuned for **verbatim**
transcription and **word-level timestamps** (~30-40 ms boundary error). It ships
as the ``crisperwhisper`` pip package (2.x) with a CTranslate2 backend
(``crisperwhisper[ct2]``) rather than plain ``transformers`` weights, so it runs
in its own venv (same pattern as the Qwen / Canary / Brouhaha backends) — the
CT2 fork (``ctranslate2-crisperwhisper``) must not leak into the senselab core.

The worker returns per-word timestamps and native per-word confidence (when the
library exposes it) so the utterance axis can consume a native uncertainty
signal via ``ScriptLine.score`` (line-level) and each word chunk's ``score``.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Optional

from senselab.audio.data_structures import Audio
from senselab.utils.data_structures import DeviceType, HFModel, ScriptLine, _select_device_and_dtype
from senselab.utils.dependencies import hf_subprocess_env
from senselab.utils.subprocess_venv import _clean_subprocess_env, ensure_venv, parse_subprocess_result, venv_python

_CRISPER_VENV = "crisperwhisper"
# Backend is platform-selected. ``ctranslate2-crisperwhisper`` only publishes
# Linux x86_64 wheels, so the fast CT2 path is used there (the GPU / CI target);
# everywhere else (e.g. macOS arm64 dev) falls back to the transformers backend,
# which loads the same model's safetensors weights. Both extras expose the same
# ``CrisperWhisperModel`` API, so the worker is backend-agnostic.
_IS_LINUX_X86 = sys.platform.startswith("linux") and platform.machine().lower() in ("x86_64", "amd64")
if _IS_LINUX_X86:
    _CRISPER_REQUIREMENTS = ["crisperwhisper[ct2]==2.0.1"]
else:
    _CRISPER_REQUIREMENTS = [
        "crisperwhisper[transformers]==2.0.1",
        # The library imports `ctranslate2` at module top (engine/hallucination)
        # even on the transformers path, but the [transformers] extra doesn't
        # install it and the CT2 *fork* is Linux-x86-only. Standard ctranslate2
        # has macOS-arm64 wheels and satisfies those imports (the transformers
        # backend doesn't actually run CT2 inference).
        "ctranslate2>=4.0",
        # Pin torch/torchaudio explicitly so ensure_venv routes them through the
        # CUDA-aware PyTorch index (the [transformers] extra pulls torch>=2.4).
        "torch>=2.4",
        "torchaudio>=2.4",
    ]
_CRISPER_PYTHON = "3.12"

# Backend token passed to CrisperWhisperModel(..., backend=...). "auto" would try
# ct2 first (and fail off Linux x86_64), so we pin it explicitly per platform.
_CRISPER_BACKEND = "ct2" if _IS_LINUX_X86 else "transformers"


# Worker — runs inside the isolated venv.
_CRISPER_WORKER_SCRIPT = r"""
import json
import sys

try:
    from crisperwhisper import CrisperWhisperModel

    args = json.loads(sys.stdin.read())
    audio_paths = args["audio_paths"]
    model_id = args["model_id"]
    backend = args.get("backend", "auto")
    device = args.get("device", "auto")
    compute_type = args.get("compute_type", "float32")
    language = args.get("language") or "en"

    model = CrisperWhisperModel(model_id, backend=backend, device=device, compute_type=compute_type)

    def _first_attr(obj, names):
        for n in names:
            v = getattr(obj, n, None)
            if v is not None:
                return v
        return None

    results = []
    for path in audio_paths:
        r = model.transcribe(path, language=language, word_timestamps=True)
        words = []
        for w in (getattr(r, "words", None) or []):
            conf = _first_attr(w, ("probability", "confidence", "score", "prob"))
            words.append({
                "text": _first_attr(w, ("word", "text")) or "",
                "start": float(w.start),
                "end": float(w.end),
                "score": (float(conf) if conf is not None else None),
            })
        line_conf = _first_attr(r, ("confidence", "avg_logprob", "score"))
        if line_conf is None:
            cs = [w["score"] for w in words if w["score"] is not None]
            line_conf = (sum(cs) / len(cs)) if cs else None
        results.append({
            "text": getattr(r, "text", "") or "",
            "language": getattr(r, "language", language),
            "words": words,
            "score": (float(line_conf) if line_conf is not None else None),
        })

    print(json.dumps({"results": results}))
except Exception as exc:
    import traceback
    err = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc(limit=5)}
    print(json.dumps({"error": err}))
    sys.exit(1)
"""


class CrisperWhisperASR:
    """CrisperWhisper 2.0 transcription via its isolated CT2 subprocess venv.

    Routed automatically by ``speech_to_text.api`` when the model id matches the
    ``nyralabs/CrisperWhisper2.0`` prefix. Returns one ``ScriptLine`` per audio
    with verbatim ``text``, per-word ``chunks`` (with ``score`` = native word
    confidence when available), and a line-level ``score``.
    """

    @classmethod
    def transcribe_with_crisperwhisper(
        cls,
        audios: List[Audio],
        model: Optional[HFModel] = None,
        device: Optional[DeviceType] = None,
        language: Optional[str] = None,
    ) -> List[ScriptLine]:
        """Transcribe audios with CrisperWhisper 2.0 via the dedicated subprocess venv.

        Args:
            audios: Audio clips (mono, 16 kHz expected).
            model: HF model id (default ``nyralabs/CrisperWhisper2.0_turbo``).
            device: CPU or CUDA (CT2 auto-uses the GPU when available).
            language: Force a language (default ``en``); CrisperWhisper is en/de.

        Returns:
            One ``ScriptLine`` per input with verbatim ``text``, word-level
            ``chunks`` carrying timestamps + ``score`` (native word confidence
            when exposed), and a line-level ``score``.
        """
        model_id = str(model.path_or_uri) if model is not None else "nyralabs/CrisperWhisper2.0_turbo"
        device_type, _ = _select_device_and_dtype(
            user_preference=device, compatible_devices=[DeviceType.CUDA, DeviceType.CPU]
        )
        # float16 only on CUDA; CPU (e.g. macOS transformers backend) needs float32.
        device_str = "cuda" if device_type == DeviceType.CUDA else "cpu"
        compute_type = "float16" if device_str == "cuda" else "float32"

        venv_dir = ensure_venv(_CRISPER_VENV, _CRISPER_REQUIREMENTS, python_version=_CRISPER_PYTHON)
        python = venv_python(venv_dir)

        with tempfile.TemporaryDirectory(prefix="senselab-crisperwhisper-") as tmpdir:
            tmp = Path(tmpdir)
            audio_paths: List[str] = []
            for i, audio in enumerate(audios):
                path = str(tmp / f"audio_{i}.wav")
                audio.save_to_file(path)
                audio_paths.append(path)

            input_json = json.dumps(
                {
                    "audio_paths": audio_paths,
                    "model_id": model_id,
                    "backend": _CRISPER_BACKEND,
                    "device": device_str,
                    "compute_type": compute_type,
                    "language": language or "en",
                }
            )
            # Stage the model once (cross-process heartbeat lock) + run the worker
            # offline so its weight fetch makes no per-call Hub version check — the
            # 429 source under parallel batch. If staging fails, hf_subprocess_env
            # leaves the env online so the worker's current fetch path still runs.
            env = hf_subprocess_env(str(model_id), "main", base_env=_clean_subprocess_env())
            result = subprocess.run(
                [python, "-c", _CRISPER_WORKER_SCRIPT],
                input=input_json,
                capture_output=True,
                text=True,
                timeout=1800,
                env=env,
            )
            output = parse_subprocess_result(result, "CrisperWhisper 2.0")

            results: List[ScriptLine] = []
            for entry in output.get("results", []):
                words = entry.get("words") or []
                chunks: Optional[List[ScriptLine]] = None
                line_start: Optional[float] = None
                line_end: Optional[float] = None
                if words:
                    chunks = [
                        ScriptLine(
                            text=w["text"],
                            start=float(w["start"]),
                            end=float(w["end"]),
                            score=(float(w["score"]) if w.get("score") is not None else None),
                        )
                        for w in words
                    ]
                    chunks.sort(key=lambda c: c.start if c.start is not None else 0.0)
                    starts = [c.start for c in chunks if c.start is not None]
                    ends = [c.end for c in chunks if c.end is not None]
                    line_start = min(starts) if starts else None
                    line_end = max(ends) if ends else None
                results.append(
                    ScriptLine(
                        text=entry.get("text", ""),
                        start=line_start,
                        end=line_end,
                        chunks=chunks,
                        score=(float(entry["score"]) if entry.get("score") is not None else None),
                    )
                )
            return results
