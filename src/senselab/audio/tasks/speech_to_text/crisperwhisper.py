"""CrisperWhisper 2.0 (verbatim, word-timed) via an isolated subprocess venv.

CrisperWhisper 2.0 (nyralabs) is a Whisper-derived model tuned for **verbatim**
transcription and **word-level timestamps** (~30-40 ms boundary error). It ships
as the ``crisperwhisper`` pip package (2.x) with a CTranslate2 backend
(``crisperwhisper[ct2]``) rather than plain ``transformers`` weights, so it runs
in its own venv (same pattern as the Qwen / Canary / Brouhaha backends) — the
CT2 fork (``ctranslate2-crisperwhisper``) must not leak into the senselab core.

The worker returns per-word timestamps and native per-word confidence (when the
library exposes it) so the asr axis can consume a native uncertainty
signal via ``ScriptLine.score`` (line-level) and each word chunk's ``score``.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path
from typing import List, Optional

from senselab.audio.data_structures import Audio
from senselab.utils.data_structures import DeviceType, HFModel, ScriptLine, _select_device_and_dtype
from senselab.utils.dependencies import hf_subprocess_env, resolve_model
from senselab.utils.subprocess_venv import _clean_subprocess_env, ensure_venv, parse_subprocess_result, venv_python

_CRISPER_VENV = "crisperwhisper"
# Backend is platform-selected. ``ctranslate2-crisperwhisper`` only publishes
# Linux x86_64 wheels, so the fast CT2 path is used there (the GPU / CI target);
# everywhere else (e.g. macOS arm64 dev) falls back to the transformers backend,
# which loads the same model's safetensors weights. Both extras expose the same
# ``CrisperWhisperModel`` API, so the worker is backend-agnostic.
_IS_LINUX_X86 = sys.platform.startswith("linux") and platform.machine().lower() in ("x86_64", "amd64")
if _IS_LINUX_X86:
    # CT2 *inference* is torch-free, but the first-run HF->CT2 *conversion* goes through
    # ``ctranslate2.converters.transformers``, whose ``try: import huggingface_hub, torch,
    # transformers`` block leaves those names unbound when absent; ``_load()`` then calls
    # ``torch.no_grad()`` + the transformers loader -> ``NameError: name 'torch' /
    # 'transformers' is not defined``. The venv "builds" but transcription fails on every
    # clip. So the ct2 venv also needs the conversion stack: use the ``[transformers]``
    # extra (== [all] with ct2: transformers + torch + accelerate) and pin torch/torchaudio
    # explicitly so ensure_venv routes them through the CUDA index. None of this is loaded
    # at CT2 inference time — only for the one-time, cached HF->CT2 conversion.
    _CRISPER_REQUIREMENTS = ["crisperwhisper[ct2,transformers]==2.0.1", "torch>=2.4", "torchaudio>=2.4"]
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

# The library's HF->CT2 conversion cache: one directory per (model, quantization),
# holding ``model.bin`` and stamped with ``.conversion_complete`` when written.
_CT2_WEIGHTS = "model.bin"
_CT2_MARKER = ".conversion_complete"


def _ct2_cache_root() -> Path:
    """Return the conversion-cache root ``crisperwhisper.converter`` reads."""
    env = os.environ.get("CRISPERWHISPER_CACHE")
    return Path(env) if env else Path.home() / ".cache" / "crisperwhisper"


def _ct2_cache_key(model_id: str, quantization: str) -> str:
    """Return the conversion-cache directory name for one model and quantization.

    Args:
        model_id: The path or repo id handed to ``CrisperWhisperModel``.
        quantization: The CT2 compute type (``float16``, ``float32``, ...).

    Returns:
        The directory name ``crisperwhisper.converter._cache_key`` would build.
    """
    slug = model_id.replace("/", "--").replace("\\", "--")
    digest = hashlib.sha256(model_id.encode()).hexdigest()[:12]
    return f"{slug}_{quantization}_{digest}"


def _ct2_entry_is_torn(entry: Path) -> bool:
    """Return whether a conversion-cache entry is stamped complete but has no weights.

    Args:
        entry: A conversion-cache directory.

    Returns:
        True when ``.conversion_complete`` exists and ``model.bin`` does not.
    """
    return (entry / _CT2_MARKER).exists() and not (entry / _CT2_WEIGHTS).exists()


def _discard_torn_ct2_entry(entry: Path) -> bool:
    """Detach and delete a conversion-cache entry that carries no weights.

    Args:
        entry: A conversion-cache directory.

    Returns:
        True when a torn entry was detached and deleted, False otherwise.
    """
    if not _ct2_entry_is_torn(entry):
        return False
    detached = entry.with_name(f"{entry.name}.torn-{uuid.uuid4().hex}")
    try:
        os.rename(entry, detached)
    except OSError:
        return False
    shutil.rmtree(detached, ignore_errors=True)
    return True


# Worker — runs inside the isolated venv.
_CRISPER_WORKER_SCRIPT = r"""
import json
import os
import shutil
import sys
from pathlib import Path

try:
    from crisperwhisper import CrisperWhisperModel

    args = json.loads(sys.stdin.read())
    audio_paths = args["audio_paths"]
    model_id = args["model_id"]
    backend = args.get("backend", "auto")
    device = args.get("device", "auto")
    compute_type = args.get("compute_type", "float32")
    language = args.get("language") or "en"

    # The CT2 backend converts the HF snapshot into a shared cache directory whose
    # writer is neither atomic nor locked. Convert into a private staging directory
    # and publish it with one rename, so a concurrent converter can neither be read
    # half-written nor delete what this process just wrote.
    if backend == "ct2":
        entry = Path(args["ct2_entry"])
        if not (entry / "model.bin").exists():
            from crisperwhisper.converter import ensure_ct2_model

            staging = Path(args["ct2_staging"])
            converted = Path(ensure_ct2_model(model_id, quantization=compute_type, cache_dir=str(staging)))
            entry.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.rename(str(converted), str(entry))
            except OSError:
                if not (entry / "model.bin").exists():
                    shutil.rmtree(str(entry), ignore_errors=True)
                    os.rename(str(converted), str(entry))
            shutil.rmtree(str(staging), ignore_errors=True)
        model_id = str(entry)

    # CrisperWhisperModel's __init__ (and both backends it dispatches to -- CT2's
    # ensure_ct2_model/_resolve_hf_or_local, and the transformers backend's
    # AutoProcessor/AutoModelForSpeechSeq2Seq.from_pretrained) take no revision kwarg
    # anywhere in the chain. `model_id` here is therefore the *local snapshot directory*
    # the parent already resolved+staged for the run-agreed commit, not a repo id --
    # `_resolve_hf_or_local`/`from_pretrained` both treat an existing local directory as
    # already pinned and never touch the Hub for it, which is what actually pins the load.
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
        if model is None:
            model = HFModel(path_or_uri="nyralabs/CrisperWhisper2.0_turbo")
        model_id = str(model.path_or_uri)
        device_type, _ = _select_device_and_dtype(
            user_preference=device, compatible_devices=[DeviceType.CUDA, DeviceType.CPU]
        )
        # float16 only on CUDA; CPU (e.g. macOS transformers backend) needs float32.
        device_str = "cuda" if device_type == DeviceType.CUDA else "cpu"
        compute_type = "float16" if device_str == "cuda" else "float32"

        venv_dir = ensure_venv(_CRISPER_VENV, _CRISPER_REQUIREMENTS, python_version=_CRISPER_PYTHON)
        python = venv_python(venv_dir)

        # CrisperWhisperModel has no revision parameter anywhere in its call chain (see
        # the worker-script comment), so resolve the ref to the run-agreed commit SHA
        # (download-once via the cross-process heartbeat lock) and point the worker at
        # that commit's already-staged local snapshot directory instead of the mutable
        # repo id -- both backends treat an existing local directory as already pinned.
        revision, snapshot_path = resolve_model(model_id, model.revision or "main")

        with tempfile.TemporaryDirectory(prefix="senselab-crisperwhisper-") as tmpdir:
            tmp = Path(tmpdir)
            audio_paths: List[str] = []
            for i, audio in enumerate(audios):
                path = str(tmp / f"audio_{i}.wav")
                audio.save_to_file(path)
                audio_paths.append(path)

            cache_root = _ct2_cache_root()
            ct2_entry = cache_root / _ct2_cache_key(str(snapshot_path), compute_type)
            if _CRISPER_BACKEND == "ct2":
                _discard_torn_ct2_entry(ct2_entry)
            input_json = json.dumps(
                {
                    "audio_paths": audio_paths,
                    "model_id": str(snapshot_path),
                    "backend": _CRISPER_BACKEND,
                    "device": device_str,
                    "compute_type": compute_type,
                    "language": language or "en",
                    "ct2_entry": str(ct2_entry),
                    "ct2_staging": str(cache_root / f".staging-{uuid.uuid4().hex}"),
                }
            )
            # Stage the model once (cross-process heartbeat lock) + run the worker
            # offline so its weight fetch makes no per-call Hub version check — the
            # 429 source under parallel batch. If staging fails, hf_subprocess_env
            # leaves the env online so the worker's current fetch path still runs.
            env = hf_subprocess_env(model_id, revision, base_env=_clean_subprocess_env())
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
