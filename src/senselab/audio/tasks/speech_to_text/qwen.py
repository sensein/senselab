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
from typing import Any, List, Optional, Sequence, Tuple

from senselab.audio.data_structures import Audio
from senselab.utils.data_structures import DeviceType, HFModel, ScriptLine, _select_device_and_dtype
from senselab.utils.dependencies import hf_subprocess_env
from senselab.utils.subprocess_venv import _clean_subprocess_env, ensure_venv, parse_subprocess_result, venv_python

_QWEN_VENV = "qwen-asr"
_QWEN_REQUIREMENTS = [
    # Pin to a known-good release; bump intentionally as Alibaba publishes new
    # versions of the wrapper. 0.0.6 is the first version that exposes both
    # Qwen3ASRModel and Qwen3ForcedAligner via from_pretrained on PyPI.
    "qwen-asr==0.0.6",
    # ``qwen-asr`` pulls ``torch`` + ``torchaudio`` transitively, but pin
    # them explicitly here so ``ensure_venv``'s CUDA-aware Stage 1 routes
    # both wheels through the matched PyTorch index. Without these pins
    # the transitives would resolve from default PyPI in Stage 2 and
    # could split across mismatched ``+cu`` local-version tags (the
    # ``torch==X+cu129`` / ``torchaudio==X`` ABI mismatch this PR
    # exists to prevent).
    "torch>=2.8,<2.9",
    "torchaudio>=2.8,<2.9",
    # qwen-asr depends on librosa without a floor; on Python 3.12 uv otherwise
    # backtracks librosa -> numba -> llvmlite to the ancient llvmlite 0.36.0,
    # which has no 3.12 support ("Cannot install on Python version 3.12.0").
    # Pin a modern numba floor so the chain resolves to llvmlite>=0.43 (3.12-ok).
    "numba>=0.60",
]
_QWEN_PYTHON = "3.12"

# Default companion forced-aligner model. Loaded on-demand only when
# return_timestamps=True; the caller-supplied model id is the ASR model
# (Qwen3-ASR-1.7B / Qwen3-ASR-3B / etc.).
_DEFAULT_FORCED_ALIGNER = "Qwen/Qwen3-ForcedAligner-0.6B"

# Standalone forced-alignment worker — aligns an existing (text, audio) pair to
# per-word timestamps via Qwen3ForcedAligner. Used to align text-only ASR output
# (e.g. Canary) that carries no native timestamps. Qwen3ForcedAligner.align()
# takes batched (audio, text, language) lists and returns one iterable
# ForcedAlignResult per input, each yielding spans with .text/.start_time/.end_time.
_QWEN_ALIGN_WORKER_SCRIPT = r"""
import json
import sys

try:
    import torch
    from qwen_asr import Qwen3ForcedAligner

    args = json.loads(sys.stdin.read())
    pairs = args["pairs"]
    aligner_name = args["aligner_model"]
    aligner_revision = args["aligner_revision"]
    device = args["device"]

    # Qwen3ForcedAligner.from_pretrained forwards **kwargs straight to
    # transformers.AutoModel.from_pretrained, which does accept `revision` -- so the
    # resolved commit SHA the parent staged pins the model weights. Its own internal
    # AutoProcessor.from_pretrained(pretrained_model_name_or_path, fix_mistral_regex=True)
    # call, however, does NOT forward revision at all (upstream hardcodes that second
    # call with no revision passthrough), so the processor/tokenizer config still
    # resolves the mutable "main" ref regardless of what's requested here -- a partial
    # pin this call site cannot close without patching qwen_asr itself.
    fa = Qwen3ForcedAligner.from_pretrained(aligner_name, revision=aligner_revision)
    if device == "cuda" and torch.cuda.is_available():
        try:
            fa.model = fa.model.cuda()
        except Exception as cuda_exc:
            print(f"WARN: Qwen aligner CUDA placement failed ({cuda_exc!r}); using CPU.", file=sys.stderr)

    audios = [p["audio_path"] for p in pairs]
    texts = [p["text"] for p in pairs]
    langs = [p.get("language") or "en" for p in pairs]
    aligned = fa.align(audios, texts, langs)

    results = []
    for r in aligned:
        chunks = []
        for span in r:
            chunks.append({"text": span.text, "start": float(span.start_time), "end": float(span.end_time)})
        results.append({"chunks": chunks})

    print(json.dumps({"results": results}))
except Exception as exc:
    import traceback
    err = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc(limit=5)}
    print(json.dumps({"error": err}))
    sys.exit(1)
"""

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
    model_revision = args["model_revision"]
    device = args["device"]
    return_timestamps = bool(args.get("return_timestamps", True))
    aligner_name = args.get("forced_aligner") if return_timestamps else None
    aligner_revision = args.get("forced_aligner_revision") if return_timestamps else None

    # revision is forwarded through **kwargs to transformers.AutoModel.from_pretrained,
    # which does accept it -- so the resolved commit SHA the parent staged pins the ASR
    # model weights. forced_aligner_kwargs is forwarded the same way into
    # Qwen3ForcedAligner.from_pretrained(forced_aligner, **forced_aligner_kwargs), which
    # forwards its own revision the same route. Both wrappers' internal
    # AutoProcessor.from_pretrained(...) calls do NOT take revision at all (see
    # align_with_qwen's worker-script comment), so the tokenizer/processor config is a
    # known, upstream-caused partial-pin gap this call site cannot close.
    load_kwargs = {"revision": model_revision}
    if aligner_name:
        load_kwargs["forced_aligner"] = aligner_name
        load_kwargs["forced_aligner_kwargs"] = {"revision": aligner_revision}

    asr = Qwen3ASRModel.from_pretrained(model_name, **load_kwargs)
    # The wrapper holds inner HF modules on .model / .forced_aligner.model;
    # try to move them onto the requested device when CUDA is available.
    if device == "cuda" and torch.cuda.is_available():
        try:
            asr.model = asr.model.cuda()
            if getattr(asr, "forced_aligner", None) is not None:
                asr.forced_aligner.model = asr.forced_aligner.model.cuda()
        except Exception as cuda_exc:
            # If the wrapper internals diverge or .cuda() fails (e.g. OOM), the
            # wrapper still runs but on CPU, which is ~50× slower. Make the
            # downgrade visible to the parent process via the subprocess log so
            # the user can choose to abort rather than wait.
            print(
                f"WARN: Qwen3-ASR CUDA placement failed ({cuda_exc!r}); "
                f"falling back to wrapper-default device (likely CPU).",
                file=sys.stderr,
            )

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
        if model is None:
            model = HFModel(path_or_uri="Qwen/Qwen3-ASR-1.7B")
        model_name = str(model.path_or_uri)
        device_type = device or _select_device_and_dtype(compatible_devices=[DeviceType.CUDA, DeviceType.CPU])[0]
        aligner_name = forced_aligner or _DEFAULT_FORCED_ALIGNER

        venv_dir = ensure_venv(_QWEN_VENV, _QWEN_REQUIREMENTS, python_version=_QWEN_PYTHON)
        python = venv_python(venv_dir)

        # Resolve both the ASR model and (when requested) its forced-aligner companion to
        # immutable commit SHAs before staging -- never the ref -- so the worker's
        # Qwen3ASRModel.from_pretrained(..., revision=...) pins the exact run-agreed
        # commit rather than whatever this host's mutable "main" ref currently points at.
        # Deferred import (not at module top) keeps this monkeypatch-friendly at
        # senselab.utils.model_revision.resolve_revision, matching the rest of the codebase.
        from senselab.utils.model_revision import resolve_revision

        revision = model.commit_sha or resolve_revision(model_name, model.revision)
        aligner_revision = resolve_revision(aligner_name, "main") if return_timestamps else None

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
                    "model_revision": revision,
                    "device": device_type.value,
                    "return_timestamps": return_timestamps,
                    "forced_aligner": aligner_name if return_timestamps else None,
                    "forced_aligner_revision": aligner_revision,
                }
            )

            # Stage the ASR model (and, when timestamps are requested, its forced-aligner
            # companion) once — cross-process — and flip the child to offline so its
            # from_pretrained loads from cache with no per-call Hub version check (the 429
            # source under many parallel jobs). The child imports fresh, so the offline
            # flag is honored (unlike an in-process toggle).
            also = [(aligner_name, aligner_revision)] if return_timestamps and aligner_revision is not None else None
            env = hf_subprocess_env(model_name, revision, also=also, base_env=_clean_subprocess_env())
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
                    # These times are the *companion aligner's*, not the recognizer's — the
                    # `forced_aligner` kwarg above is what produced them. Recording which model,
                    # not merely that a bundled one ran: the workflow aligns text-only backends
                    # (Canary) with this same aligner id, and a consumer comparing word times has
                    # to see that as one timing source rather than two agreeing ones.
                    chunks = [
                        ScriptLine(
                            text=c["text"],
                            start=float(c["start"]),
                            end=float(c["end"]),
                            timestamp_source="bundled_aligner" if return_timestamps else None,
                            timestamp_model=aligner_name if return_timestamps else None,
                        )
                        for c in chunks_raw
                    ]
                    # Sort by start so out-of-order aligner output doesn't yield a
                    # negative line span; pick the min start and max end across all
                    # chunks rather than trusting position-0/-1.
                    chunks.sort(key=lambda c: c.start if c.start is not None else 0.0)
                    valid_starts = [c.start for c in chunks if c.start is not None]
                    valid_ends = [c.end for c in chunks if c.end is not None]
                    line_start = min(valid_starts) if valid_starts else None
                    line_end = max(valid_ends) if valid_ends else None
                results.append(
                    ScriptLine(
                        text=entry.get("text", ""),
                        start=line_start,
                        end=line_end,
                        chunks=chunks,
                        timestamp_source="bundled_aligner" if return_timestamps and chunks else None,
                        timestamp_model=aligner_name if return_timestamps and chunks else None,
                    )
                )

            return results

    @classmethod
    def align_with_qwen(
        cls,
        data: Sequence[Tuple[Audio, ScriptLine, Any]],
        levels_to_keep: Optional[dict] = None,  # noqa: ARG003 — accepted for align_transcriptions parity
        aligner_model: Optional[str] = None,
        device: Optional[DeviceType] = None,
        **_kwargs: Any,  # noqa: ANN401 — parity with align_transcriptions' extra kwargs
    ) -> List[List[ScriptLine]]:
        """Force-align existing (audio, transcript) pairs with Qwen3-ForcedAligner.

        Drop-in alternative to ``forced_alignment.align_transcriptions`` for
        text-only ASR output that lacks native timestamps (e.g. Canary). Runs in
        the shared ``qwen-asr`` subprocess venv.

        Args:
            data: ``(audio, ScriptLine(text=...), Language|None)`` tuples — same
                call form the script uses for ``align_transcriptions``.
            levels_to_keep: Ignored (Qwen emits word-level spans); accepted only
                for signature parity so the caller can swap aligners freely.
            aligner_model: Aligner id (default ``Qwen/Qwen3-ForcedAligner-0.6B``).
            device: CPU or CUDA.

        Returns:
            ``List[List[ScriptLine]]`` mirroring ``align_transcriptions``: one
            inner list per input audio, holding a single asr ``ScriptLine``
            whose ``chunks`` are the aligned word spans.
        """
        aligner_name = aligner_model or _DEFAULT_FORCED_ALIGNER
        device_type = device or _select_device_and_dtype(compatible_devices=[DeviceType.CUDA, DeviceType.CPU])[0]

        venv_dir = ensure_venv(_QWEN_VENV, _QWEN_REQUIREMENTS, python_version=_QWEN_PYTHON)
        python = venv_python(venv_dir)

        # Resolve the ref to a commit SHA before staging, never the ref -- so the worker's
        # Qwen3ForcedAligner.from_pretrained(..., revision=...) pins the exact run-agreed
        # commit. Deferred import (not at module top) keeps this monkeypatch-friendly at
        # senselab.utils.model_revision.resolve_revision, matching the rest of the codebase.
        from senselab.utils.model_revision import resolve_revision

        aligner_revision = resolve_revision(aligner_name, "main")

        with tempfile.TemporaryDirectory(prefix="senselab-qwen-align-") as tmpdir:
            tmp = Path(tmpdir)
            pairs: List[dict] = []
            for i, (audio, script, language) in enumerate(data):
                path = str(tmp / f"audio_{i}.wav")
                audio.save_to_file(path)
                pairs.append(
                    {
                        "audio_path": path,
                        "text": script.text or "",
                        "language": getattr(language, "language_code", None) or "en",
                    }
                )

            input_json = json.dumps(
                {
                    "pairs": pairs,
                    "aligner_model": aligner_name,
                    "aligner_revision": aligner_revision,
                    "device": device_type.value,
                }
            )
            # Stage the aligner once (cross-process, via the heartbeat lock) + run the
            # worker offline so its from_pretrained makes no per-call Hub version check —
            # the 429 source under parallel batch. This call previously ran with no
            # staging/offline env at all, unlike every other subprocess-venv backend.
            env = hf_subprocess_env(aligner_name, aligner_revision, base_env=_clean_subprocess_env())
            result = subprocess.run(
                [python, "-c", _QWEN_ALIGN_WORKER_SCRIPT],
                input=input_json,
                capture_output=True,
                text=True,
                timeout=1800,
                env=env,
            )
            output = parse_subprocess_result(result, "Qwen forced aligner")

            aligned: List[List[ScriptLine]] = []
            for entry in output.get("results", []):
                chunks = [
                    ScriptLine(text=c["text"], start=float(c["start"]), end=float(c["end"]))
                    for c in (entry.get("chunks") or [])
                ]
                chunks.sort(key=lambda c: c.start if c.start is not None else 0.0)
                if not chunks:
                    aligned.append([])
                    continue
                starts = [c.start for c in chunks if c.start is not None]
                ends = [c.end for c in chunks if c.end is not None]
                asr = ScriptLine(
                    text=" ".join(c.text or "" for c in chunks).strip(),
                    start=min(starts) if starts else None,
                    end=max(ends) if ends else None,
                    chunks=chunks,
                )
                aligned.append([asr])
            return aligned
