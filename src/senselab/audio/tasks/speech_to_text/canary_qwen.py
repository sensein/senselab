"""NVIDIA Canary-Qwen 2.5B ASR via isolated subprocess venv.

Canary-Qwen is loaded via NeMo's ``SALM`` class (Speech-Augmented Language
Model) from ``nemo.collections.speechlm2.models`` — a different code path
than the existing NeMo ASR flow in ``nemo.py`` (which uses
``nemo.collections.asr.models.ASRModel``). It also requires a wider set
of NeMo extras (``[asr,tts]``) and currently a NeMo trunk pin, so we
isolate it in a SEPARATE venv from ``nemo-diarization`` to avoid
destabilizing the Sortformer / Conformer-CTC paths that already work.

Canary-Qwen is text-only — it has no native timestamp output. The
analyze_audio script's auto-align stage adds per-segment timestamps via
the multilingual MMS forced-aligner (see
``senselab.audio.tasks.forced_alignment``) when this backend is used
through ``transcribe_audios``.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.preprocessing import SegmentStrategy, segment_audios_at_pauses
from senselab.utils.data_structures import DeviceType, HFModel, ScriptLine, _select_device_and_dtype
from senselab.utils.dependencies import hf_subprocess_env
from senselab.utils.subprocess_venv import _clean_subprocess_env, ensure_venv, parse_subprocess_result, venv_python

# Dedicated venv — kept separate from the existing nemo-diarization venv.
# Canary-Qwen needs nemo_toolkit[asr,tts] (the [tts] extra pulls
# speechlm2 dependencies including the Qwen LM components) and a NeMo
# trunk build that publishes SALM. Pinning to the trunk keeps this venv
# updatable without affecting the stable nemo-diarization venv.
_CANARY_VENV = "nemo-canary-qwen"
# NOTE on the torch + torchaudio pins below: the version constraint here is
# necessary but not sufficient on newer-CUDA hosts. PyPI's default resolver
# can pick `torch` and `torchaudio` built for different CUDA toolchains,
# which breaks their ABI contract at import. The shared ``ensure_venv``
# routes the install through the matching PyTorch wheel index
# (``cu128``/``cu126``/``cu124``/``cu121``/``cpu``) — that's what
# guarantees the toolchain match. Do not add a backend-local install path
# that bypasses ``ensure_venv``.
_CANARY_REQUIREMENTS = [
    # NeMo trunk publishes SALM via nemo.collections.speechlm2.models.
    # When NeMo cuts a stable release that includes SALM, swap this for a
    # version pin (e.g., "nemo_toolkit[asr,tts]>=2.5"); for now trunk is
    # the only path.
    "nemo_toolkit[asr,tts] @ git+https://github.com/NVIDIA/NeMo.git",
    "torch>=2.8,<2.9",
    "torchaudio>=2.8,<2.9",
    "pyarrow<18",
    "matplotlib",
    "soundfile",
    # NeMo pulls librosa without a floor; on Python 3.12 uv otherwise backtracks
    # librosa -> numba -> llvmlite to llvmlite 0.36.0 (no 3.12 support). Pin a
    # modern numba floor so the chain resolves to llvmlite>=0.43 (same fix as the
    # qwen-asr venv).
    "numba>=0.60",
    # SALM's Qwen LM uses a LoRA adapter -> the worker imports peft at load time,
    # but NeMo doesn't always pull it transitively on a fresh resolve. Pin it
    # explicitly (the senselab core lists peft for Granite; the isolated venv needs
    # its own copy). Surfaced as "No module named 'peft'" at canary load.
    "peft>=0.13",
]
_CANARY_PYTHON = "3.12"

# Per-chunk audio ceiling for long-audio splitting. Canary-Qwen was trained on
# audio up to 40 s and has a 1024-token total budget (prompt + audio + response)
# at ~12.5 audio tokens/s, so whole long recordings truncate (validated: mean
# word recovery vs Whisper falls from ~1.0 below ~1024 total tokens to ~0.28 for
# multi-minute audio). 38 s == 475 audio tokens, leaving ample room in the budget
# while staying inside the training window. Splitting happens at pauses via
# ``segment_audios_at_pauses`` so no word is cut.
_CANARY_WINDOW_S = 38.0

# Worker script — runs inside the isolated venv.
# The chat-style prompt format with ``audio_locator_tag`` plus
# ``{"audio": [path]}`` matches the published Canary-Qwen model card
# example. Decoding via ``model.tokenizer.ids_to_text(ids)`` recovers
# the transcribed text from generated token ids.
_CANARY_WORKER_SCRIPT = r"""
import json
import sys

try:
    import torch
    from nemo.collections.speechlm2.models import SALM

    args = json.loads(sys.stdin.read())
    audio_paths = args["audio_paths"]
    model_name = args["model_name"]
    device = args["device"]
    revision = args.get("revision") or "main"

    # Load the revision the parent requested rather than the default "main".
    model = SALM.from_pretrained(model_name, revision=revision)
    if device == "cuda" and torch.cuda.is_available():
        model = model.cuda()

    all_results = []
    with torch.no_grad():
        for path in audio_paths:
            # SALM.generate expects prompts as list[list[dict]] — a batch of
            # conversations, where each conversation is a list of messages.
            prompts = [
                [
                    {
                        "role": "user",
                        "content": f"Transcribe the following: {model.audio_locator_tag}",
                        "audio": [path],
                    }
                ]
            ]
            output_ids = model.generate(prompts=prompts, max_new_tokens=512)
            # output_ids shape: (batch, seq_len). Decode the full output. NeMo
            # SALM normally returns only the completion tokens, but if a future
            # build echoes the prompt we strip the leading "Transcribe the
            # following: ..." preamble so it doesn't leak into the transcript.
            row = output_ids[0]
            ids = row.tolist() if hasattr(row, "tolist") else list(row)
            text = model.tokenizer.ids_to_text(ids)
            stripped = text.strip()
            prompt_marker = "Transcribe the following:"
            if prompt_marker in stripped:
                # Take everything after the last occurrence of the marker — covers
                # prompt-echo without dropping the marker if it appears in the
                # source audio (vanishingly rare).
                stripped = stripped.rsplit(prompt_marker, 1)[-1].strip()
            all_results.append({"text": stripped})

    print(json.dumps({"results": all_results}))
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


def _regroup_chunk_transcripts(entries: List[dict], chunk_counts: List[int]) -> List[str]:
    """Concatenate per-chunk worker transcripts back into one text per input audio.

    ``entries`` are the worker results in flattened chunk order; ``chunk_counts[i]``
    is how many chunks input ``i`` was split into. Returns one joined transcript
    per input audio, in input order.

    Raises:
        RuntimeError: if the worker returned a different number of chunk results
            than were sent. Advancing past a shortfall would silently drop text
            and misalign every downstream audio, so we fail loudly instead.
    """
    expected = sum(chunk_counts)
    if len(entries) != expected:
        raise RuntimeError(
            f"Canary-Qwen returned {len(entries)} chunk transcripts but {expected} were expected "
            "(worker likely failed on some chunks); refusing to emit a misaligned transcript."
        )
    texts: List[str] = []
    pos = 0
    for count in chunk_counts:
        parts = [(entries[pos + c].get("text") or "").strip() for c in range(count)]
        pos += count
        texts.append(" ".join(t for t in parts if t))
    return texts


class CanaryQwenASR:
    """NVIDIA Canary-Qwen 2.5B transcription via isolated subprocess venv.

    Routed automatically by ``senselab.audio.tasks.speech_to_text.api`` when
    the model id matches the ``nvidia/canary-`` prefix. Returns text-only
    ScriptLines (Canary-Qwen does not emit native timestamps); pair with
    the multilingual forced-aligner downstream to add per-segment timing.
    """

    @classmethod
    def transcribe_with_canary_qwen(
        cls,
        audios: List[Audio],
        model: Optional[HFModel] = None,
        device: Optional[DeviceType] = None,
        chunk_strategy: SegmentStrategy = "greedy",
    ) -> List[ScriptLine]:
        """Transcribe audios with Canary-Qwen via the dedicated subprocess venv.

        Args:
            audios: Audio clips to transcribe (mono, 16 kHz expected).
            model: HF model id (default: ``nvidia/canary-qwen-2.5b``).
            device: CPU or CUDA. CUDA strongly recommended; CPU works but
                is very slow for a 2.5B-parameter model.
            chunk_strategy: How to split audio longer than the model's ~40 s
                window so it does not truncate (see ``segment_audios_at_pauses``):
                ``"greedy"`` (default, pause-aware greedy packing), ``"dp"``
                (optimal pause segmentation), or ``"none"`` (no splitting —
                long audio will truncate). ``greedy`` and ``dp`` were empirically
                equivalent on annotated long recordings, so the simpler ``greedy``
                is the default. Each chunk's transcript is concatenated in time
                order into one ScriptLine per input.

        Returns:
            One ``ScriptLine`` per input audio with ``text`` populated.
            ``start``, ``end``, and ``chunks`` are intentionally None /
            empty — Canary-Qwen does not produce native timestamps.
            Downstream auto-alignment (MMS) adds per-segment timing.
        """
        model_name = model.path_or_uri if model is not None else "nvidia/canary-qwen-2.5b"
        device_type = device or _select_device_and_dtype(compatible_devices=[DeviceType.CUDA, DeviceType.CPU])[0]

        venv_dir = ensure_venv(_CANARY_VENV, _CANARY_REQUIREMENTS, python_version=_CANARY_PYTHON)
        python = venv_python(venv_dir)

        # Split any over-window audio into <=_CANARY_WINDOW_S sub-chunks at pauses
        # so the model never truncates. Audio within the window yields a single
        # chunk, so behavior is unchanged for short clips.
        chunks_per_audio = segment_audios_at_pauses(audios, max_segment_s=_CANARY_WINDOW_S, strategy=chunk_strategy)

        with tempfile.TemporaryDirectory(prefix="senselab-canary-qwen-") as tmpdir:
            tmp = Path(tmpdir)

            audio_paths: List[str] = []
            chunk_counts: List[int] = []
            for ai, chunks in enumerate(chunks_per_audio):
                chunk_counts.append(len(chunks))
                for ci, chunk in enumerate(chunks):
                    path = str(tmp / f"audio_{ai}_{ci}.wav")
                    chunk.save_to_file(path)
                    audio_paths.append(path)

            # Forward the resolved commit SHA to the worker, never the ref (e.g. "main") --
            # the worker has no senselab install and cannot re-resolve, so a bare ref would
            # load whatever "main" happens to point to on this host at this instant, which can
            # disagree with what the rest of a multi-node run resolved. model.commit_sha is
            # already populated by HFModel's constructor-time resolution when model is given;
            # resolve directly for the no-model-passed default path. Deferred import (not at
            # module top) keeps this monkeypatch-friendly at
            # senselab.utils.model_revision.resolve_revision, matching the rest of the codebase.
            from senselab.utils.model_revision import resolve_revision

            revision = (
                model.commit_sha
                if model is not None and model.commit_sha
                else resolve_revision(str(model_name), model.revision if model is not None else "main")
            )
            input_json = json.dumps(
                {
                    "audio_paths": audio_paths,
                    "model_name": model_name,
                    "device": device_type.value,
                    "revision": revision,
                }
            )

            # Stage the model once (cross-process, via the heartbeat lock) + run the
            # worker offline so its SALM.from_pretrained makes no per-call Hub version
            # check — the 429 source under parallel batch.
            env = hf_subprocess_env(str(model_name), revision, base_env=_clean_subprocess_env())
            result = subprocess.run(
                [python, "-c", _CANARY_WORKER_SCRIPT],
                input=input_json,
                capture_output=True,
                text=True,
                # Model load + per-chunk generate; a long recording is many
                # chunks, so allow 30 min (load dominates; chunks are fast).
                timeout=1800,
                env=env,
            )

            output = parse_subprocess_result(result, "Canary-Qwen ASR")
            entries = output.get("results", [])

            # Regroup the per-chunk transcripts back into one ScriptLine per input
            # audio, concatenating chunk text in time order.
            texts = _regroup_chunk_transcripts(entries, chunk_counts)
            return [ScriptLine(text=t) for t in texts]
