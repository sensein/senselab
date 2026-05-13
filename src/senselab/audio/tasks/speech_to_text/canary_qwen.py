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
from senselab.utils.data_structures import DeviceType, HFModel, ScriptLine, _select_device_and_dtype
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
]
_CANARY_PYTHON = "3.12"

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

    model = SALM.from_pretrained(model_name)
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
    ) -> List[ScriptLine]:
        """Transcribe audios with Canary-Qwen via the dedicated subprocess venv.

        Args:
            audios: Audio clips to transcribe (mono, 16 kHz expected).
            model: HF model id (default: ``nvidia/canary-qwen-2.5b``).
            device: CPU or CUDA. CUDA strongly recommended; CPU works but
                is very slow for a 2.5B-parameter model.

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

        with tempfile.TemporaryDirectory(prefix="senselab-canary-qwen-") as tmpdir:
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
                }
            )

            env = _clean_subprocess_env()
            result = subprocess.run(
                [python, "-c", _CANARY_WORKER_SCRIPT],
                input=input_json,
                capture_output=True,
                text=True,
                timeout=1200,  # 2.5B-param model load + per-audio generate; allow 20 min.
                env=env,
            )

            output = parse_subprocess_result(result, "Canary-Qwen ASR")

            results: List[ScriptLine] = []
            for entry in output.get("results", []):
                results.append(ScriptLine(text=entry.get("text", "")))

            return results
