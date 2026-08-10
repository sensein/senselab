"""Qwen3-TTS via isolated subprocess venv.

Qwen3-TTS-12Hz-1.7B-CustomVoice (``Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice``, commit
``0c0e3051f131929182e2c023b9537f8b1c68adfe``, licence apache-2.0) is loaded through
Alibaba's ``qwen-tts`` PyPI package (the ``Qwen3TTSModel`` wrapper), which itself
registers a ``qwen3_tts`` architecture onto ``transformers``' ``Auto*`` classes. It is
not a `transformers` architecture available in the host environment, so — same
template as every other backend in this package — it runs in its own subprocess venv
rather than being merged into senselab's core dependency set.

Why this model, specifically
-----------------------------
The checkpoint's config bakes in **9 named speaker identities**
(``config["talker_config"]["spk_id"]``: ``aiden``, ``dylan``, ``eric``, ``ono_anna``,
``ryan``, ``serena``, ``sohee``, ``uncle_fu``, ``vivian`` — verified against the pinned
commit's ``config.json``, not the model card's prose), reachable by name via
``generate_custom_voice(..., speaker=...)`` with **no reference audio to clone from**.
That is the property the speaker-ceiling probe
(``specs/20260809-112417-speaker-ceiling-probe/``) needs: N *distinct* speaker
identities with exact ground truth, generated directly rather than cloned. See
:func:`supported_speakers` for how a caller enumerates them.

The apache-2.0 licence (confirmed live against the Hub repo, not assumed from the model
card) also means — unlike DriftSE and unasdiff, both licence-unknown pending an
unanswered upstream issue — **no ``sensein`` mirror is needed**. This backend loads
straight from the Hub, pinned to a resolved commit like every other HF load in this
codebase.

The 25Hz-tokenizer import trap (verified, not assumed)
--------------------------------------------------------
``qwen_tts/core/__init__.py`` unconditionally imports *both* tokenizer variants —
``tokenizer_12hz`` (what this checkpoint actually uses, per its
``tokenizer_type: qwen3_tts_tokenizer_12hz``) **and** ``tokenizer_25hz`` — regardless
of which one a given checkpoint needs. ``tokenizer_25hz/vq/speech_vq.py`` does
``import sox``, ``import onnxruntime`` and ``import torchaudio.compliance.kaldi`` at
module scope. So even though this backend's runtime path (``from_pretrained`` +
``generate_custom_voice``) never touches the 25Hz tokenizer classes, all three packages
are load-bearing for the mere ``from qwen_tts import Qwen3TTSModel`` the model card
shows — this is DriftSE's ``pesq`` lesson again (a dependency that looks unrelated to
the code path you call is still required if the import chain reaches it), not
unasdiff's ``av`` (genuinely unreached). Confirmed by downloading and inspecting the
``qwen_tts`` 0.1.1 wheel directly, not by reasoning about what "should" be needed.

Two of those three needed a second check: ``import sox`` (the PyPI ``sox`` wrapper
around the SoX *binary*) only logs a warning and sets a module-level flag when the
binary is absent (``sox/__init__.py``: ``if not len(os.popen('sox -h').readlines()):
logger.warning(...)``) — it does not raise. So the SoX CLI binary itself is not a
requirement of this venv; only the pip package is. ``onnxruntime`` is the plain CPU
wheel (upstream declares no ``onnxruntime-gpu`` extra), which is fine because this
backend's call path never opens an ONNX session.

flash-attn is deliberately absent, and not on the "training-only, verify before
trusting" reasoning DriftSE and unasdiff both used — here it is simpler:
``Qwen3TTSModel.from_pretrained``'s own docstring lists
``attn_implementation="flash_attention_2"`` as one *example* value forwarded through
``**kwargs`` to ``AutoModel.from_pretrained``, not a required argument (verified
against the installed wheel's ``qwen3_tts_model.py``). Omitting it costs nothing and
avoids flash-attn's multi-minute ``--no-build-isolation`` compile in every user's cache.

Requirements pinning
---------------------
``qwen-tts==0.1.1`` is pinned exactly (the only version on PyPI at the time this
backend was written; bump intentionally as Alibaba publishes new releases, same
convention as ``qwen-asr==0.0.6`` in ``speech_to_text/qwen.py``). ``torch``/
``torchaudio`` carry a **floor**, not an exact pin: ``unasdiff``'s
``torch==2.6.0`` has no ``cu128`` wheel and failed outright on an H100 this session,
which is exactly the failure a floor avoids — ``ensure_venv``'s CUDA-aware routing
picks whichever compatible wheel actually exists on the host's index. Both are named
explicitly (rather than left to ``qwen-tts``'s own unpinned ``torchaudio`` /
transformers-transitive ``torch`` requirement) so Stage 1 of ``ensure_venv`` routes
them through the matched CUDA index — see ``subprocess_venv.py``'s
``_torch_install_specs`` docstring for why an implicit transitive pull is not enough.

A partial-pin gap in the third-party wrapper (documented, not patched)
-------------------------------------------------------------------------
``Qwen3TTSForConditionalGeneration.from_pretrained`` (the model class ``Qwen3TTSModel``
wraps) forwards its ``revision`` parameter correctly for the model weights load *and*
for ``download_weights_from_hf_specific(..., allow_patterns=["speech_tokenizer/*"],
revision=download_revision)`` — confirmed by reading the installed wheel's
``modeling_qwen3_tts.py``. But its two later reads of small config files
(``speech_tokenizer/config.json``, ``generation_config.json``) call
``cached_file(..., revision=kwargs.pop("revision", None))`` — always ``None``, because
``revision`` is consumed by the method's own named parameter and never lands in
``kwargs``. Under ``HF_HUB_OFFLINE`` that resolves against ``refs/main``. This backend
therefore stages via the *ref* the caller declared (``model.revision``, default
``"main"``) through :func:`~senselab.utils.dependencies.hf_subprocess_env` — which
writes ``refs/<ref>`` at the resolved commit — while still sending the **resolved
commit SHA** to the worker for the actual ``from_pretrained(..., revision=...)`` call.
Passing the SHA to ``hf_subprocess_env`` instead (the pattern ``speech_to_text/qwen.py``
and ``speaker_diarization/moss.py`` use) would skip writing that pointer entirely —
``_point_ref_at`` no-ops once its ``ref`` argument is already a SHA — and leave those
two unpinned reads with nothing to resolve offline. This has not been exercised
against a real download; it is a design mitigation for a gap read from source, not a
measured fix. See this task's report for what remains to verify on a GPU host.

Not wired into ``audio_analysis``
-----------------------------------
Reachable only by naming ``Qwen/Qwen3-TTS...`` explicitly through
:func:`senselab.audio.tasks.text_to_speech.api.synthesize_texts` — not a default
model and not part of any pipeline. Its first consumer is the speaker-ceiling probe,
which calls it directly.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional, Union

from senselab.audio.data_structures import Audio
from senselab.utils.data_structures import DeviceType, HFModel, _select_device_and_dtype
from senselab.utils.data_structures.logging import logger
from senselab.utils.dependencies import hf_subprocess_env
from senselab.utils.subprocess_venv import _clean_subprocess_env, ensure_venv, parse_subprocess_result, venv_python

_QWEN_TTS_VENV = "qwen-tts"
_QWEN_TTS_PYTHON = "3.12"

# torch / torchaudio: floor only, no ceiling -- see module docstring's "Requirements
# pinning" section for the H100 failure this avoids repeating. qwen-tts pulls
# torchaudio in unpinned and torch only transitively (via transformers/accelerate);
# both are named explicitly so ensure_venv's CUDA-aware Stage 1 routes them through
# the matched PyTorch wheel index instead of letting Stage 2 resolve them from
# default PyPI, where they could split across mismatched +cu local-version tags.
_QWEN_TTS_REQUIREMENTS = [
    "qwen-tts==0.1.1",
    "torch>=2.6",
    "torchaudio>=2.6",
]

_QWEN_TTS_DEFAULT_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"

# Worker script -- runs inside the isolated venv.
_WORKER_SCRIPT = r"""
import json
import sys

try:
    import numpy as np
    import soundfile as sf
    import torch
    from qwen_tts import Qwen3TTSModel

    args = json.loads(sys.stdin.read())
    texts = args["texts"]
    model_name = args["model_name"]
    model_revision = args["model_revision"]
    language = args.get("language", "Auto")
    speaker = args.get("speaker")
    instruct = args.get("instruct")
    device = args["device"]
    out_paths = args["out_paths"]

    # dtype matches the model card verbatim. device_map is set only for CUDA --
    # from_pretrained's own default (no device_map) places the model on CPU, and
    # invoking accelerate's dispatch machinery for a CPU-only run buys nothing.
    load_kwargs = {"dtype": torch.bfloat16, "revision": model_revision}
    if device == "cuda" and torch.cuda.is_available():
        load_kwargs["device_map"] = "cuda:0"

    model = Qwen3TTSModel.from_pretrained(model_name, **load_kwargs)

    if speaker is None:
        # generate_custom_voice has no default speaker; forwarding None reaches
        # model.generate(..., speakers=[None]) and fails inside the talker rather than
        # here. Default to the checkpoint's first named speaker (get_supported_speakers()
        # returns them sorted) so a bare call still produces speech -- a caller building a
        # multi-speaker session (e.g. the speaker-ceiling probe) is expected to pass
        # explicit speakers per identity instead of relying on this fallback.
        supported = model.get_supported_speakers() or []
        if not supported:
            raise RuntimeError(
                f"{model_name}@{model_revision} exposes no named speakers via get_supported_speakers()"
            )
        speaker = sorted(supported)[0]

    wavs, sr = model.generate_custom_voice(
        text=texts,
        language=language,
        speaker=speaker,
        instruct=instruct,
    )

    for wav, out_path in zip(wavs, out_paths):
        sf.write(out_path, np.asarray(wav, dtype="float32"), sr)

    print(json.dumps({"output_paths": out_paths, "sample_rate": int(sr)}))
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


def synthesize_texts_with_qwen(
    texts: List[str],
    model: Optional[HFModel] = None,
    language: Union[str, List[str]] = "Auto",
    speaker: Optional[Union[str, List[str]]] = None,
    instruct: Optional[Union[str, List[str]]] = None,
    device: Optional[DeviceType] = None,
) -> List[Audio]:
    """Synthesize speech for each text with Qwen3-TTS's CustomVoice generation path.

    ``language``/``speaker``/``instruct`` accept either a single value (applied to every
    text) or a list matched 1:1 against ``texts`` -- forwarded as-is to
    ``Qwen3TTSModel.generate_custom_voice``, which does that expansion itself.

    Args:
        texts: Texts to synthesize.
        model: HF model id (default: ``Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice``). Must
            resolve to a CustomVoice checkpoint -- ``generate_custom_voice`` raises if
            the loaded checkpoint's ``tts_model_type`` is not ``"custom_voice"``.
        language: Language name(s) recognized by the checkpoint (e.g. ``"English"``,
            ``"Auto"``). See :func:`supported_speakers`'s sibling on the model for the
            full list at runtime; validated by the worker, not here.
        speaker: Named speaker id(s) (case-insensitive), e.g. ``"Ryan"``. ``None``
            defaults to the checkpoint's first supported speaker (see the worker
            script's comment) -- pass an explicit speaker per call when identity
            matters, as the speaker-ceiling probe does.
        instruct: Optional natural-language style instruction(s), e.g. ``"Very happy."``.
        device: CPU or CUDA. CUDA strongly recommended; the checkpoint is 1.7B params.

    Returns:
        One synthesized ``Audio`` per input text, in order, at the checkpoint's native
        output sample rate.

    Raises:
        RuntimeError: if the worker fails; the upstream traceback is included.
    """
    if not texts:
        return []

    if model is None:
        model = HFModel(path_or_uri=_QWEN_TTS_DEFAULT_MODEL)
    model_name = str(model.path_or_uri)
    device_type = device or _select_device_and_dtype(compatible_devices=[DeviceType.CUDA, DeviceType.CPU])[0]

    # Resolve to the immutable commit this run pins to -- never forward model.revision
    # (a mutable ref) into the worker payload. Deferred import (not at module top) keeps
    # this monkeypatch-friendly at senselab.utils.model_revision.resolve_revision,
    # matching the rest of the codebase.
    from senselab.utils.model_revision import resolve_revision

    revision = model.commit_sha or resolve_revision(model_name, model.revision or "main")

    # Stage via the declared ref, not the resolved SHA -- see the module docstring's
    # "partial-pin gap" section for why: this writes refs/<ref> at the resolved commit,
    # which the wrapper's own unpinned cached_file() reads need to resolve offline. The
    # worker payload below still carries the resolved SHA for the actual
    # from_pretrained(..., revision=...) call, so the pin itself is unaffected. Staging
    # before ensure_venv (mirroring driftse.py's resolve_model-before-ensure_venv order)
    # is what makes this observable in a test without building a real subprocess venv.
    env = hf_subprocess_env(model_name, model.revision or "main", base_env=_clean_subprocess_env())

    venv_dir = ensure_venv(_QWEN_TTS_VENV, _QWEN_TTS_REQUIREMENTS, python_version=_QWEN_TTS_PYTHON)
    python = venv_python(venv_dir)

    logger.info(
        "Qwen3-TTS: synthesizing %d text(s) with speaker=%r, language=%r, device=%s",
        len(texts),
        speaker,
        language,
        device_type.value,
    )

    with tempfile.TemporaryDirectory(prefix="senselab-qwen-tts-") as tmpdir:
        tmp = Path(tmpdir)
        out_paths = [str(tmp / f"out_{i}.wav") for i in range(len(texts))]

        input_json = json.dumps(
            {
                "texts": texts,
                "model_name": model_name,
                "model_revision": revision,
                "language": language,
                "speaker": speaker,
                "instruct": instruct,
                "device": device_type.value,
                "out_paths": out_paths,
            }
        )

        result = subprocess.run(
            [python, "-c", _WORKER_SCRIPT],
            input=input_json,
            capture_output=True,
            text=True,
            timeout=1800,  # 1.7B model load + batch decode; generous for a cold cache.
            env=env,
        )
        output = parse_subprocess_result(result, venv_label="Qwen3-TTS")

        # Read outputs back while the temp dir is still alive: Audio(filepath=...)
        # lazy-loads on first .waveform access, and that access must happen before this
        # context manager deletes the files it points at.
        audios: List[Audio] = []
        for out_path in output["output_paths"]:
            audio = Audio(filepath=out_path)
            _ = audio.waveform
            audios.append(audio)

    return audios


def supported_speakers(model: Optional[HFModel] = None) -> List[str]:
    """Return this checkpoint's named speaker ids, sorted.

    Reads ``config.json``'s ``talker_config.spk_id`` mapping directly via
    ``huggingface_hub`` -- the same field
    ``Qwen3TTSForConditionalGeneration.__init__`` reads to build its own
    ``get_supported_speakers()`` (verified against the installed ``qwen-tts`` 0.1.1
    wheel's ``modeling_qwen3_tts.py``). This is deliberately **not** routed through the
    subprocess venv: building the venv and loading 1.7B parameters just to ask the
    model its own config would be one-thousand-times the cost of a single small-file
    download. This is what the speaker-ceiling probe uses to enumerate identities
    up front, before generating any audio.

    Args:
        model: HF model id (default: ``Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice``).

    Returns:
        Sorted list of speaker ids (lowercase, matching the checkpoint's own
        ``get_supported_speakers()`` casing).

    Raises:
        ValueError: if the resolved checkpoint's ``config.json`` carries no
            ``talker_config.spk_id`` mapping (e.g. a Base/VoiceDesign checkpoint,
            which has no fixed named-speaker set).
    """
    if model is None:
        model = HFModel(path_or_uri=_QWEN_TTS_DEFAULT_MODEL)
    model_name = str(model.path_or_uri)

    from senselab.utils.model_revision import resolve_revision

    revision = model.commit_sha or resolve_revision(model_name, model.revision or "main")

    from huggingface_hub import hf_hub_download

    config_path = hf_hub_download(model_name, "config.json", revision=revision)
    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)

    spk_id = (config.get("talker_config") or {}).get("spk_id")
    if not spk_id:
        raise ValueError(
            f"{model_name}@{revision} config.json has no talker_config.spk_id mapping -- "
            "is this a CustomVoice checkpoint?"
        )
    return sorted(spk_id.keys())
