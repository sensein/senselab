"""``pyannote/brouhaha`` — joint per-frame VAD + SNR + C50 via an isolated venv.

Brouhaha (Lavechin et al., 2022, arXiv:2210.13248) predicts, per frame and in a
single forward pass:

- **VAD** — speech-speech_presence probability in ``[0, 1]``;
- **SNR** — estimated signal-to-noise ratio in dB;
- **C50** — room-acoustics clarity in dB (higher = less reverberant).

The scene-quality workflow uses the SNR/C50 heads for the ``quality_snr`` /
``quality_reverb`` degradation scores and the VAD head as a second frame-level
speech-speech_presence voter.

**Why a subprocess venv.** The ``pyannote/brouhaha`` checkpoint is not loadable
by our main environment: its custom multitask model class lives in the
GitHub-only ``brouhaha`` package (``brouhaha-vad``), which pins
``pyannote.audio>=3.1,<3.3.1``, ``speechbrain<1.0`` and ``numpy<2.0`` — all
incompatible with senselab's ``pyannote-audio>=4.0`` / ``speechbrain>=1.0``.
So Brouhaha runs in a dedicated venv (same pattern as the NeMo / Qwen ASR
backends), isolated from the main install. The model is gated on HuggingFace;
the worker reads ``HF_TOKEN`` from the environment (preserved across the
subprocess boundary).

Inference uses upstream's ``BrouhahaInference``, which slides the model's trained 6 s window and
overlap-adds — the path brouhaha's authors ship for this model. An earlier revision here forced
``window="whole"``, which pyannote warns against for frame-based models and which measured worse
on every axis: 5.5 dB of SNR error at a true 10 dB against sliding's 0.33 dB, memory growing
linearly with duration rather than staying bounded, and the authors' own ``slide()`` override
never executing. It also did not achieve its stated purpose — brouhaha's VAD head is
high-recall and reads near 1.0 through short pauses regardless of windowing.

Frames arrive as one continuous timelineand stitched back into one continuous per-frame timeline via
``stitch_frames`` — flat memory, native ~17 ms resolution (shared with the
segmentation-3.0 extractor).
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.voice_activity_detection.frame_posteriors import stitch_frames
from senselab.utils.data_structures import DeviceType, _select_device_and_dtype
from senselab.utils.data_structures.logging import logger
from senselab.utils.data_structures.model import get_huggingface_token
from senselab.utils.dependencies import hf_subprocess_env
from senselab.utils.subprocess_venv import _clean_subprocess_env, ensure_venv, parse_subprocess_result, venv_python

BROUHAHA_MODEL_ID = "pyannote/brouhaha"
BROUHAHA_REVISION = "main"

# Dedicated venv for the old brouhaha-vad stack. torch/torchaudio are routed
# through the CUDA-aware PyTorch index by ``ensure_venv`` (see CLAUDE.md); the
# pins target the pyannote-audio 3.x era brouhaha requires. numpy<2.0 and
# speechbrain<1.0 come transitively via brouhaha-vad's own requirements.
_BROUHAHA_VENV = "brouhaha"
_BROUHAHA_REQUIREMENTS = [
    "brouhaha @ git+https://github.com/marianne-m/brouhaha-vad.git",
    "torch>=2.0,<2.3",
    "torchaudio>=2.0,<2.3",
    "numpy<2.0",
    # brouhaha-vad pins pyannote 3.x but not huggingface_hub; the latest hub
    # dropped the `use_auth_token` kwarg pyannote 3.x still passes to
    # hf_hub_download. Pin to the era that keeps it.
    "huggingface_hub>=0.19,<0.24",
    "soundfile",
]
_BROUHAHA_PYTHON = "3.11"
# The ``torch>=2.0,<2.3`` pin above has no linux-x86_64 wheel on PyTorch's newer
# CUDA indexes (cu124/cu126/cu128), so on a modern-CUDA host the default
# host-keyed index selection makes the Stage-1 install unsatisfiable and the
# venv never builds. cu121 is the newest index that still publishes a
# ``torch<2.3`` wheel (``torch==2.2.2+cu121`` + matching torchaudio), and its
# wheels are forward-compatible with 12.x/13.x drivers for inference. Cap the
# index here so brouhaha builds on every GPU host.
_BROUHAHA_MAX_CUDA_VERSION = (12, 1)

# Chunking for long recordings (mirrors the segmentation extractor's grid).

# Multitask output channel order (frames, 3): [VAD, SNR dB, C50 dB].
_VAD_CHANNEL = 0
_SNR_CHANNEL = 1
_C50_CHANNEL = 2

# Worker — runs inside the brouhaha venv. Loads the gated model (HF_TOKEN from
# env), runs whole-window inference per chunk wav, and saves each chunk's
# per-frame (VAD, SNR, C50) array as .npy for the parent to stitch.
_BROUHAHA_WORKER_SCRIPT = r"""
import json
import os
import sys

try:
    import numpy as np
    import torch
    from pyannote.audio import Inference, Model

    args = json.loads(sys.stdin.read())
    token = os.environ.get("HF_TOKEN")
    dev = "cuda" if args["device"] == "cuda" and torch.cuda.is_available() else "cpu"

    model = Model.from_pretrained(args["model_name"], use_auth_token=token)
    # Upstream's own inference class, which defaults to sliding. Three measurements say this
    # rather than window="whole": pyannote warns that "whole" on a frame-based model "might
    # lead to bad results and huge memory consumption"; at a true 10 dB SNR sliding errs by
    # 0.33 dB against whole's 5.5 dB, because the model is trained at duration=6 s and whole
    # fed it 21.5 s; and whole's memory grows linearly with duration (~494 MB above baseline
    # at 120 s) while sliding is bounded by batch_size x 6 s. BrouhahaInference also overrides
    # slide(), so under "whole" the code brouhaha's authors ship for this model never ran.
    from brouhaha.inference import BrouhahaInference

    inference = BrouhahaInference(model, device=torch.device(dev))
    try:
        hop = float(model.receptive_field.step)
    except Exception:
        hop = None

    results = []
    for ch in args["chunks"]:
        out = inference(ch["path"])
        data = out.data if hasattr(out, "sliding_window") and hasattr(out, "data") else np.asarray(out)
        data = np.asarray(data, dtype=np.float64)
        if hop is None:
            try:
                hop = float(out.sliding_window.step)
            except Exception:
                hop = 0.01
        fname = "chunk_%d_%d.npy" % (ch["audio_idx"], int(round(ch["start_s"] * 1000)))
        npy_path = os.path.join(args["out_dir"], fname)
        np.save(npy_path, data)
        results.append({"npy": npy_path, "start_s": ch["start_s"], "audio_idx": ch["audio_idx"], "hop": hop})

    print(json.dumps({"results": results}))
except Exception as exc:
    import traceback

    err = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc(limit=5)}
    print(json.dumps({"error": err}))
    sys.exit(1)
"""


@dataclass
class BrouhahaFrames:
    """Per-frame Brouhaha outputs for one audio.

    Attributes:
        vad: ``(num_frames,)`` speech-speech_presence probability in ``[0, 1]``.
        snr_db: ``(num_frames,)`` estimated SNR in dB.
        c50_db: ``(num_frames,)`` estimated C50 (clarity) in dB.
        frame_hop_s: seconds between consecutive frame starts.
    """

    vad: np.ndarray
    snr_db: np.ndarray
    c50_db: np.ndarray
    frame_hop_s: float

    def mean_in_window(self, start_s: float, end_s: float) -> tuple[float, float, float]:
        """Return ``(mean vad, mean snr_db, mean c50_db)`` over frames overlapping ``[start, end)``.

        Any component with no overlapping frames returns ``nan`` for that value.
        """
        if self.frame_hop_s <= 0 or self.vad.size == 0:
            return (float("nan"), float("nan"), float("nan"))
        lo = max(0, int(np.floor(start_s / self.frame_hop_s)))
        hi = min(self.vad.size, int(np.ceil(end_s / self.frame_hop_s)))
        if hi <= lo:
            return (float("nan"), float("nan"), float("nan"))
        return (
            float(np.nanmean(self.vad[lo:hi])),
            float(np.nanmean(self.snr_db[lo:hi])),
            float(np.nanmean(self.c50_db[lo:hi])),
        )


def _build_worker_input(
    chunks: list[dict],
    model_name: str,
    revision: str,
    device: str,
    out_dir: str,
    token: Optional[str] = None,
) -> dict:
    """Build the JSON-able payload sent to the Brouhaha worker.

    Resolves ``revision`` to its immutable commit SHA here, in the parent, rather than
    handing the worker a ref (e.g. ``"main"``) to re-resolve against its own venv's cache —
    two nodes in one run re-resolving independently can land on different commits if
    upstream moves in between. Extracted out of ``extract_brouhaha_frames`` so this step
    is unit-testable without spawning the subprocess venv.

    Raises:
        RevisionResolutionError: if ``revision`` cannot be resolved to a commit SHA
            (propagates from :func:`resolve_revision`); the caller degrades to null
            scene-quality signals rather than crash the whole run (FR-023).
    """
    # Deferred import (not at module top): keeps this module monkeypatch-friendly at
    # `senselab.utils.model_revision.resolve_revision`, matching the rest of the codebase's
    # convention for cross-module helpers (see e.g. HFModel._resolve_commit_sha).
    from senselab.utils.model_revision import resolve_revision

    resolved = resolve_revision(model_name, revision, token=token)
    return {
        "chunks": chunks,
        "model_name": model_name,
        "revision": resolved,
        "device": device,
        "out_dir": out_dir,
    }


def extract_brouhaha_frames(
    audios: list[Audio],
    device: Optional[DeviceType] = None,
    model_id: str = BROUHAHA_MODEL_ID,
    revision: str = BROUHAHA_REVISION,
) -> list[Optional[BrouhahaFrames]]:
    """Run Brouhaha (in its isolated venv) once per audio, returning per-frame VAD/SNR/C50.

    Args:
        audios: mono 16 kHz clips.
        device: inference device (CPU or CUDA).
        model_id: Brouhaha HF model id.
        revision: model revision.

    Returns:
        One ``BrouhahaFrames`` per input, or ``None`` for an audio the worker
        failed on. If the venv cannot be built or the worker fails wholesale
        (e.g. gated access not granted, install failure), every entry is
        ``None`` — the workflow then emits null quality columns rather than
        aborting (FR-023). The analyze_audio.py script treats scene quality as
        required and fails loudly on all-null instead of silently degrading.
    """
    if not audios:
        return []

    device_type, _ = _select_device_and_dtype(
        user_preference=device, compatible_devices=[DeviceType.CUDA, DeviceType.CPU]
    )

    try:
        venv_dir = ensure_venv(
            _BROUHAHA_VENV,
            _BROUHAHA_REQUIREMENTS,
            python_version=_BROUHAHA_PYTHON,
            max_cuda_version=_BROUHAHA_MAX_CUDA_VERSION,
        )
        python = venv_python(venv_dir)
    except Exception as exc:  # noqa: BLE001 — venv build failure degrades to null (FR-023)
        logger.warning(f"Failed to prepare Brouhaha venv: {exc}. Scene-quality signals will be null.")
        return [None] * len(audios)

    with tempfile.TemporaryDirectory(prefix="senselab-brouhaha-") as tmpdir:
        tmp = Path(tmpdir)
        out_dir = tmp / "out"
        out_dir.mkdir(parents=True, exist_ok=True)

        # One spec per audio: sliding inference produces a continuous per-frame timeline over
        # a file of any length at flat memory, so parent-side chunking has nothing to add.
        chunk_specs: list[dict] = []
        for ai, audio in enumerate(audios):
            path = str(tmp / f"audio_{ai}.wav")
            audio.save_to_file(path)
            chunk_specs.append({"path": path, "start_s": 0.0, "audio_idx": ai})

        token = get_huggingface_token()
        try:
            payload = _build_worker_input(chunk_specs, model_id, revision, device_type.value, str(out_dir), token)
        except Exception as exc:  # noqa: BLE001 — unresolvable revision degrades to null (FR-023), matching the
            # venv-build failure above.
            logger.warning(
                f"Failed to resolve Brouhaha revision {model_id}@{revision}: {exc}. Scene-quality signals will be null."
            )
            return [None] * len(audios)
        input_json = json.dumps(payload)
        resolved_revision = payload["revision"]

        # Stage the (gated) model once (cross-process heartbeat lock) + run the
        # worker offline so its Model.from_pretrained makes no per-call Hub version
        # check — the 429 source under parallel batch. If staging fails (no access),
        # hf_subprocess_env leaves the env online so the worker's current path still
        # runs. Staged and sent under the SAME resolved SHA -- resolving twice (once here,
        # once in _build_worker_input) risks the parent staging one commit while the worker
        # is told to load another if upstream moves between the two calls.
        env = hf_subprocess_env(str(model_id), resolved_revision, base_env=_clean_subprocess_env(), token=token)
        try:
            result = subprocess.run(
                [python, "-c", _BROUHAHA_WORKER_SCRIPT],
                input=input_json,
                capture_output=True,
                text=True,
                timeout=1800,
                env=env,
            )
            output = parse_subprocess_result(result, "Brouhaha")
        except Exception as exc:  # noqa: BLE001 — worker failure degrades to null (FR-023)
            logger.warning(f"Brouhaha worker failed: {exc}. Scene-quality signals will be null.")
            return [None] * len(audios)

        # Regroup per-chunk results by audio index, load npy, and stitch.
        by_audio: dict[int, list[dict]] = {}
        for entry in output.get("results", []):
            by_audio.setdefault(int(entry["audio_idx"]), []).append(entry)

        results: list[Optional[BrouhahaFrames]] = []
        for ai in range(len(audios)):
            entries = by_audio.get(ai)
            if not entries:
                results.append(None)
                continue
            try:
                arrays = [np.load(e["npy"]) for e in entries]
                starts = [float(e["start_s"]) for e in entries]
                hop = float(entries[0].get("hop") or 0.0)
                data = stitch_frames(arrays, starts, hop) if hop > 0 else np.zeros((0, 3))
                if data.ndim != 2 or data.shape[1] <= _C50_CHANNEL:
                    raise ValueError(f"unexpected Brouhaha output shape {data.shape}")
                results.append(
                    BrouhahaFrames(
                        vad=data[:, _VAD_CHANNEL],
                        snr_db=data[:, _SNR_CHANNEL],
                        c50_db=data[:, _C50_CHANNEL],
                        frame_hop_s=hop,
                    )
                )
            except (OSError, ValueError, KeyError) as exc:
                logger.warning(f"Failed to assemble Brouhaha frames for audio {ai}: {exc}")
                results.append(None)
        return results
