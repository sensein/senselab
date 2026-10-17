"""emotion2vec+ categorical speech emotion recognition, run through FunASR in a subprocess venv.

The ``emotion2vec/emotion2vec_plus_{seed,base,large}`` checkpoints classify an utterance into nine
classes (angry, disgusted, fearful, happy, neutral, other, sad, surprised, unknown). They load only
through FunASR, which is kept out of the core environment. The parent stages the checkpoint at its
resolved commit with :func:`~senselab.utils.dependencies.resolve_model` and hands the worker the
local ``snapshots/<sha>/`` directory; the worker passes ``disable_update=True`` and makes no network
call.

Licence: the weights are under the FunASR Model Open Source License Agreement v1.1 (free use,
modification and sharing; attribution and the model name must be retained). FunASR itself is MIT.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from senselab.audio.data_structures import Audio, AudioClassificationResult
from senselab.audio.tasks.preprocessing import downmix_audios_to_mono, resample_audios
from senselab.utils.data_structures import DeviceType, HFModel, _select_device_and_dtype, logger
from senselab.utils.dependencies import resolve_model
from senselab.utils.subprocess_venv import (
    _clean_subprocess_env,
    ensure_venv,
    parse_subprocess_result,
    stage_portable_audio_io,
    venv_python,
)

EMOTION2VEC_PREFIX = "emotion2vec/"
"""Hugging Face organisation whose repositories this backend serves."""

EMOTION2VEC_CLASSIFIERS = (
    "emotion2vec/emotion2vec_plus_seed",
    "emotion2vec/emotion2vec_plus_base",
    "emotion2vec/emotion2vec_plus_large",
)
"""Repositories with a nine-class head."""

EMOTION2VEC_LABELS = ("angry", "disgusted", "fearful", "happy", "neutral", "other", "sad", "surprised", "unknown")
"""The classes, in the checkpoints' ``tokens.txt`` order."""

SAMPLE_RATE = 16000
"""Sampling rate the models read."""

_VENV = "emotion2vec"
_REQUIREMENTS = [
    "funasr>=1.2.6,<1.3",
    "torch>=2.6,<2.9",
    "torchaudio>=2.6,<2.9",
    "numpy<2.3",
    "soundfile",
]
_PYTHON = "3.11"

WORKER_STARTUP_S = 300.0
"""Seconds allowed for the worker before it has audio to read: interpreter, FunASR, checkpoint."""

WORKER_SECONDS_PER_AUDIO_SECOND = 2.0
"""Seconds allowed per second of audio, over the startup allowance."""

_WORKER = r"""
import json
import sys
import time

import numpy as np

args = json.loads(sys.stdin.read())
sys.path.insert(0, args["io_dir"])
from portable_audio_io import read_audio

import torch
from funasr import AutoModel

torch.set_num_threads(int(args["threads"]))
started = time.perf_counter()
model = AutoModel(model=args["model_dir"], device=args["device"], disable_update=True, log_level="ERROR")
loaded = time.perf_counter()
param_device = str(next(model.model.parameters()).device)

results = []
for path in args["audio_paths"]:
    data, sr = read_audio(path)
    data = np.ascontiguousarray(np.asarray(data, dtype=np.float32).reshape(-1))
    out = model.generate(data, granularity="utterance", extract_embedding=False, disable_pbar=True)[0]
    labels = [str(label).split("/")[-1].replace("<unk>", "unknown") for label in out["labels"]]
    scores = [float(score) for score in np.asarray(out["scores"]).reshape(-1)]
    results.append({"labels": labels, "scores": scores})

if args["device"] == "mps":
    torch.mps.synchronize()
timing = {"load_s": loaded - started, "infer_s": time.perf_counter() - loaded, "device": param_device}
print(json.dumps({"results": results, "timing": timing}))
"""


def is_emotion2vec(model: HFModel) -> bool:
    """Whether ``model`` names an emotion2vec repository.

    Args:
        model: The model.

    Returns:
        True for any ``emotion2vec/...`` repository.
    """
    return str(model.path_or_uri).startswith(EMOTION2VEC_PREFIX)


def ensure_emotion2vec_venv() -> Path:
    """Build the FunASR venv if this host has not built it, and return its directory.

    Returns:
        The venv's directory.
    """
    return ensure_venv(_VENV, _REQUIREMENTS, python_version=_PYTHON)


def _worker_timeout_s(audio_seconds: float) -> float:
    return WORKER_STARTUP_S + WORKER_SECONDS_PER_AUDIO_SECOND * max(0.0, audio_seconds)


def emotion2vec_checkpoint(model: HFModel) -> Tuple[str, Path]:
    """Resolve ``model`` to a commit and stage that commit's snapshot.

    Args:
        model: An emotion2vec+ classifier (one of :data:`EMOTION2VEC_CLASSIFIERS`).

    Returns:
        ``(commit_sha, snapshot_dir)``.

    Raises:
        ValueError: If ``model`` is an emotion2vec repository without a classification head.
    """
    repo = str(model.path_or_uri)
    if repo not in EMOTION2VEC_CLASSIFIERS:
        raise ValueError(f"{repo} has no emotion classification head; use one of {', '.join(EMOTION2VEC_CLASSIFIERS)}.")
    return resolve_model(repo, model.commit_sha or model.revision or "main")


def classify_emotions_with_emotion2vec(
    audios: List[Audio],
    model: HFModel,
    device: Optional[DeviceType] = None,
    threads: int = 3,
) -> List[AudioClassificationResult]:
    """Classify each audio, as one utterance, with an emotion2vec+ model.

    Args:
        audios: Audios of any channel count and sampling rate; downmixed to mono and resampled to
            :data:`SAMPLE_RATE`.
        model: One of :data:`EMOTION2VEC_CLASSIFIERS`.
        device: CUDA, MPS or CPU; the fastest available when omitted.
        threads: Torch intra-op threads in the worker.

    Returns:
        One result per audio with the nine :data:`EMOTION2VEC_LABELS` and their posteriors.

    Raises:
        ValueError: If ``model`` has no classification head.
    """
    if not audios:
        return []
    _, snapshot = emotion2vec_checkpoint(model)
    device_type, _ = _select_device_and_dtype(
        user_preference=device, compatible_devices=[DeviceType.CUDA, DeviceType.MPS, DeviceType.CPU]
    )
    prepared = []
    for audio in audios:
        if audio.waveform.shape[0] > 1:
            audio = downmix_audios_to_mono([audio])[0]
        if audio.sampling_rate != SAMPLE_RATE:
            audio = resample_audios([audio], SAMPLE_RATE)[0]
        prepared.append(audio)
    python = venv_python(ensure_emotion2vec_venv())
    seconds = sum(a.waveform.shape[-1] / a.sampling_rate for a in prepared)

    with tempfile.TemporaryDirectory(prefix="senselab-emotion2vec-") as tmpdir:
        tmp = Path(tmpdir)
        paths = []
        for i, audio in enumerate(prepared):
            path = tmp / f"audio_{i}.wav"
            audio.save_to_file(str(path))
            paths.append(str(path))
        payload: Dict[str, object] = {
            "audio_paths": paths,
            "model_dir": str(snapshot),
            "device": device_type.value,
            "threads": int(threads),
            "io_dir": stage_portable_audio_io(tmp),
        }
        env = _clean_subprocess_env()
        env.setdefault("OMP_NUM_THREADS", str(threads))
        result = subprocess.run(
            [python, "-c", _WORKER],
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            timeout=_worker_timeout_s(seconds),
            env=env,
        )
    output = parse_subprocess_result(result, "emotion2vec")
    timing = output.get("timing", {})
    logger.info(
        "emotion2vec timing: load_s=%.3f infer_s=%.3f audio_s=%.3f device=%s",
        timing.get("load_s", float("nan")),
        timing.get("infer_s", float("nan")),
        seconds,
        timing.get("device"),
    )
    return [AudioClassificationResult(labels=r["labels"], scores=r["scores"]) for r in output.get("results", [])]
