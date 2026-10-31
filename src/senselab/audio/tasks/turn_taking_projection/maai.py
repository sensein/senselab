"""Voice Activity Projection through MaAI (``maai-kyoto/MaAI``), in an isolated venv.

MaAI wraps the stereo VAP model (Ekstedt & Skantze, 2022) and its backchannel fine-tune (Inoue et
al., NAACL 2025). Both read one speaker per channel at 16 kHz and emit one frame per
``1 / frame_rate`` seconds:

- ``vap``: ``p_now`` and ``p_future`` per speaker (normalised so the two speakers sum to 1; the
  0-600 ms and 600-2000 ms projection windows), and a per-channel voice activity probability.
- ``bc``: ``p_bc``, the probability that channel 2's speaker backchannels now while channel 1
  speaks. Run once per listener to get both speakers' backchannel opportunities.

The worker drives MaAI's own streaming step (``Maai.process``) over the whole recording, so the
frames are the ones MaAI's real-time path would emit, with the encoder state and the 20 s attention
context carried across every step. Audio is read in blocks, so memory does not grow with duration.

Checkpoints are fetched by the parent at the pinned commits below and handed to the worker as local
files, as is the CPC encoder checkpoint, verified against its SHA-256. The worker makes no Hub call.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from senselab_core.runtime.locks import SharedFileLock
from senselab_core.runtime.models import WEIGHTS, WeightsMismatch, verify_weights

from senselab.audio.data_structures import Audio
from senselab.utils.data_structures import DeviceType
from senselab.utils.subprocess_venv import (
    _cache_dir,
    _clean_subprocess_env,
    ensure_venv,
    parse_subprocess_result,
    stage_portable_audio_io,
    venv_python,
)

SAMPLING_RATE = 16000
FRAME_RATE = 10.0
CONTEXT_SECONDS = 20


@dataclass(frozen=True)
class MaaiCheckpoint:
    """One MaAI checkpoint file at a pinned commit.

    Attributes:
        repo_id: Hugging Face repository.
        revision: The 40-hex commit the file is read at.
        filename: The state-dict file in that repository.
        mode: MaAI's ``mode`` for this checkpoint (``"vap"`` or ``"bc"``).
        licence: The repository's declared licence.
    """

    repo_id: str
    revision: str
    filename: str
    mode: str
    licence: str


VAP_CHECKPOINTS: Dict[str, MaaiCheckpoint] = {
    "vap_en_kyoto_mc": MaaiCheckpoint(
        "maai-kyoto/vap_en_kyoto",
        "8450d1d8bba95f1e240a82667b29cbc105b4ef78",
        "vap_mc_state_dict_en_kyoto_10hz_20000msec.pt",
        "vap",
        "MIT",
    ),
    "vap_en_kyoto": MaaiCheckpoint(
        "maai-kyoto/vap_en_kyoto",
        "8450d1d8bba95f1e240a82667b29cbc105b4ef78",
        "vap_state_dict_en_kyoto_10hz_20000msec.pt",
        "vap",
        "MIT",
    ),
}
"""Turn-taking checkpoints, by name. ``_mc`` is the multi-condition (noise-augmented) training."""

BC_CHECKPOINTS: Dict[str, MaaiCheckpoint] = {
    "vap_bc_en": MaaiCheckpoint(
        "maai-kyoto/vap_bc_en",
        "3ff203ce14de279045eb1b145a1dd24caa8f1c9d",
        "vap-bc_state_dict_en_10hz_20000msec.pt",
        "bc",
        "MIT",
    ),
}
"""Backchannel checkpoints, by name."""

DEFAULT_VAP_CHECKPOINT = "vap_en_kyoto_mc"
DEFAULT_BC_CHECKPOINT = "vap_bc_en"

CPC_URL = "https://dl.fbaipublicfiles.com/librilight/CPC_checkpoints/60k_epoch4-d0f474de.pt"
"""The Libri-light CPC encoder MaAI's ``normal`` models are built on; its file digest is ``WEIGHTS["maai-cpc"]``."""

_MAAI_VENV = "maai"
_MAAI_REQUIREMENTS = ["maai==0.2.24", "torch>=2.6", "numpy", "soundfile"]
_MAAI_PYTHON = "3.12"
_MAAI_EXCLUDED = [
    "pyaudio",
    "pyaudiowpatch",
    "pygame-ce",
    "pyqt5",
    "pyqt5-qt5",
    "pyqt5-sip",
    "pyqtgraph",
    "onnxruntime",
    "onnxruntime-gpu",
]
"""Declared by ``maai`` for microphones, GUIs and the Mimi encoder; the offline CPC path imports none."""

_READ_BLOCK_SECONDS = 60

_MAAI_WORKER = r"""
import json
import os
import sys
import time
import types

_replies = sys.stdout
sys.stdout = sys.stderr

try:
    args = json.loads(sys.stdin.read())
    for name in ("pyaudio", "pygame"):
        sys.modules.setdefault(name, types.ModuleType(name))
    import numpy as np
    import torch

    torch.set_num_threads(int(args["threads"]))
    sys.path.insert(0, args["io_dir"])
    from portable_audio_io import audio_info, read_audio
    from maai.input import Base
    from maai.model import Maai

    began = time.monotonic()
    model = Maai(
        mode=args["mode"],
        lang="en",
        audio_ch1=Base(),
        audio_ch2=Base(),
        frame_rate=args["frame_rate"],
        context_len_sec=args["context_len_sec"],
        device=args["device"],
        cpc_model=args["cpc"],
        local_model=args["checkpoint"],
    )
    load_s = time.monotonic() - began

    hop = int(round(16000 / args["frame_rate"]))
    total = min(audio_info(args["wav_1"]).frames, audio_info(args["wav_2"]).frames)
    n_frames = total // hop
    block = max(1, int(args["block_seconds"] * args["frame_rate"])) * hop
    rows = []

    def take(result):
        if args["mode"] == "vap":
            rows.append([*result["p_now"], *result["p_future"], *result["vad"]])
        else:
            rows.append([result["p_bc"]])

    began = time.monotonic()
    for start in range(0, n_frames * hop, block):
        stop = min(start + block, n_frames * hop)
        x1, _ = read_audio(args["wav_1"], start=start, stop=stop)
        x2, _ = read_audio(args["wav_2"], start=start, stop=stop)
        x1 = np.ascontiguousarray(x1.reshape(-1), dtype=np.float32)
        x2 = np.ascontiguousarray(x2.reshape(-1), dtype=np.float32)
        for i in range(0, stop - start, hop):
            model.process(x1[i:i + hop], x2[i:i + hop])
            while not model.result_dict_queue.empty():
                take(model.result_dict_queue.get())
    run_s = time.monotonic() - began

    array = np.asarray(rows, dtype=np.float32).reshape(len(rows), -1)
    np.save(args["out"], array)
    _replies.write(json.dumps({"frames": int(array.shape[0]), "expected": int(n_frames), "load_s": load_s,
                               "run_s": run_s}) + "\n")
except Exception as exc:
    import traceback

    _replies.write(json.dumps({"error": {"type": type(exc).__name__, "message": str(exc),
                                         "traceback": traceback.format_exc(limit=5)}}) + "\n")
    sys.exit(1)
"""

VAP_COLUMNS = ("p_now_a", "p_now_b", "p_future_a", "p_future_b", "vad_a", "vad_b")
"""Column order of a ``vap`` job's output: channel 1 is speaker ``a``, channel 2 speaker ``b``."""


@dataclass(frozen=True)
class MaaiJob:
    """One worker run: a checkpoint over a channel pair.

    Attributes:
        name: The job's label in the result (``"vap"``, ``"bc_a"``, ``"bc_b"``).
        checkpoint: The checkpoint to run.
        swap: Feed channel ``b`` as MaAI's channel 1 and ``a`` as channel 2.
    """

    name: str
    checkpoint: MaaiCheckpoint
    swap: bool = False


def plan_jobs(vap_checkpoint: str, bc_checkpoint: Optional[str]) -> List[MaaiJob]:
    """The worker runs one projection needs.

    A ``bc`` checkpoint predicts channel 2's backchannels while channel 1 speaks, so each listener
    is its own run: ``bc_b`` feeds ``(a, b)``, ``bc_a`` feeds ``(b, a)``.

    Args:
        vap_checkpoint: A key of :data:`VAP_CHECKPOINTS`.
        bc_checkpoint: A key of :data:`BC_CHECKPOINTS`, or ``None`` to skip backchannel prediction.

    Returns:
        The jobs, ``vap`` first.

    Raises:
        ValueError: If a checkpoint name is unknown.
    """
    if vap_checkpoint not in VAP_CHECKPOINTS:
        raise ValueError(f"unknown VAP checkpoint {vap_checkpoint!r}; known: {sorted(VAP_CHECKPOINTS)}")
    jobs = [MaaiJob("vap", VAP_CHECKPOINTS[vap_checkpoint])]
    if bc_checkpoint is not None:
        if bc_checkpoint not in BC_CHECKPOINTS:
            raise ValueError(f"unknown backchannel checkpoint {bc_checkpoint!r}; known: {sorted(BC_CHECKPOINTS)}")
        checkpoint = BC_CHECKPOINTS[bc_checkpoint]
        jobs += [MaaiJob("bc_b", checkpoint, swap=False), MaaiJob("bc_a", checkpoint, swap=True)]
    return jobs


def stage_checkpoint(checkpoint: MaaiCheckpoint) -> str:
    """Download one checkpoint file at its pinned commit and return its local path."""
    from huggingface_hub import hf_hub_download

    return hf_hub_download(checkpoint.repo_id, checkpoint.filename, revision=checkpoint.revision)


def stage_cpc(cache_dir: Optional[Path] = None) -> str:
    """Return a local copy of the CPC encoder checkpoint, downloading it once and checking its file SHA-256.

    Args:
        cache_dir: Where to keep it; defaults to senselab's cache.

    Returns:
        The file's path.

    Raises:
        WeightsMismatch: If the file's SHA-256 is not the manifest's ``maai-cpc`` entry (a ``RuntimeError``).
    """
    directory = cache_dir or (_cache_dir().parent / "maai")
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"cpc-{WEIGHTS['maai-cpc'][:16]}.pt"
    with SharedFileLock(path):
        if not _verified(path):
            partial = path.with_name(path.name + ".partial")
            urllib.request.urlretrieve(CPC_URL, partial)  # noqa: S310 -- a fixed https URL
            try:
                verify_weights("maai-cpc", partial)
            except WeightsMismatch:
                partial.unlink(missing_ok=True)
                raise
            partial.replace(path)
    return str(path)


def _verified(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        verify_weights("maai-cpc", path)
    except WeightsMismatch:
        return False
    return True


def run_maai(
    channel_a: Audio,
    channel_b: Audio,
    jobs: List[MaaiJob],
    device: DeviceType = DeviceType.CPU,
    num_workers: int = 3,
    timeout_s: Optional[float] = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, float]]]:
    """Run each job's worker over the two channels.

    Args:
        channel_a: Speaker ``a``, mono 16 kHz.
        channel_b: Speaker ``b``, mono 16 kHz, same length as ``channel_a``.
        jobs: From :func:`plan_jobs`.
        device: CPU or CUDA.
        num_workers: Jobs run concurrently, one torch thread each.
        timeout_s: Ceiling per job; ``None`` allows one second per second of audio plus ten minutes.

    Returns:
        ``(arrays, timing)``: each job's ``(frames, columns)`` array by job name, and its load and
        run seconds.

    Raises:
        RuntimeError: If a worker fails or returns fewer frames than the audio holds.
    """
    duration = channel_a.waveform.shape[-1] / SAMPLING_RATE
    timeout = timeout_s if timeout_s is not None else duration + 600.0
    venv_dir = ensure_venv(
        _MAAI_VENV, _MAAI_REQUIREMENTS, python_version=_MAAI_PYTHON, excluded_packages=_MAAI_EXCLUDED
    )
    python = venv_python(venv_dir)
    cpc = stage_cpc()
    staged = {job.checkpoint: stage_checkpoint(job.checkpoint) for job in jobs}

    with tempfile.TemporaryDirectory(prefix="senselab-maai-") as tmpdir:
        tmp = Path(tmpdir)
        io_dir = stage_portable_audio_io(tmp)
        wav_a, wav_b = str(tmp / "a.wav"), str(tmp / "b.wav")
        channel_a.save_to_file(wav_a)
        channel_b.save_to_file(wav_b)
        env = _clean_subprocess_env()
        env.update({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
        if device is DeviceType.CPU:
            env["CUDA_VISIBLE_DEVICES"] = ""

        def run(job: MaaiJob) -> Tuple[str, np.ndarray, Dict[str, float]]:
            out = str(tmp / f"{job.name}.npy")
            payload = {
                "io_dir": io_dir,
                "wav_1": wav_b if job.swap else wav_a,
                "wav_2": wav_a if job.swap else wav_b,
                "mode": job.checkpoint.mode,
                "checkpoint": staged[job.checkpoint],
                "cpc": cpc,
                "frame_rate": FRAME_RATE,
                "context_len_sec": CONTEXT_SECONDS,
                "device": device.value,
                "threads": 1,
                "block_seconds": _READ_BLOCK_SECONDS,
                "out": out,
            }
            result = subprocess.run(  # noqa: S603 -- the interpreter is senselab's own venv
                [python, "-c", _MAAI_WORKER],
                input=json.dumps(payload),
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env,
            )
            reply = parse_subprocess_result(result, f"MaAI {job.name}")
            if reply["frames"] != reply["expected"]:
                raise RuntimeError(f"MaAI {job.name} returned {reply['frames']} frames for {reply['expected']}")
            return job.name, np.load(out), {"load_s": float(reply["load_s"]), "run_s": float(reply["run_s"])}

        with ThreadPoolExecutor(max_workers=max(1, min(num_workers, len(jobs), os.cpu_count() or 1))) as pool:
            outcomes = list(pool.map(run, jobs))

    arrays = {name: array for name, array, _ in outcomes}
    timing = {name: t for name, _, t in outcomes}
    return arrays, timing
