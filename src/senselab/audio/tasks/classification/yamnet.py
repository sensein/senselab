"""YAMNet audio classification via isolated subprocess venv.

YAMNet is a TensorFlow-based model that classifies audio into 521
AudioSet classes. It runs in an isolated subprocess venv to avoid
TF/PyTorch conflicts. The worker stays resident across calls within one
process; :func:`shutdown_yamnet_worker` ends it early.
"""

import atexit
import json
import queue
import subprocess
import tempfile
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.preprocessing import resample_audios
from senselab.utils.data_structures.logging import logger
from senselab.utils.subprocess_venv import _clean_subprocess_env, ensure_venv, venv_python


def write_worker_wav(path: "Path | str", waveform: Any, sampling_rate: int) -> Dict[str, Any]:  # noqa: ANN401
    """Write a lossless mono WAV for the YAMNet worker and report input-path artifacts.

    Args:
        path: Destination file.
        waveform: Samples; a leading channel dimension is averaged to mono.
        sampling_rate: Sample rate in Hz.

    Returns:
        A report with ``subtype``, ``clipped_fraction``, and ``requantized`` — surfaced
        rather than silently repaired, because a clamped or requantized input would make
        the classifier respond to distortion while provenance claimed clean audio
        (FR-017d, FR-019b).
    """
    import numpy as np
    import torch

    arr = np.asarray(waveform, dtype=np.float32)
    if arr.ndim > 1:
        arr = arr.mean(axis=0) if arr.shape[0] < arr.shape[-1] else arr.mean(axis=-1)
    # At or beyond full scale on the *input*, which is a different measurement from the write's
    # own out-of-range fraction: this one reports what arrived, not what the container lost.
    clipped = float(np.count_nonzero(np.abs(arr) >= 0.9999) / arr.size) if arr.size else 0.0
    report = Audio(waveform=torch.from_numpy(arr).unsqueeze(0), sampling_rate=sampling_rate).save_to_file(str(path))
    return {"subtype": report.subtype, "clipped_fraction": clipped, "requantized": False}


_YAMNET_VENV = "yamnet"
_YAMNET_REQUIREMENTS = [
    "tensorflow",
    "tensorflow-hub",
    "setuptools<70",  # tensorflow-hub needs pkg_resources
    "numpy",
    "soundfile",
]
_YAMNET_PYTHON = "3.12"

# A line protocol over the worker's stdin/stdout, one JSON object per line.
#
# Classify request (stdin):  {"audio_paths": [str, ...], "top_k": int}
# Classify reply (stdout):   {"results": [[{"label_scores": [{label: score}, ...]}, ...], ...]}
# Stop request (stdin):      {"stop": True}
# Ready reply (stdout):      {"ready": True, "load_s": float}
#
# Every reply carries ``_WORKER_MARKER``, and ``sys.stdout`` is redirected to stderr before any
# heavy import, so TensorFlow's own chatter can never be mistaken for an answer. A raised exception
# is reported as {"error": {"type": str, "message": str}} and ends the worker.
_WORKER_MARKER = "@@SENSELAB_YAMNET@@"

_YAMNET_WORKER = (
    r"""
import json
import sys
import time

MARKER = "%s"
_replies = sys.stdout
sys.stdout = sys.stderr


def emit(payload):
    _replies.write(MARKER + json.dumps(payload) + "\n")
    _replies.flush()


def load():
    import csv
    import os
    import pathlib
    import shutil

    import numpy as np
    import soundfile as sf
    import tensorflow_hub as hub

    # The TF-Hub cache defaults to $TMPDIR, which is wrong twice over: it is discarded between
    # reboots so the model is re-fetched, and a partially-written entry is reused forever
    # because TF-Hub only checks that the directory exists.
    _HUB_URL = "https://tfhub.dev/google/yamnet/1"
    cache_root = pathlib.Path(
        os.environ.get("SENSELAB_TFHUB_CACHE")
        or (pathlib.Path.home() / ".cache" / "senselab" / "tfhub")
    )
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ["TFHUB_CACHE_DIR"] = str(cache_root)

    try:
        model = hub.load(_HUB_URL)
    except (ValueError, OSError) as exc:
        # A corrupt entry must be a cache miss, not a permanent failure. Discard the
        # incomplete directory and fetch once more; a second failure is real.
        if "saved_model" not in str(exc):
            raise
        for stale in cache_root.iterdir():
            if stale.is_dir() and not any(stale.glob("saved_model.pb*")):
                shutil.rmtree(stale, ignore_errors=True)
        model = hub.load(_HUB_URL)

    class_map_path = model.class_map_path().numpy().decode("utf-8")
    with open(class_map_path) as f:
        reader = csv.DictReader(f)
        class_names = [row["display_name"] for row in reader]
    return sf, model, class_names


def classify(sf, model, class_names, audio_paths, top_k):
    all_results = []
    for audio_path in audio_paths:
        # Audio is already resampled to 16kHz mono by the caller
        data, sr = sf.read(audio_path, dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)

        scores, embeddings, spectrogram = model(data)
        scores_np = scores.numpy()

        # Each row in scores is a ~0.96s window
        windows = []
        for i, frame_scores in enumerate(scores_np):
            top_indices = frame_scores.argsort()[::-1][:top_k]
            windows.append({
                "label_scores": [{class_names[idx]: float(frame_scores[idx])} for idx in top_indices],
            })
        all_results.append(windows)
    return all_results


def main():
    started = time.monotonic()
    sf, model, class_names = load()
    emit({"ready": True, "load_s": round(time.monotonic() - started, 3)})
    while True:
        line = sys.stdin.readline()
        if not line:
            return
        if not line.strip():
            continue
        request = json.loads(line)
        if request.get("stop"):
            return
        emit({
            "results": classify(
                sf, model, class_names, request["audio_paths"], request.get("top_k", 5)
            )
        })


try:
    main()
except Exception as exc:
    emit({"error": {"type": type(exc).__name__, "message": str(exc)}})
    sys.exit(1)
"""
    % _WORKER_MARKER
)


_REQUEST_TIMEOUT_S = 600
"""Wall-clock ceiling on one classify request, and separately on the worker's start-up load."""


class _YAMNetWorker:
    """One venv subprocess with YAMNet loaded, answering classify requests until it is stopped.

    Attributes:
        load_s: How long starting it and loading the model took.
    """

    def __init__(self) -> None:
        """Record the worker's state. Starting it is :meth:`start`."""
        self.load_s = 0.0
        self._process: Optional["subprocess.Popen[str]"] = None
        self._replies: "queue.Queue[Dict[str, Any]]" = queue.Queue()
        self._noise: "deque[str]" = deque(maxlen=40)

    def start(self, timeout_s: int) -> None:
        """Build the venv if needed, spawn the worker and wait for its model.

        Args:
            timeout_s: Wall-clock ceiling on the load.

        Raises:
            RuntimeError: If the worker died, raised, or did not report ready in time.
        """
        venv_dir = ensure_venv(_YAMNET_VENV, _YAMNET_REQUIREMENTS, python_version=_YAMNET_PYTHON)
        began = time.monotonic()
        self._process = subprocess.Popen(  # noqa: S603 — the interpreter is this repo's own venv
            [venv_python(venv_dir), "-c", _YAMNET_WORKER],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=_clean_subprocess_env(),
        )
        threading.Thread(target=self._pump_replies, daemon=True).start()
        threading.Thread(target=self._pump_noise, daemon=True).start()
        self._await(timeout_s)
        self.load_s = round(time.monotonic() - began, 3)

    @property
    def alive(self) -> bool:
        """Whether the worker process is still running and still holding its model."""
        return self._process is not None and self._process.poll() is None

    def classify(self, audio_paths: List[str], top_k: int, timeout_s: int) -> List[List[Dict[str, Any]]]:
        """Ask the loaded model for one batch of per-window scores.

        Args:
            audio_paths: 16 kHz mono WAVs, already written by the caller.
            top_k: Number of top labels per window.
            timeout_s: Wall-clock ceiling on this batch alone.

        Returns:
            One list of windows per input path, in the same order.

        Raises:
            RuntimeError: If the worker died, raised, or did not answer in time.
        """
        self._send({"audio_paths": audio_paths, "top_k": int(top_k)})
        reply = self._await(timeout_s)
        return list(reply.get("results", []))

    def close(self) -> None:
        """End the worker, releasing the model. Safe to call on a worker that never started."""
        process, self._process = self._process, None
        if process is None:
            return
        try:
            if process.stdin is not None and not process.stdin.closed:
                process.stdin.write(json.dumps({"stop": True}) + "\n")
                process.stdin.flush()
                process.stdin.close()
            process.wait(timeout=5)
        except Exception:  # noqa: BLE001 — a worker that will not stop politely is killed
            process.kill()
            try:
                process.wait(timeout=30)
            except Exception:  # noqa: BLE001, S110 — nothing further is owed to an unreapable child
                pass

    def _send(self, payload: Dict[str, Any]) -> None:
        """Write one request line, turning a closed pipe into the failure the caller reports.

        Args:
            payload: The request.

        Raises:
            RuntimeError: If the worker is gone or its stdin will not take the line.
        """
        if self._process is None or self._process.stdin is None:
            raise RuntimeError("YAMNet worker is not running")
        try:
            self._process.stdin.write(json.dumps(payload) + "\n")
            self._process.stdin.flush()
        except (BrokenPipeError, OSError, ValueError) as exc:
            raise RuntimeError(f"YAMNet worker closed its input: {exc}\n{self._tail()}") from exc

    def _await(self, timeout_s: int) -> Dict[str, Any]:
        """Wait for one reply, killing the worker on anything that is not one.

        The exception a worker-reported error becomes is the one
        :func:`~senselab.utils.subprocess_venv.parse_subprocess_result` would have raised for the
        same payload, so the failure a caller sees does not depend on which path produced it.

        Args:
            timeout_s: How long to wait.

        Returns:
            The reply.

        Raises:
            RuntimeError: On a timeout, a dead worker, or a worker-reported exception whose type is
                not one of the two the one-shot parser reconstructs.
            ValueError: A worker-reported ``ValueError``.
            TypeError: A worker-reported ``TypeError``.
        """
        try:
            reply = self._replies.get(timeout=timeout_s)
        except queue.Empty:
            self.close()
            raise RuntimeError(f"YAMNet worker did not answer in {timeout_s}s\n{self._tail()}") from None
        if "error" in reply:
            self.close()
            error = reply["error"] or {}
            exc_class = {"ValueError": ValueError, "TypeError": TypeError}.get(str(error.get("type", "")), RuntimeError)
            raise exc_class(str(error.get("message", "unknown error")))
        if reply.get("eof"):
            self.close()
            raise RuntimeError(f"YAMNet worker exited without answering\n{self._tail()}")
        return reply

    def _pump_replies(self) -> None:
        """Drain stdout into the reply queue, forwarding anything unmarked to the noise tail."""
        process = self._process
        if process is None or process.stdout is None:
            return
        for line in process.stdout:
            if line.startswith(_WORKER_MARKER):
                try:
                    self._replies.put(json.loads(line[len(_WORKER_MARKER) :]))
                except ValueError:
                    self._noise.append(line.rstrip())
            elif line.strip():
                self._noise.append(line.rstrip())
        self._replies.put({"eof": True})

    def _pump_noise(self) -> None:
        """Drain stderr so a chatty loader cannot fill its pipe and deadlock the worker."""
        process = self._process
        if process is None or process.stderr is None:
            return
        for line in process.stderr:
            if line.strip():
                self._noise.append(line.rstrip())

    def _tail(self) -> str:
        """The worker's last lines of output, for a failure message that says what it was doing."""
        return "\n".join(self._noise)


_WORKER_LOCK = threading.Lock()
_WORKER: Optional[_YAMNetWorker] = None


def shutdown_yamnet_worker() -> None:
    """End the process's YAMNet worker, releasing its model.

    A later classification starts a new one. Registered to run at interpreter exit, so a caller only
    needs this to hand the memory back earlier than that.
    """
    global _WORKER
    with _WORKER_LOCK:
        worker, _WORKER = _WORKER, None
    if worker is not None:
        worker.close()


atexit.register(shutdown_yamnet_worker)


def _worker(timeout_s: int) -> _YAMNetWorker:
    """The running worker, started if there is not a live one already.

    Args:
        timeout_s: Wall-clock ceiling on a load, when one is needed.

    Returns:
        The worker with its model loaded.

    Raises:
        Exception: Whatever stopped the worker from starting.
    """
    global _WORKER
    if _WORKER is not None and _WORKER.alive:
        return _WORKER
    if _WORKER is not None:
        _WORKER.close()
        _WORKER = None
    worker = _YAMNetWorker()
    try:
        worker.start(timeout_s)
    except Exception:
        worker.close()
        raise
    _WORKER = worker
    return worker


class YAMNetClassifier:
    """YAMNet audio classification via isolated subprocess venv."""

    # YAMNet uses fixed 0.96s windows with 0.48s hop internally
    WINDOW_SECONDS = 0.96
    HOP_SECONDS = 0.48

    @classmethod
    def classify_with_yamnet(
        cls,
        audios: List[Audio],
        top_k: int = 5,
    ) -> List[List[Dict[str, Any]]]:
        """Classify audios using YAMNet (521 AudioSet classes).

        YAMNet uses its own internal windowing (0.96s windows, 0.48s hop).
        Each audio produces multiple per-window results.

        The worker holding the model outlives the call; :func:`shutdown_yamnet_worker` ends it.

        Args:
            audios: Audio objects (mono, any sample rate — resampled to 16kHz internally).
            top_k: Number of top labels per window.

        Returns:
            List of per-audio results, each containing per-window dicts
            with ``labels``, ``scores``, ``start``, ``end``.
        """
        with tempfile.TemporaryDirectory(prefix="senselab-yamnet-") as tmpdir:
            tmp = Path(tmpdir)

            audio_paths = []
            durations = []
            for i, audio in enumerate(audios):
                # Resample to 16kHz inside the loop to avoid holding all
                # resampled audios in memory simultaneously
                resampled = resample_audios([audio], resample_rate=16000)[0]
                path = str(tmp / f"audio_{i}.wav")
                # Not Audio.save_to_file: that path writes PCM_16, which replaces faint
                # residual content with quantization noise (see LOSSLESS_WAV_SUBTYPE).
                report = write_worker_wav(path, resampled.waveform.squeeze().numpy(), 16000)
                if report["clipped_fraction"] > 0.0:
                    logger.warning(
                        "yamnet input clipped: %.1f%% of samples at or beyond full scale; "
                        "the classifier will respond to distortion rather than content",
                        100.0 * report["clipped_fraction"],
                    )
                audio_paths.append(path)
                durations.append(resampled.waveform.shape[1] / resampled.sampling_rate)

            with _WORKER_LOCK:
                batched = _worker(_REQUEST_TIMEOUT_S).classify(audio_paths, top_k, _REQUEST_TIMEOUT_S)

            # Add timestamps to each window based on YAMNet's fixed windowing
            all_results: List[List[Dict[str, Any]]] = []
            for audio_idx, windows in enumerate(batched):
                duration = durations[audio_idx]
                timestamped = []
                for i, w in enumerate(windows):
                    start = i * cls.HOP_SECONDS
                    end = min(start + cls.WINDOW_SECONDS, duration)
                    timestamped.append(
                        {
                            "start": start,
                            "end": end,
                            "label_scores": w["label_scores"],
                            "win_length": cls.WINDOW_SECONDS,
                            "hop_length": cls.HOP_SECONDS,
                        }
                    )
                all_results.append(timestamped)

            return all_results


YAMNET_WINDOW_SECONDS = YAMNetClassifier.WINDOW_SECONDS
"""YAMNet's native frame. An input shorter than this is zero-padded to it inside the model."""


class SpanTooShortForYAMNet(ValueError):
    """A span shorter than YAMNet's native frame; attribute it from covering whole-file windows instead."""


def span_yamnet_input(audio: Audio, extent: tuple[float, float]) -> Audio:
    """Slice a span at least :data:`YAMNET_WINDOW_SECONDS` long, ready for YAMNet's own windowing.

    Args:
        audio: The recording the span was proposed over.
        extent: The span's ``(start, end)`` in seconds.

    Returns:
        The span's own samples, unchanged.

    Raises:
        SpanTooShortForYAMNet: The span is shorter than YAMNet's native frame.
    """
    start, end = extent
    rate = audio.sampling_rate
    first = int(round(start * rate))
    last = int(round(end * rate))
    span = audio.waveform[..., first:last]
    frame = int(round(YAMNET_WINDOW_SECONDS * rate))
    if span.shape[-1] < frame:
        raise SpanTooShortForYAMNet(f"{span.shape[-1]} samples, need at least {frame}")
    return Audio(waveform=span.clone(), sampling_rate=rate)
