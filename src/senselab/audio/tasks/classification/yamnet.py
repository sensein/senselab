"""YAMNet audio classification via isolated subprocess venv.

YAMNet is a TensorFlow-based model that classifies audio into 521
AudioSet classes. It runs in an isolated subprocess venv to avoid
TF/PyTorch conflicts.
"""

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.preprocessing import resample_audios
from senselab.utils.data_structures.logging import logger
from senselab.utils.subprocess_venv import _clean_subprocess_env, ensure_venv, parse_subprocess_result, venv_python


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

_YAMNET_WORKER = r"""
import json
import sys

try:
    import os
    import pathlib
    import shutil

    import numpy as np
    import soundfile as sf
    import tensorflow_hub as hub

    args = json.loads(sys.stdin.read())
    audio_paths = args["audio_paths"]
    top_k = args.get("top_k", 5)

    # Load YAMNet.
    #
    # The TF-Hub cache defaults to $TMPDIR, which is wrong twice over: it is discarded between
    # reboots so the model is re-fetched, and a partially-written entry is reused forever
    # because TF-Hub only checks that the directory exists. That second failure is not
    # hypothetical — it took YAMNet out of a real run with
    # "contains neither 'saved_model.pb' nor 'saved_model.pbtxt'", leaving the axis a signal
    # short with no indication the cause was a corrupt download rather than the audio.
    _HUB_URL = "https://tfhub.dev/google/yamnet/1"
    cache_root = pathlib.Path(
        os.environ.get("SENSELAB_TFHUB_CACHE")
        or (pathlib.Path.home() / ".cache" / "senselab" / "tfhub")
    )
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ["TFHUB_CACHE_DIR"] = str(cache_root)

    def _load_yamnet():
        try:
            return hub.load(_HUB_URL)
        except (ValueError, OSError) as exc:
            # A corrupt entry must be a cache miss, not a permanent failure. Discard the
            # incomplete directory and fetch once more; a second failure is real.
            if "saved_model" not in str(exc):
                raise
            for stale in cache_root.iterdir():
                if stale.is_dir() and not any(stale.glob("saved_model.pb*")):
                    shutil.rmtree(stale, ignore_errors=True)
            return hub.load(_HUB_URL)

    model = _load_yamnet()

    # Load class names from the model's assets
    import csv
    class_map_path = model.class_map_path().numpy().decode("utf-8")
    with open(class_map_path) as f:
        reader = csv.DictReader(f)
        class_names = [row["display_name"] for row in reader]

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

    print(json.dumps({"results": all_results}))
except Exception as exc:
    print(json.dumps({"error": {"type": type(exc).__name__, "message": str(exc)}}))
    sys.exit(1)
"""


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

        Args:
            audios: Audio objects (mono, any sample rate — resampled to 16kHz internally).
            top_k: Number of top labels per window.

        Returns:
            List of per-audio results, each containing per-window dicts
            with ``labels``, ``scores``, ``start``, ``end``.
        """
        venv_dir = ensure_venv(_YAMNET_VENV, _YAMNET_REQUIREMENTS, python_version=_YAMNET_PYTHON)
        python = venv_python(venv_dir)

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

            input_json = json.dumps(
                {
                    "audio_paths": audio_paths,
                    "top_k": top_k,
                }
            )

            env = _clean_subprocess_env()
            result = subprocess.run(
                [python, "-c", _YAMNET_WORKER],
                input=input_json,
                capture_output=True,
                text=True,
                timeout=600,
                env=env,
            )

            output = parse_subprocess_result(result, "YAMNet")

            # Add timestamps to each window based on YAMNet's fixed windowing
            all_results: List[List[Dict[str, Any]]] = []
            for audio_idx, windows in enumerate(output.get("results", [])):
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
