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

LOSSLESS_WAV_SUBTYPE = "FLOAT"
"""WAV subtype for the worker hand-off.

The temp WAV is the only lossy step between senselab and the TensorFlow subprocess, and
this classifier has an absolute low-level floor — so a fixed-point write is exactly the
wrong thing here. Measured with a 16-bit write: a -100 dBFS signal reads back at -93 dBFS,
because 16-bit quantization noise is louder than the content; at -120 dBFS it reads back as
exact zeros, which the model reports as silence with full confidence.

Two reasons that matters beyond losing a faint source. First, background characterization
deliberately presents this classifier quiet residual audio. Second, 16-bit quantization
noise is statistically indistinguishable from analog broadband noise, so amplifying it
yields the water-like environmental labels the noise-character guard exists to reject —
a fabricated finding rather than a missed one.
"""


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
    import soundfile as sf

    arr = np.asarray(waveform, dtype=np.float32)
    if arr.ndim > 1:
        arr = arr.mean(axis=0) if arr.shape[0] < arr.shape[-1] else arr.mean(axis=-1)
    clipped = float(np.count_nonzero(np.abs(arr) >= 0.9999) / arr.size) if arr.size else 0.0
    sf.write(str(path), arr, sampling_rate, subtype=LOSSLESS_WAV_SUBTYPE)
    return {"subtype": LOSSLESS_WAV_SUBTYPE, "clipped_fraction": clipped, "requantized": False}


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
    import numpy as np
    import soundfile as sf
    import tensorflow_hub as hub

    args = json.loads(sys.stdin.read())
    audio_paths = args["audio_paths"]
    top_k = args.get("top_k", 5)

    # Load YAMNet
    model = hub.load("https://tfhub.dev/google/yamnet/1")

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
                "labels": [class_names[idx] for idx in top_indices],
                "scores": [float(frame_scores[idx]) for idx in top_indices],
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
                            "labels": w["labels"],
                            "scores": w["scores"],
                            "win_length": cls.WINDOW_SECONDS,
                            "hop_length": cls.HOP_SECONDS,
                        }
                    )
                all_results.append(timestamped)

            return all_results
