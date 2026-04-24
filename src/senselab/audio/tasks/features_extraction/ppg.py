"""Phonetic Posteriorgrams (PPGs) via isolated subprocess venv.

ppgs depends on espnet, snorkel, and lightning which conflict with modern
torch/Python. It runs in an isolated subprocess venv managed by uv.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from matplotlib.figure import Figure

import numpy as np
import torch

from senselab.audio.data_structures import Audio
from senselab.utils.data_structures import DeviceType, _select_device_and_dtype, logger
from senselab.utils.subprocess_venv import ensure_venv, parse_subprocess_result

# Try to import the phoneme inventory from the ppgs library.
# ppgs runs in a subprocess venv and may not be importable in the main env.
try:
    from ppgs import PHONEMES as _PPGS_PHONEMES

    _PHONEME_LABELS: Tuple[str, ...] = tuple(str(p) for p in _PPGS_PHONEMES)
except (ImportError, RuntimeError):
    # Fallback: the 40 ARPAbet phonemes used by ppgs 0.0.9.
    # Source: https://github.com/interactiveaudiolab/ppgs/blob/main/ppgs/data/phonemes.py
    _PHONEME_LABELS = (
        "aa",
        "ae",
        "ah",
        "ao",
        "aw",
        "ay",
        "b",
        "ch",
        "d",
        "dh",
        "eh",
        "er",
        "ey",
        "f",
        "g",
        "hh",
        "ih",
        "iy",
        "jh",
        "k",
        "l",
        "m",
        "n",
        "ng",
        "ow",
        "oy",
        "p",
        "r",
        "s",
        "sh",
        "t",
        "th",
        "uh",
        "uw",
        "v",
        "w",
        "y",
        "z",
        "zh",
        "<silent>",
    )

# PPGs venv specification
_PPGS_VENV = "ppgs"
_PPGS_REQUIREMENTS = [
    "ppgs>=0.0.9,<0.0.10",
    "espnet",
    "snorkel>=0.10.0,<0.11.0",
    "lightning~=2.4",
    "torch>=2.8,<2.9",
    "torchaudio>=2.8,<2.9",
    "numpy",
    "soundfile",
]
_PPGS_PYTHON = "3.11"

# Worker script — runs inside the isolated venv (no senselab imports)
_WORKER_SCRIPT = r"""
import json
import sys
import traceback
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

args = json.loads(sys.stdin.read())
audio_paths = args["audio_paths"]
device = args["device"]
output_dir = args["output_dir"]

import ppgs

gpu = 0 if device == "cuda" else None

output_paths = []
for i, audio_path in enumerate(audio_paths):
    data, sr = sf.read(audio_path, dtype="float32")
    waveform = torch.from_numpy(data).unsqueeze(0) if data.ndim == 1 else torch.from_numpy(data.T)
    try:
        posteriorgram = ppgs.from_audio(
            torch.unsqueeze(waveform, dim=0),
            ppgs.SAMPLE_RATE,
            gpu=gpu,
        ).cpu()
    except RuntimeError as e:
        print(f"RuntimeError extracting PPGs for audio {i}: {e}", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        posteriorgram = torch.tensor(float("nan"))

    out_path = str(Path(output_dir) / f"ppg_{i}.npy")
    np.save(out_path, posteriorgram.float().numpy())
    output_paths.append(out_path)

print(json.dumps({"output_paths": output_paths}))
"""


def extract_ppgs_from_audios(audios: List[Audio], device: Optional[DeviceType] = None) -> List[torch.Tensor]:
    """Extracts phonetic posteriorgrams (PPGs) from every audio.

    The ppgs model runs in an isolated subprocess venv with its own
    Python and dependencies. Audio is transferred via FLAC files.

    Args:
        audios: The audios to extract PPGs from.
        device: Device to use (CUDA or CPU).

    Returns:
        List of PPG tensors, one per input audio.
    """
    device, _ = _select_device_and_dtype(user_preference=device, compatible_devices=[DeviceType.CUDA, DeviceType.CPU])

    if any(audio.waveform.shape[0] != 1 for audio in audios):
        raise ValueError("Only mono audio is supported by ppgs model.")

    venv_dir = ensure_venv(_PPGS_VENV, _PPGS_REQUIREMENTS, python_version=_PPGS_PYTHON)
    python = str(venv_dir / "bin" / "python")

    with tempfile.TemporaryDirectory(prefix="senselab-ppgs-") as tmpdir:
        tmp = Path(tmpdir)

        # Serialize audios to FLAC
        audio_paths = []
        for i, audio in enumerate(audios):
            path = str(tmp / f"audio_{i}.flac")
            audio.save_to_file(path, format="flac")
            audio_paths.append(path)

        # Run worker in isolated venv
        input_json = json.dumps(
            {
                "audio_paths": audio_paths,
                "device": device.value,
                "output_dir": str(tmp),
            }
        )

        # Scrub MPLBACKEND so the subprocess venv's matplotlib doesn't
        # choke on Jupyter's inline backend which isn't installed there.
        sub_env = {k: v for k, v in os.environ.items() if k != "MPLBACKEND"}

        result = subprocess.run(
            [python, "-c", _WORKER_SCRIPT],
            input=input_json,
            capture_output=True,
            text=True,
            timeout=600,
            env=sub_env,
        )

        output = parse_subprocess_result(result, "PPGs")

        # Load results
        posteriorgrams = []
        for out_path in output.get("output_paths", []):
            tensor = torch.from_numpy(np.load(out_path))
            posteriorgrams.append(tensor)

        return posteriorgrams


# ---------------------------------------------------------------------------
# PPG phoneme duration analysis
# ---------------------------------------------------------------------------


def _to_frame_major_posteriorgram(posteriorgram: torch.Tensor) -> torch.Tensor:
    """Normalize a PPG tensor to frame-major layout ``(frames, phonemes)``.

    The ppgs library typically returns tensors shaped ``(1, phonemes, frames)``
    or ``(phonemes, frames)``.  This helper squeezes any leading batch
    dimension and transposes so that rows correspond to time frames and
    columns correspond to phonemes.

    Args:
        posteriorgram: PPG tensor in any of the common layouts.

    Returns:
        A 2-D tensor of shape ``(frames, phonemes)``.

    Raises:
        ValueError: If the tensor has fewer than 2 dimensions after squeezing.
    """
    t = posteriorgram.detach().cpu()
    # Remove only the batch dimension (dim 0) if it's size 1
    if t.ndim == 3 and t.shape[0] == 1:
        t = t.squeeze(0)
    if t.ndim < 2:
        raise ValueError(f"Expected at least a 2-D posteriorgram after squeezing, got shape {t.shape}")
    # The ppgs library outputs (phonemes, frames) where the phoneme count
    # matches len(_PHONEME_LABELS).  Use that knowledge first; fall back to
    # the "smaller dimension = phonemes" heuristic for unknown inventories.
    n_phonemes = len(_PHONEME_LABELS)
    if t.shape[0] == n_phonemes and t.shape[1] != n_phonemes:
        # (phonemes, frames) -> (frames, phonemes)
        t = t.T
    elif t.shape[1] == n_phonemes and t.shape[0] != n_phonemes:
        # Already (frames, phonemes)
        pass
    elif t.shape[0] < t.shape[1]:
        # Fallback heuristic: smaller dim is phonemes
        t = t.T
    # else: ambiguous (e.g. square); assume already frame-major
    return t


def _extract_ppg_segments(
    audio: Audio,
    frame_major_posteriorgram: torch.Tensor,
) -> List[Dict[str, Any]]:
    """Find contiguous argmax-phoneme segments and compute their durations.

    For every time frame the dominant phoneme is determined via ``argmax``.
    Consecutive frames that share the same dominant phoneme are grouped
    into a *segment*.

    Args:
        audio: The source :class:`Audio` (used to derive real-time durations).
        frame_major_posteriorgram: PPG tensor of shape ``(frames, phonemes)``.

    Returns:
        A list of segment dicts, each containing:

        - ``phoneme_index`` (int): Index into the phoneme inventory.
        - ``phoneme`` (str): Human-readable phoneme label.
        - ``start_frame`` (int): First frame of the segment (inclusive).
        - ``end_frame`` (int): Last frame of the segment (inclusive).
        - ``frame_count`` (int): Number of frames in the segment.
        - ``start_seconds`` (float): Onset time in seconds.
        - ``end_seconds`` (float): Offset time in seconds.
        - ``duration_seconds`` (float): Segment duration in seconds.
    """
    num_frames = frame_major_posteriorgram.shape[0]
    if num_frames == 0 or frame_major_posteriorgram.shape[1] == 0:
        return []

    total_duration = audio.waveform.shape[1] / audio.sampling_rate
    seconds_per_frame = total_duration / num_frames

    argmax_indices = torch.argmax(frame_major_posteriorgram, dim=1)
    phoneme_labels = _PHONEME_LABELS
    num_labels = len(phoneme_labels)

    segments: List[Dict[str, Any]] = []
    current_idx = int(argmax_indices[0].item())
    start_frame = 0

    for frame in range(1, num_frames):
        idx = int(argmax_indices[frame].item())
        if idx != current_idx:
            # Close the current segment
            label = phoneme_labels[current_idx] if current_idx < num_labels else str(current_idx)
            frame_count = frame - start_frame
            segments.append(
                {
                    "phoneme_index": current_idx,
                    "phoneme": label,
                    "start_frame": start_frame,
                    "end_frame": frame - 1,
                    "frame_count": frame_count,
                    "start_seconds": start_frame * seconds_per_frame,
                    "end_seconds": frame * seconds_per_frame,
                    "duration_seconds": frame_count * seconds_per_frame,
                }
            )
            current_idx = idx
            start_frame = frame

    # Final segment
    label = phoneme_labels[current_idx] if current_idx < num_labels else str(current_idx)
    frame_count = num_frames - start_frame
    segments.append(
        {
            "phoneme_index": current_idx,
            "phoneme": label,
            "start_frame": start_frame,
            "end_frame": num_frames - 1,
            "frame_count": frame_count,
            "start_seconds": start_frame * seconds_per_frame,
            "end_seconds": num_frames * seconds_per_frame,
            "duration_seconds": frame_count * seconds_per_frame,
        }
    )
    return segments


def extract_mean_phoneme_durations(
    audio: Audio,
    posteriorgram: torch.Tensor,
) -> Dict[str, Any]:
    """Compute per-phoneme duration statistics from a PPG tensor.

    For each frame the dominant (argmax) phoneme is identified.  Contiguous
    runs of the same phoneme form *segments*.  This function aggregates
    segment counts and durations per phoneme and returns a summary dict.

    Args:
        audio: The source audio used to derive real-time durations.
        posteriorgram: PPG tensor as returned by
            :func:`extract_ppgs_from_audios` — typically shaped
            ``(1, phonemes, frames)`` or ``(phonemes, frames)``.

    Returns:
        A dict with the keys:

        - ``frame_count`` (int): Total number of PPG frames.
        - ``phoneme_count`` (int): Number of phonemes in the inventory.
        - ``analysis_duration_seconds`` (float): Audio duration in seconds.
        - ``seconds_per_frame`` (float): Duration of one PPG frame.
        - ``mean_segment_duration_seconds`` (float): Mean segment length
          across *all* segments.
        - ``phoneme_durations`` (list[dict]): Per-phoneme breakdown, each
          containing ``phoneme``, ``count``, ``mean_duration_seconds``, and
          ``total_duration_seconds``.

        If the input tensor contains NaN values or is empty, an empty dict
        is returned.
    """
    # Guard against NaN / degenerate tensors
    if posteriorgram.numel() == 0 or torch.isnan(posteriorgram).any():
        logger.warning("Posteriorgram is empty or contains NaN — returning empty result.")
        return {}

    frame_major = _to_frame_major_posteriorgram(posteriorgram)
    segments = _extract_ppg_segments(audio, frame_major)

    if not segments:
        return {}

    num_frames = frame_major.shape[0]
    total_duration = audio.waveform.shape[1] / audio.sampling_rate
    seconds_per_frame = total_duration / num_frames

    # Aggregate per phoneme
    phoneme_stats: Dict[str, Dict[str, float]] = {}
    for seg in segments:
        label = seg["phoneme"]
        if label not in phoneme_stats:
            phoneme_stats[label] = {"count": 0, "total_duration": 0.0}
        phoneme_stats[label]["count"] += 1
        phoneme_stats[label]["total_duration"] += seg["duration_seconds"]

    phoneme_durations = []
    for label, stats in sorted(phoneme_stats.items()):
        count = int(stats["count"])
        total_dur = stats["total_duration"]
        phoneme_durations.append(
            {
                "phoneme": label,
                "count": count,
                "mean_duration_seconds": total_dur / count,
                "total_duration_seconds": total_dur,
            }
        )

    mean_segment_duration = sum(s["duration_seconds"] for s in segments) / len(segments)

    return {
        "frame_count": num_frames,
        "phoneme_count": frame_major.shape[1],
        "analysis_duration_seconds": total_duration,
        "seconds_per_frame": seconds_per_frame,
        "mean_segment_duration_seconds": mean_segment_duration,
        "phoneme_durations": phoneme_durations,
    }


def plot_ppg_phoneme_timeline(
    audio: Audio,
    posteriorgram: torch.Tensor,
    title: str = "PPG phoneme timeline",
    show: bool = True,
) -> Figure:
    """Plot a horizontal-bar timeline of PPG phoneme segments.

    Each contiguous run of a dominant phoneme is drawn as a coloured
    horizontal bar.  Only phonemes that actually appear are shown on the
    y-axis.

    Args:
        audio: The source audio used to derive real-time durations.
        posteriorgram: PPG tensor (see :func:`extract_mean_phoneme_durations`
            for accepted shapes).
        title: Figure title.
        show: Whether to call ``plt.show()`` after creating the figure.

    Returns:
        The :class:`matplotlib.figure.Figure` object.

    Raises:
        ValueError: If the posteriorgram is empty or contains NaN values.
    """
    import matplotlib.pyplot as plt

    if posteriorgram.numel() == 0 or torch.isnan(posteriorgram).any():
        raise ValueError("Cannot plot an empty or NaN posteriorgram.")

    frame_major = _to_frame_major_posteriorgram(posteriorgram)
    segments = _extract_ppg_segments(audio, frame_major)

    if not segments:
        raise ValueError("No segments found in posteriorgram.")

    # Determine unique phonemes (preserving appearance order)
    seen: Dict[str, int] = {}
    for seg in segments:
        if seg["phoneme"] not in seen:
            seen[seg["phoneme"]] = len(seen)
    phoneme_order = list(seen.keys())

    cmap = plt.get_cmap("tab20")
    num_colors = 20  # tab20 has 20 colours

    fig, ax = plt.subplots(figsize=(14, max(3, 0.35 * len(phoneme_order))))

    for seg in segments:
        y_idx = seen[seg["phoneme"]]
        color = cmap(seg["phoneme_index"] % num_colors)
        start = seg["start_seconds"]
        width = seg["duration_seconds"]

        ax.barh(y_idx, width, left=start, height=0.7, color=color, edgecolor="none")

        # Onset/offset markers
        ax.plot(start, y_idx, "|", color="black", markersize=6, markeredgewidth=0.5)
        ax.plot(start + width, y_idx, "|", color="black", markersize=6, markeredgewidth=0.5)

    ax.set_yticks(range(len(phoneme_order)))
    ax.set_yticklabels(phoneme_order)
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Phoneme")
    ax.set_title(title)
    ax.invert_yaxis()
    fig.tight_layout()

    if show:
        plt.show()

    return fig
