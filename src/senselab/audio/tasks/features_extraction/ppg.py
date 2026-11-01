"""Phonetic Posteriorgrams (PPGs) via isolated subprocess venv.

ppgs depends on espnet, snorkel, and lightning which conflict with modern
torch/Python. It runs in an isolated subprocess venv managed by uv.
"""

from __future__ import annotations

import json
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
from senselab.utils.subprocess_venv import (
    _clean_subprocess_env,
    ensure_venv,
    parse_subprocess_result,
    provisioned_venv_dirs,
    stage_portable_audio_io,
    venv_python,
)

# Try to import the phoneme inventory from the ppgs library.
# ppgs runs in a subprocess venv and may not be importable in the main env.
try:
    from ppgs import PHONEMES as _PPGS_PHONEMES
    from ppgs import SAMPLE_RATE as _PPGS_SAMPLE_RATE

    PHONEME_LABELS: Tuple[str, ...] = tuple(str(p) for p in _PPGS_PHONEMES)
    PPGS_SAMPLE_RATE: int = int(_PPGS_SAMPLE_RATE)
except (ImportError, RuntimeError):
    # Fallback: the 40 ARPAbet phonemes used by ppgs 0.0.9.
    # Source: https://github.com/interactiveaudiolab/ppgs/blob/main/ppgs/data/phonemes.py
    PHONEME_LABELS = (
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
    PPGS_SAMPLE_RATE = 16000

# PPGs venv specification
_PPGS_VENV = "ppgs"
_PPGS_REQUIREMENTS = [
    "ppgs>=0.0.9,<0.0.10",
    "torch>=2.8,<2.9",
    "torchaudio>=2.8,<2.9",
    "numpy",
    "soundfile",
    "huggingface-hub<2",  # see specs/20261002-ppgs-venv-resolution/design.md
]
_PPGS_PYTHON = "3.11"

PPGS_FRAME_RATE: float = 100.0
"""Posteriorgram frames per second: ppgs' 160-sample hop at 16 kHz. Frame ``i`` is centred at ``(i + 0.5) / 100`` s."""

DEFAULT_WINDOW_S = 60.0
"""Audio, in seconds, the worker turns into posteriors per model call (see the sparse-ppg-boundaries spec)."""

CHUNK_BATCH = 16
"""ppgs inference chunks (5 s each) the worker stacks into one forward pass."""

PPGS_REPO_ID = "CameronChurchwell/ppgs"
"""The Hugging Face repository ppgs downloads its pretrained checkpoints from."""

PPGS_CHECKPOINT_FILE = "mel-800k.pt"
"""The ``mel`` model's checkpoint inside :data:`PPGS_REPO_ID`."""

PPGS_REVISION = "main"
"""The ref resolved, through the run manifest, to the commit every checkpoint load is pinned to."""

# Worker script — runs inside the isolated venv (no senselab imports)
_WORKER_SCRIPT = r"""
import json
import sys
import traceback
from pathlib import Path

import numpy as np
import torch

args = json.loads(sys.stdin.read())
sys.path.insert(0, args["io_dir"])
from portable_audio_io import audio_info, read_audio

import ppgs

device = torch.device({"cuda": "cuda:0", "mps": "mps"}.get(args["device"], "cpu"))
hop = int(ppgs.HOPSIZE)
stride = int(ppgs.CHUNK_LENGTH) - 2 * int(ppgs.CHUNK_OVERLAP)
window_frames = args["window_frames"]
if window_frames is not None:
    window_frames = -(-int(window_frames) // stride) * stride
context_frames = stride
model = ppgs.load.model(checkpoint=args["checkpoint"], representation="mel").to(device).eval()


chunk_batch = int(args["chunk_batch"])


def run(x):
    return model(x, torch.full((x.shape[0],), x.shape[-1], dtype=torch.long, device=device))


def logits(features):
    # ppgs.model.Transformer.forward's own chunking, with its full-length chunks batched.
    overlap, length = int(ppgs.CHUNK_OVERLAP), int(ppgs.CHUNK_LENGTH)
    if features.shape[-1] <= length:
        return run(features)
    padded = torch.nn.functional.pad(features, (overlap, 0), mode="replicate")
    splits = [padded[..., i * stride : i * stride + length] for i in range(-(-features.shape[-1] // stride))]
    kept = [None] * len(splits)
    full = [i for i, split in enumerate(splits) if split.shape[-1] == length]
    for first in range(0, len(full), chunk_batch):
        group = full[first : first + chunk_batch]
        out = run(torch.cat([splits[i] for i in group], dim=0))
        for j, i in enumerate(group):
            kept[i] = out[j : j + 1, :, overlap : length - overlap]
    for i, split in enumerate(splits):
        if kept[i] is None:
            kept[i] = run(split)[..., overlap : length - overlap]
    return torch.cat(kept, dim=-1)


def posteriors(samples):
    audio = torch.from_numpy(np.ascontiguousarray(samples, dtype=np.float32))[None, None]
    features = ppgs.preprocess.from_audio(audio, representation="mel", sample_rate=ppgs.SAMPLE_RATE)
    with torch.inference_mode():
        probabilities = torch.softmax(logits(features.float().to(device)), dim=1)
    return probabilities[0].float().cpu().numpy()


def file_posteriors(path):
    total = int(audio_info(path).frames)
    n_frames = total // hop
    if window_frames is None or n_frames <= window_frames + 2 * context_frames:
        data, _ = read_audio(path)
        return posteriors(data)
    out = np.empty((len(ppgs.PHONEMES), n_frames), dtype=np.float32)
    for start in range(0, n_frames, window_frames):
        stop = min(n_frames, start + window_frames)
        left = max(0, start - context_frames)
        right = min(n_frames, stop + context_frames)
        data, _ = read_audio(path, start=left * hop, stop=total if right == n_frames else right * hop)
        block = posteriors(data)
        out[:, start:stop] = block[:, start - left : stop - left]
    return out


outputs = []
for i, audio_path in enumerate(args["audio_paths"]):
    try:
        posteriorgram = file_posteriors(audio_path)
    except (RuntimeError, ValueError) as e:
        print(f"{type(e).__name__} extracting PPGs for audio {i}: {e}", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        outputs.append({"path": None, "error": f"{type(e).__name__}: {e}"})
        continue
    out_path = str(Path(args["output_dir"]) / f"ppg_{i}.npy")
    np.save(out_path, posteriorgram[None])
    outputs.append({"path": out_path, "error": None})

print(json.dumps({"outputs": outputs}))
"""


class PpgsPosteriorgramUnavailable(ValueError):
    """The ppgs model produced no posteriorgram for one recording; attribute it as an absence."""


def ensure_ppgs_venv() -> Path:
    """Build the isolated ppgs venv if this host has not built it, and return its directory.

    Call it once, on its own, before fanning a batch job out across an array: a cold build is far
    longer than the lock's patience window, so every task that races one waits on it.

    Returns:
        The venv's directory.
    """
    return ensure_venv(_PPGS_VENV, _PPGS_REQUIREMENTS, python_version=_PPGS_PYTHON)


def ppgs_venv_is_provisioned() -> bool:
    """Whether a completed ppgs venv already exists on this host.

    Returns:
        True when at least one ppgs venv carries the completion marker. A half-built tree is not
        provisioned.
    """
    return bool(provisioned_venv_dirs(_PPGS_VENV))


def require_posteriorgram(result: "torch.Tensor | PpgsPosteriorgramUnavailable") -> torch.Tensor:
    """Unwrap one entry of :func:`extract_ppgs_from_audios`, raising the absence it may hold.

    Args:
        result: One entry of the returned list.

    Returns:
        The posteriorgram, when the model produced one.

    Raises:
        PpgsPosteriorgramUnavailable: When the entry is the recorded absence rather than a tensor.
    """
    if isinstance(result, PpgsPosteriorgramUnavailable):
        raise result
    return result


def ppgs_checkpoint(revision: str = PPGS_REVISION) -> Tuple[str, Path]:
    """Resolve the ppgs ``mel`` checkpoint to a commit and stage that commit's file locally.

    The commit comes from :func:`~senselab.utils.model_revision.resolve_revision` (recorded in the
    run manifest); :data:`PPGS_CHECKPOINT_FILE` is downloaded at that SHA. The worker is handed the
    local file and makes no Hub call.

    Args:
        revision: A ref or a 40-hex commit of :data:`PPGS_REPO_ID`.

    Returns:
        ``(commit_sha, checkpoint_path)``; the path lies under the cache's ``snapshots/<commit_sha>/``.
    """
    from huggingface_hub import hf_hub_download

    from senselab.utils.model_revision import resolve_revision

    sha = resolve_revision(PPGS_REPO_ID, revision)
    return sha, Path(hf_hub_download(PPGS_REPO_ID, PPGS_CHECKPOINT_FILE, revision=sha))


def ppgs_model_revision(revision: str = PPGS_REVISION) -> str:
    """The commit :func:`extract_ppgs_from_audios` loads for ``revision``, for provenance.

    Args:
        revision: A ref or a 40-hex commit of :data:`PPGS_REPO_ID`.

    Returns:
        The 40-hex commit SHA (memoized for the run; no I/O after the first resolution).
    """
    from senselab.utils.model_revision import resolve_revision

    return resolve_revision(PPGS_REPO_ID, revision)


WORKER_STARTUP_S = 300.0
"""Seconds allowed for the worker before it has any audio to read: interpreter, torch, checkpoint."""

WORKER_SECONDS_PER_AUDIO_SECOND = 2.0
"""Seconds allowed per second of audio in the batch, over the startup allowance."""


def _worker_timeout_s(audio_seconds: float) -> float:
    """How long the worker may take for a batch of this much audio.

    Args:
        audio_seconds: Total duration of every audio in the batch.

    Returns:
        The subprocess timeout, in seconds.
    """
    return WORKER_STARTUP_S + WORKER_SECONDS_PER_AUDIO_SECOND * max(0.0, audio_seconds)


def extract_ppgs_from_audios(
    audios: List[Audio],
    device: Optional[DeviceType] = None,
    timeout_min_s: Optional[float] = None,
    window_s: Optional[float] = DEFAULT_WINDOW_S,
    revision: str = PPGS_REVISION,
) -> List["torch.Tensor | PpgsPosteriorgramUnavailable"]:
    """Extracts phonetic posteriorgrams (PPGs) from every audio.

    The ppgs ``mel`` model runs in an isolated subprocess venv with its own Python and dependencies,
    in float32. Audio is transferred via WAV files; the worker reads each one window by window, so
    its memory is bounded by ``window_s`` rather than by the recording's length. Windows are aligned
    to ppgs' own 4 s inference chunks and carry one chunk of context on each side, so the windowed
    posteriorgram equals the one-call posteriorgram.

    Args:
        audios: The audios to extract PPGs from. Every one must be mono and at
            :data:`PPGS_SAMPLE_RATE`.
        device: Device to use (CUDA, MPS or CPU).
        timeout_min_s: The least the worker is allowed, in seconds, whatever the audio's length; None
            allows only the length-scaled budget.
        window_s: Seconds of audio per model call, rounded up to a multiple of ppgs' 4 s chunk stride.
            None runs every recording in one call.
        revision: Ref or commit of :data:`PPGS_REPO_ID`; resolved to a SHA by :func:`ppgs_checkpoint`
            and the checkpoint loaded from that commit. :func:`ppgs_model_revision` returns the same
            SHA for provenance.

    Returns:
        One entry per input audio, in input order: a PPG tensor in ``(1, phonemes, frames)`` layout
        with the phoneme axis ordered as :data:`PHONEME_LABELS`, or a
        :class:`PpgsPosteriorgramUnavailable` carrying the model's own message for a recording it
        raised on. :func:`require_posteriorgram` unwraps one entry.

    Raises:
        ValueError: If any audio is multi-channel, or is not at :data:`PPGS_SAMPLE_RATE`, or
            ``window_s`` is not positive.
    """
    if window_s is not None and window_s <= 0:
        raise ValueError(f"window_s must be positive or None, got {window_s}")
    device, _ = _select_device_and_dtype(
        user_preference=device, compatible_devices=[DeviceType.CUDA, DeviceType.MPS, DeviceType.CPU]
    )

    if any(audio.waveform.shape[0] != 1 for audio in audios):
        raise ValueError("Only mono audio is supported by ppgs model.")
    off_rate = sorted({audio.sampling_rate for audio in audios if audio.sampling_rate != PPGS_SAMPLE_RATE})
    if off_rate:
        raise ValueError(
            f"ppgs reads every waveform at {PPGS_SAMPLE_RATE} Hz and is given {off_rate}; resample before calling."
        )

    _, checkpoint = ppgs_checkpoint(revision)
    python = venv_python(ensure_ppgs_venv())
    budget = _worker_timeout_s(sum(audio.waveform.shape[-1] / audio.sampling_rate for audio in audios))
    if timeout_min_s is not None:
        budget = max(budget, float(timeout_min_s))

    with tempfile.TemporaryDirectory(prefix="senselab-ppgs-") as tmpdir:
        tmp = Path(tmpdir)

        # Serialize audios for the worker
        audio_paths = []
        for i, audio in enumerate(audios):
            path = str(tmp / f"audio_{i}.wav")
            audio.save_to_file(path)
            audio_paths.append(path)

        # Run worker in isolated venv
        input_json = json.dumps(
            {
                "audio_paths": audio_paths,
                "device": device.value,
                "window_frames": None if window_s is None else max(1, round(window_s * PPGS_FRAME_RATE)),
                "chunk_batch": CHUNK_BATCH,
                "checkpoint": str(checkpoint),
                "output_dir": str(tmp),
                "io_dir": stage_portable_audio_io(tmp),
            }
        )

        sub_env = _clean_subprocess_env()

        result = subprocess.run(
            [python, "-c", _WORKER_SCRIPT],
            input=input_json,
            capture_output=True,
            text=True,
            timeout=budget,
            env=sub_env,
        )

        output = parse_subprocess_result(result, "PPGs")

        # Load results
        posteriorgrams: List["torch.Tensor | PpgsPosteriorgramUnavailable"] = []
        for record in output.get("outputs", []):
            message = record.get("error")
            if message:
                posteriorgrams.append(PpgsPosteriorgramUnavailable(f"ppgs produced no posteriorgram: {message}"))
                continue
            posteriorgrams.append(torch.from_numpy(np.load(record["path"])))

        return posteriorgrams


# ---------------------------------------------------------------------------
# PPG phoneme duration analysis
# ---------------------------------------------------------------------------


def to_frame_major_posteriorgram(posteriorgram: torch.Tensor) -> torch.Tensor:
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
    # matches len(PHONEME_LABELS).  Use that knowledge first; fall back to
    # the "smaller dimension = phonemes" heuristic for unknown inventories.
    n_phonemes = len(PHONEME_LABELS)
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


def load_ppg_posteriorgram(path: str | Path) -> Tuple[torch.Tensor, float, int]:
    """Read a ``ppg_posteriorgram.npz`` sidecar into a frame-major tensor and its clock.

    Args:
        path: The sidecar ``write_ppg_posteriorgram`` wrote.

    Returns:
        The posteriorgram as float32 in ``(frames, phonemes)`` layout, the duration in seconds it
        covers, and the sampling rate it was measured at.

    Raises:
        OSError: If the sidecar cannot be opened.
        KeyError: If it carries none of ``posteriorgram``, ``duration_s`` or ``sampling_rate``.
        ValueError: If the array it carries is not a posteriorgram.
    """
    with np.load(Path(path)) as archive:
        posteriorgram = torch.from_numpy(archive["posteriorgram"].astype(np.float32))
        duration_s = float(archive["duration_s"])
        sampling_rate = int(archive["sampling_rate"])
    return to_frame_major_posteriorgram(posteriorgram), duration_s, sampling_rate


def extract_ppg_segments(
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
    phoneme_labels = PHONEME_LABELS
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

    frame_major = to_frame_major_posteriorgram(posteriorgram)
    segments = extract_ppg_segments(audio, frame_major)

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
    """Plot a waveform aligned with a horizontal-bar timeline of PPG phoneme segments.

    The top panel shows the audio waveform, and the bottom panel shows
    each contiguous run of a dominant phoneme as a coloured horizontal bar.
    Both panels share the same time axis.

    Args:
        audio: The source audio used to derive real-time durations and
            the waveform display.
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
    import numpy as np

    if posteriorgram.numel() == 0 or torch.isnan(posteriorgram).any():
        raise ValueError("Cannot plot an empty or NaN posteriorgram.")

    frame_major = to_frame_major_posteriorgram(posteriorgram)
    segments = extract_ppg_segments(audio, frame_major)

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

    duration = audio.waveform.shape[1] / audio.sampling_rate
    seg_height = max(3, 0.35 * len(phoneme_order))

    fig, (ax_wav, ax_seg) = plt.subplots(
        2,
        1,
        figsize=(14, 3 + seg_height),
        sharex=True,
        gridspec_kw={"height_ratios": [1, max(1, seg_height / 3)]},
    )

    # Waveform panel
    waveform_np = audio.waveform.squeeze().cpu().numpy()
    time_axis = np.linspace(0, duration, len(waveform_np), endpoint=False)
    ax_wav.plot(time_axis, waveform_np, linewidth=0.3, color="steelblue")
    ax_wav.set_ylabel("Amplitude")
    ax_wav.set_title(title)
    ax_wav.grid(True, alpha=0.2)

    # Phoneme segment panel
    for seg in segments:
        y_idx = seen[seg["phoneme"]]
        color = cmap(seg["phoneme_index"] % num_colors)
        start = seg["start_seconds"]
        width = seg["duration_seconds"]

        ax_seg.barh(y_idx, width, left=start, height=0.7, color=color, edgecolor="none")

        # Onset/offset markers
        ax_seg.plot(start, y_idx, "|", color="black", markersize=6, markeredgewidth=0.5)
        ax_seg.plot(start + width, y_idx, "|", color="black", markersize=6, markeredgewidth=0.5)

    ax_seg.set_yticks(range(len(phoneme_order)))
    ax_seg.set_yticklabels(phoneme_order, fontsize=7)
    ax_seg.set_xlabel("Time (seconds)")
    ax_seg.set_ylabel("Phoneme")
    ax_seg.set_xlim(0, duration)
    ax_seg.invert_yaxis()
    fig.tight_layout()

    if show:
        plt.show()

    return fig
