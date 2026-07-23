"""Continuous per-frame speech posteriors from ``pyannote/segmentation-3.0``.

Unlike ``detect_human_voice_activity_in_audios`` (which runs the high-level
``Pipeline`` and returns thresholded ``ScriptLine`` segments), this extractor
uses the low-level ``Model`` + ``Inference`` path to obtain the **raw per-frame
speech probability** (~16.9 ms/frame) without any segment thresholding or
hangover smoothing — exactly what the presence axis needs to resolve brief
events and to compute a within-bucket temporal-instability signal.

``segmentation-3.0`` is a powerset model (up to 3 speakers / chunk); P(speech)
is derived as ``1 − P(no-speaker)``. The model is gated on HuggingFace; loading
reuses ``ensure_hf_model`` + ``get_huggingface_token`` so cached runs skip the
Hub (constitution VI). No new dependency.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import numpy as np

from senselab.audio.data_structures import Audio
from senselab.utils.data_structures import DeviceType, PyannoteAudioModel, _select_device_and_dtype
from senselab.utils.data_structures.logging import logger
from senselab.utils.data_structures.model import get_huggingface_token
from senselab.utils.dependencies import ensure_hf_model, hf_local_files_only, retry_on_transient_error

if TYPE_CHECKING:
    from pyannote.audio import Inference

try:
    from pyannote.audio import Inference, Model

    PYANNOTEAUDIO_AVAILABLE = True
except (ImportError, RuntimeError):
    PYANNOTEAUDIO_AVAILABLE = False

SEGMENTATION_MODEL_ID = "pyannote/segmentation-3.0"
SEGMENTATION_REVISION = "main"


@dataclass
class FramePosterior:
    """Continuous per-frame speech probability for one audio.

    Attributes:
        probs: ``(num_frames,)`` P(speech) in ``[0, 1]``.
        frame_hop_s: seconds between consecutive frame starts.
        frame_win_s: analysis window per frame (seconds).
    """

    probs: np.ndarray
    frame_hop_s: float
    frame_win_s: float = 0.0

    def mean_std_in_window(self, start_s: float, end_s: float) -> tuple[float, float]:
        """Return ``(mean, std)`` of P(speech) over frames overlapping ``[start, end)``.

        The mean is the bucket's speech-presence contribution; the std captures
        within-bucket temporal instability (a bucket straddling an onset has
        high frame variance). Returns ``(nan, nan)`` when no frame overlaps.
        """
        if self.frame_hop_s <= 0 or self.probs.size == 0:
            return (float("nan"), float("nan"))
        lo = max(0, int(np.floor(start_s / self.frame_hop_s)))
        hi = min(self.probs.size, int(np.ceil(end_s / self.frame_hop_s)))
        if hi <= lo:
            return (float("nan"), float("nan"))
        window = self.probs[lo:hi]
        return (float(np.nanmean(window)), float(np.nanstd(window)))


def _speech_prob_from_output(data: np.ndarray) -> np.ndarray:
    """Reduce a segmentation model's per-frame output to P(speech) in ``[0, 1]``.

    ``segmentation-3.0`` emits per-frame powerset class probabilities that sum
    to ~1 with class 0 = "no speaker"; P(speech) = ``1 − P(class 0)``. If the
    output instead looks multilabel (rows not summing to ~1), fall back to the
    max over the class axis. Validated end-to-end in T044.
    """
    if data.ndim == 1:
        return np.clip(data, 0.0, 1.0)
    row_sums = data.sum(axis=1)
    if np.nanmean(np.abs(row_sums - 1.0)) < 0.1:  # powerset softmax
        return np.clip(1.0 - data[:, 0], 0.0, 1.0)
    return np.clip(data.max(axis=1), 0.0, 1.0)


def _output_to_array(output: Any) -> np.ndarray:  # noqa: ANN401
    """Coerce an Inference result to a ``(num_frames, num_classes)`` float array.

    ``window="whole"`` yields a bare numpy array; sliding modes yield a
    ``SlidingWindowFeature`` whose frame data lives on ``.data``.
    """
    if hasattr(output, "sliding_window") and hasattr(output, "data"):
        return np.asarray(output.data, dtype=np.float64)
    return np.asarray(output, dtype=np.float64)


def _frame_grid(inference: "Inference", num_frames: int, dur_s: float) -> tuple[float, float]:
    """Return ``(frame_hop_s, frame_win_s)`` from the model's receptive field.

    Falls back to ``dur_s / num_frames`` (uniform tiling) when the receptive
    field isn't introspectable.
    """
    try:
        rf = inference.model.receptive_field
        step, duration = float(rf.step), float(rf.duration)
        if step > 0:
            return step, duration
    except (AttributeError, TypeError, ValueError):
        pass
    hop = dur_s / max(1, num_frames)
    return hop, hop


# segmentation-3.0 (and Brouhaha) train on 10 s chunks. For recordings longer
# than one chunk we slide a bounded window and stitch, so memory stays flat and
# we avoid pyannote's "whole-file on a frame-based model" degradation, while
# keeping native ~17 ms frame resolution.
_CHUNK_S = 10.0
_CHUNK_STEP_S = 8.0  # 2 s overlap → smooth stitching across chunk seams


def stitch_frames(
    chunk_arrays: list[np.ndarray],
    chunk_starts_s: list[float],
    hop_s: float,
) -> np.ndarray:
    """Overlap-average per-chunk ``(frames, C)`` arrays into one continuous timeline.

    Each chunk's frame ``i`` maps to absolute frame index ``round(start/hop) + i``;
    indices covered by multiple (overlapping) chunks are averaged. Trailing
    uncovered frames are trimmed. Returns a ``(num_frames, C)`` array. Pure /
    model-free so it is shared by the in-process extractor and the Brouhaha
    subprocess path.
    """
    if not chunk_arrays or hop_s <= 0:
        return np.zeros((0, 1))
    norm = [a[:, None] if a.ndim == 1 else a for a in chunk_arrays]
    n_classes = norm[0].shape[1]
    n_global = 0
    for arr, t0 in zip(norm, chunk_starts_s):
        n_global = max(n_global, int(round(t0 / hop_s)) + arr.shape[0])
    accum = np.zeros((n_global, n_classes), dtype=np.float64)
    count = np.zeros(n_global, dtype=np.float64)
    for arr, t0 in zip(norm, chunk_starts_s):
        base = int(round(t0 / hop_s))
        for i in range(arr.shape[0]):
            g = base + i
            if 0 <= g < n_global:
                accum[g] += arr[i]
                count[g] += 1
    covered = np.nonzero(count > 0)[0]
    last = int(covered[-1]) + 1 if covered.size else 0
    return accum[:last] / np.maximum(count[:last, None], 1.0)


def chunked_frame_inference(
    inference: "Inference",
    audio: Audio,
    chunk_s: float = _CHUNK_S,
    step_s: float = _CHUNK_STEP_S,
) -> tuple[np.ndarray, float, float]:
    """Run per-frame inference over the whole recording, chunking long ones.

    For clips at or under ``chunk_s`` this is a single ``window="whole"`` pass.
    For longer recordings it slides overlapping ``chunk_s`` windows (each a
    bounded whole-window pass), maps every chunk-local frame to an absolute
    frame index, and averages the overlaps — yielding one continuous
    ``(num_frames, num_classes)`` array at native frame resolution with flat
    memory. Returns ``(data, frame_hop_s, frame_win_s)``.
    """
    sr = int(audio.sampling_rate)
    total = int(audio.waveform.shape[-1])
    dur = total / sr if sr else 0.0
    hop, win = _frame_grid(inference, 1, max(dur, 1e-9))

    if dur <= chunk_s or hop <= 0:
        arr = _output_to_array(inference({"waveform": audio.waveform, "sample_rate": sr}))
        if arr.ndim == 1:
            arr = arr[:, None]
        h, w = _frame_grid(inference, arr.shape[0], max(dur, 1e-9))
        return arr, h, w

    chunk_samples = int(chunk_s * sr)
    step_samples = max(1, int(step_s * sr))
    chunk_arrays: list[np.ndarray] = []
    chunk_starts_s: list[float] = []
    start = 0
    while start < total:
        end = min(total, start + chunk_samples)
        if end - start < int(0.1 * sr):  # skip a sub-100ms tail sliver
            break
        arr = _output_to_array(inference({"waveform": audio.waveform[:, start:end], "sample_rate": sr}))
        chunk_arrays.append(arr)
        chunk_starts_s.append(start / sr)
        if end >= total:
            break
        start += step_samples

    data = stitch_frames(chunk_arrays, chunk_starts_s, hop)
    return data, hop, win


_inference_cache: dict[str, "Inference"] = {}


def _get_inference(model: PyannoteAudioModel, device: Optional[DeviceType]) -> "Inference":
    """Load (and cache) a segmentation ``Inference`` for the requested model/device."""
    import torch

    device, _ = _select_device_and_dtype(user_preference=device, compatible_devices=[DeviceType.CUDA, DeviceType.CPU])
    key = f"{model.path_or_uri}-{model.revision}-{device}"
    if key not in _inference_cache:
        ensure_hf_model(str(model.path_or_uri), revision=model.revision, token=get_huggingface_token())
        loaded = retry_on_transient_error(
            Model.from_pretrained,
            model.path_or_uri,
            revision=model.revision,
            token=get_huggingface_token(),
        )
        if loaded is None:
            raise ValueError(f"segmentation model {model.path_or_uri} could not be loaded.")
        # window="whole" returns the model's NATIVE per-frame posterior (~17 ms/frame)
        # for the entire signal. The default (sliding) mode returns a coarse
        # per-chunk aggregate (~1 s step) that would defeat the fine presence grid.
        # Trade-off: "whole" holds the whole signal in one forward pass — fine for
        # the short clinical clips this workflow targets; very long recordings would
        # need chunked stitching (future work).
        _inference_cache[key] = Inference(loaded, window="whole", device=torch.device(device.value))
    return _inference_cache[key]


def extract_speech_frame_posteriors(
    audios: list[Audio],
    model: Optional[PyannoteAudioModel] = None,
    device: Optional[DeviceType] = None,
) -> list[Optional[FramePosterior]]:
    """Return continuous per-frame P(speech) for each audio (``None`` on failure).

    If the model cannot be loaded (not installed, gated without a token,
    native-lib failure), every entry is ``None`` and the presence axis simply
    omits the frame-posterior voter (FR-023) — the workflow does not abort.
    """
    if not PYANNOTEAUDIO_AVAILABLE:
        logger.warning("pyannote-audio unavailable; segmentation frame posteriors will be null.")
        return [None] * len(audios)
    try:
        # PyannoteAudioModel validates id/revision against HF at construction
        # (ValidationError for a gated repo the token can't access), so guard it too.
        if model is None:
            model = PyannoteAudioModel(path_or_uri=SEGMENTATION_MODEL_ID, revision=SEGMENTATION_REVISION)
        hf_local_files_only(str(model.path_or_uri), revision=model.revision)
        inference = _get_inference(model=model, device=device)
    except Exception as exc:  # noqa: BLE001 — any load/access failure degrades to null (FR-023)
        logger.warning(f"Failed to load {SEGMENTATION_MODEL_ID}: {exc}. Frame posteriors will be null.")
        return [None] * len(audios)

    results: list[Optional[FramePosterior]] = []
    for audio in audios:
        try:
            t0 = time.time()
            data, hop_s, win_s = chunked_frame_inference(inference, audio)
            probs = _speech_prob_from_output(data)
            results.append(FramePosterior(probs=probs, frame_hop_s=hop_s, frame_win_s=win_s))
            logger.info(f"segmentation inference took {time.time() - t0:.2f}s ({probs.size} frames @ {hop_s:.4f}s)")
        except (RuntimeError, ValueError, OSError) as exc:
            logger.warning(f"segmentation inference failed for one audio: {exc}")
            results.append(None)
    return results
