"""The continuous per-frame posterior container, and the chunk stitching Brouhaha uses.

No longer loads ``pyannote/segmentation-3.0``: nothing consumes its per-speaker channels
(D-19 moved the count to ``occupancy`` and the binding to ``identity_binding``), and
Brouhaha's VAD head supplies the speech posterior at the same 16.9 ms hop.

Unlike ``detect_human_voice_activity_in_audios`` (which runs the high-level
``Pipeline`` and returns thresholded ``ScriptLine`` segments), this extractor
uses the low-level ``Model`` + ``Inference`` path to obtain the **raw per-frame
speech probability** (~16.9 ms/frame) without any segment thresholding or
hangover smoothing — exactly what the speech_presence axis needs to resolve brief
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


ChannelFormat = str
"""What the columns of an activation matrix mean: ``per_speaker``, ``powerset``, or ``single``."""


def channel_format_for(*, n_columns: int, declared_classes: Optional[list[str]]) -> ChannelFormat:
    """Decide what an output's columns mean, from the model's declaration and the output width.

    Args:
        n_columns: Columns in the returned frame array.
        declared_classes: ``model.specifications.classes``, or ``None`` when unavailable.

    Returns:
        ``"single"`` for one column, ``"per_speaker"`` when the width matches the declared class
        count, ``"powerset"`` when it matches the powerset size, else ``"per_speaker"``.

    **Never inferred from row sums.** ``segmentation-3.0`` declares ``powerset=True``, but
    pyannote 4.x converts to per-speaker activations before returning, so the output has one
    column per speaker. With a single speaker fully active those rows sum to exactly 1.0 — so a
    row-sum test concludes "powerset" and the caller then computes ``1 − data[:, 0]``, treating
    *speaker#1* as the no-speaker class. Measured on real audio that read exactly 1.0000 in 100%
    of frames, including 4 s of digital silence. Width against the declaration cannot make that
    mistake: powerset over 3 speakers is 7 columns, per-speaker is 3.
    """
    if n_columns <= 1:
        return "single"
    if declared_classes:
        n_declared = len(declared_classes)
        if n_columns == n_declared:
            return "per_speaker"
        # Powerset over k speakers with at most 2 concurrent: 1 + k + k(k-1)/2 classes.
        if n_columns == 1 + n_declared + n_declared * (n_declared - 1) // 2:
            return "powerset"
    return "per_speaker"


def collapse_to_speech_prob(data: np.ndarray, *, channel_format: ChannelFormat) -> np.ndarray:
    """Pool an activation matrix into per-frame P(speech) — an **L2** reduction.

    Args:
        data: ``(num_frames, num_channels)`` activations.
        channel_format: What the columns mean; see :func:`channel_format_for`.

    Returns:
        ``(num_frames,)`` P(speech) in ``[0, 1]``.

    ``per_speaker`` uses noisy-or, ``1 − Π(1 − p_k)``, which is the probability calculus for
    "at least one speaker active" — correct here precisely because the channels are still
    available to be combined. ``powerset`` uses ``1 − P(class 0)``. ``single`` bounds and returns.
    """
    if data.ndim == 1:
        return np.clip(data, 0.0, 1.0)
    clipped = np.clip(data, 0.0, 1.0)
    if channel_format == "single" or clipped.shape[1] == 1:
        return clipped[:, 0]
    if channel_format == "powerset":
        return np.clip(1.0 - clipped[:, 0], 0.0, 1.0)
    return np.clip(1.0 - np.prod(1.0 - clipped, axis=1), 0.0, 1.0)


@dataclass
class FramePosterior:
    """Per-frame activations for one audio — the L1 measurement, channels intact.

    Attributes:
        activations: ``(num_frames, num_channels)`` model output. For
            ``segmentation-3.0`` one column per speaker; for a VAD head a single column.
        frame_hop_s: seconds between consecutive frame starts.
        frame_win_s: analysis window per frame (seconds).
        channel_format: what the columns mean; see :func:`channel_format_for`.
        channel_labels: the model's own names for the columns, when it declares them.

    The pooled P(speech) is deliberately **not** a field. Storing a collapse next to the matrix
    it came from lets the two disagree, and consumers read the stored value — which is how a
    reduction that returned exactly 1.0000 everywhere went unnoticed. :meth:`speech_prob`
    computes it on demand from the channels.
    """

    activations: np.ndarray
    frame_hop_s: float
    frame_win_s: float = 0.0
    channel_format: ChannelFormat = "per_speaker"
    channel_labels: tuple[str, ...] = ()

    def speech_prob(self) -> np.ndarray:
        """Per-frame P(speech), pooled from the channels."""
        return collapse_to_speech_prob(self.activations, channel_format=self.channel_format)

    def frame_slice(self, start_s: float, end_s: float) -> Optional[tuple[int, int]]:
        """Frame index range overlapping ``[start, end)``, or ``None`` when empty."""
        n = int(self.activations.shape[0]) if self.activations.ndim >= 1 else 0
        if self.frame_hop_s <= 0 or n == 0:
            return None
        lo = max(0, int(np.floor(start_s / self.frame_hop_s)))
        hi = min(n, int(np.ceil(end_s / self.frame_hop_s)))
        return (lo, hi) if hi > lo else None

    def mean_std_in_window(self, start_s: float, end_s: float) -> tuple[float, float]:
        """Return ``(mean, std)`` of pooled P(speech) over frames overlapping ``[start, end)``.

        The mean is the bucket's speech contribution; the std captures within-bucket temporal
        instability (a bucket straddling an onset has high frame variance). Returns
        ``(nan, nan)`` when no frame overlaps.
        """
        span = self.frame_slice(start_s, end_s)
        if span is None:
            return (float("nan"), float("nan"))
        window = self.speech_prob()[span[0] : span[1]]
        return (float(np.nanmean(window)), float(np.nanstd(window)))

    def per_channel_mean_in_window(self, start_s: float, end_s: float) -> Optional[np.ndarray]:
        """Per-channel mean activation over frames overlapping ``[start, end)``.

        The basis for per-speaker speech_presence: it keeps *which* channel was active, which pooling
        discards. ``None`` when no frame overlaps.
        """
        span = self.frame_slice(start_s, end_s)
        if span is None:
            return None
        return np.asarray(np.nanmean(self.activations[span[0] : span[1], :], axis=0), dtype=np.float64)


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


MIN_SEAM_ACTIVATION = 0.5
"""Activation a chunk overlap must reach before its columns may be permuted.

Below this the overlap holds no confident speaker, the assignment cost is near-uniform, and the
"best" permutation is arbitrary. Since the permutation is applied to the *whole* chunk, acting on
an arbitrary match would scramble frames that were fine. Measured on the validation run: half the
seams of a 72 s recording fell in silence."""


def _match_columns(reference: np.ndarray, candidate: np.ndarray) -> tuple[int, ...]:
    """Column permutation of ``candidate`` that best matches ``reference``, by squared error.

    Hungarian assignment on the pairwise MSE between columns over the frames the two share. Falls
    back to the identity when scipy is unavailable, the shapes disagree, or the overlap carries no
    confident speaker — a wrong permutation is worse than none, so an uncertain match declines
    rather than guesses.
    """
    identity = tuple(range(candidate.shape[1]))
    if reference.shape != candidate.shape or reference.size == 0:
        return identity
    if max(float(reference.max()), float(candidate.max())) < MIN_SEAM_ACTIVATION:
        return identity
    cost = np.empty((candidate.shape[1], reference.shape[1]), dtype=np.float64)
    for j in range(candidate.shape[1]):
        cost[j] = np.mean((reference - candidate[:, j : j + 1]) ** 2, axis=0)
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError:  # pragma: no cover - scipy is a hard dependency in practice
        return identity
    rows, cols = linear_sum_assignment(cost)
    permutation = list(identity)
    for src, dst in zip(rows.tolist(), cols.tolist()):
        permutation[dst] = src
    return tuple(permutation)


def stitch_frames(
    chunk_arrays: list[np.ndarray],
    chunk_starts_s: list[float],
    hop_s: float,
    *,
    align_permutations: bool = False,
) -> np.ndarray:
    """Overlap-average per-chunk ``(frames, C)`` arrays into one continuous timeline.

    Each chunk's frame ``i`` maps to absolute frame index ``round(start/hop) + i``;
    indices covered by multiple (overlapping) chunks are averaged. Trailing
    uncovered frames are trimmed. Returns a ``(num_frames, C)`` array. Pure /
    model-free so it is shared by the in-process extractor and the Brouhaha
    subprocess path.

    Args:
        chunk_arrays: Per-chunk ``(frames, C)`` (or ``(frames,)``) activations.
        chunk_starts_s: Absolute start time of each chunk, seconds.
        hop_s: Frame hop, seconds.
        align_permutations: Match each chunk's columns to the already-stitched timeline before
            averaging. **Required for speaker segmentation, wrong for anything else.**

            A speaker-segmentation model assigns its output channels arbitrarily *per inference*,
            so the same person can be column 0 in one chunk and column 1 in the next. Averaging by
            column index then splits one speaker into two half-strength channels across the whole
            overlap region — which reads downstream as two speakers each half-present, i.e. as
            overlapped speech that never happened. pyannote's own pipeline solves this permutation
            before aggregating; this is the same step.

            It must stay opt-in because it is only sound where the ordering carries no meaning.
            Brouhaha's columns are ``[vad, snr, c50]``: permuting them would swap unrelated
            physical quantities.
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
        if align_permutations and arr.shape[1] > 1 and count.any():
            # Match against what the overlap already holds, not against the previous chunk: the
            # timeline is the thing every later chunk must agree with, and comparing pairwise would
            # let a permutation drift chunk by chunk.
            span = [g for g in range(base, min(n_global, base + arr.shape[0])) if count[g] > 0]
            if span:
                local = [g - base for g in span]
                reference = accum[span] / np.maximum(count[span, None], 1.0)
                arr = arr[:, list(_match_columns(reference, arr[local]))]
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

    # Speaker channels are permutation-arbitrary per inference, so they must be matched to the
    # timeline before averaging (see ``stitch_frames``). Single-column outputs are unaffected.
    data = stitch_frames(chunk_arrays, chunk_starts_s, hop, align_permutations=True)
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
        # per-chunk aggregate (~1 s step) that would defeat the fine speech_presence grid.
        # Trade-off: "whole" holds the whole signal in one forward pass — fine for
        # the short clinical clips this workflow targets; very long recordings would
        # need chunked stitching (future work).
        _inference_cache[key] = Inference(loaded, window="whole", device=torch.device(device.value))
    return _inference_cache[key]


def _declared_classes(inference: "Inference") -> Optional[list[str]]:
    """The model's own names for its output channels, or ``None`` when it declares none."""
    try:
        spec = inference.model.specifications
        # Multi-task models (e.g. Brouhaha) declare a tuple of specifications, one per head. There
        # is no single channel vocabulary in that case, so decline rather than pick a head — a
        # wrong vocabulary would produce a confident and wrong channel-format decision.
        if isinstance(spec, tuple):
            return None
        classes = spec.classes
    except (AttributeError, TypeError):
        return None
    return [str(c) for c in classes] if classes else None
