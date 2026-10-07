"""Public API for health-acoustic representations and health sound event detection.

One model family, ``google/hear``, with two capabilities and therefore two entry points:

* :func:`extract_hear_embeddings_from_audios` / :func:`extract_hear_embeddings_at_times` — the
  512-d encoder;
* :func:`detect_health_acoustic_events` — the bundled 8-label event detector.

Plus :func:`centred_cosine_similarity`, which exists because raw cosine over HeAR embeddings is
uninformative and this module refuses to ship an uncentred one (see its docstring).

All windowing is 2 s. That is not a default, it is the model's only meaningful input length; only
the hop is a parameter. See :mod:`senselab.audio.tasks.health_acoustics.hear` for the
measurements behind that and for why TensorFlow runs in an isolated venv.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.health_acoustics.hear import (
    ENCODER_SUBDIR,
    HEAR_EMBEDDING_DIM,
    HEAR_EVENT_LABELS,
    HEAR_MODEL_ID,
    HEAR_REVISION,
    HEAR_SAMPLING_RATE,
    HEAR_WINDOW_SAMPLES,
    HEAR_WINDOW_SECONDS,
    is_hear_encoder_spec,
    plan_centred_windows,
    plan_scan_windows,
    prepare_audio_for_hear,
    resolve_event_detector,
    run_hear,
    seconds_to_hop_samples,
)
from senselab.utils.compatibility import requires_compatibility
from senselab.utils.data_structures import DeviceType

DEFAULT_EMBEDDING_HOP_SECONDS = 1.0
"""50% overlap. Half a window is the coarsest hop that still sees every 2 s span of the recording
in *some* window at better than half coverage; PR #366 used the same 50% overlap."""

DEFAULT_DETECTION_HOP_SECONDS = 0.25
"""The detector is a presence gate whose response is a box-car of width (event + 2 s), so a hop
much finer than this buys resolution the model does not have; much coarser and a short event's
plateau can fall between windows."""


@dataclass
class HearEmbeddings:
    """HeAR encoder embeddings for one recording, one row per 2 s window.

    Attributes:
        embeddings: ``[n_windows, 512]`` float32.
        window_starts: Start time of each window, in seconds, on the **input recording's**
            timeline. Every window is :data:`HEAR_WINDOW_SECONDS` long and lay wholly inside the
            recording — no row here was computed on padding.
        window_seconds: Always 2.0. Present so a consumer records it rather than assuming it.
        hop_seconds: The requested hop, or ``None`` for windows placed by centre
            (:func:`extract_hear_embeddings_at_times`), where there is no single hop.
        model_id: ``google/hear``.
        revision: The pinned 40-hex commit the embeddings were computed by.
        metadata: Provenance for how the windows were placed — e.g. the times a caller asked
            :func:`extract_hear_embeddings_at_times` for, which ``window_starts`` may have had to
            clamp inward.
    """

    embeddings: torch.Tensor
    window_starts: List[float]
    window_seconds: float = HEAR_WINDOW_SECONDS
    hop_seconds: Optional[float] = None
    model_id: str = HEAR_MODEL_ID
    revision: str = HEAR_REVISION
    metadata: Dict[str, Any] = field(default_factory=dict)

    def pooled(self) -> torch.Tensor:
        """Mean of the per-window embeddings — one ``[512]`` vector for the whole recording.

        This is what PR #366 (this task's predecessor) produced per file, and it is a reasonable
        file-level summary. Two caveats worth stating where the method lives rather than in a
        release note: it is only comparable *after* centring (see
        :func:`centred_cosine_similarity`), and averaging over a recording whose windows contain
        different events averages those events together — a per-event summary wants
        :func:`extract_hear_embeddings_at_times` instead.
        """
        return self.embeddings.mean(dim=0)


def _as_tensor(array: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(array, dtype=np.float32))


@requires_compatibility("audio.tasks.health_acoustics.extract_hear_embeddings_from_audios")
def extract_hear_embeddings_from_audios(
    audios: List[Audio],
    model: str = "hear",
    device: Optional[DeviceType] = None,
    hop_length: float = DEFAULT_EMBEDDING_HOP_SECONDS,
    batch_size: int = 8,
) -> List[HearEmbeddings]:
    """Extract HeAR embeddings over a sliding 2 s window.

    The window length is fixed at 2 s and is not a parameter: the encoder accepts other lengths
    without complaint but the representation degrades sharply away from 2 s (centred class margin
    +0.91 at 2 s, +0.46 at 1 s, +0.29 at 0.3 s — and 3 s is worse than 2 s). Only ``hop_length``
    is yours to choose.

    Windows never contain padding. Each one lies wholly inside the recording, and the last is
    snapped to end at the final sample rather than starting on the hop grid, so the tail is
    covered without a short window being padded out.

    Args:
        audios: Recordings of any sampling rate; each is resampled to 16 kHz and, if
            multichannel, averaged to mono. Each must be at least 2 s **after** resampling.
        model: ``"hear"`` (equivalently ``"google/hear"`` / ``"hear-encoder"``). A plain string,
            not an ``HFModel``: nothing in transformers can load this SavedModel.
        device: ``DeviceType.CPU`` or ``DeviceType.CUDA``; ``None`` auto-selects. MPS is not a
            TensorFlow device, so it is not offered.
        hop_length: Hop between window starts, in seconds. Default 1.0 (50% overlap).
        batch_size: Windows per graph call. Raise it to trade memory for throughput; the encoder's
            batch dimension is free.

    Returns:
        One :class:`HearEmbeddings` per input, in order.

    Raises:
        ValueError: If ``model`` is not a HeAR encoder spec, if ``hop_length`` is not positive, or
            if any audio is shorter than 2 s at 16 kHz (padding it would corrupt the embedding —
            see :func:`extract_hear_embeddings_at_times` for the "short event in a long
            recording" case).

    Examples:
        >>> audios = [Audio(filepath="cough_recording.wav")]
        >>> [result] = extract_hear_embeddings_from_audios(audios, hop_length=0.5)
        >>> result.embeddings.shape[1]
        512
    """
    if not is_hear_encoder_spec(model):
        raise ValueError(
            f"{model!r} is not the HeAR encoder. Use 'hear'. For the bundled event detector, "
            "call detect_health_acoustic_events()."
        )
    if not audios:
        return []

    hop_samples = seconds_to_hop_samples(hop_length)
    prepared = [prepare_audio_for_hear(audio) for audio in audios]
    plans = [plan_scan_windows(audio.waveform.shape[-1], hop_samples) for audio in prepared]

    arrays = run_hear(prepared, plans, subdir=ENCODER_SUBDIR, batch_size=batch_size, device=device)

    results = []
    for array, starts in zip(arrays, plans):
        if array.shape[-1] != HEAR_EMBEDDING_DIM:
            raise ValueError(f"HeAR returned {array.shape[-1]}-d embeddings, expected {HEAR_EMBEDDING_DIM}")
        results.append(
            HearEmbeddings(
                embeddings=_as_tensor(array),
                window_starts=[s / HEAR_SAMPLING_RATE for s in starts],
                hop_seconds=hop_samples / HEAR_SAMPLING_RATE,
            )
        )
    return results


@requires_compatibility("audio.tasks.health_acoustics.extract_hear_embeddings_from_audios")
def extract_hear_embeddings_at_times(
    audio: Audio,
    times: Sequence[float],
    model: str = "hear",
    device: Optional[DeviceType] = None,
    batch_size: int = 8,
) -> HearEmbeddings:
    """Extract one HeAR embedding per event, from 2 s of the **real recording** around it.

    This is the entry point for "I have a 0.3 s cough at t=9.58 s". The alternative a caller
    reaches for — cutting the event out and padding it to 2 s — is measurably wrong: padding moves
    the embedding about as far as replacing the audio with something unrelated (centred cosine
    0.0-0.5, against a class margin of ~0.9). So the window is taken from the recording itself,
    centred on the requested time.

    Near an edge the window cannot be centred and stay inside the recording; it slides inward to
    the nearest legal position and ``window_starts`` reports where it landed, so the shift is
    visible rather than silent. That trade is cheap by measurement: 50-200 ms of framing error
    costs cosine 0.93-0.98, padding costs an order of magnitude more.

    Args:
        audio: The full recording — not a pre-cut clip. Must be at least 2 s at 16 kHz.
        times: Times of interest in seconds, on ``audio``'s timeline. Order is preserved, so
            result row *i* is the window for ``times[i]``.
        model: ``"hear"``.
        device: See :func:`extract_hear_embeddings_from_audios`.
        batch_size: Windows per graph call.

    Returns:
        A :class:`HearEmbeddings` with one row per requested time and ``hop_seconds=None``.

    Raises:
        ValueError: If ``times`` is empty, a time lies outside the recording, the recording is
            shorter than 2 s, or ``model`` is not a HeAR encoder spec.
    """
    if not is_hear_encoder_spec(model):
        raise ValueError(f"{model!r} is not the HeAR encoder. Use 'hear'.")
    if len(times) == 0:
        raise ValueError("times is empty; pass at least one time of interest")

    prepared = prepare_audio_for_hear(audio)
    centres = [int(round(float(t) * HEAR_SAMPLING_RATE)) for t in times]
    starts = plan_centred_windows(prepared.waveform.shape[-1], centres)

    [array] = run_hear([prepared], [starts], subdir=ENCODER_SUBDIR, batch_size=batch_size, device=device)
    return HearEmbeddings(
        embeddings=_as_tensor(array),
        window_starts=[s / HEAR_SAMPLING_RATE for s in starts],
        hop_seconds=None,
        metadata={"requested_times": [float(t) for t in times]},
    )


@requires_compatibility("audio.tasks.health_acoustics.detect_health_acoustic_events")
def detect_health_acoustic_events(
    audios: List[Audio],
    model: str = "hear-event-detector",
    device: Optional[DeviceType] = None,
    hop_length: float = DEFAULT_DETECTION_HOP_SECONDS,
    top_k: Optional[int] = None,
) -> List[List[Dict[str, Any]]]:
    """Detect health sounds with HeAR's bundled event detector, over a sliding 2 s window.

    The eight labels are :data:`HEAR_EVENT_LABELS`. They are **independent presence
    probabilities**, not a distribution: they do not sum to 1, and several can be high at once.

    **This is a presence gate, not a locator.** 40 ms of cough anywhere inside the 2 s window is
    enough to push its probability past 0.5, so the response to an event of duration D is a
    box-car roughly (D + 2 s) wide, and two events less than 2 s apart merge into one plateau.
    Read a run of high windows as "something happened in this neighbourhood", never as the
    event's extent. For onsets and offsets, use a task built to localise.

    The window is fixed at 2 s because the detector's graph rejects every other length outright
    (``InvalidArgumentError`` at 0.5, 1.0, 1.5, 3.0 and 4.0 s), and its batch dimension is pinned
    at 1, so windows are fed one at a time regardless of how many there are.

    Args:
        audios: Recordings of any sampling rate; resampled to 16 kHz, mono-averaged. Each must be
            at least 2 s after resampling — the detector cannot be given a padded short clip, and
            senselab will not pad one.
        model: ``"hear-event-detector"`` (MobileNetV3-Large, ~3 M parameters, the default) or
            ``"hear-event-detector-small"`` (MobileNetV3-Small, ~1 M). Both share one frontend
            and one label set.
        device: See :func:`extract_hear_embeddings_from_audios`.
        hop_length: Hop between window starts, in seconds. Default 0.25.
        top_k: Keep only the ``k`` highest-scoring labels per window. ``None`` (default) keeps all
            eight — for a multi-label gate, dropping labels drops the negative evidence too.

    Returns:
        Per audio, a list of per-window dicts with ``start``, ``end``, ``label_scores``
        (descending, one single-key dict per label, the shape
        :func:`senselab.audio.tasks.classification.scene_results_to_segments` consumes),
        ``win_length`` and ``hop_length``.

    Raises:
        ValueError: If ``model`` is not a HeAR detector spec, ``hop_length`` is not positive, or
            an audio is shorter than 2 s at 16 kHz.
    """
    subdir = resolve_event_detector(model)
    if not audios:
        return []

    hop_samples = seconds_to_hop_samples(hop_length)
    prepared = [prepare_audio_for_hear(audio) for audio in audios]
    plans = [plan_scan_windows(audio.waveform.shape[-1], hop_samples) for audio in prepared]

    # batch_size=1: the detectors' signature pins the batch dimension at 1 and the worker lowers
    # it anyway; saying so here keeps the payload honest about what will happen.
    arrays = run_hear(prepared, plans, subdir=subdir, batch_size=1, device=device)

    hop_seconds = hop_samples / HEAR_SAMPLING_RATE
    results: List[List[Dict[str, Any]]] = []
    for array, starts in zip(arrays, plans):
        if array.shape[-1] != len(HEAR_EVENT_LABELS):
            raise ValueError(
                f"HeAR detector returned {array.shape[-1]} scores, expected {len(HEAR_EVENT_LABELS)} "
                f"({', '.join(HEAR_EVENT_LABELS)})"
            )
        windows: List[Dict[str, Any]] = []
        for row, start in zip(array, starts):
            order = np.argsort(row)[::-1]
            if top_k is not None:
                order = order[:top_k]
            windows.append(
                {
                    "start": start / HEAR_SAMPLING_RATE,
                    "end": (start + HEAR_WINDOW_SAMPLES) / HEAR_SAMPLING_RATE,
                    "label_scores": [{HEAR_EVENT_LABELS[i]: float(row[i])} for i in order],
                    "win_length": HEAR_WINDOW_SECONDS,
                    "hop_length": hop_seconds,
                }
            )
        results.append(windows)
    return results


def centred_cosine_similarity(
    embeddings: Union[torch.Tensor, HearEmbeddings],
    reference: Optional[Union[torch.Tensor, HearEmbeddings]] = None,
    mean: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Cosine similarity between HeAR embeddings **after mean-centring**, which is not optional.

    Raw cosine over HeAR embeddings barely separates classes: measured over cough / breath /
    speech / silence windows from one recording, within-class 0.977 against between-class 0.918 —
    a 0.06 margin sitting on top of a large shared component every embedding carries. Subtract the
    mean of the pool and the same windows give +0.653 within against -0.256 between, and
    leave-one-out nearest-neighbour classification reaches 0.846 with no training at all.

    A raw-cosine helper is therefore deliberately absent from this module rather than offered
    alongside: the uncentred number looks confident and is nearly uninformative.

    Args:
        embeddings: ``[n, 512]`` (or a :class:`HearEmbeddings`).
        reference: Optional ``[m, 512]`` to compare against. ``None`` compares ``embeddings``
            against itself.
        mean: Optional centring vector, ``[512]``. Default: the mean over every row supplied
            (``embeddings`` and ``reference`` together). Pass one explicitly to centre a small
            query set against a larger pool's mean — the mean should describe the recording or
            session the comparison lives in, since that shared component is what centring removes.

    Returns:
        ``[n, m]`` cosine similarities of the centred vectors (``[n, n]`` when ``reference`` is
        ``None``).

    Raises:
        ValueError: If a tensor is not 2-D ``[*, 512]``, or if fewer than two vectors are
            available to estimate the mean from and none was supplied — a "mean" over one vector
            centres it to zero and makes every similarity undefined.
    """
    left = _matrix(embeddings, "embeddings")
    right = left if reference is None else _matrix(reference, "reference")

    if mean is None:
        pool = left if reference is None else torch.cat([left, right], dim=0)
        if pool.shape[0] < 2:
            raise ValueError(
                "centring needs at least two vectors to estimate a mean from (got "
                f"{pool.shape[0]}); pass mean=<[512] tensor> computed over the recording or "
                "session this comparison belongs to"
            )
        mean = pool.mean(dim=0)
    mean = mean.reshape(-1)
    if mean.shape[0] != left.shape[1]:
        raise ValueError(f"mean has {mean.shape[0]} dimensions, expected {left.shape[1]}")

    a = torch.nn.functional.normalize(left - mean, dim=1)
    b = torch.nn.functional.normalize(right - mean, dim=1)
    return a @ b.T


def _matrix(value: Union[torch.Tensor, HearEmbeddings], name: str) -> torch.Tensor:
    tensor = value.embeddings if isinstance(value, HearEmbeddings) else value
    if tensor.ndim != 2 or tensor.shape[1] != HEAR_EMBEDDING_DIM:
        raise ValueError(f"{name} must be [n, {HEAR_EMBEDDING_DIM}], got shape {tuple(tensor.shape)}")
    return tensor.to(torch.float32)


__all__: Tuple[str, ...] = (
    "HearEmbeddings",
    "HEAR_EVENT_LABELS",
    "centred_cosine_similarity",
    "detect_health_acoustic_events",
    "extract_hear_embeddings_at_times",
    "extract_hear_embeddings_from_audios",
)
