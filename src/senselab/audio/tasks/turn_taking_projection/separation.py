"""Split a mono two-speaker mix into one channel per speaker, given the caller's diarization.

Three methods, all returning ``(channel_a, channel_b)`` at 16 kHz with the caller's speaker labels:

- ``masking``: each channel is the mix under a smooth mask of that speaker's diarization segments.
  Overlapped speech is duplicated into both channels; nothing is removed from inside a segment.
- ``mossformer2``: ClearerVoice's ``MossFormer2_SS_16K`` two-speaker separator (Apache-2.0).
- ``pyannote``: ``pyannote/speech-separation-ami-1.0`` (MIT, gated), whose pipeline returns
  diarization and separated sources together.

The separators are run over fixed windows, and each window's sources are matched to the caller's
speakers on their own (:func:`assign_sources`), so a separator's arbitrary output order never
carries across windows. The pipeline's own diarization, where it has one, is not used for the
match: the caller's diarization defines who is who.
"""

from __future__ import annotations

import itertools
import json
from functools import lru_cache
from importlib import resources
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from senselab.audio.data_structures import Audio
from senselab.utils.data_structures import DeviceType, ScriptLine

SAMPLING_RATE = 16000

SEPARATION_METHODS = ("masking", "mossformer2", "pyannote")

PYANNOTE_SEPARATION_MODEL = "pyannote/speech-separation-ami-1.0"
PYANNOTE_SEPARATION_REVISION = "9486b106945ae0cc0784041a08bfcdba5edadfb9"
MOSSFORMER2_MODEL = "alibabasglab/MossFormer2_SS_16K"


@lru_cache(maxsize=1)
def parameters() -> Dict[str, Any]:
    """The task's constants from ``data/parameters.json``, by name, values only."""
    text = (
        resources.files("senselab.audio.tasks.turn_taking_projection").joinpath("data", "parameters.json").read_text()
    )
    raw = json.loads(text)
    return {key: entry["value"] for key, entry in raw.items() if isinstance(entry, dict) and "value" in entry}


def choose_speakers(diarization: Sequence[ScriptLine], speakers: Optional[Sequence[str]] = None) -> Tuple[str, str]:
    """The two speakers a dyad is built from.

    Args:
        diarization: Segments with ``speaker``, ``start`` and ``end``.
        speakers: The two labels to use, in channel order. ``None`` takes the two with the most
            speech, ordered by first onset.

    Returns:
        ``(speaker_a, speaker_b)``.

    Raises:
        ValueError: If fewer than two speakers are available or ``speakers`` is not two labels
            present in ``diarization``.
    """
    totals: Dict[str, float] = {}
    first: Dict[str, float] = {}
    for line in diarization:
        if line.speaker is None or line.start is None or line.end is None:
            continue
        totals[line.speaker] = totals.get(line.speaker, 0.0) + max(0.0, line.end - line.start)
        first[line.speaker] = min(first.get(line.speaker, line.start), line.start)
    if speakers is not None:
        if len(speakers) != 2 or speakers[0] == speakers[1]:
            raise ValueError(f"speakers must be two distinct labels, got {list(speakers)}")
        missing = [s for s in speakers if s not in totals]
        if missing:
            raise ValueError(f"speakers {missing} have no segments in the diarization; it has {sorted(totals)}")
        return (speakers[0], speakers[1])
    if len(totals) < 2:
        raise ValueError(f"a dyad needs two speakers in the diarization, found {sorted(totals)}")
    top = sorted(totals, key=lambda s: -totals[s])[:2]
    a, b = sorted(top, key=lambda s: first[s])
    return (a, b)


def speaker_activity(
    diarization: Sequence[ScriptLine], speaker: str, n_samples: int, sampling_rate: int = SAMPLING_RATE
) -> np.ndarray:
    """Boolean per-sample activity of one speaker."""
    active = np.zeros(n_samples, dtype=bool)
    for line in diarization:
        if line.speaker != speaker or line.start is None or line.end is None:
            continue
        lo = max(0, int(round(line.start * sampling_rate)))
        hi = min(n_samples, int(round(line.end * sampling_rate)))
        if hi > lo:
            active[lo:hi] = True
    return active


def smooth_mask(active: np.ndarray, ramp_samples: int) -> np.ndarray:
    """Turn a boolean activity track into a gain in ``[0, 1]`` with raised-cosine edges outside it.

    The gain is 1 on every active sample; each run of activity is extended by a ramp of
    ``ramp_samples`` on each side, rising into the onset and falling after the offset.

    Args:
        active: Per-sample activity.
        ramp_samples: Ramp length; 0 gives a hard mask.

    Returns:
        Float32 gain of the same length.
    """
    gain = active.astype(np.float32)
    if ramp_samples <= 0 or not active.any():
        return gain
    ramp = (0.5 - 0.5 * np.cos(np.pi * np.arange(1, ramp_samples + 1) / (ramp_samples + 1))).astype(np.float32)
    edges = np.diff(active.astype(np.int8), prepend=0, append=0)
    onsets = [int(i) for i in np.flatnonzero(edges == 1)]
    offsets = [int(i) for i in np.flatnonzero(edges == -1)]
    n = active.size
    for onset in onsets:
        lo = max(0, onset - ramp_samples)
        segment = ramp[ramp_samples - (onset - lo) :]
        np.maximum(gain[lo:onset], segment, out=gain[lo:onset])
    for offset in offsets:
        hi = min(n, offset + ramp_samples)
        segment = ramp[::-1][: hi - offset]
        np.maximum(gain[offset:hi], segment, out=gain[offset:hi])
    return gain


def mask_dyad(
    mix: np.ndarray,
    diarization: Sequence[ScriptLine],
    speakers: Tuple[str, str],
    sampling_rate: int = SAMPLING_RATE,
    ramp_seconds: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Channel ``k`` = mix x smooth mask of speaker ``k``'s segments.

    Args:
        mix: Mono samples.
        diarization: Segments.
        speakers: ``(speaker_a, speaker_b)``.
        sampling_rate: Of ``mix``.
        ramp_seconds: Ramp length; ``None`` takes ``mask_ramp_seconds`` from the parameters.

    Returns:
        ``(channel_a, channel_b)``.
    """
    ramp = parameters()["mask_ramp_seconds"] if ramp_seconds is None else ramp_seconds
    ramp_samples = int(round(ramp * sampling_rate))
    out = []
    for speaker in speakers:
        gain = smooth_mask(speaker_activity(diarization, speaker, mix.size, sampling_rate), ramp_samples)
        out.append((mix * gain).astype(np.float32))
    return out[0], out[1]


def _envelope(x: np.ndarray, hop: int) -> np.ndarray:
    n = x.size // hop
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    return np.square(x[: n * hop].astype(np.float64)).reshape(n, hop).mean(axis=1)


def assign_sources(
    sources: np.ndarray,
    diarization: Sequence[ScriptLine],
    speakers: Tuple[str, str],
    offset_seconds: float = 0.0,
    sampling_rate: int = SAMPLING_RATE,
) -> Tuple[Tuple[int, int], np.ndarray]:
    """Match separated sources to the two diarization speakers by where each source has its energy.

    Over the frames where only one speaker is active, each source's share of all sources' energy is
    taken: ``shares[k, s]`` is how much of what was heard while only ``s`` spoke came out of source
    ``k``. The pair of distinct sources maximising the summed share of their speakers wins. A speaker
    with no exclusive frames in the window contributes nothing, so the other speaker decides.

    Args:
        sources: ``(n_sources, n_samples)``, at least two sources.
        diarization: Segments, in recording time.
        speakers: ``(speaker_a, speaker_b)``.
        offset_seconds: Recording time of ``sources``' first sample.
        sampling_rate: Of ``sources``.

    Returns:
        ``((index_a, index_b), shares)``, ``shares`` of shape ``(n_sources, 2)``.

    Raises:
        ValueError: If fewer than two sources are given.
    """
    if sources.ndim != 2 or sources.shape[0] < 2:
        raise ValueError(f"need at least two sources, got shape {sources.shape}")
    hop = max(1, int(round(parameters()["assignment_hop_seconds"] * sampling_rate)))
    n = sources.shape[1]
    shifted = [
        line.model_copy(update={"start": line.start - offset_seconds, "end": line.end - offset_seconds})
        for line in diarization
        if line.start is not None and line.end is not None and line.end > offset_seconds
    ]
    activity = [speaker_activity(shifted, s, n, sampling_rate) for s in speakers]
    frames = n // hop
    act = [a[: frames * hop].reshape(frames, hop).mean(axis=1) > 0.5 for a in activity]
    exclusive = [act[0] & ~act[1], act[1] & ~act[0]]
    energies = np.stack([_envelope(src, hop)[:frames] for src in sources])
    per_speaker = np.stack([energies[:, ex].sum(axis=1) for ex in exclusive], axis=1)
    totals = per_speaker.sum(axis=0, keepdims=True)
    shares = np.divide(per_speaker, totals, out=np.zeros_like(per_speaker), where=totals > 0)
    present = np.array([ex.any() for ex in exclusive], dtype=np.float64)
    best: Tuple[int, int] = (0, 1)
    best_score = -np.inf
    for i, j in itertools.permutations(range(sources.shape[0]), 2):
        score = shares[i, 0] * present[0] + shares[j, 1] * present[1]
        if score > best_score:
            best, best_score = (i, j), score
    return best, shares


def _windows(n_samples: int, sampling_rate: int) -> List[Tuple[int, int]]:
    params = parameters()
    length = int(round(params["separation_chunk_seconds"] * sampling_rate))
    overlap = int(round(params["separation_overlap_seconds"] * sampling_rate))
    if n_samples <= length:
        return [(0, n_samples)]
    step = length - overlap
    starts = list(range(0, max(1, n_samples - overlap), step))
    windows = [(s, min(s + length, n_samples)) for s in starts]
    if windows[-1][1] < n_samples:
        windows.append((n_samples - length, n_samples))
    return windows


def stitch_windows(
    pieces: List[Tuple[int, np.ndarray]],
    n_samples: int,
) -> np.ndarray:
    """Overlap-add ``(start, (2, length))`` pieces with linear cross-fades where they overlap.

    Args:
        pieces: Window starts and their two assigned channels.
        n_samples: Output length.

    Returns:
        ``(2, n_samples)``.
    """
    out = np.zeros((2, n_samples), dtype=np.float64)
    weight = np.zeros(n_samples, dtype=np.float64)
    for index, (start, piece) in enumerate(pieces):
        length = piece.shape[1]
        w = np.ones(length, dtype=np.float64)
        if index > 0:
            prev_start, prev = pieces[index - 1]
            fade = max(0, prev_start + prev.shape[1] - start)
            if fade > 0:
                w[:fade] = np.linspace(0.0, 1.0, fade + 2)[1:-1]
        if index + 1 < len(pieces):
            next_start = pieces[index + 1][0]
            fade = max(0, start + length - next_start)
            if fade > 0:
                w[length - fade :] = np.minimum(w[length - fade :], np.linspace(1.0, 0.0, fade + 2)[1:-1])
        out[:, start : start + length] += piece * w
        weight[start : start + length] += w
    return (out / np.maximum(weight, 1e-8)).astype(np.float32)


def _match_level(mix: np.ndarray, pair: np.ndarray) -> np.ndarray:
    """Scale both channels by one factor so their sum has the mix's RMS (separators renormalise)."""
    mix_rms = float(np.sqrt(np.mean(np.square(mix, dtype=np.float64)))) if mix.size else 0.0
    sum_rms = float(np.sqrt(np.mean(np.square(pair.sum(axis=0), dtype=np.float64)))) if pair.size else 0.0
    return pair * (mix_rms / sum_rms) if sum_rms > 0 else pair


def separate_windows(
    mix: np.ndarray,
    diarization: Sequence[ScriptLine],
    speakers: Tuple[str, str],
    separate: Any,  # noqa: ANN401 -- Callable[[List[np.ndarray]], List[np.ndarray]]
    sampling_rate: int = SAMPLING_RATE,
) -> Tuple[Tuple[np.ndarray, np.ndarray], List[Dict[str, Any]]]:
    """Run a separator window by window, assign each window's sources, and stitch.

    Args:
        mix: Mono samples.
        diarization: Segments.
        speakers: ``(speaker_a, speaker_b)``.
        separate: Maps a list of mono windows to a list of ``(n_sources, length)`` arrays.
        sampling_rate: Of ``mix``.

    Returns:
        ``((channel_a, channel_b), per_window)`` where ``per_window`` records each window's start,
        chosen sources and shares.
    """
    windows = _windows(mix.size, sampling_rate)
    chunks = [mix[lo:hi] for lo, hi in windows]
    separated = separate(chunks)
    pieces: List[Tuple[int, np.ndarray]] = []
    report: List[Dict[str, Any]] = []
    for (lo, hi), chunk, sources in zip(windows, chunks, separated):
        sources = np.asarray(sources, dtype=np.float32)[:, : hi - lo]
        if sources.shape[1] < hi - lo:
            sources = np.pad(sources, ((0, 0), (0, hi - lo - sources.shape[1])))
        (ia, ib), shares = assign_sources(sources, diarization, speakers, lo / sampling_rate, sampling_rate)
        pair = _match_level(chunk, np.stack([sources[ia], sources[ib]]))
        pieces.append((lo, pair))
        report.append({"start_s": lo / sampling_rate, "sources": [int(ia), int(ib)], "shares": shares.tolist()})
    stitched = stitch_windows(pieces, mix.size)
    return (stitched[0], stitched[1]), report


def _mossformer2_separator(device: Optional[DeviceType]) -> Any:  # noqa: ANN401
    from senselab.audio.tasks.source_separation import separate_audios
    from senselab.utils.data_structures import HFModel

    model: HFModel = HFModel(path_or_uri=MOSSFORMER2_MODEL, revision="main")

    def run(chunks: List[np.ndarray]) -> List[np.ndarray]:
        audios = [Audio(waveform=torch.from_numpy(c).unsqueeze(0), sampling_rate=SAMPLING_RATE) for c in chunks]
        results = separate_audios(audios, model=model, device=device)
        return [np.stack([s.waveform.squeeze(0).numpy() for s in sources]) for sources in results]

    return run


def _pyannote_separator(device: Optional[DeviceType]) -> Any:  # noqa: ANN401
    from pyannote.audio import Pipeline
    from senselab_core.runtime.retry import retry_on_transient_error

    from senselab.utils.data_structures import _select_device_and_dtype
    from senselab.utils.data_structures.model import get_huggingface_token
    from senselab.utils.dependencies import resolve_model

    token = get_huggingface_token()
    sha, _ = resolve_model(PYANNOTE_SEPARATION_MODEL, PYANNOTE_SEPARATION_REVISION, token=token)
    pipeline = retry_on_transient_error(
        Pipeline.from_pretrained, checkpoint=PYANNOTE_SEPARATION_MODEL, revision=sha, token=token
    )
    chosen, _ = _select_device_and_dtype(user_preference=device, compatible_devices=[DeviceType.CUDA, DeviceType.CPU])
    if pipeline is None:
        raise RuntimeError(f"{PYANNOTE_SEPARATION_MODEL} did not load")
    pipeline = pipeline.to(torch.device(chosen.value))

    def run(chunks: List[np.ndarray]) -> List[np.ndarray]:
        out = []
        for chunk in chunks:
            result = pipeline({"waveform": torch.from_numpy(chunk).unsqueeze(0), "sample_rate": SAMPLING_RATE})
            sources = result[1] if isinstance(result, tuple) else result.sources
            data = np.asarray(sources.data, dtype=np.float32).T
            if data.shape[0] < 2:
                data = np.concatenate([data, np.zeros((2 - data.shape[0], data.shape[1]), dtype=np.float32)])
            out.append(data)
        return out

    return run


def separate_mix(
    mix: np.ndarray,
    diarization: Sequence[ScriptLine],
    speakers: Tuple[str, str],
    method: str,
    device: Optional[DeviceType] = None,
) -> Tuple[Tuple[np.ndarray, np.ndarray], Dict[str, Any]]:
    """Dispatch to one of :data:`SEPARATION_METHODS` over 16 kHz mono samples.

    Args:
        mix: Mono 16 kHz samples.
        diarization: Segments.
        speakers: ``(speaker_a, speaker_b)``.
        method: One of :data:`SEPARATION_METHODS`.
        device: For the separators.

    Returns:
        ``((channel_a, channel_b), provenance)``.

    Raises:
        ValueError: If ``method`` is unknown.
    """
    if method == "masking":
        return mask_dyad(mix, diarization, speakers), {"method": "masking", **_ramp()}
    if method == "mossformer2":
        pair, report = separate_windows(mix, diarization, speakers, _mossformer2_separator(device))
        return pair, {"method": method, "model": MOSSFORMER2_MODEL, "windows": report}
    if method == "pyannote":
        pair, report = separate_windows(mix, diarization, speakers, _pyannote_separator(device))
        return pair, {
            "method": method,
            "model": PYANNOTE_SEPARATION_MODEL,
            "revision": PYANNOTE_SEPARATION_REVISION,
            "windows": report,
        }
    raise ValueError(f"unknown separation method {method!r}; choose from {SEPARATION_METHODS}")


def _ramp() -> Dict[str, float]:
    return {"ramp_seconds": float(parameters()["mask_ramp_seconds"])}
