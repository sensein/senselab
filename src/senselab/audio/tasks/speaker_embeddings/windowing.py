"""Uniform windowing and per-window speaker-embedding extraction.

Lives in the task layer because two callers need it: the ``audio_analysis`` workflow's
speaker-identity path, and ``estimate_speaker_embedding_from_audios``. It was previously private to
that workflow; a task importing it there would invert the dependency direction, since workflows
compose tasks and not the reverse. Promoted rather than duplicated -- two copies of a windowing
grid drift the moment either is edited.

**The defaults here are the detection defaults, and they are deliberate.** ``window_s=1.0`` with
``hop_s=0.5`` trades embedding precision for temporal resolution: SpeechBrain speaker models are
trained on multi-second utterances so an embedding below 1 s is noisier, but a 1.0 s window on a
0.5 s hop yields one embedding per 0.5 s bucket, which is what eliminated the same-window dedup
that previously dropped half of all consecutive same-cluster comparisons. Profile *enrollment*
wants the opposite trade and passes ``window_s=2.0, hop_s=1.0`` explicitly -- measured at
cross-file centroid stability 0.890 and cross-subject separation 0.168, against 0.331 for a
0.5/0.25 grid carrying four times the windows. Two purposes, two measured settings; do not
collapse them.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from senselab.audio.data_structures import Audio

# Imported from the submodule, not the `speaker_embeddings` package `__init__`: Task 8's estimator
# (`api.py`) imports this module, so importing the package here instead would re-enter `__init__`
# while it is still mid-import and hand back a partially initialized module -- exactly the
# `ImportError: ... partially initialized module` failure this leaf-edge form avoids.
from senselab.audio.tasks.speaker_embeddings.api import extract_speaker_embeddings_from_audios
from senselab.utils.data_structures import DeviceType, SpeechBrainModel


@dataclass(frozen=True)
class WindowEmbedding:
    """One embedding vector for a fixed time window."""

    start_s: float
    end_s: float
    vector: np.ndarray


def slice_audio(audio: Audio, start_s: float, end_s: float) -> Audio:
    """Return a new ``Audio`` covering ``[start_s, end_s]`` (clamped to audio bounds).

    Assumes the input is shaped ``(channels, samples)``. Multichannel audio is
    sliced unchanged — the embedding backend's own preprocessor handles it.
    """
    sr = audio.sampling_rate
    n_samples = audio.waveform.shape[-1]
    start_sample = max(0, int(start_s * sr))
    end_sample = min(n_samples, max(start_sample + 1, int(end_s * sr)))
    waveform = audio.waveform[:, start_sample:end_sample]
    return Audio(waveform=waveform, sampling_rate=sr)


def window_starts(duration_s: float, window_s: float, hop_s: float) -> list[float]:
    """Return window start times spanning ``[0, duration_s]``.

    The last window is anchored to ``duration_s - window_s`` (or 0) so it always
    covers exactly ``window_s`` seconds — the embedding model needs the full
    minimum-length input.
    """
    if duration_s <= 0 or window_s <= 0 or hop_s <= 0:
        return []
    if duration_s <= window_s:
        return [0.0]
    starts: list[float] = []
    t = 0.0
    while t + window_s <= duration_s + 1e-9:
        starts.append(t)
        t += hop_s
    # Pin the last window to the audio tail so we don't drop the final speaker turn.
    last = max(0.0, duration_s - window_s)
    if not starts or starts[-1] < last - 1e-6:
        starts.append(last)
    return starts


def extract_per_window_embeddings(
    *,
    audio: Audio,
    models: list[str],
    window_s: float = 1.0,
    hop_s: float = 0.5,
    device: DeviceType | None = None,
    failures: dict[str, str] | None = None,
) -> dict[str, list[WindowEmbedding]]:
    """Run each embedding model on every fixed window of ``audio``.

    Args:
        audio: The pass's full ``Audio`` object.
        models: HuggingFace model ids for the embedding backends (ECAPA, ResNet, ...).
        window_s: Window length in seconds. Defaults to 1.0.
        hop_s: Window hop in seconds. Defaults to 0.5.
        device: Optional device override.
        failures: Optional dict the function will populate with
            ``{model_id → reason}`` for any model that produced an empty result.
            The caller can fold these into ``incomparable_reasons`` so silent
            empty-vote behavior is auditable.

    Returns:
        ``{model_id → [WindowEmbedding, ...]}``. Each list shares the same window
        grid across models, so ``out[m_a][i]`` and ``out[m_b][i]`` cover the same
        time span. A model that fails to load returns ``[]`` (and writes a reason
        into ``failures`` when provided).
    """
    if not models:
        return {}
    if audio.waveform.ndim < 2:
        raise ValueError(
            f"extract_per_window_embeddings expects a (channels, samples) waveform; "
            f"got shape {tuple(audio.waveform.shape)}."
        )
    duration_s = audio.waveform.shape[-1] / audio.sampling_rate
    starts = window_starts(duration_s, window_s, hop_s)
    if not starts:
        msg = (
            f"audio duration ({duration_s:.3f} s) is shorter than the embedding "
            f"window ({window_s} s); no window grid producible"
        )
        print(f"warn: {msg}", file=sys.stderr)
        if failures is not None:
            for m in models:
                failures[m] = msg
        return {m: [] for m in models}

    audio_slices: list[Audio] = []
    spans: list[tuple[float, float]] = []
    for s in starts:
        e = min(duration_s, s + window_s)
        audio_slices.append(slice_audio(audio, s, e))
        spans.append((s, e))

    out: dict[str, list[WindowEmbedding]] = {}
    for model_id in models:
        try:
            sb_model: SpeechBrainModel = SpeechBrainModel(path_or_uri=model_id, revision="main")
            tensors = extract_speaker_embeddings_from_audios(audios=audio_slices, model=sb_model, device=device)
            entries: list[WindowEmbedding] = []
            for (start_s, end_s), t in zip(spans, tensors, strict=False):
                vec = _flatten_to_1d(t)
                entries.append(WindowEmbedding(start_s=start_s, end_s=end_s, vector=vec))
            out[model_id] = entries
        except Exception as exc:  # noqa: BLE001
            # Surface per-model failure to the caller via stderr so the user can
            # tell "model crashed" from "audio too short" — both produce an empty
            # list, but only one is a real configuration problem.
            msg = f"model failed during extraction: {exc!r}"
            print(
                f"warn: speaker-embedding model {model_id!r} {msg}",
                file=sys.stderr,
            )
            if failures is not None:
                failures[model_id] = msg
            out[model_id] = []
    return out


def _flatten_to_1d(t: Any) -> np.ndarray:  # noqa: ANN401
    """Coerce an embedding tensor / array to a 1-D ``float32`` numpy vector.

    SpeechBrain returns ``(1, D)``, ``(B, D)``, or ``(K, D)`` depending on the
    backend; ``.squeeze()`` would silently leave a 2-D ``(K, D)`` if K>1, which
    the cosine helper then sees as a length-mismatch and drops to ``None``. Here
    we always reshape to a single 1-D vector — when the backend returns multiple
    candidate vectors we average them (the closest the workflow can do without
    backend-specific knowledge).
    """
    if t is None:
        return np.zeros(0, dtype=np.float32)
    if isinstance(t, torch.Tensor):
        arr = t.detach().cpu().numpy()
    else:
        arr = np.asarray(t)
    if arr.size == 0:
        return np.zeros(0, dtype=np.float32)
    while arr.ndim > 1 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim > 1:
        # Multiple candidate vectors → average to a single representative.
        arr = arr.mean(axis=tuple(range(arr.ndim - 1)))
    return arr.astype(np.float32, copy=False)
