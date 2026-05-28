"""Build a SpeakerProfile from a subject's audio files.

This module hosts the per-file speech-window extractor (T009 / FR-002 / FR-008
gating) and — in subsequent tasks — the cross-file aggregation, confidence
policy, per-file keep/drop decisions, session-weighting refinement, and the
``build_speaker_profile`` orchestration entrypoint (T012–T016).

Phase 2 deliverable in this file:

- :class:`TaggedWindowEmbedding` — the file-tagged per-window embedding used
  for leave-one-file-out scoring later (FR-012).
- :func:`extract_speech_windows_for_file` — locate speech via a
  best-available presence gate (diarization + scene-speech mask + loudness;
  opportunistic Whisper / PPG when already cached; never triggers ASR/PPG
  solely to gate) and return ≥~1s windows per model tagged with the source
  ``file_id``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from senselab.audio.data_structures import Audio
from senselab.audio.workflows.audio_analysis.embeddings import (
    WindowEmbedding,
    extract_per_window_embeddings,
)
from senselab.audio.workflows.audio_analysis.presence import speech_window_mask_for_file
from senselab.audio.workflows.speaker_profile import constants as C
from senselab.utils.data_structures import DeviceType

# Default AudioSet labels we treat as "speech-present" for the build-time gate.
# These mirror ``scripts/analyze_audio.py``'s default — keeping them in sync
# means a profile built here uses the same speech definition the identity-axis
# clustering uses inside ``analyze_audio`` (FR-002 wording: "the same signal
# the clustering step consumes").
DEFAULT_SPEECH_PRESENCE_LABELS: tuple[str, ...] = (
    "Speech",
    "Conversation",
    "Narration, monologue",
    "Female speech, woman speaking",
    "Male speech, man speaking",
    "Child speech, kid speaking",
    "Speech synthesizer",
)


@dataclass(slots=True, frozen=True)
class TaggedWindowEmbedding:
    """A :class:`WindowEmbedding` tagged with its source ``file_id`` and model.

    The ``file_id`` tag is what enables leave-one-file-out scoring at compare
    time (FR-012 / R5): when scoring recording *F*, we exclude all tagged
    windows whose ``file_id == F`` from the centroid.
    """

    file_id: str
    model_id: str
    window: WindowEmbedding


def extract_speech_windows_for_file(
    *,
    audio: Audio,
    file_id: str,
    pass_summary: dict[str, Any],
    embedding_models: Sequence[str] = C.DEFAULT_EMBEDDING_MODELS,
    device: DeviceType | None = None,
    profile_window_s: float = C.PROFILE_WINDOW_S,
    profile_hop_s: float = C.PROFILE_HOP_S,
    speech_presence_labels: Sequence[str] = DEFAULT_SPEECH_PRESENCE_LABELS,
    failures: dict[str, str] | None = None,
) -> tuple[list[TaggedWindowEmbedding], dict[str, Any]]:
    """Locate speech windows in one file and embed each per model.

    The gate is **best-available presence**: it consumes the cached
    AST/YAMNet/openSMILE outputs already present in ``pass_summary`` via the
    promoted :func:`speech_window_mask_for_file` helper (T009a). The caller
    is responsible for assembling ``pass_summary`` from whichever tasks they
    have already cached — this function never triggers ASR/PPG itself.

    Args:
        audio: Mono 16 kHz ``Audio`` for one file. The window grid is anchored
            to the audio duration, so audios shorter than ``profile_window_s``
            contribute nothing.
        file_id: Stable identifier for the source file; tagged onto every
            returned window so leave-one-file-out scoring (FR-012) can later
            exclude this file's contribution.
        pass_summary: Dict shaped like ``analyze_audio``'s per-pass summary;
            this function reads ``ast``, ``yamnet``, and ``features.opensmile``
            from it to build the speech mask. Pass an empty/partial dict to
            fall back to "every non-zero window" (the legacy behavior of
            ``speech_window_mask_for_file``).
        embedding_models: HF model ids for the embedding consensus
            (default: ECAPA + ResNet + WavLM — FR-018).
        device: Optional compute device override.
        profile_window_s: Long-window length for centroid-quality embeddings
            (default from ``constants.py``; FR-002 — windows are ≥~1 s
            contiguous speech by construction).
        profile_hop_s: Hop between consecutive long windows.
        speech_presence_labels: AudioSet labels the gate treats as speech.
        failures: Optional dict to populate with per-model load/embed failure
            reasons (mirrors the existing audio_analysis ``failures`` pattern).

    Returns:
        ``(tagged_windows, info)`` where ``tagged_windows`` is the flat list of
        speech-windows tagged with ``file_id`` and ``model_id``, and ``info``
        is a small bookkeeping dict (``speech_seconds``, ``windows_total``,
        ``windows_kept``, ``windows_dropped_non_speech``, ``drop_reason`` if
        the file contributed nothing).
    """
    sr = audio.sampling_rate
    duration_s = audio.waveform.shape[-1] / sr if sr else 0.0

    info: dict[str, Any] = {
        "file_id": file_id,
        "duration_s": float(duration_s),
        "speech_seconds": 0.0,
        "windows_total": 0,
        "windows_kept": 0,
        "windows_dropped_non_speech": 0,
        "drop_reason": None,
    }

    # Hard floor: file shorter than the long window grid can't contribute.
    if duration_s < profile_window_s:
        info["drop_reason"] = "audio_too_short"
        return [], info

    # Extract per-window embeddings per model. The function builds the same
    # window grid for every model, so the speech mask we compute once applies
    # to all of them.
    per_model_windows: dict[str, list[WindowEmbedding]] = extract_per_window_embeddings(
        audio=audio,
        models=list(embedding_models),
        window_s=profile_window_s,
        hop_s=profile_hop_s,
        device=device,
        failures=failures,
    )
    if not per_model_windows or not any(per_model_windows.values()):
        info["drop_reason"] = "no_embedding_windows"
        return [], info

    # Use the first model's window grid for the speech mask (they share the
    # same grid by construction in extract_per_window_embeddings).
    reference_windows: list[WindowEmbedding] = next((w for w in per_model_windows.values() if w), [])
    mask: list[bool] | None = speech_window_mask_for_file(
        entries=reference_windows,
        pass_summary=pass_summary,
        speech_presence_labels=list(speech_presence_labels),
    )
    # ``None`` → no AST/YAMNet/loudness available; keep every window (legacy
    # behavior matches what cluster_pass_speakers does without a mask).
    if mask is None:
        mask = [True] * len(reference_windows)

    info["windows_total"] = len(reference_windows)

    tagged: list[TaggedWindowEmbedding] = []
    kept_window_seconds: list[float] = []
    for i, w in enumerate(reference_windows):
        if i >= len(mask) or not mask[i]:
            info["windows_dropped_non_speech"] += 1
            continue
        for model_id, windows in per_model_windows.items():
            if i >= len(windows):
                continue
            mw = windows[i]
            if mw.vector.size == 0:
                continue
            tagged.append(TaggedWindowEmbedding(file_id=file_id, model_id=model_id, window=mw))
        kept_window_seconds.append(float(w.end_s) - float(w.start_s))

    info["windows_kept"] = len(kept_window_seconds)
    info["speech_seconds"] = float(sum(kept_window_seconds))
    if info["windows_kept"] == 0 and info["drop_reason"] is None:
        info["drop_reason"] = "no_speech_windows"

    return tagged, info
