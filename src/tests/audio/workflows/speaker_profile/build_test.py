"""Tests for the speaker-profile build path.

These exercise the *pure* aggregation / confidence / keep-drop logic with
injected synthetic embedding vectors (no model downloads) and the full
``build_speaker_profile`` orchestration with ``extract_per_window_embeddings``
monkeypatched to a deterministic stub. Covers:

- Exactly one profile + a usage record per file.
- Contamination tolerance: with a minority of intruder windows the centroid is
  closer to the held-out target than to the intruder.
- Non-speech / sub-window files auto-dropped with a recorded reason.
- ``ok`` / ``low`` / ``insufficient`` confidence boundaries.
- Balanced 50/50 → ``ambiguous``; dominant ~85/15 → confident.
- Optional same-session weighting steers a near-tie without erasing other
  sessions, and is a no-op without ``prefer_session``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from senselab.audio.data_structures import Audio
from senselab.audio.data_structures.audio_plus import AudioPlus, SpeakerInfo
from senselab.audio.workflows.audio_analysis.embeddings import WindowEmbedding
from senselab.audio.workflows.speaker_profile import constants as C
from senselab.audio.workflows.speaker_profile.build import (
    ProfileInput,
    TaggedWindowEmbedding,
    aggregate_dominant_cluster,
    build_source_records,
    build_speaker_profile,
    decide_confidence,
    profile_from_related_audios,
)

# Single low-dim model keeps the synthetic clustering fast and deterministic.
_MODEL = C.ECAPA_MODEL_ID
_DIM = 16
_WINDOW_S = 2.0


def _basis(idx: int, dim: int = _DIM) -> np.ndarray:
    """Unit basis vector e_idx — orthogonal speakers are trivially separable."""
    v = np.zeros(dim, dtype=np.float64)
    v[idx % dim] = 1.0
    return v


def _cluster_windows(
    rng: np.random.Generator,
    center: np.ndarray,
    n: int,
    *,
    file_id: str,
    model: str = _MODEL,
    noise: float = 0.05,
    start0: float = 0.0,
) -> list[TaggedWindowEmbedding]:
    """``n`` windows scattered tightly around ``center``, tagged to one file."""
    out: list[TaggedWindowEmbedding] = []
    t = start0
    for _ in range(n):
        vec = (center + rng.normal(0.0, noise, center.shape)).astype(np.float32)
        out.append(
            TaggedWindowEmbedding(
                file_id=file_id,
                model_id=model,
                window=WindowEmbedding(start_s=t, end_s=t + _WINDOW_S, vector=vec),
            )
        )
        t += _WINDOW_S
    return out


# ──────────────────────────────────────────────────────────────────────────
# decide_confidence


def test_confidence_ok_above_floor() -> None:
    """Aggregate speech ≥ the floor with no rival cluster → ``ok``."""
    assert decide_confidence(dominant_speech_seconds=40.0, runner_up_speech_seconds=0.0, has_dominant=True) == "ok"


def test_confidence_low_below_floor() -> None:
    """A coherent but thin dominant cluster (below the floor) → ``low``."""
    assert decide_confidence(dominant_speech_seconds=5.0, runner_up_speech_seconds=0.0, has_dominant=True) == "low"


def test_confidence_insufficient_when_no_dominant() -> None:
    """No usable dominant cluster → ``insufficient`` (declined)."""
    assert (
        decide_confidence(dominant_speech_seconds=0.0, runner_up_speech_seconds=0.0, has_dominant=False)
        == "insufficient"
    )


def test_confidence_ambiguous_for_balanced_top_two() -> None:
    """A near-equal runner-up (ratio ≥ AMBIGUITY_SHARE_RATIO) → ``ambiguous``."""
    assert (
        decide_confidence(dominant_speech_seconds=40.0, runner_up_speech_seconds=38.0, has_dominant=True) == "ambiguous"
    )


def test_confidence_not_ambiguous_for_dominant_split() -> None:
    """An ~85/15 split (ratio 0.18 < 0.80) stays confident, not ambiguous."""
    assert decide_confidence(dominant_speech_seconds=85.0, runner_up_speech_seconds=15.0, has_dominant=True) == "ok"


# ──────────────────────────────────────────────────────────────────────────
# aggregate_dominant_cluster


def test_single_speaker_profile_has_one_centroid_close_to_target() -> None:
    """A clean single-speaker pool → one centroid pointing at the speaker."""
    rng = np.random.default_rng(0)
    target = _basis(0)
    windows = _cluster_windows(rng, target, 12, file_id="sub/clean.wav")

    res = aggregate_dominant_cluster(windows, embedding_models=[_MODEL])
    assert res is not None
    assert list(res.centroids) == [_MODEL]
    centroid = np.asarray(res.centroids[_MODEL])
    assert float(centroid @ target) > 0.99
    assert res.dominant_cluster.n_windows == 12
    assert res.aggregate_speech_seconds == res.dominant_cluster.speech_seconds


def test_contamination_tolerance_centroid_closer_to_target_than_intruder() -> None:
    """A minority of intruder windows does not pull the centroid off the target."""
    rng = np.random.default_rng(1)
    target = _basis(0)
    intruder = _basis(1)
    # 16 target + 4 intruder = 20% contamination.
    windows = _cluster_windows(rng, target, 16, file_id="sub/target.wav")
    windows += _cluster_windows(rng, intruder, 4, file_id="sub/intruder.wav", start0=100.0)

    res = aggregate_dominant_cluster(windows, embedding_models=[_MODEL])
    assert res is not None
    centroid = np.asarray(res.centroids[_MODEL])
    sim_target = float(centroid @ target)
    sim_intruder = float(centroid @ intruder)
    assert sim_target > sim_intruder
    assert sim_target > 0.9
    # Intruder file contributed no dominant-cluster windows.
    assert "sub/intruder.wav" not in res.per_file_dominant


def test_balanced_two_speakers_flag_ambiguous() -> None:
    """A 50/50 pool yields a near-equal runner-up → ambiguous."""
    rng = np.random.default_rng(2)
    a = _basis(0)
    b = _basis(1)
    windows = _cluster_windows(rng, a, 10, file_id="sub/a.wav")
    windows += _cluster_windows(rng, b, 10, file_id="sub/b.wav", start0=100.0)

    res = aggregate_dominant_cluster(windows, embedding_models=[_MODEL])
    assert res is not None
    assert res.runner_up_cluster is not None
    ratio = res.runner_up_cluster.speech_seconds / res.dominant_cluster.speech_seconds
    assert ratio >= C.AMBIGUITY_SHARE_RATIO
    confidence = decide_confidence(
        dominant_speech_seconds=res.dominant_cluster.speech_seconds,
        runner_up_speech_seconds=res.runner_up_cluster.speech_seconds,
        has_dominant=bool(res.centroids),
    )
    assert confidence == "ambiguous"


def test_dominant_split_is_confident_not_ambiguous() -> None:
    """An ~85/15 pool keeps a clear dominant → confident."""
    rng = np.random.default_rng(3)
    target = _basis(0)
    other = _basis(1)
    windows = _cluster_windows(rng, target, 17, file_id="sub/target.wav")
    windows += _cluster_windows(rng, other, 3, file_id="sub/other.wav", start0=100.0)

    res = aggregate_dominant_cluster(windows, embedding_models=[_MODEL])
    assert res is not None
    runner_up_s = res.runner_up_cluster.speech_seconds if res.runner_up_cluster else 0.0
    confidence = decide_confidence(
        dominant_speech_seconds=res.dominant_cluster.speech_seconds,
        runner_up_speech_seconds=runner_up_s,
        has_dominant=bool(res.centroids),
    )
    assert confidence == "ok"


def test_aggregate_returns_none_for_empty_pool() -> None:
    """An empty pooled-window list yields ``None`` (nothing to aggregate)."""
    assert aggregate_dominant_cluster([], embedding_models=[_MODEL]) is None


def test_session_weighting_breaks_a_near_tie() -> None:
    """Equal-size clusters → ``--prefer-session`` selects the preferred one.

    Two orthogonal speakers with the same window count are a genuine tie on raw
    seconds; up-weighting one session's windows makes that cluster dominant.
    """
    rng = np.random.default_rng(4)
    a = _basis(0)
    b = _basis(1)
    win_a = _cluster_windows(rng, a, 8, file_id="sub/sesA.wav")
    win_b = _cluster_windows(rng, b, 8, file_id="sub/sesB.wav", start0=100.0)
    windows = win_a + win_b
    session_of_file = {"sub/sesA.wav": "ses-A", "sub/sesB.wav": "ses-B"}

    res = aggregate_dominant_cluster(
        windows,
        embedding_models=[_MODEL],
        prefer_session="ses-B",
        session_of_file=session_of_file,
    )
    assert res is not None
    centroid = np.asarray(res.centroids[_MODEL])
    # The preferred session's speaker (b) should win the dominant slot.
    assert float(centroid @ b) > float(centroid @ a)


# ──────────────────────────────────────────────────────────────────────────
# build_source_records


def test_source_records_keep_drop_and_reasons() -> None:
    """One record per file with keep flags and mapped drop reasons."""
    rng = np.random.default_rng(5)
    target = _basis(0)
    intruder = _basis(1)
    windows = _cluster_windows(rng, target, 16, file_id="sub/target.wav")
    windows += _cluster_windows(rng, intruder, 4, file_id="sub/intruder.wav", start0=100.0)
    res = aggregate_dominant_cluster(windows, embedding_models=[_MODEL])
    assert res is not None

    file_infos: list[dict[str, Any]] = [
        {"file_id": "sub/target.wav", "drop_reason": None},
        {"file_id": "sub/intruder.wav", "drop_reason": None},
        {"file_id": "sub/silent.wav", "drop_reason": "no_speech_windows"},
        {"file_id": "sub/tiny.wav", "drop_reason": "audio_too_short"},
    ]
    signatures = {str(fi["file_id"]): f"sig-{i}" for i, fi in enumerate(file_infos)}

    records = build_source_records(
        file_infos=file_infos,
        aggregation=res,
        audio_signatures=signatures,
    )
    by_id = {r.file_id: r for r in records}
    assert len(records) == 4

    assert by_id["sub/target.wav"].kept is True
    assert by_id["sub/target.wav"].drop_reason is None
    assert by_id["sub/target.wav"].windows_used > 0

    # Clustered but minority → outside the dominant cluster.
    assert by_id["sub/intruder.wav"].kept is False
    assert by_id["sub/intruder.wav"].drop_reason == "outside_dominant_cluster"

    # Never produced windows → mapped extractor reasons.
    assert by_id["sub/silent.wav"].kept is False
    assert by_id["sub/silent.wav"].drop_reason == "non_speech_task"
    assert by_id["sub/tiny.wav"].drop_reason == "insufficient_speech"


# ──────────────────────────────────────────────────────────────────────────
# build_speaker_profile orchestration — stubbed models


def _stub_extractor(
    monkeypatch: pytest.MonkeyPatch, vectors_by_file: dict[str, np.ndarray], n_windows: int = 12
) -> None:
    """Patch ``extract_per_window_embeddings`` to emit deterministic windows.

    Each file gets ``n_windows`` 2 s windows scattered around its assigned
    center vector, so the whole pipeline (extract → cluster → aggregate → save)
    runs without downloading any speaker-embedding model.
    """

    def _fake(*, audio: Audio, models, window_s, hop_s, device=None, failures=None, cache_dir=None):  # noqa: ANN001, ANN202
        # Identify the file by matching waveform length → center vector.
        n_samples = audio.waveform.shape[-1]
        rng = np.random.default_rng(int(n_samples))
        center = vectors_by_file[str(n_samples)]
        out: dict[str, list[WindowEmbedding]] = {}
        for m in models:
            entries: list[WindowEmbedding] = []
            t = 0.0
            for _ in range(n_windows):
                vec = (center + rng.normal(0.0, 0.05, center.shape)).astype(np.float32)
                entries.append(WindowEmbedding(start_s=t, end_s=t + window_s, vector=vec))
                t += hop_s
            out[m] = entries
        return out

    monkeypatch.setattr(
        "senselab.audio.workflows.speaker_profile.build.extract_per_window_embeddings",
        _fake,
    )


def _audio_of_length(samples: int) -> Audio:
    import torch

    return Audio(waveform=torch.zeros(1, samples), sampling_rate=16000)


def test_build_speaker_profile_one_profile_and_records(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A subject's files → exactly one profile + one record per file."""
    target = _basis(0)
    # Two target files (distinct lengths so the stub can tell them apart).
    a = _audio_of_length(160000)  # 10 s
    b = _audio_of_length(176000)  # 11 s
    centers = {"160000": target, "176000": target}
    _stub_extractor(monkeypatch, centers)

    out = tmp_path / "profile.json"
    profile = build_speaker_profile(
        "sub-001",
        [
            ProfileInput(audio=a, file_id="sub-001/ses-1/a.wav", session_id="ses-1"),
            ProfileInput(audio=b, file_id="sub-001/ses-1/b.wav", session_id="ses-1"),
        ],
        embedding_models=[_MODEL],
        output=out,
    )

    assert profile.subject_id == "sub-001"
    assert list(profile.centroids) == [_MODEL]
    assert len(profile.sources) == 2
    assert all(s.kept for s in profile.sources)
    assert profile.confidence in {"ok", "low"}
    # Artifact written and the runner-up invariant holds.
    assert out.exists()
    assert (profile.runner_up_cluster is not None) == (profile.confidence == "ambiguous")


def test_build_speaker_profile_insufficient_subject(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A sub-window file produces an ``insufficient`` declined profile."""
    # 0.5 s audio < the 2 s profile window → no windows extractable.
    a = _audio_of_length(8000)
    _stub_extractor(monkeypatch, {"8000": _basis(0)})

    profile = build_speaker_profile(
        "sub-thin",
        [ProfileInput(audio=a, file_id="sub-thin/x.wav")],
        embedding_models=[_MODEL],
    )
    assert profile.confidence == "insufficient"
    assert profile.centroids == {}
    assert len(profile.sources) == 1
    assert profile.sources[0].kept is False


# ──────────────────────────────────────────────────────────────────────────
# Enrollment from an Audio+ bundle's related recordings


def _audio_plus_with_related(refs: list[str], speaker_id: str | None = "sub-001") -> AudioPlus:
    """An Audio+ whose related refs are ``refs`` (never including its own ref)."""
    return AudioPlus(
        ref="sub-001/ses-1/analyzed.wav",
        audio=_audio_of_length(160000),
        speaker=SpeakerInfo(speaker_id=speaker_id),
        related_audio_refs=refs,
    )


def test_profile_from_related_audios_enrolls_on_siblings_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """Enrollment uses the related refs, and the analyzed recording is never a source.

    This is the leave-one-out property: the provider excludes the queried recording, so a
    profile built from what it returns cannot include the file it will be scored against.
    """
    target = _basis(0)
    _stub_extractor(monkeypatch, {"160000": target, "176000": target, "192000": target})
    lengths = {"sib-a.wav": 176000, "sib-b.wav": 192000}
    loaded: list[str] = []

    def loader(ref: str) -> Audio:
        loaded.append(ref)
        return _audio_of_length(lengths[ref])

    ap = _audio_plus_with_related(["sib-a.wav", "sib-b.wav"])
    profile = profile_from_related_audios(ap, audio_loader=loader, embedding_models=[_MODEL])

    assert profile is not None
    assert profile.subject_id == "sub-001"
    assert loaded == ["sib-a.wav", "sib-b.wav"]  # one load per sibling, lazily
    source_ids = {s.file_id for s in profile.sources}
    assert source_ids == {"sib-a.wav", "sib-b.wav"}
    assert ap.ref not in source_ids


def test_profile_from_related_audios_returns_none_without_siblings() -> None:
    """No related recordings → no profile, rather than a profile of one file."""
    ap = _audio_plus_with_related([])
    assert profile_from_related_audios(ap, audio_loader=lambda _r: _audio_of_length(160000)) is None


def test_profile_from_related_audios_skips_unloadable_siblings(monkeypatch: pytest.MonkeyPatch) -> None:
    """A sibling that fails to load is skipped and recorded, not fatal."""
    _stub_extractor(monkeypatch, {"176000": _basis(0)})
    failures: dict[str, str] = {}

    def loader(ref: str) -> Audio:
        if ref == "broken.wav":
            raise OSError("unreadable")
        return _audio_of_length(176000)

    ap = _audio_plus_with_related(["broken.wav", "ok.wav"])
    profile = profile_from_related_audios(ap, audio_loader=loader, embedding_models=[_MODEL], load_failures=failures)

    assert profile is not None
    assert {s.file_id for s in profile.sources} == {"ok.wav"}
    assert "broken.wav" in failures


def test_profile_from_related_audios_requires_a_subject_id() -> None:
    """With no speaker id and no override there is nothing to key the profile on."""
    ap = _audio_plus_with_related(["sib-a.wav"], speaker_id=None)
    with pytest.raises(ValueError, match="subject_id"):
        profile_from_related_audios(ap, audio_loader=lambda _r: _audio_of_length(176000))


def test_profile_from_related_audios_records_the_build_grid(monkeypatch: pytest.MonkeyPatch) -> None:
    """The profile stamps the grid it was built at, so a consumer can check compatibility."""
    _stub_extractor(monkeypatch, {"176000": _basis(0)})
    ap = _audio_plus_with_related(["sib-a.wav"])
    profile = profile_from_related_audios(
        ap,
        audio_loader=lambda _r: _audio_of_length(176000),
        embedding_models=[_MODEL],
        profile_window_s=2.0,
        profile_hop_s=1.0,
    )
    assert profile is not None
    assert profile.params is not None
    assert profile.params.profile_window_s == 2.0
    assert profile.params.profile_hop_s == 1.0


# ──────────────────────────────────────────────────────────────────────────
# Minority-share advisory (surfaced, NOT enforced)
#     decide_confidence checks that a dominant cluster exists, that no single runner-up is
#     within AMBIGUITY_SHARE_RATIO of it, and that there is enough speech — never the
#     dominant cluster's *absolute* share. A subject whose remaining audio is fragmented
#     across several mid-size clusters passes the runner-up test and reports "ok" while the
#     reference represents a minority of their speech. Observed on real data at share 0.459,
#     where the profile scored that subject's own held-out recording at 0.0.


def test_minority_dominant_share_is_flagged_in_provenance(monkeypatch: pytest.MonkeyPatch) -> None:
    """A below-advisory share sets a provenance flag and records the share."""
    # Two files whose windows land in distinct clusters → dominant share well under 1.0.
    a = _audio_of_length(160000)
    b = _audio_of_length(176000)
    _stub_extractor(monkeypatch, {"160000": _basis(0), "176000": _basis(5)})

    profile = build_speaker_profile(
        "sub-split",
        [ProfileInput(audio=a, file_id="a.wav"), ProfileInput(audio=b, file_id="b.wav")],
        embedding_models=[_MODEL],
    )

    assert "dominant_share" in profile.provenance
    assert profile.provenance["dominant_share"] == pytest.approx(profile.dominant_cluster.share)
    below = profile.dominant_cluster.share < C.ADVISORY_MIN_DOMINANT_SHARE
    assert profile.provenance["dominant_share_below_advisory"] is below


def test_minority_share_does_not_change_confidence(monkeypatch: pytest.MonkeyPatch) -> None:
    """The advisory must not silently downgrade a profile — semantics are unchanged.

    Enforcement is a separate decision: a low share can be legitimate for a subject whose
    every recording contains a second speaker. This asserts the flag is informational only.
    """
    a = _audio_of_length(160000)
    b = _audio_of_length(176000)
    _stub_extractor(monkeypatch, {"160000": _basis(0), "176000": _basis(5)})
    profile = build_speaker_profile(
        "sub-split",
        [ProfileInput(audio=a, file_id="a.wav"), ProfileInput(audio=b, file_id="b.wav")],
        embedding_models=[_MODEL],
    )
    # 0.0 for "no runner-up", matching how build_speaker_profile calls it.
    expected = decide_confidence(
        has_dominant=True,
        dominant_speech_seconds=profile.aggregate_speech_seconds,
        runner_up_speech_seconds=(
            profile.runner_up_cluster.speech_seconds if profile.runner_up_cluster is not None else 0.0
        ),
    )
    assert profile.confidence == expected


def test_majority_share_is_not_flagged(monkeypatch: pytest.MonkeyPatch) -> None:
    """A single coherent speaker keeps the flag clear."""
    a = _audio_of_length(160000)
    b = _audio_of_length(176000)
    _stub_extractor(monkeypatch, {"160000": _basis(0), "176000": _basis(0)})
    profile = build_speaker_profile(
        "sub-clean",
        [ProfileInput(audio=a, file_id="a.wav"), ProfileInput(audio=b, file_id="b.wav")],
        embedding_models=[_MODEL],
    )
    assert profile.dominant_cluster.share >= C.ADVISORY_MIN_DOMINANT_SHARE
    assert profile.provenance["dominant_share_below_advisory"] is False
