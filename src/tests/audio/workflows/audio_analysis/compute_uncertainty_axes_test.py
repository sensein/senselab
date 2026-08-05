"""End-to-end tests for compute_uncertainty_axes (T019, T026-T029).

Drives the workflow with synthetic ``passes`` summaries built from SimpleNamespace —
no real model invocations, no audio loading. Covers:

- T019: happy path with two diar models + two ASR models on a 4 s clip.
- T026: text-only ASR resolves through the alignment block (FR-011).
- T027: AST/YAMNet floor-based bucket→window indexing for cross-stream contributions.
- T028b: graceful degrade (FR-013) — failed pass / empty result.
- T029: multi-word AudioSet labels survive the speech_presence_labels parser.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from senselab.audio.data_structures import Audio
from senselab.audio.workflows.audio_analysis import (
    BucketGrid,
    compute_uncertainty_axes,
)


@pytest.fixture(autouse=True)
def _no_brouhaha(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep these end-to-end tests offline.

    ``compute_uncertainty_axes`` defaults to ``scene_quality=True``, which would
    otherwise try to load the gated ``pyannote/brouhaha`` model. Stub the loader
    to report the model as unavailable so the workflow exercises its null-safe
    quality path (FR-023) without any Hub call. Individual tests can re-patch it
    to inject synthetic frames.
    """
    import senselab.audio.tasks.scene_quality as sq

    monkeypatch.setattr(sq, "extract_brouhaha_frames", lambda audios, *a, **k: [None] * len(audios))


# ── Test fixture builders ─────────────────────────────────────────────


def _diar_block(segments: list[tuple[float, float, str]]) -> dict[str, Any]:
    """Build a diar by-model block from (start, end, speaker) tuples.

    senselab's diar API returns List[List[ScriptLine]]; mirror that shape.
    """
    segs = [SimpleNamespace(start=s, end=e, speaker=spk, text="") for s, e, spk in segments]
    return {"status": "ok", "result": [segs], "cache_key": "diar_k"}


def _asr_block_with_chunks(chunks: list[tuple[float, float, str]]) -> dict[str, Any]:
    """Build an ASR by-model block carrying per-token chunks (Whisper-style)."""
    chunk_objs = [SimpleNamespace(start=s, end=e, text=t, avg_logprob=-0.2, no_speech_prob=0.05) for s, e, t in chunks]
    line = SimpleNamespace(
        text=" ".join(t for _, _, t in chunks),
        chunks=chunk_objs,
        start=chunks[0][0] if chunks else None,
        end=chunks[-1][1] if chunks else None,
        avg_logprob=-0.2,
    )
    return {"status": "ok", "result": [line], "cache_key": "asr_k"}


def _asr_block_text_only(text: str) -> dict[str, Any]:
    """Text-only ScriptLine without per-token chunks (Granite / Canary-Qwen)."""
    line = SimpleNamespace(text=text, chunks=None, start=None, end=None, avg_logprob=None)
    return {"status": "ok", "result": [line], "cache_key": "asr_text_only"}


def _alignment_block_for(text: str, chunks: list[tuple[float, float, str]]) -> dict[str, Any]:
    """Build the post-MMS alignment block (List[List[ScriptLine | None]])."""
    chunk_objs = [SimpleNamespace(start=s, end=e, text=t) for s, e, t in chunks]
    line = SimpleNamespace(
        text=text,
        chunks=chunk_objs,
        start=chunks[0][0] if chunks else None,
        end=chunks[-1][1] if chunks else None,
    )
    return {"status": "ok", "result": [[line]], "cache_key": "align_k"}


def _classification_block(windows: list[dict[str, Any]]) -> dict[str, Any]:
    """Build an AST / YAMNet block — each window dict carries start/end/labels/scores."""
    return {"status": "ok", "result": [windows], "cache_key": "cls_k"}


def _silent_audio(duration_s: float, sr: int = 16000) -> Audio:
    """Build a trivial ``Audio`` object for tests that don't actually run embeddings.

    ``compute_uncertainty_axes`` accepts an audio dict; when
    ``speaker_embedding_models=[]`` the embedding extraction is skipped entirely.

    Audible rather than digitally silent. The absolute acoustic voters (LUFS, level-above-floor)
    read the waveform directly, so a silent fixture paired with mocked models reporting *speech*
    is self-contradictory: those voters correctly dissent and speech_presence uncertainty rises. Use
    :func:`_silence_audio` where silence is the thing under test.
    """
    import numpy as np
    import torch

    t = np.arange(int(duration_s * sr)) / sr
    # ~-26 LUFS: conversational level, so the level-based voters agree with mocked speech.
    y = (0.15 * np.sin(2 * np.pi * 220 * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 3 * t))).astype("float32")
    return Audio(waveform=torch.from_numpy(y).reshape(1, -1), sampling_rate=sr)


def _silence_audio(duration_s: float, sr: int = 16000) -> Audio:
    """Digital silence, for tests where the absence of signal is the point."""
    import torch

    return Audio(waveform=torch.zeros(1, int(duration_s * sr), dtype=torch.float32), sampling_rate=sr)


def _votes_at(linked: dict, perturbation: str, axis: str, start: float) -> dict:
    """The linked belief votes for one bucket, from the out-param.

    The votes are L2's input, not an L1 column: they are what the measurements mean under the
    run's policy. L1's own emission is the per-signal measurement, checked via ``signals``.
    """
    for bucket in linked[perturbation].buckets_by_axis[axis]:
        if abs(float(bucket["start"]) - start) < 1e-6:
            return dict(bucket["votes"])
    return {}


# ── T019 happy path ──────────────────────────────────────────────────


def test_compute_uncertainty_axes_happy_path() -> None:
    """Two diar models agreeing + two ASR models with one transcript edit on a 4 s clip.

    Verifies the three fused axes land — one per axis, not one per (pass, axis) — with the
    right row counts and ``uncertainty`` in [0, 1], and that L1 carries per-signal evidence
    for both passes with nothing named for an axis.
    """
    diar_segs = [(0.0, 1.0, "SPEAKER_00"), (1.0, 4.0, "SPEAKER_01")]
    raw_pass = {
        "duration_s": 4.0,
        "diarization": {
            "by_model": {
                "pyannote": _diar_block(diar_segs),
                "sortformer": _diar_block(diar_segs),
            }
        },
        "asr": {
            "by_model": {
                "whisper": _asr_block_with_chunks([(0.0, 1.0, "hello"), (1.0, 4.0, "world")]),
                # A genuine substitution, not a punctuation edit. The axis grades word agreement
                # *phonemically* (``asr.phoneme_similarity``), so "world" vs "world!!" is agreement —
                # which is the right reading and made this fixture assert on a difference that the
                # measure, correctly, does not see.
                "granite": _asr_block_with_chunks([(0.0, 1.0, "hello"), (1.0, 4.0, "planet")]),
            }
        },
    }
    enh_pass = {
        "duration_s": 4.0,
        "diarization": {
            "by_model": {
                "pyannote": _diar_block(diar_segs),
                "sortformer": _diar_block(diar_segs),
            }
        },
        "asr": {
            "by_model": {
                "whisper": _asr_block_with_chunks([(0.0, 1.0, "hello"), (1.0, 4.0, "world")]),
                "granite": _asr_block_with_chunks([(0.0, 1.0, "hello"), (1.0, 4.0, "world")]),
            }
        },
    }

    grid = BucketGrid(win_length=0.5, hop_length=0.5)
    linked: dict = {}
    signals, fused_axes, incomparable, _emb = compute_uncertainty_axes(
        linked_out=linked,
        passes={"raw": raw_pass, "enhanced": enh_pass},
        grid=grid,
        params={"win_length": 0.5, "hop_length": 0.5},
        audio={"raw": _silent_audio(4.0), "enhanced": _silent_audio(4.0)},
        speaker_embedding_models=[],  # Skip embedding extraction in this synthetic test.
        aggregator="min",
        speech_presence_labels=["Speech"],
    )

    # Every active axis, keyed by axis alone. An axis folds across passes, so a (pass, axis) key
    # cannot exist — and neither can a raw_vs_enhanced pseudo-pass to hold their difference.
    #
    # ``background_mask`` is here because it is harvested now: VAD / ASR words / speaker occupancy
    # vote on whether the target was active. A hard-coded set of three is what let the fourth axis be
    # fused and written while being absent from the index that ranks it, so this asserts against the
    # declaration rather than a literal.
    from senselab.audio.workflows.audio_analysis.axes import AXIS_NAMES

    assert set(fused_axes) == set(AXIS_NAMES)
    for axis_result in fused_axes.values():
        assert not hasattr(axis_result, "perturbation")
        for r in axis_result.rows:
            assert r["uncertainty"] is None or 0 <= r["uncertainty"] <= 1
            # The pass dimension is reported as a column on the output, never as an index.
            assert set(r["contributing_passes"]) <= {"raw", "enhanced"}

    # L1 carries per-signal evidence for both passes, and nothing there is named for an axis.
    assert set(signals) == {"raw", "enhanced"}
    for by_signal in signals.values():
        assert by_signal
        assert not ({"speech_presence", "speaker", "asr"} & set(by_signal))

    # Diar agrees across models → speech_presence and speaker uncertainty are low.
    raw_speech_presence = fused_axes["speech_presence"]
    raw_speaker = fused_axes["speaker"]
    # ``confidence`` is the probability the axis is settled here, and it is what "the models
    # agree" means. ``uncertainty`` is normalised entropy over {settled, unsettled} and is not
    # on the same scale as the fold this test used to read.
    avg_speech_presence = sum(r["confidence"] or 0 for r in raw_speech_presence.rows) / max(
        1, len(raw_speech_presence.rows)
    )
    avg_speaker = sum(r["confidence"] or 0 for r in raw_speaker.rows) / max(1, len(raw_speaker.rows))
    assert avg_speech_presence > 0.5
    assert avg_speaker > 0.5

    # Utterance: the raw pass substitutes one word (granite "planet" vs whisper "world"), so the
    # buckets that word reaches carry doubt — and the buckets no word reaches carry ``None`` rather
    # than 0.0, because nothing was said there.
    raw_asr = fused_axes["asr"]
    scored = [r for r in raw_asr.rows if r["triage_score"] is not None]
    assert scored, "the asr axis measured nothing at all"
    assert any(r["triage_score"] > 0 for r in scored)


# ── T026 text-only ASR via alignment block ───────────────────────────


def test_text_only_asr_resolves_through_alignment() -> None:
    """Granite-style text-only ASR contributes to speech_presence only via alignment block."""
    diar_segs = [(0.0, 1.0, "SPEAKER_00")]
    pass_summary = {
        "duration_s": 2.0,
        "diarization": {"by_model": {"pyannote": _diar_block(diar_segs)}},
        "asr": {
            "by_model": {
                "granite": _asr_block_text_only("hello world"),
            }
        },
        "alignment": {
            "by_model": {
                "granite": _alignment_block_for("hello world", [(0.1, 0.4, "hello"), (0.5, 0.9, "world")]),
            }
        },
    }
    linked: dict = {}
    signals, fused_axes, incomparable, _emb = compute_uncertainty_axes(
        linked_out=linked,
        passes={"raw": pass_summary},
        grid=BucketGrid(),
        params={},
        audio={"raw": _silent_audio(2.0)},
        speaker_embedding_models=[],
        aggregator="min",
        speech_presence_labels=["Speech"],
    )
    speech_presence = fused_axes["speech_presence"]
    # On the run's own 0.1 s grid, "hello" spans [0.1, 0.4]: the bucket at 0.1 is the one the word
    # reaches. Asserting at 0.0 only worked while ``BucketGrid()`` defaulted to 0.5 s — a bucket wide
    # enough to swallow the leading silence, and a default that disagreed with the declared grid.
    matching = [r for r in speech_presence.rows if abs(r["start"] - 0.1) < 1e-6]
    assert matching, "expected a row at start=0.1"
    granite_vote = _votes_at(linked, "raw", "speech_presence", 0.1).get("granite")
    assert granite_vote is not None and granite_vote["speaks"] is True


# ── T027 AST floor-based window indexing ──────────────────────────────


def test_ast_yamnet_uses_floor_window_indexing() -> None:
    """AST 10.24 s window → every 0.5 s bucket inside [0, 10.24] picks AST window 0."""
    diar_segs = [(0.0, 5.0, "SPEAKER_00")]
    pass_summary = {
        "duration_s": 4.0,
        "diarization": {"by_model": {"pyannote": _diar_block(diar_segs)}},
        "ast": _classification_block(
            [
                {
                    "start": 0.0,
                    "end": 10.24,
                    "label_scores": [{"Speech": 0.9}],
                    "win_length": 10.24,
                    "hop_length": 10.24,
                },
                {
                    "start": 10.24,
                    "end": 20.48,
                    "label_scores": [{"Music": 0.8}],
                    "win_length": 10.24,
                    "hop_length": 10.24,
                },
            ]
        ),
    }
    linked: dict = {}
    signals, fused_axes, incomparable, _emb = compute_uncertainty_axes(
        linked_out=linked,
        passes={"raw": pass_summary},
        grid=BucketGrid(),
        params={},
        audio={"raw": _silent_audio(4.0)},
        speaker_embedding_models=[],
        aggregator="min",
        speech_presence_labels=["Speech"],
    )
    speech_presence = fused_axes["speech_presence"]
    # Every bucket in [0, 4) should map to AST window 0 → Speech (in allowlist) → speaks=True.
    for r in speech_presence.rows:
        ast_vote = _votes_at(linked, "raw", "speech_presence", r["start"]).get("ast")
        assert ast_vote is not None and ast_vote["speaks"] is True


# ── T028b graceful degrade (FR-013) ──────────────────────────────────


def test_graceful_degrade_failed_models_do_not_raise() -> None:
    """Failed pass / empty result produce comparison_status entries — no exceptions."""
    pass_summary = {
        "duration_s": 2.0,
        "diarization": {
            "by_model": {
                "pyannote": _diar_block([(0.0, 2.0, "SPEAKER_00")]),
                "sortformer": {"status": "failed", "error": "OOM"},
            }
        },
        "asr": {
            "by_model": {
                "whisper": _asr_block_with_chunks([(0.0, 2.0, "hello world")]),
                "granite": {"status": "ok", "result": [], "cache_key": "empty"},
            }
        },
    }
    linked: dict = {}
    signals, fused_axes, incomparable, _emb = compute_uncertainty_axes(
        linked_out=linked,
        passes={"raw": pass_summary},
        grid=BucketGrid(),
        params={},
        audio={"raw": _silent_audio(2.0)},
        speaker_embedding_models=[],
        aggregator="min",
        speech_presence_labels=["Speech"],
    )
    # All three axes still emit; speech_presence has at least one row.
    assert "speech_presence" in fused_axes
    assert "speaker" in fused_axes
    assert "asr" in fused_axes


# ── T029 Multi-word AudioSet labels survive ──────────────────────────


def test_multi_word_audioset_labels_match() -> None:
    """The Speech allowlist contains 'Narration, monologue' — top-1 must match exactly."""
    diar_segs = [(0.0, 1.0, "SPEAKER_00")]
    pass_summary = {
        "duration_s": 1.0,
        "diarization": {"by_model": {"pyannote": _diar_block(diar_segs)}},
        "ast": _classification_block(
            [
                {
                    "start": 0.0,
                    "end": 10.24,
                    "label_scores": [{"Narration, monologue": 0.8}, {"Music": 0.2}],
                    "win_length": 10.24,
                    "hop_length": 10.24,
                },
            ]
        ),
    }
    linked: dict = {}
    signals, fused_axes, incomparable, _emb = compute_uncertainty_axes(
        linked_out=linked,
        passes={"raw": pass_summary},
        grid=BucketGrid(),
        params={},
        audio={"raw": _silent_audio(1.0)},
        speaker_embedding_models=[],
        aggregator="min",
        speech_presence_labels=["Speech", "Narration, monologue", "Conversation"],
    )
    speech_presence = fused_axes["speech_presence"]
    assert speech_presence.rows
    ast_vote = _votes_at(linked, "raw", "speech_presence", speech_presence.rows[0]["start"]).get("ast")
    assert ast_vote is not None and ast_vote["speaks"] is True


def test_speaker_robust_to_diar_label_naming_conventions() -> None:
    """Regression: pyannote ``SPEAKER_00``/``_01`` vs Sortformer ``speaker_0``/``_1``.

    Identity uncertainty no longer compares literal labels across models. Each diar
    model's label-equivalence claim is validated against the actual audio embeddings
    independently, so different naming conventions don't affect the result. The first
    E2E run on the higgs clip surfaced the bug (literal-string comparison made every
    bucket saturate at uncertainty=1.0).

    With ``speaker_embedding_models=[]`` the embedding-validation pairs are absent, but the
    cross-model signal is still measurable: H2's temporal-overlap matcher maps the two models'
    labels onto a common space from timing evidence, so two diarizers agreeing on the same timeline
    now read as *agreement* (0.0) rather than as unmeasurable. This assertion previously expected
    ``None`` — that was the old single-matcher limitation, not a requirement.
    """
    pyannote_segs = [(0.0, 2.0, "SPEAKER_00"), (2.0, 4.0, "SPEAKER_01")]
    sortformer_segs = [(0.0, 2.0, "speaker_0"), (2.0, 4.0, "speaker_1")]
    pass_summary = {
        "duration_s": 4.0,
        "diarization": {
            "by_model": {
                "pyannote": _diar_block(pyannote_segs),
                "sortformer": _diar_block(sortformer_segs),
            }
        },
    }
    linked: dict = {}
    signals, fused_axes, incomparable, _emb = compute_uncertainty_axes(
        linked_out=linked,
        passes={"raw": pass_summary},
        grid=BucketGrid(),
        params={},
        audio={"raw": _silent_audio(4.0)},
        speaker_embedding_models=[],
        aggregator="min",
        speech_presence_labels=["Speech"],
    )
    speaker = fused_axes["speaker"]
    assert speaker.rows, "expected speaker rows on a 4 s clip with diar coverage"

    # No embedding models, so there are no within-track cosines to fold; the cross-model
    # agreement signal carries the row on its own and must report agreement, never a
    # string-mismatch disagreement.
    for r in speaker.rows:
        assert r["uncertainty"] == pytest.approx(0.0), (
            f"models agree on the timeline, so speaker uncertainty must be 0, got {r['uncertainty']}"
        )
        speaker_votes = _votes_at(linked, "raw", "speaker", r["start"])
        cross = speaker_votes.get("__cross_diar_label_disagreement__")
        assert cross is not None and cross["n_disagree"] == 0
        py = speaker_votes.get("pyannote")
        sf = speaker_votes.get("sortformer")
        assert py is not None and sf is not None
        # Both labels are present per their respective convention; literal strings differ
        # but that's not what drives the aggregation.
        assert py["speaker_label"].startswith("SPEAKER_")
        assert sf["speaker_label"].startswith("speaker_")


# ── US1: scene-quality columns wired into speech_presence rows ───────────────


def _noise_audio(duration_s: float, sr: int = 16000) -> Audio:
    import torch

    rng = np.random.default_rng(0)
    y = (0.1 * rng.standard_normal(int(duration_s * sr))).astype(np.float32)
    return Audio(waveform=torch.tensor(y).reshape(1, -1), sampling_rate=sr)


def test_speech_presence_rows_carry_quality_columns_when_brouhaha_available(monkeypatch: pytest.MonkeyPatch) -> None:
    """US1: with Brouhaha frames available, speech_presence rows expose quality_* columns."""
    import senselab.audio.tasks.scene_quality as sq
    from senselab.audio.tasks.scene_quality.brouhaha import BrouhahaFrames

    n = int(2.0 / 0.02)
    frames = BrouhahaFrames(vad=np.ones(n), snr_db=np.full(n, 25.0), c50_db=np.full(n, 28.0), frame_hop_s=0.02)
    monkeypatch.setattr(sq, "extract_brouhaha_frames", lambda audios, *a, **k: [frames] * len(audios))

    raw_pass = {
        "duration_s": 2.0,
        "diarization": {"by_model": {"pyannote": _diar_block([(0.0, 2.0, "SPEAKER_00")])}},
    }
    linked: dict = {}
    signals, fused_axes, _, _emb = compute_uncertainty_axes(
        linked_out=linked,
        passes={"raw": raw_pass},
        grid=BucketGrid(win_length=0.5, hop_length=0.5),
        params={},
        audio={"raw": _noise_audio(2.0)},
        speaker_embedding_models=[],
        aggregator="min",
        speech_presence_labels=["Speech"],
    )
    speech_presence = fused_axes["speech_presence"]
    assert speech_presence.rows
    assert any(r.get("quality_snr") is not None for r in speech_presence.rows)
    for r in speech_presence.rows:
        for v in (r.get("quality_snr"), r.get("quality_reverb"), r.get("quality_bandwidth")):
            assert v is None or 0.0 <= v <= 1.0
    # Provenance of a *measurement* is L1's, so it is recorded on the pass, not on the axis.
    prov = linked["raw"].provenance["scene_quality"]
    assert prov["enabled"] is True
    assert prov["model"]["available"] is True


def test_speech_presence_rows_carry_source_columns() -> None:
    """US2: AST/YAMNet windows → per-bucket src_* category masses on speech_presence rows."""
    windows = [
        {
            "start": 0.0,
            "end": 2.0,
            "label_scores": [{"Speech": 0.8}, {"Vehicle": 0.2}],
            "win_length": 2.0,
            "hop_length": 2.0,
        }
    ]
    raw_pass = {
        "duration_s": 2.0,
        "diarization": {"by_model": {"pyannote": _diar_block([(0.0, 2.0, "SPEAKER_00")])}},
        "ast": _classification_block(windows),
    }
    linked: dict = {}
    signals, fused_axes, _, _emb = compute_uncertainty_axes(
        linked_out=linked,
        passes={"raw": raw_pass},
        grid=BucketGrid(win_length=0.5, hop_length=0.5),
        params={},
        audio={"raw": _silent_audio(2.0)},
        speaker_embedding_models=[],
        aggregator="min",
        speech_presence_labels=["Speech"],
        scene_quality=False,
    )
    speech_presence = fused_axes["speech_presence"]
    assert speech_presence.rows
    assert any(r.get("src_speech") is not None for r in speech_presence.rows)
    for r in speech_presence.rows:
        if r.get("src_speech") is not None:
            assert (
                r.get("src_people") is not None
                and r.get("src_machine") is not None
                and r.get("src_environment") is not None
            )
            total = r.get("src_speech") + r.get("src_people") + r.get("src_machine") + r.get("src_environment")
            assert abs(total - 1.0) < 1e-6
            assert r.get("src_dominant") == "speech"
    assert linked["raw"].provenance["sound_sources"]["enabled"] is True


def test_speech_presence_confidence_uncertainty_split_and_instability(monkeypatch: pytest.MonkeyPatch) -> None:
    """US3: speech_presence_confidence + speech_presence_uncertainty columns; frame instability lifts uncertainty."""
    import senselab.audio.tasks.scene_quality as sq
    from senselab.audio.tasks.scene_quality.brouhaha import BrouhahaFrames

    # Rapidly alternating VAD posterior → high within-bucket std (instability) everywhere.
    # Brouhaha's VAD head rather than segmentation-3.0's: the latter is no longer a frame voter, and
    # the instability this test is about is a property of a continuous posterior, not of that model.
    probs = np.tile([0.0, 1.0], 100)  # 200 frames @ 0.01 s hop = 2 s
    monkeypatch.setattr(
        sq,
        "extract_brouhaha_frames",
        lambda audios, *a, **k: (
            [
                BrouhahaFrames(
                    vad=probs,
                    snr_db=np.full(probs.shape, 30.0),
                    c50_db=np.full(probs.shape, 30.0),
                    frame_hop_s=0.01,
                )
            ]
            * len(audios)
        ),
    )
    raw_pass = {
        "duration_s": 2.0,
        "diarization": {"by_model": {"pyannote": _diar_block([(0.0, 2.0, "SPEAKER_00")])}},
    }
    linked: dict = {}
    signals, fused_axes, _, _emb = compute_uncertainty_axes(
        linked_out=linked,
        passes={"raw": raw_pass},
        grid=BucketGrid(win_length=0.5, hop_length=0.5),
        params={},
        audio={"raw": _silent_audio(2.0)},
        speaker_embedding_models=[],
        aggregator="min",
        speech_presence_labels=["Speech"],
        scene_quality=False,
    )
    speech_presence = fused_axes["speech_presence"]
    assert speech_presence.rows
    # ``confidence`` and ``uncertainty`` are different quantities with different estimators, and
    # the fused row keeps both rather than collapsing them.
    assert all(r.get("confidence") is not None for r in speech_presence.rows)
    assert all(r.get("uncertainty") is not None for r in speech_presence.rows)
    assert all(0.0 <= r["uncertainty"] <= 1.0 for r in speech_presence.rows)
    # Frame instability is an L1 measurement in probability units, persisted as its own signal
    # so both ingest paths can read it. It used to reach only the in-process path, which left
    # one of P2's two documented triggers structurally dead on the artifact path.
    dispersion = signals["raw"]["frame_dispersion"]
    assert dispersion.rows
    assert all(r.measurement["frame_dispersion"] > 0.0 for r in dispersion.rows)
    assert all(r.units == "probability" for r in dispersion.rows)
    # The frame voter is brouhaha's VAD head now, and its provenance is recorded whether or not
    # the model loaded — a voter with no provenance is a number nobody can reproduce.
    assert linked["raw"].provenance["frame_posteriors"]["brouhaha_vad"]["available"] is True


def test_speech_presence_quality_null_when_scene_quality_disabled() -> None:
    """scene_quality=False → no quality columns, no model load."""
    raw_pass = {
        "duration_s": 2.0,
        "diarization": {"by_model": {"pyannote": _diar_block([(0.0, 2.0, "SPEAKER_00")])}},
    }
    linked: dict = {}
    signals, fused_axes, _, _emb = compute_uncertainty_axes(
        linked_out=linked,
        passes={"raw": raw_pass},
        grid=BucketGrid(win_length=0.5, hop_length=0.5),
        params={},
        audio={"raw": _noise_audio(2.0)},
        speaker_embedding_models=[],
        aggregator="min",
        speech_presence_labels=["Speech"],
        scene_quality=False,
    )
    speech_presence = fused_axes["speech_presence"]
    assert speech_presence.rows
    assert all(r.get("quality_snr") is None for r in speech_presence.rows)
    assert linked["raw"].provenance["scene_quality"]["enabled"] is False


# ── T094a: the three axes survive the per-speaker change (SC-010) ─────


def test_the_three_axes_are_unchanged_by_the_per_speaker_derivation() -> None:
    """SC-010: per-speaker speaker is additive, not a replacement of the axis outputs.

    The per-bucket axes stay the evidence-gathering mechanism and every existing consumer
    reads them unchanged. The risk this guards is a silent one: the per-speaker derivation
    reads the same harvest, and if it mutated what it read — sorting vote dicts, promoting
    silence, renaming clusters — speech_presence and asr would shift for reasons that have
    nothing to do with either axis, and no per-speaker test would notice.
    """
    from senselab.audio.workflows.audio_analysis.speaker_identity import (
        build_speaker_identity,
        build_speech_presence_tracks,
    )

    diar_segs = [(0.0, 1.0, "SPEAKER_00"), (1.0, 4.0, "SPEAKER_01")]
    pass_summary = {
        "duration_s": 4.0,
        "diarization": {"by_model": {"pyannote": _diar_block(diar_segs), "sortformer": _diar_block(diar_segs)}},
        "asr": {
            "by_model": {
                "whisper": _asr_block_with_chunks([(0.0, 1.0, "hello"), (1.0, 4.0, "world")]),
                # A genuine substitution, not a punctuation edit. The axis grades word agreement
                # *phonemically* (``asr.phoneme_similarity``), so "world" vs "world!!" is agreement —
                # which is the right reading and made this fixture assert on a difference that the
                # measure, correctly, does not see.
                "granite": _asr_block_with_chunks([(0.0, 1.0, "hello"), (1.0, 4.0, "planet")]),
            }
        },
    }
    passes = {"raw": pass_summary, "enhanced": pass_summary}
    kwargs: dict[str, Any] = {
        "grid": BucketGrid(win_length=0.5, hop_length=0.5),
        "params": {"win_length": 0.5, "hop_length": 0.5},
        "audio": {"raw": _silent_audio(4.0), "enhanced": _silent_audio(4.0)},
        "speaker_embedding_models": [],
        "aggregator": "min",
        "speech_presence_labels": ["Speech"],
    }

    harvests: dict[str, Any] = {}
    _signals_before, before, _reasons, _emb = compute_uncertainty_axes(passes=passes, harvests_out=harvests, **kwargs)
    votes = harvests["raw"].speaker_votes

    posterior, hypotheses, correspondence = build_speaker_identity(passes, speaker_votes=votes)
    tracks = build_speech_presence_tracks(votes)
    assert hypotheses and tracks and correspondence  # the derivation did run

    _signals_after, after, _reasons2, _emb2 = compute_uncertainty_axes(passes=passes, **kwargs)
    for key in before:
        rows_before = [(r["start"], r["end"], r["uncertainty"]) for r in before[key].rows]
        rows_after = [(r["start"], r["end"], r["uncertainty"]) for r in after[key].rows]
        assert rows_before == rows_after, f"{key} changed after the per-speaker derivation ran"

    # And the speaker axis itself still reports per bucket, in range, as before.
    speaker = before["speaker"]
    assert speaker.rows
    assert all(r["uncertainty"] is None or 0.0 <= r["uncertainty"] <= 1.0 for r in speaker.rows)
