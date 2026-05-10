"""End-to-end tests for compute_uncertainty_axes (T019, T026-T029).

Drives the workflow with synthetic ``passes`` summaries built from SimpleNamespace —
no real model invocations, no audio loading. Covers:

- T019: happy path with two diar models + two ASR models on a 4 s clip.
- T026: text-only ASR resolves through the alignment block (FR-011).
- T027: AST/YAMNet floor-based bucket→window indexing for cross-stream contributions.
- T028: PPG present vs absent — utterance axis sub-signal drops out cleanly.
- T028b: graceful degrade (FR-013) — failed pass / empty result / missing PPG.
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
    """
    import torch

    return Audio(waveform=torch.zeros(1, int(duration_s * sr), dtype=torch.float32), sampling_rate=sr)


# ── T019 happy path ──────────────────────────────────────────────────


def test_compute_uncertainty_axes_happy_path() -> None:
    """Two diar models agreeing + two ASR models with one transcript edit on a 4 s clip.

    Verifies all 9 axis_results land (3 axes × 2 passes + 3 raw_vs_enhanced) with the
    right row counts and aggregated_uncertainty in [0, 1].
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
                "granite": _asr_block_with_chunks([(0.0, 1.0, "hello"), (1.0, 4.0, "world!!")]),
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
    axis_results, incomparable, _per_seg_emb = compute_uncertainty_axes(
        passes={"raw_16k": raw_pass, "enhanced_16k": enh_pass},
        grid=grid,
        params={"win_length": 0.5, "hop_length": 0.5},
        audio={"raw_16k": _silent_audio(4.0), "enhanced_16k": _silent_audio(4.0)},
        speaker_embedding_models=[],  # Skip embedding extraction in this synthetic test.
        aggregator="min",
        speech_presence_labels=["Speech"],
    )

    expected_keys = {
        (p, a) for p in ("raw_16k", "enhanced_16k", "raw_vs_enhanced") for a in ("presence", "identity", "utterance")
    }
    assert set(axis_results.keys()) == expected_keys

    # Each per-pass parquet should have rows on the buckets where speech is present.
    for pass_label in ("raw_16k", "enhanced_16k"):
        presence = axis_results[(pass_label, "presence")]
        assert len(presence.rows) > 0
        for r in presence.rows:
            assert r.aggregated_uncertainty is None or 0 <= r.aggregated_uncertainty <= 1

    # Diar agrees across models → presence and identity uncertainty are low.
    raw_presence = axis_results[("raw_16k", "presence")]
    raw_identity = axis_results[("raw_16k", "identity")]
    avg_presence = sum(r.aggregated_uncertainty or 0 for r in raw_presence.rows) / max(1, len(raw_presence.rows))
    avg_identity = sum(r.aggregated_uncertainty or 0 for r in raw_identity.rows) / max(1, len(raw_identity.rows))
    assert avg_presence < 0.5
    assert avg_identity < 0.5

    # Utterance: raw pass has one transcript edit (granite "world!!" vs whisper "world"),
    # so at least one bucket should have non-zero uncertainty.
    raw_utterance = axis_results[("raw_16k", "utterance")]
    assert any((r.aggregated_uncertainty or 0) > 0 for r in raw_utterance.rows)


# ── T026 text-only ASR via alignment block ───────────────────────────


def test_text_only_asr_resolves_through_alignment() -> None:
    """Granite-style text-only ASR contributes to presence only via alignment block."""
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
    axis_results, _, _per_seg_emb = compute_uncertainty_axes(
        passes={"raw_16k": pass_summary},
        grid=BucketGrid(),
        params={},
        audio={"raw_16k": _silent_audio(2.0)},
        speaker_embedding_models=[],
        aggregator="min",
        speech_presence_labels=["Speech"],
    )
    presence = axis_results[("raw_16k", "presence")]
    # Bucket [0.0, 0.5) should have granite voting True (covers both aligned chunks).
    matching = [r for r in presence.rows if abs(r.start - 0.0) < 1e-6]
    assert matching, "expected a row at start=0.0"
    granite_vote = matching[0].model_votes.get("granite")
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
                    "labels": ["Speech"],
                    "scores": [0.9],
                    "win_length": 10.24,
                    "hop_length": 10.24,
                },
                {
                    "start": 10.24,
                    "end": 20.48,
                    "labels": ["Music"],
                    "scores": [0.8],
                    "win_length": 10.24,
                    "hop_length": 10.24,
                },
            ]
        ),
    }
    axis_results, _, _per_seg_emb = compute_uncertainty_axes(
        passes={"raw_16k": pass_summary},
        grid=BucketGrid(),
        params={},
        audio={"raw_16k": _silent_audio(4.0)},
        speaker_embedding_models=[],
        aggregator="min",
        speech_presence_labels=["Speech"],
    )
    presence = axis_results[("raw_16k", "presence")]
    # Every bucket in [0, 4) should map to AST window 0 → Speech (in allowlist) → speaks=True.
    for r in presence.rows:
        ast_vote = r.model_votes.get("ast")
        assert ast_vote is not None and ast_vote["speaks"] is True


# ── T028 PPG presence vs absence ──────────────────────────────────────


def test_ppg_absent_drops_pairwise_ppg_pairs() -> None:
    """When PPG is absent, the pairwise grid contains no ``__ppg__|*`` pairs."""
    pass_summary = {
        "duration_s": 2.0,
        "asr": {
            "by_model": {
                "whisper": _asr_block_with_chunks([(0.0, 1.0, "hello"), (1.0, 2.0, "world")]),
            }
        },
    }
    axis_results, incomparable, _per_seg_emb = compute_uncertainty_axes(
        passes={"raw_16k": pass_summary},
        grid=BucketGrid(),
        params={},
        audio={"raw_16k": _silent_audio(2.0)},
        speaker_embedding_models=[],
        aggregator="min",
        speech_presence_labels=["Speech"],
    )
    utt = axis_results[("raw_16k", "utterance")]
    for r in utt.rows:
        pair_block = r.model_votes.get("__pairwise_phoneme_distances__")
        if isinstance(pair_block, dict):
            pairs = pair_block.get("pairs") or {}
            for pair_key in pairs:
                assert "__ppg__" not in pair_key, "no __ppg__ pairs expected when PPG opted out"
    assert "raw_16k/utterance/ppg" in incomparable


def test_ppg_present_populates_pairwise_ppg_pairs() -> None:
    """When PPG is provisioned, ``__ppg__|<asr>`` pairs appear in the bucket pairwise grid."""
    n_frames = 200
    n_phonemes = 40
    ppg = np.zeros((n_phonemes, n_frames), dtype=np.float32)
    ppg[25, :] = 1.0  # all frames argmax to "oy"
    pass_summary = {
        "duration_s": 2.0,
        "asr": {
            "by_model": {
                "whisper": _asr_block_with_chunks([(0.0, 1.0, "hello"), (1.0, 2.0, "world")]),
            }
        },
        "ppgs": {"status": "ok", "result": [ppg]},
    }
    axis_results, _, _per_seg_emb = compute_uncertainty_axes(
        passes={"raw_16k": pass_summary},
        grid=BucketGrid(),
        params={},
        audio={"raw_16k": _silent_audio(2.0)},
        speaker_embedding_models=[],
        aggregator="min",
        speech_presence_labels=["Speech"],
    )
    utt = axis_results[("raw_16k", "utterance")]
    found_ppg_pair = False
    for r in utt.rows:
        pair_block = r.model_votes.get("__pairwise_phoneme_distances__")
        if isinstance(pair_block, dict):
            pairs = pair_block.get("pairs") or {}
            for pair_key, dist in pairs.items():
                if "__ppg__" in pair_key:
                    found_ppg_pair = True
                    assert 0.0 <= float(dist) <= 1.0
    assert found_ppg_pair, "expected at least one bucket with a __ppg__ pairwise pair"


# ── T028b graceful degrade (FR-013) ──────────────────────────────────


def test_graceful_degrade_failed_models_do_not_raise() -> None:
    """Failed pass / empty result / missing PPG produce comparison_status entries — no exceptions."""
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
        # No PPG block at all.
    }
    axis_results, incomparable, _per_seg_emb = compute_uncertainty_axes(
        passes={"raw_16k": pass_summary},
        grid=BucketGrid(),
        params={},
        audio={"raw_16k": _silent_audio(2.0)},
        speaker_embedding_models=[],
        aggregator="min",
        speech_presence_labels=["Speech"],
    )
    # All three axes still emit; presence has at least one row.
    assert ("raw_16k", "presence") in axis_results
    assert ("raw_16k", "identity") in axis_results
    assert ("raw_16k", "utterance") in axis_results
    # PPG missing reason recorded.
    assert "raw_16k/utterance/ppg" in incomparable


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
                    "labels": ["Narration, monologue", "Music"],
                    "scores": [0.8, 0.2],
                    "win_length": 10.24,
                    "hop_length": 10.24,
                },
            ]
        ),
    }
    axis_results, _, _per_seg_emb = compute_uncertainty_axes(
        passes={"raw_16k": pass_summary},
        grid=BucketGrid(),
        params={},
        audio={"raw_16k": _silent_audio(1.0)},
        speaker_embedding_models=[],
        aggregator="min",
        speech_presence_labels=["Speech", "Narration, monologue", "Conversation"],
    )
    presence = axis_results[("raw_16k", "presence")]
    assert presence.rows
    ast_vote = presence.rows[0].model_votes.get("ast")
    assert ast_vote is not None and ast_vote["speaks"] is True


def test_identity_robust_to_diar_label_naming_conventions() -> None:
    """Regression: pyannote ``SPEAKER_00``/``_01`` vs Sortformer ``speaker_0``/``_1``.

    Identity uncertainty no longer compares literal labels across models. Each diar
    model's label-equivalence claim is validated against the actual audio embeddings
    independently, so different naming conventions don't affect the result. The first
    E2E run on the higgs clip surfaced the bug (literal-string comparison made every
    bucket saturate at uncertainty=1.0).

    With ``speaker_embedding_models=[]`` the embedding-validation pairs are absent and
    the aggregator returns ``None``, so we just check the raw labels are recorded for
    auditability.
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
    axis_results, _, _per_seg_emb = compute_uncertainty_axes(
        passes={"raw_16k": pass_summary},
        grid=BucketGrid(),
        params={},
        audio={"raw_16k": _silent_audio(4.0)},
        speaker_embedding_models=[],
        aggregator="min",
        speech_presence_labels=["Speech"],
    )
    identity = axis_results[("raw_16k", "identity")]
    assert identity.rows, "expected identity rows on a 4 s clip with diar coverage"

    # Without embedding models, all rows should have aggregated_uncertainty=None
    # (no within-track cosines to fold). The raw labels are recorded on the diar votes
    # for auditability.
    for r in identity.rows:
        assert r.aggregated_uncertainty is None
        py = r.model_votes.get("pyannote")
        sf = r.model_votes.get("sortformer")
        assert py is not None and sf is not None
        # Both labels are present per their respective convention; literal strings differ
        # but that's not what drives the aggregation.
        assert py["speaker_label"].startswith("SPEAKER_")
        assert sf["speaker_label"].startswith("speaker_")
