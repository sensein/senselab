"""An ASR model's word coverage must reach the speech-presence axis, whatever its result shape.

Word coverage is a presence signal by design — a model placing words in a bucket is evidence
someone spoke there. On the shipped default ASR set (CrisperWhisper 2.0 turbo, Canary-Qwen 2.5b,
Qwen3-ASR 1.7B) it did not arrive: a real run fused **8** signals onto ``speech_presence`` where
``--asr-models openai/whisper-*`` fused 12, and no ASR model appeared at all.

The harvester was not at fault, which is the part worth pinning. Fed the three models' real cached
results, ``asr_bucket_chunk_evidence`` reads every shape and reports coverage for all three, and
``speech_presence_link`` turns each into a ``speaks`` vote. What lost them was ``fuse_axis``: via
``per_signal_uncertainty`` it only understood a vote carrying a *scored* quantity, and Whisper is
the only backend that reports one (``avg_logprob`` per segment). The other three assert a direction
and score nothing, so the fold dropped them — silently, and indistinguishably from a model that had
said nothing at all.

So these tests walk the whole chain — raw result → ``resolve_asr_result`` →
``asr_bucket_chunk_evidence`` → ``link_speech_presence`` → ``per_signal_uncertainty`` →
``fuse_axis`` — and are parameterised over the *shapes* rather than the model names. A fourth
backend arriving with a new shape fails here rather than quietly contributing nothing.
"""

from __future__ import annotations

from typing import Any

import pytest

from senselab.audio.workflows.audio_analysis.fuse import (
    fuse_axis,
    is_direction_only_claim,
    per_signal_uncertainty,
)
from senselab.audio.workflows.audio_analysis.harvesters import (
    asr_bucket_chunk_evidence,
    resolve_asr_result,
)
from senselab.audio.workflows.audio_analysis.reliability import signal_stability
from senselab.audio.workflows.audio_analysis.speech_presence_link import link_speech_presence
from senselab.utils.data_structures.script_line import ScriptLine

MODEL = "some/asr-model"
BUCKET = (0.0, 0.5)


def _word(text: str, start: float, end: float, **extra: object) -> dict[str, Any]:
    """One word chunk in the shape the JSON cache deserializes to."""
    return {
        "text": text,
        "speaker": None,
        "start": start,
        "end": end,
        "chunks": None,
        "score": None,
        "avg_logprob": None,
        "no_speech_prob": None,
        "timestamp_source": None,
        "token_entropy": None,
        **extra,
    }


WORDS = [_word("I", 0.02, 0.1), _word("can't", 0.1, 0.35), _word("believe", 0.35, 0.66)]


def _line(**over: object) -> dict[str, Any]:
    base: dict[str, Any] = {
        "text": "I can't believe",
        "speaker": None,
        "start": 0.02,
        "end": 0.66,
        "chunks": None,
        "score": None,
        "avg_logprob": None,
        "no_speech_prob": None,
        "timestamp_source": None,
        "token_entropy": None,
    }
    base.update(over)
    return base


# ── the shapes, taken from real cached results ───────────────────────────────
#
# ``(id, asr_block, align_block)``. Each is a shape a shipped backend actually produces:
#
# nyralabs/CrisperWhisper2.0_turbo — native word chunks, every confidence field ``None``.
# Qwen/Qwen3-ASR-1.7B             — word chunks from its bundled Qwen3-ForcedAligner companion.
# nvidia/canary-qwen-2.5b         — text only: no line times, no chunks. Its word times arrive in
#                                   a separate alignment block, which ``resolve_asr_result`` has to
#                                   fall through to (note the extra list nesting it carries).
# openai/whisper-*                — word chunks *and* per-segment ``avg_logprob`` /
#                                   ``no_speech_prob``. The one configuration that already worked,
#                                   kept so the scored path stays covered by the same walk.
SHAPES: list[tuple[str, dict[str, Any], dict[str, Any] | None]] = [
    (
        "native_word_chunks",
        {"status": "ok", "result": [_line(chunks=WORDS)]},
        None,
    ),
    (
        "bundled_aligner_word_chunks",
        {"status": "ok", "result": [_line(start=0.0, end=0.64, chunks=WORDS)]},
        None,
    ),
    (
        "text_only_with_external_alignment",
        {"status": "ok", "result": [_line(start=None, end=None, chunks=None)]},
        {"status": "ok", "result": [[_line(start=0.0, end=0.64, chunks=WORDS)]]},
    ),
    (
        "word_chunks_with_token_logits",
        {
            "status": "ok",
            "result": [_line(chunks=WORDS, avg_logprob=-0.2, no_speech_prob=0.02)],
        },
        None,
    ),
    (
        "pydantic_script_lines_in_memory",
        {
            "status": "ok",
            "result": [
                ScriptLine(
                    text="I can't believe",
                    start=0.02,
                    end=0.66,
                    chunks=[ScriptLine(text=w["text"], start=w["start"], end=w["end"]) for w in WORDS],
                )
            ],
        },
        None,
    ),
]


def _harvest(asr_block: dict[str, Any], align_block: dict[str, Any] | None) -> dict[str, Any]:
    """The presence harvest's ASR branch for one model over ``BUCKET``, verbatim."""
    resolved = resolve_asr_result(asr_block, align_block)
    chunk_ev = asr_bucket_chunk_evidence(resolved, *BUCKET)
    return {**chunk_ev, "units": "second", "native_window_s": chunk_ev.get("claim_span_s")}


def _row(evidence: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {"start": BUCKET[0], "end": BUCKET[1], "evidence": evidence, "frame_dispersion": None}


@pytest.mark.parametrize(("shape", "asr_block", "align_block"), SHAPES, ids=[s[0] for s in SHAPES])
def test_word_coverage_survives_every_asr_result_shape(
    shape: str, asr_block: dict[str, Any], align_block: dict[str, Any] | None
) -> None:
    """L1 must measure coverage from the shape the backend actually produced.

    Parameterised on the shape so a backend whose words the harvester cannot see fails loudly
    here. It is the measurement that is asserted, not a belief: seconds of word overlap and a
    word count, in the tool's own units.
    """
    ev = _harvest(asr_block, align_block)
    assert ev["word_overlap_s"] > 0.0, f"{shape}: harvester saw no words"
    assert ev["n_words"] == len(WORDS)
    # Coverage is clipped to the bucket, so it can never exceed its width.
    assert ev["word_overlap_s"] <= BUCKET[1] - BUCKET[0]


@pytest.mark.parametrize(("shape", "asr_block", "align_block"), SHAPES, ids=[s[0] for s in SHAPES])
def test_the_axis_names_every_asr_model_that_placed_words(
    shape: str, asr_block: dict[str, Any], align_block: dict[str, Any] | None
) -> None:
    """The whole chain: coverage → vote → per-signal reading → ``contributing_signals``.

    This is the assertion the shipped configuration failed. Both link and harvest were already
    right; the fold dropped every backend that scores nothing, so the axis reported 8 signals
    where Whisper's configuration reported 12 and the loss was invisible in the output.
    """
    rows = link_speech_presence([_row({MODEL: _harvest(asr_block, align_block)})], reporting_win_s=0.5)
    assert rows[0]["votes"][MODEL]["speaks"] is True, f"{shape}: link did not read coverage as speech"

    assert MODEL in per_signal_uncertainty(rows[0]), f"{shape}: the fold cannot read this vote"

    fused = fuse_axis({"raw": rows}, weights={}, aggregator="min", snr_gate=None)
    assert MODEL in fused[0]["contributing_signals"], f"{shape}: absent from the fused axis"
    assert fused[0]["signal_weights"][MODEL] == pytest.approx(1.0)


def test_a_scored_backend_takes_its_doubt_from_its_score_where_it_reports_one() -> None:
    """Where ``avg_logprob`` reaches a bucket it sets the doubt; the bare direction does not.

    The direction-only reading is a fallback for a vote with nothing scored in it. Letting it apply
    to a bucket whose vote *does* carry a score would discard the score.

    Which buckets those are is the surprise. Whisper carries ``avg_logprob`` on the **line**, and
    ``asr_bucket_chunk_evidence`` only falls back to line-level scalars when no chunk of that line
    overlapped the bucket. So Whisper's score reaches exactly the buckets where it placed *no*
    words — and a bucket full of its words is as direction-only as any other backend's. Whisper's
    presence signal was therefore never carried by its word coverage either; it kept a foothold on
    the axis through its silent buckets, which is the whole of what made it look like the shape that
    worked.
    """
    line = _line(end=20.9, chunks=WORDS, avg_logprob=-0.2, no_speech_prob=0.02)
    block = {"status": "ok", "result": [line]}

    where_words_are = link_speech_presence([_row({MODEL: _harvest(block, None)})], reporting_win_s=0.5)
    vote = where_words_are[0]["votes"][MODEL]
    assert vote["n_words"] == len(WORDS)
    assert vote.get("avg_logprob") is None
    assert is_direction_only_claim(vote) is True
    assert per_signal_uncertainty(where_words_are[0])[MODEL] == pytest.approx(0.0)

    resolved = resolve_asr_result(block, None)
    silent_ev = asr_bucket_chunk_evidence(resolved, 8.0, 8.5)
    silent_row = {"start": 8.0, "end": 8.5, "evidence": {MODEL: {**silent_ev, "units": "second"}}}
    silent = link_speech_presence([silent_row], reporting_win_s=0.5)
    silent_vote = silent[0]["votes"][MODEL]
    assert silent_vote["n_words"] == 0
    assert silent_vote["avg_logprob"] == pytest.approx(-0.2)
    assert is_direction_only_claim(silent_vote) is False
    # 1 - exp(-0.2): the score is what the fold reads, not the direction.
    assert per_signal_uncertainty(silent[0])[MODEL] == pytest.approx(0.1813, abs=0.01)


def test_a_model_that_placed_no_words_here_fabricates_no_coverage() -> None:
    """Zero words is zero words — not coverage, and not a claim that speech happened.

    The fix must not paper over an unreadable shape by manufacturing coverage, because then a
    backend the harvester cannot parse would be indistinguishable from one that was simply silent.
    """
    ev = _harvest({"status": "ok", "result": [_line(start=8.0, end=9.0, chunks=[_word("later", 8.0, 9.0)])]}, None)
    assert ev["n_words"] == 0
    assert ev["word_overlap_s"] == pytest.approx(0.0)
    assert ev["claim_span_s"] is None
    rows = link_speech_presence([_row({MODEL: ev})], reporting_win_s=0.5)
    assert rows[0]["votes"][MODEL]["speaks"] is False


def test_a_signal_absent_from_the_evidence_is_absent_from_the_axis() -> None:
    """An unmeasured signal is still dropped, never zero-filled (FR-007).

    The two absences have to stay apart: a model the run never asked for contributes nothing,
    while a model that answered contributes even when it scored nothing.
    """
    rows = link_speech_presence([_row({MODEL: _harvest(dict(SHAPES[0][1]), None)})], reporting_win_s=0.5)
    fused = fuse_axis({"raw": rows}, weights={}, aggregator="min", snr_gate=None)
    assert fused[0]["contributing_signals"] == [MODEL]
    assert "another/asr-model" not in fused[0]["signal_weights"]


def test_stability_still_sees_a_direction_only_voter_flip_between_passes() -> None:
    """A flip is what stability asks about, and zero doubt in both directions cannot show one.

    ``per_signal_uncertainty`` now reports a direction-only voter as ``0.0`` whichever way it
    voted, which is right for the fold and constant — so ``reliability`` has to read the direction
    itself, or a diarizer and an unscored ASR model would look perfectly stable by construction.
    """
    raw = [{"start": 0.0, "end": 0.5, "votes": {MODEL: {"speaks": True, "native_confidence": None}}}]
    enhanced = [{"start": 0.0, "end": 0.5, "votes": {MODEL: {"speaks": False, "native_confidence": None}}}]
    instability = signal_stability({}, axis="speech_presence", buckets_by_pass={"raw": raw, "enhanced": enhanced})
    assert instability[MODEL] == pytest.approx(1.0)


def test_including_a_direction_only_voter_cannot_lower_the_triage_score() -> None:
    """The default fold is max-doubt, so a voter with no doubt cannot make a region look calmer.

    Worth pinning: the axis's ``uncertainty`` and ``confidence`` do move when these voters rejoin
    (seven of them rejoined on a real run), but ``triage_score`` — what the adaptive loop spends
    its budget on — is unchanged, so the fix cannot hide a region that needed attention.
    """
    doubtful = {"same_label_uncertainty": 0.8}
    without = fuse_axis(
        {"raw": [{"start": 0.0, "end": 0.5, "votes": {"x": doubtful}}]}, weights={}, aggregator="min", snr_gate=None
    )
    with_claim = fuse_axis(
        {"raw": [{"start": 0.0, "end": 0.5, "votes": {"x": doubtful, MODEL: {"speaks": True}}}]},
        weights={},
        aggregator="min",
        snr_gate=None,
    )
    assert with_claim[0]["triage_score"] == pytest.approx(without[0]["triage_score"])
    assert with_claim[0]["confidence"] > without[0]["confidence"]
