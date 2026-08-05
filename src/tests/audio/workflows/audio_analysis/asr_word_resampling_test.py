"""D-27: the asr axis is a gridded resampling of the two-part word confidence.

The defect this replaces, measured on ``english_conversation_higgs_audio_v2_20260805-034348``:
``harvest_asr_votes`` bucketed each model's words by time with ``fully_contained=True`` and took
pairwise WER over the bucket texts. Qwen3-ASR and CrisperWhisper produced **word-identical**
transcripts, yet bucket 0.5-1.5 held "you did that without" against "you did that" — a timing
difference reading as textual disagreement. Mean pairwise WER across buckets was 0.0751 while the
axis reported 0.4266.

Under D-27 the word carries the two doubts separately and the axis resamples them: doubt mass is
``1 - existence_confidence`` and how far it reaches is set by the word's temporal uncertainty. A
word every model agrees on but times loosely is a wide, low-mass contribution; a word they disagree
about is a narrow, high-mass one. The old scheme could not distinguish those, and they call for
different interventions.
"""

from __future__ import annotations

from typing import Any

import pytest

from senselab.audio.workflows.audio_analysis.asr import harvest_asr_votes, resample_word_doubt
from senselab.audio.workflows.audio_analysis.grid import BucketGrid


def _pass_summary(words_by_model: dict[str, list[tuple[float, float, str]]], *, duration_s: float) -> dict[str, Any]:
    """A pass summary in the shape the pipeline hands the harvest: ScriptLine trees per model."""
    from senselab.utils.data_structures import ScriptLine

    by_model: dict[str, Any] = {}
    for model, words in words_by_model.items():
        by_model[model] = {
            "status": "ok",
            "result": [
                ScriptLine(
                    text=" ".join(text for _s, _e, text in words),
                    start=words[0][0],
                    end=words[-1][1],
                    chunks=[
                        ScriptLine(text=text, start=start, end=end, timestamp_source="native", timestamp_model=model)
                        for start, end, text in words
                    ],
                )
            ],
        }
    return {"duration_s": duration_s, "asr": {"by_model": by_model}}


BUCKETS = [(round(i * 0.5, 6), round(i * 0.5 + 1.0, 6)) for i in range(6)]


def _word(text: str, start: float, end: float, existence: float, temporal: float | None) -> dict[str, Any]:
    return {
        "text": text,
        "start": start,
        "end": end,
        "existence_confidence": existence,
        "temporal_confidence": temporal,
    }


def test_a_bucket_no_word_reaches_is_absent_not_certain() -> None:
    """Absence of a claim is not a claim of absence — the rule this codebase turns on."""
    out = resample_word_doubt([_word("hi", 0.1, 0.3, 1.0, 1.0)], BUCKETS)
    assert out[(0.0, 1.0)] == pytest.approx(0.0), "a confident word leaves no doubt where it is"
    assert out.get((2.5, 3.5)) is None, "no word reaches here, so there is nothing to report"


def test_a_doubtful_word_puts_its_doubt_where_it_is() -> None:
    """Existence doubt is the mass; a well-localised word deposits it locally."""
    out = resample_word_doubt([_word("maybe", 2.6, 2.9, 0.25, 1.0)], BUCKETS)
    assert out[(2.5, 3.5)] == pytest.approx(0.75), out
    assert out.get((0.0, 1.0)) is None, "a well-localised word does not reach four buckets away"


def test_the_axis_carries_word_accuracy_and_nothing_else() -> None:
    """One question per axis: how much do the recognizers disagree about the word sequence.

    This axis has been through two wrong answers and the shape of both is worth keeping. Bucketed
    pairwise WER made timing jitter read as textual disagreement (0.4266 on word-identical
    transcripts). A joint of accuracy x localisation removed the conflation but not the ambiguity:
    0.788 could mean either half and a reader could not tell which. Localisation is now reported
    per edge on the word, and this axis answers only the accuracy question.

    A word every recognizer agreed on therefore leaves the axis settled, however poorly it is
    placed — and that is the honest reading, not a gap: there is no doubt about *what* was said.
    """
    agreed_but_smeared = _word("without", 1.2, 1.6, existence=1.0, temporal=0.1)
    out = resample_word_doubt([agreed_but_smeared], BUCKETS)
    values = [v for v in out.values() if v is not None]

    assert values, "the word has to reach some bucket"
    assert max(values) == pytest.approx(0.0), f"agreed text means a settled axis: {out}"


def test_localisation_changes_neither_the_mass_nor_the_reach() -> None:
    """Temporal agreement is excluded from this axis entirely, not merely down-weighted.

    Two words identical in accuracy and opposite in localisation must produce the same axis, or the
    number is once again answering two questions at once.
    """
    tight = resample_word_doubt([_word("what", 2.6, 2.9, 0.5, 1.0)], BUCKETS)
    loose = resample_word_doubt([_word("what", 2.6, 2.9, 0.5, 0.0)], BUCKETS)
    assert tight == loose, f"localisation moved the accuracy axis: tight={tight} loose={loose}"


def test_a_word_agreed_and_well_placed_leaves_no_doubt() -> None:
    """The other end of the same rule: both parts certain means the bucket is settled."""
    out = resample_word_doubt([_word("sure", 1.2, 1.6, existence=1.0, temporal=1.0)], BUCKETS)
    assert all(v == pytest.approx(0.0) for v in out.values() if v is not None), out


def test_a_word_deposits_its_doubt_over_its_own_span_only() -> None:
    """No smearing: the reach is the word, so the axis stays attributable to the words in a bucket.

    The projection may not manufacture doubt either — a bucket never reports more than the word it
    holds carries.
    """
    out = resample_word_doubt([_word("what", 2.6, 2.9, 0.5, 1.0)], BUCKETS)
    touched = {k for k, v in out.items() if v is not None}
    assert touched == {(2.0, 3.0), (2.5, 3.5)}, touched
    assert max(v for v in out.values() if v is not None) <= 0.5 + 1e-9


def test_unmeasured_localisation_does_not_smear() -> None:
    """``temporal_confidence`` is ``None`` for a lone timing source — unmeasured, not zero.

    Treating it as zero confidence would spread a single-witness word across the recording on the
    strength of a measurement nobody made.
    """
    out = resample_word_doubt([_word("solo", 2.6, 2.9, 0.5, None)], BUCKETS)
    touched = {k for k, v in out.items() if v is not None}
    assert touched == {(2.0, 3.0), (2.5, 3.5)}, touched


def test_two_words_in_one_bucket_combine_by_coverage() -> None:
    """A bucket holding one doubtful word among confident ones is partly doubtful, not fully."""
    words = [
        _word("sure", 2.55, 2.75, existence=1.0, temporal=1.0),
        _word("iffy", 2.80, 3.00, existence=0.0, temporal=1.0),
    ]
    out = resample_word_doubt(words, BUCKETS)
    value = out[(2.5, 3.5)]
    assert value is not None and 0.0 < value < 1.0, f"expected a blend, got {value}"


def test_no_words_at_all_reports_nothing() -> None:
    """A run with no transcript has an unmeasured asr axis, not a confident one."""
    assert all(v is None for v in resample_word_doubt([], BUCKETS).values())


def test_the_axis_folds_the_derivative_and_nothing_else() -> None:
    """The restructure, end to end through the fold (D-27, D-21 rule 6).

    The asr axis used to score three per-model signals whose values came from pairwise phoneme
    distance over bucketed text. Those transcripts are exactly what the consensus derivative folds,
    so scoring both is one body of evidence twice — and the per-bucket text is a reconstruction of
    what ``final/transcript.json`` holds at word resolution, which is why it is gone rather than
    kept beside the derivative as a record.

    Asserted through ``harvest_asr_votes`` rather than on a hand-built bucket, because the earlier
    version of this test constructed per-model texts and a pairwise block and then checked they were
    ignored — a shape the harvest no longer emits, so it proved nothing about the pipeline.
    """
    from senselab.audio.workflows.audio_analysis.fuse import per_signal_uncertainty

    harvested = harvest_asr_votes(
        pass_summary=_pass_summary(
            {
                "model-a": [(0.0, 0.4, "hello"), (0.4, 0.9, "there")],
                "model-b": [(0.0, 0.4, "hello"), (0.4, 0.9, "chair")],
            },
            duration_s=1.0,
        ),
        grid=BucketGrid(),
        alignment_by_model={},
    )
    scored = [b for b in harvested if b["votes"]]
    assert scored, "the harvest emitted no votes at all"
    for bucket in scored:
        assert set(bucket["votes"]) == {"consensus_words"}, f"the asr axis has one voter; got {sorted(bucket['votes'])}"
        assert set(per_signal_uncertainty(bucket)) == {"consensus_words"}


def test_the_fold_reads_the_shape_the_pipeline_actually_hands_it() -> None:
    """``resolve_asr_result`` returns ``ScriptLine`` objects, not dicts.

    Caught by a real run, not by a unit test: every test above builds dicts, and
    ``iter_word_leaves`` walks dicts only — so the fold silently produced no words, the
    ``consensus_words`` entry was never emitted, and the asr axis came out with **zero**
    contributing signals where it previously had three. An axis that vanishes is worse than one
    that is wrong, and nothing in the type system objected.
    """
    from senselab.audio.workflows.audio_analysis.asr import _consensus_word_doubt
    from senselab.utils.data_structures import ScriptLine

    line = ScriptLine(
        text="hello there",
        chunks=[
            ScriptLine(text="hello", start=0.0, end=0.4, timestamp_source="native", timestamp_model="m-a"),
            ScriptLine(text="there", start=0.4, end=0.9, timestamp_source="native", timestamp_model="m-a"),
        ],
    )
    other = ScriptLine(
        text="hello there",
        chunks=[
            ScriptLine(text="hello", start=0.02, end=0.42, timestamp_source="native", timestamp_model="m-b"),
            ScriptLine(text="there", start=0.41, end=0.92, timestamp_source="native", timestamp_model="m-b"),
        ],
    )
    doubt, provenance = _consensus_word_doubt({"m-a": [line], "m-b": [other]}, BUCKETS)

    assert provenance.get("n_words"), f"no words were folded out of the real shape: {provenance}"
    assert doubt[(0.0, 1.0)] is not None, "the first bucket holds both words and must carry a value"
    assert provenance["timing_sources"] == 2, provenance


def test_the_fold_is_exposed_so_two_axes_can_share_one_call() -> None:
    """The speaker axis needs these same words, and folding twice would run g2p twice for one answer.

    Asserted on the result rather than on a call count: what matters is that a caller can obtain the
    words once and hand them to both harvests, and that doing so gives the same axis values as letting
    the harvest fold them itself.
    """
    from senselab.audio.workflows.audio_analysis.asr import fuse_consensus_words
    from senselab.audio.workflows.audio_analysis.harvesters import resolve_asr_result

    summary = _pass_summary(
        {
            "model-a": [(0.0, 0.4, "hello"), (0.4, 0.9, "there")],
            "model-b": [(0.0, 0.4, "hello"), (0.4, 0.9, "chair")],
        },
        duration_s=1.0,
    )
    resolved = {
        m: resolve_asr_result(b, None) for m, b in summary["asr"]["by_model"].items() if b.get("status") == "ok"
    }
    words, provenance = fuse_consensus_words(resolved)
    assert words, "the fold produced no words"
    assert provenance["operator"] == "consensus_words/resample"
    assert all("temporal_confidence" in w for w in words), "the speaker axis reads this field"

    own = harvest_asr_votes(pass_summary=summary, grid=BucketGrid(), alignment_by_model={})
    shared = harvest_asr_votes(
        pass_summary=summary, grid=BucketGrid(), alignment_by_model={}, fused=(words, provenance)
    )
    assert own == shared, "handing the fold in must not change the axis"
