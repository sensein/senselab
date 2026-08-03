"""The background-mask axis becomes an ensemble: VAD, ASR words and speaker spans all vote.

Its uncertainty was `1 - confidence` of a single derived judgement, which read as a property of the
mask when it was a property of there being one producer. Three sources bear on whether the target was
active, so the axis's doubt is cross-source disagreement like every other axis's.

**What each source means depends on `--task-type`, which is why these are votes and not a formula.** In
a speech task, VAD / words / speaker spans indicate target *activity*. In a breathing task the target is
the breath, speech detection is silent through it, and a speech vote therefore indicates target
*absence* — the case that made a mask built from voice activity alone report the collected signal as a
background source.
"""

from __future__ import annotations

import pytest

from senselab.audio.workflows.audio_analysis.grid import BucketGrid
from senselab.audio.workflows.audio_analysis.mask_harvest import harvest_background_mask_evidence


def _grid() -> BucketGrid:
    return BucketGrid(win_length=0.1, hop_length=0.1)


def test_speech_evidence_indicates_target_activity_in_a_speech_task() -> None:
    """A bucket with speech is not target-free, so the mask should doubt calling it free."""
    rows = harvest_background_mask_evidence(
        duration_s=0.3,
        grid=_grid(),
        task_type="speech",
        speech_by_bucket={(0.0, 0.1): 0.95, (0.1, 0.2): 0.02},
    )
    by_bucket = {(r["start"], r["end"]): r["votes"] for r in rows}
    assert by_bucket[(0.0, 0.1)]["speech"]["target_active"] is True
    assert by_bucket[(0.1, 0.2)]["speech"]["target_active"] is False


def test_speech_evidence_indicates_target_absence_in_a_breathing_task() -> None:
    """The inversion, and the reason the task type cannot be a default.

    In a breathing task the target is the breath. Speech detection is *silent* during it, so speech
    present means something other than the target was happening — and a mask built from voice activity
    alone reported the collected signal as a background source.
    """
    rows = harvest_background_mask_evidence(
        duration_s=0.2,
        grid=_grid(),
        task_type="breathing",
        speech_by_bucket={(0.0, 0.1): 0.95},
    )
    votes = rows[0]["votes"]["speech"]
    assert votes["target_active"] is False, "speech is not the target here"


def test_the_three_sources_vote_separately() -> None:
    """Cross-source disagreement is the axis's uncertainty, so the sources must stay distinct."""
    rows = harvest_background_mask_evidence(
        duration_s=0.1,
        grid=_grid(),
        task_type="speech",
        speech_by_bucket={(0.0, 0.1): 0.9},
        word_coverage_by_bucket={(0.0, 0.1): 0.08},
        speaker_occupancy_by_bucket={(0.0, 0.1): 0.5},
    )
    assert set(rows[0]["votes"]) == {"speech", "words", "speakers"}


def test_disagreeing_sources_are_preserved_rather_than_reconciled() -> None:
    """VAD says speech, the ASR found no words: that disagreement *is* the measurement."""
    rows = harvest_background_mask_evidence(
        duration_s=0.1,
        grid=_grid(),
        task_type="speech",
        speech_by_bucket={(0.0, 0.1): 0.95},
        word_coverage_by_bucket={(0.0, 0.1): 0.0},
    )
    votes = rows[0]["votes"]
    assert votes["speech"]["target_active"] is True
    assert votes["words"]["target_active"] is False


def test_a_source_with_no_measurement_does_not_vote() -> None:
    """Absent is not "target-free". A source that said nothing must not be read as agreeing."""
    rows = harvest_background_mask_evidence(
        duration_s=0.1,
        grid=_grid(),
        task_type="speech",
        speech_by_bucket={(0.0, 0.1): 0.9},
    )
    assert set(rows[0]["votes"]) == {"speech"}, "words and speakers were never measured here"


def test_a_bucket_no_source_measured_yields_no_row() -> None:
    """No evidence is not evidence of a free region."""
    rows = harvest_background_mask_evidence(
        duration_s=0.3, grid=_grid(), task_type="speech", speech_by_bucket={(0.0, 0.1): 0.9}
    )
    assert [(r["start"], r["end"]) for r in rows] == [(0.0, 0.1)]


def test_the_votes_carry_an_uncertainty_fuse_axis_can_read() -> None:
    """The axis is fused by the same code as the others, so the vote shape must match."""
    rows = harvest_background_mask_evidence(
        duration_s=0.1, grid=_grid(), task_type="speech", speech_by_bucket={(0.0, 0.1): 0.5}
    )
    vote = rows[0]["votes"]["speech"]
    assert "same_label_uncertainty" in vote
    assert 0.0 <= vote["same_label_uncertainty"] <= 1.0


def test_an_ambiguous_speech_probability_is_more_uncertain_than_a_confident_one() -> None:
    """0.5 is the least informative reading; 0.99 and 0.01 are both informative."""

    def _u(p: float) -> float:
        rows = harvest_background_mask_evidence(
            duration_s=0.1, grid=_grid(), task_type="speech", speech_by_bucket={(0.0, 0.1): p}
        )
        return float(rows[0]["votes"]["speech"]["same_label_uncertainty"])

    assert _u(0.5) > _u(0.99)
    assert _u(0.5) > _u(0.01)


def test_an_unknown_task_type_is_refused() -> None:
    """The mapping from evidence to target activity depends on it, so a guess is a wrong answer."""
    with pytest.raises(ValueError, match="task_type"):
        harvest_background_mask_evidence(
            duration_s=0.1, grid=_grid(), task_type="interpretive-dance", speech_by_bucket={(0.0, 0.1): 0.9}
        )


def test_no_task_type_means_speech_and_says_so_in_provenance() -> None:
    """The pipeline's default, recorded on the row rather than assumed by a reader."""
    rows = harvest_background_mask_evidence(
        duration_s=0.1, grid=_grid(), task_type=None, speech_by_bucket={(0.0, 0.1): 0.9}
    )
    assert rows[0]["task_type"] == "speech"
