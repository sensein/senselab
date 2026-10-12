"""COHORT across recordings: the recording condition, the non-target voice clusters and shared non-matches."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from senselab.audio.workflows.triage import cohort_fold
from senselab.audio.workflows.triage.cohort_fold import (
    CONDITION,
    KEY_NON_TASK_SELF_TALK,
    KEY_OTHER_SPEAKER_IN_SESSION,
    KEY_RECURRING_NON_TARGET_VOICE,
    KEY_SESSION_CONDITION_SHIFT,
    KEY_SESSION_SPEAKER_INCONSISTENT,
    KEY_SPEAKER_NONMATCH_SHARED,
    LABEL_RECURRING,
    LABEL_SELF_TALK,
    LABEL_SINGLE,
    MEASURED,
    NON_TARGET_VOICE,
    NOT_APPLICABLE,
    PARTICIPANT_NONMATCH,
    POOL_NON_TASK,
    cohort_annotations,
    cohort_evidence,
    cohort_grounds,
    condition_shift,
    hot_or_clipped,
    nonmatch_spans,
    shared_nonmatch,
)
from senselab.audio.workflows.triage.cohort_stage import (
    CONSISTENCY_CHECK,
    AudioLoader,
    EnrollmentConfig,
    RecordingFacts,
    cluster_voices,
    condition_blocks,
    condition_summary,
    envelope_snr_db,
    non_target_voices,
    participant_nonmatch,
    read_participant,
)
from senselab.audio.workflows.triage.decision import REVIEW

RATE = 100
DIM = 8
CUT = 0.231
PARTICIPANT = 1.0
EXAMINER = 2.0
STRANGER = 3.0


def _direction(level: float) -> np.ndarray:
    vector = np.zeros(DIM)
    vector[int(level)] = 1.0
    vector[DIM - 1] = 0.05
    return vector / np.linalg.norm(vector)


def _embed(samples: np.ndarray, rate: int) -> np.ndarray | None:
    if samples.size == 0:
        return None
    return _direction(round(float(np.median(samples))))


def _loader(levels: list[tuple[float, float, float]], duration: float = 20.0) -> AudioLoader:
    samples = np.zeros(int(duration * RATE), dtype=np.float32)
    for start, end, level in levels:
        samples[int(start * RATE) : int(end * RATE)] = level
    return lambda: (samples, RATE)


def _speech(
    session: str, index: int, *, rms: float | None = None, clipped: float | None = None, snr: float | None = None
) -> tuple[RecordingFacts, AudioLoader]:
    facts = RecordingFacts(
        stem=f"{session}_task-rainbow-passage-{index}",
        session=session,
        family="rainbow-passage",
        lexical_task=True,
        duration_s=20.0,
        task_extent=None,
        diarization=((1.0, 9.0, "A"),),
        lexical=((1.0, 9.0),),
        target=((1.0, 9.0),),
        rms_dbfs=rms,
        clipped_s=clipped,
        snr_db=snr,
    )
    return facts, _loader([(1.0, 9.0, PARTICIPANT)])


def _breath(session: str, index: int, talker: float) -> tuple[RecordingFacts, AudioLoader]:
    facts = RecordingFacts(
        stem=f"{session}_task-respiration-and-cough-breath-{index}",
        session=session,
        family="respiration-and-cough-breath",
        lexical_task=False,
        duration_s=20.0,
        task_extent=(2.0, 4.0),
        diarization=((2.0, 4.0, "A"), (10.0, 13.0, "B")),
        lexical=((10.2, 10.8), (11.0, 12.5)),
        non_task=((10.2, 10.8), (11.0, 12.5)),
    )
    return facts, _loader([(2.0, 4.0, PARTICIPANT), (10.0, 13.0, talker)])


def _read(sessions: list) -> dict[str, list[dict[str, Any]]]:
    settings = EnrollmentConfig(model_id="m", commit="c" * 40, cut=CUT, min_s=2.0, lexical_gap_s=0.5)
    return read_participant("sub-a", sessions, settings=settings, embed=_embed, artefact=None, quantiles_ref=None)[1]


COMMON: dict[str, Any] = {"branch_decisions": {}, "ran": {}, "hint_claims": {}, "route_state": None}


# ------------------------------------------------------------------------------ D1 condition


def test_envelope_snr_is_the_active_level_over_the_floor() -> None:
    """Envelope snr is the active level over the floor."""
    levels = np.concatenate([np.full(90, -60.0), np.full(10, -20.0)])
    assert envelope_snr_db(levels, 10, 95) == pytest.approx(40.0)
    assert envelope_snr_db(None, 10, 95) is None
    assert envelope_snr_db(np.array([np.nan]), 10, 95) is None


def test_a_condition_summary_takes_medians_and_the_clipped_share() -> None:
    """A condition summary takes medians and the clipped share."""
    members = [
        _speech("s", i, rms=r, clipped=c, snr=n)[0]
        for i, (r, c, n) in enumerate([(-30, 0.0, 30), (-10, 0.2, 20), (-20, None, None)])
    ]
    summary = condition_summary(members)
    assert summary == {"recordings_n": 3, "rms_dbfs_median": -20.0, "snr_db_median": 25.0, "clipped_fraction": 0.5}
    assert condition_summary([])["rms_dbfs_median"] is None


def test_each_session_sees_its_own_the_participants_and_the_other_sessions_condition() -> None:
    """Each session sees its own the participants and the other sessions condition."""
    one = [_speech("s1", i, rms=-30.0)[0] for i in range(2)]
    two = [_speech("s2", i, rms=-10.0)[0] for i in range(2)]
    blocks = condition_blocks([("s1", one), ("s2", two)])
    assert blocks["s1"]["session"]["rms_dbfs_median"] == -30.0
    assert blocks["s1"]["other_sessions"]["rms_dbfs_median"] == -10.0
    assert blocks["s2"]["participant"]["rms_dbfs_median"] == -20.0
    assert condition_blocks([("s1", one)])["s1"]["other_sessions"] is None


def test_hot_or_clipped_reads_the_data_bounds() -> None:
    """Hot or clipped reads the data bounds."""
    bounds = cohort_fold._parameters()["condition"]
    hot = {CONDITION: {"participant": {"rms_dbfs_median": bounds["hot_rms_dbfs_min"], "clipped_fraction": 0.0}}}
    clipped = {
        CONDITION: {"participant": {"rms_dbfs_median": -60.0, "clipped_fraction": bounds["clipped_fraction_min"]}}
    }
    quiet = {CONDITION: {"participant": {"rms_dbfs_median": -60.0, "clipped_fraction": 0.0}}}
    assert hot_or_clipped(hot) is True and hot_or_clipped(clipped) is True and hot_or_clipped(quiet) is False
    assert hot_or_clipped({}) is None and hot_or_clipped(quiet, "session") is None


def _inconsistent(level_shift: float) -> dict[str, list[dict[str, Any]]]:
    sessions = [
        ("sub-a_ses-1", [_speech("sub-a_ses-1", i, rms=-30.0, snr=30.0) for i in range(2)]),
        ("sub-a_ses-2", [_speech("sub-a_ses-2", i, rms=-30.0, snr=30.0) for i in range(2)]),
    ]
    other = [_speech("sub-a_ses-3", i, rms=-30.0 + level_shift, snr=30.0) for i in range(2)]
    other = [(f, _loader([(1.0, 9.0, STRANGER)])) for f, _ in other]
    return _read([*sessions, ("sub-a_ses-3", other)])


def test_a_session_inconsistency_the_condition_explains_is_an_annotation() -> None:
    """A session whose level moved by the shift bound is a condition shift, not another speaker."""
    from senselab.audio.workflows.triage.vocabulary import fold_file_verdict

    shift = float(cohort_fold._parameters()["condition"]["shift_level_db_min"])
    reading = _inconsistent(shift + 1.0)["sub-a_ses-3"][0]
    assert reading["checks"][CONSISTENCY_CHECK]["outcome"] == REVIEW
    assert (condition_shift(reading) or {}).get("explained") is True
    assert KEY_SESSION_SPEAKER_INCONSISTENT not in [key for _, key in cohort_grounds(reading)]
    assert [key for _, key in cohort_annotations(reading)] == [KEY_SESSION_CONDITION_SHIFT]
    folded = fold_file_verdict([], cohort=reading, **COMMON)
    assert KEY_SESSION_SPEAKER_INCONSISTENT not in folded.ground_keys
    assert KEY_SESSION_CONDITION_SHIFT in folded.annotation_keys
    assert "other_speaker" not in folded.reason_keys
    assert folded.triage is fold_file_verdict([], **COMMON).triage
    items = {entry.name: entry for entry in cohort_evidence(reading)}
    assert items["cohort.session_level_shift_db"].value == pytest.approx(shift + 1.0)


def test_a_session_inconsistency_in_the_same_condition_stays_a_ground() -> None:
    """A session inconsistency in the same condition stays a ground."""
    reading = _inconsistent(0.0)["sub-a_ses-3"][0]
    assert (condition_shift(reading) or {}).get("explained") is False
    assert KEY_SESSION_SPEAKER_INCONSISTENT in [key for _, key in cohort_grounds(reading)]
    assert cohort_annotations(reading) == []


# ------------------------------------------------------------------------------ D2 non-target voices


def test_average_linkage_merges_only_voices_over_the_cut() -> None:
    """Average linkage merges only voices over the cut."""
    vectors = {"a": _direction(2), "b": _direction(2), "c": _direction(3)}
    assert cluster_voices(vectors, CUT) == [["a", "b"], ["c"]]
    assert cluster_voices({}, CUT) == []


def test_a_voice_recurring_across_recordings_is_labelled_and_a_lone_one_is_single() -> None:
    """A voice recurring across recordings is labelled and a lone one is single."""
    pools = {
        "r1": {"session": "s1", "non_task": _direction(EXAMINER), "target": None},
        "r2": {"session": "s2", "non_task": _direction(EXAMINER), "target": None},
        "r3": {"session": "s1", "non_task": _direction(STRANGER), "target": None},
        "r4": {"session": "s1", "non_task": None, "target": _direction(PARTICIPANT)},
    }
    voices = non_target_voices(pools, cut=CUT, recurring_min=2)
    assert voices["r1"]["label"] == LABEL_RECURRING and voices["r1"]["cluster_sessions_n"] == 2
    assert voices["r1"]["cluster"] == voices["r2"]["cluster"] != voices["r3"]["cluster"]
    assert voices["r3"]["label"] == LABEL_SINGLE
    assert voices["r4"]["status"] == NOT_APPLICABLE


def test_a_voice_matching_the_participants_other_recordings_is_self_talk() -> None:
    """A voice matching the participants other recordings is self talk."""
    pools = {
        "r1": {"session": "s1", "non_task": _direction(PARTICIPANT), "target": None},
        "r2": {"session": "s1", "non_task": None, "target": _direction(PARTICIPANT)},
    }
    voice = non_target_voices(pools, cut=CUT, recurring_min=2)["r1"]
    assert voice["label"] == LABEL_SELF_TALK and voice["self_cosine"] > 0.99 and voice["self_recordings_n"] == 1


def test_the_examiner_in_two_breath_recordings_is_one_recurring_voice_end_to_end() -> None:
    """Two sessions' breath recordings carry the same other talker: one cluster, labelled recurring, still flagged."""
    sessions = [
        ("sub-a_ses-1", [_speech("sub-a_ses-1", i) for i in range(2)] + [_breath("sub-a_ses-1", 1, EXAMINER)]),
        ("sub-a_ses-2", [_speech("sub-a_ses-2", i) for i in range(2)] + [_breath("sub-a_ses-2", 1, EXAMINER)]),
    ]
    readings = _read(sessions)
    first, second = readings["sub-a_ses-1"][2], readings["sub-a_ses-2"][2]
    for reading in (first, second):
        voice = reading[NON_TARGET_VOICE]
        assert voice["status"] == MEASURED and voice["label"] == LABEL_RECURRING
        assert voice["cluster_recordings_n"] == 2 and voice["cluster_sessions_n"] == 2
        assert KEY_OTHER_SPEAKER_IN_SESSION in [key for _, key in cohort_grounds(reading)]
        assert KEY_RECURRING_NON_TARGET_VOICE in [key for _, key in cohort_annotations(reading)]
    assert readings["sub-a_ses-1"][0][NON_TARGET_VOICE]["status"] == NOT_APPLICABLE
    assert first[PARTICIPANT_NONMATCH] == {
        "recordings_compared_n": 6,
        "recordings_nonmatch_n": 2,
        "nonmatch_fraction": 0.3333,
    }


def test_self_talk_is_not_another_speaker() -> None:
    """Self talk is not another speaker."""
    reading = {
        "other_speaker": {
            "status": MEASURED,
            POOL_NON_TASK: {"status": MEASURED, "match": False, "spans": [[1.0, 4.0]], "pooled_s": 3.0},
        },
        NON_TARGET_VOICE: {"status": MEASURED, "label": LABEL_SELF_TALK, "self_cosine": 0.7, "self_recordings_n": 3},
    }
    assert cohort_grounds(reading) == [] and nonmatch_spans(reading) == []
    assert [key for _, key in cohort_annotations(reading)] == [KEY_NON_TASK_SELF_TALK]


# ------------------------------------------------------------------------------ D3 shared non-matches


def _shared(label: str = LABEL_SINGLE) -> dict[str, Any]:
    bounds = cohort_fold._parameters()["shared_nonmatch"]
    count = int(bounds["recordings_min"])
    return {
        "other_speaker": {
            "status": MEASURED,
            "target": {"status": MEASURED, "match": False, "spans": [[1.0, 4.0]], "pooled_s": 3.0},
        },
        NON_TARGET_VOICE: {"status": MEASURED, "label": label},
        PARTICIPANT_NONMATCH: {
            "recordings_compared_n": count,
            "recordings_nonmatch_n": count,
            "nonmatch_fraction": 1.0,
        },
    }


def test_participant_nonmatch_counts_compared_recordings() -> None:
    """Participant nonmatch counts compared recordings."""
    blocks: list[dict[str, Any]] = [
        {"status": MEASURED, "target": {"match": True}, "nonmatch": []},
        {"status": MEASURED, "target": {"match": False}, "nonmatch": ["target"]},
        {"status": MEASURED, "target": {"match": None}, "nonmatch": []},
        {"status": "unavailable"},
    ]
    assert participant_nonmatch(blocks) == {
        "recordings_compared_n": 2,
        "recordings_nonmatch_n": 1,
        "nonmatch_fraction": 0.5,
    }
    assert participant_nonmatch([])["nonmatch_fraction"] is None


def test_a_shared_uncorroborated_nonmatch_is_an_annotation() -> None:
    """A shared uncorroborated nonmatch is an annotation."""
    from senselab.audio.workflows.triage.vocabulary import fold_file_verdict

    reading = _shared()
    assert shared_nonmatch(reading)
    assert cohort_grounds(reading) == []
    assert [key for _, key in cohort_annotations(reading)] == [KEY_SPEAKER_NONMATCH_SHARED]
    folded = fold_file_verdict([], cohort=reading, **COMMON)
    assert KEY_OTHER_SPEAKER_IN_SESSION not in folded.ground_keys
    assert KEY_SPEAKER_NONMATCH_SHARED in folded.annotation_keys


def test_a_corroborated_or_recurring_shared_nonmatch_stays_a_ground() -> None:
    """A corroborated or recurring shared nonmatch stays a ground."""
    from senselab.audio.workflows.triage.vocabulary import fold_file_verdict

    assert [key for _, key in cohort_grounds(_shared(), corroborated=True)] == [KEY_OTHER_SPEAKER_IN_SESSION]
    assert [key for _, key in cohort_grounds(_shared(LABEL_RECURRING))] == [KEY_OTHER_SPEAKER_IN_SESSION]
    heard = fold_file_verdict([], cohort=_shared(), llm_redaction={"speakers": "more_than_one"}, **COMMON)
    assert KEY_OTHER_SPEAKER_IN_SESSION in heard.ground_keys


def test_a_nonmatch_under_the_shared_bounds_stays_a_ground() -> None:
    """A nonmatch under the shared bounds stays a ground."""
    reading = _shared()
    reading[PARTICIPANT_NONMATCH] = {"recordings_compared_n": 10, "recordings_nonmatch_n": 1, "nonmatch_fraction": 0.1}
    assert not shared_nonmatch(reading)
    assert [key for _, key in cohort_grounds(reading)] == [KEY_OTHER_SPEAKER_IN_SESSION]
