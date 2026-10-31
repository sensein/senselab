"""COHORT: session enrollment and pooled comparisons on synthetic embeddings, the duration check, the fold."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from senselab_core.runtime.provenance import ProvStore

from senselab.audio.workflows.triage.cohort_fold import (
    COHORT_READING,
    KEY_OTHER_SPEAKER_IN_SESSION,
    KEY_RECORDING_DURATION_OUTLIER,
    KEY_SESSION_SPEAKER_INCONSISTENT,
    MEASURED,
    NOT_APPLICABLE,
    NOT_MEASURED,
    POOL_NON_TASK,
    POOL_TARGET,
    UNAVAILABLE,
    cohort_evidence,
    cohort_grounds,
    non_task_speech_spans,
    nonmatch_spans,
)
from senselab.audio.workflows.triage.cohort_stage import (
    CONSISTENCY_CHECK,
    NO_COHORT,
    NO_ENROLLMENT_SPANS,
    SCOPE_PARTICIPANT,
    SOURCE_NONE,
    SOURCE_PARTICIPANT_OTHER_SESSIONS,
    SOURCE_SESSION,
    TOO_SHORT_FOR_SPEAKER_ID,
    AudioLoader,
    Enrollment,
    EnrollmentConfig,
    ParticipantEnrollments,
    RecordingFacts,
    build_quantile_artefact,
    cohort_attributes,
    compare_speaker,
    enroll_participant,
    enroll_session,
    exclusive_pieces,
    non_task_spans,
    read_participant,
    run_checks,
    write_cohort_reading,
    write_cohort_unavailable,
)
from senselab.audio.workflows.triage.decision import ANNOTATION, REVIEW
from senselab.audio.workflows.triage.nodes.common import find_measurement, find_measurements

RATE = 100
DIM = 8
CUT = 0.231
MIN_S = 2.0
PARTICIPANT = 1.0
TALKER = 2.0


def _direction(level: float) -> np.ndarray:
    vector = np.zeros(DIM)
    vector[int(level)] = 1.0
    vector[DIM - 1] = 0.05
    return vector / np.linalg.norm(vector)


def _embed(samples: np.ndarray, rate: int) -> np.ndarray | None:
    """A synthetic embedder: the span's sample level names its speaker."""
    if samples.size == 0:
        return None
    return _direction(round(float(np.median(samples))))


def _loader(levels: list[tuple[float, float, float]], duration: float) -> AudioLoader:
    samples = np.zeros(int(duration * RATE), dtype=np.float32)
    for start, end, level in levels:
        samples[int(start * RATE) : int(end * RATE)] = level
    return lambda: (samples, RATE)


def _facts(  # noqa: ANN202
    stem: str,
    family: str,
    *,
    diarization=(),  # noqa: ANN001
    lexical=(),  # noqa: ANN001
    target=(),  # noqa: ANN001
    non_task=(),  # noqa: ANN001
    duration=20.0,  # noqa: ANN001
    extent=None,  # noqa: ANN001
    removed=(),  # noqa: ANN001
    events=(),  # noqa: ANN001
    session="sub-a_ses-1",  # noqa: ANN001
) -> RecordingFacts:
    """A recording's facts."""
    return RecordingFacts(
        stem=stem,
        session=session,
        family=family,
        lexical_task=family == "rainbow-passage",
        duration_s=duration,
        task_extent=extent,
        diarization=tuple(diarization),
        lexical=tuple(lexical),
        target=tuple(target),
        non_task=tuple(non_task),
        removed_speech=tuple(removed),
        task_events=tuple(events),
    )


def _session() -> tuple[list, RecordingFacts, AudioLoader]:
    speech = [
        (
            _facts(
                f"sub-a_ses-1_task-rainbow-passage-{i}",
                "rainbow-passage",
                diarization=[(1.0, 9.0, "A")],
                lexical=[(1.0, 9.0)],
                target=[(1.0, 9.0)],
            ),
            _loader([(1.0, 9.0, PARTICIPANT)], 20.0),
        )
        for i in range(2)
    ]
    breath = _facts(
        "sub-a_ses-1_task-respiration-and-cough-breath-1",
        "respiration-and-cough-breath",
        diarization=[(2.0, 4.0, "A"), (10.0, 13.0, "B")],
        lexical=[(10.2, 10.8), (11.0, 12.5)],
        non_task=[(10.2, 10.8), (11.0, 12.5)],
    )
    breath_audio = _loader([(2.0, 4.0, PARTICIPANT), (10.0, 13.0, TALKER)], 20.0)
    return speech, breath, breath_audio


def _enroll(members: list) -> Enrollment:
    return enroll_session(
        "sub-a_ses-1", members, _embed, model_id="m", commit="c" * 40, cut=CUT, min_s=MIN_S, lexical_gap_s=0.5
    )


def _compare(facts: RecordingFacts, audio: AudioLoader, enrollment: Enrollment) -> dict:
    return compare_speaker(facts, audio, enrollment, _embed, min_s=MIN_S, lexical_gap_s=0.5)


def test_a_breath_task_pools_its_talker_words_and_has_no_target() -> None:
    """A breath recording: no target comparison; its talker's words pooled, under the cut, and flagged."""
    speech, breath, breath_audio = _session()
    enrollment = _enroll(speech + [(breath, breath_audio)])
    assert enrollment.vector is not None
    assert sorted(enrollment.recordings) == sorted(f.stem for f, _ in speech), "only lexical speech enrolls"
    assert enrollment.single_speaker_only
    block = _compare(breath, breath_audio, enrollment)
    assert block[POOL_TARGET]["status"] == NOT_APPLICABLE
    pool = block[POOL_NON_TASK]
    assert pool["status"] == MEASURED and pool["spans"] == [[10.2, 12.5]] and pool["pooled_s"] == 2.3
    assert not pool["match"] and pool["cosine"] < CUT
    assert block["nonmatch"] == [POOL_NON_TASK]
    reading = cohort_attributes(breath, other_speaker=block, checks={}, quantiles=None, key="k")
    assert nonmatch_spans(reading) == [(10.2, 12.5)]
    assert non_task_speech_spans(reading) == [(10.2, 12.5)], "a breath task's other speech is non-task speech"
    assert [key for _, key in cohort_grounds(reading)] == [KEY_OTHER_SPEAKER_IN_SESSION]
    items = {entry.name: entry for entry in cohort_evidence(reading)}
    assert items["cohort.non_task_cosine"].effect == REVIEW and "cohort.target_cosine" not in items


def test_a_speech_task_pools_its_extent_and_a_mismatch_is_not_non_task_speech() -> None:
    """A speech task's target pool under the cut raises the ground; redaction reads no non-task speech from it."""
    speech, _, _ = _session()
    enrollment = _enroll(speech)
    facts = _facts("sub-a_ses-1_task-rainbow-passage-9", "rainbow-passage", target=[(1.0, 5.0)])
    block = _compare(facts, _loader([(1.0, 5.0, TALKER)], 20.0), enrollment)
    assert block[POOL_TARGET]["status"] == MEASURED and not block[POOL_TARGET]["match"]
    assert block[POOL_NON_TASK]["status"] == NOT_APPLICABLE
    reading = cohort_attributes(facts, other_speaker=block, checks={}, quantiles=None, key="k")
    assert nonmatch_spans(reading) == [(1.0, 5.0)]
    assert non_task_speech_spans(reading) == []
    assert [key for _, key in cohort_grounds(reading)] == [KEY_OTHER_SPEAKER_IN_SESSION]
    matched = _compare(facts, _loader([(1.0, 5.0, PARTICIPANT)], 20.0), enrollment)
    assert matched[POOL_TARGET]["match"] and matched["nonmatch"] == []


def test_a_pool_under_two_seconds_is_not_measured_and_never_flags() -> None:
    """Owner, 2026-10-11: speaker identification needs 2 s; a shorter pool is an annotation."""
    speech, _, _ = _session()
    facts = _facts(
        "sub-a_ses-1_task-respiration-and-cough-breath-1",
        "respiration-and-cough-breath",
        non_task=[(10.0, 10.8), (11.0, 11.9)],
    )
    block = _compare(facts, _loader([(10.0, 12.0, TALKER)], 20.0), _enroll(speech))
    pool = block[POOL_NON_TASK]
    assert pool["status"] == NOT_MEASURED and pool["reason"] == TOO_SHORT_FOR_SPEAKER_ID and pool["pooled_s"] == 1.9
    reading = cohort_attributes(facts, other_speaker=block, checks={}, quantiles=None, key="k")
    assert cohort_grounds(reading) == [] and nonmatch_spans(reading) == []
    items = {entry.name: entry for entry in cohort_evidence(reading)}
    assert items["cohort.non_task_cosine"].effect == ANNOTATION and items["cohort.non_task_cosine"].value is None


def test_a_pool_is_its_spans_concatenated_and_embedded_once() -> None:
    """Short words far apart pool into one embedding of their joined audio."""
    speech, _, _ = _session()
    seen: list[int] = []

    def counting(samples: np.ndarray, rate: int) -> np.ndarray | None:
        seen.append(samples.size)
        return _embed(samples, rate)

    facts = _facts("s", "prolonged-vowel", non_task=[(1.0, 1.8), (5.0, 5.8), (9.0, 9.6)])
    enrollment = _enroll(speech)
    block = compare_speaker(
        facts, _loader([(1.0, 10.0, PARTICIPANT)], 20.0), enrollment, counting, min_s=MIN_S, lexical_gap_s=0.5
    )
    assert block[POOL_NON_TASK]["pooled_s"] == 2.2 and block[POOL_NON_TASK]["match"]
    assert seen == [220]


def test_a_session_with_no_lexical_speech_cannot_enroll() -> None:
    """No enrolling span: the enrollment names why, and nothing is compared."""
    _, breath, breath_audio = _session()
    enrollment = _enroll([(breath, breath_audio)])
    assert enrollment.vector is None and enrollment.reason == NO_ENROLLMENT_SPANS


def test_leave_one_out_drops_a_recording_another_speaker_holds() -> None:
    """With three enrolling recordings, one voiced by someone else is left out of the pool."""
    members = [
        (
            _facts(f"sub-a_ses-1_task-rainbow-passage-{i}", "rainbow-passage", diarization=[(1.0, 9.0, "A")]),
            _loader([(1.0, 9.0, TALKER if i == 2 else PARTICIPANT)], 20.0),
        )
        for i in range(3)
    ]
    enrollment = _enroll(members)
    assert enrollment.left_out == ["sub-a_ses-1_task-rainbow-passage-2"]
    assert float(enrollment.vector @ _direction(PARTICIPANT)) > 0.99


def test_exclusive_pieces_cut_overlap() -> None:
    """Overlapped time belongs to nobody."""
    pieces = exclusive_pieces([(0.0, 5.0, "A"), (4.0, 6.0, "B")])
    assert pieces == [(0.0, 4.0, "A"), (5.0, 6.0, "B")]


def _rows() -> list[dict]:
    rows = [
        {"stem": f"s{i}", "family": "prolonged-vowel", "duration_s": 10.0, "extent_s": 8.0, "extent_fraction": 0.8}
        for i in range(9)
    ]
    rows.append(
        {"stem": "s9", "family": "rainbow-passage", "duration_s": 30.0, "extent_s": 25.0, "extent_fraction": 0.8}
    )
    return rows


def test_a_long_recording_with_a_short_task_is_a_duration_outlier() -> None:
    """Longer than 3x its family's median with the task under half of it: review, with both comparisons."""
    artefact = build_quantile_artefact(_rows(), source={"kind": "test"}, commit=None, created_at=None)
    assert artefact["families"]["prolonged-vowel"]["median_duration_s"] == 10.0
    long_short = _facts("x", "prolonged-vowel", duration=40.0, extent=(1.0, 9.0))
    check = run_checks(long_short, artefact)["recording_duration_outlier"]
    assert check["status"] == MEASURED and check["outcome"] == REVIEW
    reading = cohort_attributes(
        long_short,
        other_speaker={"status": UNAVAILABLE},
        checks={"recording_duration_outlier": check},
        quantiles=None,
        key="k",
    )
    assert [key for _, key in cohort_grounds(reading)] == [KEY_RECORDING_DURATION_OUTLIER]
    names = {entry.name: entry for entry in cohort_evidence(reading)}
    assert names["cohort.duration_over_family_median"].value == 4.0
    assert names["cohort.task_extent_fraction"].value == 0.2
    long_full = _facts("y", "prolonged-vowel", duration=40.0, extent=(1.0, 39.0))
    assert run_checks(long_full, artefact)["recording_duration_outlier"]["outcome"] == "pass"


def test_the_duration_check_reads_only_its_declared_kinds() -> None:
    """A speech family is not read; a family absent from the artefact is unavailable."""
    artefact = build_quantile_artefact(_rows(), source={"kind": "test"}, commit=None, created_at=None)
    speech = _facts("z", "rainbow-passage", duration=400.0, extent=(1.0, 9.0))
    assert run_checks(speech, artefact)["recording_duration_outlier"]["status"] == NOT_APPLICABLE
    cough = _facts("w", "voluntary-cough", duration=400.0)
    assert run_checks(cough, artefact)["recording_duration_outlier"]["status"] == UNAVAILABLE


def _store() -> ProvStore:
    store = ProvStore(run_id="r")
    store.entity(
        prov_type="stream",
        extent=(0.0, 20.0),
        attributes={"name": "recording", "path": "/x/sub-a_ses-1_task-prolonged-vowel.wav"},
    )
    return store


def test_a_single_file_run_records_cohort_unavailable_and_folds_nothing() -> None:
    """No cohort: every check is unavailable, an annotation, and no ground is raised."""
    store = _store()
    reading_id = write_cohort_unavailable(store)
    reading = find_measurement(store, COHORT_READING)
    assert reading is not None and reading.id == reading_id
    assert reading.attributes["status"] == UNAVAILABLE
    assert reading.attributes["other_speaker"]["reason"] == NO_COHORT
    assert cohort_grounds(reading.attributes) == []
    assert {entry.effect for entry in cohort_evidence(reading.attributes)} == {ANNOTATION}
    assert write_cohort_unavailable(store) == reading_id


def test_a_corpus_reading_is_not_overwritten_by_the_single_file_fallback_and_rewrites_settle() -> None:
    """A measured reading stands against the fallback; a changed one supersedes it; an equal one mints nothing."""
    store = _store()
    facts = _facts("sub-a_ses-1_task-prolonged-vowel", "prolonged-vowel")
    measured = cohort_attributes(
        facts, other_speaker={"status": MEASURED, "nonmatch": []}, checks={}, quantiles=None, key="a"
    )
    first = write_cohort_reading(store, measured)
    assert write_cohort_unavailable(store) == first
    before = store.fingerprint()
    assert write_cohort_reading(store, measured) == first
    assert store.fingerprint() == before
    second = write_cohort_reading(store, {**measured, "cohort_key": "b"})
    assert [m.id for m in find_measurements(store, COHORT_READING)] == [second]
    assert store.is_invalidated(first)


def test_no_reading_folds_to_no_evidence() -> None:
    """A store COHORT never touched contributes nothing to the fold."""
    assert cohort_evidence(None) == [] and cohort_grounds(None) == []


@pytest.mark.parametrize("status", [MEASURED, UNAVAILABLE])
def test_every_cohort_item_is_a_well_formed_row(status: str) -> None:
    """Each item is one scalar row with a comparison of its type."""
    from senselab.audio.workflows.triage.decision import row_problems

    artefact = build_quantile_artefact(_rows(), source={"kind": "test"}, commit=None, created_at=None)
    facts = _facts("x", "prolonged-vowel", duration=40.0, extent=(1.0, 9.0))
    checks = run_checks(facts, artefact if status == MEASURED else None)
    block = {
        "status": status,
        POOL_TARGET: {"status": NOT_MEASURED, "reason": TOO_SHORT_FOR_SPEAKER_ID, "spans": [], "pooled_s": 0.0},
        POOL_NON_TASK: {
            "status": MEASURED,
            "spans": [[1.0, 4.0]],
            "pooled_s": 3.0,
            "cosine": 0.1,
            "cosine_min": CUT,
            "match": False,
        },
    }
    reading = cohort_attributes(facts, other_speaker=block, checks=checks, quantiles=None, key="k")
    assert [problem for entry in cohort_evidence(reading) for problem in row_problems(entry)] == []


def test_the_fold_reviews_on_a_cohort_ground_and_passes_on_an_unavailable_reading() -> None:
    """A non-matching pool is review with reason other_speaker; an unavailable reading moves nothing."""
    from senselab.audio.workflows.triage.vocabulary import Triage, fold_file_verdict

    speech, breath, breath_audio = _session()
    block = _compare(breath, breath_audio, _enroll(speech))
    flagged = cohort_attributes(breath, other_speaker=block, checks={}, quantiles=None, key="k")
    common: dict[str, Any] = {"branch_decisions": {}, "ran": {}, "hint_claims": {}, "route_state": None}
    folded = fold_file_verdict([], cohort=flagged, **common)
    assert folded.triage is Triage.REVIEW and "other_speaker" in folded.reason_keys
    assert KEY_OTHER_SPEAKER_IN_SESSION in folded.ground_keys
    assert any(entry.name == "cohort.non_task_cosine" and entry.decisive for entry in folded.evidence)
    store = _store()
    write_cohort_unavailable(store)
    held = find_measurement(store, COHORT_READING)
    assert held is not None
    quiet = fold_file_verdict([], cohort=held.attributes, **common)
    assert KEY_OTHER_SPEAKER_IN_SESSION not in quiet.ground_keys
    assert quiet.triage == fold_file_verdict([], **common).triage


def test_a_voice_task_without_words_compares_nothing() -> None:
    """Diarized phonation with no word of speech outside the task offers no pool."""
    speech, _, _ = _session()
    vowel = _facts("sub-a_ses-1_task-prolonged-vowel", "prolonged-vowel", diarization=[(1.0, 6.0, "A")])
    block = _compare(vowel, _loader([(1.0, 6.0, TALKER)], 20.0), _enroll(speech))
    assert block[POOL_TARGET]["status"] == NOT_APPLICABLE and block[POOL_NON_TASK]["status"] == NOT_APPLICABLE
    assert block["nonmatch"] == []


def test_removed_speech_holding_a_non_task_word_joins_the_pool_less_task_events() -> None:
    """The enhancer's removed speech around a word is pooled; removed speech with no word, or on an event, is not."""
    facts = _facts(
        "s",
        "prolonged-vowel",
        non_task=[(0.5, 0.9)],
        removed=[(0.0, 2.4), (5.0, 8.0)],
        events=[(2.0, 3.0)],
    )
    assert non_task_spans(facts, lexical_gap_s=0.5) == [(0.0, 2.0)]


# ------------------------------------------------------------------------------ participant enrollment


def _speech(session: str, index: int, level: float = PARTICIPANT) -> tuple[RecordingFacts, AudioLoader]:
    stem = f"{session}_task-rainbow-passage-{index}"
    facts = _facts(
        stem,
        "rainbow-passage",
        diarization=[(1.0, 9.0, "A")],
        lexical=[(1.0, 9.0)],
        target=[(1.0, 9.0)],
        session=session,
    )
    return facts, _loader([(1.0, 9.0, level)], 20.0)


def _breath(session: str) -> tuple[RecordingFacts, AudioLoader]:
    facts = _facts(
        f"{session}_task-respiration-and-cough-breath-1",
        "respiration-and-cough-breath",
        diarization=[(2.0, 4.0, "A"), (10.0, 13.0, "B")],
        lexical=[(10.2, 10.8), (11.0, 12.5)],
        non_task=[(10.2, 10.8), (11.0, 12.5)],
        session=session,
    )
    return facts, _loader([(2.0, 4.0, PARTICIPANT), (10.0, 13.0, TALKER)], 20.0)


def _participant(sessions: list, embed: Any = _embed) -> ParticipantEnrollments:  # noqa: ANN401
    return enroll_participant(
        "sub-a", sessions, embed, model_id="m", commit="c" * 40, cut=CUT, min_s=MIN_S, lexical_gap_s=0.5
    )


def _read(sessions: list) -> tuple[ParticipantEnrollments | None, dict]:
    settings = EnrollmentConfig(model_id="m", commit="c" * 40, cut=CUT, min_s=MIN_S, lexical_gap_s=0.5)
    return read_participant("sub-a", sessions, settings=settings, embed=_embed, artefact=None, quantiles_ref=None)


def test_participant_enrollment_pools_every_session_and_embeds_each_recording_once() -> None:
    """The pool holds every session's recordings; one recording's spans are embedded once for all pools."""
    calls: list[int] = []

    def counting(samples: np.ndarray, rate: int) -> np.ndarray | None:
        calls.append(samples.size)
        return _embed(samples, rate)

    sessions = [(s, [_speech(s, i) for i in range(2)]) for s in ("sub-a_ses-1", "sub-a_ses-2")]
    found = _participant(sessions, counting)
    assert found.pooled.scope == SCOPE_PARTICIPANT
    assert len(found.pooled.recordings) == 4 and found.pooled.sessions == ["sub-a_ses-1", "sub-a_ses-2"]
    assert found.pooled.single_speaker_only and found.pooled.vector is not None
    assert float(found.pooled.vector @ _direction(PARTICIPANT)) > 0.99
    assert len(calls) == 4, "four recordings of one span each, embedded once across five pools"


def test_leave_one_session_out_excludes_the_session_compared() -> None:
    """Each session's participant enrollment is pooled from the other sessions only."""
    sessions = [(s, [_speech(s, i) for i in range(2)]) for s in ("sub-a_ses-1", "sub-a_ses-2", "sub-a_ses-3")]
    found = _participant(sessions)
    for session, _ in sessions:
        other = found.other_sessions[session]
        assert other.excluded_session == session
        assert session not in other.sessions and len(other.sessions) == 2
        assert not any(stem.startswith(session) for stem in other.recordings)
        assert len(other.recordings) == 4


def test_a_session_far_from_the_other_sessions_is_inconsistent_and_reviewed() -> None:
    """A session voiced by someone else falls under the bound against the rest; the others pass."""
    sessions = [
        ("sub-a_ses-1", [_speech("sub-a_ses-1", i) for i in range(2)]),
        ("sub-a_ses-2", [_speech("sub-a_ses-2", i) for i in range(2)]),
        ("sub-a_ses-3", [_speech("sub-a_ses-3", i, TALKER) for i in range(2)]),
    ]
    _, readings = _read(sessions)
    outcomes = {s: readings[s][0]["checks"][CONSISTENCY_CHECK]["outcome"] for s, _ in sessions}
    assert outcomes == {"sub-a_ses-1": "pass", "sub-a_ses-2": "pass", "sub-a_ses-3": REVIEW}
    flagged = readings["sub-a_ses-3"][1]
    assert KEY_SESSION_SPEAKER_INCONSISTENT in [key for _, key in cohort_grounds(flagged)]
    items = {entry.name: entry for entry in cohort_evidence(flagged)}
    assert items["cohort.session_vs_other_sessions_cosine"].effect == REVIEW
    assert items["cohort.session_vs_other_sessions_cosine"].value < CUT
    assert KEY_SESSION_SPEAKER_INCONSISTENT not in [key for _, key in cohort_grounds(readings["sub-a_ses-1"][0])]


def test_the_fold_reviews_an_inconsistent_session_as_other_speaker() -> None:
    """The ground maps to the other_speaker reason."""
    from senselab.audio.workflows.triage.vocabulary import Triage, fold_file_verdict

    sessions = [
        ("sub-a_ses-1", [_speech("sub-a_ses-1", i) for i in range(2)]),
        ("sub-a_ses-2", [_speech("sub-a_ses-2", i, TALKER) for i in range(2)]),
    ]
    _, readings = _read(sessions)
    common: dict[str, Any] = {"branch_decisions": {}, "ran": {}, "hint_claims": {}, "route_state": None}
    folded = fold_file_verdict([], cohort=readings["sub-a_ses-2"][0], **common)
    assert folded.triage is Triage.REVIEW and "other_speaker" in folded.reason_keys
    assert KEY_SESSION_SPEAKER_INCONSISTENT in folded.ground_keys


def test_a_session_without_an_enrollment_falls_back_to_the_other_sessions() -> None:
    """No lexical speech in the session: its members are compared against the other sessions' pool."""
    sessions = [
        ("sub-a_ses-1", [_speech("sub-a_ses-1", i) for i in range(2)]),
        ("sub-a_ses-2", [_breath("sub-a_ses-2")]),
    ]
    found, readings = _read(sessions)
    assert found is not None and found.sessions["sub-a_ses-2"].vector is None
    block = readings["sub-a_ses-2"][0]["other_speaker"]
    assert block["status"] == MEASURED and block["enrollment_source"] == SOURCE_PARTICIPANT_OTHER_SESSIONS
    assert block["enrollment"]["excluded_session"] == "sub-a_ses-2"
    assert block["nonmatch_spans"] == [[10.2, 12.5]]
    assert readings["sub-a_ses-1"][0]["other_speaker"]["enrollment_source"] == SOURCE_SESSION
    assert readings["sub-a_ses-2"][0]["checks"][CONSISTENCY_CHECK]["status"] == NOT_APPLICABLE, "one enrolled session"


def test_a_session_compared_with_its_own_enrollment_also_reports_the_participant_cosine() -> None:
    """Each measured pool carries its cosine against the other sessions' pool too."""
    sessions = [
        ("sub-a_ses-1", [_speech("sub-a_ses-1", i) for i in range(2)] + [_breath("sub-a_ses-1")]),
        ("sub-a_ses-2", [_speech("sub-a_ses-2", i) for i in range(2)]),
    ]
    _, readings = _read(sessions)
    block = readings["sub-a_ses-1"][2]["other_speaker"]
    assert block[POOL_NON_TASK]["participant_cosine"] < CUT
    assert readings["sub-a_ses-1"][0]["other_speaker"][POOL_TARGET]["participant_cosine"] > 0.99


def test_a_single_session_participant_has_no_consistency_check() -> None:
    """One session: its own enrollment, no participant cosine, and the check is not applicable."""
    sessions = [("sub-a_ses-1", [_speech("sub-a_ses-1", i) for i in range(2)] + [_breath("sub-a_ses-1")])]
    found, readings = _read(sessions)
    assert found is not None and found.other_sessions == {}
    reading = readings["sub-a_ses-1"][2]
    assert reading["other_speaker"]["enrollment_source"] == SOURCE_SESSION
    assert all("participant_cosine" not in reading["other_speaker"][pool] for pool in (POOL_TARGET, POOL_NON_TASK))
    assert reading["checks"][CONSISTENCY_CHECK]["status"] == NOT_APPLICABLE
    assert KEY_SESSION_SPEAKER_INCONSISTENT not in [key for _, key in cohort_grounds(reading)]
    assert not any("session" in entry.name for entry in cohort_evidence(reading))


def test_a_participant_with_no_enrollment_anywhere_compares_nothing() -> None:
    """No session enrolls: the source is none and the block names why."""
    _, readings = _read([("sub-a_ses-1", [_breath("sub-a_ses-1")]), ("sub-a_ses-2", [_breath("sub-a_ses-2")])])
    block = readings["sub-a_ses-1"][0]["other_speaker"]
    assert block["status"] == UNAVAILABLE and block["enrollment_source"] == SOURCE_NONE
    assert block["reason"] == NO_ENROLLMENT_SPANS


def test_a_recording_is_compared_with_its_session_enrollment_left_out() -> None:
    """Two enrolling recordings, one voiced by someone else: each is compared with the other's enrollment only."""
    sessions = [("sub-a_ses-1", [_speech("sub-a_ses-1", 0), _speech("sub-a_ses-1", 1, TALKER)])]
    found, readings = _read(sessions)
    assert found is not None and sorted(found.sessions["sub-a_ses-1"].recordings) == [
        "sub-a_ses-1_task-rainbow-passage-0",
        "sub-a_ses-1_task-rainbow-passage-1",
    ]
    talker = readings["sub-a_ses-1"][1]["other_speaker"]
    assert talker["enrollment"]["excluded_recording"] == "sub-a_ses-1_task-rainbow-passage-1"
    assert talker["enrollment"]["recordings"] == ["sub-a_ses-1_task-rainbow-passage-0"]
    assert not talker[POOL_TARGET]["match"] and talker["nonmatch"] == [POOL_TARGET]
