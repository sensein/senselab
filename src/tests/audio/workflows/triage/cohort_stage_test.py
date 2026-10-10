"""COHORT: session enrollment on synthetic embeddings, the duration check, the reading and its fold."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from senselab.audio.workflows.triage.cohort_fold import (
    COHORT_READING,
    KEY_OTHER_SPEAKER_IN_SESSION,
    KEY_RECORDING_DURATION_OUTLIER,
    MEASURED,
    NOT_APPLICABLE,
    UNAVAILABLE,
    cohort_evidence,
    cohort_grounds,
    non_task_speech_spans,
    nonmatch_spans,
)
from senselab.audio.workflows.triage.cohort_stage import (
    NO_COHORT,
    NO_ENROLLMENT_SPANS,
    AudioLoader,
    RecordingFacts,
    SessionEnrollment,
    build_quantile_artefact,
    candidate_runs,
    cohort_attributes,
    enroll_session,
    exclusive_pieces,
    match_runs,
    run_checks,
    window_regions,
    write_cohort_reading,
    write_cohort_unavailable,
)
from senselab.audio.workflows.triage.decision import ANNOTATION, REVIEW
from senselab.audio.workflows.triage.nodes.common import find_measurement, find_measurements
from senselab.utils.prov_store import ProvStore

RATE = 100
DIM = 8
CUT = 0.231
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
    duration=20.0,  # noqa: ANN001
    extent=None,  # noqa: ANN001
    speech=None,  # noqa: ANN001
    removed=(),  # noqa: ANN001
    events=(),  # noqa: ANN001
) -> RecordingFacts:
    """A recording's facts; speech-classified wherever diarized unless ``speech`` says otherwise."""
    return RecordingFacts(
        stem=stem,
        session="sub-a_ses-1",
        family=family,
        lexical_task=family == "rainbow-passage",
        duration_s=duration,
        task_extent=extent,
        diarization=tuple(diarization),
        lexical=tuple(lexical),
        speech=tuple((a, b) for a, b, _ in diarization) if speech is None else tuple(speech),
        removed_speech=tuple(removed),
        task_events=tuple(events),
    )


def _session() -> tuple[list, RecordingFacts, AudioLoader]:
    speech = [
        (
            _facts(f"sub-a_ses-1_task-rainbow-passage-{i}", "rainbow-passage", diarization=[(1.0, 9.0, "A")]),
            _loader([(1.0, 9.0, PARTICIPANT)], 20.0),
        )
        for i in range(2)
    ]
    breath = _facts(
        "sub-a_ses-1_task-respiration-and-cough-breath-1",
        "respiration-and-cough-breath",
        diarization=[(2.0, 4.0, "A"), (10.0, 13.0, "B")],
        lexical=[(10.2, 10.8), (11.0, 12.5)],
    )
    breath_audio = _loader([(2.0, 4.0, PARTICIPANT), (10.0, 13.0, TALKER)], 20.0)
    return speech, breath, breath_audio


def _enroll(members: list) -> SessionEnrollment:
    return enroll_session(
        "sub-a_ses-1", members, _embed, model_id="m", commit="c" * 40, cut=CUT, min_s=1.0, lexical_gap_s=0.5
    )


def test_session_enrollment_matches_the_participant_and_not_the_talker() -> None:
    """The talker's diarized and lexical runs fall under the cut; the participant's run matches."""
    speech, breath, breath_audio = _session()
    enrollment = _enroll(speech + [(breath, breath_audio)])
    assert enrollment.vector is not None
    assert sorted(enrollment.recordings) == sorted(f.stem for f, _ in speech), "only lexical speech enrolls"
    assert enrollment.single_speaker_only
    block = match_runs(breath, breath_audio, enrollment, _embed, cut=CUT, min_s=1.0, lexical_gap_s=0.5)
    by_source = {(r["source"], r["speaker"]): r for r in block["runs"]}
    assert by_source[("diarization", "A")]["match"] and by_source[("diarization", "A")]["cosine"] > 0.9
    assert not by_source[("diarization", "B")]["match"] and by_source[("diarization", "B")]["cosine"] < CUT
    assert not by_source[("lexical", None)]["match"]
    assert block["nonmatch_n"] == 2
    reading = cohort_attributes(breath, other_speaker=block, checks={}, quantiles=None, key="k")
    assert nonmatch_spans(reading) == [(10.0, 13.0)]
    assert non_task_speech_spans(reading) == [(10.0, 13.0)], "a breath task's other speech is non-task speech"
    assert [key for _, key in cohort_grounds(reading)] == [KEY_OTHER_SPEAKER_IN_SESSION]
    items = {entry.name: entry for entry in cohort_evidence(reading)}
    assert items["cohort.other_speaker_runs"].value == 2 and items["cohort.other_speaker_runs"].effect == REVIEW


def test_a_lexical_task_raises_the_ground_but_offers_no_non_task_speech() -> None:
    """Redaction reads non-matching speech as non-task speech only on a non-lexical task."""
    speech, _, _ = _session()
    enrollment = _enroll(speech)
    facts = _facts("sub-a_ses-1_task-rainbow-passage-9", "rainbow-passage", diarization=[(1.0, 5.0, "X")])
    block = match_runs(
        facts, _loader([(1.0, 5.0, TALKER)], 20.0), enrollment, _embed, cut=CUT, min_s=1.0, lexical_gap_s=0.5
    )
    reading = cohort_attributes(facts, other_speaker=block, checks={}, quantiles=None, key="k")
    assert nonmatch_spans(reading) == [(1.0, 5.0)]
    assert non_task_speech_spans(reading) == []


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


def test_exclusive_pieces_cut_overlap_and_short_runs_are_not_offered() -> None:
    """Overlapped time belongs to nobody; a run under the floor is never embedded."""
    pieces = exclusive_pieces([(0.0, 5.0, "A"), (4.0, 6.0, "B")])
    assert pieces == [(0.0, 4.0, "A"), (5.0, 6.0, "B")]
    facts = _facts("s", "rainbow-passage", diarization=[(0.0, 5.0, "A"), (4.0, 6.0, "B")], lexical=[(7.0, 7.5)])
    runs = candidate_runs(facts, min_s=1.0, lexical_gap_s=0.5)
    assert [(r.start, r.end, r.source) for r in runs] == [(0.0, 4.0, "diarization"), (5.0, 6.0, "diarization")]


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
        facts, other_speaker={"status": MEASURED, "runs": [], "nonmatch_n": 0}, checks={}, quantiles=None, key="a"
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
    block = {"status": status, "runs": [], "nonmatch_n": 0, "runs_n": 0, "cut": CUT, "lowest_cosine": 0.5}
    reading = cohort_attributes(facts, other_speaker=block, checks=checks, quantiles=None, key="k")
    assert [problem for entry in cohort_evidence(reading) for problem in row_problems(entry)] == []


def test_the_fold_reviews_on_a_cohort_ground_and_passes_on_an_unavailable_reading() -> None:
    """A non-matching run is review with reason other_speaker; an unavailable reading moves nothing."""
    from senselab.audio.workflows.triage.vocabulary import Triage, fold_file_verdict

    speech, breath, breath_audio = _session()
    block = match_runs(breath, breath_audio, _enroll(speech), _embed, cut=CUT, min_s=1.0, lexical_gap_s=0.5)
    flagged = cohort_attributes(breath, other_speaker=block, checks={}, quantiles=None, key="k")
    common: dict[str, Any] = {"branch_decisions": {}, "ran": {}, "hint_claims": {}, "route_state": None}
    folded = fold_file_verdict([], cohort=flagged, **common)
    assert folded.triage is Triage.REVIEW and "other_speaker" in folded.reason_keys
    assert KEY_OTHER_SPEAKER_IN_SESSION in folded.ground_keys
    assert any(entry.name == "cohort.other_speaker_runs" and entry.decisive for entry in folded.evidence)
    store = _store()
    write_cohort_unavailable(store)
    held = find_measurement(store, COHORT_READING)
    assert held is not None
    quiet = fold_file_verdict([], cohort=held.attributes, **common)
    assert KEY_OTHER_SPEAKER_IN_SESSION not in quiet.ground_keys
    assert quiet.triage == fold_file_verdict([], **common).triage


def test_a_diarized_run_without_speech_is_not_compared() -> None:
    """Diarized phonation or breathing, with no lexical word and no speech-classified frame, is not compared."""
    speech, _, _ = _session()
    vowel = _facts(
        "sub-a_ses-1_task-prolonged-vowel", "prolonged-vowel", diarization=[(1.0, 6.0, "A")], speech=[], lexical=[]
    )
    block = match_runs(
        vowel, _loader([(1.0, 6.0, TALKER)], 20.0), _enroll(speech), _embed, cut=CUT, min_s=1.0, lexical_gap_s=0.5
    )
    assert block["runs_n"] == 0 and block["nonmatch_n"] == 0


def test_task_events_are_cut_out_of_a_diarized_speech_run() -> None:
    """A speech-classified diarized run loses the task events inside it; what is left under 1.0 s is dropped."""
    facts = _facts("s", "prolonged-vowel", diarization=[(1.0, 6.0, "A")], events=[(1.5, 5.5)])
    assert candidate_runs(facts, min_s=1.0, lexical_gap_s=0.5) == []


def test_sliding_windows_catch_speech_no_diarized_segment_or_word_covers() -> None:
    """A removed talker before the diarizer's first segment is read by the windows: one non-matching run."""
    speech, _, _ = _session()
    facts = _facts(
        "sub-a_ses-1_task-random-item-generation",
        "random-item-generation",
        diarization=[(3.0, 12.0, "A")],
        lexical=[(3.2, 11.5)],
        speech=[(0.0, 12.0)],
        removed=[(0.0, 12.0)],
    )
    audio = _loader([(0.0, 2.4, TALKER), (2.4, 12.0, PARTICIPANT)], 20.0)
    assert window_regions(facts, min_s=1.0, lexical_gap_s=0.5) == [(0.0, 3.0)]
    block = match_runs(facts, audio, _enroll(speech), _embed, cut=CUT, min_s=1.0, lexical_gap_s=0.5)
    windows = [r for r in block["runs"] if r["source"] == "window"]
    assert [(r["start"], r["end"], r["match"]) for r in windows] == [(0.0, 2.5, False), (2.0, 3.0, True)]
    assert block["nonmatch_spans"] == [[0.0, 2.5]]


def test_sliding_windows_skip_task_events() -> None:
    """Removed speech inside a task event is the participant's task sound, not a window to read."""
    facts = _facts("s", "prolonged-vowel", speech=[(0.0, 8.0)], removed=[(0.0, 8.0)], events=[(0.5, 7.5)])
    assert window_regions(facts, min_s=1.0, lexical_gap_s=0.5) == []


def test_a_single_window_under_the_cut_is_not_a_run() -> None:
    """One window under the cut is not enough; two contiguous ones are."""
    from senselab.audio.workflows.triage.cohort_stage import window_runs

    cosines = {0.0: 0.1, 0.5: 0.9, 1.0: 0.9}
    runs = window_runs([(0.0, 2.0)], lambda span: cosines[span[0]], window_s=1.0, hop_s=0.5, cut=CUT, windows_min=2)
    assert [(r["start"], r["match"]) for r in runs] == [(0.5, True)]
