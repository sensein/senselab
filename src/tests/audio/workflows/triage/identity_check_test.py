"""The corpus identity check: what it refuses, what it only reports, under either session scope."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from senselab.audio.workflows.triage.identity import FROM_HINT, FROM_STEM, RecordingIdentity, identity_of
from senselab.audio.workflows.triage.identity_check import (
    CHECK_LEVELS,
    DUPLICATE_IDENTITY,
    DUPLICATE_RECORDING_ID,
    DUPLICATE_STEM,
    NO_RECORDED_IDENTITY,
    REFUSE,
    REPORT,
    SESSION_SCOPE_GLOBAL,
    SESSION_SCOPE_PARTICIPANT,
    SESSION_SCOPES,
    SESSION_UNDER_TWO_PARTICIPANTS,
    STEM_DISAGREES,
    TASK_NOT_DESCRIBED,
    UNREADABLE_IDENTITY,
    IdentityRecord,
    check_identities,
    identity_key,
)
from senselab.audio.workflows.triage.routing_analysis.families import UNKNOWN_TASK
from senselab.audio.workflows.triage.task_catalogue import (
    FROM_HINT_DESCRIPTION,
    FROM_TASK_LIST,
    NO_TASK,
    NOT_IN_TASK_LIST,
)


def _named(stem: str) -> RecordingIdentity:
    """This test's own file-name reader, standing in for a dataset's: a stem ``<participant>.<session>.<task id>``.

    senselab reads no file name (plan step 6a); the identity check only compares what a dataset's reader says a
    stem names against the declaration (:data:`STEM_DISAGREES`). Any other stem names nothing.
    """
    parts = stem.split(".")
    participant, session, task_id = parts if len(parts) == 3 else (None, None, UNKNOWN_TASK)
    return RecordingIdentity(
        participant, session, task_id, {"participant": FROM_STEM, "session": FROM_STEM, "task_id": FROM_STEM}
    )


def _identity(participant: str | None, session: str | None, task_id: str) -> RecordingIdentity:
    """The identity a hint declares, every field from the hint."""
    return RecordingIdentity(
        participant, session, task_id, {"participant": FROM_HINT, "session": FROM_HINT, "task_id": FROM_HINT}
    )


def _record(
    stem: str,
    participant: str | None = None,
    session: str | None = None,
    task_id: str = UNKNOWN_TASK,
    *,
    task: str | None = "listed",
    task_source: str | None = FROM_TASK_LIST,
) -> IdentityRecord:
    """A recording whose hint declares its identity, its declared task one the task list describes by default."""
    return IdentityRecord(
        stem,
        f"/corpus/{stem}.wav",
        _identity(participant, session, task_id),
        _named(stem),
        task=task,
        task_source=task_source,
    )


def _declared(stem: str, participant: str | None, session: str | None, task_id: str) -> IdentityRecord:
    """A recording whose hint declares every field, and a task the task list describes."""
    return _record(stem, participant, session, task_id)


def _check(records: list[IdentityRecord], scope: str):  # noqa: ANN202 -- an IdentityReport
    return check_identities(records, session_scope=scope)


@pytest.mark.parametrize("scope", SESSION_SCOPES)
def test_a_clean_corpus_passes_with_every_count_zero(scope: str) -> None:
    """Distinct identities, known tasks and agreeing stems make no finding."""
    report = _check(
        [
            _record("p1.s1.rainbow-passage", "p1", "s1", "rainbow-passage"),
            _record("p2.s2.harvard-sentences-list-1", "p2", "s2", "harvard-sentences-list-1"),
        ],
        scope,
    )
    assert not report.refused
    assert report.counts == dict.fromkeys(CHECK_LEVELS, 0)
    assert report.recordings == 2


@pytest.mark.parametrize(("scope", "reported"), [(SESSION_SCOPE_PARTICIPANT, False), (SESSION_SCOPE_GLOBAL, True)])
def test_a_session_identifier_under_two_participants_is_reported_only_under_global_scope(
    scope: str, reported: bool
) -> None:
    """``s1`` under two participants: two sessions per participant, or under global one session holding both.

    One session holding two participants is legitimate (a conversation), so it is reported, never refused.
    """
    report = _check(
        [_record("rec-0001", "p1", "s1", "rainbow-passage"), _record("rec-0002", "p2", "s1", "rainbow-passage")],
        scope,
    )
    assert not report.refused
    assert report.counts[SESSION_UNDER_TWO_PARTICIPANTS] == int(reported)
    if reported:
        (finding,) = report.findings
        assert finding.key == "s1"
        assert finding.detail["participants"] == ["p1", "p2"]
        assert set(finding.stems) == {"rec-0001", "rec-0002"}


@pytest.mark.parametrize("scope", SESSION_SCOPES)
def test_one_participant_session_and_task_twice_is_refused(scope: str) -> None:
    """Two recordings declaring the same identity collide, whatever their stems."""
    report = _check(
        [
            _declared("a", "p1", "s1", "harvard-sentences-list-1"),
            _declared("b", "p1", "s1", "harvard-sentences-list-1"),
        ],
        scope,
    )
    assert report.refused
    assert report.counts[DUPLICATE_IDENTITY] == 1
    (finding,) = [each for each in report.findings if each.check == DUPLICATE_IDENTITY]
    assert finding.key == "p1/s1/harvard-sentences-list-1"
    assert finding.stems == ("a", "b")
    assert finding.locations == ("/corpus/a.wav", "/corpus/b.wav")


@pytest.mark.parametrize("scope", SESSION_SCOPES)
def test_the_task_index_keeps_two_recordings_apart(scope: str) -> None:
    """``-1`` and ``-2`` of one family are two tasks."""
    report = _check(
        [
            _declared("a", "p1", "s1", "harvard-sentences-list-1"),
            _declared("b", "p1", "s1", "harvard-sentences-list-2"),
        ],
        scope,
    )
    assert report.counts[DUPLICATE_IDENTITY] == 0


@pytest.mark.parametrize("scope", SESSION_SCOPES)
def test_unnamed_recordings_are_keyed_by_stem_and_do_not_collide(scope: str) -> None:
    """Recordings declaring no participant or session, all of an unknown task, are each their own key."""
    records = [_record("clip_one"), _record("clip_two"), _record("clip_three")]
    assert {identity_key(record) for record in records} == {
        ("stem", "clip_one"),
        ("stem", "clip_two"),
        ("stem", "clip_three"),
    }
    report = _check(records, scope)
    assert report.counts[DUPLICATE_IDENTITY] == 0
    assert not report.refused


@pytest.mark.parametrize("scope", SESSION_SCOPES)
def test_a_stem_in_two_folders_is_reported_not_refused(scope: str) -> None:
    """Corpus passes key runs by place and stem, so a repeated stem is listed and nothing more."""
    stem = "rec-0001"
    report = _check(
        [
            _declared(stem, "p1", "s1", "rainbow-passage")._replace(location="/a/x.wav"),
            _declared(stem, "p1", "s2", "rainbow-passage")._replace(location="/b/x.wav"),
        ],
        scope,
    )
    assert report.counts[DUPLICATE_STEM] == 1
    assert CHECK_LEVELS[DUPLICATE_STEM] == REPORT
    (finding,) = [each for each in report.findings if each.check == DUPLICATE_STEM]
    assert finding.locations == ("/a/x.wav", "/b/x.wav")
    assert report.counts[DUPLICATE_IDENTITY] == 0
    assert not report.refused


@pytest.mark.parametrize("scope", SESSION_SCOPES)
def test_a_task_the_task_list_lacks_is_reported_per_task(scope: str) -> None:
    """A declared task nothing describes is listed under its name, every recording of it named."""
    report = _check(
        [
            _record("rec-0001", "p1", "s1", "humming-1", task="humming", task_source=NOT_IN_TASK_LIST),
            _record("rec-0002", "p1", "s1", "humming-2", task="humming", task_source=NOT_IN_TASK_LIST),
            _record("rec-0003", "p1", "s1", "rainbow-passage"),
        ],
        scope,
    )
    assert report.counts[TASK_NOT_DESCRIBED] == 1
    assert report.recordings_involved[TASK_NOT_DESCRIBED] == 2
    (finding,) = report.findings
    assert finding.key == "humming"
    assert finding.level == REPORT
    assert not report.refused


@pytest.mark.parametrize("scope", SESSION_SCOPES)
def test_the_task_is_the_declared_one_never_the_stems(scope: str) -> None:
    """A stem naming a task the list lacks is no finding when the hint declares a listed task, and vice versa."""
    report = _check(
        [
            _record("p1.s1.humming-1", "p1", "s1", "humming-1", task="rainbow-passage", task_source=FROM_TASK_LIST),
            _record(
                "p2.s2.rainbow-passage", "p2", "s2", "rainbow-passage", task="humming", task_source=NOT_IN_TASK_LIST
            ),
        ],
        scope,
    )
    assert [(finding.key, finding.stems) for finding in report.findings] == [("humming", ("p2.s2.rainbow-passage",))]


@pytest.mark.parametrize("scope", SESSION_SCOPES)
def test_a_task_described_only_in_its_hint_is_reported(scope: str) -> None:
    """No branch reads an inline description yet, so the recording would take every branch's out-of-family mode."""
    report = _check(
        [
            _record("rec-0001", "p1", "s1", "a", task="hum", task_source=FROM_HINT_DESCRIPTION),
            _record("rec-0002", "p1", "s1", "b", task=None, task_source=FROM_HINT_DESCRIPTION),
        ],
        scope,
    )
    assert sorted(finding.key for finding in report.findings) == [
        "(unnamed) (described only in the hint)",
        "hum (described only in the hint)",
    ]
    assert {finding.check for finding in report.findings} == {TASK_NOT_DESCRIBED}
    assert not report.refused


@pytest.mark.parametrize("scope", SESSION_SCOPES)
def test_no_task_declared_and_no_task_recorded_are_reported(scope: str) -> None:
    """A hint naming no task, and a run from before ADMIT recorded one, each have their own key."""
    report = _check(
        [
            _record("rec-0001", "p1", "s1", "a", task=None, task_source=NO_TASK),
            _record("rec-0002", "p1", "s1", "b", task=None, task_source=None),
        ],
        scope,
    )
    assert sorted(finding.key for finding in report.findings) == ["(no task declared)", "(no task recorded)"]
    assert not report.refused


@pytest.mark.parametrize("scope", SESSION_SCOPES)
def test_a_stem_disagreeing_with_the_declaration_is_reported(scope: str) -> None:
    """The declared identity wins; the stem's other reading (this test's reader) is listed field by field."""
    report = _check([_declared("p1.s1.rainbow-passage", "p9", "s1", "rainbow-passage")], scope)
    assert report.counts[STEM_DISAGREES] == 1
    (finding,) = report.findings
    assert finding.detail["fields"] == {"participant": {"declared": "p9", "stem": "p1"}}
    assert not report.refused


@pytest.mark.parametrize("scope", SESSION_SCOPES)
def test_a_stem_naming_nothing_cannot_disagree(scope: str) -> None:
    """A stem the dataset's reader cannot parse names nothing, so a declaration over it is the only reading."""
    report = _check([_declared("clip_one", "p1", "s1", "rainbow-passage")], scope)
    assert report.counts[STEM_DISAGREES] == 0


@pytest.mark.parametrize("scope", SESSION_SCOPES)
def test_a_run_with_no_recorded_identity_is_reported(scope: str) -> None:
    """A run from before ADMIT recorded identity is listed, keyed by its stem, not refused."""
    stem = "rec-0001"
    report = _check([IdentityRecord(stem, "p1/s1/run", None, _named(stem))], scope)
    assert report.counts[NO_RECORDED_IDENTITY] == 1
    (finding,) = report.findings
    assert finding.locations == ("p1/s1/run",)
    assert not report.refused


@pytest.mark.parametrize("scope", SESSION_SCOPES)
def test_an_identity_that_could_not_be_built_is_refused(scope: str) -> None:
    """A hint that raised leaves a recording nothing to be grouped by."""
    report = _check([IdentityRecord("x", "/corpus/x.wav", None, None, "hint: OSError: gone")], scope)
    assert report.refused
    assert report.counts[UNREADABLE_IDENTITY] == 1
    assert report.counts[NO_RECORDED_IDENTITY] == 0


def test_the_report_is_json_and_tuples_are_accepted() -> None:
    """``as_dict`` round-trips through JSON; a plain tuple reads as a record."""
    stem = "rec-0001"
    identity = _identity("p1", "s1", "rainbow-passage")
    report = check_identities(
        [
            (stem, "/a", identity, None, None, "rainbow-passage", FROM_TASK_LIST),
            (stem, "/b", identity, None, None, "rainbow-passage", FROM_TASK_LIST),
        ],
        session_scope=SESSION_SCOPE_PARTICIPANT,
    )
    payload = json.loads(json.dumps(report.as_dict()))
    assert payload["refused"] is True
    assert payload["counts"][DUPLICATE_IDENTITY] == 1
    assert payload["counts"][DUPLICATE_STEM] == 1
    assert payload["levels"][DUPLICATE_IDENTITY] == REFUSE
    assert {finding["check"] for finding in payload["findings"]} == {DUPLICATE_IDENTITY, DUPLICATE_STEM}


def test_an_unknown_scope_is_refused() -> None:
    """Only the two scopes the configuration accepts."""
    with pytest.raises(ValueError, match="session_scope"):
        check_identities([], session_scope="site")


@pytest.mark.parametrize("scope", SESSION_SCOPES)
def test_a_task_repeated_outside_any_session_is_no_duplicate(scope: str) -> None:
    """A participant with no sessions declared may take a task twice: each recording is its own session."""
    report = _check(
        [
            _declared("a", "p1", None, "harvard-sentences-list-1"),
            _declared("b", "p1", None, "harvard-sentences-list-1"),
        ],
        scope,
    )
    assert report.counts[DUPLICATE_IDENTITY] == 0


def test_a_repeated_recording_id_is_refused_before_a_run_and_reported_after() -> None:
    """Two flat-named files both called ``recording`` collide; one recording run twice is listed (step 9b)."""
    flat = [
        IdentityRecord("recording", "/c/p01/s2/recording.wav", RecordingIdentity("p01", "s2", "x", {}), None),
        IdentityRecord("recording", "/c/p02/s1/recording.wav", RecordingIdentity("p02", "s1", "x", {}), None),
    ]
    before = check_identities(flat, session_scope="participant", before_run=True)
    assert before.refused and before.counts[DUPLICATE_RECORDING_ID] == 1
    ided = [IdentityRecord(r.stem, r.location, replace(r.identity, recording_id=r.location[3:]), None) for r in flat]
    assert not check_identities(ided, session_scope="participant", before_run=True).counts[DUPLICATE_RECORDING_ID]
    after = check_identities(flat, session_scope="participant")
    assert after.counts[DUPLICATE_STEM] == 1 and not after.counts[DUPLICATE_RECORDING_ID]
