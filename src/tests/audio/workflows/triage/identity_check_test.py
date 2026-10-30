"""The corpus identity check: what it refuses, what it only reports, under either session scope."""

from __future__ import annotations

import json

import pytest

from senselab.audio.workflows.triage.identity import FROM_HINT, RecordingIdentity, identity_from_stem, identity_of
from senselab.audio.workflows.triage.identity_check import (
    CHECK_LEVELS,
    DUPLICATE_IDENTITY,
    DUPLICATE_STEM,
    NO_RECORDED_IDENTITY,
    REFUSE,
    REPORT,
    SESSION_SCOPE_GLOBAL,
    SESSION_SCOPE_PARTICIPANT,
    SESSION_SCOPES,
    SESSION_UNDER_TWO_PARTICIPANTS,
    STEM_DISAGREES,
    UNKNOWN_TASK_FAMILY,
    UNREADABLE_IDENTITY,
    IdentityRecord,
    check_identities,
    identity_key,
    triage_families,
)

KNOWN = frozenset({"rainbow-passage", "harvard-sentences-list"})


def _record(stem: str, location: str | None = None) -> IdentityRecord:
    """A recording whose identity is read off its stem, as a run with no hint declares it."""
    return IdentityRecord(stem, location or f"/corpus/{stem}.wav", identity_of(stem), identity_from_stem(stem))


def _declared(stem: str, participant: str | None, session: str | None, task_id: str) -> IdentityRecord:
    """A recording whose hint declares every field."""
    sources = {"participant": FROM_HINT, "session": FROM_HINT, "task_id": FROM_HINT}
    identity = RecordingIdentity(participant, session, task_id, sources)
    return IdentityRecord(stem, f"/corpus/{stem}.wav", identity, identity_from_stem(stem))


def _check(records: list[IdentityRecord], scope: str):  # noqa: ANN202 -- an IdentityReport
    return check_identities(records, session_scope=scope, known_families=KNOWN)


@pytest.mark.parametrize("scope", SESSION_SCOPES)
def test_a_clean_corpus_passes_with_every_count_zero(scope: str) -> None:
    """Distinct identities, known tasks and agreeing stems make no finding."""
    report = _check(
        [_record("sub-1_ses-1_task-rainbow-passage"), _record("sub-2_ses-2_task-harvard-sentences-list-1")], scope
    )
    assert not report.refused
    assert report.counts == dict.fromkeys(CHECK_LEVELS, 0)
    assert report.recordings == 2


@pytest.mark.parametrize(("scope", "reported"), [(SESSION_SCOPE_PARTICIPANT, False), (SESSION_SCOPE_GLOBAL, True)])
def test_a_session_identifier_under_two_participants_is_reported_only_under_global_scope(
    scope: str, reported: bool
) -> None:
    """``ses-1`` under two participants: two sessions per participant, or under global one session holding both.

    One session holding two participants is legitimate (a conversation), so it is reported, never refused.
    """
    report = _check([_record("sub-1_ses-1_task-rainbow-passage"), _record("sub-2_ses-1_task-rainbow-passage")], scope)
    assert not report.refused
    assert report.counts[SESSION_UNDER_TWO_PARTICIPANTS] == int(reported)
    if reported:
        (finding,) = report.findings
        assert finding.key == "ses-1"
        assert finding.detail["participants"] == ["sub-1", "sub-2"]
        assert set(finding.stems) == {"sub-1_ses-1_task-rainbow-passage", "sub-2_ses-1_task-rainbow-passage"}


@pytest.mark.parametrize("scope", SESSION_SCOPES)
def test_one_participant_session_and_task_twice_is_refused(scope: str) -> None:
    """Two recordings declaring the same identity collide, whatever their stems."""
    report = _check(
        [
            _declared("a", "sub-1", "ses-1", "harvard-sentences-list-1"),
            _declared("b", "sub-1", "ses-1", "harvard-sentences-list-1"),
        ],
        scope,
    )
    assert report.refused
    assert report.counts[DUPLICATE_IDENTITY] == 1
    (finding,) = [each for each in report.findings if each.check == DUPLICATE_IDENTITY]
    assert finding.key == "sub-1/ses-1/harvard-sentences-list-1"
    assert finding.stems == ("a", "b")
    assert finding.locations == ("/corpus/a.wav", "/corpus/b.wav")


@pytest.mark.parametrize("scope", SESSION_SCOPES)
def test_the_task_index_keeps_two_recordings_apart(scope: str) -> None:
    """``-1`` and ``-2`` of one family are two tasks."""
    report = _check(
        [
            _declared("a", "sub-1", "ses-1", "harvard-sentences-list-1"),
            _declared("b", "sub-1", "ses-1", "harvard-sentences-list-2"),
        ],
        scope,
    )
    assert report.counts[DUPLICATE_IDENTITY] == 0


@pytest.mark.parametrize("scope", SESSION_SCOPES)
def test_unnamed_recordings_are_keyed_by_stem_and_do_not_collide(scope: str) -> None:
    """Files naming no participant or session, all of an unknown task, are each their own key."""
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
    stem = "sub-1_ses-1_task-rainbow-passage"
    report = _check(
        [
            _declared(stem, "sub-1", "ses-1", "rainbow-passage")._replace(location="/a/x.wav"),
            _declared(stem, "sub-1", "ses-2", "rainbow-passage")._replace(location="/b/x.wav"),
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
def test_an_unknown_task_family_is_reported_per_family(scope: str) -> None:
    """A task no expectation table knows is listed under its family, every recording of it named."""
    report = _check(
        [
            _record("sub-1_ses-1_task-humming-1"),
            _record("sub-1_ses-1_task-humming-2"),
            _record("sub-1_ses-1_task-rainbow-passage"),
        ],
        scope,
    )
    assert report.counts[UNKNOWN_TASK_FAMILY] == 1
    assert report.recordings_involved[UNKNOWN_TASK_FAMILY] == 2
    (finding,) = report.findings
    assert finding.key == "humming"
    assert finding.level == REPORT
    assert not report.refused


@pytest.mark.parametrize("scope", SESSION_SCOPES)
def test_a_stem_disagreeing_with_the_declaration_is_reported(scope: str) -> None:
    """The declared identity wins; the stem's other reading is listed field by field."""
    report = _check([_declared("sub-1_ses-1_task-rainbow-passage", "sub-9", "ses-1", "rainbow-passage")], scope)
    assert report.counts[STEM_DISAGREES] == 1
    (finding,) = report.findings
    assert finding.detail["fields"] == {"participant": {"declared": "sub-9", "stem": "sub-1"}}
    assert not report.refused


@pytest.mark.parametrize("scope", SESSION_SCOPES)
def test_a_stem_naming_nothing_cannot_disagree(scope: str) -> None:
    """A non-BIDS stem names no entity, so a declaration over it is the only reading."""
    report = _check([_declared("clip_one", "sub-1", "ses-1", "rainbow-passage")], scope)
    assert report.counts[STEM_DISAGREES] == 0


@pytest.mark.parametrize("scope", SESSION_SCOPES)
def test_a_run_with_no_recorded_identity_is_reported(scope: str) -> None:
    """A run from before ADMIT recorded identity is listed, keyed by its stem, not refused."""
    stem = "sub-1_ses-1_task-rainbow-passage"
    report = _check([IdentityRecord(stem, "sub-1/ses-1/run", None, identity_from_stem(stem))], scope)
    assert report.counts[NO_RECORDED_IDENTITY] == 1
    (finding,) = report.findings
    assert finding.locations == ("sub-1/ses-1/run",)
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
    stem = "sub-1_ses-1_task-rainbow-passage"
    report = check_identities(
        [(stem, "/a", identity_of(stem), None), (stem, "/b", identity_of(stem), None)],
        session_scope=SESSION_SCOPE_PARTICIPANT,
        known_families=KNOWN,
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
        check_identities([], session_scope="site", known_families=KNOWN)


def test_triage_knows_every_expectation_tables_families() -> None:
    """Every branch's table, and the VOICE families awaiting a declaration."""
    families = triage_families()
    assert {"rainbow-passage", "loudness"} <= families
    assert "humming" not in families


@pytest.mark.parametrize("scope", SESSION_SCOPES)
def test_a_task_repeated_outside_any_session_is_no_duplicate(scope: str) -> None:
    """A participant with no sessions declared may take a task twice: each recording is its own session."""
    report = _check(
        [
            _declared("a", "sub-1", None, "harvard-sentences-list-1"),
            _declared("b", "sub-1", None, "harvard-sentences-list-1"),
        ],
        scope,
    )
    assert report.counts[DUPLICATE_IDENTITY] == 0
