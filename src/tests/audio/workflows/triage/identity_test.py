"""A recording's identity: declared by the hint where it says, read off the file stem where it does not."""

from __future__ import annotations

from typing import Any

from senselab.audio.data_structures import AudioHints
from senselab.audio.workflows.triage.identity import (
    DECLARED_TASK_ID,
    FROM_HINT,
    FROM_STEM,
    PARTICIPANT_KEY,
    SESSION_KEY,
    TASK_TOKEN_KEY,
    declared_session_key,
    declared_task_id,
    identity_from_stem,
    identity_of,
    recorded_identity,
    session_key_of,
)
from senselab.audio.workflows.triage.routing_analysis.families import UNKNOWN_TASK, task_family, task_id_of
from senselab.utils.prov_store import ProvStore

STEM = "sub-a_ses-b_task-respiration-and-cough-fivebreaths-2"


def _recording(store: ProvStore, path: str, **attributes: Any) -> str:  # noqa: ANN401 -- entity attributes
    return store.entity(
        prov_type="stream", extent=(0.0, 1.0), attributes={"name": "recording", "path": path, **attributes}
    )


class TestFromTheStem:
    """With no hint, the identity is exactly what the stem parsers read."""

    def test_a_bids_stem_names_participant_session_and_task(self) -> None:
        """Every field comes from the stem, and the family drops the trailing index."""
        identity = identity_from_stem(STEM)
        assert (identity.participant, identity.session) == ("sub-a", "ses-b")
        assert identity.task_id == "respiration-and-cough-fivebreaths-2"
        assert identity.task_family == "respiration-and-cough-fivebreaths"
        assert set(identity.sources.values()) == {FROM_STEM}

    def test_the_readings_match_the_stem_parsers(self) -> None:
        """Task id, family and session key equal the parsers every caller used before."""
        for stem in (STEM, "sub-a_acq-x_ses-b_task-loudness", "plain-run-id", "sub-a_task-rainbow-passage", ""):
            identity = identity_from_stem(stem)
            assert identity.task_id == task_id_of(stem)
            assert identity.task_family == task_family(task_id_of(stem))
            assert identity.session_key == session_key_of(stem)

    def test_a_stem_with_no_task_entity_declares_none(self) -> None:
        """The task id is the parsers' own placeholder."""
        assert identity_from_stem("plain-run-id").task_id == UNKNOWN_TASK


class TestFromTheHint:
    """The hint's declaration wins field by field; whatever it leaves out comes from the stem."""

    def test_a_full_declaration_replaces_the_stem(self) -> None:
        """Participant, session and task come from the hint, and the token is lower-cased."""
        hint = AudioHints(metadata={PARTICIPANT_KEY: "P7", SESSION_KEY: "visit-2", TASK_TOKEN_KEY: " Cough-3 "})
        identity = identity_of("recording-0042", hint)
        assert (identity.participant, identity.session, identity.task_id) == ("P7", "visit-2", "cough-3")
        assert identity.task_family == "cough"
        assert identity.session_key == "P7_visit-2"
        assert set(identity.sources.values()) == {FROM_HINT}

    def test_a_partial_declaration_keeps_the_stem_for_the_rest(self) -> None:
        """A hint naming only the task leaves participant and session to the stem."""
        identity = identity_of(STEM, AudioHints(metadata={TASK_TOKEN_KEY: "loudness"}))
        assert (identity.participant, identity.session, identity.task_id) == ("sub-a", "ses-b", "loudness")
        assert identity.sources == {"participant": FROM_STEM, "session": FROM_STEM, "task_id": FROM_HINT}

    def test_a_blank_declaration_is_no_declaration(self) -> None:
        """An empty or whitespace value falls back to the stem."""
        identity = identity_of(STEM, AudioHints(metadata={TASK_TOKEN_KEY: "  ", PARTICIPANT_KEY: ""}))
        assert identity == identity_from_stem(STEM)


class TestRecordedOnTheRun:
    """What ADMIT recorded is read back; a store from before it recorded anything reads the stem."""

    def test_a_store_without_a_declaration_reads_the_stem(self) -> None:
        """An earlier run replays exactly as it ran."""
        store = ProvStore(run_id="old")
        _recording(store, f"/data/{STEM}.wav")
        assert recorded_identity(store) == identity_from_stem(STEM)
        assert declared_session_key(store) == "sub-a_ses-b"

    def test_a_recorded_declaration_is_read_back(self) -> None:
        """The attributes ADMIT wrote come back as the identity."""
        store = ProvStore(run_id="new")
        declared = identity_of(
            "x", AudioHints(metadata={PARTICIPANT_KEY: "P7", SESSION_KEY: "s1", TASK_TOKEN_KEY: "a-1"})
        )
        _recording(store, "/data/x.wav", **declared.as_attributes())
        assert recorded_identity(store) == declared
        assert declared_session_key(store) == "P7_s1"

    def test_no_recording_stream_is_no_identity(self) -> None:
        """Without ADMIT's stream there is nothing to read."""
        assert recorded_identity(ProvStore(run_id="empty")) is None


class TestDeclaredTaskId:
    """The one task id every stage reads."""

    def test_without_a_recorded_declaration_the_stem_is_read(self) -> None:
        """A store that recorded no hint gives the stem's task id; there is no live hint to consult."""
        store = ProvStore(run_id="c")
        _recording(store, f"/data/{STEM}.wav")
        assert declared_task_id(store) == "respiration-and-cough-fivebreaths-2"

    def test_a_task_recorded_from_a_hint_wins_over_the_stem(self) -> None:
        """A reader without the hint still sees what it declared."""
        store = ProvStore(run_id="c")
        declared = identity_of(STEM, AudioHints(metadata={TASK_TOKEN_KEY: "loudness"}))
        _recording(store, f"/data/{STEM}.wav", **declared.as_attributes())
        assert declared_task_id(store) == "loudness"

    def test_a_hint_that_declares_an_unknown_task_is_not_overruled_by_the_stem(self) -> None:
        """The hint is the declaration: a task it names as unknown stays unknown."""
        store = ProvStore(run_id="c")
        declared = identity_of(STEM, AudioHints(metadata={TASK_TOKEN_KEY: "unknown"}))
        _recording(store, f"/data/{STEM}.wav", **declared.as_attributes())
        assert declared_task_id(store) == UNKNOWN_TASK

    def test_a_task_recorded_from_the_stem_is_the_stem(self) -> None:
        """A stem-read declaration gives what the stem gives."""
        store = ProvStore(run_id="c")
        _recording(store, f"/data/{STEM}.wav", **identity_from_stem(STEM).as_attributes())
        assert declared_task_id(store) == "respiration-and-cough-fivebreaths-2"
        assert DECLARED_TASK_ID in store.entities("stream")[0].attributes

    def test_an_explicit_stem_is_read_where_no_hint_task_was_recorded(self) -> None:
        """A caller holding the stem it means passes it; a recorded hint task still wins."""
        store = ProvStore(run_id="c")
        _recording(store, f"/data/{STEM}.wav")
        assert declared_task_id(store, "sub-1_ses-1_task-loudness-3") == "loudness-3"

    def test_a_refused_recording_reads_what_the_admit_activity_recorded(self) -> None:
        """ADMIT wrote no stream; its activity carries the file and what the hint declared."""
        store = ProvStore(run_id="c")
        declared = identity_of("recording-0042", AudioHints(metadata={TASK_TOKEN_KEY: "cough-2"}))
        store.activity(
            node="ADMIT", step=None, parameters={"audio_file": "/data/recording-0042.wav", **declared.as_attributes()}
        )
        assert declared_task_id(store) == "cough-2"

    def test_a_refused_recording_without_a_declaration_reads_the_admitted_file(self) -> None:
        """A store from before ADMIT recorded declarations on its activity: the file's stem."""
        store = ProvStore(run_id="c")
        store.activity(node="ADMIT", step=None, parameters={"audio_file": f"/data/{STEM}.wav"})
        assert declared_task_id(store) == "respiration-and-cough-fivebreaths-2"

    def test_no_admit_record_is_unknown(self) -> None:
        """Neither a stream nor an activity: nothing declares a task."""
        assert declared_task_id(ProvStore(run_id="empty")) == UNKNOWN_TASK
