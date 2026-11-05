"""A recording's identity: declared by the hint where it says, read off the file stem where it does not."""

from __future__ import annotations

from typing import Any

import pytest

from senselab.audio.data_structures import AudioHints, TargetSpeaker
from senselab.audio.workflows.triage.b2ai_hints import identity_from_name, task_id_of
from senselab.audio.workflows.triage.identity import (
    ADMIT_NODE,
    DECLARED_RECORDING_ID,
    DECLARED_TARGET_SPEAKER,
    DECLARED_TASK_ID,
    FROM_CORPUS_PATH,
    FROM_HINT,
    FROM_STEM,
    IDENTITY_FROM_KEY,
    PARTICIPANT_KEY,
    RECORDING_ID_COLUMN,
    RECORDING_ID_FIELD,
    RECORDING_ID_KEY,
    ROW_TARGET_SPEAKER,
    SESSION_KEY,
    TASK_TOKEN_KEY,
    IdentityLabelError,
    ManifestWithoutIdentity,
    MultipleTargetSpeakers,
    RecordingIdentity,
    corpus_path_recorded,
    declared_session_key,
    declared_task_id,
    identity_columns,
    identity_disagreement,
    identity_from_attributes,
    identity_of,
    recorded_identity,
    row_groups,
    row_identity,
)
from senselab.audio.workflows.triage.routing_analysis.families import UNKNOWN_TASK, task_family
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
        identity = identity_from_name(STEM)
        assert (identity.participant, identity.session) == ("sub-a", "ses-b")
        assert identity.task_id == "respiration-and-cough-fivebreaths-2"
        assert identity.task_family == "respiration-and-cough-fivebreaths"
        assert set(identity.sources.values()) == {FROM_STEM}

    def test_each_entity_is_read_on_its_own(self) -> None:
        """``sub-`` and ``ses-`` are read wherever they sit; a lone participant keeps its folder and group."""
        cases = {
            STEM: ("sub-a", "ses-b"),
            "sub-a_acq-x_ses-b_task-loudness": ("sub-a", "ses-b"),
            "sub-a_task-rainbow-passage": ("sub-a", None),
            "ses-b_task-x": (None, "ses-b"),
            "sub-_ses-_task-x": (None, None),
            "plain-run-id": (None, None),
            "": (None, None),
        }
        for stem, expected in cases.items():
            identity = identity_from_name(stem)
            assert (identity.participant, identity.session) == expected
            assert identity.task_id == task_id_of(stem)
            assert identity.task_family == task_family(task_id_of(stem))
        assert identity_from_name("sub-a_task-x").session_key is None

    def test_a_stem_with_no_task_entity_declares_none(self) -> None:
        """The task id is the parsers' own placeholder."""
        assert identity_from_name("plain-run-id").task_id == UNKNOWN_TASK


class TestFromTheHint:
    """The hint's declaration wins field by field; whatever it leaves out comes from the stem."""

    def test_a_full_declaration_replaces_the_stem(self) -> None:
        """Participant, session and task come from the hint, and the token is lower-cased."""
        hint = AudioHints(metadata={PARTICIPANT_KEY: "P7", SESSION_KEY: "visit-2", TASK_TOKEN_KEY: " Cough-3 "})
        identity = identity_of(hint)
        assert (identity.participant, identity.session, identity.task_id) == ("P7", "visit-2", "cough-3")
        assert identity.task_family == "cough"
        assert identity.session_key == "P7/visit-2"
        assert set(identity.sources.values()) == {FROM_HINT}

    def test_a_partial_declaration_declares_only_what_it_names(self) -> None:
        """A hint naming only the task leaves participant and session undeclared; no file name fills them in."""
        identity = identity_of(AudioHints(metadata={TASK_TOKEN_KEY: "loudness"}))
        assert (identity.participant, identity.session, identity.task_id) == (None, None, "loudness")
        assert identity.sources == {"task_id": FROM_HINT}

    @pytest.mark.parametrize("label", ["a/b", "..", ".", "a\\b"])
    def test_a_label_that_cannot_name_a_folder_is_refused(self, label: str) -> None:
        """A declared participant or session becomes a folder and a file name; a slash or dot path is refused."""
        with pytest.raises(IdentityLabelError):
            identity_of(AudioHints(metadata={PARTICIPANT_KEY: label}))
        with pytest.raises(IdentityLabelError):
            identity_of(AudioHints(metadata={SESSION_KEY: label}))

    def test_a_blank_declaration_is_no_declaration(self) -> None:
        """An empty or whitespace value declares nothing, and nothing is read off a file name (plan step 6a)."""
        identity = identity_of(AudioHints(metadata={TASK_TOKEN_KEY: "  ", PARTICIPANT_KEY: ""}))
        assert identity == RecordingIdentity(participant=None, session=None, task_id=UNKNOWN_TASK, sources={})

    def test_a_hint_says_its_values_were_read_off_the_file_name(self) -> None:
        """A dataset's hint builder that parsed the name may say so; ADMIT records that as the source."""
        hint = AudioHints(metadata={PARTICIPANT_KEY: "sub-a", TASK_TOKEN_KEY: "cough-2", IDENTITY_FROM_KEY: FROM_STEM})
        identity = identity_of(hint)
        assert identity.sources == {"participant": FROM_STEM, "task_id": FROM_STEM}
        with pytest.raises(ValueError, match=IDENTITY_FROM_KEY):
            identity_of(AudioHints(metadata={TASK_TOKEN_KEY: "x", IDENTITY_FROM_KEY: "guess"}))


class TestRecordedOnTheRun:
    """What ADMIT recorded is read back; a store from before it recorded anything has no identity."""

    def test_a_store_without_a_declaration_has_no_identity(self) -> None:
        """The stem is not read: a store from before ADMIT recorded identity records none."""
        store = ProvStore(run_id="old")
        _recording(store, f"/data/{STEM}.wav")
        assert recorded_identity(store) is None
        assert declared_session_key(store) is None

    def test_a_refused_recording_keeps_the_identity_on_admits_activity(self) -> None:
        """ADMIT writes no stream for a file it refuses; its activity still records the identity."""
        store = ProvStore(run_id="refused")
        declared = identity_from_name(STEM)
        store.activity(
            node=ADMIT_NODE, step=None, parameters={"audio_file": f"/data/{STEM}.wav", **declared.as_attributes()}
        )
        assert recorded_identity(store) == declared
        assert declared_session_key(store) == "sub-a/ses-b"

    def test_a_recorded_declaration_is_read_back(self) -> None:
        """The attributes ADMIT wrote come back as the identity."""
        store = ProvStore(run_id="new")
        declared = identity_of(AudioHints(metadata={PARTICIPANT_KEY: "P7", SESSION_KEY: "s1", TASK_TOKEN_KEY: "a-1"}))
        _recording(store, "/data/x.wav", **declared.as_attributes())
        assert recorded_identity(store) == declared
        assert declared_session_key(store) == "P7/s1"

    def test_no_recording_stream_is_no_identity(self) -> None:
        """Without ADMIT's stream there is nothing to read."""
        assert recorded_identity(ProvStore(run_id="empty")) is None


class TestDeclaredTaskId:
    """The one task id every stage reads."""

    def test_without_a_recorded_declaration_nothing_is_read_off_the_file_name(self) -> None:
        """A store that recorded no identity names no task, however its file is named: a replay records one."""
        store = ProvStore(run_id="c")
        _recording(store, f"/data/{STEM}.wav")
        assert declared_task_id(store) == UNKNOWN_TASK

    def test_a_task_recorded_from_a_hint_wins_over_the_stem(self) -> None:
        """A reader without the hint still sees what it declared."""
        store = ProvStore(run_id="c")
        declared = identity_of(AudioHints(metadata={TASK_TOKEN_KEY: "loudness"}))
        _recording(store, f"/data/{STEM}.wav", **declared.as_attributes())
        assert declared_task_id(store) == "loudness"

    def test_a_hint_that_declares_an_unknown_task_is_not_overruled_by_the_stem(self) -> None:
        """The hint is the declaration: a task it names as unknown stays unknown."""
        store = ProvStore(run_id="c")
        declared = identity_of(AudioHints(metadata={TASK_TOKEN_KEY: "unknown"}))
        _recording(store, f"/data/{STEM}.wav", **declared.as_attributes())
        assert declared_task_id(store) == UNKNOWN_TASK

    def test_a_task_recorded_from_the_stem_is_the_stem(self) -> None:
        """A stem-read declaration gives what the stem gives."""
        store = ProvStore(run_id="c")
        _recording(store, f"/data/{STEM}.wav", **identity_from_name(STEM).as_attributes())
        assert declared_task_id(store) == "respiration-and-cough-fivebreaths-2"
        assert DECLARED_TASK_ID in store.entities("stream")[0].attributes

    def test_a_task_read_off_the_given_stem_is_kept_when_the_stored_path_differs(self) -> None:
        """ADMIT stores the resolved path; the task it recorded came from the name it was given."""
        store = ProvStore(run_id="c")
        _recording(store, "/exports/platform-0042.wav", **identity_from_name(STEM).as_attributes())
        assert declared_task_id(store) == "respiration-and-cough-fivebreaths-2"

    def test_a_refused_recording_reads_what_the_admit_activity_recorded(self) -> None:
        """ADMIT wrote no stream; its activity carries the file and what the hint declared."""
        store = ProvStore(run_id="c")
        declared = identity_of(AudioHints(metadata={TASK_TOKEN_KEY: "cough-2"}))
        store.activity(
            node="ADMIT", step=None, parameters={"audio_file": "/data/recording-0042.wav", **declared.as_attributes()}
        )
        assert declared_task_id(store) == "cough-2"

    def test_a_refused_recording_without_a_declaration_names_no_task(self) -> None:
        """A store from before ADMIT recorded declarations on its activity: no task, whatever the file's name."""
        store = ProvStore(run_id="c")
        store.activity(node="ADMIT", step=None, parameters={"audio_file": f"/data/{STEM}.wav"})
        assert declared_task_id(store) == UNKNOWN_TASK

    def test_no_admit_record_is_unknown(self) -> None:
        """Neither a stream nor an activity: nothing declares a task."""
        assert declared_task_id(ProvStore(run_id="empty")) == UNKNOWN_TASK


class TestCorpusRows:
    """Corpus passes group by the identity columns a row carries, never by its stem."""

    def test_a_row_round_trips_through_its_columns(self) -> None:
        """The columns written from an identity read back as that identity."""
        identity = identity_of(
            AudioHints(metadata={PARTICIPANT_KEY: "sub-a", SESSION_KEY: "ses-b", TASK_TOKEN_KEY: "loudness"})
        )
        row = {"stem": STEM, **identity_columns(identity)}
        assert row_identity(row) == identity
        assert row_groups(row) == ("sub-a", "sub-a/ses-b")

    def test_a_run_that_declared_nothing_still_records_an_identity(self) -> None:
        """No hint: the row carries the unknown task, so it is told apart from a run that records no identity."""
        row = {"stem": "recording-0042", **identity_columns(identity_of(None))}
        assert row_identity(row) == identity_of(None)

    def test_a_row_whose_run_records_no_identity_is_its_own_group(self) -> None:
        """Every column null: no identity, and the stem keys both groups."""
        row = {"stem": STEM, **identity_columns(None)}
        assert row_identity(row) is None
        assert row_groups(row) == (STEM, STEM)

    def test_a_participant_without_a_session_groups_by_participant(self) -> None:
        """The session group falls back to the stem; the participant group does not."""
        row = {"stem": "sub-a_task-x", **identity_columns(identity_from_name("sub-a_task-x"))}
        assert row_groups(row) == ("sub-a", "sub-a_task-x")

    def test_a_row_without_identity_columns_is_refused(self) -> None:
        """A manifest from before rows carried identity is rebuilt, not parsed."""
        with pytest.raises(ManifestWithoutIdentity):
            row_identity({"stem": STEM, "enhanced": "/x"})

    def test_a_row_disagreeing_with_the_record_is_named(self) -> None:
        """Each differing field is listed; a missing identity on either side is a disagreement."""
        recorded = identity_from_name(STEM)
        moved = identity_of(
            AudioHints(
                metadata={
                    PARTICIPANT_KEY: "sub-a",
                    SESSION_KEY: "ses-c",
                    TASK_TOKEN_KEY: "respiration-and-cough-fivebreaths-2",
                    IDENTITY_FROM_KEY: FROM_STEM,
                }
            )
        )
        assert identity_disagreement(recorded, recorded) == []
        assert identity_disagreement(moved, recorded) == ["session"]
        assert identity_disagreement(None, recorded) == ["recorded identity"]
        assert identity_disagreement(recorded, None) == ["recorded identity"]


class TestTargetSpeakerAndRecordingId:
    """Plan step 9: the target speaker and the recording id, recorded only where they are not the defaults."""

    def test_the_default_target_is_the_participant_and_nothing_is_recorded(self) -> None:
        """B2AI's runs declare no target: its stores gain no attribute."""
        identity = identity_of(AudioHints(metadata={PARTICIPANT_KEY: "sub-07"}))
        assert identity.target_speaker is None and identity.speaker == "sub-07"
        assert DECLARED_TARGET_SPEAKER not in identity.as_attributes()

    def test_a_declared_target_is_recorded_and_read_back(self) -> None:
        """A caregiver-led recording names the child as the voice the task asks for."""
        hint = AudioHints(metadata={PARTICIPANT_KEY: "sub-07"}, target_speakers=[TargetSpeaker(id="sub-07:child")])
        identity = identity_of(hint)
        assert identity.speaker == "sub-07:child"
        assert identity_from_attributes(identity.as_attributes()) == identity

    def test_more_than_one_target_is_refused(self) -> None:
        """Multi-speaker interactions are a later item; a second target is refused, not misread."""
        hint = AudioHints(target_speakers=[TargetSpeaker(id="a"), TargetSpeaker(id="b")])
        with pytest.raises(MultipleTargetSpeakers, match="not supported yet"):
            identity_of(hint)

    def test_the_recording_id_is_declared_else_the_corpus_path_else_the_stem(self) -> None:
        """Only an id other than the stem is recorded, so a BIDS-named corpus's stores are unchanged."""
        declared = identity_of(AudioHints(metadata={RECORDING_ID_KEY: "r-1"}), stem="recording")
        assert declared.id_of("recording") == "r-1" and declared.as_attributes()[DECLARED_RECORDING_ID] == "r-1"
        flat = identity_of(None, stem="recording", corpus_path="p01/s2/recording.wav")
        assert flat.id_of("recording") == "p01/s2/recording.wav"
        bids = identity_of(AudioHints(metadata={RECORDING_ID_KEY: "sub-a_task-x"}), stem="sub-a_task-x")
        assert bids.recording_id is None and DECLARED_RECORDING_ID not in bids.as_attributes()
        assert bids.id_of("sub-a_task-x") == "sub-a_task-x"

    def test_a_recorded_id_says_where_it_came_from(self) -> None:
        """Review of step 9: a replay is given no corpus root, so only an id read off the corpus path is kept."""
        declared = identity_of(AudioHints(metadata={RECORDING_ID_KEY: "r-1"}), stem="recording")
        flat = identity_of(None, stem="recording", corpus_path="p01/s2/recording.wav")
        assert declared.sources[RECORDING_ID_FIELD] == FROM_HINT and corpus_path_recorded(declared) is None
        assert flat.sources[RECORDING_ID_FIELD] == FROM_CORPUS_PATH
        assert corpus_path_recorded(identity_from_attributes(flat.as_attributes())) == "p01/s2/recording.wav"
        assert RECORDING_ID_FIELD not in identity_of(None, stem="recording").sources

    def test_a_row_carries_its_target_speaker_and_recording_id(self) -> None:
        """Review of step 9: the table's columns rebuild both; an id equal to the stem is none."""
        row = {"stem": "recording", "participant": "p01", "session": "s2", "task": "x", "task_from": "hint"}
        identity = row_identity({**row, ROW_TARGET_SPEAKER: "p01:child", RECORDING_ID_COLUMN: "p01/s2/recording.wav"})
        assert identity is not None and identity.target_speaker == "p01:child"
        assert identity.id_of("recording") == "p01/s2/recording.wav"
        same = row_identity({**row, RECORDING_ID_COLUMN: "recording"})
        assert same is not None and same.recording_id is None

    def test_a_row_naming_another_target_speaker_disagrees_with_the_record(self) -> None:
        """Review of step 9: a stale manifest is refused rather than enrolled under the wrong voice."""
        hint = AudioHints(metadata={PARTICIPANT_KEY: "p01"}, target_speakers=[TargetSpeaker(id="p01:child")])
        recorded = identity_of(hint)
        assert identity_disagreement(identity_of(AudioHints(metadata={PARTICIPANT_KEY: "p01"})), recorded) == [
            "target_speaker"
        ]
