"""Tests for the hint a run records: its canonical form, its field digests, and reading them back."""

from __future__ import annotations

import numpy as np

from senselab.audio.data_structures import AudioHints, ExpectedSpeech
from senselab.audio.data_structures.audio_hints import SpeakerEmbeddingProvenance, TargetSpeakerEmbedding
from senselab.audio.workflows.triage.hint_record import (
    HINT,
    HINT_DIGEST,
    HINT_FIELD_DIGESTS,
    NO_HINT,
    changed_fields,
    expected_speech_digest,
    field_digests,
    hint_attributes,
    hint_digest,
    recorded_hint,
    recorded_hint_digest,
)
from senselab.utils.prov_store import ProvStore

HINT_A = AudioHints(
    metadata={"task_token": "rainbow-passage", "language": "en"},
    expected_speech=[ExpectedSpeech(text="When the sunlight strikes raindrops in the air.")],
    instructions="Read the passage aloud.",
)


def _store(**attributes: object) -> ProvStore:
    store = ProvStore(run_id="h")
    store.entity(
        prov_type="stream", extent=(0.0, 1.0), attributes={"name": "recording", "path": "/d/x.wav", **attributes}
    )
    return store


def _speaker(vector: list[float]) -> TargetSpeakerEmbedding:
    return TargetSpeakerEmbedding(vector=vector, provenance=SpeakerEmbeddingProvenance(model_id="m"))


class TestFieldDigests:
    """One digest per field, every field whether triage reads it or not."""

    def test_every_set_field_and_every_metadata_key_has_a_digest(self) -> None:
        """``metadata`` is digested key by key; fields left at their defaults are not digested."""
        assert set(field_digests(HINT_A)) == {
            "expected_speech",
            "instructions",
            "metadata.task_token",
            "metadata.language",
        }

    def test_a_field_at_its_default_is_the_same_as_an_unset_one(self) -> None:
        """So a field added to the hint later, at its default, changes no recorded digest."""
        explicit = HINT_A.model_copy(update={"may_contain": [], "environment": None})
        assert hint_digest(explicit) == hint_digest(HINT_A)
        assert field_digests(explicit) == field_digests(HINT_A)

    def test_a_metadata_value_json_cannot_hold_is_written_as_a_string(self) -> None:
        """A caller's numpy scalar does not make ADMIT fail."""
        hint = AudioHints(metadata={"snr": np.float32(12.5)})
        assert hint_attributes(hint)[HINT] == {"metadata": {"snr": "12.5"}}
        assert "metadata.snr" in field_digests(hint)

    def test_the_prompt_digest_is_the_expected_speech_field_digest(self) -> None:
        """The stimulus alignment records the same digest the hint record holds for its prompts."""
        assert expected_speech_digest(HINT_A.expected_speech) == field_digests(HINT_A)["expected_speech"]

    def test_a_changed_field_changes_only_its_own_digest(self) -> None:
        """The other fields' digests stay, which is what lets a later pass name the change."""
        moved = HINT_A.model_copy(update={"instructions": "Read it twice."})
        before, after = field_digests(HINT_A), field_digests(moved)
        assert [name for name in before if before[name] != after[name]] == ["instructions"]

    def test_the_digests_do_not_depend_on_metadata_key_order(self) -> None:
        """Canonical JSON: the same hint built in another order is the same hint."""
        reordered = AudioHints(
            metadata={"language": "en", "task_token": "rainbow-passage"},
            expected_speech=HINT_A.expected_speech,
            instructions=HINT_A.instructions,
        )
        assert hint_digest(reordered) == hint_digest(HINT_A)
        assert field_digests(reordered) == field_digests(HINT_A)


class TestHintAttributes:
    """What ADMIT writes onto the recording stream."""

    def test_no_hint_is_recorded_as_none(self) -> None:
        """Absent, explicitly: not a missing key."""
        assert hint_attributes(None) == {HINT: None, HINT_DIGEST: NO_HINT, HINT_FIELD_DIGESTS: {}}

    def test_the_hint_is_kept_in_canonical_json(self) -> None:
        """The stored copy is the hint as JSON, and its digest is the whole hint's."""
        attributes = hint_attributes(HINT_A)
        assert attributes[HINT] == HINT_A.model_dump(mode="json", exclude_defaults=True)
        assert attributes[HINT_DIGEST] == hint_digest(HINT_A)
        assert attributes[HINT_FIELD_DIGESTS] == field_digests(HINT_A)

    def test_a_target_speaker_vector_is_not_stored_but_still_digested(self) -> None:
        """A speaker embedding is a biometric: the store keeps its size and digest, the digest its value."""
        hint = AudioHints(target_speaker=_speaker([0.1, 0.2, 0.3]))
        stored = hint_attributes(hint)[HINT]["target_speaker"]
        assert "vector" not in stored
        assert stored["vector_dim"] == 3
        other = AudioHints(target_speaker=_speaker([0.1, 0.2, 0.4]))
        assert field_digests(hint)["target_speaker"] != field_digests(other)["target_speaker"]


class TestReadingTheRecordBack:
    """A later pass reads the record and compares the hint it holds."""

    def test_the_recorded_hint_round_trips(self) -> None:
        """What was recorded is what was given."""
        store = _store(**hint_attributes(HINT_A))
        assert recorded_hint(store) == HINT_A
        assert recorded_hint_digest(store) == hint_digest(HINT_A)

    def test_the_same_hint_has_no_changed_fields(self) -> None:
        """Nothing moved."""
        assert changed_fields(_store(**hint_attributes(HINT_A)), HINT_A) == []

    def test_a_changed_task_token_is_named(self) -> None:
        """The change is reported by field, ``metadata`` by key."""
        moved = HINT_A.model_copy(update={"metadata": {**HINT_A.metadata, "task_token": "caterpillar-passage"}})
        assert changed_fields(_store(**hint_attributes(HINT_A)), moved) == ["metadata.task_token"]

    def test_dropping_the_hint_changes_every_recorded_field(self) -> None:
        """A run given a hint and a pass holding none differ in every field the hint had."""
        assert changed_fields(_store(**hint_attributes(HINT_A)), None) == sorted(field_digests(HINT_A))

    def test_a_store_that_predates_the_record_is_unknown(self) -> None:
        """No record is None, not "no hint" and not "unchanged"."""
        store = _store()
        assert recorded_hint_digest(store) is None
        assert recorded_hint(store) is None
        assert changed_fields(store, HINT_A) is None

    def test_a_store_without_a_recording_stream_is_unknown(self) -> None:
        """ADMIT refused the recording, so there is nothing to read."""
        store = ProvStore(run_id="empty")
        assert recorded_hint_digest(store) is None
        assert changed_fields(store, None) is None
