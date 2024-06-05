"""Module for testing the Participant, Session, and SenselabDataset classes."""

import pytest

from senselab.utils.data_structures.dataset import Participant, SenselabDataset, Session


def test_create_participant() -> None:
    """Test creating a participant."""
    participant = Participant(metadata={"name": "John Doe"})
    assert isinstance(participant, Participant)
    assert participant.metadata["name"] == "John Doe"


def test_create_session() -> None:
    """Test creating a session."""
    session = Session(metadata={"description": "Initial session"})
    assert isinstance(session, Session)
    assert session.metadata["description"] == "Initial session"


def test_add_participant() -> None:
    """Test adding a participant to the dataset."""
    dataset = SenselabDataset()
    participant = Participant()
    dataset.add_participant(participant)
    assert participant.id in dataset.participants


def test_add_duplicate_participant() -> None:
    """Test adding a duplicate participant to the dataset."""
    dataset = SenselabDataset()
    participant = Participant()
    dataset.add_participant(participant)
    with pytest.raises(ValueError):
        dataset.add_participant(participant)


def test_add_session() -> None:
    """Test adding a session to the dataset."""
    dataset = SenselabDataset()
    session = Session()
    dataset.add_session(session)
    assert session.id in dataset.sessions


def test_add_duplicate_session() -> None:
    """Test adding a duplicate session to the dataset."""
    dataset = SenselabDataset()
    session = Session()
    dataset.add_session(session)
    with pytest.raises(ValueError):
        dataset.add_session(session)


def test_get_participants() -> None:
    """Test getting the list of participants."""
    dataset = SenselabDataset()
    participant1 = Participant()
    participant2 = Participant()
    dataset.add_participant(participant1)
    dataset.add_participant(participant2)
    participants = dataset.get_participants()
    assert len(participants) == 2
    assert participant1 in participants
    assert participant2 in participants


def test_get_sessions() -> None:
    """Test getting the list of sessions."""
    dataset = SenselabDataset()
    session1 = Session()
    session2 = Session()
    dataset.add_session(session1)
    dataset.add_session(session2)
    sessions = dataset.get_sessions()
    assert len(sessions) == 2
    assert session1 in sessions
    assert session2 in sessions
