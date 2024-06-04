"""Data structures relevant for managing datasets."""

import uuid
from typing import Any, Dict, List

from pydantic import BaseModel, Field, field_validator


class Participant(BaseModel):
    """Data structure for a participant in a dataset."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    metadata: Dict = Field(default={})

    @field_validator("id", mode="before")
    def set_id(cls, v: str) -> str:
        """Set the unique id of the participant."""
        return v or str(uuid.uuid4())

    def __eq__(self, other: object) -> bool:
        """Overloads the default BaseModel equality to correctly check that ids are equivalent."""
        if isinstance(other, Participant):
            return self.id == other.id
        return False


class Session(BaseModel):
    """Data structure for a session in a dataset."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    metadata: Dict = Field(default={})

    @field_validator("id", mode="before")
    def set_id(cls, v: str) -> str:
        """Set the unique id of the session."""
        return v or str(uuid.uuid4())

    def __eq__(self, other: object) -> bool:
        """Overloads the default BaseModel equality to correctly check that ids are equivalent."""
        if isinstance(other, Session):
            return self.id == other.id
        return False


class SenselabDataset(BaseModel):
    """Data structure for a Senselab dataset."""

    participants: Dict[str, Participant] = Field(default_factory=dict)
    sessions: Dict[str, Session] = Field(default_factory=dict)

    @field_validator("participants", mode="before")
    def check_unique_participant_id(cls, v: Dict[str, Participant], values: Any) -> Dict[str, Participant]:  # noqa: ANN401
        """Check if participant IDs are unique."""
        print("type(values)")
        print(type(values))
        input("Press Enter to continue...")
        participants = values.get("participants", {})
        for participant_id, _ in v.items():
            if participant_id in participants:
                raise ValueError(f"Participant with ID {participant_id} already exists.")
        return v

    @field_validator("sessions", mode="before")
    def check_unique_session_id(cls, v: Dict[str, Session], values: Any) -> Dict[str, Session]:  # noqa: ANN401
        """Check if session IDs are unique."""
        sessions = values.get("sessions", {})
        for session_id, _ in v.items():
            if session_id in sessions:
                raise ValueError(f"Session with ID {session_id} already exists.")
        return v

    def add_participant(self, participant: Participant) -> None:
        """Add a participant to the dataset."""
        if participant.id in self.participants:
            raise ValueError(f"Participant with ID {participant.id} already exists.")
        self.participants[participant.id] = participant

    def add_session(self, session: Session) -> None:
        """Add a session to the dataset."""
        if session.id in self.sessions:
            raise ValueError(f"Session with ID {session.id} already exists.")
        self.sessions[session.id] = session

    def get_participants(self) -> List[Participant]:
        """Get the list of participants in the dataset."""
        return list(self.participants.values())

    def get_sessions(self) -> List[Session]:
        """Get the list of sessions in the dataset."""
        return list(self.sessions.values())
