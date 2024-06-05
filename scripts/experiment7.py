"""This script is used to test the senselab classes."""

from senselab.utils.data_structures.dataset import Participant, SenselabDataset, Session

# Example usage
participant1 = Participant(metadata={"name": "John Doe", "age": 30})
session1 = Session(metadata={"session_name": "Baseline", "date": "2024-06-01"})

print(participant1)
print(session1)

# Creating another participant with a specific ID
participant2 = Participant(id="custom-id-123", metadata={"name": "Jane Smith", "age": 25})
print(participant2)

# Creating a session with default ID
session2 = Session()
print(session2)


# Example usage
dataset = SenselabDataset()

try:
    participant1 = Participant(metadata={"name": "John Doe", "age": 30})
    dataset.add_participant(participant1)

    participant2 = Participant(metadata={"name": "Jane Smith", "age": 25})
    dataset.add_participant(participant2)

    # Creating another participant with a specific ID
    participant3 = Participant(id="123", metadata={"name": "Alice"})
    dataset.add_participant(participant3)

    # Attempting to create another participant with the same ID should raise an error
    participant4 = Participant(id="123", metadata={"name": "Bob"})
    dataset.add_participant(participant4)
except ValueError as e:
    print("Value error:", e)

try:
    session1 = Session(metadata={"session_name": "Baseline", "date": "2024-06-01"})
    dataset.add_session(session1)

    session2 = Session(metadata={"session_name": "Follow-up", "date": "2024-07-01"})
    dataset.add_session(session2)

    # Attempting to create another session with the same ID should raise an error
    session3 = Session(id="123")
    dataset.add_session(session3)

    session4 = Session(id="123")
    dataset.add_session(session4)
except ValueError as e:
    print("Value error:", e)

# Print all participants and sessions
print(dataset.get_participants())
print(dataset.get_sessions())
