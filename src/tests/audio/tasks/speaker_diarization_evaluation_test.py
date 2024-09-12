"""Testing functions for speaker diarization evaluations."""

from senselab.audio.tasks.speaker_diarization_evaluation import calculate_diarization_error_rate
from senselab.utils.data_structures.script_line import ScriptLine


def test_diarization_error_rate_non_existent_speaker() -> None:
    """Tests speaker diarization error rate when a non-existent speaker is found.

    This example diarization can be found here:
    https://docs.kolena.com/metrics/diarization-error-rate/#implementation-details
    """
    ground_truth = [
        ScriptLine(start=0, end=10, speaker="A"),
        ScriptLine(start=13, end=21, speaker="B"),
        ScriptLine(start=24, end=32, speaker="A"),
        ScriptLine(start=32, end=40, speaker="C"),
    ]

    inference = [
        ScriptLine(start=2, end=14, speaker="A"),
        ScriptLine(start=14, end=15, speaker="C"),
        ScriptLine(start=15, end=20, speaker="B"),
        ScriptLine(start=23, end=36, speaker="C"),
        ScriptLine(start=36, end=40, speaker="D"),
    ]

    speaker_mapping = {"A": "A", "B": "B", "C": "C"}

    diarization = calculate_diarization_error_rate(inference, ground_truth, return_speaker_mapping=True, detailed=True)
    assert isinstance(diarization, dict)
    assert diarization["false alarm"] == 4
    assert diarization["missed detection"] == 3
    assert diarization["confusion"] == 14
    assert diarization["speaker_mapping"] == speaker_mapping


def test_diarization_error_rate_undetected_speaker() -> None:
    """Tests speaker diarization error rate when a speaker goes undetected.

    This example diarization can be found here:
    https://docs.kolena.com/metrics/diarization-error-rate/#example
    """
    ground_truth = [
        ScriptLine(start=0, end=5, speaker="C"),
        ScriptLine(start=5, end=9, speaker="D"),
        ScriptLine(start=10, end=14, speaker="A"),
        ScriptLine(start=14, end=15, speaker="D"),
        ScriptLine(start=17, end=20, speaker="C"),
        ScriptLine(start=22, end=25, speaker="B"),
    ]

    inference = [
        ScriptLine(start=0, end=8, speaker="C"),
        ScriptLine(start=11, end=15, speaker="A"),
        ScriptLine(start=17, end=21, speaker="C"),
        ScriptLine(start=23, end=25, speaker="B"),
    ]

    diarization = calculate_diarization_error_rate(inference, ground_truth, greedy=True)
    assert diarization["diarization error rate"] == 0.4
