"""Module for testing bioacoustic quality control."""

from collections import Counter
from typing import Dict, List

import pytest

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.bioacoustic_qc import audios_to_task_dict, task_to_taxonomy_tree_path
from senselab.audio.tasks.bioacoustic_qc.constants import BIOACOUSTIC_TASK_TREE


def test_audios_to_task_dict(
    mono_audio_sample: Audio,
    stereo_audio_sample: Audio,
    resampled_mono_audio_sample: Audio,
    resampled_stereo_audio_sample: Audio,
) -> None:
    """Tests the function that assigns Audio objects to task categories."""
    # Assign task metadata
    mono_audio_sample.metadata["task"] = "breathing"
    stereo_audio_sample.metadata["task"] = "cough"
    resampled_mono_audio_sample.metadata["task"] = "speech"

    audios: List[Audio] = [mono_audio_sample, stereo_audio_sample, resampled_mono_audio_sample]

    task_dict: Dict[str, List[Audio]] = audios_to_task_dict(audios)
    expected_keys = {"breathing", "cough", "speech"}

    # Ensure the function returns the expected structure
    assert set(task_dict.keys()) == expected_keys, f"Unexpected task keys: {task_dict.keys()}"

    # Ensure each task has at least one Audio object
    for task, audio_list in task_dict.items():
        assert isinstance(audio_list, list), f"Expected list for task {task}, got {type(audio_list)}"
        assert len(audio_list) > 0, f"Expected at least one audio for task {task}"

    # Test case where an audio has no task metadata (should default to "bioacoustic")
    resampled_stereo_audio_sample.metadata = {}  # Remove task metadata
    task_dict = audios_to_task_dict([resampled_stereo_audio_sample])

    assert "bioacoustic" in task_dict, "Audio without task metadata should be assigned to 'bioacoustic'"
    assert len(task_dict["bioacoustic"]) == 1, "Expected one audio under 'bioacoustic'"


@pytest.mark.parametrize(
    "taxonomy_tree",
    [BIOACOUSTIC_TASK_TREE],
)
def test_no_duplicate_subclass_keys(taxonomy_tree: Dict) -> None:
    """Tests that all subclass keys in the taxonomy are unique."""

    def get_all_subclass_keys(tree: Dict) -> List[str]:
        """Recursively extract all subclass keys from the taxonomy tree."""
        subclass_keys = []

        def traverse(subtree: Dict) -> None:
            for key, value in subtree.items():
                subclass_keys.append(key)  # Collect every key (task category)
                if isinstance(value, Dict) and "subclass" in value and value["subclass"] is not None:
                    traverse(value["subclass"])  # Continue traversal on non-null subclass

        traverse(tree)
        return subclass_keys

    subclass_keys = get_all_subclass_keys(taxonomy_tree)

    # Ensure there are no duplicate subclass keys
    subclass_counts = Counter(subclass_keys)
    duplicates = {key: count for key, count in subclass_counts.items() if count > 1}

    assert not duplicates, f"Duplicate subclass keys found: {duplicates}"


def test_task_to_taxonomy_tree_path() -> None:
    """Tests that the function correctly retrieves the taxonomy path for a given task."""
    # Test valid task paths
    assert task_to_taxonomy_tree_path("sigh") == [
        "bioacoustic",
        "human",
        "respiration",
        "breathing",
        "sigh",
    ], "Incorrect path for 'sigh'"

    assert task_to_taxonomy_tree_path("cough") == [
        "bioacoustic",
        "human",
        "respiration",
        "exhalation",
        "cough",
    ], "Incorrect path for 'cough'"

    assert task_to_taxonomy_tree_path("diadochokinesis") == [
        "bioacoustic",
        "human",
        "vocalization",
        "speech",
        "repetitive_speech",
        "diadochokinesis",
    ], "Incorrect path for 'diadochokinesis'"

    # Test task not in taxonomy
    with pytest.raises(ValueError, match="Task 'nonexistent_task' not found in taxonomy tree."):
        task_to_taxonomy_tree_path("nonexistent_task")
