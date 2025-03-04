"""Module for testing bioacoustic quality control."""

from collections import Counter
from typing import Dict, List

import pytest
import torch

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.bioacoustic_qc import (
    audios_to_task_dict,
    check_node,
    check_quality,
    run_taxonomy_subtree_checks_recursively,
    task_dict_to_dataset_taxonomy_subtree,
    task_to_taxonomy_tree_path,
)
from senselab.audio.tasks.bioacoustic_qc.checks import audio_intensity_positive_check, audio_length_positive_check
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


def test_task_dict_to_dataset_taxonomy_subtree(mono_audio_sample: Audio) -> None:
    """Tests that the function correctly prunes the taxonomy based on dataset tasks."""
    # Case 1: Valid task in the taxonomy (should return a pruned tree with 'sigh')
    task_dict = {"sigh": [mono_audio_sample]}
    expected_subtree = {
        "bioacoustic": {
            "checks": [audio_length_positive_check, audio_intensity_positive_check],
            "subclass": {
                "human": {
                    "checks": None,
                    "subclass": {
                        "respiration": {
                            "checks": None,
                            "subclass": {
                                "breathing": {"checks": None, "subclass": {"sigh": {"checks": None, "subclass": None}}}
                            },
                        }
                    },
                }
            },
        }
    }
    pruned_tree = task_dict_to_dataset_taxonomy_subtree(task_dict, task_tree=BIOACOUSTIC_TASK_TREE)
    assert pruned_tree == expected_subtree, f"Expected {expected_subtree}, but got {pruned_tree}"

    # Case 2: Task not in the taxonomy (should raise ValueError)
    task_dict = {"nonexistent_task": [mono_audio_sample]}
    with pytest.raises(ValueError, match="Task 'nonexistent_task' not found in taxonomy tree."):
        task_dict_to_dataset_taxonomy_subtree(task_dict, task_tree=BIOACOUSTIC_TASK_TREE)

    # Case 3: Empty task_dict (should return 'bioacoustic' with empty subclass)
    task_dict = {}
    expected_empty_tree = {
        "bioacoustic": {"checks": [audio_length_positive_check, audio_intensity_positive_check], "subclass": None}
    }
    pruned_tree = task_dict_to_dataset_taxonomy_subtree(task_dict, task_tree=BIOACOUSTIC_TASK_TREE)
    assert pruned_tree == expected_empty_tree, f"Expected {expected_empty_tree}, but got {pruned_tree}"

    # Case 4: Multiple valid tasks ('sigh' and 'cough')
    task_dict = {"sigh": [mono_audio_sample], "cough": [mono_audio_sample]}
    expected_subtree_multiple = {
        "bioacoustic": {
            "checks": [audio_length_positive_check, audio_intensity_positive_check],
            "subclass": {
                "human": {
                    "checks": None,
                    "subclass": {
                        "respiration": {
                            "checks": None,
                            "subclass": {
                                "breathing": {"checks": None, "subclass": {"sigh": {"checks": None, "subclass": None}}},
                                "exhalation": {
                                    "checks": None,
                                    "subclass": {"cough": {"checks": None, "subclass": None}},
                                },
                            },
                        }
                    },
                }
            },
        }
    }
    pruned_tree = task_dict_to_dataset_taxonomy_subtree(task_dict, task_tree=BIOACOUSTIC_TASK_TREE)
    assert pruned_tree == expected_subtree_multiple, f"Expected {expected_subtree_multiple}, but got {pruned_tree}"

    # Case 5: Deeply nested task ('voluntary cough')
    task_dict = {"voluntary": [mono_audio_sample]}
    expected_subtree_deep = {
        "bioacoustic": {
            "checks": [audio_length_positive_check, audio_intensity_positive_check],
            "subclass": {
                "human": {
                    "checks": None,
                    "subclass": {
                        "respiration": {
                            "checks": None,
                            "subclass": {
                                "exhalation": {
                                    "checks": None,
                                    "subclass": {
                                        "cough": {
                                            "checks": None,
                                            "subclass": {"voluntary": {"checks": None, "subclass": None}},
                                        }
                                    },
                                }
                            },
                        }
                    },
                }
            },
        }
    }
    pruned_tree = task_dict_to_dataset_taxonomy_subtree(task_dict, task_tree=BIOACOUSTIC_TASK_TREE)
    assert pruned_tree == expected_subtree_deep, f"Expected {expected_subtree_deep}, but got {pruned_tree}"


def test_check_node(mono_audio_sample: Audio) -> None:
    """Tests that `check_node` correctly applies checks and updates the taxonomy tree."""
    # Create a test tree node with sample checks
    tree = {"checks": [audio_length_positive_check, audio_intensity_positive_check]}

    # Create valid and invalid audio samples
    empty_audio = Audio(waveform=torch.tensor([]), sampling_rate=16000, metadata={})
    silent_audio = Audio(waveform=torch.zeros(1, 16000), sampling_rate=16000, metadata={})

    # List of audio files for testing
    audios = [mono_audio_sample, empty_audio, silent_audio]
    task_audios = [mono_audio_sample, empty_audio, silent_audio]

    # Run the check_node function
    check_node(audios=audios, task_audios=task_audios, tree=tree)

    # Verify the check results are stored in the tree
    assert "checks_results" in tree, "Check results should be stored in the tree."
    assert audio_length_positive_check.__name__ in tree["checks_results"], "Audio length check missing."
    assert audio_intensity_positive_check.__name__ in tree["checks_results"], "Audio intensity check missing."

    # Verify excluded audios
    if not isinstance(tree["checks_results"], dict):
        raise TypeError("Expected 'checks_results' to be a dict.")

    length_check_results = tree["checks_results"][audio_length_positive_check.__name__]
    intensity_check_results = tree["checks_results"][audio_intensity_positive_check.__name__]

    assert empty_audio in length_check_results["exclude"], "Empty audio should be excluded for length."
    assert silent_audio in intensity_check_results["exclude"], "Silent audio should be excluded for intensity."

    # Verify passing audio
    assert mono_audio_sample in length_check_results["passed"], "Valid audio should pass length check."
    assert mono_audio_sample in intensity_check_results["passed"], "Valid audio should pass intensity check."


def test_run_taxonomy_subtree_checks_recursively(mono_audio_sample: Audio) -> None:
    """Tests that checks correctly applied."""
    # Create test taxonomy tree with sample checks
    test_tree = {
        "bioacoustic": {
            "checks": [audio_length_positive_check, audio_intensity_positive_check],
            "subclass": {
                "human": {
                    "checks": None,
                    "subclass": {
                        "respiration": {
                            "checks": None,
                            "subclass": {
                                "breathing": {
                                    "checks": None,
                                    "subclass": {"sigh": {"checks": None, "subclass": None}},
                                }
                            },
                        }
                    },
                }
            },
        }
    }

    # Create valid and invalid audio samples
    empty_audio = Audio(waveform=torch.tensor([]), sampling_rate=16000, metadata={"task": "sigh"})
    silent_audio = Audio(waveform=torch.zeros(1, 16000), sampling_rate=16000, metadata={"task": "sigh"})

    # Create task_dict mapping tasks to audios
    task_dict = {"sigh": [mono_audio_sample, empty_audio, silent_audio]}

    # Run the function
    updated_tree = run_taxonomy_subtree_checks_recursively(
        audios=[mono_audio_sample, empty_audio, silent_audio],
        dataset_tree=test_tree,
        task_dict=task_dict,
    )

    # Verify that check results were added **only at the bioacoustic level**
    bioacoustic_node = updated_tree["bioacoustic"]
    assert "checks_results" in bioacoustic_node, "Check results should be stored at the 'bioacoustic' level."

    # Ensure the correct check functions were applied at `bioacoustic`
    assert audio_length_positive_check.__name__ in bioacoustic_node["checks_results"], "Audio length check missing."
    assert (
        audio_intensity_positive_check.__name__ in bioacoustic_node["checks_results"]
    ), "Audio intensity check missing."

    # Ensure all lower levels contain **empty check results (`{}`)** if no checks are defined
    lower_nodes = [
        updated_tree["bioacoustic"]["subclass"]["human"],
        updated_tree["bioacoustic"]["subclass"]["human"]["subclass"]["respiration"],
        updated_tree["bioacoustic"]["subclass"]["human"]["subclass"]["respiration"]["subclass"]["breathing"],
        updated_tree["bioacoustic"]["subclass"]["human"]["subclass"]["respiration"]["subclass"]["breathing"][
            "subclass"
        ]["sigh"],
    ]

    for node in lower_nodes:
        assert "checks_results" in node, f"'checks_results' missing in {node}."
        assert node["checks_results"] == {}, f"Expected empty check results at this level: {node}"

    # Verify excluded audios at the correct level (`bioacoustic`)
    length_check_results = bioacoustic_node["checks_results"][audio_length_positive_check.__name__]
    intensity_check_results = bioacoustic_node["checks_results"][audio_intensity_positive_check.__name__]

    assert empty_audio in length_check_results["exclude"], "Empty audio should be excluded for length."
    assert silent_audio in intensity_check_results["exclude"], "Silent audio should be excluded for intensity."

    # Verify passing audio
    assert mono_audio_sample in length_check_results["passed"], "Valid audio should pass length check."
    assert mono_audio_sample in intensity_check_results["passed"], "Valid audio should pass intensity check."


def test_check_quality() -> None:
    """Tests that `check_quality` correctly applies quality checks and updates the taxonomy tree."""
    # Create valid and invalid audio samples
    valid_audio = Audio(waveform=torch.rand(1, 16000), sampling_rate=16000, metadata={"task": "breathing"})
    empty_audio = Audio(waveform=torch.tensor([]), sampling_rate=16000, metadata={"task": "breathing"})
    silent_audio = Audio(waveform=torch.zeros(1, 16000), sampling_rate=16000, metadata={"task": "breathing"})

    # Run check_quality
    updated_tree, remaining_audios = check_quality([valid_audio, empty_audio, silent_audio])

    # Ensure `checks_results` is stored only at the bioacoustic level
    bioacoustic_node = updated_tree["bioacoustic"]
    assert "checks_results" in bioacoustic_node, "Check results should be stored at the 'bioacoustic' level."

    # Verify the correct checks were applied
    assert audio_length_positive_check.__name__ in bioacoustic_node["checks_results"], "Audio length check missing."
    assert (
        audio_intensity_positive_check.__name__ in bioacoustic_node["checks_results"]
    ), "Audio intensity check missing."

    # Ensure no check results exist at lower levels
    breathing_node = updated_tree["bioacoustic"]["subclass"]["human"]["subclass"]["respiration"]["subclass"][
        "breathing"
    ]
    assert breathing_node["checks_results"] == {}

    # Verify that excluded audios were correctly filtered out
    length_check_results = bioacoustic_node["checks_results"][audio_length_positive_check.__name__]
    intensity_check_results = bioacoustic_node["checks_results"][audio_intensity_positive_check.__name__]

    assert empty_audio in length_check_results["exclude"], "Empty audio should be excluded for length."
    assert silent_audio in intensity_check_results["exclude"], "Silent audio should be excluded for intensity."

    # Verify passing audio remains in `remaining_audios`
    assert valid_audio in remaining_audios, "Valid audio should not be excluded."
    assert empty_audio not in remaining_audios, "Empty audio should be removed from the final list."
    assert silent_audio not in remaining_audios, "Silent audio should be removed from the final list."
