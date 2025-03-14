"""Module for testing bioacoustic quality control."""

from collections import Counter
from typing import Dict, List

import pytest
import torch

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.bioacoustic_qc import (
    activity_dict_to_dataset_taxonomy_subtree,
    activity_to_taxonomy_tree_path,
    audios_to_activity_dict,
    check_quality,
    evaluate_node,
    run_taxonomy_subtree_checks_recursively,
)
from senselab.audio.tasks.bioacoustic_qc.checks import audio_intensity_positive_check, audio_length_positive_check
from senselab.audio.tasks.bioacoustic_qc.constants import BIOACOUSTIC_ACTIVITY_TAXONOMY


def test_audios_to_activity_dict(
    mono_audio_sample: Audio,
    stereo_audio_sample: Audio,
    resampled_mono_audio_sample: Audio,
    resampled_stereo_audio_sample: Audio,
) -> None:
    """Tests the function that assigns Audio objects to activity categories."""
    # Assign activity metadata
    mono_audio_sample.metadata["activity"] = "breathing"
    stereo_audio_sample.metadata["activity"] = "cough"
    resampled_mono_audio_sample.metadata["activity"] = "speech"

    audios: List[Audio] = [mono_audio_sample, stereo_audio_sample, resampled_mono_audio_sample]

    activity_dict: Dict[str, List[Audio]] = audios_to_activity_dict(audios)
    expected_keys = {"breathing", "cough", "speech"}

    # Ensure the function returns the expected structure
    assert set(activity_dict.keys()) == expected_keys, f"Unexpected activity keys: {activity_dict.keys()}"

    # Ensure each activity has at least one Audio object
    for activity, audio_list in activity_dict.items():
        assert isinstance(audio_list, list), f"Expected list for activity {activity}, got {type(audio_list)}"
        assert len(audio_list) > 0, f"Expected at least one audio for activity {activity}"

    # Test case where an audio has no activity metadata (should default to "bioacoustic")
    resampled_stereo_audio_sample.metadata = {}  # Remove activity metadata
    activity_dict = audios_to_activity_dict([resampled_stereo_audio_sample])

    assert "bioacoustic" in activity_dict, "Audio without activity metadata should be assigned to 'bioacoustic'"
    assert len(activity_dict["bioacoustic"]) == 1, "Expected one audio under 'bioacoustic'"


@pytest.mark.parametrize("taxonomy_tree", [BIOACOUSTIC_ACTIVITY_TAXONOMY])
def test_no_duplicate_subclass_keys(taxonomy_tree: Dict) -> None:
    """Tests that all subclass keys in the taxonomy are unique."""

    def get_all_subclass_keys(tree: Dict) -> List[str]:
        """Recursively extract all subclass keys from the taxonomy tree."""
        subclass_keys = []

        def traverse(subtree: Dict) -> None:
            for key, value in subtree.items():
                subclass_keys.append(key)  # Collect every key (activity category)
                if isinstance(value, Dict) and "subclass" in value and value["subclass"] is not None:
                    traverse(value["subclass"])  # Continue traversal on non-null subclass

        traverse(tree)
        return subclass_keys

    subclass_keys = get_all_subclass_keys(taxonomy_tree)

    # Ensure there are no duplicate subclass keys
    subclass_counts = Counter(subclass_keys)
    duplicates = {key: count for key, count in subclass_counts.items() if count > 1}

    assert not duplicates, f"Duplicate subclass keys found: {duplicates}"


def test_activity_to_taxonomy_tree_path() -> None:
    """Tests that the function correctly retrieves the taxonomy path for a given activity."""
    # Test valid activity paths
    assert activity_to_taxonomy_tree_path("sigh") == [
        "bioacoustic",
        "human",
        "respiration",
        "breathing",
        "sigh",
    ], "Incorrect path for 'sigh'"

    assert activity_to_taxonomy_tree_path("cough") == [
        "bioacoustic",
        "human",
        "respiration",
        "exhalation",
        "cough",
    ], "Incorrect path for 'cough'"

    assert activity_to_taxonomy_tree_path("diadochokinesis") == [
        "bioacoustic",
        "human",
        "vocalization",
        "speech",
        "repetitive_speech",
        "diadochokinesis",
    ], "Incorrect path for 'diadochokinesis'"

    # Test activity not in taxonomy
    with pytest.raises(ValueError, match="Activity 'nonexistent_activity' not found in taxonomy tree."):
        activity_to_taxonomy_tree_path("nonexistent_activity")


def test_activity_dict_to_dataset_taxonomy_subtree(mono_audio_sample: Audio) -> None:
    """Tests that the function correctly prunes the taxonomy based on dataset activities."""
    # Case 1: Valid activity in the taxonomy (should return a pruned tree with 'sigh')
    activity_dict = {"sigh": [mono_audio_sample]}
    expected_subtree = {
        "bioacoustic": {
            "checks": [audio_length_positive_check, audio_intensity_positive_check],
            "metrics": [],
            "subclass": {
                "human": {
                    "checks": [],
                    "metrics": [],
                    "subclass": {
                        "respiration": {
                            "checks": [],
                            "metrics": [],
                            "subclass": {
                                "breathing": {
                                    "checks": [],
                                    "metrics": [],
                                    "subclass": {
                                        "sigh": {
                                            "checks": [],
                                            "metrics": [],
                                            "subclass": None,
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
    pruned_tree = activity_dict_to_dataset_taxonomy_subtree(activity_dict, activity_tree=BIOACOUSTIC_ACTIVITY_TAXONOMY)
    assert pruned_tree == expected_subtree, f"Expected {expected_subtree}, but got {pruned_tree}"

    # Case 2: Activity not in the taxonomy (should raise ValueError)
    activity_dict = {"nonexistent_activity": [mono_audio_sample]}
    with pytest.raises(ValueError, match="Activity 'nonexistent_activity' not found in taxonomy tree."):
        activity_dict_to_dataset_taxonomy_subtree(activity_dict, activity_tree=BIOACOUSTIC_ACTIVITY_TAXONOMY)

    # Case 3: Empty activity_dict (should return 'bioacoustic' with empty subclass)
    activity_dict = {}
    expected_empty_tree = {
        "bioacoustic": {
            "checks": [audio_length_positive_check, audio_intensity_positive_check],
            "metrics": [],
            "subclass": None,
        }
    }
    pruned_tree = activity_dict_to_dataset_taxonomy_subtree(activity_dict, activity_tree=BIOACOUSTIC_ACTIVITY_TAXONOMY)
    assert pruned_tree == expected_empty_tree, f"Expected {expected_empty_tree}, but got {pruned_tree}"

    # Case 4: Multiple valid activities ('sigh' and 'cough')
    activity_dict = {"sigh": [mono_audio_sample], "cough": [mono_audio_sample]}
    expected_subtree_multiple = {
        "bioacoustic": {
            "checks": [audio_length_positive_check, audio_intensity_positive_check],
            "metrics": [],
            "subclass": {
                "human": {
                    "checks": [],
                    "metrics": [],
                    "subclass": {
                        "respiration": {
                            "checks": [],
                            "metrics": [],
                            "subclass": {
                                "breathing": {
                                    "checks": [],
                                    "metrics": [],
                                    "subclass": {
                                        "sigh": {
                                            "checks": [],
                                            "metrics": [],
                                            "subclass": None,
                                        }
                                    },
                                },
                                "exhalation": {
                                    "checks": [],
                                    "metrics": [],
                                    "subclass": {
                                        "cough": {
                                            "checks": [],
                                            "metrics": [],
                                            "subclass": None,
                                        }
                                    },
                                },
                            },
                        }
                    },
                }
            },
        }
    }
    pruned_tree = activity_dict_to_dataset_taxonomy_subtree(activity_dict, activity_tree=BIOACOUSTIC_ACTIVITY_TAXONOMY)
    assert pruned_tree == expected_subtree_multiple, f"Expected {expected_subtree_multiple}, but got {pruned_tree}"

    # Case 5: Deeply nested activity ('voluntary cough')
    activity_dict = {"voluntary": [mono_audio_sample]}
    expected_subtree_deep = {
        "bioacoustic": {
            "checks": [audio_length_positive_check, audio_intensity_positive_check],
            "metrics": [],
            "subclass": {
                "human": {
                    "checks": [],
                    "metrics": [],
                    "subclass": {
                        "respiration": {
                            "checks": [],
                            "metrics": [],
                            "subclass": {
                                "exhalation": {
                                    "checks": [],
                                    "metrics": [],
                                    "subclass": {
                                        "cough": {
                                            "checks": [],
                                            "metrics": [],
                                            "subclass": {
                                                "voluntary": {
                                                    "checks": [],
                                                    "metrics": [],
                                                    "subclass": None,
                                                }
                                            },
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
    pruned_tree = activity_dict_to_dataset_taxonomy_subtree(activity_dict, activity_tree=BIOACOUSTIC_ACTIVITY_TAXONOMY)
    assert pruned_tree == expected_subtree_deep, f"Expected {expected_subtree_deep}, but got {pruned_tree}"


def test_evaluate_node(mono_audio_sample: Audio) -> None:
    """Tests that `evaluate_node` correctly applies checks and updates the taxonomy tree."""
    # Create a test tree node with sample checks
    tree = {"checks": [audio_length_positive_check, audio_intensity_positive_check]}

    # Create valid and invalid audio samples
    empty_audio = Audio(waveform=torch.tensor([]), sampling_rate=16000, metadata={})
    silent_audio = Audio(waveform=torch.zeros(1, 16000), sampling_rate=16000, metadata={})

    # List of audio files for testing
    audios = [mono_audio_sample, empty_audio, silent_audio]
    activity_audios = [mono_audio_sample, empty_audio, silent_audio]

    # Run the evaluate_node function
    evaluate_node(audios=audios, activity_audios=activity_audios, tree=tree)

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
            "metrics": [],
            "subclass": {
                "human": {
                    "checks": [],
                    "metrics": [],
                    "subclass": {
                        "respiration": {
                            "checks": [],
                            "metrics": [],
                            "subclass": {
                                "breathing": {
                                    "checks": [],
                                    "metrics": [],
                                    "subclass": {
                                        "sigh": {
                                            "checks": [],
                                            "metrics": [],
                                            "subclass": None,
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

    # Create valid and invalid audio samples
    empty_audio = Audio(waveform=torch.tensor([]), sampling_rate=16000, metadata={"activity": "sigh"})
    silent_audio = Audio(waveform=torch.zeros(1, 16000), sampling_rate=16000, metadata={"activity": "sigh"})

    # Create activity_dict mapping activities to audios
    activity_dict = {"sigh": [mono_audio_sample, empty_audio, silent_audio]}

    # Run the function
    updated_tree = run_taxonomy_subtree_checks_recursively(
        audios=[mono_audio_sample, empty_audio, silent_audio],
        dataset_tree=test_tree,
        activity_dict=activity_dict,
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
    valid_audio = Audio(waveform=torch.rand(1, 16000), sampling_rate=16000, metadata={"activity": "breathing"})
    empty_audio = Audio(waveform=torch.tensor([]), sampling_rate=16000, metadata={"activity": "breathing"})
    silent_audio = Audio(waveform=torch.zeros(1, 16000), sampling_rate=16000, metadata={"activity": "breathing"})

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
