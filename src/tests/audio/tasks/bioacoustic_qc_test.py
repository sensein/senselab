"""Module for testing bioacoustic quality control."""

from collections import Counter
from typing import Dict, List

import pandas as pd
import pytest
import torch

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.bioacoustic_qc import (
    activity_to_dataset_taxonomy_subtree,
    # activity_to_taxonomy_tree_path,
    # audios_to_activity_dict,
    check_quality,
    # evaluate_node,
    # run_taxonomy_subtree_checks_recursively,
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


@pytest.mark.parametrize(
    "activity_name,expected_subtree",
    [
        (
            "sigh",
            {
                "bioacoustic": {
                    "checks": [audio_length_positive_check, audio_intensity_positive_check],
                    "metrics": BIOACOUSTIC_ACTIVITY_TAXONOMY["bioacoustic"]["metrics"],
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
            },
        ),
        (
            "voluntary",
            {
                "bioacoustic": {
                    "checks": [audio_length_positive_check, audio_intensity_positive_check],
                    "metrics": BIOACOUSTIC_ACTIVITY_TAXONOMY["bioacoustic"]["metrics"],
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
            },
        ),
    ],
)
def test_activity_to_dataset_taxonomy_subtree(activity_name: str, expected_subtree: Dict) -> None:
    result = activity_to_dataset_taxonomy_subtree(activity_name, BIOACOUSTIC_ACTIVITY_TAXONOMY)
    assert result == expected_subtree, f"Expected {expected_subtree}, but got {result}"


def test_activity_to_dataset_taxonomy_subtree_errors() -> None:
    with pytest.raises(ValueError, match="Activity 'nonexistent_activity' not found in taxonomy tree."):
        activity_to_dataset_taxonomy_subtree("nonexistent_activity", BIOACOUSTIC_ACTIVITY_TAXONOMY)


def test_evaluate_node(mono_audio_sample: Audio) -> None:
    """Tests that `evaluate_node` applies checks and updates the node dict with results."""
    tree = {"checks": [audio_length_positive_check, audio_intensity_positive_check]}
    empty_audio = Audio(waveform=torch.tensor([]), sampling_rate=16000, metadata={})
    silent_audio = Audio(waveform=torch.zeros(1, 16000), sampling_rate=16000, metadata={})

    audios = [mono_audio_sample, empty_audio, silent_audio]
    activity_audios = [mono_audio_sample, empty_audio, silent_audio]

    # For the new code, we need a DataFrame to store results
    df = pd.DataFrame({"audio_path_or_id": ["valid", "empty", "silent"]})
    mono_audio_sample.orig_path_or_id = "valid"
    empty_audio.orig_path_or_id = "empty"
    silent_audio.orig_path_or_id = "silent"

    updated_df = evaluate_node(audios, activity_audios, tree, df)

    # The tree itself doesn't hold "checks_results" now; the DataFrame does
    # but let's confirm we have columns for each check
    assert "audio_length_positive_check" in updated_df.columns
    assert "audio_intensity_positive_check" in updated_df.columns

    # Check values
    length_vals = updated_df["audio_length_positive_check"].values
    intensity_vals = updated_df["audio_intensity_positive_check"].values

    # Expected booleans:
    # - valid (waveform rand): True length, True intensity
    # - empty (waveform=[]): False length, False intensity
    # - silent (waveform=all zeros): True length, False intensity
    assert list(length_vals) == [True, False, True], "Unexpected length check booleans"
    assert list(intensity_vals) == [True, False, False], "Unexpected intensity check booleans"


def test_run_taxonomy_subtree_checks_recursively(mono_audio_sample: Audio) -> None:
    """Tests that checks are correctly applied across the taxonomy subtree, storing results in a DataFrame."""
    # Create a minimal taxonomy tree with sample checks
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
                                    "subclass": {"sigh": {"checks": [], "metrics": [], "subclass": None}},
                                }
                            },
                        }
                    },
                }
            },
        }
    }

    # Create valid/invalid audio
    empty_audio = Audio(waveform=torch.tensor([]), sampling_rate=16000, metadata={"activity": "sigh"})
    silent_audio = Audio(waveform=torch.zeros(1, 16000), sampling_rate=16000, metadata={"activity": "sigh"})
    mono_audio_sample.metadata["activity"] = "sigh"

    # Assign IDs to each audio and build a DataFrame for storing check results
    mono_audio_sample.orig_path_or_id = "valid"
    empty_audio.orig_path_or_id = "empty"
    silent_audio.orig_path_or_id = "silent"
    df = pd.DataFrame({"audio_path_or_id": ["valid", "empty", "silent"]})

    # Create the activity_dict
    activity_dict = {"sigh": [mono_audio_sample, empty_audio, silent_audio]}

    # Run the function, now providing `results_df=df`
    results_df = run_taxonomy_subtree_checks_recursively(
        audios=[mono_audio_sample, empty_audio, silent_audio],
        dataset_tree=test_tree,
        activity_dict=activity_dict,
        results_df=df,
    )

    # Verify the DataFrame contains columns for each check
    for col in ["audio_length_positive_check", "audio_intensity_positive_check"]:
        assert col in results_df.columns, f"Missing column {col} in results DataFrame."

    # Check the boolean values
    length_vals = results_df["audio_length_positive_check"].tolist()
    intensity_vals = results_df["audio_intensity_positive_check"].tolist()

    # For [valid, empty, silent]:
    # - valid => length=True, intensity=True
    # - empty => length=False, intensity=False (no samples)
    # - silent => length=True, intensity=False (samples but all zeros)
    assert length_vals == [True, False, True], "Unexpected length check booleans."
    assert intensity_vals == [True, False, False], "Unexpected intensity check booleans."


def test_check_quality() -> None:
    """Tests that `check_quality` produces a DataFrame with correct boolean columns."""
    # Create valid/invalid Audio samples
    valid_audio = Audio(
        waveform=torch.rand(1, 16000),
        sampling_rate=16000,
        metadata={"activity": "breathing"},
    )
    empty_audio = Audio(
        waveform=torch.tensor([]),
        sampling_rate=16000,
        metadata={"activity": "breathing"},
    )
    silent_audio = Audio(
        waveform=torch.zeros(1, 16000),
        sampling_rate=16000,
        metadata={"activity": "breathing"},
    )

    # Assign identifiers for the DataFrame
    valid_audio.orig_path_or_id = "valid"
    empty_audio.orig_path_or_id = "empty"
    silent_audio.orig_path_or_id = "silent"

    # Create the initial DataFrame
    df = pd.DataFrame({"audio_path_or_id": ["valid", "empty", "silent"]})

    # Run check_quality, which returns the updated DataFrame
    results_df = check_quality(audios=[valid_audio, empty_audio, silent_audio], audio_df=df)

    # Ensure columns for the two checks exist
    for col in ["audio_length_positive_check", "audio_intensity_positive_check"]:
        assert col in results_df.columns, f"Expected column {col} not found in results DataFrame."

    # Extract booleans from each column
    length_vals = results_df["audio_length_positive_check"].tolist()
    intensity_vals = results_df["audio_intensity_positive_check"].tolist()

    # For [valid, empty, silent]:
    # valid => length=True, intensity=True
    # empty => length=False, intensity=False (no samples)
    # silent => length=True, intensity=False (samples, but all zeros)
    assert length_vals == [True, False, True], f"Unexpected length results: {length_vals}"
    assert intensity_vals == [True, False, False], f"Unexpected intensity results: {intensity_vals}"
