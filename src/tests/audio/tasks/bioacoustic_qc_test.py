"""Module for testing bioacoustic quality control."""

from collections import Counter
from typing import Dict, List

import pandas as pd
import pytest
import torch

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.bioacoustic_qc import (
    subtree_to_evaluations,
    activity_to_dataset_taxonomy_subtree,
    activity_to_taxonomy_tree_path,
    check_quality,
)
from senselab.audio.tasks.bioacoustic_qc.checks import audio_intensity_positive_check, audio_length_positive_check
from senselab.audio.tasks.bioacoustic_qc.constants import BIOACOUSTIC_ACTIVITY_TAXONOMY


@pytest.mark.parametrize("taxonomy_tree", [BIOACOUSTIC_ACTIVITY_TAXONOMY])
def test_no_duplicate_subclass_keys(taxonomy_tree: Dict) -> None:
    """Tests that all subclass keys in the taxonomy are unique."""

    def get_all_subclass_keys(tree: Dict) -> List[str]:
        """Recursively extract all subclass keys from the taxonomy tree."""
        subclass_keys = []

        def traverse(subtree: Dict) -> None:
            for key, value in subtree.items():
                subclass_keys.append(key)  # Collect every key (activity category)
                if isinstance(value, dict) and "subclass" in value and value["subclass"] is not None:
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
    """Tests that the function correctly creates a pruned taxonomy subtree."""
    result = activity_to_dataset_taxonomy_subtree(activity_name, BIOACOUSTIC_ACTIVITY_TAXONOMY)
    assert result == expected_subtree, f"Expected {expected_subtree}, but got {result}"


def test_activity_to_dataset_taxonomy_subtree_errors() -> None:
    """Tests error handling in activity_to_dataset_taxonomy_subtree."""
    with pytest.raises(ValueError, match="Activity 'nonexistent_activity' not found in taxonomy tree."):
        activity_to_dataset_taxonomy_subtree("nonexistent_activity", BIOACOUSTIC_ACTIVITY_TAXONOMY)


def test_subtree_to_evaluations() -> None:
    """Tests that evaluations are correctly extracted from a taxonomy subtree."""
    # Create a test subtree
    subtree = {
        "bioacoustic": {
            "checks": [audio_length_positive_check, audio_intensity_positive_check],
            "metrics": BIOACOUSTIC_ACTIVITY_TAXONOMY["bioacoustic"]["metrics"],
            "subclass": {"human": {"checks": [], "metrics": [], "subclass": None}},
        }
    }

    evaluations = subtree_to_evaluations(subtree)
    expected_evals = BIOACOUSTIC_ACTIVITY_TAXONOMY["bioacoustic"]["metrics"] + [
        audio_length_positive_check,
        audio_intensity_positive_check,
    ]
    assert evaluations == expected_evals, "Incorrect evaluations extracted from subtree"


def test_check_quality(tmp_path) -> None:
    """Tests that check_quality correctly processes audio files and returns results."""
    # Create a test audio file
    audio = Audio(
        waveform=torch.rand(1, 16000),
        sampling_rate=16000,
    )
    audio_path = str(tmp_path / "test.wav")
    audio.save(audio_path)

    # Run check_quality
    results_df = check_quality(
        audio_paths=[audio_path],
        output_dir=tmp_path,
    )

    # Verify results
    assert isinstance(results_df, pd.DataFrame), "Expected DataFrame output"
    assert not results_df.empty, "Expected non-empty results"
    assert "path" in results_df.columns, "Expected 'path' column in results"
