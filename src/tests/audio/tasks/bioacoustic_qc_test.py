"""Module for testing bioacoustic quality control."""

from collections import Counter
from pathlib import Path
from typing import Callable, Dict, List, Union, cast

import pandas as pd
import pytest
import torch

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.bioacoustic_qc import (
    activity_to_dataset_taxonomy_subtree,
    activity_to_taxonomy_tree_path,
    check_quality,
    create_activity_to_evaluations,
    evaluate_audio,
    evaluate_batch,
    run_evaluations,
    subtree_to_evaluations,
)
from senselab.audio.tasks.bioacoustic_qc.checks import (
    audio_intensity_positive_check,
    audio_length_positive_check,
)
from senselab.audio.tasks.bioacoustic_qc.constants import BIOACOUSTIC_ACTIVITY_TAXONOMY as TAXONOMY


@pytest.mark.parametrize("taxonomy_tree", [TAXONOMY])
def test_no_duplicate_subclass_keys(taxonomy_tree: Dict) -> None:
    """Tests that all subclass keys in the taxonomy are unique."""

    def get_all_subclass_keys(tree: Dict) -> List[str]:
        """Recursively extract all subclass keys from the taxonomy tree."""
        keys: List[str] = []

        def traverse(subtree: Dict) -> None:
            for key, value in subtree.items():
                # Collect every key (activity category)
                keys.append(key)
                # Continue traversal on non-null subclass
                if isinstance(value, dict) and "subclass" in value and value["subclass"] is not None:
                    traverse(value["subclass"])

        traverse(tree)
        return keys

    # Ensure there are no duplicate subclass keys
    subclass_keys = get_all_subclass_keys(taxonomy_tree)
    subclass_counts = Counter(subclass_keys)
    duplicates = {key: count for key, count in subclass_counts.items() if count > 1}

    assert not duplicates, f"Duplicate subclass keys found: {duplicates}"


def test_activity_to_taxonomy_tree_path() -> None:
    """Tests taxonomy path retrieval for activities."""
    # Test valid activity paths
    assert activity_to_taxonomy_tree_path("sigh") == [
        "bioacoustic",
        "human",
        "respiration",
        "breathing",
        "sigh",
    ], "Incorrect path for 'sigh'"

    # Test activity not in taxonomy
    error_msg = "Activity 'nonexistent_activity' not found in taxonomy tree."
    with pytest.raises(ValueError, match=error_msg):
        activity_to_taxonomy_tree_path("nonexistent_activity")


@pytest.mark.parametrize(
    "activity_name,expected_subtree",
    [
        (
            "sigh",
            {
                "bioacoustic": {
                    "checks": [
                        audio_length_positive_check,
                        audio_intensity_positive_check,
                    ],
                    "metrics": TAXONOMY["bioacoustic"]["metrics"],
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
    ],
)
def test_activity_to_dataset_taxonomy_subtree(
    activity_name: str,
    expected_subtree: Dict,
) -> None:
    """Tests pruned taxonomy subtree creation."""
    result = activity_to_dataset_taxonomy_subtree(
        activity_name,
        TAXONOMY,
    )
    assert result == expected_subtree, f"Expected {expected_subtree}, but got {result}"


def test_activity_to_dataset_taxonomy_subtree_errors() -> None:
    """Tests error handling in taxonomy subtree creation."""
    error_msg = "Activity 'nonexistent_activity' not found in taxonomy tree."
    with pytest.raises(ValueError, match=error_msg):
        activity_to_dataset_taxonomy_subtree(
            "nonexistent_activity",
            TAXONOMY,
        )


def test_subtree_to_evaluations() -> None:
    """Tests evaluation extraction from taxonomy subtree."""
    # Create a test subtree
    subtree = {
        "bioacoustic": {
            "checks": [
                audio_length_positive_check,
                audio_intensity_positive_check,
            ],
            "metrics": TAXONOMY["bioacoustic"]["metrics"],
            "subclass": {
                "human": {
                    "checks": [],
                    "metrics": [],
                    "subclass": None,
                }
            },
        }
    }

    evaluations = subtree_to_evaluations(subtree)
    metrics = cast(list, TAXONOMY["bioacoustic"]["metrics"])
    expected_evals = metrics + [
        audio_length_positive_check,
        audio_intensity_positive_check,
    ]
    assert evaluations == expected_evals, "Incorrect evaluations extracted from subtree"


def test_subtree_to_evaluations_empty() -> None:
    """Tests that empty subtrees return empty evaluation lists."""
    empty_subtree: Dict[str, Dict] = {"node": {"checks": [], "metrics": [], "subclass": None}}
    evaluations = subtree_to_evaluations(empty_subtree)
    assert evaluations == [], "Expected empty list for empty subtree"


def test_subtree_to_evaluations_nested() -> None:
    """Tests that nested subtrees correctly collect all evaluations.

    Verifies that evaluations are collected from all levels of the tree,
    including root and child nodes.
    """

    # Mock evaluation functions
    def mock_metric1(audio: Audio) -> float:
        return 0.0

    def mock_metric2(audio: Audio) -> float:
        return 0.0

    def mock_check1(audio: Audio) -> bool:
        return True

    nested_subtree = {
        "root": {
            "checks": [mock_check1],
            "metrics": [mock_metric1],
            "subclass": {"child": {"checks": [], "metrics": [mock_metric2], "subclass": None}},
        }
    }

    evaluations = subtree_to_evaluations(nested_subtree)
    assert len(evaluations) == 3, "Expected all evaluations from nested structure"
    assert mock_check1 in evaluations, "Missing check from root"
    assert mock_metric1 in evaluations, "Missing metric from root"
    assert mock_metric2 in evaluations, "Missing metric from child"


def test_subtree_to_evaluations_duplicates() -> None:
    """Tests that duplicate evaluations are only included once.

    Verifies that when the same evaluation function appears multiple times
    in the tree, it is only included once in the final list.
    """

    # Mock evaluation function
    def mock_eval(audio: Audio) -> float:
        return 0.0

    subtree_with_duplicates = {
        "node1": {
            "checks": [mock_eval],
            # Same function in checks and metrics
            "metrics": [mock_eval],
            "subclass": {
                "node2": {
                    # Same function in child
                    "checks": [mock_eval],
                    "metrics": [],
                    "subclass": None,
                }
            },
        }
    }

    evaluations = subtree_to_evaluations(subtree_with_duplicates)
    assert len(evaluations) == 1, "Expected duplicates to be removed"
    assert evaluations == [mock_eval], "Expected single instance of duplicate function"


def test_subtree_to_evaluations_order() -> None:
    """Tests that evaluation order is preserved from the taxonomy structure.

    Verifies that evaluations are returned in the same order they appear
    in the tree structure.
    """

    # Mock evaluation functions
    def mock_eval1(audio: Audio) -> float:
        return 0.0

    def mock_eval2(audio: Audio) -> float:
        return 0.0

    def mock_eval3(audio: Audio) -> float:
        return 0.0

    ordered_subtree = {
        "root": {
            "checks": [mock_eval1],
            "metrics": [mock_eval2],
            "subclass": {"child": {"checks": [mock_eval3], "metrics": [], "subclass": None}},
        }
    }

    evaluations = subtree_to_evaluations(ordered_subtree)
    expected = [mock_eval2, mock_eval1, mock_eval3]
    assert evaluations == expected, "Expected order to be preserved"


def test_check_quality(tmp_path: Path) -> None:
    """Tests that check_quality correctly processes audio files and returns results."""
    # Create a test audio file
    audio = Audio(
        waveform=torch.rand(1, 16000),
        sampling_rate=16000,
    )
    audio_path = str(tmp_path / "test.wav")
    audio.save_to_file(audio_path)

    # Run check_quality
    results_df = check_quality(
        audio_paths=[audio_path],
        output_dir=tmp_path,
    )

    # Verify results
    assert isinstance(results_df, pd.DataFrame), "Expected DataFrame output"
    assert not results_df.empty, "Expected non-empty results"
    assert "path" in results_df.columns, "Expected 'path' column in results"


def test_create_activity_to_evaluations() -> None:
    """Tests mapping of audio paths to their evaluation functions."""
    # Setup test data
    audio_paths = {
        "audio1.wav": "sigh",
        "audio2.wav": "sigh",  # Same activity to test deduplication
    }

    # Get evaluations mapping
    activity_evals = create_activity_to_evaluations(
        audio_path_to_activity=audio_paths,
        activity_tree=TAXONOMY,
    )

    # Verify results
    assert len(activity_evals) == 1, "Expected one activity"
    assert "sigh" in activity_evals, "Expected 'sigh' activity"

    # Verify evaluations for sigh
    sigh_evals = activity_evals["sigh"]
    assert isinstance(sigh_evals, list), "Expected list of evaluations"
    assert audio_length_positive_check in sigh_evals, "Missing length check"
    assert audio_intensity_positive_check in sigh_evals, "Missing intensity check"


def test_evaluate_audio(tmp_path: Path) -> None:
    """Tests that evaluate_audio correctly processes audio and handles existing results."""
    # Create test audio file
    audio = Audio(
        waveform=torch.rand(1, 16000),
        sampling_rate=16000,
    )
    audio_path = str(tmp_path / "test.wav")
    audio.save_to_file(audio_path)

    # Create test evaluation functions with proper names
    def test_float(x: Audio) -> float:
        return 0.5

    def test_bool(x: Audio) -> bool:
        return True

    def test_str(x: Audio) -> str:
        return "test"

    evaluations: List[Callable[[Audio], Union[float, bool, str]]] = [
        test_float,
        test_bool,
        test_str,
    ]

    # Test basic evaluation
    results = evaluate_audio(audio_path, "test_activity", evaluations)
    assert results["id"] == Path(audio_path).stem
    assert results["path"] == audio_path
    assert results["activity"] == "test_activity"
    assert results["test_float"] == 0.5
    assert results["test_bool"] is True
    assert results["test_str"] == "test"

    # Test with existing results
    existing = {
        "test_float": 1.0,  # Should be preserved
        "test_str": "old",  # Should be preserved
    }
    results = evaluate_audio(audio_path, "test_activity", evaluations, existing)
    assert results["test_float"] == 1.0, "Existing float result was not preserved"
    assert results["test_str"] == "old", "Existing string result was not preserved"
    assert results["test_bool"] is True, "New evaluation was not computed"


def test_evaluate_batch(tmp_path: Path) -> None:
    """Tests that evaluate_batch correctly processes multiple audio files and handles caching."""
    # Create test audio files
    audio = Audio(
        waveform=torch.rand(1, 16000),
        sampling_rate=16000,
    )

    # Create two test files
    audio_path1 = str(tmp_path / "test1.wav")
    audio_path2 = str(tmp_path / "test2.wav")
    audio.save_to_file(audio_path1)
    audio.save_to_file(audio_path2)

    # Setup test evaluation function
    def test_metric(_: Audio) -> Union[float, bool, str]:
        return 0.5

    # Setup test data
    batch_audio_paths = [audio_path1, audio_path2]
    audio_path_to_activity = {audio_path1: "test_activity", audio_path2: "test_activity"}
    activity_to_evaluations: Dict[
        str,
        List[Callable[[Audio], Union[float, bool, str]]],
    ] = {"test_activity": [test_metric]}

    # Run evaluate_batch
    results = evaluate_batch(
        batch_audio_paths=batch_audio_paths,
        audio_path_to_activity=audio_path_to_activity,
        activity_to_evaluations=activity_to_evaluations,
        output_dir=tmp_path,
    )

    # Verify results
    assert len(results) == 2, "Expected results for both audio files"
    for result in results:
        assert "id" in result, "Expected 'id' in result"
        assert "path" in result, "Expected 'path' in result"
        assert "activity" in result, "Expected 'activity' in result"
        assert "test_metric" in result, "Expected metric result"
        assert result["test_metric"] == 0.5, "Expected metric value of 0.5"

    # Verify caching - files should exist
    results_dir = tmp_path / "audio_results"
    assert results_dir.exists(), "Results directory should be created"
    assert (results_dir / "test1.parquet").exists(), "Cache file for test1 should exist"
    assert (results_dir / "test2.parquet").exists(), "Cache file for test2 should exist"

    # Test with existing results
    cached_results = evaluate_batch(
        batch_audio_paths=batch_audio_paths,
        audio_path_to_activity=audio_path_to_activity,
        activity_to_evaluations=activity_to_evaluations,
        output_dir=tmp_path,
    )

    # Verify cached results match original results
    assert cached_results == results, "Cached results should match original results"


def test_run_evaluations(tmp_path: Path) -> None:
    """Tests that run_evaluations correctly processes batches in parallel."""
    # Create test audio files
    audio = Audio(
        waveform=torch.rand(1, 16000),
        sampling_rate=16000,
    )

    # Create multiple test files to test batching
    audio_paths = []
    for i in range(3):  # Create 3 files to test batching
        path = str(tmp_path / f"test{i}.wav")
        audio.save_to_file(path)
        audio_paths.append(path)

    # Setup test evaluation function
    def test_metric(_: Audio) -> Union[float, bool, str]:
        return 0.5

    # Setup test data
    audio_path_to_activity = {path: "test_activity" for path in audio_paths}
    activity_to_evaluations: Dict[
        str,
        List[Callable[[Audio], Union[float, bool, str]]],
    ] = {"test_activity": [test_metric]}

    # Run evaluations with different configurations
    # Test serial execution
    results_serial = run_evaluations(
        audio_path_to_activity=audio_path_to_activity,
        activity_to_evaluations=activity_to_evaluations,
        output_dir=tmp_path,
        batch_size=2,  # Should create 2 batches
        n_cores=1,  # Force serial execution
        plugin="serial",
    )

    # Test parallel execution
    results_parallel = run_evaluations(
        audio_path_to_activity=audio_path_to_activity,
        activity_to_evaluations=activity_to_evaluations,
        output_dir=tmp_path,
        batch_size=2,  # Should create 2 batches
        n_cores=2,  # Use parallel execution
        plugin="cf",
    )

    # Verify results
    assert isinstance(results_serial, pd.DataFrame), "Expected DataFrame output"
    assert isinstance(results_parallel, pd.DataFrame), "Expected DataFrame output"
    assert len(results_serial) == 3, "Expected results for all files"
    assert len(results_parallel) == 3, "Expected results for all files"

    # Verify both methods give same results
    pd.testing.assert_frame_equal(
        results_serial.sort_values("id").reset_index(drop=True),
        results_parallel.sort_values("id").reset_index(drop=True),
    )

    # Verify results content
    for df in [results_serial, results_parallel]:
        for _, row in df.iterrows():
            assert "id" in row, "Expected 'id' in result"
            assert "path" in row, "Expected 'path' in result"
            assert "activity" in row, "Expected 'activity' in result"
            assert "test_metric" in row, "Expected metric result"
            assert row["test_metric"] == 0.5, "Expected metric value of 0.5"

    # Verify cache files exist
    results_dir = tmp_path / "audio_results"
    assert results_dir.exists(), "Results directory should be created"
    for i in range(3):
        assert (results_dir / f"test{i}.parquet").exists(), f"Cache file for test{i} should exist"
