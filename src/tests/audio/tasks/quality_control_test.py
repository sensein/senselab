"""Module for testing bioacoustic quality control."""

from collections import Counter
from pathlib import Path
from typing import List

import pandas as pd
import pytest

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.quality_control import (
    activity_to_evaluations,
    check_quality,
)
from senselab.audio.tasks.quality_control.taxonomy import TaxonomyNode
from senselab.audio.tasks.quality_control.trees import (
    BIOACOUSTIC_ACTIVITY_TAXONOMY as TAXONOMY,
)


@pytest.mark.parametrize("taxonomy_tree", [TAXONOMY])
def test_no_duplicate_subclass_keys(taxonomy_tree: TaxonomyNode) -> None:
    """Tests that all subclass keys in the taxonomy are unique."""

    def get_all_subclass_keys(node: TaxonomyNode) -> List[str]:
        """Recursively extract all node names from the taxonomy tree."""
        keys: List[str] = []

        def traverse(current_node: TaxonomyNode) -> None:
            # Collect the current node's name
            keys.append(current_node.name)
            # Traverse all children
            for child_node in current_node.children.values():
                traverse(child_node)

        traverse(node)
        return keys

    # Ensure there are no duplicate subclass keys
    subclass_keys = get_all_subclass_keys(taxonomy_tree)
    subclass_counts = Counter(subclass_keys)
    duplicates = {key: count for key, count in subclass_counts.items() if count > 1}

    assert not duplicates, f"Duplicate subclass keys found: {duplicates}"


def test_activity_to_evaluations() -> None:
    """Tests mapping of audio paths to their evaluation functions."""
    # Setup test data
    audio_paths = {
        "audio1.wav": "sigh",
        "audio2.wav": "sigh",  # Same activity to test deduplication
    }

    # Get evaluations mapping
    activity_evals = activity_to_evaluations(
        audio_path_to_activity=audio_paths,
        activity_tree=TAXONOMY,
    )

    # Verify results
    assert len(activity_evals) == 1, "Expected one activity"
    assert "sigh" in activity_evals, "Expected 'sigh' activity"

    # Verify evaluations for sigh
    sigh_evals = activity_evals["sigh"]
    assert isinstance(sigh_evals, list), "Expected list of evaluations"
    # Since we removed dependency on specific check functions,
    # just verify we get a non-empty list of evaluations
    assert len(sigh_evals) > 0, "Expected non-empty evaluation list"


def test_check_quality(tmp_path: Path, resampled_mono_audio_sample: Audio) -> None:
    """Tests that check_quality correctly processes audio files."""
    # Save test audio file
    audio_path = str(tmp_path / "test.wav")
    resampled_mono_audio_sample.save_to_file(audio_path)

    # Run check_quality
    results_df = check_quality(
        audio_paths=[audio_path],
        output_dir=tmp_path,
    )

    # Verify results
    assert isinstance(results_df, pd.DataFrame), "Expected DataFrame output"
    assert not results_df.empty, "Expected non-empty results"
    assert "path" in results_df.columns, "Expected 'path' column in results"
