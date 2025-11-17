"""Module for testing bioacoustic quality control."""

from collections import Counter
from pathlib import Path
from typing import List, Optional

import pandas as pd
import pytest

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.quality_control import (
    activity_to_evaluations,
    check_quality,
)
from senselab.audio.tasks.quality_control.taxonomies import (
    BIOACOUSTIC_ACTIVITY_TAXONOMY,
    BRIDGE2AI_VOICE_TAXONOMY,
)
from senselab.audio.tasks.quality_control.taxonomy import TaxonomyNode


@pytest.mark.parametrize("taxonomy_tree", [BIOACOUSTIC_ACTIVITY_TAXONOMY, BRIDGE2AI_VOICE_TAXONOMY])
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


@pytest.mark.parametrize(
    "taxonomy_tree,activity_name",
    [(BIOACOUSTIC_ACTIVITY_TAXONOMY, "bioacoustic"), (BRIDGE2AI_VOICE_TAXONOMY, "human")],
)
def test_activity_to_evaluations(taxonomy_tree: TaxonomyNode, activity_name: str) -> None:
    """Tests mapping of audio paths to their evaluation functions."""
    # Setup test data - use activities that exist in the taxonomy
    audio_paths = {
        "audio1.wav": activity_name,
        "audio2.wav": activity_name,  # Same activity to test deduplication
    }

    # Get evaluations mapping
    activity_evals = activity_to_evaluations(
        audio_path_to_activity=audio_paths,
        activity_tree=taxonomy_tree,
    )

    # Verify results
    assert len(activity_evals) == 1, "Expected one activity"
    assert activity_name in activity_evals, f"Expected '{activity_name}' activity"

    # Verify evaluations
    evals = activity_evals[activity_name]
    assert isinstance(evals, list), "Expected list of evaluations"
    # Since we removed dependency on specific check functions,
    # just verify we get a non-empty list of evaluations
    assert len(evals) > 0, "Expected non-empty evaluation list"


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


@pytest.mark.parametrize(
    "taxonomy_tree,has_children,expected_child",
    [
        (BIOACOUSTIC_ACTIVITY_TAXONOMY, False, None),
        (BRIDGE2AI_VOICE_TAXONOMY, True, "human"),
    ],
)
def test_taxonomy_tree_structure(
    taxonomy_tree: TaxonomyNode, has_children: bool, expected_child: Optional[str]
) -> None:
    """Tests that taxonomy trees have the expected structure."""
    # Verify root node exists and has expected properties
    assert taxonomy_tree.name in ["bioacoustic", "bridge2ai_voice"], f"Unexpected root name: {taxonomy_tree.name}"
    assert hasattr(taxonomy_tree, "checks"), "Root node should have checks attribute"
    assert hasattr(taxonomy_tree, "metrics"), "Root node should have metrics attribute"
    assert hasattr(taxonomy_tree, "children"), "Root node should have children attribute"

    # Verify root node children based on taxonomy type
    if has_children:
        assert len(taxonomy_tree.children) > 0, "Root node should have children"
        if expected_child:
            assert expected_child in taxonomy_tree.children, f"Taxonomy should have '{expected_child}' child"
            child_node = taxonomy_tree.children[expected_child]
            assert isinstance(child_node, TaxonomyNode), f"'{expected_child}' node should be TaxonomyNode instance"
    else:
        assert len(taxonomy_tree.children) == 0, "Bioacoustic taxonomy should have no children"


@pytest.mark.parametrize(
    "taxonomy_tree,expected_min_nodes",
    [(BIOACOUSTIC_ACTIVITY_TAXONOMY, 1), (BRIDGE2AI_VOICE_TAXONOMY, 2)],
)
def test_taxonomy_node_traversal(taxonomy_tree: TaxonomyNode, expected_min_nodes: int) -> None:
    """Tests that we can traverse the entire taxonomy tree."""
    visited_nodes = []

    def traverse(node: TaxonomyNode, depth: int = 0) -> None:
        """Recursively traverse the taxonomy tree."""
        visited_nodes.append((node.name, depth))
        for child in node.children.values():
            traverse(child, depth + 1)

    traverse(taxonomy_tree)

    # Verify we visited at least the expected minimum nodes
    assert len(visited_nodes) >= expected_min_nodes, f"Should visit at least {expected_min_nodes} node(s) in taxonomy"

    # Verify root is at depth 0
    assert visited_nodes[0][1] == 0, "Root should be at depth 0"

    # For taxonomies with children, verify we have nodes at different depths
    if expected_min_nodes > 1:
        depths = [depth for _, depth in visited_nodes]
        assert max(depths) > 0, "Should have nodes at depth > 0"


def test_activity_to_evaluations_with_missing_activity() -> None:
    """Tests handling of activities not found in taxonomy."""
    # Setup test data with non-existent activity
    audio_paths = {
        "audio1.wav": "nonexistent_activity",
    }

    # Should raise ValueError for missing activities (consistent with review.py behavior)
    with pytest.raises(ValueError, match="Activity 'nonexistent_activity' not found in taxonomy"):
        activity_to_evaluations(
            audio_path_to_activity=audio_paths,
            activity_tree=BIOACOUSTIC_ACTIVITY_TAXONOMY,
        )


def test_activity_to_evaluations_with_multiple_activities() -> None:
    """Tests mapping with multiple different activities."""
    # Setup test data with activities that exist in bridge2ai_voice taxonomy
    # (bioacoustic only has the root "bioacoustic" node)
    audio_paths = {
        "audio1.wav": "human",
        "audio2.wav": "breathing",
        "audio3.wav": "vocalization",
    }

    # Get evaluations mapping
    activity_evals = activity_to_evaluations(
        audio_path_to_activity=audio_paths,
        activity_tree=BRIDGE2AI_VOICE_TAXONOMY,
    )

    # Verify we get evaluations for activities that exist
    assert isinstance(activity_evals, dict), "Should return dictionary"

    # Check that we have some activities mapped
    found_activities = [activity for activity in audio_paths.values() if activity in activity_evals]
    assert len(found_activities) > 0, "Should find at least some activities in taxonomy"
