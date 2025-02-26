"""Module for testing bioacoustic quality control."""

from collections import Counter

import pytest

from senselab.audio.tasks.bioacoustic_qc.taxonomy import BIOACOUSTIC_TASK_TREE


@pytest.mark.parametrize(
    "taxonomy_tree",
    [BIOACOUSTIC_TASK_TREE],
)
def test_unique_leaf_nodes(taxonomy_tree: dict) -> None:
    """Tests that all leaf nodes (keys with value None) in the taxonomy are unique."""

    def get_leaf_nodes(tree: dict) -> list[str]:
        """Recursively extract all leaf nodes (keys with value None) from the taxonomy tree."""
        leaf_nodes = []

        def traverse(subtree: dict) -> None:
            for key, value in subtree.items():
                if isinstance(value, dict):
                    traverse(value)
                elif value is None:
                    leaf_nodes.append(key)  # Append instead of adding to a set

        traverse(tree)
        return leaf_nodes

    leaf_nodes = get_leaf_nodes(taxonomy_tree)
    leaf_counts = Counter(leaf_nodes)
    duplicates = {node: count for node, count in leaf_counts.items() if count > 1}
    assert not duplicates, f"Duplicate leaf nodes found: {duplicates}"
