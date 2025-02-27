"""Module for testing bioacoustic quality control."""

from collections import Counter
from typing import Dict, List

import pytest

from senselab.audio.tasks.bioacoustic_qc.constants import BIOACOUSTIC_TASK_TREE


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
