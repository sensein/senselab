"""Tests batching functions."""

import pytest

from senselab.utils.tasks.batching import batch_list


def test_batch_list() -> None:
    """Tests the batch_list function with various examples."""
    # Test case 1: Normal case with even division
    items = [1, 2, 3, 4, 5, 6]
    n_batches = 3
    expected_output = [[1, 2], [3, 4], [5, 6]]
    assert batch_list(items, n_batches) == expected_output

    # Test case 2: Normal case with uneven division
    items = [1, 2, 3, 4, 5, 6, 7]
    n_batches = 3
    expected_output = [[1, 2, 3], [4, 5, 6], [7]]
    assert batch_list(items, n_batches) == expected_output

    # Test case 3: Single batch
    items = [1, 2, 3, 4, 5, 6]
    n_batches = 1
    expected_output = [[1, 2, 3, 4, 5, 6]]
    assert batch_list(items, n_batches) == expected_output

    # Test case 4: More batches than items
    items = [1, 2, 3]
    n_batches = 5
    expected_output = [[1], [2], [3]]
    assert batch_list(items, n_batches) == expected_output

    # Test case 5: Empty list
    items = []
    n_batches = 3
    expected_output = []
    assert batch_list(items, n_batches) == expected_output


if __name__ == "__main__":
    pytest.main()
