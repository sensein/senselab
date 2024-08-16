"""Provides functionality for batch processing."""

from typing import Any, List


def batch_list(items: List[Any], n_batches: int) -> List[List[Any]]:
    """Splits a list into a specified number of batches.

    Args:
        items (List[Any]): The list of items to be batched.
        n_batches (int): The number of batches to divide the list into.

    Returns:
        List[List[Any]]: A list of lists, where each sublist is a batch of items.
    """
    if not items:
        return []

    n_items = len(items)
    n_batches = min(n_batches, n_items)  # ensure n_batches does not exceed n_items
    batch_size = -(n_items // -n_batches)  # fast ceiling division
    return [items[i : i + batch_size] for i in range(0, n_items, batch_size)]
