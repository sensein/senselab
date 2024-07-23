""".. include:: ../../README.md"""  # noqa: D415

from typing import Iterator

import nest_asyncio
import numpy as np
import torch
from pydra.utils.hash import Cache, bytes_repr_sequence_contents, register_serializer

nest_asyncio.apply()


@register_serializer(torch.Tensor)
def bytes_repr_arraylike(obj: torch.Tensor, cache: Cache) -> Iterator[bytes]:
    """Register a serializer for Torch tensors that allows Pydra to properly use them."""
    yield f"{obj.__class__.__module__}{obj.__class__.__name__}:".encode()
    array = np.asanyarray(obj)
    yield f"{array.size}:".encode()
    if array.dtype == "object":
        yield from bytes_repr_sequence_contents(iter(array.ravel()), cache)
    else:
        yield array.tobytes(order="C")
