"""Functionality for Pydra."""

from typing import Iterator

import numpy as np
import torch
from pydra.utils.hash import Cache, bytes_repr_sequence_contents, register_serializer


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


# TODO: Ignore this for now but need to decide how to incorporate Pydra into the package
# Pydra runner
# need function that allows for marking a task (could be obfuscated internally)
# need function that runs the actual code/does the parallelization
# getting results is a bit clunky right now so could clean that up
# internal Pydra wouldn't need obfuscations for add/split but external API might be useful
# perhaps dictionary structure could be intuitive? linked list?

# what if we had a simple split/run structure:
# user: run X task with Y audios (Satra example); perhaps extraneous information of DeviceType, Model to use
#   - kinda easy example of just create the pydra setup ourselves, no user worries
# Senselab "pipelines"/Tasks:

# example:
# RAVDESS audios:
# - need to resample
# - perhaps run loudness normalization
# - run SER
# - run evaluation

# get RAVDESS audios
# split into batches
#   - run batches through resmapling
#   - pass to normalization
#   - run SER
#   - get partial evaluation
# - combine evaluation
# - give average
