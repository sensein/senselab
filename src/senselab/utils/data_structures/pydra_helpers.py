"""Functionality for Pydra."""

from typing import Iterator

import numpy as np
import torch
from pydra.utils.hash import Cache, bytes_repr_sequence_contents, register_serializer

try:
    import opensmile

    OPENSMILE_AVAILABLE = True
except ImportError:
    OPENSMILE_AVAILABLE = False


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


if OPENSMILE_AVAILABLE:

    @register_serializer(opensmile.Smile)
    def bytes_repr_smile(obj: opensmile.Smile, _cache: Cache) -> Iterator[bytes]:
        """Serializer for opensmile.Smile.

        This function registers a custom serializer for instances of `opensmile.Smile`,
        allowing Pydra's caching system to recognize and hash these objects based on
        their configurations. By encoding essential attributes to bytes, we ensure that
        identical configurations produce the same hash, facilitating efficient workflow caching.

        Key Attributes Serialized:
            - `feature_set`: The OpenSMILE feature set, e.g., `eGeMAPSv02`.
            - `feature_level`: The feature level, e.g., `Functionals` or `LowLevelDescriptors`.
            - `options`: A dictionary containing additional configurations for feature extraction.
            - `logfile`: The log file path, if logging is enabled.
            - `verbose`: Boolean indicating verbosity in logging.
            - `column_names`: Column names of features generated by OpenSMILE, represented as an index.
            - `feature_names`: List of specific feature names extracted by OpenSMILE.
            - `hop_dur`: The hop duration for windowed feature extraction, if applicable.
            - `name`: Name identifier for the OpenSMILE instance.
            - `num_channels`: Number of audio channels expected by the instance.
            - `num_features`: Number of features generated for each frame.
            - `params`: Dictionary of internal configuration parameters such as `sampling_rate`, `channels`,
            `mixdown`, `resample`, and other settings impacting feature extraction.
            - `process_func_applies_sliding_window`: Indicates if a sliding window is applied in feature extraction.
            - `win_dur`: Duration of each window frame, if applicable.

        Args:
            obj (opensmile.Smile): The `opensmile.Smile` instance to be serialized.
            _cache (Cache): The Pydra cache object.

        Usage:
            This serializer is automatically used by Pydra to calculate a unique hash for `opensmile.Smile`
            objects in workflows, ensuring consistent hashing based on the object's configurations. The
            serializer helps avoid hash collisions in cases where `opensmile.Smile` instances have the same
            internal settings but different object IDs in memory.

        Returns:
            Iterator[bytes]: Byte-encoded representations of each serialized attribute.
        """
        _ = _cache  # This is just to silence the unused parameter warning

        yield f"{obj.__class__.__module__}{obj.__class__.__name__}:".encode()

        # Serialize key configuration attributes
        yield f"feature_set:{obj.feature_set}".encode()
        yield f"feature_level:{obj.feature_level}".encode()
        yield f"options:{obj.options}".encode()
        yield f"logfile:{obj.logfile}".encode()
        yield f"verbose:{obj.verbose}".encode()
        yield f"column_names:{obj.column_names}".encode()
        yield f"feature_names:{obj.feature_names}".encode()
        yield f"hop_dur:{obj.hop_dur}".encode()
        yield f"name:{obj.name}".encode()
        yield f"num_channels:{obj.num_channels}".encode()
        yield f"num_features:{obj.num_features}".encode()
        yield f"params:{obj.params}".encode()
        yield f"process_func_applies_sliding_window:{obj.process_func_applies_sliding_window}".encode()
        yield f"win_dur:{obj.win_dur}".encode()


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
