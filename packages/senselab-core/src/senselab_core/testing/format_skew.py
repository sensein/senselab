"""The format-skew value contract: a reader of a later minor ignores fields it does not declare, at every level."""

import contextlib
import copy
import pickle
from collections.abc import Mapping, Sequence

from pydantic import ValidationError

from senselab_core.kernel import formats
from senselab_core.kernel._base import Frozen, _reads
from senselab_core.kernel.formats import Formatted


def inject_unknown(value: object, dumped: object, *, root: bool = True) -> None:
    """Add a field no type declares at every level of `dumped`, the dump of `value`, but a nested `Formatted`'s own."""
    if isinstance(value, Formatted) and not root:
        return
    if isinstance(value, Frozen) and isinstance(dumped, dict) and set(dumped) <= set(type(value).model_fields):
        dumped["added_by_a_later_minor"] = 1
        for name in type(value).model_fields:
            inject_unknown(getattr(value, name), dumped.get(name), root=False)
    elif isinstance(value, Mapping) and isinstance(dumped, Mapping):
        for key, item in value.items():
            inject_unknown(item, dumped.get(key), root=False)
    elif isinstance(value, (tuple, list)) and isinstance(dumped, (tuple, list)):
        for item, held in zip(value, dumped, strict=False):
            inject_unknown(item, held, root=False)


def later_minor_ignored(samples: Sequence[Frozen]) -> None:
    """Check each sample reads past later-minor fields at every level, keeps that in its copies, else refuses."""
    for v in samples:
        cls = type(v)
        expected: Frozen
        scope: contextlib.AbstractContextManager[object]
        if isinstance(v, Formatted):
            if v.model_fields["format"].is_required():
                continue
            later = bump_minor(v.format)
            expected, dumped = v.model_copy(update={"format": later}), {**v.model_dump(), "format": later}
            strict = {**dumped, "format": v.format}
            inject_unknown(v, dumped)
            inject_unknown(v, strict)
            scope = contextlib.nullcontext()
        else:
            expected, dumped = v, v.model_dump()
            inject_unknown(v, dumped)
            strict, scope = dumped, formats.reading("sample@1.0", "sample@1.1")
        with scope:
            read = cls.model_validate(dumped)
        assert read == expected, cls.__qualname__
        assert read.read_under() is not None and read.read_under().dropped, cls.__qualname__  # type: ignore[union-attr]
        copies = (pickle.loads(pickle.dumps(read)), copy.copy(read), copy.deepcopy(read), read.model_copy())
        assert all(_reads(c) == _reads(read) for c in copies), f"{cls.__qualname__}: a copy lost what the read cost"
        try:
            cls.model_validate(strict)
        except ValidationError:
            continue
        raise AssertionError(f"{cls.__qualname__} read an unknown field under its own minor")


def bump_minor(tag: str) -> str:
    """Return `tag` with its minor version one later."""
    name, version = tag.split("@")
    major, minor = version.split(".")
    return f"{name}@{major}.{int(minor) + 1}"
