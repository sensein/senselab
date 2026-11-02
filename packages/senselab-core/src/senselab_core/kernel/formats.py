"""Format tags carried by every persisted artifact."""

import re
from typing import Annotated

from pydantic import ConfigDict, StringConstraints, model_validator

from senselab_core.kernel._base import Frozen

FORMAT_PATTERN = r"^[a-z0-9_.-]+@\d+\.\d+$"

SENSELAB_VOCAB_FORMAT = "senselab-vocab@1.0"
"""The format of a `senselab:` vocabulary document, carried as its ontology node's `owl:versionInfo` (ADR-010)."""
PROV_ACTIVITY_TAG = "prov-activity@1"
"""The identity tag of a PROV activity: its compact JSON-LD node without `@id`."""
PROV_NODE_TAG = "prov-node@1"
"""The identity tag of a PROV agent or environment: its compact JSON-LD node without `@id`."""
PROVENANCE_FORMAT = "provenance@3.0"
"""The format of a PROV-O JSON-LD provenance document, carried as its top-level `format` (ADR-010)."""

LABELS_FORMAT = "labels@1.0"
"""The format of a run's private label map, `<run>/private/labels@1.json` (ADR-007)."""

BENCHMARK_RESULT_FORMAT = "benchmark-result@1.0"
"""The format of a benchmark run's result document, `<run>/result.json`."""

TOLERANCE_FORMAT = "tolerance@1.0"
"""The format of a backend's tolerance profile, `senselab/data/tolerance/<backend>.yaml`."""
TOLERANCE_OBSERVATIONS_FORMAT = "tolerance-observations@1.0"
"""The format of one measurement job's output, the input of `scripts/generate_tolerance.py generate`."""

FormatTag = Annotated[str, StringConstraints(pattern=FORMAT_PATTERN)]
"""`<name>@<major>.<minor>`."""


def _name_and_major(tag: str) -> tuple[str, str]:
    name, version = tag.split("@")
    return name, version.split(".")[0]


def readable(written: str, found: object) -> bool:
    """Whether `found` is a format tag of `written`'s name and major version: one a reader of `written` reads."""
    return (
        isinstance(found, str)
        and re.fullmatch(FORMAT_PATTERN, found) is not None
        and _name_and_major(found) == _name_and_major(written)
    )


class Formatted(Frozen):
    """A model carrying `format: "<name>@<major>.<minor>"`.

    A subclass declares the tag it writes as the default, `format: FormatTag = "<name>@<major>.<minor>"`,
    and then reads only tags with that name and major version; unknown fields are ignored. A type that
    is not `Formatted` refuses unknown fields.
    """

    model_config = ConfigDict(extra="ignore")

    format: FormatTag

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: object) -> None:
        """Refuse a subclass whose declared default tag is malformed."""
        super().__pydantic_init_subclass__(**kwargs)
        written = cls.model_fields["format"].default
        if isinstance(written, str) and not re.match(FORMAT_PATTERN, written):
            raise TypeError(f"{cls.__name__} declares a malformed format {written!r}")

    @model_validator(mode="after")
    def _check_readable(self) -> "Formatted":
        written = type(self).model_fields["format"].default
        if isinstance(written, str) and not readable(written, self.format):
            raise ValueError(f"{type(self).__name__} reads {written} and its minor versions, got {self.format}")
        return self
