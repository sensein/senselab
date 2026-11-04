"""Format tags carried by every persisted artifact."""

import re
from collections.abc import Collection
from enum import Enum
from typing import Annotated, ClassVar

from pydantic import ConfigDict, StringConstraints, model_validator

from senselab_core.kernel._base import Frozen

FORMAT_PATTERN = r"^[a-z0-9_.-]+@\d+\.\d+$"

SENSELAB_VOCAB_FORMAT = "senselab-vocab@1.3"
"""The format of a `senselab:` vocabulary document, carried as its ontology node's `owl:versionInfo` (ADR-010)."""
PROV_ACTIVITY_TAG = "prov-activity@1"
"""The identity tag of a PROV activity: its compact JSON-LD node without `@id`."""
PROV_NODE_TAG = "prov-node@1"
"""The identity tag of a PROV agent or environment: its compact JSON-LD node without `@id`."""
PROVENANCE_FORMAT = "provenance@3.0"
"""The format of a PROV-O JSON-LD provenance document, carried as its top-level `format` (ADR-010)."""

RETIRED_PROVENANCE_FORMATS = frozenset({"provenance@1", "provenance@2"})
"""Provenance formats 2.x no longer reads, as `<name>@<major>`."""

LABELS_FORMAT = "labels@2.0"
"""The format of a run's private label map, canonical JSON in `<run>/private/labels@1.json` (ADR-007)."""
RETIRED_LABELS_FORMATS = frozenset({"labels@1"})
"""Label-map formats 2.x no longer reads, as `<name>@<major>`."""

BENCHMARK_RESULT_FORMAT = "benchmark-result@2.0"
"""The format of a benchmark run's result document, `<run>/result.json`."""
RETIRED_BENCHMARK_RESULT_FORMATS = frozenset({"benchmark-result@1"})
"""Result-document formats 2.x no longer reads, as `<name>@<major>`."""

EQUIVALENCE_FORMAT = "equivalence@3.0"
"""The format of the measured equivalence classes, `runtime/data/equivalence.yaml`."""
EQUIVALENCE_EVIDENCE_FORMAT = "equivalence-evidence@4.0"
"""The format of one verify-sampling comparison, a record in the result store."""

PARITY_REFERENCE_FORMAT = "parity-reference@3.0"
"""The format of the 1.x outputs of the held-out items of one backend, `benchmarks/parity_reference.py convert`."""
PARITY_OUTPUTS_FORMAT = "parity-outputs@4.0"
"""The format of the 2.x outputs of the held-out items of one backend and stack, `parity_reference.py run`."""
PARITY_COMPARISON_FORMAT = "parity-comparison@1.0"
"""The format of one parity decision and the thresholds it was made under, `parity_reference.py compare`."""

GRAPH_FORMAT = "graph@1.0"
"""The format of a graph document, task nodes joined by plain, `map` and `gather` edges (ADR-001)."""

INPUT_MANIFEST_FORMAT = "input-manifest@1.0"
"""The format of an input manifest: each record of a dataset with the content identity of every input it binds."""

LOGICAL_PLAN_FORMAT = "logical-plan@1.0"
"""The format of a logical plan: a graph with its pinned models, profile slices, manifest digest and reuse policy."""

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


class FormatVerdict(str, Enum):
    """What a reader of one format makes of the tag a record declares (`check`)."""

    ACCEPTED = "accepted"
    """The written format's name and major: the reader reads it, ignoring fields of a later minor."""
    RETIRED = "retired"
    """A retired `<name>@<major>`: refused with the migration statement, never treated as corrupt."""
    UNKNOWN = "unknown"
    """Another format tag, such as a later major: version skew, not corruption."""
    MALFORMED = "malformed"
    """Not a format tag at all: the record is corrupt."""


def check(written: str, found: object, retired: Collection[str] = ()) -> FormatVerdict:
    """Return the verdict of a reader of `written` on a record declaring `found`, `retired` as `<name>@<major>`."""
    if not isinstance(found, str) or re.fullmatch(FORMAT_PATTERN, found) is None:
        return FormatVerdict.MALFORMED
    if readable(written, found):
        return FormatVerdict.ACCEPTED
    return FormatVerdict.RETIRED if "@".join(_name_and_major(found)) in retired else FormatVerdict.UNKNOWN


def later_minor(written: str, found: str) -> bool:
    """Whether `found`, of `written`'s name and major, is a later minor than `written`."""
    return int(found.rsplit(".", 1)[1]) > int(written.rsplit(".", 1)[1])


class Formatted(Frozen):
    """A model carrying `format: "<name>@<major>.<minor>"`.

    A subclass declares the tag it writes as the default, `format: FormatTag = "<name>@<major>.<minor>"`,
    and then reads only tags with that name and major version; unknown fields are ignored. A type that
    is not `Formatted` refuses unknown fields.
    """

    model_config = ConfigDict(extra="ignore")

    format: FormatTag

    port_value: ClassVar[bool] = True
    """Whether a task port may name this type; a document that never flows on a port, such as a graph, sets False."""

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
