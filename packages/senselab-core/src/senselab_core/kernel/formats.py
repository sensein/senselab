"""Format tags carried by every persisted artifact."""

import logging
import re
from collections.abc import Collection, Iterator, Mapping
from contextlib import AbstractContextManager, contextmanager, nullcontext
from enum import Enum
from typing import Annotated, Any, ClassVar, TypeVar

from pydantic import StringConstraints, model_validator

from senselab_core.kernel._base import _READING, Frozen, ReadUnder, _Reading

F = TypeVar("F", bound=Frozen)
_LOG = logging.getLogger(__name__)
_REPORTED: set[tuple[str, str]] = set()

FORMAT_PATTERN = r"^[a-z0-9_.-]+@\d+\.\d+$"

SENSELAB_VOCAB_FORMAT = "senselab-vocab@1.4"
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

EQUIVALENCE_FORMAT = "equivalence@4.0"
"""The format of the measured equivalence classes, a group's table file (`load_equivalence`)."""
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

PLACEMENT_FORMAT = "placement@3.0"
"""The format of a plan's placement: each node's backend, env, device, batch, resource class and licenses."""

RETIRED_PLACEMENT_FORMATS = frozenset({"placement@2"})
"""Placement formats 2.x no longer reads, as `<name>@<major>`."""

PLAN_FRAGMENT_FORMAT = "plan-fragment@3.0"
"""The format of a plan fragment: jobs of one node of a plan, their input identities, the placement and the attempt."""

RUN_EVENT_FORMAT = "run-event@2.3"
WORKER_READY_FORMAT = "worker-ready@1.0"
"""The format of one run event, a line of `<run>/events/<writer>.jsonl`: a job's state change and its fault."""

SERVICE_ENTRY_FORMAT = "service-entry@1.0"
"""The format of a service's registry entry, `<registry>/svc-<identity digest>.json`: its identity and endpoint."""

EXECUTION_PROFILE_FORMAT = "execution-profile@5.0"
"""The format of an execution profile, `runtime/data/profiles/<name>.yaml` or a user's override."""
RETIRED_EXECUTION_PROFILE_FORMATS = frozenset(
    {"execution-profile@1", "execution-profile@2", "execution-profile@3", "execution-profile@4"}
)
"""Execution-profile formats 2.x no longer reads, as `<name>@<major>`."""

CATALOGUE_FORMAT = "catalogue@2.0"
"""The format of a model catalogue."""
RETIRED_CATALOGUE_FORMATS = frozenset({"catalogue@1"})
"""Catalogue formats 2.x no longer reads, as `<name>@<major>`."""

MEASURES_LAYOUT_TAG = "measures@1"
"""The layout tag of a measures root: the directory `<root>/measures@1/` holds every record kind."""
MEASURE_UNIT_TAG = "measure-unit@1"
"""The identity tag of a measure unit: its digest is over the unit's identity fields."""
MEASURE_CLASS_TAG = "hardware-class@1"
"""The identity tag of a seen hardware class: its digest names the index entry."""
MEASURE_DECLARED_TAG = "hardware-declared@1"
"""The identity tag of a resource class's declared hardware fields."""
MEASURE_UNIT_FORMAT = "measure-unit@1.0"
"""The format of a unit's identity record, `units/<u[:2]>/<u>/unit.json`."""
MEASURE_CLASS_FORMAT = "measure-class@1.0"
"""The format of a hardware index entry, `classes/<declared digest>/<seen digest>.json`: the seen `HardwareClass`."""
MEASURE_OBSERVATIONS_FORMAT = "measure-observations@1.1"
"""The format of one probe's observations, `units/<u[:2]>/<u>/obs/<probe id>.json`."""
MEASURE_DRIFT_FORMAT = "measure-drift@1.0"
"""The format of production evidence that an estimate is suspect."""
MEASURE_SUPERSEDE_FORMAT = "measure-supersede@1.0"
"""The format of a record that a confirming probe disagreed, so earlier observations stop pooling."""
MEASURE_REGIME_FORMAT = "measure-regime@1.0"
"""The format of a co-tenancy factor of an estimate."""
MEASURE_OOM_FORMAT = "measure-oom@1.1"
"""The format of a resource fault at a batch and length."""
MEASURE_UNFIT_FORMAT = "measure-unfit@1.0"
"""The format of a unit's load that ran out of memory in a grant: the unit does not fit that class."""

RUN_CONFIG_FORMAT = "run-config@2.1"
"""The format of a run config, `runtime/data/run_config.yaml` or a user's: comparison thresholds and calibration."""

PLACEMENT_REQUEST_FORMAT = "placement-request@1.0"
"""The format of what a run needs, handed to a site's placement hook."""
PLACEMENT_ANSWER_FORMAT = "placement-answer@1.0"
"""The format of a site's placement hook's answer."""

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


@contextmanager
def reading(written: str, found: object) -> Iterator[None]:
    """Scope in which a record declaring `found` is validated by a reader of `written`.

    Under a later minor of `written`'s name and major, every `Frozen` validated inside drops the fields it does not
    declare, at every nesting level, and records what it dropped (`Frozen.read_under`); the newer minor is logged the
    first time it is met. Under any other tag unknown fields are refused. A `Formatted` decides from its own tag; a
    bare `Frozen` inherits the scope it is validated in.
    """
    newer = check(written, found) is FormatVerdict.ACCEPTED and later_minor(written, str(found))
    parent, current = _READING.get(), _Reading(str(found), []) if newer else None
    token = _READING.set(current)
    try:
        yield
    finally:
        _READING.reset(token)
        if current is not None:
            if parent is not None:
                parent.dropped.extend(current.dropped)
            if (written, current.tag) not in _REPORTED:
                _REPORTED.add((written, current.tag))
                _LOG.info("reading %s as %s: ignoring %s", current.tag, written, sorted(set(current.dropped)))


class FieldsOfLaterMinor(ValueError):
    """A value read under a later minor lost fields: it is not part of an identity, and it is not rewritten."""

    def __init__(self, read: ReadUnder) -> None:
        """Name the minor and the fields."""
        super().__init__(f"read under {read.tag}, which declares fields this version dropped: {list(read.dropped)}")
        self.read = read


def require_writable(value: Frozen) -> None:
    """Refuse a value that was read under a later minor and lost fields.

    Raises:
        FieldsOfLaterMinor: naming the newer minor and the fields.
    """
    read = value.read_under()
    if read is not None and read.dropped:
        raise FieldsOfLaterMinor(read)


def validated(model: type[F], record: Mapping[str, Any], written: str, found: str) -> F:
    """Return `record` as `model`, read under `found` by a reader of `written`.

    Raises:
        FieldsOfLaterMinor: if `record` held fields `model` does not declare, so the value is not the whole of what
            was written and its digest would differ.
    """
    with reading(written, found):
        value = model.model_validate(record)
    require_writable(value)
    return value


class Formatted(Frozen):
    """A model carrying `format: "<name>@<major>.<minor>"`.

    A subclass declares the tag it writes as the default, `format: FormatTag = "<name>@<major>.<minor>"`,
    and then reads only tags with that name and major version. Under a later minor, unknown fields are ignored
    at every nesting level (`reading`); under its own or an earlier minor they are refused.
    """

    format: FormatTag

    port_value: ClassVar[bool] = True
    """Whether a task port may name this type; a document that never flows on a port, such as a graph, sets False."""

    @classmethod
    def _reading(cls, data: object) -> AbstractContextManager[object]:
        written = cls.model_fields["format"].default
        if isinstance(data, Mapping) and isinstance(written, str):
            return reading(written, data.get("format", written))
        return nullcontext()

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


ESTIMATE_FORMAT = "estimate@1"
"""The format tag of an `Estimate`: the estimator that derived it."""
