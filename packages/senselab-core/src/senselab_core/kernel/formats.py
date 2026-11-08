"""Format tags carried by every persisted artifact."""

import re
from collections.abc import Collection
from enum import Enum
from typing import Annotated, ClassVar

from pydantic import ConfigDict, StringConstraints, model_validator

from senselab_core.kernel._base import Frozen

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

PLACEMENT_FORMAT = "placement@1.0"
"""The format of a plan's placement: each node's backend, env, device, batch, resource class, licenses and cost."""

PLAN_FRAGMENT_FORMAT = "plan-fragment@2.0"
"""The format of a plan fragment: jobs of one node of a plan, their input identities, the placement and the attempt."""

RUN_EVENT_FORMAT = "run-event@1.0"
"""The format of one run event, a line of `<run>/events/<writer>.jsonl`: a job's state change and its fault."""

SERVICE_ENTRY_FORMAT = "service-entry@1.0"
"""The format of a service's registry entry, `<registry>/svc-<identity digest>.json`: its identity and endpoint."""

EXECUTION_PROFILE_FORMAT = "execution-profile@3.0"
"""The format of an execution profile, `runtime/data/profiles/<name>.yaml` or a user's override."""
RETIRED_EXECUTION_PROFILE_FORMATS = frozenset({"execution-profile@1", "execution-profile@2"})
"""Execution-profile formats 2.x no longer reads, as `<name>@<major>`."""

CATALOGUE_FORMAT = "catalogue@2.0"
"""The format of a model catalogue; its resource tables are separate `resource-table@1.0` files."""
RETIRED_CATALOGUE_FORMATS = frozenset({"catalogue@1"})
"""Catalogue formats 2.x no longer reads, as `<name>@<major>`."""

RESOURCE_TABLE_FORMAT = "resource-table@1.0"
"""The format of one task and backend's measured resources, `data/resources/<task>/<backend>.yaml`."""

RESOURCE_TABLE_2_FORMAT = "resource-table@2.0"
"""The format of a measured resource table whose rows are measure units with their estimates (`resource-table@1.0`'s
successor)."""

MEASURES_LAYOUT_TAG = "measures@1"
"""The layout tag of a measures root: the directory `<root>/measures@1/` holds every record kind."""
MEASURE_UNIT_TAG = "measure-unit@1"
"""The identity tag of a measure unit: its digest is over the unit's identity fields."""
MEASURE_UNIT_FORMAT = "measure-unit@1.0"
"""The format of a unit's identity record, `units/<u[:2]>/<u>/unit.json`."""
MEASURE_OBSERVATIONS_FORMAT = "measure-observations@1.0"
"""The format of one probe's observations, `units/<u[:2]>/<u>/obs/<probe id>.json`."""
MEASURE_DRIFT_FORMAT = "measure-drift@1.0"
"""The format of production evidence that an estimate is suspect."""
MEASURE_SUPERSEDE_FORMAT = "measure-supersede@1.0"
"""The format of a record that a confirming probe disagreed, so earlier observations stop pooling."""
MEASURE_REGIME_FORMAT = "measure-regime@1.0"
"""The format of a co-tenancy factor of an estimate."""
MEASURE_OOM_FORMAT = "measure-oom@1.0"
"""The format of a resource fault at a batch and length."""

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


ESTIMATE_FORMAT = "estimate@1"
"""The format tag of an `Estimate`: the estimator that derived it."""
