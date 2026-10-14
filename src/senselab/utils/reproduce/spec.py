"""Manifest and profile schema for the reproduce driver.

A **manifest** is ``{name, steps}``: the steps of one analysis, each pinned to a repository commit,
a hashed env and an argv template. A **manifest set** is the steps of several manifests taken
together; step names are unique across the set. A **profile** is the caller's site description:
data roots, result roots, partitions and the keyed environment map. A **source spec** is the
reduced step a draft manifest holds for ``prefetch --sources-only``.

Files are YAML (``.yaml``/``.yml``) or JSON (``.json``).
"""

import copy
import datetime
import json
import math
import os
import re
import stat
import tempfile
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path, PurePosixPath
from typing import Annotated, Any, Literal, cast

import yaml
from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationInfo,
    field_validator,
    model_validator,
)

from senselab.utils.reproduce.metrics import (
    CLASSIFICATION_KINDS,
    POSITIVE_FLOOR_METRICS,
    SCALED_QUANTITIES,
    SECONDARY_BY_KIND,
    HeadlineRule,
    MetricName,
    MetricsKind,
    SplitOptOutReason,
    headline_rule_error,
    metric_address_error,
    metric_quantity,
)

Commit = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{40}$")]
Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
Name = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9_][A-Za-z0-9_.-]*$")]
Label = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9_+-]{0,63}$")]
CountName = Annotated[str, StringConstraints(pattern=r"^[a-z][a-z0-9_]{0,63}$")]
SourceRoot = Annotated[str, StringConstraints(pattern=r"^(up:)?[A-Za-z0-9_.-]+$")]


def _relative_path(value: str) -> str:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or "\\" in value or ".." in path.parts:
        raise ValueError(f"path must be relative with no '..': {value!r}")
    return value


RelPath = Annotated[str, AfterValidator(_relative_path)]

FIXED_PLACEHOLDERS = frozenset({"out", "tmp", "checkout", "venv", "cache"})
RESERVED_ROOT_NAMES = frozenset({"checkout"})
_TOKEN = re.compile(r"\{\{|\}\}|\{([^{}]*)\}|[{}]")
_KEYED_PLACEHOLDER = re.compile(r"^(root|up|input):([A-Za-z0-9_.-]+)$")
_SECRET_KEY = re.compile(r"(^|_)(TOKEN|KEY|SECRET|PASSWORD|PASSWD|CREDENTIALS?)S?(_|$)", re.IGNORECASE)
ALLOWED_SECRET_LIKE_KEYS = frozenset({"TOKENIZERS_PARALLELISM"})
_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
R_LIB_TEMPLATE = re.compile(r"^\{cache\}/r/(?!.*\.\.)[A-Za-z0-9_][A-Za-z0-9_.-]*$")
"""The only form an R library may take: ``{cache}/r/<name>``, one plain name with no leading dot and no ``..``."""


def placeholders(template: str) -> list[str]:
    """Return the placeholders in a template, in order; ``{{`` and ``}}`` are literal braces.

    Args:
        template: An argv element, ``cwd``, ``pythonpath`` entry or other templated string.

    Returns:
        The text inside each single-brace pair, e.g. ``["out", "up:train"]``.

    Raises:
        ValueError: If a placeholder is outside ``{out}``, ``{tmp}``, ``{checkout}``, ``{venv}``,
            ``{cache}``, ``{root:<name>}``, ``{up:<step>}`` and ``{input:<id>}``, or a brace is unmatched.
    """
    found = []
    for match in _TOKEN.finditer(template):
        text, name = match.group(0), match.group(1)
        if text in ("{{", "}}"):
            continue
        if name is None:
            raise ValueError(f"unmatched brace in {template!r}")
        if name not in FIXED_PLACEHOLDERS and not _KEYED_PLACEHOLDER.match(name):
            raise ValueError(f"unknown placeholder {{{name}}} in {template!r}")
        found.append(name)
    return found


def render(template: str, values: Mapping[str, str | os.PathLike[str]]) -> str:
    """Expand a template with the grammar of `placeholders`; ``{{`` and ``}}`` become literal braces.

    Args:
        template: A templated string.
        values: Placeholder text to value, e.g. ``{"out": "/r/x", "root:data": "/d"}``.

    Returns:
        The expanded string.

    Raises:
        ValueError: If the template is malformed or names an unknown placeholder.
        KeyError: If a placeholder has no value.
    """
    placeholders(template)

    def expand(match: re.Match[str]) -> str:
        text, name = match.group(0), match.group(1)
        if text in ("{{", "}}"):
            return text[0]
        return os.fspath(values[name])

    return _TOKEN.sub(expand, template)


def _upstream(template: str) -> list[str]:
    return [p.split(":", 1)[1] for p in placeholders(template) if p.startswith("up:")]


def _registered(template: str) -> list[str]:
    return [p.split(":", 1)[1] for p in placeholders(template) if p.startswith("input:")]


def _reject_secret_keys(env: dict[str, str]) -> dict[str, str]:
    bad = sorted(k for k in env if _SECRET_KEY.search(k) and k.upper() not in ALLOWED_SECRET_LIKE_KEYS)
    if bad:
        raise ValueError(f"secret-like environment keys are refused: {bad}")
    return env


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class LockRef(_Model):
    """A hashed lock file, relative to the manifest repository."""

    lock: RelPath
    lock_sha256: Sha256


class FileRef(_Model):
    """A hashed file, relative to the manifest repository."""

    file: RelPath
    sha256: Sha256


class TorchSpec(LockRef):
    """The torch-family lock and the index it was compiled against."""

    index_url: str


class BodySpec(LockRef):
    """The body lock: compiled from a sidecar ``requirements.in`` or exported from a repository ``uv.lock``."""

    source: Literal["requirements.in", "uv-export"]
    requirements_in: RelPath | None = None
    requirements_in_sha256: Sha256 | None = None
    export_cmd: list[str] | None = None
    index_url: str = "https://pypi.org/simple"
    config_file: RelPath | None = None
    config_file_sha256: Sha256 | None = None
    overrides: RelPath | None = None
    overrides_sha256: Sha256 | None = None

    @model_validator(mode="after")
    def _paired(self) -> "BodySpec":
        if self.source == "requirements.in" and (self.requirements_in is None or self.requirements_in_sha256 is None):
            raise ValueError("source requirements.in needs requirements_in and requirements_in_sha256")
        if self.source == "uv-export" and not self.export_cmd:
            raise ValueError("source uv-export needs export_cmd")
        for file in ("config_file", "overrides"):
            if (getattr(self, file) is None) != (getattr(self, f"{file}_sha256") is None):
                raise ValueError(f"{file} and {file}_sha256 come together")
        return self


ModuleName = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9_.+-]+(/[A-Za-z0-9_.+-]+)*$")]


class SdistSpec(_Model):
    """An sdist built into a wheel at prefetch, in its own hashed build env.

    ``build_modules`` are environment modules loaded, in order, for the build (e.g. a compiler and a
    CUDA toolkit).
    """

    name: str
    url: str
    sha256: Sha256
    patches: list[FileRef] = []
    build_env: LockRef
    build_env_vars: dict[str, str] = {}
    build_modules: list[ModuleName] = []
    torch_cuda_arch_list: str | None = None

    _no_secrets = field_validator("build_env_vars")(_reject_secret_keys)

    @field_validator("build_modules")
    @classmethod
    def _unique_modules(cls, value: list[str]) -> list[str]:
        if len(set(value)) != len(value):
            raise ValueError("build_modules has duplicates")
        return value


class PipCheckExpected(_Model):
    """A declared ``uv pip check`` deviation."""

    package: str
    kind: Literal["requirement", "platform"]
    detail: str


class RPackage(_Model):
    """A pinned R package."""

    name: str
    version: str


class EnvSpec(_Model):
    """A step's env: the hashed locks and the checks run against the built venv."""

    python: Annotated[str, StringConstraints(pattern=r"^\d+\.\d+$")]
    python_exact: Annotated[str, StringConstraints(pattern=r"^\d+\.\d+\.\d+$")]
    torch: TorchSpec | None
    body: BodySpec
    sdist_allow: list[SdistSpec] = []
    exclude_newer: datetime.date
    extra_no_emit: list[str] = []
    first_resolve: FileRef | None = None
    wheels: LockRef | None = None
    test: LockRef | None = None
    notebook_tools: LockRef | None = None
    pip_check_expected: list[PipCheckExpected] = []
    smoke_imports: list[str] = []
    max_cuda: str | None = None
    r_packages: list[RPackage] = []
    r_snapshot: datetime.date | None = None
    r_lib: str | None = None

    @field_validator("exclude_newer", "r_snapshot", mode="before")
    @classmethod
    def _iso_date(cls, value: object, info: ValidationInfo) -> object:
        if value is None or type(value) is datetime.date:
            return value
        if isinstance(value, str) and _ISO_DATE.match(value):
            return datetime.date.fromisoformat(value)
        raise ValueError(f"{info.field_name} must be a YYYY-MM-DD date")

    @field_validator("r_lib")
    @classmethod
    def _r_lib_under_cache(cls, value: str | None) -> str | None:
        if value is not None and not R_LIB_TEMPLATE.match(value):
            raise ValueError("E_SPEC_BAD_R_LIB: r_lib must be {cache}/r/<name>, one plain name")
        return value

    @model_validator(mode="after")
    def _torch_consistency_input(self) -> "EnvSpec":
        if self.torch is not None and self.body.source == "requirements.in" and self.first_resolve is None:
            raise ValueError("a torch lock with a requirements.in body needs first_resolve")
        return self


class InputSpec(_Model):
    """Patterns, relative to a data root or ``up:<step>``'s result dir, digested before and after the step."""

    root: SourceRoot
    patterns: list[str]


class StageSpec(_Model):
    """Files copied into ``{out}/stage/<dest>`` before the step.

    ``root`` is a data-root name, ``checkout`` or ``up:<step>`` (an upstream step's result dir).
    """

    root: SourceRoot
    patterns: list[str]
    dest: str


class Resources(_Model):
    """The profile partition key and the step's hardware and native-tool needs.

    ``native`` names entries of the profile's ``native`` map; only those entries' dirs reach the step's
    ``PATH`` and ``LD_LIBRARY_PATH``.
    """

    profile: str
    gpu_arch: list[str] | None = None
    min_driver: str | None = None
    native: list[str] = []


class FromStep(_Model):
    """A target taken from an upstream step's ``metrics.json``."""

    step: str
    metric: str


class ToleranceFrom(_Model):
    """A tolerance computed from upstream calibration steps: ``max(a, k * sd)`` of their metric."""

    steps: list[str] = Field(min_length=2)
    rule: Literal["max(a, k*sd)"]
    a: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    k: Annotated[float, Field(ge=0, allow_inf_nan=False)]

    @field_validator("steps")
    @classmethod
    def _unique_steps(cls, value: list[str]) -> list[str]:
        if len(set(value)) != len(value):
            raise ValueError("tolerance_from.steps has duplicates")
        return value


Finite = Annotated[float, Field(allow_inf_nan=False)]
NonNegative = Annotated[float, Field(ge=0, allow_inf_nan=False)]
BAND_A_MIN = {"fraction": 0.002, "pp": 0.2}
"""The default ``a_min`` of a band, by unit."""


class BandSpec(_Model):
    """A ``seed_band`` expectation: the stored headline and the chains or re-draws whose spread sets its half-width.

    ``floor`` (declared) or ``floor_from: entry`` (the metric entry's own ``floor``) gives the chance floor,
    and ``floor_source`` names where it comes from. ``chains`` are the step names of the independent chains
    (at least two, all in the step's ``needs``); ``chain1``, when set, is the originally seeded chain and
    gets a sub-verdict. An inference row gives ``redraws`` instead: exactly three re-draw steps over a subset
    of ``n_subset`` of the ``n_full`` items, with ``max(ceil(0.10 * n_full), min(200, n_full)) <= n_subset
    <= n_full`` (``E_SPEC_BAND_SUBSET``); ``chain1`` is refused with them. ``a_min`` defaults by ``unit``
    (`BAND_A_MIN`); ``rho`` must be below ``max_width_frac`` (``E_SPEC_BAND_RHO``).
    """

    stored: Finite
    unit: Literal["fraction", "pp"]
    floor: Finite | None = None
    floor_from: Literal["entry"] | None = None
    floor_source: Label
    chains: Annotated[list[Name], Field(min_length=2)] | None = None
    chain1: Name | None = None
    redraws: list[Name] | None = None
    n_subset: Annotated[int, Field(ge=1)] | None = None
    n_full: Annotated[int, Field(ge=1)] | None = None
    k: NonNegative = 2.0
    rho: NonNegative = 0.10
    a_min: Annotated[float, Field(gt=0, allow_inf_nan=False)]
    max_width_frac: Annotated[float, Field(gt=0, le=1, allow_inf_nan=False)] = 0.25
    characterise: bool = False

    @model_validator(mode="before")
    @classmethod
    def _default_a_min(cls, data: object) -> object:
        if isinstance(data, Mapping) and "a_min" not in data and data.get("unit") in BAND_A_MIN:
            return {**data, "a_min": BAND_A_MIN[data["unit"]]}
        return data

    @model_validator(mode="after")
    def _shape(self) -> "BandSpec":
        if (self.floor is None) == (self.floor_from is None):
            raise ValueError("E_SPEC_BAND_SHAPE: a band takes exactly one of floor and floor_from")
        if (self.chains is None) == (self.redraws is None):
            raise ValueError("E_SPEC_BAND_SHAPE: a band takes exactly one of chains and redraws")
        if self.redraws is not None and len(self.redraws) != 3:
            raise ValueError("E_SPEC_BAND_SHAPE: a band takes exactly three redraws")
        if len(set(self.sources())) != len(self.sources()):
            raise ValueError("E_SPEC_BAND_SHAPE: band steps have duplicates")
        if self.chain1 is not None and (self.chains is None or self.chain1 not in self.chains):
            raise ValueError("E_SPEC_BAND_SHAPE: chain1 must be one of the chains")
        counts = (self.n_subset is not None) + (self.n_full is not None)
        if counts != (2 if self.redraws is not None else 0):
            raise ValueError("E_SPEC_BAND_SHAPE: n_subset and n_full go with redraws, both or neither")
        if self.n_subset is not None and self.n_full is not None:
            least = max(math.ceil(0.10 * self.n_full), min(200, self.n_full))
            if not least <= self.n_subset <= self.n_full:
                raise ValueError(f"E_SPEC_BAND_SUBSET: n_subset must be in [{least}, n_full]")
        if self.rho >= self.max_width_frac:
            raise ValueError("E_SPEC_BAND_RHO: rho must be below max_width_frac")
        return self

    def sources(self) -> list[str]:
        """The upstream steps the band reads: its chains or its re-draws."""
        return list(self.chains if self.chains is not None else self.redraws or [])


def _check_band_scale(step: str, metric: str, band: BandSpec) -> None:
    """Refuse ``pp`` on an unscaled metric, and a ``stored`` or ``floor`` outside the metric's range in the unit."""
    quantity = metric_quantity(metric)
    if quantity not in SCALED_QUANTITIES:
        if band.unit == "pp":
            raise ValueError(f"E_SPEC_EXPECT_UNIT: step {step!r} band on {metric!r} takes unit fraction")
        return
    scale = 100.0 if band.unit == "pp" else 1.0
    low = -scale if quantity == "mcc" else 0.0
    for value in (band.stored, band.floor):
        if value is not None and not low <= value <= scale:
            raise ValueError(f"E_SPEC_BAND_SHAPE: step {step!r} band on {metric!r} is outside its range in {band.unit}")


class Expect(_Model):
    """One expected metric.

    ``abs`` takes one target (``value`` or ``from_step``) and one tolerance (``tolerance`` or
    ``tolerance_from``); ``exact`` takes one target and no tolerance; ``none`` records the metric
    and takes no tolerance; ``seed_band`` takes a ``band`` and nothing else (``E_SPEC_BAND_SHAPE``).
    """

    metric: str
    value: Finite | None = None
    from_step: FromStep | None = None
    tolerance: NonNegative | None = None
    tolerance_from: ToleranceFrom | None = None
    unit: Literal["fraction", "pp"] | None = None
    band: BandSpec | None = None
    kind: Literal["abs", "exact", "none", "seed_band"]

    @field_validator("value", "tolerance")
    @classmethod
    def _no_negative_zero(cls, value: float | None) -> float | None:
        return 0.0 if value == 0 else value

    @model_validator(mode="after")
    def _shape(self) -> "Expect":
        targets = (self.value is not None) + (self.from_step is not None)
        tolerances = (self.tolerance is not None) + (self.tolerance_from is not None)
        if self.kind == "seed_band":
            if self.band is None or targets or tolerances or self.unit is not None:
                raise ValueError("E_SPEC_BAND_SHAPE: kind seed_band takes a band and no value, tolerance or unit")
            return self
        if self.band is not None:
            raise ValueError("E_SPEC_BAND_SHAPE: only kind seed_band takes a band")
        if targets > 1 or tolerances > 1:
            raise ValueError("give at most one of value/from_step and one of tolerance/tolerance_from")
        if self.kind == "abs" and (targets, tolerances) != (1, 1):
            raise ValueError("kind abs needs a target and a tolerance")
        if self.kind == "exact" and (targets, tolerances) != (1, 0):
            raise ValueError("kind exact needs a target and no tolerance")
        if self.kind == "none" and tolerances:
            raise ValueError("kind none takes no tolerance")
        return self


HEADLINE_RULE_CONTEXT = "headline_rule"
"""The validation-context key that carries the profile policy's ``headline_rule`` into manifest parsing."""


def headline_rule_of(info: ValidationInfo) -> HeadlineRule:
    """The headline rule a manifest is parsed under: the context's, else ``balanced_accuracy_only``.

    Raises:
        ValueError: If the context names an unknown rule.
    """
    context = info.context if isinstance(info.context, Mapping) else {}
    rule = context.get(HEADLINE_RULE_CONTEXT, "balanced_accuracy_only")
    if rule not in ("balanced_accuracy_only", "kind_vocabulary"):
        raise ValueError(f"unknown headline rule {rule!r}")
    return cast(HeadlineRule, rule)


def _no_duplicates(value: list[str], info: ValidationInfo) -> list[str]:
    if len(set(value)) != len(value):
        raise ValueError(f"{info.field_name} has duplicates")
    return value


class SplitOptOut(_Model):
    """A declared reason a classification or detection step has no split report."""

    none: SplitOptOutReason


class MetricsSchema(_Model):
    """What a step's ``metrics.json`` may hold, within the fixed vocabulary of `metrics`.

    ``headline`` names the headline metrics the adapter emits (each must appear, and each must belong
    to ``kind``); ``secondary`` the secondary metrics (each must appear; classification only);
    ``headline_labels`` the labels that tell apart several headline or secondary entries of one name
    (e.g. systems); ``counts`` the named counts (each must appear); ``per_class`` the class labels.
    ``positive`` names the positive class of a binary classification step, which fixes the PR-AUC and
    F1 floors; a step declaring ``pr_auc`` or ``f1`` as secondary must be binary
    (``E_SPEC_SECONDARY_BINARY_ONLY``) and declare it (``E_SPEC_POSITIVE_REQUIRED``). ``split`` opts a
    classification or detection step out of the split report. Only classification and detection steps
    take ``per_class`` and ``positive``.

    A classification step declares at least two classes and, under the profile policy's
    ``headline_rule: balanced_accuracy_only``, exactly ``["balanced_accuracy"]`` as its headline
    (``E_SPEC_HEADLINE_NOT_BA`` otherwise); under ``kind_vocabulary`` any of the kind's headline or
    secondary names (none both). The rule comes from the validation context key ``headline_rule``
    (`headline_rule_of`); without one, the stricter ``balanced_accuracy_only`` applies, and
    `check_step_profile` checks the step again against its profile's rule. Accuracy, top-1, macro F1,
    AUROC, PR-AUC, MCC and F1 are ``secondary`` (``E_SPEC_SECONDARY_NOT_ALLOWED`` for anything else, and
    for any secondary metric on another kind). A detection step declares at least one class.
    """

    kind: MetricsKind
    headline: list[MetricName] = Field(min_length=1)
    secondary: list[MetricName] = []
    headline_labels: list[Label] = []
    counts: list[CountName] = []
    per_class: list[Label] = []
    positive: Label | None = None
    split: SplitOptOut | None = None

    _unique = field_validator("headline", "secondary", "headline_labels", "counts", "per_class")(_no_duplicates)

    @model_validator(mode="after")
    def _kind(self, info: ValidationInfo) -> "MetricsSchema":
        rule = headline_rule_of(info)
        problem = headline_rule_error(self.kind, list(self.headline), rule)
        if problem is not None:
            raise ValueError(problem)
        secondary: set[str] = set(self.secondary)
        not_allowed = sorted(secondary - SECONDARY_BY_KIND[self.kind])
        if not_allowed:
            raise ValueError(f"E_SPEC_SECONDARY_NOT_ALLOWED: secondary {not_allowed} not allowed for kind {self.kind}")
        overlap = sorted(secondary & set(self.headline))
        if overlap:
            raise ValueError(f"names {overlap} are both headline and secondary")
        if self.kind in CLASSIFICATION_KINDS and not self.per_class:
            raise ValueError(f"kind {self.kind} needs per_class labels")
        if self.kind == "classification":
            if len(self.per_class) < 2:
                raise ValueError("kind classification needs at least two per_class labels")
        if self.kind not in CLASSIFICATION_KINDS and (self.per_class or self.positive is not None):
            raise ValueError(f"kind {self.kind} takes no per_class labels or positive class")
        if self.positive is not None and (
            self.kind != "classification" or len(self.per_class) != 2 or self.positive not in self.per_class
        ):
            raise ValueError("positive must be one of exactly two per_class labels of a classification step")
        positive_floor = sorted(secondary & POSITIVE_FLOOR_METRICS)
        if positive_floor and len(self.per_class) != 2:
            raise ValueError(f"E_SPEC_SECONDARY_BINARY_ONLY: secondary {positive_floor} needs a binary step")
        if positive_floor and self.positive is None:
            raise ValueError(f"E_SPEC_POSITIVE_REQUIRED: secondary {positive_floor} needs a declared positive class")
        return self


_HF_REPO = r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*"
HFRepo = Annotated[str, StringConstraints(pattern=rf"^{_HF_REPO}$")]


class HFFile(_Model):
    """One file of a Hugging Face snapshot: its path inside the repository and its content sha256."""

    path: RelPath
    sha256: Sha256


class HFWeights(_Model):
    """Hugging Face weights a step loads: a repository at a 40-hex commit, and every file it reads, hashed.

    ``revision`` is a commit, never a branch or tag, so the snapshot cannot move under the step.
    """

    repo: HFRepo
    repo_type: Literal["model", "dataset"] = "model"
    revision: Commit
    files: list[HFFile] = Field(min_length=1)

    @field_validator("files")
    @classmethod
    def _unique_paths(cls, value: list[HFFile]) -> list[HFFile]:
        if len({f.path for f in value}) != len(value):
            raise ValueError("files has duplicate paths")
        return value


_MODEL_REF = re.compile(rf"^{_HF_REPO}(@[0-9a-f]{{40}})?$")
_LABEL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_+-]{0,63}$")


class ExtractSpec(_Model):
    """One categorical value a step's adapter may emit beside its metrics (``extracted`` in ``summary.json``).

    ``model_id``: one of ``allowed`` (repository ids, optionally ``@<commit>``); ``yesno``: ``yes`` or ``no``;
    ``index``: a position in ``allowed`` (labels such as run-dir names, never emitted themselves); ``series``:
    a list of at most ``max_len`` finite numbers; ``count``: a non-negative integer, suppressed below the
    policy's ``min_cell``. ``allowed`` goes with ``model_id`` and ``index`` only, ``max_len`` with ``series``
    only (``E_SPEC_EXTRACT``).
    """

    name: CountName
    kind: Literal["model_id", "yesno", "index", "series", "count"]
    allowed: list[str] = []
    max_len: Annotated[int, Field(ge=1)] | None = None

    @model_validator(mode="after")
    def _shape(self) -> "ExtractSpec":
        listed = self.kind in ("model_id", "index")
        if listed != bool(self.allowed):
            raise ValueError(f"E_SPEC_EXTRACT: {self.name!r}: allowed is required for model_id and index, else refused")
        if (self.kind == "series") != (self.max_len is not None):
            raise ValueError(f"E_SPEC_EXTRACT: {self.name!r}: max_len is required for series, else refused")
        if len(set(self.allowed)) != len(self.allowed):
            raise ValueError(f"E_SPEC_EXTRACT: {self.name!r}: allowed has duplicates")
        pattern = _MODEL_REF if self.kind == "model_id" else _LABEL
        if any(not pattern.match(a) for a in self.allowed):
            raise ValueError(f"E_SPEC_EXTRACT: {self.name!r}: an allowed entry is not a {self.kind} value")
        return self


class StepSpec(_Model):
    """One step: a repository commit, an env, an argv template, its inputs, outputs and expectations.

    ``hf_weights`` has no default: a manifest states ``[]`` for a step that loads no Hugging Face weights.
    Stage covered by inputs: every pattern of a ``stage`` entry drawn from a data root must equal a
    pattern of some ``inputs`` entry with the same root (``E_SPEC_STAGE_UNDIGESTED``), so its content is
    digested and keyed; ``checkout`` and ``up:`` stages are
    keyed by the commit, patches and upstream keys. An expectation ``unit`` is allowed only on a rate
    or ``mcc`` metric (`metrics.metric_quantity`). An ``argv`` or ``pythonpath`` entry may not be an absolute
    path without a placeholder (``E_SPEC_ABSOLUTE_PATH``): a program or module outside the checkout, venv,
    cache and roots would run unkeyed. ``lock_inputs`` names registered inputs of the manifest's sources
    lock, and every ``{input:<id>}`` placeholder must name one of them (``E_SPEC_INPUT_UNDECLARED``);
    ``extract`` declares the categorical values its adapter may emit (`ExtractSpec`). ``restricted`` has no
    default: a restricted step's results live under the profile's ``restricted_results_root``, and a step on
    a restricted input, on a data-root pattern reaching a restricted dir, or needing a restricted step must
    declare it (`restricted.check_restricted`).
    """

    name: Name
    repo: str
    replicate: Annotated[int, Field(ge=0)] = 0
    commit: Commit
    submodules: bool = True
    kind: Literal["python", "notebook", "r", "ir_notebook"]
    env: EnvSpec
    argv: list[str] = Field(min_length=1)
    cwd: str = "."
    pythonpath: list[str] = []
    inputs: list[InputSpec] = []
    stage: list[StageSpec] = []
    outputs: list[str] = []
    needs: list[str] = []
    resources: Resources
    patches: list[FileRef] = []
    adapter: str
    hf_weights: list[HFWeights]
    metrics_schema: MetricsSchema
    expect: list[Expect] = []
    extract: list[ExtractSpec] = []
    lock_inputs: list[Name] = []
    restricted: bool

    _unique_lock_inputs = field_validator("lock_inputs")(_no_duplicates)

    @model_validator(mode="after")
    def _consistent(self) -> "StepSpec":
        needs = set(self.needs)
        if len({e.name for e in self.extract}) != len(self.extract):
            raise ValueError(f"E_SPEC_EXTRACT: step {self.name!r} has duplicate extract names")
        if len(needs) != len(self.needs):
            raise ValueError(f"step {self.name!r} has duplicate needs")
        if self.name in needs:
            raise ValueError(f"step {self.name!r} needs itself")
        weights = [(w.repo_type, w.repo) for w in self.hf_weights]
        if len(set(weights)) != len(weights):
            raise ValueError(f"step {self.name!r} has duplicate hf_weights repositories")
        literal = [t for t in (*self.argv, *self.pythonpath) if t.startswith("/") and not placeholders(t)]
        if literal:
            raise ValueError(
                f"E_SPEC_ABSOLUTE_PATH: step {self.name!r} names absolute paths with no placeholder: {literal}"
            )
        templates = [*self.argv, self.cwd, *self.pythonpath, *(s.dest for s in self.stage)]
        if self.env.r_lib is not None:
            templates.append(self.env.r_lib)
        referenced = [s for t in templates for s in _upstream(t)]
        undeclared = sorted({i for t in templates for i in _registered(t)} - set(self.lock_inputs))
        if undeclared:
            raise ValueError(
                f"E_SPEC_INPUT_UNDECLARED: step {self.name!r} names inputs outside its lock_inputs: {undeclared}"
            )
        roots = [i.root for i in self.inputs] + [s.root for s in self.stage]
        referenced += [r.split(":", 1)[1] for r in roots if r.startswith("up:")]
        referenced += [e.from_step.step for e in self.expect if e.from_step is not None]
        referenced += [s for e in self.expect if e.tolerance_from is not None for s in e.tolerance_from.steps]
        referenced += [s for e in self.expect if e.band is not None for s in e.band.sources()]
        missing = sorted(set(referenced) - needs)
        if missing:
            raise ValueError(f"step {self.name!r} refers to steps outside its needs: {missing}")
        for expect in self.expect:
            code = metric_address_error(expect.metric, self.metrics_schema)
            if code is not None:
                raise ValueError(f"step {self.name!r} expect metric {expect.metric!r} is not in metrics_schema: {code}")
            if expect.unit is not None and metric_quantity(expect.metric) not in SCALED_QUANTITIES:
                raise ValueError(
                    f"E_SPEC_EXPECT_UNIT: step {self.name!r} expect metric {expect.metric!r} takes no unit"
                )
            if expect.band is not None:
                _check_band_scale(self.name, expect.metric, expect.band)
            floor_from_entry = expect.band is not None and expect.band.floor_from == "entry"
            if floor_from_entry and not (self.metrics_schema.kind in CLASSIFICATION_KINDS and "." not in expect.metric):
                raise ValueError(
                    f"E_SPEC_BAND_SHAPE: step {self.name!r} band on {expect.metric!r} has no entry floor to read"
                )
        digested = {(i.root, p) for i in self.inputs for p in i.patterns}
        for index, stage in enumerate(self.stage):
            data_root = stage.root not in RESERVED_ROOT_NAMES and not stage.root.startswith("up:")
            if data_root and any((stage.root, p) not in digested for p in stage.patterns):
                raise ValueError(
                    f"E_SPEC_STAGE_UNDIGESTED: step {self.name!r} stage[{index}] has a pattern that no inputs "
                    "entry of the same root lists"
                )
        if self.kind in ("notebook", "ir_notebook") and self.env.notebook_tools is None:
            raise ValueError(f"kind {self.kind} needs env.notebook_tools")
        if self.kind in ("r", "ir_notebook") and (
            not self.env.r_packages or self.env.r_snapshot is None or self.env.r_lib is None
        ):
            raise ValueError(f"kind {self.kind} needs env.r_packages, env.r_snapshot and env.r_lib")
        return self


def _unique_names(steps: list[StepSpec]) -> list[StepSpec]:
    dupes = sorted(name for name, n in Counter(s.name for s in steps).items() if n > 1)
    if dupes:
        raise ValueError(f"duplicate step names: {dupes}")
    return steps


SOURCES_LOCK = re.compile(r"^sources/[A-Za-z0-9_][A-Za-z0-9_.-]*\.lock\.yaml$")


def _sources_lock_path(value: str) -> str:
    if not SOURCES_LOCK.match(value) or ".." in value.split("/"):
        raise ValueError(f"a sources lock lives at sources/<name>.lock.yaml: {value!r}")
    return value


class SourcesRef(_Model):
    """The sources lock a manifest references: ``sources/<name>.lock.yaml`` in the manifest repository, and its sha."""

    lock: Annotated[str, AfterValidator(_sources_lock_path)]
    sha256: Sha256


class Manifest(_Model):
    """One analysis: a name, the sources lock its steps run from, and its steps.

    The analysis and step names name directories and appear in batch job names (``reproduce-<step>``),
    which every user of the cluster can list; class labels, count names and headline labels of each
    ``metrics_schema`` are copied into the agent-readable ``metrics.json``. None of them may carry anything
    sensitive. A ``python`` or ``notebook`` step's program must live in its venv, so a shell wrapper is a
    script run by the venv's Python (or a program in an R module or native bin dir, for an ``r`` step).
    """

    name: Name
    sources: SourcesRef
    steps: list[StepSpec] = Field(min_length=1)

    _unique = field_validator("steps")(_unique_names)


class SourceSpec(_Model):
    """A draft step for ``prefetch --sources-only``: only the code to check out."""

    name: Name
    repo: str
    commit: Commit
    submodules: bool = True


class PartitionSpec(_Model):
    """A batch partition and its limits."""

    partition: str
    gres: str | None = None
    cpus: Annotated[int, Field(ge=1)]
    mem: str
    time: str
    fallback: str | None = None


class NativeSpec(_Model):
    """A native tool the profile offers: the dirs a step that selects it gets on ``PATH`` and ``LD_LIBRARY_PATH``.

    ``bin_patterns`` and ``lib_patterns`` select the files whose content is digested into the step key
    (`cache.native_digests`), relative to each bin and lib dir: by default every top-level entry of a
    bin dir (what ``PATH`` lookup can reach) and every shared object, versioned or not, at any depth of
    a lib dir.
    """

    bin_dirs: list[Path] = []
    lib_dirs: list[Path] = []
    bin_patterns: list[str] = Field(default=["*"], min_length=1)
    lib_patterns: list[str] = Field(default=["**/*.so", "**/*.so.*"], min_length=1)


class RModule(_Model):
    """The site R module, the version it provides, and modules loaded before it (e.g. a compiler).

    ``build_modules`` are loaded, in order, before ``module`` for the R prefetch and the captured module
    env, so packages with C++ sources (e.g. lme4) build with that compiler and find its runtime libraries.
    """

    module: ModuleName
    version: str
    build_modules: list[ModuleName] = []

    @field_validator("build_modules")
    @classmethod
    def _unique_modules(cls, value: list[str]) -> list[str]:
        if len(set(value)) != len(value):
            raise ValueError("r.build_modules has duplicates")
        return value


def _norm(path: Path) -> PurePosixPath:
    text = str(path)
    return PurePosixPath(os.path.normpath("/" + text.lstrip("/") if text.startswith("/") else text))


def _strictly_under(path: PurePosixPath, root: PurePosixPath) -> bool:
    return path != root and path.is_relative_to(root)


class Policy(_Model):
    """A site's project rules, stated in its profile; the library implies none.

    Attributes:
        min_cell: The small-cell threshold: a suppressible count below it is written as ``"<min_cell>"``.
        headline_rule: Which classification headlines a manifest may declare (`metrics.headline_rule_error`).
        require_deny_patterns: The profile must name ``deny_patterns_file`` (``E_PROFILE_POLICY`` otherwise).
        require_canary: The profile must name ``canary_host`` (``E_PROFILE_POLICY`` otherwise).
        confidential_provenance: True records no paths and only the argv's sha256; False also records each
            input's paths and the rendered argv.
        umask: The umask of batch scripts and submissions; the dated log dirs get mode ``2777`` less it.
    """

    min_cell: int = Field(ge=1)
    headline_rule: HeadlineRule
    require_deny_patterns: bool
    require_canary: bool
    confidential_provenance: bool
    umask: Literal["007", "022"]


class Profile(_Model):
    """The caller's site: data roots, result roots, partitions, keyed env, R module, deny patterns, policy.

    ``modules_init`` is the environment-modules init script sourced before ``module load`` (needed by any
    sdist with ``build_modules`` and by R envs; without it those are refused with ``E_MODULES_INIT``).
    ``job_tmp_root`` is where each batch job makes its private ``TMPDIR`` (removed on exit); without it a
    job keeps the node's default temporary dir.

    The restricted fields are set together or not at all (``E_PROFILE_RESTRICTED``): ``restricted_dirs``
    are masked in the sandbox of every public step; ``restricted_results_root`` holds restricted steps'
    results and claims, lies strictly under a restricted dir and under an ``rw_under_roots`` entry;
    ``released_metrics_root`` receives released metrics, under an ``rw_under_roots`` entry and no restricted
    dir; ``scan_hook`` is the argv of the release scan (the file path is appended). No other result, run,
    log or job tmp root may overlap a restricted dir, as written or resolved.

    ``mirror_map`` maps repository URLs to local bare repositories (absolute paths) that every source fetch
    of a mapped URL uses instead of the network (`source.mirror`); its keys are stored as
    `source.canonical_repo` names, and two keys naming one repository with different paths are refused.
    Under ``mirror_map_strict`` an unmapped repository or submodule URL is refused (``mirror-unmapped``).
    """

    roots: dict[Name, Path] = Field(min_length=1)
    results_root: Path
    runs_root: Path
    slurm_logs_root: Path
    rw_under_roots: list[Path] = []
    env: dict[str, str] = {}
    partitions: dict[str, PartitionSpec] = Field(min_length=1)
    build: str
    native: dict[Name, NativeSpec] = {}
    r: RModule | None = None
    deny_patterns_file: Path | None = None
    canary_host: str | None = None
    modules_init: Path | None = None
    job_tmp_root: Path | None = None
    policy: Policy
    restricted_dirs: list[Path] = []
    restricted_results_root: Path | None = None
    released_metrics_root: Path | None = None
    scan_hook: list[str] = []
    mirror_map: dict[str, Path] = {}
    mirror_map_strict: bool = False

    _no_secrets = field_validator("env")(_reject_secret_keys)

    @field_validator("mirror_map")
    @classmethod
    def _canonical_keys(cls, value: dict[str, Path]) -> dict[str, Path]:
        from senselab.utils.reproduce.source import canonical_map

        return canonical_map(value)

    @model_validator(mode="after")
    def _consistent(self) -> "Profile":
        if self.policy.require_deny_patterns and self.deny_patterns_file is None:
            raise ValueError("E_PROFILE_POLICY: the policy requires deny_patterns_file")
        if self.policy.require_canary and self.canary_host is None:
            raise ValueError("E_PROFILE_POLICY: the policy requires canary_host")
        paths = [*self.roots.values(), self.results_root, self.runs_root, self.slurm_logs_root, *self.rw_under_roots]
        paths += [p for p in (self.modules_init, self.job_tmp_root) if p is not None]
        paths += [*self.restricted_dirs, *(p for p in (self.restricted_results_root, self.released_metrics_root) if p)]
        paths += list(self.mirror_map.values())
        native = [d for n in self.native.values() for d in (*n.bin_dirs, *n.lib_dirs)]
        relative = sorted(str(p) for p in (*paths, *native) if not p.is_absolute())
        if relative:
            raise ValueError(f"profile paths must be absolute: {relative}")
        colon = sorted(str(d) for d in native if ":" in str(d))
        if colon:
            raise ValueError(f"native dirs may not contain ':': {colon}")
        if self.build not in self.partitions:
            raise ValueError(f"build key {self.build!r} is not a partition key")
        reserved = sorted(RESERVED_ROOT_NAMES & set(self.roots))
        if reserved:
            raise ValueError(f"reserved root names: {reserved}")
        roots = [_norm(p) for p in self.roots.values()]
        rw = [_norm(p) for p in self.rw_under_roots]
        outside = sorted(str(w) for w in rw if not any(_strictly_under(w, r) for r in roots))
        if outside:
            raise ValueError(f"rw_under_roots entries must lie strictly under a data root: {outside}")
        for field in ("results_root", "runs_root", "slurm_logs_root"):
            path = _norm(getattr(self, field))
            for root in (r for r in roots if path.is_relative_to(r)):
                if not any(path.is_relative_to(w) and _strictly_under(w, root) for w in rw):
                    raise ValueError(f"{field} lies inside a data root and under no rw_under_roots entry inside it")
        self._restricted_consistent(rw)
        return self

    def _restricted_consistent(self, rw: list[PurePosixPath]) -> None:
        given = [
            self.restricted_results_root is not None,
            bool(self.restricted_dirs),
            self.released_metrics_root is not None,
            bool(self.scan_hook),
        ]
        if not any(given):
            return
        code = "E_PROFILE_RESTRICTED"
        if not all(given):
            raise ValueError(
                f"{code}: restricted_results_root, restricted_dirs, released_metrics_root and scan_hook go together"
            )
        assert self.restricted_results_root is not None and self.released_metrics_root is not None
        dirs = [_norm(d) for d in self.restricted_dirs]
        root, released = _norm(self.restricted_results_root), _norm(self.released_metrics_root)
        if not any(_strictly_under(root, d) for d in dirs):
            raise ValueError(f"{code}: restricted_results_root lies strictly under no restricted_dirs entry")
        for field, path in (("restricted_results_root", root), ("released_metrics_root", released)):
            if not any(path.is_relative_to(w) for w in rw):
                raise ValueError(f"{code}: {field} lies under no rw_under_roots entry")
        dirs += [PurePosixPath(os.path.realpath(d)) for d in self.restricted_dirs]
        public = {"released_metrics_root": self.released_metrics_root}
        public |= {f: getattr(self, f) for f in ("results_root", "runs_root", "slurm_logs_root", "job_tmp_root")}
        for field, value in public.items():
            if value is None:
                continue
            for path in (_norm(value), PurePosixPath(os.path.realpath(value))):
                if any(path.is_relative_to(d) or d.is_relative_to(path) for d in dirs):
                    raise ValueError(f"{code}: {field} overlaps a restricted_dirs entry")


def check_step_profile(step: StepSpec, profile: Profile) -> None:
    """Check what a step asks of the profile.

    Args:
        step: The step.
        profile: The caller's profile.

    Raises:
        ValueError: If the step selects a native entry the profile lacks, names a partition key it lacks, or
            declares headline names the profile's ``policy.headline_rule`` refuses.
    """
    schema = step.metrics_schema
    problem = headline_rule_error(schema.kind, list(schema.headline), profile.policy.headline_rule)
    if problem is not None:
        raise ValueError(f"step {step.name!r}: {problem}")
    missing = sorted(set(step.resources.native) - set(profile.native))
    if missing:
        raise ValueError(f"step {step.name!r} selects native entries the profile lacks: {missing}")
    if step.resources.profile not in profile.partitions:
        raise ValueError(f"step {step.name!r} names partition key {step.resources.profile!r}, not in the profile")


def _read(path: Path) -> Any:  # noqa: ANN401
    text = path.read_text()
    if path.suffix == ".json":
        return json.loads(text)
    if path.suffix in (".yaml", ".yml"):
        return yaml.safe_load(text)
    raise ValueError(f"expected a .yaml, .yml or .json file: {path}")


def parse_manifest(data: bytes, suffix: str, *, headline_rule: HeadlineRule = "balanced_accuracy_only") -> Manifest:
    """Validate a manifest from bytes already read (so they can be hashed first).

    Args:
        data: The file's bytes.
        suffix: ``.json``, ``.yaml`` or ``.yml``.
        headline_rule: The profile policy's headline rule the manifest is checked against.

    Returns:
        The validated manifest.
    """
    if suffix not in (".json", ".yaml", ".yml"):
        raise ValueError(f"expected a .yaml, .yml or .json file, not {suffix!r}")
    return Manifest.model_validate(_parse(data.decode(), suffix), context={HEADLINE_RULE_CONTEXT: headline_rule})


def load_manifest(path: str | os.PathLike[str], *, headline_rule: HeadlineRule = "balanced_accuracy_only") -> Manifest:
    """Load and validate one manifest file.

    Args:
        path: A YAML or JSON manifest.
        headline_rule: The profile policy's headline rule the manifest is checked against.

    Returns:
        The validated manifest.
    """
    return Manifest.model_validate(_read(Path(path)), context={HEADLINE_RULE_CONTEXT: headline_rule})


def load_manifest_set(
    paths: Iterable[str | os.PathLike[str]], *, headline_rule: HeadlineRule = "balanced_accuracy_only"
) -> list[StepSpec]:
    """Load several manifests as one set of steps.

    Args:
        paths: YAML or JSON manifests.
        headline_rule: The profile policy's headline rule the manifest is checked against.

    Returns:
        Every step, in file order then step order.

    Raises:
        ValueError: If a step name occurs in more than one place across the set.
    """
    return _unique_names([s for p in paths for s in load_manifest(p, headline_rule=headline_rule).steps])


def load_sources(path: str | os.PathLike[str]) -> list[SourceSpec]:
    """Load a draft manifest for ``prefetch --sources-only``.

    The file is one source mapping, a list of them, or a manifest-shaped ``{name, steps}`` whose steps
    may still lack ``env`` and every other field: only each step's ``name``, ``repo``, ``commit`` and
    ``submodules`` are read and validated.

    Args:
        path: A YAML or JSON draft manifest.

    Returns:
        The validated sources, in file order.
    """
    data = _read(Path(path))
    if isinstance(data, dict) and "steps" in data:
        steps = data["steps"]
        if not isinstance(steps, list):
            raise ValueError("a draft manifest's steps must be a list")
        fields = SourceSpec.model_fields
        data = [{k: v for k, v in step.items() if k in fields} if isinstance(step, dict) else step for step in steps]
    return [SourceSpec.model_validate(d) for d in (data if isinstance(data, list) else [data])]


def _dump(data: Any, suffix: str) -> str:  # noqa: ANN401
    if suffix == ".json":
        return json.dumps(data, indent=2) + "\n"
    return yaml.safe_dump(data, sort_keys=False)


def _parse(text: str, suffix: str) -> Any:  # noqa: ANN401
    return json.loads(text) if suffix == ".json" else yaml.safe_load(text)


def _string_changes(old: Any, new: Any, out: dict[str, str]) -> bool:  # noqa: ANN401
    """Collect ``{old: new}`` for string leaves that differ; False if anything else differs."""
    if isinstance(old, dict) and isinstance(new, dict):
        return old.keys() == new.keys() and all(_string_changes(old[k], new[k], out) for k in old)
    if isinstance(old, list) and isinstance(new, list):
        return len(old) == len(new) and all(_string_changes(a, b, out) for a, b in zip(old, new))
    if old == new and type(old) is type(new):
        return True
    if isinstance(old, str) and isinstance(new, str) and out.get(old, new) == new:
        out[old] = new
        return True
    return False


def _substituted(text: str, old: Any, new: Any, suffix: str) -> str | None:  # noqa: ANN401
    """The file text with each changed string replaced in place, when each occurs exactly once as a token."""
    changes: dict[str, str] = {}
    if not _string_changes(old, new, changes):
        return None
    for before, after in changes.items():
        token = re.compile(rf"(?<![A-Za-z0-9_]){re.escape(before)}(?![A-Za-z0-9_])")
        if not before or len(token.findall(text)) != 1 or not _PLAIN_WORD.match(after):
            return None
        text = token.sub(lambda _: after, text)
    return text if _parse(text, suffix) == new else None


_PLAIN_WORD = re.compile(r"^[A-Za-z0-9_.-]+$")


def rewrite_manifest(path: str | os.PathLike[str], edit: Callable[[Any], None]) -> bool:
    """Apply ``edit`` to a manifest file's parsed data and write the result back atomically.

    When every change replaces a plain word (a sha256, say) that occurs once in the file, the file text
    is edited in place, so comments, anchors and layout survive; otherwise the data is dumped again
    (``yaml.safe_dump`` keeps key order but drops comments and expands anchors). The result must still
    be a valid `Manifest` under the most permissive headline rule (the rule is checked where the manifest
    is loaded for a profile).

    Args:
        path: The manifest file (YAML or JSON).
        edit: Changes the parsed data in place.

    Returns:
        True if the file changed.

    Raises:
        ValueError: If the edited data is not a valid manifest.
    """
    path = Path(path)
    text = path.read_text()
    data = _parse(text, path.suffix)
    new = copy.deepcopy(data)
    edit(new)
    if new == data:
        return False
    Manifest.model_validate(new, context={HEADLINE_RULE_CONTEXT: "kind_vocabulary"})
    out = _substituted(text, data, new, path.suffix)
    if out is None:
        out = _dump(new, path.suffix)
        if _parse(out, path.suffix) != new:
            raise ValueError(f"{path.name}: the edited manifest does not round-trip")
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(out)
        os.chmod(tmp, stat.S_IMODE(path.stat().st_mode))
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return True


def load_profile(path: str | os.PathLike[str]) -> Profile:
    """Load and validate a profile file.

    Args:
        path: A YAML or JSON profile.

    Returns:
        The validated profile.
    """
    return Profile.model_validate(_read(Path(path)))
