"""Manifest and profile schema for the reproduce driver.

A **manifest** is ``{name, steps}``: the steps of one analysis, each pinned to a repository commit,
a hashed env and an argv template. A **manifest set** is the steps of several manifests taken
together; step names are unique across the set. A **profile** is the caller's site description:
data roots, result roots, partitions and the keyed environment map. A **source spec** is the
reduced step a draft manifest holds for ``prefetch --sources-only``.

Files are YAML (``.yaml``/``.yml``) or JSON (``.json``).
"""

import datetime
import json
import os
import re
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path, PurePosixPath
from typing import Annotated, Any, Literal

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
    CLASSIFICATION_HEADLINE,
    CLASSIFICATION_KINDS,
    HEADLINES_BY_KIND,
    SCALED_QUANTITIES,
    SECONDARY_BY_KIND,
    MetricName,
    MetricsKind,
    SplitOptOutReason,
    metric_address_error,
    metric_quantity,
)

Commit = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{40}$")]
Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
Name = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9_.-]+$")]
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
_KEYED_PLACEHOLDER = re.compile(r"^(root|up):([A-Za-z0-9_.-]+)$")
_SECRET_KEY = re.compile(r"(^|_)(TOKEN|KEY|SECRET|PASSWORD|PASSWD|CREDENTIALS?)S?(_|$)", re.IGNORECASE)
ALLOWED_SECRET_LIKE_KEYS = frozenset({"TOKENIZERS_PARALLELISM"})
_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def placeholders(template: str) -> list[str]:
    """Return the placeholders in a template, in order; ``{{`` and ``}}`` are literal braces.

    Args:
        template: An argv element, ``cwd``, ``pythonpath`` entry or other templated string.

    Returns:
        The text inside each single-brace pair, e.g. ``["out", "up:train"]``.

    Raises:
        ValueError: If a placeholder is outside ``{out}``, ``{tmp}``, ``{checkout}``, ``{venv}``,
            ``{cache}``, ``{root:<name>}`` and ``{up:<step>}``, or a brace is unmatched.
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
        if value is not None and not value.startswith("{cache}/"):
            raise ValueError("r_lib must start with {cache}/")
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


class Expect(_Model):
    """One expected metric.

    ``abs`` takes one target (``value`` or ``from_step``) and one tolerance (``tolerance`` or
    ``tolerance_from``); ``exact`` takes one target and no tolerance; ``none`` records the metric
    and takes no tolerance.
    """

    metric: str
    value: Annotated[float, Field(allow_inf_nan=False)] | None = None
    from_step: FromStep | None = None
    tolerance: Annotated[float, Field(ge=0, allow_inf_nan=False)] | None = None
    tolerance_from: ToleranceFrom | None = None
    unit: Literal["fraction", "pp"] | None = None
    kind: Literal["abs", "exact", "none"]

    @field_validator("value", "tolerance")
    @classmethod
    def _no_negative_zero(cls, value: float | None) -> float | None:
        return 0.0 if value == 0 else value

    @model_validator(mode="after")
    def _shape(self) -> "Expect":
        targets = (self.value is not None) + (self.from_step is not None)
        tolerances = (self.tolerance is not None) + (self.tolerance_from is not None)
        if targets > 1 or tolerances > 1:
            raise ValueError("give at most one of value/from_step and one of tolerance/tolerance_from")
        if self.kind == "abs" and (targets, tolerances) != (1, 1):
            raise ValueError("kind abs needs a target and a tolerance")
        if self.kind == "exact" and (targets, tolerances) != (1, 0):
            raise ValueError("kind exact needs a target and no tolerance")
        if self.kind == "none" and tolerances:
            raise ValueError("kind none takes no tolerance")
        return self


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
    F1 floors. ``split`` opts a classification or detection step out of the split report. Only
    classification and detection steps take ``per_class`` and ``positive``.

    A classification step declares at least two classes and exactly ``["balanced_accuracy"]`` as its
    headline (``E_SPEC_HEADLINE_NOT_BA`` otherwise); accuracy, top-1, macro F1, AUROC, PR-AUC, MCC and
    F1 are ``secondary`` (``E_SPEC_SECONDARY_NOT_ALLOWED`` for anything else, and for any secondary
    metric on another kind). A detection step declares at least one class.
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
    def _kind(self) -> "MetricsSchema":
        names: set[str] = set(self.headline)
        if self.kind == "classification" and self.headline != [CLASSIFICATION_HEADLINE]:
            raise ValueError(
                f"E_SPEC_HEADLINE_NOT_BA: a classification headline is exactly [{CLASSIFICATION_HEADLINE}]"
            )
        foreign = sorted(names - HEADLINES_BY_KIND[self.kind])
        if foreign:
            raise ValueError(f"headline names {foreign} do not belong to kind {self.kind}")
        secondary: set[str] = set(self.secondary)
        not_allowed = sorted(secondary - SECONDARY_BY_KIND[self.kind])
        if not_allowed:
            raise ValueError(f"E_SPEC_SECONDARY_NOT_ALLOWED: secondary {not_allowed} not allowed for kind {self.kind}")
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
        return self


HFRepo = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*$")]


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


class StepSpec(_Model):
    """One step: a repository commit, an env, an argv template, its inputs, outputs and expectations.

    ``hf_weights`` has no default: a manifest states ``[]`` for a step that loads no Hugging Face weights.
    Stage covered by inputs: every pattern of a ``stage`` entry drawn from a data root must equal a
    pattern of some ``inputs`` entry with the same root (``E_SPEC_STAGE_UNDIGESTED``), so its content is
    digested and keyed; ``checkout`` and ``up:`` stages are
    keyed by the commit, patches and upstream keys. An expectation ``unit`` is allowed only on a rate
    or ``mcc`` metric (`metrics.metric_quantity`).
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

    @model_validator(mode="after")
    def _consistent(self) -> "StepSpec":
        needs = set(self.needs)
        if len(needs) != len(self.needs):
            raise ValueError(f"step {self.name!r} has duplicate needs")
        if self.name in needs:
            raise ValueError(f"step {self.name!r} needs itself")
        weights = [(w.repo_type, w.repo) for w in self.hf_weights]
        if len(set(weights)) != len(weights):
            raise ValueError(f"step {self.name!r} has duplicate hf_weights repositories")
        templates = [*self.argv, self.cwd, *self.pythonpath, *(s.dest for s in self.stage)]
        if self.env.r_lib is not None:
            templates.append(self.env.r_lib)
        referenced = [s for t in templates for s in _upstream(t)]
        roots = [i.root for i in self.inputs] + [s.root for s in self.stage]
        referenced += [r.split(":", 1)[1] for r in roots if r.startswith("up:")]
        referenced += [e.from_step.step for e in self.expect if e.from_step is not None]
        referenced += [s for e in self.expect if e.tolerance_from is not None for s in e.tolerance_from.steps]
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


class Manifest(_Model):
    """One analysis: a name and its steps."""

    name: str
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


class Profile(_Model):
    """The caller's site: data roots, result roots, partitions, keyed env, R module, deny patterns."""

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
    deny_patterns_file: Path
    canary_host: str

    _no_secrets = field_validator("env")(_reject_secret_keys)

    @model_validator(mode="after")
    def _consistent(self) -> "Profile":
        paths = [*self.roots.values(), self.results_root, self.runs_root, self.slurm_logs_root, *self.rw_under_roots]
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
        return self


def check_step_profile(step: StepSpec, profile: Profile) -> None:
    """Check what a step asks of the profile.

    Args:
        step: The step.
        profile: The caller's profile.

    Raises:
        ValueError: If the step selects a native entry the profile lacks, or names a partition key it lacks.
    """
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


def load_manifest(path: str | os.PathLike[str]) -> Manifest:
    """Load and validate one manifest file.

    Args:
        path: A YAML or JSON manifest.

    Returns:
        The validated manifest.
    """
    return Manifest.model_validate(_read(Path(path)))


def load_manifest_set(paths: Iterable[str | os.PathLike[str]]) -> list[StepSpec]:
    """Load several manifests as one set of steps.

    Args:
        paths: YAML or JSON manifests.

    Returns:
        Every step, in file order then step order.

    Raises:
        ValueError: If a step name occurs in more than one place across the set.
    """
    return _unique_names([s for p in paths for s in load_manifest(p).steps])


def load_sources(path: str | os.PathLike[str]) -> list[SourceSpec]:
    """Load a draft manifest: one source mapping or a list of them.

    Args:
        path: A YAML or JSON draft manifest.

    Returns:
        The validated sources.
    """
    data = _read(Path(path))
    return [SourceSpec.model_validate(d) for d in (data if isinstance(data, list) else [data])]


def load_profile(path: str | os.PathLike[str]) -> Profile:
    """Load and validate a profile file.

    Args:
        path: A YAML or JSON profile.

    Returns:
        The validated profile.
    """
    return Profile.model_validate(_read(Path(path)))
