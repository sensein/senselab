"""Site placement: what a run needs, explicit placement values, and the site's hook that answers.

senselab holds no site scheduling policy and runs no scheduler query. A run's needs are a `PlacementRequest`
(`placement-request@1.0`); a site's `PlacementHook` answers with a `PlacementAnswer` (`placement-answer@1.0`);
`check_answer` refuses an answer that could not finish the work. The values a class is placed with come from three
layers, in order: the profile, the explicit overrides, the hook's answer (`resolve_placement`, `Placer`).
"""

import importlib
import logging
import os
import re
import threading
from collections.abc import Callable, Iterable, Mapping
from importlib import metadata
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import ConfigDict, Field, ValidationError

from senselab_core.kernel._base import Frozen, NonEmptyStr
from senselab_core.kernel.formats import PLACEMENT_ANSWER_FORMAT, PLACEMENT_REQUEST_FORMAT, FormatTag, Formatted
from senselab_core.runtime.clock import Clock, SystemClock
from senselab_core.runtime.config import expand_variables, load_yaml
from senselab_core.runtime.execution_settings import PositiveInt, PositiveSeconds, ResourceClass
from senselab_core.runtime.profiles import ExecutionProfile

logger = logging.getLogger("senselab")

Layer = Literal["profile", "override", "hook"]
"""Who set a placement value: the profile, the explicit overrides or the site's hook."""
Label = Literal["<partition>", "<node>", "<account>"]
"""What a site name is rendered as in a text other users read."""
FIELDS = ("partition", "time_s", "max_workers", "exclude")
"""The placement values a class takes, in the order the layers are recorded."""


class PlacementAnswerRefused(ValueError):
    """A placement value, an override or a hook's answer, that could not run the work: its class, field and values."""

    def __init__(self, resource_class: str, field: str, value: object, limit: object, why: str) -> None:
        """Name the class, the field, the value given and the value it was held against."""
        super().__init__(f"placement of class {resource_class!r}: {field} {value!r} is refused ({why}; {limit!r})")
        self.resource_class, self.field, self.value, self.limit = resource_class, field, value, limit


class PlacementHookFailed(RuntimeError):
    """The site's hook raised, returned something else than a `PlacementAnswer`, or did not answer in time."""


class PlacementValues(Frozen):
    """Placement values for one class; None is unset. An explicit override or a hook's answer holds these."""

    partition: NonEmptyStr | None = None
    time_s: float | None = None
    max_workers: int | None = None
    exclude: tuple[NonEmptyStr, ...] | None = None


class PlacementAnswer(Formatted):
    """A `placement-answer@1.0` document: the values per class the site chose, and why."""

    format: FormatTag = PLACEMENT_ANSWER_FORMAT
    classes: Mapping[str, PlacementValues] = Field(default_factory=dict)
    why: str = ""


class UnmeasuredFigure(Frozen):
    """A figure of a request that is not measured, and why."""

    figure: NonEmptyStr
    why: NonEmptyStr


class CurrentValues(Frozen):
    """The placement values a class has before any hook, and the layer that set each."""

    partition: NonEmptyStr | None = None
    time_s: PositiveSeconds | None = None
    max_workers: PositiveInt
    layers: Mapping[str, Layer]


class ClassNeeds(Frozen):
    """What one resource class of a run needs: its hardware, workers, time, wall time, GPU hours and current values.

    `time_floor_s` is the load's measured seconds plus the per-item cost, given when `time_per_worker_s` is not
    measured; `unmeasured` names each figure that is not measured and why.
    """

    model_config = ConfigDict(populate_by_name=True)

    resource_class: Annotated[NonEmptyStr, Field(alias="class")]
    device: Literal["cpu", "cuda", "mps"]
    gpu_type: NonEmptyStr | None = None
    device_memory_bytes: int | None = None
    cpus: int | None = None
    mem_bytes: int | None = None
    workers: Annotated[int, Field(ge=0)]
    workers_max: PositiveInt
    time_per_worker_s: float | None = None
    time_floor_s: float | None = None
    wall_s: float | None = None
    gpu_hours: float | None = None
    unmeasured: tuple[UnmeasuredFigure, ...] = ()
    current: CurrentValues


class PlacementRequest(Formatted):
    """A `placement-request@1.0` document: the needs of each resource class a run will use."""

    format: FormatTag = PLACEMENT_REQUEST_FORMAT
    classes: tuple[ClassNeeds, ...]

    def only(self, name: str) -> "PlacementRequest":
        """Return the request narrowed to class `name`; empty when the request holds no such class."""
        return self.model_copy(update={"classes": tuple(c for c in self.classes if c.resource_class == name)})

    def held(self, name: str) -> ClassNeeds | None:
        """Return class `name`'s needs, None when the request holds none."""
        return next((c for c in self.classes if c.resource_class == name), None)

    def document(self) -> dict[str, Any]:
        """Return the request as the JSON document `placement-request@1.0` describes."""
        return self.model_dump(mode="json", by_alias=True)


PlacementHook = Callable[[PlacementRequest], PlacementAnswer]
"""A site's hook: the request of one class in, the site's answer out."""

_HOOK_PATH = re.compile(r"^[\w.]+:[\w.]+$")


def load_overrides(given: Mapping[str, Any] | Path | str | None) -> dict[str, PlacementValues]:
    """Return the explicit placement values per class from a mapping or a YAML or JSON file; None gives none.

    Raises:
        ValueError: a class holds a key other than `partition`, `time_s`, `max_workers` and `exclude`, or a value of
            the wrong type.
        ConfigError: the file cannot be read.
    """
    if given is None:
        return {}
    raw = given if isinstance(given, Mapping) else load_yaml(Path(given), "placement")
    try:
        return {name: PlacementValues.model_validate(values) for name, values in raw.items()}
    except (ValidationError, AttributeError, TypeError) as err:
        raise ValueError(f"placement overrides are refused: {err}") from err


def check_values(name: str, values: PlacementValues) -> None:
    """Raise `PlacementAnswerRefused` for a time that is not positive or fewer than one worker in `values`."""
    if values.time_s is not None and values.time_s <= 0:
        raise PlacementAnswerRefused(name, "time_s", values.time_s, 0, "a time must be positive")
    if values.max_workers is not None and values.max_workers < 1:
        raise PlacementAnswerRefused(name, "max_workers", values.max_workers, 1, "at least one worker")


def check_answer(request: PlacementRequest, answer: PlacementAnswer) -> str | None:
    """Refuse `answer` unless each of its values lets the work in `request` finish; return a note on what was unchecked.

    A time must hold the longest worker (`time_per_worker_s`), else the load plus one item (`time_floor_s`), else
    it is accepted unchecked and the note says so. A worker count is at least 1 and at most `workers_max`.

    Raises:
        PlacementAnswerRefused: a field names a class the request does not hold, or a value is refused.
    """
    notes = []
    for name, values in answer.classes.items():
        needs = request.held(name)
        if needs is None:
            raise PlacementAnswerRefused(
                name,
                "class",
                name,
                sorted(c.resource_class for c in request.classes),
                "the request holds no such class",
            )
        check_values(name, values)
        if values.time_s is not None:
            need, label = (needs.time_per_worker_s, "time_per_worker_s")
            if need is None:
                need, label = needs.time_floor_s, "time_floor_s"
            if need is None:
                notes.append(f"{name}: time_s accepted unchecked, the worker time is unmeasured")
            elif values.time_s < need:
                raise PlacementAnswerRefused(
                    name, "time_s", values.time_s, need, f"below {label}: a worker could not finish"
                )
        if values.max_workers is not None and values.max_workers > needs.workers_max:
            raise PlacementAnswerRefused(
                name, "max_workers", values.max_workers, needs.workers_max, "above workers_max"
            )
    return "; ".join(notes) or None


class ClassPlacement(Frozen):
    """The values one class is placed with and the layer that set each, with the hook's verdict on its answer.

    `names` holds every site name of every layer, a replaced one too, with the label it is rendered as (`rendered`).
    """

    values: CurrentValues
    exclude: tuple[NonEmptyStr, ...] = ()
    verdict: Literal["no-hook", "accepted", "refused", "failed"] = "no-hook"
    note: str | None = None
    answer: PlacementAnswer | None = None
    names: Mapping[NonEmptyStr, Label] = Field(default_factory=dict)

    def grant(self, base: ResourceClass) -> ResourceClass:
        """Return `base` with the partition, time and worker count these values give."""
        return base.model_copy(
            update={
                "partition": self.values.partition,
                "time_s": self.values.time_s,
                "max_workers": self.values.max_workers,
            }
        )


class HookRef(Frozen):
    """A placement hook as recorded: its path, whether `load_hook` can import it, its distribution's version."""

    path: NonEmptyStr
    version: NonEmptyStr | None = None
    importable: bool = True


def hook_ref(hook: PlacementHook | str) -> HookRef:
    """Return the path of `hook` and the version of the distribution that provides its module.

    A lambda or a function defined inside another has a path no import can resolve; the reference says so.
    """
    path = hook if isinstance(hook, str) else f"{hook.__module__}:{getattr(hook, '__qualname__', type(hook).__name__)}"
    module = path.partition(":")[0].partition(".")[0]
    try:
        owner = next(iter(metadata.packages_distributions().get(module, ())), None)
        version = None if owner is None else metadata.version(owner)
    except metadata.PackageNotFoundError:
        version = None
    return HookRef(path=path, version=version, importable=_HOOK_PATH.match(path) is not None and "<" not in path)


def load_hook(path: str) -> PlacementHook:
    """Return the callable `module:callable` names.

    Raises:
        PlacementHookFailed: the path is malformed, its module does not import, or it names no callable.
    """
    if not _HOOK_PATH.match(path):
        raise PlacementHookFailed(f"placement hook {path!r} is not a `module:callable` path")
    module, _, name = path.partition(":")
    try:
        found: Any = importlib.import_module(module)
        for part in name.split("."):
            found = getattr(found, part)
    except (ImportError, AttributeError) as err:
        raise PlacementHookFailed(f"placement hook {path!r} cannot be imported: {err}") from err
    if not callable(found):
        raise PlacementHookFailed(f"placement hook {path!r} is not callable")
    return found  # type: ignore[no-any-return]


def ask(hook: PlacementHook, request: PlacementRequest, *, bound_s: float, clock: Clock) -> PlacementAnswer:
    """Call `hook(request)` on a thread, waiting up to `bound_s` seconds on `clock`.

    A hook that outlasts the bound is left running on its daemon thread; its answer is dropped.

    Raises:
        PlacementHookFailed: the hook raised (chained), returned another type, or had not answered in `bound_s`.
    """
    done = threading.Event()
    box: list[PlacementAnswer | BaseException] = []

    def run() -> None:
        try:
            box.append(hook(request))
        except BaseException as err:  # noqa: BLE001 - handed to the caller as the failure
            box.append(err)
        finally:
            done.set()

    threading.Thread(target=run, name="senselab-placement-hook", daemon=True).start()
    if not clock.wait(done, bound_s):
        raise PlacementHookFailed(f"the placement hook did not answer within {bound_s:g} s")
    [result] = box
    if isinstance(result, BaseException):
        raise PlacementHookFailed(f"the placement hook raised {type(result).__name__}: {result}") from result
    if not isinstance(result, PlacementAnswer):
        raise PlacementHookFailed(f"the placement hook returned {type(result).__name__}, not a PlacementAnswer")
    return result


def site_names(partitions: Iterable[str | None] = (), nodes: Iterable[str] = ()) -> dict[str, Label]:
    """Return each of `partitions` and `nodes`, as written and expanded, and the accounts of the environment, labelled.

    An account is the one `SBATCH_ACCOUNT` or `SLURM_ACCOUNT` holds.
    """
    names: dict[str, Label] = {
        os.environ[v]: "<account>" for v in ("SBATCH_ACCOUNT", "SLURM_ACCOUNT") if os.environ.get(v)
    }
    labelled: list[tuple[str, Label]] = [(p, "<partition>") for p in partitions if p]
    for given, label in labelled + [(n, "<node>") for n in nodes]:
        names.update({name: label for name in (given, expand_variables(given)) if name})
    return names


def _answered(answer: PlacementAnswer) -> dict[str, Label]:
    given = answer.classes.values()
    return site_names((v.partition for v in given), (n for v in given for n in v.exclude or ()))


def _layered(
    name: str, base: ResourceClass, overrides: Mapping[str, PlacementValues], profile_exclude: tuple[str, ...] = ()
) -> ClassPlacement:
    given = overrides.get(name, PlacementValues())
    check_values(name, given)
    values = {f: getattr(base, f) for f in ("partition", "time_s", "max_workers")}
    layers: dict[str, Layer] = dict.fromkeys(FIELDS, "profile")
    for field in FIELDS:
        if getattr(given, field) is not None:
            layers[field] = "override"
            if field != "exclude":
                values[field] = getattr(given, field)
    exclude = given.exclude if given.exclude is not None else profile_exclude
    names = site_names((base.partition, given.partition), (*profile_exclude, *(given.exclude or ())))
    return ClassPlacement(values=CurrentValues(**values, layers=layers), exclude=tuple(exclude), names=names)


def current_values(
    profile_classes: Mapping[str, ResourceClass], overrides: Mapping[str, PlacementValues]
) -> dict[str, CurrentValues]:
    """Return each class's values under the profile and the explicit overrides, each with the layer that set it.

    Raises:
        PlacementAnswerRefused: an override names no class of the profile, or holds a refused value.
    """
    for name in overrides:
        if name not in profile_classes:
            raise PlacementAnswerRefused(name, "class", name, sorted(profile_classes), "the profile has no such class")
    return {name: _layered(name, base, overrides).values for name, base in profile_classes.items()}


class Placer:
    """Resolves each class's placement once per epoch, calling the site's hook the first time a class is asked for.

    The hook is `hook`, else the profile's `substrate.placement_hook`; it is called with the request narrowed to the
    class, bounded by `substrate.placement_hook_s` seconds on `clock`. An answer is checked (`check_answer`); a
    failed hook falls back to the explicit and profile values and says so. Every resolution, a refused one too, is
    handed to `record` as `(epoch, class, the class's needs, the hook's reference, its placement)`. The profile's
    layer of `exclude` is `substrate.exclude_nodes`; senselab adds none.

    Args:
        profile: The execution profile: its classes, excluded nodes, hook and the hook's bound.
        overrides: The explicit values per class.
        hook: The site's hook, a callable or its `module:callable` path; None takes the profile's.
        request: Builds the run's request; called once per epoch, when the first class of it needs the hook.
        clock: Times the hook's bound.
        record: Told of each resolution; set before the first `for_class`.
        inert: Said in each recorded note when the substrate applies only the worker cap.
    """

    def __init__(  # noqa: PLR0913 - the profile and layers, then keyword-only call settings
        self,
        profile: ExecutionProfile,
        overrides: Mapping[str, PlacementValues] | None,
        hook: PlacementHook | str | None,
        *,
        request: Callable[[], PlacementRequest],
        clock: Clock = SystemClock(),  # noqa: B008 - stateless
        record: Callable[[int, str, ClassNeeds | None, HookRef | None, ClassPlacement], None] | None = None,
        inert: str | None = None,
    ) -> None:
        """Resolve placements under `profile`; nothing is called until `for_class`.

        Raises:
            PlacementAnswerRefused: an override names no class of the profile or holds a refused value.
            PlacementHookFailed: the hook's path does not import.
        """
        self.classes, self.overrides = dict(profile.resource_classes), dict(overrides or {})
        self._exclude = tuple(profile.substrate.exclude_nodes)
        current_values(self.classes, self.overrides)
        chosen = hook if hook is not None else profile.substrate.placement_hook
        self._hook = None if chosen is None else load_hook(chosen) if isinstance(chosen, str) else chosen
        self._ref = None if chosen is None else hook_ref(chosen)
        self._request = request
        self._built: dict[int, PlacementRequest] = {}
        self.hook_s, self._clock, self.record, self._inert = profile.substrate.placement_hook_s, clock, record, inert
        self._done: dict[tuple[int, str], ClassPlacement] = {}
        self._lock = threading.Lock()

    def _needs(self, name: str, epoch: int) -> PlacementRequest:
        if epoch not in self._built:
            self._built[epoch] = self._request()
        return self._built[epoch].only(name)

    def for_class(self, name: str, epoch: int) -> ClassPlacement:
        """Return class `name`'s placement for `epoch`, asking the hook the first time this epoch.

        Raises:
            PlacementAnswerRefused: the hook's answer is refused; nothing is remembered, so the next ask calls again.
            ValueError: `name` is no class of the profile.
        """
        with self._lock:
            if (epoch, name) in self._done:
                return self._done[epoch, name]
            if name not in self.classes:
                raise ValueError(f"no resource class {name!r}; the profile has {sorted(self.classes)}")
            placed, needs, refusal = _layered(name, self.classes[name], self.overrides, self._exclude), None, None
            if self._hook is not None:
                request = self._needs(name, epoch)
                needs = request.held(name)
                if needs is not None:
                    placed = placed.model_copy(
                        update={"names": {**placed.names, **site_names([needs.current.partition])}}
                    )
                placed, refusal = self._asked(self._hook, name, request, placed)
            if self._inert is not None:
                placed = placed.model_copy(update={"note": "; ".join(n for n in (placed.note, self._inert) if n)})
            if self.record is not None:
                self.record(epoch, name, needs, self._ref, placed)
            if refusal is not None:
                raise refusal
            self._done[epoch, name] = placed
            return placed

    def _asked(
        self, hook: PlacementHook, name: str, request: PlacementRequest, placed: ClassPlacement
    ) -> tuple[ClassPlacement, PlacementAnswerRefused | None]:
        try:
            answer = ask(hook, request, bound_s=self.hook_s, clock=self._clock)
        except PlacementHookFailed as err:
            logger.warning("%s; class %s keeps its explicit and profile values", err, name)
            return placed.model_copy(update={"verdict": "failed", "note": str(err)}), None
        names = {**placed.names, **_answered(answer)}
        try:
            note = check_answer(request, answer)
        except PlacementAnswerRefused as err:
            update = {"verdict": "refused", "note": str(err), "answer": answer, "names": names}
            return placed.model_copy(update=update), err
        given = answer.classes.get(name, PlacementValues())
        update = {f: getattr(given, f) for f in ("partition", "time_s", "max_workers") if getattr(given, f) is not None}
        layers = {**placed.values.layers, **{f: "hook" for f in FIELDS if getattr(given, f) is not None}}
        values = placed.values.model_copy(update={**update, "layers": layers})
        exclude = given.exclude if given.exclude is not None else placed.exclude
        accepted = ClassPlacement(
            values=values, exclude=tuple(exclude), verdict="accepted", note=note, answer=answer, names=names
        )
        return accepted, None


class ResolvedPlacement(Frozen):
    """Each class's placement for one epoch."""

    classes: Mapping[str, ClassPlacement]


def resolve_placement(
    profile: ExecutionProfile,
    overrides: Mapping[str, PlacementValues] | None,
    hook: PlacementHook | str | None,
    *,
    request: PlacementRequest,
    clock: Clock = SystemClock(),  # noqa: B008 - stateless
) -> ResolvedPlacement:
    """Return each class's placement: the profile's values, the explicit `overrides`, then `hook`'s answer.

    The classes are those `request` holds. `hook` (a callable or its path) defaults to
    the profile's `substrate.placement_hook`; it is called per class with `request` narrowed to the class.

    Raises:
        PlacementAnswerRefused: an override or the hook's answer is refused.
    """
    placer = Placer(profile, overrides, hook, request=lambda: request, clock=clock)
    return ResolvedPlacement(classes={c.resource_class: placer.for_class(c.resource_class, 0) for c in request.classes})


def rendered(text: str, names: Mapping[str, str]) -> str:
    """Return `text` as a record other users read: the account `--account` names and `names` as labels, then paths.

    The one helper of every text that may name a site: the refusal of a submission and the placement record. `names`
    maps each site name to its label (`site_names`, `ClassPlacement.names`). The operator's own text goes to the log
    (`stored_message` renders paths, user and host).
    """
    from senselab_core.runtime.execution.executor import stored_message  # noqa: PLC0415 - executor imports the planner

    shown = re.sub(r"(--account[= ])\S+", r"\1<account>", text)
    for name, label in sorted(names.items(), key=lambda n: len(n[0]), reverse=True):
        shown = re.sub(rf"(?<![\w-]){re.escape(name)}(?![\w-])", label, shown)
    if shown != text:
        logger.info("a placement text was rendered for the record; the operator's text was: %s", text)
    return stored_message(shown)
