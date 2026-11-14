"""What a measure is of, where its observations are kept, and where a unit's estimate is read from.

A `MeasureUnit` names the task, backend, parameters, model, env, service and hardware a measure holds for; its digest
is the key every measures root is looked up by. `MeasureRoots` is the layered store of
`measures@1` roots: write-once observation and record files under a trust filter by owner uid, a hardware index, and a
probe lease. `MeasureSource` reads a unit's `Estimate` from the roots (pooled); a unit no root holds is unmeasured.
"""

import contextlib
import errno
import json
import logging
import os
import re
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Final, Literal, Protocol, Self, TypeVar

from pydantic import AfterValidator, Field, PlainSerializer, ValidationError, model_validator
from senselab_worker.canonical import require_json_depth
from senselab_worker.digest import COMMIT_HEX, SHA256_HEX, identity_digest
from senselab_worker.ownership import UUID_HEX

from senselab_core.kernel._base import Frozen, FrozenMapping, NonEmptyStr, StrMap
from senselab_core.kernel.formats import (
    MEASURE_CLASS_FORMAT,
    MEASURE_CLASS_TAG,
    MEASURE_DECLARED_TAG,
    MEASURE_DRIFT_FORMAT,
    MEASURE_OBSERVATIONS_FORMAT,
    MEASURE_OOM_FORMAT,
    MEASURE_REGIME_FORMAT,
    MEASURE_SUPERSEDE_FORMAT,
    MEASURE_UNFIT_FORMAT,
    MEASURE_UNIT_FORMAT,
    MEASURE_UNIT_TAG,
    MEASURES_LAYOUT_TAG,
    FormatVerdict,
    check,
    reading,
)
from senselab_core.kernel.identity import LogicalKey, ModelIdentity, Params, ProfileSliceIdentity
from senselab_core.runtime.calibration import (
    DriftRecord,
    Estimate,
    HardwareClass,
    Observation,
    OomRecord,
    Positive,
    RegimeFactor,
    RegimeRecord,
    SupersedeRecord,
    UnfitRecord,
)
from senselab_core.runtime.clock import Clock, SystemClock
from senselab_core.runtime.config import expand
from senselab_core.runtime.execution_settings import CalibrationSettings, JobLease, ResourceClass, Trust
from senselab_core.runtime.keys import Placement
from senselab_core.runtime.locks import Lease
from senselab_core.runtime.storage import (
    Staging,
    StorageDir,
    StorePolicy,
    WrongKindError,
    committed_modes,
    shared_dir_mode,
)

logger = logging.getLogger("senselab")
T = TypeVar("T")

DRIVER_KEYS = frozenset({"torch_build", "engine", "engine_version"})
"""The driver entries a measure holds."""
_UNMEASURED_DRIVER_KEYS = frozenset({"launch_env", "served_model", "model_identity", "config_digest", "observed"})

Formation = Annotated[str, Field(pattern=r"^(mixed|single|bucket:\S+)$")]
"""How a backend forms batches: `mixed`, `single`, or `bucket:<grid id>`."""

OBSERVATION_FORMAT = MEASURE_OBSERVATIONS_FORMAT.split(".")[0]
"""The observation format an identity names, as `<name>@<major>`: an observation of another cannot be pooled."""


def measure_driver(driver: Mapping[str, Any]) -> dict[str, Any]:
    """Return the driver entries a measure holds: `driver` without the launch digest and the attestation."""
    return {k: v for k, v in driver.items() if k not in _UNMEASURED_DRIVER_KEYS}


ENV_ENTRY_KEYS = ("env_lock", "wheel_set", "platform", "numerics", "threads")
"""The fields of an env a worker's ready mark carries beside its `driver`; the decoder is the job's, not the env's."""


def env_entry(fields: Mapping[str, Any]) -> dict[str, Any]:
    """Return a ready mark's entry for an env: `fields` held to `MeasureEnv`'s filter, without the job's decoder.

    Raises:
        ValueError: a field is missing or outside `MeasureEnv`'s domain.
    """
    entry = {**{k: fields[k] for k in ENV_ENTRY_KEYS}, "driver": measure_driver(fields["driver"])}
    MeasureEnv.model_validate({**entry, "decoder": None})
    return entry


def env_identity(entry: Mapping[str, Any]) -> dict[str, Any]:
    """Return what two env entries must share for one measure to hold for both: the torch build and the rest whole."""
    driver = entry.get("driver") or {}
    return {**{k: entry.get(k) for k in ENV_ENTRY_KEYS}, "torch_build": driver.get("torch_build")}


class MeasureEnv(Frozen):
    """The env's fingerprinted context fields that decide cost: its lock and wheels, driver, platform and numerics.

    The driver holds only `DRIVER_KEYS`: the torch build (`cu128`, `cpu`) and, for a server, the engine and its version.
    `threads` are the worker launch's measured thread counts (`Launch.measured`).
    """

    env_lock: NonEmptyStr
    wheel_set: NonEmptyStr
    driver: StrMap
    platform: NonEmptyStr
    decoder: NonEmptyStr | None
    numerics: StrMap
    threads: Annotated[Mapping[str, Positive], AfterValidator(FrozenMapping), PlainSerializer(dict)]

    @model_validator(mode="after")
    def _check_driver(self) -> Self:
        if undeclared := sorted(set(self.driver) - DRIVER_KEYS):
            raise ValueError(f"driver holds {undeclared}; a measure holds only {sorted(DRIVER_KEYS)}")
        return self

    @classmethod
    def from_fields(cls, fields: Mapping[str, Any]) -> Self:
        """Return the env of `fields` as `EnvHandle.context_fields()`, the ready mark and `Launch.measured` give them.

        The driver's launch digest and attestation are dropped.

        Raises:
            ValueError: a field is missing.
        """
        if missing := sorted(set(cls.model_fields) - set(fields)):
            raise ValueError(f"env fields lack {missing}")
        driver = measure_driver(fields["driver"])
        return cls.model_validate({**{k: fields[k] for k in cls.model_fields}, "driver": driver})


LABEL_HEX: Final = 12  # invariant: 48 bits of a digest, as the store and fetch messages abbreviate digests
"""Hex digits of a digest that name a unit in text."""


class MeasureUnit(Frozen):
    """What a measure is of: the logical key without its inputs, plus env, hardware and formation.

    The input lengths, the senselab source, the regime and the run are not part of it.
    """

    task: NonEmptyStr
    task_version: NonEmptyStr
    backend: NonEmptyStr
    backend_version: NonEmptyStr
    params: Params
    backend_params: Params
    profile_slices: Annotated[
        Mapping[str, ProfileSliceIdentity], AfterValidator(FrozenMapping), PlainSerializer(dict)
    ] = FrozenMapping({})
    formation: Formation
    chunk_grid: NonEmptyStr | None
    model: ModelIdentity | None
    env: MeasureEnv
    service: NonEmptyStr | None
    hardware: HardwareClass
    observation_format: NonEmptyStr = OBSERVATION_FORMAT

    @classmethod
    def of(
        cls, placement: Placement, key: LogicalKey, env_fields: Mapping[str, Any], hardware: HardwareClass
    ) -> "MeasureUnit":
        """Return the unit of the call `key` run as `placement` in the env `env_fields` on `hardware`.

        `env_fields` holds `env_lock`, `wheel_set`, `driver` (`env.context_fields()`), `platform`, `decoder` and
        `numerics` (the worker's ready mark), and `threads` (`Launch.measured`, `{}` for a host env). The service is
        the attestation's `config_digest`.

        Raises:
            ValueError: an env field is missing, or the placement's formation is not yet decided.
        """
        if placement.batch_formation is None:
            raise ValueError("the placement's batch formation is not decided; each job's length picks it")
        return cls(
            task=key.task,
            task_version=key.task_version,
            backend=key.backend,
            backend_version=key.backend_version,
            params=key.params,
            backend_params=key.backend_params,
            profile_slices=key.profile_slices,
            formation=placement.batch_formation,
            chunk_grid=placement.chunk_grid,
            model=key.model,
            env=MeasureEnv.from_fields(env_fields),
            service=None if key.service is None else key.service.config_digest,
            hardware=hardware,
        )

    def digest(self) -> str:
        """Return `identity_digest("measure-unit@1", fields)`."""
        return identity_digest(MEASURE_UNIT_TAG, self.model_dump(mode="json"))

    def label(self) -> str:
        """Return the first `LABEL_HEX` hex digits of the digest: how a log line or `explain` names the unit."""
        return self.digest()[:LABEL_HEX]


class MeasuresRoot(Protocol):
    """A measures root, read: the observations of a unit that still pool (none of a superseded probe)."""

    def observations(self, unit: MeasureUnit) -> Sequence[Observation]:
        """Return the observations of `unit` that still pool."""


class Estimates(Protocol):
    """Where a unit's estimate is read: `MeasureSource`, or a double that gives one."""

    def get(self, unit: "MeasureUnit") -> Estimate | None:
        """Return the estimate of `unit`, None when it has none."""


class MeasureSource:
    """Reads a unit's estimate: the roots' pooled observations, estimated once."""

    def __init__(
        self,
        sources: Sequence[MeasuresRoot],
        *,
        estimate: Callable[[MeasureUnit, Sequence[Observation]], Estimate],
    ) -> None:
        """Read `sources` in order; `estimate` derives an estimate from the pooled observations."""
        self._sources = tuple(sources)
        self._estimate = estimate

    def get(self, unit: MeasureUnit) -> Estimate | None:
        """Return the estimate of `unit` over the observations of every root, None when it is not measured."""
        pooled = [o for root in self._sources for o in root.observations(unit)]
        if pooled and (found := self._estimate(unit, pooled)).status == "measured":
            return found
        return None


MEASURES_DIR = MEASURES_LAYOUT_TAG
UNIT_FILE = "unit.json"
PROBE_LEASE = "probe"
CACHE_ROOT_TOKEN = "${cache_root}"
"""Stands in a profile's `measures_roots` for the cache root (`revision.cache_root()`), so the default lives once."""
RECORD_FORMATS: Mapping[str, str] = FrozenMapping(
    {
        "drift": MEASURE_DRIFT_FORMAT,
        "supersede": MEASURE_SUPERSEDE_FORMAT,
        "regime": MEASURE_REGIME_FORMAT,
        "oom": MEASURE_OOM_FORMAT,
        "unfit": MEASURE_UNFIT_FORMAT,
    }
)
"""The record kinds below a unit, each with the format its files carry."""
RECORD_MODELS: Mapping[str, type[Frozen]] = FrozenMapping(
    {
        "drift": DriftRecord,
        "supersede": SupersedeRecord,
        "regime": RegimeFactor,
        "oom": OomRecord,
        "unfit": UnfitRecord,
    }
)
"""The type of each record kind's payload: `put_record` validates it on write."""
ESTIMATE_RECORDS = ("regime", "oom")
"""The kinds `regime_records` reads back typed: the ones an estimate is derived from."""
RecordKind = Literal["drift", "supersede", "regime", "oom", "unfit"]
_JSON_NAME = re.compile(r".+\.json")


PACKAGE_IN_CHECKOUT = Path("packages") / "senselab-core" / "src"
"""Where senselab-core lies inside its own checkout: a checkout's commit is this package's only from there."""


def _git_dir(start: Path) -> Path | None:
    """Return the git directory of the checkout whose work tree holds senselab-core and `start`; None otherwise.

    A `.git` found above `start` is another project's (a virtual environment inside its repository) unless
    `packages/senselab-core/src` lies under the same work tree and `start` lies under it.
    """
    for parent in (start, *start.parents):
        dot = parent / ".git"
        found = None
        if dot.is_dir():
            found = dot
        elif dot.is_file() and (named := dot.read_text().strip()).startswith("gitdir:"):
            found = (parent / named.removeprefix("gitdir:").strip()).resolve()
        if found is not None:
            return found if start.is_relative_to(parent / PACKAGE_IN_CHECKOUT) else None
    return None


def _ref_commit(git: Path, ref: str) -> str | None:
    """Return the commit of `ref`, loose or packed, in `git` or the common directory a worktree's names."""
    common = git / "commondir"
    base = (git / common.read_text().strip()).resolve() if common.is_file() else git
    loose = base / ref
    if loose.is_file():
        return loose.read_text().strip()
    packed = base / "packed-refs"
    lines = packed.read_text().splitlines() if packed.is_file() else []
    return next((line.split()[0] for line in lines if line.endswith(f" {ref}")), None)


def checkout_commit(root: Path | None = None) -> str | None:
    """Return the 40-hex commit `HEAD` names in the git checkout this package lives in (`root`: where it lives).

    Read from the git directory, not by running git. None unless the package lies inside the work tree of that
    checkout, so a package installed into a virtual environment inside another repository gets none of that
    repository's; and None where it cannot be read.
    """
    try:
        git = _git_dir(Path(__file__).resolve().parent if root is None else root.resolve())
        if git is None:
            return None
        head = (git / "HEAD").read_text().strip()
        found = _ref_commit(git, head.removeprefix("ref:").strip()) if head.startswith("ref:") else head
    except OSError:
        return None
    return found if found is not None and COMMIT_HEX.fullmatch(found) else None


class Build(Frozen):
    """The senselab release a record was written by: its version and, when known, its 40-hex commit."""

    version: NonEmptyStr
    commit: Annotated[str, Field(pattern=COMMIT_HEX.pattern)] | None = None


class Provenance(Frozen):
    """What every measures record carries; never an account name or a host name.

    Attributes:
        job: The scheduler job id, None off a scheduler.
        run: The run's id.
        senselab: The release, None when unknown.
        profile: The execution profile's canonical digest.
        hardware: The worker's `HardwareClass`.
    """

    job: NonEmptyStr | None
    run: NonEmptyStr
    senselab: Build | None
    profile: Annotated[str, Field(pattern=SHA256_HEX.pattern)]
    hardware: HardwareClass


class MeasureRecord(Frozen):
    """A trusted `drift`, `supersede`, `regime`, `oom` or `unfit` file as `MeasureRoots.records` reads it.

    Attributes:
        kind: The record kind.
        id: The file's id.
        uid: The owner uid of the file, read from the file system.
        written_s: The file's mtime.
        provenance: Where it came from.
        record: The kind's payload.
    """

    kind: RecordKind
    id: NonEmptyStr
    uid: Annotated[int, Field(ge=0)]
    written_s: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    provenance: Provenance
    record: Params


def declared_digest(declared: ResourceClass) -> str:
    """Return the digest of the hardware fields of `declared`: device, gpu type, memory cap, cpus and partition."""
    fields = declared.model_dump(mode="json", include={"device", "gpu_type", "memory_cap_bytes", "cpus", "partition"})
    return identity_digest(MEASURE_DECLARED_TAG, fields)


def class_digest(seen: HardwareClass) -> str:
    """Return the digest of a seen `HardwareClass`, the name of its index entry."""
    return identity_digest(MEASURE_CLASS_TAG, seen.model_dump(mode="json"))


_VARIABLE = re.compile(r"\$\{(\w+)\}|\$(\w+)")


def resolve_roots(raw: Sequence[str], cache: Path) -> tuple[Path, ...]:
    """Return the roots `raw` names, in order: `cache` for the cache-root token, then `expand`.

    A root naming a variable that is unset or empty, a root that is not absolute, a blank one and a repeat are left
    out, logged and never raised.
    """
    roots: list[Path] = []
    for entry in raw:
        text = entry.replace(CACHE_ROOT_TOKEN, str(cache))
        if empty := [n for m in _VARIABLE.finditer(text) for n in m.groups() if n and os.environ.get(n) == ""]:
            logger.info("measures root %r is not a root: %s is empty", entry, empty[0])
            continue
        try:
            path = Path(expand(text))
        except ValueError as err:
            logger.info("measures root %r is not a root: %s", entry, err)
            continue
        if not entry.strip() or path in roots:
            continue
        if not path.is_absolute():
            logger.info("measures root %r is not a root: %s is not absolute", entry, path)
            continue
        roots.append(path)
    return tuple(roots)


@dataclass(frozen=True)
class _File:
    """A file read: its parsed value, owner uid and mtime."""

    value: Any
    uid: int
    mtime: float


def _dump(document: Mapping[str, Any]) -> bytes:
    return json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


class ProbeFile(Frozen):
    """One probe's observations of a unit as stored: its id, who wrote it and what it reported."""

    probe: NonEmptyStr
    provenance: Provenance
    observations: tuple[Observation, ...]


def _parse_observations(document: Mapping[str, Any], info: os.stat_result) -> ProbeFile:
    return ProbeFile.model_validate({k: document[k] for k in ("probe", "provenance", "observations")})


def _parse_unit(document: Mapping[str, Any], info: os.stat_result) -> MeasureUnit:
    return MeasureUnit.model_validate(document["unit"])


def _parse_class(document: Mapping[str, Any], info: os.stat_result) -> HardwareClass:
    return HardwareClass.model_validate(document["class"])


def outdates(record: Mapping[str, Any]) -> tuple[str, ...]:
    """Return the probe ids a `supersede` payload names.

    Raises:
        ValueError: `outdates` is not a non-empty list of probe ids.
    """
    ids = record.get("outdates")
    if not isinstance(ids, (list, tuple)) or not ids or not all(isinstance(i, str) and i for i in ids):
        raise ValueError("a supersede record names the probe ids it outdates")
    return tuple(ids)


def _typed(record: MeasureRecord) -> RegimeRecord:
    """Return the payload of a `regime` or `oom` record as its model, read under the tag the file declared."""
    model: type[RegimeFactor] | type[OomRecord] = RegimeFactor if record.kind == "regime" else OomRecord
    written, read = RECORD_FORMATS[record.kind], record.read_under()
    with reading(written, written if read is None else read.tag):
        return model.model_validate(record.record)


def _record_parser(kind: str) -> Callable[[Mapping[str, Any], os.stat_result], MeasureRecord]:
    def parse(document: Mapping[str, Any], info: os.stat_result) -> MeasureRecord:
        if kind == "supersede":
            outdates(document["record"])
        return MeasureRecord.model_validate(
            {
                "kind": kind,
                "id": document["id"],
                "uid": info.st_uid,
                "written_s": max(info.st_mtime, 0.0),
                "provenance": document["provenance"],
                "record": document["record"],
            }
        )

    return parse


@dataclass(frozen=True)
class PeerProbing:
    """A coordinator the trust filter admits holds a live probe lease: wait for its observations."""


@dataclass(frozen=True)
class Unleased:
    """No probe lease is to be held: probe without one and do not wait.

    Attributes:
        reason: `no writable root`, or `foreign lease`: under `trust: own`, a live lease of another uid that
            cannot be taken and whose observations this reader would not pool.
    """

    reason: Literal["no writable root", "foreign lease"]


class _Skipped(Exception):  # noqa: N818 - control flow inside one class: the unit directory takes no write
    """A unit directory a write must not use."""


class MeasureRoots:
    """Layered `measures@1` roots: read pooled under a trust filter, written to the first writable root.

    Every file is write-once: staged in `tmp/<uuid>/` and linked onto a name no other writer uses. A file's writer is
    its owner uid; `trust` `own` reads only files of `uid`, `group` reads every file. An `OSError` on a root is
    logged once per run and never raised: reads go on over the other roots and a write moves to the next writable root,
    else the measure is not persisted. A file of an unknown format is skipped. A malformed or misfiled file is
    quarantined by its owner in a writable root and skipped by anyone else. `collect` removes only files `uid` owns.
    """

    def __init__(  # noqa: PLR0913 - the roots, the reader, then keyword-only staleness and clock
        self,
        roots: Sequence[Path],
        policy: StorePolicy,
        trust: Trust,
        uid: int,
        *,
        stale_after_s: float,
        lease: JobLease | None = None,
        clock: Clock = SystemClock(),  # noqa: B008 - stateless
    ) -> None:
        """Hold `roots` in order, the reader's `trust` and `uid`; probe leases go stale after `stale_after_s`.

        `lease` is the profile's lease section the probe leases are taken under (the packaged profile's when None).
        """
        self.roots = tuple(roots)
        self.policy = policy
        self.trust = trust
        self.uid = uid
        self.stale_after_s = stale_after_s
        self._lease = lease
        self._clock = clock
        self._dir_mode = shared_dir_mode(policy)
        self._file_mode = committed_modes(policy)[0]
        self._probed: dict[Path, bool] = {}
        self._said: set[tuple[Any, ...]] = set()
        self._digests: dict[MeasureUnit, str] = {}
        self._verified: set[tuple[Path, str]] = set()

    @classmethod
    def from_profile(  # noqa: PLR0913 - the calibration and policy, then keyword-only staleness, cache, reader and lease
        cls,
        calibration: CalibrationSettings,
        policy: StorePolicy,
        *,
        stale_after_s: float,
        cache: Path,
        uid: int | None = None,
        lease: JobLease | None = None,
    ) -> Self:
        """Return the roots `calibration` names, resolved against `cache`, read as `uid` (default: this process's)."""
        roots = resolve_roots(calibration.measures_roots, cache)
        return cls(
            roots,
            policy,
            calibration.trust,
            os.geteuid() if uid is None else uid,
            stale_after_s=stale_after_s,
            lease=lease,
        )

    def _say(self, key: tuple[Any, ...], message: str, *args: object) -> None:
        if key not in self._said:
            self._said.add(key)
            logger.warning(message, *args)

    def _admits(self, uid: int) -> bool:
        return self.trust == "group" or uid == self.uid

    def _in_first_writable(self, action: Callable[[Path], T]) -> T | None:
        for root in list(self._writable()):
            try:
                return action(root)
            except _Skipped:
                continue
            except (OSError, WrongKindError) as err:
                self._lost(root, err)
        return None

    def _owner(self, uid: int) -> str:
        return "you" if uid == self.uid else "another user" if self.trust == "own" else f"uid {uid}"

    def _note(self, root: Path, direction: str, err: BaseException) -> None:
        self._say((root, direction), "measures root %s cannot be %s: %s", root, direction, err)

    def _digest(self, unit: MeasureUnit) -> str:
        if unit not in self._digests:
            self._digests[unit] = unit.digest()
        return self._digests[unit]

    @contextlib.contextmanager
    def _guard(self, root: Path, direction: str) -> Iterator[None]:
        try:
            yield
        except FileNotFoundError:
            pass
        except (OSError, WrongKindError) as err:
            self._note(root, direction, err)

    def _open(self, root: Path, *, create: bool) -> StorageDir:
        with StorageDir.open_root(root, create=create, mode=self._dir_mode) as top:
            return top.child(MEASURES_DIR, create=create, mode=self._dir_mode)

    def _can_write(self, root: Path) -> bool:
        if root not in self._probed:
            try:
                with self._open(root, create=True) as m, m.walk("tmp", create=True, mode=self._dir_mode) as tmp:
                    probe = uuid.uuid4().hex
                    tmp.make_dir(probe, grammar=UUID_HEX)
                    tmp.remove_dir(probe, grammar=UUID_HEX)
                self._probed[root] = True
            except (OSError, WrongKindError) as err:
                self._note(root, "written", err)
                self._probed[root] = False
        return self._probed[root]

    def _writable(self) -> Iterator[Path]:
        for root in self.roots:
            if self._can_write(root):
                yield root

    def _lost(self, root: Path, err: BaseException) -> None:
        self._note(root, "written", err)
        self._probed[root] = False

    @property
    def write_root(self) -> Path | None:
        """The first root a probe directory can be made in, checked once; None (logged) when there is none."""
        chosen = next(self._writable(), None)
        if chosen is None:
            self._say(("none",), "no measures root is writable; measures are kept in memory only")
        elif chosen != self.roots[0]:
            self._say(("fallback", chosen), "measures root %s is not writable; writing to %s", self.roots[0], chosen)
        return chosen

    def _load(  # noqa: PLR0913 - where the file is, what it must be, how it is parsed, and whether trust filters it
        self,
        root: Path,
        directory: StorageDir,
        name: str,
        expected: str,
        parse: Callable[[Mapping[str, Any], os.stat_result], Any],
        *,
        filtered: bool = True,
    ) -> _File | None:
        info = directory.lstat(name)
        if info is None or (filtered and not self._admits(info.st_uid)):
            return None
        try:
            raw = directory.read_regular(name, limit=self.policy.measures_file_bytes_max)
        except (WrongKindError, ValueError) as err:
            self._malformed(root, directory, name, info.st_uid, type(err).__name__)
            return None
        if raw is None:
            return None
        path = directory.path / name
        try:
            require_json_depth(raw)
            document = json.loads(raw)
        except ValueError:
            document = None
        found = document.get("format") if isinstance(document, dict) else None
        verdict = check(expected, found)
        if verdict in (FormatVerdict.UNKNOWN, FormatVerdict.RETIRED):
            self._say(
                (path,),
                "measures file %s (owner %s) has a format not read here; skipped",
                path,
                self._owner(info.st_uid),
            )
            return None
        try:
            if verdict is FormatVerdict.MALFORMED:
                raise ValueError("no format tag")
            with reading(expected, found):
                return _File(parse(document, info), info.st_uid, info.st_mtime)
        except (ValueError, KeyError, TypeError, ValidationError) as err:
            self._malformed(root, directory, name, info.st_uid, type(err).__name__)
            return None

    def _aside(self, root: Path, move: Callable[[StorageDir], object]) -> bool:
        try:
            with (
                self._open(root, create=True) as m,
                m.child("quarantine", create=True, mode=self._dir_mode) as quarantine,
            ):
                move(quarantine)
        except (OSError, WrongKindError) as err:
            self._note(root, "quarantined into", err)
            return False
        return True

    def _malformed(self, root: Path, directory: StorageDir, name: str, owner: int, why: str) -> None:
        path = directory.path / name
        if owner == self.uid and self._can_write(root):
            if self._aside(root, lambda q: directory.rename(name, q, uuid.uuid4().hex, grammar=UUID_HEX)):
                self._say((path,), "malformed measures file %s quarantined (%s)", path, why)
                return
        self._say((path,), "malformed measures file %s (owner %s) skipped (%s)", path, self._owner(owner), why)

    def _misfiled(self, root: Path, parent: StorageDir, digest: str) -> None:
        path = parent.path / digest
        info = parent.lstat(digest)
        owner = self.uid if info is None else info.st_uid
        if owner == self.uid and self._can_write(root) and self._aside(root, lambda q: parent.set_aside(digest, q)):
            self._say((path,), "misfiled measures unit %s quarantined", path)
        else:
            self._say((path,), "misfiled measures unit %s (owner %s) skipped", path, self._owner(owner))

    def _unit_status(
        self, root: Path, parent: StorageDir, unit_dir: StorageDir, digest: str
    ) -> Literal["ok", "missing", "bad"]:
        if (root, digest) in self._verified:
            return "ok"
        found = self._load(root, unit_dir, UNIT_FILE, MEASURE_UNIT_FORMAT, _parse_unit, filtered=False)
        if found is None:
            return "missing" if unit_dir.kind(UNIT_FILE) is None else "bad"
        if self._digest(found.value) != digest:
            self._misfiled(root, parent, digest)
            return "bad"
        self._verified.add((root, digest))
        return "ok"

    def _each_unit(self, digest: str, visit: Callable[[Path, StorageDir], None]) -> None:
        for root in self.roots:
            with self._guard(root, "read"), self._open(root, create=False) as m, m.walk("units", digest[:2]) as parent:
                with parent.child(digest) as unit_dir:
                    if self._unit_status(root, parent, unit_dir, digest) == "ok":
                        visit(root, unit_dir)

    def _read_dir(
        self,
        root: Path,
        directory: StorageDir,
        expected: str,
        parse: Callable[[Mapping[str, Any], os.stat_result], Any],
    ) -> list[tuple[str, _File]]:
        found: list[tuple[str, _File]] = []
        for name in sorted(n for n in directory.listdir() if _JSON_NAME.fullmatch(n)):
            with self._guard(root, "read"):
                loaded = self._load(root, directory, name, expected, parse)
                if loaded is not None:
                    found.append((name, loaded))
        return found

    def _files(
        self,
        root: Path,
        unit_dir: StorageDir,
        sub: str,
        expected: str,
        parse: Callable[[Mapping[str, Any], os.stat_result], Any],
    ) -> list[tuple[str, _File]]:
        with self._guard(root, "read"), unit_dir.child(sub) as directory:
            return self._read_dir(root, directory, expected, parse)
        return []

    def _records(self, digest: str) -> list[MeasureRecord]:
        records: list[MeasureRecord] = []

        def visit(root: Path, unit_dir: StorageDir) -> None:
            for kind, expected in RECORD_FORMATS.items():
                for _, f in self._files(root, unit_dir, kind, expected, _record_parser(kind)):
                    records.append(f.value)

        self._each_unit(digest, visit)
        return sorted(records, key=lambda r: (r.written_s, r.id))

    def _outdated(self, digest: str) -> frozenset[str]:
        return frozenset(i for r in self._records(digest) if r.kind == "supersede" for i in r.record["outdates"])

    def records(self, unit: MeasureUnit) -> tuple[MeasureRecord, ...]:
        """Return the `drift`, `supersede`, `regime`, `oom` and `unfit` records of `unit` the trust admits, by age."""
        return tuple(self._records(self._digest(unit)))

    def regime_records(self, unit: MeasureUnit) -> tuple[RegimeRecord, ...]:
        """Return the typed `regime` and `oom` records of `unit` the trust admits, oldest first."""
        return tuple(_typed(r) for r in self.records(unit) if r.kind in ESTIMATE_RECORDS)

    def unfit_records(self, unit: MeasureUnit) -> tuple[UnfitRecord, ...]:
        """Return the `unfit` records of `unit` the trust admits, oldest first."""
        written = RECORD_FORMATS["unfit"]
        found = []
        for record in self.records(unit):
            if record.kind == "unfit":
                read = record.read_under()
                with reading(written, written if read is None else read.tag):
                    found.append(UnfitRecord.model_validate(record.record))
        return tuple(found)

    def remove_unfit(self, unit: MeasureUnit) -> int:
        """Remove the `unfit` records of `unit` this process's user owns, in every root; return how many.

        A root this reader cannot write, or a record another user owns, is left as it is.
        """
        removed = 0

        def visit(root: Path, unit_dir: StorageDir) -> None:
            nonlocal removed
            with self._guard(root, "written"), unit_dir.child("unfit") as directory:
                for name in directory.listdir():
                    info = directory.lstat(name)
                    if _JSON_NAME.fullmatch(name) and info and info.st_uid == self.uid:
                        removed += directory.remove_owned(name, pattern=_JSON_NAME)

        self._each_unit(self._digest(unit), visit)
        return removed

    def probes(self, unit: MeasureUnit) -> tuple[ProbeFile, ...]:
        """Return the probe files of `unit` that still pool, oldest first.

        A probe whose id a trusted `supersede` record names in `outdates` does not pool.
        """
        digest = self._digest(unit)
        outdated = self._outdated(digest)
        files: list[tuple[float, str, ProbeFile]] = []

        def visit(root: Path, unit_dir: StorageDir) -> None:
            for name, f in self._files(root, unit_dir, "obs", MEASURE_OBSERVATIONS_FORMAT, _parse_observations):
                if name.removesuffix(".json") not in outdated:
                    files.append((f.mtime, name, f.value))

        self._each_unit(digest, visit)
        return tuple(file for _, _, file in sorted(files, key=lambda f: f[:2]))

    def observations(self, unit: MeasureUnit) -> tuple[Observation, ...]:
        """Return the observations of `unit` that still pool, oldest file first (`probes`)."""
        return tuple(o for probe in self.probes(unit) for o in probe.observations)

    def units(self) -> tuple[MeasureUnit, ...]:
        """Return the units any root holds, by digest, each read from a unit file whose name is its digest."""
        found: dict[str, MeasureUnit] = {}
        for root in self.roots:
            with self._guard(root, "read"), self._open(root, create=False) as m, m.walk("units") as units:
                for prefix in sorted(units.listdir()):
                    self._read_units(root, units, prefix, found)
        return tuple(found[d] for d in sorted(found))

    def _read_units(self, root: Path, units: StorageDir, prefix: str, found: dict[str, MeasureUnit]) -> None:
        with self._guard(root, "read"), units.child(prefix) as parent:
            for digest in sorted(parent.listdir()):
                with self._guard(root, "read"), parent.child(digest) as unit_dir:
                    if self._unit_status(root, parent, unit_dir, digest) == "ok":
                        loaded = self._load(root, unit_dir, UNIT_FILE, MEASURE_UNIT_FORMAT, _parse_unit, filtered=False)
                        if loaded is not None:
                            found.setdefault(digest, loaded.value)

    def _stage(self, root: Path, target: StorageDir, name: str, data: bytes) -> None:
        with self._open(root, create=True) as m, m.walk("tmp", create=True, mode=self._dir_mode) as tmp:
            with Staging(tmp, mode=self._dir_mode) as stage:
                stage.write_new(name, data, self._file_mode, grammar=_JSON_NAME)
                with contextlib.suppress(FileExistsError):
                    stage.link(name, target, name, grammar=_JSON_NAME)

    @contextlib.contextmanager
    def _unit_dir(self, root: Path, unit: MeasureUnit, digest: str) -> Iterator[StorageDir]:
        with (
            self._open(root, create=True) as m,
            m.walk("units", digest[:2], create=True, mode=self._dir_mode) as parent,
        ):
            with parent.child(digest, create=True, mode=self._dir_mode) as unit_dir:
                status = self._unit_status(root, parent, unit_dir, digest)
                if status == "missing":
                    document = {"format": MEASURE_UNIT_FORMAT, "unit": unit.model_dump(mode="json")}
                    self._stage(root, unit_dir, UNIT_FILE, _dump(document))
                    status = self._unit_status(root, parent, unit_dir, digest)
                if status != "ok":
                    raise _Skipped
                yield unit_dir

    def _write(self, parts: tuple[str, ...], name: str, document: Mapping[str, Any], unit: MeasureUnit) -> Path | None:
        data, digest = _dump(document), self._digest(unit)

        def put(root: Path) -> Path:
            with self._unit_dir(root, unit, digest) as unit_dir:
                with unit_dir.walk(*parts, create=True, mode=self._dir_mode) as target:
                    self._stage(root, target, name, data)
                    return target.path / name

        return self._in_first_writable(put)

    def put_observations(
        self, unit: MeasureUnit, probe_id: str, obs: Sequence[Observation], provenance: Provenance
    ) -> Path | None:
        """Write one probe's observations of `unit` as `obs/<probe id>.json`; return the file, None if none persists."""
        document = {
            "format": MEASURE_OBSERVATIONS_FORMAT,
            "probe": probe_id,
            "provenance": provenance.model_dump(mode="json"),
            "observations": [o.model_dump(mode="json") for o in obs],
        }
        return self._write(("obs",), f"{probe_id}.json", document, unit)

    def put_record(
        self, unit: MeasureUnit, kind: RecordKind, record: Frozen | Mapping[str, Any], provenance: Provenance
    ) -> Path | None:
        """Write a `kind` record of `unit` as `<kind>/<id>.json`; return the file, None when none persists.

        A `supersede` payload names the probe ids it outdates (`outdates`).

        Raises:
            ValueError: `kind` is not a record kind.
            pydantic.ValidationError: a payload is not its kind's type (`RECORD_MODELS`).
        """
        if kind not in RECORD_FORMATS:
            raise ValueError(f"{kind!r} is not a measures record kind; they are {sorted(RECORD_FORMATS)}")
        payload = record.model_dump(mode="json") if isinstance(record, Frozen) else dict(record)
        payload = RECORD_MODELS[kind].model_validate(payload).model_dump(mode="json")
        record_id = uuid.uuid4().hex
        document = {
            "format": RECORD_FORMATS[kind],
            "id": record_id,
            "provenance": provenance.model_dump(mode="json"),
            "record": payload,
        }
        return self._write((kind,), f"{record_id}.json", document, unit)

    def note_seen(self, declared: ResourceClass, seen: HardwareClass) -> Path | None:
        """Record in the write root that a worker of `declared` reported `seen`; write-once. None when none persists."""
        name = f"{class_digest(seen)}.json"
        data = _dump({"format": MEASURE_CLASS_FORMAT, "class": seen.model_dump(mode="json")})

        def put(root: Path) -> Path:
            with self._open(root, create=True) as m:
                with m.walk("classes", declared_digest(declared), create=True, mode=self._dir_mode) as target:
                    self._stage(root, target, name, data)
                    return target.path / name

        return self._in_first_writable(put)

    def seen_classes(self, declared: ResourceClass) -> tuple[HardwareClass, ...]:
        """Return the hardware classes workers of `declared` have reported in any root, ordered by digest."""
        found: dict[str, HardwareClass] = {}
        for root in self.roots:
            with self._guard(root, "read"), self._open(root, create=False) as m:
                with m.walk("classes", declared_digest(declared)) as directory:
                    for name, loaded in self._read_dir(root, directory, MEASURE_CLASS_FORMAT, _parse_class):
                        if f"{class_digest(loaded.value)}.json" != name:
                            self._malformed(root, directory, name, loaded.uid, "name is not the class digest")
                        else:
                            found[name] = loaded.value
        return tuple(found[n] for n in sorted(found))

    def _lease_age(self, unit_dir: StorageDir) -> float | None:
        try:
            return Lease(
                unit_dir.path / PROBE_LEASE,
                settings=self._lease,
                stale_after=self.stale_after_s,
                directory=unit_dir,
                clock=self._clock,
            ).age()
        except PermissionError:
            pass
        except OSError as err:
            if err.errno != errno.EROFS:
                raise
        self._say(
            ("local-clock", unit_dir.path.parents[3]),
            "lease liveness in %s is judged on the local clock",
            unit_dir.path.parents[3],
        )
        newest = 0.0
        with unit_dir.child(PROBE_LEASE + ".lease") as lease:
            for token in lease.listdir():
                with lease.child(token) as held:
                    beats = (i.st_mtime for n in held.listdir() if (i := held.lstat(n)) is not None)
                    newest = max([newest, os.fstat(held.fd).st_mtime, *beats])
        return self._clock.local_now() - newest if newest else None

    def _lease_owner(self, unit_dir: StorageDir) -> int | None:
        name = PROBE_LEASE + ".lease"
        info = unit_dir.lstat(name)
        if info is None or unit_dir.kind(name) != "directory":
            return None
        age = self._lease_age(unit_dir)
        return info.st_uid if age is not None and age <= self.stale_after_s else None

    def peer_holds(self, unit: MeasureUnit) -> bool:
        """Return whether a live probe lease of `unit` is held in any root by an owner the trust filter admits."""
        owners: list[int] = []

        def visit(root: Path, unit_dir: StorageDir) -> None:
            if (owner := self._lease_owner(unit_dir)) is not None:
                owners.append(owner)

        self._each_unit(self._digest(unit), visit)
        return any(self._admits(o) for o in owners)

    def probe_lease(self, unit: MeasureUnit) -> Lease | PeerProbing | Unleased:
        """Return the probe lease of `unit` in the write root, not taken, or why none is to be held.

        `PeerProbing`: an admitted peer holds one in some root; wait for its observations up to `peer_wait_s`.
        `Unleased`: probe without a lease and do not wait. A `Lease` is taken with `with lease` (`LeaseTimeout`: a peer
        won the race; `LeaseUnavailable`: a stale lease this uid cannot break), renewed with `keep_alive` and
        `prove`d before the probe's observations are written.
        """
        if self.peer_holds(unit):
            return PeerProbing()
        digest = self._digest(unit)

        def take(root: Path) -> Lease | PeerProbing | Unleased:
            with self._unit_dir(root, unit, digest) as unit_dir:
                if (owner := self._lease_owner(unit_dir)) is not None:
                    return PeerProbing() if self._admits(owner) else Unleased("foreign lease")
                path = unit_dir.path / PROBE_LEASE
            return Lease(
                path,
                settings=self._lease,
                timeout=0,
                stale_after=self.stale_after_s,
                mode=self._dir_mode,
                clock=self._clock,
            )

        taken = self._in_first_writable(take)
        return Unleased("no writable root") if taken is None else taken

    def collect(self, kept: int) -> int:
        """Remove what `uid` owns and no reader needs; return how many.

        That is staging and quarantine older than the policy's ages on the clock's storage time, and the
        observation files of probes a trusted `supersede` record names in `outdates`, beyond the newest `kept`.
        Another user's file is never removed.
        """
        removed = 0
        ages = (("tmp", self.policy.staging_stale_s), ("quarantine", self.policy.quarantine_retention_s))
        for root in list(self._writable()):
            with self._guard(root, "written"), self._open(root, create=False) as m:
                now = self._clock.storage_now(m)
                for sub, age in ages:
                    with self._guard(root, "written"), m.child(sub) as directory:
                        removed += self._remove_owned_before(directory, now - age)
                removed += self._collect_units(root, m, kept)
        return removed

    def _remove_owned_before(self, directory: StorageDir, cutoff: float) -> int:
        removed = 0
        for name in directory.listdir():
            info = directory.lstat(name)
            if info is None or not UUID_HEX.fullmatch(name) or info.st_uid != self.uid or info.st_mtime >= cutoff:
                continue
            with contextlib.suppress(OSError):
                removed += directory.remove_owned(name, pattern=UUID_HEX)
        return removed

    def _collect_units(self, root: Path, m: StorageDir, kept: int) -> int:
        removed = 0
        with self._guard(root, "written"), m.child("units") as units:
            for prefix in units.listdir():
                with self._guard(root, "written"), units.child(prefix) as parent:
                    for digest in parent.listdir():
                        with self._guard(root, "written"), parent.child(digest) as unit_dir:
                            if self._unit_status(root, parent, unit_dir, digest) == "ok":
                                removed += self._collect_obs(unit_dir, self._outdated(digest), kept)
        return removed

    def _collect_obs(self, unit_dir: StorageDir, outdated: frozenset[str], kept: int) -> int:
        with unit_dir.child("obs") as obs:
            old = []
            for name in obs.listdir():
                info = obs.lstat(name)
                if (
                    _JSON_NAME.fullmatch(name)
                    and info
                    and info.st_uid == self.uid
                    and name.removesuffix(".json") in outdated
                ):
                    old.append((info.st_mtime, name))
            return sum(obs.remove_owned(name, pattern=_JSON_NAME) for _, name in sorted(old)[: max(len(old) - kept, 0)])
