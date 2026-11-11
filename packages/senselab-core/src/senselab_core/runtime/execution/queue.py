"""Queues of plan fragments in a shared directory: items by identity, slots in segments, oldest-first claims (ADR-003).

`QueueSet(root)` holds one queue per name: a resource class, with `#<model identity>` for a resident model
(`gpu:h100#<id>`). A queue is the directory `<root>/<identity_digest("label@1", name)>`, holding:

- `name`: the queue's name, written once.
- `items/<digest>.json`: a `plan-fragment@3.0` document as JCS, named by `identity_digest("plan-fragment@3", …)` of
  its JSON form; the first writer wins.
- `segments/<seq>/slot-<n>`: the index, a symlink to `<digest>` (never followed) made by an exclusive `symlink` at
  the next free `n`; `n` runs from 0 to `pool.segment_items - 1`, and a put that finds them all taken makes
  `segments/<seq+1>`. Order is slot order within a segment and segment order across segments. Nothing in a segment is
  written in place.
- `segments/<seq>/jobs-<n>-<count>`: the number of jobs of slot `n`'s item, written by the put that took the slot,
  so a listing of the segment gives the queue's work without reading an item.
- `segments/<seq>/res-<n>`: the mark that slot `n`'s item has an outcome, written only once `outcome/<digest>`
  exists and the slot, read again, names that digest.
- `slotted/<digest>`: a symlink to `<seq>/<n>`, made after the item's first slot; a put of a slotted item adds none.
- `segments/<seq>/retired`: the mark, made by an exclusive create, of a segment whose slots are all taken and all
  marked. A segment stays where it is once retired, so a segment number is made once.
- `claims/<digest>.lease`: the `Lease` of a claim on the item.
- `serve.lease`: on a probe queue, the `Lease` of the one worker that serves it (`Queue.serve_lease`).
- `hardware`, `env`: on a probe queue, the `HardwareClass` and the backend's env its serving worker must have,
  written once by the coordinator
  (`Queue.set_probe_unit`); a queue without it is served by no worker.
- `closed`: on a probe queue, the mark (`Queue.close_probe`) that no worker takes its serve lease or claims from it.
- `probe/<digest>`: on a probe queue, a JCS record `{"phase", "stratum"}` of the probe item (`Queue.mark_probe`).
- `outcome/<digest>`: the item's one outcome, a JCS record `{"outcome": "done" | "dead", ...}` linked from a staged
  file, so the first writer wins and an item is never both done and dead.
- `held-<epoch>-<env label>`: the mark, made by the coordinator of that epoch, that no item placed in that env is
  claimed; the env label is `identity_digest("label@1", env)`. Only a hold of the current epoch or a newer one counts.

A probe queue is named `<queue>~probe-<id>` (`probe_queue`), under the queue it measures.

A claim walks the segments from the oldest live one, lists each, and takes the lease of the first slot that is
neither marked nor under a live claim; the outcome is checked again once the lease is held. An outcome is staged,
the lease proven, and the staged record linked at `outcome/<digest>` in the one call after the proof.
"""

import contextlib
import logging
import re
import uuid
from collections.abc import Callable, Generator, Iterable, Iterator, Mapping
from pathlib import Path
from typing import Optional

from senselab_worker.canonical import jcs_dumps, jcs_loads
from senselab_worker.digest import identity_digest

from senselab_core.runtime.calibration import HardwareClass
from senselab_core.runtime.clock import Clock, SystemClock
from senselab_core.runtime.execution.epoch import Epoch
from senselab_core.runtime.execution.executor import FaultRecord, PlanFragment
from senselab_core.runtime.locks import Lease, LeaseState, LeaseTimeout
from senselab_core.runtime.profiles import ExecutionProfile
from senselab_core.runtime.storage import (
    STAGING_DIR_MODE,
    STAGING_FILE_MODE,
    TEMPORARY_NAME,
    StorageDir,
    WrongKindError,
    temporary_name,
)

logger = logging.getLogger("senselab")

RUN_QUEUES = "queues"
"""The directory of a run's queues, under its run directory."""
ITEM_TAG = "plan-fragment@3"
"""The identity tag of a queue item: the digest of a fragment's JSON form."""
LABEL_TAG = "label@1"
"""The identity tag of a queue's directory name: the digest of the queue's name."""
QUEUE_NAME = re.compile(r"^[^#\s/]+(#\S+)?$")
"""A queue name: a resource class, with `#<model identity>` for a resident model."""
ITEM_NAME = re.compile(r"^([0-9a-f]{64})\.json$")
"""An item: `<digest>.json`."""
DIGEST_NAME = re.compile(r"^[0-9a-f]{64}$")
"""An outcome, a `slotted` entry, or a slot's target: `<digest>`."""
SEQ_NAME = re.compile(r"^[0-9]{12}$")
"""A segment: `<seq>`, twelve digits."""
SLOT_NAME = re.compile(r"^slot-([0-9]+)$")
"""A slot of a segment: `slot-<n>`."""
JOBS_NAME = re.compile(r"^jobs-([0-9]+)-([1-9][0-9]*)$")
"""The job count of a slot's item: `jobs-<n>-<count>`."""
RESOLVED_NAME = re.compile(r"^res-([0-9]+)$")
"""The mark of a slot whose item has an outcome: `res-<n>`."""
RETIRED_MARK = "retired"
"""The mark of a retired segment, inside it."""
RETIRED_NAME = re.compile(r"^retired$")
"""The grammar of `RETIRED_MARK`."""
HELD_NAME = re.compile(r"^held-([1-9][0-9]*)-([0-9a-f]{64})$")
"""The mark of an env held on a queue, naming the epoch that held it and the env's label."""
NAME_FILE = re.compile(r"^name$")
"""The file holding a queue's name."""
CLOSED_FILE = re.compile(r"^closed$")
"""The mark that no worker serves a probe queue any longer."""
ENV_FILE = re.compile(r"^env$")
"""The file holding the backend and env a probe queue is served in."""
HARDWARE_FILE = re.compile(r"^hardware$")
"""The file holding the hardware class a probe queue is served on."""
PROBE_MARK = "~probe-"
"""Between a queue's name and the id of a probe, in the name of the probe's queue."""
SERVE_LEASE = "serve"
"""The resource whose lease pins a probe queue to one worker."""
DIRECTORIES = ("items", "segments", "slotted", "claims", "outcome", "probe")
"""The directories of a queue."""
OUTCOMES = ("done", "dead")
"""The outcomes an item can have."""


def probe_queue(queue: str, probe_id: str) -> str:
    """Return the name of the probe queue of `probe_id` under `queue`.

    Raises:
        ValueError: `queue` is itself a probe queue, or the result is not a queue name.
    """
    name = f"{queue}{PROBE_MARK}{probe_id}"
    if PROBE_MARK in queue or not QUEUE_NAME.fullmatch(name):
        raise ValueError(f"{name!r} is not the probe queue of {queue!r}")
    return name


def probe_of(name: str) -> tuple[str, str] | None:
    """Return `(queue, probe id)` of a probe queue's name, None for any other queue."""
    queue, found, probe_id = name.partition(PROBE_MARK)
    return (queue, probe_id) if found and probe_id else None


def item_digest(fragment: PlanFragment) -> str:
    """Return the digest that names `fragment` as a queue item: `identity_digest("plan-fragment@3", …)` of its JSON."""
    return identity_digest(ITEM_TAG, fragment.model_dump(mode="json"))


def _seq(seq: int) -> str:
    return f"{seq:012d}"


def _resolve(q: StorageDir, seq: int, n: int, digest: str) -> None:
    """Mark slot `n` of segment `seq` resolved, only when `outcome/<digest>` exists and the slot names `digest`."""
    with q.child("outcome") as outcome:
        if outcome.lstat(digest) is None:
            return
    with contextlib.suppress(FileNotFoundError), q.child("segments") as segments:
        with segments.child(_seq(seq)) as segment:
            if segment.readlink(f"slot-{n}") != digest:
                logger.warning("queue segment %s slot %d no longer names %s; not marked", _seq(seq), n, digest)
                return
            with contextlib.suppress(FileExistsError):
                segment.write_new(f"res-{n}", b"", STAGING_FILE_MODE, grammar=RESOLVED_NAME)


def _slots(segment: StorageDir) -> tuple[list[int], set[int], dict[int, int]]:
    """Return the slot numbers of a segment, in order, the resolved ones, and each recorded job count by slot."""
    slots, resolved, jobs = [], set(), {}
    for name in segment.listdir():
        if m := SLOT_NAME.fullmatch(name):
            slots.append(int(m.group(1)))
        elif m := RESOLVED_NAME.fullmatch(name):
            resolved.add(int(m.group(1)))
        elif m := JOBS_NAME.fullmatch(name):
            jobs[int(m.group(1))] = int(m.group(2))
    return sorted(slots), resolved, jobs


def _outcome_record(outcome: str, worker: str, epoch: int, **extra: object) -> dict[str, object]:
    """Return the record of `outcome/<digest>`: who wrote it, in which coordinator epoch, and `extra`."""
    return {"outcome": outcome, "worker": worker, "epoch": epoch, "finish": uuid.uuid4().hex, **extra}


def settle(lease: Lease, q: StorageDir, digest: str, record: dict[str, object]) -> Optional[bool]:
    """Write `record` as the one outcome of `digest` under `lease`: staged, then proven, then linked.

    Returns True when `outcome/<digest>` holds this record (written now, or by this caller before a lost reply),
    False when it holds another, and None when the proof failed and nothing was linked.
    """
    data = jcs_dumps(record)
    with q.child("outcome") as outcome:
        temporary = temporary_name(digest)
        outcome.write_new(temporary, data, STAGING_FILE_MODE, grammar=TEMPORARY_NAME)
        try:
            if not lease.prove():
                return None
            try:
                outcome.link(temporary, outcome, digest, grammar=DIGEST_NAME)
            except FileExistsError:
                return outcome.read_regular(digest) == data
            return True
        finally:
            outcome.remove_owned(temporary, pattern=TEMPORARY_NAME)


class _Bound:
    """The size of a claim: at most `items` items and, when `jobs` is set, at most `jobs` jobs beyond its first item.

    With `singles`, only one-job items join a one-job first item.
    """

    def __init__(self, items: Optional[int], jobs: Optional[int], singles: bool = False) -> None:
        self.items = items if items is not None else (1 if jobs is None else None)
        self.jobs = jobs
        self.singles = singles

    def fits(self, taken: list["_Held"], count: Optional[int]) -> bool:
        """Whether an item of `count` jobs (None when unknown) may join `taken`."""
        if not taken:
            return True
        if self.singles and (count != 1 or len(taken[0].fragment.jobs) != 1):
            return False
        if self.jobs is None or count is None:
            return True
        return sum(len(h.fragment.jobs) for h in taken) + count <= self.jobs

    def full(self, taken: list["_Held"]) -> bool:
        """Whether `taken` fills the claim."""
        jobs = sum(len(h.fragment.jobs) for h in taken)
        return (self.items is not None and len(taken) >= self.items) or (self.jobs is not None and jobs >= self.jobs)


class _Held:
    """One item a claim holds: its digest, slot, fragment, lease and the stack that releases it."""

    def __init__(  # noqa: PLR0913 - the parts of one held item
        self,
        digest: str,
        slot: tuple[int, int],
        fragment: PlanFragment,
        lease: Lease,
        state: LeaseState,
        stack: contextlib.ExitStack,
    ) -> None:
        self.digest, self.slot, self.fragment = digest, slot, fragment
        self.lease, self.state, self.stack = lease, state, stack


class Claim:
    """Items one worker claimed from a queue, each under its own beating `Lease` until it is finished.

    `outcomes` records, per digest, what became of each item this claim gave up: `committed`, `dead`, `duplicate`
    (another claim's outcome was written first), `released` or `lost` (the lease failed its proof; nothing written).
    An outcome record names `epoch`, the coordinator epoch current when the claim was taken.
    """

    def __init__(self, queue: "Queue", worker: str, held: list[_Held], epoch: int) -> None:
        """Hold `held` for `worker`, taken in coordinator epoch `epoch`; use `Queue.claim`."""
        self.queue = queue
        self.worker = worker
        self.epoch = epoch
        self._held = {h.digest: h for h in held}
        self.outcomes: dict[str, str] = {}

    @property
    def items(self) -> tuple[str, ...]:
        """The digests still held, in claim order."""
        return tuple(self._held)

    @property
    def fragments(self) -> tuple[PlanFragment, ...]:
        """The fragments still held, in claim order."""
        return tuple(h.fragment for h in self._held.values())

    def slot(self, digest: str) -> tuple[int, int]:
        """Return the `(segment, slot)` a held item was claimed from."""
        return self._held[digest].slot

    def broke(self, digest: str) -> int:
        """Return how many stale claims on a held item its lease broke when it was taken."""
        return len(self._held[digest].lease.broke)

    def _chosen(self, items: Optional[Iterable[str]]) -> list[_Held]:
        if items is None:
            return list(self._held.values())
        chosen = list(items)
        if missing := [d for d in chosen if d not in self._held]:
            raise KeyError(f"this claim does not hold {missing}")
        return [self._held[d] for d in chosen]

    def _end(self, held: _Held, outcome: str) -> None:
        del self._held[held.digest]
        self.outcomes[held.digest] = outcome
        held.stack.close()

    def renew(self) -> bool:
        """Prove every held lease; an item whose lease was lost is given up as `lost`. Return whether all were held."""
        lost = [h for h in self._held.values() if h.state.lost or not h.lease.prove()]
        for held in lost:
            self._end(held, "lost")
        return not lost

    def _finish(self, held: _Held, outcome: str, extra: dict[str, object]) -> str:
        """Write `outcome` for `held` (`settle`), mark its slot and release it; return what became of it."""
        record = _outcome_record(outcome, self.worker, self.epoch, **extra)
        written = None
        if not held.state.lost:
            with self.queue.open() as q:
                written = settle(held.lease, q, held.digest, record)
                if written is not None:
                    _resolve(q, *held.slot, held.digest)
        result = (
            "lost" if written is None else ("committed" if outcome == "done" else "dead") if written else "duplicate"
        )
        self._end(held, result)
        return result

    def commit(
        self, items: Optional[Iterable[str]] = None, extra: Optional[dict[str, object]] = None
    ) -> tuple[str, ...]:
        """Commit the held `items` (all by default) as `done`, with the fields of `extra`; return those committed."""
        outcomes = {h.digest: self._finish(h, "done", dict(extra or {})) for h in self._chosen(items)}
        return tuple(d for d, o in outcomes.items() if o == "committed")

    def dead(self, fault: FaultRecord, items: Optional[Iterable[str]] = None) -> tuple[str, ...]:
        """Give the held `items` (all by default) the outcome `dead` with `fault`; return the digests this call did."""
        extra: dict[str, object] = {"fault": fault.model_dump(mode="json")}
        outcomes = {h.digest: self._finish(h, "dead", extra) for h in self._chosen(items)}
        return tuple(d for d, o in outcomes.items() if o == "dead")

    def release(self, items: Optional[Iterable[str]] = None) -> None:
        """Give the held `items` (all by default) back: their leases are released and they are ready again."""
        for held in self._chosen(items):
            self._end(held, "released")

    def __enter__(self) -> "Claim":
        """Return this claim."""
        return self

    def __exit__(self, *exc: object) -> None:
        """Release every item still held."""
        self.release()


class ProbeQueueError(ValueError):
    """A production put on a probe queue, or a probe put on another."""


class UnmarkedProbeItem(LookupError):
    """An item of a probe queue has no phase and stratum: it was not put with `put_probe`."""


class ServeLease:
    """The lease pinning a probe queue to the worker that holds it; `held` says whether its beats still prove it."""

    def __init__(self, stack: contextlib.ExitStack, state: LeaseState) -> None:
        """Hold what `stack` holds, its beats recorded in `state`; use `Queue.serve_lease`."""
        self._stack, self._state = stack, state

    @property
    def held(self) -> bool:
        """Whether the lease has not been lost."""
        return not self._state.lost

    def release(self) -> None:
        """Give the lease up."""
        self._stack.close()


class Queue:
    """One queue of plan fragments under a `QueueSet`.

    Attributes:
        name: The queue's name.
        path: The queue's directory.
        faults: Each item a claim found missing, with its infrastructure fault; the item stays ready.
        backoffs: How many times a claim found an item under another live claim and moved on.
    """

    def __init__(  # noqa: PLR0913 - the queue's place, name and profile, then the epoch and the clock
        self,
        root: Path,
        name: str,
        profile: ExecutionProfile,
        epoch: Optional[Callable[[], int]] = None,
        clock: Clock = SystemClock(),  # noqa: B008 - stateless
    ) -> None:
        """Open, creating it if missing, the queue `name` under `root`; use `QueueSet.queue`.

        `epoch` gives the current coordinator epoch, below which a hold does not count; every hold counts when None.
        Every lease, the storage clock and the claim backoffs are read on `clock`.

        Raises:
            ValueError: `name` is not a resource class with an optional `#<model identity>`.
        """
        if not isinstance(name, str) or not QUEUE_NAME.fullmatch(name):
            raise ValueError(f"{name!r} is not a queue name: a resource class, with #<model identity> for a model")
        self.name = name
        self.path = Path(root) / identity_digest(LABEL_TAG, name)
        self.faults: list[tuple[str, FaultRecord]] = []
        self.backoffs = 0
        self._segment_items = profile.pool.segment_items
        self._stale_after = profile.lease.stale_after_s
        self._renew_fraction = profile.lease.renew_fraction
        self._judged: dict[str, float] = {}
        self._missing: dict[str, float] = {}
        self._targets: dict[tuple[int, int], str] = {}
        self._envs: dict[str, str] = {}
        self._jobs: dict[str, int] = {}
        self._epoch, self._clock = epoch, clock
        self._head = 0
        self._oldest = 0
        with StorageDir.open_root(self.path, create=True, mode=STAGING_DIR_MODE) as q:
            for directory in DIRECTORIES:
                q.child(directory, create=True, mode=STAGING_DIR_MODE).close()
            if q.lstat("name") is None:
                q.publish("name", name.encode(), grammar=NAME_FILE, mode=STAGING_FILE_MODE)

    @contextlib.contextmanager
    def open(self) -> Iterator[StorageDir]:
        """Yield the queue's directory, open."""
        with StorageDir.open_root(self.path) as q:
            yield q

    def outcome(self, digest: str) -> Optional[dict[str, object]]:
        """Return the outcome record of `digest`, None while it has none."""
        with self.open() as q, q.child("outcome") as outcome:
            data = outcome.read_regular(digest)
        record = jcs_loads(data) if data is not None else None
        return record if isinstance(record, dict) else None

    def serve_lease(self) -> Optional[ServeLease]:
        """Take the lease that pins this queue to one worker, beating until released; None while another's is live."""
        stack = contextlib.ExitStack()
        lease = Lease(self.path / SERVE_LEASE, timeout=0, stale_after=self._stale_after, clock=self._clock)
        try:
            stack.enter_context(lease)
        except LeaseTimeout:
            return None
        try:
            state = stack.enter_context(lease.keep_alive(self._renew_fraction))
        except BaseException:
            stack.close()
            raise
        return ServeLease(stack, state)

    def set_probe_unit(self, hardware: HardwareClass, backend: str, env: Mapping[str, object]) -> None:
        """Record the class and the `backend`'s env a worker must have to serve this probe queue; the first stays.

        Raises:
            ProbeQueueError: this is not a probe queue.
        """
        if probe_of(self.name) is None:
            raise ProbeQueueError(f"{self.name!r} is not a probe queue")
        with self.open() as q:
            q.publish(
                "env", jcs_dumps({"backend": backend, "env": dict(env)}), grammar=ENV_FILE, mode=STAGING_FILE_MODE
            )
            q.publish(
                "hardware", jcs_dumps(hardware.model_dump(mode="json")), grammar=HARDWARE_FILE, mode=STAGING_FILE_MODE
            )

    def probe_env(self) -> tuple[str, dict[str, object]] | None:
        """Return the backend and env set by `set_probe_unit`; None while none is recorded or it cannot be read."""
        with self.open() as q:
            data = q.read_regular("env")
        try:
            found = None if data is None else jcs_loads(data)
            return None if not isinstance(found, dict) else (str(found["backend"]), dict(found["env"]))
        except (ValueError, KeyError, TypeError):
            return None

    def close_probe(self) -> None:
        """Mark this probe queue closed: no worker serves it from now on. Write-once.

        Raises:
            ProbeQueueError: this is not a probe queue.
        """
        if probe_of(self.name) is None:
            raise ProbeQueueError(f"{self.name!r} is not a probe queue")
        with self.open() as q:
            q.publish("closed", b"1", grammar=CLOSED_FILE, mode=STAGING_FILE_MODE)

    def probe_closed(self) -> bool:
        """Return whether `close_probe` was called on this queue."""
        with self.open() as q:
            return q.lstat("closed") is not None

    def probe_hardware(self) -> HardwareClass | None:
        """Return the class set by `set_probe_unit`; None while none is recorded or it cannot be read."""
        with self.open() as q:
            data = q.read_regular("hardware")
        try:
            return None if data is None else HardwareClass.model_validate(jcs_loads(data))
        except ValueError:
            return None

    def put_probe(self, fragment: PlanFragment, phase: str, stratum: int) -> None:
        """Put a probe item: its mark (`phase`, length `stratum`) is written before its slot, the first mark staying.

        Raises:
            ProbeQueueError: this is not a probe queue.
        """
        if probe_of(self.name) is None:
            raise ProbeQueueError(f"{self.name!r} is not a probe queue; put production items with `put`")
        digest = item_digest(fragment)
        with self.open() as q, q.child("probe") as probe:
            probe.publish(
                digest, jcs_dumps({"phase": phase, "stratum": stratum}), grammar=DIGEST_NAME, mode=STAGING_FILE_MODE
            )
        self._put_all([fragment])

    def probe_mark(self, digest: str) -> tuple[str, int]:
        """Return the `(phase, stratum)` item `digest` was put with.

        Raises:
            UnmarkedProbeItem: the item has no readable mark.
        """
        with self.open() as q, q.child("probe") as probe:
            data = probe.read_regular(digest)
        record = jcs_loads(data) if data is not None else None
        phase, stratum = (record.get("phase"), record.get("stratum")) if isinstance(record, dict) else (None, None)
        if not (isinstance(phase, str) and isinstance(stratum, int)):
            raise UnmarkedProbeItem(f"item {digest} of probe queue {self.name!r} has no phase and stratum")
        return phase, stratum

    def put(self, fragment: PlanFragment) -> None:
        """Write `fragment`'s item and give it a slot, unless it has a slot or an outcome.

        A put is idempotent, so putting a run's unfinished fragments again repairs a put that was interrupted. A
        fragment with an outcome is not queued again; a retry is a fragment of a later `attempt`, another digest.

        Raises:
            ProbeQueueError: this is a probe queue (`put_probe`).
        """
        self.put_all([fragment])

    def put_all(self, fragments: Iterable[PlanFragment]) -> int:
        """Put each fragment in order (`put`); return how many slots were taken.

        Raises:
            TypeError: an element is not a `PlanFragment`.
            ProbeQueueError: this is a probe queue (`put_probe`).
        """
        if probe_of(self.name) is not None:
            raise ProbeQueueError(f"{self.name!r} is a probe queue; put its items with `put_probe`")
        return self._put_all(fragments)

    def _put_all(self, fragments: Iterable[PlanFragment]) -> int:
        digests: dict[str, Optional[int]] = {}
        with self.open() as q, q.child("items") as items:
            for fragment in fragments:
                if not isinstance(fragment, PlanFragment):
                    raise TypeError(f"a queue item is a PlanFragment, not {type(fragment).__name__}")
                digest = item_digest(fragment)
                if items.lstat(f"{digest}.json") is None:
                    items.publish(
                        f"{digest}.json",
                        jcs_dumps(fragment.model_dump(mode="json")),
                        grammar=ITEM_NAME,
                        mode=STAGING_FILE_MODE,
                    )
                digests[digest] = len(fragment.jobs)
            return self._index(q, digests)

    def _index(self, q: StorageDir, digests: Mapping[str, Optional[int]]) -> int:
        """Give each of `digests` that has neither a slot nor an outcome a slot, in order; return how many.

        Each maps to its item's job count, read from the item when None.
        """
        taken = 0
        with (
            q.child("outcome") as outcome,
            q.child("slotted") as slotted,
            q.child("segments") as segments,
            q.child("items") as items,
        ):
            for digest, count in digests.items():
                if slotted.lstat(digest) is not None or outcome.lstat(digest) is not None:
                    continue
                jobs = count if count is not None else self._job_count(items, digest)
                seq, n = self._take_slot(segments, digest, jobs)
                with contextlib.suppress(FileExistsError):
                    slotted.symlink(f"{_seq(seq)}/{n}", digest, grammar=DIGEST_NAME)
                taken += 1
        return taken

    def _take_slot(self, segments: StorageDir, digest: str, jobs: int) -> tuple[int, int]:
        """Make the next free slot of the head segment point at `digest`, rolling to a new segment when it is full.

        The slot's job count `jobs` is recorded beside it once the slot is taken.

        A segment's directory is made by `mkdir`, which the storage refuses for a number made before, so a segment
        number, retired or not, is opened and never made again.
        """
        seq = max(self._head, self._last(segments))
        while True:
            segment = segments.child(_seq(seq), create=True, mode=STAGING_DIR_MODE)
            with segment:
                slots, _, _ = _slots(segment)
                n = slots[-1] + 1 if slots else 0
                while n < self._segment_items:
                    try:
                        segment.symlink(digest, f"slot-{n}", grammar=SLOT_NAME)
                    except FileExistsError:
                        n += 1
                        continue
                    with contextlib.suppress(FileExistsError):
                        segment.write_new(f"jobs-{n}-{jobs}", b"", STAGING_FILE_MODE, grammar=JOBS_NAME)
                    self._head = seq
                    return seq, n
            seq += 1

    def _last(self, segments: StorageDir) -> int:
        """Return the highest segment number this queue knows of, listing `segments/` when it knows none."""
        if self._head:
            return self._head
        return max((int(n) for n in segments.listdir() if SEQ_NAME.fullmatch(n)), default=0)

    def reconcile(self) -> int:
        """Put again every item that has neither a slot nor an outcome; return how many slots were taken.

        It lists every item and looks each up, so it costs O(items), once per run start. Repaired items take slots in
        digest order, behind the work already queued.
        """
        with self.open() as q, q.child("items") as items:
            digests = sorted(m.group(1) for n in items.listdir() if (m := ITEM_NAME.fullmatch(n)))
            return self._index(q, dict.fromkeys(digests))

    def _current(self) -> int:
        """Return the current coordinator epoch, 0 when the queue has no epoch source."""
        return self._epoch() if self._epoch is not None else 0

    def hold(self, epoch: int, env: str) -> None:
        """Hold `env` on this queue for the coordinator of `epoch`: no item placed in it is claimed while it counts."""
        with self.open() as q:
            name = f"held-{epoch}-{identity_digest(LABEL_TAG, env)}"
            q.make_dir(name, grammar=HELD_NAME, mode=STAGING_DIR_MODE, exist_ok=True)

    def _holds(self) -> dict[str, int]:
        """Return each env label held by the current epoch or a newer one, with the newest such epoch."""
        current = self._current()
        found: dict[str, int] = {}
        with self.open() as q:
            for name in q.listdir():
                if (m := HELD_NAME.fullmatch(name)) and int(m.group(1)) >= current:
                    found[m.group(2)] = max(found.get(m.group(2), 0), int(m.group(1)))
        return found

    def held(self, env: str) -> Optional[int]:
        """Return the newest counting epoch that holds `env` on this queue, None when it is not held."""
        return self._holds().get(identity_digest(LABEL_TAG, env))

    def unhold(self, before: int) -> None:
        """Remove every hold made by an epoch older than `before`."""
        with self.open() as q:
            for name in q.listdir():
                if (m := HELD_NAME.fullmatch(name)) and int(m.group(1)) < before:
                    with contextlib.suppress(FileNotFoundError):
                        q.remove_dir(name, grammar=HELD_NAME)

    def claimed_any(self) -> bool:
        """Return whether any item of this queue is under a claim stamped within the stale bound."""
        with self.open() as q, q.child("claims") as claims:
            now = self._clock.storage_now(q)
            leased = [n.removesuffix(".lease") for n in claims.listdir() if n.endswith(".lease")]
            return any(self._live(claims, digest, now) for digest in leased)

    def claimed(self, digest: str) -> bool:
        """Return whether `digest` is under a claim stamped within the stale bound, on the storage clock."""
        with self.open() as q, q.child("claims") as claims:
            return self._live(claims, digest, self._clock.storage_now(q))

    def claim(
        self, worker: str, max_items: Optional[int] = None, max_jobs: Optional[int] = None, *, singles: bool = False
    ) -> Optional[Claim]:
        """Claim up to `max_items` items without an outcome, oldest first, for `worker`; None when none is claimable.

        `max_items` is 1 when neither bound is given and unbounded when only `max_jobs` is. With `max_jobs`, an item
        whose recorded job count would take the claim past it is passed over, except as the claim's first item. With
        `singles`, items join a claim only while it and they hold one job each.

        Each item is held under a `Lease` beating from its own renewer until the claim commits, kills or releases
        it. An item under a live claim is skipped; a stale claim is broken, within one renewal interval of going
        stale, and its item claimed again. A missing item is left ready and recorded in `faults`. An item whose env is
        held (`hold`) is skipped.

        Raises:
            ValueError: `max_items` or `max_jobs` is below 1.
        """
        if (max_items is not None and max_items < 1) or (max_jobs is not None and max_jobs < 1):
            raise ValueError(f"a claim takes at least one item of one job (got {max_items}, {max_jobs})")
        skip = set(self._holds())
        taken: list[_Held] = []
        try:
            with self.open() as q, q.child("segments") as segments, q.child("claims") as claims:
                self._fill(q, segments, claims, taken, _Bound(max_items, max_jobs, singles), skip)
        except BaseException:
            for held in taken:
                held.stack.close()
            raise
        return Claim(self, worker, taken, self._current()) if taken else None

    def _segments(
        self, segments: StorageDir
    ) -> Generator[tuple[int, StorageDir, list[int], set[int], dict[int, int]], None, None]:
        """Yield each live segment, open, oldest first from the oldest this queue has seen live; retire full ones.

        A segment is retired by its `retired` mark, made in place once all its slots are taken and marked.
        """
        seq, prefix = self._oldest, True
        while True:
            try:
                segment = segments.child(_seq(seq))
            except FileNotFoundError:
                return
            with segment:
                live = segment.lstat(RETIRED_MARK) is None
                if live:
                    slots, resolved, jobs = _slots(segment)
                    live = len(slots) < self._segment_items or not resolved.issuperset(slots)
                    if not live:
                        with contextlib.suppress(FileExistsError):
                            segment.write_new(RETIRED_MARK, b"", STAGING_FILE_MODE, grammar=RETIRED_NAME)
                if live:
                    prefix = False
                    yield seq, segment, slots, resolved, jobs
                elif prefix:
                    self._oldest = seq + 1
                    self._targets = {k: d for k, d in self._targets.items() if k[0] > seq}
            seq += 1

    def _fill(  # noqa: PLR0913 - the queue's open directories, the items taken and the bound
        self,
        q: StorageDir,
        segments: StorageDir,
        claims: StorageDir,
        taken: list[_Held],
        bound: _Bound,
        skip: set[str],
    ) -> None:
        """Take claims into `taken`, oldest first, until `bound` is full or no live segment has an item that fits."""
        leased = {n.removesuffix(".lease") for n in claims.listdir() if n.endswith(".lease")}
        now = self._clock.now()
        self._judged = {d: at for d, at in self._judged.items() if d in leased and now - at < self._interval}
        self._missing = {d: at for d, at in self._missing.items() if now - at < self._interval}
        walk = self._segments(segments)
        with contextlib.closing(walk):
            self._fill_from(q, claims, walk, taken, bound, skip)

    def _fill_from(  # noqa: PLR0913 - as `_fill`, with the walk of live segments
        self,
        q: StorageDir,
        claims: StorageDir,
        walk: Iterator[tuple[int, StorageDir, list[int], set[int], dict[int, int]]],
        taken: list[_Held],
        bound: _Bound,
        skip: set[str],
    ) -> None:
        for seq, segment, slots, resolved, jobs in walk:
            for n in slots:
                if n in resolved or not bound.fits(taken, jobs.get(n)):
                    continue
                digest = self._target(segment, seq, n)
                if digest is None or digest in self._judged or digest in self._missing:
                    continue
                if self._envs.get(digest) in skip:
                    continue
                held = self._take(q, claims, (seq, n), digest, skip)
                if held is not None:
                    taken.append(held)
                    if bound.full(taken):
                        return

    @property
    def _interval(self) -> float:
        return self._renew_fraction * self._stale_after

    def _target(self, segment: StorageDir, seq: int, n: int) -> Optional[str]:
        """Return the digest slot `n` names, read once per process; None if it names none."""
        if (seq, n) not in self._targets:
            target = segment.readlink(f"slot-{n}")
            if not DIGEST_NAME.fullmatch(target):
                logger.warning("queue %s: segment %s slot %d names no item", self.name, _seq(seq), n)
                return None
            self._targets[(seq, n)] = target
        return self._targets[(seq, n)]

    def _take(  # noqa: PLR0913 - the queue's open directories, the slot, the item and the held envs
        self, q: StorageDir, claims: StorageDir, slot: tuple[int, int], digest: str, skip: set[str]
    ) -> Optional[_Held]:
        """Take the claim on `digest` and read its fragment; None if it is claimed, has an outcome or is missing.

        None too when its env's label is in `skip`.
        """
        stack = contextlib.ExitStack()
        lease = Lease(
            claims.path / digest, timeout=0, stale_after=self._stale_after, directory=claims, clock=self._clock
        )
        try:
            stack.enter_context(lease)
        except LeaseTimeout:
            self.backoffs += 1
            self._judged[digest] = self._clock.now()
            return None
        try:
            with q.child("outcome") as outcome:
                if outcome.lstat(digest) is not None:
                    _resolve(q, *slot, digest)
                    stack.close()
                    return None
            fragment = self._fragment(q, lease, slot, digest)
            if fragment is not None:
                self._envs[digest] = identity_digest(LABEL_TAG, fragment.placement.env)
            if fragment is None or self._envs[digest] in skip:
                stack.close()
                return None
            state = stack.enter_context(lease.keep_alive(self._renew_fraction))
        except BaseException:
            stack.close()
            raise
        return _Held(digest, slot, fragment, lease, state, stack)

    def _fragment(self, q: StorageDir, lease: Lease, slot: tuple[int, int], digest: str) -> Optional[PlanFragment]:
        """Return the fragment of `digest`, held under `lease`; None when it is missing or is not that fragment.

        A missing item is an infrastructure fault: it is recorded in `faults` and stays ready. An item that is not the
        fragment its name digests is a content fault, and its outcome is `dead`.
        """
        with q.child("items") as items:
            data = items.read_regular(f"{digest}.json")
        if data is None:
            fault = FaultRecord(
                fault="infrastructure", type="MissingItem", message=f"queue item {digest}", gap=False, charged=False
            )
            logger.warning("queue %s: item %s is missing; it stays ready", self.name, digest)
            self.faults.append((digest, fault))
            self._missing[digest] = self._clock.now()
            return None
        try:
            document = jcs_loads(data)
            if identity_digest(ITEM_TAG, document) != digest:
                raise ValueError("not the item its name digests")
            return PlanFragment.model_validate(document)
        except ValueError as err:
            fault = FaultRecord(fault="content", type="CorruptItem", message=str(err), gap=False, charged=True)
            logger.warning("queue %s: item %s cannot be read (%s); its outcome is dead", self.name, digest, err)
            record = _outcome_record("dead", "queue", self._current(), fault=fault.model_dump(mode="json"))
            if settle(lease, q, digest, record) is not None:
                _resolve(q, *slot, digest)
            return None

    def depth(self) -> int:
        """Return an upper bound on the items without an outcome: taken slots less resolved ones, per live segment.

        Claimed items, items given two slots and items whose slot is not yet marked are counted.
        """
        return sum(1 for _ in self._waiting())

    def work(self) -> int:
        """Return an upper bound on the jobs of the items `depth` counts, from the slots' recorded job counts.

        A slot without a recorded count is counted from its item, and as one job when the item cannot be read.
        """
        return sum(self._waiting())

    def _waiting(self) -> Iterator[int]:
        """Yield the job count of each taken, unresolved slot of each live segment."""
        with self.open() as q, q.child("segments") as segments, q.child("items") as items:
            for name in segments.listdir():
                if SEQ_NAME.fullmatch(name):
                    with contextlib.suppress(FileNotFoundError), segments.child(name) as segment:
                        if segment.lstat(RETIRED_MARK) is None:
                            slots, resolved, jobs = _slots(segment)
                            for n in sorted(set(slots) - resolved):
                                yield jobs[n] if n in jobs else self._unrecorded(items, segment, int(name), n)

    def _unrecorded(self, items: StorageDir, segment: StorageDir, seq: int, n: int) -> int:
        """Return the job count of slot `n` of segment `seq`, which has none recorded, from its item."""
        digest = self._target(segment, seq, n)
        return 1 if digest is None else self._job_count(items, digest)

    def _job_count(self, items: StorageDir, digest: str) -> int:
        """Return the jobs of item `digest`, read once per process; 1 while it cannot be read."""
        if digest not in self._jobs:
            data = items.read_regular(f"{digest}.json")
            try:
                document = jcs_loads(data) if data is not None else None
            except ValueError:
                document = None
            jobs = document.get("jobs") if isinstance(document, dict) else None
            if not isinstance(jobs, list) or not jobs:
                return 1
            self._jobs[digest] = len(jobs)
        return self._jobs[digest]

    def oldest_age_s(self) -> Optional[float]:
        """Return the storage-clock age of the oldest slot whose item has no outcome and no live claim.

        An item behind a stale claim (its holder dead) counts as waiting. None when no item waits.
        """
        with self.open() as q, q.child("segments") as segments, q.child("claims") as claims:
            now = self._clock.storage_now(q)
            for name in sorted(n for n in segments.listdir() if SEQ_NAME.fullmatch(n) and int(n) >= self._oldest):
                with contextlib.suppress(FileNotFoundError), segments.child(name) as segment:
                    if segment.lstat(RETIRED_MARK) is not None:
                        continue
                    slots, resolved, _ = _slots(segment)
                    for n in slots:
                        digest = None if n in resolved else self._target(segment, int(name), n)
                        if digest is not None and not self._live(claims, digest, now):
                            st = segment.lstat(f"slot-{n}")
                            return None if st is None else max(0.0, now - st.st_mtime)
        return None

    def _live(self, claims: StorageDir, digest: str, now: float) -> bool:
        """Return whether `digest` has a claim stamped within `stale_after` of `now` on the storage clock."""
        try:
            with claims.child(f"{digest}.lease") as lease:
                stamps = [st.st_mtime for n in lease.listdir() if (st := lease.lstat(n)) is not None]
                own = claims.lstat(f"{digest}.lease")
        except (FileNotFoundError, WrongKindError):
            return False
        stamps += [own.st_mtime] if own is not None else []
        return bool(stamps) and now - max(stamps) <= self._stale_after


class QueueSet:
    """The queues under one shared directory, one per name.

    Args:
        root: The directory holding the queues; made if missing.
        profile: The execution profile whose `pool` and `lease` sections size segments and pace claims; the default
            profile when None.
    """

    def __init__(
        self,
        root: Path,
        profile: Optional[ExecutionProfile] = None,
        *,
        epoch: Optional[Callable[[], int]] = None,
        clock: Clock = SystemClock(),  # noqa: B008 - stateless
    ) -> None:
        """Hold the queues under `root`; `epoch` and `clock` are as `Queue` reads them.

        A run's queues live in `<run dir>/queues`, so `epoch` is the current epoch of the run at `root`'s parent
        (`Epoch(root.parent).current`, 0 where no epoch was taken) when None.
        """
        self.root = Path(root)
        self.profile = profile if profile is not None else ExecutionProfile.load()
        self.epoch = epoch if epoch is not None else Epoch(self.root.parent).current
        self._clock = clock
        self._queues: dict[str, Queue] = {}
        self._labels: dict[str, str] = {}

    def queue(self, name: str) -> Queue:
        """Return the queue `name`, made if missing."""
        if name not in self._queues:
            self._queues[name] = Queue(self.root, name, self.profile, self.epoch, self._clock)
        return self._queues[name]

    def names(self) -> list[str]:
        """Return the name of every queue under `root`, sorted; a name is read once and a removed queue's is dropped."""
        if not self.root.is_dir():
            return []
        with StorageDir.open_root(self.root) as root:
            labels = {n for n in root.listdir() if DIGEST_NAME.fullmatch(n)}
            self._labels = {k: v for k, v in self._labels.items() if k in labels}
            for label in sorted(labels - self._labels.keys()):
                with root.child(label) as directory:
                    name = directory.read_regular("name")
                if name is not None and identity_digest(LABEL_TAG, name.decode()) == label:
                    self._labels[label] = name.decode()
        return sorted(self._labels.values())

    def claimed_any(self) -> bool:
        """Return whether any item of any queue is under a live claim, on the storage clock."""
        return any(self.queue(name).claimed_any() for name in self.names())

    def reconcile(self) -> int:
        """Put again every item of every queue under `root` that has neither a slot nor an outcome; return how many."""
        return sum(self.queue(name).reconcile() for name in self.names())


__all__ = [
    "ITEM_TAG",
    "LABEL_TAG",
    "OUTCOMES",
    "PROBE_MARK",
    "RUN_QUEUES",
    "Claim",
    "ProbeQueueError",
    "Queue",
    "QueueSet",
    "ServeLease",
    "UnmarkedProbeItem",
    "item_digest",
    "probe_of",
    "probe_queue",
    "settle",
]
