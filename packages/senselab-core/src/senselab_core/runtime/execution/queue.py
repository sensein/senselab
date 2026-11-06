"""Queues of plan fragments in a shared directory: items by identity, index segments, oldest-first claims (ADR-003).

`QueueSet(root)` holds one queue per name: a resource class, with `#<model identity>` for a resident model
(`gpu:h100#<id>`). A queue is the directory `<root>/<identity_digest("label@1", name)>`, holding:

- `name`: the queue's name, written once.
- `items/<digest>.json`: a `plan-fragment@1.0` document as JCS, named by `identity_digest("plan-fragment@1", …)` of
  its JSON form; the first writer wins.
- `segments/<seq>.jsonl`: the index, one line `{"item": <digest>, "t": <storage time>}` per indexed item, appended
  under the segment's `Lease` after a `prove()`; the head segment (the highest `<seq>`) rolls at `pool.segment_items`
  lines. `<seq>.retired` marks a segment below the head whose items are all finished.
- `claims/<digest>.lease`: the `Lease` of a claim on the item.
- `done/<digest>.json`, `dead/<digest>.json`: an item's outcome, by hard link, so the first writer wins; `dead` holds
  the item's `FaultRecord`.
- `progress/<seq>/<digest>`: a mark, written after the outcome, that an item indexed in segment `<seq>` is finished.

A claim lists `segments/`, reads the oldest live segment, lists its `progress/<seq>/` and `claims/`, and takes the
lease of the first item that is neither finished nor live-claimed; the outcome is checked again once the lease is
held. A commit proves its lease and then links `done/<digest>.json`, so an item is committed once whatever the number
of claimers. A segment whose items are all finished is retired, so finished items are not read again.
"""

import contextlib
import json
import logging
import os
import re
import time
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Optional

from senselab_worker.canonical import jcs_dumps, jcs_loads
from senselab_worker.digest import identity_digest

from senselab_core.runtime.execution.executor import FaultRecord, PlanFragment
from senselab_core.runtime.locks import TIMEOUT_S, Lease, LeaseState, LeaseTimeout
from senselab_core.runtime.profiles import ExecutionProfile
from senselab_core.runtime.storage import (
    STAGING_DIR_MODE,
    STAGING_FILE_MODE,
    TEMPORARY_NAME,
    StorageDir,
    path_component,
    temporary_name,
)

logger = logging.getLogger("senselab")

ITEM_TAG = "plan-fragment@1"
"""The identity tag of a queue item: the digest of a fragment's JSON form."""
LABEL_TAG = "label@1"
"""The identity tag of a queue's directory name: the digest of the queue's name."""
QUEUE_NAME = re.compile(r"^[^#\s/]+(#\S+)?$")
"""A queue name: a resource class, with `#<model identity>` for a resident model."""
ITEM_NAME = re.compile(r"^[0-9a-f]{64}\.json$")
"""An item, a `done` or a `dead` entry: `<digest>.json`."""
MARK_NAME = re.compile(r"^[0-9a-f]{64}$")
"""A progress mark: `<digest>`."""
SEGMENT_NAME = re.compile(r"^([0-9]{12})\.jsonl$")
"""An index segment: `<seq>.jsonl`, `<seq>` twelve digits."""
RETIRED_NAME = re.compile(r"^([0-9]{12})\.retired$")
"""The mark of a retired segment: `<seq>.retired`."""
SEQ_NAME = re.compile(r"^[0-9]{12}$")
"""A segment's progress directory: `<seq>`."""
NAME_FILE = re.compile(r"^name$")
"""The file holding a queue's name."""
DIRECTORIES = ("items", "segments", "claims", "done", "dead", "progress")
"""The directories of a queue."""
APPEND_ATTEMPTS = 8
"""How many times `put_all` takes the head segment's lease for one batch before it raises `QueueUnavailable`."""


class QueueUnavailable(RuntimeError):
    """The head segment could not be appended to: its lease was lost or the head kept rolling."""


def _segment(seq: int) -> str:
    return f"{seq:012d}.jsonl"


def _publish(directory: StorageDir, name: str, data: bytes, grammar: re.Pattern[str]) -> bool:
    """Write `data` at `name` unless something is there; return whether this call wrote it.

    The bytes are written to a temporary, fsynced and hard-linked at `name`, so a reader sees all of them or nothing.
    """
    temporary = temporary_name(name)
    directory.write_new(temporary, data, STAGING_FILE_MODE, grammar=TEMPORARY_NAME)
    try:
        directory.link(temporary, directory, name, grammar=grammar)
    except FileExistsError:
        return False
    finally:
        directory.remove_owned(temporary, pattern=TEMPORARY_NAME)
    return True


def _mark(progress: StorageDir, seq: int, digest: str) -> None:
    """Write the progress mark of `digest` in segment `seq`."""
    with progress.child(f"{seq:012d}", create=True, mode=STAGING_DIR_MODE) as marks:
        with contextlib.suppress(FileExistsError):
            marks.write_new(digest, b"", STAGING_FILE_MODE, grammar=MARK_NAME)


def _marks(progress: StorageDir, seq: int) -> set[str]:
    """Return the digests marked finished in segment `seq`."""
    try:
        with progress.child(f"{seq:012d}") as marks:
            return {n for n in marks.listdir() if MARK_NAME.fullmatch(n)}
    except FileNotFoundError:
        return set()


def _parse(data: bytes) -> list[tuple[str, float]]:
    """Return the `(digest, t)` of each line of a segment, in order; a line that is not one is skipped and logged."""
    lines = data.split(b"\n")
    if lines and lines[-1] == b"":
        lines.pop()
    entries = []
    for number, line in enumerate(lines):
        try:
            entry = json.loads(line)
            digest, t = entry["item"], float(entry["t"])
            if not isinstance(digest, str) or not MARK_NAME.fullmatch(digest):
                raise ValueError(digest)
        except (ValueError, TypeError, KeyError):
            logger.warning("queue segment: line %d is not an index entry; skipped", number + 1)
            continue
        entries.append((digest, t))
    return entries


def _index(segments: StorageDir, seq: int) -> list[tuple[str, float]]:
    """Return each item indexed in segment `seq` once, at its first line, in order."""
    seen: dict[str, float] = {}
    for digest, t in _parse(segments.read_regular(_segment(seq)) or b""):
        seen.setdefault(digest, t)
    return list(seen.items())


def _settle(q: StorageDir, outcome: str, seq: int, digest: str, record: dict[str, object]) -> bool:
    """Link the outcome `record` of `digest` in `outcome`'s directory and mark it in segment `seq`.

    Returns whether this call wrote the outcome.
    """
    with q.child(outcome) as target, q.child("progress") as progress:
        written = _publish(target, f"{digest}.json", jcs_dumps(record), ITEM_NAME)
        _mark(progress, seq, digest)
    return written


class _Held:
    """One item a claim holds: its digest, segment, fragment, lease and the stack that releases it."""

    def __init__(  # noqa: PLR0913 - the parts of one held item
        self,
        digest: str,
        seq: int,
        fragment: PlanFragment,
        lease: Lease,
        state: LeaseState,
        stack: contextlib.ExitStack,
    ) -> None:
        self.digest, self.seq, self.fragment = digest, seq, fragment
        self.lease, self.state, self.stack = lease, state, stack

    def held(self) -> bool:
        """Return whether the lease is still held, proven by the storage."""
        return not self.state.lost and self.lease.prove()


class Claim:
    """Items one worker claimed from a queue, each under its own beating `Lease` until it is finished.

    `outcomes` records, per digest, what became of each item this claim gave up: `committed`, `duplicate` (another
    claim committed it first), `dead`, `released` or `lost` (its lease was lost before the outcome was written).
    """

    def __init__(self, queue: "Queue", worker: str, held: list[_Held]) -> None:
        """Hold `held` for `worker`; use `Queue.claim`."""
        self.queue = queue
        self.worker = worker
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

    def segment(self, digest: str) -> int:
        """Return the segment a held item was claimed from."""
        return self._held[digest].seq

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
        lost = [h for h in self._held.values() if not h.held()]
        for held in lost:
            self._end(held, "lost")
        return not lost

    def _finish(self, held: _Held, outcome: str, record: dict[str, object]) -> str:
        """Prove the lease, link the outcome `record` in `outcome`'s directory and mark it; return what became of it."""
        if not held.held():
            self._end(held, "lost")
            return "lost"
        with self.queue.open() as q:
            written = _settle(q, outcome, held.seq, held.digest, record)
        result = ("committed" if outcome == "done" else "dead") if written else "duplicate"
        self._end(held, result)
        return result

    def commit(self, items: Optional[Iterable[str]] = None) -> tuple[str, ...]:
        """Commit the held `items` (all by default) to `done/`; return the digests this call committed.

        An item whose lease is lost is not committed (`lost`); one another claim committed first is `duplicate`.
        """
        record = {"worker": self.worker}
        outcomes = {h.digest: self._finish(h, "done", {**record, "segment": h.seq}) for h in self._chosen(items)}
        return tuple(d for d, o in outcomes.items() if o == "committed")

    def dead(self, fault: FaultRecord, items: Optional[Iterable[str]] = None) -> tuple[str, ...]:
        """Move the held `items` (all by default) to `dead/` with `fault`; return the digests this call moved."""
        record = {"worker": self.worker, "fault": fault.model_dump(mode="json")}
        outcomes = {h.digest: self._finish(h, "dead", {**record, "segment": h.seq}) for h in self._chosen(items)}
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


class Queue:
    """One queue of plan fragments under a `QueueSet`.

    Attributes:
        name: The queue's name.
        path: The queue's directory.
    """

    def __init__(self, root: Path, name: str, profile: ExecutionProfile) -> None:
        """Open, creating it if missing, the queue `name` under `root`; use `QueueSet.queue`.

        Raises:
            ValueError: `name` is not a resource class with an optional `#<model identity>`.
        """
        if not isinstance(name, str) or not QUEUE_NAME.fullmatch(name):
            raise ValueError(f"{name!r} is not a queue name: a resource class, with #<model identity> for a model")
        self.name = name
        self.path = Path(root) / identity_digest(LABEL_TAG, name)
        self._segment_items = profile.pool.segment_items
        self._stale_after = profile.lease.stale_after_s
        self._renew_fraction = profile.lease.renew_fraction
        self._judged: dict[str, float] = {}
        with StorageDir.open_root(self.path, create=True, mode=STAGING_DIR_MODE) as q:
            for name_ in DIRECTORIES:
                q.child(name_, create=True, mode=STAGING_DIR_MODE).close()
            if q.lstat("name") is None:
                _publish(q, "name", name.encode(), NAME_FILE)

    @contextlib.contextmanager
    def open(self) -> Iterator[StorageDir]:
        """Yield the queue's directory, open."""
        with StorageDir.open_root(self.path) as q:
            yield q

    def _lease(self, directory: StorageDir, name: str, timeout: float) -> Lease:
        return Lease(directory.path / name, timeout=timeout, stale_after=self._stale_after, directory=directory)

    def put(self, fragment: PlanFragment) -> None:
        """Write `fragment`'s item and index it in the head segment, unless it is finished."""
        self.put_all([fragment])

    def put_all(self, fragments: Iterable[PlanFragment]) -> int:
        """Write each fragment's item and index the unfinished ones, in order, a segment at a time; return the lines.

        Raises:
            TypeError: an element is not a `PlanFragment`.
            QueueUnavailable: the head segment's lease was lost, or the head rolled, `APPEND_ATTEMPTS` times running.
        """
        pending: dict[str, None] = {}
        with self.open() as q, q.child("items") as items, q.child("done") as done, q.child("dead") as dead:
            for fragment in fragments:
                if not isinstance(fragment, PlanFragment):
                    raise TypeError(f"a queue item is a PlanFragment, not {type(fragment).__name__}")
                document = fragment.model_dump(mode="json")
                digest = identity_digest(ITEM_TAG, document)
                name = f"{digest}.json"
                if items.lstat(name) is None:
                    _publish(items, name, jcs_dumps(document), ITEM_NAME)
                if done.lstat(name) is None and dead.lstat(name) is None:
                    pending[digest] = None
            order = list(pending)
            with q.child("segments") as segments:
                while order:
                    for _ in range(APPEND_ATTEMPTS):
                        if appended := self._append(segments, order):
                            order = order[appended:]
                            break
                    else:
                        raise QueueUnavailable(f"queue {self.name}: could not append to {segments.path}")
        return len(pending)

    def _head(self, segments: StorageDir) -> int:
        """Return the head segment's number, making segment 0 when there is none."""
        seqs = [int(m.group(1)) for n in segments.listdir() if (m := SEGMENT_NAME.fullmatch(n))]
        if seqs:
            return max(seqs)
        with contextlib.suppress(FileExistsError):
            segments.write_new(_segment(0), b"", STAGING_FILE_MODE, grammar=SEGMENT_NAME)
        return 0

    def _append(self, segments: StorageDir, digests: list[str]) -> int:
        """Append the first of `digests` that fit to the head segment under its lease; return how many.

        A full head rolls (0 is returned); a lost lease returns 0. The lines are read back under the lease, and any
        missing is appended again.
        """
        head = self._head(segments)
        name = _segment(head)
        lease = self._lease(segments, name, TIMEOUT_S)
        with lease, lease.keep_alive(self._renew_fraction) as state:
            data = segments.read_regular(name) or b""
            room = self._segment_items - len(_parse(data))
            if room <= 0:
                with contextlib.suppress(FileExistsError):
                    segments.write_new(_segment(head + 1), b"", STAGING_FILE_MODE, grammar=SEGMENT_NAME)
                return 0
            batch = digests[:room]
            t = segments.dir_now()
            for _ in range(2):
                written = {d for d, _ in _parse(data)}
                missing = [d for d in batch if d not in written]
                if not missing:
                    return len(batch)
                lines = b"".join(jcs_dumps({"item": d, "t": t}) + b"\n" for d in missing)
                if state.lost or not lease.prove():
                    return 0
                self._write(segments, name, (b"\n" if data and not data.endswith(b"\n") else b"") + lines)
                data = segments.read_regular(name) or b""
            return len(batch) if all(d in {e for e, _ in _parse(data)} for d in batch) else 0

    @staticmethod
    def _write(segments: StorageDir, name: str, data: bytes) -> None:
        """Append `data` to the segment `name` in one write, fsynced."""
        fd = os.open(path_component(name), os.O_WRONLY | os.O_APPEND | os.O_NOFOLLOW, dir_fd=segments.fd)
        try:
            os.write(fd, data)
            os.fsync(fd)
        finally:
            os.close(fd)

    def _live(self, segments: StorageDir) -> tuple[list[int], int]:
        """Return the live segments, oldest first, and the head's number (-1 when there is none)."""
        names = segments.listdir()
        seqs = {int(m.group(1)) for n in names if (m := SEGMENT_NAME.fullmatch(n))}
        retired = {int(m.group(1)) for n in names if (m := RETIRED_NAME.fullmatch(n))}
        return sorted(seqs - retired), max(seqs, default=-1)

    def claim(self, worker: str, max_items: int = 1) -> Optional[Claim]:
        """Claim up to `max_items` unfinished items, oldest first, for `worker`; None when there is none to claim.

        Each item is held under a `Lease` beating from its own renewer until the claim commits, kills or releases
        it. An item under a live claim is skipped; a stale claim is broken and the item claimed again.

        Raises:
            ValueError: `max_items` is below 1.
        """
        if max_items < 1:
            raise ValueError(f"a claim takes at least one item (got {max_items})")
        taken: list[_Held] = []
        try:
            with self.open() as q, q.child("segments") as segments, q.child("claims") as claims:
                self._fill(q, segments, claims, taken, max_items)
        except BaseException:
            for held in taken:
                held.stack.close()
            raise
        return Claim(self, worker, taken) if taken else None

    def _fill(self, q: StorageDir, segments: StorageDir, claims: StorageDir, taken: list[_Held], most: int) -> None:
        """Take claims into `taken`, oldest first, until it holds `most` or no live segment has a claimable item."""
        leased = {n.removesuffix(".lease") for n in claims.listdir() if n.endswith(".lease")}
        self._judged = {d: at for d, at in self._judged.items() if d in leased}
        live, head = self._live(segments)
        with q.child("progress") as progress:
            for seq in live:
                finished = _marks(progress, seq)
                waiting = [d for d, _ in _index(segments, seq) if d not in finished]
                if not waiting and seq != head:
                    with contextlib.suppress(FileExistsError):
                        segments.write_new(f"{seq:012d}.retired", b"", STAGING_FILE_MODE, grammar=RETIRED_NAME)
                for digest in waiting:
                    if digest in leased and not self._due(digest):
                        continue
                    held = self._take(q, claims, progress, seq, digest)
                    if held is not None:
                        taken.append(held)
                        if len(taken) == most:
                            return

    def _due(self, digest: str) -> bool:
        """Return whether the live claim on `digest` is due to be judged again: not judged within one beat."""
        judged = self._judged.get(digest)
        return judged is None or time.monotonic() - judged >= self._renew_fraction * self._stale_after

    def _take(self, q: StorageDir, claims: StorageDir, progress: StorageDir, seq: int, digest: str) -> Optional[_Held]:
        """Take the claim on `digest` and read its fragment; None if it is claimed or finished."""
        name = f"{digest}.json"
        stack = contextlib.ExitStack()
        lease = self._lease(claims, digest, 0)
        try:
            stack.enter_context(lease)
        except LeaseTimeout:
            self._judged[digest] = time.monotonic()
            return None
        self._judged.pop(digest, None)
        try:
            with q.child("done") as done, q.child("dead") as dead:
                if done.lstat(name) is not None or dead.lstat(name) is not None:
                    _mark(progress, seq, digest)
                    stack.close()
                    return None
            fragment = self._fragment(q, digest)
            if fragment is None:
                fault = FaultRecord(
                    fault="infrastructure",
                    type="UnreadableItem",
                    message=f"queue item {digest}",
                    gap=False,
                    charged=False,
                )
                if lease.prove():
                    _settle(q, "dead", seq, digest, {"worker": "queue", "fault": fault.model_dump(mode="json")})
                stack.close()
                return None
            state = stack.enter_context(lease.keep_alive(self._renew_fraction))
        except BaseException:
            stack.close()
            raise
        return _Held(digest, seq, fragment, lease, state, stack)

    def _fragment(self, q: StorageDir, digest: str) -> Optional[PlanFragment]:
        """Return the fragment of `digest`; None when its item is missing or is not that fragment."""
        with q.child("items") as items:
            data = items.read_regular(f"{digest}.json")
        try:
            document = jcs_loads(data) if data is not None else None
            if document is None or identity_digest(ITEM_TAG, document) != digest:
                raise ValueError("missing, or not the item its name digests")
            return PlanFragment.model_validate(document)
        except ValueError as err:
            logger.warning("queue %s: item %s cannot be read (%s); it goes to dead/", self.name, digest, err)
            return None

    def depth(self) -> int:
        """Return how many items of the live segments are unfinished, claimed ones included."""
        total = 0
        with self.open() as q, q.child("segments") as segments, q.child("progress") as progress:
            for seq in self._live(segments)[0]:
                finished = _marks(progress, seq)
                total += sum(d not in finished for d, _ in _index(segments, seq))
        return total

    def oldest_age_s(self) -> Optional[float]:
        """Return the storage-clock age of the oldest unfinished, unclaimed item; None when there is none."""
        with self.open() as q, q.child("segments") as segments, q.child("progress") as progress:
            with q.child("claims") as claims:
                leased = {n.removesuffix(".lease") for n in claims.listdir() if n.endswith(".lease")}
            for seq in self._live(segments)[0]:
                finished = _marks(progress, seq) | leased
                waiting = [t for d, t in _index(segments, seq) if d not in finished]
                if waiting:
                    return max(0.0, q.dir_now() - waiting[0])
        return None


class QueueSet:
    """The queues under one shared directory, one per name.

    Args:
        root: The directory holding the queues; made if missing.
        profile: The execution profile whose `pool` and `lease` sections size segments and pace claims; the default
            profile when None.
    """

    def __init__(self, root: Path, profile: Optional[ExecutionProfile] = None) -> None:
        """Hold the queues under `root`."""
        self.root = Path(root)
        self.profile = profile if profile is not None else ExecutionProfile.load()
        self._queues: dict[str, Queue] = {}

    def queue(self, name: str) -> Queue:
        """Return the queue `name`, made if missing."""
        if name not in self._queues:
            self._queues[name] = Queue(self.root, name, self.profile)
        return self._queues[name]


__all__ = ["ITEM_TAG", "LABEL_TAG", "Claim", "Queue", "QueueSet", "QueueUnavailable"]
