# Locks that work when the cache is shared between users

Status: design approved 2026-08-09, not yet implemented.

senselab's caches are worth sharing. On a cluster, a group points `HF_HOME` at one large tree
(`/orcd/data/satra/002/huggingface` — 790 GB, group-writable to `orcd_rg_hstor004_pi_satra`) so a
16 GB checkpoint is downloaded once rather than once per person. senselab's own resolution cache is
equally shareable: a repo's resolved commit SHA is the same for everyone.

What is not shareable today is the **coordination**. Both of senselab's file locks assume a single
user, in ways that fail silently rather than loudly.

## What is actually there

Two implementations, which have already diverged:

**`dependencies.py::_HeartbeatLock`** — the real one. Acquires with a 60 s timeout; on timeout it
checks the heartbeat's mtime against a 90 s staleness threshold, and either keeps waiting (holder
alive, download in progress) or unlinks the lock and takes over (holder crashed). This is the
behaviour a shared cache needs.

**`subprocess_venv.py::_FileLockWithHeartbeat`** — decorative. It writes a heartbeat every 15 s and
deletes it on exit, and **nothing in the repository ever reads it**. Its docstring says "Other
processes can check the heartbeat to distinguish a live lock holder from a crashed one"; no code
does. It has no staleness threshold and no takeover path, so a crashed holder blocks every waiter
until the 300 s timeout, then raises.

That second one guards venv creation — the operation `CLAUDE.md` already records as the cause of a
5.5-hour CI hang, where "`ensure_venv` takes no lock and does `shutil.rmtree(venv_dir)` before
installing, so two workers that want the same subprocess venv delete each other's tree mid-install."

## Why neither works across users

**1. File modes.** Both create lock and heartbeat files through `FileLock(...)` and `Path.touch()`,
which use the process umask — typically `0644`, owner-writable only. In a shared tree:

- user B cannot `touch` the heartbeat, so B's own heartbeat writes silently fail (`except Exception:
  pass` in both implementations swallows it);
- B's heartbeat therefore never refreshes, so a *third* user reads a stale heartbeat and breaks a
  lock that is very much alive;
- B may not be able to `unlink` A's stale lock, so the takeover path raises where it should recover.

The ORCD tree carries setgid (`drwxrwsr-x`), so group *ownership* is inherited — but setgid does not
make files group-*writable*. That is the gap.

**2. No holder identity.** Neither lock records who holds it. On a cluster the holder is frequently
on a different node, and the only useful diagnostic — "held by `alice@node1234` pid 4211, last beat
300 s ago" — cannot be produced. The current log line names only the path.

**3. Unstated cross-node assumptions.** Staleness compares `time.time()` on the waiting node against
an mtime set by the file server. Node clock skew therefore reads directly as heartbeat age. And
`filelock` uses `fcntl`, whose behaviour over NFS depends on the protocol version and server
configuration. Neither assumption is written down anywhere.

## Design

**One lock, used by both call sites.** Consolidating is not tidiness: the two implementations
diverged, and the divergence is precisely that one of them lost its stale detection. Two copies of a
concurrency primitive is how that happens again.

**Always group-writable.** Lock and heartbeat files are created `0664`, directories `2775`
(setgid, so group ownership propagates). Not behind a flag, and not inferred from the directory's
setgid bit:

- On a single-user machine, group-writable lock files are harmless.
- Behind a flag, someone who points `HF_HOME` at a shared tree without knowing the flag exists gets
  a permission error whose cause is several layers from its symptom — and worse, gets the *silent*
  failure above, where a swallowed heartbeat write leads a third party to break a live lock.
- Inferred from setgid, the behaviour depends on a directory bit almost nobody inspects.

The failure this prevents is silent and confusing; the cost of preventing it unconditionally is
nil.

**Record the holder.** The lock file carries a small JSON payload: user, host, pid, and the time the
lock was taken. When a waiter breaks a stale lock it logs all of it. This is the difference between
"stale lock detected, breaking" and "breaking lock held by alice@node1234 pid 4211, taken 22 min ago,
last heartbeat 340 s ago" — on a shared filesystem, the second is the one that lets someone find out
whether alice's job actually died.

**State the assumptions in the code.** The staleness threshold is a wall-clock comparison across
nodes, so the threshold must exceed plausible clock skew; and `fcntl` locking requires NFSv4 or
equivalent. Both belong in the module docstring, because both are invisible until they misbehave.

## What this does not do

- It does **not** introduce a lock manager, a lease service, or anything requiring a daemon. A file
  lock with a heartbeat is adequate for this workload — a handful of processes contending to
  download a model or build a venv.
- It does **not** try to make locking correct on filesystems without working `fcntl`. It documents
  the requirement instead of pretending to detect it.
- It does **not** change where any cache lives. `SENSELAB_CACHE` and `HF_HOME` keep their current
  meanings; this makes a shared location safe, which is what makes pointing them at one reasonable.
- It does **not** add locking to `ensure_venv` itself. That gap is real and recorded in `CLAUDE.md`,
  but adding a lock to venv creation is a behavioural change to a hot path and deserves its own
  review — this work makes the lock it would use trustworthy first.

## Testing

The hard part is that the interesting cases are multi-process and multi-user, and a test suite runs
as one user. So:

- **Modes are directly assertable.** Create a lock, stat the files, assert `0664`/`2775`. This is the
  fix's core and needs no concurrency.
- **Staleness is assertable by writing an old mtime.** `os.utime` the heartbeat into the past and
  assert the waiter breaks the lock; write a recent one and assert it keeps waiting. No sleeping.
- **The holder payload is assertable** by reading the lock file while held.
- **Cross-user is not directly testable** in CI. Simulate it: a lock file owned by the test user but
  written by a *different recorded identity*, and assert the takeover path reports that identity
  rather than assuming self. That covers the logic; it cannot cover the kernel's permission check,
  which is what the mode assertions are for.
- **Two live processes contending** is worth one real test, using `multiprocessing` with the second
  process asserting it waited rather than proceeding — the one case where a mocked test would prove
  nothing.
