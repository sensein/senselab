# ADR-003: Storage and lease protocol on shared filesystems

**Status:** accepted, 2026-10-30.

## Context
Architect review §6 re-opened four items:
- R-3: two coordinators can run at once, and a check-then-act fencing scheme races;
- R-8: starvation, given today's lexicographic claim order and O(N) directory scans;
- R-9: service lifecycle;
- R-12: commits to the packed tier.

None of the recorded resolutions had a mechanism on plain shared files.

## Decision

The protocol uses only primitives that `work_pool` has measured on NFS v4.2: atomic `rename`,
`O_EXCL` hard link, and server-clock timestamps.

- **Safety by idempotence.**
  - Jobs are content-addressed (ADR-002), and results are written to their key by rename, first
    writer wins.
  - A duplicate coordinator, a duplicate claim or a resumed run can only cause **duplicate work,
    never a wrong or mixed result**.
  - The coordinator epoch file (incremented by `O_EXCL` link) is a fence for efficiency only. A
    stale coordinator sees a newer epoch and stops releasing.
- **Claim order and fairness.**
  - Each queue keeps append-only, time-ordered index segments, and workers claim oldest-first
    from the head segment. This replaces lexicographic scans; claiming costs O(1) amortised.
  - A worker holding a resident model yields after N items or T seconds (its quantum) whenever
    another queue it can serve has older work.
- **GPU memory admission.**
  - Devices are host-local, so admission is a host-local `flock` token budget per device, not an
    NFS lock.
  - **In 2.0:** static caps per resource class, and an OOM is classified as a resource fault (it
    re-places the job and does not charge the item). The token budget is added when the scale
    ladder shows the need (design §6.4).
- **Services.**
  - A service is a registry entry `{identity, endpoint, heartbeat}` under a lease.
  - Each consumer holds its own lease on the service. The service exits when no consumer lease has
    been live for T.
  - The service identity is attested on every response (I-8).
- **Packed tier (deferred, design §6.4).** When added: per-shard append-only segments, written only
  by that shard's lease holder, plus an index committed by rename. Readers trust only indexed
  entries.

**Guaranteed:**
- no wrong, torn or mixed results;
- eventual completion while workers are alive;
- results that are never lost once committed.

**Not guaranteed:**
- exactly-once execution;
- bounded latency when the cluster grants no workers;
- fairness across users of a shared store.

## Consequences
- `work_pool` gains index segments and quantum yielding. Items become plan fragments.
- The packed tier and the token arbiter wait for measurements.

## Alternatives rejected
- **A database or lock service for coordination.** It adds an operated dependency to every site.
- **Strict fencing on NFS.** It cannot be made atomic with check-then-act; idempotence makes it
  unnecessary for correctness.
