# Parked items: closed and open (rulings audit I-12)

The controller ledgers (git-excluded) and the task reports held parked, deferred and "follows" items. This closes
rulings audit I-12 and R35. Checked against `2x/phase-1` at `7ec0d169` on 2026-11-01.

- **Open: 61.** Each has an R-row in [`review-register.md`](../../review-register.md) (R-110..R-131), except
  three that point at existing rows: A-39 branch protection is R-97, M-4 whole-file hashing is R-109, and its
  STATE follow is the same row.
- **Resolved: 44** and **obsolete: 13**, listed below with their evidence. They get no register row.
- Open items by row:
  - R-110: H-F3c ST-1
  - R-111: ST-2
  - R-112: P1-T6b ST-3
  - R-113: ST-4
  - R-114: ST-5
  - R-115: ST-6
  - R-116: P1-T6e H-Gb ST-7
  - R-117: ST-8
  - R-118: P1-T8a
  - R-119: P1-T2e P1-T4a
  - R-120: H-H4b H-H4c H-H4d
  - R-121: P1-T5f H-H1a H-H1b H-F3a H-F3b
  - R-122: P1-T12a P1-T12b P1-T12c
  - R-123: P1-T4b H-H9a1 H-H9a3 H-H9d3 H-H9e2 H-H11b H-Ga
  - R-124: H-H9a2
  - R-125: P1-T10c P1-T13b H-H3b H-H9b2
  - R-126: H-H9c4 H-H9c5 H-H9d2 H-H9cm1 H-H9cm3
  - R-127: P1-T11b H-F1a H-F1b H-F4a
  - R-128: P1-T8b P1-T8c P1-T8d P1-T11a H-H1c H-H9b3 H-H10a H-H10b
  - R-129: H-H9cm2
  - R-130: H-DPa
  - R-131: P1-T2a

## Resolved and obsolete

| Id | Source | Item | State | Evidence |
|---|---|---|---|---|
| P1-T1a | T1 minor | Core hook is a tracked git symlink (Windows checkouts) | resolved | no mode-120000 entry in the tree at `7ec0d169`; the shared hook is `hatch_build_pins.py` at the root (H6/H7) |
| P1-T1b | T1 minor | worker-floor does not pin numpy==1.23.5 | resolved | `.github/locks/worker-floor.in:5` pins `numpy==1.23.5` |
| P1-T1c | T1 concern | Version reads `31.dev33` (1.x freeze tag matches the default tag regex) | resolved | `git_describe_command ... --match '[0-9]*.[0-9]*.[0-9]*'` in all three pyprojects; freeze marker is a branch (H-R41) |
| P1-T2b | T2 minor | Scalar NaN/Inf via `.item()` unstated | resolved | ruling R13; `canonical.py:378` refuses non-finite values |
| P1-T2c | T2 minor | Handler returning `{"error": <non-exception>}` unspecified | resolved | `serve.py:138` accepts only a `BaseException` as `error` |
| P1-T2d | T2 minor | R-55 lens column holds "R-37" | resolved | housekeeping at T15a (R40) |
| P1-T3a | T3 concern | Integer range rule vs float round-trip | resolved | T3 fix round 1 and H2: float path, `jcs_loads` round trip |
| P1-T3b | T3 concern | Content root omits `chunk_size` | resolved | R-89 (H2): the content root covers `chunk_size` |
| P1-T4c | T4 minor | `scripted_env` replaces PYTHONPATH | resolved | `runtime/conftest.py:42-43` and `shutdown_schedule_fixture.py:283-284` prepend |
| P1-T4d | T4 minor | `_remove_arrays` unlink without `missing_ok` in `finally` | obsolete | function rewritten on `remove_owned` with total cleanup (H4, `worker_client.py:794-806`) |
| P1-T4e | T4 minor | `WorkerClient.run` not documented as non-thread-safe | resolved | `worker_client.py:369` states one `run` in flight; `close()` thread rule at `:528` |
| P1-T4f | T4 minor | Launcher imports numpy before claiming fd 1 | resolved | H9a `1c13f572`: the launcher claims fds 0-2 before any import |
| P1-T4g | T4 concern | Orphan array files after a partly encoded item | resolved | H9a: cleanup in `finally`, `__del__` completes unfinished steps (R-99) |
| P1-T5a | T5 minor | Models holding numpy arrays raise on `==` and `hash()` | resolved | `kernel/_base.py:366,372` define both |
| P1-T5b | T5 minor | Header trusted from validation context | obsolete | context key removed when stamping moved to `runtime/audio_io.py` (F3 `56563623..2cd31918`); the remaining gap is R-121 |
| P1-T5c | T5 minor | `model_dump(mode="json")` fails on ndarray | resolved | arrays cross as canonical `.npy` (codec, store, provenance `$nd`; R-104) |
| P1-T5d | T5 minor | Arrays lose the read-only flag through pickle/deepcopy | resolved | `is_frozen`/`frozen_array` (`_base.py:75-90`), sealed values (H9c, R-102) |
| P1-T5e | T5 concern | File-backed group caches frame count; `Recording.groups` mutable | resolved | source stamps (R-103) and `FrozenMapping` (R-95) |
| P1-T6a | T6 concern | `protocol` is part of the context fingerprint | resolved | decided in `identity.py:245`: recorded, checked at the handshake, excluded from the digest |
| P1-T6c | T6 concern | Canonical params turn `1.0` into `1` | obsolete | H-R29: JCS numeric equality accepted |
| P1-T6d | T6 concern | `LogicalKey` not hashable | resolved | `Frozen.__hash__` (`_base.py:372`) |
| P1-T7a | T7 minor | Pre-existing mypy failure `host.py:148` forces `SKIP=mypy` | resolved | T15a `fafb2d11..94e81479`; mypy strict hook `73931a6d` |
| P1-T7b | T7 minor | Historical plan names the old `_get_cached_commit_hash` path | obsolete | historical specs are left as written |
| P1-T8e | T8 minor | A url `ModelRef` with empty ref can be constructed | resolved | by design in `identity.py` (ref empty only for an undeclared url hash; `resolve` refuses it) |
| P1-T9a | T9 minor | Nine pre-existing mypy errors | resolved | T15a: pre-commit all-files passes with mypy; `73931a6d` |
| P1-T9b | T9 minor | Incidental import re-sort in a 1.x script | obsolete | harmless; `src/` and `scripts/` restored to `alpha` by D-6 |
| P1-T10a | T10 minor | `_guarded` maps every OSError to corrupt | resolved | `faults.is_transient`, F1 `7e179167..e363dc87`; `store.py:25` |
| P1-T10b | T10 minor | Unwritable root: uncaught PermissionError from quarantine | resolved | H3: `get` skips with a warning, `put` raises `StoreUnavailable`; `owner_remedy` |
| P1-T10d | T10 minor | `quarantine/` has no cleanup | resolved | `collect` covers `quarantine/`, `tmp/` and key dirs (`store.py:477`); the root-level set-aside residue is R-125 |
| P1-T10e | T10 minor | `_ensure_dir` should be public | obsolete | replaced by `StorageDir` (H3/H9b) |
| P1-T13a | T13 minor | `.complete` that is a directory; malformed marker not checked against the manifest | resolved | H3 marker rule; H9b wrong-kind handling (R-100); `dataset-marker@2.0` |
| P1-T14a | T14 minor | `check()` no longer falls back for paths outside the root | obsolete | `scripts/check_fixture_budget.py` rewritten (H6, H9f) and trimmed by D-1 |
| H-H2a | H2 minor | `to_wire` recurses without the shared cycle guard | resolved | H5 total wire path (R-92); `entered` guard |
| H-H4a | H4 deferred | Concurrent second `close()` can reach about 2x the shutdown bound | resolved | H9a one-step-machine shutdown, idempotent from any thread (`worker_client.py:528`, R-99) |
| H-H3a | H3 deferred | `path_key` should use simple case folding | resolved | `fetch.py:129-148`: NFD, upper, casefold, NFC (H8a item 4, H-R35) |
| H-H3c | H3 deferred | Group members can rename or remove a sealed entry (2770 parent, no sticky bit); `os.rmdir` allowance pinned by call name | obsolete | the call-name allowance lint was cut by D-1; the remaining limit is R-125 |
| H-H3d | H3 deferred | `os.link` FileExistsError path keeps an existing blob without re-verification | resolved | H9b/H-R40: blobs hashed from the fd they are written through, linked by directory fd |
| H-H5a | H5 parked | Aggregate oversize is plain ValueError; `_check_depth` re-walks shared DAGs; mechanism docstrings | resolved | H8a item 3; H5 round 1 memoised `to_wire`; `ValueTooLarge` |
| H-H6a | H6 parked | `DYNAMIC_ALLOWED` needs a dead-entry check | obsolete | moved-module guard rewritten without `DYNAMIC_ALLOWED` (H9b) |
| H-H6b | H6 parked | Aggregate code-file cap | resolved | R-85 (H6), fixture budget over every tracked object (H9f) |
| H-H7a | H7 parked | `.data/scripts` refusal, wheel RECORD and `entry_points.txt` checks | resolved | H8a item 6; R-107 gate completeness (H9f) |
| H-H7b | H7 parked | Modified-tracked-file in tagged CI: document only | obsolete | release gate reworked by D-5 (`a3ea1894`) |
| H-H7c | H7 parked | H4 flake `test_running_leader_is_stopped_without_asking_ps` on macOS | resolved | H8a item 1: root cause was the test's EPERM probe, now reads `ps` state |
| H-H6c | H6 concern | `requests`/`httpx` imported optionally and undeclared | resolved | F1 added `httpx` to core's dependencies |
| H-H9a4 | H9a concern | Ctrl-C held up to the shutdown bound; handler `close()` on the run thread | resolved | documented in the `close()` docstring and `wire-protocol.md` |
| H-H9b1 | H9b concern | macOS cannot apply `RLIMIT_AS` | resolved | stated in `testing.md`; Linux applies it (`bounded_probe_fixture.py`) |
| H-H9b4 | H9b concern | `store@3` trees not collected | obsolete | pre-release layout, no released store at that version (pre-alpha rule: replace outright) |
| H-H9b5 | H9b concern | Hard-linking a live lock breaks exclusion | resolved | ADR-003 states it as not guaranteed; `owns()` for irreversible steps |
| H-H9b6 | H9b concern | Heartbeat write can land after release | obsolete | D-2: leases by rename and `O_EXCL`, no heartbeat thread |
| H-H9c2 | H9c N-2 | Two frozen mapping types (`FrozenMapping`, `FrozenDict`) | resolved | ruling H-R45, documented in ADR-006 |
| H-H9c3 | H9c medium | `EnumMember` `eq=False` makes `recorded(x) == recorded(x)` False | resolved | H9-cleanup (`11992718..3c0e0cdb`), ADR-006 read-back equality |
| H-H9d1 | H9d concern | senselab-core wheel ships without `data/*.yaml` | resolved | H9d fix round 1: package data artifacts in all three pyprojects; `package_data_test` |
| H-H9e1 | H9e concern | Computed-name `getattr`, `vars(m)[..]`, non-literal `__import__` are not decidable statically | resolved | recorded in ADR-006 and R-106 as the residual |
| H-H9f1 | H9f concern | Member derivation unverified against a real hatch build | resolved | `package_data_test` and the tagged-workspace tests; 2.x CI green at `97b617c5` |
| H-H11a | H11b concern | Later-boundary sweep, env tier and ORCD runs not made locally | obsolete | Phase 1 exit item (STATE, next actions 3); not parked |
| H-H11c | H11b concern | Linux `/proc` liveness branch unexercised | resolved | Linux CI green at `b76681a9` |
| H-D2a | D2 concern | Stale-break race window cannot be closed on POSIX | resolved | ADR-003 amended: correctness rests on idempotence; overlap only with a displaced holder that knows it (`2c899203`) |
