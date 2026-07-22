# Phase 0 Research: HuggingFace Model Cache & Version Consistency

All findings grounded in the installed packages: `huggingface_hub` 1.11.0, `transformers` 4.55.4, `speechbrain` 1.0.3, `pyannote.audio` 3.4.0, and the existing `src/senselab/utils/dependencies.py`.

---

## D1. How to load without hitting the Hub (the core of FR-001/002)

**Decision**: Resolve the requested ref to a full commit SHA once, then load with `revision=<sha>, local_files_only=True` (in-process) or a fresh-import offline subprocess env (workers). Do **not** rely on mid-process `os.environ["HF_HUB_OFFLINE"]` toggling.

**Rationale**: In `huggingface_hub` 1.11 / `transformers` 4.55, `constants.HF_HUB_OFFLINE` and transformers' `_is_offline_mode` are evaluated at **import time** (`constants.py:171`; `transformers/utils/hub.py`). Setting the env var *after* import does not reliably suppress the network `repo_info`/HEAD check — which is precisely the call that 429s. This is the concrete defect behind the maintainer's "this PR doesn't pass my smell test" on #527. By contrast, `local_files_only=True` is honored at call time and, combined with a full-SHA `revision`, resolves purely from `snapshots/<sha>/` with **zero** network I/O.

**Alternatives considered**:
- *Env-var toggling in-process (`hf_offline_loading` today)* — rejected: import-time capture makes it unreliable in current versions.
- *Setting `huggingface_hub.constants.HF_HUB_OFFLINE = True` at runtime* — works (read dynamically by `is_offline_mode()`) but is a global mutation with the same concurrency hazards as env toggling; SHA-pinning is cleaner and per-call.
- *Subprocess env (`hf_subprocess_env`)* — kept for workers: the child imports fresh with the env already set, so it is correct. Extended to also inject the resolved SHA.

## D2. Determining version identity at the cache/hub layer (FR-003/004/015)

**Decision**: Requested ref → SHA via `HfApi().model_info(repo_id, revision=ref, expand=["sha"]).sha` (one lightweight `GET /api/models/{id}/revision/{ref}`). Local SHA via the `refs/<ref>` cache file / `try_to_load_from_cache(...).parent.name` — no network. Verification = compare the local snapshot's resolved SHA against the requested/resolved SHA; loaders that expose no `revision` param are simply pointed at the version-identified local snapshot dir, so verification never depends on the loader.

**Rationale**: Satisfies FR-015's "cache/hub-layer, loader-independent" requirement exactly. `expand=["sha"]` trims the payload (mutually exclusive with `files_metadata`/`securityStatus`). Both calls raise `RepositoryNotFoundError` / `RevisionNotFoundError`, giving the clear, actionable errors FR-008 needs.

**Alternatives considered**: reading the loader's own reported version — rejected (loaders like SpeechBrain/pyannote/NeMo expose none, and it would make verification loader-dependent, violating FR-015).

## D3. Freshness window without per-load Hub calls (FR-005/006, US3)

**Decision**: A per-`(repo_id, ref)` sidecar JSON stores `resolved_sha` + `last_verified_epoch`. A load re-checks the Hub only when `now - last_verified_epoch >= window` (default `7*86400`s). The re-check is serialized by the existing `_HeartbeatLock`; waiters re-read the sidecar after acquiring (double-checked locking) so concurrent stale-finders collapse into **exactly one** `model_info` call (FR-006). Unchanged SHA → bump timestamp; changed SHA → `snapshot_download(revision=new_sha)` then rewrite record.

**Rationale**: The window lives in the shared filesystem record (visible to all nodes sharing the cache), making it coordinated rather than per-process. Reuses `_HeartbeatLock` (heartbeat + stale-break) already in `dependencies.py`. Atomic write-then-`os.replace` prevents torn records.

**Alternatives considered**: per-process in-memory TTL — rejected (not coordinated across the 100+ jobs that cause 429s); checking on every load — rejected (the originating problem).

## D4. Freeze scopes and precedence (FR-013/014, US3 scenarios 3–4)

**Decision**:
- **System freeze** — env `SENSELAB_HF_FREEZE=1`: skip all Hub calls; resolve SHA from the sidecar (or local `refs/<ref>`) and load SHA-pinned local-only. Highest precedence.
- **Run freeze** — a `run_version_freeze()` context manager backed by a `ContextVar` dict `(repo_id, ref) → sha`: the first resolution per repo in the run is recorded and reused for the run's duration, so no SHA shifts mid-run even if the window lapses.
- **Default** — the 7-day window (D3).
- Precedence system ⊇ run ⊇ default, enforced by checking system freeze first, then run dict, then window.

**Rationale**: Directly encodes the clarified policy. `ContextVar` is async/thread-safe and auto-scopes to the `with` block. System freeze is a single cluster/job-level switch for reproducible-science runs (SC-009); run freeze keeps a multi-day batch internally consistent (SC-008).

**Alternatives considered**: a single global freeze flag — rejected (can't express "stable within this run but re-checkable across runs"); a config file — deferred (env var matches existing `SENSELAB_HF_*` conventions and is simplest to set per SLURM job).

## D5. Uniform loading across heterogeneous loaders (FR-009/010, US4)

**Decision**: One resolve-then-load pattern. `resolve_model(repo_id, ref)` returns `(resolved_sha, snapshot_path)`; backends then either (a) pass `revision=sha, local_files_only=True` to `from_pretrained`/`pipeline`/`SentenceTransformer` (loaders with a `revision` param), or (b) pass the local `snapshot_path` as the source (SpeechBrain `source=`, pyannote `checkpoint=`, NeMo `.nemo` path) for loaders without one. Subprocess workers receive the SHA + offline env via `hf_subprocess_env`.

**Rationale**: A single entry point with two thin adapters covers all HF-backed families (evidence: research table of per-loader local-path support). New backends implement only the model-specific load step (SC-006).

**Gotchas baked into the plan**:
1. **pyannote**: `revision` + local path raises `ValueError` — pass the local dir *without* `revision`. Diarization pipelines fan out to sub-models (segmentation, embedding) by their own repo ids → pre-stage the whole set, not just the top-level repo.
2. **SpeechBrain**: writes into CWD; keep `speechbrain_savedir()` + `speechbrain_loading_cwd()`; point `source` at `snapshots/<sha>/`; `local_strategy=COPY` on no-symlink filesystems.
3. **NeMo**: no Hub `revision`; pin by the pre-staged `.nemo` artifact path and control offline via the subprocess env route.
4. **Gated repos**: keep the existing rule — never cache `GatedRepoError` as a definitive failure; treat `model_info` auth errors during a re-check as non-fatal (don't overwrite a good cached SHA).
5. **`_CACHED_NO_EXIST`**: compare `try_to_load_from_cache` results against the sentinel, not truthiness.
6. **Symlink cache layout**: `snapshots/<sha>/*` are symlinks into `blobs/`; detect corruption via `scan_cache_dir().warnings` (partial/aborted downloads).

## D6. Non-HF-Hub loaders (scope boundary)

**Decision**: TF-Hub (yamnet), s3prl, SPARC, and Coqui backends are **out of scope for HF version verification** (no HF Hub version concept — matches spec Assumptions/Out-of-Scope), but they already run through `ensure_venv`; leave their download behavior unchanged. Document them as explicitly excluded so SC-010 ("every existing model backend loads through the shared mechanism") is scoped to HF-backed backends.

**Rationale**: Spec states a source with no version concept at all is out of scope. Forcing them through the HF resolver would add complexity with no 429 benefit.

## D7. Defects in the existing (pre-#527) mechanism this rework must fix, not inherit

A high-effort review of the pre-#527 code (`utils/data_structures/model.py` + the download-once helpers in `utils/dependencies.py`) found the "already implemented" caching does download-once coordination but has concrete defects. The rework must **fix** these, not reuse them as-is:

| Pre-#527 defect | Why it bites | Design response |
|---|---|---|
| **`_HeartbeatLock` stale-break `unlink()`s the lockfile then re-acquires** | A live-but-CPU-starved holder (routine under 100-job load) is misjudged dead; `filelock` then locks a *new* inode at the same path → two processes write the same cache dir → **corruption**. The heartbeat proves only that the daemon thread scheduled, not that the download is healthy. | Replace with a **safe lease**: record owner identity (pid/host) + heartbeat; only take over when the owner is **provably gone**; never unlink a lock held by a live owner; bound the wait. (new task T010a) |
| **`ensure_hf_model` returns a branch name, not a SHA** (`_get_cached_commit_hash` falls through to `return revision`) | The whole rework pins by *immutable SHA*; trusting this return would pin to `"main"`. | `resolve_model` derives the SHA **independently** via `model_info` / the `refs/<ref>` file and never trusts `ensure_hf_model`'s return. (T008) |
| **`is_hf_model_cached` returns True under `HF_HUB_OFFLINE=1` for uncached models** | Reports success for a model that isn't present → confusing deep loader failure. | Cache-identity check verifies an actual snapshot exists; offline mode never fabricates presence. (T008) |
| **`config.json` presence treated as "fully cached"** | A partial/aborted snapshot is treated as complete and never re-downloaded (no self-heal). | Corruption-gated fast path: `scan_cache_dir().warnings` / missing-file check → treat as absent and re-obtain. (T008) |
| **No TTL anywhere** — cached models never re-checked; error results cached forever | Silent version staleness; a since-published revision keeps failing. | Bounded 7-day window on both success and error records (FR-004/005; data-model E3). |
| **Only guards download + existence validation, never the load-time revision check** | The actual `from_pretrained`/`pipeline`/`from_hparams` still HEADs the Hub → the 429 source was never covered. | SHA-pin + `local_files_only=True` at the loader (D1); subprocess offline env for workers. |
| **`_write_result_cache` is non-atomic** | Crash/concurrent write → truncated JSON → silent re-fetch. | Atomic write-then-`os.replace` (T005). |

**Conclusion:** the 429 problem is not merely "not every model goes through the existing channel." Even full adoption of the pre-#527 channel would still 429 (it never covered the load path) and could corrupt the cache (the lock). A refactor, not a patch, is warranted.

---

## Resolved unknowns

| Unknown | Resolution |
|---|---|
| Resolve ref→SHA in one call? | `HfApi().model_info(repo_id, revision=ref, expand=["sha"]).sha` |
| Read cached SHA offline? | `refs/<ref>` file / `try_to_load_from_cache(...).parent.name` / `scan_cache_dir` |
| Reliable in-process offline? | SHA-pin + `local_files_only=True` (NOT env toggling — import-time capture) |
| Download-once coordination? | `snapshot_download` per-file `WeakFileLock` + senselab `_HeartbeatLock` per `(repo,ref)` |
| Coordinated single re-check? | shared sidecar `last_verified_epoch` + `_HeartbeatLock` + double-checked re-read |
| Freeze precedence? | env `SENSELAB_HF_FREEZE` (system) ⊇ `run_version_freeze()` ContextVar (run) ⊇ 7-day window |
| Loaders without `revision`? | point at local `snapshot_path` (SpeechBrain/pyannote/NeMo) |

**No `[NEEDS CLARIFICATION]` markers remain.**
