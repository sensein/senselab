# Contract: Shared HF Model-Cache API

This library's external interface is a Python API in `src/senselab/utils/hf_model_cache.py` (re-exported from `dependencies.py` for back-compat). Signatures are the contract; behavior clauses map to FRs/SCs and become the acceptance tests.

---

## `resolve_model(repo_id, ref="main", *, repo_type="model", token=None) -> ResolvedModel`

The single entry point. Ensures the model is present and version-verified, without avoidable Hub traffic, honoring freeze precedence.

**Behavior contract**
- **C1 (FR-001/SC-002)**: if a valid `resolved_sha` is known and fresh (system freeze, or run freeze, or `now - last_verified_epoch < window`), returns with `from_cache=True` and performs **zero** network calls.
- **C2 (FR-002/SC-003)**: if the SHA's snapshot is absent, downloads it exactly once across concurrent processes (`_HeartbeatLock` + `snapshot_download`); concurrent callers reuse it.
- **C3 (FR-005/006/SC-005)**: when the window has lapsed (and no freeze), performs **at most one** coordinated `HfApi().model_info(repo_id, revision=ref, expand=["sha"]).sha`; double-checked re-read collapses concurrent stale-finders into one call. Adopts a changed SHA; otherwise bumps `last_verified_epoch`.
- **C4 (FR-003/004/015/SC-004)**: guarantees `ResolvedModel.resolved_sha` is a full commit SHA and that `snapshot_path` contains that SHA's snapshot; never returns a different/stale version silently.
- **C5 (FR-007/SC-007)**: if the Hub is unreachable during a due re-check but a valid cached SHA exists, returns the cached SHA and logs a warning (no hard failure).
- **C6 (FR-008)**: if the model/ref is neither cached nor retrievable, raises an error naming `repo_id` and `ref` (`RepositoryNotFoundError` / `RevisionNotFoundError`). `GatedRepoError` is raised, not cached.
- **C7 (FR-011)**: two distinct refs of the same repo resolve to distinct records/paths, no cross-contamination.
- **C8 (FR-013/014, precedence)**: `SENSELAB_HF_FREEZE=1` ⇒ no network, SHA from record/local refs; else active `run_version_freeze()` ⇒ run-pinned SHA; else the window governs.

## `load_hf_resilient(loader, *args, repo_id, ref="main", pass_revision=True, **kwargs) -> T`

Loader-agnostic in-process wrapper. Resolves first (`resolve_model`), then invokes `loader`:
- if `pass_revision=True`: injects `revision=resolved_sha, local_files_only=True` into `kwargs` (transformers `from_pretrained`/`pipeline`, `SentenceTransformer`);
- if `pass_revision=False`: the caller uses `ResolvedModel.snapshot_path` as the loader's source (SpeechBrain/pyannote/NeMo).
Retries the load on residual transient errors (429/5xx/timeout). **Never** relies on `HF_HUB_OFFLINE` env toggling for suppression (research D1).

## `hf_subprocess_env(repo_id, revision="main", *, also=None, base_env=None) -> dict` (reworked)

For subprocess-venv workers. Resolves each referenced model (`resolve_model`), then returns `base_env` copy with `HF_HUB_OFFLINE`/`TRANSFORMERS_OFFLINE=1` **and** `SENSELAB_HF_RESOLVED_<n>` hints so the child loads SHA-pinned. Correct because the child imports fresh with the env already set. Falls back to unchanged env if a model cannot be staged (child may download online).

## `run_version_freeze() -> ContextManager[None]`

Enters a run-scoped freeze: the first `resolve_model` per `(repo_id, ref)` inside the block records its SHA in a `ContextVar` dict; subsequent resolutions in the block reuse it regardless of window lapse (FR-013 / SC-008). Re-entrant; nesting shares the same dict.

## `ResolvedModel` (dataclass)

Fields per data-model E2: `repo_id, requested_ref, resolved_sha, snapshot_path, last_verified_epoch, from_cache`.

## Retained primitives (unchanged public behavior)

`ensure_hf_model`, `is_hf_model_cached`, `retry_on_transient_error`, `speechbrain_savedir`, `speechbrain_loading_cwd`, `_HeartbeatLock` — kept; `resolve_model` builds on them.

---

## Contract test matrix (Phase 2 → tasks)

| Test | Asserts | FR/SC |
|---|---|---|
| `test_fresh_within_window_no_network` | monkeypatched `model_info` never called when record fresh | C1 / SC-002 |
| `test_stale_triggers_single_recheck` | `model_info` called exactly once across N threads | C3 / SC-005 |
| `test_changed_sha_adopted` | new SHA downloaded + record rewritten after window | C3 |
| `test_version_mismatch_fails_loud` | raises naming repo+ref; no silent substitution | C4 / SC-004 |
| `test_hub_unreachable_uses_cache` | cached SHA returned + warning on network error | C5 / SC-007 |
| `test_missing_model_clear_error` | `Repository/RevisionNotFoundError` names repo+ref | C6 / FR-008 |
| `test_gated_not_cached` | `GatedRepoError` raised, record not written | C6 |
| `test_system_freeze_no_network` | `SENSELAB_HF_FREEZE=1` ⇒ zero Hub calls across runs | C8 / SC-009 |
| `test_run_freeze_stable_mid_run` | SHA constant across window lapse inside `run_version_freeze()` | C8 / SC-008 |
| `test_two_refs_no_crosstalk` | distinct records/paths per ref | C7 / FR-011 |
| `test_download_once_concurrent` | one `snapshot_download` across N first-time callers | C2 / SC-003 |
| per-backend `test_<backend>_behavior_preserved` | migrated backend output unchanged vs baseline | FR-009 / SC-010 |
