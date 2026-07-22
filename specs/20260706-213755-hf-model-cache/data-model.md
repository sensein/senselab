# Phase 1 Data Model: HuggingFace Model Cache & Version Consistency

Entities derived from the spec's **Key Entities** plus the resolved design (research.md). Storage is filesystem-only; no database. All persisted records carry an explicit `schema_version` and are written atomically (write temp + `os.replace`).

---

## E1. ModelReference

What the caller asks for. Not persisted (constructed per call); usually derived from an existing `HFModel`.

| Field | Type | Notes |
|---|---|---|
| `repo_id` | `str` | e.g. `"openai/whisper-large-v3"` |
| `requested_ref` | `str` | selector: exact sha, branch, tag, or channel (`"main"`). Default `"main"`. |
| `repo_type` | `str` | `"model"` (default) / `"dataset"` / `"space"` |

**Validation**: `repo_id` non-empty and of form `org/name` (or a bare id). `requested_ref` non-empty. Invalid → `ValueError` before any Hub/cache access.

## E2. ResolvedModel

The concrete, immutable identity a `ModelReference` maps to, plus where it lives locally. Returned by `resolve_model()`; consumed by every loader. Not persisted directly (its durable projection is E3).

| Field | Type | Notes |
|---|---|---|
| `repo_id` | `str` | echoes the reference |
| `requested_ref` | `str` | echoes the reference |
| `resolved_sha` | `str` | full 40-hex commit SHA — the version everything is verified against |
| `snapshot_path` | `pathlib.Path` | local `…/models--org--name/snapshots/<sha>/` — for loaders without a `revision` param |
| `last_verified_epoch` | `int` | when `resolved_sha` was last confirmed against the Hub |
| `from_cache` | `bool` | `True` if resolved without any network call this load |

**Invariant**: `resolved_sha` is always a full commit SHA (never a branch/tag) once resolution succeeds — so downstream `from_pretrained(revision=resolved_sha)` / `snapshot_path` loads are immutable and reproducible.

## E3. CachedModelRecord (persisted sidecar)

The durable freshness/version record. One JSON file per `(repo_id, requested_ref)` at `{HF_HOME}/senselab_cache/<safe_key>.json`, where `<safe_key> = "<org>--<name>--<ref>"` (existing `_safe_key`). Extends the file `_write_result_cache` writes today by adding `resolved_sha` + `last_verified_epoch`.

```json
{
  "schema_version": 1,
  "repo_id": "openai/whisper-large-v3",
  "requested_ref": "main",
  "resolved_sha": "1ecca609b4600b02d5c0f68e6c1f8b2a3d4e5f60",
  "last_verified_epoch": 1737504000,
  "status": "ok",
  "error_type": null,
  "error_message": null
}
```

| Field | Type | Notes |
|---|---|---|
| `schema_version` | `int` | currently `1`; bump on incompatible change |
| `repo_id`, `requested_ref` | `str` | identity |
| `resolved_sha` | `str \| null` | full SHA when `status == "ok"` |
| `last_verified_epoch` | `int` | freshness anchor (Unix seconds) |
| `status` | `"ok" \| "error"` | definitive-failure caching (existing behavior) |
| `error_type`, `error_message` | `str \| null` | for reconstructing `Repository/RevisionNotFoundError`; **never** set for `GatedRepoError` |

**State transitions**
```
(absent) --resolve--> ok{sha, now}
ok{sha, t} --load within window (now-t < W)--> ok{sha, t}            # no network
ok{sha, t} --window lapsed, hub sha unchanged--> ok{sha, now}        # 1 coordinated call, timestamp bump
ok{sha, t} --window lapsed, hub sha changed--> ok{sha', now}         # 1 call + snapshot_download(sha')
(absent) --repo/rev not found--> error{type,msg}                     # cached; blocks repeat API calls
error{...} --caller gains access / repo appears--> (re-resolve)      # gated errors are NOT cached, so retried
```

**Validation / integrity**: reject a record whose `schema_version` is unknown (treat as absent → re-resolve). If `status=="ok"` but the referenced `snapshots/<sha>/` is missing/corrupt (per `scan_cache_dir` warnings or a `try_to_load_from_cache` miss), treat as absent and re-obtain (FR: partial/corrupt cache detection).

## E4. FreezeState (in-memory, not persisted)

Controls whether re-checks may occur and pins SHAs. Precedence **system ⊇ run ⊇ default**.

| Scope | Mechanism | Effect |
|---|---|---|
| System | env `SENSELAB_HF_FREEZE=1` | all loads SHA-pinned local-only; zero re-checks across all runs (SC-009) |
| Run | `run_version_freeze()` context manager → `ContextVar[dict[(repo_id,ref), sha]]` | first resolution per repo recorded; reused for the run's duration; no mid-run SHA change (SC-008) |
| Default | `DEFAULT_FRESHNESS_WINDOW_DAYS = 7` (env `SENSELAB_HF_FRESHNESS_DAYS`) | re-check only after the window |

## Configuration constants (FR-012 / Constitution III)

| Name | Default | Override | Meaning |
|---|---|---|---|
| `DEFAULT_FRESHNESS_WINDOW_DAYS` | `7` | `SENSELAB_HF_FRESHNESS_DAYS` | days before a coordinated version re-check |
| `SENSELAB_HF_MAX_RETRIES` | `3` | env (existing) | transient-error retry attempts |
| `SENSELAB_HF_FREEZE` | unset | env | system-level version freeze (`1` = on) |
| heartbeat interval / stale threshold | `30` / `90` s | (existing, `_HeartbeatLock`) | cross-process download-lock liveness |

All are named module constants documented at their definition site and surfaced in the PR description.
