# Migration ledger (P-39)

One row per 1.x area: the last `alpha` commit whose 1.x changes are reconciled into 2.x. Update it
in the same PR as the reconciliation. A 1.x module may be deleted only when its mark equals
`alpha`.

Base: the design branch was cut from `alpha` at `19834fdd`. No commits had landed on `alpha`
since then as of 2026-10-30.

| Area (1.x paths) | Mark | Reconciled changes (commit → 2.x handling) |
|---|---|---|
| `src/senselab/audio/tasks/` | `19834fdd` | — |
| `src/senselab/video/tasks/`, `src/senselab/text/tasks/` | `19834fdd` | — |
| `src/senselab/audio/workflows/audio_analysis/` | `19834fdd` | — |
| `src/senselab/audio/workflows/triage/` | `19834fdd` | — |
| `src/senselab/utils/` | `19834fdd` | — |
| `scripts/`, `tutorials/` | `19834fdd` | — |

## Untagged digests in 1.x (H-R20)

The strict form of the tagged-digest guard (`untagged_digests(..., strict=True)` in
`packages/senselab-core/tests/lint/tagged_digest_test.py`), run over every allowlisted module outside
`packages/`, reports 28 sites where `json.dumps`, `str`, `repr` or `format` output reaches a hash call. They are
migration debt under ADR-006, not fixed in Phase 1: each is ported when its module moves (P-39). Line numbers are
at `deb84049`. Replacements name the 2.x mechanism: `tagged_digest(<tag>, value)` from
`senselab_worker.canonical` with a tag registered in `DIGEST_TAGS`, `sha256_bytes`/`sha256_file` from
`senselab_worker.hashing` for bytes that are already canonical, or a typed identity from
`senselab_core.kernel.identity`.

An earlier count of 32 included four `dict.update` calls fed by `str()` (`cohort_stage.py:317`,
`recording_vectors.py:1144`, `speaker_vectors.py:479`, `reproduce/source.py:830`). The guard now counts
`.update` only on a receiver a hash constructor made, so they are no longer reported and have no row.

| Site | Function | What it keys | 2.x replacement |
|---|---|---|---|
| `scripts/free_speech_review_page.py:496` | `finding_key` | review-page finding id from stem, categories and extent (`sha1` of a pipe-joined string) | `tagged_digest` over `[stem, categories, start, end]` under a registered review-finding tag |
| `scripts/verify_grid_unification.py:232` | `check` | printed fingerprint of word texts and confidences, a diagnostic only | `tagged_digest` over the same mapping, or drop: it keys nothing |
| `src/senselab/audio/workflows/audio_analysis/adaptive/policy.py:62` | `load_policy` | `policy_hash` of the merged adaptive policy | `tagged_digest` under a registered policy tag; the policy becomes a `Frozen` config with `canonical_digest()` |
| `src/senselab/audio/workflows/audio_analysis/run_config.py:301` | `load_run_config` | `config_hash` of the merged run config (`default=str` stringifies non-JSON values) | `ExecutionContext` config field digested by `tagged_digest("context@2", …)`; non-JSON values refused, not stringified |
| `src/senselab/audio/workflows/triage/cohort_stage.py:144` | `_digest` | cohort-stage parameter and input digests (`default=str`) | `tagged_digest` under a registered cohort tag; the stage's key becomes a `LogicalKey` |
| `src/senselab/audio/workflows/triage/cohort_stage.py:1715` | `write_quantile_artefact` | sha256 of the artefact file's bytes, written as indented `json.dumps` | write the artefact as JCS (`jcs_dumps`) and hash it with `sha256_bytes`, or key it by `tagged_digest` over the record |
| `src/senselab/audio/workflows/triage/config.py:203` | `load_triage_config` | `config_hash` of the merged triage config (`default=str`) | as `load_run_config`: one config identity through `tagged_digest("context@2", …)` |
| `src/senselab/audio/workflows/triage/review_page/page.py:259` | `index_of` | review-page build id over the page index (`default=str`) | `tagged_digest` under a registered review-page tag |
| `src/senselab/audio/workflows/triage/task_audio.py:293` | `cut_key` | cache key of a task-audio cut: input digest, definition, sample range, extras (`default=str`) | `LogicalKey` of the cut node, digested by `tagged_digest("logical-key@2", …)` |
| `src/senselab/audio/workflows/triage/task_squim.py:322` | `measure_key` | cache key of a SQUIM measurement: input, extent, windows, weights | `LogicalKey` with the weights as a `ModelIdentity` (`weights@2`) |
| `src/senselab/utils/reproduce/build.py:663` | `build_config_sha256` | key of an sdist build configuration | `tagged_digest` under a registered reproduce-build tag |
| `src/senselab/utils/reproduce/env.py:191` | `env_key` | key of a reproduce environment spec | `tagged_digest` under a registered reproduce-env tag |
| `src/senselab/utils/reproduce/env.py:263` | `_write_json` | sha256 of a JSON marker file as written (indented `json.dumps`) | write JCS and hash the bytes with `sha256_bytes` |
| `src/senselab/utils/reproduce/env.py:420` | `checks_digest` | digest of the venv checks (`pip_check_expected`, `smoke_imports`) | `tagged_digest` under the reproduce-env tag |
| `src/senselab/utils/reproduce/env.py:490` | `kernels_sha256` | digest of an installed kernel tree's listing | a content root over the tree (`content-root@2`) |
| `src/senselab/utils/reproduce/run.py:1736` | `_r_env` | recorded sha of the R session environment | `tagged_digest` under a registered run-env tag |
| `src/senselab/utils/reproduce/run.py:2219` | `_record` | `argv_sha` of a step's argument vector in provenance | `tagged_digest` under the run tag |
| `src/senselab/utils/reproduce/sandbox.py:413` | `_sha` | sha of a sandbox environment mapping | `tagged_digest` under the run-env tag |
| `src/senselab/utils/reproduce/source.py:345` | `patch_set_sha256` | digest of a patch set | `tagged_digest` under a registered patch-set tag |
| `src/senselab/utils/subprocess_venv.py:262` | `venv_environment` | `dependencies_digest` over a venv's full distribution listing | `tagged_digest` under a registered environment tag; `Environment.dependencies_digest` in `ProvStore` |
| `src/senselab/utils/tasks/cached_inference.py:353` | `audio_signature` | audio identity: `str(sampling_rate)` into a hash update | `ContentIdentity` of the audio (`audio@2` over the canonical `.npy`) |
| `src/senselab/utils/tasks/cached_inference.py:355` | `audio_signature` | audio identity: `str(shape)` into the same hash | as the row above: one `audio@2` identity |
| `src/senselab/utils/uv_tools.py:88` | `requirements_sha256` | digest of a requirements set with its header fields | `tagged_digest` under the environment tag |
| `src/senselab/utils/venv_worker.py:116` | `_digest` | short digest of a worker call payload (`default=str`) | the worker wire's canonical form: `tagged_digest("value@1", …)` with non-JSON values refused |
| `src/senselab/utils/vllm_server.py:476` | `_path_label` | path digest shown in an error message instead of the name, a label only | a label, not a key: `tagged_digest` under a registered label tag, or `sha256_bytes` of the path's UTF-8 |
| `src/senselab/utils/vllm_server.py:527` | `content_part` | digest of the media directory shown in an error message, a label only | as the row above |
| `src/senselab/utils/work_pool.py:286` | `Item.payload_digest` | idempotence key of a work-pool item | `tagged_digest` under a registered work-item tag |
| `src/senselab/utils/work_pool.py:1315` | `pool_tag` | eight-hex pool name in Slurm job names, from the resolved root path | a label, not a key: `tagged_digest` under a registered label tag |
