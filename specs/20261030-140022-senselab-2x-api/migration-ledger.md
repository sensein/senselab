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

The strict form of the tagged-digest data-flow lint (`untagged_digests(..., strict=True)`, deleted in H9e with
R-106), run over every allowlisted module outside `packages/`, reported 28 sites where `json.dumps`, `str`,
`repr` or `format` output reaches a hash call. They are migration debt under ADR-006, not fixed in Phase 1: each
is ported when its module moves (P-39). Line numbers are at `deb84049`. Replacements name the 2.x mechanism:
`identity_digest(<tag>, value)` from `senselab_worker.digest` with a tag registered in `IDENTITY_TAGS`,
`blob_digest(<tag>, data)` with a tag in `BLOB_TAGS` for bytes that are already canonical, or a typed identity
from `senselab_core.kernel.identity`. The lint is gone; the table is the list it produced.

An earlier count of 32 included four `dict.update` calls fed by `str()` (`cohort_stage.py:317`,
`recording_vectors.py:1144`, `speaker_vectors.py:479`, `reproduce/source.py:830`). The guard now counts
`.update` only on a receiver a hash constructor made, so they are no longer reported and have no row.

| Site | Function | What it keys | 2.x replacement |
|---|---|---|---|
| `scripts/free_speech_review_page.py:496` | `finding_key` | review-page finding id from stem, categories and extent (`sha1` of a pipe-joined string) | `identity_digest` over `[stem, categories, start, end]` under a registered review-finding tag |
| `scripts/verify_grid_unification.py:232` | `check` | printed fingerprint of word texts and confidences, a diagnostic only | `identity_digest` over the same mapping, or drop: it keys nothing |
| `src/senselab/audio/workflows/audio_analysis/adaptive/policy.py:62` | `load_policy` | `policy_hash` of the merged adaptive policy | `identity_digest` under a registered policy tag; the policy becomes a `Frozen` config with `canonical_digest()` |
| `src/senselab/audio/workflows/audio_analysis/run_config.py:301` | `load_run_config` | `config_hash` of the merged run config (`default=str` stringifies non-JSON values) | `ExecutionContext` config field digested by `identity_digest("context@3", …)`; non-JSON values refused, not stringified |
| `src/senselab/audio/workflows/triage/cohort_stage.py:144` | `_digest` | cohort-stage parameter and input digests (`default=str`) | `identity_digest` under a registered cohort tag; the stage's key becomes a `LogicalKey` |
| `src/senselab/audio/workflows/triage/cohort_stage.py:1715` | `write_quantile_artefact` | sha256 of the artefact file's bytes, written as indented `json.dumps` | write the artefact as JCS (`jcs_dumps`) and hash it with `sha256_bytes`, or key it by `identity_digest` over the record |
| `src/senselab/audio/workflows/triage/config.py:203` | `load_triage_config` | `config_hash` of the merged triage config (`default=str`) | as `load_run_config`: one config identity through `identity_digest("context@3", …)` |
| `src/senselab/audio/workflows/triage/review_page/page.py:259` | `index_of` | review-page build id over the page index (`default=str`) | `identity_digest` under a registered review-page tag |
| `src/senselab/audio/workflows/triage/task_audio.py:293` | `cut_key` | cache key of a task-audio cut: input digest, definition, sample range, extras (`default=str`) | `LogicalKey` of the cut node, digested by `identity_digest("logical-key@3", …)` |
| `src/senselab/audio/workflows/triage/task_squim.py:322` | `measure_key` | cache key of a SQUIM measurement: input, extent, windows, weights | `LogicalKey` with the weights as a `ModelIdentity` (`weights@3`) |
| `src/senselab/utils/reproduce/build.py:663` | `build_config_sha256` | key of an sdist build configuration | `identity_digest` under a registered reproduce-build tag |
| `src/senselab/utils/reproduce/env.py:191` | `env_key` | key of a reproduce environment spec | `identity_digest` under a registered reproduce-env tag |
| `src/senselab/utils/reproduce/env.py:263` | `_write_json` | sha256 of a JSON marker file as written (indented `json.dumps`) | write JCS and hash the bytes with `sha256_bytes` |
| `src/senselab/utils/reproduce/env.py:420` | `checks_digest` | digest of the venv checks (`pip_check_expected`, `smoke_imports`) | `identity_digest` under the reproduce-env tag |
| `src/senselab/utils/reproduce/env.py:490` | `kernels_sha256` | digest of an installed kernel tree's listing | a content root over the tree (`content-root@2`) |
| `src/senselab/utils/reproduce/run.py:1736` | `_r_env` | recorded sha of the R session environment | `identity_digest` under a registered run-env tag |
| `src/senselab/utils/reproduce/run.py:2219` | `_record` | `argv_sha` of a step's argument vector in provenance | `identity_digest` under the run tag |
| `src/senselab/utils/reproduce/sandbox.py:413` | `_sha` | sha of a sandbox environment mapping | `identity_digest` under the run-env tag |
| `src/senselab/utils/reproduce/source.py:345` | `patch_set_sha256` | digest of a patch set | `identity_digest` under a registered patch-set tag |
| `src/senselab/utils/subprocess_venv.py:262` | `venv_environment` | `dependencies_digest` over a venv's full distribution listing | `identity_digest` under a registered environment tag; `Environment.dependencies_digest` in `ProvStore` |
| `src/senselab/utils/tasks/cached_inference.py:353` | `audio_signature` | audio identity: `str(sampling_rate)` into a hash update | `ContentIdentity` of the audio (`audio@3` over the canonical `.npy`) |
| `src/senselab/utils/tasks/cached_inference.py:355` | `audio_signature` | audio identity: `str(shape)` into the same hash | as the row above: one `audio@3` identity |
| `src/senselab/utils/uv_tools.py:88` | `requirements_sha256` | digest of a requirements set with its header fields | `identity_digest` under the environment tag |
| `src/senselab/utils/venv_worker.py:116` | `_digest` | short digest of a worker call payload (`default=str`) | the worker wire's canonical form: `identity_digest("value@2", …)` with non-JSON values refused |
| `src/senselab/utils/vllm_server.py:476` | `_path_label` | path digest shown in an error message instead of the name, a label only | a label, not a key: `identity_digest` under a registered label tag, or `sha256_bytes` of the path's UTF-8 |
| `src/senselab/utils/vllm_server.py:527` | `content_part` | digest of the media directory shown in an error message, a label only | as the row above |
| `src/senselab/utils/work_pool.py:286` | `Item.payload_digest` | idempotence key of a work-pool item | `identity_digest` under a registered work-item tag |
| `src/senselab/utils/work_pool.py:1315` | `pool_tag` | eight-hex pool name in Slurm job names, from the resolved root path | a label, not a key: `identity_digest` under a registered label tag |

## Modules allowed to import a digest module (H-R19, H9e)

The import contract `digests-are-tagged` (`.importlinter`, ADR-006) forbids every root package to import
`hashlib`, `hmac` and the other digest modules. Its allowances outside `senselab_worker.digest` are these
1.x modules and scripts; each leaves the list when it is ported to `identity_digest` / `blob_digest`, and an
allowance that no longer matches an import fails the contract.

| Module | Imports |
|---|---|
| `scripts.analyze_audio` | `hashlib` |
| `scripts.build_classifier_ontology_map` | `hashlib` |
| `scripts.check_fixture_budget` | `hashlib` |
| `scripts.extend_cohort` | `hashlib` |
| `scripts.free_speech_review_page` | `hashlib` |
| `scripts.release_gate` | `hashlib` |
| `scripts.verify_grid_unification` | `hashlib` |
| `senselab.audio.tasks.speech_to_text.crisperwhisper` | `hashlib` |
| `senselab.audio.workflows.audio_analysis.adaptive.belief` | `hashlib` |
| `senselab.audio.workflows.audio_analysis.adaptive.policy` | `hashlib` |
| `senselab.audio.workflows.audio_analysis.fuse` | `hashlib` |
| `senselab.audio.workflows.audio_analysis.run_config` | `hashlib` |
| `senselab.audio.workflows.triage.cohort_stage` | `hashlib` |
| `senselab.audio.workflows.triage.config` | `hashlib` |
| `senselab.audio.workflows.triage.nodes.second_opinion` | `hashlib` |
| `senselab.audio.workflows.triage.recording_vectors` | `hashlib` |
| `senselab.audio.workflows.triage.review_page.page` | `hashlib` |
| `senselab.audio.workflows.triage.speaker_vectors` | `hashlib` |
| `senselab.audio.workflows.triage.task_audio` | `hashlib` |
| `senselab.audio.workflows.triage.task_squim` | `hashlib` |
| `senselab.text.tasks.decision_model.ollama` | `hashlib` |
| `senselab.text.tasks.pii_detection.redaction_review` | `hashlib` |
| `senselab.text.tasks.pii_detection.subprocess_backend` | `hashlib` |
| `senselab.utils.reproduce.build` | `hashlib` |
| `senselab.utils.reproduce.cache` | `hashlib` |
| `senselab.utils.reproduce.cli` | `hashlib` |
| `senselab.utils.reproduce.digest` | `hashlib` |
| `senselab.utils.reproduce.engine` | `hashlib` |
| `senselab.utils.reproduce.env` | `hashlib` |
| `senselab.utils.reproduce.inputs` | `hashlib` |
| `senselab.utils.reproduce.inputs` | `xxhash` |
| `senselab.utils.reproduce.lock` | `hashlib` |
| `senselab.utils.reproduce.lockgen` | `hashlib` |
| `senselab.utils.reproduce.locks` | `hashlib` |
| `senselab.utils.reproduce.run` | `hashlib` |
| `senselab.utils.reproduce.sandbox` | `hashlib` |
| `senselab.utils.reproduce.source` | `hashlib` |
| `senselab.utils.reproduce.weights` | `hashlib` |
| `senselab.utils.subprocess_venv` | `hashlib` |
| `senselab.utils.tasks.cached_inference` | `hashlib` |
| `senselab.utils.uv_tools` | `hashlib` |
| `senselab.utils.venv_lock` | `hashlib` |
| `senselab.utils.venv_worker` | `hashlib` |
| `senselab.utils.vllm_server` | `hashlib` |
| `senselab.utils.work_pool` | `hashlib` |
