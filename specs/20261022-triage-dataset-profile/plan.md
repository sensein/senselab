# Implementation plan: triage dataset profiles — 2026-10-22

This is the staged work for `design.md` in this folder.

**End state:** senselab's triage ships no dataset-specific data, only the engine's own files and neutral
defaults. Every dataset, B2AI included, supplies its task rows, families, word lists, fitted values and
policy through its own profile. Until the last steps land, senselab still ships the B2AI files, so a
profile's files are read instead of them; by the end there is nothing left to take over from.
 The work happens on an integration branch,
`design/triage-dataset-profile`:
- Each step is its own PR against that branch, reviewed by the triage owner. A step starts only after the
  one before it has been reviewed.
- The branch's own PR (#607) merges into `alpha` once, when every step is in. A published alpha therefore
  holds either none of the refactor or all of it.
- `alpha` is merged into the branch regularly, with a merge rather than a rebase, so the step PRs keep
  their history while triage keeps changing on `alpha`.

Every step compares B2AI output before and after. A refactor is expected to change nothing. Where a change
is intended or reasonable, such as step 2's hash, the PR names it and the reviewer accepts it case by case.
An unexplained difference blocks the step.

Citations are to `origin/alpha` at `98c8048b6`, relative to `src/senselab/audio/workflows/triage/`.

## Step 1 — identity from the declaration

**Goal:** participant, session and task are declared once, by the caller's hint, and recorded on the run.
Graph code reads the record instead of parsing file names. Parsing stays as the fallback, so runs and
stores without the new fields behave exactly as now.

**Today:**
- The task id is gathered in two copies of one helper, `_declared_task_ids` (`nodes/branches.py:1186`) and
  `declared_task_ids` (`nodes/airway.py:585`). Each tries `hint.metadata["task_token"]`, then the `_task-`
  regex on the recording path.
- Nothing writes `task_token`, so the regex always decides.
- Three readers see only the store and never the hint, so they always parse the stem:
  `nodes/background.py:168` (`declared_family`), `live_evidence.py:92` (`declared_task`) and the session
  key at `run.py:309` (`session_key(recording_stem(store))`).

**Changes:**
1. A new module, `identity.py`:
   - a `RecordingIdentity` with `participant`, `session`, `task_id`, `task_family` and `source`
     (`"hint"` or `"stem"`);
   - `identity_of(stem, hint)`, which prefers the hint metadata keys `participant_id`, `session_id` and
     `task_token`, and falls back to today's parsers unchanged (`routing_analysis/families.py:124-146`,
     `nodes/background.py` `session_key`);
   - `recorded_identity(store)`, which reads what ADMIT recorded and otherwise re-derives the identity
     from the stem, so older stores replay unchanged.
2. **ADMIT** records the identity on the `recording` stream entity it already writes (`nodes/admit.py:92`)
   as `declared_participant`, `declared_session`, `declared_task_id` and `declared_from`. The hint is
   already passed to `admit` and is unread today.
3. **The readers:** both task-id helpers, `background.declared_family`, `live_evidence.declared_task` and
   the session key at `run.py:309` read through `recorded_identity`. When ADMIT refuses a recording it
   writes no stream, so the fallback to the `audio_file` parameter stays as it is.
4. **`b2ai_hints.build_hint`** writes `task_token`, `participant_id` and `session_id`, taken from the
   recording's stem.
   - `task_token` is the stem's own task entity, exactly what the regex reads today, not the sidecar's
     `task_name`. The two can differ: the 09_04 tree's `high-to-low` is its own row in
     `VOICE_EXPECTATIONS`.
   - Moving B2AI identity to the sidecars or the manifest is a separate, later decision.

**Not in this step:**
- The corpus-level parsers read finished runs, not the graph: `cohort_stage.py:527` and `:852`,
  `recording_vectors.py:486`, `speaker_vectors.py:172` with its `sub-*` directory walk,
  `triage_table.py:529`, and the `scripts/extend_*` grouping. They move once runs record identity, which
  step 1 starts.
- The output layout, `run.py:167` (`entity_subdir`), keeps mirroring the input tree.

**Checks:**
1. **Unit tests** in the existing test file for each module:
   - `identity_of`, with and without a hint, for every carrier and the fallback;
   - an old store with no `declared_*` attributes;
   - ADMIT's new attributes;
   - `build_hint` writing the three keys.
2. **Equivalence over the corpus.** For every recording in the adult 09_04 tree (62,550) and the FLAC
   source tree, compare the identity from `build_hint`'s hint with the stem parsing: task family,
   participant and session must agree for all of them. It runs as a SLURM job, since reading 62k sidecars
   takes more than a minute or two.
3. **The existing triage test suite passes unchanged.**
4. **One end-to-end run on a few B2AI recordings**, before and after the change. The decision tables are
   compared, and any difference is explained. The recordings cover a breath task, whose route comes from its index, plus a DDK task and a
   reading task.

**Size:** about 150 lines of code plus tests.

## Step 2 — profile loader and hash

- **Layers:** `load_triage_config` accepts packaged defaults, then a profile, then a run override.
- **Profile manifest:** `profile.yaml` gives the name, version, provenance and the data files the profile
  supplies. The engine's own files (the packaged config, the output schemas, the decision vocabularies)
  cannot be supplied.
- **One resolver for data files:** each data read goes through it. It reads the file the profile supplies,
  and otherwise the packaged one.
- **One profile per process:** loaders cache what they read, so activating a second, different profile
  raises.
- **A wider hash, only with a profile:** `config_hash` also covers the profile's name, version and every
  byte it supplies, and `run.json` records the profile. Without a profile, the hash is unchanged.
- **`scripts/triage_audio.py --profile`.** The `extend_*` drivers get the same option next.
- **Acceptance:** the decision tables on the step-1 sample are compared, and any difference other than
  the hash is explained. Every packaged file is still found through the fallback.

## Step 3 — tasks and families as data

- The three `*_EXPECTATIONS` tables and the `families.py` sets load from `tasks.yaml` and `families.yaml`
  through accessors. The about 40 reads by name change to those accessors.
- `CINDERELLA_CAST`, `STROOP_COLOURS`, `REPETITION_ALLOWED_CATEGORIES` and `data/task_lexicon/` move to
  `lexicons/`.
- The DDK dispatch moves from family sets to `Pattern` (`nodes/speech.py:1622`, `:1774`).
- The B2AI rows become a test-fixture profile. `nodes/branches_test.py:471-480` checks them through the
  loader.

## Step 4 — fitted values and policy

- The family-keyed packaged YAMLs and the owner-policy keys in `default.yaml` move into the B2AI profile.
- The packaged maps keyed by task are left with no dataset entries.
- The neutral defaults follow open question 1.

## Step 5 — inputs outside triage

- `pii_detection/data/redaction_policy.yaml`, `task_guidance.yaml` and the protocol wording in
  `redaction_review.py` become caller-supplied, and `decision_model/second_opinion.py` follows.
- `prov_store.PROV_TYPE` accepts types registered by triage.
- The two tests in `src/tests/text/tasks/` that import triage move under the triage tests.

## Step 6 — hand-over to b2aiprep

- The B2AI profile, `b2ai_hints.py`, `task_registry.py` and `data/b2ai_task_registry/` move to b2aiprep.
- This lands together with b2aiprep's paired PR.

## Step 7 — tests on the fixture profile

- The 59 triage test files that name B2AI tasks load the fixture profile.

## After this plan

- **A pediatric profile:** the first second protocol.
- **senselab's data folder holds no dataset data:** checked by a test that fails if a packaged data file
  names a dataset's tasks.
- **Fitting a new dataset's values programmatically:** a follow-up issue. It needs train/test split
  support in senselab.
