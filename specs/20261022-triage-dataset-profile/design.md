# Dataset profiles for triage — 2026-10-22

**Status:** proposal, for review by the triage owner. No code. Citations are to `origin/alpha` at
`98c8048b6`, relative to `src/senselab/audio/workflows/triage/` unless a path says otherwise. The staged
work is in `plan.md` beside this file.

## What this is

The triage workflow stays in senselab. What it knows about one dataset moves into a **dataset
profile**: a folder the caller supplies, which the workflow loads beside its packaged configuration.
The Bridge2AI-Voice (B2AI) adult profile is the first one. It is assembled from what triage packages
today, and it moves to b2aiprep once senselab can load it from outside the package.

This spec settles the profile's contents, how it is loaded and hashed, the few code hooks it needs, and
the order of the changes. It builds nothing.

## Why

Triage was built and validated on one corpus, the B2AI adult release. The graph, the nodes, the store
and the instruments are general. The dataset reaches them through Python tables, fixed package paths
and file-name parsing, so a second dataset cannot run by supplying configuration:

- **Task rows are Python constants.** There are 48 `Expectation` rows keyed by B2AI task names:
  `VOICE_EXPECTATIONS` (`nodes/branches.py:830`), `SPEECH_EXPECTATIONS` (`:945`) and
  `AIRWAY_EXPECTATIONS` (`:1047`). About 25 call sites read them by name. No config path reaches them:
  `EXPECTATIONS` is not in `DATA_MAP_PATHS` (`config.py:30`).
- **Family sets are Python constants.** `SYLLABLE_REPETITION`, `LEXICAL_SPEECH` and `DECLARED_KIND`
  (`routing_analysis/families.py:18`, `:34`, `:111`) have about 15 importers, among them `nodes/verdict.py`
  and `triage_table.py`. The ruleset can name only the sets this code defines.
- **The task comes from the file name.** `declared_task_family` (`nodes/branches.py:1157`) prefers
  `hint.metadata["task_token"]`, but nothing in `src/senselab` or `scripts/` writes it. Every run therefore
  falls back to `_task-(?P<task>[^_]+)$` and strips trailing `-\d+` (`routing_analysis/families.py:12-13`).
  Participant and session are parsed from `sub-`/`ses-` in about six places, among them `run.py:83`,
  `nodes/background.py` (`session_key`), `cohort_stage.py` (`participant_key`), `recording_vectors.py:65`
  and `speaker_vectors.py:66`.
- **Data files are read from package paths.** There are 39 package-relative loads under the triage package,
  among them `task_minimum_duration`, `discard_replaced_task`, `airway_event_requirements`,
  `task_content`, `task_speech`, `task_extent`, `breath_pattern`, `cough_pattern`, `voice_phonation`,
  `cohort_stage` and `task_lexicon/*`. Several are keyed by B2AI family names, and their numbers are fitted
  on B2AI runs (r8 to r37) or on the owner's listening sessions.
- **The default config carries B2AI values.** In `data/config/default.yaml` these include
  `stimulus.task_lexicons` and `residue.cue_families` (`:158-159`), `airway.route_by_task_index`
  (`:183-187`), `verdict.cohort_conditions: bridge2ai_voice_adult_2026-09-04` (`:333`), the
  hint-mismatch-exempt and model-speaker family lists (`:343`, `:349`), and `verdict.gates.by_family`.
- **Only one override can be applied.** `load_triage_config(override)` (`config.py:185`) merges a single
  file. Data maps merge additively (`config.py:165-166`), so an override can add entries to a packaged map
  but cannot remove them.

The branch code itself is general. No `.py` file outside `nodes/branches.py` and
`routing_analysis/families.py` branches on a B2AI task name; the other mentions are in docstrings.
The breath, cough, DDK, phonation, task-event and stimulus-alignment instruments are signal processing
whose parameters happen to be B2AI's.

## What stays in senselab

- **Engine:** the graph and its fixed order (`GRAPH_ORDER`, `vocabulary.py:53`), the node contract
  (`nodes/common.py`), the provenance store, config loading and validation (`UnmeasuredConfigKey`), the
  run and replay tools (`extend_*`, `replay_diff`) and the viewers.
- **Task model:** the `Pattern` kinds (`nodes/gates.py:107`), the `Expectation` dataclass with its
  `as_mapping`/`from_mapping` (`nodes/branches.py:782`, `:802`), the gate layers (default → group →
  family) and every instrument.
- **Declarations:** `AudioHints` (`senselab/audio/data_structures/audio_hints.py:129`) and the
  `hints.py` plug-in loaded by `extend.load_hint_builder` (`extend.py:794`).

## What a profile holds

| File | Contents | Replaces |
|---|---|---|
| `profile.yaml` | name, version, provenance (source dataset, runs it was fitted on), the list of files below, and a config override | the single override argument |
| `tasks.yaml` | the task rows: family → branch, `Pattern` and `Expectation` fields, loaded through `Expectation.from_mapping` | the three `*_EXPECTATIONS` tables |
| `families.yaml` | named family sets and declared kinds | `families.py` frozensets and `DECLARED_KIND` |
| `lexicons/` | story casts, colour words, category word lists, and which categories allow repeats | `CINDERELLA_CAST`, `STROOP_COLOURS` (`nodes/branches.py:865`, `:894`), `data/task_lexicon/`, `REPETITION_ALLOWED_CATEGORIES` (`nodes/speech.py:154`) |
| `fitted/` | per-family durations, rates, percentiles and listening-derived bounds, each with the run it came from | the family-keyed packaged YAMLs |
| `policy/` | the verdict and redaction switches the dataset owner decided, and the reviewer's task guidance | owner-policy keys in `default.yaml`; see "Outside triage" for the PII files |
| `hints.py` (optional) | the dataset's `build_hint` | today's campaign `hints.py` |

A profile supplies files, not code, except for the optional `hints.py` and the hooks below.

## How a profile is loaded

1. **Stacked layers:** packaged defaults, then the profile's override, then a run override.
   `load_triage_config` accepts more than one layer.
   - **Packaged maps keyed by task carry no dataset entries.** A data map such as
     `verdict.gates.by_family` (`rainbow-passage`, `caterpillar-passage`, `word-color-stroop` today) merges
     add-only, so an override can add or change a key but never delete one (`config.py:165-166`).
   - The B2AI entries move into the B2AI profile, and the packaged maps keyed by task are left empty.
   - Deleting a packaged entry is then needed only for senselab's own general entries, such as classifier
     label thresholds. That is an open question below, not a requirement.
2. **One data-path resolver.** Every one of the 39 package-relative loads asks the resolver. The resolver
   looks in the profile first and falls back to the package.
3. **One hash over everything that decides.** Today `config_hash` covers the merged YAML only
   (`config.py:201`). The packaged data files are pinned only by the senselab commit, but files supplied
   from outside are not. The hash must cover the merged config plus the bytes of every profile file that
   was loaded, or two runs with different rules would record the same hash. The run records the
   profile's name and version beside it.
4. **Identity from the declaration.** One function returns participant, session, task family and task
   index from the hint or the manifest row, with BIDS file-name parsing as the fallback. The B2AI
   `build_hint` writes `task_token`, so the regex stops being the source.

## Code hooks the profile needs

Four rules are written in code today and cannot be expressed as data.

| Hook | Where today | Proposed |
|---|---|---|
| Task id → family and index | `routing_analysis/families.py:12-13`, `:124`, `:137` | the identity function above; the index-stripping rule comes from the profile |
| Airway index → nose or mouth route | the parse in `nodes/airway.py`; the map is already config (`default.yaml:183-187`) | the index comes from identity; the map stays in config |
| Item-list category from the instruction text | `CATEGORY_LINE` (`nodes/branches.py:1315`); letter/number rules in `task_lexicon.py` | the category becomes a declared field in the hint; the rules move to the category files in `lexicons/` |
| DDK dispatch by family set | `task_family in SYLLABLE_REPETITION` (`nodes/speech.py:1622`, `:1774`) | dispatch on `Pattern.SYLLABLE_TRAIN`/`SYLLABLE_SEQUENCE`, as `align_*` already does |

`Pattern` stays a closed set of 12 matchers. A dataset whose task needs a new kind of matcher needs new
code; a matcher registry is listed under "Not in scope".

## Changes, in order

Each step is its own PR. Each compares B2AI output before and after the change, and any difference is
explained in the PR (see "Acceptance"). `plan.md` has the detail.

1. **Identity.** ADMIT records participant, session and task from the hint, with today's file-name
   parsing as the fallback. `b2ai_hints.build_hint` writes `task_token`, `participant_id` and `session_id`.
   The graph's readers use the record. The corpus-level parsers follow in a later step.
2. **Loader.** Add stacked layers, the data-path resolver and the extended hash. Every packaged file is
   still found through the fallback.
3. **Tasks and families as data.** Load the task rows and family sets through accessors. Extract the B2AI
   rows, sets and lexicons into a B2AI profile, kept at first as a senselab test fixture. Switch the DDK
   dispatch to `Pattern`.
4. **Fitted values, policy and neutral defaults.** Move the family-keyed YAMLs and owner-policy keys into
   the B2AI profile. The packaged values become explicit, as open question 1 decides.
5. **Outside triage.** Make the PII and second-opinion inputs caller-supplied (see "Outside triage").
6. **Hand-over.** The B2AI profile, `b2ai_hints.py`, `task_registry.py` and the vendored
   `data/b2ai_task_registry/` move to b2aiprep, which already owns the registry's source
   (`data/b2ai_task_registry/PROVENANCE.yaml`). b2aiprep gets a paired change.
7. **Tests.** The 59 triage test files that use B2AI family names load the B2AI fixture profile.
   `nodes/branches_test.py:471-480` (48 rows) moves with it.

## Acceptance

- **B2AI output is compared, and every difference is explained.** A fixed B2AI sample is replayed before
  and after each step, and the decision tables are compared with `replay_diff`. Each step's PR includes the
  diff.
  - A refactor is expected to change nothing.
  - Where a change is intended or a fair consequence of the step, the PR names it and the reviewer accepts
    it case by case.
  - An unexplained difference blocks the step.
- **The engine runs without the profile.** A synthetic recording with no hint and no profile runs to a
  verdict through detect mode (`nodes/branches.py:1253`).
- **A second profile comes after this plan.** B2AI pediatric is the planned next extension and the first
  real test of a second protocol. Its tasks (identifying pictures, repeating words, the sounds tasks) are
  named nowhere in triage today.

## Outside triage

- **PII policy:** `senselab/text/tasks/pii_detection/data/redaction_policy.yaml` (owner rules, v7 to v11; its
  word lists are the forms present in the B2AI adult release) and `data/task_guidance.yaml` (keyed by B2AI
  families) become caller-supplied, from the profile's `policy/`.
  - The general reviewer in `redaction_review.py` keeps its mechanism. Its protocol wording (examiner,
    item-set, non-lexical tasks) becomes a template the caller fills.
  - `decision_model/second_opinion.py` reads guidance from the caller in the same way.
- **Store vocabulary:** `senselab/utils/prov_store.py:21-37` lists triage's entity types in `PROV_TYPE`.
  Triage registers them with the store, so the store keeps no triage-specific list.
- **Tests:** `src/tests/text/tasks/redaction_policy_test.py` and `pii_redaction_review_vllm_test.py` import
  triage. They move under `src/tests/audio/workflows/triage/`.
- **Clean already:** the other general modules changed during triage (spans, envelope, phonation,
  clipping, disruptions, gammatone, spectral continuity, classification, speech enhancement, feature
  extraction, plotting). They import nothing from triage and take every tuned value as an argument.

## Decided (2026-10-22)

- **Who does the work.** The first attempt is by the b2aiprep side (Jordan Wilke, with Claude Code). The
  triage owner reviews each PR and changes it as needed.
- **Where the B2AI profile lives:** in the b2aiprep package, not a separate data package.
- **Fitted values belong to the caller.** Whoever runs triage on a dataset maintains that dataset's
  values in its profile.
  - For B2AI this is the release's triage runner, and changes go through a reviewed b2aiprep PR.
  - A way to fit a new dataset's values programmatically is not part of this plan. It needs train/test
    split support that senselab does not have yet, and is a follow-up issue after the refactor.
- **Pediatric** is the planned next extension, after this plan.
- **Dependency.** b2aiprep points at a senselab alpha on PyPI (`>=1.4.0a7`). 1.4.0a7 is `origin/alpha` as
  of 2026-10-21, triage included.

## Open questions

1. **Neutral defaults.** Every value fitted today comes from B2AI adult data. The values are of three
   kinds, and each suits a different default:
   - **Defined by signal or physics** (clipping at full scale, band limits at the resample rate): keep
     them as plain defaults.
   - **What a task normally produces** (DDK syllable rate, maximum phonation time, breath and cough
     durations): literature exists but depends on the population. Use cited values at the `Pattern`
     level, or none.
   - **About the models and pipeline** (classifier score cut-offs, the speaker-embedding cosine floor,
     ASR agreement): only a corpus can give these. Either fail through `UnmeasuredConfigKey` until a
     profile supplies them, or ship the B2AI value with a warning that names its source.

   Proposal: keep the defined values, cite literature where it exists, and decide fail-or-warn for the
   model and pipeline values. `specs/20260817-triage-workflow-dag/config-derivations.md` is the starting
   point for sorting each value.
2. **Deleting packaged entries.** Should an override be able to delete one of senselab's own general
   data-map entries, such as a classifier label threshold? If so, how: a `null` value, or a separate
   `remove:` list. The B2AI profile does not need this.
3. **Profile versioning:** whether a profile's version is free-form or must match a profile schema
   version that senselab checks.
4. **Where `hints.py` lives:** in the profile folder, or as an entry point the dataset package exposes.

## Not in scope

- A node or matcher registry. It is needed only for a dataset that requires a new branch or `Pattern`.
- Any change to what triage measures or decides.
- b2aiprep's triage step: see its paired design (`docs/design/triage-workflow-step.md` on b2aiprep branch
  `design/triage-workflow-step`).
