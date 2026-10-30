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

## Step 2 — the B2AI data moves out, and profiles supply it

Redone on 2026-10-22, after the review of the first version (see "Review of steps 1 and 2" below). The B2AI data
leaves senselab in this step, so the mechanism is tested as it will be used.

- **Three kinds of data file** (`dataset_profile.py`; the file-by-file split is in `inventory.md`):
  - **engine:** always the packaged copy;
  - **dataset** (`DATASET_PATHS`): only a profile supplies them, and without one triage raises
    `DatasetDataMissing`;
  - **defaulted** (`DEFAULTED_PATHS`): packaged with defaults; a profile may supply its own complete copy.
- **No mixing.** A path comes from the profile or from the package. A profile supplies a directory whole, and may
  list only those paths, so typos, engine files and files inside a directory are refused.
- **The B2AI adult profile** holds every moved file and key at its pre-profile value:
  `src/tests/audio/workflows/triage/profiles/bridge2ai_voice_adult/`, a test fixture until step 7. With it active,
  the merged config and every data loader's output equal step 1's (checked: 51 items, two hash seeds).
- **senselab's own values:**
  - the packaged config and defaulted files keep only defaults;
  - values only a corpus can give are unset, and `require_dataset_values` lists them at the start of a run;
  - policy switches that only add review keep their values.
- **Loaders:** they cache per active profile (`profile_cache`). The detector catalogue is built on first use, not
  at import.
- **Config:** the profile is activated only after the merged config validates. A config loaded without a profile
  while one is active uses that profile, so the hash always names the files read.
- **Hash:** unchanged without a profile. With one, it adds the profile's name, version and digest.
- **Tests:** they run with the B2AI profile active (fixture `b2ai_triage_profile` in `src/tests/conftest.py`).
  The no-profile behaviour is tested with `without_triage_profile`.
- **`scripts/triage_audio.py --profile`.** The `extend_*` drivers get it in step 3.

### Review of steps 1 and 2 (2026-10-22)

A code review and five verifiers confirmed ten findings. This redo removes the ones that came from mixing profile
and packaged files (findings 1, 3, 4 and 5), and makes the import-time reads lazy (finding 2). Where the rest went:
- 10 (cleanup): the imports step 1 left unused are removed on the step-1 branch.
- 8 (evidence): rerun on clean commits before the step-2 PR.
- 6 (`--profile` on the drivers), 7 (two readings of the task on replay) and 9 (corpus grouping by file name) are
  one problem: a finished run does not record enough about its inputs for a later pass to check against. Step 3
  handles them together.

## Step 3 — the run records its inputs; replay and corpus passes read the record

Citations in this step are to `refactor/triage-b2ai-profile` at `eaa38d665` (alpha, step 1 and step 2).

**Goal:** every later pass over a finished run (replay, refold, the `extend_*` drivers, the corpus scripts) reads
one record of what the run was given. A changed hint is the caller's decision and is accepted, and every stage then
follows it. What changed is recorded. Replay refuses only where a change is almost certainly a mistake.

**Terms:**
- **Fresh run** (`run_triage`): every stage, ADMIT through REPORT.
- **Replay** (`replay_decisions`, `scripts/extend_replay_decisions.py`): reruns the decision stages, BACKGROUND
  through VERDICT and REPORT except COHORT (`extend.py:162`). It keeps PREPROCESS's stored model outputs, and it
  is handed a hint again, rebuilt from the sidecars (`extend_replay_decisions.py:206`).
- **Refold** (`refold_verdict`): reruns VERDICT only.
- **Extend** (`scripts/extend_*.py`): adds one measurement to finished runs.
- **Corpus pass:** groups many runs by participant or session (cohort, session floor, speaker vectors, review
  pages).

**Today:**
- **Two task readers.**
  - The live hint first: `identity.task_id_candidates` (`identity.py:211`), used by branch routing, the airway
    route, REVIEW and replay's word arbitration (`extend.py:1110`).
  - What ADMIT recorded: `live_evidence.declared_task_of` (`live_evidence.py:119`), used by VERDICT
    (`nodes/verdict.py:1456`), DDK (`nodes/ddk.py:617`, `:631`) and cohort facts.
  - The hint-less `declared_task_family(store)` calls in REDACT and QUALITY (`nodes/redact.py:2283`, `:2317`,
    `:2428`; `nodes/quality.py:269`) also land on ADMIT's value, while `nodes/redact.py:869` passes the live hint.
  - On a replay with a changed task, routing follows the new task while VERDICT scores against the old one.
- **What the hint reaches upstream of replay:**
  - ADMIT's recorded identity (`nodes/admit.py:106`);
  - the stimulus alignment, from `expected_speech` (`nodes/preprocess.py:882`, `:1004`);
  - word arbitration, from the task family (`nodes/preprocess.py:2695`), which replay already reruns.

  ASR, the classifiers, diarization and enhancement never read the hint.
- **The hint itself is not recorded.** Parts of it appear in routing's `hint_tags`, REVIEW's `task_context` and
  the stimulus sidecar, but there is no fingerprint to compare against.
- **The audio digest is recorded, never checked.** ADMIT stores `checksum_sha256` (`nodes/admit.py:78`, `:92`,
  `:105`). Neither replay nor refold compares it.
- **A corrected sidecar is skipped.** `find_replay_marker` matches only the config hash and the commit
  (`extend.py:711-715`), so a second replay returns PRESENT (`extend_replay_decisions.py:204`).
- **A missing hint is silent.** Replaying without `--hints`, or with no sidecar (`b2ai_hints.py:111`), drops the
  expected speech. REDACT loses its stimulus exemptions and nothing is flagged.
- **New prompts are ignored.** If the original hint had no expected speech, SPEECH reports the alignment absent
  even when the new hint has some (`nodes/speech.py:1018`).
- **The profile is not checked.** The `extend_*` drivers load the config without `--profile`
  (e.g. `scripts/extend_cohort.py:504`), and nothing compares a replay's profile with the run's.
- **Corpus passes parse file names**, with five parsers that disagree on edge cases:
  - `cohort_stage.participant_key` (first `_`);
  - `scripts/triage_review_page.py:270` (`_ses-`);
  - `speaker_vectors.py` (regex and the `sub-*` walk);
  - `run.entity_subdir` (entity chunks);
  - `identity.session_key_of`.

  For B2AI the hint's identity is the file name by construction (`b2ai_hints.py:116-125`), so nothing differs
  today.
- **The replay design doc assumes the same hint** (`specs/20260922-replay-decisions-over-a-finished-corpus/design.md:148-153`):
  rebuilt from the sidecars, deterministic. It never considers a corrected hint. Its node table and "ADMIT does
  not store it" are out of date.

**Changes:**
1. **One task reader.** Every stage reads the identity in the store through `recorded_identity`. That includes
   `declared_task_family`, `task_id_candidates` and the hint-less calls. On a fresh run the store holds what the
   hint declared, so B2AI output does not change.
2. **The input record, written at ADMIT** on the `recording` stream:
   - the audio digest (already there);
   - **the hint itself**, canonical JSON (`model_dump(mode="json")`), inside the store rather than as a file per run
     (no new inodes). `target_speaker` is a speaker-embedding vector, so it is reduced to its provenance and digest.
     B2AI hints never set it (`b2ai_hints.py:138`).
   - a digest of the hint as a whole and of each field, with one digest per `metadata` key, and an explicit "no
     hint". Every field is digested, read or not, so there is no list to keep in step with the code. A change to a
     field nothing reads is noted on the replay marker and never refused.
   - the resolved participant, session and task, each with its source (already there).

   This closes the gap the replay design doc names: "the one genuine gap in the run's self-description" (`:148-153`).

   The profile and config are covered by `config_hash`, which `run.json` already records with the profile's name,
   version and digest (step 2). No separate hash of the identity rule is needed.
3. **Replay checks the record first**, before anything is retired, in this order:
   1. **Store missing:** refuse.
   2. **Audio digest differs** from ADMIT's: refuse. It is a different recording.
   3. **Profile name differs**, or the run had a profile and the replay has none: refuse, unless
      `--allow-profile-change`.
   4. **Profile version or contents, or config, differ:** accept and record them on the marker. Re-deciding after a
      rule or data change is what replay is for.
   5. **`--hints` given and it differs from the recorded hint:** accept. The replay re-resolves the identity from the
      new hint. Where it differs, it writes a new `recording` record and retires the old one with the existing
      `supersede` (`extend.py:507`), so every stage reads the new one. The marker records the old and new digest of
      each changed field.
   6. **No `--hints`, and the store holds a hint:** reuse the recorded hint. A hint that carried a `target_speaker`
      cannot be reused, since only the speaker vector's digest is stored: such a replay refuses without `--hints`.
   7. **No `--hints`, and the store holds no hint** (stores written before this step): refuse, unless `--hints` or an
      explicit `--no-hint` is given. The B2AI corpus runs used a hint, so silence would drop the expected speech.
   8. **Expected speech changed, or prompts appear where there were none:** redo the stimulus alignment from the
      stored consensus words. A hint change never needs a fresh run.
   9. **Participant or session changed:** list the affected participants on the marker and in the log, so
      `extend_cohort.py apply --participants` and the session floor rerun only those.

   **Old stores:**
   - Stores written before `c0fe77f0b` (2026-09-08) have no audio digest, so check 2 is skipped.
   - No store written before this step has a saved hint, so check 7 applies.
   - The replay writes the missing parts of the record.
4. **Refold refuses a changed hint**, and says to use replay. Refold reruns VERDICT only and keeps branch reports and
   REDACT output made under the old hint (`extend.py:882-946`), so accepting a changed hint would mix old and new
   inputs. Checks 1–3 and 6–7 apply to refold as well.
5. **The marker key** is the config hash, the commit and the hint digest (today the first two only,
   `extend.py:711-715`). A replay after a sidecar fix runs, and a second replay with the same inputs is still PRESENT.
6. **The `extend_*` drivers:**
   - every driver that loads the config takes `--profile`, through one shared argument helper;
   - each runs checks 1–3;
   - drivers that read the hint (`extend_llm_review`, `extend_second_opinion`) use the recorded hint by default.
7. **The stimulus alignment records the hint it was made from.** It is the one output a replay keeps that read
   the hint directly (`expected_speech`, `nodes/preprocess.py:882`), so its activity records the digest of the
   `expected_speech` it aligned against. The other candidates need nothing:
   - word arbitration is rerun by every replay and records the family it used (`word_arbitration.py:284`);
   - the branch reports a refold keeps cannot mix hints, because a refold refuses a changed hint (change 4).

   There is no central list of which stage reads which field: the whole hint reaches every stage, so such a list
   would drift from the code.
8. **One chain from the record to every corpus grouping** (decided 2026-10-27 to 10-30; no backward compatibility
   is kept, since B2AI is the only user and its stored runs can be replayed). Every place that grouped or filed runs
   by file name now reads the recorded identity:
   - the session and participant groups of `extend_cohort`, `extend_session_floor` and
     `measure_session_background` (`session_key`, `cohort_stage.participant_key` removed);
   - the enrollment `subject_id`, passed in by the participant being processed;
   - the corpus table, `decision_tables` and the free-speech page (`recording_vectors.identity` removed);
   - `triage_review_page`'s already-labelled check, looked up from the extract by stem; a labelled stem the extract
     lacks is reported, not parsed;
   - `speaker_vectors`, which reads the corpus table (`--table`) instead of walking `sub-*` folders and using the
     folder name as the speaker;
   - `run.entity_subdir(identity)`.

   **Strict reading.** `recorded_identity` reads ADMIT's record (its stream, or its activity where it refused the
   recording) and returns None where there is none; it no longer reads the stem, nor does `declared_session_key`. The
   one file-name reader left is ADMIT's default for a hint that declares no identity (`identity_from_stem`, which now
   reads `sub-` and `ses-` each on its own). `declared_task_id` keeps its stem reading for stores from before step 1,
   so they can still be replayed. Helpers that generate an identity for old stores are later work.

   **The chain:**
   1. ADMIT's record is the source.
   2. The corpus table writes `participant`, `session`, `task` (the recorded `declared_task_id`) and
      `participant_from`, `session_from`, `task_from`. A run with no recorded identity still gets a row, with the six
      columns null, and is counted (schema 30; decision tables 4, `triage.tsv` 5, speaker vectors 3).
   3. `select_replay_manifest` copies the six columns into each row and refuses a table without them.
   4. Corpus passes group by the row's columns. A row without them is refused ("rebuild the manifest"). A row whose
      run records no identity is its own group, keyed by its stem, and is refused when its store is opened. Where a
      pass opens the store, a row that disagrees with the record refuses its whole participant or session group,
      before anything is written.
   5. A recording naming no participant and session is its own group, keyed by its stem.

   **Run discovery and folders:**
   - The corpus walk finds runs at any depth; a directory holding `run/store.jsonl` is a run root (`run.json` is
     written last, so an interrupted run has none). The latest run is kept per (folder, stem), so two recordings
     sharing a stem in different folders are both kept.
   - `entity_subdir(identity)` places a run under `<participant>/<session>`, labels as declared, a level dropped
     where the identity does not name it. B2AI paths do not change. A label that is empty, `.`, `..` or holds a slash
     is refused in `identity_of`.
   - At run start the identity is `identity_of(stem, hint)`. A replay builds its hint before it mirrors, and places
     an `--out-root` mirror by the identity it will record; `triage_replay_diff` finds it by the row's identity, then
     by name.
   **Sessions** (decided 2026-10-30; revised after the framing review the same day). A session is one recording
   occasion: its recordings are expected to share its conditions (room, background, device). It is only a grouping
   of recordings; what sessions share with one another (participant, location, time of day) is not encoded in it, and
   a pass that ever needs those would read them as separate declared context attributes (not in this step).
   - **Two uses.** The session floor and the session background need only shared conditions. The level-against-
     session reading, per-session enrollment and the session speaker checks (`other_speaker_in_session`,
     `session_speaker_inconsistent`, `session_condition_shift`) compare one speaker within a session. Every current
     pass therefore keys a session by its participant and its identifier together. "One participant per session" is
     an expectation, not an engine rule: a session holding several participants (a conversation, a caregiver and
     child) is reported by the identity check, never refused. A future pass that needs shared conditions only, across
     speakers, would group by the identifier alone.
   - **The key** is built once, `RecordingIdentity.session_key`, as `<participant>/<session>`. No label may hold a
     slash (`check_label`), so the key names one pair; labels may hold `_`. The same form names the enrollment file
     (`<participant>/<session>.npy`).
   - **No session declared.** For the session floor and the session background, a recording that declares no session
     is its own session: it shares conditions with no other recording. For the cohort, a participant's recordings that
     declare no session are pooled (`undeclared_session`) and compared leave-one-out within that pool; no
     session-level check reads the pool (consistency and condition shift are not applicable), and it is no other
     session's. A recording that names no participant is its own group everywhere.
   - **Identifier namespace.** Whether session identifiers are unique within their participant (as BIDS `ses-1`) or
     across the corpus is a fact about whoever minted them (B2AI source trees: corpus-wide; released trees: per
     participant). It changes no triage output, since keys always carry the participant, so it is not in the run
     config or the profile: `check_identity --session-scope` takes it, default `participant`.
9. **An identity check**, `identity_check.check_identities` plus `scripts/check_identity.py`, run at two points:
   - before a corpus run, over the recording list and the hints it will use (`--recordings`, `--hints`);
   - after a run, over the corpus table (`--table`).

   **Session scope** (change 8, Sessions): `--session-scope global` says identifiers are corpus-wide, so one seen
   under two participants is one session holding both; it is reported (check 1).

   | Check | On failure |
   |---|---|
   | 1. A session identifier under two participants (`--session-scope global` only) | report |
   | 2. A (participant, session, task with index) appearing twice; a recording that does not name both is its own session, keyed by its stem | refuse |
   | 3. A task in no family triage knows (the `*_EXPECTATIONS` tables until step 4) | report |
   | 4. A BIDS-like file name and the hint's identity disagree | report; the hint wins |
   | 5. A run with no recorded identity (after a run) | report |
   | 6. A stem appearing in more than one folder | report |
   | 7. A hint that cannot be built, or a label that cannot name a folder | refuse |

   Every finding is listed with its stems and run directories; the script exits 1 on any refusal. The report is
   written mode 600, since it names participants and sessions. Check 4 cannot fire on B2AI, whose hints are built
   from the stem, so it is not evidence for B2AI.
10. **The replay design doc** is updated: hints may change, the node table includes REVIEW, and ADMIT now records
    identity and the hint.

**Not in this step:**
- SECOND_OPINION: replay retires its readings but does not rerun it (`extend.py:162`, `run.py:414`). This was
  found by reading the code and needs its own fix.
- Enrollment is still not replayable (the replay design doc, `:157-160`).

**Three PRs**, each against the integration branch once the one before it is reviewed:

| PR | Changes | Estimate (not measured) |
|---|---|---|
| **3a** | 1 (one task reader), 2 (hint saved in the store, with digests), 7 (the stimulus alignment records its `expected_speech` digest) | about 150 lines |
| **3b** | 3–6 (replay, refold and extend checks; marker key; alignment redo; `--profile` on the drivers), 10 (replay design doc) | about 250 lines |
| **3c** | 8 (corpus chain, output folders), 9 (identity check, `session_scope`) | about 200 lines |

3a only adds records, so it cannot change B2AI output. 3b and 3c each get their own before/after comparison.

**Checks:**
1. **Unit tests**, in each module's existing test file:
   - `identity_test.py`:
     - no stage reads the hint's task directly;
     - the four identity checks under both session scopes;
     - grouping over a flat, non-BIDS folder.
   - `nodes/admit_test.py`:
     - the hint saved in the store;
     - per-field and per-`metadata`-key digests;
     - "no hint" recorded;
     - `target_speaker` reduced.
   - `scripts/extend_replay_decisions_test.py` and `extend_refold_test.py`:
     - each of checks 1–9 and each `--allow-*` and `--no-hint` override;
     - refold refusing a changed hint;
     - the marker key with a changed hint;
     - the recorded hint reused;
     - old stores;
     - the identity superseded on a hint change.
   - `nodes/speech_test.py` and `nodes/preprocess_test.py`: the alignment redone when expected speech changes or
     prompts appear.
   - The corpus chain:
     - `recording_vectors_test.py`: columns from the record;
     - `select_replay_manifest_test.py`: the columns copied;
     - `extend_cohort_test.py`, `extend_session_floor_test.py`, `triage_review_page_test.py`: grouping by columns,
       and a stale row refused;
     - `speaker_vectors_test.py`: runs found by `run.json`.
   - `run_test.py`: `entity_subdir` from the identity, with B2AI paths unchanged.
   - The wide suite, as for step 2.
2. **B2AI before and after**, from clean commits (the wrapper shows `dirty 0`):
   - **The five end-to-end recordings:**
     - decisions identical;
     - entity differences limited to run-derived IDs and the documented float noise;
     - the new hint entity and digests named in the PR as expected additions.
   - **Replay over stored corpus runs:**
     - on copies of about 500 runs stratified by task family, never the original runs;
     - replayed with step-2 code and with step-3 code, decisions compared;
     - the step-2 replay is given `--hints`, since those stores hold no hint;
     - inode headroom on the target store checked before copying.
   - **Corpus grouping:** on the same sample, every corpus pass groups recordings identically under old and new code.
   - **The identity check over the full adult corpus table:** counts for each check. Any refusal on B2AI goes to the
     user before it is treated as a bug.
3. **Deliberate changes**, on copies of a few runs:
   - a changed task token: every stage reports the new task, the marker lists the field, and a second replay is
     PRESENT;
   - a hint omitted on a new store: the recorded hint is reused;
   - a hint omitted on an old store: refused;
   - the audio swapped: refused;
   - a changed profile name: refused;
   - a refold with a changed hint: refused;
   - prompts added: the alignment is rebuilt.

## Step 4 — tasks and families as data

**Framing amendment (2026-10-30 review).** The task id carries several meanings in one string: the trailing index is
a repetition, a protocol variant (B2AI's airway route, `nodes/airway.py:598`) or a stimulus selector (a Harvard
list), and the engine decodes it (`families.py:13`). In this step the hint declares them apart: the protocol item
(kind and family), its variant attributes (such as the route), the stimulus (`expected_speech`) and a repetition
index; B2AI's hint builder maps its tokens onto these. "One take per task per session" becomes an expectation a
profile declares, not a uniqueness rule. The family-keyed lexicons and `voice_task_text` are a second carrier of the
stimulus beside `expected_speech`; a family that holds different stimuli would read the wrong words, so the hint's
stimulus is the one source.

- The three `*_EXPECTATIONS` tables and the `families.py` sets load from `tasks.yaml` and `families.yaml`
  through accessors. The about 40 reads by name change to those accessors.
- `CINDERELLA_CAST`, `STROOP_COLOURS`, `REPETITION_ALLOWED_CATEGORIES` and `data/task_lexicon/` move to
  `lexicons/`.
- The DDK dispatch moves from family sets to `Pattern` (`nodes/speech.py:1622`, `:1774`).
- The B2AI rows become a test-fixture profile. `nodes/branches_test.py:471-480` checks them through the
  loader.

## Step 5 — fitted values and policy

- The family-keyed packaged YAMLs and the owner-policy keys in `default.yaml` move into the B2AI profile.
- The packaged maps keyed by task are left with no dataset entries.
- The neutral defaults follow open question 1.

## Step 6 — inputs outside triage

- **File-name parsing leaves the engine** (framing review, 2026-10-30). ADMIT's default reads `sub-`/`ses-`/`task-`
  off the stem when the hint declares nothing (`identity_from_stem`), and identity check 4 compares a BIDS-like name
  with the hint. Both move to the hint builder (the profile); the engine's default is "nothing declared". Values
  read off a file name are recorded as such, not as "declared".
- `pii_detection/data/redaction_policy.yaml`, `task_guidance.yaml` and the protocol wording in
  `redaction_review.py` become caller-supplied, and `decision_model/second_opinion.py` follows.
- `prov_store.PROV_TYPE` accepts types registered by triage.
- The two tests in `src/tests/text/tasks/` that import triage move under the triage tests.

## Step 7 — hand-over to b2aiprep

- The B2AI profile, `b2ai_hints.py`, `task_registry.py` and `data/b2ai_task_registry/` move to b2aiprep.
- This lands together with b2aiprep's paired PR.

## Step 8 — tests on the fixture profile

- The 59 triage test files that name B2AI tasks load the fixture profile.

## Step 9 — speakers, recordings and corpus inputs (framing review, 2026-10-30)

- **Target speaker.** Enrollment, `speaker_vectors` and the cohort treat the participant as the speaker. That holds
  for adults recording themselves, not where a caregiver speaks for a child (the pediatric profile), nor for
  clinician-prompted or conversational data. "Target speaker" names whose voice the task asks for; it defaults to the
  participant and may be declared otherwise. `hint.target_speaker` exists, but the speech branch ignores it
  (`nodes/speech.py:1753`). The participant stays the unit of consent, leakage and grouping.
- **Recording id.** The stem is the run folder's name and the review pages' join key. It is unique only because BIDS
  puts sub, ses and task in it; a flat `p01/s2/recording.wav` collides. The recording id becomes the source path
  relative to the corpus root, or a declared id; the stem is a display label.
- **Corpus-pass inputs on the replay marker.** The marker is the config hash, commit and hint digest. BACKGROUND and
  VERDICT also read the session floor and the cohort readings, which `extend_session_floor` and `extend_cohort` write
  without touching it, so rerunning those and then replaying returns PRESENT. The marker gains a digest of those
  inputs. [UNVERIFIED — how `--force` is set in practice.]
- **"Cohort" names two things:** the COHORT corpus stage (`cohort_stage.py`) and the recruited study cohort
  (`cohort.py`, `verdict.cohort_conditions`). One is renamed.

## After this plan

- **A pediatric profile:** the first second protocol.
- **senselab's data folder holds no dataset data:** checked by a test that fails if a packaged data file
  names a dataset's tasks.
- **Fitting a new dataset's values programmatically:** a follow-up issue. It needs train/test split
  support in senselab.
