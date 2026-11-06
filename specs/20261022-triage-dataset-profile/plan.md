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

## Step 4 — what each recording was asked to contain (agreed 2026-11-02)

**What this step is for.** Today senselab knows B2AI's tasks by name. It cuts the trailing number off a task id
(`respiration-and-cough-fivebreaths-2` becomes `respiration-and-cough-fivebreaths`) and uses that name to look up
everything: which analysis runs (speech, voice or airway), what the recording should contain, and which word lists
apply. A dataset that does not name its tasks B2AI's way gets none of it. After this step senselab knows no task by
name: the dataset says, for each recording, what it was asked to contain, and senselab analyses that.

**Example: one B2AI recording.** `…_task-respiration-and-cough-fivebreaths-2.wav` is the second five-breaths
recording, done through the mouth.

| | Today | After this step |
|---|---|---|
| Which task | senselab cuts `-2` off the file name | B2AI's hint says: task `respiration-and-cough-fivebreaths`, take 2 |
| What it asks for (five breaths, an inhale first) | a row in senselab's code | an entry in B2AI's task list |
| Nose or mouth | senselab maps the `2` to mouth with a table in B2AI's profile | B2AI's hint says: route mouth |
| Which analysis (airway, breath sounds) | senselab looks the name up in hand-written lists | worked out from the entry: an airway task about breath sounds |

What triage reports for B2AI does not change; only who supplies the information.

**1. What senselab keeps.** The description of what a task asks for (today's `Expectation`: the pattern, the words
or count, the duration, the route, whether repeats are allowed, the vocabulary), the analyses that read it, and the
meaning of each kind of task. senselab ships no task.

**2. The task list, and how a recording points into it.** The dataset profile holds a task list: one entry per task,
under a name the dataset chooses, saying which analysis (SPEECH, VOICE or AIRWAY) reads it and what it asks for.
Each recording's hint names its entry exactly, in new typed fields of `AudioHints`: the task, its variant (B2AI's
route) and its take number. senselab never derives the task from a file name and never decodes the number. A hint may
instead carry the whole description itself, for a dataset with no task list. ADMIT records what it resolved (the
entry, its analysis, the description applied and its digest, the variant and the take), and every stage, replay and
refold read that record. A recording whose task the list lacks, or that names none, gets the default treatment and is
reported by the identity check (as B2AI's `audio-check`, which should never have been triaged).

**3. Kinds are worked out, not listed.** Today nine hand-written lists say which tasks are speech, syllable
repetition, words and sentences, voice, pitch glides, held vowels, airway, coughs and breaths. They are replaced by
rules over each entry: speech, voice and airway are the analysis that reads it; syllable repetition is a syllable
pattern; glides are the glide pattern; held vowels are the other voice tasks; coughs and breaths are the airway sound
the entry names. What a rule cannot work out is a field of the entry: B2AI's `loudness` is a word task that is also
read for its voice ("hey"), today's `voice_task_text`. Settings that already name kinds keep their names.

Checked 2026-11-02 on the 48 descriptions senselab applies to B2AI: every worked-out grouping equals today's list
(speech 31, syllable repetition 10, words and sentences 21, voice 6, glides 3, held vowels 3, airway 11, coughs 3,
breaths 8). Four descriptions senselab holds but never applies (voice readings of `loudness`, `loudness-v2`,
`cape-v-sentences`, `cape-v-sentences-v2`) would move those tasks into voice; decision (user, 2026-11-02): they stay
speech tasks as today, and the unused voice descriptions are carried on the same entries marked not used.

**4. Word lists.** A task's own words belong to its entry: the Cinderella characters and the Stroop colours leave
senselab's code for B2AI's entries, and an entry names its lexicon file. For word-generation tasks ("name as many
animals as you can") the hint declares the recording's category, and the entry lists the categories it allows, each
with its word list and whether repeating a word is a mistake. senselab keeps only the kinds of category rule (a
fixed list, "starts with a letter", numbers); the number words become a profile list, since today they are English
only and B2AI has Spanish participants. The three readers of a task's words (the PII residue, REDACT's exemption, the
mask plan) read them through the recorded task.

**5. What B2AI provides.** A task list (about 48 entries: today's rows and word lists, moved), and, per recording in
its hint, the task, the take, the route where the protocol fixes one, the category for word-generation tasks and the
text to be read. B2AI's hint builder works these out from its file names, sidecars and questionnaire; it moves to
b2aiprep in step 7. This step absorbs part of step 6's first item: the task id is no longer decoded by senselab.

**6. Checks.**
1. Before code: the worked-out groupings equal today's lists (done, above).
2. The full test suite.
3. Five real recordings (breaths, "pa-pa-pa", the rainbow passage, free speech, a cough), old code against new, each
   pair on one machine: decisions identical; the only new output is the recorded task description.
4. The 192 stored adult runs, replayed under old and new code on scratch copies: decisions identical.
5. All 70,520 adult recordings through the identity check with B2AI's hints: every task is found in B2AI's task list
   except `audio-check`.
6. A small made-up dataset that is not B2AI (two tasks, other names, no BIDS-like file names): senselab triages it
   from its own task list. A test, not a SLURM run.

Checks 1, 3 and 4 must show no difference on B2AI; any difference goes to the user, not explained away.

**Results (2026-11-02; old code 6fc1c8011, new code snapshot 9d524ecc8; each side its own B2AI profile and hints
written by its own builder; outputs under the triage scratch environment's `s4/`).**
1. Done before code, above.
2. The wide suite (triage, scripts, the two redaction test files): 4,266 passed, 20 skipped, 1 failed; the failure was
   the viewer's copy of the table schema number (30, now 31), fixed and rebuilt; the viewer and table tests then pass.
3. Five recordings, each pair on one node (node2803): decisions identical (fivebreaths review/as_is; pa, rainbow,
   free speech, cough pass/as_is). The only differing store entity is the `recording` stream, and on it only the
   recorded task (`declared_task`, `task_from`, `task_branch`, `task_expects` and its digest, `task_variant`,
   `task_take`, `item_category`) and the hint's new fields (`task`, `task_variant`, `task_take`) with the hint digests.
4. The 192 stored runs, replayed in place on two fresh private copies, slice by slice on one node, base then new: 192
   ok on both, 192 decisions identical, the same 8 needing the reviewer again. Differences: the recorded task and the
   hint fields as in check 3 on every run; and on 59 PII scans and 6 separations the result cache's `hit` flag, with
   identical cache keys on both sides (the base half computed, the new half read what the base half had just cached).
5. 70,520 recordings: passed; every task is in B2AI's task list except `audio-check` (4,202) and `audio-check-v2`
   (3,768), as in step 3c; no other finding.
6. `task_catalogue_test.py` `TestADatasetThatIsNotB2AI`: a two-task lab corpus (`hum-a-note`, `read-the-card`) with
   plain clip names passes the run's start check; the real ADMIT node records each hint's task; each branch aligns its
   own task and detects the other; a B2AI task name means nothing to it; its identity check reports only an unlisted
   task.

Found, not changed in this step: `airway_event_requirements.yaml` and `airway_event_tokens.yaml` are hand-written lists
of B2AI's cough and breath tasks, repeating the kinds the task list now gives (step 5); a task described inline in the
hint is recorded, but branch selection, expected names and categories still look the task up in the task list by
name, so an inline description alone gets each branch's detect mode.

## Step 5 — fitted values and policy (agreed 2026-11-03)

**What this step is for.** senselab still ships numbers that were tuned on B2AI recordings or chosen by B2AI's data
owner. A new dataset inherits them without knowing it. After this step senselab ships the method, and the profile
ships every number that comes from data or from a decision. This answers open question 1 of `design.md`.

**Why not neutral defaults.** Most of these values only mean something for the data they were tuned on: the cough
detector's 25 settings come from the owner listening to 15 B2AI coughs. A dataset that wants this kind of triage has
to listen to and tune its own data anyway. A default that looks neutral but is really B2AI's runs silently untuned.
senselab carries no "borrowed from B2AI" defaults and no B2AI warnings (user, 2026-11-03).

**1. Settings that name tasks** (agreed 2026-11-03).
- *Repeating the task list:* eight lists in B2AI's profile equal what its task list already says, task for task
  (checked 2026-11-03): the breath and cough tasks of `airway_event_requirements.yaml`, the cough tasks of
  `airway_event_tokens.yaml`, `stimulus.task_lexicons`, `open_vocabulary_families`, the two loudness lists of
  `discard_replaced_task.yaml`, and the task-text comparison of `cohort_stage.yaml`. senselab works them out from
  the task list; the lists are deleted.
- *Facts about a task:* five become fields of the task's entry: a cue word the participant defines
  (`cue_families`), someone may say the sentence first (`model_speaker_families`), quiet breathing the classifier
  does not hear is no flag (`hint_mismatch_exempt_families`), breaths close together (`quick_families`), never
  trimmed at the edges (`exclude_families` of the DDK trim).
- *Numbers measured per task on B2AI:* the per-task gate bounds, response-length medians and minimum durations stay
  in B2AI's profile, keyed by task name; senselab ships them empty.
- The PII task guidance names tasks too; it is step 6.

**2. Where each value lives.** One rule, read off each value's written derivation, so no value is decided by taste:
- *The profile supplies* every value fitted on recordings, set by listening, chosen by the owner (release, review
  and redaction policy), taken as a task norm only from B2AI's own recordings, or with no written derivation.
  senselab ships none of them; a run whose profile lacks any is refused, with one message naming every missing
  value (the step-2 mechanism, extended).
- *senselab keeps* values set by the signal, task norms with a published source (a clinical DDK rate band), and its
  own design choices (window lengths, model pins).
- A data file that holds both kinds is split: the fitted part becomes profile data, the rest stays packaged.
- The four packaged files a profile cannot replace today (`voice_phonation.yaml`, `background_model.yaml`,
  `background_speech.yaml`, `task_squim.yaml`) follow the same rule.
- Cases the rule cannot settle (e.g. classifier cut-offs the owner chose without a fit, a model cut fitted on the
  model vendor's corpus) go to the user with a recommendation.
- *Settled 2026-11-03* (user took all three recommendations):
  - values reasoned from the task's own wording or from anatomy ("sustained" is at least half a second, breathing
    is 0.1-1.2 Hz) are not B2AI's and stay in senselab;
  - a value with no written derivation stays in senselab when it is a processing setting (window, hop, frame,
    smoothing, the percentile a floor is estimated at), marked "derivation owed", and goes to the profile when it
    is a detection threshold;
  - of the unclear cases: classifier cut-offs the owner chose without a fit, values B2AI's own counts decided (the
    airway bracket words, which gates route AIRWAY and VOICE, `r` admitted for "buttercup", the breath flatness
    0.08, the YAMNet silence and coverage cuts) and owner decisions go to the profile; fits on general data (the
    speaker cut-off 0.25, SpeechBrain's own default; the discontinuity factor), design the owner directed (the AST
    10 s window, the hops, the Praat multipliers) and principles (a presence threshold of 1) stay in senselab.
- *Result:* of 685 values, 378 go to the profile and 306 stay in senselab (39 of them "derivation owed"); 16 kHz is
  fixed. Two agents sorted them independently and agreed on 515; the 170 differences are settled above. The tables
  are kept in the triage scratch environment (`s5/step5_final.tsv`, with both agents' evidence per value).
- Ten places where `specs/20260817-triage-workflow-dag/config-derivations.md` contradicts the shipped value or its
  comment are corrected in this step (e.g. the speaker cut-off it gives as 0.231, B2AI's; `diarization.streams`;
  senselab's minimum task length of 0.5 s, which is B2AI's cough floor, while B2AI's own default is 1.0 s).

**3. Everything senselab keeps is overridable** (user, 2026-11-03): a profile or run override can change any
packaged value, one value at a time rather than by copying the whole file. Two exceptions stay fixed: the 16 kHz
working rate the models rely on, and files that define the output format (the corpus table's column dictionary,
the decision-reason vocabulary), since readers of the outputs depend on them.

**4. A reference profile, and profiles that build on one** (user, 2026-11-03). Moving every value to B2AI's profile
alone would make senselab configurable in form only: about 230 of the values are detector settings tuned by
listening to B2AI recordings (breath, cough, voice phonation, background, task events and extent), which describe the
sounds and the recording setup more than B2AI, and no second dataset would re-listen its way to them. It would copy
B2AI's profile, and B2AI's profile would become everyone's hidden default. Instead:
- A profile may name another to build on (`extends:`). It inherits that profile's values and supplies only its own;
  a value file merges senselab's packaged values, then the extended profile's, then its own, then the run's.
- senselab ships a **reference profile**, "adult clinical speech recordings, fitted on Bridge2AI-Voice adult 2026":
  the detector tuning and the model cut-offs fitted on labelled or listened recordings. It is named and opt-in; no
  run uses it unless its profile says so, and it records what it was fitted on.
- B2AI's profile extends it and supplies what is B2AI's own: its task list, the owner's policy, its cohort
  conditions, and the values computed from its own corpus (percentiles, fences, per-task medians).
- A new dataset of the same kind extends the reference profile, sets its policy, fits its corpus values from a pilot
  (step 10), and overrides the few detector values its recordings need.

**5. How a dataset gets each value.** Every profile value is one of four, and the per-value entries and the template
are ordered by it: (a) *decide* (policy switches); (b) *compute from a pilot run* (percentiles, fences, medians:
step 10 fits them); (c) *needs listening* (detector tuning; inherited from the reference profile); (d) *per task*.

*Sorted 2026-11-03* (user agreed): of the 378 profile values, 336 are (c) and go to the reference profile; 42 are
B2AI's own: 30 (a) policy, 10 (b) corpus statistics (the cohort condition fences and consistency floor, the
free-speech word rate, the count-in gap, the cough and DDK atypical bounds) and 2 (d) per task (the per-task
duration ranges, the alveolar `r` admitted for "buttercup"). Two sorts agreed on 327; the check's answer was taken on
the 19 they placed differently (unfitted placeholders and the recognizer's word duration are (c), listening-round
extent numbers are (c), three owner switches are (a)); of 32 unclear values, the owner's unfitted classifier cut-offs,
the sweeps against the corpus's own labels (routing gates, speaker-similarity floor, speaker minimum extent), the
extent feature switches, the never-fitted detector numbers and SQUIM's skipped branches are (c), and the release
audible-share minimum and the default minimum task length are B2AI's. Step 10 refits the 10 (b) values, B2AI's
per-task response medians and, on request, the sweep values. The assignment with both sorts' evidence:
`s5/step5_assignment.tsv` in the triage scratch environment.

**6. Help for a new dataset.** Each profile value gets a short entry in senselab: what it means, its unit, which of
(a)-(d) it is, and how to fit it. A command writes a template profile with every required value blank beside its
entry, extending the reference profile or another profile the dataset chooses; that choice is recorded in the new
profile, not warned about by senselab.

**7. Decided for 5c** (user, 2026-11-03): B2AI's copies hold only its own values and the few places it deliberately
differs from senselab or the reference profile (D1); a profile value is required where B2AI's runs use it today and
optional where it is empty today and B2AI leaves it empty, i.e. it switches a feature off or nothing reads it (D2);
the start-of-run check guards every entry point senselab ships, and a library caller runs it itself (D3).

**8. Checks.**
1. The split: every value's place follows its derivation; an independent agent re-derives it (done before code),
   and so does each profile value's place, reference or B2AI.
2. The full test suite, including a test that a profile missing a required value is refused, naming it.
3. B2AI unchanged: its profile and the reference profile it extends supply every moved value, so five fresh
   paired runs and the 192 replayed runs give
   identical decisions and stores (the step-4 procedure).
4. The made-up dataset of step 4 runs from a template filled in by hand.
5. senselab's packaged data names no task and holds no B2AI-fitted value: a test over every packaged file.

**Results (2026-11-03; old code 140e446f8 = step 4 with its review fixes, new code 32616ea9d = 5a-5c; each side
its own B2AI profile, hints written by each tree's builder, which were identical; outputs under the triage scratch
environment's `s5/`).**
1. The split: two independent sorts agreed on 515 of 685 values, then on 327 of the 378 profile values' kinds; the
   rest were settled with the user (sections 2 and 5).
2. The wide suite after 5c: 4,292 passed, 20 skipped; after 5d: see the 5d commit.
3. Five recordings, each pair on one node: decisions identical, no store entity differs; run.json differs only in
   the config hash and the profile record, which now names the profile it extends (adult-clinical-speech). The 192
   stored runs, replayed in place on fresh private copies, slice by slice on one node: 192 ok, 192 decisions
   identical, no store entity differs. Before any run, every value B2AI's runs read (25 value files and the
   config) was compared with its value before 5c: identical.
4. The made-up lab corpus's profile is written from the template and filled in by hand (task_catalogue_test).
5. `dataset_profile_test`: no packaged value file or config holds a listed profile value, and no packaged data file
   names a B2AI task (the vendored task registry, which moves in step 7, aside).

## Step 6 — inputs outside triage (agreed 2026-11-04)

**What it is for.** senselab still assumes things about a dataset's inputs that only hold for B2AI: that a file name
is BIDS-like, that the redaction policy and the reviewer's wording are B2AI's, and that the provenance store's record
types are triage's. Walked through topic by topic with the user on 2026-11-04.

**6a — file names.** (a) senselab reads nothing off a file name: ADMIT's identity is what the hint declares, and a
recording declaring none records no participant or session (the run sits at the run root, as `entity_subdir`
already allows). (b) B2AI's hint builder keeps reading `sub-`/`ses-`/`task-` off the stem (`b2ai_hints.py`), so
B2AI's identities are unchanged. (c) A hint may say a value was read off a file name, and ADMIT records that as its
source; B2AI's builder keeps declaring its values, so B2AI's records stay "declared" (user: "b2ai could just still
state as declared if we declare them after generating them from the filename"). (d) The identity check's
`stem_disagrees` takes an optional file-name parser from the dataset's hints module and runs only where there is one;
B2AI's lives beside its hints. (e) The replay and corpus tools that fall back to the file name for a store recording
no identity use the same optional parser, and record none without one. `identity_from_stem` leaves the engine.

**6b — redaction policy and the reviewer's wording.** (a) senselab keeps the reviewer's frame (how the transcript is
shown, what it answers), HIPAA Safe Harbor (`safe_harbor.yaml`), and the matching code. (b) The dataset supplies the
policy prose (with its examples, USF and Vanderbilt among them), the rules of what is masked and released, and the
task guidance (`task_guidance.yaml`); the text functions take them as arguments, triage passes them from the profile,
and `decision_model/second_opinion.py` takes the same. B2AI's profile carries today's text unchanged. (c) For B2AI
the reviewer's instructions are byte-identical, so no stored reading needs reading again: the 192-run replay's
carried and needs-reread counts stay the same. (d) The word lists (months, holidays, number and kinship words,
countries, states; English and Spanish) go to the reference profile `adult_clinical_speech`, a starting point a
dataset may replace; the rules of what to mask stay each dataset's. (e) Dataset files name the stages that read
them, as profile values do since step 11: the word lists REDACT, the prose and guidance REVIEW and SECOND_OPINION;
a file is required only when one of its stages runs.

**6a, refined in implementation (user, 2026-11-04).** (f) and (g): the engine's fallbacks for stores recording no
identity (`identity.declared_task_id`, `task_squim.declared_family`, `live_evidence`) are removed, not handed a
parser: such a store is replayed first, which records its identity from the dataset's hint (option A, "replay first
is enough"); every corpus pass but replay refuses it, saying so (`extend.require_recorded_identity`). The report shows
the recorded task, not the `task-` token of the run's folder name.

**6b, refined (user, 2026-11-04): the redaction rules wait for a module.** The fold's rules
(`nodes/redact.py:2505-2840`) are B2AI's (owner, 2026-10-03 to 10-09): dates and states hidden wherever they occur;
ages, places below a country and names a detector marked locked; a country released unless the reviewer hides it;
kinship words never hidden; the time words of `time_release.yaml` released; health conditions never held. The user
agrees with the nine as B2AI's. They are logic, not values, so they become a redaction-policy module (a function from
the run's words, their kinds, the detectors' marks and the reviewer's proposals to each word's decision) rather than
switches; that is the first module of the modular-design effort (option B), and until it is built they stay in
senselab. Step 6 still moves the data: the word lists to `adult_clinical_speech`, the reviewer's policy prose and the
task guidance to B2AI's profile. When the module is built, `adult_clinical_speech`'s default policy is the set
closest to HIPAA (user: "the set of rules that best matches HIPAA"): Safe Harbor, 45 CFR 164.514(b)(2), which
releases years, states and ages under 90 where B2AI's rules hide them; the Common Rule governs consent and review,
not which words are hidden.

**6c — the store's record types.** (a) `prov_store` keeps only the general type (`artifact`); other code registers
its own, and triage registers its fourteen, the historical `kind` among them, when its package loads. (b) Reading
back checks against the registered types. (c) The BEP-028 exporter reads any type: it converts and decides nothing.
(d) Triage keeps its own `Literal` for type checking, and its registration is built from it. Triage's stores do not
change.

**6d — the two text tests.** The word-rule tests stay in `src/tests/text/tasks/redaction_policy_test.py` with word
lists of their own; "REDACT records the policy version" moves to `audio/workflows/triage/nodes/redact_test.py`. The
vLLM engine and mocked-server tests stay in `pii_redaction_review_vllm_test.py` with inline engine settings; "the
cache key names the engine" moves to `nodes/review_test.py`. Nothing under `src/tests/text/` imports triage.

**Checks.** B2AI unchanged, as in 5e and 11e: 5 paired fresh runs, the 192-run replay (with its review carried and
needs-reread counts), the baseline values check; the reviewer's instructions for B2AI hashed before and after; the
suite as a SLURM job.

**Results (2026-11-04, branch `refactor/triage-inputs` off dbf3b3fc6, evidence in `triage_env_2026_10_22/s6/`).**
- B2AI's reviewer instructions are byte-identical: the policy text, both prompts, and every task's guidance, nature
  and description, for all 48 tasks; the guidance file's digest, which every reading's cache key carries, unchanged.
- B2AI unchanged: 5 paired fresh runs, the same decisions and every store entity the same (entity ids normalized);
  the 192-run replay, 192/192 entity-identical and the same decisions, and the reviewer readings each side reused or
  must read again the same (59 needing a reread, 133 with none, on both); the baseline values identical to step 11's.
- Suite: 4,368 passed, 20 skipped (triage and scripts); the 12 failures under `src/tests/text` and `src/tests/utils`
  fail on the base tree too.
- Found while building it: the redaction plan's new word-list variable shadowed its task lexicon, which would have
  changed redactions (fixed before any evidence ran); a profile's template now writes `review_policy.txt` and
  `task_guidance.yaml` when REVIEW runs, and the start check refuses a policy file still holding the template text.

## Step 7 — hand-over to b2aiprep (agreed 2026-11-06; after step 8)

**What it is for.** The last B2AI code and data leave senselab: the B2AI profile, its hint builder and file-name
reader (`b2ai_hints.py`), its task-registry reader (`task_registry.py`) and the frozen registry copy
(`data/b2ai_task_registry/`). b2aiprep's design (sensein/b2aiprep#361, `docs/design/triage-workflow-step.md`)
already places them. Walked through with the user on 2026-11-06.

**Facts checked first.** senselab's frozen registry copy (b2aiprep 8c43256) is byte-identical, file for file, to
b2aiprep `release/v4.1.0`, so a builder reading b2aiprep's own registry writes the same hints. No published senselab
carries this refactor (PyPI's newest is 1.4.0a9, from `alpha`; the stack #607-#638 is unmerged), and the hint builder
imports senselab's identity record, task list and profile loader, so b2aiprep's copy needs a senselab with the stack.

**7a — what moves where** (agreed). The profile (the fixture's content, named `bridge2ai-voice-adult` again) to
b2aiprep `src/b2aiprep/prepare/resources/triage_profile/adult/`; `b2ai_hints.py` to a new
`src/b2aiprep/prepare/triage_hints.py`, with the profile's `hints.py` exposing it to the identity check;
`task_registry.py` beside it, reading `resources/task_registry/` directly; their tests to b2aiprep's `tests/`.
senselab deletes the frozen copy. senselab's test fixture stays: same content, neutral name.

**7b — order: add first, then remove** (option a). Both PRs are built now as drafts: b2aiprep's on
`release/v4.1.0` (adds), senselab's removal stacked on #638. Merge order: senselab's stack into `alpha`, an alpha
published, b2aiprep's senselab pin moved to it and its PR merged, then senselab's removal merged, so no release is
left without a hint builder.

**7c — checks** (agreed). Hints from b2aiprep's builder and senselab's current one byte-identical for the 5 test
recordings and the 192 stored runs; with b2aiprep's profile active, every value B2AI's runs read equal to the
fixture's (the 25 value files and the config), the profile's name aside; the 5 paired runs and the 192-run replay
identical, senselab with the modules against senselab without them plus b2aiprep's profile and hints; senselab's suite
without the modules, the copy or the naming check's exemptions; b2aiprep's suite with the moved tests, against the
senselab branch.

**Results (2026-11-06; senselab `refactor/triage-handover` off f06526391, b2aiprep `feat/triage-profile-handover` off
`release/v4.1.0` 6bb52c7; evidence in `triage_env_2026_10_22/s7/`).**
- b2aiprep: the profile at `prepare/resources/triage_profile/adult/` (3aeed5433's B2AI profile, its header now
  naming its home) with a `hints.py` exposing `build_hint` and `identity_from_name`; `prepare/triage_hints.py` and
  `prepare/triage_task_registry.py`, the latter reading `resources/` directly; `tests/test_triage_hints.py` and
  `tests/test_triage_task_registry.py`, which skip under a senselab without the refactor; the profile added to the
  package data. b2aiprep's senselab pin is unchanged until the alpha with the stack is published (7b).
- senselab: `b2ai_hints.py`, `task_registry.py`, `data/b2ai_task_registry/` and their tests deleted; the naming
  check's exemptions and the registry's pre-commit excludes removed; packaged comments that cited B2AI's values as
  examples made neutral (the reference profile's record of what it was fitted on stays). Two scripts still name B2AI
  tasks: `evaluate_task_events.py` (scores against B2AI's listening labels, in the fixture) and
  `triage_viewer_fixture.py` (viewer demo data); left for the user to place.
- Hints: b2aiprep's builder and senselab's write byte-identical hints for the 5 test recordings and the 192 stored
  runs' sources. Values: every value B2AI's runs read identical, but the recruited-conditions list's name (the
  fixture's neutral one against `bridge2ai_voice_adult_2026-09-04`).
- B2AI unchanged: 5 paired runs and the 192-run replay (senselab with the modules, the fixture and its hints, against
  senselab without them, b2aiprep's profile and its `hints.py`): every store entity identical (ids normalized), the
  same decisions and replay rows, the same rereads (59 / 133); run.json names `bridge2ai-voice-adult` again.
- Suites: senselab 6,700 passed, the same 12 failures as the base tree; b2aiprep with the senselab branch 625 passed
  (25 moved tests), with its installed senselab 1.3.0 600 passed and 2 skipped, the same 13 failures as
  `release/v4.1.0` in both.

## Step 8 — tests on a fixture profile and a made-up one (agreed 2026-11-06; before step 7)

**What it is for.** Every triage and script test (136 files) runs with B2AI's profile active, 84 name B2AI tasks,
48 use B2AI-style file names (`sub-…_ses-…_task-…`, 262 of them) and 18 read identity off those names with B2AI's
reader. Step 7 moves B2AI's profile and reader to b2aiprep, which senselab's tests cannot import. Walked through with
the user on 2026-11-06.

**8a — two profiles** (user: option 2, "so we have additional test coverage", and "without ever naming anything as
b2ai"). (1) The existing tests run on a fixture with the same content, under a neutral name: a test snapshot of a
real adult clinical-speech dataset's profile. (2) A made-up profile adds coverage that needs no B2AI-derived file.

**8b — plain file names, identity declared** (user: "move away from b2ai bids naming convention for testing to prove
that the new mechanisms work"). Test recordings take names that carry no identity (`rec-0007`); a test helper
builds the hint each test declares (participant, session, task), as a dataset's hint builder would, in place of the
18 uses of B2AI's reader. Three or four decoy tests give a BIDS-looking name that contradicts the hint and check the
hint's identity is recorded. Tests of B2AI's reader move with it in step 7; the identity check's "the name disagrees
with the hint" report gets a test-local reader. Run folders still follow participant and session.

**8c — the made-up profile.** One invented task per kind of task the engine reads (all 12 patterns, or the ones that
run, said so). It extends the reference profile, makes up its policy and per-task values, and takes its corpus
statistics from a `fitted.yaml`. Its tests, per kind, at unit level and with no model loaded: the start check passes,
ADMIT records the declared identity and task, the owning branch reads it with its matcher, the gates resolve, and
VERDICT folds a prepared reading. It grows the two-task lab corpus of `task_catalogue_test.py`.

**8d — names.** The fixture: `profiles/adult_speech_fixture/`, named `adult-speech-fixture`, its manifest a test
snapshot taken 2026-11-06, comments citing B2AI's runs made neutral, its recruited-conditions list renamed. The
made-up profile: `profiles/lab_corpus/`. Test helpers: `FIXTURE_PROFILE`, `triage_profile`,
`declared_identity_attributes`, `hint_for(participant, session, task)`.

**Checks.** The suite keeps its pass count, give or take the moved and added tests, each listed; a test fails if any
file under senselab's tests names Bridge2AI or B2AI, but the tests of the two modules step 7 moves; B2AI unchanged:
the 5 paired runs and the 192-run replay on the renamed fixture (identical content), every store entity identical,
the run record naming the new profile (the replays take `--allow-profile-change` for it).

**Results (2026-11-06, branch `refactor/triage-tests` off 3aeed5433, evidence in `triage_env_2026_10_22/s8/`).**
- The fixture: `profiles/adult_speech_fixture/` (`adult-speech-fixture`), same content; its own text and its
  recruited-conditions list renamed; `task_guidance.yaml` left byte-identical, since every reviewer reading's cache
  key hashes it, and exempted by name from the naming check. Stores record neither the profile's nor the list's name.
- Plain names: 58 test files converted by four agents on disjoint groups (746, 532, 740 and 646 passed), with
  `hint_for`, `declared_identity_attributes` and `seed_recorded_identity(store, stem, participant, session, task)`.
  The test helper no longer derives a task from a task token through the dataset's parser, so 11 hints in three more
  files name their task. The identity check's name-disagreement tests use a test-local reader
  (`<participant>.<session>.<task>`). Tests whose point is that a BIDS-looking name is not read keep one.
- Decoys: four tests (ADMIT, identity, run folder, corpus table) give `sub-x9_ses-y9_task-voluntary-cough-3` and a
  hint declaring p07, s2, rainbow-passage; the hint's identity is what is recorded.
- The dataset's reader: its five tests moved to `b2ai_hints_test.py`, which step 7 hands over with the module
  (`task_registry_test.py` too). Three scripts still read the task off file names with it and now take the recorded
  task: the Free Speech review page (from the run record; every run is a candidate, the recorded task decides;
  its language now comes from the recorded hint's `metadata.language`, not the source dataset's sidecar, so a run
  from before the hint record shows none), `verify_task_squim.py` (the driver log row's declared family) and
  `analyze_routing_evidence.py` (the run summary's recorded task). No script or library module imports the reader.
- The made-up profile: `profiles/lab_corpus/`, written from the template, ten invented tasks (one per kind a branch
  reads; `effort` and `per_sentence` have no matcher), its corpus statistics in `fitted.yaml`; `lab_corpus_test.py`,
  42 tests (start check, ADMIT, the owning branch's matcher, the gates, the fold, per kind), all passing.
- Naming check: `no_dataset_name_test.py` fails on any line naming the source dataset under the triage, scripts and
  text tests, but the two moving modules' tests, the byte-hashed task guidance and the packaged registry's directory
  name (until step 7). `quality_control_test.py` (outside triage) names senselab's own Bridge2AI taxonomy.
- Suite: 6,732 passed, 34 skipped (job 28099312; 48 more than at step 10: the lab corpus, the decoys and the check);
  the 12 failures under `src/tests/text` and `src/tests/utils` fail on the base tree too. mypy over the touched
  scripts and modules: no issues.
- B2AI unchanged: 5 paired fresh runs and the 192-run replay, every store entity identical (ids normalized), the
  same decisions and replay rows, the same rereads (59 / 133). run.json names `adult-speech-fixture` (config hash
  434248ea993153c6, was 40722e4438d7e559 under the old name).
- Review (2026-11-06, high effort, on 05b5bd5e9; findings checked against the code): ten confirmed, fixed:
  - 1-2: `verify_task_squim.py` takes a log row's family from the task its run's store recorded (the row's
    declared family as fallback); `analyze_routing_evidence.py` takes the task from each store, since no writer of
    the `<stem>.summary.json` field it read exists in the repo.
  - 3-4: the review page reads only the head of each store, up to ADMIT's recording record, to pick candidates
    (a replay that later moved a recording's task into scope is not seen there); the language is read through
    ADMIT's record, refused recordings included, and the light reader keeps activities.
  - 5-6: seven "the name is not read" tests in identity, run and the identity check take a BIDS-looking decoy
    name again.
  - 7-10: the r37 measurement script points at the renamed fixture; the lab corpus has its own time words and its
    test refuses a file copied from the fixture; the fold test that checked nothing is dropped (32 per-kind tests
    remain); the fixture's header says it was extracted on 2026-10-22 and kept through step 10.

  Suite after the fixes: 6,725 passed, 34 skipped (job 28107150), the same 12 failures as the base tree. mypy over
  the touched files: one error, in the r37 script's own older code (`analyse.py:210`).

## Step 9 — speakers, recordings and corpus inputs (agreed 2026-11-04)

**What it is for.** Four assumptions that hold for B2AI's recordings and not for other datasets': that the participant
is the voice a task asks for, that a file's stem is unique, that a replay's marker covers every input it reads, and
that "cohort" means one thing. Walked through topic by topic with the user on 2026-11-04.

**9a — target speaker.** (a) The participant is whose recording it is, the unit of consent, grouping and leakage;
the target speaker is whose voice the task asks for, and defaults to the participant (B2AI unchanged). (b) A hint may
declare its target speakers as a list, each with an id, an optional role (participant, examiner, partner, caregiver)
and an optional voice vector; the default is one entry, the participant. Enrollment (`speaker_vectors`) and COHORT
group by the target speaker's id, which is the participant's unless declared. The engine accepts exactly one target
and refuses more, saying multi-speaker interactions are not supported yet (user: "list now, record multi-speaker as
a later item"). (c) SPEECH uses a declared voice vector, which wins over the enrollment, and records which it used;
today `hint.target_speaker` is ignored (`nodes/speech.py:1760`).

**9b — recording id.** (a) Each recording has an id unique in its corpus: the hint's declared `recording_id`, else
its path relative to the `--corpus-root` a driver is given (user: "corpus-root fallback is fine"), else its stem;
B2AI's hint builder declares its stem, so every B2AI key is unchanged. (b) ADMIT records it with the identity; the
corpus table gains a `recording_id` column, and the triage table, the review pages, the replay diff and the manifests
join on it; the stem is a display label. (c) The identity check refuses a repeated recording id (today it reports a
repeated stem). (d) Run folders keep `<stem>_<stamp>` under participant and session.

**9c — the replay marker.** (a) The marker also records a fingerprint of the session-floor and cohort readings the
replay folded, so a replay after `extend_session_floor` or `extend_cohort` reran replays again; both passes are
confirmed not to touch the marker. (b) A marker from before this records none and does not match, so each such run is
replayed once more (option i, CPU only). How the release runs set `--force` is [UNVERIFIED — assumption]; the
evidence jobs set it only through `FORCE` (`item8/replay.sbatch:27`).

**9d — "cohort".** (a) The recruited study cohort is renamed "recruited conditions": `cohort.py` to
`recruited_conditions.py`, `verdict.cohort_conditions` to `verdict.recruited_conditions`, the profile folder
`cohort_conditions/` to `recruited_conditions/`, B2AI's profile with them. The COHORT stage keeps its name, which
every store records. (b) The ledger's condition label becomes `recruited` (was `cohort`), readers accepting `cohort`
in older stores (option i): B2AI's ledger entries for listed conditions differ in that value, its decisions do not.

**Checks.** B2AI unchanged, as in 5e, 11e and 6: 5 paired fresh runs and the 192-run replay identical (entity ids
normalized), but for the ledger's condition label (9d) and the marker's new fingerprint (9c), each listed; the
baseline values check; the suite as a SLURM job. A flat-named corpus (`p01/s2/recording.wav`) keeps its recordings
apart in the tables; a hint declaring two target speakers is refused; a replay after `extend_cohort` reran replays.

**Results (2026-11-04, branch `refactor/triage-speakers` off 252941fc6, evidence in `triage_env_2026_10_22/s9/`).**
- B2AI unchanged but for the renamed ledger keys: in the 5 paired fresh runs and all 192 replay copies, exactly one
  entity per recording differs, the newest PII ledger, and only in four key names (`cohort_diagnoses` to
  `recruited_diagnoses`, `counts.cohort_condition_n` to `counts.recruited_condition_n`), each with the same value
  (entity ids normalized). Decisions are the same, and so are the reviewer readings each side reused or must read
  again (59 needing a reread, 133 with none, on both).
- The `recruited` label value itself was not exercised: no newest ledger in either set lists a condition. The two
  replay copies whose earlier ledgers list one (anxiety) keep `cohort` there, as history should, and the replay's
  own ledger lists none on both sides. The reading of older stores is covered by
  `recruited_conditions_test.py::test_a_store_from_before_the_rename_reads_under_the_new_names`.
- Replay marker: all 192 new-side stores carry `corpus_inputs` on the marker, none on the base side. The marker is
  an activity, so the entity comparison does not see it.
- Baseline values: identical to step 6's except the renamed setting (`verdict.cohort_conditions` to
  `verdict.recruited_conditions`, same value `bridge2ai_voice_adult_2026-09-04`).
- Suite: 4,380 passed, 20 skipped (triage and scripts) on the snapshot; on the working tree, triage, scripts, text
  and utils gave 6,634 passed with the same 12 failures under `src/tests/text` and `src/tests/utils` as the base tree.
- Deviation from 9b(a), same outcome: B2AI's hint builder does not declare `recording_id`. The id is recorded only
  when it differs from the stem, so B2AI's stores gain no attribute and its keys stay its stems.
- The checks above are unit tests: `identity_test.py::test_the_recording_id_is_declared_else_the_corpus_path_else_the_stem`,
  `identity_check_test.py::test_a_repeated_recording_id_is_refused_before_a_run_and_reported_after`,
  `identity_test.py::test_more_than_one_target_is_refused`, and
  `extend_replay_decisions_test.py::test_a_marker_matches_only_the_corpus_readings_it_left`.
- Found while building it: the first version applied a declared voice after SPEECH read its enrollment settings, so
  the tests raised a KeyError (`enrollment_model_id`); the declared voice is now applied first, before any evidence ran.
- Review (2026-11-05, high effort, on 538143e01; each finding checked by a verifier): ten findings, eight confirmed,
  one real but mostly older (7), one rejected (8). Fixed in 732231a18:
  - 1: `--corpus-root` does not follow symbolic links (a git-annex corpus's id is the path under its root).
  - 2: a replay drops a target speaker or declared id its hint no longer declares, and keeps an id read off the
    corpus path; the identity records each id's source.
  - 3: replay manifests carry `target_speaker` and `recording_id`, so COHORT groups by the declared target.
  - 4: a review record's id is the one its run recorded.
  - 5: a declared voice the probe cannot use leaves the enrollment given as the target, with a note.
  - 6: the after-run identity check reads the table's `recording_id` and `target_speaker`.
  - 7: `triage_audio.py` checks the identity before the run (exit 2).
  - 9: a manifest row naming another target speaker than the run's is stale and refused.
  - 10: the Free Speech review page explains the `recruited` kind.

  Finding 8 (replay and cohort passes never settling) was rejected: they settle after one extra round, now tested.
  B2AI takes the unchanged paths, so the evidence was not rerun. Suite: 6,645 passed, the same 12 failures under
  `src/tests/text` and `src/tests/utils` as the base tree (job 27984791).

## Step 10 — values fitted from a pilot run (added 2026-11-03; agreed 2026-11-05)

**What it is for.** Triage flags some recordings as unusual for their dataset: a cough much longer than the
dataset's coughs, a participant louder than almost every other, a session much noisier than that participant's
others. Each needs a number that depends on the dataset: B2AI's cough cut-off is 1.877 s because 97.5% of its
coughs are shorter. Eleven profile values are of this kind (kind b). Until this step a second dataset had to supply
all eleven before triage would run, and B2AI's came from one-off analyses that mostly survive only in a scratch
folder. A `fit` command computes them from a first run's corpus table, the way B2AI's were computed. It changes
nothing for B2AI; it is what makes the reference profile usable by another dataset. Walked through with the user on
2026-11-05 (user: "just finish step 10 if we think it is actually a useful functionality").

**10a — a pilot run** (option a). A profile with `pilot: true` in its manifest may leave its corpus statistics
empty. The start check does not ask for them; each check that reads one is skipped (a COHORT bound flags nothing,
the count-in joins no extent, the replaced-task rule's non-lexical part reads no talk, an atypical bound reads
nothing), and the consistency check records its cosine for the fit. The run record says `pilot: true` and lists
the values it left unfitted. After the fit the pilot runs are replayed (CPU only), and `pilot: true` comes out.
None of these values feeds a reading the fit reads, so one pilot, one fit and one replay settle it.

**10b — the inputs are corpus-table columns** (option a). The table (schema 32) gains: COHORT's participant median
level and clipped share, the session's level and SNR shift from the other sessions, the session consistency cosine,
the participant's non-match share and compared count (each repeated on the group's recordings), the median cough
event, the syllable rate and whether the syllable task was found, the count-in-to-phonation gap, and lexical words
per second of the response. `fit` reads only the table. The order: pilot run, COHORT pass, corpus table, fit,
replay.

**10c — each value's method is written once** (agreed). Each kind-b entry in `data/profile_values.yaml` carries a
`method`: the column, one value per recording, session or participant, filters, the statistic (median, percentile,
percentile band, Tukey fence with k = 1.5) and the rounding. `fit.py` follows only that. Four of B2AI's values do
not follow a clean method: the count-in limit (p99 2.9 s, rounded up by hand to 3.0), the cough cut-off (pooled over
every cough task though its note said "its family's"; the note is corrected), the DDK band (only recordings where the
task was found; now written into the method), and the free-speech word rate and per-task response medians (no
script, and a description that cannot be literally true). B2AI keeps its numbers, each marked in its profile as a
deliberate difference (option i). The per-task response medians gain a profile-values entry, so a new dataset's
template asks for them.

**10d — the check** (option a): (1) *method*: fit.py's calculation, over the r36 readings the original analyses read
(COHORT `--readings-out` over r36, and the r37 extent measure over r36), against B2AI's values; (2) *whole chain*:
the corpus table rebuilt from the release tree's r37 runs (read only; written to scratch), `fit` run on it, each value
listed beside B2AI's with the r36-to-r37 difference; and the made-up dataset's template profile as a pilot.

**10e — fitted.yaml** (option a). `scripts/fit_profile.py` writes the profile's `fitted.yaml`: the values, nested by
data file, with what they were fitted on (table, sha256, rows, schema, senselab version, date) and each value's count.
It sits above the profile it extends and below the profile's own files, so a value a dataset sets itself (B2AI's
3.0) wins, and a refit rewrites only `fitted.yaml`. The driver lists each value the profile sets itself beside the
fit. A refit needs `--force`; a table older than the fit columns is refused. The template writes a new profile as a
pilot and says how to fit it.

**Results (2026-11-05, branch `refactor/triage-fit` off 732231a18, evidence in `triage_env_2026_10_22/s10/`).**
- Method, over the r36 readings: fit.py reproduces B2AI's value exactly for the loudness bound (-9.343, n = 1,527
  participants), both session-shift bounds (17.783, 15.117; 351 sessions), the consistency floor (0.63; 313
  sessions), the shared non-match share (0.169; 214 participants) and the DDK band ([2.235, 7.692]; 7,815
  recordings); the count-in limit fits 2.9 (n = 1,232), B2AI's deliberate 3.0. The clipped share fits 0.0 here: those
  readings carry an older clip reading, which B2AI's analysis replaced with the r36 table's (unreadable to us). The
  cough cut-off, word rate and response medians are not in those readings. A first run found the shift bounds 0.001
  off: the new column rounded the shift before the fence; it no longer does.
- Whole chain: the corpus table rebuilt from the release tree's 62,550 r37 runs (64 slices, read only; no release
  file changed), then `fit_profile.py` with B2AI's profile, written to scratch. Identical to B2AI: the cough cut-off
  (1.877; 2,643), the DDK band, and, with participant and session filled from the B2AI stems (the r37 runs predate the
  identity record), the clipped share (0.475), the loudness bound, both shift bounds and the consistency floor. The
  count-in limit fits 2.9. r37 differs from r36 on three: the shared non-match share 0.182 (108 participants; r36's
  readings give 0.169), the free-speech word rate 2.07 (B2AI 2.03), and three of nine response medians (free-speech
  26.77 vs 26.75, free-speech-v2 25.87 vs 25.89, story-recall-v2 73.59 vs 73.38; every count equals B2AI's). The fit
  also gives productive-vocabulary a median (5.19), a free-response task B2AI's list leaves out. B2AI keeps all its
  values; the driver lists each as the profile's own, which wins.
- B2AI unchanged: 5 paired fresh runs and the 192-run replay, every store entity identical (ids normalized), the
  same decisions and replay rows, the same reviewer readings reused and reread (59 / 133); the baseline values
  identical to step 9's. Suite: 6,677 passed, 34 skipped; the 12 failures under `src/tests/text` and
  `src/tests/utils` fail on the base tree too (job 28009253). mypy over the touched files: two errors, both on the base
  tree (`profile_template.py:271`, `run.py:187`).
- Made-up dataset: the template writes a pilot profile, whose start check asks for every value but the ten
  required corpus statistics and lists them unfitted (`profile_template_test`), and `fit_profile_test` fits a
  pilot table into such a profile, which then reads the fitted values. A pilot run, fit and replay over real
  recordings of a second dataset was not run: there is no second corpus.
- Review (2026-11-05, high effort, on bc6ea7fb3; findings checked by verifiers): nine confirmed, one rejected for
  the fitted values (8). Fixed:
  - 1: the template wrote each corpus statistic as a blank in the profile's own file, which hid the fit; it now
    writes them commented out (and a mapping holding only them), so fitted.yaml supplies them.
  - 2: the consistency outcome is decided when COHORT runs and the replay skips COHORT, so the order is now: pilot,
    COHORT, table, fit, COHORT again, replay.
  - 3: the word rate counts words as the replaced-task rule counts talk (no unconfirmed word, no filler). B2AI's task
    list marks productive-vocabulary free-response, which pulls the fitted rate down (on r37, counting as before this fix:
    2.07 with it, 2.16 without); left as the task list says, B2AI keeping its own 2.03.
  - 4: the driver names a value as winning only where the profile's own file or the run's data section writes it,
    names an own blank as hiding the fit, and says when --out writes a file the profile does not read.
  - 5: a fitted per-family mapping may add families a lower profile's mapping lacks.
  - 6: a missing or unreadable table, a malformed profile and a profile without a task list exit 2; the write is
    atomic; --force sets an unloadable fitted.yaml aside as fitted.yaml.previous.
  - 7: the shared non-match fit keeps B2AI's population (participants with recordings_min compared recordings); its
    method text now says so and that the rule counts non-matching recordings (13 participants on r36, too few to fence).
  - 8: no fitted input reads the voice extent; the driver notes that a pilot's count-ins stay outside it until the
    replay.
  - 9: B2AI's markers say only "deliberate difference from fit (plan.md, step 10)"; the reasons are above.
  - 10: misplaced docstrings fixed; the table code uses COHORT's own reading names; unfitted values read each file
    once.

## Step 11 — stages on and off, models by name (light version, agreed 2026-11-04; before step 7)

**What it is for.** Steps 1-5 make what triage reads a dataset's, but not which stages run or which models they use:
the made-up lab corpus of step 4, with no breathing tasks and no recruited conditions, still has to supply AIRWAY and
cohort values (found in step 5c), and a pediatric dataset cannot swap the adult speech recognizers without editing
senselab. This step gives a dataset those two choices and nothing more; the larger modular redesign is a separate
effort (`specs/20261104-triage-modular-design/design.md`, decision 3).

**11a — the stage table.** `run.py` drives a table of stages, each with its name, its call, the stages it needs and
whether a workflow may leave it off, instead of hand-written calls. The order and the calls are unchanged, so a run
writes the same store. ADMIT, PREPROCESS, SESSION, BACKGROUND, TAXONOMY, routing and VERDICT may not be off: every
later stage and the verdict read them. AIRWAY, SPEECH, VOICE, QUALITY, COHORT, REDACT, REVIEW and SECOND_OPINION may.
REDACT and REVIEW need SPEECH; SECOND_OPINION needs REVIEW.

**11b — `workflow.stages`.** A config setting lists the stages that run; senselab's default lists all of them. A stage
left off is recorded `skipped` with the note "off in this workflow", the same way an unrouted branch is recorded now;
a stage whose need is off is skipped with "off in this workflow: needs SPEECH". A list that leaves off a stage that may
not be off, or names an unknown stage, is refused before the run. The release stays conservative without new rules:
with SPEECH off a recording with words has no transcript, so the verdict releases nothing (`no_transcript`), and with
REDACT off a recording where PII was found is withheld (`redaction_owed`). Replay (`drive_decisions`) reads the same
setting, and the corpus pass for a stage that is off (`extend_cohort`, `extend_quality`, `extend_llm_review`,
`extend_second_opinion`) refuses to run.

**11c — models by name.** Six model roles are already settings (the diarizer, the second diarizer, the speaker
embedding, the separator, the PII detectors, the LLM reviewer and second opinion). The ones still fixed in code
become settings with today's values as senselab's defaults:
- `asr.recognizers`: the list of speech recognizers, each with a name, model, revision, how it times words and its
  options; the consensus transcript is already built over however many there are, so one recognizer or three work;
- `ast.model` and `ast.revision`: the AudioSet sound-event classifier (a checkpoint with the same label set);
- `residual.enhancer`: the enhancement model the background residual is taken against.
YAMNet, HeAR, the phonetic posteriorgram and SQUIM stay fixed: senselab has one implementation of each, so another
model is a new component, not a setting (deferred with the modular design).

**11d — values follow stages.** Each `data/profile_values.yaml` entry gains `stages:`, the optional stages that read
it; an entry without it is read whatever runs. A required value is required only when one of its stages runs, so the
start-of-run check and the template ask only for what the workflow reads. Assigned by the stage that owns the file
or section (breathing and cough → AIRWAY, phonation → VOICE, speaker, DDK, PII and task speech → SPEECH, redaction →
REDACT, cohort → COHORT, quality → QUALITY, the reviewer's flags → REVIEW, the second opinion's → SECOND_OPINION);
where the verdict reads a value of an optional stage it must cope with the stage being off, which the 11e run checks.

**11e — checks.** B2AI, running every stage, is unchanged: 5 paired fresh runs, the 192-run replay, and the baseline
values check, as in 5e; the suite as a SLURM job. A profile extending the reference with AIRWAY, VOICE, QUALITY and
COHORT off, and their values left blank, is accepted by the start-of-run check, and a run of it over recordings of
each task kind completes with those stages skipped and the others' decisions unchanged.

**Results (2026-11-04, branch `refactor/triage-workflow` off 6e546cfe6, evidence in
`triage_env_2026_10_22/s11/`).**
- B2AI unchanged: 5 paired fresh runs, same decisions and every store entity the same once entity ids (derived from
  the run id) are normalized, as in 5e; the 192-run replay, 192/192 entity-identical and the same decisions; the
  baseline values, unchanged except the six new keys (`asr.recognizers`, `ast.model`/`revision`,
  `residual.enhancer.model`/`revision`, `workflow.stages`).
- The speech-only profile (B2AI's, with AIRWAY, VOICE, QUALITY and COHORT off, their cohort and quality files removed,
  SPEECH tasks only) passes the start check with nothing missing; over the DDK, Rainbow Passage and free-speech
  recordings it completes with the four stages skipped as off, and each decision (pass, as is, same reasons) is the
  full workflow's.
- Found by that run and fixed: the verdict expected QUALITY's join whenever a background reading existed, so with
  QUALITY off every recording went to review (user, 2026-11-04: option (a), a stage left off leaves no uncomputed
  reading); and a config naming a cohort-conditions list the profile lacks failed VERDICT on every recording, now
  refused by the start check. `verdict.cohort_conditions` is read for the reviewer's condition quotes, so its stage
  is REVIEW, not COHORT.
- Suite: 4,339 passed, 20 skipped (job 27899647).

## After this plan

- **A pediatric profile:** the first second protocol.
- **senselab's data folder holds no dataset data:** checked by a test that fails if a packaged data file
  names a dataset's tasks.
- **Fitting the rest programmatically:** step 10 fits the statistics of a pilot run; fitting the listened detector
  values (kind (c)) needs labelled recordings and train/test split support in senselab, a follow-up issue.
- **A modular triage** (`specs/20261104-triage-modular-design/design.md`): stage descriptors and a workflow spec,
  sub-workflows, corpus passes as workflows, PREPROCESS as tasks, the verdict as a fold, the explicit settings
  context; weighed against pydra. A separate effort after step 11's light version.
- **Multi-speaker interactions** (user, 2026-11-04): dyadic and conversational recordings, where more than one voice
  is expected. Step 9 shapes the target-speaker declaration as a list and refuses more than one; making triage read
  interactions (which voices are expected, each one's task conformance, consent and release per person) is its own
  walkthrough.
- **senselab alpha's resampler** (#632, c4bb902b7, 2026-11-04): `resample_audios` now band-limits with a Kaiser sinc
  filter, which PREPROCESS uses; merging alpha into the stack changes every resampled stream, so it needs its own
  evidence run.
- **A shared store of dataset profiles** (user, 2026-11-03): a separate project where datasets publish and copy
  profiles. Out of scope until there is a second dataset.
