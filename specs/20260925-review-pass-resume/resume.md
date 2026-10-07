# Resuming the review pass

## State at r17, 2026-10-07 — AIRWAY move, VOICE redesign, r17 parquet

Read this section first. Everything below it is history.

### Where things are

| | |
|---|---|
| Code branch | `fix/policy-v8` (PR #581 into `triage`). Tip at writing: `8b068a02`. |
| Corpus | `/orcd/scratch/bcs/002/satra/triage_r9_20260929/out` (62,550 runs), re-folded in place (r17c re-fold) |
| Latest parquet | `recording_vectors_r17` (schema 23), at `/orcd/scratch/bcs/002/satra/recording_vectors_r17/`; local copy `~/Downloads/recording_vectors_20261006_r17/` |
| Pinned checkout | `senselab-r26` (moved forward through `c3e97dd4` → `a8a95cea` → `8b068a02`, each time only with nothing running from it) |
| Owner labels | `~/Downloads/triage_listening_labels_20261006.csv` (~180 rows), plus per-set `index.csv` in `voice_check_20261006/` (and `discard_cause/`), `discard_contested_check_20261006/` and `breath_review_check_20261006/` |
| Derivatives | still at the r11 state; the r17 sync is next |

**Vocabulary (feat/task-events, 2026-10-07; recording_vectors schema 25).** Verdict is
`pass` · `review` · `discard`; `flag` is now `review`, and `rerun` is `run_status = incomplete` with
verdict `review` and reason `not_measured`. Release is `as_is` · `redacted` · `withheld` (redaction
holds only), empty on a discard. Each recording also carries a `reason` from
`data/decision_reasons.yaml` and its decision evidence; `triage_decisions` and `triage_evidence`
are written beside recording vectors. The r17 counts below use the old names.

### r17 counts

| Branch | discard | flag | pass | rerun | Total |
|---|---|---|---|---|---|
| AIRWAY | 1,814 | 688 | 10,363 | 144 | 13,009 |
| SPEECH | 1,390 | 2,568 | 37,071 | 178 | 41,207 (unchanged from r16) |
| VOICE | 79 | 268 | 7,870 | 88 | 8,305 |
| no family | 29 | – | – | – | 29 |
| Total | 3,312 | 3,524 | 55,304 | 410 | 62,550 |

From r16 (3,914 / 6,741 / 51,435 / 460):
- 599 discards and 3,876 flags now pass;
- 616 passes now flag, mostly breath review and background speech;
- 32 passes are now discarded, quiet or noise-only breath recordings (not yet listened to);
- VOICE flags fell from 3,164 to 268; AIRWAY contested discards are 42, down from 335 before the narrowing.

### What landed (fix/policy-v8)

- `3eede30d`: **AIRWAY move.** Breath, cough and background speech are measured in the AIRWAY branch, and VERDICT only reads them. Discard is released as `discarded` with no task-audio cuts. `discard_contested` added; Clef is out of airway decisions.
- `8864ed00`, `dd8635c9`: **breath fixes.** Vocalised exhales count, along with edge phases and acoustic speech; word timing uses consensus; speech splits the train only across more than two cycles; an edge phase needs ≥5 dB over the floor.
- `c3e97dd4`: **the breath train decides presence.** Inside the review band the old measure separates review from discard.
- `7075d730`: **VOICE phonation redesign** (schema 23).
- `a8a95cea`: **replay driver.** A node error now gives row status `errored`, and `--force` replays past the marker.
- `8b068a02`: **contested narrowed.** A contest needs a train rise ≥10 dB, plus ≥3 detector events for the uncounted families.
- **Specs:** `20261006-voice-phonation/`, `20261006-airway-move/`, `20261007-task-events-in-background/` (the next iteration).

### The r17 run

- Replay of 21,492 recordings: the airway families, all 460 reruns and all 8,305 VOICE recordings. Then the full re-fold, task audio, recording vectors and merge, using a chain helper because of the QOS job-count limit.
- **FFmpeg incident:** the scratch miniforge (`~/orcd/scratch/miniforge/lib`) was found emptied. Every re-fold slice died at the first released-audio decode, and 144 replay SPEECH runs failed silently. `env_common.sh` now prepends `~/ffmpeg/lib` (backup `env_common.sh.bak-20261006`). The 144 were re-replayed with `--force`.

### Owner decisions this cycle

- **Disordered voice is data.** Rough, breathy, hoarse, creaky or unsustained phonation is measured, not flagged.
- **The VOICE extent is the attempt,** from energy continuity starting at the inhale, and weakly voiced attempts are included. Discard only when nothing rises above the floor.
- **Broken or restarted holds merge into one extent.** There is no minimum hold. A mic shutoff during phonation flags.
- **Hum:** the extent comes from raw, with a residual mains-line guard; the enhanced stream is not the default.
- **Annotations, not flags:** `route_mismatch` when the owning branch found its task, and `task_mismatch`.
- "it's ok for now to triage some noisy recordings"; "don't overoptimize on specifics. determine the more general components".
- **Clef is out of VOICE and AIRWAY decisions.**

### Open

- **Next iteration:** the general task-events design (`specs/20261007-task-events-in-background/`): a background model, event detection, rhythm as a prior, an activity-bounded extent, and evidence kept separate from the decision. The five owner questions are in that spec.
- **The enhanced stream is not a decider**; it is only a hint for reviewers. On the owner labels it releases noise as readily as it rescues breathing.
- **The breath review band is 486 flags**, 18.9% of the breath family against ≤3.4% for counted families. Acceptable for now.
- **VOICE:** the break minimum (0.25 s) and the glide bound (6 st) are unfitted. 68de3829 reads as the wrong direction, 5b4817c2's extent runs past the glide, and the hum guard's false-fire rate is unchecked.
- **Deferred:** multi-speaker, and background speech in voiced families.
- **318 `route_unexplained` reruns** need a routing-rules fix or a fresh run; the replay cannot clear them.

### Working rules learnt this round

- Listen before changing a rule. Every listen batch overturned at least one fitted cut-off.
- Fix classes, not recordings. Fit a few parameters jointly over all labels, with a held-out split.
- Pull samples with original + enhanced + dyngain audio and a four-panel figure, and give an `index.csv` with an empty `owner_note` column.
- Agents stall on large single edits: ask for edits under ~60 lines, commit per unit, and background tests.
- Never hold a Slurm wait inside an agent. For a re-fold-only change, pin the checkout and re-submit `refold` → chain helper (`task_audio` → `rvec` → merge).
- After any replay, grep the rows for non-ok status and the logs for `libtorchcodec`.

## History, 2026-10-06 — redaction v9, readable triage state, breath and cough measures

### Where things are

| | |
|---|---|
| Code branch | `fix/policy-v8` (PR #581 into `triage`; #567 merged 2026-10-04). Tip at writing: `217b22fb`. |
| Corpus | `/orcd/scratch/bcs/002/satra/triage_r9_20260929/out` (62,550 runs), re-folded in place |
| Latest parquet | `recording_vectors_r16` (schema 21, policy 9, built at `1a73d8d0`); local `~/Downloads/recording_vectors_20261006_r16/`. Superseded by the work below; not yet re-built. |
| Pinned checkouts | `senselab-r15` … `senselab-r25`, one per run (job ids in `triage_r9_20260929/jobs.txt`) |
| Owner labels | `~/Downloads/triage_listening_labels_20261006.csv` (103 rows: breath, empty-route, cough listens, each with the owner's note); copy at `/orcd/scratch/bcs/002/satra/tmp_bext/` |
| Review figures | `~/Downloads/breath_extent_check_20261006/` (`fixed/`, `fixed2/` pending) and `~/Downloads/cough_check_20261006/fixed/` |
| Derivatives | still at the r11 state (finalized 2026-10-03); not re-synced since |

### Settled since r11 (all on `fix/policy-v8`)

- **Redaction policy v7 → v9** (`specs/20261003-redaction-policy-v7/`, `specs/20261004-redaction-policy-v8/`). The rules:
  - Names are masked by default. A public figure is released only with human approval (`redaction.name_approvals`); kinship words are released.
  - Years, months, holidays, day-of-month and ages are locked masked. Seasons, weekdays, relative time and time of day are released (v9).
  - Places below country are locked masked, unless the reviewer gives a closed-set reason (historical, fictional, landmark, task content). Countries are released unless the reviewer judges triangulation.
  - Conditions are never masked or withheld. Specific organisations are masked.
  - Lower-case detector masks are released, in every language. Task text (stimulus, target word, instructions) is never masked; per-family guidance is in `task_guidance.yaml`.
  - The reviewer can re-label a "person" that isn't a person. Spanish closed-class words count as function words.
  - Prompt v9. Clef question set 3 (other_voice, instructions_spoken, policy_identifier_present).
  - `task_guidance.yaml` is hashed by bytes into the cache identity, so it's excluded from the YAML formatter (`7b409a81`).
- **Readable triage state** (DAG-review proposals 1, 3, 4; `specs/20261004-dag-review/`):
  - `ground_keys` for every flag, discard and withhold, with no transcript words in grounds.
  - Empty recordings discard.
  - A `rerun` state for pipeline failures.
- **Truncated and absent tasks** (`specs/20261005-truncated-capture-discard/`):
  - `too_short_for_task` (per-family minimum durations in data/).
  - `declared_task_absent`: the declared family's owning branch decides.
  - Bracketed tokens aren't words; `[cough]`/`[咳]` support a cough task.
  - `owning_branch_input_absent` → rerun, not discard.
- **Breath tasks** (`specs/20261005-breathing-pattern/`):
  - A breath train on the denoised pre-emphasised spectrogram: 9 subbands to 7.5 kHz, coherent bursts, template matching, runs split at gaps or speech.
  - Phases are counted; breaths = phases / 2.
  - A vocalised exhale counts as the exhale phase (owner, 988c1609).
  - The extent is the run's span and supersedes AIRWAY's.
  - A narrow review band, `breath_review_low_confidence`, about 4% of kept breath recordings.
  - Agreement with owner labels: 53/56.
  - HeAR/YAMNet and the old vetoes are context only. Classifier-evidence rules were tried and rejected on the owner's labels.
- **Cough tasks** (`specs/20261006-cough-pattern/`, `cough_pattern.py`):
  - Sharp coherent broadband onsets (≥ 12 dB rise across ≥ 60% of bands); tails, a second phase and the preparatory inhale attach to the cough.
  - The extent is the cough train, and the inhale start runs back to 8 dB over the floor (`217b22fb`).
  - `no_cough_captured`, `cough_review_low_confidence`; a hard cough needs at least 1 onset.
  - 23/23 owner labels agree. AIRWAY's HeAR-gated detector missed coughs that the classifiers scored low.
- **Background speech in airway tasks** (`background_speech.py`): `background_speech_in_task` (flag) reads what enhancement removed, from the residual stream and its YAMNet scores. It fires on 6ca9935e's intercom; ~1% of the cough and breath samples.

### Owner decisions not yet in code

1. **`task_mismatch` is an annotation, not a flag, for every airway family.** It goes in a separate `annotation_keys` list; a recording with only that key passes. The breath agent is implementing it; the cough code reuses its constant.
2. **`rerun` should be rerun.** The 460 r16 rerun recordings get their missing derivatives recomputed (HeAR, TAXONOMY) in the airway replay, and are held until then.
3. **Discard is not released** (owner, 2026-10-06: "yes, discard should not be released"). Today 3,880 discarded recordings carry a release value and 2,111 have task-audio cuts. Planned: release `discarded` for a triage discard, and no task-audio cuts.
4. **Move breath, cough and background-speech measurement into the AIRWAY branch** (owner: "yes"). VERDICT then only decides, restoring branches-measure / VERDICT-decides. This replaces AIRWAY's HeAR-gated detector outright, and needs a targeted CPU replay of the airway families (~13,000 recordings).
5. **Figures use the fixed-gain pre-emphasised spectrogram only.** Dynamic gain inverted loudness on 988c1609; keep `_dyngain.wav` for listening.

### In flight at writing

- **Breath-phase agent, not pushed:**
  - split phases at voicing boundaries, no overlapping phases;
  - trailing speech detected acoustically per segment (ba1d1459), the ≥ 5-word cut was too coarse;
  - edge phases already elevated at file start (ecc63817);
  - the `task_mismatch` annotation;
  - re-render into `fixed2/`.

### Next, in order

1. Land the breath-phase work. Then the AIRWAY move, the discard release, rerun handling and the annotation, all one branch unit.
2. One cluster run:
   - an airway replay of ~13,000 recordings plus the 460 reruns, recomputing missing HeAR/TAXONOMY;
   - a full re-fold;
   - task audio, then the parquet (r17).
3. Free-speech review page, evaluations, derivatives sync with `finalize_*.sbatch` and the verifier; merge #581.

### Open and deferred

- Lone "El": released; owner, 2026-10-06: it "does not matter". Closed.
- Multi-speaker in speech tasks: deferred again (owner, 2026-10-06). Signals are noisy; better models later, not gate tweaks.
- Clef: left out of all airway (breath and cough) decisions (owner, 2026-10-06). It stays the second opinion on lexical review only.
- 6ca9935e breath extent: fixed in `fixed2/` (owner, 2026-10-06: "looks good"). 0dc15213: some modulation but no clean signal; flagged or discarded is acceptable (owner). Closed.
- Background speech: cough and breath tasks now; the voiced non-speech families (sustained vowels, glides, DDK…) are tested on a labelled sample later, since residual speech evidence there is unproven (owner agreed, 2026-10-06).

### Working rules learnt this round

- Listen before changing a rule. Every listen batch overturned at least one fitted cut-off.
- Pull samples with original + enhanced + dyngain audio and a four-panel figure, and give an `index.csv` with an empty `your_listen` column.
- Agents stall on large single edits: ask for edits under ~60 lines, commit per unit, and background tests.
- Never hold a Slurm wait inside an agent. For a re-fold-only change, pin a fresh clone and re-submit `refold_*` → `task_audio_*` → `rvec_*`.

## Settled, 2026-10-03 — r11: the second opinion is Clef 27B

r11 is r10 with Nimble replaced by Clef 27B (`ollama clef:27b`, Ollama 0.35.1, weights
`sha256:6c02216a…`, pinned in config; `clef-flash` fails on `/v1/systemone`, ollama#18769). Code:
model-neutral `second_opinion_*` columns, parquet schema 18 (`4266c2db`); load the model by a decision
request, not `/api/generate` (`af416ba0`); `second_opinion.load_timeout_s` 1200 (`72a45e6e`). Run:
Clef over the 15,113 review recordings from `senselab-r13`/`senselab-r14`, then a full re-fold and
`recording_vectors_r11` (local `~/Downloads/recording_vectors_20261003_r11/`). 13 slices first failed
on the 300 s model load, with weights read from the /orcd/data capacity tier by several slices per
node; the store is now copied to flash at `/orcd/scratch/bcs/002/satra/ollama-models-flash/`.

Result: 15,111 answered, 2 nothing to read. Release unchanged (60,297 / 1,140 / 1,081 / 32); flagged
9,315 (−104), all from the second-opinion ground, 315 → 147 recordings (named diagnosis 70, another
voice 67, instructions spoken 11). Clef vs Nimble: agreement at 0.5 is 91.6–99.4% per question; Clef
is confident "yes" far less often on another voice (112 vs 228) and Safe Harbor (318 vs 1,286).
Evaluations: `evaluations_r11_20261003/README.md`. Derivatives synced and finalized
(`tmp_deriv/finalize_r11.sbatch`): 2,221,738 files and 1,404,927,276,966 bytes on both sides, 0
symlinks, 0 stores naming scratch. Thresholds stay 0.8/0.2, unfitted.

## Settled, 2026-10-02 — r10; history from here

The corpus is the r9 tree, re-read on reviewer prompt v6, second-opinioned by Nimble, re-folded in full
and cut to task extents. r9c and r9d are superseded.

| | path |
|---|---|
| corpus (62,550 = every in-scope BIDS WAV) | `/orcd/scratch/bcs/002/satra/triage_r9_20260929/out` |
| checkouts, pinned | `senselab-r5` @ `29357489` (v6 review); `senselab-r10` @ `ac0bd3ba` (r10 chain, parquet, page); `senselab-r11` @ `b013e274` (fix17); `senselab-r12` @ `9095b81a` (fix20) |
| job ids, submit order | `triage_r9_20260929/jobs.txt`, `RUN.md` |
| review manifest (15,113) / v6 readings | `triage_r9_20260929/review/`, `rows_v6/` |
| Nimble readings | `triage_r9_20260929/second_opinion/rows/` |
| parquet, schema 17, dictionary embedded | `recording_vectors_r10/`; laptop `~/Downloads/recording_vectors_20261002_r10/` |
| page / evaluations | `free_speech_page_20261002_r10/`, `evaluations_r10_20261002/README.md`; laptop `~/Downloads/free_speech_review_20261002_r10/` |
| derivatives (inside the release) | `/orcd/data/satra/002/datasets/b2aivoice/4.0-release/adult/bids_adult_2026_09_04/derivatives/senselab-triage/` |

Code since r9c, on `design/triage-workflow-dag`:

| commit | what |
|---|---|
| `29357489` | reviewer prompt v6: time expressions, a named identifier must be proposed, productive-vocabulary target word is task content, tolerant quotes |
| `d95b0b2a` | merge of `design/nimble` (schema 15), `design/task-extent-audio` (renumbered to 16), `design/airway-extent`: AIRWAY and RIG place a `task_extent` or record why not |
| `e1d9b203` | time-of-day and duration findings released by kind (`data/time_release.yaml`), schema 17 |
| `ac0bd3ba` | Nimble driver: no per-recording re-fold (the full re-fold follows), 4 parallel requests; atomic `ProvStore.write_jsonl` |
| `284a493c` | Ollama server: Vulkan off, model must be fully GPU-resident, a slice aborts on a dead server |
| `b013e274` | a replay carries an unchanged REVIEW reading forward, or reports `needs_reread` and exits 3; an unrun owning branch records `owning_branch_not_run` |
| `9095b81a` | `mint_live` in every writer (a retire-then-rewrite reused the retired id); ADMIT-refused recordings record their owner reason |

Runs, in order:

1. v6 full re-review of the 15,113 (`--force`), re-fold, parquet `r9d`.
2. r10 chain at `ac0bd3ba`: AIRWAY/RIG extent replay of 1,614; Nimble over the review manifest
   (15,111 ok, 2 nothing to read; one sweep array after slice 29 fell back to Vulkan and crashed); full
   re-fold under `second_opinion_on.yaml`; task-audio cuts; parquet `r10`.
3. `fix105/`: the extent replay had retired the v6 REVIEW of 105 recordings (an overlap check matched
   `stem` across manifests that disagree on the run timestamp, and reported 0). Re-reviewed, re-folded,
   recut.
4. `fix25/` at `b013e274`: 17 recordings with neither an extent nor a reason, replayed.
5. `fix20/` at `9095b81a`: 7 born-retired reviews re-minted, 5 born-retired cuts recut, 8 ADMIT-refused
   recordings given their reason.

Outcome: flagged 9,419. Release: 60,297 without redaction, 1,140 with, 1,081 withheld (cohort-condition
review 580, other-condition review 267, reviewer proposed hiding more 169, REDACT 65), 32 not assessed.
`llm_status` disabled 47,437 (exactly the recordings outside the review manifest), null 0. Nimble
disagrees confidently with the reviewer (0.8 / 0.2, `UNFITTED`) on 315. Every AIRWAY and RIG recording
carries an extent or a reason. Task-audio cuts: plain 60,560, enhanced 60,559, redacted 1,125; the 1,990
without a cut are exactly those without an extent.

Derivatives: every run root copied with symlinks followed — 2,221,738 files and 1,404,319,166,145 bytes
on both sides, 0 symlinks, 0 of 62,550 stores naming `/orcd/scratch`, checked by
`scripts/verify_derivatives_copy.py` (threaded walk, `senselab.utils.fastio`; ~8 min where serial `find`
and `grep -r` passes ran over 3 h). The earlier ~200 MB gap was `du` counting directory sizes.
`tmp_deriv/finalize_r10.sbatch` rewrites the top level (parquet, summary, dictionary, viewer, review page,
evaluations with `--delete`) and `dataset_description.json`, sets group `orcd_rg_hstor004_pi_satra`
2775/664 on them, and runs the verifier from the `senselab-verify` checkout.

Subprocess venvs install from committed hashed locks (`src/senselab/utils/data/venv_locks/`, 601f7670,
crisperwhisper fixed at be8220e4); all 22 built and imported on an H100 against their CUDA index. Model
loads under the new pins are not yet exercised.

Open for the owner:

- A lone "El" (Spanish article, or a place-name fragment) is released; unanswered.
- Nimble thresholds are unfitted (left as is, 2026-10-02): hand-label the disagreements per question, then fit.
- Second-speaker signals (separation, diarization) are noisy; the owner wants better models, not gate tweaks.
- The evaluate-triage panel has no r9c baseline; those parquets lack the columns.

The "future run for all" scheduled on 2026-09-30 (re-read every review-manifest recording on one prompt)
is done: the v6 full re-review covered all 15,113.

Cluster traps met this round: `squeue -j <id>` errors once a finished job is purged, so a watcher read
that as an unreachable cluster — list `squeue -u satra -h -r -o '%F %R'` and filter by id instead;
`afterok` on an array with one failed task never runs, so put a resumable sweep array between; an
agent holding a Slurm wait stalls — submit from the agent, hold the wait in the main session.

## r7, 2026-09-28 — history

The corpus is **r7**, replayed, fully re-reviewed and folded at `77e9273e`. r6 and r5 are superseded.
The cache commits (`7089b05f`..`e4301967`) sit on top and are merged; r7 predates them.

| | path |
|---|---|
| corpus (r7, 62,550 = every in-scope BIDS WAV) | `/orcd/scratch/bcs/002/satra/triage_r7_20260927/out` |
| checkout, pinned | `/orcd/scratch/bcs/002/satra/senselab-r5` @ `77e9273e` |
| job ids, submit order | `triage_r7_20260927/jobs.txt`, `RUN.md` |
| review manifest (15,193) / readings | `triage_r7_20260927/review/` |
| parquet, schema 11, dictionary embedded | `recording_vectors_r7/`; laptop `~/Downloads/recording_vectors_20260928_r7/` |
| page / evaluations | `free_speech_page_20260928_r7/`, `evaluations_r7_20260928/` (laptop `~/Downloads/…`) |

What r7 carries beyond r6: SPEECH places each finding on its own words (no whole-transcript or bridged
spans; unplaced findings stored with their text); one mask per finding; a reviewer redact entry on words
already masked is agreement, not hiding more; a released term is released at every occurrence; a
`speakers: more_than_one` reading flags for review; the reviewer is given the task's instructions,
stimulus and speech type (sidecar, with b2aiprep's curated registry vendored at 8c43256); recall tasks
subtract their story as task content; located gates are bound-only (schema 10→11); a data dictionary
for every parquet column, verified against the code by three independent passes.

Outcome: triage pass 52,830 / flag 9,691 / discard 29. Release: 60,578 without redaction, 1,422 with,
515 withheld (196 cohort-condition review, 151 other-condition review, 114 new reviewer redaction,
54 REDACT), 35 not assessed. 0 released with a new redact proposal; 0 with-redaction copies masking nothing.

Open for the owner: second-speaker flags on non-lexical tasks (~60); the other-condition review list
(164 recordings, many one-off phrases, some not conditions); the two fill recordings (one ASR hypothesis,
`not_assessed`); subprocess-venv dependency versions are not in any cache key.

Cluster traps met this run: node2119 (CUDA fault), node2621 and node3002 (slices hang to the time
limit) — exclude all three; the replay's wall time is dominated by hung nodes and by recomputation,
which the result cache (`SENSELAB_CACHE/results/`) now removes for the next run.

## In flight right now

| what | job | state |
|---|---|---|
| REVIEW over the corpus | **23879493** | 128 slices, `%36`, a100, 31/128 done, ~97 elements left |

It is **resumable**: a recording whose store already carries a `redaction_llm_annotation` written by
a **REVIEW** activity comes back `present` and costs nothing. So a cancelled or preempted array is
resumed by resubmitting the same sbatch — never by starting over.

```bash
ssh orcd 'cd /orcd/scratch/bcs/002/satra/triage_review_20260925 && sbatch review.sbatch'
```

Watch it:

```bash
ssh orcd 'ls /orcd/scratch/bcs/002/satra/triage_review_20260925/rows/slices/*.summary.json | wc -l'
```

## Monitors do not survive the session — restart them first

Every watcher in this campaign is a background shell held by the session that started it. A new
session inherits **none** of them, and nothing on the cluster notices: the array keeps running and
no one is told when it finishes. So the first act on resuming is to re-arm a watcher, before
anything else, or the pass completes silently and sits idle.

The watchers live in the job's scratch directory, which is **also not durable** — it goes when the
job is deleted. Treat them as disposable and rewrite them; the shape is what matters:

```bash
# poll the array, report only real progress, exit when it leaves the queue
prev=-1
while :; do
  read -r slices rows state <<<"$(ssh orcd '
    R=/orcd/scratch/bcs/002/satra/triage_review_20260925
    s=$(ls $R/rows/slices/*.summary.json 2>/dev/null | wc -l)
    r=$(cat $R/rows/slices/*.jsonl 2>/dev/null | wc -l)
    q=$(squeue -j <JOBID> -h -r -o "%T" 2>/dev/null | wc -l)
    echo "$s $r $q"')"
  [ "$state" = "0" ] && { echo "done: $slices/128 slices, $rows rows"; break; }
  bucket=$(( slices / 16 ))
  [ "$bucket" != "$prev" ] && { echo "slices $slices/128 · rows $rows"; prev=$bucket; }
  sleep 900
done
```

Two things learned the hard way about these:

- **Report on progress, not on state jitter.** A watcher keyed to the running-count fires every few
  minutes on a preemptable partition and says nothing. Bucket by completed slices instead.
- **A waiter that treats `PREEMPTED` as terminal exits early**, because a requeued job passes
  through that state and then carries on. Wait for the job to leave the queue.

Re-arm one for whichever of the three steps below is in flight, and one for the next.

## Pinned checkouts — do not cross them

Two checkouts, two jobs, and they must stay separate. Checking one out to a different commit while
a job runs from it fails that job's guard (`exit 74`), and I did exactly this once today.

| checkout | used by | pin |
|---|---|---|
| `/orcd/scratch/bcs/002/satra/senselab-review` | the REVIEW array | `5165925f` |
| `/orcd/scratch/bcs/002/satra/senselab-rvec` | parquet, re-fold, page | current HEAD |

Both now have `origin` at GitHub. `senselab-rvec` pointed at another scratch checkout until today.

## Trees and artefacts

| | path |
|---|---|
| corpus (r4, verified 62,548) | `/orcd/scratch/bcs/002/satra/triage_r4_20260924/out` |
| original corpus, for `run.json` | `/orcd/scratch/bcs/002/satra/triage_design_20260919/run/out` |
| hints (required by every driver) | `/orcd/scratch/bcs/002/satra/triage_design_20260919/scope` |
| review manifest, 62,519 rows | `/orcd/scratch/bcs/002/satra/triage_review_20260925/review_manifest.jsonl` |
| reviewer-on config | `/orcd/scratch/bcs/002/satra/triage_review_20260925/review_on.yaml` |
| parquet shards | `/orcd/scratch/bcs/002/satra/recording_vectors_r4` |
| page extract + html | `/orcd/scratch/bcs/002/satra/free_speech_page_20260925` |
| **on the laptop** | `~/Downloads/recording_vectors_20260925/`, `~/Downloads/free_speech_review_20260925/` |

The r4 tree is a **mirror**: `store.jsonl`, `streams/`, a `derivatives` symlink, and **no
`run.json`**. That is why the manifest carries `source` — `source_of()` has nothing to read there.

## The three remaining steps, in order

Each needs the r4 tree to be settled, so do them after the review array leaves the queue. A census
taken while the array writes is a snapshot mid-write; that contaminated the r4 report once already.

**1. Final re-fold**, to make the whole corpus one fold. See the open question below for the config.

```bash
ssh orcd 'cd /orcd/scratch/bcs/002/satra/triage_refold_20260925 && sbatch refold.sbatch'
```

**2. Parquet at schema 6**, then merge and retrieve mode 600.

```bash
ssh orcd 'cd /orcd/scratch/bcs/002/satra/recording_vectors_r4 && sbatch rvec_r4.sbatch'
# then
ssh orcd '... python scripts/triage_recording_vectors.py --merge <dir> --out <dir>'
```

**3. The two evaluations** the owner asked for, with `~/evaluate_flags.py` (staged on the cluster):
flags per family against the **pre-review baseline** (`~/flags_r4_prereview.json`, 8,915 of 61,113
flagged — taken before any reviewer data landed, so it separates the reviewer's effect from the
r3→r4 code), and the free-response population on its own.

Then rebuild the page from the settled tree.

## Decided, and measured

- **story-recall**: 93.7% → 5.0% flagged, `coverage_min` gone from the failing-gate table, 45 of 47
  other families within 1pp. The gate change did one thing.
- **release split**: 26,757 / 12,016 / 4,677 / 19,098 landing exactly on its predicted grounds.
- **non-lexical cleared** (owner, 2026-09-25): `not_assessed` 19,098 → **1**. Totals then reproduce
  r3's permissive classification but with a named ground per recording.
- **brackets dropped for the reviewer**: 29.9% of recordings render empty and cost no GPU. Not a
  false-positive fix — a probe over 20 marker-only recordings read all 20 clean.
- **loop convergence**: 99.2% of multi-round recordings gained nothing after round one, because
  masking an empty proposal is a no-op and the next round re-read an identical string. 39% of GPU
  time. Fixed.

## Open for the owner

**`verdict.llm_redaction_withholds`.** Marked `UNFITTED` in the packaged config. On, essentially
every flagged reading becomes a withholding: the sample put withheld at 22.6% against 7.5% before,
so roughly 14,000 rather than 4,647. The review array runs with it **on** (`review_on.yaml`); the
last re-fold ran with the packaged config, which is **off**, and silently reverted it.

This is cheap either way and that is the point: the **reading** is the expensive, permanent half,
and the weighting is pure fold. Whichever way it goes, one re-fold settles the corpus.

## Traps this campaign has actually hit

- A census over a tree a job is writing is a snapshot mid-write.
- `MaxSubmitPU` counts **expanded array elements** (380 once), not job ids.
- Entitlement is not availability: 36 cards allowed, 8–13 obtainable.
- The L40S OOMs this checkpoint at 38.3 GiB; it needs an 80 GB card.
- `--gres=gpu:1` untyped reaches an L40S first.
- Print a record's keys before filtering on them. `tokens_n`, not `n_words`.
- A substring assertion cannot see a JavaScript syntax error. 112 of them missed a broken page.
- Derive a commit SHA from `git rev-parse`, never from recall.
