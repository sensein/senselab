# What a task extent is, and how much of the corpus has one

Measured on the finished design corpus,
`/orcd/scratch/bcs/002/satra/triage_design_20260919/run/out`, 62,548 recordings over 1,527
subjects, at commit `f3da4081`. Slurm job `23470827` (32 shards, `pi_satra`); the per-recording
census rows are at `/orcd/scratch/bcs/002/satra/speaker_vectors_20260922/census_design/`.

## The thing itself

Task extent is **not** an entity type and **not** a measurement. It is a `span` entity whose
`attributes["role"]` is the literal `"task_extent"`:

```json
{"record": "entity", "prov_type": "span", "id": "span-<sha>",
 "extent": [1.0916432707667731, 5.448201277955271],
 "attributes": {"role": "task_extent", "family": "speech", ...}}
```

- `extent` is `[start, end]`, **floats in seconds on the recording's own time axis**, not samples.
  Every PREPROCESS stream (`plain`, `preemphasised`, `normalized`, `enhanced`, `residual`,
  `redacted`) shares that axis and that duration, so one pair of seconds indexes all of them.
- **Exactly one interval, never a list.** `hull()` collapses candidates to a single span and
  `proposer()` refuses a non-positive one, so "found nothing" is unrepresentable as a degenerate
  span. Measured: of 54,923 recordings carrying any, **every one carries exactly one**.
- `family` is one of `airway`, `speech`, `voice` — the branch that minted it. DDK mints under
  `family: "speech"`.
- Minted in **align mode only**. A detect-mode run mints `phonation` / `lexical_run_<i>` /
  `<label>_event` spans carrying `evaluates_no_task: true` and no task extent.
- **There is no `extent_found` flag and no fallback to the whole recording.** Absence of the span
  is the branch's record that it found no task. Anything that substituted the whole file would be
  inventing a task boundary the branch declined to place.

Minters, one per branch, by declared family:

| branch | matcher | task families |
| --- | --- | --- |
| VOICE | `_voice_sustained`, `_voice_glide` | `prolonged-vowel`, `maximum-phonation-time{,-v2}`, `glides-{low-to-high,high-to-low}`, `high-to-low` |
| SPEECH | `_speech_ordered`, `_speech_free_response`, `_speech_item_list`, DDK's `task_extent_span` | 31 families: harvard/cape-v/rainbow/caterpillar/stroop/loudness, the free-response set, `animal-fluency`, `random-item-generation{,-v2}`, the ten `diadochokinesis-*` |
| AIRWAY | `_airway_event_series`, `_airway_alternation`, `_airway_coverage` | the `respiration-and-cough-*` set, `breath-sounds`, `voluntary-cough` |

## Coverage

| | recordings | share |
| --- | ---: | ---: |
| total | 62,548 | |
| carry ≥1 task extent | **54,923** | **87.8%** |
| carry exactly one | 54,923 | 100% of those |
| carry none | 7,625 | 12.2% |

Stores read: 62,548. Unreadable: 0. Missing: 0.

### By branch family

| family | extents | min | p5 | p25 | median | p75 | p95 | max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| speech | 39,235 | 0.04 | 1.91 | 2.72 | **3.72** | 15.57 | 59.47 | 330.74 |
| airway | 11,066 | 0.05 | 0.50 | 4.23 | **8.20** | 14.45 | 28.02 | 101.63 |
| voice | 4,622 | 0.29 | 1.38 | 3.20 | **6.79** | 11.41 | 20.23 | 50.07 |

All seconds. Pooled median 4.32 s, mean 12.37 s.

### Duration distribution, all 54,923

| bucket (s) | n | share |
| --- | ---: | ---: |
| [0, 0.25) | 329 | 0.6% |
| [0.25, 0.5) | 364 | 0.7% |
| [0.5, 1.0) | 530 | 1.0% |
| [1.0, 1.5) | 822 | 1.5% |
| [1.5, 2.0) | 2,001 | 3.6% |
| [2.0, 3.0) | 12,248 | 22.3% |
| [3.0, 5.0) | 13,566 | 24.7% |
| [5.0, 10.0) | 8,064 | 14.7% |
| [10.0, 30.0) | 12,321 | 22.4% |
| ≥ 30 | 4,678 | 8.5% |

**7.4% of extents are under 2.0 s** — under the profile-enrollment window this module uses. That
is the population the refusal floor in `design.md` is about.

### Per subject

| | value |
| --- | ---: |
| subjects with ≥1 usable extent | 1,523 / 1,527 |
| extents per subject: min / median / mean / max | 1 / 35 / 36.1 / 140 |
| extent seconds per subject: min / median / mean / max | 0.6 / 410.7 / 446.3 / 3,408.1 |
| subjects with ≥2 extents | 1,518 |
| subjects with ≥5 | 1,514 |
| subjects with ≥10 | 1,508 |
| subjects with ≥20 | 1,485 |

The supply is not the constraint: a median subject brings 35 extents and nearly seven minutes.
Four subjects bring nothing, and a tail of ~40 brings fewer than ten.

### The 12.2% that mints nothing: structural or per-recording?

Almost entirely per-recording, but at a rate the task family sets.

Of the 796 distinct task labels: **2 never mint an extent** (472 recordings), 184 always do
(3,596 recordings), and 610 sometimes do (58,480 recordings). So of the 7,625 recordings with no
extent:

| | recordings | share of missing |
| --- | ---: | ---: |
| in a task label that mints one on **no** recording | 472 | **6.2%** |
| in a task label that mints one on **some** recordings | 7,153 | **93.8%** |

The 472 are both `random-item-generation` variants, and they are structural: the branch never
places an extent on that task, on any recording, so no amount of better audio would change it.

The other 93.8% is a per-recording outcome — the branch looked at that recording and did not find
its task — but **the probability is set by the family**. The five VOICE families
(`glides-high-to-low`, `glides-low-to-high`, `prolonged-vowel`, `maximum-phonation-time`,
`maximum-phonation-time-v2`) contribute **3,660 of the 7,625 missing, 48.0%, from 8,264
recordings — 13.2% of the corpus**. Read the other way: a diadochokinesis or cape-v recording
mints an extent ~99% of the time and a glide recording 44.5% of the time. So the honest summary
is *a per-recording failure whose rate is a property of the task*, not a property of either alone.

Collapsed family coverage, worst first (n ≥ 200):

| task family | with extent / total | | missing |
| --- | ---: | ---: | ---: |
| `random-item-generation` | 0 / 265 | 0.0% | 265 |
| `random-item-generation-v2` | 0 / 207 | 0.0% | 207 |
| `glides-high-to-low` | 692 / 1,554 | 44.5% | 862 |
| `glides-low-to-high` | 711 / 1,596 | 44.5% | 885 |
| `prolonged-vowel` | 786 / 1,604 | 49.0% | 818 |
| `maximum-phonation-time-v2` | 557 / 813 | 68.5% | 256 |
| `maximum-phonation-time` | 1,858 / 2,697 | 68.9% | 839 |
| `respiration-and-cough-breath` | 1,332 / 1,788 | 74.5% | 456 |
| `respiration-and-cough-fivebreaths` | 2,921 / 3,576 | 81.7% | 655 |
| `harvard-sentences-list` | 12,684 / 13,705 | 92.6% | 1,021 |
| `free-speech` | 2,996 / 3,074 | 97.5% | 78 |
| `diadochokinesis-*` (10 families) | ~99% each | | ~100 total |
| `cape-v-sentences-v2` | 1,218 / 1,224 | 99.5% | 6 |
| `picture-description` | 886 / 889 | 99.7% | 3 |

`harvard-sentences-list` is the largest single contributor in absolute terms (1,021 missing)
purely because it is the largest family; its *rate* is unremarkable at 92.6%.

### Where the 12.2% goes, by task label

Coverage is not uniform across tasks. Lowest, among tasks with ≥100 recordings:

| task | with extent / total | |
| --- | ---: | ---: |
| `random-item-generation` | 0 / 265 | **0.0%** |
| `random-item-generation-v2` | 0 / 207 | **0.0%** |
| `glides-high-to-low` | 692 / 1,554 | 44.5% |
| `glides-low-to-high` | 711 / 1,596 | 44.5% |
| `prolonged-vowel` | 786 / 1,604 | 49.0% |
| `maximum-phonation-time-v2-2` | 66 / 113 | 58.4% |
| `maximum-phonation-time-3` | 607 / 899 | 67.5% |
| `maximum-phonation-time-1` | 620 / 899 | 69.0% |
| `respiration-and-cough-breath-2` | 622 / 894 | 69.6% |

Highest: the cape-v, picture-description, animal-fluency and diadochokinesis families all sit at
99–100%.

Two things worth naming:

- **`random-item-generation` mints an extent on none of its 472 recordings**, although
  `ITEM_LIST` is a declared minter and `animal-fluency` — the same pattern — sits at 99.5%. Not
  investigated here; it is a coverage fact this module inherits, not one it can fix.
- **VOICE is the weak family**, 44–70% across sustained and glide, which is the 38–76% the held
  note in `specs/20260922-compact-recording-vectors/design.md` predicted. A speaker whose session
  is voice-heavy contributes proportionally less.

## Subject as the grouping key

Every run directory is `sub-<label>/ses-<label>/<stem>_<stamp>/`, and the store holds no subject
entity — subject identity lives only in the stem and the path. The BIDS `sub-<id>` is therefore
the only speaker key available, and this module uses it.

**What that assumes, stated plainly:** that one `sub-` id is one human, and that the participant
is the only voice inside a task extent. Neither is verified here. The corpus is a protocol of
prompted single-speaker tasks, so the second is the design intent rather than a measurement — and
`specs/20260915-preprocess-diarization/design.md` already documents one recording in which a
second speaker's labels appeared inside the participant's own narration. A task extent is the
portion a branch judged to serve the declared task, which biases towards the participant but does
not exclude an interviewer. The per-extent agreement columns on each row
(`cos_extent_pairwise_q50`, `leave_one_extent_out_cos_min`, `auc_same_extent_vs_diff_extent`) are
what a reader uses to notice a speaker whose extents do not agree with each other; nothing in this
module reaches a verdict about it.
