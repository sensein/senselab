# The 2,291-recording pilot, 2026-09-19

The first run of the whole graph at scale on the fixed code: commit `f2729d7c`, CPU, 128 slices,
a family-stratified sample covering all 796 task tokens. 2,269 of 2,291 rows read here; the
remaining 22 were still running.

## The pipeline itself is sound

| | |
|---|---|
| decisions read | **2,269**, none missing |
| rows that failed | **0** |
| node errors | **0** |
| config paths nobody measured | **0** |
| gates no branch could read | **0** |

SPEECH errored on 1 of 20 recordings in the smoke that preceded the clamp fix. It errors on none of
2,269 now. The `unmeasured` and `critical_absences` tables being empty is the readable-rulesets work
holding at corpus scale.

TAXONOMY flags **38 of 2,269 (1.7%)**. In the 2026-09-08 corpus it flagged 62,521 of 62,521, because
four thresholds were null. Three now carry values and the fourth no longer exists.

## The flag rate is VOICE

**929 of 2,269 recordings flag (40.9%).** Which node contributed a flag verdict:

| node | flag records | share of all recordings |
|---|---:|---:|
| **VOICE** | **917** | **40.4%** |
| AIRWAY | 426 | 18.8% |
| SPEECH | 274 | 12.1% |
| TAXONOMY | 38 | 1.7% |
| routing | 11 | 0.5% |

VOICE alone accounts for essentially the whole flagged population. Its own numbers say why: of 756
recordings it ran on, its finding is **absent on 575 (76%)**, and where it reported a conformance it
is **False on 187 against True on 71**. That is the defect measured over 300 recordings in
`specs/20260817-triage-workflow-dag/voice-flag-grounds.md` — a gate reading the maximum over ~2,500
windows against a threshold derived as a per-window bound — reproducing at eight times the scale.

**AIRWAY is second and unexamined.** 426 flag records, findings `uncertain` on 1,467 and `absent` on
267, conformance `UNDETERMINED` on 394 against `True` on 305. Whether that is the same shape as
VOICE's is the next thing to measure, not to assume.

SPEECH, by contrast, reads `True` on 1,403 against `False` on 182 and `UNDETERMINED` on 81. The two
blockers fixed today are the difference.

## Short recordings are caught, and they are all flagged

| duration | recordings | flagged |
|---|---:|---:|
| 0–1 s | 80 (3.5%) | **80 (100%)** |
| 1–3 s | 113 (5.0%) | 22 (19.5%) |
| 3–10 s | 1,259 (55.5%) | 386 (30.7%) |
| 10–30 s | 527 (23.2%) | 272 (51.6%) |
| 30–60 s | 208 (9.2%) | 124 (59.6%) |
| ≥ 60 s | 82 (3.6%) | 45 (54.9%) |

Every recording under one second flags. That answers the question of whether task verification also
catches unusually short recordings: it does, and the duration cross-tab is what separates them from
a task that was attempted and failed.

## The LLM re-read did not run

`redaction.llm_check.enabled` is `false` — "off; a run with a GPU turns it on". Across the 746
recordings that reached REDACT, `status` is `disabled` on 588 and `not_run` on 158, `model_id` is
empty on all of them.

This is not a defect, but it is a gap against what was asked for. The sequencing that makes it
affordable: run the corpus on CPU without it, then run the re-read as a GPU second pass over only
the recordings carrying a finding. The PII scan gate makes that set much smaller than it was, since
a recording that produced only the words its task asked for is no longer scanned at all.

## Deviations

SPEECH: `stimulus_mismatch` 289, `omission` 126, `off_task_extent` 103, `filler` 57, `truncation` 26,
`repeat_reading` 13, `repeated_item` 5. AIRWAY: `off_task_extent` 121, `truncation` 15. VOICE:
`omission` 10, `repeat_attempt` 5, `lexical_content` 3, `truncation` 3, `sweep_direction_mismatch` 2.

VOICE's deviation counts being two orders of magnitude below its flag count is itself the finding:
it is not reporting what a participant did differently, it is failing to find the carrier at all.

## The largest flag ground was routing's leniency, and it is no longer a ground

The grounds table, read over the pilot, separates what the flag count could not:

| ground | records | of all recordings |
|---|---:|---:|
| **`mismatch: routing routed VOICE, it found no subject`** | **569** | **25.1%** |
| **`mismatch: routing routed AIRWAY, it found no subject`** | **261** | **11.5%** |
| `VOICE reported that what the instruction asked for did not happen` | 187 | 8.2% |
| `SPEECH reported that what the instruction asked for did not happen` | 182 | 8.0% |
| `hint mismatch: VOICE was declared and did not find it` | 157 | 6.9% |
| `REDACT verification found pii` | 158 | 7.0% |
| `AIRWAY reported that what the instruction asked for did not happen` | 102 | 4.5% |
| `hint mismatch: SPEECH was declared and did not find it` | 68 | 3.0% |
| `hint mismatch: AIRWAY was declared and did not find it` | 63 | 2.8% |
| `TAXONOMY: no per-span classifier produced scores` | 38 | 1.7% |
| `mismatch: routing declined SPEECH, it found it` | 24 | 1.1% |
| `routing: no branch routed and the recording was not measurably empty` | 11 | 0.5% |

The top two are one ground reaching 830 records, and it is not a finding about the recording.
Routing is lenient by design — the owner's rule is that screening is lenient and the branches
discard — so sending VOICE to a read passage and AIRWAY to a DDK train is routing working as
intended, and the branch finding nothing of its kind is the branch being right. The owner confirmed
it directly: *it's possible for voice to find nothing.*

**382 of 2,269 recordings (16.8%) flagged on that ground and nothing else** — led by
`harvard-sentences-list` (104), `productive-vocabulary` (30), `free-speech-v2` (21), `free-speech`
(21), `picture-description` (17), `rainbow-passage` (14) and the DDK families. Every one of them
performed its declared task.

So that direction of the mismatch is no longer a flag ground. The other direction — declined and
found it anyway — is the ruleset being wrong about the recording, and still flags. Both remain in
the `agreement` table, which is where a reader checks the ruleset against the detectors, and
`findings` and `routes` already record the observation on every recording, so nothing auditable is
lost.

The informative absence is untouched: `hint mismatch: X was declared and did not find it` stays a
ground on all three branches, which is the case where the recording said it held a kind and does not.

**Corpus flag rate: 41.0% → 24.2%.**
