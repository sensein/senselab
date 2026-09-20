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
