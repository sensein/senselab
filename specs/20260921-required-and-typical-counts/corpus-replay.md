# The corpus replay: nothing moved, on conformance or on any reading

**Zero recordings out of 62,299 change their task conformance, and zero change any measurement
VERDICT could read.** Every one of the 48 declared families answers exactly what it answered
before, to the recording.

## How it was measured

[`corpus-replay.py`](corpus-replay.py), run over the finished 2026-09-19 corpus run at
`/orcd/scratch/bcs/002/satra/triage_design_20260919/run/` — read-only, nothing under that tree was
written. Each pass loads a recording's `store.jsonl` and its PREPROCESS derivatives, re-runs the
**one branch that owns the declared family** in its in-family mode, and records both what that
task's conformance is and every `measure` finding it wrote, as a name-to-value mapping. No model is
loaded and no audio is decoded.

It is [`specs/20260921-gates-in-verdict/corpus-replay.py`](../20260921-gates-in-verdict/corpus-replay.py)
with one change — the readings are recorded as a mapping rather than a list of names — because
**conformance alone would have been too weak a check here**. No gate reads a count, so the
conformance could not move whatever this change did to the counts; what could move is a reading the
branch writes beside them, and the mapping is how that is seen. The report script is that
directory's with one added column.

| side | code | commit |
| --- | --- | --- |
| `old` | `design/triage-workflow-dag` | `608b4bd6` |
| `new` | this change | `be61ddb0` |

Both sides are post-gates, so both reach the conformance the same way — VERDICT's gates over the
readings the branch reported — and one script produces both halves, so a difference is attributable
to the change and not to a difference between two measurement scripts.

Two array jobs of 64 tasks each on `mit_preemptable`; all 128 exited 0.

62,299 recordings carry a declared family with an expectation row and completed on both sides. 0
errored on either side.

## The result

```
| family | n | before True/False/UND | after True/False/UND | moved | readings moved |
| --- | --- | --- | --- | --- | --- |
| `animal-fluency` | 195 | 194/1/0 | 194/1/0 | 0 | 0 |
| `breath-sounds` | 326 | 310/16/0 | 310/16/0 | 0 | 0 |
| `cape-v-sentences` | 2362 | 2269/93/0 | 2269/93/0 | 0 | 0 |
| `cape-v-sentences-v2` | 1217 | 1195/22/0 | 1195/22/0 | 0 | 0 |
| `caterpillar-passage` | 597 | 306/290/1 | 306/290/1 | 0 | 0 |
| `cinderella-story` | 255 | 235/20/0 | 235/20/0 | 0 | 0 |
| `diadochokinesis-buttercup` | 893 | 889/4/0 | 889/4/0 | 0 | 0 |
| `diadochokinesis-ka` | 892 | 884/6/2 | 884/6/2 | 0 | 0 |
| `diadochokinesis-pa` | 895 | 892/3/0 | 892/3/0 | 0 | 0 |
| `diadochokinesis-pataka` | 893 | 889/3/1 | 889/3/1 | 0 | 0 |
| `diadochokinesis-ta` | 894 | 889/3/2 | 889/3/2 | 0 | 0 |
| `diadochokinesis-v2-buttercup` | 699 | 687/12/0 | 687/12/0 | 0 | 0 |
| `diadochokinesis-v2-kuh` | 696 | 687/9/0 | 687/9/0 | 0 | 0 |
| `diadochokinesis-v2-puh` | 697 | 685/12/0 | 685/12/0 | 0 | 0 |
| `diadochokinesis-v2-puhtuhkuh` | 697 | 688/9/0 | 688/9/0 | 0 | 0 |
| `diadochokinesis-v2-tuh` | 702 | 694/8/0 | 694/8/0 | 0 | 0 |
| `free-speech` | 3063 | 2978/84/1 | 2978/84/1 | 0 | 0 |
| `free-speech-v2` | 2111 | 2064/46/1 | 2064/46/1 | 0 | 0 |
| `glides-high-to-low` | 1541 | 684/0/857 | 684/0/857 | 0 | 0 |
| `glides-low-to-high` | 1587 | 706/0/881 | 706/0/881 | 0 | 0 |
| `harvard-sentences-list` | 13660 | 12050/1605/5 | 12050/1605/5 | 0 | 0 |
| `high-to-low` | 43 | 18/0/25 | 18/0/25 | 0 | 0 |
| `loudness` | 892 | 863/28/1 | 863/28/1 | 0 | 0 |
| `loudness-v2` | 702 | 627/75/0 | 627/75/0 | 0 | 0 |
| `maximum-phonation-time` | 2686 | 1849/0/837 | 1849/0/837 | 0 | 0 |
| `maximum-phonation-time-v2` | 807 | 553/0/254 | 553/0/254 | 0 | 0 |
| `open-response-questions` | 198 | 193/5/0 | 193/5/0 | 0 | 0 |
| `picture-description` | 882 | 878/4/0 | 878/4/0 | 0 | 0 |
| `picture-description-option1` | 370 | 365/5/0 | 365/5/0 | 0 | 0 |
| `picture-description-option2` | 327 | 323/4/0 | 323/4/0 | 0 | 0 |
| `productive-vocabulary` | 2899 | 2827/70/2 | 2827/70/2 | 0 | 0 |
| `prolonged-vowel` | 1602 | 785/0/817 | 785/0/817 | 0 | 0 |
| `rainbow-passage` | 896 | 684/211/1 | 684/211/1 | 0 | 0 |
| `random-item-generation` | 265 | 0/0/265 | 0/0/265 | 0 | 0 |
| `random-item-generation-v2` | 206 | 0/0/206 | 0/0/206 | 0 | 0 |
| `respiration-and-cough-breath` | 1781 | 0/0/1781 | 0/0/1781 | 0 | 0 |
| `respiration-and-cough-cough` | 1778 | 1558/194/26 | 1558/194/26 | 0 | 0 |
| `respiration-and-cough-fivebreaths` | 3564 | 3075/418/71 | 3075/418/71 | 0 | 0 |
| `respiration-and-cough-threequickbreaths` | 1709 | 1539/121/49 | 1539/121/49 | 0 | 0 |
| `respiration-and-cough-v2-breath` | 695 | 0/0/695 | 0/0/695 | 0 | 0 |
| `respiration-and-cough-v2-hardcough` | 698 | 645/52/1 | 645/52/1 | 0 | 0 |
| `respiration-and-cough-v2-threebreaths` | 695 | 673/21/1 | 673/21/1 | 0 | 0 |
| `respiration-and-cough-v2-threebreathsmouth` | 694 | 663/30/1 | 663/30/1 | 0 | 0 |
| `respiration-and-cough-v2-threebreathsnose` | 694 | 638/56/0 | 638/56/0 | 0 | 0 |
| `story-recall` | 889 | 56/832/1 | 56/832/1 | 0 | 0 |
| `story-recall-v2` | 657 | 260/397/0 | 260/397/0 | 0 | 0 |
| `voluntary-cough` | 327 | 318/9/0 | 318/9/0 | 0 | 0 |
| `word-color-stroop` | 471 | 322/149/0 | 322/149/0 | 0 | 0 |
| **all** | **62299** | **50587/4927/6785** | **50587/4927/6785** | **0** | **0** |

```

`readings moved` is the added column: the number of recordings whose `measure` name-to-value
mapping differs between the sides. It is 0 in every family.

## Why zero was the prediction

Nothing a gate reads is a count. The change rewrites `count` findings — a different name, different
evidence keys, and none written where a row declares no count of that kind — plus two `task_extent`
span attributes on AIRWAY and one covariate on `ddk_repetition_count_from_ppg_decode`. The
covariate sits beside the value rather than in it: `PPG_REPETITIONS` still reports `decode.count`.
So the reading mapping is untouched by construction, and the table says the construction is right.

The one thing the replay would have caught and did not have to: `decode_evidence` lost the
`declared_event_count // vowel_positions` division. Had that division been load-bearing for the
repetition count rather than only for a covariate, `ddk_repetition_count_from_ppg_decode` would
have moved on all five counted syllable families. It reads the same on all 4,467 of them.

## What the replay also shows, and does not attribute to this change

The `before` side differs from what the corpus run itself recorded on **2,795 recordings**, all
AIRWAY. This is the same population `specs/20260921-gates-in-verdict/corpus-replay.md` reports at
2,787 — the `SOUND_COVERAGE` families now answering `UNDETERMINED` always, and the YAMNet
breath-spelling fix — plus the handful the slightly larger denominator adds. Both predate this
change and both are present identically on both sides, which is why the A/B moves nothing.

## Reproducing

```bash
# both sides, 64 tasks each, read-only over the run tree
cd /orcd/scratch/bcs/002/satra/counts-replay
REPLAY_SIDE=old sbatch --array=0-63 replay.sbatch
REPLAY_SIDE=new sbatch --array=0-63 replay.sbatch
REPLAY_OUT=out python corpus-replay-report.py
```

`old/` holds `git archive 608b4bd6 src/senselab`; `new/` holds `git archive be61ddb0 src/senselab`.
Both run under the corpus checkout's own interpreter with `PYTHONPATH` pointed at the side under
test, so the two sides share every dependency and differ only in the senselab tree.
[`replay.sbatch`](replay.sbatch) is in this directory.
