# The corpus replay: conformance per family, before and after

**Zero recordings out of 62,139 change their task conformance.** Every one of the 48 declared
families answers exactly what it answered before, to the recording.

## How it was measured

`corpus-replay.py`, run over the finished 2026-09-19 corpus run at
`/orcd/scratch/bcs/002/satra/triage_design_20260919/run/` — read-only, nothing under that tree was
written. Each pass loads a recording's `store.jsonl` and its PREPROCESS derivatives, re-runs the
**one branch that owns the declared family** in its in-family mode, and records what that task's
conformance is. No model is loaded and no audio is decoded, so a pass is minutes rather than the
run's fourteen hours.

The two passes differ only in which checkout is on `PYTHONPATH`:

| side | code | how the conformance is reached |
| --- | --- | --- |
| `old` | `design/triage-workflow-dag` at `1af94c2a` | the branch's own `Result.done` |
| `new` | this change | VERDICT's gates over the readings the branch reported |

One script produces both halves — it reads which side it is on off `Result._fields` — so a
difference is attributable to the change and not to a difference between two measurement scripts.

The `new` side was replayed twice over: once against the single-layer gate table the design first
called for, and again against the three-layer one it was corrected to. Both carry the same bounds —
`by_family` and `default` ship empty, so every bound resolves from `by_group` — and both moved
nothing. The table below is the second pass.

Two array jobs of 64 tasks each on `mit_preemptable`; `corpus-replay-report.py` joins them by stem.
Only the in-family branch is replayed, because it is the only conformance this change can move: an
out-of-family report answered `UNDETERMINED` before and answers `UNDETERMINED` after — the branch
writes it and no gate is applied to it — and QUALITY's conformance is about the store's own
assertions, which no gate reads.

62,139 recordings carry a declared family with an expectation row and completed. 0 errored on
either side.

## The result

```
| family | n | before True/False/UND | after True/False/UND | moved |
| --- | --- | --- | --- | --- |
| `animal-fluency` | 195 | 194/1/0 | 194/1/0 | 0 |
| `breath-sounds` | 324 | 308/16/0 | 308/16/0 | 0 |
| `cape-v-sentences` | 2359 | 2266/93/0 | 2266/93/0 | 0 |
| `cape-v-sentences-v2` | 1214 | 1192/22/0 | 1192/22/0 | 0 |
| `caterpillar-passage` | 594 | 305/288/1 | 305/288/1 | 0 |
| `cinderella-story` | 256 | 236/20/0 | 236/20/0 | 0 |
| `diadochokinesis-buttercup` | 887 | 883/4/0 | 883/4/0 | 0 |
| `diadochokinesis-ka` | 891 | 883/6/2 | 883/6/2 | 0 |
| `diadochokinesis-pa` | 893 | 890/3/0 | 890/3/0 | 0 |
| `diadochokinesis-pataka` | 893 | 889/3/1 | 889/3/1 | 0 |
| `diadochokinesis-ta` | 895 | 890/3/2 | 890/3/2 | 0 |
| `diadochokinesis-v2-buttercup` | 698 | 686/12/0 | 686/12/0 | 0 |
| `diadochokinesis-v2-kuh` | 694 | 685/9/0 | 685/9/0 | 0 |
| `diadochokinesis-v2-puh` | 694 | 682/12/0 | 682/12/0 | 0 |
| `diadochokinesis-v2-puhtuhkuh` | 698 | 689/9/0 | 689/9/0 | 0 |
| `diadochokinesis-v2-tuh` | 701 | 693/8/0 | 693/8/0 | 0 |
| `free-speech` | 3059 | 2974/84/1 | 2974/84/1 | 0 |
| `free-speech-v2` | 2096 | 2049/46/1 | 2049/46/1 | 0 |
| `glides-high-to-low` | 1538 | 683/0/855 | 683/0/855 | 0 |
| `glides-low-to-high` | 1583 | 705/0/878 | 705/0/878 | 0 |
| `harvard-sentences-list` | 13618 | 12011/1602/5 | 12011/1602/5 | 0 |
| `high-to-low` | 43 | 18/0/25 | 18/0/25 | 0 |
| `loudness` | 892 | 863/28/1 | 863/28/1 | 0 |
| `loudness-v2` | 698 | 623/75/0 | 623/75/0 | 0 |
| `maximum-phonation-time` | 2679 | 1844/0/835 | 1844/0/835 | 0 |
| `maximum-phonation-time-v2` | 801 | 549/0/252 | 549/0/252 | 0 |
| `open-response-questions` | 199 | 194/5/0 | 194/5/0 | 0 |
| `picture-description` | 880 | 876/4/0 | 876/4/0 | 0 |
| `picture-description-option1` | 369 | 364/5/0 | 364/5/0 | 0 |
| `picture-description-option2` | 325 | 321/4/0 | 321/4/0 | 0 |
| `productive-vocabulary` | 2887 | 2815/70/2 | 2815/70/2 | 0 |
| `prolonged-vowel` | 1600 | 785/0/815 | 785/0/815 | 0 |
| `rainbow-passage` | 892 | 680/211/1 | 680/211/1 | 0 |
| `random-item-generation` | 265 | 0/0/265 | 0/0/265 | 0 |
| `random-item-generation-v2` | 206 | 0/0/206 | 0/0/206 | 0 |
| `respiration-and-cough-breath` | 1778 | 0/0/1778 | 0/0/1778 | 0 |
| `respiration-and-cough-cough` | 1774 | 1555/193/26 | 1555/193/26 | 0 |
| `respiration-and-cough-fivebreaths` | 3557 | 3068/418/71 | 3068/418/71 | 0 |
| `respiration-and-cough-threequickbreaths` | 1705 | 1535/121/49 | 1535/121/49 | 0 |
| `respiration-and-cough-v2-breath` | 692 | 0/0/692 | 0/0/692 | 0 |
| `respiration-and-cough-v2-hardcough` | 696 | 643/52/1 | 643/52/1 | 0 |
| `respiration-and-cough-v2-threebreaths` | 695 | 673/21/1 | 673/21/1 | 0 |
| `respiration-and-cough-v2-threebreathsmouth` | 692 | 661/30/1 | 661/30/1 | 0 |
| `respiration-and-cough-v2-threebreathsnose` | 695 | 639/56/0 | 639/56/0 | 0 |
| `story-recall` | 887 | 56/830/1 | 56/830/1 | 0 |
| `story-recall-v2` | 657 | 261/396/0 | 261/396/0 | 0 |
| `voluntary-cough` | 326 | 317/9/0 | 317/9/0 | 0 |
| `word-color-stroop` | 469 | 322/147/0 | 322/147/0 | 0 |
| **all** | **62139** | **50455/4916/6768** | **50455/4916/6768** | **0** |
```

## The one change predicted in advance, and why it never fires

[`implementation.md`](implementation.md) § *The one behaviour change this cannot avoid* predicted
that a `SUSTAINED` recording whose selected carrier has a **non-finite F0 spread** would move from
`True` to `UNDETERMINED`, because the old qualifier left an unmeasurable reading's gate unapplied
and rule 3 does not. `prolonged-vowel`, `maximum-phonation-time` and `maximum-phonation-time-v2`
move 0 of 5,080 recordings between them, so the case does not occur on this corpus: a carrier that
clears `production_min_s` at 0.5 s and `voiced_strength_min` at 0.45 always has enough voiced
frames for `typical_windowed_spread` to resolve one at a 0.5 s window. The prediction stands as a
statement about the code; its rate here is zero.

## What the replay also shows, and does not attribute to this change

The replay's `before` side is the base branch running now. On **2,787 recordings it differs from
what the corpus run itself recorded** — every one of them AIRWAY, and every one of them explained by
commits landed between the run and this branch's base, not by this change:

| move | n |
| --- | --- |
| `respiration-and-cough-breath`: True/False → UNDETERMINED | 1,639 |
| `respiration-and-cough-v2-breath`: True/False → UNDETERMINED | 685 |
| the five event-series breath families and `breath-sounds`: False → True | 320 |
| the same, plus `-cough` and `-hardcough`: False → UNDETERMINED | 143 |

The first two are `SOUND_COVERAGE`, which now answers `UNDETERMINED` always — it has no conformance
term, and never had one that was reasoned. The `False → True` group is the YAMNet breath-spelling
fix `airway-flag-grounds.md` measured. Both predate this change and both are present identically on
both replay sides, which is why the A/B moves nothing.

## Reproducing

```bash
# both sides, 64 tasks each, read-only over the run tree
cd /orcd/scratch/bcs/002/satra/gates-replay
REPLAY_SIDE=new sbatch --array=0-63 replay.sbatch
REPLAY_SIDE=old sbatch --array=0-63 replay.sbatch
REPLAY_OUT=out python corpus-replay-report.py
```

`old/` holds `git archive design/triage-workflow-dag src/senselab`; `new/` holds this branch's.
Both run under the corpus checkout's own interpreter with `PYTHONPATH` pointed at the side under
test, so the two sides share every dependency and differ only in the senselab tree.
