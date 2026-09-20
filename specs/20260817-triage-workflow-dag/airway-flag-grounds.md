# What AIRWAY had in hand when it flagged

A measurement of the three AIRWAY flag families over the in-flight full-graph corpus run:

- `AIRWAY reported that what the instruction asked for did not happen on <family>` — AIRWAY's own
  `conformance` reaching `False`;
- `hint mismatch: AIRWAY was declared and did not find it` — VERDICT's hint table, folding what
  AIRWAY *found* (the spans it proposed) against the declaration;
- `mismatch: routing declined AIRWAY, it found it`.

Every `False`, every `claimed_not_found` and every `unavailable` below was traced back into the
store: which spans AIRWAY was handed, which classifier windows covered them, which gate decided,
and on what value against what bound.

This document reports and proposes. **Nothing in `src/` is changed by it.**

## What was measured

| | |
| --- | --- |
| recordings | **29,484** completed rows, read 2026-09-20 while the array was still running |
| rows | `/orcd/scratch/bcs/002/satra/triage_design_20260919/run/rows/**/*.row.json` |
| stores | `.../run/out/**/run/store.jsonl`, opened for every in-family AIRWAY row (**6,080**) |
| sidecars | `.../run/out/**/run/derivatives/hear_scores.json`, opened for those plus **3,183** controls |
| controls | ten families that elicit no breathing task: `maximum-phonation-time`, `harvard-sentences-list`, `prolonged-vowel`, `free-speech`, `glides-low-to-high`, `word-color-stroop`, `animal-fluency`, `diadochokinesis-pataka`, `rainbow-passage`, `cinderella-story` |
| graph | the full triage DAG, one recording per run directory |
| commit | **`b7d882a9aa4492890c9fa15aae918ecaf2743415`** (`senselab-corpus2`), pinned |
| config | `senselab-triage/default`, hash `23c7df85d999d771`, no override |
| where | ORCD, read-only over the existing corpus; **nothing was submitted and nothing was run** |
| probes | `airway-flag-grounds-census.py` and `airway-flag-grounds-tables.py`, beside this file |

`airway.py`, `branches.py`, `vocabulary.py` and every `branch.*` / `airway.*` key of the packaged
config are byte-identical between `b7d882a9` and the branch tip, so the line numbers below hold for
both. The only config change between the two commits is `diarization.streams`, which AIRWAY does
not read.

The brief quotes 29,265 rows; the array added 219 while this was measured. Every rate below is over
the 29,484 actually read, so the absolute counts are slightly above the brief's.

## A correction to the framing

The brief says `respiration-and-cough-*` is one family to look at hardest and that "its variants
(breath, cough, fivebreaths) ask for different things". They do — and the split is not a matter of
degree. The eleven in-family rows use **three** matchers, and the pass rate separates on the
matcher, not on the task:

| reader | families | n | True | False | UND | pass |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `EVENT_SERIES` / `EVENT_ALTERNATION`, cough | `respiration-and-cough-cough`, `-v2-hardcough`, `voluntary-cough` | 1,311 | 1,163 | 148 | 0 | **88.7 %** |
| `EVENT_SERIES`, breath | `-fivebreaths`, `-threequickbreaths`, `-v2-threebreaths`, `-v2-threebreathsmouth`, `-v2-threebreathsnose`, `breath-sounds` | 3,605 | 3,094 | 511 | 0 | **85.8 %** |
| `SOUND_COVERAGE`, breath | `respiration-and-cough-breath`, `-v2-breath` | 1,164 | 389 | 707 | 68 | **33.4 %** |

**Two families carry 51.8 % of every AIRWAY `False` in the corpus, and they are the only two the
`SOUND_COVERAGE` matcher evaluates.** The headline "AIRWAY passes 76 % of its targeted family" is
an average over a bimodal population: 86–89 % under the event matcher, 33 % under the coverage one.

## 1, 2, 3 — the three rates

```
conformance   True 4,646   False 1,366   UNDETERMINED 5,258      (11,270 written)
findings      present 7,827  uncertain 18,198  absent 3,443
routes        routed 11,162  declined 18,286  unavailable 20
hints         claimed_and_found 5,160  no_claim 20,721
              found_unclaimed 2,667     claimed_not_found 920

flag grounds
  1,366  4.6%  AIRWAY reported that what the instruction asked for did not happen
                 (1,347 respiration-and-cough-*, 14 breath-sounds, 5 voluntary-cough)
    920  3.1%  hint mismatch: AIRWAY was declared and did not find it
     37  0.1%  mismatch: routing declined AIRWAY, it found it
```

Per family, over the 6,080 in-family rows:

```
family                                        n    True   False   UND    pass    reader
respiration-and-cough-fivebreaths          1662    1374     288     0   82.7%    event
respiration-and-cough-breath                847     263     517    67   31.1%    coverage
respiration-and-cough-cough                 839     725     114     0   86.4%    event
respiration-and-cough-threequickbreaths     812     718      94     0   88.4%    event
respiration-and-cough-v2-threebreaths       338     315      23     0   93.2%    event
respiration-and-cough-v2-threebreathsmouth  333     291      42     0   87.4%    event
respiration-and-cough-v2-hardcough          332     303      29     0   91.3%    event
respiration-and-cough-v2-threebreathsnose   326     276      50     0   84.7%    event
respiration-and-cough-v2-breath             317     126     190     1   39.7%    coverage
voluntary-cough                             140     135       5     0   96.4%    alternation
breath-sounds                               134     120      14     0   89.6%    event
```

### The 920 `claimed_not_found`

They are not a separate population. `claimed_not_found` is set when AIRWAY proposed no span of its
own kind; `conformance` is `False` when it found no event. The two coincide almost exactly:

```
claimed_and_found  x  True            4,646
claimed_not_found  x  False             852
claimed_and_found  x  False             514
claimed_not_found  x  UNDETERMINED       68
```

**The 514 in row three are the finding.** On those recordings AIRWAY proposed breath spans, reported
`findings: present`, and reported `conformance: False` in the same breath: *it found breath and
said the breathing task did not happen*. All 514 are `SOUND_COVERAGE` rows. Section "Defect B"
is about them.

### The 5,258 `UNDETERMINED`

5,190 of them are `detect_airway` — the out-of-family mode, which returns `UNDETERMINED` by
construction and evaluates no task (`airway.py:922,981`). Their families are `harvard-sentences-list`
(959), `maximum-phonation-time` (566), `productive-vocabulary` (454) and so on; every one is a
family AIRWAY does not own. `verdict.undetermined_flags: false`, so none of them flags. **This arm
is honest and is not a silent absence.**

The remaining **68** are in-family: `instrument_absent("hear_scores")` from `_airway_coverage`
(`airway.py:607,825`). All 68 are recordings under 1 s; HeAR's window is 2 s and PREPROCESS writes
no `hear_scores` sidecar for a file shorter than one window. Their report carries
`notes: ["the hear_scores derivative is absent"]`. **Also honest.**

### The 20 `unavailable` routes

One gate, one cause. Every one of the 20 carries

```
unavailable: {"AIRWAY": {"airway.cough": "span_yamnet: LookupError: spans are absent"}}
```

and every one is 0.12–0.37 s long. PREPROCESS proposed no span at all, so `span_yamnet` wrote no
window and the `airway.cough` gate had nothing to read. Routing reports `unavailable` — a branch
never judged — rather than a negative. **This is the correct behaviour**, and it is the direct
contrast with Defect C below: on the very same evidence state, routing says *unavailable* and
`align_airway` says *False*.

## 4 — the structural trap, pattern 1

**Ruled out for every derivative AIRWAY names, with one exception, which is a defect.**

`airway` declares its reads at `airway.py:1020–1026`: `energy_envelope`, `span_hear`,
`span_yamnet`, `hear_scores`, `silence`, `spectrogram_wideband`, plus `band_profile` via
`content_band_hz`. Over all 6,080 in-family rows:

| derivative | writer | absent |
| --- | --- | ---: |
| `energy_envelope` | PREPROCESS | **0** |
| `spectrogram_wideband` | PREPROCESS | **0** |
| `silence` | PREPROCESS | **0** |
| `band_profile` | PREPROCESS | **0** |
| `span_hear` | PREPROCESS | 65 (the rows with no span at all) |
| `span_yamnet` | PREPROCESS | 65 (the same rows) |
| `hear_scores` | PREPROCESS | 351 (every recording under 2 s) |

`Breathe` and `Cough` are live keys of `span_hear.raw_scores` on **6,015 / 6,080** rows — every row
that has a span. No AIRWAY reader is keyed to a measurement nothing writes.

**The exception.** `sounds_like` (`branches.py:2026`) is handed
`classifier_windows(store)` — `span_hear` **and** `span_yamnet` together (`airway.py:145`) — and
tests them against one label set. The shipped sets are

```yaml
branch.label_sets:            # default.yaml:288-290
  cough: [Cough]
  breath: [Breathe]
```

`Cough` and `Breathe` are HeAR head names. `span_yamnet.raw_scores` is keyed by **AudioSet display
names**. AudioSet has a class called `Cough` — so the cough set is readable by both writers — and it
has **no class called `Breathe`**; its class is `Breathing`, with `Gasp`, `Pant`, `Snoring`, `Snort`
and `Wheeze` beneath it.

So for cough the branch reads two classifiers, and for breath it reads one. Half of the declared
evidence base is silently unreadable, for one of the two label sets only. The repo already ships the
translation it needs and AIRWAY does not call it — `classifier_ontology`'s packaged profile
(`data/classifier_ontology/2026-09-10.json`) maps

```
Cough   -> roots [Cough]      corroborating {Cough, Throat clearing}
Breathe -> roots [Breathing]  corroborating {Breathing, Gasp, Pant, Snoring, Snort, Wheeze}
```

**Verdict: defect.** Measured cost, using the packaged profile's own corroboration set and counting
only spans that are `amplitude` carriers (the ones `_airway_event_series` looks inside):

```
family                                      True n   yamnet breath   False n   yamnet-only
                                                     fires on                  on an amplitude span
breath-sounds                                  120        90.8%           14      6  (42.9%)
respiration-and-cough-fivebreaths             1374        91.3%          288     71  (24.7%)
respiration-and-cough-threequickbreaths        718        94.3%           94     17  (18.1%)
respiration-and-cough-v2-threebreaths          315        92.1%           23     10  (43.5%)
respiration-and-cough-v2-threebreathsmouth     291        94.2%           42     20  (47.6%)
respiration-and-cough-v2-threebreathsnose      276        85.5%           50     18  (36.0%)
                                                           TOTAL          511    142  (27.8%)
```

Labels involved, by rows in which each fires: `Breathing` 209, `Snort` 180, `Snoring` 145, `Gasp`
78, `Wheeze` 42, `Pant` 4. `Snort` and `Snoring` on the nasal-route families is what a nose breath
sounds like to AudioSet.

The YAMNet breath vote is **informative, not noise**: it fires on 85.5–94.3 % of the rows AIRWAY
already calls `True` and on 18–48 % of those it calls `False`. It is **not calibrated** — this
measurement says the vote exists and is discriminating, not that all 142 are breaths. The fix is to
make the label set readable by both writers and re-derive `score_min` against the union, not to
assume the 142 flip.

On the cough side the same test rescues **1** row of 148: `cough: [Cough]` already reads both.

### The `Baby Cough` head, checked and not claimed

HeAR carries eight heads; AIRWAY's cough set names one. Of the 148 cough-family `False` rows,
137 carry a classifier window at all (the other 11 are the no-span stores of Defect C), and
`Baby Cough` clears 0.2 on an amplitude carrier in **96** of those 137 (70.1 %), with `Cough` at
p50 0.007. That looks like a second unread vote — it is not. `Baby Cough` also clears 0.2 on an
amplitude carrier in **81.1 %** of the cough rows AIRWAY calls `True` (max p50 0.43), so at this
cut it is a non-specific head, not a cough detector: 70 % against 81 % is not a discrimination.
Adding it would be an unmeasured decision. **No defect claimed.**

## Defect A — the cough gate is correct and its 148 `False` are real findings

The one place the evidence is unambiguous. `sounds_like` against `cough: [Cough]` at
`branch.score_min = 0.2`:

```
family                                conf     n   hear_max(Cough)          rows with a span >= 0.2
                                                   p10      p50      p90
respiration-and-cough-cough           True   725   0.7714   0.9977   1.0000   706  (97.4%)
respiration-and-cough-cough           False  114   0.0016   0.0043   0.0747     0  ( 0.0%)
respiration-and-cough-v2-hardcough    True   303   0.5716   0.9958   0.9999   293  (96.7%)
respiration-and-cough-v2-hardcough    False   29   0.0029   0.0197   0.1500     0  ( 0.0%)
voluntary-cough                       True   135   0.9244   0.9994   1.0000   134  (99.3%)
voluntary-cough                       False    5   0.0039   0.0329   0.1996     0  ( 0.0%)
```

The distribution is bimodal with the cut in the empty middle: p10 of the positives is 0.57–0.92, p90
of the negatives is 0.07–0.20, and **not one** `False` row carries a span over the cut. A cut
anywhere in [0.2, 0.55] gives the same answer. The statistic the gate reads is the statistic the
threshold was derived for, and it separates.

**Verdict: real finding.** The 148 are recordings on which no cough-like transient occurred.

## Defect B — `SOUND_COVERAGE` reads a duty cycle against a bound derived as a majority

707 of the 1,366 `False` (51.8 %) come from the two families `_airway_coverage` evaluates.

### The mechanism

```python
# airway.py:829-838, 877-878
covered = merge([extent for extent, scores in scored
                 if any(float(scores.get(label, 0.0)) >= minimum for label in labels)])
total = sum(duration(extent) for extent in covered)
whole = duration(stream_extent(store))
coverage = total / whole if whole > 0.0 else 0.0
...
coverage_min = params.point("breath_coverage_min")            # 0.5
return Result(UNDETERMINED if coverage_min is None else coverage >= coverage_min, ...)
```

`scored` is the `hear_scores` sidecar on HeAR's **own non-overlapping 2 s grid** (`win_length_s
2.0`, `hop_s 2.0`). So `coverage` is not a fraction of anything continuous: it is
`k / n_windows`, where `n` is 16 for a 30 s file and 11 for a 20 s one. The bound's derivation
(`config-derivations.md:1256`) reads in full:

> `branch.breath_coverage_min: 0.5` — A majority. A breathing task asks for breathing throughout the
> extent.

That is a prior, not a fitted value, and it is stated about "the extent" while the code divides by
the whole stream. It was never checked against the distribution it decides on.

### The statistic has no structure at the bound

The measured distribution of `coverage` over `respiration-and-cough-breath` (n = 780), in 0.1 bins:

```
0.0:141  0.1:167  0.2: 52  0.3: 80  0.4: 39  0.5: 65  0.6: 25  0.7: 70  0.8: 30  0.9: 72  1.0: 39
```

It is flat. There is no valley at 0.5 and no valley anywhere else — unlike the cough gate above,
where the cut sits in an empty region. A threshold on a flat distribution is a quantile cut: it
splits a continuum and reports the split as a decision. Moving it moves the answer smoothly:

```
respiration-and-cough-breath (n=780)          respiration-and-cough-v2-breath (n=316)
  coverage >= 0.05  ->  639 pass (81.9%)        coverage >= 0.05  ->  263 pass (83.2%)
  coverage >= 0.10  ->  547 pass (70.1%)        coverage >= 0.10  ->  244 pass (77.2%)
  coverage >= 0.20  ->  432 pass (55.4%)        coverage >= 0.20  ->  210 pass (66.5%)
  coverage >= 0.30  ->  377 pass (48.3%)        coverage >= 0.30  ->  178 pass (56.3%)
  coverage >= 0.50  ->  263 pass (33.7%)        coverage >= 0.50  ->  126 pass (39.9%)
```

Changing the **denominator** is not the lever. Against the declared 30 s rather than the whole
stream the pass rate moves 33.7 % → 32.2 %, because the files are 30 s anyway. The bound itself is
the lever.

### The gate discards evidence the branch itself proposed

On the `False` rows of these two families:

```
family                            False n   >=1 span over score_min   >=1 2 s window over 0.2
respiration-and-cough-breath          517      286  (55.3%)               376  (72.7%)
respiration-and-cough-v2-breath       190      125  (65.8%)               137  (72.1%)
```

`>=1 span over score_min` is **exactly** the condition `_airway_event_series` treats as sufficient
to read `True`. On 411 of the 707 rejected recordings, the branch holds evidence its own other
matcher would have accepted, proposes spans off it, writes `findings: present` — and then reports
that the breathing task did not happen. That is the 514 in the hint table.

### Control contrast — does the statistic discriminate at all?

Yes, and that is what makes the bound the defect rather than the instrument. The same statistic
computed over 3,183 recordings from ten families that elicit **no** breathing task:

```
family                            n   cov p50  cov p90  cov p99   >= 0.5     any window >= 0.2
maximum-phonation-time          385    0.0794   0.3988   0.7659    27 ( 7.0%)   204 (53.0%)
harvard-sentences-list          391    0.0      0.0      0.3001     0 ( 0.0%)     6 ( 1.5%)
prolonged-vowel                 400    0.0      0.3384   0.6700    20 ( 5.0%)   156 (39.0%)
free-speech                     393    0.0      0.1113   0.4380     2 ( 0.5%)   119 (30.3%)
glides-low-to-high              400    0.1196   0.5820   1.0000    52 (13.0%)   202 (50.5%)
word-color-stroop               214    0.0792   0.3163   0.5039     4 ( 1.9%)   178 (83.2%)
animal-fluency                   81    0.0666   0.2332   0.3668     0 ( 0.0%)    65 (80.2%)
diadochokinesis-pataka          399    0.0      0.0      0.2373     0 ( 0.0%)     7 ( 1.8%)
rainbow-passage                 397    0.0      0.0638   0.1971     1 ( 0.3%)    67 (16.9%)
cinderella-story                123    0.0209   0.1048   0.3109     1 ( 0.8%)    77 (62.6%)
ALL CONTROLS                   3183    0.0      0.2757   0.7040   107 ( 3.4%)

respiration-and-cough-breath    263    0.7994   1.0      1.0                    <- AIRWAY True
respiration-and-cough-breath    517    0.1332   0.3998   0.4671                 <- AIRWAY False
respiration-and-cough-v2-breath 126    0.8002   1.0      1.0                    <- AIRWAY True
respiration-and-cough-v2-breath 190    0.1998   0.4063   0.4996                 <- AIRWAY False
```

On material that genuinely contains no sustained breathing the statistic is near zero: median 0.0,
3.4 % over the bound, and 0.0 % on `harvard-sentences-list` and `diadochokinesis-pataka`. The
statistic is measuring breath. **The rejected population sits strictly between the controls and the
accepted population** — median 0.13/0.20 against 0.0 for controls and 0.80 for accepted. They are
not recordings with no breath in them; they are recordings with less breath than a majority of a
30 s file.

### Control contrast — the within-session pairing

The stronger control. 1,130 sessions contributed both a sustained `-breath` recording (coverage
gate) and at least one counted breath recording (event gate) — same participant, same microphone,
same sitting, same target sound.

```
sustained -breath verdict   x   counted breath tasks in the SAME session
  False        | ALL counted pass     466
  False        | some pass            202
  False        | NONE pass             22
  True         | ALL counted pass     322
  True         | some pass             45
  True         | NONE pass              5
  UNDETERMINED | ALL / some / NONE      7 / 52 / 9
```

Of the 795 sustained-breath recordings from sessions where **every** counted breath task passed —
sessions that demonstrably produced breath the same instrument detects, minutes apart — the coverage
gate called **466 (58.6 %) `False`**. Of the 36 from sessions where **no** counted breath task
passed, it called **22 (61.1 %) `False`**.

**58.6 % against 61.1 %.** The coverage gate's verdict carries no measurable information about
whether the session produced detectable breath. (The `NONE pass` arm is only 36 recordings, so the
95 % interval on that 61.1 % is roughly ±16 pp; the honest statement is that no difference is
detectable and the arm that could show one is small. The 466 is not small, and it is the number
that matters: 466 recordings called "the instruction was not followed" where the same participant
followed the same instruction, in the same session, under a different matcher.)

### The rejection rate trends with recording level

Quintiles of `level.lufs` over the 1,096 coverage-family rows of 2 s or more:

```
lufs quintile      [-inf,-50.3] [-50.3,-45.3] [-45.3,-37.7] [-37.7,-25.1] [-25.1,-4.9]
False rate            74.4%        55.3%         52.5%         55.3%        85.0%
coverage p50           0.200        0.400         0.467         0.406        0.102
```

A 32-point swing in the rejection rate driven by how loud the recording is, with the **loudest**
quintile the worst. A statistic meant to say whether a person breathed should not be U-shaped in
gain. This is not diagnosed further here — it is reported as the nuisance trend the brief asked
for, and it is a second reason the bound cannot be a constant.

**Verdict: defect.** The threshold is a prior asserted against a statistic it was never fitted to;
the statistic is quantised to a 2 s grid, is flat across its whole range, and its rejection rate
moves 32 points with recording level and none at all with whether the participant demonstrably
breathed.

## Defect C — `align` says `False` where `routing` says `unavailable`

65 in-family rows carry **no span at all**: PREPROCESS proposed nothing, so there is no `span_hear`
window, no `span_yamnet` window, and no carrier for `sounds_like` to test. Their durations are
0.23 s (p10) to 0.42 s (p90).

- 8 of them are `respiration-and-cough-breath`. `_airway_coverage` checks its instrument
  (`airway.py:823–825`) and returns `UNDETERMINED`.
- **57 of them are event-pattern families. `_airway_event_series` checks only `energy_envelope`
  (`airway.py:641–642`), never the classifier windows, so `airway_events` returns `[]` and
  `_events_reading` (`airway.py:590–604`) returns `False` — because `score_min` was readable.**

```python
def _events_reading(events, params):
    if events:
        return True
    return UNDETERMINED if params.point("score_min") is None else False   # airway.py:604
```

The guard asks whether the *operating point* was measurable. It does not ask whether the
*instrument that consumes it* produced anything. On these 57 recordings no window existed that could
have cleared any cut, and the branch reports that the participant did not do the task. Routing, on
the identical evidence state, reports `airway.cough: unavailable — span_yamnet: spans are absent`.
Two nodes, one store, opposite answers.

**Verdict: defect.** 57 rows, and the mechanism is the same class as VOICE's "never say `False` on
nothing".

Adjacent, and **not** claimed as a defect: 351 in-family rows are under 2 s, of which 268 read
`False`. 193 of those have classifier windows that simply scored nothing over 0.2 on a file shorter
than a single breath. Whether a 0.4 s file should return `False` ("no five breaths occurred", true)
or `UNDETERMINED` ("the instruction asks for 30 s and the recording is 0.4 s") is a design question
for ADMIT, not a reader keyed to a missing writer. It is flagged here because it is 19.6 % of all
`False` and it is not what the flag text says it is.

## The two patterns, answered

| pattern | verdict |
| --- | --- |
| **1 — a reader keyed to something no writer produces** | **Defect**, in one specific form. Ruled out for `energy_envelope`, `hear_scores`, `span_hear`, `span_yamnet`, `silence`, `spectrogram_wideband`, `band_profile`: all seven are written by PREPROCESS on all 6,015 in-family rows that carry a span, and `Breathe`/`Cough` are live keys of `span_hear.raw_scores` on all of them. **Not** ruled out for `branch.label_sets.breath = [Breathe]` against the `span_yamnet` half of `classifier_windows`: AudioSet has no class of that name, so YAMNet's breath vote is unreadable while its cough vote is read. 142 of 511 event-pattern breath `False`. |
| **2 — evidence present and a gate discards it** | **Defect**, in `_airway_coverage`. 514 recordings report `findings: present` and `conformance: False` together; 411 of the 707 rejected hold evidence the branch's own event matcher treats as sufficient. The bound is a prior on a flat, grid-quantised statistic, and it rejects 58.6 % of sustained-breath recordings from sessions in which every counted breath task passed, against 61.1 % from sessions in which none did. |
| — the cough families | **Real finding.** 148 `False`, clean bimodal separation, zero overlap at the cut. |
| — the 5,258 `UNDETERMINED` | **Honest.** 5,190 are `detect_airway` by construction; 68 are `instrument_absent("hear_scores")` on sub-2 s files, reported as an absence with a note. |
| — the 20 `unavailable` routes | **Honest.** One gate, `airway.cough`, unreadable because `span_yamnet` wrote no window on a 0.12–0.37 s file. |
| — the duration trend | **Undetermined on this evidence.** Restricted to files of 2 s or more, the `False` rate rises with duration on the breath event families (`fivebreaths` 5 %→19 %, `-v2-threebreathsmouth` 6 %→20 %, `breath-sounds` 6 %→18 % across duration quartiles) and is flat or falling on the cough ones (`-cough` 8 %→7 %, `threequickbreaths` 6 %→3 %). The median amplitude-span count *rises* over the same quartiles (11→15 on `fivebreaths`), so it is not fewer carriers. A mechanism is not established from store reads alone and none is claimed. |

## How many recordings this would move

Over the 29,484 read, scaled to the 62,578 the campaign targets (×2.122):

| | measured | scaled |
| --- | ---: | ---: |
| every AIRWAY `False` | 1,366 | ~2,899 |
| every `claimed_not_found` | 920 | ~1,952 |
| **Defect B** — `SOUND_COVERAGE` `False` | 707 | ~1,500 |
| — of which `present` **and** `False` at once | 514 | ~1,091 |
| **Pattern 1** — breath `False` a readable YAMNet vote covers | 142 | ~301 |
| **Defect C** — `False` on a store with no span | 57 | ~121 |
| unaffected: cough-family `False` (real findings) | 148 | ~314 |

Defects B and C are disjoint (C's 57 are all event-pattern; B's 707 all coverage). Pattern 1's 142
are a subset of the 511 event-pattern breath `False` and disjoint from both. **The three together
move 906 of the 1,366 `False`, ~1,923 of the projected ~2,899.**

## The fixes, as proposed — none made

### 1 — `branch.label_sets` must be readable by both writers it is tested against

`sounds_like` is handed two classifiers' windows and one list of names. Give it the names both
classifiers use, from the profile the repo already ships and validates rather than by hand.

`airway.py:145` `classifier_windows` and `branches.py:2026` `sounds_like` stay as they are. The
label set becomes a per-classifier resolution: for `span_hear`, the HeAR head name; for
`span_yamnet`, `corroboration_sets()[name]` from `classifier_ontology` — which is
`{Breathing, Gasp, Pant, Snoring, Snort, Wheeze}` for `Breathe` and `{Cough, Throat clearing}` for
`Cough`, both already validated as closures over the profile's node table.

Pre-alpha, so `branch.label_sets` is replaced, not aliased: it becomes a map from kind to the HeAR
head, and the AudioSet side is derived. `config-derivations.md` records that `Cough` matched both
vocabularies by coincidence of spelling and `Breathe` did not, which is why the mapping is now
explicit.

**`branch.score_min` must then be re-derived.** Its derivation already records the owed
measurement — "one cut across two classifiers whose scales were never compared must be measured"
(`config-derivations.md:1251`). Making the second classifier readable for breath is the point at
which that debt comes due. Do not ship the union at 0.2 on the strength of the 142 above; fit the
cut against the paired-session contrast this document establishes (a session's counted breath tasks
are a within-subject label for whether that participant produces detectable breath).

### 2 — `_airway_coverage` must not decide on an unfitted bound

Three things are wrong and they need separating.

**(a) The bound.** `branch.breath_coverage_min: 0.5` is a prior. Per the repo rule, a threshold
belongs in `data/` with a written derivation, regenerated from measured verdicts. The pairing in
this document supplies the verdicts: 1,130 sessions in which the counted breath tasks say whether
the participant produced detectable breath and the sustained recording is the item to be graded.
Fit the cut against that, write the derivation, and put it in a profile — do not hand-edit 0.5 to
0.2 because the sweep above makes 0.2 look better.

**(b) The statistic.** `k / n_windows` on a non-overlapping 2 s grid cannot resolve a respiratory
duty cycle: at 12–16 breaths/min the audible portion of each cycle is of the order of 1–1.5 s out
of 4–5 s, so a correct performance cannot occupy a majority of 2 s windows unless the breathing is
loud and near-continuous. Either score on an overlapping grid, or stop asking for a duty cycle and
ask what the row actually asks: *was breath present throughout the extent, rather than only at its
start* — which is a question about the **gaps** between covered runs, and is invariant to how loud
each breath is. The second is the smaller change and the one that matches the row's own words.

**(c) The denominator.** `whole = duration(stream_extent(store))` (`airway.py:837`) contradicts the
derivation's "throughout the extent" and contradicts `declared_duration_s`, which the row already
carries (30.0 / 20.0). It is worth only 1.5 pp on this corpus, so fix it for correctness, not for
yield, and say so.

Until (a) and (b) are settled the honest return from `_airway_coverage` is **`UNDETERMINED` with the
coverage fraction reported as a measurement**, not `False`. That alone removes 707 `False` and the
514 self-contradictions, and it costs nothing: `breath_coverage_fraction` is already written
(`airway.py:856`) and VERDICT already declines to flag `UNDETERMINED`.

### 3 — `_events_reading` must not say `False` on a store with no instrument

`_airway_event_series` (`airway.py:641–642`) and `_airway_alternation` (`airway.py:736–737`) check
`read_envelope_track` and return `instrument_absent("energy_envelope")`. They do not check that
`classifier_windows` returned anything. Add the second check, in the same shape as the first:

```python
windows = classifier_windows(store)
if not windows:
    return instrument_absent("span_hear")
```

`instrument_absent` already returns `UNDETERMINED` with a recorded `event_instrument` measurement
and already surfaces as `notes: ["the span_hear derivative is absent"]` through `_detail`. 57 rows,
and it makes `align_airway` agree with what routing already reports on the same store.

## Reproducing this

```bash
# on ORCD, read-only over the existing corpus. Nothing is submitted; both scripts are
# I/O-bound readers over rows/ and out/ and need no cluster job.
D=/orcd/scratch/bcs/002/satra/triage_design_20260919
PY=/orcd/scratch/bcs/002/satra/senselab-corpus2/.venv/bin/python3.12   # the login node's is 3.6

nice -n 19 $PY specs/20260817-triage-workflow-dag/airway-flag-grounds-census.py \
    --run $D/run --out $D/airwaygrounds --workers 10
nice -n 19 $PY specs/20260817-triage-workflow-dag/airway-flag-grounds-tables.py \
    --out $D/airwaygrounds --workers 10
```

The census takes five steps and about twenty minutes over 30,000 stores; naming a step
(`... --out <dir> spans`) re-runs just that one. Every table in this document is printed by the
second script, in the order the sections appear above, and every figure quoted here was checked
against that output rather than transcribed from the working notes.
