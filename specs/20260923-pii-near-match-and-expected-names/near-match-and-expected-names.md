# Near-spelling matches, expected names, and the reviewer's population

Owner-directed, 2026-09-23:

> "let's make sure LLM review for PII runs on anything that goes through the detectors
> gliner/presidio/etc. also we do need to ensure that expected words/near spelling mismatches don't
> trigger redaction. analyze the task to determine expected names."

Three changes over one corpus. Every number below is from `triage_rerun_20260923`, 62,548 completed
run directories, reduced by `measure-near-match.py` in this directory. The reduction emits counts,
categories and distance histograms only; no transcript token and no detected surface leaves the
host, which is why `tables.json` is committed beside it.

## 1. The reviewer's population

REDACT concluded on **18,850** of the 62,548 recordings — it runs only where SPEECH's scan found
something. The reviewer had run on **none** of them:

| REDACT outcome | recordings | `redaction_llm_annotation.status` |
| --- | --- | --- |
| `pass` | 13,598 | `disabled` |
| `fail` | 5,252 | `not_run` |
| `flag` | 0 | — |

`disabled` because the corpus run left `redaction.llm_check.enabled` false; `not_run` because the
ladder tested the detector outcome *before* the config, so `enabled: true` could not have reached
the withheld population either. The fix and the rule are in
`specs/20260817-triage-workflow-dag/llm-check.md`.

### What the withheld population actually is

Every one of the 5,252 was withheld on one ground:

| ground | recordings |
| --- | --- |
| `verification found pii on the redacted transcript` | 5,252 |
| `the store's pii scan is incomplete` | 0 |
| `the re-scan over the redacted text is incomplete` | 0 |

That is what settles what the reviewer reads there. The question a human has about a withheld
recording in this corpus is not abstract — *the second detector pass says something identifying
survived this redaction; did it?* — and it is a question about the redacted transcript, which is
the text the reviewer already reads under the prompt it already has. Showing it the findings in the
clear would answer a different question, *were the detectors right to mark these?*, and answering
that is being an authority on whether a redaction was warranted, which the step's contract forbids
in the other direction and should forbid in this one.

### What it costs

`llm-check-amortised-load.md` sized ~13.5 GPU-hours amortised over ~13,600 recordings — that is
the *passing* population, 13,598 here, so the figure is ~3.6 s of amortised GPU per recording.
Widening to every recording the detectors touched takes the population to 18,850, a **1.386×**
increase: **~18.7 GPU-hours** for this corpus at the same per-recording cost, +5.2 hours.

Two caveats the arithmetic does not carry. A withheld recording is one whose redacted transcript
still scans dirty, so its reviews are more likely to flag and therefore to use more of the
`max_iterations: 3` budget than a passing one; the per-recording cost on that population is an
upper-bounded unknown until it is measured, and **it has not been measured**. And
`keep_worker_resident: true` remains off: the worker holds 70.4 GiB and OOMed a corpus driver
(19 of 22 recordings died), so the load is paid per recording and the widening pays 5,252 more
loads, which is where the 5.2 hours actually goes.

## 2. The near-match bound

### What is being fitted, and against what oracle

`_expected_exemptions` (REDACT) asks whether a finding's covered consensus words occur, in order
and contiguously, inside one declared stimulus unit. The comparison at each position was exact
string equality on the normalised key. Exact equality is too brittle for ASR output.

The **stimulus aligner** is an instrument already in the graph and independent of the exemption
pass: `derivatives/stimulus_alignment.npz` records, per consensus word, whether the aligner placed
it on a declared stimulus position (`realised`/`substituted`) or nowhere in the stimulus
(`unexpected`). That judgment is the oracle.

* **positives** — a live `pii` finding every one of whose covered words the aligner placed *on* a
  stimulus position. n = 29,485. The stimulus accounts for it; the exemption pass may still redact
  it.
* **negatives, in-corpus** — a finding every one of whose covered words the aligner placed
  *nowhere* in the stimulus. n = 10,438. This population is contaminated: exact matching already
  admits 5.54% of it, because a word the aligner could not place at a position may still equal a
  token elsewhere in the unit (a repetition). It is reported but does not decide.
* **negatives, donor control** — every finding in the corpus (n = 224,769), evaluated against the
  Rainbow Passage as a stimulus it has nothing to do with. Any admission here is unambiguously a
  false admit. This is the population that decides.

The predicate evaluated is the shipped one — contiguous, ordered, within one unit — with **only**
the per-token comparison widened. The grid is 40 `(exact_below, two_edits_from)` pairs; a token
pair shorter than `exact_below` must match exactly, one shorter than `two_edits_from` may differ by
one edit, one at or above it by two. `(99, 99)` is today's rule.

### The result

| rule | admits, on-stimulus | yield | admits, donor | false | admits, off-stimulus |
| --- | --- | --- | --- | --- | --- |
| `(99, 99)` exact, today | 13,147 | 44.59% | 385 | 0.171% | 5.54% |
| `(8, 8)` | 13,307 | 45.13% | 385 | 0.171% | 5.86% |
| `(6, 8)` | 13,484 | 45.73% | 390 | 0.174% | 5.97% |
| **`(5, 8)` shipped** | **13,662** | **46.34%** | **478** | **0.213%** | **6.05%** |
| `(5, 6)` | 13,817 | 46.86% | 713 | 0.317% | 6.23% |
| `(4, 7)` | 13,952 | 47.32% | 1,982 | 0.882% | 6.26% |
| `(3, 6)` | 14,170 | 48.06% | 2,903 | 1.292% | 6.63% |

The knee is at `exact_below = 5`. Relaxing it to 4 **quadruples** the donor false-admit rate,
0.213% → 0.882%, to buy one further point of yield. `two_edits_from = 8` is the loosest second
bound that costs nothing at `exact_below = 5`: donor admits are 478 at both 8 and 99 while yield is
46.34% against 46.03%; below 8 the cost starts (500 at 7, 713 at 6).

### The error rate in both directions, at the shipped bound

**False-admit direction — the one that matters.** *"A tolerance that also admits genuinely
different names is worse than none."* Against a donor stimulus, by detector category:

| category | findings | exact admits | shipped admits | added |
| --- | --- | --- | --- | --- |
| PERSON | 114,371 | 307 (0.268%) | 327 (0.286%) | **+20** |
| NAME | 65,041 | 38 (0.058%) | 42 (0.065%) | **+4** |
| LOCATION | 12,230 | 22 (0.180%) | 22 (0.180%) | **0** |
| DATE_TIME | 19,860 | 11 (0.055%) | 78 (0.393%) | +67 |
| MISC | 3,669 | 7 | 9 | +2 |
| every other category (AGE, NRP, LOC, UNIQUE_IDENTIFIER, VEHICLE_IDENTIFIER, US_SSN, …) | 9,598 | 0 | 0 | 0 |

**24 further false admits across 191,642 PERSON / NAME / LOCATION findings — 1.3 per 10,000.** The
one category that moves materially is DATE_TIME (+67, +0.34 points), whose surfaces are short and
numeric and which sits furthest from what "a name the recogniser spelled differently" means.

**Miss direction, and the honest statement about it.** 53.66% of aligner-confirmed on-stimulus
findings are still not admitted, down from 55.41%. The widening recovers 515 of the 16,338 the
exact rule missed — **3.2% of the residual**. Near-spelling is a real but small part of the miss.
The rest is structural: `_expected_exemptions` also requires the covered words' hulls to span the
whole extent, every covered key to be non-empty, the finding not to reach every consensus word, and
the run to lie inside *one* unit. The aligner is bound by none of those. **Widening the string
comparison does not fix the bulk of the miss, and this change should not be reported as though it
does.**

### The tri-state is preserved

`in_stimulus` stays `true` / `false` / `null`. `stimulus_tokens(None)` is `None` and
`in_stimulus(surface, None, near)` is `None` on every path; only the comparison between a surface
and a *declared* stimulus was widened. 4,537 of 22,319 findings reading `null` was the artefact
that hid this family, and collapsing "checked and not present" into "never checked" would hide it
again.

### Substitution distances, for the shape of the rule

The aligner's own substitutions — expected token against the consensus word it was paired with,
both normalised — are the ASR's spelling variation as this corpus produces it:

| normalised length | n | d=1 | d=2 | cum d≤1 | cum d≤2 |
| --- | --- | --- | --- | --- | --- |
| 3 | 56,443 | 1,248 | 6,298 | 2.21% | 13.37% |
| 4 | 42,376 | 1,778 | 2,262 | 4.20% | 9.53% |
| 5 | 24,084 | 510 | 799 | 2.12% | 5.44% |
| 6 | 17,750 | 338 | 442 | 1.90% | 4.39% |
| 7 | 11,863 | 152 | 189 | 1.28% | 2.87% |
| 8 | 10,529 | 519 | 113 | 4.93% | 6.00% |
| 9 | 2,685 | 62 | 47 | 2.31% | 4.06% |
| 10 | 4,215 | 21 | 10 | 0.50% | 0.74% |

The population is dominated by `d ≥ 4` at every length: the aligner's "substituted" is mostly a
genuinely different word, not a respelling. That is why the yield a near-match bound can buy is
bounded at a couple of points, and it is the measurement behind not reaching for a larger one.

## 3. Expected names

### Which families can carry a declaration, and which cannot

| family | can declare expected names | why |
| --- | --- | --- |
| `cinderella-story` | **yes** | A closed, knowable cast. `stimulus_text` is empty on all 258 recordings — the source is a physical storybook — so `in_stimulus` is `null` on all 5,833 of its findings and the question was never asked. |
| `harvard-sentences-list`, `cape-v-sentences`, `rainbow-passage`, `caterpillar-passage`, `word-color-stroop` | **no, and they need none** | They declare `stimulus_text`; the haystack is the declaration. |
| `story-recall`, `story-recall-v2`, `free-speech`, `open-response-questions`, `productive-vocabulary` | **no, and they need none** | `token_source="stimulus_text"`; a declaration is present per recording. |
| `picture-description`, `picture-description-option1`, `picture-description-option2` | **no, and no list can be written** | The stimulus is an **image**. There is no text a faithful performance is expected to contain, and any list would be a guess at what a participant chose to name. Declaring one would turn `null` — "never checked" — into a false `false`. |
| `free-speech-v2`, `animal-fluency`, `random-item-generation`, `random-item-generation-v2` | **no** | The instruction asks the participant to choose the content. A cast here is a vocabulary nobody fitted. |
| the ten `diadochokinesis-*` families | **no, already covered** | `declared_carrier()` derives the carrier from the family name (`buttercup`, `pataka`, …); the scan gate already reads it. |

### Cinderella: the measurement

5,833 live findings over 258 recordings, 230 of which reached REDACT. Their distance to the
declared cast:

| distance to nearest cast name | findings | share |
| --- | --- | --- |
| 0 | 4,492 | 77.16% |
| 1 | 197 | 3.38% |
| 2 | 422 | 7.25% |
| ≥3 | 711 | 12.21% |

**80.54% of this family's findings sit within one edit of a name the task itself puts in the
speaker's mouth.** By category:

| category | findings | within one edit of the cast |
| --- | --- | --- |
| PERSON | 2,633 | 2,427 (92.18%) |
| NAME | 2,163 | 2,067 (95.56%) |
| MISC | 249 | 178 (71.49%) |
| DATE_TIME | 740 | 13 (1.76%) |
| LOCATION | 21 | 2 (9.52%) |
| NRP, VEHICLE_IDENTIFIER | 14 | 0 (0%) |

The shape is exactly right: the declaration accounts for the name categories and leaves the date,
location and identifier categories almost untouched. A `cinderella-story` retelling's PERSON
findings are the cast, and its DATE_TIME findings are not.

### Where the declaration lives

On the family's own `Expectation` row (`SPEECH_EXPECTATIONS` in `nodes/branches.py`), as
`expected_names`. Not in `data/`, not as a fitted threshold: it is a *declaration* about what the
instruction asked for, the same kind of thing as `sequence=BUTTERCUP` or `tokens=("hey",)` on the
rows beside it, and the architecture already puts a declaration there.
`specs/20260923-free-speech-review-page/no-stimulus-to-check-against.md` argues the same. It is
task metadata, not detected data, which is what makes writing the list down permissible at all.

The list is matched under the same fitted `near_match` bound as a declared stimulus, so the 3.38%
at distance 1 are admitted with the 77.16% at distance 0.

## What could not be measured

* **The reviewer's per-recording cost on the withheld population.** No reviewer has ever run on a
  withheld recording, so the extra iterations a dirtier transcript provokes are an unknown; the
  ~18.7 GPU-hour figure assumes the passing population's per-recording cost.
* **The reviewer's false-positive rate on this corpus**, unchanged from
  `llm-check.md` — still unmeasured, and `verdict.llm_redaction_flags` still ships `true` on an
  asymmetry argument rather than on a rate.
* **Whether an admitted near match is *correct*.** The oracle is the stimulus aligner, not a human.
  A finding the aligner placed on a stimulus position is treated as accounted for; if the aligner
  is wrong there, the yield figure inherits its error. The donor control has no such dependence,
  which is why the false-admit rate is the number the bound was chosen on.
* **`cinderella-story` cast recall.** 77.16% at distance 0 says the list covers most findings; it
  does not say what fraction of the *cast actually spoken* the list contains, because that would
  need the transcripts read.
