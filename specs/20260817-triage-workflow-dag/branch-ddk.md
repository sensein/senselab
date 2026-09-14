# DDK branch

What the branch answers: **is there a repeated syllable train here, how fast is it, how regular, and
does it hold across the effort?**

The frame is [`../20260913-branch-contract-and-hints/design.md`](../20260913-branch-contract-and-hints/design.md);
shared rules are in [`branch-conventions.md`](branch-conventions.md); owed ground truth is in
[`branch-listening-sample.md`](branch-listening-sample.md).

## The state of this branch

**There is no node.** No `nodes/ddk.py` exists, `DDK` is absent from `GRAPH_ORDER`
(`vocabulary.py:14-25`) and from `run.py`'s dispatch table (`run.py:297-301`). `DDK` *is* in
`BRANCHES` (`vocabulary.py:31`), so ROUTING writes it a `branch_decision` and `_drive_branches`
records it `SKIPPED` with `NO_NODE` (`run.py:302-305`).

**Since ruleset stage 2 it is routable for the first time.** No `kind` ever mapped to DDK, so under
the old fold it could not be selected at all. The consequence is live and large: a recording whose
DDK gates fire now flags the file, because VERDICT records "was asked to run and never ran" — and
that is **all 22,363 recordings DDK routes, of which 14,878 declare no DDK family**. An earlier
version of this document implied it fires only on recordings with DDK content, understating it by
roughly threefold against numbers the same document carries below. Since this flag is the standing
argument for building the node, the number matters.

Everything below is **not built** unless it says otherwise.

## The tasks this branch serves

Declared families, not ground truth. `SYLLABLE_REPETITION` (`families.py`) holds exactly these ten,
and they sum to 7,989 — matching the declared DDK count exactly, with no residual.

| family | n | what the task asks for |
| --- | --- | --- |
| `diadochokinesis-buttercup` | 896 | repeat a trisyllabic word |
| `diadochokinesis-ka` | 896 | repeat /ka/ |
| `diadochokinesis-pa` | 896 | repeat /pa/ |
| `diadochokinesis-pataka` | 896 | repeat /pa-ta-ka/ |
| `diadochokinesis-ta` | 896 | repeat /ta/ |
| `diadochokinesis-v2-buttercup` | 702 | as above |
| `diadochokinesis-v2-kuh` | 702 | repeat /kuh/ |
| `diadochokinesis-v2-puh` | 702 | repeat /puh/ |
| `diadochokinesis-v2-tuh` | 702 | repeat /tuh/ |
| `diadochokinesis-v2-puhtuhkuh` | 701 | repeat /puh-tuh-kuh/ |

The protocol's own instruction says what is wanted — a v2 DDK `_acoustictask-metadata.json` reads
*"imitate the speaker by repeating the syllables 'puhtuhkuh' as quickly and consistently as possible
until the timer runs out"*.

**"As quickly and consistently as possible" is the measurement.** Diadochokinetic rate is a standard
clinical measure of articulatory motor control with two components: **rate** and **regularity**. The
families split into *alternating motion rate* — a single repeated syllable, 4,794 declarations — and
*sequential motion rate*, a repeated sequence of different syllables, 3,195. Sequential motion rate
stresses the ability to switch place of articulation and is the more sensitive of the two.

### A recording routed here whose declared task is not DDK

DDK routed 22,363 recordings against 7,989 declaring a DDK family. Repetition occurs in ordinary
speech — a stutter, a false start, a repeated word. The branch measures what it finds and says what
it is; it does not assert that a Harvard sentence failed to be a DDK task.

**DDK's gate evidence is `unavailable` on 2,345 recordings** (`ruleset_score.json`,
`totals.unavailable`). Ascending, the four are SPEECH 0, VOICE 29, **DDK 2,345**, AIRWAY 56,505 — so
DDK is second-*highest*. An earlier version called it the highest, and the first correction called it
the second-lowest; both were wrong. The design conclusion is unchanged either way: an unavailable
posteriorgram is an absence, never a negative.

## How DDK is routed today

Two gates, either of which routes (`default.yaml:218`, `:264-271`):

| gate | feature | threshold |
| --- | --- | --- |
| `ddk.lexical_repetition` | `transcript_repeat` — largest repeat count of any normalised token | `>= 3`, marked **UNMEASURED** in the config |
| `ddk.ppg_segment_rate_per_s` | `ppg.segment_rate_per_s` — contiguous argmax-phoneme segments per second | `>= 10` |

The second reads no transcript, so it routes a syllable train an ASR declines to transcribe — which
is most of them, since `/pa-pa-pa/` is not lexical.

**Neither may be refit.** Both would be fitted against declared DDK families, encoding which
recordings the protocol labelled rather than which carry a syllable train.

## Capabilities

### D1 — Locate the train and measure its modulation (**not built; primary**)

**Question.** Where is the repeated production, how fast is it, and how regular?

**Propose the train from the envelope, then qualify it by repetition structure** — the same
inversion VOICE V1 applies. A train defined *by* detected repetition cannot be found when the
repetition is irregular, which is the presentation the task exists to detect. An envelope-proposed
train with weak or absent modulation structure is a finding about the production; a train that was
never proposed is an absence of data, and would concentrate `DDK: FAIL` on the most impaired
speakers.

**Reads.** The audio, from **`plain`** — not `enhanced`, and not PREPROCESS's stored
`energy_envelope`. Both exclusions matter: the stored envelope is wrong for the reasons below, and
the `enhanced` stream is FRCRN output, which is out of domain on a DDK train. It also feeds the PPG
and therefore this branch's own routing gate — **which is a permanent state, not a pending repair**:
audit step 1b is withdrawn and the posteriorgram reads `enhanced` by decision
([`praat-instrument-audit.md`](praat-instrument-audit.md) finding 0 and step 1b). D1's exclusion is
D1's own, and stands on the envelope's own grounds.

**Computes.** The amplitude-envelope modulation spectrum over the proposed train. A syllable train is
an amplitude modulation at the repetition frequency; the spectral peak gives the **rate** directly.

**The stored envelope is the wrong input, for three reasons.** `energy_envelope` is
`hilbert_envelope_dbfs` over the **pre-emphasised** stream, Butterworth-lowpassed, stored in
**dBFS** (`preprocess.py:1283-1288`):

1. **A logarithmic envelope generates harmonics of the modulation fundamental through the
   nonlinearity alone.** The f-versus-3f peak ratio is this capability's entire unit-disambiguation
   mechanism, and on a dB envelope part of that structure is an artefact of the conversion rather
   than of production unevenness. **D1 needs a linear-amplitude or power envelope** and says so.
2. **Pre-emphasis (+6 dB/octave) reweights burst against vowel energy** — which is exactly the
   asymmetry that creates f/3f structure on `/pa-ta-ka/`. So the harmonic ratio would depend on
   `preemphasis.coefficient`, a key derived for cough and mouth-sound contrast rather than for this.
3. The lowpass is `envelope.lowpass_hz: 40.0`, derived at `config-derivations.md:43-47`. D1 listed
   "a lowpass cutoff" among the conventions it must declare; **one exists and is in force**, so D1
   inherits it by name rather than declaring a second.

**The unit is ambiguous by a factor of three on the sequential families, and this must be resolved
in the output.** For `/pa-ta-ka/`, `/puh-tuh-kuh/` and `/buttercup/` — **3,195 declarations** —
syllables within a cycle differ in amplitude, so the envelope carries energy at the **cycle** rate
(syllable rate ÷ 3) as well as at the syllable rate. Which is the global peak depends on how uneven
the production is. Reporting whichever peak is larger as "the rate" would put a normal speaker at 5
syllables/s into the severe-dysarthria range at 1.7, or the reverse.

**So report the harmonic peak structure and state which unit the number is in.** For the alternating
families the ambiguity does not arise, but the output is explicit there too rather than implicitly
different.

**f and 3f do not by themselves resolve the unit**, because an alternating train is also a
quasi-periodic pulse train carrying energy at 3×. The peak structure says *there is harmonic
structure*; it does not say which harmonic is the syllable rate.

**D2's nucleus count over the train extent supplies the unit** — syllables ÷ duration is unambiguous.
That does not reconcile the two rates into one number, which stays forbidden; it uses the unambiguous
one to label the precise one.

**But the unit then inherits a detector D2 itself declares compromised, and the bias runs toward the
error this is preventing.** D2's minimum sounding interval of 0.1 s "sits on the measurand" at
6–7 syll/s and merges syllables — which biases the nucleus rate **downward, toward the cycle rate**.
That is precisely the 3× confusion the disambiguation exists to resolve, so a merged-syllable
recording can have its unit assigned in the wrong direction. D3 makes this argument about CV; it
applies to the unit as well.

**The harmonic pattern can resolve the unit, under a condition that must be stated.** For a
three-syllable cycle with **one odd syllable** — pattern (A, B, B) — the 3-point DFT gives
|X₁| = |X₂| = |A − B| and |X₃| = A + 2B, so **`|X1| ≈ |X2| < |X3|` is a signature** that the
fundamental is the cycle rate and 3f the syllable rate.

**But `/pa-ta-ka/` generally has three *distinct* amplitudes** — labial, alveolar and velar releases
differ — where |X₁| ≠ |X₂| and the signature does not apply. And after pulse-shape weighting the
`< |X₃|` half can vanish while the `≈` half survives, so **the robust part is the equality of the
first two harmonics**, not the full three-term pattern.

This is therefore **conditional on the (A, B, B) pattern, not the usual case** — and 3,195
recordings' unit assignment rests on it. **The tolerance for `≈` is an owed operating point**, filed
in the table above.

**Not parameter-free, and only one of the four has a value.** An earlier version claimed the method
was parameter-free and then listed four conventions without supplying three of them:

| convention | status |
| --- | --- |
| lowpass cutoff | **in force, but inherited across a condition change** — `envelope.lowpass_hz: 40.0`, derived at `config-derivations.md:43-47` **on the dB, pre-emphasised envelope**, while D1 requires a linear-amplitude envelope on `plain`. That is a value derived under one condition applied under another — the structure [`praat-instrument-audit.md`](praat-instrument-audit.md) finding 8 condemns. Probably harmless at 40 Hz against a syllable rate of 5–7 Hz, but stated rather than assumed |
| envelope extraction method | **owed** — linear-amplitude or power, per the argument above, but the method is unspecified |
| analysis window | **owed** — and per [`branch-conventions.md`](branch-conventions.md) a measurement with no stated window is comparable to nothing |
| **search band** | **owed, and decisive** — it determines which harmonic is findable at all, which is this capability's whole unit problem |
| harmonic-equality tolerance | **owed** — how close `|X1|` and `|X2|` must be to read as equal, which is what the (A, B, B) unit signature below turns on |

D3 marks its analogous parameters owed; this does the same rather than listing conventions as though
declaring them were the same as having them.

**Peak sharpness is not a regularity measure.** Spectral resolution scales as 1/T, so a short train
gives a broad peak however regular the production was, and the measure would report brevity as
irregularity. Either normalise sharpness by analysis duration or do not report it as regularity —
**D3's interval coefficient of variation is duration-robust and carries regularity instead**.

**Emits.** A `propose` span, `family: "ddk"`, subject to the `propose`/`refine` rule; a per-span
measurement carrying the harmonic peak structure with its stated unit and the declared conventions.

**Owed.** A gap criterion for where the train breaks. Stating it relative to the train's own
inter-onset interval keeps it scale-free, but the factor is owed.

### D2 — Syllable-nucleus rate (**not built; cross-check**)

**Question.** What rate does nucleus counting give, and does it agree with D1?

`extract_speech_rate` (`praat_parselmouth.py:103`) returns `speaking_rate`, `articulation_rate`,
`phonation_ratio`, `pause_rate` and `mean_pause_dur`.

**Report rate over the train extent as primary, articulation rate as secondary.** Articulation rate
excludes intra-train pauses — and in DDK those pauses are part of the deficit the task probes.

**Four undeclared operating points, one of them data-dependent.** They are literals inside the
helper, in no config:

| what | where | value |
| --- | --- | --- |
| silence threshold | `praat_parselmouth.py:154` | `silence_db = -25` |
| minimum dip between peaks | `:161` | `min_dip = 4` |
| **…dropped to 2 when mean HNR < 60** | `:166-167` | data-dependent |
| minimum pause duration | `:171` | `min_pause = 0.3` |

**The HNR switch is the serious one.** It changes syllable-detection behaviour based on a
voice-quality measurement of the recording itself, on a corpus of dysphonic speakers — so the
detector's sensitivity is conditioned on the thing being measured. In practice mean HNR is far below
60 dB for any real recording, so the `min_dip = 2` branch is effectively always taken.

**And it fails in the wrong direction on the degenerate case.** If Praat returns undefined for the
mean, `NaN < 60` evaluates `False`, so `min_dip` stays at **4** — the stricter setting — on exactly
the recordings where pitch could not be measured at all. The one population where the branch flips is
the one least able to bear it, and it flips silently.

**Two further deviations from Praat** in the same helper: `min_pause` **0.3 s against Praat's 0.1 s**
and minimum sounding interval **0.1 s against 0.05 s** — both of which change what counts as a
syllable in a rapid train, which is this branch's entire measurement. And its `to_pitch_ac` differs
from Praat on six parameters, with the code's own comments recording that no reason was found.

**All of it is owed**, and the measurements are in
[`praat-instrument-audit.md`](praat-instrument-audit.md).

**Demote the PPG segment rate.** `ddk.ppg_segment_rate_per_s` is a fine *gate* — it routes without a
transcript — but as a measurement an argmax-change rate has no interpretable units: one syllable with
an onset consonant and a vowel yields two or more segments, and the count depends on the
posteriorgram's inventory rather than on articulation. Keep it for routing; do not report it as a
rate.

**Do not reconcile D1 and D2 into one number.**

### D3 — Inter-onset interval structure (**not built; carries regularity**)

**Question.** How does the interval between productions behave across the train?

**Onsets come from the time domain, never from D1's modulation phase.** Narrowband phase tracking
around a spectral peak **imposes near-constant intervals by construction**, so an interval CV derived
from it is biased toward zero — and biased *most* where the production is most irregular, which is
the presentation the task exists to detect. Since D3 now carries regularity (D1's peak sharpness
having been demoted), taking intervals from the phase would re-create, inside the replacement, the
exact bias structure the envelope inversion removed.

**Pick onsets in the time domain from the linear envelope.** D2's nuclei are an acceptable source;
D1's phase is not. **The time-domain peak-picking parameters are owed** and were not previously
listed.

**Computes.** From those onsets: the sequence of inter-onset intervals, then

- its **dispersion** — the coefficient of variation, dimensionless and so comparable across speakers
  at different rates, and **duration-robust** in a way D1's peak sharpness is not;
- its **trend across the train** — festination, slowing, or irregular-without-trend. Among the most
  discriminative DDK features and currently absent entirely;
- **amplitude regularity** across productions, the intensity analogue.

**But CV rests on a detector D2 declares compromised, and the bias runs the wrong way.**
`extract_speech_rate`'s **minimum sounding interval is 0.1 s against Praat's 0.05 s** — and at
6–7 syll/s the syllable period is 140–170 ms with only 80–110 ms voiced, so **the minimum sits on the
measurand**. Merged or dropped syllables produce doubled intervals, which **inflates CV at high
rates — the fastest, healthiest speakers.** That is the opposite direction from the phase-tracking
bias just removed, and not smaller.

**And CV over a sequential train is not comparable to CV over an alternating one.** In `/pa-ta-ka/`
the envelope onset lands differently relative to the release for /p/, /t/ and /k/, so the three
within-cycle intervals are **unequal by measurement convention** and a pooled CV has a floor set by
syllable identity rather than by motor control. **Compute CV within syllable position** on sequential
trains, or report both — 4,794 alternating declarations against 3,195 sequential.

**Parameter-free as measurements.** Dispersion and trend of a sequence need no threshold.

**Emits.** A per-span measurement, with its support count per
[`branch-conventions.md`](branch-conventions.md). **No verdict.**

### D4 — Segment inventory over the train (**not built**)

**Question.** What phonetic segments make up the train?

**Reads.** `extract_ppg_segments`
(`tasks/features_extraction/ppg.py:349`), which returns contiguous argmax-phoneme segments carrying
`phoneme_index`, `phoneme`, `start_frame`, `end_frame`, `frame_count`, `start_seconds`,
`end_seconds` and `duration_seconds` — eight keys.

**Emits.** A per-span measurement over the D1 train. The train span itself is D1's; this adds the
segment inventory to it.

**Owed.** Whether the posteriorgram is usable here at all — see D6.

### D5 — Train duration as a fraction of recording duration (**not built**)

The conformance check the task actually admits. The instruction is *"until the timer runs out"*, so
there is no declared syllable count — but a train occupying a small fraction of the recording is a
departure from *"until the timer runs out"* that is measurable without any norm.

**Emits.** A `counts` entry carrying `found` (train duration, and the fraction) with no `declared`
half.

### D6 — Sequence conformance (**not built**)

**Question.** Were the syllables produced in the order the task asked for?

For `/pa-ta-ka/` the expected cycle is three places of articulation in order. A speaker producing
`/pa-ta-pa/` has substituted; one producing `/pa-pa-pa/` has collapsed a sequential task into an
alternating one — a clinically meaningful finding, and the reason the sequential families exist.

**Emits.** One `deviate` assertion per departure with its extent, typed
**`syllable_sequence_mismatch`** — *not* `stimulus_mismatch`. The contract defines
`stimulus_mismatch` as a lexical word differing from the stimulus text; DDK tasks carry no stimulus
text and expect no lexical content, and the provenance here is PPG argmax over out-of-domain rapid
nonsense syllables. Sharing the name would pool two different objects.

**Owed.** The mapping from PPG phoneme labels to expected syllables. Whether `/puh/` reliably
surfaces as a particular argmax phoneme is **unmeasured**, and the posteriorgram is out of domain on
rapid nonsense repetition.

## Deviations

| type | evidence |
| --- | --- |
| `syllable_sequence_mismatch` | a produced syllable that is not the one the sequence expected (D6) |
| `truncation` | the train runs to the recording boundary |

**A deviation is not evidence of a bad recording.** See
[`branch-conventions.md`](branch-conventions.md): `syllable_sequence_mismatch` is produced by apraxia
of speech and by dysarthria — it is the finding, not a fault of the recording.

**`off_task_extent` is withdrawn from this branch.** It previously covered *"lexical speech, or
silence, where the task expected sustained repetition"* — which makes every breath pause in a DDK
train a deviation and needs an undeclared minimum duration to avoid firing constantly. **Pause
structure is a measurement**: D3's interval and trend analysis reports it, with locations, and
reports it better.

`filler` does not apply — DDK expects no lexical content. Train duration and fraction are `counts`
entries (D5).

## Quality covariates

D1's modulation spectrum and D2's nucleus detection are both amplitude-envelope measures and are
therefore directly sensitive to AGC, which is the default in consumer capture. Every measurement
carries the quality covariates of its extent, per
[`branch-conventions.md`](branch-conventions.md).

## A branch `FAIL` is an absence of detected content

`DDK: FAIL` means **this branch's detector found no train**, never that the speaker produced none.
A detector keyed on regular repetition fails most often on irregular trains — the presentation the
task exists to detect — so without care `FAIL` would concentrate on the most impaired speakers.
**D1's envelope-first proposal removes most of that cause**: a train with weak modulation structure
is still a train, reported with weak structure. See [`branch-conventions.md`](branch-conventions.md)
and Unresolved on `Outcome.FAIL`'s wording.

## What exists today

| capability | status |
| --- | --- |
| D1 propose and measure the train | **not built**; no module computes an envelope modulation spectrum; unit ambiguity must be resolved in the output |
| D2 nucleus rate | **not built**; `extract_speech_rate` exists, no branch consumes it; four undeclared operating points |
| D3 interval structure | **not built**; nothing computes an inter-onset sequence, its dispersion or its trend |
| D4 segment inventory | **not built**; `extract_ppg_segments` exists but no *branch* calls it — `features.py:421` does, to compute the `ddk.ppg_segment_rate_per_s` gate feature |
| D5 train fraction | **not built** |
| D6 sequence conformance | **not built**; PPG-to-syllable mapping owed |

**The routing side is built and works.** What is missing is everything downstream of the decision.

**What senselab provides**: `extract_ppg_segments` (`ppg.py:349`),
`extract_mean_phoneme_durations` (`ppg.py:429`), `to_frame_major_posteriorgram` (`ppg.py:287`),
`load_ppg_posteriorgram` (`ppg.py:327`), `extract_speech_rate` (`praat_parselmouth.py:103`).

**What is missing**: an envelope modulation-spectrum computation, an inter-onset interval sequence
with dispersion and trend, and a PPG-phoneme-to-syllable mapping.

## What the branch would emit

```
spans        family: "ddk", one per syllable train (D1)
assertions   refine (corrected_extent) where a PREPROCESS span's extent is wrong;
             deviate (syllable_sequence_mismatch, truncation)
measurements harmonic peak structure f and 3f with its stated unit (D1),
             nucleus rates with their four declared operating points (D2),
             interval dispersion and trend, amplitude regularity (D3),
             segment inventory (D4) — each with its support count
counts       train duration and fraction of recording {found}
verdict      { trains_n, train_s, train_fraction, modulation_peak_hz,
               modulation_unit, articulation_rate_praat_nuclei_per_s, interval_cv,
               interval_trend, flags }
```

**The verdict's basis, proposed**: `FAIL` when no syllable train was found; `FLAG` when one
measurement contradicts another — a modulation peak that routed the branch over a region where
nucleus detection finds nothing; `PASS` otherwise. Rates and regularity are carried as measurements
and **no normative judgement enters the verdict**.

**There is no `contest` capability.** An earlier version of this document listed one; no capability
here produces an object for it, and contesting the branch's own proposal has no object. If DDK is to
contest, the object must be a PREPROCESS span.

## Out of scope

Normative interpretation of rate or regularity. Reconciling D1 and D2 into one number. Any refit of
`ddk.lexical_repetition` or `ddk.ppg_segment_rate_per_s` against declared families.

## Unresolved

- **`Outcome.FAIL`'s wording is a hazard** — `no_content_found` would carry the meaning — but
  `Outcome` is a closed vocabulary with readers, so this is recorded rather than changed.
- **D6 reads an expected syllable sequence the contract's declaration does not define.** The
  contract says `instructions` is prose and must not be parsed, and it froze the entry keys.
  `data/task_expectations/` does not exist yet, so the extension is free — but it must be asked for
  explicitly.
- Whether the PPG posteriorgram is usable at all on rapid nonsense syllables.

### SPARC as a shared PREPROCESS derivative (proposed; not decided)

The owner proposes extracting **SPARC** (speech articulatory coding) across the corpus as a shared
PREPROCESS derivative, in the same shape as the PPG and the whole-file diarization that
[`../20260913-branch-contract-and-hints/design.md`](../20260913-branch-contract-and-hints/design.md)
already owes — because articulator trajectories would give DDK rhythm directly rather than by
inference. It is recorded here because DDK is the motivating case. **Nothing below is decided, and
none of it is built.**

It is **not cross-recording**: every product is a per-recording trajectory, so it does not belong in
[`corpus-level-node.md`](corpus-level-node.md). `spk_emb` is the only key that even looks
cross-recording, and nothing here proposes to pool it.

#### What exists in senselab today

`SparcFeatureExtractor` (`audio/tasks/features_extraction/sparc.py:188`) with three classmethods —
`extract_sparc_features` (`:191`), `decode_sparc_features` (`:312`), `convert_voice` (`:398`). Only
the first is relevant. It saves one WAV per input, runs `sparc.load_model(language).encode(waveform)`
(`:64`, `:72`) inside the `sparc` subprocess venv (`:31`, `:44` — Python 3.11, `torch>=2.8,<2.9`) and
returns one dict per audio. `lang` admits only English (`"en+"`) or `None` (`"multi"`); anything else
raises (`:220-225`). Mono only (`:236-237`). With `resample=False` — **the default** — a sampling rate
other than the model's raises after the worker has already run (`:265-271`); `resample=True` re-runs
the whole subprocess on resampled audio (`:274-278`).

**The documented per-recording output**, from `features_extraction/api.py:422-470` — the only place
in the tree where any shape is written down:

| key | shape in that docstring |
| --- | --- |
| `ema` | `(2106, 12)` float32 — frame-major, 12 articulator channels |
| `loudness` | `(2107, 1)` |
| `pitch` | `(2107, 1)` |
| `periodicity` | `(2107, 1)` |
| `pitch_stats` | `(2,)` |
| `spk_emb` | `(64,)` |
| `ft_len` | `2107`, a plain int |

Four qualifications on that table, because it is a docstring rather than an assertion:

1. **`ema` is one frame shorter than the other three tracks**, and `ft_len` matches the longer set.
   Any per-frame alignment of articulation against pitch or loudness inherits that off-by-one.
2. **The key set is not enforced.** The worker serialises whatever `coder.encode` returned
   (`sparc.py:75-85`) and the parent rebuilds it key-by-key (`:302-307`); only the *failure* path
   names the seven keys explicitly. The two tests that inspect the dict assert key presence only, never a shape
   (`src/tests/audio/tasks/features_extraction_test.py:547-549`, `:561-563`).
3. **The registry disagrees with the docstring on the channel count** — `14 (EMA)` at
   `src/senselab/model_registry.yaml:562` and `model_registry.md:113` against 12 above.
4. **Channel semantics are recorded nowhere in `src/`.** The only naming in the tree is
   `CHANNEL_NAMES` in `tutorials/audio/speech_representations_lab.ipynb` — `TT`, `TB`, `TD`, `UL`,
   `LL`, `LI` in x and y — and that notebook also transposes `ema` by a shape heuristic because it
   does not know the orientation either.

**Triage uses it nowhere.** `grep -rni sparc src/senselab/audio/workflows/ scripts/` returns nothing.
Its only in-package caller is `extract_features_from_audios(sparc=True)`
(`features_extraction/api.py:567-568`), plus that tutorial.

#### Failure is a NaN, not a typed absence — and that is the gap against the PPG

When `coder.encode` raises for one recording, the parent appends a dict carrying **0-dimensional NaN
tensors under all seven keys**, with `ft_len` a bare `float('nan')` (`sparc.py:289-300`). A consumer
that does not test `torch.isnan` or `.dim()` cannot tell a failure from a measurement.

The PPG does the opposite: `PpgsPosteriorgramUnavailable` (`ppg.py:141-142`) with
`require_posteriorgram` to unwrap it (`:167-181`), which is what lets an absence be attributed as
one — [`dag.md`](dag.md):157 records it reaching `extend.attempt_derivation`. **A corpus-wide SPARC
derivative needs the PPG's typed absence before it needs anything else**, or absences become
NaN-shaped measurements. Per [`branch-conventions.md`](branch-conventions.md) and this document's own
rule, an absence is never a negative.

#### No revision is pinned, and no guard catches that

`load_model(language, device=device)` (`sparc.py:64`) passes no revision, and the worker's payload
carries no revision-shaped key (`:243-250`). `src/tests/utils/revision_pinning_guard_test.py` sweeps
for exactly that key, so **`features_extraction/sparc.py` is invisible to it** — and it appears in
neither `REVISION_RESOLVED_SUBPROCESS_FILES` nor `LOADER_CANNOT_PIN_SUBPROCESS_FILES`. That test's
docstring says every subprocess backend lands in exactly one list, but
`test_the_two_revision_allowlists_are_disjoint_and_current` (`:370-389`) checks only disjointness and
that the named files still exist; **it does not check coverage.** `features_extraction/ppg.py` is
absent the same way — but for the PPG that is the honest answer, because ppgs ships its checkpoint in
the PyPI release and there is no commit to resolve ([`dag.md`](dag.md):496-498). **SPARC is not in
that position**: `specs/20260819-091500-wav-subtype-sweep/design.md:69` names a Hugging Face repo,
`cheoljun95/Speech-Articulatory-Coding`, so a commit exists to resolve and none is recorded. Nothing
today *claims* a SHA, which per CLAUDE.md is the one outcome worse than recording none — but a corpus
pass would publish provenance naming no commit at all.

#### Three operational gaps against the PPG's own extension, all small

- **No `ensure_sparc_venv`.** The PPG has one (`ppg.py:145-154`) whose docstring exists precisely to
  say *call it once before fanning a batch out across an array*, because a cold build outlasts the
  lock's patience window. SPARC calls `ensure_venv` inline at three sites (`sparc.py:227`, `:351`,
  `:435`), so a Slurm array would race the build — the trap CLAUDE.md records under the venv lock.
- **The worker timeout is a flat `timeout=600`** on all three paths (`sparc.py:257`, `:382`, `:462`),
  where the PPG's is duration-proportional — `WORKER_STARTUP_S + 2.0 × audio_seconds`
  (`ppg.py:184-200`, applied at `:236` and `:264`).
- **No availability probe.** `ppgs_venv_available` exists (`ppg.py:164`); SPARC has no equivalent.
- `utils/compatibility.py` marks the PPG entry `gpu_required=True` (`:159`) and the SPARC entry not
  at all (`:166-171`), which is a claim about SPARC that nothing in this tree measured.

**The host constraint is the PPG's, exactly.** In
`specs/20260908-120042-cuda-wheel-availability-matrix/design.md:76-77` the `sparc` and `ppgs` rows are
identical cell for cell — same `torch>=2.8,<2.9`, same `no wheel` on cu121, cu124 and cu130. So a
SPARC pass adds no new CUDA-host problem beyond the one the PPG pass already has. The two venvs are
nonetheless **separate trees**: `sparc.py:31` names `"sparc"`, `ppg.py:87` names `"ppgs"`, so
`sparc.py`'s module docstring claim of a venv *"shared with ppgs where possible"* is not what the code
does. The sharing that does hold is with `voice_cloning/sparc.py:27-40`, whose venv name, requirement
list and Python version are byte-identical, so the two cannot thrash each other's install.

#### 1. Why EMA would be the direct instrument, not another cross-check

`/pa-ta-ka/` is a labial, then alveolar, then velar closure — this document already says so twice, at
`:150-151` and `:298`. EMA channels track those articulators: a labial closure is a lip-aperture
minimum, an alveolar closure a tongue-tip maximum, a velar closure a tongue-dorsum maximum, and the
trajectory carries all three in one place with the order built in.

Every DDK capability here infers rhythm from something else, and each inherits a defect recorded
above:

- **D1** proposes the train from an amplitude envelope. It explicitly *rejects* PREPROCESS's stored
  `energy_envelope` — dBFS, pre-emphasised, 40 Hz-lowpassed — for three reasons at `:104-117`, and
  requires a linear-amplitude envelope on `plain` instead. What it cannot escape is that **three of
  its five conventions are owed** (`:159-171`), including the search band, which it calls decisive;
  and the one convention in force, `envelope.lowpass_hz: 40.0`, was derived on the dB pre-emphasised
  envelope and would be inherited across a condition change.
- **D2**'s nucleus detector has a **minimum sounding interval of 0.1 s against Praat's 0.05 s**, which
  at 6–7 syll/s sits on the measurand (`:214-216`, `:253-258`), biasing merged syllables *toward* the
  cycle rate — the same 3× confusion D1's unit disambiguation exists to resolve.
- **The PPG is out of domain on rapid nonsense syllables**, which is this document's closing
  Unresolved item and D6's Owed.

EMA would **measure the closures rather than infer them**, which is what makes it a candidate
*primary* instrument for D1 rather than a fourth cross-check. It does not rescue the unit problem for
free — a closure sequence still has to be segmented — but a labial-alveolar-velar *ordering* in the
trajectories is an unambiguous cycle marker in a way an envelope peak is not, because the three
closures are distinguishable by channel and not only by amplitude.

#### 2. The stream question is the PPG's question again — and the PPG's answer is `enhanced`

**This entry previously said the opposite, and the framing it carried was withdrawn by the owner on
2026-09-14.** It read that plan 1 moves the posteriorgram **off `enhanced` onto `plain`** on D1's
reasoning, and that SPARC used for rhythm has identical exposure because *the transients whose timing
is the measurement are the ones an enhancer can smear*. Both halves were wrong.

**The posteriorgram does not move.** Audit step 1b is withdrawn
([`praat-instrument-audit.md`](praat-instrument-audit.md), step 1b); plan 1's Task 5 moves the Praat
scalars alone. The reason is the one that matters here: **a trained model reading the stream closest
to its training domain is the defensible default**, and many recordings in this corpus carry
background noise a phoneme classifier would handle worse on `plain`. FRCRN's out-of-domain risk on a
DDK train is real, but nothing in this tree has measured transient smearing — that borrowed argument
belongs to **D1's amplitude envelope**, where peak timing genuinely *is* the measurement, and D1
already reads `plain` for its own reasons (`:95-101`).

**The consequence for SPARC runs the other way from what this entry used to claim.** SPARC is
likewise a trained model, so **the same argument makes `enhanced` its defensible default too** — which
*weakens* the case for a raw-versus-enhanced pilot rather than strengthening it. The owner's proposal
already says `enhanced`; that is now the reasoned default, not an unexamined one.

**The pilot stays owed, narrowed to the DDK-rhythm use.** The training-domain argument settles what
the default is; it does not establish that a denoiser preserves inter-onset *timing*, which is the
one thing a rhythm derivative reads and the one thing no benchmark here measures — the four FRCRN
benchmarks measure which non-speech events survive, not when they occur. That is a narrow,
answerable question and it is what the pilot is for. The structurally identical whole-file
diarization decision has the same contract requirement
(`../20260913-branch-contract-and-hints/design.md:664-667`, `:740`, `:798`), but **the default there
is undecided and here it is not.** A pilot run on DDK material does not transfer to the VOICE use in
(4), which is sustained phonation — two populations, so two pilots, or one pilot whose scope is
stated.

#### 3. Place-class grouping refines D6 — and does not need SPARC

**The brief for this entry attributed the /pa-ta-ka/ collapse to D4; it is D6.** D4 is the segment
inventory over the train; D6 is sequence conformance, and it is D6 that names `/pa-ta-pa/` as a
substitution and `/pa-pa-pa/` as a sequential task collapsed into an alternating one (`:294-310`).
D4 supplies the segments D6 would read.

D6's standing objection is a domain mismatch, not a per-phoneme error rate: whether a given syllable
*"reliably surfaces as a particular argmax phoneme is unmeasured, and the posteriorgram is out of
domain on rapid nonsense repetition"* (`:308-310`). The owner's sharper version — that reduced velar
/k/ is the most frequently weakened and mislabelled segment, so an automatic finding would largely be
recogniser error — is the right shape of worry but **is not measured anywhere in this tree**, and is
recorded here as a hypothesis, not as a citation.

**Grouping by place of articulation is robust to within-class confusion**, which is the argument for
it: if /k/'s probability mass leaks to /g/ or /ng/, a labial/alveolar/velar partition absorbs the leak
because all three are velar, while an exact-identity argmax reports a substitution. That is
structural, not empirical, and it is what would make the collapse detectable at all.

**It needs no SPARC.** The posteriorgram already carries the full distribution:
`load_ppg_posteriorgram` returns `(frames, phonemes)` float32 (`ppg.py:327-346`), the sidecar and the
measurement both carry the phoneme order (`preprocess.py:800`, `:816`), and the inventory is 40
ARPAbet labels plus `<silent>` (`ppg.py:42-81`). **Sum the class members and then take the argmax**,
which is strictly more robust than argmax-then-group: the latter can pick a phoneme whose own mass is
a minority of its class's. `extract_ppg_segments` (`ppg.py:349`) is the existing argmax-then-segment
path and does not do this; the class-summed variant is a different reduction over the same array.

Record it as **a refinement of D6, not a new capability**. What it adds to D6's Owed is narrower than
what is there now: a phoneme-to-place-class table rather than a phoneme-to-syllable mapping. That
table is a lookup, not a fitted operating point, so under CLAUDE.md it belongs in `data/` with its
source written beside it and needs no corpus measurement to exist. What it does **not** resolve is
whether the posteriorgram is in domain on rapid nonsense syllables at all — grouping cannot repair a
posteriorgram that is confidently wrong about *place*, only one confused *within* a place.

#### 4. An independent F0 estimate for VOICE

SPARC returns `pitch` and `periodicity` per frame, and its venv pins `torchcrepe==0.0.23` and
`penn==0.0.14` (`sparc.py:40-41`) — so those come from a neural tracker in that family, not from
Praat's autocorrelation. That makes it a genuinely independent second estimate rather than a second
reading of the same method.

[`branch-voice.md`](branch-voice.md):401-412 records that **no type-2 instrument works today**: the
period-length modality test provably fails in the case it was written for (a tracker locked to the
subharmonic gives a distribution unimodal at 2T), the octave-jump count cannot separate period
doubling from the tracker's own octave error on the same material, the subharmonic-to-harmonic ratio
is absent from the inventory, and Sample C's two-floor substitute is confounded. **Two independent
trackers disagreeing by an octave is evidence a single tracker cannot produce**, which is why this is
worth recording rather than dismissing.

Record it as **a candidate owed a measurement, not as a solution.** Four reasons it is not one yet:
disagreement localises a problem without typing it, so an octave disagreement is not by itself a
type-2 finding; SPARC's own tracker has its own octave errors and nothing here measures their rate;
`periodicity` is not a subharmonic-to-harmonic ratio and must not be read as one; and any
disagreement criterion is an operating point, which the corpus may not be used to fit.
Cross-referenced from [`branch-voice.md`](branch-voice.md)'s Unresolved.

#### 5. Cost, and what is owed before any of it is built

**A model pass over the whole corpus in a subprocess venv** — the same cost statement the diarization
derivative carries, *"a model pass over the corpus, comparable to the PPG extension, delivered as an
extend driver"* (`../20260913-branch-contract-and-hints/design.md:670-672`). It is an **extend driver
against finished stores**, under the same contract as the original pass, and it would land as a
`derivatives/` sidecar with the path, SHA-256, size and shape on the entity and never the array —
the shape [`dag.md`](dag.md):490-500 records for `ppg_posteriorgram`. No wall-clock or GPU-hour figure
for a SPARC pass exists in this tree, and none is invented here.

**Owed before any of it is built:**

1. **The stream decision** (2) — `enhanced` is the reasoned default on the training-domain argument, so
   what is owed is narrower than a stream choice: a pilot on the DDK-rhythm population asking whether
   the denoiser preserves inter-onset **timing**, which nothing in this tree measures.
2. **Whether EMA resolves the measurand.** The only statement of SPARC's frame rate anywhere in the
   tree is a comment in a tutorial cell (`speech_representations_lab.ipynb`, *"SPARC runs at 50 Hz"*);
   nothing in `src/` asserts it and no test pins it. If it holds, a 20 ms frame against the 140–170 ms
   syllable period this document cites at 6–7 syll/s (`:254-256`) leaves single-digit frames per
   syllable — arithmetic on two in-tree numbers, conditional on an unverified third, and **not a
   measurement**. Whether that resolves a closure, and whether the channel set is 12 or 14, are both
   owed.
3. **Whether SPARC is in domain on non-lexical nonsense syllables at all** — the same question D6 and
   this document's closing Unresolved item ask of the PPG, unanswered for either model, and the
   proposal's weakest point: a model trained on speech has no guarantee on `/puh-tuh-kuh/`.
4. **The typed absence, the named venv builder and the proportional timeout** — the gaps above, each
   cheap, each a prerequisite for a corpus pass rather than a follow-up.
5. **No threshold may be fitted against this corpus.** Any rhythm criterion built on EMA arrives
   parameter-free — a dispersion, a trend, an ordering — or arrives owed, exactly as D3's do. A
   closure detector needs a threshold on a trajectory, and where that value comes from is the question
   to answer before the pass, not after it.
