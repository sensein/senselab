# AIRWAY branch

What the branch answers: **is there airway content in this recording, where is it, and what kind?**

The frame is [`../20260913-branch-contract-and-hints/design.md`](../20260913-branch-contract-and-hints/design.md).
Conventions shared by all five branch documents — span family spelling, the `propose`/`refine` rule,
how deviations and counts are stored — are in [`branch-conventions.md`](branch-conventions.md).
Ground truth owed across all of them is in [`branch-listening-sample.md`](branch-listening-sample.md).

## The tasks this branch serves

Declared families are what the protocol says a recording is, not what a listener confirmed. Counts
are over the 62,547 recordings in `runs/ruleset-score-20260912/ruleset_score.json`.

| family | n | what the task asks for |
| --- | --- | --- |
| `respiration-and-cough-fivebreaths` | 3,576 | five respiratory cycles |
| `respiration-and-cough-breath` | 1,788 | breathing |
| `respiration-and-cough-cough` | 1,788 | a cough |
| `respiration-and-cough-threequickbreaths` | 1,718 | three rapid cycles |
| `respiration-and-cough-v2-breath` | 699 | breathing |
| `respiration-and-cough-v2-threebreaths` | 699 | three cycles |
| `respiration-and-cough-v2-threebreathsmouth` | 699 | three cycles, oral |
| `respiration-and-cough-v2-threebreathsnose` | 699 | three cycles, nasal |
| `respiration-and-cough-v2-hardcough` | 698 | a forceful cough |

These nine sum to 12,364 against 13,017 declared AIRWAY. The residual 653 is `breath-sounds` and
`voluntary-cough`, both in `AIRWAY_ELICITING` (`families.py`) and neither present in the corpus
profile above a count this table carries.

**Two task shapes wanting different measurements.** A **cough** task asks for one or a few forced
expulsive events; the clinical question is about the event. A **breath** task asks for a *count* of
respiratory cycles and the question is about the cycle — how many, how regular, through which route.
For `threequickbreaths` the *interval* is the measurement: a count of three says nothing about
whether they were quick.

### A recording routed here whose declared task is not airway

AIRWAY routed 23,606 recordings where 13,017 declare an airway family. A cough inside a Harvard
sentence reading is a real cough and is recorded as one, with no penalty for the recording not
matching its label. Whether that cough bears on the *sentence* task is a question about the sentence
declaration, and the contract gives every branch the declaration — so AIRWAY could ask it. It does
not, because the finding it would produce is about routing-versus-declaration across the whole file
rather than about airway content, and that belongs at VERDICT (see Unresolved).

### The number that governs this branch

**AIRWAY's gate evidence is `unavailable` on 56,505 of 62,547 recordings** — against SPEECH 0, VOICE
29, DDK 2,345 (`ruleset_score.json`, `totals.unavailable`). It is the largest unavailability in the
graph by a factor of 24, and it bears directly on the contract's finding that `airway.cough` becomes
structurally unable to fire on the class it names once covering-window YAMNet labels are ineligible.
Any capability here must treat an unavailable measurement as an absence, never as a negative.

## Capabilities

### A1 — Label an airway event (**built**)

**Question.** Does this span carry cough or breath?

**Reads.** PREPROCESS's general spans (`airway.py:198`) and its per-span HeAR measurements, bucketed
by the `span_id` attribute (`airway.py:241-245`). A span overlapped by a lexical consensus word is
skipped as transcribed content (`airway.py:253`; `_is_transcribed` at `:69-84`); a span overlapped
only by bracketed words stays eligible.

**Computes.** The intersection of the span's HeAR labels with `airway.labels_of_interest`.

**Emits.** One `label` assertion per member, `wasDerivedFrom` the span (`airway.py:265-277`).

**Derived, and the exemption holds.** `airway.labels_of_interest` ships `[Cough, Breathe]`
(`default.yaml:136`) with a written derivation at `config-derivations.md:555-557`: the two are drawn
from HeAR's eight labels, the confirmation map `Cough -> {Cough}`, `Breathe -> {Breathing, Sigh,
Gasp}` is recorded beside them, and the entry states in as many words — **"Vocabulary, not
thresholds."**

A round of review had me withdraw that exemption and mark the key owed, on the grounds that "a label
set, not a threshold" is not one of the no-refits rule's three outs. It is: a value with a recorded
derivation is the second out, and this one has had it all along. The withdrawal was made by asserting
an absence without checking `config-derivations.md`, and it is restored here.

**A3's contest set is a different case** and stays owed — not because it is the same kind of object,
but because `config-derivations.md:566` records it as declared-but-underived while this one records
the derivation.

**And the two-label vocabulary erases clinically distinct events.** Under `[Cough, Breathe]` a
musical breath noise surfaces as `Breathe` and a throat clear as `Cough`. Tonal breath noise, gasp
and throat clearing should be labels in their own right — a vocabulary decision, not a threshold, and
therefore available now.

**Do not carry the classifier's label `Wheeze` through to output without naming the recording
site.** *Wheeze* is a term of art in auscultation: a continuous musical **lung** sound of at least
~100 ms, conventionally heard with a stethoscope on the chest. Mouth- and trachea-recorded
forced-expiratory wheeze detection is an established method, so a phone microphone at mouth level is
not simply the wrong instrument — but it is a **different one**, and what a general-purpose audio
classifier labels "Wheeze" on this material may equally be upper-airway turbulence. A clinician
reading "wheeze detected" with no site named imports a chest finding nobody made. **The term must
never appear without its recording site**, or use an acoustic name — *musical/tonal breath noise* —
instead.

### A2 — Corroborate against AudioSet (**built**)

**Reads.** PREPROCESS's whole-file `yamnet_window` measurements overlapping the HeAR window the
label came from (`_windows_covering`, `airway.py:51-66`, called at `:295`). The corroboration set is
each HeAR label's mapped AudioSet node and its descendants, from the classifier-ontology profile
(`_corroboration`, `airway.py:99-125`).

**Emits.** A corroboration assertion when an overlapping YAMNet label is in the closure, a `contest`
when it is in `airway.contest_labels`, and an `abstain` carrying `colocated_windows_n` when neither
fired (`airway.py:303-341`).

**The corroboration verb changes from `confirm` to `label`.** The contract's verb set is `label`,
`contest`, `refine`, `trim`, `propose`, plus `abstain` and `flag` keeping their meanings.
`confirm` is on neither list, and the contract's piece 7 widens REPORT's assertion read **by verb** —
so AIRWAY's corroborations would go invisible in the very change designed to make branch assertions
visible. The migration is a one-line change at `airway.py:303` — **but only because `confirm` and `contest`
share the tuple on that line**, `(("confirm", confirms), ("contest", contests))`, so a naive edit
touches both verbs. The resulting assertion is distinguished from A1's by its `yamnet_labels` and
`yamnet_window_ids` attributes.

### A3 — Contest a label (**gated behind null config; structurally dead**)

`airway.contest_labels` is `null` (`default.yaml:138`), so `_contest_labels` returns the empty set
(`airway.py:150`) and no YAMNet label can contest. `contested_n` is structurally zero and the flag at
`airway.py:323-326` is unreachable.

**Replace it rather than wait for it.** A declared contest set cannot be fitted — a set fitted
against declared families encodes which labels co-occur with the declaration, not which deny a cough.
The contract's threshold-free definition needs no list: AIRWAY contests a span for which an airway
label was proposed and no evidence of any kind survives within its extent.

**Its firing rate is unmeasured and must be counted before it ships.** The contract's part (a)
enlarges the set of spans with no admissible evidence, so a definition keyed on "no evidence" could
move from firing never to firing almost always.

### A4 — Off-task content during an airway task (**built as a file-level flag; needs two fixes**)

**Question.** Did the participant produce content the airway task did not ask for, and where?

**Reads today.** Every live lexical word against the hull of the airway-labelled spans
(`airway.py:343-373`), emitting one `flag` assertion with `reason: "lexical_contamination"`.

**Fix one — it must be conditioned on the declaration.** The check is guarded only by
`if labels_by_span:` (`airway.py:345`), so it runs on every recording AIRWAY routes. On a Harvard
reading with a cough near the start and a breath near the end, the hull spans the whole recording
and **every word between them is contamination** — on a branch that routes 23,606 recordings of
which 13,017 declare airway. That is exactly the penalty this document's own off-declaration section
promises never to impose. Condition it on the declaration's `expected_content` naming an airway task.

**Fix two — it becomes a deviation, and that removes AIRWAY's only reachable FLAG.** A lexical word
inside a breath task is a located observation, so it is an `off_task_extent` deviation, one per
contaminating extent rather than one file-level flag. With A3 dead, `lexical_contamination` is the
only flag AIRWAY can currently raise, so after the migration **AIRWAY has no FLAG path at all** until
A3's replacement lands. The verdict section below states the consequence rather than leaving the emit
block and the verdict disagreeing.

**This is the one correct `off_task_extent` in the five documents.** It keys on positively-identified
off-task content that has its own extent. The versions previously in SPEECH, VOICE and DDK keyed on
the *absence* of the target and were withdrawn.

**And the lexical channel is the wrong instrument on its own.** Humming, laughter and speech-like
voicing are off-task content that produces no lexical word, and on a noisy breath recording an ASR
transcribes little. Voicing detection from the F0 track (`phonation_tracks`, `preprocess.py:919`)
catches them and is available now.

**The voicing channel needs the same exclusion the lexical one has.** A cough has a voiced phase —
which A6 measures as an ordinary cough descriptor — and voiced exhalation is normal in several breath
families. A bare voicing detector would flag both as off-task. **An extent already carrying an airway
label is not off-task**, exactly as a span overlapping only bracketed words is not transcribed
content.

### A5 — Respiratory cycle count (**not built**)

**Question.** How many respiratory cycles are there, and how long is each?

This is what the breath families need and what the branch most conspicuously lacks. `labelled_n`
counts *spans carrying a label*, not breaths: PREPROCESS merges adjacent proposals
(`merged_proposals`, `airway.py:262-263`), so one span can cover several cycles.

**Invert the architecture.** Classifier windows are on the order of a second and a cough is
300–500 ms, so counting events from window labels is structurally limited. Segment candidate events
from the energy envelope and use the classifiers to *type* them, rather than reading counts off
labels. A5 is where that inversion starts, and it should be the general principle for this branch.

**Which phase is the primary event is task-dependent.** Quiet nasal inhalation is frequently below
the noise floor on consumer capture, so an inhale-based count biases downward — but
exhalation-primary is not the universal answer either:

- on `v2-threebreathsnose` **both** phases are near-silent, so exhalation-primary rescues nothing;
- on `fivebreaths` and `threequickbreaths` a deep mouth inhalation is often the **louder** event, so
  exhalation-primary biases downward there instead.

The primary-event choice is therefore per-family, driven by the declaration, and stated per family
rather than fixed once.

**Name the outputs acoustically, not spirometrically.** An acoustic breath event is not a respiratory
cycle. "Cycle duration" and "I:E ratio" import meaning from spirometry onto what are breath-event
durations and the intervals between them — measured at a phone microphone, with no airflow
measurement anywhere. Emit **breath-event duration** and **inter-event interval**.

**And the I:E ratio cannot be computed by the detector this capability specifies.** The justification
for exhalation-primary counting is that inhalation is frequently undetectable — in which case I is
not measurable, and a ratio requires both. **Condition any inspiratory-to-expiratory measure on
inhalation actually being detected, and mark it `unavailable` otherwise**, per the branch's own
absence rule.

**Emits.** `propose` spans for events PREPROCESS did not find, `family: "airway"`, subject to the
`propose`/`refine` rule in [`branch-conventions.md`](branch-conventions.md); a per-span measurement
carrying breath-event durations and inter-event intervals, with the I:E measure present only where
both phases were detected; and a `counts` entry `expected_event_count` carrying `found` and
`declared`.

**Owed — the operating points, not only the validation.** Envelope smoothing window, peak/trough
criterion, and minimum breath-event duration are all required for the capability to execute and none
exists. These have no config key and no entry in `config-derivations.md` because **the capability
itself does not exist yet** — unlike `spans.k_db`, which is derived at `config-derivations.md:74`,
`:133` and `:238` for a proposer that does. An earlier version marked only the validation owed, which
was insufficient: without the detection parameters the capability cannot run at all.

### A6 — Cough event descriptors (**not built**)

**Question.** What kind of cough was it?

Duration, rise time, presence of a voiced phase, and spectral distribution over a cough-labelled
event. All are available from the envelope and from Praat's spectral machinery
(`extract_spectral_moments`, `praat_parselmouth.py:945`), all are non-normative descriptions, and
none is currently computed.

**Two instrument problems land here, both from
[`praat-instrument-audit.md`](praat-instrument-audit.md).** The spectral moments are silently
band-limited to **5 kHz** (finding 7) — and a cough is the most broadband event in this corpus, so
the band excludes much of what distinguishes one cough from another. And the scalars are computed on
the **FRCRN-enhanced** stream (finding 0), which is out of domain on a cough: a denoiser trained to
reconstruct speech from noise has no defined behaviour on a forced expulsive event. Serves `respiration-and-cough-cough` (1,788) and `v2-hardcough` (698).

**Cough counting is undefined, and the choice changes the count two- to threefold.** A bout of three
coughs on one expiration is one cough epoch or three cough events, and nothing here says which.
Across the 2,486 cough-declaring recordings that is the difference between two incompatible measures
sharing a name. **Report both** — epochs and events, each named for what it counts — or declare one
convention explicitly. Do not emit an unqualified "cough count".

**Effort stays out of scope** — `hardcough` asks for it and nothing here measures it.

### A7 — Nasal versus oral route (**not built; may not be measurable**)

`v2-threebreathsmouth` and `v2-threebreathsnose` (699 each) differ only in route. Nasal breathing is
lower-amplitude with reduced high-frequency energy, and the spectral machinery exists. **Whether it
separates these two families is unmeasured**, and measuring it against the declared families would
fit the declaration.

## Deviations

| type | evidence |
| --- | --- |
| `off_task_extent` | positively-identified off-task content inside an airway task — a lexical word, or voiced production from the F0 track outside any airway-labelled extent — each with its own extent (A4) |

**A deviation is not evidence of a bad recording.** See
[`branch-conventions.md`](branch-conventions.md).

`expected_event_count` is a `counts` entry, not a deviation (A5). AIRWAY emits no
`stimulus_mismatch` and no `filler`: no airway task carries a stimulus text.

## Quality covariates

Every acoustic measurement this branch emits carries the covariates of its own extent — clipping,
SNR, support count — plus the **file-level** ones referenced rather than recomputed per span:
effective bandwidth, and any AGC or noise-suppression signature. See
[`branch-conventions.md`](branch-conventions.md), which draws that split; an earlier version of this
section required all four per extent and was the one document not updated when the split was made.

A spectral descriptor computed over a clipped or noise-suppressed span is not a measurement of the
airway.

## What exists today

| capability | status |
| --- | --- |
| A1 label | built; `labels_of_interest` owed a derivation; vocabulary too narrow |
| A2 corroborate | built; `confirm` → `label` migration owed |
| A3 contest | gated behind null config, structurally dead; replace rather than fit |
| A4 off-task content | built as an unconditional file-level flag; two fixes owed |
| A5 cycle count | not built; detection operating points owed |
| A6 cough descriptors | not built |
| A7 route | not built; separability unmeasured |

**The branch runs no model.** It reads PREPROCESS's `span_hear` and `yamnet_window` measurements
(`airway.py:166-176`).

**There is no airway-specific span gate.** `airway.k_db`, `airway.k_db_by_task` and
`airway.k_margin_db` are retired — none is in `default.yaml` and `airway.py` reads no `k_db`. A prior
version of this document described all three as live and claimed the branch re-runs HeAR per span;
both were wrong.

**AIRWAY's selector must widen to see its own proposals.** `airway.py:198` selects `family is None`,
so A5's cycle spans — carrying `family: "airway"` — would be invisible to A1, A2, A4 and
`_windows_covering`. The selector becomes `family is None or family == "airway"`.

**A gap span can carry the whole verdict.** Gap spans carry no `family` key, so they are full members
of this branch's evidence set, and `labelled_n` separates `pass` from `fail`
(`airway.py:376-383`, detail at `:393-398`). How often that decides a verdict is unmeasured
(`dag.md`). [`branch-quality.md`](branch-quality.md) says what concludes on them if the contract's
part (b) types them background.

## A branch `FAIL` is an absence of detected content

`AIRWAY: FAIL` means **no span carried a label of interest**, never that the recording lacks airway
content. With gate evidence `unavailable` on 56,505 of 62,547 recordings, that distinction carries
more weight here than anywhere else in the graph. See
[`branch-conventions.md`](branch-conventions.md).

## What the branch emits

```
spans        A5 would propose family: "airway" cycle spans
assertions   label (A1), label with yamnet_* attributes (A2 corroboration),
             contest (A3, once replaced), abstain, deviate/off_task_extent (A4)
interval     airway_labelled_interval, the hull of the labelled spans
measurements A5 breath-event durations and inter-event intervals, with the
             inspiratory:expiratory duration measure present only where both
             phases were detected; A6 cough descriptors — each with its
             extent's covariates and support count
counts       expected_event_count {found, declared} (A5)
verdict      { labelled_n, by_label, contested_n, merged_n, flags }
```

**The verdict's basis, exactly** (`airway.py:376-383`, detail at `:393-398`):

- `FAIL` when no span was proposed at all or PREPROCESS reported `no_contrast`
  (`airway.py:203-211`), or when spans exist and none carries a label of interest.
- `FLAG` when any flag accumulated.
- `PASS` otherwise.

**After A4's migration the FLAG path is empty** until A3's replacement lands, because
`lexical_contamination` is currently the only reachable flag. The verdict becomes `FAIL` or `PASS`
only, and the document says so rather than listing a flag that cannot fire.

## Out of scope

Cough effort. Severity. Any conclusion about a kind that is not airway. Any refit of `spans.k_db`.

## Unresolved

- **`task_content_mismatch` belongs at VERDICT**, not here. AIRWAY routes 23,606 against 13,017
  declared; VOICE 22,277 against 8,306; DDK 22,363 against 7,989. No branch can claim the finding,
  because each holds only its own question — so it should be recorded at VERDICT as a described
  observation with routing and declaration side by side, asserting nothing about which is right.
- Whether A7 is measurable at all.
- The count of AIRWAY verdicts resting on a gap span.
