# Branch — voice / no-words. The residual, and the two tracks that carry it

Drafted 2026-08-19. Branch node for the third kind in [`taxonomy.md`](taxonomy.md), reached when the
residual's acoustic gate admits voicing that neither airway nor speech claimed. `admit.md` and
`taxonomy.md` govern upstream of here; `design.md` are stale and are
not a source of structure for this file.

Members: sustained vowel, pitch glide, loud phonation, maximum phonation time, laughter, crying, vocal
imitation of a non-vocal target.

## What it decides, and what it does not

**It measures.** The product is pitch periods, per-period amplitudes, and vocalization spans. It is not
a classification of the recording into the members above, and the branch does not own a label space —
`Human voice`, `Human sounds` and `Respiratory sounds` are all absent from YAMNet's 521 labels, so no
classifier in the screening set could name a member even if one were run here, and none is.

Two of the seven members are not acoustic classes at all and cannot be labels:

| member | what it actually is |
| --- | --- |
| maximum phonation time | the **duration** of a sustained-vowel span under a named offset criterion. A task, and its measurement. |
| loud phonation | a **contrast between two spans** in the same recording — `adult.loudness.v2` asks for "hey" spoken normally and then loudly (D9, D12). Not a property of one span. |

`taxonomy.md`'s residual feature table lists maximum phonation time as a member row whose
distinguishing feature is "the duration of the voiced run is itself the measurement". That row is a
measurement, not a discriminator, and this branch treats it as one. Its kinds table omits the member
entirely, which is the consistent reading.

## Signature

```
voice(audio, residual_windows, hint?)
  -> fail(reason)
   | flag(reason, partial)
   | pass(voicing)
```

| port | direction | kind | type | meaning |
| --- | --- | --- | --- | --- |
| `audio` | in | data | decoded audio | from ADMIT, **the whole recording as supplied** |
| `residual_windows` | in | data | time regions | the gate's voiced windows that neither other kind claimed |
| `hint` | in | param | `AudioHints` or absent | optional, per D5; reaches two nodes only, see below |
| `fail` | out | reason | — | the instrument cannot measure this recording |
| `flag` | out | reason, partial | — | a judgement that could have gone either way; the partial product travels with it |
| `pass` | out | `voicing` | product record | period marks, tracks, spans, features, proposals |

`audio` is the whole recording and not the residual's regions, and that is forced rather than
convenient: **loud phonation is defined by energy relative to the rest of the recording**, so a node
handed only the admitted regions could not compute the reference it is measured against.
`residual_windows` therefore *selects* which voiced runs are this branch's product; it never bounds
what is analysed.

### `residual_windows` has no producer, and that is the F-187 shape

`taxonomy.md` declares exactly one output port, `kinds`, one presence `Estimate` per kind. Its residual
section also says the residual "is the one place grids must be compared" — the gate's voiced windows
checked for time overlap against the airway and speech detectors' confident windows. That comparison is
computed there and appears on no declared port, so the input this branch needs is produced by nothing.
an earlier airway draft caught four dangling ports of exactly this shape in its own table
(`crisper_tokens`, `c50_db`, `rms_db`, `community1_seg`), and it is the reason ports are declared before
code exists.

Two ways to close it, and this file picks the first:

1. **TAXONOMY grows the port.** The comparison already runs there. Recomputing it here would be a second
   copy of one decision, parameterised separately, which `ports.md` rule 5 forbids in spirit — one
   producer per product name.
2. The branch recomputes the overlap from sibling branch spans. Rejected: it makes the voice branch
   downstream of the airway and speech branches, and `ports.md` rule 7 — absence is not a value, a task
   whose input port has no product does not run — then stops this branch dead on every recording where
   airway is absent. A branch that cannot run when a *different* kind is absent is wired wrong.

The product is empty, not missing, when nothing was claimed. An empty `residual_windows` is a
well-formed input meaning "the whole recording is the residual's to look at"; it is distinct from no
product at all, which does not run the branch.

## The two tracks, and why nothing else enters

Everything in this branch derives from an **energy** track and a **periodicity / F0** track. No
classifier, no ASR, no enhancer, no speaker embedder. Each exclusion is measured, not stylistic:

| instrument | why it is absent from this branch |
| --- | --- |
| YAMNet, AST | no label in the 521 names any member; the only available construction is a union of specific labels, and none of the members has one. |
| HeAR | a box-car presence gate: 40 ms of cough inside its 2 s window — 2% — saturates the posterior, so it cannot locate, bound or count. And it is **amplitude-invariant** — gains ×0.1 to ×10 give cosine 1.0000 — so loud phonation is invisible to it by construction. |
| CrisperWhisper / any ASR | it imposes a speech prior on non-speech: it labelled a voiced cough phase `[UH]`. On a sustained vowel or a glide it would return words that were never produced, and word-gated paths null wordless material outright (F-165, D9, D10). |
| enhancement | D19 makes enhancement an operation a node may invoke; this branch invokes none. DriftSE v1 hallucinated `[laughter] [laughter]` at an input-output correlation of 0.204 — an enhancer inventing this branch's own vocabulary. `MossFormer2_SE_48K` takes breath to −39 to −45 dB and the repo default takes breath 1 to −26.4 dB, which says nothing about voicing but says enough about the class of tool. |
| speaker embedders | D13 records them as weak on sustained vowels, which is most of this branch's material. Attribution is the shared sub-workflow of D14, downstream, and its input is this branch's spans. |
| praat HNR | measured on a real recording it returns `nan` nearly everywhere, valid only at two cough onsets (span-probe Finding 6). It is an alternative or an addition to normalised autocorrelation and it is **unmeasured on voicing**; it is not a substitute. |

## The nodes, in order

Nodes 0-7 are the measurement. Node 8 is the only decision, node 9 the only gate.

| # | node | kind | in | out |
| --- | --- | --- | --- | --- |
| 0 | `energy_track` | pure | `audio`, `cfg.voice.frame_s`, `cfg.voice.hop_s` | `energy_track`, `energy_reference` |
| 1 | `periodicity_track` | pure | `audio`, `cfg.voice.frame_s`, `cfg.voice.hop_s`, `cfg.voice.f0_search_hz` | `periodicity_track`, `f0_candidates` |
| 2 | `voicing_gate` | pure | `energy_track`, `periodicity_track`, `cfg.voice.periodicity_floor`, `cfg.voice.rms_floor`, `cfg.voice.min_run_s` | `voiced_runs[]` |
| 3 | `claim_reconcile` | pure | `voiced_runs[]`, `residual_windows` | `owned_runs[]`, `foreign_runs[]` |
| 4 | `period_mark` | pure | `audio`, `owned_runs[]`, `f0_candidates` | `period_marks[run]` |
| 5 | `span_edges` | pure | `owned_runs[]`, `period_marks`, `energy_track`, `periodicity_track`, `cfg.voice.offset_criteria` | `vocalization_spans[]` |
| 6 | `run_group` | pure | `vocalization_spans[]`, `energy_track` | `run_groups[]` |
| 7 | `span_features` | pure | `period_marks`, `energy_track`, `energy_reference`, `vocalization_spans[]`, `run_groups[]` | `features[span]`, `energy_contrast` |
| 8 | `member_propose` | decision | `features`, `run_groups`, **`hint?`**, `cfg.voice.member_prior` | `member_proposals[]`, `hint_conflicts[]` |
| 9 | `voice_outcome` | gate | all of the above, `kinds` | `fail` / `flag` / `pass` |

**The hint port appears twice and never before node 8.** That placement is the enforcement of "a hint
conditions the decision, never the measurement": no node that computes a track, a period, a span edge or
a feature declares a hint port, so a hint cannot reach a measurement by any route, and the property is
checkable by reading the table rather than by trusting a convention. Node 9 reads the hint only through
`hint_conflicts`, which node 8 produced.

`energy_reference` is a recording-level product — the level distribution over the whole file — and it is
the reason node 0 cannot be scoped to `residual_windows`.

## The gate, and the one parameter with a measurement behind it

Normalised autocorrelation with an RMS floor, so periodic room tone cannot pass. Reproduced from
`taxonomy.md`:

| region | RMS | F0 | periodicity |
| --- | --- | --- | --- |
| sustained voicing, 3.20-3.40 s | 0.0188 | 87.4 Hz | **0.933** |
| sustained voicing, 4.40-4.60 s | 0.0161 | 88.1 Hz | **0.934** |
| quiet stretches | 0.0004-0.0007 | unstable | **0.22-0.44** |

**These two observations constrain an interval, not a value.** Any periodicity floor in (0.44, 0.933)
separates them and any RMS floor in (0.0007, 0.0161) does too — a factor of 2.1 and a factor of 23. So
there is no fitted number here, only a wide gap on one recording, and a wide gap on one recording is
precisely what cannot tell you where the boundary sits on another. Picking a midpoint would be inventing
a decision the measurement does not contain. The parameter table below therefore records the interval and
leaves the derivation slot empty.

### The provenance gap in it, which has to be closed before anything rests on it

`taxonomy.md` is the only place these six numbers appear anywhere in the repository — no measurement note
produces them, and the table names no recording. That matters because the timestamps are checkable
against the one file that does have verified labels
(`streaming-audio-2026-07-30T04-21-56-487Z.wav`, `ground-truth-2026-08-18.md`):

| region in the gate's table | what is verified at that time in the labelled file |
| --- | --- |
| 3.20-3.40 s | inside **breath 1**, 2.2995-3.5205 s, a verified exhalation — and breath is unvoiced |
| 4.40-4.60 s | inside a **verified-empty** stretch, 3.5205-5.3285 s |

Either the table comes from a different recording, which is the likely reading — the labelled file
contains no sustained voicing at all, only a mouth sound, two exhalations, two coughs and 1.554 s of
speech — or it is the labelled file, in which case both "sustained voicing" rows are a breath and a
silence, and the 0.933 is something other than a vocal tract. The same file carries stationary tones at
**85.0** and 108.4 Hz, within 3% of the table's 87.4 and 88.1 Hz, now believed to be music partials
rather than interference.

This is not a claim that the numbers are wrong. It is that the branch's only empirical parameter cites no
source, its stated timestamps land on verified non-voicing in the only labelled file, and a periodic
non-vocal competitor sits at the same frequency. **Naming the recording is the cheapest thing on this
page and it must happen before the floor is used anywhere.**

### Runs are elementary, and nothing is merged

`voicing_gate` emits maximal runs of frames that clear both floors and last at least `min_run_s`. It has
**no gap-merging parameter**, and the omission is the design rather than an oversight.

A gap-merging tolerance would silently decide the laughter/vowel distinction. `taxonomy.md`'s own feature
table separates them exactly there — a sustained vowel is "high periodicity held for a long run",
laughter is "periodicity intermittent in bursts" — so a `max_gap_s` large enough to hold a sustained
vowel together across creak is large enough to fuse a laughter burst train into one span, and the
parameter would be answering the question the branch exists to report. Merging also destroys the one
member the feature set genuinely resolves: an intermittent `prolonged-vowel` is a finding, and merging
converts it into a clean vowel.

So intermittency is a **product**, not a cleanup. `run_group` collects runs separated by less than a
declared reporting horizon and reports the gap durations, the burst count and the energy modulation rate
over the group, without altering the runs. A maximum phonation time computed over a group with interior
gaps is not a maximum phonation time, and the group carries what is needed to say so.

Fragmentation control lives in `min_run_s` alone, which is unfitted, and whose value trades laughter-burst
recall against creak fragmentation. Nothing measured here bounds it.

## Period marks, not an F0 contour

The primary product is a **point process**, not a series: per voiced run, an ordered sequence of glottal
period boundaries, each carrying its period duration, its amplitude, and the autocorrelation peak that
placed it.

| product | grid | defined where |
| --- | --- | --- |
| `energy_track` | analysis grid, `cfg.voice.hop_s` | everywhere in the recording |
| `periodicity_track`, `f0_candidates` | analysis grid | everywhere; the F0 value is meaningless below the gate's floor and is carried with its periodicity so a reader cannot separate them |
| `period_marks` | **no grid** | inside voiced runs only; **absent** elsewhere, not zero and not interpolated |

The reason the product is periods and not a contour is arithmetic. At 87.4 Hz one period is **11.44 ms**.
Any contour on a fixed hop has already committed to a resolution coarser than or comparable to the
quantity it is sampling, and every period-to-period measure a consumer would want — jitter, and shimmer
from the amplitudes — is defined between consecutive periods and is unrecoverable from a resampled
contour. The branch therefore publishes the periods and **does not publish jitter or shimmer**: those are
derived statistics, each needing its own validity rule about how many consecutive periods are enough, and
a consumer that needs one can compute it from the process with its own rule stated.

F0 and its trajectory are read off the period process inside runs, and off `f0_candidates` outside them
where they are only ever used to explain why the gate did not open.

### The F0 search range is one parameter serving two irreconcilable populations

An adult male sustained vowel sits at 87.4 Hz. An infant cry sits far above it — `taxonomy.md`'s own
discriminator for crying is "high periodicity at a high absolute F0". A range set for adults
half-octave-errors or rails on infants; measured on a real recording, `pyin` railed at its 60 Hz floor
through every quiet stretch, locking onto low-frequency rumble (span-probe Finding 6), which is what
railing looks like.

The range must therefore be wide and **must not be conditioned on the hint**, because narrowing a search
range on a declared expectation is a hint reaching a measurement. The consequence is stated rather than
hidden: a wide range readmits periodic non-vocal competitors inside it, including the 85.0 Hz partial
above. The wide range and the music vulnerability are the same parameter, and this design cannot have one
without the other.

What it does instead of resolving that: `f0_candidates` retains the top-k autocorrelation peaks per frame,
so an octave error is **visible in the product** rather than baked into it, and a reader can see that the
chosen peak had a competitor at half or double.

## Spans: the onset is a period, the offset is a criterion

Onset and offset are not comparably measurable here, and the type says so rather than reporting one
number twice.

**The onset is sharp, and its uncertainty is `1/F0`.** Voicing onset is a phase transition, not a ramp:
periodicity steps from 0.22-0.44 to 0.933 between the gate's two regimes. The finest thing that can locate
it is the first period mark, and a period mark cannot be placed to better than the period it bounds:

| onset F0 | one period | plausible member |
| --- | --- | --- |
| 87.4 Hz | 11.44 ms | adult sustained vowel |
| 200 Hz | 5.00 ms | child, adult female |
| 400 Hz | 2.50 ms | infant cry |

So onset uncertainty is **not a constant**, and reporting it as one would over-state precision on the low
voices and under-state it on the high ones. It is unfitted as a coefficient — whether the true figure is
one period, two, or a fraction of one is unmeasured — but its *scaling* with F0 is a property of the
instrument and is not a free choice. Elsewhere in this project a DSP envelope held a cough onset to ±5 ms,
which is the right order for the same reason: a transient with a 9 ms rise cannot be bounded finer than
its rise.

**The offset is definitional, and one number would report a choice as a measurement.** D12 records that
phonation degrades into creak and irregularity before it stops, and that voicing-based, amplitude-based and
regularity-based offsets disagree by hundreds of milliseconds. The envelope's own sensitivity is measured:
moving an offset threshold from floor+12 dB to floor+3 dB moved a breath offset by **2.03 s** and a cough
offset by 1.04-1.10 s. For cough that ambiguity was later shown to be an artifact of the envelope method —
CrisperWhisper bounded cough 1's offset to within 14 ms against the verified window — but that correction
came from a *speech* model on a transient, and there is no equivalent instrument for a gradual phonation
offset. The ambiguity stands here.

So the span type carries a mapping, not a scalar:

| field | type | notes |
| --- | --- | --- |
| `onset_s` | time | first period mark of the run |
| `onset_uncertainty_s` | time | one period at the onset F0, scaled per the table above; coefficient unfitted |
| `offset` | **absent**, or `{criterion -> time}` | one entry per criterion in `cfg.voice.offset_criteria`; never reduced to one |
| `offset_status` | `resolved` / `contested` / `truncated` | see below |
| `duration` | `measurement` / `lower_bound` / `unresolved` | derived, and it carries which of the three it is |

The three criteria, all with empty derivation slots:

| criterion | rule | what it is sensitive to |
| --- | --- | --- |
| `periodicity` | last frame at or above the gate's periodicity floor | creak, which drops periodicity while phonation continues |
| `amplitude` | last frame within `cfg.voice.offset_drop_db` of the run's peak | the drop parameter, per the 2.03 s sweep above |
| `regularity` | last period mark whose period is within `cfg.voice.offset_period_tol` of the run median | the same creak, from the other side |

`offset_status` is `resolved` when the criteria agree to within `cfg.voice.offset_agreement_s`,
`contested` when they do not, and **`truncated` when the last frame of the recording is voiced**. A
truncated span has **no offset at all** — the field is absent, not the file duration — and its `duration`
is a `lower_bound`. That is the case the whole asymmetry exists for: a maximum-phonation-time recording
that ran out of file has produced a lower bound on the measurement being asked for, and publishing the
file duration as the answer would be publishing the recording length as a physiological measure.

## The six features, and the three members no feature names

From the two tracks and the period process:

| feature | computed from | separates |
| --- | --- | --- |
| periodicity level | `periodicity_track` over the run | voiced from not — the gate |
| F0 level | period process median | infant cry from adult vowel |
| F0 trajectory | period-to-period F0 over the run | held vowel from swept glide from incoherent noise |
| energy level | run energy against `energy_reference` | loud phonation, **only relative to the recording** |
| energy modulation rate | `energy_track` over a `run_group` | laughter's burst amplitude modulation |
| voiced-run duration | span, under a named offset criterion | maximum phonation time, where the duration *is* the measurement |

Against the members:

| member | resolvable from the six? |
| --- | --- |
| sustained vowel | yes — high periodicity, long run, F0 trajectory flat |
| pitch glide | yes — F0 trajectory monotonic and smooth over a long run |
| maximum phonation time | not a class; the span's `duration` and its `offset_status` |
| loud phonation | not a per-span class; `energy_contrast` between two spans in one recording |
| crying | weakly — high F0 with modulation, but the F0 boundary against a high child voice is unfitted and there is no material here |
| laughter | weakly — burst intermittency at a group level, and `min_run_s` decides whether the bursts survive at all |
| **vocal imitation** | **no.** No trajectory distinguishes it. |

**Imitation needs a hint or it cannot be named, and that is a property of the feature set.** It passes the
gate because a vocal tract genuinely made it — which is exactly what D11 records as the protection for
`pediatric.noisy-sounds`, where a source classifier will confidently and correctly identify the imitated
animal — and its trajectory can be anything. Nothing in this branch papers over that. An unnameable span
is still fully measured: its periods, amplitudes and edges are the product, and only the proposal is
empty.

## Member labels in the product: proposals, never claims

They are in the product, and they are `Estimate`s over the member vocabulary, one list per span, marked as
proposals.

**Why they are there at all.** The measurement is the product, but an F0 sweeping monotonically over two
seconds is a finding, and a reader cannot act on it unless something names it. More sharply: the
hint-contradiction finding is only expressible if there is a name to contradict. Remove the proposals and
`prolonged-vowel` against an intermittent periodicity track has nothing to disagree with.

**Why they are proposals and not claims.** Of the seven members, two are not classes, one is unnameable by
construction, two are weak, and none of the discriminating boundaries has been fitted on anything. A
single argmax label would publish an unfitted decision as a fact, which is the failure this project has
already paid for twice — a silhouette coefficient read as a probability, and a 2→10 dB HNR ramp under which
ordinary voiced speech read as only partly voiced.

Each proposal is an `Estimate` (`utils/data_structures/estimate.py:28`), which makes the honesty
structural rather than a convention:

| `Estimate` field | what this branch puts in it |
| --- | --- |
| `raw` | the discriminating statistic, or `None` where nothing discriminated — imitation, always |
| `n_evidence` | how many of the six features actually separated this span from the alternatives; `0` where none did, which is legal and means "nothing observed" |
| `prior` | the hint's expectation where a hint names a task, the flat residual prior where it does not |
| `prior_key` | the config key holding that prior, so its derivation is findable |
| `population` | honest, and currently narrow: everything behind these features is one adult on a close mic |

`n_evidence = 0` with `raw = None` collapses `value` to the prior, which is the correct reading of an
imitation span: the only thing known about its identity is what was declared, and the measurement added
nothing. That is not a gap papered over — it is the gap, represented.

**Nothing downstream gates on a proposal.** They reach the report and the flag. The measurement is what a
consumer computes on.

## The hint: what it can and cannot do

`AudioHints` (`audio/data_structures/audio_hints.py`) is an assertion by an operator or a protocol, not an
observation, and D5 makes it an optional parameter port that conditions the decision.

| hint | what node 8 does with it |
| --- | --- |
| `prolonged-vowel`, `long-sounds` | expects one long run, flat F0. Raises the prior on sustained vowel. |
| `glides` | expects a monotonic F0 sweep. Raises the prior on pitch glide. |
| `loudness` | expects two spans and reads `energy_contrast` between them; a single span is a conflict. |
| `maximum-phonation-time` | expects one long run whose `offset_status` is `resolved`; `truncated` or `contested` is a conflict, because the requested measurement is the thing that is unresolved. |
| `noisy-sounds` (imitation) | the only route by which imitation can be named. Names it, at `n_evidence = 0`. |
| absent | every proposal falls back to the flat prior; the members the trajectory resolves are still proposed, imitation is not. |

**A hint the tracks contradict is a finding, not an error to suppress.** `hint_conflicts` is a declared
output of node 8 for exactly that. A `prolonged-vowel` hint whose periodicity track shows intermittent
bursts is telling you the participant could not sustain phonation, or did not understand the prompt, or
that the wrong file was attached to the wrong task. All three matter, and none is distinguishable here, so
the conflict is reported and it flags rather than being resolved.

## Boundary fact: a leaked cough is indistinguishable from a downward glide by trajectory

Cough is voiced. A diarizer's raw posterior reads 0.574 and 0.906 on the two verified coughs, so voicing
alone cannot separate airway from this branch, and detection order is what does: airway claims a cough
before this branch sees it.

**What this branch does when airway under-claims.** The leaked cough arrives inside `residual_windows`, so
`claim_reconcile` cannot catch it — by definition nothing claimed it. It becomes an `owned_run` and is
measured like anything else. And then the collision:

| | cough's voiced phase, measured | pitch glide, expected |
| --- | --- | --- |
| F0 trajectory | descending harmonic chirp, 9.65-10.00 s | monotonic sweep |
| periodicity | high enough for a diarizer to read 0.906 | high |
| duration | ~350 ms of a 568-640 ms event | seconds |
| 10-90% rise | **9-17 ms** | a phonation onset, far slower |
| level step | **44.9-48.5 dB** | not measured on phonation |

**The trajectory is the same shape.** A cough's voiced phase and a descending pitch glide are both smooth
monotonic F0 descents, so the feature `taxonomy.md` nominates to separate the residual's members is the
one feature that cannot separate this. What does separate them on the probe recording is rise time and
level step — and those figures are explicitly forbidden from becoming thresholds: n=2 from one healthy
adult on a close mic, and `ground-truth-2026-08-18.md` lists four populations that move them (reduced peak
cough flow in neuromuscular disease and post-stroke, absent glottic closure, infant and child cough, COPD
and asthma), failing hardest where the signal matters most.

So the honest behaviour, and it is a deliberate choice:

1. The span is measured and published, with its periods and amplitudes.
2. It carries a `short_run_fast_onset` marker — a short run whose rise and level step sit in the transient
   regime — as a **continuous feature with its uncertainty**, never a threshold verdict.
3. `member_propose` emits **no proposal** for it. `n_evidence = 0`.
4. The branch **flags**.

Not `fail`: the instrument worked and produced a correct period process. Not a silent `pass`: a 350 ms
monotonic descent published with no marker would be read as a glide, and a glide computed on a cough is a
clinical number about the wrong event. The flag names the suspicion — "voicing consistent with a
transient, unclaimed upstream" — rather than naming the class, because naming the class would be this
branch claiming a member of a kind it does not own.

**And it is the only place airway's recall failure becomes visible.** A cough airway misses is otherwise
invisible: airway does not know it missed it, TAXONOMY's presence answer for airway may still be `present`
from the other cough, and nothing else in the graph looks at that region. This flag is the detector for a
defect in a sibling branch, which is worth stating because a later change that suppresses it to reduce
flag volume would remove the only such detector.

## Boundary fact: breath is unvoiced, the gate rejects it, and that is correct

A diarizer reads exactly **0.0000** on both verified breaths, and Brouhaha's VAD reads 0.0049 and 0.0055,
against 0.689 on real speech. The gate's periodicity floor will never admit them.

**This is correct behaviour and must not be "fixed."** Two reasons, and the second is the stronger:

1. Breath belongs to airway, by taxonomy. Admitting it here would give one element two producers.
2. **Nothing measured anywhere in this project can bound a breath's extent.** Coverage of the verified
   breath windows ran 10-52% across five independent instruments — CrisperWhisper 26.2% and 10.2%, HeAR's
   event detector 52.4% and 24.4% — against 64-98% for coughs and 87-98% for speech, and the breaths are
   *longer* than the coughs. Every instrument marks where a breath begins and then loses it.

This branch's product is spans. Admitting a class whose extent no instrument recovers would mean publishing
`vocalization_spans` whose offsets are the offset criterion and nothing else, on the one element where that
is known in advance. The gate's rejection is not a coverage gap; it is the branch declining material it
could not measure.

## fail, flag, pass

`fail` is about the instrument. `flag` is a judgement that could have gone either way.

| outcome | condition |
| --- | --- |
| **fail** | the sample rate cannot support `cfg.voice.f0_search_hz`; the recording is shorter than one analysis frame; the periodicity track cannot be computed |
| **fail** | the F0 estimate sits at a boundary of the search range for more than `cfg.voice.rail_fraction` of admitted frames — the railing failure, measured as `pyin` locking to 60 Hz rumble. A railed estimator is not a low-confidence measurement, it is no measurement. |
| **flag** | **zero owned runs**, where TAXONOMY said the residual kind is `present`. Upstream and the tracks contradict each other, and that is a finding either way — a gate that admitted something this branch cannot reproduce, or a residual whose voicing was all claimed after all. |
| **flag** | any span with `short_run_fast_onset` — the leaked-transient case above |
| **flag** | any entry in `hint_conflicts` |
| **flag** | `offset_status` is `contested` or `truncated` on a span whose duration is the requested measurement — `maximum-phonation-time`, `glides`, `prolonged-vowel` |
| **flag** | a periodic non-vocal competitor inside the F0 search band: an admitted run whose F0 is stationary to within `cfg.voice.stationary_tol_hz` across its whole extent **and** whose energy does not vary. The measured case is the 85.0, 108.4, 164.1, 1564.5 and 1757.8 Hz tones on the probe file, now read as music partials rather than interference. A human vocal fold does not hold F0 that still. |
| **pass** | at least one owned run with a period process, and none of the flag conditions |

**An imitation span passes.** It is not a flag, and that is a choice the evidence does not force. The
argument for flagging it is that no member could be proposed. The argument against, which wins here: a
`pediatric.noisy-sounds` recording is imitation and little else, so flagging on unnameability would flag
100% of a whole task type and the flag would stop meaning anything. Unnameability is a known property of
the feature set, not a borderline judgement, and `flag` is for judgements that could have gone either way.
The product is complete: periods, amplitudes and spans, with an empty proposal and a stated reason.

**`foreign_runs` do not flag.** Voicing that fell outside `residual_windows` was claimed by another kind
and is another branch's product. It is counted and reported so the residual's share of the recording is
legible, and nothing more.

## Parameters

Every value in this branch is unfitted. The gate is the only one with an observation behind it, and that
observation constrains an interval rather than a value.

| key | what it sets | status | derivation |
| --- | --- | --- | --- |
| `voice.periodicity_floor` | the gate | observed interval **(0.44, 0.933)** on one unnamed recording | *empty — and the recording must be named first* |
| `voice.rms_floor` | the gate | observed interval **(0.0007, 0.0161)**, a factor of 23, same recording | *empty — same* |
| `voice.frame_s`, `voice.hop_s` | analysis grid | must resolve laughter's burst modulation and must not be confused with the period process | *empty* |
| `voice.f0_search_hz` | search range | must span infant cry to adult male; narrowing it on a hint is forbidden | *empty* |
| `voice.min_run_s` | fragmentation | trades laughter-burst recall against creak fragmentation | *empty* |
| `voice.offset_criteria` | which offsets are reported | a set, never reduced to one | *empty* |
| `voice.offset_drop_db` | amplitude offset | the 2.03 s / 1.76 s breath sweep is the sensitivity, on a different class | *empty* |
| `voice.offset_period_tol` | regularity offset | creak tolerance | *empty* |
| `voice.offset_agreement_s` | `resolved` vs `contested` | *empty* |
| `voice.rail_fraction` | the railing `fail` | *empty* |
| `voice.stationary_tol_hz` | the music-partial flag | *empty* |
| `voice.member_prior` | proposal priors, per member, and the hint's substitutions | *empty* |

**No thresholds as code literals.** Per CLAUDE.md these are config keys with `derivation:` blocks, and
`derivation: unfitted` is an acceptable value where nothing has been fitted. Two of this project's defects
came from literals that were never fitted, and one of them — the 2→10 dB HNR ramp under which ordinary
voiced speech at a median 8.12 dB read as only partly voiced — was a voicing threshold, in this branch's
own territory.

## Choices the evidence did not force

Stated so a later reader can reverse one without re-deriving the whole file.

| choice | the alternative | why this one, and what would settle it |
| --- | --- | --- |
| the product is a period point process, not an F0 contour | a contour on the analysis grid | one period is 11.44 ms at 87.4 Hz, so a contour cannot express jitter; but nothing here measured whether a consumer actually needs period-level resolution. A jitter measurement on verified pathological and healthy vowels would settle it. |
| jitter and shimmer are not published | publish them per run | each needs its own validity rule about sufficient consecutive periods, and none is fitted. If a consumer needs them under a stated rule, publishing them here with that rule stated is a small change. |
| no gap merging; intermittency is a product | a `max_gap_s` | merging decides the laughter/vowel distinction. Laughter material with verified burst boundaries would settle both this and `min_run_s`. |
| member labels are proposals in the product | omit labels entirely, or claim one | omitting them makes the hint-contradiction finding inexpressible. Claiming one publishes an unfitted decision. A labelled residual corpus would let a claim be earned. |
| imitation passes, unnameable | flag it | flagging would flag every `pediatric.noisy-sounds` recording. Reversible if flag volume turns out not to matter. |
| a leaked transient flags | fail, or pass silently | the instrument worked, so not fail; a silent pass publishes a glide computed on a cough. What would settle it is airway's recall measured on material containing coughs *and* phonation, which nothing here has. |
| the branch measures the whole recording and `residual_windows` only selects | bound the analysis to the admitted regions | loud phonation is relative to the recording, so the reference cannot be computed from a subset. Forced, not chosen. |
| TAXONOMY grows the `residual_windows` port | the branch recomputes the overlap | recomputation duplicates one decision and makes the branch fail to run when a sibling kind is absent. |
| onset uncertainty scales as `1/F0` | a constant tolerance | a period mark cannot be placed finer than its period; the *coefficient* is unfitted and a constant would misreport both ends of the F0 range. |

## What would settle the parameters

The gate needs a corpus, and `test-examples.md` names candidates per member: Coswara vowels, the
Saarbruecken Voice Database and PVQD for sustained vowels with pathology ratings; donateacry and ESC-50
`crying_baby` for infant cry F0; AudioSet and FSD50K for laughter. None is verified for availability,
licence or label granularity, and none carries verified *spans* — they carry clip labels, which fits the
gate's frame-level question badly.

The scoring shape is available from the airway branch's own proposal and transfers directly: voiced-frame
recall against **false-voiced frames per minute** over verified-empty stretches, on the originals and on
degraded copies with added noise and reverberation. That last part is the one that matters most here,
because every number in this file comes from one quiet close-miked recording, and the gate's RMS floor is
justified against room tone rather than against a reverberant room or background music.

Three measurements would move this branch further than any design change:

1. **Name the gate's recording** and re-measure the two floors on material with verified voiced spans.
2. **Measure a phonation offset** the way the cough offset was measured — against a human-verified span —
   so the offset criteria can be compared to something rather than only to each other.
3. **Measure onset uncertainty against verified voicing onsets** across the F0 range, to fit the
   coefficient on `1/F0`.

## What this branch does not do

- No classifier, no ASR, no enhancer, no speaker embedder. Each exclusion is measured, above.
- No source attribution and no target selection. D14 puts attribution downstream of per-source
  measurement; this branch's spans and period processes are its input, and the branch reports per run, not
  per source.
- No breath, ever, by gate construction.
- No jitter, shimmer or HNR as published measures.
- No member claim, and no gate on a member proposal.
- No merging of voiced runs.
- No offset reduced to a single number.
- No hint reaching any node numbered 0 through 7.
