# VOICE branch

What the branch answers: **is there sustained phonation here, what are its acoustic properties, and
did the voice do what the task asked?**

The frame is [`../20260913-branch-contract-and-hints/design.md`](../20260913-branch-contract-and-hints/design.md).

## The state of this branch

**VOICE fails on every recording.** Its subject is every live span whose `family` is `phonation`
(`voice.py:232`, `_PHONATION_FAMILY` at `:39`). Nothing reachable proposes one: the detector that
did was retired on 2026-09-04, and the only `prov_type="span"` write in the module
(`voice.py:336-348`) sits downstream of the no-span path the branch always takes at `:236-266`. So
the branch resolves to `Outcome.FAIL` with a `why` naming the retirement, and roughly 380 of its 420
lines are unreachable.

That single fact organises this document. VOICE does not need repair; it needs a proposer, and under
the branch contract **the branch is the proposer**.

**And VOICE is the worked example of what the contract forbids.** `voice.py:336-348` mints a *second*
span from an input span, re-keyed by period-aligned onset and carrying `onset_kind`. That is exactly
the re-minting the contract replaces with a `refine` assertion, and it is why `report.py:291-308`
needs `_spans_of_family` to split one family into two populations on `("onset_kind" in attributes)`.
Once the minting becomes a `refine`, that split has nothing to separate and the two-population
problem dissolves. Note the consequence for REPORT: `onset_kind` is written *only* at `voice.py:343`,
so on stores written under the contract the `voice=True` reads at `report.py:715` and `:1154` return
empty, and REPORT's VOICE arms need the span-versus-assertion distinction substituted in rather than
the family merely widened.

## The tasks this branch serves

Declared families, not ground truth.

| family | n | what the task asks for |
| --- | --- | --- |
| `maximum-phonation-time` | 2,696 | sustain a vowel as long as possible on one breath |
| `prolonged-vowel` | 1,604 | sustain a vowel at comfortable pitch and loudness |
| `glides-low-to-high` | 1,596 | glide F0 upward across the range |
| `glides-high-to-low` | 1,554 | glide F0 downward across the range |
| `loudness` | 897 | produce at varying intensity |
| `maximum-phonation-time-v2` | 813 | as above |
| `loudness-v2` | 705 | as above |

Three clinical measurement shapes. **Maximum phonation time is a duration** — the single number the
task exists to produce, and the one measurement where getting the offset right *is* the result.
**Glides are an F0 trajectory** — range in semitones, monotonicity, and direction, which is what
separates the two glide families from each other. **Prolonged vowel and loudness are voice quality
and intensity** over a steady segment, which is where jitter, shimmer, HNR and CPP live.

### A recording routed here whose declared task is not voice

VOICE routed 22,277 recordings against 8,306 declaring a voice family. Sustained phonation appears
inside sentence reading, inside free speech, and inside DDK. The branch marks it wherever it finds
it and says what it measured. A long steady vowel inside a Harvard sentence is a real vowel; VOICE
records its properties and asserts nothing about whether the sentence task was performed.

## Capabilities

### V1 — Propose phonation spans (**not built; the branch's missing foundation**)

**Question.** Where is the sustained phonation?

**Reads.** PREPROCESS's general spans, its `phonation_tracks` measurement — written whole-file by
`phonation_tracks` (`preprocess.py:919`, called at `:2147`), which runs `f0_track` over the
pre-emphasised stream for per-frame F0 and voicing strength and `formant_track` over `plain` for the
first four formants — and the HNR track VOICE can compute itself via `hnr_track`
(`tasks/phonation/api.py`, exported at `tasks/phonation/__init__.py:3-12`).

**Computes.** A contiguous region of voiced frames. A voiced frame is one where Praat returns an F0
and a voicing strength; the region is the maximal run of them. That formulation is **parameter-free
in its core** — "F0 exists here" is Praat's own decision, not a threshold this branch picks.

**Emits.** `propose` spans, `family: "VOICE"`, `wasDerivedFrom` the phonation track. Where a proposed
region coincides with a PREPROCESS span whose extent differs, a `refine` assertion carrying
`corrected_extent` rather than a second span — the fix for the re-minting described above.

**The offset is the hard part and it is definitional, not measurable.** Where a sustained
phonation ends depends on the criterion used to decide it, and for maximum phonation time that
choice *is* the measurement. How far an offset moves between plausible criteria is **unmeasured**
in any source this document can stand on. The honest resolution is to state the offset convention as
a convention — voicing ceases where Praat reports no F0 — and report the duration against it, rather
than to fit a dB criterion no ground truth supports.

**Owed ground truth.** `phonation.hnr_floor_interval_db` and `phonation.rms_floor_interval` are both
null (`default.yaml:146-147`), and their own comment says why: *Praat calibrates no dB floor*. The
branch already computes a tri-state `gate_interval` from whether both are set
(`voice.py:196-203`). Neither can be fitted against the corpus.

### V2 — Maximum phonation time (**not built**)

**Question.** How long was the longest sustained phonation?

**Computes.** The duration of the longest V1 span. That is the whole measurement — no threshold, no
model, nothing owed. It is available the moment V1 exists.

**Emits.** A `counts` entry, `found` being the measured duration; `declared` from the declaration's
expected duration where the per-task table carries one.

**Serves.** `maximum-phonation-time` (2,696), `-v2` (813), and `prolonged-vowel` (1,604) as a
secondary reading.

**Today there is a `_task_range` check** (`voice.py:131-163`) reading `voice.task_duration_ranges`,
which is null (`default.yaml:162`), so it returns `not_evaluated` immediately. Under the contract
this becomes a `counts` entry asserting no discrepancy, not a gate — and that is the better form,
because a duration range per task is exactly the kind of number that would otherwise be fitted
against declarations.

### V3 — F0 trajectory (**detectors exist; branch consumption not built**)

**Question.** What did F0 do across the phonation — how far did it travel, how monotonically, and in
which direction?

**Reads.** The per-span F0 track.

**Computes.** Semitone range, semitone IQR, monotonicity, monotone fraction, sweep rate, rank
correlation with time, and direction bias. These are already defined as detectors and already held
for this branch: `BRANCH_DETECTORS = {"VOICE": _PITCH_TRAJECTORY}` (`detectors.py:1555`) holds the
22 trajectory detectors defined at `detectors.py:1477-1549` — 19 under the `glide` kind and 3 under
`voice` — specifically because no routing profile could promote them. They are branch measurements,
not gates, which is why each carries an empty `thresholds` and is absent from `DETECTORS`.

**Direction is what separates the two glide families, and the detectors are already signed for it.**
`detectors.py:1477-1549` defines `_rising`/`_falling` pairs reading one signed key at both
polarities — `direction_bias`, `rank_correlation`, `net_over_variation`, `sweep_semitones` — which
is exactly the shape a direction question needs. **How well any of them separates the two families
is unmeasured** in any source this document can stand on, and measuring it against the declared
families would fit the declaration rather than the phenomenon. The branch should therefore report
the trajectory values and let a reader compare them, rather than emit a direction verdict against a
cut nobody has earned.

**Emits.** A per-span measurement carrying the trajectory values, plus `label` assertions naming the
direction where the branch chooses to name it.

**Serves.** `glides-low-to-high` (1,596), `glides-high-to-low` (1,554).

**What is missing.** `_f0_trajectory` and the sweep computation live in the routing-analysis module,
not in `tasks/phonation`. A branch consuming them needs them promoted to
`senselab.audio.tasks.phonation` with a per-span reduction, so the branch and the router read one
definition. Nothing about that requires a fit.

### V4 — Voice quality (**machinery exists; branch consumption not built**)

**Question.** What are the perturbation and noise properties of the sustained phonation?

**Reads.** The V1 span and the audio.

**Computes.** Via `tasks/features_extraction/praat_parselmouth.py`, all of which exists:
`extract_jitter` (`:1112`), `extract_shimmer` (`:1168`), `extract_harmonicity_descriptors` (`:580`),
`extract_cpp_descriptors` (`:706`), `extract_pitch_descriptors` (`:448`),
`extract_intensity_descriptors` (`:515`), `extract_slope_tilt` (`:638`),
`measure_f1f2_formants_bandwidths` (`:824`).

Jitter, shimmer, HNR and CPP over a sustained vowel are the standard acoustic correlates of voice
quality — CPP in particular is the measure most consistently associated with perceived dysphonia. A
speech scientist would expect exactly this set on `prolonged-vowel` and on the CAPE-V protocol's
sustained-vowel component.

**Emits.** A per-span measurement. **No verdict** — these are numbers, and mapping them to normal or
disordered requires norms this project does not have and cannot derive from declarations.

**Serves.** `prolonged-vowel` (1,604), `maximum-phonation-time` as a secondary reading, and the
sustained-vowel portion of the CAPE-V families that SPEECH holds.

**Owed ground truth.** Every normative cut. None should be attempted.

### V5 — Intensity dynamics (**not built**)

**Question.** Did intensity vary as the loudness task asked?

**Computes.** The intensity contour over the V1 span via `extract_intensity_descriptors`
(`praat_parselmouth.py:515`), reported as range and trajectory.

**The recording is not calibrated.** Absolute SPL is not recoverable from a file whose gain is
unknown — and `recording_input_gain` does not make it recoverable, as the owner ruled in a different
context. So only *relative* dynamics within the recording are measurable, and the design must say so
rather than reporting a dB number that reads as absolute.

**Serves.** `loudness` (897), `loudness-v2` (705).

### V6 — Population-conditioned F0 range (**gated behind null config**)

`_f0_range` (`voice.py:48-83`) reads `hint.metadata["population"]` and looks it up in
`voice.f0_range_by_population`, which is null (`default.yaml:160`); it falls back to
`derive_f0_range` over the wide `voice.f0_search_range_hz`, which ships `[50.0, 600.0]`
(`default.yaml:159`). Under the contract the population comes from the declaration rather than from
`metadata`. The per-population ranges are **owed ground truth** — but unlike most owed numbers,
these are published clinical norms rather than something to fit, so the right move is to cite a
source in `data/` with its derivation, not to measure them here.

## Deviations

| type | evidence |
| --- | --- |
| `off_task_extent` | lexical speech inside a sustained-phonation task; a region with no phonation where the task asked for one |

`expected_event_count` does not apply — voice tasks ask for one production, not a count. The
duration comparison is a `counts` entry from V2.

VOICE emits no `stimulus_mismatch`: none of its tasks carries a stimulus text.

## What exists today

| capability | status |
| --- | --- |
| V1 propose phonation spans | **not built** — the branch has no subject |
| V2 maximum phonation time | not built; trivial once V1 exists |
| V3 F0 trajectory | 22 detectors held in `BRANCH_DETECTORS["VOICE"]`; branch consumption not built |
| V4 voice quality | Praat machinery complete; branch consumption not built |
| V5 intensity dynamics | Praat machinery exists; not built |
| V6 population F0 range | gated behind null config |

Reachable today: `_f0_range` resolution, `resolve_stream`, the activity write, and the
`gate_interval` tri-state (`voice.py:196-203`). Everything from `:268` on — the HNR and RMS tracks,
the per-span slicing, `period_marks`, the span minting, the `voice_tracks.npz` sidecar, the
period-doubling alias check (`_alias_in_range`, `:126-128`) and the task-duration check — is
unreachable because the branch always returns at `:266`.

## What the branch emits

```
spans        V1 would propose family: "VOICE" phonation spans
assertions   refine (corrected_extent) where a PREPROCESS span's extent is wrong;
             label naming glide direction; contest where a proposed phonation
             span carries no voiced frame
measurements per-span trajectory (V3), voice quality (V4), intensity (V5)
counts       maximum phonation time {found, declared}
verdict      { spans_n, phonation_s, longest_span_s, longest_span_criterion,
               production, ambiguous_spans_n, marks_skipped_short_n, task_range,
               gate_interval, flags }
```

**The verdict's basis, exactly.** Today: `FAIL` when no phonation span exists (`voice.py:242`) —
which is every recording. On the reachable-but-unreached path: `FLAG` when flags accumulated,
`PASS` when spans were measured and nothing contested (`voice.py:396-399`).

Under the contract the basis should become: `FAIL` when no phonation was found, `FLAG` when a
measurement contradicts another, `PASS` otherwise — with the numbers carried as measurements and no
normative judgement in the verdict.

## Out of scope

Any normative interpretation of jitter, shimmer, CPP or HNR; absolute intensity; any refit of a
threshold against declared families.
