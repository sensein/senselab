# Branch 1 — airway. DAG draft, pending confirmation

Elements: inhalation, exhalation, cough, throat clear, **mouth non-speech sound** (added after
verification; 202 ms, oral, event-shaped, and a target event type for speech tasks in the existing
pipeline).

The branch's job is to **label events and generate spans**, per source, and to say plainly where it
cannot.

## Ordering — propose, corroborate, refine, reconfirm

Established against the verified labels rather than assumed. DSP is **not** the entry: it found 6 of 6
events on this recording but its precision is unmeasured, and a peak-pick delta set slightly too small
produced 119-120 false onsets at a uniform 85 ms spacing. Global DSP entry floods the moment there is
noise or reverberation. It earns its place as a *refiner* inside already-proposed windows, where that
failure mode is unreachable.

| # | node | kind | in | out |
| --- | --- | --- | --- | --- |
| 0 | `transient_propose` | pure | `audio_raw`, `cfg.transient_max_dur_s`, `cfg.transient_isolation_s`, `hear_proposals[]` | `short_proposals[]` |
| 1 | `hear_scan` | model, cached | `audio_raw`, `cfg.hear_hop_s` | `hear_proposals[]` |
| 2 | `class_corroborate` | model, cached | proposals, `audio_raw` | proposals + YAMNet/AST posteriors, agreement recorded |
| 3 | `span_refine` | pure | proposals, `audio_raw`, `crisper_tokens` | `spans[]` — edges adjusted **only inside a proposal** |
| 4 | `span_reconfirm` | model, cached | `spans[]`, `audio_raw` | `events[]` — class `Estimate` per refined span |
| 4b | `spectrogram_review` | model, **optional** | `spans[]`, render params | `visual_verdicts[]` |
| 5 | `group_events` | pure | `events[]`, `crisper_tokens` | grouped `events[]` — **unsolved** |
| 6 | `attribute_sources` | pure | `events[]`, `c50_db`, `rms_db`, `community1_seg` | `source_assignments[]` |
| 7 | `measure_per_source` | pure | grouped events, assignments | `airway_measures[source]` |

`crisper_tokens` feeds 3 and 5 in parallel: its timings were the best measured (cough 1 offset to
within 14 ms) so it proposes span edges, while its labels are unreliable — it called a voiced cough
phase `[UH]` — so it never sets class.

**Node 4 is not a cycle.** It takes refined spans where node 1 took scan windows: different inputs,
different node, acyclic. Contract detail: reconfirming a refined span means re-windowing 2 s of *real*
surrounding audio centred on it, never padding the span, because padding moved a HeAR embedding as far
as substituting unrelated audio.

## `transient_propose` — the short-event path, and its gate

HeAR's class margin is +0.91 at 2 s, +0.46 at 1 s and +0.29 at 0.3 s, and the 202 ms mouth sound was
missed by HeAR, YAMNet, AST and CrisperWhisper alike — only the envelope found it. So a dedicated
short-transient proposer is required, gated so it cannot reintroduce the flood:

1. duration bound — longer candidates belong to `hear_scan`;
2. **isolation requirement** — a click in silence is detectable and a click inside speech is not, so
   isolation is a declared input rather than a hidden heuristic;
3. no overlap with `hear_proposals` — it speaks only where the scan is silent;
4. its candidates still pass through refine and reconfirm, so anything no classifier will confirm dies
   there rather than reaching the output.

Representation candidates, in order of expected value. Not in senselab today: gammatone, cochleagram,
PCEN, madmom. Available without a new dependency: `librosa` (which has `pcen`), `torchaudio`, `scipy`,
and ERB machinery already touched by 37 files.

- **PCEN + per-band flux** — adaptive per-channel gain control, aimed directly at the noise-robustness
  failure rather than hoping a threshold holds.
- **ERB or gammatone per-channel envelope onsets** — the auditory periphery trades frequency resolution
  for time resolution at high frequencies, which is where a click's energy is; a uniform STFT does the
  opposite. True gammatone needs a dependency; an ERB approximation may not.
- **SuperFlux / complex-domain flux** at a 2.7-5.3 ms hop — vibrato suppression targets the spurious-onset
  mode that produced the 120 false onsets.
- **A trained frame-level onset detector** — new dependency, music-trained, transfer unmeasured.

## `spectrogram_review` — optional, and the only non-audio evidence

Every other proposed source consumes a waveform, so their failure modes correlate. A model reading a
rendered spectrogram fails differently, and the strongest non-classifier evidence obtained so far was
visual: the descending harmonic striations after each cough burst, which is what argued against AST's
`Throat clearing` verdict.

**Render parameters are ports, not implementation detail**, because the image is the input:
`cfg.review_context_s`, `cfg.review_freq_max_hz`, `cfg.review_db_range`, `cfg.review_freq_scale`,
`cfg.review_colormap`. This is HeAR's padding problem in visual form — a wrong dynamic range or
frequency ceiling yields a confident answer about the rendering. 98.6% of this file's energy is below
8 kHz; rendered to 24 kHz every event is a smear at the bottom.

Two constraints: it is a **confirmer only**, never a proposer, since it needs a span to render around;
and its open-vocabulary output splits — observations inside the taxonomy contribute evidence, anything
outside reaches the report and the flag but never the gate.

## What this branch will not report

**Breath duration.** Coverage of the verified breath extents ran 10-52% across every instrument, so
`extent_estimate` returns *unresolved* for breath rather than a number. Respiratory rate and
inter-breath interval, both derived from onsets, are supportable. Inspiratory-to-expiratory ratio is
not, and would need an instrument nobody has.

**Grouped events, reliably.** HeAR fragmented 3 of 4 verified events and CrisperWhisper split cough 2
into two mislabelled tokens. `group_events` is declared with no solution so the gap fails at graph
build rather than hiding inside whichever consumer notices first.

`rise_ms` and `level_step_db` travel as continuous features carrying uncertainty, never as thresholds:
they are n=2 from one healthy adult and encode a *healthy adult voluntary* cough, which reduced peak
cough flow, absent glottic closure and infant cough all move.

## The measurement that would settle the proposer choice

Six verified windows and roughly 8.5 s of verified-empty audio in the same file (0-0.78, 1.0-2.3,
3.5-5.3, 6.3-7.9, 8.5-9.6, 10.25-11.65, 13.2-14.03). So a proposer can be scored on recall over the six
and **false positives per minute** over the empty stretches, on the original and on degraded copies with
added noise and reverberation. That answers the robustness question the current draft can only assert.

---

## Correction: four dangling input ports, and what Brouhaha is actually doing

The table above consumes four products that **no node in it produces**:

| product | consumed by | producer |
| --- | --- | --- |
| `crisper_tokens` | `span_refine`, `group_events` | **none** |
| `c50_db` | `attribute_sources` | **none** |
| `rms_db` | `attribute_sources` | **none** |
| `community1_seg` | `attribute_sources` | **none** |

They had producers in the first draft — a `crisper_tokens` node and a `vad_pair` node — and both were
dropped when the ordering was restructured to propose / corroborate / refine / reconfirm. This is the
F-187 failure exactly: a consumer reading a product nothing writes. It was caught here by the port
declaration itself, in a table, before any code existed, which is the reason for declaring ports.

**Brouhaha's role in this branch is one derived quantity, not its VAD.** Only `c50_db` — reverberation,
as proximity evidence for attribution — and even that had no producer. The vocal-versus-lexical
discriminator (Brouhaha's VAD against community-1's segmentation) answers *which branch a file belongs
to*, which is a TAXONOMY question upstream of here. It does not belong in the airway branch, and its
appearance in this table was a leftover.

## Correction: the grouping rule has no detector behind it

`group_events` was described as solvable by a physiological rule — a voiced phase within ~400 ms of a
broadband burst belongs to that burst. **There is no instrument here for "voiced phase".** Measured on
this recording, praat HNR returns `nan` almost everywhere, valid only at the two cough onsets, and pyin
rails at its 60 Hz floor through every quiet stretch, locking onto low-frequency rumble. So the cue the
rule depends on is not detectable by anything currently in the graph.

Two candidates exist in senselab and neither has been tested for this: `features_extraction/ppg.py`
(phonetic posteriorgrams — frame-level phone posteriors, where a voiced cough phase might read as a
vowel with characteristic entropy) and `features_extraction/sparc.py` (articulatory inversion). Both are
speech-trained, so their behaviour on non-speech events is unknown.

A second problem the rule ignores: **grouping consumes sub-event labels, and those degrade in noise**.
Every label measured in this project comes from one quiet close-miked recording. How `hear_classify` and
`span_reconfirm` behave on noisy or reverberant input is unmeasured, so the inputs to grouping are of
unknown quality exactly where grouping is hardest.

`group_events` therefore remains unsolved, and the earlier suggestion that it was "not a modelling
problem" was wrong — it needs either a cue detector that works on non-speech, or a different
formulation.

---

## `span_refine`'s CrisperWhisper input, after the revision finding

`span_refine` was specified to take `crisper_tokens` as span-edge candidates, on the strength of
timings that matched the verified windows closely — cough 1 bounded to within 26 ms at onset and 14 ms
at offset. Two things now qualify that.

**The producer does not exist in this table.** `crisper_tokens` is consumed by `span_refine` and
`group_events` and produced by nothing, along with `c50_db`, `rms_db` and `community1_seg`. Whatever is
decided below, a producing node has to be declared.

**Which model produces it is now a deliberate choice, not a default.**
`nyralabs/CrisperWhisper2.0_turbo` was retrained upstream on 2026-08-17, and the two revisions differ in
exactly the output this node depends on:

| revision | non-speech tokens on the probe recording |
| --- | --- |
| `831f87e1` (2026-08-03) | `[breath] [breath] [cough] [UH] [breath]` |
| `de0369c8` (2026-08-17, "2.1-generation two-stage retrain") | `[cough]` |

Both recover the speech identically. So tracking `main` gives a model that does not annotate breath at
all, and every earlier measurement of this node's usefulness was made against the pre-retrain weights,
reached through a stale cached ref.

**Consequences for the node as designed:**

- For **cough**, the node stands on either revision — `[cough]` is the one token both emit, and it was
  the best-bounded span of anything measured.
- For **breath**, the node has an input only if the branch pins `831f87e1`. That means deliberately
  running a superseded model, which needs its own justification: the newer weights may well be better at
  what they were retrained for, and the older model's `[UH]` is a mislabelled cough, so its extra tokens
  are not uniformly more correct.
- If the branch tracks the current model instead, **breath has no span source at all** — not the
  envelope (10-52% coverage of the verified windows), not HeAR (fragments, 24-52%), and not
  CrisperWhisper. That is not a gap awaiting a better tool; it is unmeasured by everything tried, and it
  is what makes the "no breath duration" restriction structural rather than provisional.

Either way the node must **pin a revision explicitly** rather than resolve `main`, and record which,
because the choice determines whether one of its two consumers has any input.
