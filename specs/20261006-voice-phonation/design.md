# Voice tasks decided on the phonation attempt

2026-10-06. This covers the six VOICE families: `maximum-phonation-time`, `-v2`,
`prolonged-vowel`, `glides-low-to-high`, `glides-high-to-low` and `high-to-low`. There are 8,305
recordings in r16. The owner asked for "equivalent checks and evaluations for the voice branch
including how clef performed", then listened to 41 recordings (34 sampled across flag keys plus 7
discards traced to a cause). The design below is implemented; what the implementation does where it
differs from the design, and how it reads on the labels, is in "Implementation" at the end.

## What r16 does and why it fails

r16 kept 5,855 of the 8,305 VOICE recordings as pass, flagged 3,164 (38 %), discarded 181 and held
105 for rerun. Of the flags, 2,471 carry a single key:

| Only key | n |
|---|---|
| `conformance:VOICE` (glide dominant fraction or range; prolonged-vowel declared duration) | 1,228 |
| `gate:f0_spread_max_semitones` | 615 |
| `gate:voiced_fraction_min` | 536 |
| `route_mismatch:VOICE` | 57 |

The bounds on those keys are in `data/config/default.yaml` under `verdict.gates.by_group`:
voiced fraction 0.5, f0 spread 2.0 st, dominant segment 0.5, glide range 6.0 st, and declared
duration 0.5 of 12 s. The glide range carries the label UNFITTED. None of these bounds was fitted
against labelled verdicts.

Every failure below traces to one of two things: the span finder working on the pre-emphasised
stream, or the F0 track computed only on that stream.

### Discards on `declared_task_absent` (130 recordings)

VOICE proposes no carrier, so it reports the task absent and VERDICT discards
(`vocabulary.py:2120`). `hint_mismatch:VOICE` follows from that and is not a cause. There are two
causes:

- **A: the floor is the phonation (50 recordings).** PREPROCESS sets one floor per recording at
  the 5th percentile of the pre-emphasised envelope (`nodes/preprocess.py:1753`,
  `floor.percentile: 5.0`). An amplitude span opens 6 dB over that floor (`spans.k_db: 6.0`). When
  the vowel fills the file, the floor is the vowel. In 685fb824 the median envelope sits 2.0 dB over
  the floor, so the only spans are 0.08–0.46 s fragments. Every one of them falls under
  `production_min_s: 0.5` (`nodes/voice.py:414`; glides `:778`). The F0 track there is fine, with
  2,107 voiced frames.
- **B: the stored F0 track is unvoiced (80 recordings, 49 participants).** F0 is tracked only over
  the pre-emphasised stream (`nodes/preprocess.py:1414`). A carrier exists (c35284b1: 1.49–6.55 s),
  but it has no frame at `voiced_strength_min: 0.45`, so it is rejected as `no_voicing`
  (`nodes/voice.py:430`; glides `:788`).
  - Praat cc re-run over the same per-recording F0 range gives a median of 0 voiced frames on the
    pre-emphasised stream and 285 on plain.
  - The silence threshold is not the cause: the result is unchanged at 0.
  - The F0 range is not the cause: c35284b1 runs 50–248 Hz, 68de3829 70–286 Hz.
  - These are low, hoarse or rough voices (plain median F0 131 Hz, p10 72 Hz), so the failure
    follows the speaker. Every voice task of c35284b1 and 68de3829 is discarded.

On plain voicing, 65 of the 130 hold ≥ 3 s voiced and 94 hold ≥ 1 s. The 36 under 1 s were called
empty on voicing. That count is wrong by the owner's rule below (an unvoiced attempt is still an
attempt), so it must be recounted on energy. The 51 `too_short_for_task` discards are 0.21–0.81 s
files and are correct.

### Extents on kept recordings

- **Glides.** The task extent is the longest monotone run of the smoothed stored F0
  (`nodes/voice.py:792`, ranked by duration whatever the declared direction, `:815`). The run breaks
  at any reversal beyond `sweep_reversal_tolerance_semitones: 2.0`
  (`nodes/branches.py:2218`, `:2241`) and skips non-finite frames, so it bridges unvoiced gaps.
  Every tracker octave error against the glide's direction is therefore an extent boundary:
  - b442b86a: the extent starts at a 296→586 Hz jump and ends at a 113→228 Hz jump;
  - 028943cf: the same at 97→148 Hz and 130→247 Hz;
  - 5b4817c2: the winning run crosses an unvoiced gap from the glide's tail to an unrelated event
    at 3.1 s.
- **Rising glides lose their start.** Pre-emphasis lifts the envelope as pitch rises, so the low
  opening stays under +6 dB. In 7ae91a65 the only long carrier is the top, 7.31–12.31 s. Its length
  of exactly 5.00 s is a coincidence of the walk; there is no tail window.
- **Sustained families.** The longest carrier is trimmed to its first and last frame voiced on the
  stored track (`nodes/voice.py:563`, `voiced_extent` at `nodes/branches.py:2126`). A breathy or
  hoarse stretch therefore ends the extent: 5e68a500 ends at 6.45 s, while the vowel runs to 10.9 s.
- **Octave errors.** The plain and stored tracks both halve or double on many recordings. f0
  spread then reads 8–12 st on a vowel that varies by 1 st (74a221d7, 3856c95a, ad6bfe11). A glide
  starting at 1150 Hz (3b06709b) is tracked at half its pitch.

### Route mismatch (183 recordings)

The ruleset routes VOICE when any of three gates fires (`taxonomy.ruleset.gates`):

- `voice.sustained`: the longest amplitude span is ≥ 3.0 s;
- `voice.glide`: the YAMNet singing-subtree peak on plain is ≥ 0.05;
- `voice.chant`: the YAMNet Chant peak is ≥ 0.02.

A glide of about 2 s, or a short MPT, clears none of them. 359 VOICE recordings were declined.
VOICE ran on all of them anyway, because the declared family names it, and found its task on 183:

| Family | n |
|---|---|
| glides-high-to-low | 79 |
| glides-low-to-high | 46 |
| maximum-phonation-time | 43 |
| maximum-phonation-time-v2 | 15 |

There are none on prolonged-vowel, whose holds exceed 3 s. VERDICT flags "declined and found"
(`vocabulary.py:2059`). On a declared voice task, that flag says only that the router cannot
recognise a short phonation.

### Clef on VOICE

The production second-opinion run covered only the 319 voice recordings sent to lexical review. A
spectrogram probe on 325 recordings (asked blind, then with the task) showed:

- **Glide direction echoes the instruction.** It matched the declared direction 49 % of the time
  blind and 87 % informed. On 60 glides with no measurable travel it still answered falling 23
  times and rising 19.
- **`pitch_jumps` and `other_sound` never discriminate.** Neither reached 0.5 on any recording.
- **`voicing_breaks` is weak.** It caught one of three owner-labelled breaks (6b0c36f0 at 0.87;
  74a221d7 0.33, 4464b02f 0.20).
- **`voice_at_end` separates.** It reads 0.60 on vowels still sounding at the file end, against
  0.15 otherwise.

Clef stays out of VOICE decisions. `voice_at_end` may be carried as context on the shutoff and
truncation readings.

## Owner rules (listens, 2026-10-06)

- **Disordered voicing is data, not a flag.** That covers rough, breathy, hoarse, creaky and
  glottalized voicing, register shifts, tremor and unsustained voicing. On 3856c95a high-to-low the
  owner said: "task done, voicing not sustained (that's ok) there are many disordered voices". On
  709b34e1: "glide goes through many phonation changes (ok for disordered voices)".
- **A weakly voiced attempt is included.** On 3c6a97e9 and dbd096ea: "these unvoiced ones are fine
  to include. and extents should be marked."
- **The extent is the attempt, measured from energy continuity.** Voicing does not decide whether
  an extent exists.
- **Discard only when nothing rises above the floor.** On 889afc08: "no audio".
- **The extent starts at the preparatory inhale.** This came up on fa6befa4, e421c733, a9500875,
  ad6bfe11 and 8cec2c76.
- **A held vowel or a wrong-direction glide in a glide task is a `task_mismatch` annotation.** This
  is the airway convention (e0b4d428).
- **Route mismatch is an annotation when the owning branch found its task.** The owner asked "why
  is this flagged?" on 63779be0, a real 1.5 s MPT.
- **Leading speech is detected and annotated with its ASR words.** It was missed on 74a221d7,
  3856c95a, 4464b02f, 840041f9 and 6b7bd347, often as three short voiced segments, likely a count-in.
- **Breaks are measured; brief dips are not breaks.** ad6bfe11 has 20+ dips of ≲ 0.15 s and was
  heard as one hold. 6b0c36f0 (≈ 0.5 s), 4464b02f (0.4–1 s) and 74a221d7 (≈ 1.3 s) are breaks.
- **A microphone shutoff during phonation makes MPT a lower bound and flags for review.** In
  f47eeda7 every band drops ≈ 50 dB to a flat digital floor at 12.95 s while voicing continues.
  The owner confirmed "flag". A shutoff after the task has ended is an annotation.
- **Broken or restarted holds merge into one extent.** The extent runs from the first onset to the
  last offset; a break or a restart is not a flag (4464b02f's five restarted holds, 74a221d7's
  break). Owner: "broken/restarted should be seen as merging spans".
- **There is no minimum hold.** Owner: "i don't think there is a good minimum." Hold duration is a
  measurement, not a ground.

## Design

All of this reads stored derivatives except where noted. Every threshold goes in `data/` with its
derivation from the labels below; none is a code literal.

1. **Phonation extent on plain (raw).**
   - Energy continuity is subband energy at least 10 dB over the floor, plus harmonic structure
     where any exists, as the breath train uses. Pitch-tracker strength alone is not voicing: it
     counts mains hum as voiced (item 10).
   - The floor is estimated outside the phonation: from leading and trailing quiet when they exist,
     otherwise from the session group's floor, as in the group-relative dBFS work. It is never the
     recording's own 5th percentile.
   - The preparatory inhale attaches to the start, run back to 8 dB over the floor as in the cough
     measure.
   - The extent ends at the energy offset.
   - Discard (`no_phonation_captured`) only when no extent is found on energy.
2. **F0 on plain, over about 50–1600 Hz, with an octave check against the harmonic ridge.** A track
   at half or double the lowest strong ridge is corrected. PREPROCESS gains the plain track. That is
   a stage change, so its version is bumped.
3. **Glides: net direction and range over the whole extent.**
   - The extremes are taken in the declared direction. A held start, an initial counter-move
     (1c139a67), a register jump in the declared direction (445c8cdf, 709b34e1) and a return at the
     end (028943cf) are recorded shape, not boundaries.
   - Range under the bound, or the wrong net direction: `task_mismatch`, with the shape named (for
     example "held 190 Hz for 6.1 s").
   - The 6 st bound is refitted on the labels.
4. **Hold and break segmentation, merged into one extent.**
   - Holds separated by breaks or restarts merge: the extent runs from the first onset to the last
     offset. A break is never a flag.
   - A break is a fall to the floor longer than a minimum, fitted from the labels: between 0.15 s
     (ad6bfe11's dips) and 0.4 s (4464b02f's shortest gap). Brief dips under it are part of the
     hold.
   - Readings: number of holds, each hold's duration, the longest hold, total voiced time, break
     durations. All are measurements.
   - There is no minimum hold. The prolonged-vowel declared-duration conformance
     (`declared_duration_min_fraction`, 0.5 of 12 s, 585 single-key flags in r16) is removed as a
     ground; 4.55 s (6b7bd347) and 5.35 s (840041f9) holds were heard as good. No duration ladder is
     needed.
5. **Voice quality as annotations.** Voiced share, f0 spread, register shifts, roughness and creak
   are read inside the extent. `f0_spread_max_semitones` and `voiced_fraction_min` stop being flag
   gates.
6. **Leading and trailing speech.** Speech-like voiced runs outside the extent are found
   acoustically, and the ASR words over them are attached. They are annotated outside the extent and
   flagged only if they overlap it.
7. **Shutoff detection.** Every band drops abruptly to a flat floor well under the noise floor, the
   same signature as the truncated-capture check.
   - Within phonation, or within a short gap of its end: MPT is reported as a lower bound and the
     recording flags `capture_cut_during_task`.
   - Otherwise it is an annotation.
8. **A narrow low-confidence review band.** The decision is computed again at strict and lenient
   settings (floor margin, break minimum, glide bound). Where the two disagree, the recording flags
   `voice_review_low_confidence`.
9. **Route mismatch as an annotation for declared voice families.**
10. **Hum guard on the residual; the extent stays on raw.**
    - 3bbc69ef has mains hum across the file: the plain track locks onto 60 Hz, the stored track
      onto 120 Hz, voiced fraction reads 1.00, and the r16 extent sits on hum at 7.23–8.1 s. With
      the item 1 energy condition the raw extent is 3.93–4.28 s, on the glide.
    - Guard: at least 3 mains lines at multiples of 50 or 60 Hz in the residual, each 10 dB over
      its 2–8 Hz neighbourhood. On 36 hum-locked and 30 clean VOICE recordings it caught 31/36 hum
      and 0/30 clean. At 2 lines it catches 34/36 but also 1 clean. An f0-lock test (half the
      voiced frames within 1 Hz of 50/60/100/120 Hz) catches 25/36 and 0/30 clean; line-level
      stationarity does not separate the two groups.
    - When the guard fires, f0 frames within 1 Hz of a mains multiple are dropped, and the extent
      is compared with the enhanced-stream extent. A disagreement over 1 s flags
      `voice_review_low_confidence` rather than choosing one.
    - The enhanced stream is not the default. On 7cd07b02 its extent ends at 3.37 s against about
      8.25 s on raw, because the enhancer moved the quieter later voice into the residual; on 1 of
      30 clean recordings it shortened the extent from 10.2 s to 7.1 s. fa6befa4 (the control)
      gives the same extent on both streams.
    - An enhanced level 20–33 dB under raw is a supporting "nothing captured" signal. It held on
      the 4 hum-only recordings (6e17edae, a28e5022, 1ec7ddad, 9de82d6f), whose r16 discard stands.
    - The 36 hum-locked recordings come from about 15 participants who recur across tasks.

Task-audio cuts follow the new extent, passes included. 028943cf passed with an extent missing the
start and the return of its glide.

## Labels (owner listens, r16 stores)

| Subject | Family | r16 | Heard |
|---|---|---|---|
| fa6befa4 | prolonged-vowel | flag voiced_fraction | task present; extent misses inhale and onset |
| 74a221d7 | prolonged-vowel | flag f0_spread | two holds with a break; leading speech; extent second half only |
| 3856c95a | prolonged-vowel | flag f0_spread | leading speech uncaptured; vowel extent right |
| 4464b02f | prolonged-vowel | flag duration | five holds restarted; leading speech |
| 840041f9 | prolonged-vowel | flag duration | 5.35 s hold good; leading speech |
| 6b7bd347 | prolonged-vowel | flag duration | 4.55 s hold good; leading speech |
| b7990104 | prolonged-vowel | flag conformance | ok |
| 6112f3a9 | prolonged-vowel | flag conformance + f0_spread | ok |
| 164584f3 | prolonged-vowel | pass | ok |
| a9500875 | mpt-v2 | flag f0_spread + voiced_fraction | rough long hold good; inhale uncaptured |
| e421c733 | mpt-v2 | pass | vowel good; inhale uncaptured |
| 6b0c36f0 | mpt-v2 | flag voiced_fraction | breaks with a 4 st step; extent ends mid-segment |
| ad6bfe11 | mpt-v2 | flag f0_spread | inhale uncaptured; brief dips are one hold |
| 85a22592 | mpt-v2 | flag f0_spread | ok |
| 685fb824 | mpt-v2 | discard | whole-file vowel; not discard (cause A) |
| 68d62b1e | mpt-v2 | discard | performed (cause A) |
| f47eeda7 | mpt-v2 | discard | long phonation then mic shutoff (cause B) |
| 889afc08 | mpt-v2 | discard | no audio; discard correct |
| 5e68a500 | mpt | flag voiced_fraction | vowel to 10.9 s; extent ends where tracking is lost |
| c35284b1 | mpt | discard | low-pitched phonation; not discard (cause B) |
| 0581355f | mpt | discard | hoarse, performed (cause B) |
| eaa742f4 | mpt | discard | performed (cause A) |
| 63779be0 | mpt | flag route_mismatch | real 1.5 s MPT |
| ebd757bf | mpt | pass | ok |
| 3b06709b | high-to-low | flag voiced_fraction | glide from 1150 Hz; trackers at half; extent middle third |
| 3856c95a | high-to-low | flag dominant | task done, unsustained voicing ok; extent short |
| 028943cf | glides-low-to-high | pass | extent misses start and return |
| 8cec2c76 | glides-low-to-high | pass | inhale uncaptured; vowel extent good |
| 68de3829 | glides-low-to-high | discard | task done (cause B) |
| 3bbc69ef | glides-low-to-high | flag glide range | mains hum; short glide; extent on hum |
| 7ae91a65 | glides-low-to-high | flag glide range | 15 st rise; extent its flat top |
| 445c8cdf | glides-low-to-high | flag dominant | register shift mid-glide; extent part of it |
| 3c6a97e9 | glides-low-to-high | discard | weakly voiced, creaky; include (cause B) |
| 5b4817c2 | glides-high-to-low | flag route_mismatch | 12 st fall; extent tail plus a later event |
| 2558e9da | glides-high-to-low | flag route_mismatch | 11 st fall; extent trims the end |
| e0b4d428 | glides-high-to-low | flag glide range | held, no glide; extent half |
| 1c139a67 | glides-high-to-low | flag glide range | inhale and an irregular 19 st fall; extent the tail |
| 709b34e1 | glides-high-to-low | flag dominant | 30+ st fall through phonation changes |
| b442b86a | glides-high-to-low | flag dominant | as 709b34e1 with a glottalized tail |
| dbd096ea | glides-high-to-low | discard | breathy, weakly voiced; include (cause B) |
| 2f122abe | glides-high-to-low | discard | performed (cause A) |

The owner heard 11 `declared_task_absent` discards; 10 were performed tasks, and only 889afc08 was
empty. Five recordings were heard as "ok" with no further note.

## Decided (owner, 2026-10-06)

- **No minimum hold.** Hold duration is a measurement; the declared-duration conformance ground is
  removed (item 4).
- **Broken or restarted holds merge into one extent**, not flagged (item 4).
- **A mic shutoff during phonation flags** (`capture_cut_during_task`); after the task it is an
  annotation (item 7).
- **Hum guard on the residual, extent on raw**, with the enhanced extent as a cross-check (item 10).

## Still open

- **Fitted values from the labels**: the break minimum (bracket 0.15–0.4 s) and the glide range
  bound under which a glide becomes `task_mismatch` (6 st, unfitted). These are fits, not owner
  decisions.
- **The empty count on energy.** 36 of the 130 `declared_task_absent` discards had under 1 s voiced
  on plain, but voicing is the wrong test (3c6a97e9 and dbd096ea are weakly voiced attempts). The
  count is redone on the item 1 energy extent before any discard is confirmed.

## Implementation (2026-10-06)

`voice_phonation.py` is the measure, `data/voice_phonation.yaml` its parameters. VOICE's align arm
(`nodes/voice.py:align_voice`) takes the reading, proposes the extent as its `task_extent` span,
writes the gate readings over it (`carrier_duration_s`, `carrier_voiced_fraction`,
`carrier_f0_spread_semitones`, `glide_extent_semitones`, `phonation_onset_to_offset_s`,
`voiced_duration_s`, `longest_hold_s`) and the whole reading as the `voice_phonation_reading`
measurement. VERDICT reads that measurement and nothing else of the measure
(`nodes/verdict.py:_voice_reading`), the same pattern as the AIRWAY move. The detect arm is
unchanged.

### Where it differs from the design above

- **F0 on plain is computed inside VOICE, from the stored `plain` stream, not as a new PREPROCESS
  derivative.** The replay re-decides from TAXONOMY on and runs no PREPROCESS stage; reading the
  stored stream in the branch lets the replay land the change without a reprocess. Praat cc over
  50–1600 Hz, strength 0.45, then the octave check (odd against even harmonic level on the frame
  spectrum, then a fold toward the running median).
- **The floor is the quietest 100 ms of the recording's own non-digital frames**, not a session
  floor. Cause A was the 5th percentile: a vowel filling all but 0.2 s of the file left that 0.2 s
  as the only quiet, and a percentile never sees it; the minimum of a 100 ms running mean does.
  Frames under −90 dBFS (digital silence, as in 3856c95a's opening) are excluded, or every frame
  would sit 10 dB over a −120 dB floor. The session-group floor stays a fallback to build if a
  recording with no quiet at all turns up.
- **A plain-stream voiced frame counts as phonation only 6 dB over the floor** (`voiced_rise_db`):
  Praat voices room noise at strength ≥ 0.45 (889afc08, heard as no audio, read 34 % voiced at
  2.5 dB over its floor).
- **Speech is told apart by its ASR words.** A lexical consensus word that is not spelled like the
  vowel (`^[aeiouhym]+$` after dropping non-letters: "Ah", "E", "Eeeeee"), carries Latin letters and
  lasts at most 1.5 s is speech. Its frames are taken out of the phonation mask, and no hold merges
  across one. This is what separates the count-in (one, two, three) from the vowel in every
  prolonged-vowel label, and it keeps the vowel transcribed as a word (2558e9da "E.", e421c733
  "Ah.") or in another script (68de3829 "一") as the vowel.
- **A further hold merges only if it is voiced** (≥ 0.3 of its frames) or 20 dB over the floor, so a
  soft inhale 1 s before the vowel is not a hold (63779be0).
- **The shutoff is a dead stretch, not a single drop.** A frame is dead where it is digital silence
  or sits in a 0.3 s window whose p10–p90 spread is ≤ 1 dB and 8 dB under the floor of the live
  frames; dead runs join across 0.15 s, must last 0.3 s, start after the first phonation, and be
  entered by a 30 dB fall within 0.1 s. f47eeda7's shutoff is a click, a decay through digital
  silence, then a −82 dB plateau flat to ±0.5 dB, 20 dB under its room noise; no single-drop rule
  read it.
- **`extent_min_s` is 0.3 s.** 889afc08 holds one 0.22 s bracketed "[UM]" and nothing else.
- **The review band** flags `voice_review_low_confidence` where the extent's existence differs at
  `phonation_db` ± 2 dB, where the glide's shape differs at the bound ± 1.5 st, or where the hum
  guard fired and the enhanced-stream extent's edges differ from raw's by more than 1 s.

### Gates and grounds

- `dominant_segment_min_fraction` and `declared_duration_min_fraction` are removed (gate table,
  config, recording-vectors columns, viewer); recording-vectors schema 23.
- For a declared voice family VERDICT applies no flag gate (`voiced_fraction_min`,
  `f0_spread_max_semitones`, `continuity_min`), reads no VOICE conformance or uncomputed-gate ground,
  and does not flag a Clef disagreement. The readings are still recorded in the `gate_*` columns.
- New grounds: `no_phonation_captured` (discard), `voice_review_low_confidence` and
  `capture_cut_during_task` (flags). New annotations: `task_mismatch` for a glide, `capture_cut_after_task`,
  `speech_outside_task`, and `route_mismatch:VOICE` when VOICE owns the family and found it.
- A store VOICE reported on without a `voice_phonation_reading` (every pre-change store) names
  `VOICE:voice_phonation_reading` as an absent owner input, so it reruns until it is replayed.

### Labels, before and after

"After" is the VOICE-owned part of the verdict read off the measure on the r9 stores (the replay's
other grounds — a second speaker, redaction — are not re-read here). The extent is the reading's.

| Subject | Family | r16 | Owner label | After | Extent (s) |
|---|---|---|---|---|---|
| 028943cf | glides-low-to-high | pass | task_present_extent_short | pass | 0.345–4.025 |
| 0581355f | maximum-phonation-time | discard | task_present_hoarse_voice | pass | 0.665–16.565 |
| 164584f3 | prolonged-vowel | pass | ok | pass | 0.705–12.095 |
| 1c139a67 | glides-high-to-low | flag | task_present_irregular_glide_extent_wrong | pass | 0.075–4.595 |
| 2558e9da | glides-high-to-low | flag | task_present_extent_wrong | pass | 1.425–3.755 |
| 2f122abe | glides-high-to-low | discard | task_present_whole_file | pass + task_mismatch:held | 0.015–8.965 |
| 3856c95a-g | high-to-low | flag | task_present_disordered_voice_extent_short | pass | 0.895–7.185 |
| 3856c95a-pv | prolonged-vowel | flag | task_present_leading_speech | pass + speech_outside_task | 3.965–11.165 |
| 3b06709b | high-to-low | flag | task_present_extent_short_f0_tracks_off | pass | 0.405–4.945 |
| 3bbc69ef | glides-low-to-high | flag | task_short_background_hum | flag (voice_review_low_confidence) | 3.925–4.625 |
| 3c6a97e9 | glides-low-to-high | discard | task_present_weakly_voiced_include | pass | 0.805–8.895 |
| 445c8cdf | glides-low-to-high | flag | task_present_register_shift_extent_wrong | pass | 1.455–12.275 |
| 4464b02f | prolonged-vowel | flag | task_present_broken_hold_leading_speech | pass + speech_outside_task | 1.885–12.105 |
| 5b4817c2 | glides-high-to-low | flag | task_present_extent_wrong | pass | 0.355–3.355 |
| 5e68a500 | maximum-phonation-time | flag | task_present_extent_short_rough_voice | pass | 0.505–10.875 |
| 6112f3a9 | prolonged-vowel | flag | ok | pass | 5.045–10.655 |
| 63779be0 | maximum-phonation-time | flag | task_present_route_mismatch | pass | 2.105–3.685 |
| 685fb824 | maximum-phonation-time-v2 | discard | task_present_whole_file | pass | 0.045–21.305 |
| 68d62b1e | maximum-phonation-time-v2 | discard | task_present_whole_file | pass | 0.605–38.045 |
| 68de3829 | glides-low-to-high | discard | task_present_no_extent | pass + task_mismatch:opposite | 1.125–8.215 |
| 6b0c36f0 | maximum-phonation-time-v2 | flag | task_present_breaks | pass | 1.625–19.725 |
| 6b7bd347 | prolonged-vowel | flag | task_present_duration_good_leading_speech | pass + speech_outside_task | 3.425–9.455 |
| 709b34e1 | glides-high-to-low | flag | task_present_disordered_voice_extent_short | pass | 0.015–5.905 |
| 74a221d7 | prolonged-vowel | flag | task_present_broken_hold_leading_speech | pass + speech_outside_task | 1.425–11.255 |
| 7ae91a65 | glides-low-to-high | flag | task_present_extent_wrong | pass | 0.555–12.285 |
| 840041f9 | prolonged-vowel | flag | task_present_duration_good_leading_speech | pass + speech_outside_task | 4.025–9.775 |
| 85a22592 | maximum-phonation-time-v2 | flag | ok | pass | 0.025–6.745 |
| 889afc08 | maximum-phonation-time-v2 | discard | no_task_no_audio | discard (no_phonation_captured) | – |
| 8cec2c76 | glides-low-to-high | pass | task_present_inhale_uncaptured | flag (voice_review_low_confidence) | 1.405–9.125 |
| a9500875 | maximum-phonation-time-v2 | flag | task_present_inhale_uncaptured_rough_voice | pass | 0.065–19.985 |
| ad6bfe11 | maximum-phonation-time-v2 | flag | task_present_inhale_uncaptured | pass | 1.585–16.195 |
| b442b86a | glides-high-to-low | flag | task_present_glottalized_tail_extent_wrong | pass | 0.765–10.275 |
| b7990104 | prolonged-vowel | flag | ok | pass + speech_outside_task | 1.225–5.315 |
| c35284b1 | maximum-phonation-time | discard | task_present_no_extent | pass | 1.555–5.855 |
| dbd096ea | glides-high-to-low | discard | task_present_weakly_voiced_include | pass + speech_outside_task | 1.325–7.525 |
| e0b4d428 | glides-high-to-low | flag | held_not_glide_extent_wrong | pass + task_mismatch:held | 0.225–6.435 |
| e421c733 | maximum-phonation-time-v2 | pass | task_present_inhale_uncaptured | pass + speech_outside_task | 2.575–12.975 |
| eaa742f4 | maximum-phonation-time | discard | task_present_whole_file | pass | 0.175–41.455 |
| ebd757bf | maximum-phonation-time | pass | ok | pass | 0.405–14.355 |
| f47eeda7 | maximum-phonation-time-v2 | discard | task_present_mic_shutoff | flag (capture_cut_during_task) | 0.085–13.015 |
| fa6befa4 | prolonged-vowel | flag | task_present_extent_short | pass + speech_outside_task | 3.375–11.165 |

Of the 41: the owner heard a performed task on 40 and no audio on one. After the change all 40 are
kept (37 pass, 3 flag) and 889afc08 is the one discard; r16 discarded 11, 10 of them performed. The
three flags are review-band or shutoff cases the design asks to leave for review: 3bbc69ef (mains hum,
a 0.4 s glide), 8cec2c76 (a 6.75 st rise against the 6 st bound) and f47eeda7 (shutoff during
phonation). Where the extent can be compared with the owner's times it starts within about 0.3 s,
or earlier where an inhale or breath noise before the onset joined it (1c139a67, 709b34e1,
a9500875 start near 0.0).

Over the 158 random, 30 clean and 35 hum-locked recordings in the eval set: no shutoff was read on a
random or clean recording; the four hum-only recordings (6e17edae, a28e5022, 1ec7ddad, 9de82d6f)
read no extent, as their r16 discard has it; 14 of 35 hum-locked recordings go to review on the
raw/enhanced disagreement.

### Left unfitted, or wrong on a label

- **The break minimum** stays 0.25 s and **the glide bound** 6 st (both marked UNFITTED in data/).
- **68de3829** ("task done", a rising glide) reads 3.7 st up and 6.6 st down: an `opposite`
  mismatch annotation the owner's label does not support; the F0 track there still needs a look.
- **2f122abe** ("fine") reads as a held vowel in a falling-glide task (2.0 st): annotation only.
- **5b4817c2**'s extent runs to 3.36 s, taking in the noise after the glide (owner: glide to 2.35 s).
- **The hum guard fires on 8 of 158 random recordings**, against about 36 of 8,305 found hum-locked;
  on a clean recording it only drops mains-near F0 frames, but its false-fire rate needs a look on
  the replay.
