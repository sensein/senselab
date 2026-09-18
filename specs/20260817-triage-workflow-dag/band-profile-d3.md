# D3 `band_profile` — the decisions behind the build

The specification is
[preprocess-derivatives-for-expected-patterns.md](preprocess-derivatives-for-expected-patterns.md)
§ D3. The values and their derivations are in
[config-derivations.md](config-derivations.md#band_profile). This file is the rest: what was
verified before building, what was decided, and what was deliberately not claimed.

## 1. "No existing derivative serves this" — verified, not assumed

The spec asserts it; three checks confirm it against the tree as it stands at `f49f902b`.

**Everything spectral is post-resample.** `preprocess`'s conditioning step resamples to
`resample.target_hz` before anything spectral runs, and every spectral block downstream reads
`plain` or `preemphasised`: `spectrogram_wideband`, `spectrogram_narrowband`, `continuity_trace`,
`gammatone` (whose top channel is a config key, `high_hz: 7800.0`, chosen under an 8 kHz ceiling),
`praat_features`, `phonation_tracks`. None of them can see above the working rate's Nyquist, and
none of them records whether the file had content there. A run at the packaged 16 kHz therefore has
no entity from which "does this file stop at 4 kHz?" is answerable.

**The declared rate is not the content's band.** ADMIT writes `sampling_rate` onto the `recording`
stream from the decoded container (`admit.py:96-103`). That is what the file says it is stored at,
which is a property of the container. An 8 kHz-sourced recording resampled to 48 kHz before it ever
reached this pipeline declares 48000 and carries nothing above 4 kHz, and no existing entity
separates it from a genuine 48 kHz capture. `test_a_band_limited_file_reports_its_content_band_not_
its_declared_rate` is exactly that file, and it bounds the answer away from 24000, 48000 and 8000.

**The one other un-resampled derivative measures something else.** `disruptions_file` is the only
block reading the `recording` stream as supplied, and it is clip runs, dropouts, discontinuities, DC
and zero-crossing rate. ZCR is the nearest thing to a band statistic in it and is not one: it is a
time-domain count that confounds bandwidth with periodicity and level, and it is reported as a
single scalar with no spectrum beside it.

## 2. Promoted to `tasks/`, not lifted into triage

`_rolloff_hz` (`audio_analysis/quality.py:156`) is the core and is a private function in a sibling
workflow. Of the two options the brief allows, this build promotes it to
`senselab/audio/tasks/band_profile/`, for four reasons:

1. Triage cannot import from `audio_analysis`, so lifting means a second independent copy of a
   cumulative-energy quantile in the tree. Two copies of one statistic drift, and the failure when
   they do is silent: two workflows reporting the same field name from different arithmetic.
2. Every other signal-processing primitive PREPROCESS uses already lives in `tasks/` —
   `gammatone`, `spectral_continuity`, `envelope`, `disruptions`, `clipping`, `spans`. Both of the
   first two were created for this workflow and put there anyway. A private copy in a node body
   would be the only exception, and the precedent is unambiguous.
3. `audio_analysis` now calls it, so the promotion is not speculative sharing: the two workflows
   report one statistic by construction, which is the property that makes triage's `rolloff_hz` and
   `audio_analysis`'s `rolloff_95_hz` comparable at all.
4. The LTAS half has no implementation anywhere in the tree and would have had to be written
   regardless. Writing it beside the quantile it shares a transform with is what makes "one STFT"
   true rather than aspirational.

**The rewire is bit-identical, so no cache bump is owed.** `audio_analysis`'s `_rolloff_hz` keeps
its own transform sizing — an analysis window there is a slice of arbitrary length, so `n_fft` is
sized to the slice rather than to a configured duration — and delegates only the quantile. Checked
against the pre-change implementation over 72 cases (four sampling rates × six lengths spanning the
256-sample floor × mono/stereo/silent): 0 mismatches, `None` cases included. `CACHE_SCHEMA_VERSION`
is therefore untouched, and no `analyze_audio_cache` entry is invalidated.

## 3. What this does and does not buy

**It does not make the nasal/oral route (A7) measurable, and nothing here claims it does.** The
design's conclusion stands unchanged and was re-read against the code before building: the
discriminating band sits largely above the 8 kHz ceiling, and the residual tilt below it is
confounded one-for-one with mouth-to-microphone geometry, which changes *with* the route by
construction — a participant told to breathe through the mouth points the mouth at the phone, one
told to breathe through the nose does not. `branch-airway.md:321` has A7 returning
`NOT_SEPARABLE_BY_THIS_DESIGN` with `band_profile` carried as an absent covariate, and
`airway.measured_route` still returns `NOT_SEPARABLE_BY_THIS_DESIGN` after this change. Nothing in
the code contradicted that reading.

What changes is one word: the covariate stops being **absent** and becomes **measured**. A route
negative on a corpus whose files stop at 4 kHz is now attributable to a band limit that was
measured, rather than recorded as "route is not measurable" for every recording alike.
`test_the_content_band_covariate_carries_the_measured_roll_off_when_one_exists` pins the covariate
carrying a real number *and* the verdict staying `NOT_SEPARABLE_BY_THIS_DESIGN`, so the two cannot
be confused later.

## 4. `verdict.tilt_max_db_per_octave` stays null

The recorded reason for the null — "the instrument this cut would be taken on does not exist yet" —
expired with this build, and the derivation has been rewritten rather than left standing. The key
stays null for a different and narrower reason: a cut needs labelled verdicts and there are none. No
threshold was invented. What would settle it is written into the derivation: recordings independently
judged occluded and not — a hand over the microphone, a phone face-down, a pocket — with the slope
measured on each, and the cut wherever the two distributions separate, if they do. The `ltas_bands`
vector and its edges are what such a slope would be taken over, which is why both ship in the
sidecar rather than the slope itself: a slope is a reduction, and which two bands it is taken
between is part of the unfitted decision.

## 5. Failure is an absence, never a fatal

Two paths, both non-fatal by the handler PREPROCESS already has (`ValueError` and `LookupError` are
cascading absences; anything else is a hard failure that still lets every remaining block run and
then raises):

- no live `recording` stream → `LookupError`, as `disruptions_file` does on the same stream;
- the file ADMIT digested is gone or no longer digests to what ADMIT recorded →
  `_check_recording_unchanged`'s `ValueError`, the guard `extend_clip_amplitudes` already uses.

A signal too short for the transform, or carrying no energy at all, is also a `ValueError` from the
`tasks/` function rather than a padded transform or a measured zero — a padded transform reports the
padding's spectrum, and a silent file has no band edge to report.

Both paths are pinned, and both mutations that would make them fatal are caught: raising a class
outside the non-fatal pair, and dropping the digest guard so an unreadable file reaches the decoder.

## 6. What the built instrument reads, measured

Four synthetic cases, packaged config (20 ms / 5 ms / 0.95 / 24 bands / 50 Hz), `tilt` fitted by
least squares over the finite `level_db` against `log2(band_centre_hz)` — the reduction
`verdict.tilt_max_db_per_octave` would cut on:

| input | declared rate | Nyquist | roll-off | n_fft | tilt |
|---|---|---|---|---|---|
| 8 kHz-sourced, stored at 48 kHz (lowpass 4 kHz) | 48000 | 24000 | **3700 Hz** | 960 | −10.27 dB/oct |
| telephone band, stored at 48 kHz (lowpass 3.4 kHz) | 48000 | 24000 | **3100 Hz** | 960 | −7.83 dB/oct |
| true full-band 48 kHz capture | 48000 | 24000 | **22850 Hz** | 960 | +0.02 dB/oct |
| full-band at the 16 kHz working rate | 16000 | 8000 | **7650 Hz** | 320 | −0.02 dB/oct |

Rows 1 and 3 are the same container and the same declared rate and read 3700 against 22850, which
is the separation no existing derivative could make. The roll-off sits a little below each nominal
cutoff because a 10th-order Butterworth rolls off gradually and the 95% cumulative point falls
inside the transition band; that is the statistic behaving correctly, not an offset to correct.

**And this table is also the argument against defaulting the tilt cut.** The band-limited rows read
−10.27 and −7.83 dB/oct on recordings that are not occluded at all — they are merely narrowband. A
hand over the microphone would also read steeply negative. So the tilt alone does not separate
"occluded" from "narrowband source", and a cut fitted without labels would call every telephone-band
recording in the corpus an occluded microphone. Any fit has to condition on the roll-off, which is
why both ship and neither is reduced away here.
