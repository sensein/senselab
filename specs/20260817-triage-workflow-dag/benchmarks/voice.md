# Voice branch

## The gate constrains an interval, not a value

| region | RMS | F0 | periodicity |
| --- | --- | --- | --- |
| sustained voicing, 3.20–3.40 s | 0.0188 | 87.4 Hz | **0.933** |
| sustained voicing, 4.40–4.60 s | 0.0161 | 88.1 Hz | **0.934** |
| quiet stretches | 0.0004–0.0007 | unstable | **0.22–0.44** |

Two observations. Any periodicity floor in `(0.44, 0.933)` separates them, and any RMS floor in
`(0.0007, 0.0161)` does too — a factor of 2.1 and a factor of 23. **A wide gap on one recording is
precisely what cannot tell you where the boundary sits on another**, so the config records the interval
and leaves the derivation empty. A midpoint would be an invented decision.

**Provenance caveat.** These periodicity figures (0.933, 0.934) come from an *unlabelled* recording
(2026-08-20), not the labelled one. On the labelled file the same timestamps read **0.558 and 0.134**.
The two files are not interchangeable and the numbers above describe only the first. An earlier note
described "sustained voicing" at those timestamps as justification for the gate — which was circular,
since the voicing was inferred from those very periodicity values.

## F0 search range serves two irreconcilable populations

A range wide enough for a low adult male fundamental (~88 Hz here) admits period-doubling artefacts; a
range narrow enough to exclude them cuts off infant and high-F0 voices. No single range serves both, so a
run whose F0 sits where the range is ambiguous is flagged rather than resolved.

## Why the product is periods and not a contour

At 87.4 Hz one glottal period is **11.44 ms**. A fixed-hop contour has already committed to a resolution
coarser than or comparable to the quantity it samples. Jitter, and shimmer from the amplitudes, are
defined *between consecutive periods* and are unrecoverable from a resampled contour — so the primary
product is a point process and the tracks are secondary.

Both F0 routes measured on the same 400 ms agree: waveform autocorrelation 87.75 Hz, and autocorrelation
along the time axis of the 5 ms-window spectrogram 86.96 Hz. See
[`preprocess-params.md`](preprocess-params.md).

## Two boundary facts

**A leaked cough is indistinguishable from a downward glide by trajectory alone.** Both show falling F0
across a short run, so trajectory shape cannot separate them and the residual's exclusion of
airway-labelled spans is doing the work instead. This is why the branch reads the other branches'
assertions rather than re-detecting.

**Breath is unvoiced, the gate rejects it, and that is correct.** Exhalations carry energy without
periodicity, so the energy-and-periodicity conjunction excludes them. A gate on energy alone would admit
every breath in the recording.

## The residual had no producer before the store

Earlier designs declared a `residual_windows` input that nothing in the graph produced — the shape of
finding F-187. With an append-only store the residual is a fold over what AIRWAY and SPEECH asserted, so
it has a producer for the first time. An unlabelled span is deliberately *not* excluded: a span AIRWAY
proposed and declined to label is exactly where unclaimed vocalic activity would sit.
