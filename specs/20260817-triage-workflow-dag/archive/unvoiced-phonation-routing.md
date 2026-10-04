# Unvoiced phonation routing — scoped design note

## Problem

`TAXONOMY` correctly derives `voice` only from PREPROCESS `phonation_spans`; it does not and must
not use YAMNet labels as a diagnostic proxy.  PREPROCESS nevertheless admitted an unvoiced span from
stable Burg F1/F2 alone.  That condition is also possible for steady broadband noise, breath/airway
material, and other non-phonatory sound, so the stated voiced/unvoiced/mixed contract was not safe
enough to reach routing.

## Decision and boundary

For a non-periodic span, retain the stable-formant continuity limb only when F1 and F2 are both
resolved and narrow enough under `phonation_spans.unvoiced_max_formant_bandwidth_hz`.  This is an
acoustic screening condition for stable resonant structure, not a diagnosis of vocal-fold impairment
and not a YAMNet-derived class.  Periodic F0 continuity remains eligible without this extra limb.

The bandwidth limit is a new **null** configuration value: no corpus has fitted a clinically valid
limit, so the packaged configuration must not silently choose one.  A deployment that supplies it
can admit sufficiently sustained aperiodic, resonance-bearing production; broad/noisy or unresolved
formant tracks do not become phonation spans.

## Timed-word acoustic path

Consensus words provide only time boundaries.  PREPROCESS evaluates the same periodic-or-resonant
acoustic evidence inside each bounded segment and may write a `phonation` span with
`member: word_aligned`, linked to the word entity and the analysed stream.  Word text, word identity,
and any presumed vocal-fold state are not inputs.  The path is positive-only and complementary: no
words, rejected words, or a missing consensus never remove a sustained span.  Existing VOICE code can
consume this span representation without a second subject type; an overlapping sustained/glide span
suppresses the redundant word-aligned copy.  Its evidence fraction is the null, fitted key
`phonation_spans.word_aligned_min_evidence_fraction`.

## Checklist

- [x] Keep TAXONOMY and routing as consumers of `phonation_spans`; do not add a YAMNet voice rule.
- [x] Add a configurable, provenance-recorded non-periodic formant-quality condition.
- [x] Keep voiced and mixed production handling compatible with VOICE's no-period-mark path.
- [x] Add positive-only acoustic assessment over timed consensus-word segments without reading word text.
- [x] Cover an unvoiced chain through TAXONOMY, routing, and VOICE plus silence/noise/short controls.
- [x] Document that this is an acoustic screening rule, not a clinical diagnosis or validated measure.
