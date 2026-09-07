# Benchmarks

Every measurement behind a parameter or a rule in the design files. The design files state values;
these state where the values came from, what was rejected, and what the numbers do not license.

**Reference recording** for all of these unless stated: `streaming-audio-2026-07-30T04-21-56-487Z.wav`,
14.03 s, six scored labels and one unresolved region. Its labels and SHA-256 are in
[`../ground-truth-2026-08-18.json`](../ground-truth-2026-08-18.json), read via `../labels.py`.

**Standing limit on all of it.** One recording, one healthy adult, close mic, 1.58 s of speech, six
events. These justify the *shape* of a rule. Every constant needs more files.

| file | what it settles |
| --- | --- |
| [`spans.md`](spans.md) | the propose / onset / offset rules, and why the anchor is local and absolute |
| [`snr.md`](snr.md) | how far into noise span extraction survives, and the per-consumer `K` |
| [`hear-yamnet.md`](hear-yamnet.md) | how each classifier is fed, and what it reports where there is nothing |
| [`squim.md`](squim.md) | why quality is measured per span and never per file |
| [`diarization.md`](diarization.md) | pyannote's count against its spans |
| [`separation.md`](separation.md) | MossFormer separation and enhancement survival across SNR |
| [`separation-buzz-2026-09-06.md`](separation-buzz-2026-09-06.md) | seven models against a continuous buzz; why unasdiff is not a downstream backend, and why sepformer-wham16k must not be a default |
| [`open.md`](open.md) | **what is still unmeasured**, and what died with the round-based workflow |
| [`preprocess-params.md`](preprocess-params.md) | pre-emphasis, the two spectrograms, spectral continuity — which spectrogram feeds it, why the gate is a rank cut, and why IQR was the wrong metric — clip spans and what 50/30 does not catch, the envelope filter, the sample rate |
| [`taxonomy.md`](taxonomy.md) | the screening set: what each detector contributes, and what it is barred from |
| [`voice.md`](voice.md) | the phonation gate's interval, why the product is periods, the two boundary facts |
| [`disruptions.md`](disruptions.md) | what stream disruptions are measured on, and the local discontinuity reference |
| [`span-fill-recovery-2026-09-08.md`](span-fill-recovery-2026-09-08.md) | controlled ground-truth test of `span_yamnet_input`'s periodic fill: does trimming and refilling a native ≥0.96s span recover its label, and down to what duration — uses the b2ai 10-subject batch, not the single reference recording above |
| [`residual-without-speech-2026-09-08.md`](residual-without-speech-2026-09-08.md) | whether `original - g*FRCRN_SE_16K(original)` still means "background" with no speech present — eight b2ai `Respiration-and-cough-*` recordings across three subjects plus two `Story-recall` comparators; not the single reference recording above |

Scripts that produced these live in [`scripts/`](scripts/). They read the labels through
`../../labels.py`, so ground truth has exactly one owner in this repository.
