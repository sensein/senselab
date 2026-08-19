# SpeechBrain: a separation checkpoint returned interleaved samples as audio

## The defect

`speech_enhancement/speechbrain.py` reshaped `SepformerSeparation.separate_batch`'s output with
`reshape(1, -1)`. That output is `(batch, samples, sources)`. For an enhancement checkpoint there is
one source and the reshape is correct. For a separation checkpoint it **interleaves the sources
sample-by-sample**, and senselab returned the result as audio with no warning.

Measured through `enhance_audios` on a 14.03 s recording, every reachable SepFormer separation
checkpoint: `sepformer-wsj02mix`, `-wsj03mix`, `-libri2mix`, `-whamr16k`, `-wham`, `-whamr`,
`resepformer-wsj02mix`.

| symptom | value |
| --- | --- |
| output length against input | exactly **2.000×** (3.000× for `wsj03mix`) |
| cross-correlation peak with input | **\|r\| ≤ 0.08**, at a lag of 50k-154k samples |
| ASR transcript | **empty**, for every one of the seven |

De-interleaving post hoc recovered streams matching direct-call `src0`/`src1` measurements exactly, so
nothing was lost — the output was purely mis-shaped.

## The fix

`_single_source` derives the source count from the **output shape**, not from the model name. A name
rule would misclassify any checkpoint whose id does not advertise itself; the shape is authoritative.
More than one source raises, naming the model, the count, and pointing the caller at
`source_separation`. One source has its source axis dropped; a 2-D output passes through, since
`enhance_batch` returns `(batch, samples)` with no source axis.

Raising is the correct outcome rather than a workaround: `enhance_audios` returns one `Audio` per
input, so it structurally cannot represent an N-source separation. Exposing SpeechBrain separation
under `source_separation/`, where the return type fits, is the useful follow-up and is left for the
pass that is concurrently adding ClearerVoice separation there.

## Prior work included by cherry-pick

PR #525 by Jordan Wilke is included as its two original commits, authorship preserved, rebased onto
`triage`. It replaces the previous try-one-class-then-fall-back loader selection with `_loader_for`,
mapping `"sepformer"` in the id to `SepformerSeparation` and everything else to
`SpectralMaskEnhancement`, plus a fallback for off-convention names.

**That PR identified this defect and scoped it out deliberately**, writing in `_loader_for`'s docstring
that "true multi-speaker separation checkpoints … are out of scope here, because
`enhance_audios_with_speechbrain` reshapes the model output to a single waveform and would garble
multiple separated sources." This branch adds the part they left, and does not correct them.

Two conflicts arose because `speechbrain.py` moved under three merges since June, and both were
resolved in favour of the newer code: the load now uses `source=str(snapshot_path)` — SpeechBrain's
`from_hparams` takes no `revision`, so the immutable snapshot path is how a revision is pinned — and
`run_opts={"device": device_run_opt(device)}` rather than `device.value`.

## Tests

Six, in `TestSpeechBrainSourceGuard`. Five exercise `_single_source` directly. The sixth is the one
that matters: it drives the **public** `enhance_audios_with_speechbrain` path with a stub separator
returning `(1, T, 2)`, and pre-fix it reaches a length assertion that fails with
`interleaved: 32000 samples from 16000 in (2.000x)` — pinning the defect rather than the presence of a
helper. The other five fail pre-fix on `AttributeError`, which is weaker evidence and is recorded as
such.

## Not fixed here

`mtl-mimic-voicebank` still fails, at inference rather than load: `SpectralMaskEnhancement` rejects it
(`Need hparams['compute_stft']`), the `SepformerSeparation` fallback constructs and then dies in
`separate_batch` with `AttributeError: 'ModuleDict' object has no attribute 'encoder'`. It is a
`WaveformEnhancement` checkpoint, and `speechbrain` 1.1.0 also ships `SGMSEEnhancement`; senselab tries
neither. Routing it needs a third and fourth loader class and a decision about how they are selected —
`_loader_for`'s name mapping would need extending, and a name rule is exactly what this branch avoided
for the source count. Left as a separate change rather than half-done alongside a guard.
