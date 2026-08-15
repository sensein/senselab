# Speaker embedding extraction


<button class="tutorial-button" onclick="window.location.href='https://github.com/sensein/senselab/blob/main/tutorials/audio/extract_speaker_embeddings.ipynb'">Tutorial</button>


## Overview

Speaker embeddings are fixed-dimensional vector representations that capture the unique characteristics of a speaker's
voice, allowing for tasks such as speaker identification, verification, and diarization.

Speaker embedding extraction is a crucial task in speaker recognition systems. It involves transforming variable-length
audio signals into fixed-size vector representations that encapsulate speaker-specific information while being robust
to variations in speech content, background noise, and recording conditions.

## Model Architecture:
The default model used in this module (speechbrain/spkrec-ecapa-voxceleb) is based on the ECAPA-TDNN architecture,
which has shown strong performance across various speaker recognition tasks.
Other supported models include ResNet TDNN (speechbrain/spkrec-resnet-voxceleb) and
xvector (speechbrain/spkrec-xvect-voxceleb).

**Note**: Performance can vary significantly depending on the specific dataset, task, and evaluation protocol used.
Always refer to the most recent literature for up-to-date benchmarks.

## Two entry points

- **`extract_speaker_embeddings_from_audios`** returns one embedding per input `Audio`, whole-file. Reach for this
  when the caller already knows each file is a single, clean utterance from one speaker and only wants the vector --
  e.g. scoring a verification pair, or feeding a downstream classifier that expects exactly one vector per file.
  It makes no windowing decision and reports no statistics: it is the plain per-audio extractor everything else in
  this module is built from.
- **`estimate_speaker_embedding_from_audios`** takes a *set* of files that may contain the target speaker, windows
  and embeds each one, pools the windows, and returns a `TargetSpeakerEmbedding` carrying both the centroid and the
  `EmbeddingDistribution` it was estimated from. Reach for this for profile enrollment: several recordings of one
  person, of unknown per-file purity, where the caller wants both an estimate and the evidence for how well-supported
  it is. It decides nothing about whether the input set was clean -- see "Contamination rejection" below for the one
  place a decision is made, and only on request.

## Two window settings, two separate measurements

Windowing appears in two places in this task family, tuned for two different jobs, and the two settings must not be
collapsed into one "the" default:

- **Detection / temporal resolution** -- `windowing.extract_per_window_embeddings`'s own fallback defaults,
  `window_s=1.0, hop_s=0.5`, exercised only when a caller passes nothing. Neither production caller does:
  `audio_analysis`'s `data/run_config/default.yaml` runs the workflow at `window_s=0.5, hop_s=0.25`, and that file's
  own derivation measured 1.0 s as *worse* for this job -- ARI 0.70 at 0.5 s against 0.48 at 1.0 s on a 4-speaker
  validation clip, because a 1.0 s window straddles turn boundaries. See
  `workflows/audio_analysis/embeddings.py`'s "Why 1.0 s / 0.5 s defaults" section for the function-level rationale
  behind the fallback pair, and `default.yaml`'s own derivation comment for the measurement that supersedes it in
  every real run.
- **Profile enrollment / centroid stability** -- `estimate_speaker_embedding_from_audios`'s defaults,
  `window_s=2.0, hop_s=1.0` (`PROFILE_WINDOW_S`, `PROFILE_HOP_S` in `api.py`). Measured directly for this job: a
  2.0 s window on a 1.0 s hop gave cross-file centroid stability 0.890 and cross-subject separation 0.168, against
  0.331 for a finer 0.5/0.25 grid carrying four times the windows. Finer windows do not make a better centroid here
  -- they make a noisier one, because each window is a shorter, less certain sample of the same speaker.

Both defaults can be overridden by the caller; neither is "the" senselab default, because they were fitted for
different trade-offs (temporal resolution against centroid stability), not for the same objective at different
resolutions.

## Reading the statistics block against its analytic nulls

`EmbeddingDistribution` (`senselab.utils.tasks.embedding_distribution`) **describes and never decides**: there is no
verdict field, no boolean, no thresholded label. Every statistic is either bounded on an interpretable scale, or
paired with a closed-form null it should be read against rather than against a fitted or remembered number:

- `nulls.cos_sd_null = 1/sqrt(d)` (0.0722 at ECAPA's d=192) -- the sd of pairwise cosines between *independent random
  directions* at this dimension.
- `nulls.rbar_null = 1/sqrt(n)` -- the mean resultant length `n` independent directions would produce.
- `nulls.participation_ratio_null = d*n/(d+n)` -- the Marchenko-Pastur value for white noise at this shape.
- `nulls.auc_null = 0.5` -- the Mann-Whitney AUC of same-file vs. different-file cosines under exchangeability.

**The counter-intuitive one, worth stating explicitly**: a *small* sd of cosines is not evidence of a coherent
speaker. At d=192, independent random directions already give sd ~= 0.072 (`nulls.cos_sd_null`); an observed sd of
0.05 is *below* that random-vector null, which is a property of the dimensionality, not of the speaker. This is why
`sd` is never reported as a headline dispersion figure on its own in this codebase -- only next to
`nulls.cos_sd_null`, so a reader compares the two rather than reading `sd` as if lower always meant tighter.

The rest of the block follows the same discipline: `spectrum.participation_ratio` reads against
`nulls.participation_ratio_null`; `file_effect.auc_same_file_vs_diff_file` reads against `nulls.auc_null`; and
`file_effect.permutation_quantile` is a block-permutation reference built from the data itself rather than a second
closed-form null, because windows overlap and shuffling individual vectors would destroy dependence the observed
statistic retains. See the module docstring in `embedding_distribution.py` for the full derivation, including why
silhouette, k-NN purity, intrinsic-dimensionality estimators, and von Mises-Fisher concentration are all deliberately
absent from this block (each for a specific, mechanical reason, not a judgement call).

## Contamination rejection

`select_dominant_vectors` groups the pooled windows (agglomerative clustering on geodesic angular distance) and
returns the dominant group -- **this is the one function in the distribution-statistics module that decides
something**, which is why it is a separate, opt-in call rather than folded into `describe_embedding_distribution`.
`estimate_speaker_embedding_from_audios(..., reject_contamination=True)` calls it and keeps only the dominant group
before describing the result; it is off by default, because choosing a dominant group is a decision the function
does not make on the caller's behalf unasked.

What it removed is recorded, not silently discarded: `provenance.method` gains a `+dominant_cluster` suffix,
`provenance.n_windows_dropped` counts what was cut, and the full `DominantSelection` (cluster summaries, the merge
heights, which cut rule fired) is available from `select_dominant_vectors` directly if the caller wants to inspect
the decision rather than just its outcome.

Once contamination rejection has run, **the statistics describe the curated set, not the raw input**. A caller
reading `EmbeddingDistribution` off a contamination-rejected pool is reading "how coherent is what we kept", not
"how coherent was everything we were given" -- the two questions look identical in the returned fields and are not
the same question.

## Suggested vocabulary for hint tags

`AudioHints.may_contain` and `AudioHints.environment` (`senselab.audio.data_structures.audio_hints`) are open
strings, not an enum, because a closed vocabulary here would be a taxonomy nobody fitted: every corpus that didn't
fit it would force an edit to a type definition rather than to a tag. The following is a **suggestion, not a
contract** -- nothing downstream validates against it, and a caller is free to use any string:

- `may_contain`: `read-speech`, `spontaneous-speech`, `sustained-vowel`, `cough`, `breath`, `singing`, `music`,
  `multiple-speakers`, `silence`
- `environment`: `quiet-room`, `clinic`, `home`, `telephone`, `outdoors`, `unknown`

Using this vocabulary where it fits is useful for cross-corpus consistency, but it should be extended or ignored
freely rather than treated as exhaustive.

## Learn more:
- [SpeechBrain](https://speechbrain.github.io/)
- [ECAPA-TDNN](https://arxiv.org/abs/2005.07143)
- [ResNet TDNN](https://doi.org/10.1016/j.csl.2019.101026)
- [xvector](https://doi.org/10.21437/Odyssey.2018-15)
