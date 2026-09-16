# AST's minimum-input guard

## Incidence

53 of 61,442 triage recordings were lost entirely — not just their AST derivatives — because they
are shorter than AST's feature extractor can window:

```
RuntimeError: PREPROCESS: 3 block(s) failed unexpectedly:
  ast_scores: AssertionError: choose a window size 400 that is [2, 372];
  enhanced_ast: AssertionError: choose a window size 400 that is [2, 372];
  residual_ast: AssertionError: choose a window size 400 that is [2, 372]
```

`preprocess.py:1955` classifies `ValueError`/`LookupError` as a cascading absence and continues;
anything else (`AssertionError` included) is appended to `hard_failures`, and `preprocess.py:1963-1965`
raises `RuntimeError` once every block has run — discarding the whole recording's already-computed
`PreprocessResult`, including YAMNet, HeAR, spans and SQUIM. `ast_scores` (`preprocess.py:965-980`),
`enhanced_ast` and `residual_ast` (`preprocess.py:1876-1895`, both via `_stream_ast`) all reach the
model through the same `classify_audios(..., model=_ast_model())` call, so all three fail on the
same recording at once.

## Where the assertion comes from

`transformers.ASTFeatureExtractor._extract_fbank_features` (`feature_extraction_audio_spectrogram_transformer.py:104-121`)
calls `torchaudio.compliance.kaldi.fbank(waveform, sample_frequency=self.sampling_rate, ...)`. Inside
`kaldi.py:_get_waveform_and_window_properties` (`kaldi.py:125-144`):

```python
window_shift = int(sample_frequency * frame_shift * MILLISECONDS_TO_SECONDS)
window_size = int(sample_frequency * frame_length * MILLISECONDS_TO_SECONDS)
...
assert 2 <= window_size <= len(waveform), "choose a window size {} that is [2, {}]".format(window_size, len(waveform))
```

`ASTFeatureExtractor` calls `fbank` with `sample_frequency` set but `frame_length` left at `fbank`'s
own default (`frame_length: float = 25.0`, `kaldi.py:520`). At AST's `sampling_rate = 16000`
(`ASTFeatureExtractor.__init__` default, `feature_extraction_audio_spectrogram_transformer.py:71`),
`window_size = int(16000 * 25.0 * 0.001) = 400`. The assertion fires whenever the audio handed to
the feature extractor — after resampling to 16 kHz — has fewer than 400 samples.

## Derived minimum, and how it was derived

**400 samples at 16 kHz = 0.025 s = 25 ms.** Reproduced directly (`AudioTooShortForAST` test in
`src/tests/audio/tasks/classification/huggingface_test.py`, and a scratch script against a live
`MIT/ast-finetuned-audioset-10-10-0.4593` pipeline): 399 samples at 16 kHz raises, 400 passes;
372 samples at 16 kHz reproduces the exact `[2, 372]` bound from the field incident. Resampled from
48 kHz: 1104 samples (0.023 s) raises with `[2, 368]`; 1200 samples (0.025 s) passes — the bound is a
duration, invariant to the recording's native rate, because `classify_audios_with_transformers`
resamples to `feature_extractor.sampling_rate` before this check runs.

`huggingface.py:_ast_min_samples` computes this from the extractor's own state rather than
hard-coding `400`: `feature_extractor.sampling_rate` (an attribute of the loaded `ASTFeatureExtractor`)
and `frame_length_ms = inspect.signature(torchaudio.compliance.kaldi.fbank).parameters["frame_length"].default`
(reads `fbank`'s own default off its signature rather than copying the literal `25.0`), combined with
`torchaudio.compliance.kaldi.MILLISECONDS_TO_SECONDS` (`0.001`) using the identical formula
`kaldi.py:139` uses. If AST's checkpoint ever ships a different `sampling_rate`, or a future
`torchaudio` changes `fbank`'s default `frame_length`, this recomputes rather than silently drifting
from the assertion it stands in for.

## Where the guard lives

AST has no dedicated module the way YAMNet has `yamnet.py` — it is an `HFModel` routed through the
generic pipeline in `huggingface.py`, which is also where the `feature_extractor` and
`expected_sampling_rate` are already resolved (`huggingface.py:_get_hf_audio_classification_pipeline`,
`classify_audios_with_transformers`). `AudioTooShortForAST` and `_ast_min_samples` live there,
gated on `isinstance(feature_extractor, ASTFeatureExtractor)` so no other HF audio classifier is
affected. `classify_audios_with_transformers` raises it, after resampling and before calling the
pipeline, for every audio (whole-clip or windowed) shorter than the derived minimum — covering
`ast_scores`, `enhanced_ast` and `residual_ast` from one call site, since `_classify_windowed` and
`_classify_whole` both route through it.

`AudioTooShortForAST` subclasses `ValueError`, so `preprocess.py:1955`'s existing
`except (ValueError, LookupError)` classification catches it unchanged — no widening of that clause.
