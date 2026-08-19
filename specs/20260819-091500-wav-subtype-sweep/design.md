# The PCM_16 default, swept repo-wide instead of one backend at a time

`soundfile.write` takes its subtype from the file extension when the caller does not name one, and
for both WAV and FLAC that default is `PCM_16`. Samples beyond ±1 are hard-clipped; everything else
is quantized with a floor at −96 dBFS. The call succeeds, the file is readable, and nothing reports
that the samples are not the ones that were handed over.

That default has now cost this project four measurements:

1. two analysis harnesses;
2. a GPU re-run in which three SepFormer streams lost up to 8.9% of their samples and disagreed
   with the CPU pass by 9.5 dB;
3. DriftSE's plain-checkpoint output, whose input-output correlation fell from 0.995 to 0.204
   because 98.5% of samples were clipped into a square wave.

Two backends were fixed in PRs #564 and #566 —
`specs/20260818-071500-unasdiff-device-timeout-pcm16` §D-3 fixed four sites and closed with an
inventory of the ones it did not touch, plus the note that the constant then existed under two
names in three modules. This change is that inventory, done: every remaining write, the shared
constant, and a guard so the next one fails a test instead of a measurement.

## The inventory, re-verified

The starting list came from `grep -rn "sf.write\|soundfile.write" src/senselab --include="*.py"`.
Re-running it found the same twelve call sites and no multi-line call the grep had missed. Two
things in the inherited classification were wrong:

- **`features_extraction/sparc.py:126` and `:165` are FLAC sites, not WAV sites.** Both write
  `Path(output_dir) / "decoded.flac"` and `"converted.flac"`, so libsndfile infers FLAC from the
  extension. They took `PCM_16` all the same — FLAC's default is also `PCM_16` — but the fix
  available to them is not the one a WAV site gets, so they belong with the FLAC group. That makes
  `text_to_speech/qwen_tts.py:212` the only true WAV site left in the repository.
- **`video/tasks/input_output.py:70` is not a third category so much as the only site where the
  container is the caller's.** Everything else writes a temp file senselab controls end to end.

## What each site did, and does now

| site | container | before | after |
| --- | --- | --- | --- |
| `audio/tasks/features_extraction/sparc.py` decode worker | `.flac` → `.wav` | default `PCM_16` | `subtype=wav_subtype` (`FLOAT`) |
| `audio/tasks/features_extraction/sparc.py` convert worker | `.flac` → `.wav` | default `PCM_16` | `subtype=wav_subtype` (`FLOAT`) |
| `audio/tasks/voice_cloning/sparc.py` worker | `.flac` → `.wav` | `format="FLAC"`, default `PCM_16` | `subtype=wav_subtype` (`FLOAT`) |
| `audio/tasks/voice_cloning/coqui.py` worker | `.flac` → `.wav` | `format="FLAC"`, default `PCM_16` | `subtype=wav_subtype` (`FLOAT`) |
| `audio/tasks/text_to_speech/coqui.py` worker | `.flac` → `.wav` | `format="FLAC"`, default `PCM_16` | `subtype=wav_subtype` (`FLOAT`) |
| `audio/tasks/text_to_speech/qwen_tts.py` worker | `.wav` | default `PCM_16` | `subtype=wav_subtype` (`FLOAT`) |
| `video/tasks/input_output.py` | caller's | default for that format | `widest_subtype(fmt, dtype)` + a clipping warning |
| `audio/tasks/classification/yamnet.py` | `.wav` | already `FLOAT`, own constant | imports the shared constant |
| `audio/tasks/source_separation/unasdiff.py` (2 sites) | `.wav` | already `FLOAT`, own constant | imports the shared constant |
| `audio/tasks/speech_enhancement/driftse.py` (2 sites) | `.wav` | already `FLOAT`, own constant | imports the shared constant |

Every one of the five FLAC sites writes into a `tempfile.TemporaryDirectory` whose paths the parent
reads back through `Audio(filepath=...)` and then discards, so changing the container changes
nothing a caller can observe. No parent constructs those paths; the worker returns them in its JSON
result.

## The FLAC judgement, and the evidence behind it

FLAC is an integer codec: `soundfile.available_subtypes("FLAC")` is `{PCM_S8, PCM_16, PCM_24}`, so
`subtype="FLOAT"` raises there and "write FLOAT everywhere" is not available as a policy. For each
FLAC site the question was whether the data is guaranteed in-range (then `PCM_24` costs nothing but
quantization at −144 dBFS), or whether the container has to change.

**Measured, by calling the models rather than reasoning about them.** Both venvs
(`~/.cache/senselab/venvs/{sparc,coqui}`) and all four checkpoints were already warm, so each of
these is a real call on `src/tests/data_for_testing/` audio:

| path | model | peak | fraction over ±1 |
| --- | --- | --- | --- |
| SPARC `coder.convert(src, trg)` | `cheoljun95/Speech-Articulatory-Coding`, multi | 0.3216 | 0 |
| SPARC `coder.decode(...)` | same | 0.4194 | 0 |
| Coqui VC `tts.voice_conversion(...)` | `knnvc` + `wavlm-hifigan_prematched` | 0.6930 | 0 |
| Coqui TTS `tts.tts(...)` | `xtts_v2`, one plain English sentence | 0.8846 | 0 |

Nothing clipped, which is why no one has reported these as broken. But the *bound* is what matters,
and it is not uniform:

- **SPARC is bounded by construction.** `sparc.sparc.SPARC.decode` returns the generator output
  with no post-processing, and the loaded checkpoint's `generator.output_conv` is
  `Sequential(LeakyReLU, Conv1d(32→1, k=7), Tanh)` — inspected on the real model, not inferred from
  `use_tanh`'s default. `convert` is `decode` after a pitch shift, so it inherits the bound. Both
  SPARC sites are therefore genuinely in-range.
- **Coqui voice conversion is bounded, but one layer further out.** `KNNVC.voice_conversion`
  returns *features*; `Synthesizer.voice_conversion` then runs `self.vocoder_model.inference(...)`,
  and that generator ends in `torch.tanh` (`TTS/vocoder/models/hifigan_generator.py:287`).
  FreeVC — the other model a caller might pass — ends in `torch.tanh` too
  (`TTS/vc/models/freevc.py:156`). But the model is a `CoquiTTSModel` the caller supplies, so the
  bound is a property of today's default rather than of the code path.
- **Coqui TTS is not bounded.** XTTS-v2 decodes through its own `HifiganGenerator` in
  `TTS/tts/layers/xtts/hifigan_decoder.py`, which contains zero occurrences of `tanh`; the module
  returns `self.waveform_decoder(z, g=g)` directly. `TTS.api.TTS.tts()` returns that raw output.
  Upstream never writes it as-is: `TTS/utils/audio/numpy_transforms.py:save_wav` does
  `wav * (32767 / max(0.01, np.max(np.abs(wav))))` — an unconditional peak normalization — which
  senselab bypasses by calling `tts()` rather than `tts_to_file()`. So the one measurement is
  0.885, i.e. 1.1 dB of headroom, on a model whose own maintainers rescale rather than trust the
  range.

**Decision: WAV/`FLOAT` for all five, not FLAC/`PCM_24`.**

- One of the five is provably unbounded, and the two Coqui sites take a caller-supplied model, so a
  per-site "this model is bounded" argument would have to be re-litigated on every checkpoint
  change — the kind of premise that stops holding without anyone noticing. Uniform policy is worth
  more here than the smaller per-site optimum.
- FLAC's compression is buying nothing measurable. These files exist inside a
  `TemporaryDirectory` for the duration of one subprocess hand-off; halving their size on disk is
  not a cost anyone has measured, and it is being paid for with a hard ±1 ceiling.
- It matches what PRs #564 and #566 established for the other four hand-offs, so all nine worker
  hand-offs now encode identically and the guard can check one rule rather than three.

**Rejected: FLAC with `PCM_24` plus a clamp-and-warn.** It keeps the container and raises the
quantization floor to −144 dBFS, which is below any measurement senselab makes. But at the XTTS
site the clamp is the loss: the excursion is signal, and reporting that it was destroyed is worse
than not destroying it. The clamp also has to be written, tested and kept correct in five worker
strings that cannot import shared code.

**Rejected: peak-normalizing before the write, with the gain recorded in the payload.** This is
what upstream Coqui does, and it is lossless in the container. It changes the returned `Audio`'s
level relative to what the model produced, so every downstream level measurement (loudness, SNR,
the uncertainty workflow's own gates) would be reading a number senselab invented. Recording the
gain makes it recoverable, not correct.

## `video/tasks/input_output.py`: the subtype is the caller's format's problem

`extract_audios_from_local_videos(files, audio_format="wav", acodec="pcm_s16le")`. Both knobs are
free strings, and the write was `sf.write(path, audio_data, sample_rate, format=fmt.upper())`.

Two facts decide the fix:

- **The float path is reachable and was clipping.** `audio_data` is `float32` unless `"s16" in
  codec`, because PyAV hands back `fltp` for AAC/MP3/Vorbis streams (verified on
  `src/tests/data_for_testing/video_48khz_stereo_16bits.mp4`: `codec=aac`, `format=fltp`). So any
  `acodec` without `s16` in it — `pcm_f32le`, `pcm_s24le`, `flac`, `aac`, `""` — wrote float samples
  into `PCM_16`. Lossy decoders routinely overshoot ±1 on transients, and that overshoot was being
  clipped with no report.
- **A lossy container is also reachable, so `FLOAT` cannot be hardcoded.** `audio_format="ogg"`
  resolves to `format="OGG"`, whose only subtypes are `VORBIS` and `OPUS`; `FLOAT` raises. Same for
  `mp3`.

So the subtype is resolved from the format and the dtype by `widest_subtype`, and where the
resolved subtype is fixed-point while the data is float, the fraction of samples that will clip is
logged. Reported rather than repaired: rescaling would change the extracted track's level relative
to the video's own audio, which is not this function's decision.

The `acodec="pcm_s16le"` default is behaviourally unchanged — `int16` data resolves to `PCM_16`,
which is what the default already produced.

## The shared constant: `senselab/utils/audio_write.py`

Before: `LOSSLESS_WAV_SUBTYPE` in `audio/tasks/classification/yamnet.py`, `_WAV_SUBTYPE` in
`audio/tasks/source_separation/unasdiff.py`, `_WAV_SUBTYPE` in
`audio/tasks/speech_enhancement/driftse.py` — the same literal under two names in three modules,
each with its own rationale comment. That is precisely why the sweep was needed: a module that had
*not* been fixed looked, locally, exactly as deliberate as the three that had.

It lives in `utils/` rather than under `audio/` because the importers span three task packages plus
`video/tasks/input_output.py`, and the video path must not have to import `senselab.audio` (and its
torch/transformers chain) to learn how to encode a WAV. It is a top-level leaf module importing
nothing from senselab, so there is no cycle to reason about — the constraint recorded in
`dependencies.py`'s own history, that a utility module must not reach into
`senselab.utils.data_structures`, is satisfied trivially.

Contents:

- `LOSSLESS_WAV_SUBTYPE = "FLOAT"` — the subtype for every WAV senselab writes for itself.
- `widest_subtype(fmt, dtype)` — for the one site where the container is the caller's. Float data
  prefers `FLOAT`, then the widest fixed-point subtype the format has (this is how FLAC resolves to
  `PCM_24`); integer data prefers the subtype that represents it exactly, since widening `int16` to
  `PCM_24` gains nothing and costs 50% more bytes; a lossy container with no PCM subtype at all
  falls back to its own codec default, because there is no "widest" there to pick.
- `out_of_range_fraction(data)` — the number the video site logs.

`FLOAT` over `PCM_24` for the constant, for the same reason as above plus one measurement of the
low end. Against libsndfile 1.2.2: a −100 dBFS signal written at 16 bits reads back at −93 dBFS, and
so does a −120 dBFS one. libsndfile truncates toward negative infinity rather than rounding, so at
−120 dBFS the surviving int16 codes are `{0, −1}` alone — a rectified, DC-offset artifact sitting at
the 1-LSB floor. Background characterization hands YAMNet exactly this kind of quiet residual audio,
and a broadband artifact at −93 dBFS is indistinguishable from analog noise, so amplifying it yields
water-like environmental labels: a fabricated finding rather than a missed one.

**A claim that did not reproduce.** `yamnet.py`'s constant docstring said a −120 dBFS signal "reads
back as exact zeros" at 16 bits. It does not, under soundfile 0.13.1 / libsndfile 1.2.2 — it reads
back at −93 dBFS as the `{0, −1}` artifact above, which is worse, because zeros would at least be
recognisable as absent signal. The replacement wording is what the test now measures.

## The guard: `src/tests/utils/audio_write_guard_test.py`

Modelled on `revision_pinning_guard_test.py`, including its allowlist discipline: a new unguarded
write fails the test until a human reviews it and either fixes it or records why the default is safe
there.

Two sweeps, because the writes live in two places. Nine of the twelve call sites are inside
subprocess-worker `r"""..."""` strings, invisible to a plain AST walk of the parent, so
`_writes_in_source` recurses into any string constant containing `sf.write(` and parses it as
Python. A worker string that mentions `sf.write(` and does *not* parse is reported as unguarded
rather than skipped — the alternative is a write nobody ever checks.

| test | what it enforces |
| --- | --- |
| `test_every_sf_write_names_its_subtype` | every write in `src/senselab`, parent-side or worker-side, passes `subtype=` (keyword or 4th positional) or is in `UNGUARDED_SF_WRITE` with a reason |
| `test_the_allowlist_is_current_and_explained` | allowlist entries name real files, carry a reason, and still have an unguarded write — a stale entry fails |
| `test_the_detector_actually_rejects_the_default` | the detector's verdicts on synthetic sources, including the pre-fix spellings and the worker-string recursion |
| `test_worker_payloads_carry_the_shared_constant` | every parent whose worker reads `args["wav_subtype"]` builds `{"wav_subtype": LOSSLESS_WAV_SUBTYPE}` — a literal there would be a second copy of the constant |
| `test_the_constant_home_check_tells_a_definition_from_a_use` | the definition-vs-use distinction, so a lowercase local is not mistaken for a re-fork |
| `test_the_subtype_constant_has_one_home` | no constant-cased `*SUBTYPE*` assignment outside `utils/audio_write.py` |
| `test_widest_subtype_is_what_each_container_can_actually_carry` | the resolved subtype passes `soundfile.check_format` for WAV, FLAC, AIFF, OGG and MP3 |

`UNGUARDED_SF_WRITE` ships empty, because every write now names its subtype. It is kept rather than
deleted along with its assertions so the next one lands there after review instead of being waved
through, and its stale-entry test keeps the mechanism from rotting while unused.

**Against the pre-fix tree** the sweep names all seven defective sites and nothing else:

```
E       AssertionError: Audio write(s) take libsndfile's default subtype, which is PCM_16 for WAV and FLAC -- it clips every sample beyond +-1 and quantizes the rest at -96 dBFS, silently:
E           audio/tasks/features_extraction/sparc.py:92 (worker script):35
E           audio/tasks/features_extraction/sparc.py:136 (worker script):30
E           audio/tasks/text_to_speech/coqui.py:30 (worker script):42
E           audio/tasks/text_to_speech/qwen_tts.py:161 (worker script):52
E           audio/tasks/voice_cloning/coqui.py:31 (worker script):61
E           audio/tasks/voice_cloning/sparc.py:39 (worker script):37
E           video/tasks/input_output.py:70
```

The other three pre-fix failures were `test_worker_payloads_carry_the_shared_constant` (five
backends read no subtype from their payload), `test_the_subtype_constant_has_one_home` (three
modules defining it), and the five `widest_subtype` cases (no such module).

## Tests

`src/tests/utils/audio_write_test.py` — the policy, checked against libsndfile rather than against
itself. Every assertion writes a real file and reads it back, because the defect was invisible
exactly because the code *looked* right.

| test | what it measures |
| --- | --- |
| `test_the_wav_default_really_does_clip_and_float_really_does_not` | a 1.6-peak float32 signal: `PCM_16` clips >50% of samples to 1.0; `FLOAT` round-trips bit-exactly |
| `test_the_16_bit_floor_replaces_quiet_content_rather_than_dropping_it` | −100 and −120 dBFS both read back at −93 dBFS; the surviving codes at −120 dBFS are `{0, −1}` |
| `test_flac_cannot_take_the_wav_subtype` | `check_format("FLAC", "FLOAT")` is false; FLAC's subtypes are exactly the three PCM ones |
| `test_widest_subtype_picks_something_the_container_accepts` | ten (format, dtype) pairs, each verified with `check_format` |
| `test_widest_subtype_writes_are_actually_writable` | WAV/FLAC/OGG round-trip through a real write, `sf.info(...).subtype` matching what was asked |
| `test_widest_subtype_refuses_what_it_cannot_answer` | an unknown format raises; a format with no default subtype raises rather than returning `None`, which `sf.write` would read as "use the default" |
| `test_out_of_range_fraction_counts_only_what_a_fixed_point_write_would_lose` | exactly ±1 is representable and must not count; an empty array is 0.0 |

`src/tests/video/tasks/input_output_test.py` — the one changed site whose behaviour is observable
without a model, tested end to end through the real extractor by intercepting
`read_files_from_disk` (which is called while the `TemporaryDirectory` is still alive) and copying
the written files out.

| test | what it measures |
| --- | --- |
| `test_the_s16_codec_hint_still_writes_pcm_16` | the default path is unchanged |
| `test_a_float_codec_hint_writes_float_not_pcm_16` | `acodec="pcm_f32le"` → `sf.info(...).subtype == "FLOAT"` (pre-fix: `PCM_16`) |
| `test_a_flac_container_gets_the_widest_integer_subtype_it_has` | `audio_format="flac"` → `PCM_24`, not `FLOAT` (which would raise) and not `PCM_16` |
| `test_a_lossy_container_is_still_writable` | `audio_format="ogg"` → `VORBIS`, i.e. the fix did not break the lossy path |
| `test_a_video_with_no_audio_track_is_skipped` | pre-existing behaviour, kept while the function was edited |

The five worker sites have no test that runs them: each needs its own venv and a multi-GB
checkpoint. What covers them is the guard's two AST sweeps — the write names a subtype, and the
value it names comes from the shared constant — which is the same trade
`revision_pinning_guard_test.py` makes for the same reason.

## Out of scope, with a measurement so the follow-up is concrete

`Audio.save_to_file` is the *other* way senselab writes audio, and it has the same defect through a
different API. Measured on a 1.6-peak float32 signal:

| `format=` | resulting subtype | fraction of samples clipped |
| --- | --- | --- |
| `"wav"` | `PCM_16` | 0.575 |
| `"flac"` | `PCM_24` | 0.575 |

It has no subtype parameter at all on the torchcodec path (`AudioEncoder(...).to_file(path)`), which
is why PRs #564/#566 stopped using it for their hand-offs and called `soundfile.write` directly.
Three sites still feed workers through it — the input serialization in
`features_extraction/sparc.py`, `voice_cloning/sparc.py` and `voice_cloning/coqui.py` — so a
recording peaking above full scale is clipped before the worker sees it. Fixing that means either a
subtype parameter on `save_to_file` or replacing those three calls, and it needs its own decision
about the public API; it is not folded in here. The guard does not see it, because the guard
watches `sf.write`.
