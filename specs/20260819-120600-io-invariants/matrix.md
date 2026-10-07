# senselab audio I/O audit — measured matrix

All measurements on this host, 2026-08-19. Reproduction scripts:
`/Users/satra/.claude/jobs/295c3f8a/tmp/io-audit/{matrix,summarize,probes,invariants,invariants2}.py`
Raw results: `matrix.json` in the same directory (copied here as `matrix.json`).

## Versions

| component | version |
|---|---|
| Python | 3.12.0 |
| numpy | 2.4.4 |
| torch | 2.11.0 |
| torchaudio | 2.11.0 |
| torchcodec | 0.11.1 |
| soundfile | 0.13.1 |
| libsndfile | 1.2.2 |
| librosa | 0.11.0 |
| PyAV | 17.0.1 |
| ffmpeg / ffprobe CLI | 8.1 (homebrew, `--enable-libopus --enable-libmp3lame`; **no `--enable-libvorbis`**) |

Test signals, 22050 Hz unless noted:

- `q16` — 2000 float32 samples on the exact 16-bit grid (`k/32768`, integer `k`). Any
  lossless integer path of ≥16 bits should return this bit-for-bit; failure means a
  scaling-convention mismatch.
- `f32` — 2000 float32 samples uniform in (−0.9, 0.9). Not representable at any integer depth.
- `oor` — `f32` scaled to peak 3.0, plus a 400-sample flat block at 1.5.
- `q16_stereo` — two independent `q16` channels.

---

## Finding 0: `torchaudio.save` / `torchaudio.load` are not a fallback — they *are* torchcodec

`torchaudio/__init__.py` in 2.11.0 routes `save` → `save_with_torchcodec` and `load` →
`load_with_torchcodec`, each of which does `from torchcodec.encoders import AudioEncoder`
(resp. `.decoders import AudioDecoder`) and raises `ImportError` if torchcodec is missing
(`torchaudio/_torchcodec.py`, import guards at lines 82–86 and 246–250). Its own docstring:

> As of TorchAudio 2.9, this function relies on TorchCodec's encoding capabilities under the
> hood. […] Because of the reliance on Torchcodec, the parameters `format`, `encoding`,
> `bits_per_sample`, `buffer_size`, and `backend`, are ignored and accepted only for
> backwards compatibility.

Measured consequence: every `torchaudio.save` cell in the matrix below is **byte-identical in
size and content** to the corresponding `torchcodec.AudioEncoder` cell, and `torchaudio.save`
emits a `UserWarning` per ignored argument rather than honouring it. `torchaudio.info`,
`torchaudio.list_audio_backends`, `torchaudio.io`, `torchaudio.backend` and
`torchaudio.AudioMetaData` are **gone** in 2.11.0 (verified by attribute check; the migration
landed in pytorch/audio#3975 "Add save_with_torchcodec, modify save()'s warnings" and #4039
"Let `torchaudio.load()` and `torchaudio.save()` rely on `load_with_torchcodec()` and
`save_with_torchcodec()`", both closed).

So senselab's `TORCHCODEC_AVAILABLE ? AudioEncoder : torchaudio.save` structure has no second
branch in practice: if torchcodec is importable the first branch runs, and if it is not, the
second branch raises `ImportError` from inside torchaudio. The only real fallbacks are
`soundfile`, `ffmpeg`, or `scipy.io.wavfile`.

---

## Matrix 1 — what each encoder actually writes

Requested target vs. what landed on disk (`ffprobe` codec/`sample_fmt`, and libsndfile's view).
`torchaudio.save` is omitted where it is identical to torchcodec (it always is).

| target requested | torchcodec / torchaudio | soundfile.write | ffmpeg CLI |
|---|---|---|---|
| wav PCM_16 | `pcm_s16le` s16 → WAV/PCM_16 | WAV/PCM_16 | WAV/PCM_16 |
| wav PCM_24 | **`pcm_s16le` s16 → WAV/PCM_16** | WAV/PCM_24 | WAVEX/PCM_24 |
| wav PCM_32 | **`pcm_s16le` s16 → WAV/PCM_16** | WAV/PCM_32 | WAVEX/PCM_32 |
| wav FLOAT32 | **`pcm_s16le` s16 → WAV/PCM_16** | WAV/FLOAT | WAV/FLOAT |
| wav FLOAT64 | **`pcm_s16le` s16 → WAV/PCM_16** | WAV/DOUBLE | WAV/DOUBLE |
| flac 16 | **`flac` s32 / 24 bit → FLAC/PCM_24** | FLAC/PCM_16 | FLAC/PCM_16 |
| flac 24 | `flac` s32 / 24 → FLAC/PCM_24 | FLAC/PCM_24 | FLAC/PCM_24 |
| ogg / vorbis | **`flac` in Ogg — libsndfile CANNOT READ** | OGG/VORBIS | mono FAILS (native `vorbis` encoder, no libvorbis in this build); stereo OK |
| opus | FAILS: `invalid sample rate=22050. Supported … 48000, 24000, 16000, 12000, 8000` | FAILS: same constraint, libsndfile message | OK, but resampled (see below) |
| mp3 | MP3/MPEG_LAYER_III | MP3/MPEG_LAYER_III | MP3/MPEG_LAYER_III |
| m4a / aac | `aac` in mp4 — libsndfile CANNOT READ | **not supported by libsndfile** | `aac` in mp4 |

The torchcodec extension→codec map, measured exhaustively (there is **no parameter to override
it** — `to_file` takes only `bit_rate`, `num_channels`, `sample_rate`):

| ext | codec / sample_fmt | libsndfile can read? |
|---|---|---|
| `.wav`, `.w64` | `pcm_s16le` / s16 | yes |
| `.caf`, `.aiff`, `.au` | `pcm_s16be` / s16 | yes |
| `.flac` | `flac` / s32, 24 bits_per_raw_sample | yes |
| `.ogg`, `.oga` | **`flac` in Ogg** / s32, 24 | **no** — `unknown error in flac decoder` |
| `.opus` | `opus` / fltp | yes (OGG/OPUS) |
| `.mp3` | `mp3` / fltp | yes |
| `.m4a`, `.mp4`, `.aac` | `aac` / fltp | no |
| `.webm` | `opus` / fltp | no |
| `.wv` | **`wavpack` / fltp** | no |
| `.tta` | `tta` / s32 | no |
| `.mka` | fails (`validateSampleRate`) | — |
| `.rf64`, `.ape`, `.als`, `.dts`, `.wma` | `RuntimeError` in `AudioEncoder`/`initializeEncoder` | — |

Two consequences worth naming:

- **`.ogg` from torchcodec is Ogg-FLAC, not Ogg-Vorbis.** It is lossless, which is *better*
  than asked for, but libsndfile refuses it — so a file senselab writes as `.ogg` cannot be
  read by `soundfile`, which is exactly the reader senselab's own metadata fallback
  (`Audio.sampling_rate` → `sf.info`) and `Audio.from_stream` use.
- **`.wv` (WavPack, fltp) is the only torchcodec extension that preserves float.** Writing the
  out-of-range signal (peak 4.0) to `.wv` and reading it back with `AudioDecoder` is
  **bit-exact, peak 4.0 preserved**. Every other lossless torchcodec target clamps to 1.0.

### Input-type constraints, `torchcodec.AudioEncoder`

| input | result |
|---|---|
| `torch.float32` (C, N) | OK |
| `torch.float32` (N,) 1-D | OK (treated as mono) |
| `torch.float32` non-contiguous | OK |
| `torch.float64` | `ValueError: Expected float32 samples, got samples.dtype = torch.float64` |
| `torch.float16`, `torch.int16`, `torch.int32` | same `ValueError` |
| `numpy.ndarray` float32 | `ValueError: Expected samples to be a Tensor` |
| (N, 1) time-first | `RuntimeError` in `validateSamples` (reads dim 0 as channels) |

`soundfile.write` accepts float32, float64, int16, int32 numpy arrays **and** torch tensors,
and writes the requested subtype regardless of input dtype.

### float → int16 conversion rule (all three agree on scale, differ on rounding)

| input float | torchcodec | soundfile PCM_16 | ffmpeg pcm_s16le |
|---|---|---|---|
| 1.0 | 32767 | 32767 | 32767 |
| −1.0 | −32768 | −32768 | −32768 |
| 0.5 | 16384 | 16384 | 16384 |
| 1.5/32768 | **2** | **1** | **2** |
| 2.5/32768 | 2 | 2 | 2 |

Scale is `×32768` with a clamp at `+32767` in all three — no convention mismatch. Rounding
differs: measured max error writing arbitrary float32 to WAV/PCM_16 is **1.53e-05 (½ LSB) for
torchcodec and ffmpeg, 3.05e-05 (1 full LSB) for soundfile** — i.e. round-to-nearest vs.
truncation. This is the only encoder disagreement on lossless integer targets.

---

## Matrix 2 — round-trip exactness, `read(write(x)) == x`

Every cell was read back by all six decode paths. **In every case all six decoders agreed
exactly with each other for WAV and FLAC**, so the tables below collapse the decoder axis; the
lossy-format decoder disagreements are called out separately.

### Signal on the exact 16-bit grid (`q16`)

| target | torchcodec / torchaudio | soundfile | ffmpeg |
|---|---|---|---|
| wav PCM_16 | **EXACT** | **EXACT** | **EXACT** |
| wav PCM_24 | **EXACT** (file is really PCM_16) | **EXACT** | **EXACT** |
| wav PCM_32 | **EXACT** (really PCM_16) | **EXACT** | **EXACT** |
| wav FLOAT32 | **EXACT** (really PCM_16) | **EXACT** | **EXACT** |
| wav FLOAT64 | **EXACT** (really PCM_16) | **EXACT** | **EXACT** |
| flac 16 | **EXACT** (really 24-bit) | **EXACT** | **EXACT** |
| flac 24 | **EXACT** | **EXACT** | **EXACT** |
| ogg/vorbis | **EXACT** (it's Ogg-FLAC) — but soundfile cannot open it | 8.8e−01 (lossy) | write failed (mono) |
| opus | write failed | write failed | 2.2e+00 (lossy + resampled) |
| mp3 | 1.2e+00 | 1.0e+00 | 1.2e+00 |
| m4a/aac | 1.4e+00 | unsupported | 1.4e+00 |

### Arbitrary in-range float32 (`f32`), max abs difference

| target | torchcodec / torchaudio | soundfile | ffmpeg |
|---|---|---|---|
| wav PCM_16 | 1.5e−05 | 3.0e−05 | 1.5e−05 |
| wav PCM_24 | **1.5e−05** (silently 16-bit) | 1.2e−07 | 1.2e−07 |
| wav PCM_32 | **1.5e−05** (silently 16-bit) | 2.3e−10 | 2.3e−10 |
| wav FLOAT32 | **1.5e−05** (silently 16-bit) | **EXACT** | **EXACT** |
| wav FLOAT64 | **1.5e−05** (silently 16-bit) | **EXACT** | **EXACT** |
| flac 16 | 1.2e−07 (silently 24-bit) | 1.5e−05 | 1.5e−05 |
| flac 24 | 1.2e−07 | 6.0e−08 | 1.2e−07 |
| ogg/vorbis | 1.2e−07 (Ogg-FLAC) | 7.6e−01 | — |
| mp3 / opus / m4a | 1.5e+00 / — / 1.7e+00 | 1.1e+00 / — / — | 1.5e+00 / 2.1e+00 / 1.7e+00 |

**The exactness answer, stated plainly.** For arbitrary float32 data there are exactly two
`(encode, decode)` families that are bit-exact, and neither of them is torchcodec:

- `soundfile.write(..., subtype="FLOAT")` or `subtype="DOUBLE"` → read by **any** of
  torchcodec, torchaudio, soundfile, librosa, ffmpeg. EXACT.
- `ffmpeg -c:a pcm_f32le` / `pcm_f64le` → read by any of the six. EXACT.

Every torchcodec encode path is inexact for arbitrary float32, because the best lossless
container it will select is PCM_16 (WAV) or 24-bit (FLAC) — with the sole exception of `.wv`,
which is bit-exact but readable only by torchcodec/ffmpeg.

### Out-of-range data (peak 3.0, flat 1.5 block)

Warnings emitted at write time, across all 176 encode cells: **zero**, from every encoder. The
only warnings anywhere were `torchaudio.save`'s `UserWarning`s about ignored arguments.

| target | torchcodec / torchaudio | soundfile | ffmpeg |
|---|---|---|---|
| wav PCM_16/24/32 | peak → 1.0000, silently | peak → 1.0000, silently | peak → 1.0000, silently |
| **wav FLOAT32** | **peak → 1.0000** (it wrote PCM_16) | **peak 2.9985 preserved** | **peak 2.9985 preserved** |
| **wav FLOAT64** | **peak → 1.0000** (it wrote PCM_16) | **peak 2.9985 preserved** | **peak 2.9985 preserved** |
| flac 16/24 | peak → 1.0000 | peak → 1.0000 | peak → 1.0000 |
| ogg/vorbis | peak → 1.0000 (Ogg-FLAC) | peak 3.5540 (overshoot) | — |
| mp3 | peak 4.8797 | peak 5.2681 | peak 4.8797 |
| opus | — | — | peak 4.5479 |
| m4a/aac | peak 5.0081 | — | peak 5.0081 |

Clamping is inherent to integer PCM and is not torchcodec-specific. What *is* torchcodec-
specific is that **there is no way to opt out**: no argument selects a float sample format for
`.wav` or `.flac`. Note also that lossy codecs do not clamp — they *overshoot*, so a signal
that leaves at peak 3.0 comes back at 4.9.

The flat-1.5 collapse reported in the brief reproduces exactly: through `Audio.save_to_file` to
`.wav`, a 500-sample block at 1.5 becomes 500 identical samples at 1.0, one distinct value, and
`save_to_file` returns normally.

### Sample rate, channels, length, metadata

- **Sample rate preserved** by every path for wav, flac, mp3, ogg-vorbis. **Not preserved for
  opus**: writing 22050 Hz forces a resample, ffprobe then reports 48000 while libsndfile
  reports 24000 for the same file — two readers disagreeing about the rate of one file.
- **Channel count preserved** by every path, every format tested (mono and stereo).
- **Length**: exact for wav, flac, and mp3 (LAME's Xing header lets every decoder trim). Not
  exact for `m4a`/`aac`: 2000 in → 2048 out via torchcodec/ffmpeg, and `librosa`/`audioread`
  returns yet another length. AAC padding is not trimmed by torchcodec.
- **Metadata**: `soundfile` writes and reads tags (`SoundFile.title`, `.comment`); `ffmpeg`
  writes and reads tags. **torchcodec has no metadata parameter at all** — it writes only
  `encoder=Lavf62.12.100` — and `AudioDecoder.metadata` exposes no tag fields
  (`begin_stream_seconds, bit_rate, codec, duration_seconds, num_channels, sample_format,
  sample_rate, stream_index` only). So no torchcodec path can carry provenance in the file.

### Audio from a video container

| path | mp4 (h264+aac) | mkv (vp9+opus) | mov (h264+pcm_s16le) |
|---|---|---|---|
| `torchcodec.AudioDecoder` | OK (48128 samples) | OK (48000) | OK (48000) |
| `torchaudio.load` (= torchcodec) | OK (48128) | OK (48000) | OK (48000) |
| **`soundfile.read`** | **`Format not recognised`** | **`Format not recognised`** | **`Format not recognised`** |
| `librosa.load` | OK (48000) via `audioread` + `UserWarning: PySoundFile failed` + `FutureWarning: audioread_load deprecated as of librosa 0.10.0, will be removed` | OK (48000) | OK (48000) |
| `ffmpeg` CLI | OK (48128) | OK (48000) | OK (48000) |

libsndfile reads **no** video container. librosa reads them only through the deprecated
`audioread` fallback. So for video, torchcodec or ffmpeg are the only durable options — and
torchcodec/ffmpeg return 128 more samples than librosa for AAC-in-MP4, i.e. the paths disagree
on where the audio starts and ends.

---

## Priority question 1 — is decode source-independent, and does anything normalise?

### Amplitude convention: unanimous

A WAV/PCM_16 file containing the literal int16 values `[-32768, -32767, -16384, 0, 16384,
32766, 32767]`, read by all six paths:

| reader | −32768 | 32767 |
|---|---|---|
| `/32768` reference | −1.00000000 | 0.99996948 |
| torchcodec.AudioDecoder | −1.00000000 | 0.99996948 |
| torchaudio.load | −1.00000000 | 0.99996948 |
| soundfile.read(float32) | −1.00000000 | 0.99996948 |
| soundfile.read(float64) | −1.00000000 | 0.99996948 |
| librosa.load | −1.00000000 | 0.99996948 |
| ffmpeg CLI | −1.00000000 | 0.99996948 |

All six use the `/32768` convention, identically, to the last bit. dtype returned: `float32`
from torchcodec, torchaudio, soundfile(f32), librosa; `float64` only if you ask soundfile for it.

### Does the read path clamp out-of-range float? **No.**

A WAV/FLOAT and a WAV/DOUBLE file with samples spanning ±4.0:

| reader | peak read back | bit-exact vs. source |
|---|---|---|
| torchcodec.AudioDecoder | 4.00000 | **YES** |
| torchaudio.load | 4.00000 | **YES** |
| soundfile.read(f32) | 4.00000 | **YES** |
| soundfile.read(f64) | 4.00000 | **YES** |
| librosa.load | 4.00000 | **YES** |
| ffmpeg CLI | 4.00000 | **YES** |

Identical result whether the file was written by soundfile or by ffmpeg. **The asymmetry is
precise: torchcodec's read path is fully range-transparent; its write path is not.** So
out-of-range audio can be read into senselab but cannot be written back out of it, except via
`.wv` or a non-torchcodec encoder.

Lossy sources also come back un-clamped: the same reference signal (peak 0.7999) decoded from
mp3 returns peak 1.0603, from opus 1.6327, from AAC 1.5213. Decoders do not clip codec
overshoot, which is correct but means "peak ≤ 1" is not an invariant a caller may assume.

### Does anything normalise? **No — measured, not assumed.**

Two files identical except for a `×0.1` gain, decoded by all six paths:

| source format | peak ratio (expected 0.1) | RMS ratio (expected 0.1) |
|---|---|---|
| WAV/FLOAT | 0.100000 | 0.100000 |
| WAV/PCM_16 | 0.100010 | 0.100000 |
| FLAC/PCM_24 | 0.100000 | 0.100000 |

Identical for every reader. The 0.100010 on PCM_16 is 16-bit quantisation of the peak sample,
not normalisation — the RMS ratio is exactly 0.1. **No decode path in this set performs peak,
RMS, or filter-graph normalisation, per file or per segment.** The per-segment hazard the brief
warns about (DriftSE's per-chunk peak normalisation) does **not** exist at the decode layer;
it is introduced by senselab's own worker code, above the I/O boundary.

### Source-independence, one signal in 11 containers, decoded by torchcodec

| source | dtype | n | sr | peak | RMS | bit-exact vs. reference |
|---|---|---|---|---|---|---|
| reference float32 | float32 | 44100 | 22050 | 0.799973 | 0.461799 | — |
| wav PCM_16 | float32 | 44100 | 22050 | 0.799988 | 0.461799 | no (quantised) |
| wav PCM_24 | float32 | 44100 | 22050 | 0.799973 | 0.461799 | no (quantised) |
| wav PCM_32 | float32 | 44100 | 22050 | 0.799973 | 0.461799 | no (quantised) |
| **wav FLOAT32** | float32 | 44100 | 22050 | 0.799973 | 0.461799 | **YES** |
| **wav FLOAT64** | float32 | 44100 | 22050 | 0.799973 | 0.461799 | **YES** |
| flac 16 | float32 | 44100 | 22050 | 0.799988 | 0.461799 | no |
| flac 24 | float32 | 44100 | 22050 | 0.799973 | 0.461799 | no |
| mp3 192k | float32 | 44100 | 22050 | 1.060308 | 0.418184 | no |
| opus | float32 | 44100 | 48000 | 1.632722 | 0.423055 | no |
| m4a aac | float32 | **45056** | 22050 | 1.521272 | 0.450459 | no |
| mp4 (video+aac) | float32 | **45056** | 22050 | 1.521272 | 0.450459 | no |

Decode is source-independent in **dtype** (always float32) and in **amplitude convention**
(always `/32768`, never normalised, never clamped). It is *not* source-independent in **length**
(AAC adds 956 samples) or **sample rate** (preserved from the file, so an Opus file reports 48000
however it was made). Audio from a video container is indistinguishable from the same audio in a
bare container — same length, same values.

---

## Priority question 2 — does a streamed chunk equal the slice of a full decode?

Method note: the first run compared every chunker against a *torchcodec* full decode, which
conflates decoder identity with chunking. The table below compares **each library against its
own full decode**, with the alignment searched over ±4000 samples so that a shifted chunk is
reported as a shift rather than as noise. Source: 6 s of 16-bit-grid float32.

`off` = how far the returned chunk actually sits from where it was requested, in samples.

| format | torchcodec range | torchaudio (=tc) | soundfile `start=` | ffmpeg `-ss` **before** `-i` | ffmpeg `-ss` **after** `-i` |
|---|---|---|---|---|---|
| wav PCM_16 | **EXACT**, off=0 | **EXACT** | **EXACT** | **EXACT** | **EXACT** |
| wav FLOAT32 | **EXACT**, off=0 | **EXACT** | **EXACT** | **EXACT** | **EXACT** |
| flac PCM_24 | **EXACT**, off=0 | **EXACT** | **EXACT** | **EXACT** | **EXACT** |
| mp3 192k | exact values, **off = −1105** (and a short read at t=0: 943 of 2048) | **EXACT**, off=0 | off=0, diff 1.2e−07 | off=0, **diff 1.2e+00 — wrong data** | **EXACT** |
| m4a aac | **EXACT**, off=0 | **EXACT** | libsndfile cannot open | off=0, **diff up to 1.1e+00** | **EXACT** |
| opus 48k | **EXACT**, off=0 | **EXACT** | **EXACT** | **EXACT** | **EXACT** |

**Lossless formats: the invariant holds, bit-for-bit, for every path, at aligned, odd and
mid-file offsets.** Contiguity too: concatenating 2048-sample chunks reconstructs the full
decode bit-for-bit, 0 mismatched samples, for WAV/PCM_16, WAV/FLOAT32 and FLAC/PCM_24 via
torchcodec, soundfile and torchaudio alike. senselab's `Audio.from_stream` (soundfile blocks)
also reconstructs the full decode bit-for-bit at 0.25 s chunks on all three.

**MP3 breaks it, and the mechanism is exact.** torchcodec's two entry points use two different
timelines for the same file:

| requested start | `n` returned (2048 asked) | `pts_seconds` reported | offset from requested, vs. `get_all_samples()` |
|---|---|---|---|
| 0 | **943** | 0.050113379 (= sample 1105) | 0 |
| 1 | 944 | 0.050113379 (sample 1105) | −1 |
| 100 | 1043 | 0.050113379 (sample 1105) | −100 |
| 576 | 1519 | 0.050113379 (sample 1105) | −576 |
| 1105 | 2048 | 0.050113379 (sample 1105) | −1105 |
| 2048 | 2048 | 0.092879819 (sample 2048) | −1105 |
| 5001 | 2048 | 0.226802721 (sample 5001) | −1105 |
| 22050 | 2048 | 1.000000000 (sample 22050) | −1105 |

`get_all_samples()` prepends **1105 samples** of MP3 decoder pre-roll (576 + 529, the standard
MDCT + LAME delay) that `get_samples_played_in_range()` excludes. So for an MP3, index `i` of a
full decode is presentation sample `i − 1105`, and **a windowed measurement taken via offsets
is shifted 1105 samples — 50.1 ms at 22.05 kHz — relative to a whole-file measurement of the
same file.** Below `start=1105` the range API additionally silently short-reads, returning
`n − (1105 − start)` samples with no warning.

The chunk values themselves are exact once aligned (diff 0.00e+00 at off = −1105), so this is a
pure indexing defect, not a decoding one. `pts_seconds` on the returned `AudioSamples` reports
the true position and would let a caller detect and correct it — `align_vs_pts` is exact in
every row. **senselab ignores `pts_seconds`**: `Audio._lazy_load_data_from_filepath` returns
`samples.data` and discards the rest, so `Audio(filepath=..., offset_in_sec=...)` inherits the
shift verbatim — measured off = −1105 for mp3, 0 for wav/flac/m4a/opus.

`torchaudio.load(frame_offset=, num_frames=)`, despite being a torchcodec wrapper, is **exact
for mp3** — it decodes and slices rather than seeking, so it does not hit the bug.

**ffmpeg `-ss` placement matters and is a real trap**: `-ss` *before* `-i` (fast/keyframe seek)
returns wrong data for mp3 and aac — max diff 1.2 and 1.1 respectively, i.e. a different part of
the signal. `-ss` *after* `-i` (decode-and-discard) is bit-exact for every format tested.

**Cross-library agreement on a full decode** (decoder identity, separate from chunking):

| format | torchcodec vs. torchaudio | vs. soundfile | vs. ffmpeg |
|---|---|---|---|
| wav PCM_16 / FLOAT32 | EXACT | EXACT | EXACT |
| flac PCM_24 | EXACT | EXACT | EXACT |
| mp3 192k | EXACT | diff 2.46e−06 | EXACT |
| m4a aac | EXACT | cannot open | EXACT |
| opus 48k | EXACT | diff 1.13e−06 | EXACT |

For lossless formats all four decoders are bit-identical. For lossy formats libsndfile's
decoders (mpg123, libopusfile) differ from ffmpeg's by ~1–2e−06 — small, but it means "which
library decoded this mp3" is a recorded-provenance question, not a free choice.

---

## What senselab does today, measured end-to-end

`Audio.save_to_file(path, encoding="PCM_F", bits_per_sample=32)` then `Audio(filepath=path)`:

| input | ext | file actually written | round-trip | in peak | out peak | warnings |
|---|---|---|---|---|---|---|
| in-range float32 | wav | WAV/PCM_16 | max diff 1.53e−05 | 0.8998 | 0.8998 | **0** |
| in-range float32 | flac | FLAC/PCM_24 | max diff 1.19e−07 | 0.8998 | 0.8998 | **0** |
| in-range float32 | ogg | **libsndfile cannot read** | max diff 1.19e−07 | 0.8998 | 0.8998 | **0** |
| 16-bit grid | wav | WAV/PCM_16 | **exact** | 0.9999 | 0.9999 | 0 |
| **out-of-range (peak 3.0)** | wav | WAV/PCM_16 | **max diff 2.00** | **3.0000** | **1.0000** | **0** |
| **out-of-range (peak 3.0)** | flac | FLAC/PCM_24 | **max diff 2.00** | **3.0000** | **1.0000** | **0** |

`encoding` and `bits_per_sample` are accepted by `Audio.save_to_file`
(`src/senselab/audio/data_structures/audio.py:310`) and are **never passed to torchcodec** —
the torchcodec branch constructs `AudioEncoder(samples=..., sample_rate=...)` and calls
`encoder.to_file(file_path)`. They reach `torchaudio.save` only in the dead fallback branch,
which would itself ignore them (with a `UserWarning`). So today they are pure decoration, and
not even the upstream warning surfaces: **zero warnings**.

Two further senselab-side notes from the same read:

- `Audio.convert_to_tensor` ends with `.to(torch.float32)` — so float64 input is silently
  narrowed on the way in, independently of any file format. Bit-exact float64 cannot survive
  the `Audio` object regardless of the encoder chosen.
- The workers already converged on the right answer without saying so: `driftse.py:49`,
  `unasdiff.py:237` and `yamnet.py:19` all define a module constant `= "FLOAT"` and pass it as
  `sf.write(..., subtype=...)`, which the matrix above shows is the only bit-exact,
  range-transparent path. But `sparc.py:126,165`, `qwen_tts.py:212` and
  `video/tasks/input_output.py:70` call `sf.write` with **no subtype**, and libsndfile's default
  for WAV is `PCM_16` — so three worker boundaries quantise to 16 bits and clamp, while three
  others do not. That inconsistency is inside senselab, not upstream.
