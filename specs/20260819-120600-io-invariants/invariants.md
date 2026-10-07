# The decode side: which invariants hold, measured

Companion to `matrix.md` (which covers the format matrix and the encode side). Versions:
Python 3.12.0, torch/torchaudio 2.11.0, torchcodec 0.11.1, soundfile 0.13.1 / libsndfile 1.2.2,
librosa 0.11.0, numpy 2.4.4, ffmpeg 8.1. Scripts: `scripts/invariants*.py`; raw output in
`raw/invariants_run{1,2,3}.txt`.

Method note that changes a conclusion: the first pass compared every chunker against a
*torchcodec* full decode, which conflates "this library decodes differently" with "this
library's chunking is broken". Everything below compares **each library against its own full
decode**, with the alignment searched over ±3000–4000 samples so a shifted chunk is reported as
a shift rather than as noise. Cross-library agreement is reported separately.

---

## Invariant 1 — decode is source-independent in dtype and amplitude. HOLDS.

One float32 signal (peak 0.799973, RMS 0.461799, 44100 samples at 22050 Hz) encoded into 11
containers, all decoded by `torchcodec.AudioDecoder`:

| source | dtype | n | sr | peak | RMS | bit-exact vs. reference |
|---|---|---|---|---|---|---|
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
| **mp4 (h264 video + aac)** | float32 | **45056** | 22050 | 1.521272 | 0.450459 | no |

- **dtype**: always `float32`, from every source, from torchcodec, torchaudio, soundfile(f32)
  and librosa. `float64` only if you explicitly ask soundfile for it.
- **amplitude convention**: identical everywhere. A WAV/PCM_16 holding the literal int16 values
  `[-32768, -32767, -16384, 0, 16384, 32766, 32767]` decodes to
  `[-1.0, -0.99996948, -0.5, 0.0, 0.5, 0.99993896, 0.99996948]` — the `/32768` convention, to
  the last bit, from **all six** readers (torchcodec, torchaudio, soundfile f32, soundfile f64,
  librosa, ffmpeg). No reader disagrees.
- **audio from a video container is indistinguishable** from the same audio in a bare container:
  identical length, identical values.
- What is **not** source-independent: **length** (AAC adds 956 samples; nothing trims the padding)
  and **sample rate** (taken from the file, so an Opus file reports 48000 however it was made,
  and for one Opus file ffprobe says 48000 while libsndfile says 24000).

## Invariant 2 — decode does NOT clamp out-of-range float. HOLDS.

A WAV/FLOAT and a WAV/DOUBLE file with samples spanning ±4.0:

| reader | peak read back | bit-exact vs. source |
|---|---|---|
| torchcodec.AudioDecoder | 4.00000 | **YES** |
| torchaudio.load | 4.00000 | **YES** |
| soundfile.read(float32) | 4.00000 | **YES** |
| soundfile.read(float64) | 4.00000 | **YES** |
| librosa.load | 4.00000 | **YES** |
| ffmpeg CLI | 4.00000 | **YES** |

Same result whether the file was written by soundfile or by ffmpeg. **The asymmetry is precise:
torchcodec's read path is fully range-transparent; its write path is not.** Out-of-range audio
can be read into senselab; before PR #570 it could not be written back out again.

Lossy sources also return un-clamped: the reference signal at peak 0.7999 comes back at peak
1.0603 from mp3, 1.6327 from opus, 1.5213 from AAC. So `peak <= 1` is **not** an invariant a
caller may assume after decode, for any source.

## Invariant 3 — nothing normalises, at any scope. HOLDS.

Two tests, because whole-file and per-segment normalisation fail differently.

**Whole-file.** Two files identical except a `x0.1` gain, all six readers:

| source format | peak ratio (expect 0.1) | RMS ratio (expect 0.1) |
|---|---|---|
| WAV/FLOAT | 0.100000 | 0.100000 |
| WAV/PCM_16 | 0.100010 | 0.100000 |
| FLAC/PCM_24 | 0.100000 | 0.100000 |

The 0.100010 on PCM_16 is 16-bit quantisation of the single peak sample, not gain: the RMS
ratio is exactly 0.1.

**Per-segment — the one that would invalidate every windowed measurement in this project.** A
6 s file whose first half is at peak 0.899982 and second half at peak 0.000900 (−60 dB). Decode
the quiet half as a seeked chunk, and compare to the same range sliced out of that library's
own full decode. A per-segment peak or RMS normaliser would boost the quiet chunk; the ratio
would be ~1000, not 1.

| format | reader | quiet-chunk peak | slice peak | chunk/slice | bit-exact |
|---|---|---|---|---|---|
| wav FLOAT32 | torchcodec / torchaudio / soundfile / ffmpeg | 0.000899961 | 0.000899961 | **1.000000** | **True** |
| wav PCM_16 | all four | 0.000915527 | 0.000915527 | **1.000000** | **True** |
| flac PCM_24 | all four | 0.000899911 | 0.000899911 | **1.000000** | **True** |
| m4a aac | torchcodec / torchaudio / ffmpeg | 0.150671810 | 0.150671810 | **1.000000** | **True** |
| **mp3 192k** | **torchcodec** | **1.163850546** | 0.117209934 | **9.929624** | **False** |
| mp3 192k | torchaudio / soundfile / ffmpeg | 0.117209934 | 0.117209934 | 1.000000 | True |

**No decode path in this set normalises, per file or per segment**, at 60 dB of dynamic range.
The single failing cell is *not* normalisation — it is Invariant 4's MP3 offset defect showing
its consequences: the chunk landed 1105 samples early, straddling the loud/quiet boundary, so a
windowed measurement of the quiet half reads **9.93x too loud**. That is the concrete shape of
the corruption, and it is an indexing bug, not a gain bug.

(Per-segment normalisation *does* exist in this codebase — DriftSE's per-chunk peak
normalisation — but it is introduced by senselab's own worker code, above the I/O boundary. The
decode layer is innocent.)

## Invariant 4 — chunk equals slice. HOLDS for WAV and FLAC. FAILS for MP3 via torchcodec.

Each library vs. its own full decode; `off` = how far the returned chunk actually sits from
where it was requested, in samples; requests at an aligned offset (0), an odd offset (5001) and
mid-file (44877).

| format | torchcodec range | torchaudio (=tc) | soundfile `start=` | ffmpeg `-ss` **before** `-i` | ffmpeg `-ss` **after** `-i` |
|---|---|---|---|---|---|
| wav PCM_16 | **EXACT** 0.0e+00 | **EXACT** | **EXACT** | **EXACT** | **EXACT** |
| wav FLOAT32 | **EXACT** 0.0e+00 | **EXACT** | **EXACT** | **EXACT** | **EXACT** |
| flac PCM_24 | **EXACT** 0.0e+00 | **EXACT** | **EXACT** | **EXACT** | **EXACT** |
| mp3 192k | values exact but **off = −1105**; short read at t=0 (943 of 2048) | **EXACT** 0.0e+00 | off=0, **1.8e−07** | off=0, **1.2e+00 — wrong data** | **EXACT** 0.0e+00 |
| m4a aac | **EXACT** 0.0e+00 | **EXACT** 0.0e+00 | cannot open | off=0, **up to 1.1e+00 — wrong data** | **EXACT** 0.0e+00 |
| opus 48k | **EXACT** 0.0e+00 | **EXACT** 0.0e+00 | **EXACT** 0.0e+00 | **EXACT** | **EXACT** 0.0e+00 |

**Contiguity** (concatenate consecutive 2048-sample chunks, compare to the full decode):
bit-for-bit identical, 0 mismatched samples, for WAV/PCM_16, WAV/FLOAT32 and FLAC/PCM_24 via
torchcodec, soundfile and torchaudio alike. senselab's `Audio.from_stream` (soundfile blocks)
also reconstructs the full decode bit-for-bit at 0.25 s chunks on all three. For MP3 via
torchcodec the concatenation is **1105 samples short** (131195 vs 132300).

### The MP3 defect, exactly

torchcodec's two entry points use two different timelines for the same file:

| requested start | n returned (2048 asked) | `pts_seconds` reported | offset vs. `get_all_samples()` |
|---|---|---|---|
| 0 | **943** | 0.050113379 (= sample 1105) | 0 |
| 1 | 944 | 0.050113379 (sample 1105) | −1 |
| 100 | 1043 | 0.050113379 (sample 1105) | −100 |
| 576 | 1519 | 0.050113379 (sample 1105) | −576 |
| 1105 | 2048 | 0.050113379 (sample 1105) | −1105 |
| 2048 | 2048 | 0.092879819 (sample 2048) | −1105 |
| 5001 | 2048 | 0.226802721 (sample 5001) | −1105 |
| 22050 | 2048 | 1.000000000 (sample 22050) | −1105 |

`get_all_samples()` prepends **1105 samples** of MP3 decoder pre-roll (576 + 529 — the standard
MDCT + LAME delay) which `get_samples_played_in_range()` excludes. So for an MP3, index `i` of a
full decode is presentation sample `i − 1105`, and **a windowed measurement taken via offsets is
shifted by 1105 samples — 50.1 ms at 22.05 kHz — relative to a whole-file measurement of the same
file.** Below `start = 1105` the range API additionally short-reads, returning
`n − (1105 − start)` samples with no warning.

The chunk *values* are exact once aligned (0.0e+00 at off = −1105), so this is purely an
indexing defect. Two further properties, both measured:

- **It is not frame-boundary dependent.** Sweeping the seek offset across a full MP3 frame
  (1152 samples) at +0, +1, +288, +576, +1151, +1152, +1153, +2304 gives off = −1105 and
  diff = 0.0e+00 at *every* offset. A constant shift, not jitter — which makes it correctable.
- **`pts_seconds` on the returned `AudioSamples` reports the true position**, and aligning
  against it is bit-exact in every row. So the information needed to detect and fix this is
  present in the API. **senselab discards it**:
  `Audio._lazy_load_data_from_filepath` returns `samples.data` and drops the rest, so
  `Audio(filepath=..., offset_in_sec=...)` inherits the shift verbatim — measured off = −1105
  for mp3, 0 for wav / flac / m4a / opus.

`torchaudio.load(frame_offset=, num_frames=)` is exact for mp3 despite being a torchcodec
wrapper, because it decodes then slices rather than seeking.

### Two traps worth pinning in a test

- **ffmpeg `-ss` placement.** `-ss` *before* `-i` (fast/keyframe seek) returns **wrong data** for
  mp3 (max diff 1.2) and aac (up to 1.1) — a different part of the signal, silently. `-ss`
  *after* `-i` (decode-and-discard) is bit-exact for every format tested. Same for AAC and Opus,
  which are otherwise clean at every offset swept across their frame sizes (1024 and 960).
- **soundfile + mp3** is positionally correct (off = 0) but **not** bit-identical to its own full
  decode: tolerance **1.2e−07 to 1.8e−07**, present at every offset except a full-frame multiple.
  Correct to float32 rounding, not to the bit.

### Cross-library agreement on a full decode (decoder identity, not chunking)

| format | torchcodec vs torchaudio | vs soundfile | vs ffmpeg |
|---|---|---|---|
| wav PCM_16 / FLOAT32 | EXACT | EXACT | EXACT |
| flac PCM_24 | EXACT | EXACT | EXACT |
| mp3 192k | EXACT | **2.46e−06** | EXACT |
| m4a aac | EXACT | cannot open | EXACT |
| opus 48k | EXACT | **1.13e−06** | EXACT |

For lossless formats all four decoders are bit-identical, so the choice is free. For lossy
formats libsndfile's decoders (mpg123, libopusfile) differ from ffmpeg's by ~1–2e−06, so "which
library decoded this mp3" is a provenance fact, not a free choice.

---

## The invariants a caller may rely on, stated for a test to pin

Given the primary decode path below, these hold and are cheap to assert:

1. **dtype is always `float32`**, shape `(channels, samples)`, for every format and container,
   including audio inside a video file.
2. **Amplitude is the `/32768` convention** and is never rescaled: full-scale negative int16 is
   exactly `−1.0`, and a `x g` gain in the file is a `x g` gain in the array, to float precision.
3. **Nothing is normalised** — not per file, not per segment. Two windows of one recording are
   directly comparable, at 60 dB of dynamic range.
4. **Out-of-range float survives decode unclamped and bit-exact.** `peak <= 1` must not be
   assumed, especially from a lossy source (measured up to 1.63).
5. **`read(path)[a:b] == read(path, offset=a, duration=b-a)`, bit-for-bit**, for WAV (any
   subtype) and FLAC, at aligned, odd and mid-file offsets, and consecutive chunks concatenate
   back to the whole file with 0 mismatched samples.
6. **Invariant 5 does not hold for MP3 through `torchcodec.get_samples_played_in_range`** — a
   constant −1105-sample shift plus a silent short read below sample 1105. It does hold via
   `torchaudio.load(frame_offset=...)`, `soundfile.read(start=...)` (to 1.8e−07) and
   `ffmpeg -i … -ss` (bit-exact).
7. **Sample rate and channel count are preserved** by every path for wav / flac / mp3 /
   ogg-vorbis. **Not for Opus** (forced to a supported rate; two readers can disagree about
   which). **Length is not preserved for AAC** (up to one 1024-sample frame of untrimmed
   padding).

`portable_audio_io`'s read side must honour 1–5 identically, or a worker's read will disagree
with the parent's. Since the workers have only numpy + soundfile, and since soundfile is
bit-identical to torchcodec on WAV and FLAC for both whole-file and chunked reads, that is
achievable **for lossless formats only** — which is the right constraint for an IPC boundary
anyway.

---

## Recommendation

### Decode: `torchcodec.AudioDecoder` primary; `soundfile` first fallback; `ffmpeg -i … -ss` last.

Argued against the alternatives:

- **Why not soundfile primary?** It cannot open any video container (`Format not recognised` on
  mp4, mkv, mov), cannot open m4a/aac at all, and cannot open the Ogg-FLAC that torchcodec
  itself writes for `.ogg`. senselab handles video. That rules it out as the primary.
- **Why not librosa?** It reads m4a and video only through `audioread`, which emits
  `FutureWarning: Deprecated as of librosa version 0.10.0. It will be removed in …`. Building on
  a path that announces its own removal, to get a capability torchcodec already has, is a bad
  trade. It also gave a different length for AAC-in-MP4 than every other reader (48000 vs 48128).
- **Why not ffmpeg CLI primary?** It is bit-identical to torchcodec everywhere and the most
  capable, but it is a subprocess per read, it has the `-ss` placement trap, and it is an
  external binary rather than a pinned wheel. Ideal as the last resort, wrong as the default.
- **Why not torchaudio?** In 2.11.0 it *is* torchcodec (`torchaudio/_torchcodec.py` imports
  `AudioDecoder` and raises `ImportError` without it), with `torchaudio.info`,
  `list_audio_backends`, `io` and `backend` already removed. Depending on it adds a deprecation
  surface and buys nothing — with one exception worth exploiting: its `frame_offset` slicing is
  exact for MP3 where the torchcodec range API is not.
- **So: torchcodec primary**, because it is the only single path covering every format *and*
  video, it returns float32 with a consistent amplitude convention, it never normalises, and it
  is bit-identical to soundfile and ffmpeg on all lossless formats. Its decode side has none of
  the defects its encode side has.

With two required guards, both measured, not speculative:

- **MP3 must not use `get_samples_played_in_range`.** Either read the whole file and slice, or
  use the returned `pts_seconds` to correct the offset (bit-exact in every row tested), or route
  MP3 offsets through `soundfile.read(start=)`. Left alone, every windowed MP3 measurement in
  the project is shifted 50.1 ms against its whole-file counterpart — and at a level transition
  that read 9.93x too loud in the test above.
- **`Audio` should stop discarding `pts_seconds`.** It is the only evidence available at runtime
  that a chunk is not where it was asked for.

Formats covered: everything measured — wav (all subtypes), flac, ogg-vorbis, ogg-flac, opus,
mp3, m4a/aac, wavpack, and audio inside mp4/mkv/mov. Not covered by the fallback (soundfile):
m4a/aac, ogg-flac, wavpack, and all video. That gap is exactly why the fallback is second.

### Exactness: achievable, and where it is impossible in principle

- **Exact for arbitrary float32**: WAV `FLOAT` / `DOUBLE` only, and WavPack (`.wv`). Written by
  soundfile or ffmpeg, read by any of the six paths — bit-exact, and out-of-range values
  preserved (peak 4.0 survived).
- **Exact only for values already on the target integer grid**: WAV PCM_16/24/32, FLAC 16/24.
  A signal on the 16-bit grid round-trips bit-exactly through all of them; arbitrary float32
  does not, by construction (1.5e−05 at 16 bits, 1.2e−07 at 24, 2.3e−10 at 32).
- **Impossible in principle**: **FLAC cannot carry float at any depth** — the format is integer
  PCM only (libsndfile offers PCM_S8/16/24 and nothing else). No encoder choice fixes this;
  FLAC is the wrong container for a float measurement artifact, full stop. Likewise every lossy
  codec, which additionally *overshoots* rather than clamps (peak 3.0 in, 4.9 out).

### Sample-format control on `Audio.save_to_file`

**Keep the capability; #570 changed its spelling to `subtype`.** Before #570 they were pure
decoration: `save_to_file` accepted both and passed neither to `AudioEncoder`, so the measured
result was zero warnings and a silent PCM_16 write — worse than not offering them, because a
caller who set `encoding="PCM_F", bits_per_sample=32` got 16-bit integer and was told nothing.
The argument for removal ("only some backends honour them") is answered the other way round: the
two backends that *can* honour them, soundfile and ffmpeg, are the two that can write float at
all, and the one that cannot — torchcodec — cannot express a float write under any argument. So
the parameters are not backend-specific decoration, they are the only way to say the one thing
that matters, and the correct place to resolve them is exactly where #570 put them: above the
backend, in senselab, where an unsatisfiable request can be refused rather than silently
downgraded.

What #570 shipped is that argument with a different interface. `encoding` and `bits_per_sample`
are gone, replaced by one `subtype` plus an `out_of_range` policy. The pair said the same thing
twice and could disagree with itself (`encoding="PCM_S"` with `bits_per_sample=32` is not a
libsndfile subtype), whereas `subtype` is libsndfile's own vocabulary — `PCM_16`, `PCM_24`,
`FLOAT` — so every accepted value names a format that exists. `resolve_subtype` supplies the
default (`FLOAT` for a float-capable container), and `subtype_preference` separates a demand,
which raises when the container cannot carry it, from a codec hint, which degrades to the widest
the container has.

One inconsistency inside senselab that the matrix exposed, **measured against `triage` before
#570 merged and resolved by it** — all six sites now route through the layer: `driftse.py:49`, `unasdiff.py:237` and `yamnet.py:19` each define a module constant
`= "FLOAT"` and pass it to `sf.write(..., subtype=...)` — the correct, bit-exact,
range-transparent path. But `sparc.py:126,165`, `qwen_tts.py:212` and
`video/tasks/input_output.py:70` call `sf.write` with **no subtype**, and libsndfile's default
for WAV is `PCM_16`. Three worker boundaries quantise to 16 bits and clamp; three do not.

### Different paths for different purposes

| purpose | path | why |
|---|---|---|
| read anything, including video | `torchcodec.AudioDecoder` (guard MP3 offsets) | only path covering every format; float32, no normalisation, no clamping |
| write a measurement artifact | WAV `FLOAT` via soundfile (or ffmpeg `pcm_f32le`) | the only bit-exact, range-transparent target |
| worker IPC (numpy + soundfile only) | WAV `FLOAT` via soundfile both ways | bit-identical to torchcodec on WAV, chunk-exact, no senselab import |
| archive / distribute | FLAC 24 via soundfile, **after** an explicit in-range check | lossless and small, but integer-only — the check cannot be delegated |
| read a lossy or video source at an offset | `soundfile.read(start=)` for mp3; `ffmpeg -i … -ss` otherwise | avoids torchcodec's MP3 shift and ffmpeg's `-ss`-before-`-i` trap |

How a caller knows: by the invariant list above, asserted in tests, rather than by reading
backend source. That is the deliverable.

---

## Version-dependence of these invariants (added after cross-checking 0.16.0)

The invariants above were measured on the pinned **torchcodec 0.11.1**. Re-measured on **0.16.0**
(isolated venv, torch 2.13.0, FFmpeg 8.1; raw output `raw/torchcodec_0.16.0_check.txt`):

- Invariants 1–4 (dtype, amplitude, no normalisation, no clamp on read) are **unchanged**.
- Invariant 5 holds on both for WAV, FLAC and Opus.
- Invariant 6 (the MP3 exception) is **unchanged on 0.16.0** — same −1105 shift, same 943-sample
  short read, same 1105 samples lost over a chunked pass. Not fixed, and unreported upstream.
- **New on 0.11.1 and fixed in 0.16.0:** chunked decoding *with the decoder's own resampler*
  (`AudioDecoder(path, sample_rate=N)`) does **not** equal one-go decoding on 0.11.1 — max abs
  difference **2.0e−01 at 16 kHz** and **9.96e−01 at 8 kHz**. Order-unity, not rounding. Fixed by
  upstream PR #1614, bit-exact on 0.16.0 at both rates.

  senselab does not currently trip this: all three `AudioDecoder(...)` call sites
  (`audio.py:144`, `audio.py:204`, `video/data_structures/video.py:186`) decode at the native rate
  and resample separately. So this is a **latent** hazard, not an active corruption — but it means
  "let the decoder resample, it's cheaper" is not a safe optimisation on 0.11.1, and someone will
  eventually propose it.
- **New on 0.16.0 and absent on 0.11.1:** chunked AAC contiguity is no longer bit-exact — 256
  samples differ from index 132160 (the file tail), max abs difference **9.9e−01**. Consistent
  with upstream #1601 ("drops 1 sample in final fractional-second chunk"), which is marked
  completed but which the reporter still reproduces and the maintainer describes as "not trivial".

**Implication for the recommendation.** Upgrading torchcodec is not a free win: 0.16.0 buys the
resampling-chunk fix and costs an AAC tail discrepancy, while the MP3 shift survives both. So the
MP3 offset guard and a chunked-AAC check are required at either version, and the invariant list
above should be asserted in senselab's own tests rather than assumed from a version pin. That is
the practical argument for writing them down as tests: they are the only thing that will notice
when a torchcodec upgrade moves one of them.
