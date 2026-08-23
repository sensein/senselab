# Part 1 — upstream status

Searched via the GitHub API. **Note the repository moved**: it is
`meta-pytorch/torchcodec`, not `pytorch/torchcodec` (the latter 422s on search). The C++ error
paths emitted by the installed wheel confirm it —
`/Users/runner/work/torchcodec/torchcodec/meta-pytorch/torchcodec/src/torchcodec/_core/Encoder.cpp`.

## The silent clamping is acknowledged, intended, and closed as `not_planned`

**[meta-pytorch/torchcodec#1576](https://github.com/meta-pytorch/torchcodec/issues/1576) —
"AudioEncoder silently clips out-of-range float32 samples instead of rejecting or warning"**
Opened 2026-07-30 by `John6666cat`, closed 2026-07-30 (~6 hours later), state reason
`not_planned`. The report is essentially identical to ours: the constructor validates tensor
type, dimensions, dtype and sample rate but not the documented value range, so out-of-range
float32 is accepted and silently clipped, and "for sufficiently large input values, almost the
entire encoded waveform becomes full-scale PCM. The operation succeeds, but most amplitude
information is lost."

The maintainer's reply is the whole answer, verbatim:

> **NicolasHug:** Hi @John6666cat , we leave it up to the user to validate their input. We are
> usually quite defensive about user input, but in such a case it would significantly affect
> performance: checking the input is another `O(n)` check that most users would not want to pay.
> I don't think there's an easy way to validate the input on the fly as they're being encoded,
> since the encoding happens within FFmpeg.

The reporter accepted and closed it. So:

- **acknowledged?** Yes, explicitly, by the maintainer.
- **intended?** Yes — declined on `O(n)` performance grounds, not deferred.
- **documented?** The *contract* is: `AudioEncoder` documents that samples must be float32 in
  `[-1, 1]`, and `torchaudio.save`'s docstring repeats it ("Must be a 1D or 2D tensor with
  float32 values in the range [-1, 1]"). What is **not** documented is the consequence of
  violating it — nothing states that violation means silent clipping, and nothing documents that
  `.wav` means `pcm_s16le`.
- **will it change?** No. This is the settled upstream position.

**Consequence for senselab: the range check cannot be delegated below senselab, ever.** Upstream
has declined to do it, for a stated and reasonable reason. That is precisely the case for putting
the resolution and range policy in `Audio.save_to_file` plus a senselab-free
`portable_audio_io`, which is what PR #570 does.

## A bit-depth / encoding parameter is *unreported*

Searched `bits_per_sample` (**0 results**), `encoding parameter audio in:title` (0 results),
`sample_fmt`, `pcm_s24le`/`pcm_f32le`, `bit depth` (7 results, all video/HDR/pixel-format), and
every issue and PR with `AudioEncoder` in the title (11 results — `#692` public API, `#698` use
`AudioStreamOptions`, `#700` user-defined encoded sample rate, `#701` rename, `#717` docs, `#754`
file-like, `#836` FFmpeg 5 on Windows, `#850`/`#852` `pathlib.Path`, `#1411`/`#1478`/`#1480`
refactors, `#1576` above).

**Nobody has asked for PCM subtype / encoding control.** So ask #3 in the issue draft is novel,
and worth filing on its own — separately from #1576, which is closed and settled.

Also relevant, still **open**:
**[pytorch/audio#4211](https://github.com/pytorch/audio/issues/4211) — "torchaudio.save silently
saturates PCM-scale int16 input after the TorchCodec migration"** (opened 2026-07-30, no
comments). `torchaudio.save` casts `int16` to `float32` **without rescaling**, so int16 `26212`
becomes float `26212.0` and everything saturates. A wrapper-level bug distinct from #1576, and a
live hazard for any code that still hands `torchaudio.save` integer PCM.

## The chunking / seeking area is under active repair — this is the important part

Our Invariant 4 work landed in the middle of an open upstream workstream. Chronologically:

| ref | state | what |
|---|---|---|
| [#1448](https://github.com/meta-pytorch/torchcodec/issues/1448) | closed | `get_samples_played_in_range` decodes from the start up to `start_seconds` rather than seeking, so a window near the end of a long file costs a full decode. Reporter notes `soundfile.seek + read` is effectively constant-time. |
| [#1449](https://github.com/meta-pytorch/torchcodec/pull/1449) | closed | "Fix and speed-up AudioDecoder seeking logic" |
| [#1601](https://github.com/meta-pytorch/torchcodec/issues/1601) | closed/completed, **but reporter still reproducing** | `AudioDecoder` intermittently drops 1 sample in the final fractional-second chunk, ~1 in 3 runs. Reporter states it appears only after 0.14.0. Maintainer after the first fix: *"working on it. It's not trivial"*. |
| [#1610](https://github.com/meta-pytorch/torchcodec/issues/1610) | closed/completed | `get_samples_played_in_range` **fails outright** on MPEG/MPG audio with `start_seconds > 0` |
| [#1614](https://github.com/meta-pytorch/torchcodec/pull/1614) | **merged** 2026-08-12 | "Fix audio resampling: decoding in chunks should return identical samples to decoding in one go" — *"a resampled stream in consecutive chunks must return exactly the same samples as decoding it in one go … we must align the input and output 'grid' for the resampler … we enforce that through preroll + skipping some samples."* Adds `test_resample_chunked_matches_full`. |
| [#1645](https://github.com/meta-pytorch/torchcodec/issues/1645) | closed | `get_samples_played_in_range()` fails to seek in SWF audio |

**No open or closed issue matches the MP3 −1105-sample offset we measured** (searched `mp3 seek`,
`priming`/`pre-roll`/`delay samples`, `get_samples_played_in_range`, `get_all_samples mismatch`).
That specific defect appears unreported, and it reproduces on 0.16.0 — see below. It is the
better candidate for a new issue than the clamping, which is settled.

## Version comparison: 0.11.1 (pinned here) vs 0.16.0 (current)

Measured in an isolated venv, torchcodec 0.16.0 + torch 2.13.0 + FFmpeg 8.1. (The 0.16.0 macOS
wheel needs `DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib` — its dylibs ship with no `LC_RPATH`
and fail to load otherwise. Worth knowing before anyone upgrades.) Raw output:
`raw/torchcodec_0.16.0_check.txt`.

| behaviour | 0.11.1 | 0.16.0 |
|---|---|---|
| `AudioEncoder` signature | `(samples, *, sample_rate)` / `to_file(dest, *, bit_rate, num_channels, sample_rate)` | **identical — no encoding control added** |
| flat 1.5 → `.wav` | PCM_16, 1 distinct value, **no warning** | PCM_16, 1 distinct value, **no warning** |
| `.wav` / `.flac` / `.ogg` / `.wv` | `pcm_s16le` / 24-bit flac / **Ogg-FLAC** / `wavpack fltp` | **unchanged** |
| float32 round-trip via `.wav` | inexact, 1.53e−05 | inexact, 1.53e−05 |
| float32 round-trip via `.wv` | **bit-exact** | **bit-exact** |
| chunk == slice, wav / flac / opus | EXACT | EXACT |
| chunk == slice, **mp3** | **off = −1105**, short read 943 at t=0 | **off = −1105**, short read 943 at t=0 — **not fixed** |
| mp3 chunk contiguity | loses exactly 1105 samples | loses exactly 1105 samples |
| decode range-transparency (float WAV peak 4.0) | bit-exact, unclamped | bit-exact, unclamped |
| amplitude convention | `/32768` | `/32768` |
| **chunked decode with resampling vs one go** | **BROKEN: max diff 2.0e−01 at 16 kHz, 9.96e−01 at 8 kHz** | **bit-exact at both** (PR #1614) |
| **m4a chunk contiguity** | bit-exact | **NOT bit-exact**: 256 samples differ from index 132160, max diff 9.9e−01 (tail of the file) |

Two conclusions from this table, both actionable:

1. **The pinned 0.11.1 has a serious latent hazard: `AudioDecoder(path, sample_rate=N)` read in
   chunks does not equal the same file read in one go — max abs difference 0.20 at 16 kHz and
   0.996 at 8 kHz.** Order-unity, not rounding. senselab does **not** currently trip it: all
   three `AudioDecoder(...)` call sites (`audio.py:144`, `audio.py:204`,
   `video/data_structures/video.py:186`) decode at the native rate and resample separately. But
   this forecloses the obvious optimisation of letting the decoder resample, on 0.11.1, and it is
   exactly the kind of thing someone adds later as a speedup. PR #1614 fixes it in 0.16.0.
2. **Upgrading is not a free win.** 0.16.0 fixes the resampling-chunk bug but leaves the MP3
   −1105 shift untouched and introduces an AAC tail discrepancy of order 1.0 that 0.11.1 did not
   have — consistent with #1601 still being open in substance. Whichever version is pinned, the
   MP3 offset guard is required, and chunked AAC needs its own check.

## `torchaudio` I/O deprecation timeline

Verified directly against the installed 2.11.0 rather than from release notes:

- `torchaudio.load` / `torchaudio.save` **exist**, and are thin wrappers over torchcodec.
  `torchaudio/_torchcodec.py` does `from torchcodec.decoders import AudioDecoder` (line 82) and
  `from torchcodec.encoders import AudioEncoder` (line 246), each guarded to raise `ImportError`
  with *"TorchCodec is required for load_with_torchcodec"* / `save_with_torchcodec`.
- `torchaudio.info`, `torchaudio.list_audio_backends`, `torchaudio.io`, `torchaudio.backend`,
  `torchaudio.AudioMetaData` — **all absent** (`hasattr` → `False` for every one).
- The `save` docstring states the migration landed in **2.9** and that `format`, `encoding`,
  `bits_per_sample`, `buffer_size` and `backend` "are ignored and accepted only for backwards
  compatibility"; `load`'s says the same of `normalize`, `buffer_size` and `backend`. Both
  recommend porting to torchcodec directly.
- Relevant PRs, both closed: pytorch/audio#3975 "Add save_with_torchcodec, modify save()'s
  warnings" and #4039 "Let `torchaudio.load()` and `torchaudio.save()` rely on
  `load_with_torchcodec()` and `save_with_torchcodec()`"; also #4089 "Clarify load and save
  behavior".

**I did not find an authoritative announced removal version for `torchaudio.load`/`save`
themselves.** What is established is stronger than a timeline for our purposes: they are already
not an independent implementation. A senselab "torchaudio fallback" for encode or decode cannot
work when torchcodec is missing, because that is the exact condition under which torchaudio's own
I/O raises `ImportError`. The real fallbacks are `soundfile`, `ffmpeg`, and `scipy.io.wavfile`.
