# DRAFTS — not filed. Proposed issues for `meta-pytorch/torchcodec`

**Do not file without a final duplicate check.** Note the repo is `meta-pytorch/torchcodec`.

**The silent-clamping issue is already settled and must NOT be re-filed.**
[#1576](https://github.com/meta-pytorch/torchcodec/issues/1576) reported exactly our clamping
behaviour and was closed `not_planned` within six hours: the maintainer's position is that range
validation is the caller's job because it costs an extra `O(n)` pass. Re-filing it would be a
duplicate of a deliberate decision. See `upstream.md` for the verbatim exchange.

What *is* unreported, and worth filing, is below: **Draft A** (an MP3 correctness bug, the
stronger of the two) and **Draft B** (the encoding-parameter feature request, which no issue or PR
has ever raised — `bits_per_sample` returns zero search results).

Measured on torchcodec **0.11.1** *and independently reproduced on **0.16.0*** (torch 2.13.0,
FFmpeg 8.1, Python 3.12, macOS arm64) unless noted.

---

# Draft A — `get_all_samples()` and `get_samples_played_in_range()` disagree by 1105 samples on MP3

## Summary

For an MP3 file, `AudioDecoder`'s two entry points index the same stream on two different
timelines. `get_all_samples()` returns 1105 extra leading samples of decoder pre-roll that
`get_samples_played_in_range()` excludes. Consequently, for an MP3:

- index `i` of a full decode is presentation sample `i − 1105`;
- concatenating consecutive ranges over the whole file yields **1105 fewer samples** than
  `get_all_samples()`;
- a request with `start_seconds` below sample 1105 **silently short-reads**, returning
  `n − (1105 − start)` samples with no warning.

The sample *values* are correct once aligned, so this is purely an indexing inconsistency. It is
also a **constant** shift, independent of where the seek lands relative to a frame boundary, which
makes it straightforward to reason about and to fix.

Why it matters: any analysis that windows a file via `get_samples_played_in_range` and compares
against a whole-file analysis of the same file is misaligned by 1105 samples — 50.1 ms at
22.05 kHz. In our case a windowed level measurement of a quiet passage read **9.9x too loud**,
because the shifted window straddled a loud/quiet transition.

## Reproducer

```python
import subprocess
import numpy as np, soundfile as sf
from torchcodec.decoders import AudioDecoder

SR = 22050
sig = (np.random.default_rng(23).integers(-32768, 32768, SR * 6) / 32768.0).astype(np.float32)
subprocess.run(["ffmpeg","-hide_banner","-v","error","-y","-f","f32le","-ar",str(SR),
                "-ac","1","-i","-","-c:a","libmp3lame","-b:a","192k","c.mp3"],
               input=sig.tobytes(), check=True)

full = AudioDecoder("c.mp3").get_all_samples().data.numpy()
print(full.shape)                                    # (1, 132300)

# a window in the middle
a, n = 5001, 2048
s = AudioDecoder("c.mp3").get_samples_played_in_range(
        start_seconds=a / SR, stop_seconds=(a + n) / SR)
ch = s.data.numpy()

print(np.abs(full[:, a:a+n] - ch).max())             # 1.8   <- not the requested window
print(np.abs(full[:, a-1105:a-1105+n] - ch).max())   # 0.0   <- shifted by exactly 1105
print(s.pts_seconds * SR)                            # 5001.0 (the request WAS honoured on the
                                                     #         container timeline)

# at the start, a silent short read
s0 = AudioDecoder("c.mp3").get_samples_played_in_range(start_seconds=0.0, stop_seconds=n / SR)
print(s0.data.shape[-1], s0.pts_seconds * SR)        # 943  1105.0   (asked for 2048)

# and the whole file, chunked, loses 1105 samples
parts, t = [], 0
while t < full.shape[1]:
    m = min(2048, full.shape[1] - t)
    parts.append(AudioDecoder("c.mp3").get_samples_played_in_range(
        start_seconds=t / SR, stop_seconds=(t + m) / SR).data.numpy())
    t += 2048
print(np.concatenate(parts, axis=1).shape[-1])       # 131195  vs  132300
```

1105 = 576 + 529, the standard MPEG MDCT + LAME encoder delay.

## Expected vs. actual

| | expected | actual |
|---|---|---|
| `get_all_samples()` vs. concatenated ranges | same samples, same count | ranges total 1105 fewer |
| `full[a:a+n]` vs. `range(a/sr, (a+n)/sr)` | identical | shifted by 1105 |
| `range(0, n/sr)` | `n` samples, or a documented reason for fewer | `n − 1105` samples, silently |

WAV, FLAC, Opus and AAC are all exact at every offset tested, so MP3 (and, per
[#1610](https://github.com/meta-pytorch/torchcodec/issues/1610), MPEG/MPG) look like the affected
family. This is adjacent to
[#1601](https://github.com/meta-pytorch/torchcodec/issues/1601) and
[#1614](https://github.com/meta-pytorch/torchcodec/pull/1614) — #1614 established the principle
that "decoding in chunks should return identical samples to decoding in one go" and fixed it for
the resampling case; this is the same principle failing for MP3 pre-roll.

## The ask

Make the two entry points agree — either by having `get_all_samples()` drop the pre-roll (so both
start at the first presented sample), or by having the range API address the same timeline
`get_all_samples()` exposes. Failing that, document the offset and stop silently short-reading
below sample 1105. Note that `AudioSamples.pts_seconds` already reports the true position
correctly, so the information needed for a fix — or for a caller-side workaround — is present.

## Also reproduced on 0.16.0

Identical numbers: `off = −1105` at every offset, 943-sample short read at `t=0`, 1105 samples
lost over a chunked full pass.

---

# Draft B — no way to select a sample format / bit depth when encoding

## Summary

`AudioEncoder` chooses the output sample format from the file extension alone, and for every
lossless target it selects an integer format:

| extension | codec / `sample_fmt` |
|---|---|
| `.wav`, `.w64` | `pcm_s16le` / s16 |
| `.caf`, `.aiff`, `.au` | `pcm_s16be` / s16 |
| `.flac` | `flac` / s32, 24 bits_per_raw_sample |
| `.ogg`, `.oga` | `flac` in Ogg / s32, 24 |
| `.wv` | `wavpack` / **fltp** |

There is no argument that changes this: `to_file` takes only `bit_rate`, `num_channels` and
`sample_rate`, and `bit_rate` is bits-per-second for lossy codecs, not PCM bit depth. So a caller
cannot request `pcm_f32le`, `pcm_s24le` or `pcm_s32le`, all of which FFmpeg supports.

Combined with the settled decision in #1576 that range validation is the caller's job, this leaves
callers with float data outside `[-1, 1]` no correct target at all: the check is theirs to make,
but the format that would carry the data is not expressible. Writing a float WAV is a normal
requirement for measurement and research pipelines, where intermediate signals routinely exceed
unity (enhancement, source separation, spectral inversion, gain staging).

## Reproducer

```python
import numpy as np, soundfile as sf, torch
from torchcodec.decoders import AudioDecoder
from torchcodec.encoders import AudioEncoder

z = (np.random.rand(1000).astype(np.float32) * 6 - 3)          # peak ~3.0
AudioEncoder(samples=torch.from_numpy(z[None]), sample_rate=16000).to_file("x.wav")
print(sf.info("x.wav").subtype)                                 # PCM_16  (not requestable)
print(AudioDecoder("x.wav").get_all_samples().data.abs().max())  # 1.0     (was 3.0)

# both alternatives can express it, and both round-trip bit-exactly:
sf.write("y.wav", z, 16000, format="WAV", subtype="FLOAT")
assert sf.read("y.wav", dtype="float32")[0].tobytes() == z.tobytes()
# ffmpeg -f f32le -ar 16000 -ac 1 -i - -c:a pcm_f32le y2.wav   -- likewise bit-exact
```

`AudioDecoder` reads such a file back **correctly and unclamped** — peak 4.0 stays 4.0, bit-exact
— so the capability gap is only on the write side.

## The ask

1. A `sample_format=` (or `encoding=` / `bits_per_sample=`, mirroring the `torchaudio.save`
   signature this replaced) on `to_file` / `to_tensor` / `to_file_like`. FFmpeg already exposes
   the formats, so this is plumbing rather than new capability.
2. Failing that, **document the extension → sample-format mapping.** Neither the `AudioEncoder`
   API reference nor the encoding tutorial states that `.wav` means 16-bit integer, so a caller
   has no way to learn that a float write is lossy other than by inspecting the output.
3. Consider defaulting `.wav` to `pcm_f32le` when the input is float32 and the container supports
   it — this would make the documented `[-1, 1]` contract and the default behaviour consistent.

Context: `torchaudio.save` has forwarded to `AudioEncoder` since 2.9 and warns that `encoding` and
`bits_per_sample` are unsupported, so for code migrating off torchaudio this is a capability
regression. Searching this repo, `bits_per_sample` returns **zero** results and no issue or PR in
the `AudioEncoder` history requests sample-format control, so as far as I can tell this has simply
never been raised.

## Two smaller observations, same root cause

- **`.ogg` produces Ogg-FLAC, not Ogg-Vorbis** (`ffprobe`: `codec_name=flac`, container `ogg`).
  Lossless, so arguably better than asked for, but libsndfile refuses it —
  `soundfile.read` fails with `unknown error in flac decoder` — so a file written as `.ogg` cannot
  be read by a large part of the Python audio ecosystem. `.oga` behaves the same way.
- **`.wv` is the only float-preserving target**, and it round-trips out-of-range data bit-exactly
  (peak 4.0 in, 4.0 out). That the one float-capable option is undocumented and reachable only by
  guessing an extension is itself the argument for making the format choice explicit.

## Environment

torchcodec 0.11.1 and 0.16.0; torch 2.11.0 / 2.13.0; torchaudio 2.11.0; FFmpeg 8.1;
libsndfile 1.2.2 via soundfile 0.13.1; Python 3.12.0; numpy 2.4.4; macOS arm64.
