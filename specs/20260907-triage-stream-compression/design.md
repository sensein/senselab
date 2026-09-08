# Persisted triage streams: FLAC instead of float32 WAV

A 388-recording corpus run wrote 1262 stream files (`plain`, `preemphasised`, `normalized`,
`residual` per recording) totalling 6.7 GB. All four are float32 WAV via
`LOSSLESS_WAV_SUBTYPE = "FLOAT"` in `portable_audio_io.py`, chosen on purpose (`yamnet.py:178`,
`hear.py:493`): PCM_16 replaces faint residual content below roughly −96 dBFS with quantization
noise that reads back louder than it should, and a classifier then responds to the noise.

## What torchcodec decodes

Verified in this environment (`torchcodec==0.11.1`, ffmpeg 8.1 with `libavcodec` `flac`, `wavpack`,
`alac` built in):

| container/codec | write path | decode via torchcodec | round-trip on a real residual |
| --- | --- | --- | --- |
| FLAC / PCM_24 | `soundfile` (libsndfile) | correct | max abs diff `1.19e-7` (quantization floor) |
| WavPack (`.wv`) / float32 | `torchcodec.encoders.AudioEncoder` (ffmpeg) | correct | max abs diff `0` (bit-exact) |
| ALAC in CAF / `ALAC_32` | `soundfile` (libsndfile) | **drops samples**: `57448` decoded vs `59444` true (`ffprobe duration_ts`) | disqualified — this is data loss, not quantization, and reproduces with plain `ffmpeg -i … -f f32le` (not a torchcodec-only defect) |
| ALAC in CAF / `ALAC_16`/`20`/`24` | `soundfile` | same sample-count loss as `ALAC_32` | disqualified |

`soundfile.available_formats()` in this environment does not include `WAVPACK` at all —
`portable_audio_io.py` (soundfile-only by its own import contract, staged into subprocess workers)
cannot write or read WavPack regardless of what torchcodec can do with it. That, not the round-trip
result, is why WavPack is not the chosen format: it would solve the range problem below by
construction (float32, no ceiling to clip against) but the writer that must stay usable inside a
staged worker cannot produce it.

Code run for this table: `AudioEncoder(samples=…).to_file("test.{flac,wv,m4a,caf}")` then
`AudioDecoder(dest).get_all_samples()`, and `soundfile.write(…, format="CAF", subtype="ALAC_32")`
followed by both `soundfile.read` and `AudioDecoder`/raw `ffmpeg -f f32le` on the result.

## Compression, measured on the 98 real cluster residuals

`~/Downloads/frcrn_cluster_residuals_20260907/` — 98 `(plain.wav, residual.wav)` pairs, 196 files,
224.57 MB of float32 WAV.

| format | total size | ratio | size reduction |
| --- | --- | --- | --- |
| FLAC / PCM_24 | 104.82 MB | 2.142x | 53.3% |
| WavPack / float32 (measured, not chosen — see above) | 164.71 MB | 1.363x | 26.7% |

Projected for the 388-recording run that produced 6.7 GB: `6.7 GB / 2.142 ≈ 3.1 GB`, extrapolating
the measured ratio from `plain`/`residual` to `preemphasised`/`normalized`/`separated_*` (not in the
downloaded sample, but the same class of signal at the same rate).

## Range: how many samples exceed ±1

Across the 98 residuals: 153 of 28,069,572 samples (`5.4e-6`), confined to 10 of the 98 files, peaks
1.01–1.42. Across the 98 `plain` files: zero — `plain` is already peak-normalized in `preprocess.py`
before it is persisted, so it never approaches the ceiling. `residual = original − g·enhanced` is
the one persisted stream this corpus shows going out of range, and only rarely.

FLAC/PCM_24 cannot represent a sample beyond ±1; `soundfile` clips it there without wraparound
(verified: writing `[1.4193, −1.4193]` reads back `[0.9999999, −1.0]`, not an overflowed value).
Clipping is data loss and is not used. The write goes through `portable_audio_io`'s existing
`apply_range_policy` with `out_of_range="normalize"`: a peak already at or below 1.0 (a clipped
*original* recording — content, not a write hazard) passes through unchanged; a peak the write
would otherwise clip is scaled down by `1/peak` first, and the gain is recorded on the stream
entity as `write_gain` rather than silently disappearing.

## Classifier acceptance test

`classify_audios([...], model="yamnet", top_k=5)`, one batched call, on the 10 out-of-range
residuals + 5 further random residuals + 5 random `plain` files (20 files, 1034 YAMNet windows).

- **FLAC with clipping** (not the chosen policy, measured for comparison): 2/1034 (0.19%) per-window
  top-label disagreements, max score delta 0.10.
- **FLAC with the chosen `normalize`-to-fit policy**: 17/1034 (1.6%) per-window top-label
  disagreements, max score delta 0.37. A whole-clip summary metric (single highest-scoring window
  per clip) showed 1/20 apparent mismatches, but that metric compares different time windows across
  clips and is misleading; isolating the worst case (peak 1.29 from 8 outlier samples in a
  480,251-sample file) shows the ~2.2 dB whole-file gain reduction needed to fit those 8 samples
  moves a handful of already-low-confidence (<0.6) windows near the Speech/Silence boundary, while
  every high-confidence window (>0.9) stayed within ~0.01–0.03 of its original score. A parallel
  float32-WAV control at the same gain reproduced the FLAC result almost exactly, isolating the
  effect to the required rescale, not FLAC's quantization.
- No high-confidence label changed in the tested sample under either policy. The measurable cost of
  the "never write a clipped sample" invariant is real but small, and confined to the same rare
  files that already needed rescaling.

## Decision

FLAC (`soundfile`'s default subtype resolution picks `PCM_24`), `out_of_range="normalize"`, via one
helper — `write_stream` in
`src/senselab/audio/workflows/triage/nodes/common.py` — that every node writing a persisted stream
calls instead of `Audio.save_to_file` directly. Reading needs no format-specific code: `Audio`'s
lazy load (torchcodec, torchaudio fallback) and `resolve_stream` already resolve whatever extension
is recorded, so existing runs' `.wav`-referencing stores keep working unchanged.

Transient hand-off WAVs (`write_worker_wav` in `yamnet.py`, the HeAR equivalent, and every
`audio.save_to_file(temp_path)` a subprocess-invoking function writes just before calling a worker)
are untouched: traced through `unasdiff.py`, `driftse.py`, `voice_cloning/{coqui,sparc}.py`, and the
`classify_audios` workers, every one of these re-serializes a fresh temp WAV from the in-memory
`Audio` immediately before the subprocess call — none of them ever decodes a persisted stream's own
file. So the container choice for persisted streams does not touch the subprocess-venv boundary in
this pipeline today, though `soundfile` remains worth declaring explicitly (not transitively) in any
venv that does gain a direct stream-decode path later.

## Open item for the concurrent residual-block change

`preprocess.py`'s residual block (owned by a different concurrent change) must call `write_stream`
for its `residual` write and, when it lands, the new `enhanced` stream — the same as `plain`,
`preemphasised`, `normalized`, and `speech.py`'s `separated_*` already do here. Nothing else about
the residual computation itself was touched.
