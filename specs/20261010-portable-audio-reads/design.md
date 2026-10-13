# Every audio read goes through `portable_audio_io`

## Why this was believed done

`portable_audio_io` arrived with PR #570 (`2f5a9141`, 2026-08-19, "route every audio write through one
policy"). That change covered **writes only**: thirteen write sites across four APIs were routed
through `write_audio` and `audio_write_boundary_test.py` was added. The module also gained a
`read_audio`, but it was a thin `sf.read` wrapper used by five subprocess workers (driftse,
unasdiff, voice-cloning coqui/sparc, clearvoice) for the files they read back. Nothing in-process
used it. `Audio` kept its own torchcodec → torchaudio branches, and triage, Praat, the video audio
track and six more workers decoded with soundfile, torchaudio or torchcodec directly. The I/O
invariants audit (#571, `specs/20260819-120600-io-invariants/`) measured decode behaviour but did not
reroute reads. The libsndfile fallback in `34d40953` patched `Audio`'s whole-file decode only.

## The failure that made it matter

torchcodec decodes FLAC with FFmpeg's parser. On corpus streams written by senselab itself:

| file (r18 run) | header frames | torchcodec whole-file decode | libsndfile |
|---|---|---|---|
| `sub-8c990ecd…_task-productive-vocabulary-4…/run/streams/plain.flac` | 378,579 | 352,256 frames, no error | 378,579 |
| `sub-a99dde7b…_task-maximum-phonation-time-2…/run/streams/preemphasised.flac` | 272,324 | `RuntimeError` | 272,324 |

Both are intact: libsndfile decodes the full header count, and every other stream in both run
directories decodes identically with either decoder.

## The read policy

`read_audio(path, always_2d=False, channels_first=True, *, start=None, stop=None)`,
`audio_info(path)` and `read_audio_blocks(source, block_seconds)`:

- FLAC, WAV, WAVEX, W64, RF64 and AIFF are decoded by libsndfile. Their header frame count is exact,
  so a decode returning a different count raises. A WAV whose data chunk is longer than the file
  (libsndfile reports `data : N (should be M)`) is read as the frames present with a warning; this
  matches what both decoders did before, and raising would reject recorders that leave a
  placeholder size.
- Every other format is decoded by torchcodec, imported inside the function so a worker venv reading
  WAVs never needs it. An empty or non-finite decode raises. Where torchcodec cannot decode a file
  libsndfile opens (OGG, MP3 on a host without torchcodec), libsndfile reads it.
- `start`/`stop` are frame indices. `Audio` converts its seconds with `round()`, which is what the
  old code did for both decoders; on 1,200 random windows over 16/44.1/48 kHz FLAC and WAV,
  torchcodec's `get_samples_played_in_range` returned the same samples as the libsndfile slice.
- `audio_info(path).decoder` names the decoder used.

No clipping logic on read: the range policy is a write-side guard (`apply_range_policy`).

## Writes

`Audio.save_to_file` no longer carries its own torchcodec/torchaudio encode for containers
libsndfile cannot write. `write_audio` handles them with torchcodec's `AudioEncoder` at `PCM_16`,
under the same range check, and refuses an explicit subtype. The write boundary is one file.

## Guards

- `src/tests/utils/portable_audio_read_guard_test.py`: AST sweep of `src/senselab`, `scripts/` and
  worker-script string literals for `sf/soundfile.read|info|blocks|SoundFile`,
  `torchaudio.load|info`, `librosa.load|stream`, `wavfile.read`, `AudioDecoder`,
  `parselmouth.Sound(<path>)` and their bare-name imports. Allowlist empty. Video frame decoding is
  out of scope. Every subprocess-venv backend is classified as reading through the staged module,
  handing a parent-written WAV to a third-party loader, or taking no audio input.
- `src/tests/utils/audio_write_boundary_test.py`: now one layer file, sweeps `scripts/`, detects
  bare-name writer imports and skips docstrings.

Measured on the commit before the change, the read sweep finds 33 direct decodes; after it, none.

## Numeric verification (ORCD, mit_quicktest, libavutil 56 as in production)

83 files: all streams of both known-bad run directories, every stream and the source WAV of eight
random r18d recordings, and synthetic stereo FLAC/PCM_24, WAV/FLOAT and WAV/PCM_16 built from a real
stream. For each: whole-file `read_audio` against `sf.read`, against torchcodec's whole-file decode,
and three frame windows against both. All 83 are bit-identical to libsndfile, whole and windowed.
81 are also bit-identical to torchcodec's whole-file decode; the two that are not are the two known
files above. Windowed torchcodec reads matched on all 83, the two bad files included: seeking
resynchronises past the frame the linear decode stumbles on.
