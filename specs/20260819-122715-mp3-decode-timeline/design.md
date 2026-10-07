# A windowed read of an MP3 returned content it did not contain

`Audio(filepath=..., offset_in_sec=..., duration_in_sec=...)` served, for MP3 only, samples from
about 50 ms before the requested offset, while reporting the requested offset as the timestamp of
what it returned. The misalignment certified itself as correct, so nothing downstream could notice.

## Measurement

A 6 s file, 22050 Hz mono: 3 s of uniform noise, then the same noise 60 dB down. Written to
`.wav`, `.flac` and `.mp3` from one signal. The comparison that matters is against **another
decoder reading the same MP3**, not against the lossless source — an amplitude step in an MP3
carries codec ringing either side of it, and comparing to the source conflates that ringing with
the shift under investigation.

Requesting `[3.000, 3.200)` s, the window that begins exactly at the step:

| reader | peak returned |
| --- | --- |
| `torchcodec` `get_samples_played_in_range` | 1.1639 |
| `torchcodec` `get_all_samples()` sliced `[66150:70560]` | 0.1172 |
| `torchaudio.load(frame_offset=66150, num_frames=4410)` | 0.1172 |
| `soundfile.read(start=66150, frames=4410)` | 0.1172 |
| `ffmpeg -i … -ss 3.0 -t 0.2` | 0.1172 |

Ratio against the three decoders that agree: **9.93×** (`raw/invariants_run3.txt` lines 20-23 of the
I/O audit). The returned `pts_seconds` was `3.000000` — the value asked for.

Three further measurements fix the mechanism:

- The shift is a constant **1105 samples earlier**, at every seek position swept across an MP3
  frame (offsets +0, +1, +288, +576, +1151, +1152, +1153, +2304), and the returned block is
  **bit-identical** to `get_all_samples()[…, start-1105 : …]`. Not a rounding or frame-boundary
  effect: a whole-timeline offset.
- Tiling consecutive ranges over the file yields **131195 samples against 132300** for the full
  decode — 1105 short, the same number lost once at the end.
- `get_all_samples()` on the MP3 reports `pts_seconds = 0.050113` = **1105 samples**, and
  `metadata.begin_stream_seconds` reports the same value without decoding anything. For `.wav`,
  `.flac`, `.m4a` and `.opus` both are exactly `0.000000`, and their range reads are bit-exact and
  tile without loss.

1105 = 576 + 529, the libmp3lame encoder delay.

## Mechanism

An MP3 stream begins at a non-zero presentation timestamp because the encoder prepends its delay.
Decoders trim that delay, so the first *decoded sample* is source time 0 while the first
*timestamp* is 0.050113 s. `get_samples_played_in_range` seeks on the timestamp timeline;
`get_all_samples()` returns data on the trimmed-sample timeline and labels it with the untrimmed
origin. Asking for 3.0 s therefore lands at sample 3.0 × sr on a timeline whose zero sits 1105
samples later than the data's zero — 1105 samples early — and the returned `pts_seconds` is echoed
back from the request rather than derived from what was decoded, so the error is invisible at the
call site. The lost tail is the same disagreement at the other end: the last range stops 1105
samples before the data ends.

Upstream is `meta-pytorch/torchcodec`; present in 0.11.1 and still in 0.16.0. Chunked decoding is
an active workstream there (#1614 merged, #1601 reopened in practice), but this specific timeline
disagreement appears unreported. Nothing was filed.

## Fix

`Audio._lazy_load_data_from_filepath` asks the decoder whether the two origins coincide
(`metadata.begin_stream_seconds == 0`) and only then uses the range API. Otherwise it decodes the
stream and slices by sample index — `_window_of_all_samples`.

The gate is the file's own reported timeline, not a codec allowlist. A stream that starts at a
non-zero timestamp is exactly the condition under which the two origins can differ, and it is
readable from metadata before any decoding. A decoder that does not report the field reads as
"cannot confirm" and takes the slicing path.

### Cost accepted

A windowed read of a file with a non-zero start timestamp now decodes the whole stream. Measured on
a 10-minute 22050 Hz mono MP3 (single run, loaded laptop, so upper bounds):

| operation | time |
| --- | --- |
| range read of a 0.2 s window | 52 ms |
| full decode, then slice | 146 ms |
| `soundfile.read(start=…, frames=…)` | 11 ms |

2.8× on time, and the full decode's buffer is transient — `convert_to_tensor` clones the slice, so
only the window is retained. Peak memory is the real cost: 4 bytes per sample per channel for the
duration of the call, ~2.8 GB for a 2-hour stereo 48 kHz MP3. Windowed reads of `.wav`, `.flac`,
`.m4a` and `.opus` are untouched and still seek. The full-file read path is untouched for every
format.

## Rejected alternatives

**Compensate: add `begin_stream_seconds` to the requested range.** Bit-exact at all seven swept
seek positions, recovers the lost tail, and keeps the 52 ms seek. Rejected because it depends on an
upstream convention that no call can verify, and inverts the defect exactly: the day torchcodec
aligns the two timelines, every compensated window goes 1105 samples wrong in the other direction,
silently. Slicing a decode we already trust cannot fail that way.

**Seek with `soundfile` (11 ms) or `torchaudio` (bit-exact) for the affected formats.** Rejected
for consistency: libsndfile's MP3 decode differs from torchcodec's by up to 2.4e-06, so a window
would no longer be exactly the corresponding slice of `Audio(filepath=...).waveform`. That
invariant is asserted with `torch.equal` in
`test_consecutive_windows_concatenate_to_the_full_decode`, and it is worth more than 135 ms.
libsndfile also cannot open `.m4a`.

**Slice a full decode for every format.** Rejected: the range API is measured bit-exact and
loss-free for `.wav`, `.flac`, `.m4a` and `.opus`, so this would pay the whole-file decode on every
windowed read of every format to fix one.

**Honour `pts_seconds` on the full-decode path.** Rejected, and it would introduce an error rather
than remove one — see below.

## `pts_seconds` on the full-decode path corresponds to nothing in the data

`Audio` discards `pts_seconds`, which for an MP3 full decode is 0.050113 s. Whether honouring it
would fix or create an alignment error is settled by encoding a burst at a known source time:
silence for exactly 1.000 s (22050 samples), then noise, at 22050 Hz.

| decode | first sample above 0.05 |
| --- | --- |
| lossless `.wav` | 22050 |
| `.mp3`, `torchcodec` `get_all_samples()` | 21839 |
| `.mp3`, `torchaudio.load()` | 21839 |
| `.mp3`, `soundfile.read()` | 21839 |

The MP3 full decode is 66150 samples for a 66150-sample source. Index 0 is source time 0: the onset
lands 211 samples early in all three decoders alike, which is the encoder's analysis window
spreading energy backwards across the onset, not a timeline offset. Honouring `pts_seconds` would
place index 0 at 0.050113 s and shift every absolute time derived from the waveform — ASR
timestamps, diarization turns, alignment spans — 1105 samples late. The full-decode path is
therefore correct as it stands, and `pts_seconds` is correctly ignored there.

## Blast radius

`get_samples_played_in_range` has exactly one call site in the tree, the one fixed here. Nothing in
`src/senselab` constructs an `Audio` with `offset_in_sec`; the only in-tree users are
`src/tests/audio/data_structures/audio_test.py` and callers outside the repository. Every other
windowed read in the tree goes through paths measured clean: `Audio.from_stream` uses sequential
`soundfile` blocks (bit-exact against the full decode on every format tested),
`Audio.window_generator` slices an already-decoded waveform, `Video` decodes its audio with
`get_all_samples()`, and every other `sf.read` call in `src/senselab` reads a whole file.

## Tests

`src/tests/audio/data_structures/audio_test.py`, four assertions parametrised over `.wav`, `.flac`
and `.mp3` — fixture written from one signal through `Audio.save_to_file`:

- a window holds what `soundfile` reads over the same sample range;
- a window lying wholly inside the quiet half peaks below 5% of the loud half (pre-fix: 64%);
- an offset-only read returns exactly the tail (pre-fix: 67255 samples for a 66150-sample tail);
- windows tiling the file concatenate back to the full decode, bit for bit (pre-fix: 131195 against
  132300).

Pre-fix, all four fail for `.mp3` and pass for `.wav` and `.flac`. The window begins 512 samples
inside the quiet half, which is far enough for the codec ringing at the step to have decayed to
0.0016 and near enough that a 1105-sample shift reaches back into content peaking at 1.81.

Raw measurements: `~/Downloads/senselab-io-audit/raw/invariants_run{1,2,3}.txt`, scripts in that
audit's `scripts/`.
