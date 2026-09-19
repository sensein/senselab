# Diarization turns past the decode

A corpus-scoping run over a 30-recording sample lost its SPEECH branch on 3/26 recordings (CPU),
3/23 (GPU) and 4/30 on a third path — 12–13%. Over the 62,548-recording corpus that projects to
≈8,100 recordings with no SPEECH report at all: not "the instrument did not run" and not "it ran
and found nothing", which is why it was a blocker rather than a defect.

## The composition path

The raise came from `nodes/common.py`'s `clamp_extent`, called at `nodes/speech.py`'s Step 4. It is
not a word timing and not a hull over spans. It is **PREPROCESS's whole-file diarization
derivative, read back**:

```
diarize_audios (pyannote) -> diarized_segments -> write_diarization -> derivatives/<stream>_diarization.npz
                                                                    -> SPEECH `_read_diarization`
                                                                    -> clamp_extent((start, end), plain)  # raised
```

SPEECH runs no diarizer of its own. It reads `enhanced_diarization` (the first stream in
`diarization.streams` that has one), and clamped each turn against the `plain` stream it slices.

## The source was wrong, not the audio

Three measurements, all over the Sept-8 corpus at
`/orcd/scratch/bcs/002/satra/triage_full_20260908/out/` (62,578 stores on disk):

| measurement | result |
| --- | --- |
| consensus words ending past the `plain` stream's decode, by > 1 sample | **0 / 62,578** |
| stream entity extent != the stream file's header duration | **0**, every stream, every store |
| stores whose diarization derivative holds a turn ending > 1 sample past `plain` | **10,958 / 62,578 (17.5%)** |
| — of those, `enhanced_diarization` (the one SPEECH reads) | **6,246 (10.0%)** |
| — `residual_diarization` | 7,167 (11.5%) |

So the ASR side is clean: `preprocess.py`'s `_bound_to_duration` already bounds every recognizer
chunk by the stream's duration and counts what it dropped (`out_of_bounds_chunks_n`). The diarizer's
turns were the one model reading in PREPROCESS that reached a consumer unbounded — `diarized_segments`
dropped a line with no speaker, no start or no end, and passed every other line's times through.

Overshoot distribution, over the 13,413 overshooting derivatives:

```
min 1.5   p50 1000.5   p90 7375.5   p99 15033.5   max 52155.5   samples at 16 kHz
         (62.5 ms)    (461 ms)     (940 ms)      (3.26 s)
```

## Why it is borderline, and why the failing set moves between CPU and GPU

The recurring value in the worst cases is an end of exactly **9.970344 s**, on files of 6.71 s,
7.57 s, 7.64 s, 8.24 s. That is pyannote's own grid: a file shorter than the segmentation model's
10-second window is processed as one chunk **padded to the full window**, and the turns come back
timed on the padded window rather than on the file. 9.970344 s is the last frame of that grid.
For a file longer than one window the same thing happens to the final partial chunk, which is why
the median overshoot is 62 ms rather than seconds.

Nothing about that is deterministic at the file's edge: whether the binarizer's activation stays
above threshold into the padded tail is a numerical question, decided differently by a CPU kernel
and a CUDA kernel on the same weights. That is the whole of the CPU/GPU sensitivity — the overshoot
is a property of the grid, and only *whether a given file's last turn runs into it* moves.

The per-family rates match the families the scoping run named:

```
diadochokinesis-v2-puh        60.8%  427/702
diadochokinesis-v2-kuh        50.7%  356/702
open-response-questions       46.2%   92/199
diadochokinesis-pa            25.0%  224/896
random-item-generation-v2     27.5%   57/207
```

Short, single-token DDK prompts rank highest because they are the files most often shorter than one
window.

## The fix, in two parts

**1. PREPROCESS bounds the reading where it is taken.** `diarized_segments(lines, duration_s)`
returns a `DiarizedTurns` carrying the turns bounded by the audio the diarizer read, plus
`bounded_n`, `past_end_n` and `max_overshoot_s`, which `write_diarization` records on the
measurement. This is the same admission rule `_bound_to_duration` applies to a recognizer's chunks:
an instrument cannot report a region of a recording it was not given. It also fixes
`speaker_activity`, which was totalling speech time the recording does not hold.

**2. SPEECH reports an over-long foreign reading instead of raising on it — a branch-contract
change, stated as one.** `clamp_extent`'s one-sample tolerance is unchanged and still guards every
extent SPEECH composes itself (the word runs, the speaker runs, the cross-clamp against
`recording`). A diarization turn is not one of those: it is another node's instrument reading, taken
on a *different stream*, on a grid that is not SPEECH's. `nodes/common.py`'s new `bound_reading`
bounds it and returns None when it names no part of the audio; SPEECH counts both and puts
`segments_bounded_n`, `segments_past_end_n` and `max_overshoot_s` in its report's `diarization`
detail with a note beside them. A branch reports; VERDICT decides.

Part 2 is what clears the 10,958 stores already on disk without re-diarizing them. Part 1 is what
stops new runs writing the reading unbounded in the first place.

The tolerance was **not** widened. `test_an_extent_this_branch_composed_itself_still_raises` pins
that: a word run 100 ms past the decode still raises, with the fix in place.

## What else composes an extent the same way

An audit of the other branches found none of them exposed to *this* failure, because none of them
decodes audio: `grep "waveform\["` over `airway.py voice.py quality.py ddk.py taxonomy.py` is empty,
and only `speech.py` imports `clamp_extent`. They are exposed to the adjacent, silent version — an
extent past the end of the file written into the graph, or an array slice that quietly truncates:

- `voice.py`'s `voiced_extent` ends a span at `times_s[last] + track.hop_s`, up to one hop (~10 ms)
  past the file.
- `ddk.py`'s `Decode.extent` ends at `(repetitions[-1][1] + 1) * seconds_per_frame`, up to one PPG
  frame (~20 ms) past it.
- `airway.py`'s HeAR run hulls are taken over the model's own 2-second window grid, whose last
  window is padded exactly as pyannote's is.
- `preprocess.py`'s span-YAMNet crop slices `plain.waveform` with no clamp at all, so a span past
  the end yields a zero-length `Audio` recorded as `unmeasured`.
- `propose_span` refuses a span naming no evidence and refuses non-positive duration. It takes no
  audio and bounds nothing.

None of those is this blocker and none is fixed here. They are listed so the next person does not
have to find them again.

## Measurement

Before/after rates are in [`measurement.md`](measurement.md).
