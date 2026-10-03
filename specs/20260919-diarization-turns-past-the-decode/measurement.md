# Measurement

Every number here comes from the Sept-8 corpus at
`/orcd/scratch/bcs/002/satra/triage_full_20260908/out/` (62,578 run directories, 62,521 with a
complete set of decodable streams). The corpus driver ran PREPROCESS and TAXONOMY and **no branch**,
so a SPEECH replay over those stores is the first time the branch sees them — the same thing a
corpus SPEECH pass would do, and with no hint, exactly as the driver runs PREPROCESS.

## Scan 1 — is it the words?

`scan_overshoot.py`, over every store: for each `word` entity, `(extent[1] - plain_duration) * rate`.

```
62578 stores
stores with an overshooting word: 0 / 62578
```

Not the words. `preprocess.py`'s `_bound_to_duration` already bounds every recognizer chunk.

## Scan 2 — the streams and the diarization derivative

`scan2.py`, over every store: each stream's FLAC header frame count against its stream entity's
extent, and each `derivatives/*_diarization.npz`'s largest `ends` against the `plain` stream.

```
stores scanned: 62578
stores with no diarization npz: 691
entity extent != header duration, per stream: {}          <- every stream entity is exact
stores where streams differ in length: 748                <- enhanced/residual vs plain, never recording vs plain
diarization max_end > plain duration + 1 sample:
    enhanced_diarization  6246   (10.0%)   <- the one SPEECH reads
    residual_diarization  7167   (11.5%)
stores with at least one overshooting derivative: 10958 / 62578  (17.5%)

overshoot, samples at 16 kHz:
    min 1.5   p50 1000.5   p90 7375.5   p99 15033.5   max 52155.5
             (62.5 ms)    (461 ms)     (940 ms)      (3.26 s)
```

The worst cases all end at exactly **9.970344 s**, on files of 6.71, 7.57, 7.64 and 8.24 s — the
last frame of the segmentation model's 10-second window. A file shorter than one window is decoded
as a single chunk padded to the full window, and the turns come back on the padded grid.

Per-family rates (families with ≥ 50 stores, top of the list):

```
diadochokinesis-v2-puh         60.83%   427/702
diadochokinesis-v2-tuh         53.85%   378/702
diadochokinesis-v2-puhtuhkuh   52.07%   365/701
diadochokinesis-v2-kuh         50.71%   356/702
diadochokinesis-v2-buttercup   48.43%   340/702
open-response-questions        46.23%    92/199
free-speech-2                  36.75%   330/898
diadochokinesis-pataka         31.44%   282/897
caterpillar-passage            30.94%   185/598
random-item-generation-v2      27.54%    57/207
diadochokinesis-pa             25.00%   224/896
diadochokinesis-ka             23.94%   215/898
```

`diadochokinesis-v2-kuh`, `diadochokinesis-pa`, `open-response-questions` and
`random-item-generation` are four of the five families the scoping run named.

## Reproduction — six real recordings, both source trees

`replay_speech.py` calls `speech(store, "plain", config, run_dir=..., enrollment=None)` over a run
directory. Six recordings: the three worst-overshooting `enhanced_diarization` derivatives and three
with none. `PYTHONPATH` selects the source tree; the log line prints which was imported.

| | base (`e7dc979f`) | fixed |
| --- | --- | --- |
| overshooting store 1 (`diadochokinesis-ka`) | `ValueError: extent ends at 6.74721875s, past the 6.710625s this audio decoded to by 585.500 samples` | ok, `segments_bounded_n=1`, `max_overshoot_s=0.0366` |
| overshooting store 2 (`glides-low-to-high`) | `ValueError: extent ends at 8.08034375s, past the 7.56975s this audio decoded to by 8169.500 samples` | ok, `segments_bounded_n=1`, `max_overshoot_s=0.5106` |
| overshooting store 3 (`maximum-phonation-time-2`) | ok | ok (its SPEECH took the no-lexical path before Step 4) |
| three control stores | ok | ok, `segments_bounded_n=0` |

## The larger sample

A uniform random sample of **800** run directories drawn from the 62,521 complete stores
(seed 20260919), spanning **219 task families**, replayed through both source trees on
`mit_preemptable` as an 8-way array (`replay.sbatch`).

```
base rows 800, fixed rows 800, compared 800

BASE:  {'ok': 732, 'errored': 68}
  SPEECH lost to a raise: 68/800 = 8.50%
  error types: {'ValueError': 68}   — all 68 the extent clamp

FIXED: {'ok': 800}
  SPEECH lost to a raise: 0/800 = 0.00%

errored -> ok: 68        ok -> errored: 0
```

Families recovered, most first: `diadochokinesis-v2-tuh` 8, `diadochokinesis-v2-buttercup` 5,
`free-speech-v2-3` 4, `diadochokinesis-pataka` 4, `diadochokinesis-v2-puh` 3, `free-speech-v2-2` 3,
`free-speech-2` 3, `diadochokinesis-v2-kuh` 2, `prolonged-vowel` 2, `story-recall` 2,
`free-speech-v2-1` 2, `caterpillar-passage` 2, `picture-description` 2, `free-speech-1` 2,
`diadochokinesis-v2-puhtuhkuh` 2, and a tail of singletons.

What the fixed pass reports instead of raising:

```
runs that bounded at least one turn:            64/800
  max_overshoot_s  min 0.0007  p50 0.0612  p90 0.2457  max 0.9430
runs that dropped a turn wholly past the end:    6/800
```

**8.50% → 0.00% on 800 recordings, no run that completed before now fails.** The 68 are a lower
bound on what a corpus pass would lose, because this replay carries no hint: a hint routing more
recordings into SPEECH's in-family mode would only add. The scoping run's 12–13% was measured on
30 recordings, three samples of 26, 23 and 30; 8.50% of 800 sits inside that interval's range.

## The other exposure the scan found

748 stores have an `enhanced`/`residual` stream whose decoded length differs from `plain`'s. In
those, a diarization turn taken on `enhanced` can legitimately reach past `plain` with no padding
involved at all. `recording` and `plain` never disagree in this corpus, so `speech.py`'s cross-clamp
of a plain-derived extent against `recording` is not implicated here — but it would be on a corpus
whose source files are not already 16 kHz.
