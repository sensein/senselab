
### The rescale fix, measured on this repo's code — 2026-08-18

`x / x.abs().max() * norm`, guarded for a denormal peak, restored at the ISTFT return.

| variant | | output peak | clipped | corr(in, out) |
| --- | --- | --- | --- | --- |
| plain | before | 1.000000 | 98.5% | **0.2041** |
| plain | after | 0.958740 | 0% | **0.9439** |
| † | before | 0.995667 | 0% | 0.9756 |
| † | after | 0.958740 | 0% | 0.9756 |

Raw ISTFT peak on this host for a peak-1 input: **53 160** (plain) against **1.030** (†). Earlier
figures quoted from a probe outside this repository (0.6284 → 0.9950, 91.8% clipped) did not reproduce
through `enhance_audios`; same mechanism and direction, different numbers, and only these came from the
shipped code path.

**Per-window normalisation was kept, against expectation, because the model is not level-equivariant.**
The same content at input scales 1.0 / 0.5 / 1/6 / 0.05, fed without renormalising, gives output RMS
927.9 / 842.8 / 638.2 / 349.0 — roughly a square-root law. A per-file peak match therefore *preserves*
that compression between windows. On a 56.1 s file with a 15.6 dB level jump, per-file gives corr 0.9160
with boundary steps of +1.28 / −2.59 / −3.43 dB; per-window gives 0.9435 with −0.36 / −0.74 / −1.20 dB,
at or below the frame-wise difference between segmentations. Per window is also exactly upstream's
procedure for any file within one chunk.

The trailing-remainder loss was smaller than described: the previous window always reached the end of
the file, so the skipped remainder was redundant, and what was actually lost was ~10 samples at each
edge where the overlap-add weight fell under its `1e-8` clamp. Fixed by fixed-length anchored windows
plus flattening the outer half of the first and last Hann tapers.

**Source note.** A review of this fix reported that our documented figures 15.8 and 3.50 / 20.2 are
absent from upstream's README. They are — they come from the maintainer's own issue comments (issue #2
gives `ema` PESQ 3.00 / SI-SDR 15.8; issue #1 gives † PESQ 3.50 / SI-SDR 20.2), which is a legitimate
source. The reviewer could not access the paper, so the earlier citations to §3.1, §4.1 and Table 1
remain single-sourced and should be treated accordingly.
