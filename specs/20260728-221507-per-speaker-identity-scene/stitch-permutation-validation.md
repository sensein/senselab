# Do segmentation-3.0's speaker channels actually flip between stitched chunks?

The fix in `a949b541` was justified by a deterministic demonstration that `stitch_frames` had no
defence against a permutation flip. That establishes the code was wrong; it says nothing about how
often flips occur on real model output, which is what decides whether earlier runs' overlap and
speaker-count figures were inflated. This measures it.

## Method

Four distinct voices from Bark (`suno/bark-small`, presets `v2/en_speaker_{0,3,6,9}`), trimmed to
≤ 3 s per turn and tiled with a rotating order so no single speaker owns every seam. Two conditions,
both 72 s, both giving 9 inference chunks and 8 seams at the 8 s chunk step:

- **sequential** — one speaker at a time, 24 turns. True simultaneous count is 1 during speech and
  0 in the gaps, so any reported overlap is an artefact.
- **overlapping** — two speakers straddling every seam, 17.3% of frames with two active by
  construction, plus solo turns between seams.

Both stitched twice from *identical* model output — the two paths differ only in the stitch — and
compared frame by frame. `pyannote/segmentation-3.0`, CPU (the frame-posterior path excludes MPS
upstream in `_select_device_and_dtype`; device changes throughput, not activations). Output is
`(frames, 3)`, i.e. per-speaker columns, consistent with the D-5 finding.

Scripts and artifacts: `artifacts/stitch_validation/` (`sequence.wav`, `overlap.wav`,
`stitch_report_{sequence,overlap}.json`, per-condition logs).

## Result

| condition | informative seams | flips | max \|naive − aligned\| | seam `p_overlap` naive → aligned |
|---|---|---|---|---|
| sequential | 4 | **0** | 0.0000 | 0.0021 → 0.0021 |
| overlapping | 4 | **1** | **1.0000** | 0.0096 → **0.0000** |

"Informative" means the seam carried a confident speaker: 4 of 8 seams in each condition fell in
silence and cannot flip.

**Flips are real and their effect is full-scale.** One of four informative seams in the overlapping
condition permuted (`[0, 2, 1]`), and where it happened the two stitches differ by **1.0** on a
`[0, 1]` activation — one speaker at full strength in one channel versus split across two at half.
Not a marginal numerical difference.

**The naive stitch fabricates overlap at seams, in the predicted direction.** Seam `p_overlap` falls
from 0.0096 to exactly 0.0000 once the channels are matched, while non-seam `p_overlap` is unchanged
at 0.0224 — the correction acts where the defect was and nowhere else.

**Sequential speech is not evidence of absence.** Zero flips there is what the mechanism predicts,
not a contradiction: channel assignment is ambiguous when several speakers are active *within one
chunk*, and with one speaker at a time the model has nothing to permute. The first experiment was
run before this was thought through, and on its own it would have wrongly retired the concern.

## What this does not establish

- **n is very small.** Four informative seams per condition. "1 in 4" is a point estimate with an
  interval wide enough to include both "rare" and "most seams"; it establishes that flips happen,
  not their rate.
- **The absolute overlap numbers are not representative.** The model detected almost no overlap
  anywhere on this material (non-seam `p_overlap` 0.0224 against a designed 17.3% of frames with
  two speakers), because digitally summing two TTS voices at equal level is not what overlapping
  speech sounds like. The flip finding does not depend on those magnitudes; a rate estimate would.
- **Real conversational audio was not tested.** That is the measurement that would give a usable
  rate, and it needs material with known overlap ground truth.

## Change this prompted

Half of all seams fell in silence. There the assignment cost is near-uniform and the "best"
permutation is arbitrary — and since it is applied to the *whole* chunk, acting on it would scramble
frames that were fine. Exact zeros happen to make the assignment degenerate to identity, so the
hazard only appears with faint noise. `_match_columns` now declines below `MIN_SEAM_ACTIVATION`
(0.5). Re-running both conditions with the guard reproduced the table above exactly, confirming it
closes the hazard without touching the correction.
