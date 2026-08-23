# What the 33 `nontarget_active` buckets contain

F-187 recorded a reviewer's open question and declined to wire the mask regions through until it
was answered: **does `nontarget_active` belong in `speaker._VOCAL_ACTIVITY` at all?** 33 of the 34
buckets that wiring would spare from the wordless-bucket clear are `nontarget_active`, and the
register states plainly that "nobody has measured what those 33 buckets contain".

Measured now, from the completed runs' own artifacts. **Answer: no.**

## Method

Run `streaming-audio-2026-08-07T17-22-57-815Z_20260807-191739`. Its `L2/background_mask.parquet`
holds exactly three `nontarget_active` regions — 0.0-2.0 s, 8.7-9.4 s, 10.6-11.2 s — which on the
shipped 0.1 s grid is 20 + 7 + 6 = **33 buckets**, matching the register's count.

Per-bucket rather than per-region, because YAMNet's window is 0.96 s on a 0.48 s hop and a window
straddling a region edge would otherwise import content from outside it. Each bucket takes the
per-label maximum over the windows covering its midpoint. Vocal mass is the sum over AudioSet's
vocal labels (`Speech`, `Cough`, `Breathing`, `Sneeze`, `Throat clearing`, `Sigh`, `Laughter`,
`Babbling`, `Baby cry, infant cry`, the three speaker-demographic speech labels, and 12 more).

## Result

| region | buckets | dominant label | median vocal mass | median `Music` |
|---|---|---|---|---|
| 0.0-2.0 | 20 | **Music** ×19, Speech ×1 | 0.0049 | 0.860 |
| 8.7-9.4 | 7 | **Music** ×7 | 0.1492 | 0.904 |
| 10.6-11.2 | 6 | **Silence** ×6 | 0.6714 | 0.115 |

**32 of 33 buckets carry a non-vocal dominant label** — 26 Music, 6 Silence. **16 of 33 have vocal
mass below 0.01.** The single Speech-dominant bucket sits at 1.9 s, where the region's edge lands on
a speech onset.

One honest caveat on the last row: its median vocal mass of 0.67 and its `Silence` dominance both
come from two overlapping windows that disagree — 10.08-11.04 reads `Speech` 0.659 while
10.56-11.26 reads `Silence` 0.674. The max-merge keeps both. What is not in doubt is that a
`Silence` score of 0.674 is present inside a region the mask calls `nontarget_active`.

## Why this settles the question

`_VOCAL_ACTIVITY` exempts a wordless bucket from the speaker-vote clear on the reading that *a
vocalization was measured here, it simply had no lexical content*. That reading is false for this
state: the bulk case is music at 0.86-0.90 with vocal mass at the third decimal place, and the
tail case is scored silence. Admitting `nontarget_active` would spare 26 music buckets and 6
silent ones from a clear that exists precisely to remove speaker claims where there is no speaker.

This confirms the reviewer's concern by measurement, and confirms the mechanism they proposed:
`_classify_bucket` reaches its `nontarget_confidence >= 0.5` test only after the bucket has already
passed `confidence <= free_at`, so `nontarget_active` means *the target is confidently absent and
some non-speech source scored >= 0.5* — and `nontarget_confidence_by_bucket` maxes over every
non-speech category, of which `environment` holds 292 of the 527 labels in
`data/audioset_source_map.json` and is also the map's default. Music-dominated audio satisfies it
without any voice present, which is exactly what these three regions are.

## The discriminator already exists

`background_mask.parquet` carries `contains_nontarget_speech`, and it is **`False` on all three
regions**. So the table already distinguishes the case `_VOCAL_ACTIVITY` wants — a
`nontarget_active` region holding non-target *speech*, which is genuinely vocal — from the case it
must not admit. Nothing new needs measuring to tell them apart.

Recommendation, for the wiring decision F-187 still defers: `_VOCAL_ACTIVITY` should admit
`target_active` unconditionally, and `nontarget_active` only where `contains_nontarget_speech` is
true. Under that rule this run spares 1 bucket of the 33 rather than 33, and the wordless clear
keeps doing its job on music and silence.

## What this does not establish

These 33 buckets are from one recording, and **not the one with verified labels**. The hand-verified
ground truth in `specs/20260817-triage-workflow-dag/ground-truth-2026-08-18.md` is
`streaming-audio-2026-07-30T04-21-56-487Z.wav` (14.027 s), while every completed run under
`artifacts/analyze_audio/` is on `2026-08-07T17-22-57-815Z.wav` (11.264 s) or the synthetic
conversation. Different files, different durations, different hashes.

So the labels cannot check any register measurement, and the region boundaries here appearing to sit
in the labelled recording's event gaps is coincidence — it was checked and it does not hold. Running
the analysis over the labelled recording is what would make the ground truth usable, and is worth
doing before any further mask-state reasoning is trusted on a single run.
