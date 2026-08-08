# Next: one axis grid, and a CLI that takes only audio in and results out

Handoff for a fresh session. Two directives, both from 2026-08-05, and they are related: the grids
are the largest thing the CLI currently lets a caller get wrong, and unifying them removes most of
the flags that would otherwise need defaults.

## Directive 1 — every axis on one grid

**Measured, not assumed.** `audio_48khz_mono_16bits_20260805-134600/final/estimates/`, a 4.92 s clip:

| axis | rows | window | hop |
|---|---|---|---|
| `speech_presence` | 242 | 0.1 | **0.02** |
| `background_mask` | 242 | 0.1 | **0.02** |
| `speaker` | 19 | **0.25** | 0.25 |
| `asr` | 8 | **1.0** | 0.5 |

No two axes share a grid. Three consequences, each verifiable from the code:

- **`axes.DEFAULT_TIME_GRID = (0.1, 0.1)` is declared and nothing uses it.** Every axis is built on a
  grid the CLI supplied instead.
- **`asr` declares `grid="word"`** (`axes.py:201`) and runs on 1.0 s / 0.5 s time buckets. Neither
  the word grid it claims nor the shared grid the others are supposed to use.
- **Presence and mask run at an 0.02 s hop inside a 0.1 s window** — 80% overlap, so 242 rows are not
  242 independent measurements. D-24 settled *window equals hop* precisely to stop this, and gave the
  reason: "reporting five near-duplicate rows per window is not the same thing" as fine resolution.

And the coupling this breaks is not hypothetical: `fuse.project_axis_onto`'s own docstring records
that on real audio "the four axes carried 85 / 41 / 1070 / 1 buckets on four different grids and
shared **zero** keys, so coupling did nothing and every round came out byte-identical to the last."
A gridded measure one axis can hand another does not currently exist for *any* axis.

### The work

1. **`harvest_asr_votes` emits one voter and no text.** The axis is a resampling of fused word
   confidence onto the grid — `consensus_words`, already implemented in
   `asr._consensus_word_doubt` + `asr.resample_word_doubt`. Delete the per-bucket text,
   `asr_phoneme_sequence_in_window`, the `__pairwise_phoneme_distances__` block and the
   `avg_logprob` / `token_entropy` / `alignment_ctc_score` per-bucket reads. The wide 1.0/0.5 grid
   exists only because the axis used to be built from bucketed text with `fully_contained=True`;
   with the derivative as the voter, the reason is gone.
2. **All four axes harvest on `DEFAULT_TIME_GRID`**, window equal to hop, so row *i* of one axis is
   row *i* of another and cross-axis coupling needs no projection. `project_axis_onto` should end up
   unused for same-grid axes — check whether it can go.
3. **`compute.py`'s `consensus_votes` block and the LS asr TextArea read `final/transcript.json`**
   instead of per-bucket votes. The per-bucket text always was a reconstruction of what the
   transcript holds at word resolution; `labelstudio._utterance_text_payload` is the consumer.
4. **Remove the grid flags** — `--asr-win-length/-hop-length`, `--speech-presence-grid-*`,
   `--cross-stream-*`. A knob that no longer binds anything is worse than no knob. (Directive 2
   removes them anyway; listed here because step 2 is what makes them dead.)

### What this invalidates — say so in the PR, do not discover it later

Every axis's row count and every downstream number changes. Anything fitted or tuned against the
current grids must be re-measured, not carried over: the scene-quality calibration profile,
convergence thresholds, triage gates, `detection_margin` mask thresholds. `d45aaf1d` recorded the
same caveat for a smaller change and it still applies. Bump `CACHE_SCHEMA_VERSION`.

## Directive 2 — the CLI takes audio in and results out, nothing else

`scripts/analyze_audio.py` currently exposes dozens of flags, and the run recipes in `CLAUDE.md`
differ from each other only in flags a reader has no basis to choose between. Replace with:

```bash
uv run python scripts/analyze_audio.py <audio> [--out <dir>]
```

Everything else moves into **one versioned default config of agreed values** — model ids, grids,
window/hop, aggregator, task type, triage and enhancement gates, ASR set, aligner backend. Design
notes, so the next session does not have to re-derive them:

- **One file, versioned, with a recorded derivation** — the pattern
  `data/detection_margin/<version>.json` already follows, and the reason is the same: a value with no
  derivation is a literal someone will change without measuring. Reuse that shape rather than
  inventing a second one.
- **The config's identity travels into every artifact's provenance**, as the detection-margin profile
  already does. A run whose config cannot be named is a run that cannot be reproduced.
- **Overrides, if any, are a single `--config <path>`** — not per-knob flags creeping back. The
  policy file (`adaptive/policy/default.yaml`) is the precedent and should probably absorb the new
  values rather than sit beside them; check before adding a third config location.
- Keep `--out`. Everything else that survives must justify itself against "a caller has no basis to
  choose".

## Landmines this session hit — do not re-trip them

Each cost a wrong artifact before it was caught, and each was invisible to unit tests:

- **`iter_word_leaves` walks dicts only.** `resolve_asr_result` returns `ScriptLine` *objects* from a
  live backend and dicts from the cache. Passing objects silently yields no words — it produced an
  asr axis with zero contributing signals on a real run. `asr._as_plain` normalises; use it.
- **An onset does not identify a word.** Aligners emit words sharing an onset and words of zero
  duration (measured: `"Josh"` at `[2.72, 2.72]`). Matching lattice slots back to word dicts by
  `(model, onset)` corrupted a transcript — `"wanted to take"` became `"wanted take take"`. Use
  `TranscriptSlot.indices`.
- **A populated, plausible-looking axis can still be wrong.** The asr axis read `0.0000` across a
  whole recording and every test passed; the cause was using existence confidence alone as the doubt
  mass when only the temporal part varied. Check *distributions*, not just presence.
- **Verify against the pipeline, not against your own fixtures.** Three defects in a row survived
  because the tests were built from the same assumption as the code. A cache-cleared run on both
  clips, then reading the numbers, is what caught each one.
- **Look for what already exists before building.** Three components were fully implemented, tested
  and had no production caller: the cross-ASR pairwise phoneme distance, `harmonize_transcripts`
  (star-shaped sequence alignment — exactly what the filler requirement needed), and
  `aggregate.aggregate_{speech_presence,speaker,asr}`. The last three are still uncalled and should
  be deleted with their tests.

## Verification, and what counts as evidence

Not "tests pass". For each clip in `src/tests/data_for_testing/`
(`english_conversation_higgs_audio_v2.wav`, `audio_48khz_mono_16bits.wav` — the second is the harder
one and caught what the first could not):

1. Clear `artifacts/analyze_audio_cache/`, run, exit 0.
2. **All four axes report the same row count and the same `(window, hop)`.** That is the directive,
   stated as a check.
3. Cross-axis coupling moves something: rounds are no longer byte-identical, which is the symptom
   `project_axis_onto` was written for.
4. The asr axis still varies (it read mean 0.3632 over 6 distinct values of 8 rows before this
   change, on the 48 kHz clip) — a unified grid must not flatten it.
5. `final/transcript.json` unchanged in words and confidences: this change is about the axis grid,
   and a transcript diff means something else moved.

## Still open, carried forward

- **Insertion preservation is unexercised on real audio.** Both test clips disagree by substitution
  and one deletion; neither contains a filler, so `single-source words: 0` is correct rather than a
  pass. A disfluent recording would settle it.
- **The per-edge onset/offset confidences describe member spread, while published timings come from
  one aligner** (`consensus_alignment_backend`, now `qwen`, 80 ms grid — accepted). The figure draws
  the marks at the published boundary and colours them by member agreement. Either relabel, or
  measure the spread against the published boundary.
- **`uncertainty` is normalised binary entropy while `confidence` is a weighted mean.** Both correct,
  not comparable, and plotted next to each other. H(0.0444) = 0.2621 is the worked example.
