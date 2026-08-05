# Results: one axis grid, and a two-argument CLI

What `next-grid-unification-and-cli-config.md` asked for, what it measured out at, and the one thing
it exposed that nobody had seen before because it could not run.

Measured on `audio_48khz_mono_16bits.wav` (4.92 s, 5 named speakers) and
`english_conversation_higgs_audio_v2.wav`, both with `artifacts/analyze_audio_cache/` cleared first.

Reproduce, or re-check any later run:

```bash
uv run python scripts/analyze_audio.py src/tests/data_for_testing/audio_48khz_mono_16bits.wav
uv run python scripts/analyze_audio.py src/tests/data_for_testing/english_conversation_higgs_audio_v2.wav
uv run python scripts/verify_grid_unification.py artifacts/analyze_audio/*
```

The runs behind the figures below:
`artifacts/analyze_audio/audio_48khz_mono_16bits_20260805-182654/` and
`artifacts/analyze_audio/english_conversation_higgs_audio_v2_20260805-183211/`. Both exit 0 and the
checks exit 0.

**Words and confidences reproduce exactly.** The 48 kHz transcript digest is `ad7dfa13a6971e1a` on two
independent cache-cleared runs into different output directories — so the fold is deterministic in
the deliverable, not merely in its decision log.

## Directive 1 — every axis on one grid

Measured on both clips' last round, before and after. "Before" was read back off the pre-change run
directories the handoff named, not copied from it:

**`audio_48khz_mono_16bits.wav`** (4.92 s)

| axis | before: rows @ win/hop | mean `u` | distinct | after: rows @ win/hop | mean `u` | distinct |
|---|---|---|---|---|---|---|
| `speech_presence` | 242 @ 0.1/**0.02** | 0.5328 | 174 | 49 @ 0.1/0.1 | 0.4759 | 49 |
| `background_mask` | 242 @ 0.1/**0.02** | 0.0512 | 76 | 49 @ 0.1/0.1 | 0.1004 | 49 |
| `speaker` | 19 @ **0.25**/0.25 | 0.8174 | 18 | 49 @ 0.1/0.1 | 0.7860 | 49 |
| `asr` | 8 @ **1.0**/0.5 | 0.3632 | 6 | 49 @ 0.1/0.1 | 0.2146 | 7 |

**`english_conversation_higgs_audio_v2.wav`**

| axis | before: rows @ win/hop | mean `u` | distinct | after: rows @ win/hop | distinct |
|---|---|---|---|---|---|
| `speech_presence` | 1070 @ 0.1/**0.02** | 0.3394 | 744 | 214 @ 0.1/0.1 | — |
| `background_mask` | 1070 @ 0.1/**0.02** | 0.0949 | 415 | 214 @ 0.1/0.1 | — |
| `speaker` | 85 @ **0.25**/0.25 | 0.7917 | 84 | 214 @ 0.1/0.1 | — |
| `asr` | 41 @ **1.0**/0.5 | **0.0000** | **1** | 214 @ 0.1/0.1 | **1** |

Four grids became one. The row count *falls* on the fine axes — 242 → 49, 1070 → 214 — because window
now equals hop: the old rows were a 0.1 s window at a 0.02 s hop, sharing 80% of their audio, so they
were never that many independent measurements.

Same count, same `(window, hop)`, and the same `(start, end)` keys — asserted as a set comparison,
not inferred from the counts matching. `axes.DEFAULT_TIME_GRID` is now what `BucketGrid()` defaults
to, so the declared constant and the used default cannot disagree; it was `(0.5, 0.5)` against a
declared `(0.1, 0.1)` that nothing read.

### The asr axis reflects its evidence on both clips

The check the handoff asked for (item 4), on the 48 kHz clip, last round:

```
uncertainty    n=45/49  mean=0.2146  min=0.0  max=0.9183  distinct=7
triage_score   n=45/49  mean=0.0776  min=0.0  max=0.6667  distinct=10
unmeasured buckets (no word reached): 4/49
contributing signals: ['consensus_words']
```

Seven distinct values over 45 measured buckets. The 4 nulls are buckets no word reaches — reported
as unmeasured rather than as `0.0`, since "nothing was said here" and "nothing is in doubt here" are
different claims.

The pre-change figure was mean 0.3632 over 6 distinct values of 8 rows. The mean is not comparable
across the two: the old 8 rows were 1.0 s buckets that each absorbed several words, and buckets with
no words were folded into their neighbours rather than reported as null. What is comparable is that
the axis still varies, over more distinct values than before, from a single named voter.

### On the conversation clip the asr axis is 0 everywhere, and that is the right answer

An axis reads zero when the algorithms agree, and cleanly-extracted audio should produce exactly
that. Measured: all three recognizers emit 62 words and agree on every surface form, so the fused
words carry `existence_confidence == 1.0` throughout and there is nothing in doubt about *what was
said*. The pre-change run reports the same thing — mean 0.0000 over a single distinct value across
its 41 buckets — so this is a property of the recording, not of the grid, and no bucket width can
change it (a coverage-weighted mean of zeros is zero at any width).

What *would* be a defect is a zero axis whose inputs disagree, i.e. the fold not reaching the axis —
which is the failure that once produced an asr axis with zero contributing signals. So the check
distinguishes the two by measuring the input rather than the axis, and only the second case fails.
`scripts/verify_grid_unification.py` does this by re-folding the run's own cached ASR outcomes.

The timing evidence on this clip is real and large (`onset_confidence` down to 0.25,
`offset_confidence` to 0.325) and deliberately lives on the word rather than in the axis, per D-27's
revised split. That trade-off is item 2 of "Still open" below.

### `final/transcript.json` unchanged in words and confidences

19 words, `"This is Peter This is Johnny Kenny and Josh We just wanted to take a minute to thank
you"`, confidence mean 0.4704 / min 0.0 / max 0.9118. `"wanted to take"` is intact — the
`(model, onset)` corruption that produced `"wanted take take"` stays fixed, and no word is duplicated
or dropped. This change is about the axis grid; a transcript diff would have meant something else
moved.

## What the unified grid exposed: cross-axis coupling actually runs now

This is the finding that was not in the handoff, and it is the first thing to decide next.

`fuse.project_axis_onto`'s docstring records that on real audio "the four axes carried 85 / 41 /
1070 / 1 buckets on four different grids and shared **zero** keys, so coupling did nothing and every
round came out byte-identical to the last." With one grid, it runs. Measured across the five rounds
of the 48 kHz run, reading `contributing_signals`:

| round | asr signals | asr `uncertainty` | nulls |
|---|---|---|---|
| 0 | `consensus_words` | `[0.000, 0.918]` | 4 |
| 1 | `consensus_words`, `axis::speaker`, `axis::speech_presence`, `axis::background_mask` | `[0.473, 1.000]` | 0 |
| 2 | same four | `[0.432, 1.000]` | 0 |
| 3 | `consensus_words` | `[0.000, 0.918]` | 4 |
| 4 | `consensus_words` | `[0.000, 0.918]` | 4 |

Every axis behaves the same way: `background_mask` goes from `[0.000026, 0.650]` at round 0 to
`[0.712, 1.000]` at round 2, `speech_presence` from `[0.388, 0.671]` to `[0.581, 0.891]`. So the
handoff's check 3 passes — rounds are no longer byte-identical, and it is specifically the coupling
that moves them.

**But what it does when it runs is saturate, not refine.** Two properties combine badly:

1. A cross-axis input carries **full weight** by design — `cross_axis_inputs` argues, correctly, that
   a factor never measured must not act as a discount, so it deleted the fixed 0.4 multiplier an
   earlier draft had.
2. The default aggregator is `min` over confidences, i.e. **max doubt wins**. Each axis now receives
   three more voters, each carrying another axis's most-doubtful reading.

The result is monotone inflation, and it hit every axis at once. The loop measured no improvement and
terminated `no_improvement`; `final/estimates/` is extracted from round 4, which carries no `axis::`
inputs and the honest 4 nulls, so **the deliverable is not contaminated**. But the mechanism is now
live and its observable effect is to make everything look more doubtful for as long as it is applied.

A second, sharper symptom: the asr axis's 4 unmeasured buckets are **filled in** during rounds 1–2.
Those are buckets no word reaches. Their round-1 values are sourced entirely from other axes — an
echo that is indistinguishable from a measurement in the output, which is the exact pathology
`cross_axis_inputs`' own docstring warns about for a different reason ("coupling informs an axis's
grid; it never extends it"). That guard was written against an axis holding *one* datum; it does not
cover an axis that legitimately has a row per bucket but has measured nothing in some of them.

Deliberately **not** fixed here, because the fix is a measurement rather than a number:
`measure_axis_overlap` already exists for exactly this question — how much of a contributing axis's
evidence the receiving axis already holds — and returns `None` (no discount) when the source
contributed no evidence of its own. Wiring it in, or deciding that a bucket an axis did not measure
must stay null through coupling, are both design decisions with evidence attached. Picking a
multiplier now would reintroduce the constant this module twice removed.

## Directive 2 — the CLI takes audio in and results out

```
usage: analyze_audio.py [-h] [--out OUT] [--config CONFIG] audio
```

`vars(args)` is exactly `{audio, out, config}`, asserted in `analyze_audio_test.py` so the surface
cannot grow back quietly. Every other value lives in
`src/senselab/audio/workflows/audio_analysis/data/run_config/default.yaml`, one versioned file with a
`derivation:` block and per-value comments, following `data/detection_margin/<version>.json`'s shape:
`version` is the *schema* version, and the identity of a set of values is `name` plus the merged
hash.

The adaptive loop's policy is that file's `adaptive:` section; `adaptive/policy/default.yaml` is
deleted. A file with `thresholds:` / `fusion:` / `rules:` at the top level is **refused** rather than
deep-merged into keys nothing reads — silence there would mean a run proceeding under the packaged
policy while reporting an override.

The config's identity travels: `{name, version, config_hash, sources}` is stamped on
`final/summary.json`, the comparator params on every fused row, and `disagreements.json`. Verified on
both runs — `config_hash=ec3e307cca3f88fe…` appeared in both places.

### Flags deleted rather than moved, because they bound nothing

- `--phoneme-disagreement-threshold`, `--asr-reference-model`, `--diarization-boundary-shift-ms` —
  echoed into provenance and read by nothing. `disagreements.json` reported a threshold that gated
  no decision.
- `--max-influence-rounds`, `--skip-comparisons` — declared in argparse and never read at all.

## What this invalidates

`CACHE_SCHEMA_VERSION` 10 → 11, with the reasons written into its docstring. Every axis's row count
and every number downstream of it changed, so **anything fitted against the old grids must be
re-measured, not carried over**: the scene-quality calibration profile, the convergence thresholds,
the triage gates, and the `detection_margin` mask thresholds were all fitted at spacings that no
longer exist.

## Also found while doing this

- The calibration profile's `temperature` and `token_entropy_reference_nats` **reach no fold.** Their
  only consumers were `aggregate.aggregate_asr` / `aggregate_speech_presence`, which had no
  production caller and are deleted; `fuse.fuse_axis` takes no temperature. This was already true
  before the change — deleting the aggregators only made it visible. Kept validated, with the state
  recorded in `calibration.py` and `axes.CALIBRATED_AXES`, so fitted values survive until
  `fuse_axis` takes them.
- `token_entropy` (FR-017) is now produced by the ASR task layer and consumed nowhere in the
  workflow. It remains on `ScriptLine` as a legitimate model output.
- `stage_alignment(language: str = "en")` and `PassPlan.asr_language: str = "en"` were annotated as
  non-optional while every caller passed the CLI's unset `None` straight through, so the declared
  default never applied and the real one lived in an `or "en"` inside the body. Typing the config
  properly surfaced it; both are `str | None = None` now.
- `adaptive/policy._AXIS_PRIORITY` was a hand-written dict of three axes; it now reads
  `axes.AXIS_PRIORITY`. Same values for the four active axes, so intervention ranking is unchanged.
- The `[final uncertainty] N axis map(s)` line counted `final_rows` as an axis, reporting 5 for four
  axes. It filtered by excluding known non-axis keys; it now intersects with `AXIS_NAMES`, because a
  denylist goes stale on the next key added and an allowlist cannot.

## Follow-up landed: the loop's gates were reading the wrong scale

Reported from the clean conversation — the speaker axis did not reflect the confidence of the
individual speaker information. It did not, and the cause was a scale mismatch rather than a bad
measurement.

What every other reading of that clip says about its speakers:

| evidence | value |
|---|---|
| `final/speakers.json` count posterior | **2 speakers at 0.978**, `is_multimodal: false` |
| `speakers[0].existence_uncertainty` | **0.0**, supported by all four sources |
| per-signal doubt on the speaker axis | median **0.0000**, mean 0.2072, 77.7% of 1257 readings ≤ 0.25 |

And what the axis reported: `uncertainty` **0.666**, seeding **114 of 214** buckets as
high-uncertainty regions and letting **23** converge.

`uncertainty` is normalised **binary entropy** of the mean per-signal doubt, and entropy climbs
steeply away from zero: `H(0.10) = 0.469`, `H(0.20) = 0.722`. `theta_high = 0.66` and
`theta_low = 0.33` are doubt-scaled — they are the Label Studio high/low bins — so comparing them
against entropy silently meant:

    theta_high 0.66 on H  ==  "flag anything above 17.1% doubt"
    theta_low  0.33 on H  ==  "converged only below 6.1% doubt"

Thresholds nobody chose. Three consumers made that comparison: `regions.propose_regions`,
`convergence.apply_convergence_marks` and `belief.BeliefState.uncertainty_mass`.

Compounding it, most of what the loop chased was unfixable: of the speaker axis's 0.666, aleatoric
was 0.391 and epistemic 0.275, so **59% of the mass driving region proposal was doubt no further
measurement can remove** — the waste `statistics.py` says the decomposition exists to prevent.

**Fix:** the three gates read `estimates.control_doubt`, i.e. `1 - confidence`. `confidence` is
documented as a probability, which is the scale `theta_*` are on. Convergence's round-over-round
improvement test reads the same quantity (the history carries `doubt` beside `uncertainty`), so
"stalled" and "converged" are no longer judged on two different scales; and `aleatoric_floor`, built
from `[0, 1]` degradation scores, is only now comparable to the value it floors.

**Why not `epistemic_uncertainty`**, which is the reducible part and looks like the principled
choice. It is inter-signal disagreement, so it is structurally `0.0` for a single-voter axis — and
`asr` now has exactly one voter. Gating on it would have made that axis permanently
un-investigatable while its doubt was real (measured mean 0.215, max 0.918 on the 48 kHz clip). A
lone confident-but-doubtful voter is a reason to add a second, not a reason to stop looking. Each
rule keeps its own reducibility test where the question belongs: `U1`/`U2` already gate on
`epistemic_uncertainty` themselves.

Measured effect at the gates, seeds ⇄ converged out of 214 (conversation) and 49 (48 kHz):

| axis | before | after |
|---|---|---|
| `speaker` (conv) | 114 ⇄ 23 | **13 ⇄ 152** |
| `speech_presence` (conv) | 0 ⇄ 109 | 0 ⇄ 214 |
| `asr` (48 kHz) | 9 ⇄ 33 | 1 ⇄ 41 |
| `background_mask` (conv) | 12 ⇄ 188 | 0 ⇄ 214 |

`control_scale_test.py` pins it, including that genuine doubt still seeds and that a single-voter
axis stays investigatable — the two ways a fix here could quietly disable the loop instead.

### Verified on a live run

`english_conversation_higgs_audio_v2`, cache warm, same input and config, gate column the only
difference:

| | before | after |
|---|---|---|
| termination | `no_improvement` | **`converged`** |
| speaker bucket status | 205 open, 6 irreducible, 3 converged | **0 open**, 78 irreducible, 136 converged |
| speech_presence status | 109 converged, 105 open | **214 converged** |
| background_mask status | 188 converged, 26 open | **214 converged** |
| residual mass, speaker | 9.211 | **1.278** |
| residual mass, presence / mask | 1.066 / 0.849 | **0.0 / 0.0** |
| interventions fired | 12 (S1×10, I1×1, I2×1) | 18 (S1×6, I1×6, I2×6) |

The asr axis keeps 23 open buckets in both: those are the buckets no word reached, so they have no
confidence and cannot converge. That is correct — an unmeasured bucket is not a settled one — and it
is unchanged by this fix.

Note the intervention mix. Fewer regions are proposed, yet *more* interventions fire: with 114 seeds
the budget was spent re-electing streams over buckets that were never contested, and the identity
repairs got one shot each. With 13 seeds the budget reaches them six times.

### What that exposed: the per-speaker deliverables disagree on the count

Words and spans of the transcript are otherwise stable — all 62 surface forms identical, mean
confidence 0.7232 → 0.7315 — but the speaker labels changed from `['C0']` to `['R0' … 'R4']`, and one
run now publishes three inconsistent answers about the same speakers:

| deliverable | says |
|---|---|
| `final/speakers.json` | **2 speakers** (`S0`, `S1`), posterior 0.978, `is_multimodal: false` |
| `final/diarization.json` | **5 clusters** (`R0`–`R4`), 7 segments |
| `final/transcript.json` | every word labelled from `R0`–`R4` |

**This is latent, not new.** `I2_recluster` reported `n_clusters: 5, n_segments: 7` in the *old* run
too; what changed is that the repair now reaches the deliverable, where before every word was
attributed to `C0` — one speaker, on a two-speaker recording. Both readings are wrong about the
count; the fix changed which wrong answer is published and made the disagreement visible.

The underlying issue is that `identity_repair` re-clusters against
`speaker.recluster_cosine_threshold` (0.45) and never consults the count posterior, so a confident
unimodal "2 speakers at 0.978" does not constrain a repair that emits 5. It is also a fourth id
namespace — CLAUDE.md already records that `SPEAKER_00` / `C0` / `S0` must stay distinct, and `R*`
joins them without a stated relation to `S0`.

Deliberately not fixed here: whether re-clustering should be *constrained* to the modal count,
*weighted* by the posterior, or left free with the disagreement recorded is a design decision with
evidence attached, and the same class of choice as the coupling weight above. Next session's first
speaker task.

## Still open, carried forward from the handoff

- **Insertion preservation is unexercised on real audio.** Neither clip contains a filler, so
  `single-source words: 0` is correct rather than a pass. A disfluent recording would settle it.
- **Per-edge onset/offset confidences describe member spread while published timings come from one
  aligner** (`consensus_alignment_backend: qwen`). Either relabel, or measure the spread against the
  published boundary.
- **`uncertainty` is normalised binary entropy while `confidence` is a weighted mean.** Both correct,
  not comparable, and plotted next to each other. H(0.0444) = 0.2621 is the worked example.
