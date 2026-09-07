# Span-fill recovery: does the periodic fill recover the true label, or manufacture a false one?

`span_yamnet_input` (`src/senselab/audio/tasks/classification/yamnet.py`) fills a span shorter than
YAMNet's 0.96 s native frame by centring it and extending it periodically from its own samples.
Measured elsewhere, that fill is implicated in 68 of 69 "artefact family" labels seen across ten
recordings, but the same short durations also carry genuine airway events. This experiment supplies
ground truth — a span YAMNet classifies with **no padding at all** — and asks whether trimming that
span down and refilling it recovers the same label, and down to what duration.

**No pipeline file is modified.** `span_yamnet_input` is imported and called exactly as the pipeline
calls it. The zero-pad control is implemented locally in the experiment script and is not used by
the pipeline itself. This document and its script are the only artifacts.

## Method

**Reference set.** A reference span is ≥ 0.96 s (YAMNet's own frame, so it is classified with no
padding) and its native top-1 score is ≥ 0.5. The reference label/score is that native
classification.

Two sources of reference spans, both restricted to the three subjects for which local audio was
actually available (see Data, below):

- **`store_span`** — spans the triage pipeline itself proposed, read from a completed run's
  `store.jsonl`. The pipeline's own `span_yamnet` measurement is reused as the reference rather
  than recomputed: for a span ≥ 0.96 s, `span_yamnet_input` is a no-op (`frame_filled: False` is
  asserted for every one), so what the store already recorded *is* the unpadded native
  classification. When a span produces more than one native YAMNet window (span duration > 0.96 s),
  the reference is the highest-scoring window's label; 44 of the 47 `store_span` references have
  every one of their native windows agree on that label (full agreement), which is the check for
  treating "the span" as one coherent label rather than an average over drifting content.
- **`native_window`** — YAMNet's own native hop windows (0.96 s, 0.48 s hop), taken directly from
  *other* recordings of the same three subjects with no span-detection step at all. This source
  exists because the three subjects' analysed recordings are narration tasks (`Story-recall`,
  `Story-recall-(v2)`): their `store_span` references are 41/47 `Speech` and the rest `Silence` —
  no transient/airway content clears the 0.5 floor natively. `Respiration-and-cough-*` recordings
  for the same three subjects were classified with YAMNet's own windowed mode
  (`classify_audios(model="yamnet")`, no window-length override — this *is* how YAMNet classifies
  natively), and eligible windows (full 0.96 s, top-1 ≥ 0.5) were thinned by greedy non-overlap
  selection in time order so a single ~6 s breathing bout's 50%-overlapping windows are not counted
  as a dozen independent references.

**Manipulation.** Each reference span is trimmed to 0.06, 0.10, 0.15, 0.20, 0.30, 0.40, 0.48, 0.70 s
— skipping any trim ≥ the span's own length (never triggered here: the shortest reference span is
0.9597 s, longer than every trim). Trimming is done twice, from the **start** of the span and from
its **centre**. Each trimmed fragment is then classified two ways:

- **filled** — `span_yamnet_input(plain, fragment_extent)`, the pipeline's own periodic fill.
- **zero** — the same centring, silence instead of the tiled repeat (what the pipeline did before
  the fill existed).

`plain` is reconstructed exactly as `preprocess()` builds it: mono downmix, resampled to 16 kHz,
peak-scaled only if clipping. Every fragment is then classified with
`classify_audios(model="yamnet", top_k=10)`. All 222 references × 8 trims × 2 positions × 2 arms =
**7104 fragments** were classified in one script run, in four `classify_audios` calls (chunk size
1000) — `classify_audios` spawns one subprocess venv per call, so batching this way rather than one
call per fragment is what keeps the run under two minutes.

**Reproduce:**

```bash
uv run python specs/20260817-triage-workflow-dag/benchmarks/scripts/span_fill_recovery.py \
    --out results.json
uv run python specs/20260817-triage-workflow-dag/benchmarks/scripts/summarize_span_fill_recovery.py \
    results.json
```

## Data

Stores: `~/Downloads/triage_10subj_20260908/out/<subject>/<run>/run/store.jsonl` (10 subjects, one
run each). Audio: `~/Downloads/b2ai_v31_bids_07_01_v3/<subject>/<session>/audio/*.wav`, matched to a
store by the basename of its `recording` stream entity's `path` attribute.

**Only 3 of the 10 subjects had a locally-present matching recording** at the time of this
experiment; the BIDS tree on this machine holds full audio for `sub-17578482-…`, `sub-17cee767-…`
and `sub-1f4ea26f-…` only. The other seven stores' recordings (`sub-0032892c-…`, `sub-004d42e9-…`,
`sub-00e88aac-…`, `sub-00f74092-…`, `sub-014813db-…`, `sub-016023f6-…`, `sub-01a1f0fd-…`) are not on
disk and were skipped — not sampled around, skipped, because there is no ground truth to compute
without the audio. This is a hard constraint of the available data, not a choice; it means the
537-span population named in the brief is not the population measured here, only the ≈ 22% of it
belonging to these three subjects (183 of their spans are ≥ 0.96 s before any confidence filter).

The three subjects' analysed recordings are narration tasks, so the `native_window` supplement
(25 `Respiration-and-cough-*` recordings, one set per subject) is what populates the
transient/airway stratum. No other node of the pipeline ran on those 25 recordings; their reference
labels come only from YAMNet's own native windowing, independent of the triage span-proposal step.

## Reference set: 222 spans

| stratum | store_span | native_window | total | mean native score | dominant labels |
| --- | --- | --- | --- | --- | --- |
| stationary | 5 | 43 | 48 | 0.835 | Silence (32), Mains hum (7), Hum (5), Noise (2), Buzz (1), Static (1) |
| speech | 41 | 17 | 58 | 0.898 | Speech (58) |
| transient | 0 | 69 | 69 | 0.750 | Breathing (35), Explosion (11), Sigh (5), Snort (4), Snoring (4), Gasp (3), Burst/pop (2), Whack (1), Thump (1), Sneeze (1) |
| other | 1 | 46 | 47 | 0.643 | Synthesizer (12), Heart sounds/heartbeat (9), Music (2), Electric shaver (2), Animal (2), Writing (2), Owl (2), + singles |

`transient` includes the design brief's own broadening, "anything impulsive" — `Explosion`,
`Burst, pop`, `Whack, thwack`, `Thump, thud` alongside the named airway labels — because a periodic
tiling turns any single-shot burst into a pulse train the same way it does an airway burst.
`stationary`'s 32 `Silence` references matter for reading the tables below: silence is a trivial
case for both arms (padding with more silence, or tiling near-zero samples, both stay near-silent),
so it is broken out separately from genuine tonal content (`Mains hum`/`Hum`/`Noise`/`Buzz`/`Static`,
n = 16) below.

## Recovery tables

Cells are `top-1 fraction / top-4 fraction (n)` — the fraction of reference spans whose native label
is still the fragment's top-1 (resp. in the top-4) after trim + fill/zero-pad.

### Trimmed from the START

| trim (s) | stationary/filled | stationary/zero | speech/filled | speech/zero | transient/filled | transient/zero | other/filled | other/zero |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.06 | 0.38/0.46 (48) | 0.67/0.67 (48) | 0.00/0.05 (58) | 0.00/0.31 (58) | 0.00/0.00 (69) | 0.00/0.00 (69) | 0.19/0.28 (47) | 0.02/0.11 (47) |
| 0.10 | 0.48/0.56 (48) | 0.67/0.67 (48) | 0.03/0.09 (58) | 0.09/0.34 (58) | 0.00/0.03 (69) | 0.00/0.00 (69) | 0.15/0.28 (47) | 0.00/0.09 (47) |
| 0.15 | 0.56/0.65 (48) | 0.67/0.67 (48) | 0.22/0.31 (58) | 0.21/0.38 (58) | 0.03/0.12 (69) | 0.00/0.01 (69) | 0.17/0.38 (47) | 0.02/0.09 (47) |
| 0.20 | 0.62/0.73 (48) | 0.67/0.67 (48) | 0.47/0.53 (58) | 0.47/0.66 (58) | 0.06/0.07 (69) | 0.00/0.03 (69) | 0.23/0.36 (47) | 0.04/0.09 (47) |
| 0.30 | 0.71/0.79 (48) | 0.65/0.67 (48) | 0.66/0.78 (58) | 0.64/0.81 (58) | 0.04/0.12 (69) | 0.00/0.01 (69) | 0.36/0.45 (47) | 0.09/0.26 (47) |
| 0.40 | 0.62/0.71 (48) | 0.60/0.67 (48) | 0.67/0.76 (58) | 0.76/0.86 (58) | 0.09/0.22 (69) | 0.00/0.01 (69) | 0.40/0.55 (47) | 0.02/0.19 (47) |
| 0.48 | 0.69/0.75 (48) | 0.65/0.69 (48) | 0.79/0.88 (58) | 0.84/0.93 (58) | 0.17/0.35 (69) | 0.03/0.09 (69) | 0.45/0.62 (47) | 0.13/0.30 (47) |
| 0.70 | 0.65/0.75 (48) | 0.65/0.67 (48) | 0.97/0.98 (58) | 0.93/0.98 (58) | 0.23/0.57 (69) | 0.10/0.42 (69) | 0.55/0.74 (47) | 0.11/0.19 (47) |

### Trimmed from the CENTRE

| trim (s) | stationary/filled | stationary/zero | speech/filled | speech/zero | transient/filled | transient/zero | other/filled | other/zero |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.06 | 0.46/0.58 (48) | 0.67/0.67 (48) | 0.00/0.02 (58) | 0.00/0.17 (58) | 0.00/0.01 (69) | 0.00/0.00 (69) | 0.21/0.28 (47) | 0.00/0.09 (47) |
| 0.10 | 0.46/0.65 (48) | 0.67/0.67 (48) | 0.00/0.05 (58) | 0.09/0.41 (58) | 0.00/0.07 (69) | 0.00/0.00 (69) | 0.23/0.30 (47) | 0.00/0.11 (47) |
| 0.15 | 0.54/0.81 (48) | 0.67/0.67 (48) | 0.24/0.38 (58) | 0.29/0.48 (58) | 0.01/0.06 (69) | 0.01/0.04 (69) | 0.36/0.47 (47) | 0.00/0.15 (47) |
| 0.20 | 0.65/0.85 (48) | 0.67/0.67 (48) | 0.52/0.62 (58) | 0.55/0.79 (58) | 0.03/0.07 (69) | 0.01/0.06 (69) | 0.34/0.49 (47) | 0.00/0.09 (47) |
| 0.30 | 0.71/0.85 (48) | 0.65/0.67 (48) | 0.67/0.79 (58) | 0.67/0.93 (58) | 0.06/0.14 (69) | 0.03/0.09 (69) | 0.43/0.53 (47) | 0.09/0.09 (47) |
| 0.40 | 0.69/0.73 (48) | 0.65/0.69 (48) | 0.79/0.84 (58) | 0.81/0.91 (58) | 0.13/0.22 (69) | 0.06/0.10 (69) | 0.47/0.72 (47) | 0.09/0.19 (47) |
| 0.48 | 0.77/0.83 (48) | 0.65/0.71 (48) | 0.83/0.93 (58) | 0.88/0.98 (58) | 0.07/0.29 (69) | 0.06/0.14 (69) | 0.51/0.79 (47) | 0.06/0.17 (47) |
| 0.70 | 0.71/0.83 (48) | 0.65/0.67 (48) | 0.95/0.98 (58) | 0.91/1.00 (58) | 0.26/0.65 (69) | 0.19/0.49 (69) | 0.62/0.85 (47) | 0.13/0.28 (47) |

Start vs. centre make little qualitative difference to any stratum's ranking — the onset-vs-interior
distinction the design asked about does not separate the strata further than trim duration and
stratum already do.

### The `stationary` numbers above are inflated by `Silence`. Broken out (start-anchored):

| trim (s) | Silence only: filled top1/top4 | Silence only: zero top1/top4 | Hum/Buzz/Noise/Static (n=16): filled top1/top4 | Hum/Buzz/Noise/Static: zero top1/top4 |
| --- | --- | --- | --- | --- |
| 0.06 | 0.56/0.62 | 1.00/1.00 | 0.00/0.12 | 0.00/0.00 |
| 0.10 | 0.72/0.78 | 1.00/1.00 | 0.00/0.12 | 0.00/0.00 |
| 0.15 | 0.84/0.88 | 1.00/1.00 | 0.00/0.19 | 0.00/0.00 |
| 0.20 | 0.88/0.88 | 1.00/1.00 | 0.12/0.44 | 0.00/0.00 |
| 0.30 | 0.91/0.94 | 0.97/1.00 | 0.31/0.50 | 0.00/0.00 |
| 0.40 | 0.91/0.97 | 0.91/0.97 | 0.06/0.19 | 0.00/0.06 |
| 0.48 | 0.91/0.94 | 0.97/1.00 | 0.25/0.38 | 0.00/0.06 |
| 0.70 | 0.91/0.91 | 0.97/1.00 | 0.12/0.44 | 0.00/0.00 |

`Silence` is trivially "recovered" by zero-padding (padding *is* silence) and nearly as trivially by
tiling (near-zero samples tiled stay near-zero). It carries two-thirds of the `stationary` stratum's
weight, and it is the only reason `stationary`'s top-line numbers look strong. The 16 genuinely
tonal references (`Mains hum`, `Hum`, `Noise`, `Buzz`, `Static`) never exceed 31% top-1 recovery at
any trim under either arm, and zero-padding is at essentially 0% for all of them (n = 16 is small;
read this as a direction, not a precise rate).

## What the fill produces when it loses the reference (filled arm, all trims/positions, n = 3552)

Top 15 false top-1 labels overall:

| label | count |
| --- | --- |
| Synthesizer | 254 |
| Silence | 215 |
| Noise | 194 |
| Music | 141 |
| Engine | 121 |
| Inside, small room | 113 |
| White noise | 77 |
| Vehicle | 60 |
| Hum | 52 |
| Sound effect | 38 |
| Heart sounds, heartbeat | 35 |
| Writing | 33 |
| Animal | 32 |
| Snake | 31 |
| Hiss | 31 |

**Yes — the artefact family the design brief opened with is exactly what dominates.**
`Synthesizer` is the single most common false label the fill produces, by a wide margin, echoing the
9-of-10-recordings finding that motivated this experiment. But the false-label identity is
stratum-dependent:

- **`stationary`** false labels: `Synthesizer` (69), `Noise` (31), `Heart sounds, heartbeat` (25),
  `Hum` (21), `Music` (18), `Electric shaver, electric razor` (16) — mechanical/tonal confusions,
  consistent with a comb-filtered short fragment reading as a machine.
- **`speech`** false labels: `Synthesizer` (59), `Music` (56), `Silence` (54), `Noise` (45),
  `Engine` (32) — a striking concrete case: a 962 ms `Speech` span (native score 0.827) trimmed to
  60 ms and tiled scores `Synthesizer` 0.939, beating the true label outright.
- **`transient`** false labels: `Silence` (107), `Noise` (86), `Engine` (72),
  `Inside, small room` (57), `White noise` (51) — here the dominant failure mode is not
  "Synthesizer" but "ambient/nothing": a 60 ms fragment of a `Sigh` (native score 0.65–0.70) tiled to
  0.96 s reads as `Silence` 1.00; a 60 ms `Snort` fragment reads as `Noise` 0.46–0.65. The comb
  artefact that manufactures `Synthesizer` on stationary/speech material more often manufactures
  `Silence`/`Noise` on airway material, because the airway fragment itself is quieter and less
  spectrally structured at these durations.
- **`other`** false labels: `Synthesizer` (96), `Silence` (50), `Music` (36) — same mechanical
  skew as `stationary`.

## Does the fill beat zero-padding at all?

Aggregated over every trim, position and stratum:

| arm | top-1 | top-4 | n |
| --- | --- | --- | --- |
| filled | 0.357 | 0.468 | 3552 |
| zero | 0.294 | 0.386 | 3552 |

Filled wins on the aggregate. Per stratum it is not uniform:

| stratum | filled top-1 | zero top-1 | filled top-4 | zero top-4 |
| --- | --- | --- | --- | --- |
| stationary | 0.605 | **0.654** | 0.721 | 0.672 |
| speech | 0.488 | **0.509** | 0.562 | **0.685** |
| transient | **0.074** | 0.031 | **0.187** | 0.094 |
| other | **0.355** | 0.049 | **0.505** | 0.153 |

Zero-padding is *at least as good as* filling on `stationary` and `speech` top-1, and clearly better
on `speech` top-4 — because zero-padding a mostly-silent or already-quiet fragment does nothing
worse than filling would, while tiling a short speech fragment manufactures the `Synthesizer` comb
described above. Filling only wins decisively on `transient` and `other`, and on `transient` both
arms are poor in absolute terms (7.4% vs 3.1% top-1). Head-to-head per (span, trim, position): filling
recovers something zero-padding misses in 394 cases; zero-padding recovers something filling misses
in 172; both recover the same 873; neither recovers the remaining 2113 (out of 3552 pairs). Filling
is the better default on net, but it is not a universal improvement, and the pipeline's own rationale
for it — "a span that is mostly inserted silence is classified as silence" — is true of `speech` and
`stationary` less than the design assumed: for those two strata, reading it as silence loses less
than tiling it does about as often as it loses more.

## Verdict

**The stationary/transient hypothesis, as literally stated, does not hold.** "Periodic filling
preserves stationary content at any duration" is refuted directly: genuinely tonal `stationary`
content (`Hum`, `Buzz`, `Noise`, `Static` — excluding trivial `Silence`) is recovered by tiling at
0–31% top-1 across every trim tested, never reaching a duration at which it is reliably preserved.
The half of the hypothesis that does hold is the second half — periodic filling corrupts transient
content — and even there, filling is measurably *less bad* than the alternative (zero-padding), not
acceptable in an absolute sense: 7–26% top-1 recovery at the largest trim tested (0.70 s, 73% of the
native frame) is not a duration a downstream consumer should trust.

**Duration threshold:** there is no trim at or below which the fill is trustworthy for
transient/airway content — recovery is near zero through 0.40 s and still under 30% top-1 at 0.70 s,
the largest trim tested (73% of the native 0.96 s frame). For `speech`, recovery crosses 50% top-1
between 0.20 and 0.30 s and is reliable (≥ 90%) only at 0.48–0.70 s. Genuinely tonal `stationary`
content never reaches a reliable duration in this data. `Silence` is recovered at every duration
tested by both arms, but that is a property of silence, not of the fill.

**Stratum-dependence:** yes, strongly, but not along the axis the hypothesis proposed. The working
distinction this data supports is *spectrally structured vs. not*, and within "structured" content,
tonal/stationary material fares no better than transient material — both are poorly recovered below
~0.3 s. `Speech` is the only stratum that recovers well, and only past ~0.3 s.

**Does the fill beat zero-padding?** On aggregate, yes (0.357 vs. 0.294 top-1). Per stratum, no —
it loses or ties on `stationary` and `speech` top-1, and wins clearly only on `transient` and
`other`, where both arms remain weak in absolute terms.

## What this licenses, and what it does not

**Licenses:** treating `frame_filled: True` as a real reliability signal for anything under ~0.3 s,
regardless of the span's `measure` (`amplitude`/`gap`/`continuity`/`asr`) or duration alone —
neither separated the populations in the motivating background, and this experiment shows why: the
relevant axis is spectral structure, which duration and `measure` both proxy for badly. It also
licenses treating `Synthesizer`, `Music`, `Noise`, `Engine`, and `Silence` as fill-artefact-suspect
labels specifically when `frame_filled: True` and the source span is short, since those are the
labels the fill manufactures far out of proportion to anything the recordings actually contain.

**Does not license:** a numeric duration cutoff for production use — the reference set here is 222
spans from three subjects' recordings (mostly narration plus one respiration-and-cough task per
subject), not the ten-subject, 535-span population the pipeline runs over, and seven of those ten
subjects were entirely unmeasured for lack of local audio. It does not license a verdict on `measure`
(amplitude vs. gap) as a factor — this design held it constant per stratum rather than crossing it,
so a `measure`-specific effect could exist and not show here. It does not resolve which specific
label thresholds should gate a `span_yamnet` window's admission into a downstream verdict; it only
establishes that duration and `measure` are the wrong knobs to gate on, and that a spectral-content
test (comparable to the `stationary` vs. genuinely-tonal split found by accident here) is the
direction a real fix would need to go. The `store_span`/`native_window` reference-source split is
itself a methodological compromise this experiment introduces to reach the transient stratum at
all — the two sources use different definitions of "a span" (pipeline-proposed region vs. raw YAMNet
hop window) and are not a controlled comparison of each other.
