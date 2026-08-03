# Removal ledger

What D-17 – D-22 make unnecessary, what is already unnecessary and nobody noticed, and the order
the removals have to happen in. A worklist, not a design record — the reasoning lives in
`layered-architecture.md`.

Baseline: **29,063 lines** across `audio_analysis/`.

## Finding: there is almost no dead code, and a lot of unwired code

An AST sweep for definitions referenced nowhere returns **4 results** out of the whole package. The
29k lines are reachable. So "what can be removed" is not a dead-code question.

Reachability from *non-test* code is a different question, and it has a much worse answer. Three
modules are built, documented, and tested, and **the pipeline never calls them**.

### `sources.py` — 526 lines, zero non-test callers

The corroborated **3 / 6 / 10 dB tier ladder** above the band floor, plus the four fabrication
guards. Fifteen exports; `grep` finds two hits in `src/senselab`, and both are false positives — a
same-named local variable in `quality_control/metrics.py:248` and a `noqa` comment in `io.py:306`.

**The pipeline detects background sources a different way.** `stages.stage_background_sources` (wired,
runs by default) calls `noise_floor.detect_stationary_sources` — ECMA-74 prominence, ≥9 dB. So there
are two implementations of one capability, one wired, and **CLAUDE.md documents the unwired one** as
the mechanism.

`io.write_background_sources(findings: Any, ...)` is annotated `Any` because the type it wants is
`SourceFinding`, from the module nothing calls.

*Decision needed.* D-21's derivative list has no tier ladder — background sources arrive as
`(background_source, project_labels/<map>, (scene_labels, T, p))`. So the ladder is not part of the
new design, and its four fabrication guards are the reasoning at risk of being lost with it.
Wire it, or delete it and move the guards' reasoning into the projection?

### `foreground.py` — 191 lines src + 201 test, test-only

`suppress_foreground` is not in `__all__`, is called by nothing, and **the documented CLI flag does
not exist**: `--foreground-suppression` appears in CLAUDE.md and zero times in
`scripts/analyze_audio.py`.

*Settled by D-23 — wire it, as a **pathway**, not a perturbation.* Foreground suppression is an
alternate route to enhancing a signal component (the background), and a perturbation is by definition
a transform that does not remove the primary information targets. Suppression removes the foreground
target on purpose, even ideally, so it is what makes a *different* component primary — the `background`
pathway's constitutive transform. Enhancement stays a perturbation: ideally it removes only non-target
content, so the primary target survives. `project_onto` / `suppression_depth_db` /
`leakage_margin_db` become what the pathway reports about itself in the route register.

### `level.py` — 389 lines, one live export

`apply_gain_db` is called by `scripts/probe_classifier_levels.py`. Everything else — `integrated_lufs`,
`true_peak_dbtp`, `loudness_range_lu`, `measure_variant`, `normalization_gain_db`,
`peak_limited_gain_db`, `AudioVariant`, `GainCapExceededError` — has no caller. `contracts.py:882`
already records the symptom as a deviation: *"write_level_json writes `<run>/level.json`, which no
stage declares and no run produces."*

*No decision needed — wire it.* D-20 asks for `(loudness, pyloudnorm, p)` and `(clipping, pcm, p)`
and this module already computes both. The signals the design wants exist and are not plugged in.

### Orphaned definitions

| site | note |
|---|---|
| `embeddings.window_embedding_at` | dies with the nine embedding signals |
| `embeddings.silhouette_voice_score` | ditto — silhouette becomes internal to clustering |
| `background_mask.MaskedRegionIntrospection` | never constructed |
| `foreground.suppress_foreground` | above |

## Removals, in dependency order

The removals are **not independent cleanups** — each is a consequence of a restructure step, and
doing it early breaks the pipeline. Ordered by what unblocks what.

### Step 1 — row types, keys, StageIO (no removals; unblocks most of them) — **done**

`shapes.py` (six L1 output kinds), `keys.py` (`Route`, `SignalKey`, `DerivativeKey`, `EstimateKey`,
arity, source closure), `stage_io.py` (the capability), `measurements.py` (native-shape parquet/JSON
with the schema in the metadata). 78 tests, ruff clean, `mypy .` clean across 389 files. Nothing can be removed before this, because every reduction that dies needs a
native-shaped artifact to be replaced *by*.

The capability turned out **smaller than the guard it replaces** because the problem changed shape:
paths derive from keys, so a stage never holds a path and there is no expression to resolve. The
predicate is over *key kinds and rounds* — finite, and acyclicity is exhaustively checkable over the
stage×round product rather than inferred from pattern overlap. `stage_io.py` is 260 lines against
`contracts.py`'s 1,883.

One design change from D-17: a round's `timeline.png` and `summary.json` move to
`L2/round/<n>/report/`. At the round root they would force a write root meaning "the files here but
not the subdirectories", and every stage's write root being one whole directory is what makes
containment structural rather than checked.

### Step 2 — StageIO replaces the guard

| removed | lines | condition |
|---|---|---|
| `contracts.py` static AST guard + `_PathResolver` | ~900 of 1,883 | StageIO exists and L1/L2/FINAL are converted |
| `KNOWN_DEVIATIONS` (47 entries) | ~350 | each entry closed or migrated |
| `stage_contract_test.py` static half | ~600 of 1,143 | ditto |

The declarations **stay** — `Artifact` patterns and their `key` tuples are what StageIO is built
from. What goes is inspection-after-the-fact of an undecidable property, which was defeated in ten
ways and whose real-run test skips-or-fails.

### Step 3 — signals removed at L1

| removed | condition |
|---|---|
| ~~`pyannote/segmentation-3.0`~~ | **done.** No code path loads it. J1/J4/C2 rebuilt on spans; P2 on Brouhaha's VAD; I4 on cross-diarizer spans |
| 9 embedding-derived signals (`speaker_distance` ×4, `speaker_change` ×2, `embedding_silhouette` ×3) | the embedder-plus-clusterer diarizer emits `speaker_spans` |
| ~~`ppgs` signal + the PER sub-signal~~ | **done** — 987 net lines out of `src/`, across 11 modules + the CLI. `features_extraction/ppg.py` stays a senselab task; only the *signal* went |
| the `scene_quality` bundle (`units: "mixed"`) | the 8 per-target scene signals exist |

`ppgs` was the only one removable without building a replacement first, and it is done.

**Two things it took with it, and one it did not.** `joint.phoneme_transcript_agreement` (J7 — "which
reading do the acoustics support") was PPG-dependent and **test-only**, the same unwired-capability
pattern as `sources.py`. The `plot.py` PPG stripe row went too. What did *not* go is
`__pairwise_phoneme_distances__`: that compares two **ASRs'** g2p sequences with each other and never
involved PPG — it is the dominant asr sub-signal and survives intact. Two things were renamed rather
than deleted, because their justification outlived their original consumer:

- `arpabet_to_ppg_inventory` → `normalize_arpabet`. It normalises ARPAbet (lowercase, stress
  stripped), which is what makes two ASRs' sequences comparable — `AH0` and `AH1` are the same
  phoneme and counting them as a difference inflates every pairwise distance.
- `ppg_per_signal_enabled` → `phoneme_signal_enabled`. The English gate was never PPG's inventory; it
  was `g2p_en`, which maps English and nothing else.

`SILENCE_LABEL` also stayed, rehomed: `speaker_claims_from_votes` needs it, because a model reporting
silence has *claimed nothing* and treating that as a claim would let the mask discount a model for
agreeing with it.

### Step 4 — the harvest/link/vote layer dissolves into derivatives

| removed | replaced by |
|---|---|
| `L2/round0/votes/<axis>.parquet` | `project` derivatives, keyed |
| `speech_presence_link` (588 lines) | a set of `project` derivatives, policy in the operator tag |
| `PassHarvest.speech_presence_evidence`, `votes_for_harvest` | the merged input pool |
| `signal_stability(..., axis=...)`'s axis parameter | stability is per-derivative (D-21) |
| `votes.py` (470), parts of `support.py` / `aggregate.py` / `estimates.py` | `fuse_axis` over the pool |

### Step 5 — the second speaker lineage

| removed | condition |
|---|---|
| `adaptive/plot._fused_axis` — self-documented as *"scaffolding for a defect, should be deleted rather than maintained"* | the store stops persisting estimates (D-22) |
| the store's materialised estimate copy + its parity oracle | ditto |

## Live defects found while surveying, not removals

- `aleatoric_floor` reads a `quality_snr`-family name present in neither ingest path → takes `None` →
  floors at `0.0` on every bucket of every run. D-21 gives it a source:
  `(scene_score, anchor/<profile>, (snr, T, p))`.
- `background_sources.parquet` is written with `[]` when detection finds nothing and the finding type
  comes from an uncalled module — so an empty artifact and an unwired subsystem look identical from
  the run directory. The same absent-vs-empty confusion this design keeps hitting.


## `segmentation-3.0`: three capabilities D-19 did not enumerate

D-19 said remove it and resolved J1, J4 and C2. Those are done: rebuilt on spans
(`occupancy.py`, `identity_binding.py`), rewired, and the frame-channel implementations deleted along
with `PassHarvest.frame_posteriors`, which was written and never read once fusion stopped using it.

**But `extract_speech_frame_posteriors` has three other consumers**, none of them named in D-19, and
each is a capability rather than a call site:

| consumer | what it does | what removal costs |
|---|---|---|
| `compute.py` → `frame_voters["frame_segmentation"]` | a speech-presence voter | **one fewer voter**, not a lost capability. `_rule_for` dispatches on `frame_mean` rather than on a model name, so brouhaha's VAD keeps its identical path with no registry edit |
| `adaptive/backends.py` (I4, FR-016) | overlap detection via `FramePosterior.overlap_probs()` | **less than first claimed — see the correction below.** Overlap *is* derivable from diarization spans; `segmentation-3.0` is the local segmentation model inside `community-1`, so the pipeline already computes it |
| `scripts/analyze_audio.py` | triage's round-0 speech gate | the gate, unless it moves to brouhaha VAD |

### Correction: overlap comes from diarization output, and one line was discarding it

I claimed no other tool exposes overlap, so I4 made `segmentation-3.0` irreplaceable. That was wrong,
and the reason is worth recording because it hid a live defect.

`segmentation-3.0` **is a diarizer** — specifically the local segmentation model inside
`community-1`'s pipeline. So `community-1` already computes overlap; the question was only whether
senselab keeps it. It does not:

```python
results.append(_annotation_to_script_lines(diarization.exclusive_speaker_diarization))
```

`exclusive_speaker_diarization` is a **partition**: every instant belongs to at most one speaker, and
concurrent speech has been resolved away before senselab sees it.

**The consequence is a defect in the J1 replacement, not just a missed substitution.**
`count_at(spans, t)` over `community-1` spans is capped at 1 by construction, so
`count_posterior_in_window` reports `p_overlap = 0.0` — *confidently*, as a measurement, for input
that could not have expressed overlap. The absent-vs-zero failure this design keeps finding, arrived at
through correct code fed structurally-impossible input.

`nvidia/diar_sortformer_4spk-v1` is unaffected: it emits per-speaker activity, so its concurrent
segments survive. So span-derived overlap works today via sortformer only.

`diarize_audios_with_pyannote` now takes `exclusive: bool = True`, and `exclusive=False` raises rather
than falling back if the overlapping view is absent — silently returning a partition when overlap was
asked for is the same failure one level down.

#### Verified live, and the cost is worse than "overlap is discarded"

`DiarizeOutput` exposes `speaker_diarization`, `exclusive_speaker_diarization` and
`speaker_embeddings`; the first two are **distinct objects**, so `exclusive=False` genuinely reaches
the overlapping view.

On the real conversation clip both views agree exactly — 5 spans, 2 speakers, **0.000 s of overlap**.
That clip is clean turn-taking, so it cannot test the question: identical output there means "nothing
to resolve", not "the switch works". Distinguishing those required audio that *does* overlap.

On a constructed clip with 3 s of genuine concurrent speech:

| view | spans | speakers | overlap |
|---|---|---|---|
| `exclusive=True` | 1 | **1** | 0.00 s |
| `exclusive=False` | 2 | **2** | **3.14 s** (2.95–6.09 s; designed 3.0–6.0 s) |

**The partition did not merely discard the overlap — it lost the second speaker entirely.** A speaker
talking for three seconds is absent from the exclusive view. So `exclusive=False` is not a refinement
for overlap accounting; without it, concurrent speakers can vanish from the run, and every downstream
count, binding and per-speaker presence inherits that.

This settles the `segmentation-3.0` question: overlap is available from diarization spans, so I4 does
not require it.

What genuinely remains distinct about `segmentation-3.0`'s overlap is that it is a **soft,
pre-threshold probability** at 16.9 ms, where spans give a hard post-threshold decision at segment
boundaries. That is a real difference — but it is the *same* difference this design already ruled on
(a tool's reported confidence vs cross-tool disagreement, both recorded, neither substituting), and for
an intervention meant to resolve doubt, two independent diarizers agreeing on overlap is the stronger
evidence.

Options, re-ranked after the correction:

1. **Wire `exclusive=False` for the workflow's diarization**, then remove `segmentation-3.0`
   entirely. Overlap becomes cross-tool and the speech voter and triage gate move to brouhaha VAD.
   Needs verification against a live `community-1` run first.
2. Keep `segmentation-3.0` for I4 only, as a soft overlap probability recorded as one tool's
   confidence — defensible, but no longer necessary.
3. Drop I4.


## TODO (deferred): overlap + capacity test fixtures via NeMo's multi-speaker simulator

The overlap clip above was hand-mixed, which is enough to answer one question and wrong to build on:
its ground truth is "designed at 3.0–6.0 s and hopefully detected" rather than a known annotation.

NeMo's multi-speaker data simulator generates sessions **with ground-truth RTTM** — controllable
speaker count, overlap ratio and silence. Two things that buys, in order of value:

1. **D-19's censoring becomes testable.** It is currently *latent*: the corpus is ≤3 concurrent
   speakers, so a 4-capacity tool never hits its ceiling and the lower-bound path has never run
   against real model output. A simulated 5- and 8-speaker session exercises it — and censoring is
   precisely the kind of logic that is invisible in the output when it bites.
2. **Overlap regression at known ratios**, so "did the exclusive view lose a speaker" becomes a
   measured recall against an annotation rather than a hand-checked span list.

Dependency to note: the simulator concatenates real single-speaker speech from a source manifest, so
it needs one (LibriSpeech-style). NeMo lives in a subprocess venv, so this runs there rather than in
the main environment.

**Deferred deliberately** — the rewiring for a full run comes first. Until these fixtures exist, two
things are verified only by a hand-built clip and should be read as such: that the overlapping view
preserves a concurrent speaker (it does — a speaker was *lost* without it), and that censoring works
at all (it has never run against real model output).
## Run findings (2026-08-03, cache cleared, 21.5 s two-speaker clip)

The rewiring works: `__overlap_count__` votes from spans, `exclusive: False` is in the diarization
provenance and therefore the cache key, and the run reports **2 speakers at p=0.977** — both diarizers
agreeing, the embedding clusterer dissenting at 1 and down-weighted to 0.046. Correct for this clip.

Five defects the run exposed, none of them caused by the rewiring:

### 1. The asr axis reports `triage_score: 1.0` in the index and `None` in the parquet

`final/estimates/asr.parquet` has `uncertainty`, `epistemic_uncertainty`, `confidence` and
`triage_score` **all None on every one of its 41 rows**. `L2/disagreements.json` lists **18 asr
entries with `triage_score: 1.0`** — the maximum, so they sort to the top of the index.

Two artifacts of one run disagree about whether the axis was measured. An unmeasured axis is being
ranked as maximally doubtful, which is the absent-vs-value confusion in its most consequential form:
the top of the triage index is phantom findings.

### 2. `n_sources: 2` while `contributing_signals: []`

Same asr rows. Two sources counted, none contributing. A count that does not agree with the list it
counts.

### 3. `high_uncertainty_rate: 0.994`

1189 of 1196 rows flagged high-uncertainty. An index that flags 99.4% of rows ranks nothing, so the
threshold is not doing the work its name claims.

### 4. `background_mask` is absent from the disagreements index

`rows_by_axis` lists `speech_presence`, `speaker`, `asr` — three. The fourth axis is fused, written to
`final/estimates/background_mask.parquet`, drawn on the timeline, and **missing from the index that
decides what a reader looks at**. This is the same fourth-axis omission `axes.py` was created to end,
still live in this artifact.

### 5. Four axes, four grids, ranked against each other

| axis | window | hop | rows |
|---|---|---|---|
| speech_presence | 0.1 s | **0.02 s** | 1070 |
| speaker | 0.25 s | 0.25 s | 85 |
| asr | 1.0 s | 0.5 s | 41 |
| background_mask | **21.0 s** | — | **1** |

The index's 100 entries mix widths of 1.0 s (18), 0.1 s (76) and 0.25 s (6), ranked by `triage_score`
with nothing recording how the spans were reconciled — exactly what D-24 predicted. Two further
observations from the same table:

- **speech_presence is 0.1 s windows at a 0.02 s hop**, so adjacent rows share 80% of their audio.
  1070 rows are not 1070 independent measurements, and no consumer is told that.
- **background_mask is a single row spanning the whole recording.** It has no time resolution at all,
  which is why its uncertainty is 0.000: one region, confident. A degenerate axis, not a timeline.
- **No axis is on the 0.1 s grid D-24 specifies.** speech_presence has a 0.1 s *window* but a 0.02 s
  hop.

### 6. The speaker axis is dominated by signals already marked for deletion

Mean uncertainty 0.859 while the speaker *count* is 0.977 confident. Of the 6 contributing signals, 4
are the embedding-derived ones D-20 removes (`::change_point` ×2, `embedding_silhouette::…` ×2). Under
the default `min` (max-doubt) aggregator they decide the axis. This is the "saturated embedding check
outvotes unanimous diarizer agreement" failure, reproduced from a real run, and it is the concrete
argument for finishing the nine-signal removal.

## Decision: `background_mask` becomes a harvested axis (VAD + ASR + speakers)

Settles the question D-22 left open. The mask's uncertainty should be **cross-source disagreement**
over VAD, ASR words and diarizer spans — not one derived judgement's self-reported confidence.
`harvested=False` read as a property of the mask when it was a property of there being one producer.

**What each source contributes depends on `--task-type`, which is why these must be votes and not a
formula.** In a speech task, VAD / words / speaker spans indicate target activity. In a breathing task
the target is the breath, speech detection is *silent* through it, and a speech vote therefore
indicates target **absence** — the case that made a mask built from voice activity alone report the
collected signal as a background source.

**Not yet implemented, and the flag stays `False` until it is.** Flipping it was tried and breaks the
pipeline: the axis enters `HARVESTED_AXES`, and every consumer then asks for evidence nothing produces
— `reliability._AXIS_SIGNALS` is a three-member map of axis → (`PassHarvest` field, key), and the mask
has no field because it never had voters. Two new test failures confirmed it immediately, one of them
`unknown axis 'background_mask'`.

So the work is, in order:

1. A `PassHarvest` field for the mask's evidence, and a `harvest_background_mask_evidence` that emits
   VAD / ASR-word / speaker-span votes **on `speech_presence`'s grid** (they share it — see the D-24
   correction).
2. A `_AXIS_SIGNALS` entry, and a link rule mapping each source to target-activity *under the declared
   task type*.
3. Flip `harvested=True`, and delete the region-count assumption in the mask writer that currently
   yields one row for the whole recording.

Worth noting for step 1: the mask and `speech_presence` will then **share evidence** (both read VAD,
ASR words and diarizer spans). Their agreement is therefore not corroboration, and D-21 rule 6's
source-closure test is what makes that computable rather than assumed.

## Verification run 2 (cache cleared, two ASR models, 2026-08-03)

Two models on purpose, to separate "the asr axis is broken" from "one recognizer has no pairs".

| check | result |
|---|---|
| `frame_segmentation` signal | **gone** — site 1 confirmed |
| `frame_brouhaha_vad` signal | present, carrying the speech evidence |
| asr axis | **41 rows, 41 measured**, mean uncertainty 0.299 |
| speaker axis | 85 rows, mean uncertainty **0.858** — unchanged |
| background_mask | still **1 row spanning 21 s**, still absent from the disagreements index |
| `high_uncertainty_rate` | still **0.9941** |

### Corrected: the asr `triage 1.0` vs `None` finding was a diagnosis, not an index bug

Run 1 had 41 asr rows with every quantity `None` while the index ranked 18 of them at
`triage_score: 1.0`. I recorded that as two artifacts disagreeing. With two recognizers the axis
measures on all 41 rows and both agree, so **the axis genuinely had no evidence in run 1** — one model
gives no pairwise comparison.

The defect is narrower than I wrote and **still real**: an unmeasured axis was ranked at the maximum
triage score, so it sorted to the top of the index. That is now *latent* rather than fixed — it needs
one ASR model to trigger, which is a supported configuration.

### Confirmed: the speaker axis is dominated by signals already marked for deletion

Mean uncertainty 0.858 (0.859 in run 1) while the count posterior is confident. Six of its eight
contributing signals are the embedding-derived ones D-20 removes — four `::spkrec-*` distances, two
`embedding_silhouette::*`. Two runs agreeing on 0.86 while both diarizers agree on the speaker count is
the strongest available argument for finishing the nine-signal removal.

### Open: the VAD provenance is verified in-process but not in an artifact

`frame_posteriors_provenance["brouhaha_vad"]` is asserted by a unit test and passes, but
`final/summary.json` has no `frame_posteriors` block, so I could not confirm it reaches a run artifact.
Not claimed either way — the in-memory record exists; whether any writer surfaces it is unverified.

### Why the speaker axis has six embedding signals: it is a cross product

`speaker.py` builds them in a nested loop — `for m in diarizers: for emb in embedders:
votes[f"{m}::{emb}"]` — so the count is **diarizers × embedders**. The run has 3 × 2 = 6:

- diarizers: `community-1`, `sortformer`, **and** `embedding_silhouette/spkrec-ecapa-voxceleb`
- embedders: `spkrec-ecapa-voxceleb`, `spkrec-resnet-voxceleb`

It grows multiplicatively. A third embedder makes 9; a fourth diarizer makes 12. The two genuinely
cross-tool signals (`__cross_diar_label_disagreement__`, `__overlap_count__`) stay at one each, because
each already folds across every diarizer.

**Corrects D-20's arithmetic**, which predicted "4 × `speaker_distance` (2 diarizers × 2 embedders)".
The synthetic embedding-derived diarizer was not counted, so the removal is larger than written.

**The count is not the defect; the correlation is** — and it is precisely the within-bucket
independence that matters for correctness:

- All six share the same two embedders. `community-1::ecapa` and `sortformer::ecapa` differ *only* in
  whose segments were embedded; the model the cosine distance comes from is identical.
- `embedding_silhouette/ecapa::ecapa` is degenerate — the same model as both diarizer and embedder,
  comparing ecapa against itself.
- Under the default `min` (max-doubt) aggregator a single high value decides the axis, so six
  correlated signals reliably outvote two independent folds.

That is the mechanism behind 0.858 uncertainty against a 0.977-confident count posterior: not two views
disagreeing, but one family of near-duplicate signals outvoting the folds. D-21 rule 6's source-closure
test is what makes this computable rather than argued — all six closures contain the same embedders.

#### Correction: the cross product is a claim × verifier matrix, and only one cell is circular

The explanation above is wrong about the mechanism. The `::` vote is a **claim validation**, not a
distance labelled with two model names. The *diarizer* supplies a claim — "same speaker as the previous
bucket on this track", or "the speaker changed" — and the *embedder* supplies a verdict on it via cosine
distance. `prev_emb_per_track` is keyed `(diarizer, embedder, cluster_id)`, so the comparison points
differ per diarizer.

So `community-1::ecapa` and `sortformer::ecapa` are **not the same distance twice**. They validate
different claims with the same yardstick, which is a real second dimension. "They differ only in whose
segments were embedded" was wrong.

The three diarizers stand in *different* relations to the embedders, which is what the uniform-cross-
product reading missed:

| cell | claim from | verdict from | independent |
|---|---|---|---|
| `community-1::{ecapa,resnet}` | spans, no embeddings of its own in play | ecapa / resnet | yes |
| `sortformer::{ecapa,resnet}` | spans, no embeddings | ecapa / resnet | yes |
| `embedding_silhouette/ecapa::resnet` | ecapa clustering | resnet | yes |
| `embedding_silhouette/ecapa::ecapa` | ecapa clustering | **ecapa** | **no — circular** |

**Exactly one of the six is degenerate**, not all six. So the cross product is not the defect; it is a
legitimate claim × verifier matrix with one self-validating cell.

What survives is narrower and is about **aggregation, not duplication**: six validations against two
folds under the default `min` (max-doubt) aggregator means the validation family decides the axis, which
is why 0.858 sits against a 0.977-confident count. That is an argument for weighting the folds against
the validations — and for removing the circular cell — rather than for deleting the family wholesale.

**This weakens the case for the nine-signal removal as stated in D-20.** Its premise was that each was
"a computation over vectors carrying an unrecorded estimator choice". The estimator choice is real
(cosine, and which lag), but these are not redundant re-measurements, and removing all of them removes
genuine diarizer-claim validation. Worth re-deciding rather than executing as written.

Also unused and worth noting: `community-1` exposes `speaker_embeddings` on its pipeline output, which
nothing reads. A diarizer's own embeddings validating its own claims would be a fourth circular cell if
it were wired the same way.

#### Correction 2: no cell is circular, and D-20's nine-signal removal is not justified by this evidence

The "one circular cell" claim above is also wrong. `embedding_silhouette/ecapa::ecapa` is not
self-confirming, because the claim and the verdict are **different computations over the same
vectors**:

- the claim comes from `cluster_pass_speakers` — a **global** partition of every window embedding in
  the pass, sweeping *k* and selecting by silhouette;
- the verdict is a **local pairwise** cosine between this bucket and the previous one on the track.

A global partition can group two windows that are locally distant, joined transitively through
intermediates, and it chooses *k* by a whole-pass criterion the local check knows nothing about. So the
pairwise check *can* contradict the clustering it is validating. Shared representation is not shared
computation.

**So all six cells are legitimate claim × verifier pairs, and nothing here supports removing them.**

Two claims withdrawn in sequence, recorded because the reasoning drifted the same way twice — from
"these signals share a model" to "therefore they are redundant", which does not follow:

1. *"They differ only in whose segments were embedded"* — wrong; the diarizers supply different claims.
2. *"Exactly one is circular"* — wrong; a global partition and a local pairwise distance are different
   questions.

**What is left of the 0.858-vs-0.977 finding is purely about aggregation.** Six validations and two
folds under the default `min` (max-doubt) aggregator means the validation family decides the axis. That
is a weighting question — the folds measure cross-tool agreement about *who*, the validations measure
whether an embedder corroborates a diarizer's local same/change claim, and max-doubt lets the more
numerous family win regardless of which is more informative.

**D-20's nine-signal removal should not be executed on this evidence.** Its premise was that each was a
redundant computation carrying an unrecorded estimator choice. The estimator choice is real and worth
naming (cosine, and which lag); the redundancy is not established, and twice now I inferred it from a
shared model name.

Design observation, unresolved: the synthetic diarizer is *parameterised by embedder*
(`embedding_silhouette/<embedder>`), and only the ecapa one is instantiated. The same clustering
algorithm over resnet embeddings would be a legitimate fourth diarizer, making the matrix 4 × 2.

#### `emb_cluster` should not be a consumed product — two things ride on one object

The clustering's **diarization output is not consumed through `emb_cluster`.** It goes into
`diarization.by_model` alongside pyannote and sortformer, and every downstream consumer reads it there,
uniformly. That is correct and is what "clustering output is a diarization estimate" means.

What `emb_cluster` is still consumed for is one thing, at `compute.py:342`:

```python
sf = emb_cluster.get("empirical_same_speaker_floor")
df = emb_cluster.get("empirical_diff_speaker_floor")
```

That is a **calibration band** — the empirical cosine distances separating same-speaker from
different-speaker in that embedding space — not a diarization estimate. Two unrelated products were
bundled on one object, and only one of them is diarization.

**The defect this exposes: the band is per-embedder and is applied globally.** ecapa's cosine
distribution is not resnet's, but `same_floor_eff` / `diff_floor_eff` are a single pass-level pair
handed to `harvest_speaker_votes`, which uses them to calibrate **every** embedder's distances. So
resnet distances are calibrated with ecapa's floors — and with the `break` in place it was always
whichever embedder sorted first, silently.

That is why `emb_clusters[0]` was the wrong instinct. Not a poor choice of representative: there should
be no pass-level representative for a per-embedder quantity.

The fix, not yet made: `harvest_speaker_votes` takes `{embedder → (same_floor, diff_floor)}` instead of
one pair, and the `::` validation looks up the band for the embedder it is using. `emb_cluster` then has
no consumer and goes away, leaving the clustering to be exactly what it is — a diarizer that emits
spans into `by_model`.

Worth checking when that lands: whether the calibration band, being derived from the same clustering
whose claims it calibrates, is measuring the embedding space or the clustering's own separation. That is
the circularity question I asked in the wrong place earlier — it belongs here, about the band, not about
the claim × verifier cells.

## Verification run 3: four models, and the uncertainty fell

Cache cleared, two ASR models, same clip.

| | run 2 | run 3 |
|---|---|---|
| count-posterior sources | 3 | **4** (`spkrec-resnet-voxceleb` clustering now a model) |
| `modal_count` / p | 2 @ 0.977 | 2 @ 0.978 |
| speaker mean uncertainty | 0.858 | **0.7914** |
| `::` validation signals | 6 | 8 |

**The discriminating result: uncertainty fell while the signal count rose.** Two more validations were
added and doubt dropped 0.067. That distinguishes the two competing diagnoses, and it favours the one
that survives:

- *"too many correlated signals"* predicts uncertainty rising or holding as signals are added under
  max-doubt;
- *"the calibration band was applied to the wrong embedder"* predicts it falling once resnet's
  distances are read against resnet's own separation, because the excess doubt was borrowed
  calibration rather than redundancy.

The second is what happened. Adding a genuine independent model *reduced* the axis's uncertainty, which
is what should happen when an independent source agrees.

**This is the third and clearest refutation of D-20's nine-signal removal.** The evidence I had been
citing for it — the 0.858-vs-0.977 gap — was substantially a calibration defect, and it moved when the
calibration was fixed rather than when signals were removed. The removal should not be executed on this
reasoning. What remains worth doing is naming the estimator choice on each key (cosine, which lag), which
is a keying task, not a deletion.

Both same-model pairings now exist (`…/ecapa::ecapa`, `…/resnet::resnet`) and neither is circular, for
the reason established earlier: a global silhouette-optimised partition and a local pairwise cosine are
different computations over the same vectors, and the second can contradict the first.

## The P2 / I4 decision, with the resolution question settled

**P2 is the driver, not I4.** `_p2_trigger` fires when a speech-presence region's votes are dominated
by **coarse** voters — sentence-level ASR, 30 s Whisper `no_speech`, AST's 10.24 s window — each casting
one identical vote across every bucket it spans, so *agreement among them is an artifact of window size
rather than evidence about that bucket*. Or when `frame_dispersion` says the bucket straddles an onset.
Either way the response is to re-measure locally at frame resolution. That is the capability at stake,
and it has nothing to do with overlap. `I4_overlap_detection` fires on a speaker region co-located with
an asr region and reuses P2's output.

**The resolution objection is settled: brouhaha's VAD hop is `0.016875 s`, the same 16.9 ms as
`segmentation-3.0`.** Both are pyannote models on the same frame grid, so substituting brouhaha costs P2
nothing at the one thing P2 exists for. That was the only reason to keep `segmentation-3.0`.

**Recommended: option 1.**

1. **Repoint P2's backend at brouhaha on the crop, and I4's overlap at diarizer spans.**
   `segmentation-3.0` then has no consumer and goes. Overlap from spans is verified — 3.14 s detected on
   a constructed clip, where the exclusive view had *lost* the second speaker entirely.
2. Keep `segmentation-3.0` for P2 only — no longer justified, since the hop is identical.
3. Drop both interventions — loses a real capability for no reason.

One consequence to accept with option 1, stated rather than discovered later: P2 re-measures with the
*same model* that already voted in round 0 (`frame_brouhaha_vad`), where it previously brought in a
second, independent model. What P2 buys is then purely **locality** — the same estimator on a crop,
which is a genuine re-measurement because a model given a short span sees different context — but not
independence. Under `segmentation-3.0` it bought both. Whether that matters depends on whether P2's
value was ever the second opinion or always the finer localisation; its trigger says localisation.

Also resolved: `frame_posteriors` **does** reach an artifact — it is in
`L1/signals/frame_brouhaha_vad.parquet`'s `signal_provenance` metadata. An earlier entry recorded this as
unverified.


## Option 1 executed: `segmentation-3.0` is no longer loaded

- **P2** (`_p2_execute`) calls `backends.speech_posteriors` — Brouhaha's VAD head on the crop, at the
  **same 16.9 ms hop**, so nothing is lost at the localisation P2 exists for.
- **I4** (`_i4_execute`) calls `backends.overlap_track_from_spans` — per-frame overlap from
  cross-diarizer spans via `occupancy.count_at`, reading `ctx["passes"]`.
- `extract_speech_frame_posteriors`, `collapse_to_overlap_prob` and `FramePosterior.overlap_probs` are
  deleted. `FramePosterior`, `stitch_frames` and the chunking helpers stay — Brouhaha uses them.

**Three consequences, each a decision rather than a discovery:**

1. **P2 and I4 are now independent.** The contract let I4 run "light (reuses P2 output)" because P2
   emitted `overlap_posterior` as a side effect. Brouhaha's VAD is single-channel and reports no
   overlap, so P2 writes none — and it writes *nothing* rather than a zero, because a reader cannot
   tell an overlap of 0.0 from an unmeasured one. The test asserting the old side effect is inverted to
   assert its absence.
2. **P2 buys locality, not a second opinion.** It re-measures with the same model that already voted in
   round 0. A model given a short span sees different context, so it is a genuine re-measurement — but
   under `segmentation-3.0` it was also an independent one. If independence is wanted back, a second
   continuous VAD as a frame voter restores it without reviving a model whose channels nothing uses.
3. **I4's overlap is a decision, not a posterior** — 1.0 where two or more distinct speakers cover the
   instant. A soft probability would need a model that reports one, and manufacturing one from hard
   spans would fabricate confidence. It depends on `exclusive=False`, which is wired and verified.

`SEGMENTATION_MODEL_ID` remains as an unused constant, deliberately left rather than removed by a
regex that had already over-stripped the module once.
