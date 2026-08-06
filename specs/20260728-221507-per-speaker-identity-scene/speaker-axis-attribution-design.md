# Design: the speaker axis measures attribution, not change

Date: 2026-08-05. Approved in conversation; supersedes the speaker axis's current composition.

## The problem, measured

On a clean two-speaker conversation (`english_conversation_higgs_audio_v2.wav`, 21.1 s), everything
that describes the speakers is confident:

| evidence | value |
|---|---|
| `final/speakers.json` count posterior | 2 speakers at **0.978**, `is_multimodal: false` |
| `speakers[0].existence_uncertainty` | **0.0**, converged |
| `speakers[1].existence_uncertainty` | **0.0221**, converged |
| `final/per_speaker_presence.parquet` `speech_presence_confidence` | mean **0.8899**, 4 distinct values |
| `final/per_speaker_presence.parquet` `speech_presence_uncertainty` | mean **0.1683**, 3 distinct values |

The speaker axis reported `uncertainty` **0.666**.

The axis is not missing information. It and the per-speaker tracks derive from the *same*
`harvest_speaker_votes` output, and read it to answer different questions:

- **`speaker.per_speaker_tracks`** — per bucket, `speech_presence_confidence` is the share of diar
  models placing this speaker here and `speech_presence_uncertainty` the binary entropy of that
  share. Question: *who is here, and do the models agree?* Mean doubt **0.118**.
- **the axis** — `same_label_uncertainty` and `change_inconsistency_uncertainty` per
  (diar model × embedder) pair, plus `__cross_diar_label_disagreement__` and `__overlap_count__`.
  Question: *did the speaker change since the previous bucket?* Mean **0.666**.

The change-detection framing is structurally noisy on the run's 0.1 s grid: it asks "same as
before?" ten times a second and validates each answer against embeddings whose windows are 0.5 s at
a 0.25 s hop. Every disagreement between a diarizer's continuity claim and the embedding cosine
registers as doubt, at 10 Hz, on a recording where who-is-speaking is not actually in doubt.

This is a separate defect from the threshold-scale bug fixed in `a38d5292`. That one made the loop's
*gates* read doubt instead of entropy; this one is about how the axis is *composed*.

## The design

**The axis's question changes** to: *how sure are we who is speaking here?*

### Three scored voters

`harvest_speaker_votes` keeps emitting each model's cluster assignment per bucket — `per_speaker_tracks`,
`cluster_active_time`, `label_correspondence_rows`, `speaker_claims_from_votes` and
`build_speaker_identity` all read it — but the axis's **scored** voters become three:

| voter | value | measured (conversation clip) |
|---|---|---|
| `per_speaker_presence` | `max` over the speakers present in the bucket of `speech_presence_uncertainty` (binary entropy of the model share) | mean 0.120, 3 distinct |
| `asr_location` | coverage-weighted mean of `1 - temporal_confidence` over the words reaching the bucket | mean 0.220, 79 distinct |
| `target_activity` | the mask **region's** `uncertainty`, **only where its `state != "target_active"`** | 1.0 over the 25 indeterminate buckets (12% of the clip) |

**`max` over speakers, not mean.** If any speaker's presence in this bucket is contested, attribution
in this bucket is contested; averaging that against a confidently-placed speaker would let one
certain speaker hide doubt about another. Approved explicitly.

**Why `asr_location` belongs here.** Word boundaries are what assign words to a speaker's span, so
not knowing where a word starts is not knowing whose it is. This is the quantity D-27 deliberately
moved *off* the asr axis and onto the word (`onset_confidence` / `offset_confidence` /
`temporal_confidence`) — it has had no consumer since, and this is its consumer. It is also where the
axis's resolution comes from: 79 distinct values against the per-speaker term's 3.

The mask input is the **region** table (`L2/background_mask.parquet`: `region_id`, `start`, `end`,
`state`, `uncertainty`), not the `background_mask` *axis* rows — only the regions carry `state`, and
the gate needs the direction, not just the magnitude. A bucket takes the state and uncertainty of the
region it overlaps most.

**Why `target_activity` is gated on state.** Not knowing whether the target was active is not knowing
whether there is anyone to attribute, so an indeterminate mask region raises attribution doubt. But
applying the mask's uncertainty *everywhere* was measured and rejected: the mask has 14 coarse
regions against 214 buckets, so folding it unconditionally collapses the axis from 80 distinct values
to 35 — the coarse measurement overwrites the fine one. Where the mask confidently reports
`target_active`, the attribution question is live and the mask contributes nothing.

### Null semantics

The axis reports **`None`** — no claim — in two cases:

1. **No speaker present in the bucket** (3 of 214 here). Consistent with `per_speaker_tracks`, which
   gives an absent speaker no row because "an absent row is not the same claim as presence at
   confidence zero".
2. **The mask confidently reports `target_free`.** There is no one to attribute, so there is nothing
   to be sure or unsure about. `0.0` would assert confident attribution where no attribution was
   made. Approved explicitly.

`None` rather than `0.0` has a consequence worth stating: `estimates.control_doubt` returns `None`
for such a row, so `apply_convergence_marks` leaves it `open` rather than marking it converged. That
matches the asr axis's 23 word-free buckets today and is the honest reading — an unmeasured bucket is
not a settled one — but it means a recording that is mostly target-free keeps an open speaker axis.
If that proves noisy in practice the fix is a `not_applicable` status, not a zero.

### Where it lives, and why it is not a derived formula

The three voters go through `fuse.fuse_axis` exactly like every other axis's. That is deliberate:

- they get **measured stability weights** across the raw and enhanced passes, like every other signal;
- `uncertainty` stays the normalised entropy, `confidence` the weighted mean, `triage_score` the
  aggregator fold — no new column and no fourth definition of "uncertainty";
- the composition is expressed as *which voters exist*, which is this codebase's idiom (`axes.AXES`
  declares properties; `fuse_axis` is the one fold). A derived formula bolted beside the fold would
  be a second fold, and removing second folds is what the module has been doing.

So the change is confined to what `harvest_speaker_votes` emits as scored entries, plus the two new
inputs it needs threaded in.

### What stops being a scored voter

`same_label_uncertainty`, `change_inconsistency_uncertainty`, `__cross_diar_label_disagreement__`
and `__overlap_count__`. They are the 0.666.

They are *not* deleted wholesale: `identity_repair`'s change-point detection and
`speaker_claims_from_votes` read the harvest, and the cluster assignments they read stay. What stops
is emitting these four as scored vote entries that `fuse.per_signal_uncertainty` folds into the axis.

Anything that becomes genuinely uncalled by this change gets deleted with its tests rather than left
as a recorded-but-unread block — the `__pairwise_phoneme_distances__` lesson.

## Consequences

**Expected axis value, and on which scale.** The measured figure is **mean 0.333 with 80 distinct
values, on the doubt scale** — a max fold over the three voters, which is what the default `min`
aggregator produces and therefore what lands in `triage_score`. Composed of a clean per-speaker term
(0.120), real word-localisation doubt (0.220), and 1.0 across the 12% of buckets where target
activity is genuinely indeterminate.

`uncertainty` is the normalised **entropy** of those voters and will read higher than 0.333 by
construction — `H(0.333) = 0.915` for a single value, less once the three disagree and the fold takes
the entropy of their mean. That is not a regression and must not be read as one: it is the same
scale distinction as `a38d5292`, and it is why the loop's gates read `control_doubt` rather than the
entropy column. **The 0.666 → 0.333 comparison in this document is doubt against doubt**; today's
axis has no doubt-scale reading to compare because its voters are the change-detection signals.
The number to hold this design to is the doubt one, and the check is that it tracks the per-speaker
presence.

**The mean does not drop to 0.12.** The `asr_location` term is doing most of the work, and that is by
design: the recognizers place these word boundaries with `temporal_confidence` averaging 0.735, and
that disagreement is real. An attribution measure that ignored it would be reporting confidence it
has not earned.

**A new cross-stage dependency.** The speaker axis now depends on the ASR stage. With
`stages.asr: false` the `asr_location` voter is absent and the axis degrades to the other two rather
than failing; `contributing_signals` records which voters contributed, so a reader can tell. Stated
here because it is a real coupling between stages that did not exist before.

**Re-measurement.** Every downstream number keyed to the speaker axis changes: region proposal,
convergence, residual mass, the disagreements ranking, the LS bins. `theta_low` / `theta_high` were
not tuned against this composition. `CACHE_SCHEMA_VERSION` must be bumped.

## Testing

- **Unit, on the composition:** each voter is emitted with the right value from a synthetic harvest;
  `max` over speakers rather than mean (a fixture with one contested and one certain speaker);
  `target_activity` absent where the mask is `target_active` and present where indeterminate;
  `None` where no speaker is present and where the mask is confidently `target_free`.
- **Unit, on the regression this fixes:** a bucket every diar model agrees on must read low even when
  the previous bucket held a different speaker — the change-detection framing scored that high.
- **Resolution guard:** the axis must keep more distinct values than the per-speaker term alone,
  pinning that the coarse mask does not overwrite the fine word evidence (the D/E rejection).
- **Live, both clips, cache cleared:** the axis tracks the per-speaker presence; the run still
  converges; the transcript's words are unchanged; `verify_grid_unification.py` still passes.

## Not in scope

- The **2-vs-5 speaker-count disagreement** between `speakers.json` and `diarization.json` /
  `transcript.json`, from `identity_repair` re-clustering against a cosine threshold without
  consulting the count posterior. Recorded in `grid-unification-results.md`; a separate decision.
- The **cross-axis coupling saturation** exposed by grid unification. Same file, also separate.
- Re-tuning `theta_low` / `theta_high` against the new composition. They need re-measuring against
  ground truth, which is the evaluation harness's job, not this change's.

---

## Outcome (implemented 2026-08-06)

Commits `0c3bfd8f` → Task 6. Verified with `scripts/verify_grid_unification.py` on both clips,
cache cleared, exit 0.

### The axis now reflects its evidence

`english_conversation_higgs_audio_v2`, per round, speaker axis:

| round | `triage_score` (doubt) | contributing voters |
|---|---|---|
| 0 | **0.2878** | `per_speaker_presence`, `asr_location` |
| 1 | 0.3927 | + `axis::asr`, `axis::speech_presence`, `axis::background_mask` |
| 2 | 0.8222 | + the same three |
| 3–4 | 0.6082 | `per_speaker_presence`, `asr_location` |

**Round 0 is the design, met:** 0.288 against a predicted 0.333, composed of
`per_speaker_presence` mean 0.1196 and `asr_location` mean 0.2228 — it tracks the clean per-speaker
presence (0.1196) plus the real word-location doubt, against 0.666 before. The 48 kHz clip reads
0.62, which is also correct there: its per-speaker presence doubt is genuinely 0.576 on a five-speaker
recording with a multi-modal 1-vs-5 count posterior, so the axis is reflecting contested evidence
rather than manufacturing it.

Both transcripts are byte-identical to their pre-change digests (`ad7dfa13a6971e1a`,
`a033983bab339bf4`), and checks 2–5 pass unchanged.

### Rounds 1–2: the cross-axis coupling saturation

Already documented in `grid-unification-results.md`: the `axis::*` voters join at full weight against
a max-doubt aggregator and inflate every axis. Nothing new, and the speaker axis is now one more place
it is visible. It also means the axis is *not* a function of only the three named terms in those
rounds, which is a deviation from this design and is tracked there.

### Rounds 3–4: `I2_recluster` was overwriting the per-speaker term, and that was a bug

Rounds 3–4 initially settled at 0.6082 rather than returning to round 0's 0.2878. The cause was
`I2_recluster` shadowing the harvest's `per_speaker_presence` vote with a value recomputed over its
own repaired clusters: it emits **5 clusters**, five clusters spread across the sources drop each
share to ~0.2, and `H(0.2) = 0.722`.

**I first described that as the axis correctly surfacing a real disagreement. That was wrong**, on two
counts:

1. **The per-speaker answer is correct for this recording.** It is a two-person conversation; the
   count posterior is 2 at 0.978 and unimodal; `S0`/`S1` carry existence uncertainty 0.0 and 0.022.
   So `I2_recluster`'s 5 clusters are not competing evidence — they are simply wrong, and inflating
   the axis with them reports doubt where the run's own accurate answer has none.
2. **It reintroduced the defect this axis exists to remove.**
   `final/per_speaker_presence.parquet` is built by `build_speech_presence_tracks(speaker_harvest)` —
   from the *harvest*, never from `refined_identity` — so it still read 0.1196 while the axis read
   0.608. A deliverable and the axis describing it must not disagree.

So the overwrite is removed. `I2_recluster`'s product is the refined segmentation it writes; a repair
that contradicts a confident count posterior belongs in the identity deliverables where the
contradiction is visible, not folded into how sure we are who is speaking.

**`I2_recluster` over-segmenting is now the open defect**, not an uncertainty signal. It cuts at 6
change-points (local maxima above `mean + cp_k·std`, `cp_k: 1.0`) and agglomerates the pooled segment
vectors at `recluster_cosine_threshold: 0.45`, and it never consults the count posterior — so on a
clip whose speaker count is known to 0.978 it returns 5. Constraining it to the posterior, or gating
it on agreeing with one, is the next speaker task.

### What the verification caught that the unit tests could not

`contributing_signals` still contained `__cross_diar_label_disagreement__` on a live run while
`speaker_attribution_test.py` asserted it was gone. The tests exercise `harvest_speaker_votes`;
`I2_recluster` is a **second producer** that added its own scored copy after re-clustering — which is
how it propagated the repair into the axis, since `per_speaker_presence` is computed at harvest and is
stale the moment a re-cluster lands. It now recomputes `per_speaker_presence` instead, a faithful
translation because both read the same cluster assignments. `cross_source_disagreement` became
uncalled and is deleted.

Exactly the handoff's landmine — *verify against the pipeline, not against your own fixtures* — and
the reason check `[6]` asserts on the voter names rather than only on the value.
