# Summary — analyze_audio audit

## 1. Five cross-sweep patterns (the actionable unit, not 176 individual rows)

`candidates/deduped.md`'s "Cross-sweep patterns" section is the headline of this audit: a defect
class recurring across N locations is one fix, not N. In order of how many locations each spans:

### Pattern 1 — an unfitted numeric threshold gates a binary downstream verdict (7 locations)
`F-139` (`fuse.py`, `settled_below=0.35`), `F-140` (`fuse.py`, `unsettled_above=0.6`), `F-143`
(`support.py`, `MIN_LOW_FRACTION=0.02`, numbers the docstring itself disowns), `F-144`
(`speaker_identity.py`, `multimodal_threshold=0.15`), `F-145` (`speaker_identity.py`,
`_SUPPORTED_THRESHOLD=0.5`), `F-151/REFUTED` (`noise_floor.py`, `recorder_margin_db=3.0`),
`F-149` (`global_summary.py`, PESQ/STOI/SI-SDR ramp bounds contradicting their own docstring).
Every one gates settled-or-not / multimodal-or-not / supported-or-not / recorder-or-perceptual /
acceptable-or-not on a boundary with no citation and no `run_config.py` override. This is the
same shape as the already-fixed HNR ramp (2→10 dB, median voiced speech at 8.12 dB — the reason
this pattern was looked for at all). **One sweep** — require every threshold in this module
family to carry a citation or a `data/`-derivation file, per this repo's own stated convention —
closes 6 of these 7 at once (`F-151` is REFUTED: a derivation does exist, just uncomfortably far
from the constant it backs).

### Pattern 2 — `0.0`/`1.0`/absence reads as "not measured" (5 locations)
`F-83` (`belief.py`, aleatoric floor defaults to `0.0` on every lookup miss), `F-146`
(`identity_binding.py`, `binding_agreement=0.0` for `eligible==0`, indistinguishable from
unanimous rejection), `F-147` (`speaker.py`, `speech_presence_confidence=1.0` when 3 of 4
diarizers crashed), `F-150` (`disagreements.py`, `high_uncertainty_rate=0.0` on total harvest
failure — reads as a dramatic improvement against a stored 0.9941 baseline), `F-156`
(`identity_repair.py`, `boundary_confidence` fabricates `0.5`, outranking a genuine weak
detection). Same class the codebase's own `l1-post-processing-register.md` already tracks for
other signals (silhouette-as-probability, SNR floor saturating to 0.0): "did we measure this"
and "what did we measure" share one field in five different places, independently.

### Pattern 3 — an adult/clean-corpus anchor applied with no population or task conditioning (4+ locations)
`F-164` (0.30/0.70/0.5-0.55 cosine-similarity family, VoxCeleb-derived), `F-169` (`degradation.py`'s
25 dB SNR / 30 dB C50 anchor, no `task_type` parameter), `F-172` (`global_summary.py`'s
solo-speaker assumption penalizing correct caregiver-mediated recordings), `F-171` (`compute.py`'s
2.0s ECAPA-minimum embedding window). `background_mask.py` already has task-aware machinery a few
files away (`F-168`'s vocabulary gap notwithstanding); the degradation/global-summary/embedding
path never adopted it.

### Pattern 4 — the same rationale narrated near-verbatim in 3+ files, no canonical home (~7 copies)
"Silhouette coefficient is not a probability" (`F-24`, `F-52`); "5 speakers vs 2 diarizers
reported" (`F-22`, echoed in `influence.py`/`support.py`/`reliability.py`, `F-35`); "three id
namespaces once rendered as S0" (`F-23`, duplicated in `harmonize.py`/`clustering.py`, told again
in `F-28`'s `joint.py` section); `asr.py`'s own triple-telling of its `consensus_words` removal
history (`F-16`/`F-18`/`F-19`). Each pair stays a separate finding in the register (different
files, independent migration/deletion), but `prose-migration.md` stages the canonical text once
per cluster so the eventual write-once pass doesn't recreate the duplication one level up.

### Pattern 5 — the hardcoded `"ast"`/`"yamnet"` trust ladder recurs at four call sites (`F-163`)
The largest single repeated-location finding in the merge — until refutation narrowed it.
`compute.py:890-1009` does implement a real priority ladder; `stages.py:763-786`,
`sound_sources.py:193-197`, and `background_mask.py:520-535` all **aggregate** both classifiers
(`max()`/union) rather than replicate a ladder. `F-163` is REFUTED as a 4x-repeated bug, but
survives, narrower, as a real one-site limitation — see §4 (model pluggability) below.

## 2. Layer measurements (`measure.py`) and what they imply for reuse

```
orchestration :  20 files    5423 code   (imports senselab.audio.tasks / senselab.utils.tasks)
computation   :  61 files   11721 code   (does not)
prose         :              10888       (8796 docstring + 2092 comment)
prose:code    : 0.64 : 1
```

The package is 68% computation files by line count (11,721 of 17,144 code lines) — logic that
imports nothing from the shared task libraries and could run on plain arrays/dicts, versus 5,423
lines that actually orchestrate calls into `senselab.audio.tasks`/`senselab.utils.tasks`. Sweep B
independently found and demonstrated 5 concrete instances of this being real, unclaimed reuse debt
rather than an artifact of the split: `F-142` (`level.py` — BS.1770/EBU-Tech-3342 loudness/gain,
target `quality_control`), `F-148` (`statistics.py` — generic uncertainty statistics, target a
new `senselab/utils/tasks/uncertainty.py`, which is also what `project_mc_dropout_optional`
already wants), `F-152` (`acoustic.py`'s `lufs_track`/`level_above_floor_track`, target
`features_extraction/loudness.py`), `F-153` (`occupancy.py`'s interval algebra, target
`senselab/utils/tasks/`), `F-160` (`identity_repair.py`'s clustering/L2-norm/change-point-trajectory
primitives, target `senselab/utils/tasks/` or `speaker_embeddings/`). All five are `demonstrated`,
`low` severity (no functional defect — placement only) and `irrelevant` to the triage graph today.
The 11,721-line computation layer is therefore not evidence the package is bloated; it is evidence
that a workflow-specific package accumulated general-purpose numerical code with nowhere shared to
put it, which is exactly the shape `senselab-core` (this session's future-package memory) is meant
to fix.

## 3. Prose ratio and Sweep A's own restates-code fraction

Prose (docstrings + comments) runs 0.64 lines for every 1 line of code — well above what a
"self-documenting code" convention alone would produce, consistent with this repo's own stated
practice of recording *why*, not *what*, next to non-obvious choices. Sweep A's own
classification of the 138 raw prose findings it examined (not a sample of all 10,888 prose
lines — a targeted sweep over rationale-bearing modules) split three ways: 8 `stale-or-false`
(5.8%), 98 `rationale-to-migrate` (71.0%, after two reclassifications moved `A-111`/`A-122` here
from `restates-code`), and 32 `restates-code` (23.2%) — pure documentation redundancy with no
migration destination, safe to delete once the code it restates is read directly. The 71%
majority is the load-bearing case for this repo's prose convention: most of what Sweep A looked
at is real design rationale that would be lost, not noise that would be improved by deleting it.

## 4. What the register implies for the graph

Of the 44 gated-and-survived findings, **27 are `consumed`** by at least one of the seven planned
triage outputs (human-review flag, transcript, speaker count, PII, recording quality, task match,
trim) — the graph cannot treat these as someone else's problem; a fix or an explicit acceptance
is needed before the triage numbers can be trusted. **3 are `routed-around`**: `F-146`
(`binding_agreement`, currently unwired into any production call path — dead until someone wires
it in), `F-157` and `F-158` (both shape internal adaptive-loop search priority/budget, not a
published value directly — the graph reads whatever the loop converges to, regardless of how
efficiently it got there). **14 are `irrelevant`**: the 8 prose `stale-or-false` docstring
mismatches (no runtime effect) and 6 dev-experience/placement findings (5 promotion-candidates,
plus `F-161`'s `types.py` stdlib shadow, which only bites a process whose cwd is inside the
package — not how the triage graph invokes it). Practical reading: the graph can be built today
treating the `routed-around`/`irrelevant` 17 as out of scope, but the 27 `consumed` findings —
19 of them `high` severity (13 already `demonstrated`, 6 `verified-latent` pending a pediatric
corpus) — are live defects in exactly the signals the graph is supposed to read, and the
register's `failure` column names the concrete corrupted value for each.

One `high`-severity item deserves its own line rather than folding into a pattern: refutation
found, while checking `F-144`, that `default.yaml`'s `multimodal_threshold: 0.15` is never
threaded into the `speaker_count_posterior(claims, gates=gates)` call site (`speaker_identity.py:524`
omits `multimodal_threshold=`) — the YAML entry is decorative, and the hardcoded literal always
governs. This was not raised by any sweep; it surfaced only because the refuter checked the
config wiring while verifying `F-144`. It is the same class of defect as `F-162`
(`policy=` dead on every `fuse_consensus_words` call path, including the U1 live-re-ASR route) —
a config knob that reads as live but is not.

## 5. What this implies for each deferred concern

**Model pluggability.** `F-163` (the 4-call-site hardcoded ladder claim) is REFUTED as filed — 3
of the 4 sites aggregate both classifiers via `max()`/union rather than replicate a ladder — but
this *weakens, not eliminates*, the underlying worry: `compute.py:890-1009` does implement a real
YAMNet-over-AST veto, `PassPlan` only ever exposes `ast_model`/`yamnet_model` as named fields
(unlike the ASR/diarization/embedding lists elsewhere in the package, which take arbitrary
model-id lists through config), and `F-170` names the concrete population harm this veto enables
(child-voice-as-Music/Singing) with no automatic mitigation. Add `F-162` (SURVIVED-CORRECTED:
`policy=` dead on every `fuse_consensus_words` call site, including the one live intervention
route) and `F-144`'s decorative `multimodal_threshold` (§4, above) — three independent places
where a value that looks configurable is not. Scope: `F-163` (refuted, narrowed), `F-170`,
`F-162`, `F-144`.

**The 1-vs-more speaker decision.** `F-172` (demonstrated): `single_speaker_uncertainty` scores
any 2+-speaker recording as maximally (`1.0`) noncompliant with no task-aware exception, so a
correct caregiver-mediated recording is indistinguishable from an unexpected-second-speaker
failure. `F-167` (verified-latent): the speaker-count posterior carries no population signal at
all, so a child-specific diarizer failure (merge or over-split) surfaces as ordinary posterior
disagreement. `F-164` (verified-latent, SURVIVED-CORRECTED): the fixed cosine-similarity family
deciding same-vs-different speaker is adult-derived, with the narrower surviving claim on
`cluster_cosine_threshold=0.5`/`merge_threshold=0.55` specifically (the per-embedder empirical
floors already correct for `same_speaker_floor`/`diff_speaker_floor`). Scope: `F-172`, `F-167`,
`F-164`, and `F-144` (the multimodal-vs-unimodal threshold that gates whether the loop treats the
count as converged in the first place).

**New speech-extraction models.** The 5 promotion-candidates (`F-142`, `F-148`, `F-152`, `F-153`,
`F-160`, §2 above) are the direct answer to "how hard would it be to add a new backend that needs
this math": today, none of these generic numerical routines live anywhere a new backend module
could import them without reaching into `audio_analysis`'s internals. `F-163`'s narrowed
model-pluggability gap (above) is the second half — the two-classifier assumption baked into
`_speech_window_mask` at `compute.py:890` specifically. Scope: `F-142`, `F-148`, `F-152`, `F-153`,
`F-160`, `F-163`.

**Lifespan validation (pediatric/non-adult populations).** This is where nearly the entire
`verified-latent` tier lives: `F-164`, `F-166`, `F-167`, `F-168`, `F-169`, `F-170`, `F-171`,
`F-173`, `F-174`, `F-175`, `F-176` (11 of 11 verified-latent findings), plus `F-165` and `F-172`,
which reached `demonstrated` without a pediatric corpus because a synthetic bucket/call built
the exact precondition (zero ASR word coverage; a 2-speaker posterior) directly, rather than
needing a recording. The 11 `verified-latent` findings each name, in `verdicts/reproduction.md`,
a specific corpus, metric, and comparison an experiment would need — none could be executed here
because no child/pediatric/non-verbal-vocalization corpus is available in this environment, which
is the expected (not evasive) outcome for this class. Read together, the pattern is structural,
not a handful of independent oversights: ASR (`F-166`), diarization/embeddings (`F-164`, `F-167`,
`F-170`, `F-171`, `F-173`), overlap detection
(`F-174`), the missed-speech correction mechanism (`F-175`), the background/task vocabulary
(`F-168`), the quality anchor (`F-169`), the headline compliance score (`F-172`), and the
certifying WER metric itself (`F-176`) all carry the same unstated assumption — validated on
adult, largely conversational/read speech — independently, at every layer of the pipeline. Scope:
`F-164`, `F-165`, `F-166`, `F-167`, `F-168`, `F-169`, `F-170`, `F-171`, `F-172`, `F-173`, `F-174`,
`F-175`, `F-176` (13 findings — all of Sweep D minus none, since none of Sweep D was refuted).
