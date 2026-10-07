# Statistical review of the register and the code

A Bayesian statistician's review of `analyze_audio`, reading the code rather than taking the
register at its word. The register's authors were software auditors; this pass asks the questions
they were not equipped to ask. Every number below is a measured output from running the live
modules, not an estimate.

Register findings are cited as `F-n`; new findings introduced here are `N1`–`N10`.

## The frame

`analyze_audio` is an **iterative refiner**: it harvests per-window votes from several models,
fuses them into per-axis doubt over a time grid, and an adaptive loop proposes interventions where
doubt is high, re-running to refine. Five axes, an L1-measures / L2-decides discipline.

## Q1 — Is the uncertainty arithmetic coherent?

**Partly. The estimators are sound; their inputs are not.**

`statistics.py` is correct and well argued — `confidence` renormalises weights, `variability`
refuses `n=1`, `epistemic_uncertainty` is textbook BALD with the Jensen clamp. The defect is what
is fed into them.

`fuse.py:426` builds each signal's "distribution" as `{"unsettled": v, "settled": 1-v}`, where `v`
is a doubt *score* — `1-exp(avg_logprob)`, `1-p_voice`, `same_label_uncertainty`. None of these is
`P(the axis is unsettled)`. Worse, **"settled" is not an observable**, so this entropy can never be
calibrated against anything. The correct construction sits eight lines away in the same function
(`fuse.py:433-436`, `_source_distributions`): point masses over the actual latent — which speaker —
used only as a fallback when exactly one signal is present. The default path uses the pseudo-outcome
space.

**Independence is assumed everywhere and is false.** Four diarizers (pyannote 3.1 / community /
sortformer / a derived clusterer) share AMI + VoxConverse + DIHARD training data; three ASR backends
share LibriSpeech / CommonVoice lineage. `influence.py`'s `derivation_gate` is the only correlation
control, a single hand-set `0.4` for one binary category — it cannot express "these two diarizers
are 0.9 correlated." This inflates confidence.

**Counts used where a likelihood ratio is needed.** `speaker_identity.py:158-175` does
`mass[count] += weight; p = mass/total` and calls the result a posterior. Measured: 4 unanimous
sources → P = 1.000; **20** unanimous sources → P = 1.000. 2-vs-2 → 0.5/0.5; 10-vs-10 → 0.5/0.5.
A posterior's width is invariant to sample size, which no posterior is. Same shape at
`speaker.py:640` (`share`) and `disagreements.py:152`.

## Q2 — `speaker.py:636` (F-147): right about the bug, wrong about the fix

Changing the denominator to enrolled-models is *also* wrong. A diarizer that crashes on hard audio
is **MNAR** — the failure is caused by the same acoustic difficulty the statistic is estimating —
so neither `k/n_surviving` nor `k/n_enrolled` is estimable from the data.

Correct treatment is three parts:

1. Carry `n_observed` and `n_enrolled` as columns, so the reader can see the shrinkage at all.
2. Replace the plug-in proportion with a posterior — Beta(½+k, ½+n−k) under Jeffreys. This matters
   because `_binary_entropy(share)` at `speaker.py:648` is a plug-in entropy of an MLE and is
   structurally **0.0 at n=1 regardless of data**: 1/1 and 4/4 are byte-identical *even with a
   corrected denominator*. That is the same defect one level down, and the register does not have it.
3. For the MNAR part, publish the bracket over all completions `[k/N, (k+m)/N]` rather than a point.
   No single number answers this question.

Miller–Madow (`−(k−1)/2n`) is the cheap partial fix if a point estimate is mandatory.

## Q3 — The adaptive loop as inference

**Not a Bayesian update; a re-fold. That is defensible. The stopping rule is not.**

`belief.reaggregate_bucket:810` recomputes the whole fold from the current active vote set rather
than multiplying in a likelihood, and `add_vote:492-498` shadows a same-source file-scope vote with
a region-scope one. Together these avoid sequential double-counting, and this design is preferable
to a fake sequential Bayes.

But there is an **acknowledged** double-count the register missed. `belief.py:526-530` states
outright that the corroborating evidence used to size a claimant's attenuation *also* votes in the
same fold, "so weighting a claimant by it does pull that fold toward the evidence a second time."
The docstring calls this bounded by the floor and the trigger gate — that bounds the *multiplier*,
not the *bias*. Evidence E both moves the posterior and shrinks E's rival: E enters twice.

**Convergence** (`convergence.py:78`): `stalled = (prev_doubt − last_doubt) < epsilon` with
`epsilon=0.05` and `theta_low=0.33` from `default.yaml:379-381`, annotated *"matches labelstudio
HIGH_THRESHOLD"* — **the run's certification threshold is a UI colour bin.** It is a one-sample
delta on a noisy statistic with no variance estimate, so the loop cannot distinguish "improvement
below ε" from "noise above ε", and it scores a *worsening* as stalled → `irreducible`. A fixed point
on a heuristic, not a stopping rule. F-159 is a subset of this.

## Q4 — Thresholds: fit, calibrate, decide, or eliminate

| Threshold | Verdict |
| --- | --- |
| `aggregators.py:86` `disagreement_weighted` (F-141) | **Eliminate.** It computes `mean(u)·max(u)`; the real dispersion statistics are already in the same row. |
| `fuse.py:559` `settled_below=0.35`, `fuse.py:831` `unsettled_above=0.6` (F-139/140) | **Eliminate the binary.** Both gate "offer this bucket to a hook." Rank by continuous doubt, take top-k under the budget the loop already tracks. The cut is a resource constraint, not an inference. |
| `speaker_identity.py:121` `multimodal_threshold=0.15` (F-144) | **Eliminate — it is not scale-free.** Measured: 3-vs-1 → P=0.25 → multimodal; 6-vs-1 → P=0.143 → *unimodal*; 7-vs-1 → 0.125 → unimodal. It encodes "one dissenter out of ≤6" and flips when an operator adds a diarizer, with no change in the audio. Report the posterior's entropy or a credible set. |
| `speaker_identity.py:60` `_SUPPORTED_THRESHOLD=0.5` (F-145) | **Caller-supplied, with a stated loss function.** Missing a real off-target speaker costs far more than reviewing a spurious one; ship `source_support` continuous and let the caller cut at `c_FN/(c_FN+c_FP)`. |
| `global_summary.py:208` PESQ/STOI/SI-SDR ramps (F-149) | **Fit from labelled data.** These are proxies for a human usability judgement; the anchors must come from regressing the intrusive metric on annotated usability. |
| `support.py:276` `MIN_LOW_FRACTION=0.02` (F-143) | **Remove and replace** — see N6. |

## Q5 — New findings, not in the register

**N1 — a low-reliability signal *increases* confidence.** `aggregators.py:67` builds
`values = [u_i · w_i]`, scaling doubt toward certainty, and unlike `fuse.py:422` it never
renormalises by `Σw`. Measured: `mean` over one signal at doubt 0.80 → **0.800**; add a second
signal at the same doubt but reliability 0.05 → **0.420**. Adding a source measured to be
untrustworthy halves published doubt. Two quantities in the same row treat the same weights under
contradictory algebra. Masked under the default `min`; appears whenever config selects
`mean`/`harmonic_mean`, both of which `AGGREGATORS` offers. **High, consumed by every axis.**

**N2 — direction-only votes are entered as zero doubt and dominate the convergence gate.**
`fuse.py:146` correctly identifies voters that assert without scoring (both diarizers, plus
Canary-Qwen / Qwen3-ASR / CrisperWhisper, which expose `avg_logprob=None`) and then assigns
`out[name] = 0.0`. The justification analogises to `per_source_voice` mapping such a vote to
`p=1.0`, but that is a *direction* probability, not a *precision*. Measured on one bucket: Whisper
alone at doubt 0.699 → `control_doubt` **0.699** (open, above `theta_high`); add four direction-only
voters → **0.140**, below `theta_low=0.33`, so `convergence.py:58` marks it **converged**. On the
shipped default ASR set the majority of presence voters are direction-only, so presence buckets
converge by voter *composition*. `epistemic_uncertainty` simultaneously rises 0.000 → 0.407 — the
fold reports new reducible disagreement contributed by voters that scored nothing. **High, consumed
by the human-review flag and the speaker count.**

**N3 — the headline source weight is a 20× step function of a two-sample coin flip.**
`speaker_identity.py:441` + `:395-398`: `perturbation_uncertainty` is normalised entropy over answers
across passes, and a run has exactly two (`perturbations.py:57-64`, raw + speech_enhanced). Measured:
agree → **0.0**, differ → **1.0**. Via `effective_weight` (floor `MIN_EVIDENCE_WEIGHT=0.05`) a
source's weight is therefore `1.0·support` or `0.05·support` — nothing between. A diarizer reporting
2 speakers on raw and 3 on enhanced is silenced entirely; the same diarizer with enhancement disabled
carries full weight. F-167 discusses population but never that the weight is degenerate.

**N4 — the `snr_floor` irreducibility verdict equates incommensurable scales.**
`convergence.py:84` compares `u` (a weighted mean of doubt scores) against `aleatoric_floor`
(`belief.py:1140`, a `max` of anchored *degradation* scores). Both live in [0,1]; no mapping from
degradation to achievable doubt has ever been fitted. Concretely, 15 dB SNR yields
`snr_degradation = (25−15)/(25−5) = 0.50`, so any touched bucket with doubt ≤ 0.55 gets
`irreducible_reason: "snr_floor"` written into `final/convergence.json` — a causal claim produced by
a numeric coincidence. (The `max` fold over floor terms is separately conservative and defensible.)

**N5 — `n_sources` double-counts shared backers.** `fuse.py:463` sums across signals, so when
`speaker_assignment` folds 4 diarizers and `__cross_diar_label_disagreement__` folds the same 4, the
row publishes `n_sources: 8`. The fix exists three lines below (`signal_sources`, the named-set
expansion `measure_axis_overlap` uses) and is not applied to the count.

**N6 — `informative_evidence` is in-sample screening with a powerless null.**
`support.py:301-353` selects the evidence pool by a criterion evaluated **on the same recording** the
pool then weighs — selection on the outcome. The criterion also has no power: `min_low_fraction=0.02`
over ~697 buckets means "dips below 0.20 at least 14 times", which any noisy-but-uninformative signal
clears with probability ≈1. And the nominal *n* is inflated: presence is strongly autocorrelated
across 0.1 s buckets, so the effective sample size is one to two orders of magnitude below 697.
Replace with an out-of-sample screen, or a discrimination statistic whose null comes from a **block**
permutation respecting the autocorrelation.

**N7 — no multiple-comparison exposure anywhere.** ~1070 buckets × 4 active axes ≈ 4280 per-bucket
threshold decisions per run, plus per-word ASR rows, with zero FDR/FWER control (no repo-wide
occurrence of `fdr` / `bonferroni` / `p_value`). `disagreements.py`'s top-N index selects the extreme
tail of thousands of noisy statistics and reports it as "where the run is uncertain" — winner's-curse
selection, so the reported extremes are systematically overstated and regress on re-run.

**N8 — overlapping embedding windows treated as independent.** `compute.py:101-102` uses
`window_s=2.0`, `hop_s=1.0`, and `identity_repair._agglomerative_cosine` /
`change_point_trajectory` then run on adjacent windows sharing one second of audio. This inflates
adjacent cosine toward 1 — attenuating prominence exactly at the boundary the change-point detector
is looking for — and inflates the observation count feeding the derived clusterer's speaker count.
`recluster_cosine_threshold=0.45` was not derived under that overlap. Note `axes.py:60-72` correctly
eliminated this at the axis grid (win == hop == 0.1 s); it survives at the embedding grid.

**N9 — `combined_uncertainty = max(...)` makes F-172 absorbing.** `global_summary.py:398`:
`single_speaker_uncertainty` is exactly 1.0 for any recording with ≥2 speakers (`:302`), and a `max`
over four incommensurable components pins the run's headline at 1.0 for every such recording,
regardless of transcript, quality or PII. The headline has zero discriminating power on multi-speaker
audio — stronger than F-172's "scored as maximally noncompliant".

**N10 — nothing degrades toward a prior as evidence vanishes.** The codebase is rigorous about
`None` ≠ `0.0` (the convention appears in a dozen docstrings and is correct), but has no notion of
shrinkage: `confidence` with one voter equals that voter's certainty exactly (`fuse.py:428-430` names
this "an identity mapping into the axis" and fixes only the *reducibility* half). One confident voter
and forty confident voters produce identical published confidence at the fold, the count posterior,
and `per_speaker_tracks`. A hierarchical prior, or Beta/Dirichlet smoothing with the prior weight
recorded, fixes all three at once. **The single highest-leverage change in this review.**

## Q6 — Register findings to reduce, reframe, or raise

- **F-148** (promote `statistics.py`) — endorse, but with a warning: these are the best functions in
  the package, and moving them invites reuse by callers who will feed them the same pseudo-
  distributions `fuse.py:426` does. Promote with a docstring precondition that inputs be predictive
  distributions over an observable outcome space.
- **F-146** (`binding_agreement=0.0` when `eligible==0`) — **reduce to low**. Same defect as
  F-147/N10; fix by the same one-line change (report `None`, carry `n_eligible`), not tracked
  separately.
- **F-156** (`boundary_confidence` fabricates 0.5) — **reframe**. 0.5 is the only value on that scale
  meaning "no information"; the defect is that a fabricated 0.5 and a measured 0.5 are
  indistinguishable. Fix is `None` plus a `boundary_source` column, not a different number.
- **F-158** (`priority = gain / cost`) — **raise to high**. This is the sharpest finding in the
  register. `priority` is a value-of-information ratio and VOI must be in one currency (expected
  doubt-seconds resolved). `_n_candidates_gain` returning a raw count and `_u2_gain` carrying an
  arbitrary ×10 are not "unnormalised" — they are *different units*.
- **F-151, F-163** — both refutations are correct. ISO 1996-2 minimum measurability genuinely backs
  `recorder_margin_db=3.0`; `max()` and union over two classifiers are legitimate aggregators, not
  ladders.
- Nothing else in the register is statistically wrong. The prose findings and promotion candidates
  are outside statistics. The lifespan cluster (F-164…F-176) is well specified — each names a corpus,
  a metric and a comparison, which is more than most such claims get.

## What the single-pass triage graph should inherit, and avoid

**Inherit:** the L1/L2 split (weights cannot be applied after a fold — correct and load-bearing);
`None` ≠ `0.0` enforced at the schema (`estimates.ESTIMATE_COLUMNS`); the four-quantity row
(`uncertainty` / `epistemic` / `confidence` / `variability`) kept separate rather than collapsed; the
non-overlapping axis grid (`axes.py:60`) and its reasoning; `SnrGate` recording what it withheld
rather than silently shrinking `contributing_passes`; the `statistics.py` estimators themselves.

**Avoid:**

1. Any entropy over `{settled, unsettled}` — define outcome spaces over observables (which speaker,
   which word), or report doubt as a plain score and stop calling it entropy.
2. Vote shares named posteriors. If the graph publishes a speaker count it needs per-model confusion
   likelihoods and a prior, or it must publish the raw votes and refuse to publish a probability.
3. Multiplicative weight-times-doubt (N1) — renormalise, or drop the signal, never both.
4. Zero-doubt imputation for unscored assertions (N2) — carry direction and precision as separate
   fields, so a voter that cannot report precision cannot manufacture certainty.
5. Thresholds inherited from UI bins.
6. Per-bucket binary verdicts at all. Every one of the graph's seven outputs is better served by a
   continuous score plus a caller-supplied, loss-function-documented cut than by an unfitted constant
   inside the pipeline.

**If the graph carries exactly one thing forward from this review, make it N10:** an evidence count
on every published confidence, and shrinkage toward a stated prior when that count is small.
