# Audio hints and target-speaker embedding estimation

**Status:** design approved 2026-08-14. Branch `feat/audio-hints-speaker-embedding`, cut from
`alpha` at `9e78187a`.

## Goal

Two things, neither of which changes any existing output:

1. **Declared hints on an `Audio`** — what the recording may contain, how many speakers the
   acquisition protocol targeted, the environment, the expected text for a read task, and a
   target-speaker embedding with its provenance.
2. **An estimator** that takes a set of files that *may* contain a speaker and returns one
   embedding for that speaker, together with statistics describing the distribution it came
   from.

## Scope, explicitly

**In:** the `AudioHints` type and its attachment to `Audio`; a vector-distribution descriptor
in `utils/`; an opt-in contamination-rejection selector beside it;
`estimate_speaker_embedding_from_audios` in the speaker-embeddings task; two primitives promoted
down a layer; two defect fixes in `audio_analysis/embeddings.py`.

**Out, by decision:**

- **No consumer wiring.** Hints are declared and carried. No task changes behaviour because a
  hint is present — not diarization speaker bounds, not enhancement, not `analyze_audio`. How a
  hint should inform a decision is a decision, and it gets its own derivation in a later change.
- **No prompt matcher.** `expected_speech` makes matching *possible*; WER, alignment and
  skipped-sentence detection are consumers. That work would want designing alongside PR #542's
  unlanded task-compliance half rather than piecemeal.
- **No dataset provider.** PR #543's `AudioPlus`, `MetadataProvider`, `NullMetadataProvider`
  and `audio/metadata/b2ai.py` do not come across. Its metadata is *resolved by lookup*; hints
  are *asserted by a declarer*. Different provenance, different trust, different failure modes,
  and hints must work with no provider at all.
- **No artifact I/O, no `workflows/speaker_profile/` package, no `compare.py` /
  `score_voice_groups`.** The last belongs with the per-speaker uncertainty work, as #543 says
  itself.
- **No clustering inside the descriptor.** Contamination rejection exists, but as an opt-in
  component layered above it. See "Contamination rejection" below.
- **No multi-model embedding.** One model per call. #543 defaulted to ECAPA+ResNet because
  `analyze_audio` scores both per-window; decoupled from that consumer, a second model is
  unused cost. `provenance.model_id` records which model produced a vector.
- **No `min`/`max` range on `targeted_speaker_count`.** Plausible but unrequested; it goes in
  `metadata` until a caller needs it, rather than shipping parallel fields now.

## Component 1 — the hints layer

New file `src/senselab/audio/data_structures/audio_hints.py`. `Audio` gains one field:
`hints: AudioHints | None = None`.

```python
class ExpectedSpeech(BaseModel):
    text: str | None          # verbatim prompt the speaker was asked to produce
    prompt_id: str | None     # id in an external reference set
    reference: str | None     # which reference set (name / version / URI)


class SpeakerEmbeddingProvenance(BaseModel):
    model_id: str
    model_commit_sha: str | None     # resolved 40-hex, or None with unresolved_reason set
    unresolved_reason: str | None
    method: str                      # e.g. "spherical_mean"
    source_files: list[str]
    window_s: float
    hop_s: float
    n_windows_used: int
    n_windows_dropped: int
    created_at: str | None           # ISO-8601, caller-stamped


class TargetSpeakerEmbedding(BaseModel):
    vector: list[float]              # unit-norm; see "Geometry"
    provenance: SpeakerEmbeddingProvenance
    distribution: EmbeddingDistribution | None


class AudioHints(BaseModel):
    may_contain: list[str] = []
    targeted_speaker_count: int | None = None
    environment: str | None = None
    expected_speech: list[ExpectedSpeech] = []
    target_speaker: TargetSpeakerEmbedding | None = None
    metadata: dict[str, Any] = {}
```

### Decisions inside this component

**`may_contain`, not `contains`.** A hint is a declaration of intent or expectation, never a
measurement. The name keeps that epistemic status so nothing downstream reads it as ground
truth.

**`hints=None` by default, not an empty `AudioHints`.** An empty object makes "nobody declared
anything" indistinguishable from "declared nothing". That is the same collapse as reading a
`None` confidence as `0.0`, which `pii_detection` documents at length; absent stays absent.

**Open tags carry a suggested vocabulary in `doc.md`, not an enum.** `may_contain` and
`environment` are open string tags. A `Literal` enum would be a taxonomy nobody has fitted, and
every corpus that did not fit would need the enum edited. Numbers stay numbers:
`targeted_speaker_count` is an `int`.

**`expected_speech` is an ordered list, not one string.** A single file often holds several
sentences (the Harvard/IEEE sets are read as sequences). Concatenating them destroys the
boundaries a matcher needs to say *which* sentence was skipped or reordered — "did they read
all five" is a different question from "how close was the whole thing".

**`prompt_id` + `reference` alongside `text`.** The verbatim text makes a hint self-contained;
the id and reference set let it be traced to an external prompt corpus without vendoring that
corpus. PR #542's `scripts/task_reference.json` (797 task definitions, 720 Harvard/IEEE
sentences) is the motivating case: it is available elsewhere and deliberately not committed
here, so this hint is its injection point.

**Provenance records a resolved commit SHA, never a ref.** Reusing the pinning machinery merged
in #550. When resolution genuinely fails, `model_commit_sha` is `None` *and*
`unresolved_reason` says why — recording a ref in that field would be provenance that is
confidently wrong, which #550 established is worse than recording none.

**Adding a field to `Audio` does not disturb cache keys.** Verified:
`utils/tasks/cached_inference.py` hashes `audio.waveform`, not the model. This is not a
backwards-compatibility concern — cache and schema compatibility are explicitly not constraints
during alpha, and invalidation is free. It is a *correctness* one: a hint that nothing consumes
must not change what a computation returns, or "carried only" would be false. A test asserts it.

## Component 2 — the distribution descriptor

New file `src/senselab/utils/tasks/embedding_distribution.py`, beside `cosine_similarity.py`.

```python
def describe_embedding_distribution(
    vectors: Sequence[Sequence[float]] | torch.Tensor | np.ndarray,
    file_ids: Sequence[str] | None = None,
    aggregator: str = "spherical_mean",   # | "trimmed_mean" | "medoid"
    window_s: float | None = None,
    hop_s: float | None = None,
) -> tuple[list[float], EmbeddingDistribution]
```

It describes **one** set of vectors: a centroid and statistics. It does not cluster, does not
select a subset, and returns no verdict. It lives in `utils/` because it is a function over
vectors, not over audio, and anything holding embeddings can use it.

### Geometry

**L2-normalise every vector first.** Not a convenience. ECAPA is trained with an angular-margin
objective and scored by cosine, so identity information is angular by construction, while the
embedding norm covaries with window energy and how much speech fills the window — a cough, or
0.4 s of speech in a 2.0 s window, gets a systematically different norm. Any unnormalised
statistic mixes that loudness/occupancy nuisance into what a reader takes for speaker
dispersion. Rows with zero norm are dropped and counted.

After normalisation, cosine and Euclidean are the **same** geometry:
`‖x−y‖² = 2(1−cos θ)`, a strictly monotone reparametrisation. So the common "Euclidean is
unusable at high d" objection does not apply to any rank- or neighbour-based quantity; what
differs is *mean-of-distances* statistics, where Jensen makes `s_euc < s_cos` for identical
geometry. Working scale is cosine. Where a true metric is required (medoid, any linkage) use
the geodesic `θ = arccos(clip(cos, −1, 1))`; `cos` is not a metric and neither is `1 − cos`.

**Centroid: spherical mean by default**, `ĉ = S/‖S‖` with `S = Σ xᵢ`. It is the von
Mises–Fisher MLE direction and its error shrinks as `O(n^{−1/2})`, where the medoid *is* one
real window and so inherits that window's phonetic content with `O(1)` error that does not
shrink with `n`. An arithmetic mean of unnormalised vectors is rejected outright: it weights
each window by loudness, so a loud cough outvotes a quiet target utterance.

Because there is no clustering to reject contamination, `aggregator` is exposed as a tool
parameter (`trimmed_mean`, `medoid`), and the block always reports the cosine between the mean
and both alternatives. A gap between them tells a caller the estimate is contamination
sensitive — a robustness statement carrying no threshold and no verdict.

### Reference scales — all analytic, none fitted

| statistic | null | value at d=192 |
| --- | --- | --- |
| sd of pairwise cosines | `1/√d` | 0.0722 |
| mean resultant length `R̄` | `1/√n` | — |
| participation ratio | `d·n/(d+n)` | — |
| separability AUC | `0.5` (exact) | — |

Every reported field is either bounded on an interpretable scale or paired with one of these,
and `dim`, `n_windows_scored` and `n_effective` are in the block so a consumer can recompute
all four and check ours.

**The counter-intuitive one, which belongs in the docstring:** a *small* sd of cosines is not
evidence of a coherent single speaker. At d=192 independent random vectors give sd ≈ 0.072, so
an observed 0.05 is *below* the random-vector null. sd therefore never appears as a headline
dispersion figure — only next to `1/√d`.

### Fields

```
geometry          metric="cosine", l2_normalised=True, dim, distance="angular",
                  centroid_rule=<aggregator>
counts            n_vectors_total, n_scored, n_zero_norm_dropped, n_files,
                  vectors_per_file{file→n}, window_s, hop_s, n_effective
nulls             cos_sd_null, rbar_null, participation_ratio_null, auc_null
cos_to_centroid_loo   {min, q05, q25, q50, q75, q95}
rbar              mean resultant length of the scored set
within_file       per file: {n_vectors, rbar, cos_to_own_centroid_q05, _q50}
cross_file        per file: cos(file_centroid, pooled_centroid);
                  file_centroid_pairwise_cos {q05, q50, q95}
file_effect       auc_same_file_vs_diff_file, permutation_quantile,
                  permutation_block_len, n_permutations, guard_band_s
spectrum          participation_ratio, pc1_share_centred, eigenvalue_shares_top5
centroid_robustness   cos_mean_vs_trimmed10, cos_mean_vs_medoid,
                      leave_one_file_out_cos{file→cos}
```

**Within-file and cross-file stay strictly separate.** The prior measurement on this exact
pipeline is within-file cosine stability 0.984 against cross-file 0.891 — essentially the whole
error budget is cross-file. A single pooled dispersion would average those into one
uninterpretable number and destroy the most informative split known about this data.

**Leave-one-file-out centroid stability is the most valuable field.** `cos(ĉ_full, ĉ_−f)` per
file is a jackknife along exactly that cross-file axis, and it answers a consumer's real
question — "is this centroid an artefact of one file?" — more directly than any dispersion
number. Cost is `n_files` matmuls and no model calls.

**Cosine-to-centroid is leave-one-out.** Scoring a vector against a centroid it helped define
is optimistically biased. Closed form, one matmul, no loop:
`cos_loo(i) = xᵢ·(S − xᵢ) / ‖S − xᵢ‖`.

**`n_effective = total_windowed_duration / window_s`,** which is ≈ `n/2` at a 2.0 s window with
1.0 s hop. Overlapping windows mean any null whose width scales as `n^{−1/2}` is ~√2
overconfident; reporting `n_effective` lets a consumer discount correctly instead of us
pretending independence.

**The file-effect permutation uses block shuffling.** Statistic: between-file share of angular
variance, `η² = 1 − Σ_f n_f(1 − R̄_f²) / [n(1 − R̄²)]`. Reference: shuffle `file_id` across
vectors keeping per-file counts, `B = 1000` times, report the observed value's permutation
quantile. Windows are autocorrelated (50% overlap plus prosodic continuity), so a naive
per-vector shuffle destroys dependence the observed statistic retains and the p-value comes out
anti-conservative; blocks of `L = ceil(window_s/hop_s)` vectors fix that. `L`, `B` and
`guard_band_s` are stored so the number is auditable, and the docstring states it is a
diagnostic on a dependent sample, not an exact test.

**The guard band is what keeps `file_effect` from measuring the hop size.** At a 2.0 s window
on a 1.0 s hop, two temporally adjacent windows share half their audio, so a same-file pair drawn
from neighbouring windows is a near-duplicate. Left in, same-file similarity is inflated toward
1.0 for *any* input and the AUC reports the windowing configuration rather than a speaker
effect. Same-file pairs closer together than `guard_band_s` (default `window_s`) are therefore
excluded from both sides of the comparison, and the value used is reported.

The AUC in this block is specifically **same-file pairs versus different-file pairs**; with no
clustering there is no within-cluster/between-cluster contrast to compute. Its exact null is
0.5 under exchangeability, which is what makes it readable with no fitted scale. Cost of the
whole permutation is `B` label shuffles and zero model calls.

### Deliberately absent, and why

- **Any silhouette coefficient.** `silhouette(metric="cosine")` and `silhouette(metric="euclidean")` return
  different numbers for *identical* geometry on unit vectors, so a threshold on silhouette is a
  threshold on a parameterisation choice. It is also a property of a chosen partition, not of
  the data. Replaced by the Mann–Whitney separability AUC, which is rank-based and therefore
  parameterisation-invariant, with an exact null of 0.5.
- **Any k-NN purity measure.** Two independent failures. Hubness is severe at this dimension
  (measured skew of k-occurrence 3.29 at d=192, with 64 of 1000 points appearing in no
  neighbour list), so neighbour counts are biased by something unrelated to speaker identity.
  And at 50% overlap the 1-NN of a window is almost always the temporally adjacent window, so a
  same-file purity statistic would read ≈1.0 for any input — it would be measuring the hop size.
- **Any intrinsic-dimensionality estimate.** Two-NN's `r₁` becomes the distance to a
  near-duplicate window, so `d̂` is driven down by the hop size. Same artefact, and ID estimators
  disagree by 2–3× on the same embedding set with no consensus for speaker embeddings.
- **vMF concentration `κ`.** `κ̂ = R̄(d − R̄²)/(1 − R̄²)` is a deterministic function of `R̄` and
  `d`, both already in the block, so it stores no new information; it is unbounded as `R̄→1`, so
  it is not interpretable on its own scale; and vMF assumes isotropic concentration, which
  embedding spaces violate (participation ratio well below `d`). Recoverable in one line by
  anyone who wants it.
- **`sd` of cosines as a standalone number, any `n_speakers` field, any boolean or `p_*`
  field, anything computed on unnormalised vectors.**

## Component 3 — the estimator

`src/senselab/audio/tasks/speaker_embeddings/` gains:

```python
def estimate_speaker_embedding_from_audios(
    audios: list[Audio],
    model: SenselabModel | None = None,     # default ECAPA
    device: DeviceType | None = None,
    window_s: float = 2.0,
    hop_s: float = 1.0,
    aggregator: str = "spherical_mean",
    reject_contamination: bool = False,
) -> TargetSpeakerEmbedding
```

Window each file → embed each window → L2-normalise → `describe_embedding_distribution` over
the pooled set with `file_ids` → wrap the centroid, provenance and distribution.

`2.0 / 1.0` are PR #543's **measured** defaults for a profile centroid, not picks: their grid
gave cross-file centroid stability 0.890 and cross-subject separation 0.168, against 0.331 for
a 0.5/0.25 grid carrying four times the windows. Their finding that cross-file variation is the
entire error budget is also why the descriptor splits within-file from cross-file.

### Contamination rejection — opt-in, layered above the descriptor

PR #543 rejected contamination by clustering and keeping the dominant cluster, measured as 24 of
32 non-speech recordings dropped with the centroid preserved at cos ≥ 0.99 in 7 of 8 subjects.

**That clustering is cross-file, once, over the pooled window set** — its own section header
reads "Cross-file dominant-cluster aggregation". Per file it does speech gating only (dropping
non-speech windows); every surviving window from every file then pools into one set, and the
clustering runs on that. So a recording is excluded as a *side effect* of a cross-file decision,
never by a per-file judgement — which is why #543 also has to report `per_file_dominant`
(`file_id → windows_in_dominant_cluster`), since a pooled clustering decision is opaque without
it.

This design keeps that capability but moves it **out of the descriptor and behind a flag**,
because selecting a dominant cluster is a decision and `describe_embedding_distribution`'s job is
to describe what it is handed.

```python
def select_dominant_vectors(
    vectors,
    file_ids: Sequence[str] | None = None,
    linkage: str = "average",
    cut_theta: float | None = None,
    min_file_share: float | None = None,
) -> DominantSelection
```

`estimate_speaker_embedding_from_audios` gains `reject_contamination: bool = False`. With the
flag on it selects first and hands only the retained subset to the descriptor, so the descriptor
is unchanged and its statistics then describe the *kept* set.

**Off by default.** The default path stays pure description, and the flag is the only place a
decision enters this component.

**Agglomerative hierarchical clustering, average linkage, angular distance** — not #543's
spectral, for three reasons. It is deterministic, where `SpectralClustering(assign_labels=
"kmeans")` is stochastic and #543 pins `random_state=0` / `n_init=5`, which hides that variance
rather than removing it. k-means-family clustering carries an equal-size bias and will split one
speaker's prosodic halves before isolating an 8% intruder — the exact failure
`_merge_close_clusters` was written to undo after the fact. And AHC turns "choose *k*" into a
merge-height profile, which is reportable rather than decided.

**The cut gets no numeric default, because that is where an unfitted literal would enter.**
`cut_theta=None` means the cut is derived from the data by a stated rule: the largest gap in the
merge-height sequence. That is a *rule*, not a fitted constant, which is what keeps it inside
this repository's prohibition on literals nobody measured. A caller who disagrees passes an
explicit `cut_theta`. Either way the value used and the full merge profile are recorded in
`DominantSelection.rule_used`, so the choice is auditable and reversible.

**#543's four literals are deliberately not carried over** — `coherent_silhouette_threshold=0.10`,
`merge_threshold=0.55`, `min_cluster_fraction=0.10`, `n_clusters_max=6`. Each is justified against
`analyze_audio`'s bucket structure rather than against this use, and the first gates on silhouette,
which this design rejects as parameterisation-dependent (see "Deliberately absent").

**Dominant selection is by file-balanced share**, with raw window share reported alongside. The
target is the speaker present in *most files*, not the one occupying most seconds — otherwise a
single ten-minute off-target recording outvotes three one-minute target recordings. When the two
shares disagree that is diagnostic in itself, so both are reported, along with the runner-up's
share and `cos(dominant_centroid, runner_up_centroid)`: "0.52 / 0.46 at cos 0.31" and
"0.94 / 0.05 at cos 0.88" are different situations and both must stay legible.

**Rejection is recorded, never silent.** `provenance.method` becomes
`"spherical_mean+dominant_cluster"`, `provenance.n_windows_dropped` reflects what rejection
removed, and `DominantSelection` names which files lost windows and how many.

**The honest cost.** With the flag on, the returned statistics describe a curated set, so they
will look better than the raw input warranted. That is inherent to rejection rather than a flaw
in the reporting — and it is why the flag defaults off, why `n_windows_dropped` sits in
provenance rather than buried, and why `cos_mean_vs_trimmed10` and the leave-one-file-out cosines
remain the check on whether rejection actually cleaned anything up.

**With the flag off**, contamination is still *visible* without being acted on: the descriptor
reports per-file centroids, their cosine to the pooled centroid, and leave-one-file-out
stability, so the caller can curate the input set and re-run. `cluster_pass_speakers` stays
untouched in the workflow where it belongs; the AHC step here is a generic vector-level
primitive, not a second copy of that workflow's calibrated clustering.

### Layering: two primitives move down

`_window_starts` and `extract_per_window_embeddings` already exist — in
`audio/workflows/audio_analysis/embeddings.py`. A task importing from a workflow inverts the
dependency direction (`workflows` compose `tasks`, not the reverse), so both are **promoted
into `audio/tasks/speaker_embeddings/`** and the workflow imports them from there. PR #543 set
this precedent by promoting the shared speech gate out of `compute.py`'s private copy, which
also cut that file by 105 lines.

**What is not promoted:** `cluster_pass_speakers`, `silhouette_voice_score` and
`calibrate_cosine_uncertainty`. The first is saturated with workflow semantics — a
YAMNet/AST-derived speech mask, a `"NOISE"` label, `p_voice`, per-pass calibration bands, and
four literals justified against that workflow's bucket structure. The second returns
`p_voice = 0.5·(silhouette + 1)`, which is the silhouette-as-probability defect named in
`CLAUDE.md`. The third is a thresholded piecewise-linear map whose `(0.30, 0.70)` defaults its
own sibling docstring records as having "sat below *every* distance the embedding produced".
Importing any of them would import a mislabelled semantic.

## Defect fixes folded in

Both in `audio/workflows/audio_analysis/embeddings.py`, both pre-existing on `alpha`:

1. **`p_voice = 0.5·(silhouette + 1)` at lines ~484 and ~531.** A silhouette coefficient read
   as a probability — the exact class `CLAUDE.md` calls out, and the L1 post-processing register
   closed its item 12 by removing the *consumer* rather than the computation. The register
   records the cost: `embedding_silhouette` produced 0.4022–0.4996 doubt across 214 buckets with
   stdev 0.0227, and earned the highest fusion weight of fifteen signals precisely because it
   was near-constant; removing it moved published presence doubt from 0.0682 to 0.0385. The
   computation is removed or renamed to what it measures, with no `p_`-prefixed name and no
   probability claim.
2. **A stale module docstring** claiming "default 2.0 s with 1.0 s hop" while the signature is
   `window_s=1.0, hop_s=0.5` and the docstring's own "Why 1.0 s / 0.5 s defaults" section agrees
   with the signature. The stale line is corrected.

**The `1.0/0.5` signature default is deliberately not changed, and cache invalidation is not
the reason.** Cache and schema compatibility are explicitly not constraints during alpha, and
`CLAUDE.md` says invalidation is free. The reason is a *quality* one, written into that module's
own docstring: ECAPA/ResNet embeddings are noisier below 1 s, and the module knowingly trades
embedding precision for temporal resolution because a 1.0 s window on a 0.5 s hop "gives one
embedding per 0.5 s bucket, eliminating the same-window dedup that previously dropped half of
consecutive same-cluster comparisons". The value is tied to the workflow's 0.5 s bucket grid,
and `calibrate_cosine_uncertainty` is calibrated against that noisier same-speaker baseline.
Moving it to 2.0/1.0 would halve temporal resolution and reintroduce the dedup it was chosen to
remove.

So the stale artefact is the docstring's opening line, which claims "default 2.0 s with 1.0 s
hop" while both the signature and the module's own "Why 1.0 s / 0.5 s defaults" section say
otherwise. That line is corrected.

The result is two window settings for two purposes, each with its own derivation: **2.0/1.0 for
a profile centroid** (measured: cross-file stability 0.890, cross-subject separation 0.168) and
**1.0/0.5 for detection** (measured against the 0.5 s bucket grid). This mirrors #543's own
`PROFILE_WINDOW_S` / `DETECT_WINDOW_S` split rather than collapsing them.

## Cherry-picked from PR #543

- the measured `2.0 / 1.0` window/hop values **with their derivation table**, not just the
  numbers;
- the contamination evidence (24 of 32 non-speech recordings dropped, centroid preserved at
  cos ≥ 0.99 in 7 of 8 subjects) recorded as the measurement this design consciously trades
  away;
- the promote-a-shared-primitive precedent.

**Not taken:** `scripts/gen_synthetic_test_audio.py` (638 lines). #543 needs synthetic *audio*
because it tests end to end. This design's aggregation and statistics are functions of vectors,
so tests inject synthetic **embeddings** — deterministic, instant, no model download, and
consistent with the standing rule against constructing an unmocked `HFModel` in a test.

## Testing

1. **Statistics** — known vector configurations with analytically known answers: one tight
   cone, two well-separated cones, one cone plus outliers, and uniform-random vectors at
   d=192 whose `sd`, `R̄` and participation ratio must land on the analytic nulls
   (`1/√d`, `1/√n`, `d·n/(d+n)`). No model involved.
2. **Null correctness** — each `nulls` field equals its closed form for the reported `dim`,
   `n_scored`. This is what keeps the block free of fitted numbers.
3. **LOO correctness** — `cos_to_centroid_loo` matches a naive recomputed-centroid loop on a
   small input, proving the closed form.
4. **Within/cross-file separation** — a construction where within-file dispersion is tiny and
   cross-file dispersion large must produce visibly different `within_file` and `cross_file`
   figures, and a `file_effect` AUC far from 0.5.
5. **Block permutation** — with `file_ids` shuffled at random, the permutation quantile is
   uniform-ish; with a real file effect it is extreme. Asserted as a direction, not a
   threshold.
6. **Robustness diagnostics** — adding contaminant vectors opens a measurable gap in
   `cos_mean_vs_trimmed10` and `cos_mean_vs_medoid`, and drops the contaminated file's
   `leave_one_file_out_cos`.
7. **Hints** — pydantic validation; `hints=None` distinguishable from an empty `AudioHints`;
   provenance either carries a 40-hex SHA or sets `unresolved_reason`.
8. **Cache-key invariance** — two `Audio` objects with the same waveform and different `hints`
   produce the same cache key. The design depends on this.
9. **A layering guard** — an AST test asserting nothing under `audio/tasks/` imports from
   `audio/workflows/`. The repo already uses this pattern (`hf_load_coverage_test`,
   `revision_pinning_guard_test`), and since two primitives move down a layer, a guard is what
   stops them drifting back.
10. **Contamination rejection** — with `reject_contamination=True` on a set where one file is a
    different speaker, that file's windows are dropped, `provenance.method` records
    `+dominant_cluster`, `n_windows_dropped` is non-zero, and `DominantSelection` names the file.
    With the flag off, the same input keeps every window and the contamination is instead visible
    in `cross_file` / `leave_one_file_out_cos`.
11. **The cut rule is a rule, not a literal** — `cut_theta=None` derives from the largest
    merge-height gap and records the value used; an explicit `cut_theta` overrides it and is
    recorded verbatim. A test asserts no numeric cut default exists in the signature.
12. **AHC determinism** — the same input produces byte-identical `kept_indices` and
    `merge_heights` across repeated calls, with no seed passed.
13. **File-balanced vs raw share** — a set with one long off-target file and several short
    on-target files selects the on-target group, and both shares are reported so the
    disagreement is visible.
14. **One skip-gated integration test** that runs only when an embedding model is already
    cached locally.

## File structure

| Path | Responsibility | Action |
| --- | --- | --- |
| `src/senselab/audio/data_structures/audio_hints.py` | `AudioHints` and its nested types | Create |
| `src/senselab/audio/data_structures/audio.py` | gains `hints: AudioHints \| None` | Modify |
| `src/senselab/utils/tasks/embedding_distribution.py` | `describe_embedding_distribution`, `EmbeddingDistribution`, `select_dominant_vectors`, `DominantSelection` | Create |
| `src/senselab/audio/tasks/speaker_embeddings/windowing.py` | `_window_starts`, `extract_per_window_embeddings`, promoted | Create |
| `src/senselab/audio/tasks/speaker_embeddings/api.py` | gains `estimate_speaker_embedding_from_audios` | Modify |
| `src/senselab/audio/tasks/speaker_embeddings/doc.md` | suggested hint vocabulary; estimator contract | Create |
| `src/senselab/audio/workflows/audio_analysis/embeddings.py` | import promoted primitives; two defect fixes | Modify |
| `src/tests/utils/embedding_distribution_test.py` | statistics, nulls, LOO, permutation | Create |
| `src/tests/audio/data_structures/audio_hints_test.py` | hint validation, cache-key invariance | Create |
| `src/tests/audio/tasks/speaker_embeddings_estimate_test.py` | estimator, robustness diagnostics | Create |
| `src/tests/audio/tasks/task_layer_guard_test.py` | AST guard: `tasks` must not import `workflows` | Create |

## Success criteria

- An `Audio` can carry every hint in the request, and a hint's presence changes no existing
  output and no cache key.
- `estimate_speaker_embedding_from_audios` returns a unit-norm centroid plus a statistics block
  in which every field is bounded on an interpretable scale or paired with an analytic null.
- No field in the block is a verdict, a boolean, a probability, or a thresholded label.
- Contamination rejection is off by default; when on, what it removed is recorded in provenance
  and in `DominantSelection`, never silent.
- No numeric cut threshold exists as a code default; the cut is caller-supplied or derived by a
  stated rule, and the value used is always reported.
- Within-file and cross-file dispersion are separately readable.
- Nothing under `audio/tasks/` imports from `audio/workflows/`.
- The two `embeddings.py` defects are gone, and the `1.0/0.5` detection default is unchanged.
