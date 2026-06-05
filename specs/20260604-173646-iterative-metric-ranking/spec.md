# Feature Specification: Iterative Metric-Driven Ranking

**Feature Branch**: `20260604-173646-iterative-metric-ranking`
**Created**: 2026-06-04
**Status**: Draft
**Input**: User description: "Given metrics that might describe the quality of audio, or combined metrics of different signals that might tell us how well ASR did or with what confidence there is only one speaker in an audio, we want to create functionality that can create a rank ordering of all of the audios and/or the segments of the audios such that the metric/quality being assessed clearly changes as one moves through the ranking. The rankings relative to close by don't have to be exactly right but something in the top 25% should be differentiable from the bottom 25%. Additionally, this system should allow for iterative processes to update these metrics by spot checking files, annotating them as necessary, and updating the metric to account for the new information. The system should track how files move about the ranking when the metric changes."

## Clarifications

### Session 2026-06-04

- Q: Given imperfect metrics, how should top/bottom bands and boundary crossings be treated? → A: Bands are a configurable coarse lens (default 20%); movement tracking reports ordinal/rank shift — most meaningful at the region level — rather than exact per-item boundary-crossing accounting. The exact-crossing verification criterion is dropped.
- Q: What is the primary measure of a "good" ranking (and what does assisted recalibration optimize)? → A: Primary = rank agreement between the metric order and quality annotations (a rank-correlation measure); secondary = top/bottom-band separation as a coarse check. Assisted recalibration maximizes rank agreement. The driving purpose is to place a confidence threshold/cut point along the ranking: items above it are treated as confidently good (release-ready, auto-accepted) and the rest are routed to human review — applied to checks such as PII presence, multiple-speaker presence, and general audio quality when releasing a dataset.
- Q: What scale do spot-check annotations use? → A: A small fixed ordinal scale by default (good / acceptable / poor), with a numeric score optionally allowed for finer judgments.
- Q: What corpus size should a single ranking run handle comfortably? → A: Up to ~100k items per run (full re-rank per metric version), without committing to streaming/distributed scale.
- Q: How are repeat/conflicting annotations on the same item resolved? → A: Latest-wins — the newest annotation for an item is the active one used for evaluation and recalibration; superseded annotations are retained (not deleted) for history/provenance.
- Q: What data sensitivity does the ranking system itself hold? → A: Only derived signal values, item identifiers, and quality labels/notes — never raw audio, transcripts, or extracted PII content. PII is consumed solely as a numeric indicator signal; any inspection of raw media happens in the separate human-review step, outside this system.
- Q: What is the default behavior when a required signal is missing? → A: Missing signal ⇒ item is unscorable by default; a metric definition may opt a signal into an explicit per-signal fallback (neutral contribution or fixed fill). Unscorable items are never placed in a triage auto-accept region — they count as auto-fail and are routed to human review.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Rank a corpus by a quality metric (Priority: P1)

A researcher has a corpus of audio recordings (and/or segments within them) and a metric that combines one or more signals — for example an audio-quality score, an ASR-confidence indicator, or a single-speaker-confidence indicator. They want a single ordered list of items, best-to-worst (or worst-to-best) on that metric, so they can triage: review the most problematic items first, or select the cleanest items for downstream use. The ordering does not need to be precise between neighboring items, but the top of the list must be clearly better on the assessed quality than the bottom.

**Why this priority**: This is the core value of the feature and a usable product on its own. Without a ranking, none of the iterative refinement or movement-tracking has anything to operate on.

**Independent Test**: Provide a corpus and a metric definition; confirm the system emits a complete ordered ranking of the chosen unit (files or segments), and that a quality-separation check confirms the top band is differentiable from the bottom band (default band = 20%).

**Acceptance Scenarios**:

1. **Given** a corpus of audio items and a metric that references one or more available signals, **When** the researcher requests a ranking at the file level, **Then** the system produces an ordered list covering every item, each with its metric score and rank position.
2. **Given** the same corpus, **When** the researcher requests a ranking at the segment level instead, **Then** the system produces an ordered list over segments using the same metric, without re-specifying the metric from scratch.
3. **Given** a produced ranking and a set of quality annotations, **When** the researcher runs the separation check, **Then** the system reports whether annotated items in the top band of the ranking score better on the reference quality than annotated items in the bottom band, and by what margin.
4. **Given** a metric that references a signal missing for some items, **When** the ranking is produced, **Then** those items are not silently dropped — they are surfaced as unscorable (or scored by an explicit fallback) and reported separately.

---

### User Story 2 - Iteratively refine the metric via spot-checking and annotation (Priority: P2)

After an initial ranking, the researcher spot-checks a sample of items (often drawn from across the ranking — top, middle, bottom, and disagreement zones), listens/inspects, and records a ground-truth quality judgment for each (a score or an ordinal label such as good / acceptable / poor). They then update the metric to better agree with what they observed — either by manually revising the metric definition (which signals it combines, weights, thresholds) or by asking the system to assist by recalibrating the metric against the accumulated annotations. The system re-scores and re-ranks the corpus under the revised metric, producing a new metric version.

**Why this priority**: This is what makes the metric improve over time and converge toward the researcher's notion of quality. It depends on P1 producing a ranking to spot-check.

**Independent Test**: Starting from an existing ranking, annotate a handful of items with quality judgments, update the metric (both by manual revision and by assisted recalibration), and confirm a new, versioned ranking is produced and that its separation check is no worse than before on the annotated set.

**Acceptance Scenarios**:

1. **Given** an existing ranking, **When** the researcher selects items to spot-check, **Then** the system supports sampling strategies including across-the-ranking coverage (e.g. spread over rank regions, including near a candidate triage threshold) and can present items so they can be reviewed and annotated.
2. **Given** an item under review, **When** the researcher records a quality judgment (numeric score or ordinal label), **Then** the annotation is stored, attributed to the item (and metric version under which it was reviewed), and retained for future metric updates.
3. **Given** accumulated annotations, **When** the researcher manually revises the metric definition and applies it, **Then** the system re-scores all items and produces a new metric version with its own ranking.
4. **Given** accumulated annotations, **When** the researcher requests assisted recalibration, **Then** the system proposes an updated metric (e.g. adjusted weights/thresholds) that increases agreement with the annotations, and the researcher can accept it to create a new metric version.
5. **Given** any metric update, **When** the new version is created, **Then** prior versions, their rankings, and their annotations remain intact and retrievable (nothing is overwritten).

---

### User Story 3 - Track how items move when the metric changes (Priority: P3)

When a new metric version produces a new ranking, the researcher wants to understand the impact: which items moved up, which moved down, which shifted into or out of the top/bottom bands (a coarse lens), and whether items they annotated ended up where they expected. This tells them whether the update did what they intended and helps catch regressions.

**Why this priority**: It builds confidence in the iterative loop and guards against silent regressions, but the loop is still usable without it (the researcher could eyeball two rankings). It depends on P2 producing at least two versions to compare.

**Independent Test**: With two metric versions present, request a movement report and confirm it correctly identifies, for every item, its old and new rank, the ordinal shift, and band-region movement (into/out of the top/bottom band).

**Acceptance Scenarios**:

1. **Given** two metric versions over the same corpus, **When** the researcher requests a movement comparison, **Then** the system reports, per item, the rank under each version and the change (positions and/or percentile).
2. **Given** the same comparison, **When** the researcher asks where items shifted, **Then** the system reports ordinal/rank shift per item and summarizes movement at the band-region level (e.g. items that moved into or out of the top/bottom band), treating band membership as a coarse lens rather than an exact ledger.
3. **Given** items that were annotated, **When** the movement report is generated, **Then** annotated items are highlighted so the researcher can see whether each landed consistently with its recorded quality judgment.
4. **Given** an item present in one version but unscorable in the other, **When** the comparison runs, **Then** that item is reported as added/removed/became-unscorable rather than being omitted.

---

### Edge Cases

- **Ties in metric score**: Many items may share an identical metric score. The system must order them deterministically (stable ordering) so that rankings are reproducible and movement between versions is not spurious noise from tie-breaking.
- **Missing or partial signals**: An item may lack one or more signals a metric needs. The system must not silently drop it; by default it is reported as unscorable, unless the metric definition opts the missing signal into an explicit per-signal fallback. Unscorable items are excluded from rank/band positions but always reported, and on triage they count as auto-fail (routed to human review, never auto-accepted).
- **Single-item / empty corpus**: Ranking and separation checks must behave sensibly (or report "insufficient data") rather than error.
- **Too few annotations for recalibration**: Assisted recalibration must refuse or warn when there are not enough annotations (or not enough spread across quality levels) to fit a meaningful update, rather than over-fit to a handful of points.
- **Band thresholds with small N**: When the corpus is too small for the configured band fraction to be meaningful, the separation check must report that the top/bottom-band differentiability target cannot be evaluated.
- **Annotation conflicts**: The same item may be annotated more than once (same or different reviewers/versions) with differing judgments. The system applies **latest-wins**: the newest annotation is the active one used for evaluation and recalibration; superseded annotations are retained (not deleted) for history/provenance.
- **Mixed units**: A file-level ranking and a segment-level ranking are distinct ranking spaces; the system must not conflate ranks or movement across units.
- **Metric references an unavailable signal**: If a revised metric references a signal that does not exist for the corpus, the system must reject the metric definition with a clear message rather than producing an all-unscorable ranking.

## Requirements *(mandatory)*

### Functional Requirements

#### Ranking

- **FR-001**: System MUST accept a metric definition that combines one or more available signals (e.g. audio-quality measures, ASR-confidence indicators, single-speaker-confidence indicators, PII-presence indicators) into a single comparable score per item. The intended quality concerns include PII presence, multiple-speaker presence, and general audio quality for dataset release.
- **FR-002**: System MUST produce a complete rank ordering over the chosen unit, assigning every scorable item a metric score and a rank position.
- **FR-003**: System MUST allow the ranking unit to be selected per run as either whole audio files or segments within audios, using the same metric definition for either unit.
- **FR-004**: System MUST order items so that the assessed quality changes monotonically along the ranking at the macro scale, even though neighboring items need not be in exactly correct relative order.
- **FR-005**: System MUST break ties deterministically so that repeated runs over unchanged inputs yield identical rankings.
- **FR-006**: System MUST handle items with missing or partial signals by reporting them as **unscorable by default**, never silently dropping them. A metric definition MAY opt an individual signal into an explicit, documented per-signal fallback (neutral contribution or fixed fill value); only signals without such a fallback render an item unscorable.
- **FR-007**: System MUST persist each ranking together with the metric version that produced it and the corpus/unit it covers, so it can be retrieved and compared later.

#### Ranking quality / separation

- **FR-008**: System MUST provide a separation check that, using available quality annotations, quantifies whether top-band items differ from bottom-band items in the assessed quality (band fraction configurable, default 20%), and reports the result against a configurable target.
- **FR-009**: System MUST expose the band boundaries (the band fraction is configurable, default top/bottom 20%) so the researcher can see which items fall in the top and bottom bands, treating these as a coarse lens rather than precise per-item guarantees.
- **FR-010**: System MUST report when the separation check cannot be evaluated (e.g. too few annotations or too small a corpus) rather than reporting a misleading pass/fail.
- **FR-010a**: System MUST report a rank-agreement measure (rank correlation between the metric order and available quality annotations) as the primary indicator of ranking quality, with the top/bottom-band separation as a secondary coarse check.

#### Triage threshold (release vs. human review)

- **FR-010b**: System MUST let the researcher place a threshold (cut point) along the ranking such that items above it are treated as confidently good (release-ready / auto-accepted) and the remainder are routed to human review. Unscorable items MUST NOT fall in the auto-accept region — they count as auto-fail and are always routed to human review.
- **FR-010c**: System MUST report, for a chosen threshold, how the annotated items fall relative to it (e.g. how many annotated-good vs. annotated-poor items sit above vs. below) so the researcher can judge the confidence of the auto-accept region and adjust the threshold.

#### Spot-checking & annotation

- **FR-011**: System MUST let the researcher select items to spot-check, including a sampling strategy that spreads selection across the ranking (e.g. across rank regions, near a candidate triage threshold, and/or disagreement zones).
- **FR-012**: System MUST let the researcher record, per reviewed item, a ground-truth quality judgment using a small fixed ordinal scale by default (good / acceptable / poor), with a numeric score optionally allowed for finer judgments.
- **FR-013**: System MUST store each annotation with provenance: the item it applies to, the metric version under which the item was reviewed, and reviewer/time. Repeat annotations on the same item are resolved by **latest-wins** — exactly one annotation per item is active; superseded ones are retained for history (never deleted).
- **FR-014**: System MUST retain annotations across metric updates so the full annotation history is available to every later metric version.

#### Metric update

- **FR-015**: System MUST support manual metric revision — the researcher changes which signals are combined, their weights, and/or thresholds — and re-scores/re-ranks the corpus under the revised definition.
- **FR-016**: System MUST support assisted recalibration — proposing an updated metric (e.g. adjusted weights/thresholds) that increases rank agreement with accumulated annotations — which the researcher can review and accept.
- **FR-017**: System MUST refuse or clearly warn on assisted recalibration when annotations are insufficient or insufficiently varied to support a meaningful update.
- **FR-018**: System MUST treat every metric change as a new, immutable version, preserving prior metric definitions, their rankings, and their annotations.
- **FR-019**: System MUST reject a metric definition that references a signal unavailable for the target corpus, with a clear explanation.

#### Movement tracking

- **FR-020**: System MUST compute, for any two metric versions over the same corpus and unit, each item's rank (and/or percentile) under each version and the ordinal shift between them.
- **FR-021**: System MUST summarize movement at the band-region level — e.g. which items moved into or out of the configurable top/bottom band between versions — reported as a coarse, approximate signal rather than an exactly-audited per-item boundary ledger.
- **FR-022**: System MUST highlight annotated items in movement reports so the researcher can judge whether each landed consistently with its recorded quality judgment.
- **FR-023**: System MUST account for items that are scorable in one version but not the other (added / removed / became-unscorable) rather than omitting them from the comparison.

### Key Entities *(include if feature involves data)*

- **Corpus**: The set of audio items under consideration, at a chosen unit (files or segments). For segments, includes the segment's source audio and its time span.
- **Signal**: A single measurable input attached to an item (e.g. an audio-quality measure, an ASR-confidence indicator, a single-speaker-confidence indicator). Has a name and a value; may be absent for some items.
- **Metric**: A versioned, named definition that combines one or more signals into a single comparable score, with the rule for combination (signals used, weights, thresholds) and a direction (higher = better or worse).
- **Metric Version**: An immutable snapshot of a metric definition, plus how it was produced (manual revision or assisted recalibration) and from which prior version it derived.
- **Ranking**: An ordered list of items for a given metric version, corpus, and unit; each entry has the item, its metric score, its rank position, and its band (top / middle / bottom under the configurable band fraction).
- **Annotation**: A ground-truth quality judgment recorded by a reviewer for one item — by default a fixed ordinal level (good / acceptable / poor), optionally a numeric score — with provenance (item, metric version under review, reviewer/time, resolution status).
- **Movement Report**: A comparison of two rankings over the same corpus/unit, giving per-item ordinal shift, band-region movement (into/out of the top/bottom band, as a coarse lens), annotation highlights, and added/removed/unscorable items.
- **Separation Result**: The outcome of the ranking-quality check for a ranking against available annotations — the primary rank-agreement measure plus the secondary top/bottom-band differentiability — including the measured values, the targets, and whether each was evaluable.
- **Triage Threshold**: A chosen cut point along a ranking that partitions items into a confidently-good (release-ready / auto-accepted) region above it and a human-review region below it, together with the reported counts of annotated-good vs. annotated-poor items on each side.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: For a ranking judged "good," a reviewer comparing a randomly drawn top-band item against a randomly drawn bottom-band item (default band = 20%) agrees that the top-band item is higher quality in at least 80% of such pairings.
- **SC-002**: Every scorable item in the corpus appears exactly once in the ranking with a rank and metric score (100% coverage), and unscorable items are reported separately (0% silently dropped).
- **SC-003**: Re-running a ranking over unchanged inputs and metric version reproduces the identical order (100% reproducibility), including tie-breaking.
- **SC-004**: After a metric update informed by annotations, the rank-agreement measure (and secondary band separation) on the annotated set is no worse than the prior version's, and the researcher can confirm this from the reported diagnostic before adopting the new version.
- **SC-005**: A researcher can complete one full refinement cycle — spot-check a sample, annotate them, update the metric, and obtain a new ranking — without losing any prior version, ranking, or annotation.
- **SC-006**: For any two versions, the movement report accounts for 100% of items (moved, unchanged, added, removed, or became-unscorable) with correct old/new ranks.
- **SC-007**: For any two versions, per-item ordinal shift is computed for 100% of items shared between them, and the band-region movement summary is consistent with those shifts (no item reported as moving into a band it did not, at the coarse band granularity).
- **SC-008**: Ranking quality is reported as a rank-agreement measure against quality annotations; a ranking is acceptable when this measure indicates positive ordinal agreement and the top-band-vs-bottom-band separation target (SC-001) is met.
- **SC-009**: For a chosen triage threshold, the researcher can read how many annotated-good and annotated-poor items fall above vs. below it, enabling them to pick a cut point above which the auto-accept (release-ready) region contains a known, acceptably low share of poor items.

## Assumptions

- **Signals are provided as inputs.** Computing the underlying signals (audio-quality measures, ASR confidence, single-speaker confidence, etc.) is upstream of this feature; this feature consumes already-available signal values per item. New signal types can be added as inputs without changing the ranking machinery.
- **Ranking unit is chosen per run.** A given ranking run targets either whole files or segments; file-level and segment-level rankings are independent ranking spaces and are never conflated. Segment definitions (source audio + time span), when segment-level ranking is requested, are supplied as input.
- **Metric updates use both mechanisms.** The system supports manual metric revision and assisted recalibration from annotations; assisted recalibration is advisory (proposes; the researcher accepts).
- **Annotations are quality judgments.** Spot-check annotations capture a ground-truth quality value (numeric score) or an ordinal label; per-signal corrections and free-form notes are out of scope for v1 (an annotation may carry a short note, but the metric update is driven by the quality judgment).
- **"Differentiable" is evaluated against annotations.** The top-band-vs-bottom-band differentiability target (default band = 20%) is assessed using the quality annotations available; with no annotations, separation cannot be asserted and the system reports it as not-yet-evaluable.
- **Bands are a coarse lens, not a precise ledger.** Top/bottom bands (default 20%, configurable) are a coarse evaluation and triage device; given that the metrics under consideration are imperfect, the system tracks ordinal/rank shift and summarizes movement at the band-region level rather than asserting exact per-item boundary correctness.
- **Low-sensitivity store.** The ranking system holds only derived signal values, item identifiers, and quality labels/notes — never raw audio, transcripts, or extracted PII content. PII enters solely as a numeric indicator signal; raw-media inspection happens in the separate human-review step, outside this system. Item identifiers and annotation notes are expected to be free of PII content.
- **Single-corpus scope.** A ranking and its versions operate over one defined corpus at a time; cross-corpus comparison is out of scope for v1.
- **Reproducibility over scale.** A single ranking run is expected to handle up to ~100k items, where a full re-rank per metric version is acceptable (re-ranking is recomputed rather than incrementally patched); streaming/distributed scale is out of scope for v1. Deterministic, reproducible output is prioritized over incremental-update performance.
