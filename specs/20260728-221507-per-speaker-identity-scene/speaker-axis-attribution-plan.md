# Speaker-axis attribution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Recompose the speaker axis so it answers "how sure are we who is speaking here?" from three
scored voters — per-speaker presence doubt, ASR word-location doubt, and target-activity doubt —
instead of from per-bucket speaker-change detection.

**Architecture:** A new pure module `attribution.py` holds the three composition functions, each
taking plain data and returning plain numbers so they are unit-testable without a model.
`speaker.harvest_speaker_votes` calls them to emit three scored vote entries and stops emitting the
four change-detection entries as scored. `compute.harvest_pass` folds the consensus words once and
hands them to both the asr and speaker harvests. Everything still goes through `fuse.fuse_axis`, so
there is no second fold and no new column.

**Tech Stack:** Python 3.11–3.12, uv, pytest, mypy, ruff. No new dependencies.

## Global Constraints

- Design doc: `specs/20260728-221507-per-speaker-identity-scene/speaker-axis-attribution-design.md`.
  Read it before starting.
- Every command runs under `uv run` — never bare `python`/`pytest`/`pip`.
- Google-style docstrings, 120-char lines, type hints required. `uv run ruff format` before every
  commit; `uv run ruff check` and `uv run mypy .` must pass.
- Tests live in `src/tests/` mirroring the package, named `*_test.py`.
- `max` over speakers, never mean, for the per-speaker term.
- `None`, never `0.0`, where the mask confidently reports `target_free` or no speaker is present.
- The mask input is the **region** table with `state` (from `pass_summary["background_mask"]["result"]`),
  not the `background_mask` axis rows.
- Do not delete anything that still has a caller. Anything that becomes genuinely uncalled gets
  deleted **with its tests** in Task 5.

---

### Task 1: The three composition functions

**Files:**
- Create: `src/senselab/audio/workflows/audio_analysis/attribution.py`
- Test: `src/tests/audio/workflows/audio_analysis/attribution_test.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `per_speaker_attribution_doubt(clusters: Mapping[str, str], *, silent_cluster_id: str = "SIL") -> float | None`
  - `word_location_doubt(words: Sequence[Mapping[str, Any]], buckets: Sequence[tuple[float, float]]) -> dict[tuple[float, float], float | None]`
  - `target_activity_doubt(mask_regions: Sequence[Mapping[str, Any]], buckets: Sequence[tuple[float, float]]) -> dict[tuple[float, float], tuple[float | None, str | None]]`

- [ ] **Step 1: Write the failing tests**

Create `src/tests/audio/workflows/audio_analysis/attribution_test.py`:

```python
"""The speaker axis's three composition terms, each measurable on its own.

Pure functions over plain data, so the axis's composition can be checked without running a model —
which is the point: the change-detection composition these replace could only be judged from a full
run, and it read 0.666 on a clean two-speaker conversation whose per-speaker presence doubt was 0.168.
"""

from __future__ import annotations

import pytest

from senselab.audio.workflows.audio_analysis.attribution import (
    per_speaker_attribution_doubt,
    target_activity_doubt,
    word_location_doubt,
)

BUCKETS = [(0.0, 0.1), (0.1, 0.2), (0.2, 0.3)]


def test_unanimous_models_carry_no_attribution_doubt() -> None:
    """Every model placing the same speaker here is the case that must read zero."""
    clusters = {"pyannote": "C0", "sortformer": "C0", "emb/ecapa": "C0"}
    assert per_speaker_attribution_doubt(clusters) == pytest.approx(0.0)


def test_an_even_split_between_two_speakers_saturates() -> None:
    """Two models each, on different speakers: a 50/50 share is maximal doubt for both."""
    clusters = {"a": "C0", "b": "C0", "c": "C1", "d": "C1"}
    assert per_speaker_attribution_doubt(clusters) == pytest.approx(1.0)


def test_the_doubt_is_the_max_over_speakers_not_the_mean() -> None:
    """A confidently-placed speaker must not hide doubt about another one.

    Three of four models agree on C0 (share 0.75, H = 0.811); the fourth claims C1 alone
    (share 0.25, H = 0.811). Both are 0.811 here, so use an asymmetric split to make max != mean:
    C0 held by 3 of 5 (H(0.6) = 0.971), C1 by 1 of 5 (H(0.2) = 0.722). Max is 0.971; mean is 0.846.
    """
    clusters = {"a": "C0", "b": "C0", "c": "C0", "d": "C1", "e": "SIL"}
    doubt = per_speaker_attribution_doubt(clusters)
    assert doubt == pytest.approx(0.9710, abs=1e-3), "must be the max over speakers"
    assert doubt != pytest.approx(0.8463, abs=1e-3), "must not be the mean over speakers"


def test_models_reporting_silence_stay_in_the_denominator() -> None:
    """A lone detection among silent models must not read as certain."""
    clusters = {"a": "C0", "b": "SIL", "c": "SIL", "d": "SIL"}
    # share 0.25 -> H(0.25) = 0.8113
    assert per_speaker_attribution_doubt(clusters) == pytest.approx(0.8113, abs=1e-3)


def test_no_speaker_present_is_no_claim() -> None:
    """All models silent, or none reporting: None rather than 0.0."""
    assert per_speaker_attribution_doubt({"a": "SIL", "b": "SIL"}) is None
    assert per_speaker_attribution_doubt({}) is None


def test_word_location_doubt_is_coverage_weighted() -> None:
    """A bucket's location doubt is the coverage-weighted mean over the words reaching it."""
    words = [
        {"start": 0.0, "end": 0.1, "temporal_confidence": 0.5},
        {"start": 0.1, "end": 0.2, "temporal_confidence": 1.0},
    ]
    out = word_location_doubt(words, BUCKETS)
    assert out[(0.0, 0.1)] == pytest.approx(0.5)
    assert out[(0.1, 0.2)] == pytest.approx(0.0)


def test_a_bucket_no_word_reaches_has_no_location_doubt() -> None:
    """None, not 0.0: nothing was said there, so nothing localises it."""
    words = [{"start": 0.0, "end": 0.1, "temporal_confidence": 0.5}]
    assert word_location_doubt(words, BUCKETS)[(0.2, 0.3)] is None


def test_a_word_without_a_temporal_confidence_is_skipped() -> None:
    """An unmeasured word contributes nothing rather than counting as fully confident."""
    words = [{"start": 0.0, "end": 0.1, "temporal_confidence": None}]
    assert word_location_doubt(words, BUCKETS)[(0.0, 0.1)] is None


def test_target_active_contributes_no_doubt() -> None:
    """Where the mask is confident the target is active, the attribution question is simply live."""
    regions = [{"start": 0.0, "end": 0.3, "state": "target_active", "uncertainty": 0.24}]
    out = target_activity_doubt(regions, BUCKETS)
    assert out[(0.0, 0.1)] == (None, "target_active")


def test_indeterminate_contributes_its_uncertainty() -> None:
    """Not knowing whether the target was active is not knowing whether anyone is here."""
    regions = [{"start": 0.0, "end": 0.3, "state": "indeterminate", "uncertainty": 1.0}]
    out = target_activity_doubt(regions, BUCKETS)
    assert out[(0.0, 0.1)] == (pytest.approx(1.0), "indeterminate")


def test_target_free_is_reported_as_a_state_for_the_caller_to_null() -> None:
    """The function reports the state; the caller turns target_free into no claim at all."""
    regions = [{"start": 0.0, "end": 0.3, "state": "target_free", "uncertainty": 0.05}]
    assert target_activity_doubt(regions, BUCKETS)[(0.0, 0.1)][1] == "target_free"


def test_a_bucket_takes_the_region_it_overlaps_most() -> None:
    """Regions are coarse and a bucket can straddle two; the dominant one wins, deterministically."""
    regions = [
        {"start": 0.0, "end": 0.12, "state": "indeterminate", "uncertainty": 1.0},
        {"start": 0.12, "end": 0.3, "state": "target_active", "uncertainty": 0.0},
    ]
    out = target_activity_doubt(regions, BUCKETS)
    assert out[(0.1, 0.2)][1] == "target_active", "0.08 s of target_active beats 0.02 s indeterminate"


def test_a_bucket_no_region_covers_has_no_state() -> None:
    """No mask region here means the mask said nothing, which is not 'target active'."""
    regions = [{"start": 1.0, "end": 2.0, "state": "target_active", "uncertainty": 0.0}]
    assert target_activity_doubt(regions, BUCKETS)[(0.0, 0.1)] == (None, None)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest src/tests/audio/workflows/audio_analysis/attribution_test.py -q`
Expected: collection error, `ModuleNotFoundError: No module named
'senselab.audio.workflows.audio_analysis.attribution'`.

- [ ] **Step 3: Write the implementation**

Create `src/senselab/audio/workflows/audio_analysis/attribution.py`:

```python
"""The speaker axis's composition: how sure are we *who* is speaking here.

The axis used to ask "did the speaker change since the previous bucket?", validated per (diar model ×
embedder) pair against embedding cosine. On the run's 0.1 s grid that asks ten times a second against
embeddings windowed at 0.5 s, so every disagreement between a diarizer's continuity claim and the
cosine registered as doubt: it read 0.666 on a clean two-speaker conversation whose count posterior
was 2 at 0.978, whose per-speaker existence uncertainty was 0.0, and whose per-speaker presence doubt
averaged 0.168.

It now asks about **attribution**, from three terms, each a function here:

- :func:`per_speaker_attribution_doubt` — do the diarization models agree about who is here?
- :func:`word_location_doubt` — do we know where the words are? Word boundaries are what assign a
  word to a speaker's span, so not knowing where a word starts is not knowing whose it is. This
  consumes the per-edge temporal confidences D-27 moved onto the word, which had no consumer until
  now.
- :func:`target_activity_doubt` — do we know whether the target was active at all? Not knowing that
  is not knowing whether there is anyone to attribute.

Pure functions over plain data, deliberately: the composition they define can then be checked without
running a model, which the change-detection composition could not be.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

__all__ = [
    "per_speaker_attribution_doubt",
    "target_activity_doubt",
    "word_location_doubt",
]

SILENT_CLUSTER_ID = "SIL"
"""The pseudo-cluster standing for "no speaker here" — bookkeeping, never a person."""


def _binary_entropy(p: float) -> float:
    """Normalised Shannon entropy of a two-outcome split; 0 unanimous, 1 evenly split."""
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -(p * math.log(p) + (1.0 - p) * math.log(1.0 - p)) / math.log(2.0)


def per_speaker_attribution_doubt(
    clusters: Mapping[str, str],
    *,
    silent_cluster_id: str = SILENT_CLUSTER_ID,
) -> float | None:
    """How much the diarization models disagree about *who* is in this bucket.

    Per speaker present, the share of models placing them here, read as binary entropy — the same
    quantity ``speaker.per_speaker_tracks`` publishes per speaker in
    ``final/per_speaker_presence.parquet``, so the axis and that deliverable can no longer disagree
    about how confident the run is.

    **Folded by ``max`` over the speakers present, not by mean.** If any speaker's presence here is
    contested, attribution here is contested; averaging a contested speaker against a confidently
    placed one lets the confident one hide the doubt.

    Models reporting silence stay in the denominator: a lone detection among four silent models is
    exactly the case that must not read as certain.

    Args:
        clusters: ``{diar model → cluster id}`` for one bucket.
        silent_cluster_id: The id standing for "no speaker", excluded from the speakers but kept in
            the denominator.

    Returns:
        Doubt in ``[0, 1]``, or ``None`` when no model placed a speaker here — which is the absence
        of a claim rather than confident attribution of nobody.
    """
    if not clusters:
        return None
    n_models = len(clusters)
    active = sorted({c for c in clusters.values() if c != silent_cluster_id})
    if not active:
        return None
    return max(_binary_entropy(sum(1 for c in clusters.values() if c == cluster) / n_models) for cluster in active)


def word_location_doubt(
    words: Sequence[Mapping[str, Any]],
    buckets: Sequence[tuple[float, float]],
) -> dict[tuple[float, float], float | None]:
    """Per bucket, how poorly localised the words reaching it are.

    ``1 - temporal_confidence`` per word, coverage-weighted over the words overlapping the bucket.
    ``temporal_confidence`` is the fused word's own agreement about its span (the per-edge
    ``onset_confidence`` / ``offset_confidence`` folded), so this is the run's own measure of "do we
    know where this word is" — projected onto the axis grid the same way
    ``asr.resample_word_doubt`` projects accuracy.

    Args:
        words: Fused words carrying ``start``, ``end`` and ``temporal_confidence``.
        buckets: ``(start, end)`` pairs on the axis grid.

    Returns:
        ``{bucket → doubt}``, ``None`` where no word with a measured temporal confidence reaches the
        bucket. ``None`` rather than ``0.0``: nothing was said there, so nothing localises it, and
        zero would assert that we know exactly where a word we never heard was.
    """
    out: dict[tuple[float, float], float | None] = {}
    for bucket in buckets:
        weighted = 0.0
        total = 0.0
        for word in words:
            confidence = word.get("temporal_confidence")
            if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
                continue
            try:
                start, end = float(word["start"]), float(word["end"])
            except (KeyError, TypeError, ValueError):
                continue
            overlap = min(end, bucket[1]) - max(start, bucket[0])
            if overlap > 0:
                weighted += overlap * max(0.0, min(1.0, 1.0 - float(confidence)))
                total += overlap
        out[bucket] = (weighted / total) if total > 0 else None
    return out


def target_activity_doubt(
    mask_regions: Sequence[Mapping[str, Any]],
    buckets: Sequence[tuple[float, float]],
) -> dict[tuple[float, float], tuple[float | None, str | None]]:
    """Per bucket, the mask's doubt about whether the target was active — and its verdict.

    Returns the **state** alongside the number because the number alone cannot be acted on: low
    uncertainty means the mask is sure, and "sure the target is active" and "sure the region is
    target-free" call for opposite treatment. The caller nulls the axis where the state is
    ``target_free``, because there is nobody to attribute there.

    Doubt is contributed **only where the state is not** ``target_active``. Folding the mask's
    uncertainty in unconditionally was measured and rejected: 14 coarse regions against 214 fine
    buckets collapsed the axis from 80 distinct values to 35, the coarse measurement overwriting the
    fine one.

    A bucket takes the region it overlaps most, so a bucket straddling a boundary gets one answer
    rather than a blend of two verdicts, and ties break on region order for determinism.

    Args:
        mask_regions: Region dicts carrying ``start``, ``end``, ``state`` and ``uncertainty``.
        buckets: ``(start, end)`` pairs on the axis grid.

    Returns:
        ``{bucket → (doubt, state)}``. ``doubt`` is ``None`` where the state is ``target_active`` (the
        question is simply live) or where no region covers the bucket (the mask said nothing, which is
        not the same as saying the target was active). ``state`` is ``None`` in the latter case.
    """
    out: dict[tuple[float, float], tuple[float | None, str | None]] = {}
    for bucket in buckets:
        best_overlap = 0.0
        best: Mapping[str, Any] | None = None
        for region in mask_regions:
            try:
                start, end = float(region["start"]), float(region["end"])
            except (KeyError, TypeError, ValueError):
                continue
            overlap = min(end, bucket[1]) - max(start, bucket[0])
            if overlap > best_overlap:
                best_overlap, best = overlap, region
        if best is None:
            out[bucket] = (None, None)
            continue
        state = str(best.get("state")) if best.get("state") is not None else None
        if state == "target_active":
            out[bucket] = (None, state)
            continue
        raw = best.get("uncertainty")
        doubt = max(0.0, min(1.0, float(raw))) if isinstance(raw, (int, float)) and not isinstance(raw, bool) else None
        out[bucket] = (doubt, state)
    return out
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest src/tests/audio/workflows/audio_analysis/attribution_test.py -q`
Expected: `13 passed`.

- [ ] **Step 5: Lint, type-check, commit**

```bash
uv run ruff format src/senselab/audio/workflows/audio_analysis/attribution.py src/tests/audio/workflows/audio_analysis/attribution_test.py
uv run ruff check
uv run mypy .
git add src/senselab/audio/workflows/audio_analysis/attribution.py src/tests/audio/workflows/audio_analysis/attribution_test.py
git commit -m "feat(speaker): the three attribution terms, as pure functions"
```

---

### Task 2: Split the consensus word fold so it runs once

**Files:**
- Modify: `src/senselab/audio/workflows/audio_analysis/asr.py` (`_consensus_word_doubt`, `harvest_asr_votes`)
- Test: `src/tests/audio/workflows/audio_analysis/asr_word_resampling_test.py`

**Interfaces:**
- Consumes: nothing from Task 1.
- Produces: `asr.fuse_consensus_words(asr_resolved: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]`
  returning `(fused_words, provenance)`; and `harvest_asr_votes(..., fused: tuple[list[dict[str, Any]], dict[str, Any]] | None = None)`.

The speaker axis needs the same fused words the asr axis folds. Folding twice would run g2p phoneme
similarity twice per pass for one answer, so `harvest_pass` folds once and hands the result to both.

- [ ] **Step 1: Write the failing test**

Append to `src/tests/audio/workflows/audio_analysis/asr_word_resampling_test.py`:

```python
def test_the_fold_is_exposed_so_two_axes_can_share_one_call() -> None:
    """The speaker axis needs these same words, and folding twice would run g2p twice for one answer.

    Asserted on identity of the result rather than on the call count: what matters is that a caller
    can obtain the words once and hand them to both harvests, and that doing so gives the same axis
    values as letting the harvest fold them itself.
    """
    from senselab.audio.workflows.audio_analysis.asr import fuse_consensus_words

    summary = _pass_summary(
        {
            "model-a": [(0.0, 0.4, "hello"), (0.4, 0.9, "there")],
            "model-b": [(0.0, 0.4, "hello"), (0.4, 0.9, "chair")],
        },
        duration_s=1.0,
    )
    from senselab.audio.workflows.audio_analysis.harvesters import resolve_asr_result

    resolved = {
        m: resolve_asr_result(b, None) for m, b in summary["asr"]["by_model"].items() if b.get("status") == "ok"
    }
    words, provenance = fuse_consensus_words(resolved)
    assert words, "the fold produced no words"
    assert provenance["operator"] == "consensus_words/resample"
    assert all("temporal_confidence" in w for w in words), "the speaker axis reads this field"

    own = harvest_asr_votes(pass_summary=summary, grid=BucketGrid(), alignment_by_model={})
    shared = harvest_asr_votes(
        pass_summary=summary, grid=BucketGrid(), alignment_by_model={}, fused=(words, provenance)
    )
    assert own == shared, "handing the fold in must not change the axis"
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest src/tests/audio/workflows/audio_analysis/asr_word_resampling_test.py -q -k two_axes`
Expected: FAIL with `ImportError: cannot import name 'fuse_consensus_words'`.

- [ ] **Step 3: Implement the split**

In `src/senselab/audio/workflows/audio_analysis/asr.py`, replace the body of `_consensus_word_doubt`
with a call to a new public `fuse_consensus_words`, keeping every existing comment and the
provenance construction inside the new function:

```python
def fuse_consensus_words(
    asr_resolved: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Fold the recognizers' words once, returning the fused words and the fold's provenance.

    Split out of :func:`_consensus_word_doubt` because **two axes read these words**: the asr axis
    resamples their accuracy, and the speaker axis reads their ``temporal_confidence`` as
    word-location doubt. The fold runs g2p phoneme similarity per word pair, so doing it twice per
    pass would double that cost to obtain one answer — and worse, would let the two axes disagree
    about a fold neither of them owns.

    Returns:
        ``(fused_words, provenance)``. ``provenance`` travels onto every row that uses the fold,
        because the fold's parameters *are* its policy (D-21 rule 4).
    """
    from senselab.audio.tasks.speech_to_text_ensemble import fuse_word_streams, iter_word_leaves

    slot_overlap, slot_mid_tol_s = 0.3, 0.15
    streams: dict[str, list[dict[str, Any]]] = {}
    unreadable: list[str] = []
    for model_id, resolved in asr_resolved.items():
        words = iter_word_leaves(_as_plain(resolved))
        if words:
            streams[str(model_id)] = words
        else:
            unreadable.append(str(model_id))
    if unreadable:
        print(
            f"warn: asr fold extracted no words from {sorted(unreadable)} — either the model produced "
            "no transcript, or its result shape is one `_as_plain` could not convert (the axis then "
            "reports nothing for it, which is not the same as reporting no doubt)",
            file=sys.stderr,
        )
    if not streams:
        return ([], {})

    grading_languages = _warn_if_grading_is_out_of_language(asr_resolved)
    fused = fuse_word_streams(
        streams,
        slot_overlap=slot_overlap,
        slot_mid_tol_s=slot_mid_tol_s,
        text_similarity=phoneme_similarity,
        columns=aligned_columns(streams),
    )
    counts = sorted({int(w["timing_sources"]) for w in fused if w.get("timing_sources") is not None})
    provenance = {
        "operator": "consensus_words/resample",
        "sources": sorted(streams),
        "n_words": len(fused),
        "slot_overlap": slot_overlap,
        "slot_mid_tol_s": slot_mid_tol_s,
        "grading_languages": grading_languages,
        "timing_sources": (counts[0] if len(counts) == 1 else counts) if counts else None,
    }
    return (list(fused), provenance)


def _consensus_word_doubt(
    asr_resolved: Mapping[str, Any],
    buckets: Sequence[tuple[float, float]],
    *,
    fused: tuple[list[dict[str, Any]], dict[str, Any]] | None = None,
) -> tuple[dict[tuple[float, float], float | None], dict[str, Any]]:
    """Resample the fused words' accuracy onto the grid (D-27).

    ``fused`` lets a caller supply a fold it already performed — ``compute.harvest_pass`` does, so the
    asr and speaker axes share one call.
    """
    words, provenance = fused if fused is not None else fuse_consensus_words(asr_resolved)
    if not words:
        return ({b: None for b in buckets}, {})
    return resample_word_doubt(words, buckets), provenance
```

Then give `harvest_asr_votes` the passthrough parameter. Change its signature and the call:

```python
def harvest_asr_votes(
    *,
    pass_summary: dict[str, Any],
    grid: BucketGrid,
    alignment_by_model: dict[str, Any],
    fused: tuple[list[dict[str, Any]], dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
```

and inside it:

```python
    word_doubt, word_doubt_provenance = _consensus_word_doubt(asr_resolved, buckets, fused=fused)
```

Add to `harvest_asr_votes`'s docstring, under the existing text:

```
    ``fused`` accepts a consensus fold the caller already performed (``compute.harvest_pass`` shares
    one with the speaker axis, which reads the same words' ``temporal_confidence``). Omitted, the
    harvest folds them itself, which is what a standalone caller wants.
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest src/tests/audio/workflows/audio_analysis/asr_word_resampling_test.py src/tests/audio/workflows/audio_analysis/compute_uncertainty_axes_test.py -q`
Expected: all pass — the asr axis's values must be unchanged by this refactor.

- [ ] **Step 5: Lint, type-check, commit**

```bash
uv run ruff format src/senselab/audio/workflows/audio_analysis/asr.py src/tests/audio/workflows/audio_analysis/asr_word_resampling_test.py
uv run ruff check
uv run mypy .
git add src/senselab/audio/workflows/audio_analysis/asr.py src/tests/audio/workflows/audio_analysis/asr_word_resampling_test.py
git commit -m "refactor(asr): expose the consensus fold so two axes share one call"
```

---

### Task 3: Emit the three scored voters from the speaker harvest

**Files:**
- Modify: `src/senselab/audio/workflows/audio_analysis/speaker.py` (`harvest_speaker_votes`)
- Test: `src/tests/audio/workflows/audio_analysis/speaker_attribution_test.py`

**Interfaces:**
- Consumes: `attribution.per_speaker_attribution_doubt`, `attribution.word_location_doubt`,
  `attribution.target_activity_doubt` (Task 1).
- Produces: `harvest_speaker_votes(..., fused_words: Sequence[Mapping[str, Any]] | None = None)`
  emitting per bucket the scored entries `per_speaker_presence`, `asr_location`, `target_activity`,
  and no longer emitting `same_label_uncertainty` / `change_inconsistency_uncertainty` /
  `__cross_diar_label_disagreement__.value` / `__overlap_count__.value` as scored fields.

- [ ] **Step 1: Write the failing tests**

Create `src/tests/audio/workflows/audio_analysis/speaker_attribution_test.py`:

```python
"""The speaker axis emits attribution voters, and stops emitting change-detection ones.

The regression being fixed, restated as a test: a bucket every diarizer agrees on must read low even
when the previous bucket held a different speaker. The change-detection composition scored exactly
that case high, which is how a clean two-speaker conversation reported 0.666.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from senselab.audio.workflows.audio_analysis.fuse import per_signal_uncertainty
from senselab.audio.workflows.audio_analysis.grid import BucketGrid
from senselab.audio.workflows.audio_analysis.speaker import harvest_speaker_votes

SCORED_FIELDS = ("value", "same_label_uncertainty", "change_inconsistency_uncertainty")


def _diar(segments: list[tuple[float, float, str]]) -> dict[str, Any]:
    segs = [SimpleNamespace(start=s, end=e, speaker=spk, text="") for s, e, spk in segments]
    return {"status": "ok", "result": [segs], "cache_key": "k"}


def _summary(**extra: Any) -> dict[str, Any]:
    """Two diarizers agreeing on one speaker for the first half and another for the second."""
    a = [(0.0, 0.5, "SPEAKER_00"), (0.5, 1.0, "SPEAKER_01")]
    return {
        "duration_s": 1.0,
        "diarization": {"by_model": {"pyannote": _diar(a), "sortformer": _diar(a)}},
        **extra,
    }


def _votes(**kwargs: Any) -> list[dict[str, Any]]:
    return harvest_speaker_votes(
        pass_summary=kwargs.pop("pass_summary", _summary()),
        grid=BucketGrid(),
        per_window_embeddings={},
        **kwargs,
    )


def test_agreeing_diarizers_read_low_even_across_a_speaker_change() -> None:
    """The regression. Both models change speaker at 0.5 s and agree about it throughout."""
    buckets = _votes()
    assert buckets, "the harvest produced no buckets"
    for bucket in buckets:
        doubt = per_signal_uncertainty(bucket)
        assert doubt.get("per_speaker_presence") == pytest.approx(0.0), (
            f"agreeing models must carry no attribution doubt at {bucket['start']}"
        )


def test_the_change_detection_entries_are_no_longer_scored() -> None:
    """They are the 0.666. Nothing the fold reads may carry them."""
    for bucket in _votes():
        for name, entry in (bucket["votes"] or {}).items():
            if not isinstance(entry, dict):
                continue
            for field in ("same_label_uncertainty", "change_inconsistency_uncertainty"):
                assert field not in entry, f"{name} still carries {field}"
        read = set(per_signal_uncertainty(bucket))
        assert not {n for n in read if "::" in n}, f"a (diar::emb) pair is still scored: {read}"
        assert "__cross_diar_label_disagreement__" not in read
        assert "__overlap_count__" not in read


def test_the_cluster_assignments_survive_for_their_other_readers() -> None:
    """`per_speaker_tracks`, `cluster_active_time` and identity repair all read these."""
    from senselab.audio.workflows.audio_analysis.speaker import cluster_active_time, per_speaker_tracks

    buckets = _votes()
    assert per_speaker_tracks(buckets), "the per-speaker deliverable lost its input"
    assert cluster_active_time(buckets), "cluster ranking lost its input"


def test_word_location_doubt_reaches_the_axis() -> None:
    """A poorly localised word raises attribution doubt: we do not know whose it is."""
    words = [{"start": 0.0, "end": 0.5, "temporal_confidence": 0.2}]
    buckets = _votes(fused_words=words)
    first = per_signal_uncertainty(buckets[0])
    assert first.get("asr_location") == pytest.approx(0.8)


def test_an_indeterminate_mask_raises_attribution_doubt() -> None:
    """Not knowing whether the target was active is not knowing whether anyone is here."""
    summary = _summary(
        background_mask={
            "status": "ok",
            "result": {"regions": [{"start": 0.0, "end": 1.0, "state": "indeterminate", "uncertainty": 1.0}]},
        }
    )
    buckets = _votes(pass_summary=summary)
    assert per_signal_uncertainty(buckets[0]).get("target_activity") == pytest.approx(1.0)


def test_a_confidently_target_free_bucket_makes_no_claim() -> None:
    """No one to attribute, so no vote at all — None, never 0.0."""
    summary = _summary(
        background_mask={
            "status": "ok",
            "result": {"regions": [{"start": 0.0, "end": 1.0, "state": "target_free", "uncertainty": 0.02}]},
        }
    )
    for bucket in _votes(pass_summary=summary):
        assert bucket["votes"] == {}, "a target-free bucket must carry no attribution vote"
        assert per_signal_uncertainty(bucket) == {}


def test_a_target_active_mask_adds_nothing() -> None:
    """Where the mask is sure the target is active, the attribution question is simply live."""
    summary = _summary(
        background_mask={
            "status": "ok",
            "result": {"regions": [{"start": 0.0, "end": 1.0, "state": "target_active", "uncertainty": 0.1}]},
        }
    )
    for bucket in _votes(pass_summary=summary):
        assert "target_activity" not in (bucket["votes"] or {})
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest src/tests/audio/workflows/audio_analysis/speaker_attribution_test.py -q`
Expected: FAIL — `per_speaker_presence` absent from `per_signal_uncertainty`, and the change-detection
fields still present.

- [ ] **Step 3: Implement**

In `src/senselab/audio/workflows/audio_analysis/speaker.py`:

1. Add the import at the top, beside the existing ones:

```python
from senselab.audio.workflows.audio_analysis.attribution import (
    per_speaker_attribution_doubt,
    target_activity_doubt,
    word_location_doubt,
)
```

2. Add the parameter to `harvest_speaker_votes`:

```python
    fused_words: Sequence[Mapping[str, Any]] | None = None,
```

and document it in the Args block:

```
        fused_words: The consensus words from ``asr.fuse_consensus_words``, read for their
            ``temporal_confidence`` — word boundaries are what assign a word to a speaker's span, so
            not knowing where a word starts is not knowing whose it is. Omitted, the ``asr_location``
            voter is simply absent and the axis degrades to its other two terms; the row's
            ``contributing_signals`` records which voted.
```

3. Where the per-bucket `votes` dict is complete (immediately before the bucket is appended to the
output list), stop emitting the scored change-detection fields and add the three voters. Delete the
`same_label_uncertainty` and `change_inconsistency_uncertainty` keys from the `<diar>::<emb>` entries
and delete the `"value"` keys from `__cross_diar_label_disagreement__` and `__overlap_count__`,
leaving their remaining diagnostic keys (`cluster_ids` is read by `_bucket_clusters`).

Then, after the per-bucket loop has built `out`, add the attribution voters in a second pass:

```python
    # ── the attribution voters ──
    # The axis asks "how sure are we who is speaking here?", so it is composed from the three terms
    # that bear on that, not from per-bucket change detection against a coarser embedding grid. See
    # ``attribution`` for why each belongs and what the change-detection composition cost.
    buckets = [(round(float(b["start"]), 6), round(float(b["end"]), 6)) for b in out]
    mask_doc = ((pass_summary.get("background_mask") or {}).get("result")) or {}
    mask_regions = mask_doc.get("regions") or []
    location = word_location_doubt(list(fused_words or ()), buckets)
    activity = target_activity_doubt(mask_regions, buckets)

    for bucket_dict in out:
        key = (round(float(bucket_dict["start"]), 6), round(float(bucket_dict["end"]), 6))
        votes = bucket_dict["votes"]
        doubt, state = activity[key]
        if state == "target_free":
            # Nobody to attribute, so no claim at all. ``0.0`` would assert confident attribution
            # where no attribution was made.
            bucket_dict["votes"] = {}
            continue
        presence = per_speaker_attribution_doubt(_bucket_clusters(bucket_dict))
        if presence is not None:
            votes["per_speaker_presence"] = {"value": presence, "operator": "max_over_speakers/entropy"}
        if location[key] is not None:
            votes["asr_location"] = {"value": location[key], "operator": "1-temporal_confidence/coverage_mean"}
        if doubt is not None:
            votes["target_activity"] = {"value": doubt, "operator": "mask_region/gated_on_state", "state": state}

    return out
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest src/tests/audio/workflows/audio_analysis/speaker_attribution_test.py -q`
Expected: `7 passed`.

- [ ] **Step 5: Run the neighbours that read this harvest**

Run: `uv run pytest src/tests/audio/workflows -q`
Expected: failures only in tests that assert the old speaker composition. Fix each by pointing it at
the new voters — do **not** weaken an assertion to make it pass. If a test's premise no longer exists
(it was asserting change-detection scoring), delete it and say so in the commit message.

- [ ] **Step 6: Lint, type-check, commit**

```bash
uv run ruff format
uv run ruff check
uv run mypy .
git add -A
git commit -m "feat(speaker)!: the axis measures attribution, not change"
```

---

### Task 4: Thread the shared fold through harvest_pass

**Files:**
- Modify: `src/senselab/audio/workflows/audio_analysis/compute.py` (`harvest_pass`)
- Test: `src/tests/audio/workflows/audio_analysis/compute_uncertainty_axes_test.py`

**Interfaces:**
- Consumes: `asr.fuse_consensus_words` (Task 2), `harvest_speaker_votes(fused_words=...)` (Task 3).
- Produces: no new public names. `harvest_pass` folds once and hands the words to both harvests.

- [ ] **Step 1: Write the failing test**

Append to `src/tests/audio/workflows/audio_analysis/compute_uncertainty_axes_test.py`:

```python
def test_the_speaker_axis_reads_the_words_the_asr_axis_folded() -> None:
    """One fold per pass, shared. The speaker axis's location term must actually arrive.

    Both axes read the same fused words: the asr axis resamples their accuracy, the speaker axis their
    temporal confidence. If ``harvest_pass`` fails to thread them, the speaker axis silently loses a
    voter — the same class of silent-omission failure that once left the asr axis with zero
    contributing signals.
    """
    diar_segs = [(0.0, 1.0, "SPEAKER_00"), (1.0, 4.0, "SPEAKER_01")]
    raw_pass = {
        "duration_s": 4.0,
        "diarization": {"by_model": {"pyannote": _diar_block(diar_segs), "sortformer": _diar_block(diar_segs)}},
        "asr": {
            "by_model": {
                "whisper": _asr_block_with_chunks([(0.0, 1.0, "hello"), (1.0, 4.0, "world")]),
                "granite": _asr_block_with_chunks([(0.0, 1.0, "hello"), (1.0, 4.0, "planet")]),
            }
        },
    }
    _signals, fused_axes, _incomparable, _emb = compute_uncertainty_axes(
        passes={"raw": raw_pass},
        grid=BucketGrid(),
        params={},
        audio={"raw": _silent_audio(4.0)},
        speaker_embedding_models=[],
        aggregator="min",
        speech_presence_labels=["Speech"],
    )
    signals = {s for row in fused_axes["speaker"].rows for s in (row.get("contributing_signals") or ())}
    assert "asr_location" in signals, f"the speaker axis lost its location voter; got {sorted(signals)}"
    assert "per_speaker_presence" in signals
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest src/tests/audio/workflows/audio_analysis/compute_uncertainty_axes_test.py -q -k reads_the_words`
Expected: FAIL — `asr_location` absent, because `harvest_pass` does not pass `fused_words` yet.

- [ ] **Step 3: Implement**

In `src/senselab/audio/workflows/audio_analysis/compute.py`, inside `harvest_pass`, replace the asr
harvest block and move it **above** the speaker harvest so the fold is available:

```python
    # ── the consensus word fold, once per pass ──
    # Two axes read these words: the asr axis resamples their accuracy, the speaker axis reads their
    # ``temporal_confidence`` as word-location doubt. The fold runs g2p per word pair, so folding it
    # twice would double that cost for one answer — and would let the two axes disagree about a fold
    # neither owns.
    from senselab.audio.workflows.audio_analysis.asr import fuse_consensus_words
    from senselab.audio.workflows.audio_analysis.harvesters import resolve_asr_result

    asr_blocks = (harvest_summary.get("asr") or {}).get("by_model") or {}
    asr_resolved = {
        m: resolve_asr_result(b, align_by_model.get(m))
        for m, b in asr_blocks.items()
        if isinstance(b, dict) and b.get("status") == "ok"
    }
    consensus_fold = fuse_consensus_words(asr_resolved)

    # ── asr harvest ──
    asr_votes = harvest_asr_votes(
        pass_summary=harvest_summary,
        grid=grid,
        alignment_by_model=align_by_model,
        fused=consensus_fold,
    )
```

and give the existing `harvest_speaker_votes(...)` call the words:

```python
        fused_words=consensus_fold[0],
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest src/tests/audio/workflows/audio_analysis/compute_uncertainty_axes_test.py src/tests/audio/workflows/audio_analysis/grid_test.py -q`
Expected: all pass. `grid_test` must still show every axis on one grid with equal row counts.

- [ ] **Step 5: Lint, type-check, commit**

```bash
uv run ruff format
uv run ruff check
uv run mypy .
git add -A
git commit -m "feat(compute): fold the consensus words once and share them across both axes"
```

---

### Task 5: Declaration, dead code, cache version

**Files:**
- Modify: `src/senselab/audio/workflows/audio_analysis/axes.py:186-203` (the `asr` and `speaker` Axis entries)
- Modify: `src/senselab/utils/tasks/cached_inference.py` (`CACHE_SCHEMA_VERSION`)
- Modify: `src/senselab/audio/workflows/audio_analysis/speaker.py` (delete what became uncalled)
- Modify: `src/senselab/audio/workflows/audio_analysis/doc.md`, `CLAUDE.md`

- [ ] **Step 1: Update the axis declaration**

In `axes.py`, change the `speaker` entry's `question` and add the reason:

```python
    Axis(
        name="speaker",
        question="who is speaking here?",
```

and beneath it, replacing the existing comment about attenuation, add:

```python
        # Was "was it the same speaker as before?". That framing asked a change question at the grid
        # rate and validated it against embeddings windowed ten times coarser, so it read 0.666 on a
        # conversation whose per-speaker presence doubt was 0.168. The axis is composed from
        # ``attribution``'s three terms now; see ``speaker-axis-attribution-design.md``.
```

- [ ] **Step 2: Find what became uncalled and delete it with its tests**

```bash
for f in $(grep -o "^def [a-z_A-Z0-9]*" src/senselab/audio/workflows/audio_analysis/speaker.py | sed 's/def //'); do
  n=$(grep -rn "\b$f\b" src/senselab scripts | grep -v "speaker.py:" | wc -l | tr -d ' ')
  echo "$n $f"
done | sort -n | head -20
```

Any function reporting `0` external references **and** not called inside `speaker.py` is now dead:
delete it and delete its tests. Do not leave a recorded-but-unread helper — that is the
`__pairwise_phoneme_distances__` mistake. Expect the embedding same-label / change-inconsistency
helpers to appear here; verify each before deleting, because `identity_repair` reads some of them.

- [ ] **Step 3: Bump the cache schema version**

In `src/senselab/utils/tasks/cached_inference.py`, change `CACHE_SCHEMA_VERSION = 11` to `12` and
append to its docstring:

```
Bumped 11 → 12 when the speaker axis stopped measuring change and started measuring attribution. Its
scored voters are now ``per_speaker_presence`` / ``asr_location`` / ``target_activity`` rather than
per-(diar × embedder) ``same_label_uncertainty`` and ``change_inconsistency_uncertainty``, so a
cached row's ``contributing_signals`` names voters this axis no longer has and lacks the three it
does. Every number keyed to the speaker axis moves with it: region proposal, convergence, residual
mass, the disagreements ranking and the LS bins. ``theta_low`` / ``theta_high`` were not tuned
against this composition and must be re-measured rather than carried over.
```

- [ ] **Step 4: Update the prose**

In `src/senselab/audio/workflows/audio_analysis/doc.md`, under the axis list, change the `speaker`
line to "**speaker** — who is speaking here?" and add a paragraph after the asr-axis one:

```markdown
The `speaker` axis measures **attribution**: how sure we are *who* is speaking, composed by
`attribution.py` from three voters — per-speaker presence doubt (`max` over the speakers present of
the entropy of the model share), ASR word-location doubt (`1 - temporal_confidence`, coverage-weighted
over the words reaching the bucket), and target-activity doubt (the mask region's uncertainty, only
where its `state` is not `target_active`). It asked "did the speaker change since the previous
bucket?" until 2026-08-05, validated per (diar × embedder) pair against embedding cosine — which on a
0.1 s grid asks ten times a second against 0.5 s windows, and read 0.666 on a clean two-speaker
conversation whose per-speaker presence doubt was 0.168. A bucket the mask confidently calls
`target_free` carries no vote at all: there is nobody to attribute.
```

Mirror the same two facts (new question, three voters) in `CLAUDE.md`'s "Three-axis uncertainty
workflow" section where the axes are listed.

- [ ] **Step 5: Full check and commit**

```bash
uv run ruff format
uv run ruff check
uv run mypy .
uv run pytest src/tests/audio/workflows src/tests/scripts -q
git add -A
git commit -m "feat(axes)!: the speaker axis asks who is speaking, and the cache knows it"
```

---

### Task 6: Verify against real runs

**Files:**
- Modify: `scripts/verify_grid_unification.py`

- [ ] **Step 1: Add the attribution check to the verifier**

In `scripts/verify_grid_unification.py`, inside `check`, after the existing `[4]` block, add:

```python
    # ── 6. the speaker axis tracks the per-speaker presence it is supposed to reflect ──
    print("\n[6] speaker axis vs the per-speaker presence it describes:")
    speaker = frames.get("speaker")
    psp_path = run_dir / "final" / "per_speaker_presence.parquet"
    if speaker is None or not psp_path.exists():
        print("    (no speaker axis or no per-speaker presence table)")
    else:
        psp = pd.read_parquet(psp_path)
        per_bucket: dict[tuple[float, float], float] = {}
        for row in psp.itertuples():
            key = (round(float(row.start), 6), round(float(row.end), 6))
            per_bucket[key] = max(per_bucket.get(key, 0.0), float(row.speech_presence_uncertainty))
        doubt = [1.0 - float(c) for c in speaker["confidence"].dropna()]
        names = {s for row in speaker["contributing_signals"].dropna() for s in list(row)}
        print(f"    per-speaker presence doubt (max/bucket): mean={sum(per_bucket.values()) / len(per_bucket):.4f}")
        print(f"    speaker axis doubt (1 - confidence):      mean={sum(doubt) / len(doubt):.4f}")
        print(f"    contributing voters: {sorted(names)}")
        expected = {"per_speaker_presence", "asr_location", "target_activity"}
        if not (names & expected):
            failures.append(f"[6] the speaker axis carries none of {sorted(expected)}; got {sorted(names)}")
        stale = {n for n in names if "::" in n} | {"__cross_diar_label_disagreement__", "__overlap_count__"}
        if names & stale:
            failures.append(f"[6] change-detection voters are still scored: {sorted(names & stale)}")
```

- [ ] **Step 2: Run both clips with the cache cleared**

```bash
rm -rf artifacts/analyze_audio/* artifacts/analyze_audio_cache
uv run python scripts/analyze_audio.py src/tests/data_for_testing/audio_48khz_mono_16bits.wav
uv run python scripts/analyze_audio.py src/tests/data_for_testing/english_conversation_higgs_audio_v2.wav
```

Expected: both exit 0.

- [ ] **Step 3: Verify**

```bash
uv run python scripts/verify_grid_unification.py artifacts/analyze_audio/*
```

Expected: exit 0. Specifically, on the conversation clip:
- every axis still on one grid with 214 rows (checks 2 and 3 unaffected);
- the speaker axis's doubt mean lands near **0.333**, not 0.666, and tracks the per-speaker presence
  doubt (≈0.118) plus the location term (≈0.220);
- `contributing_signals` contains `per_speaker_presence` and `asr_location`, and **no** `::` pair,
  `__cross_diar_label_disagreement__` or `__overlap_count__`;
- `final/transcript.json`'s 62 words are unchanged in text.

If the doubt mean is far from 0.333, stop and diagnose rather than adjusting the expectation — the
figure came from the same clip's own artifacts, so a large gap means a term is not arriving.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "test(verify): check the speaker axis tracks its per-speaker presence"
```

---

## Self-Review

**Spec coverage.** Axis question change → Task 5 Step 1. Three voters → Tasks 1 and 3. `max` over
speakers → Task 1 (`test_the_doubt_is_the_max_over_speakers_not_the_mean`). `asr_location` from
`temporal_confidence` → Tasks 1–4. Mask gated on `state` → Task 1, Task 3. `None` for target-free →
Task 3 (`test_a_confidently_target_free_bucket_makes_no_claim`). `None` for no speaker present →
Task 1. Composition via `fuse_axis` rather than a derived formula → Task 3 emits votes; no new fold
anywhere. Change-detection voters unscored → Task 3. Uncalled code deleted with tests → Task 5
Step 2. Cross-stage ASR dependency → Task 3's `fused_words` docstring plus the degradation path
(voter simply absent). Re-measurement → Task 5 Step 3. Testing section → Tasks 1, 3, 4, 6. The
resolution guard named in the spec's testing section is Task 6 Step 3's distinct-value expectation.

**Placeholder scan.** No TBD/TODO. Every code step carries the code. Task 5 Step 2 is a discovery
command rather than a fixed edit — deliberate, because which helpers become dead depends on Task 3's
final shape, and the step states the rule (`0` external references and no internal caller) and the
hazard (`identity_repair` reads some).

**Type consistency.** `fuse_consensus_words` returns `tuple[list[dict[str, Any]], dict[str, Any]]` in
Task 2 and is consumed as `consensus_fold` / `consensus_fold[0]` in Task 4 and as
`fused=(words, provenance)` in Task 2's own test. `harvest_asr_votes`'s new parameter is `fused` in
both Task 2 and Task 4. `harvest_speaker_votes`'s is `fused_words` in Task 3 and Task 4.
`target_activity_doubt` returns `(doubt, state)` in Task 1 and is unpacked as `doubt, state` in
Task 3. `per_speaker_attribution_doubt` takes the mapping `_bucket_clusters` returns
(`dict[str, str]`).
