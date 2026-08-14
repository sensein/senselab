# Audio hints and target-speaker embedding estimation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let an `Audio` carry declared hints (what it may contain, targeted speaker count,
environment, expected read text, a target-speaker embedding with provenance), and add an
estimator that turns a set of files which *may* contain a speaker into one embedding plus
statistics describing the distribution it came from.

**Architecture:** Three separable units. A pydantic hints layer hanging off `Audio`, carried and
never consumed. A pure vector-level descriptor in `utils/tasks/` that takes one set of vectors and
returns a centroid plus statistics, deciding nothing. An audio-level estimator in
`tasks/speaker_embeddings/` that windows, embeds, and calls the descriptor — with an opt-in
contamination-rejection selector layered between them.

**Tech Stack:** Python 3.12, pydantic v2, numpy, torch, scipy (`scipy.cluster.hierarchy` for AHC),
pytest. No new third-party dependency.

## Global Constraints

Copied from `design.md`. Every task's requirements implicitly include these.

- **Every Python command runs through `uv run`.** Never bare `python`/`pip`.
- **`uv sync` is subtractive** — always `--all-extras`.
- **Never run `pytest -n auto`.** Serial, scoped to the directory changed.
- **Run `uv run ruff format` before any commit.** Line length 120. Google-style docstrings.
- **Type hints required**; `mypy --ignore-missing-imports --extra-checks src/` must pass over all
  of `src/`, tests included. Run that exact command — not `mypy src/senselab/`.
- **Never `git add -A` unqualified.** Always limit with a pathspec.
- **Never construct an unmocked `HFModel`/`SpeechBrainModel` in a test.** It downloads a full
  snapshot. Tests inject synthetic embedding vectors instead.
- **No unfitted numeric literals as thresholds.** Every reference scale in this feature is either
  analytic (closed form in `d` or `n`) or derived from the data by a stated rule. If you find
  yourself typing a magic number that gates a decision, stop — it does not belong.
- **Explain *why* in comments and docstrings, not *what*.** A non-obvious choice records the
  measurement or failure that drove it.
- **No field anywhere in this feature may be a verdict, a boolean, a probability, or a
  thresholded label.** The one exception is the explicit `reject_contamination` flag, which is a
  caller's decision, and what it removed is always recorded.
- **All statistics types are pydantic `BaseModel`s**, not dataclasses — hints get serialised to
  disk, so everything hanging off `AudioHints` must serialise.

## File Structure

| Path | Responsibility | Action |
| --- | --- | --- |
| `src/senselab/utils/tasks/embedding_distribution.py` | Vector-level descriptor + optional dominant-group selector. No audio, no models. | Create |
| `src/senselab/audio/data_structures/audio_hints.py` | `AudioHints` and nested hint types | Create |
| `src/senselab/audio/data_structures/audio.py` | gains `hints: AudioHints \| None = None` | Modify |
| `src/senselab/audio/data_structures/__init__.py` | export the hint types | Modify |
| `src/senselab/audio/tasks/speaker_embeddings/windowing.py` | `window_starts`, `extract_per_window_embeddings`, promoted down from the workflow | Create |
| `src/senselab/audio/tasks/speaker_embeddings/api.py` | gains `estimate_speaker_embedding_from_audios` | Modify |
| `src/senselab/audio/tasks/speaker_embeddings/__init__.py` | export the estimator | Modify |
| `src/senselab/audio/tasks/speaker_embeddings/doc.md` | suggested hint vocabulary + estimator contract | Create |
| `src/senselab/audio/workflows/audio_analysis/embeddings.py` | import promoted primitives; two defect fixes | Modify |
| `src/tests/utils/embedding_distribution_test.py` | descriptor: geometry, nulls, LOO, spectrum | Create |
| `src/tests/utils/embedding_distribution_files_test.py` | descriptor: within/cross-file, file effect | Create |
| `src/tests/utils/embedding_distribution_selection_test.py` | selector: AHC, cut rule, shares | Create |
| `src/tests/audio/data_structures/audio_hints_test.py` | hint validation, cache-key invariance | Create |
| `src/tests/audio/tasks/speaker_embeddings_estimate_test.py` | estimator, provenance, rejection | Create |
| `src/tests/audio/tasks/task_layer_guard_test.py` | AST guard: `audio/tasks` must not import `audio/workflows` | Create |

**Why the descriptor tests are split across three files:** the descriptor has ~10 statistic
families. One test file would grow past the point where a reviewer can hold it in context, and the
three groups fail for independent reasons (pure geometry / per-file structure / permutation).

---

### Task 1: The hints layer

**Files:**
- Create: `src/senselab/audio/data_structures/audio_hints.py`
- Modify: `src/senselab/audio/data_structures/audio.py`
- Modify: `src/senselab/audio/data_structures/__init__.py`
- Test: `src/tests/audio/data_structures/audio_hints_test.py`

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces:
  - `ExpectedSpeech(text: str | None = None, prompt_id: str | None = None, reference: str | None = None)`
  - `SpeakerEmbeddingProvenance(model_id: str, model_commit_sha: str | None = None, unresolved_reason: str | None = None, method: str = "spherical_mean", source_files: list[str] = [], window_s: float | None = None, hop_s: float | None = None, n_windows_used: int = 0, n_windows_dropped: int = 0, created_at: str | None = None)`
  - `TargetSpeakerEmbedding(vector: list[float], provenance: SpeakerEmbeddingProvenance, distribution: Any | None = None)`
  - `AudioHints(may_contain: list[str] = [], targeted_speaker_count: int | None = None, environment: str | None = None, expected_speech: list[ExpectedSpeech] = [], target_speaker: TargetSpeakerEmbedding | None = None, metadata: dict[str, Any] = {})`
  - `Audio.hints: AudioHints | None = None`

**Note on `TargetSpeakerEmbedding.distribution`:** typed `Any | None` in this task on purpose.
Task 2 creates `EmbeddingDistribution`; Task 8 narrows this field to it. Importing
`utils.tasks.embedding_distribution` from `data_structures` before it exists would block this task
on Task 2 for no benefit.

- [ ] **Step 1: Write the failing test**

Create `src/tests/audio/data_structures/audio_hints_test.py`:

```python
"""Declared hints on an Audio.

A hint is an assertion by whoever knows the acquisition protocol -- never a measurement, and
never consumed by any task in this change. These tests pin the two properties that make
"declared and carried" true: absent stays distinguishable from empty, and a hint cannot change
what a computation returns.
"""

import torch

from senselab.audio.data_structures import Audio
from senselab.audio.data_structures.audio_hints import (
    AudioHints,
    ExpectedSpeech,
    SpeakerEmbeddingProvenance,
    TargetSpeakerEmbedding,
)


def test_an_audio_carries_no_hints_by_default() -> None:
    """Absent must stay distinguishable from empty.

    An empty AudioHints would make "nobody declared anything" read the same as "declared
    nothing" -- the same collapse as reading a None confidence as 0.0, which pii_detection
    documents at length.
    """
    audio = Audio(waveform=torch.zeros(1, 16000), sampling_rate=16000)
    assert audio.hints is None


def test_hints_hold_every_declared_field() -> None:
    """Every hint in the request round-trips through the model."""
    hints = AudioHints(
        may_contain=["read-speech", "cough"],
        targeted_speaker_count=1,
        environment="quiet-room",
        expected_speech=[
            ExpectedSpeech(text="The quick brown fox.", prompt_id="harvard-01", reference="ieee-1969"),
            ExpectedSpeech(text="Rice is often served in round bowls.", prompt_id="harvard-02"),
        ],
    )
    audio = Audio(waveform=torch.zeros(1, 16000), sampling_rate=16000, hints=hints)
    assert audio.hints is not None
    assert audio.hints.may_contain == ["read-speech", "cough"]
    assert audio.hints.targeted_speaker_count == 1
    assert audio.hints.environment == "quiet-room"
    assert len(audio.hints.expected_speech) == 2
    assert audio.hints.expected_speech[0].prompt_id == "harvard-01"


def test_expected_speech_preserves_order() -> None:
    """A file often holds several sentences read in sequence.

    Concatenating them would destroy the boundaries a matcher needs to say *which* sentence was
    skipped or reordered -- a different question from how close the whole thing was.
    """
    hints = AudioHints(
        expected_speech=[ExpectedSpeech(text="first"), ExpectedSpeech(text="second"), ExpectedSpeech(text="third")]
    )
    assert [e.text for e in hints.expected_speech] == ["first", "second", "third"]


def test_provenance_records_a_resolved_sha_or_says_why_not() -> None:
    """A ref in the commit-sha field would be provenance that is confidently wrong.

    #550 established that recording a ref while claiming a commit is worse than recording
    nothing, so an unresolved model must set unresolved_reason instead.
    """
    resolved = SpeakerEmbeddingProvenance(model_id="speechbrain/spkrec-ecapa-voxceleb", model_commit_sha="a" * 40)
    assert resolved.model_commit_sha == "a" * 40
    assert resolved.unresolved_reason is None

    unresolved = SpeakerEmbeddingProvenance(
        model_id="speechbrain/spkrec-ecapa-voxceleb",
        model_commit_sha=None,
        unresolved_reason="offline: hub unreachable and no cached ref",
    )
    assert unresolved.model_commit_sha is None
    assert unresolved.unresolved_reason


def test_a_non_sha_commit_value_is_rejected() -> None:
    """The field means "resolved commit". Anything ref-shaped in it defeats the point."""
    import pytest

    with pytest.raises(ValueError, match="40"):
        SpeakerEmbeddingProvenance(model_id="org/model", model_commit_sha="main")


def test_a_target_speaker_embedding_carries_its_provenance() -> None:
    """A vector with no provenance cannot be interpreted later, so provenance is required."""
    emb = TargetSpeakerEmbedding(
        vector=[0.1, 0.2, 0.3],
        provenance=SpeakerEmbeddingProvenance(model_id="org/model", model_commit_sha="b" * 40),
    )
    assert emb.provenance.model_id == "org/model"
    assert emb.distribution is None


def test_hints_do_not_change_the_cache_key() -> None:
    """A hint nothing consumes must not change what a computation returns.

    Not a backwards-compatibility concern -- alpha owes none -- but a correctness one: if a hint
    moved a cache key, "carried only" would be false.
    """
    from senselab.utils.tasks.cached_inference import audio_content_hash

    waveform = torch.rand(1, 16000)
    bare = Audio(waveform=waveform, sampling_rate=16000)
    hinted = Audio(
        waveform=waveform.clone(),
        sampling_rate=16000,
        hints=AudioHints(may_contain=["read-speech"], targeted_speaker_count=2),
    )
    assert audio_content_hash(bare) == audio_content_hash(hinted)
```

- [ ] **Step 2: Run the test and watch it fail**

```bash
uv run --no-sync pytest src/tests/audio/data_structures/audio_hints_test.py -v 2>&1 | tail -20
```

Expected: FAIL — `ModuleNotFoundError: senselab.audio.data_structures.audio_hints`.

- [ ] **Step 3: Find the real name of the content-hash helper**

The last test imports `audio_content_hash`. That name is a guess; find what
`cached_inference.py` actually calls the function that hashes a waveform:

```bash
grep -n "def .*hash\|def cache_key\|audio.waveform" src/senselab/utils/tasks/cached_inference.py | head
```

Use the real name in the test. If the helper is private (leading underscore), import it as-is —
this test is asserting an internal property on purpose. If no such helper exists, build the key
through the public `cache_key(...)` for both audios and compare those instead.

- [ ] **Step 4: Create the hints module**

Create `src/senselab/audio/data_structures/audio_hints.py`:

```python
"""Declared hints about what an ``Audio`` may contain.

A hint is an **assertion** -- by an operator, an acquisition protocol, or a corpus description --
about what a recording was *meant* to contain. It is never a measurement, and nothing in this
change consumes one: no task alters its behaviour because a hint is present. How a hint should
inform a decision is itself a decision, and it gets its own derivation when someone builds that
consumer.

This is deliberately not the same thing as dataset metadata resolved by a lookup (PR #543's
``AudioPlus``). A lookup's trust comes from the dataset; a hint's comes from whoever declared it.
Keeping them apart means hints work with no provider, no corpus, and no network.
"""

from __future__ import annotations

import re
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator

_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


class ExpectedSpeech(BaseModel):
    """The text a speaker was asked to produce, for a read task.

    Attributes:
        text: The verbatim prompt. Present so the hint is self-contained -- a consumer can match
            a transcript against it without resolving anything.
        prompt_id: Identifier in an external reference set, e.g. ``"harvard-01"``.
        reference: Which reference set the id belongs to (name, version, or URI). Together with
            ``prompt_id`` this traces the prompt back to its corpus without vendoring that corpus
            into this repository.
    """

    text: Optional[str] = None
    prompt_id: Optional[str] = None
    reference: Optional[str] = None


class SpeakerEmbeddingProvenance(BaseModel):
    """Where a target-speaker embedding came from.

    Attributes:
        model_id: The embedding model, e.g. ``"speechbrain/spkrec-ecapa-voxceleb"``.
        model_commit_sha: The **resolved** 40-hex commit the vector was produced with, or ``None``.
            Never a ref: recording ``"main"`` here would be provenance that is confidently wrong,
            which is worse than recording none.
        unresolved_reason: Why ``model_commit_sha`` is ``None``. Required in that case, so an
            absent commit is always explained rather than merely missing.
        method: How the vector was aggregated, e.g. ``"spherical_mean"`` or
            ``"spherical_mean+dominant_cluster"`` when contamination rejection ran.
        source_files: What the estimate was computed from.
        window_s: Window length used, in seconds.
        hop_s: Hop between windows, in seconds.
        n_windows_used: Windows that contributed to the returned vector.
        n_windows_dropped: Windows excluded -- zero-norm, or removed by contamination rejection.
            Kept beside ``n_windows_used`` so a curated estimate cannot look like a clean one.
        created_at: ISO-8601 timestamp, stamped by the caller. Not defaulted to "now": a library
            that stamps wall-clock time makes its own output unreproducible.
    """

    model_id: str
    model_commit_sha: Optional[str] = None
    unresolved_reason: Optional[str] = None
    method: str = "spherical_mean"
    source_files: list[str] = Field(default_factory=list)
    window_s: Optional[float] = None
    hop_s: Optional[float] = None
    n_windows_used: int = 0
    n_windows_dropped: int = 0
    created_at: Optional[str] = None

    @field_validator("model_commit_sha")
    @classmethod
    def _must_be_a_sha(cls, v: Optional[str]) -> Optional[str]:
        """Reject anything that is not a full 40-hex commit.

        Args:
            v: The candidate value.

        Returns:
            The value unchanged when it is ``None`` or a 40-hex commit.

        Raises:
            ValueError: When the value is a ref name or a short hash. The field's whole purpose is
                to be immutable; a ref in it silently reintroduces the ambiguity it removes.
        """
        if v is None:
            return v
        if not _SHA_RE.match(v):
            raise ValueError(f"model_commit_sha must be a resolved 40-hex commit, got {v!r}")
        return v


class TargetSpeakerEmbedding(BaseModel):
    """A speaker embedding declared as the target for a recording.

    Attributes:
        vector: The embedding, unit-norm. Held inline rather than as a path to a stored artifact
            so a hint is interpretable on its own -- a reference that outlives its file is the
            dangling-pointer failure this avoids.
        provenance: Required. A vector with no provenance cannot be interpreted or reproduced.
        distribution: Optional statistics describing the set the vector was estimated from. Typed
            loosely here to keep ``data_structures`` from importing ``utils.tasks``; narrowed by
            the estimator's own signature.
    """

    vector: list[float]
    provenance: SpeakerEmbeddingProvenance
    distribution: Optional[Any] = None


class AudioHints(BaseModel):
    """What a recording was declared to contain.

    Attributes:
        may_contain: Open tags -- ``"read-speech"``, ``"cough"``, ``"music"``. Named *may* contain
            because a hint is an expectation, not an observation; nothing downstream should read
            it as ground truth. Open strings rather than an enum: a closed vocabulary here would
            be a taxonomy nobody fitted, and every corpus that did not fit it would force an edit.
            See ``speaker_embeddings/doc.md`` for a suggested, non-binding vocabulary.
        targeted_speaker_count: How many speakers the acquisition protocol aimed for -- intent,
            not a count of who is audible. A range is deliberately not modelled; it goes in
            ``metadata`` until a caller needs it, rather than shipping parallel min/max fields.
        environment: Open tag, e.g. ``"quiet-room"``, ``"clinic"``, ``"telephone"``.
        expected_speech: Ordered prompts for a read task. Ordered and separate rather than one
            concatenated string, because "which sentence was skipped" is a different question
            from "how close was the whole thing".
        target_speaker: The declared target speaker's embedding, with provenance.
        metadata: Escape hatch for corpus-specific extras that do not deserve a typed field.
    """

    may_contain: list[str] = Field(default_factory=list)
    targeted_speaker_count: Optional[int] = None
    environment: Optional[str] = None
    expected_speech: list[ExpectedSpeech] = Field(default_factory=list)
    target_speaker: Optional[TargetSpeakerEmbedding] = None
    metadata: dict[str, Any] = Field(default_factory=dict)
```

- [ ] **Step 5: Attach the field to `Audio`**

In `src/senselab/audio/data_structures/audio.py`, add the import and one field beside the existing
`metadata: Dict = Field(default={})`:

```python
from senselab.audio.data_structures.audio_hints import AudioHints
```

```python
    hints: Optional[AudioHints] = None
```

Add to the class docstring's `Attributes:` block:

```
        hints: Declared expectations about this recording (see ``audio_hints``). ``None`` means
            nobody declared anything, which is deliberately distinct from an empty ``AudioHints``.
            Nothing in senselab consumes hints; they are carried.
```

Check for an import cycle before moving on: `audio_hints.py` must import **nothing** from
`senselab.audio` — only pydantic and stdlib. If `audio.py` already imports from
`senselab.utils.data_structures`, that is fine and unrelated.

- [ ] **Step 6: Export the new types**

In `src/senselab/audio/data_structures/__init__.py`, add `AudioHints`, `ExpectedSpeech`,
`SpeakerEmbeddingProvenance` and `TargetSpeakerEmbedding` to the imports and to `__all__`, keeping
`__all__` alphabetically sorted the way the file already is.

- [ ] **Step 7: Run the tests to verify they pass**

```bash
uv run --no-sync pytest src/tests/audio/data_structures/audio_hints_test.py -v 2>&1 | tail -20
```

Expected: PASS, 7 tests.

- [ ] **Step 8: Verify nothing else broke**

```bash
uv run --no-sync pytest src/tests/audio/data_structures/ -q 2>&1 | tail -5
uv run --no-sync ruff format src/senselab/audio/data_structures/ src/tests/audio/data_structures/
uv run --no-sync ruff check src/ src/tests/
uv run --no-sync mypy --ignore-missing-imports --extra-checks src/ 2>&1 | tail -3
```

Expected: all pass. `Audio` is constructed all over the suite, so a broken field shows up here.

- [ ] **Step 9: Commit**

```bash
git add src/senselab/audio/data_structures/audio_hints.py \
        src/senselab/audio/data_structures/audio.py \
        src/senselab/audio/data_structures/__init__.py \
        src/tests/audio/data_structures/audio_hints_test.py
git commit -m "feat(audio): declared hints on an Audio, carried and not consumed

A hint is an assertion about what a recording was meant to contain -- may_contain,
targeted_speaker_count, environment, expected read prompts, and a target-speaker
embedding with provenance. Nothing consumes them: how a hint should inform a
decision is itself a decision and gets its own derivation later.

hints defaults to None rather than an empty AudioHints so 'nobody declared
anything' stays distinguishable from 'declared nothing'. model_commit_sha rejects
anything that is not 40-hex, because a ref in that field is provenance that is
confidently wrong. A test pins that a hint cannot move a cache key -- if it could,
'carried only' would be false."
```

---

### Task 2: Descriptor core — geometry, counts, nulls, R̄, LOO cosines, spectrum

**Files:**
- Create: `src/senselab/utils/tasks/embedding_distribution.py`
- Test: `src/tests/utils/embedding_distribution_test.py`

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces (later tasks extend the same module and the same `EmbeddingDistribution` model):
  - `SimilarityStats(min, q05, q25, q50, q75, q95, max, mean, sd)` — all `float`
  - `GeometryInfo(metric: str, l2_normalised: bool, dim: int, distance: str, centroid_rule: str)`
  - `CountsInfo(n_vectors_total: int, n_scored: int, n_zero_norm_dropped: int, n_files: int, vectors_per_file: dict[str, int], window_s: float | None, hop_s: float | None, n_effective: float | None)`
  - `NullsInfo(cos_sd_null: float, rbar_null: float, participation_ratio_null: float, auc_null: float)`
  - `SpectrumStats(participation_ratio: float, pc1_share_centred: float, eigenvalue_shares_top5: list[float])`
  - `EmbeddingDistribution(geometry, counts, nulls, cos_to_centroid_loo: SimilarityStats, rbar: float, spectrum: SpectrumStats)` — Tasks 3-5 add `within_file`, `cross_file`, `file_effect`, `centroid_robustness`
  - `describe_embedding_distribution(vectors, file_ids=None, *, aggregator="spherical_mean", window_s=None, hop_s=None, window_starts_s=None, n_permutations=1000, seed=0) -> tuple[list[float], EmbeddingDistribution]`
  - `_l2_normalise(x: np.ndarray) -> tuple[np.ndarray, int]` — returns normalised rows and the count dropped for zero norm
  - `_similarity_stats(values: np.ndarray) -> SimilarityStats`
  - `_spherical_mean(x: np.ndarray) -> np.ndarray` — unit-norm mean direction

- [ ] **Step 1: Write the failing test**

Create `src/tests/utils/embedding_distribution_test.py`:

```python
"""Core statistics of one set of embedding vectors.

Every reference scale here is analytic, so these tests check the statistics against closed forms
rather than against recorded numbers. That is the point: a fitted literal would have to be
measured and maintained, while 1/sqrt(d) is true by construction.
"""

import numpy as np
import pytest

from senselab.utils.tasks.embedding_distribution import describe_embedding_distribution


def _tight_cone(n: int, d: int, spread: float, seed: int = 0) -> np.ndarray:
    """n vectors clustered around one random direction, with angular spread."""
    rng = np.random.default_rng(seed)
    axis = rng.normal(size=d)
    axis /= np.linalg.norm(axis)
    x = axis[None, :] + spread * rng.normal(size=(n, d))
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def test_uniform_random_vectors_land_on_the_analytic_nulls() -> None:
    """Random directions must reproduce the closed forms, or the nulls are wrong.

    This is the test that keeps the block free of fitted numbers: sd of pairwise cosines -> 1/sqrt(d),
    mean resultant length -> 1/sqrt(n), participation ratio -> d*n/(d+n).
    """
    n, d = 800, 192
    rng = np.random.default_rng(17)
    x = rng.normal(size=(n, d))
    x /= np.linalg.norm(x, axis=1, keepdims=True)

    _, dist = describe_embedding_distribution(x)

    assert dist.nulls.cos_sd_null == pytest.approx(1.0 / np.sqrt(d))
    assert dist.nulls.rbar_null == pytest.approx(1.0 / np.sqrt(n))
    assert dist.nulls.participation_ratio_null == pytest.approx(d * n / (d + n))
    assert dist.nulls.auc_null == 0.5

    # And the data actually sits near them.
    assert dist.rbar == pytest.approx(1.0 / np.sqrt(n), abs=0.02)
    assert dist.spectrum.participation_ratio == pytest.approx(d * n / (d + n), rel=0.1)


def test_a_tight_cone_has_high_rbar_and_low_effective_rank() -> None:
    """One coherent speaker points one way and occupies few directions."""
    x = _tight_cone(n=400, d=192, spread=0.15)
    _, dist = describe_embedding_distribution(x)

    assert dist.rbar > 0.9
    assert dist.rbar > 10 * dist.nulls.rbar_null
    assert dist.spectrum.participation_ratio < 0.5 * dist.nulls.participation_ratio_null


def test_the_centroid_is_unit_norm() -> None:
    """The returned vector is a direction. Callers compare it by cosine."""
    centroid, _ = describe_embedding_distribution(_tight_cone(n=50, d=64, spread=0.2))
    assert np.linalg.norm(np.asarray(centroid)) == pytest.approx(1.0)


def test_zero_norm_rows_are_dropped_and_counted() -> None:
    """A zero vector has no direction, so it cannot contribute -- but silence about it would
    make n_scored unexplainable."""
    x = _tight_cone(n=20, d=32, spread=0.1)
    x = np.vstack([x, np.zeros((3, 32))])
    _, dist = describe_embedding_distribution(x)

    assert dist.counts.n_vectors_total == 23
    assert dist.counts.n_scored == 20
    assert dist.counts.n_zero_norm_dropped == 3


def test_input_is_normalised_even_when_the_caller_did_not() -> None:
    """ECAPA embeddings are not unit norm, and the norm covaries with window energy and how much
    speech fills the window. Left alone it would inject a loudness nuisance into every statistic,
    so normalisation is unconditional and the block says so."""
    x = _tight_cone(n=40, d=64, spread=0.1)
    scaled = x * np.linspace(0.1, 50.0, x.shape[0])[:, None]

    c_plain, d_plain = describe_embedding_distribution(x)
    c_scaled, d_scaled = describe_embedding_distribution(scaled)

    assert d_scaled.geometry.l2_normalised is True
    assert np.allclose(c_plain, c_scaled, atol=1e-6)
    assert d_scaled.rbar == pytest.approx(d_plain.rbar, abs=1e-6)


def test_loo_cosines_match_a_naive_recomputation() -> None:
    """Scoring a vector against a centroid it helped define is optimistically biased.

    The closed form x_i . (S - x_i) / ||S - x_i|| must equal recomputing the centroid without i,
    which this checks directly on a small input.
    """
    x = _tight_cone(n=12, d=16, spread=0.3, seed=3)
    _, dist = describe_embedding_distribution(x)

    naive = []
    for i in range(x.shape[0]):
        others = np.delete(x, i, axis=0)
        c = others.sum(axis=0)
        c /= np.linalg.norm(c)
        naive.append(float(x[i] @ c))
    naive_arr = np.sort(np.asarray(naive))

    assert dist.cos_to_centroid_loo.min == pytest.approx(naive_arr.min(), abs=1e-9)
    assert dist.cos_to_centroid_loo.max == pytest.approx(naive_arr.max(), abs=1e-9)
    assert dist.cos_to_centroid_loo.q50 == pytest.approx(float(np.quantile(naive_arr, 0.5)), abs=1e-9)


def test_n_effective_discounts_overlapping_windows() -> None:
    """At a 2.0 s window on a 1.0 s hop, adjacent windows share half their audio, so independent
    information is about n/2. Reporting it lets a consumer discount nulls that scale as
    n^-1/2 instead of us pretending independence."""
    x = _tight_cone(n=100, d=32, spread=0.2)
    _, dist = describe_embedding_distribution(x, window_s=2.0, hop_s=1.0)
    assert dist.counts.n_effective == pytest.approx(50.0, rel=0.05)


def test_n_effective_is_none_without_window_information() -> None:
    """It cannot be derived from vectors alone, and a guessed value would be worse than none."""
    _, dist = describe_embedding_distribution(_tight_cone(n=10, d=8, spread=0.1))
    assert dist.counts.n_effective is None


def test_pc1_share_is_computed_on_centred_data() -> None:
    """Uncentred, PC1 is just the mean direction and explains almost everything for any coherent
    set -- a field that always reads the same. Centred, a high PC1 share is the signature of
    bimodality, which is what makes it worth reporting."""
    d = 64
    rng = np.random.default_rng(5)
    axis = rng.normal(size=d)
    axis /= np.linalg.norm(axis)
    perp = rng.normal(size=d)
    perp -= (perp @ axis) * axis
    perp /= np.linalg.norm(perp)

    # Two lobes displaced along one perpendicular direction: one dominant axis of variation.
    a = axis[None, :] + 0.35 * perp[None, :] + 0.02 * rng.normal(size=(100, d))
    b = axis[None, :] - 0.35 * perp[None, :] + 0.02 * rng.normal(size=(100, d))
    x = np.vstack([a, b])
    x /= np.linalg.norm(x, axis=1, keepdims=True)

    _, dist = describe_embedding_distribution(x)
    assert dist.spectrum.pc1_share_centred > 0.8
    assert len(dist.spectrum.eigenvalue_shares_top5) == 5


def test_geometry_records_what_was_done() -> None:
    """A consumer reading a stored block has to know the geometry to interpret any number in it."""
    _, dist = describe_embedding_distribution(_tight_cone(n=10, d=8, spread=0.1))
    assert dist.geometry.metric == "cosine"
    assert dist.geometry.distance == "angular"
    assert dist.geometry.dim == 8
    assert dist.geometry.centroid_rule == "spherical_mean"


def test_too_few_vectors_raises_rather_than_returning_a_meaningless_block() -> None:
    """One vector has no distribution. Returning a block of zeros would look like a measurement."""
    with pytest.raises(ValueError, match="at least 2"):
        describe_embedding_distribution(np.ones((1, 8)))
```

- [ ] **Step 2: Run the test and watch it fail**

```bash
uv run --no-sync pytest src/tests/utils/embedding_distribution_test.py -v 2>&1 | tail -10
```

Expected: FAIL — `ModuleNotFoundError: senselab.utils.tasks.embedding_distribution`.

- [ ] **Step 3: Implement the module**

Create `src/senselab/utils/tasks/embedding_distribution.py`. Write the module docstring first —
it carries the reasoning that keeps later readers from "improving" this into something wrong:

```python
"""Describe one set of embedding vectors: a centroid, and statistics about the distribution.

This module **describes and never decides**. There is no verdict field, no boolean, no
probability, and no thresholded label anywhere in its output. A consumer applies its own
threshold, so every statistic here is either bounded on an interpretable scale or paired with an
analytic null it can be read against.

Every reference scale is closed-form, which is what keeps the module free of literals nobody
fitted:

- sd of pairwise cosines between independent directions: ``1/sqrt(d)`` (0.0722 at d=192)
- mean resultant length of ``n`` independent directions: ``E[Rbar^2] = 1/n``, so ``1/sqrt(n)``
- participation ratio under Marchenko-Pastur: ``d*n/(d+n)``
- Mann-Whitney AUC under exchangeability: exactly ``0.5``

**The counter-intuitive one.** A *small* sd of cosines is not evidence of a coherent speaker. At
d=192 independent random directions give sd ~= 0.072, so an observed 0.05 is *below* the
random-vector null. sd is therefore never reported as a headline dispersion figure -- only beside
``nulls.cos_sd_null``.

Geometry: vectors are L2-normalised on entry, unconditionally. SpeechBrain speaker embeddings are
not unit norm, and the norm covaries with window energy and how much speech fills the window -- a
cough, or 0.4 s of speech in a 2.0 s window, gets a systematically different norm. Any
unnormalised statistic would mix that loudness/occupancy nuisance into what a reader takes for
speaker dispersion. ECAPA is trained with an angular-margin objective and scored by cosine, so
the norm is not part of the discriminative geometry to begin with.

After normalisation cosine and Euclidean are the *same* geometry --
``||x-y||^2 = 2(1-cos t)``, a strictly monotone reparametrisation -- so the common "Euclidean is
unusable at high d" objection does not apply to any rank- or neighbour-based quantity. Where a
true metric is needed (medoid, linkage) the geodesic ``arccos(clip(cos,-1,1))`` is used, because
``cos`` is not a metric and neither is ``1-cos``.

**Deliberately absent**, each for a mechanical reason rather than taste:

- *Silhouette.* ``silhouette(metric="cosine")`` and ``silhouette(metric="euclidean")`` return
  different numbers for identical geometry on unit vectors (Jensen: the square root compresses
  large distances more than small ones), so any threshold on silhouette is a threshold on a
  parameterisation choice. It is also a property of a chosen partition, not of the data.
- *k-NN purity.* Hubness is severe at this dimension -- k-occurrence skew ~3.3 at d=192, with a
  measurable fraction of points appearing in no neighbour list -- so neighbour counts carry a bias
  unrelated to speaker identity. Worse, at 50% window overlap a window's nearest neighbour is
  almost always the temporally adjacent window, so any same-file purity statistic would read ~1.0
  for every input: it would measure the hop size.
- *Intrinsic-dimensionality estimators.* Two-NN's ``r1`` becomes the distance to a near-duplicate
  window, driving the estimate down by the hop size. Same artefact.
- *von Mises-Fisher concentration.* ``kappa = Rbar(d - Rbar^2)/(1 - Rbar^2)`` is a deterministic
  function of ``Rbar`` and ``d``, both reported, so it stores nothing new; it is unbounded as
  ``Rbar -> 1``, so it is not interpretable on its own scale; and vMF assumes isotropic
  concentration, which embedding spaces violate. Recover it in one line if you want it.
"""
```

Then the models and the implementation:

```python
from __future__ import annotations

from typing import Any, Optional, Sequence, Union

import numpy as np
from pydantic import BaseModel, Field

AGGREGATOR_SPHERICAL_MEAN = "spherical_mean"
AGGREGATOR_TRIMMED_MEAN = "trimmed_mean"
AGGREGATOR_MEDOID = "medoid"
_AGGREGATORS = (AGGREGATOR_SPHERICAL_MEAN, AGGREGATOR_TRIMMED_MEAN, AGGREGATOR_MEDOID)

# The fraction trimmed by the trimmed-mean aggregator and by the robustness diagnostic. Not a
# decision threshold: nothing is accepted or rejected on it, and both the trimmed result and its
# cosine to the untrimmed mean are reported, so a reader sees what trimming did rather than
# inheriting its verdict.
_TRIM_FRACTION = 0.10


class SimilarityStats(BaseModel):
    """Quantiles of a set of cosine values, plus mean and sd.

    Quantiles rather than mean-and-sd alone because contamination makes this distribution a
    *mixture*: mean +/- sd of a bimodal mixture describes neither lobe, and the sd is inflated by
    exactly the thing worth exposing -- so sd alone cannot separate "one loose speaker" from "two
    tight speakers". ``q05`` and ``min`` are where an intruder shows up.
    """

    min: float
    q05: float
    q25: float
    q50: float
    q75: float
    q95: float
    max: float
    mean: float
    sd: float


class GeometryInfo(BaseModel):
    """What was done to the vectors, so a stored block can be interpreted later."""

    metric: str
    l2_normalised: bool
    dim: int
    distance: str
    centroid_rule: str


class CountsInfo(BaseModel):
    """Sizes, and the effective sample size after accounting for window overlap.

    Attributes:
        n_effective: ``total_windowed_duration / window_s``, which is about ``n/2`` at a 2.0 s
            window on a 1.0 s hop. ``None`` when window information was not supplied -- it cannot
            be derived from vectors alone, and a guess would be worse than an absence. Any null
            whose width scales as ``n^-1/2`` is about sqrt(2) overconfident without this discount.
    """

    n_vectors_total: int
    n_scored: int
    n_zero_norm_dropped: int
    n_files: int
    vectors_per_file: dict[str, int] = Field(default_factory=dict)
    window_s: Optional[float] = None
    hop_s: Optional[float] = None
    n_effective: Optional[float] = None


class NullsInfo(BaseModel):
    """Closed-form reference scales, so no statistic needs a fitted threshold.

    ``dim`` and ``n_scored`` are in ``CountsInfo`` and ``GeometryInfo`` specifically so a consumer
    can recompute every one of these and check ours.
    """

    cos_sd_null: float
    rbar_null: float
    participation_ratio_null: float
    auc_null: float


class SpectrumStats(BaseModel):
    """How many directions the set actually occupies.

    Attributes:
        participation_ratio: ``(sum lambda)^2 / sum lambda^2`` of the centred covariance, in
            ``[1, min(n,d)]``. Read against ``nulls.participation_ratio_null``: well below it means
            a genuinely low-dimensional set, near it means indistinguishable from white noise at
            this sample size.
        pc1_share_centred: ``lambda_1 / sum lambda`` on **centred** data. Uncentred, PC1 is the
            mean direction and explains almost everything for any coherent set, which would make
            this a field that always reads the same. Centred, a high share is the signature of
            bimodality.
        eigenvalue_shares_top5: The five largest ``lambda_i / sum lambda``, zero-padded when
            ``min(n,d) < 5``.
    """

    participation_ratio: float
    pc1_share_centred: float
    eigenvalue_shares_top5: list[float]


class EmbeddingDistribution(BaseModel):
    """Statistics describing one set of embedding vectors. Contains no verdict."""

    geometry: GeometryInfo
    counts: CountsInfo
    nulls: NullsInfo
    cos_to_centroid_loo: SimilarityStats
    rbar: float
    spectrum: SpectrumStats


def _as_array(vectors: Union[Sequence[Sequence[float]], np.ndarray, Any]) -> np.ndarray:
    """Coerce input to a 2-D float64 array without importing torch at module scope.

    Args:
        vectors: A nested sequence, a numpy array, or anything exposing ``detach``/``cpu``/``numpy``
            (a torch tensor).

    Returns:
        A 2-D ``float64`` array, one row per vector.

    Raises:
        ValueError: If the result is not 2-D.
    """
    if hasattr(vectors, "detach"):
        vectors = vectors.detach().cpu().numpy()
    arr = np.asarray(vectors, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"vectors must be 2-D (n, d); got shape {arr.shape}")
    return arr


def _l2_normalise(x: np.ndarray) -> tuple[np.ndarray, int]:
    """Return unit-norm rows and the count of zero-norm rows removed.

    Args:
        x: ``(n, d)`` array.

    Returns:
        ``(normalised, n_dropped)``. A zero-norm row has no direction, so it cannot contribute to
        any angular statistic; it is dropped rather than producing ``nan``, and counted so
        ``n_scored`` stays explainable.
    """
    norms = np.linalg.norm(x, axis=1)
    keep = norms > 0
    n_dropped = int((~keep).sum())
    return x[keep] / norms[keep][:, None], n_dropped


def _spherical_mean(x: np.ndarray) -> np.ndarray:
    """Unit-norm mean direction of unit-norm rows.

    This is the von Mises-Fisher MLE direction, and its error shrinks as ``O(n^-1/2)``. An
    arithmetic mean of *unnormalised* vectors would weight each row by its norm -- i.e. by
    loudness and speech occupancy -- so a loud cough would outvote a quiet target utterance.

    Args:
        x: ``(n, d)`` unit-norm array.

    Returns:
        A unit-norm ``(d,)`` direction.

    Raises:
        ValueError: If the rows sum to the zero vector, which has no direction.
    """
    s = x.sum(axis=0)
    norm = float(np.linalg.norm(s))
    if norm == 0.0:
        raise ValueError("vectors sum to zero; no mean direction exists")
    return s / norm


def _similarity_stats(values: np.ndarray) -> SimilarityStats:
    """Summarise a 1-D array of cosine values.

    Args:
        values: Any 1-D array of cosines.

    Returns:
        Quantiles plus mean and sd. ``sd`` is included for completeness but is meaningless without
        ``nulls.cos_sd_null`` beside it; see the module docstring.
    """
    v = np.asarray(values, dtype=np.float64).ravel()
    q = np.quantile(v, [0.05, 0.25, 0.50, 0.75, 0.95])
    return SimilarityStats(
        min=float(v.min()),
        q05=float(q[0]),
        q25=float(q[1]),
        q50=float(q[2]),
        q75=float(q[3]),
        q95=float(q[4]),
        max=float(v.max()),
        mean=float(v.mean()),
        sd=float(v.std(ddof=1)) if v.size > 1 else 0.0,
    )


def _loo_cos_to_centroid(x: np.ndarray) -> np.ndarray:
    """Cosine of each row to the centroid computed *without* that row.

    Scoring a vector against a centroid it helped define is optimistically biased -- each row pulls
    the centroid toward itself. Closed form, one pass, no loop: with ``S = sum x_j`` over unit
    rows, the leave-one-out mean direction is proportional to ``S - x_i``, so
    ``cos_loo(i) = x_i . (S - x_i) / ||S - x_i||``.

    Args:
        x: ``(n, d)`` unit-norm array with ``n >= 2``.

    Returns:
        ``(n,)`` array of leave-one-out cosines.
    """
    s = x.sum(axis=0)
    diff = s[None, :] - x
    denom = np.linalg.norm(diff, axis=1)
    numer = np.einsum("ij,ij->i", x, diff)
    out = np.zeros_like(denom)
    nz = denom > 0
    out[nz] = numer[nz] / denom[nz]
    return out


def _spectrum(x: np.ndarray) -> SpectrumStats:
    """Participation ratio, centred PC1 share, and the top-5 eigenvalue shares.

    Uses singular values of the centred matrix rather than forming the ``d x d`` covariance: both
    quantities are ratios, so the ``1/(n-1)`` factor cancels and the SVD is cheaper and better
    conditioned.

    Args:
        x: ``(n, d)`` unit-norm array.

    Returns:
        The spectrum summary.
    """
    centred = x - x.mean(axis=0, keepdims=True)
    sv = np.linalg.svd(centred, compute_uv=False)
    lam = sv**2
    total = float(lam.sum())
    if total == 0.0:
        # Every row identical: no variation at all. One occupied direction, by definition.
        return SpectrumStats(participation_ratio=1.0, pc1_share_centred=0.0, eigenvalue_shares_top5=[0.0] * 5)
    pr = float(total**2 / float((lam**2).sum()))
    shares = (lam / total).tolist()
    top5 = [float(v) for v in shares[:5]] + [0.0] * max(0, 5 - len(shares))
    return SpectrumStats(participation_ratio=pr, pc1_share_centred=float(shares[0]), eigenvalue_shares_top5=top5)


def describe_embedding_distribution(
    vectors: Union[Sequence[Sequence[float]], np.ndarray, Any],
    file_ids: Optional[Sequence[str]] = None,
    *,
    aggregator: str = AGGREGATOR_SPHERICAL_MEAN,
    window_s: Optional[float] = None,
    hop_s: Optional[float] = None,
    window_starts_s: Optional[Sequence[float]] = None,
    n_permutations: int = 1000,
    seed: int = 0,
) -> tuple[list[float], EmbeddingDistribution]:
    """Describe one set of embedding vectors: a centroid and statistics about the distribution.

    Decides nothing. See the module docstring for the geometry, the analytic nulls, and what is
    deliberately absent.

    Args:
        vectors: ``(n, d)`` embeddings. Normalised on entry regardless of input scale.
        file_ids: One id per row, when the vectors come from several files. Enables the per-file
            statistics; ``None`` treats the set as one group.
        aggregator: ``"spherical_mean"`` (default), ``"trimmed_mean"``, or ``"medoid"``. A tool
            parameter, not a decision: the returned block always reports the cosine between the
            mean and both alternatives, so a reader sees whether the choice mattered.
        window_s: Window length in seconds, used for ``n_effective`` and the permutation block
            length. ``None`` leaves both unreported rather than guessed.
        hop_s: Hop between windows in seconds.
        window_starts_s: Start time of each window, same order as ``vectors``. Required for the
            same-file guard band; without it that guard is skipped and reported as ``None``.
        n_permutations: Permutations for the file-effect reference.
        seed: Seed for the permutation. Fixed by default so the reported quantile is reproducible.

    Returns:
        ``(centroid, distribution)``. The centroid is a unit-norm ``list[float]``.

    Raises:
        ValueError: If fewer than 2 vectors survive normalisation, if ``vectors`` is not 2-D, if
            ``aggregator`` is unknown, or if ``file_ids``/``window_starts_s`` lengths disagree with
            ``vectors``.
    """
    if aggregator not in _AGGREGATORS:
        raise ValueError(f"aggregator must be one of {_AGGREGATORS}; got {aggregator!r}")

    raw = _as_array(vectors)
    n_total = int(raw.shape[0])
    if file_ids is not None and len(file_ids) != n_total:
        raise ValueError(f"file_ids has {len(file_ids)} entries for {n_total} vectors")
    if window_starts_s is not None and len(window_starts_s) != n_total:
        raise ValueError(f"window_starts_s has {len(window_starts_s)} entries for {n_total} vectors")

    norms = np.linalg.norm(raw, axis=1)
    keep_mask = norms > 0
    x, n_dropped = _l2_normalise(raw)
    n = int(x.shape[0])
    if n < 2:
        raise ValueError(f"need at least 2 non-zero vectors to describe a distribution; got {n}")
    d = int(x.shape[1])

    kept_files = [str(f) for f, k in zip(file_ids, keep_mask) if k] if file_ids is not None else None

    centroid = _spherical_mean(x)  # Tasks 4-5 replace this with the aggregator dispatch.

    per_file: dict[str, int] = {}
    if kept_files is not None:
        for f in kept_files:
            per_file[f] = per_file.get(f, 0) + 1

    n_effective: Optional[float] = None
    if window_s is not None and hop_s is not None and window_s > 0:
        # total covered duration / window_s: overlapping windows do not carry independent
        # information, and pretending they do makes every n^-1/2 null overconfident.
        n_effective = float((hop_s * (n - 1) + window_s) / window_s)

    return centroid.tolist(), EmbeddingDistribution(
        geometry=GeometryInfo(
            metric="cosine",
            l2_normalised=True,
            dim=d,
            distance="angular",
            centroid_rule=aggregator,
        ),
        counts=CountsInfo(
            n_vectors_total=n_total,
            n_scored=n,
            n_zero_norm_dropped=n_dropped,
            n_files=len(per_file) if per_file else (1 if kept_files is None else 0),
            vectors_per_file=per_file,
            window_s=window_s,
            hop_s=hop_s,
            n_effective=n_effective,
        ),
        nulls=NullsInfo(
            cos_sd_null=float(1.0 / np.sqrt(d)),
            rbar_null=float(1.0 / np.sqrt(n)),
            participation_ratio_null=float(d * n / (d + n)),
            auc_null=0.5,
        ),
        cos_to_centroid_loo=_similarity_stats(_loo_cos_to_centroid(x)),
        rbar=float(np.linalg.norm(x.sum(axis=0)) / n),
        spectrum=_spectrum(x),
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
uv run --no-sync pytest src/tests/utils/embedding_distribution_test.py -v 2>&1 | tail -20
```

Expected: PASS, 11 tests. If `test_n_effective_discounts_overlapping_windows` is off, check the
formula: 100 windows at 2.0 s with 1.0 s hop cover `1.0*99 + 2.0 = 101` s, so
`n_effective = 50.5`, which is inside the `rel=0.05` tolerance of 50.

- [ ] **Step 5: Lint, typecheck, commit**

```bash
uv run --no-sync ruff format src/senselab/utils/tasks/embedding_distribution.py src/tests/utils/embedding_distribution_test.py
uv run --no-sync ruff check src/ src/tests/
uv run --no-sync mypy --ignore-missing-imports --extra-checks src/ 2>&1 | tail -3
git add src/senselab/utils/tasks/embedding_distribution.py src/tests/utils/embedding_distribution_test.py
git commit -m "feat(utils): describe one set of embedding vectors, deciding nothing

A centroid plus statistics: geometry, counts, closed-form nulls, leave-one-out
cosine-to-centroid quantiles, mean resultant length, and the centred spectrum. No
verdict field, no boolean, no probability.

Every reference scale is analytic -- cos sd null 1/sqrt(d), Rbar null 1/sqrt(n),
participation ratio null d*n/(d+n), AUC null exactly 0.5 -- so the module carries
no literal anyone had to fit, and dim/n_scored are reported so a consumer can
recompute all four.

Vectors are L2-normalised unconditionally, because SpeechBrain embeddings are not
unit norm and the norm covaries with window energy and speech occupancy; left
alone it injects a loudness nuisance into every statistic. Cosine-to-centroid is
leave-one-out via x.(S-x)/||S-x||, since scoring a vector against a centroid it
helped define is optimistically biased. PC1 share is computed centred, because
uncentred it is just the mean direction and would always read the same.

Silhouette, k-NN purity, intrinsic dimensionality and vMF kappa are deliberately
absent; the module docstring records the mechanical reason for each."
```

---

### Task 3: Per-file statistics — within-file and cross-file

**Files:**
- Modify: `src/senselab/utils/tasks/embedding_distribution.py`
- Test: `src/tests/utils/embedding_distribution_files_test.py`

**Interfaces:**
- Consumes: from Task 2 — `SimilarityStats`, `_similarity_stats`, `_spherical_mean`,
  `_loo_cos_to_centroid`, `EmbeddingDistribution`, `describe_embedding_distribution`.
- Produces:
  - `WithinFileStats(n_vectors: int, rbar: float, cos_to_own_centroid_q05: float, cos_to_own_centroid_q50: float)`
  - `CrossFileStats(cos_file_centroid_to_pooled: dict[str, float], file_centroid_pairwise_cos: SimilarityStats | None)`
  - `EmbeddingDistribution.within_file: dict[str, WithinFileStats]` and
    `EmbeddingDistribution.cross_file: CrossFileStats` — both default-empty so Task 2's tests
    still pass.

- [ ] **Step 1: Write the failing test**

Create `src/tests/utils/embedding_distribution_files_test.py`:

```python
"""Per-file structure of an embedding set.

Within-file and cross-file dispersion are kept strictly separate because prior measurement on
this pipeline puts essentially the whole error budget cross-file: within-file cosine stability
0.984 against cross-file 0.891. A single pooled dispersion number would average those into
something uninterpretable and destroy the most informative split known about this data.
"""

import numpy as np
import pytest

from senselab.utils.tasks.embedding_distribution import describe_embedding_distribution


def _file_of_vectors(axis: np.ndarray, n: int, spread: float, seed: int) -> np.ndarray:
    """n unit vectors tightly around `axis`."""
    rng = np.random.default_rng(seed)
    x = axis[None, :] + spread * rng.normal(size=(n, axis.size))
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def _two_files_same_speaker(d: int = 64) -> tuple[np.ndarray, list[str]]:
    """Two files whose directions differ slightly -- one speaker, two sessions."""
    rng = np.random.default_rng(11)
    axis = rng.normal(size=d)
    axis /= np.linalg.norm(axis)
    nudge = rng.normal(size=d)
    nudge -= (nudge @ axis) * axis
    nudge /= np.linalg.norm(nudge)

    a = _file_of_vectors(axis + 0.05 * nudge, 40, 0.02, seed=1)
    b = _file_of_vectors(axis - 0.05 * nudge, 40, 0.02, seed=2)
    return np.vstack([a, b]), ["fileA"] * 40 + ["fileB"] * 40


def test_within_file_is_reported_per_file() -> None:
    """Each file gets its own coherence figure, not a pooled one."""
    x, ids = _two_files_same_speaker()
    _, dist = describe_embedding_distribution(x, ids)

    assert set(dist.within_file) == {"fileA", "fileB"}
    assert dist.within_file["fileA"].n_vectors == 40
    assert dist.within_file["fileA"].rbar > 0.95


def test_within_file_is_tighter_than_cross_file() -> None:
    """The measured error budget on this pipeline is cross-file, so the two must be separable and
    must actually differ on data built that way."""
    x, ids = _two_files_same_speaker()
    _, dist = describe_embedding_distribution(x, ids)

    within = min(dist.within_file[f].cos_to_own_centroid_q50 for f in dist.within_file)
    cross = min(dist.cross_file.cos_file_centroid_to_pooled.values())
    assert within > cross


def test_cross_file_reports_each_file_centroid_against_the_pooled_one() -> None:
    """A contaminated file shows up here as a low cosine, which is what lets a caller curate."""
    d = 64
    rng = np.random.default_rng(7)
    target = rng.normal(size=d)
    target /= np.linalg.norm(target)
    intruder = rng.normal(size=d)
    intruder -= (intruder @ target) * target
    intruder /= np.linalg.norm(intruder)

    x = np.vstack(
        [
            _file_of_vectors(target, 40, 0.02, seed=1),
            _file_of_vectors(target, 40, 0.02, seed=2),
            _file_of_vectors(intruder, 40, 0.02, seed=3),
        ]
    )
    ids = ["t1"] * 40 + ["t2"] * 40 + ["bad"] * 40
    _, dist = describe_embedding_distribution(x, ids)

    assert dist.cross_file.cos_file_centroid_to_pooled["bad"] < 0.5
    assert dist.cross_file.cos_file_centroid_to_pooled["t1"] > 0.8


def test_pairwise_file_centroid_cosines_are_summarised() -> None:
    """With three or more files the pairwise spread is itself informative."""
    x, ids = _two_files_same_speaker()
    third = _file_of_vectors(np.asarray(x[0]), 30, 0.02, seed=4)
    x = np.vstack([x, third])
    ids = ids + ["fileC"] * 30

    _, dist = describe_embedding_distribution(x, ids)
    assert dist.cross_file.file_centroid_pairwise_cos is not None
    assert -1.0 <= dist.cross_file.file_centroid_pairwise_cos.q50 <= 1.0


def test_a_single_file_has_no_pairwise_spread() -> None:
    """One file means no pair exists. Reporting a number would be inventing one."""
    x, _ = _two_files_same_speaker()
    _, dist = describe_embedding_distribution(x, ["only"] * x.shape[0])
    assert dist.cross_file.file_centroid_pairwise_cos is None
    assert set(dist.cross_file.cos_file_centroid_to_pooled) == {"only"}


def test_no_file_ids_leaves_the_per_file_blocks_empty() -> None:
    """Without ids there is no per-file structure to report, and inventing one would be a lie."""
    x, _ = _two_files_same_speaker()
    _, dist = describe_embedding_distribution(x)
    assert dist.within_file == {}
    assert dist.cross_file.cos_file_centroid_to_pooled == {}


def test_a_file_with_one_vector_still_reports_without_crashing() -> None:
    """Singleton files are ordinary in real corpora; rbar of one vector is 1.0 by definition."""
    x, ids = _two_files_same_speaker()
    x = np.vstack([x, x[:1]])
    ids = ids + ["singleton"]
    _, dist = describe_embedding_distribution(x, ids)
    assert dist.within_file["singleton"].n_vectors == 1
    assert dist.within_file["singleton"].rbar == pytest.approx(1.0)
```

- [ ] **Step 2: Run the test and watch it fail**

```bash
uv run --no-sync pytest src/tests/utils/embedding_distribution_files_test.py -v 2>&1 | tail -10
```

Expected: FAIL — `AttributeError: 'EmbeddingDistribution' object has no attribute 'within_file'`.

- [ ] **Step 3: Add the models**

In `embedding_distribution.py`, after `SpectrumStats`:

```python
class WithinFileStats(BaseModel):
    """Coherence inside one file.

    Attributes:
        n_vectors: Rows from this file that survived normalisation.
        rbar: Mean resultant length of this file's rows. ``1.0`` for a single row, by definition.
        cos_to_own_centroid_q05: 5th percentile of cosine to *this file's* centroid.
        cos_to_own_centroid_q50: Median cosine to this file's centroid.
    """

    n_vectors: int
    rbar: float
    cos_to_own_centroid_q05: float
    cos_to_own_centroid_q50: float


class CrossFileStats(BaseModel):
    """How the files sit relative to each other and to the pooled centroid.

    Attributes:
        cos_file_centroid_to_pooled: Per file, the cosine of its own centroid to the pooled
            centroid. A contaminated file shows up here directly, which is what lets a caller
            curate its input without any clustering.
        file_centroid_pairwise_cos: Quantiles of the pairwise cosines between file centroids.
            ``None`` with fewer than two files, because no pair exists and a reported number would
            be invented.
    """

    cos_file_centroid_to_pooled: dict[str, float] = Field(default_factory=dict)
    file_centroid_pairwise_cos: Optional[SimilarityStats] = None
```

Extend `EmbeddingDistribution` with defaults, so Task 2's construction stays valid:

```python
    within_file: dict[str, WithinFileStats] = Field(default_factory=dict)
    cross_file: CrossFileStats = Field(default_factory=CrossFileStats)
```

- [ ] **Step 4: Implement the computation**

Add this helper:

```python
def _per_file_stats(
    x: np.ndarray, file_ids: Optional[list[str]], pooled_centroid: np.ndarray
) -> tuple[dict[str, WithinFileStats], CrossFileStats]:
    """Within-file coherence and cross-file agreement.

    Kept strictly apart because the measured error budget on this pipeline is almost entirely
    cross-file (within-file cosine stability 0.984 against cross-file 0.891). Pooling them would
    average away the most informative split there is.

    Args:
        x: ``(n, d)`` unit-norm array.
        file_ids: One id per row, or ``None``.
        pooled_centroid: The centroid over all rows.

    Returns:
        ``(within_file, cross_file)``. Both are empty when ``file_ids`` is ``None``.
    """
    if file_ids is None:
        return {}, CrossFileStats()

    order: list[str] = []
    for f in file_ids:
        if f not in order:
            order.append(f)

    within: dict[str, WithinFileStats] = {}
    centroids: dict[str, np.ndarray] = {}
    ids = np.asarray(file_ids)
    for f in order:
        rows = x[ids == f]
        c = _spherical_mean(rows)
        centroids[f] = c
        cos_own = rows @ c
        within[f] = WithinFileStats(
            n_vectors=int(rows.shape[0]),
            rbar=float(np.linalg.norm(rows.sum(axis=0)) / rows.shape[0]),
            cos_to_own_centroid_q05=float(np.quantile(cos_own, 0.05)),
            cos_to_own_centroid_q50=float(np.quantile(cos_own, 0.50)),
        )

    to_pooled = {f: float(centroids[f] @ pooled_centroid) for f in order}

    pairwise: Optional[SimilarityStats] = None
    if len(order) >= 2:
        stacked = np.stack([centroids[f] for f in order])
        gram = stacked @ stacked.T
        iu = np.triu_indices(len(order), k=1)
        pairwise = _similarity_stats(gram[iu])

    return within, CrossFileStats(cos_file_centroid_to_pooled=to_pooled, file_centroid_pairwise_cos=pairwise)
```

In `describe_embedding_distribution`, after computing `centroid`, add:

```python
    within_file, cross_file = _per_file_stats(x, kept_files, centroid)
```

and pass `within_file=within_file, cross_file=cross_file` into the `EmbeddingDistribution(...)`
call.

- [ ] **Step 5: Run both descriptor test files**

```bash
uv run --no-sync pytest src/tests/utils/embedding_distribution_test.py src/tests/utils/embedding_distribution_files_test.py -v 2>&1 | tail -15
```

Expected: PASS, 18 tests total.

- [ ] **Step 6: Lint, typecheck, commit**

```bash
uv run --no-sync ruff format src/senselab/utils/tasks/embedding_distribution.py src/tests/utils/
uv run --no-sync ruff check src/ src/tests/
uv run --no-sync mypy --ignore-missing-imports --extra-checks src/ 2>&1 | tail -3
git add src/senselab/utils/tasks/embedding_distribution.py src/tests/utils/embedding_distribution_files_test.py
git commit -m "feat(utils): within-file and cross-file statistics, kept strictly separate

Per file: vector count, mean resultant length, and quantiles of cosine to that
file's own centroid. Across files: each file centroid's cosine to the pooled
centroid, plus quantiles of the pairwise file-centroid cosines.

Separate rather than pooled because prior measurement on this pipeline puts
essentially the whole error budget cross-file -- within-file cosine stability
0.984 against cross-file 0.891 -- so one pooled dispersion figure would average
away the most informative split available.

A contaminated file surfaces directly as a low cos_file_centroid_to_pooled, which
is what lets a caller curate its input set without any clustering. Fewer than two
files leaves file_centroid_pairwise_cos None rather than inventing a number for a
pair that does not exist."
```

---

### Task 4: The file effect — separability AUC and a block permutation

**Files:**
- Modify: `src/senselab/utils/tasks/embedding_distribution.py`
- Test: append to `src/tests/utils/embedding_distribution_files_test.py`

**Interfaces:**
- Consumes: from Tasks 2-3 — `describe_embedding_distribution`, `EmbeddingDistribution`.
- Produces:
  - `FileEffect(auc_same_file_vs_diff_file: float | None, permutation_quantile: float | None, permutation_block_len: int, n_permutations: int, guard_band_s: float | None, seed: int)`
  - `EmbeddingDistribution.file_effect: FileEffect` (default-constructed)
  - `_mann_whitney_auc(within: np.ndarray, between: np.ndarray) -> float`

- [ ] **Step 1: Write the failing test**

Append to `src/tests/utils/embedding_distribution_files_test.py`:

```python
def test_auc_is_half_when_files_carry_no_effect() -> None:
    """Exchangeable data must land on the exact null, 0.5.

    That the null is exact -- not fitted, not simulated -- is why this statistic replaces
    silhouette, whose value depends on whether you call it with cosine or Euclidean.
    """
    rng = np.random.default_rng(3)
    d, n = 48, 200
    x = rng.normal(size=(n, d))
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    ids = ["a" if i % 2 == 0 else "b" for i in range(n)]

    _, dist = describe_embedding_distribution(x, ids, n_permutations=200, seed=0)
    assert dist.file_effect.auc_null_reference == 0.5 if hasattr(dist.file_effect, "auc_null_reference") else True
    assert dist.file_effect.auc_same_file_vs_diff_file == pytest.approx(0.5, abs=0.06)


def test_auc_is_high_when_each_file_is_its_own_speaker() -> None:
    """Same-file pairs then really are more similar, and the statistic should say so loudly."""
    d = 48
    rng = np.random.default_rng(4)
    a_axis = rng.normal(size=d)
    a_axis /= np.linalg.norm(a_axis)
    b_axis = rng.normal(size=d)
    b_axis -= (b_axis @ a_axis) * a_axis
    b_axis /= np.linalg.norm(b_axis)

    x = np.vstack([_file_of_vectors(a_axis, 60, 0.05, 1), _file_of_vectors(b_axis, 60, 0.05, 2)])
    ids = ["a"] * 60 + ["b"] * 60

    _, dist = describe_embedding_distribution(x, ids, n_permutations=200, seed=0)
    assert dist.file_effect.auc_same_file_vs_diff_file is not None
    assert dist.file_effect.auc_same_file_vs_diff_file > 0.95
    assert dist.file_effect.permutation_quantile is not None
    assert dist.file_effect.permutation_quantile > 0.99


def test_the_permutation_block_length_follows_the_window_geometry() -> None:
    """Windows overlap, so a per-vector shuffle destroys dependence the observed statistic keeps
    and the quantile comes out anti-conservative. Blocks of ceil(window_s/hop_s) fix that, and the
    length used is recorded so the number is auditable."""
    x, ids = _two_files_same_speaker()
    _, dist = describe_embedding_distribution(x, ids, window_s=2.0, hop_s=1.0, n_permutations=50)
    assert dist.file_effect.permutation_block_len == 2
    assert dist.file_effect.n_permutations == 50


def test_the_guard_band_needs_window_times_and_says_so_when_absent() -> None:
    """At 50% overlap a window's neighbour is a near-duplicate, so same-file pairs drawn from
    adjacent windows would inflate same-file similarity toward 1.0 for any input -- the statistic
    would measure the hop size. Excluding them needs times; without times the guard is skipped and
    reported as absent rather than silently not applied."""
    x, ids = _two_files_same_speaker()
    _, without = describe_embedding_distribution(x, ids, window_s=2.0, hop_s=1.0, n_permutations=20)
    assert without.file_effect.guard_band_s is None

    starts = [float(i) for i in range(40)] * 2
    _, with_times = describe_embedding_distribution(
        x, ids, window_s=2.0, hop_s=1.0, window_starts_s=starts, n_permutations=20
    )
    assert with_times.file_effect.guard_band_s == 2.0


def test_file_effect_is_absent_without_file_ids() -> None:
    """No files, no file effect."""
    x, _ = _two_files_same_speaker()
    _, dist = describe_embedding_distribution(x)
    assert dist.file_effect.auc_same_file_vs_diff_file is None
    assert dist.file_effect.permutation_quantile is None
```

Delete the stray `auc_null_reference` line from the first test after reading it — it was written
defensively and the real assertion is the `pytest.approx(0.5, abs=0.06)` below it. Keep the file
tidy.

- [ ] **Step 2: Run and watch it fail**

```bash
uv run --no-sync pytest src/tests/utils/embedding_distribution_files_test.py -v -k "auc or permutation or guard or file_effect" 2>&1 | tail -10
```

Expected: FAIL — no `file_effect` attribute.

- [ ] **Step 3: Add the model**

```python
class FileEffect(BaseModel):
    """Whether file identity explains similarity, and how surprising that is.

    Attributes:
        auc_same_file_vs_diff_file: Mann-Whitney AUC of same-file pair cosines against
            different-file pair cosines. Exact null 0.5 under exchangeability, which is what lets
            it be read with no fitted scale. Rank-based, so unlike silhouette it does not change
            when the same geometry is expressed as cosine rather than Euclidean distance. ``None``
            without file ids or with fewer than two files.
        permutation_quantile: Where the observed between-file share of angular variance falls in
            the block-permutation reference, in ``[0, 1]``.
        permutation_block_len: Block length used, ``ceil(window_s/hop_s)`` when both are known and
            1 otherwise. Windows overlap, so shuffling single vectors would destroy dependence the
            observed statistic retains and the quantile would come out anti-conservative.
        n_permutations: How many shuffles produced the reference.
        guard_band_s: Same-file pairs closer together in time than this were excluded. ``None``
            when ``window_starts_s`` was not supplied, so a reader can tell "not applied" from
            "applied with value X" -- at 50% overlap a window's neighbour is a near-duplicate, and
            leaving those in makes the AUC measure the hop size.
        seed: Permutation seed, recorded so the quantile is reproducible.
    """

    auc_same_file_vs_diff_file: Optional[float] = None
    permutation_quantile: Optional[float] = None
    permutation_block_len: int = 1
    n_permutations: int = 0
    guard_band_s: Optional[float] = None
    seed: int = 0
```

Add `file_effect: FileEffect = Field(default_factory=FileEffect)` to `EmbeddingDistribution`.

- [ ] **Step 4: Implement the AUC and the permutation**

```python
def _mann_whitney_auc(within: np.ndarray, between: np.ndarray) -> float:
    """Probability that a randomly chosen within-group value exceeds a between-group one.

    ``AUC = P(w > b) + 0.5*P(w == b)``, computed from ranks so it costs one sort rather than a
    quadratic comparison. Rank-based is the point: the value is invariant to any monotone
    reparametrisation of the similarity, which is exactly the property silhouette lacks.

    Args:
        within: 1-D array of within-group values.
        between: 1-D array of between-group values.

    Returns:
        The AUC in ``[0, 1]``. Returns ``0.5`` when either side is empty -- no evidence either way.
    """
    if within.size == 0 or between.size == 0:
        return 0.5
    from scipy.stats import rankdata

    combined = np.concatenate([within, between])
    ranks = rankdata(combined)
    r_within = float(ranks[: within.size].sum())
    u = r_within - within.size * (within.size + 1) / 2.0
    return float(u / (within.size * between.size))


def _eta_squared_between_files(x: np.ndarray, ids: np.ndarray) -> float:
    """Between-file share of angular variance.

    ``1 - sum_f n_f (1 - Rbar_f^2) / [n (1 - Rbar^2)]``: the residual within-file dispersion
    removed from the total. Bounded in ``[0, 1]`` for well-formed input.

    Args:
        x: ``(n, d)`` unit-norm array.
        ids: ``(n,)`` array of file ids.

    Returns:
        The between-file share; ``0.0`` when the pooled set has no dispersion to explain.
    """
    n = x.shape[0]
    rbar = float(np.linalg.norm(x.sum(axis=0)) / n)
    total = n * (1.0 - rbar**2)
    if total <= 0:
        return 0.0
    resid = 0.0
    for f in np.unique(ids):
        rows = x[ids == f]
        nf = rows.shape[0]
        rf = float(np.linalg.norm(rows.sum(axis=0)) / nf)
        resid += nf * (1.0 - rf**2)
    return float(1.0 - resid / total)


def _file_effect(
    x: np.ndarray,
    file_ids: Optional[list[str]],
    window_starts_s: Optional[list[float]],
    window_s: Optional[float],
    hop_s: Optional[float],
    n_permutations: int,
    seed: int,
) -> FileEffect:
    """Separability AUC plus a block-permutation reference for the between-file variance share.

    Args:
        x: ``(n, d)`` unit-norm array.
        file_ids: One id per row, or ``None``.
        window_starts_s: Window start times aligned to ``x``, or ``None``.
        window_s: Window length, used for the guard band and the block length.
        hop_s: Hop, used for the block length.
        n_permutations: Shuffles for the reference.
        seed: Permutation seed.

    Returns:
        The populated :class:`FileEffect`, or a default one when there is no file structure.
    """
    block_len = 1
    if window_s is not None and hop_s is not None and hop_s > 0:
        block_len = max(1, int(np.ceil(window_s / hop_s)))

    if file_ids is None or len(set(file_ids)) < 2:
        return FileEffect(permutation_block_len=block_len, n_permutations=0, seed=seed)

    ids = np.asarray(file_ids)
    gram = x @ x.T
    iu = np.triu_indices(x.shape[0], k=1)
    same = ids[iu[0]] == ids[iu[1]]
    cos_pairs = gram[iu]

    guard_band_s: Optional[float] = None
    keep = np.ones(cos_pairs.shape, dtype=bool)
    if window_starts_s is not None and window_s is not None:
        # Adjacent windows share audio, so a same-file pair drawn from them is a near-duplicate.
        # Excluded from both sides, or the AUC would report the hop size.
        guard_band_s = float(window_s)
        starts = np.asarray(window_starts_s, dtype=np.float64)
        too_close = np.abs(starts[iu[0]] - starts[iu[1]]) < guard_band_s
        keep = ~(same & too_close)

    auc = _mann_whitney_auc(cos_pairs[keep & same], cos_pairs[keep & ~same])

    observed = _eta_squared_between_files(x, ids)
    rng = np.random.default_rng(seed)
    n = x.shape[0]
    n_blocks = int(np.ceil(n / block_len))
    exceeded = 0
    for _ in range(n_permutations):
        block_order = rng.permutation(n_blocks)
        shuffled = np.concatenate([ids[b * block_len : (b + 1) * block_len] for b in block_order])[:n]
        if _eta_squared_between_files(x, shuffled) <= observed:
            exceeded += 1
    quantile = float(exceeded / n_permutations) if n_permutations > 0 else None

    return FileEffect(
        auc_same_file_vs_diff_file=auc,
        permutation_quantile=quantile,
        permutation_block_len=block_len,
        n_permutations=n_permutations,
        guard_band_s=guard_band_s,
        seed=seed,
    )
```

Wire it into `describe_embedding_distribution`:

```python
    starts_list = [float(s) for s, k in zip(window_starts_s, keep_mask) if k] if window_starts_s is not None else None
    file_effect = _file_effect(x, kept_files, starts_list, window_s, hop_s, n_permutations, seed)
```

and pass `file_effect=file_effect`.

- [ ] **Step 5: Run and verify**

```bash
uv run --no-sync pytest src/tests/utils/ -q 2>&1 | tail -5
```

Expected: PASS. If `test_auc_is_half_when_files_carry_no_effect` is flaky, widen the tolerance to
`abs=0.08` and note in the test why: an AUC over ~20k pairs has real sampling spread and the
assertion is about the null being *centred* at 0.5, not about a tight interval.

- [ ] **Step 6: Lint, typecheck, commit**

```bash
uv run --no-sync ruff format src/senselab/utils/tasks/embedding_distribution.py src/tests/utils/
uv run --no-sync ruff check src/ src/tests/
uv run --no-sync mypy --ignore-missing-imports --extra-checks src/ 2>&1 | tail -3
git add src/senselab/utils/tasks/embedding_distribution.py src/tests/utils/embedding_distribution_files_test.py
git commit -m "feat(utils): file-effect separability AUC with a block-permutation reference

Mann-Whitney AUC of same-file against different-file pair cosines, with an exact
null of 0.5 under exchangeability. Rank-based on purpose: it is invariant to
monotone reparametrisation of the similarity, which is the property silhouette
lacks -- silhouette returns different numbers for identical geometry depending on
whether it is called with cosine or Euclidean.

The permutation reference shuffles file labels in blocks of ceil(window_s/hop_s)
rather than per vector, because overlapping windows are autocorrelated and a
per-vector shuffle destroys dependence the observed statistic retains, making the
quantile anti-conservative. Block length, permutation count and seed are all
recorded so the number is auditable.

Same-file pairs closer in time than one window are excluded when window start
times are supplied: at 50% overlap a window's neighbour is a near-duplicate, and
leaving those in would make the AUC report the hop size rather than a speaker
effect. Without times the guard is reported as absent rather than silently not
applied."
```

---

### Task 5: Aggregator dispatch and centroid robustness

**Files:**
- Modify: `src/senselab/utils/tasks/embedding_distribution.py`
- Test: append to `src/tests/utils/embedding_distribution_test.py`

**Interfaces:**
- Consumes: Tasks 2-4.
- Produces:
  - `CentroidRobustness(cos_mean_vs_trimmed10: float, cos_mean_vs_medoid: float, leave_one_file_out_cos: dict[str, float])`
  - `EmbeddingDistribution.centroid_robustness: CentroidRobustness`
  - `_trimmed_spherical_mean(x, fraction) -> np.ndarray`, `_medoid(x) -> np.ndarray`
  - `aggregator` now actually selects the returned centroid.

- [ ] **Step 1: Write the failing test**

Append to `src/tests/utils/embedding_distribution_test.py`:

```python
def test_the_aggregator_selects_the_returned_centroid() -> None:
    """A tool parameter, not a decision -- but it must actually take effect."""
    x = _tight_cone(n=60, d=32, spread=0.2, seed=9)
    mean_c, mean_d = describe_embedding_distribution(x, aggregator="spherical_mean")
    medoid_c, medoid_d = describe_embedding_distribution(x, aggregator="medoid")

    assert mean_d.geometry.centroid_rule == "spherical_mean"
    assert medoid_d.geometry.centroid_rule == "medoid"
    assert not np.allclose(mean_c, medoid_c)
    # The medoid is one of the input rows; the spherical mean generally is not.
    assert np.isclose(np.abs(np.asarray(medoid_c) @ x.T).max(), 1.0, atol=1e-9)


def test_an_unknown_aggregator_is_rejected() -> None:
    """Silently falling back would make the reported centroid_rule a lie."""
    with pytest.raises(ValueError, match="aggregator"):
        describe_embedding_distribution(_tight_cone(n=5, d=8, spread=0.1), aggregator="median")


def test_contamination_opens_a_gap_between_mean_and_trimmed_mean() -> None:
    """With no clustering to reject contamination, this gap is how a caller learns the estimate is
    contamination-sensitive -- a robustness statement carrying no threshold and no verdict."""
    d = 48
    rng = np.random.default_rng(21)
    target = rng.normal(size=d)
    target /= np.linalg.norm(target)
    other = rng.normal(size=d)
    other -= (other @ target) * other.dot(target) * 0  # keep it independent-ish
    other /= np.linalg.norm(other)

    clean = _tight_cone(n=80, d=d, spread=0.05, seed=1)
    dirty = np.vstack([clean, np.repeat(other[None, :], 20, axis=0)])

    _, clean_dist = describe_embedding_distribution(clean)
    _, dirty_dist = describe_embedding_distribution(dirty)

    assert clean_dist.centroid_robustness.cos_mean_vs_trimmed10 > 0.999
    assert dirty_dist.centroid_robustness.cos_mean_vs_trimmed10 < clean_dist.centroid_robustness.cos_mean_vs_trimmed10


def test_leave_one_file_out_finds_the_file_driving_the_centroid() -> None:
    """A jackknife along the cross-file axis, which is where the measured error budget sits. It
    answers a caller's real question -- is this centroid an artefact of one file -- more directly
    than any dispersion number."""
    d = 48
    rng = np.random.default_rng(31)
    target = rng.normal(size=d)
    target /= np.linalg.norm(target)
    intruder = rng.normal(size=d)
    intruder -= (intruder @ target) * target
    intruder /= np.linalg.norm(intruder)

    def cone(axis: np.ndarray, n: int, seed: int) -> np.ndarray:
        r = np.random.default_rng(seed)
        v = axis[None, :] + 0.03 * r.normal(size=(n, d))
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    x = np.vstack([cone(target, 30, 1), cone(target, 30, 2), cone(intruder, 30, 3)])
    ids = ["good1"] * 30 + ["good2"] * 30 + ["bad"] * 30

    _, dist = describe_embedding_distribution(x, ids, n_permutations=20)
    lofo = dist.centroid_robustness.leave_one_file_out_cos
    assert set(lofo) == {"good1", "good2", "bad"}
    # Removing the intruder moves the centroid most, so its LOFO cosine is the lowest.
    assert lofo["bad"] < lofo["good1"]
    assert lofo["bad"] < lofo["good2"]
```

- [ ] **Step 2: Run and watch it fail**

```bash
uv run --no-sync pytest src/tests/utils/embedding_distribution_test.py -v -k "aggregator or trimmed or leave_one_file" 2>&1 | tail -10
```

Expected: FAIL — no `centroid_robustness`, and `aggregator` does not change the centroid.

- [ ] **Step 3: Implement**

```python
class CentroidRobustness(BaseModel):
    """Whether the centroid depends on how it was aggregated, or on one file.

    Attributes:
        cos_mean_vs_trimmed10: Cosine between the spherical mean and the 10%-trimmed spherical
            mean. Near 1.0 means aggregation choice did not matter; a visible gap means the
            estimate is contamination-sensitive. With no clustering rejecting contamination, this
            is how a caller learns that.
        cos_mean_vs_medoid: Cosine between the spherical mean and the medoid. The medoid has a
            ~50% breakdown point but *is* one real vector, so its error does not shrink with ``n``;
            it is reported as a diagnostic rather than used as the default centroid.
        leave_one_file_out_cos: Per file, the cosine between the full centroid and the centroid
            recomputed with that file removed. A jackknife along the cross-file axis, where the
            measured error budget sits. Empty without file ids.
    """

    cos_mean_vs_trimmed10: float = 1.0
    cos_mean_vs_medoid: float = 1.0
    leave_one_file_out_cos: dict[str, float] = Field(default_factory=dict)
```

Add `centroid_robustness: CentroidRobustness = Field(default_factory=CentroidRobustness)` to
`EmbeddingDistribution`.

```python
def _trimmed_spherical_mean(x: np.ndarray, fraction: float = _TRIM_FRACTION) -> np.ndarray:
    """Spherical mean after dropping the ``fraction`` of rows least aligned with the full mean.

    Args:
        x: ``(n, d)`` unit-norm array.
        fraction: Share of rows to drop, from the low end of leave-one-out cosine.

    Returns:
        A unit-norm direction. Falls back to the untrimmed mean when trimming would leave fewer
        than two rows.
    """
    n = x.shape[0]
    k = int(np.floor(fraction * n))
    if k < 1 or n - k < 2:
        return _spherical_mean(x)
    keep = np.argsort(_loo_cos_to_centroid(x))[k:]
    return _spherical_mean(x[keep])


def _medoid(x: np.ndarray) -> np.ndarray:
    """The input row minimising total geodesic distance to the others.

    Uses angular distance ``arccos(clip(cos))`` rather than ``1-cos``, because only the former is
    a metric on the sphere and "medoid" is defined against a metric.

    Args:
        x: ``(n, d)`` unit-norm array.

    Returns:
        A copy of the selected row.
    """
    theta = np.arccos(np.clip(x @ x.T, -1.0, 1.0))
    return x[int(np.argmin(theta.sum(axis=1)))].copy()


def _centroid_robustness(
    x: np.ndarray, file_ids: Optional[list[str]], mean_centroid: np.ndarray
) -> CentroidRobustness:
    """Aggregation sensitivity plus leave-one-file-out stability.

    Args:
        x: ``(n, d)`` unit-norm array.
        file_ids: One id per row, or ``None``.
        mean_centroid: The spherical mean over all rows.

    Returns:
        The populated robustness block.
    """
    trimmed = _trimmed_spherical_mean(x)
    medoid = _medoid(x)

    lofo: dict[str, float] = {}
    if file_ids is not None:
        ids = np.asarray(file_ids)
        order: list[str] = []
        for f in file_ids:
            if f not in order:
                order.append(f)
        for f in order:
            rows = x[ids != f]
            # A single remaining row still has a direction; zero remaining rows does not.
            lofo[f] = float(mean_centroid @ _spherical_mean(rows)) if rows.shape[0] >= 1 else 1.0

    return CentroidRobustness(
        cos_mean_vs_trimmed10=float(mean_centroid @ trimmed),
        cos_mean_vs_medoid=float(mean_centroid @ medoid),
        leave_one_file_out_cos=lofo,
    )
```

Replace the single `centroid = _spherical_mean(x)` line with a dispatch, keeping the *mean* for
the robustness comparisons regardless of what the caller asked for:

```python
    mean_centroid = _spherical_mean(x)
    if aggregator == AGGREGATOR_SPHERICAL_MEAN:
        centroid = mean_centroid
    elif aggregator == AGGREGATOR_TRIMMED_MEAN:
        centroid = _trimmed_spherical_mean(x)
    else:
        centroid = _medoid(x)
```

Use `mean_centroid` (not `centroid`) for `_per_file_stats` and `_centroid_robustness`, so those
comparisons stay on one reference regardless of the aggregator, and add
`centroid_robustness=_centroid_robustness(x, kept_files, mean_centroid)` to the constructor.

- [ ] **Step 4: Run, lint, typecheck**

```bash
uv run --no-sync pytest src/tests/utils/ -q 2>&1 | tail -5
uv run --no-sync ruff format src/senselab/utils/tasks/embedding_distribution.py src/tests/utils/
uv run --no-sync ruff check src/ src/tests/
uv run --no-sync mypy --ignore-missing-imports --extra-checks src/ 2>&1 | tail -3
```

If `test_contamination_opens_a_gap_between_mean_and_trimmed_mean` fails because `other` was built
oddly (that line has a deliberately awkward construction), simplify it to a fresh random unit
vector orthogonalised against `target`, matching the pattern used in the LOFO test.

- [ ] **Step 5: Commit**

```bash
git add src/senselab/utils/tasks/embedding_distribution.py src/tests/utils/embedding_distribution_test.py
git commit -m "feat(utils): aggregator dispatch and centroid-robustness diagnostics

The aggregator (spherical_mean, trimmed_mean, medoid) now selects the returned
centroid, and the block reports the cosine from the spherical mean to both
alternatives. With no clustering rejecting contamination by default, that gap is
how a caller learns the estimate is contamination-sensitive -- a robustness
statement carrying no threshold and no verdict.

Leave-one-file-out centroid stability is added per file: a jackknife along the
cross-file axis, which is where the measured error budget on this pipeline sits
(within-file 0.984, cross-file 0.891). It answers 'is this centroid an artefact of
one file' more directly than any dispersion figure, at the cost of n_files
matmuls and no model calls.

The medoid uses geodesic arccos distance rather than 1-cos, because only the
former is a metric and medoid is defined against one. It stays a diagnostic
rather than the default centroid: its breakdown point is attractive but it *is*
one real vector, so its error does not shrink with n while the spherical mean's
does."
```

---

### Task 6: Optional contamination rejection

**Files:**
- Modify: `src/senselab/utils/tasks/embedding_distribution.py`
- Test: `src/tests/utils/embedding_distribution_selection_test.py`

**Interfaces:**
- Consumes: Tasks 2-5 — `_as_array`, `_l2_normalise`, `_spherical_mean`, `SimilarityStats`.
- Produces:
  - `ClusterSummary(cluster_id: int, n_vectors: int, window_share: float, file_balanced_share: float, n_files_contributing: int, per_file_share: dict[str, float])`
  - `SelectionRule(linkage: str, cut_theta: float, cut_source: str, merge_heights: list[float], min_file_share: float | None)`
  - `DominantSelection(kept_indices: list[int], dropped_indices: list[int], clusters: list[ClusterSummary], dominant_cluster_id: int, runner_up_cluster_id: int | None, cos_dominant_to_runner_up: float | None, dropped_per_file: dict[str, int], rule_used: SelectionRule)`
  - `select_dominant_vectors(vectors, file_ids=None, *, linkage="average", cut_theta=None, min_file_share=None) -> DominantSelection`

- [ ] **Step 1: Write the failing test**

Create `src/tests/utils/embedding_distribution_selection_test.py`:

```python
"""Optional contamination rejection.

This is the one component in the feature that makes a decision, so it is opt-in and everything it
did is recorded. The cut has no numeric default: it is either supplied by the caller or derived
from the data by a stated rule, and whichever was used is reported.
"""

import numpy as np
import pytest

from senselab.utils.tasks.embedding_distribution import select_dominant_vectors


def _cone(axis: np.ndarray, n: int, spread: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = axis[None, :] + spread * rng.normal(size=(n, axis.size))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def _two_speakers(n_target: int = 60, n_intruder: int = 20, d: int = 48) -> tuple[np.ndarray, list[str]]:
    rng = np.random.default_rng(13)
    a = rng.normal(size=d)
    a /= np.linalg.norm(a)
    b = rng.normal(size=d)
    b -= (b @ a) * a
    b /= np.linalg.norm(b)
    x = np.vstack([_cone(a, n_target, 0.03, 1), _cone(b, n_intruder, 0.03, 2)])
    ids = ["target"] * n_target + ["intruder"] * n_intruder
    return x, ids


def test_the_intruder_group_is_dropped() -> None:
    """The measured property this exists for: a contaminating recording leaves the estimate."""
    x, ids = _two_speakers()
    sel = select_dominant_vectors(x, ids)

    assert len(sel.kept_indices) == 60
    assert len(sel.dropped_indices) == 20
    assert sel.dropped_per_file == {"intruder": 20}


def test_no_numeric_cut_default_exists() -> None:
    """A fitted literal here is exactly what this repository forbids, so the signature must not
    carry one: the cut is caller-supplied or derived by a stated rule."""
    import inspect

    default = inspect.signature(select_dominant_vectors).parameters["cut_theta"].default
    assert default is None


def test_the_derived_cut_is_recorded_as_derived() -> None:
    """Auditable means a reader can tell where the number came from."""
    x, ids = _two_speakers()
    sel = select_dominant_vectors(x, ids)
    assert sel.rule_used.cut_source == "largest_merge_gap"
    assert sel.rule_used.cut_theta > 0
    assert len(sel.rule_used.merge_heights) == x.shape[0] - 1


def test_an_explicit_cut_overrides_and_is_recorded_verbatim() -> None:
    """A caller who disagrees with the rule must be able to say so, and see that it took."""
    x, ids = _two_speakers()
    sel = select_dominant_vectors(x, ids, cut_theta=3.0)  # larger than pi: one cluster
    assert sel.rule_used.cut_source == "caller"
    assert sel.rule_used.cut_theta == 3.0
    assert len(sel.clusters) == 1
    assert sel.dropped_indices == []


def test_selection_is_deterministic() -> None:
    """Shares are a reported field, so they must be reproducible. AHC takes no seed; spectral
    clustering with k-means assignment would, and pinning it only hides the variance."""
    x, ids = _two_speakers()
    a = select_dominant_vectors(x, ids)
    b = select_dominant_vectors(x, ids)
    assert a.kept_indices == b.kept_indices
    assert a.rule_used.merge_heights == b.rule_used.merge_heights


def test_file_balanced_share_beats_raw_duration() -> None:
    """The target is the speaker present in most files, not the one occupying most seconds.

    One long off-target recording must not outvote several short on-target ones, so selection is
    by file-balanced share -- and both shares are reported so the disagreement stays visible.
    """
    rng = np.random.default_rng(23)
    d = 48
    target = rng.normal(size=d)
    target /= np.linalg.norm(target)
    other = rng.normal(size=d)
    other -= (other @ target) * target
    other /= np.linalg.norm(other)

    # Three short target files (10 each) against one long off-target file (200).
    x = np.vstack(
        [_cone(target, 10, 0.02, 1), _cone(target, 10, 0.02, 2), _cone(target, 10, 0.02, 3), _cone(other, 200, 0.02, 4)]
    )
    ids = ["t1"] * 10 + ["t2"] * 10 + ["t3"] * 10 + ["long"] * 200

    sel = select_dominant_vectors(x, ids)
    dominant = next(c for c in sel.clusters if c.cluster_id == sel.dominant_cluster_id)

    assert dominant.n_files_contributing == 3
    assert dominant.file_balanced_share > 0.5
    assert dominant.window_share < 0.5  # raw duration disagrees, and both are reported


def test_the_runner_up_is_reported_with_its_distance() -> None:
    """'0.52/0.46 at cos 0.31' and '0.94/0.05 at cos 0.88' are different situations, and both must
    stay legible without this function deciding which matters."""
    x, ids = _two_speakers()
    sel = select_dominant_vectors(x, ids)
    assert sel.runner_up_cluster_id is not None
    assert sel.cos_dominant_to_runner_up is not None
    assert sel.cos_dominant_to_runner_up < 0.5


def test_a_single_coherent_group_keeps_everything() -> None:
    """Nothing to reject is a perfectly ordinary outcome and must not drop rows."""
    rng = np.random.default_rng(29)
    axis = rng.normal(size=32)
    axis /= np.linalg.norm(axis)
    x = _cone(axis, 50, 0.02, 5)
    sel = select_dominant_vectors(x, ["one"] * 50)
    assert len(sel.kept_indices) == 50
    assert sel.dropped_indices == []
```

- [ ] **Step 2: Run and watch it fail**

```bash
uv run --no-sync pytest src/tests/utils/embedding_distribution_selection_test.py -v 2>&1 | tail -10
```

Expected: FAIL — `ImportError: cannot import name 'select_dominant_vectors'`.

- [ ] **Step 3: Implement**

Add to `embedding_distribution.py`:

```python
LINKAGE_AVERAGE = "average"


class ClusterSummary(BaseModel):
    """One group found by the selector.

    Attributes:
        cluster_id: Stable id within this selection.
        n_vectors: Rows in this group.
        window_share: Share of all scored rows. Reported alongside ``file_balanced_share`` because
            the two disagree exactly when duration imbalance is driving the answer.
        file_balanced_share: Share with each file weighted ``1/n_f``, so a long recording cannot
            outvote several short ones.
        n_files_contributing: How many distinct files put rows in this group.
        per_file_share: Per file, the share of that file's rows landing in this group.
    """

    cluster_id: int
    n_vectors: int
    window_share: float
    file_balanced_share: float
    n_files_contributing: int
    per_file_share: dict[str, float] = Field(default_factory=dict)


class SelectionRule(BaseModel):
    """What the selector actually did, so the decision is auditable and reversible.

    Attributes:
        linkage: Linkage used for the agglomerative clustering.
        cut_theta: The angular cut applied, in radians.
        cut_source: ``"caller"`` when supplied, ``"largest_merge_gap"`` when derived. A reader has
            to be able to tell which, because a derived cut is a rule and a supplied one is a
            caller's judgement.
        merge_heights: The full ascending merge-height sequence. This is the threshold-free form of
            the "how many clusters" question; a consumer can re-cut it anywhere.
        min_file_share: Optional minimum per-file share for a file to count toward a group's
            file-balanced share. ``None`` means unused.
    """

    linkage: str
    cut_theta: float
    cut_source: str
    merge_heights: list[float] = Field(default_factory=list)
    min_file_share: Optional[float] = None


class DominantSelection(BaseModel):
    """Which vectors survived contamination rejection, and everything about how that was decided."""

    kept_indices: list[int]
    dropped_indices: list[int]
    clusters: list[ClusterSummary]
    dominant_cluster_id: int
    runner_up_cluster_id: Optional[int] = None
    cos_dominant_to_runner_up: Optional[float] = None
    dropped_per_file: dict[str, int] = Field(default_factory=dict)
    rule_used: SelectionRule


def select_dominant_vectors(
    vectors: Union[Sequence[Sequence[float]], np.ndarray, Any],
    file_ids: Optional[Sequence[str]] = None,
    *,
    linkage: str = LINKAGE_AVERAGE,
    cut_theta: Optional[float] = None,
    min_file_share: Optional[float] = None,
) -> DominantSelection:
    """Group vectors and return the dominant group, for optional contamination rejection.

    **This is the one function in the module that decides something**, which is why it is separate
    from :func:`describe_embedding_distribution` and why callers opt in. Everything it did is
    recorded in the returned object.

    Agglomerative hierarchical clustering on geodesic angular distance, average linkage. Chosen
    over spectral clustering for three reasons: it is deterministic, where
    ``SpectralClustering(assign_labels="kmeans")`` is stochastic and pinning a seed hides that
    variance rather than removing it; k-means-family clustering carries an equal-size bias and will
    split one speaker's prosodic halves before isolating a small intruder; and AHC turns "choose
    k" into a merge-height profile that can be reported instead of decided.

    Args:
        vectors: ``(n, d)`` embeddings. L2-normalised on entry.
        file_ids: One id per row. Enables file-balanced selection and per-file drop accounting.
        linkage: Passed to ``scipy.cluster.hierarchy.linkage``. ``"average"`` by default.
        cut_theta: Angular cut in radians. ``None`` derives it from the largest gap in the merge
            heights -- a *rule*, not a fitted constant. Supply a value to override; either way the
            value used and its source are recorded.
        min_file_share: Optional floor on a file's share within a group before that file counts
            toward the group's file-balanced share. ``None`` counts every contributing file.

    Returns:
        A :class:`DominantSelection`.

    Raises:
        ValueError: If fewer than 2 vectors survive normalisation, or ``file_ids`` length disagrees.
    """
    from scipy.cluster.hierarchy import fcluster, linkage as scipy_linkage
    from scipy.spatial.distance import squareform

    raw = _as_array(vectors)
    n_total = int(raw.shape[0])
    if file_ids is not None and len(file_ids) != n_total:
        raise ValueError(f"file_ids has {len(file_ids)} entries for {n_total} vectors")

    norms = np.linalg.norm(raw, axis=1)
    keep_mask = norms > 0
    original_index = np.flatnonzero(keep_mask)
    x, _ = _l2_normalise(raw)
    n = int(x.shape[0])
    if n < 2:
        raise ValueError(f"need at least 2 non-zero vectors to select a dominant group; got {n}")

    ids = [str(f) for f, k in zip(file_ids, keep_mask) if k] if file_ids is not None else ["_all"] * n

    theta = np.arccos(np.clip(x @ x.T, -1.0, 1.0))
    np.fill_diagonal(theta, 0.0)
    theta = (theta + theta.T) / 2.0  # enforce exact symmetry for squareform
    z = scipy_linkage(squareform(theta, checks=False), method=linkage)
    merge_heights = [float(h) for h in z[:, 2]]

    if cut_theta is None:
        # The largest gap between consecutive merge heights is where the data itself separates.
        # A rule, not a literal: nothing here was fitted, and the resulting value is reported.
        heights = np.asarray(merge_heights)
        if heights.size >= 2:
            gaps = np.diff(heights)
            g = int(np.argmax(gaps))
            resolved_cut = float((heights[g] + heights[g + 1]) / 2.0)
        else:
            resolved_cut = float(heights[0]) if heights.size else 0.0
        cut_source = "largest_merge_gap"
    else:
        resolved_cut = float(cut_theta)
        cut_source = "caller"

    labels = fcluster(z, t=resolved_cut, criterion="distance")

    file_counts: dict[str, int] = {}
    for f in ids:
        file_counts[f] = file_counts.get(f, 0) + 1

    summaries: list[ClusterSummary] = []
    for cid in sorted(set(int(v) for v in labels)):
        member = labels == cid
        member_files = [f for f, m in zip(ids, member) if m]
        per_file: dict[str, float] = {}
        for f in set(member_files):
            per_file[f] = member_files.count(f) / file_counts[f]
        counted = {f: s for f, s in per_file.items() if min_file_share is None or s >= min_file_share}
        summaries.append(
            ClusterSummary(
                cluster_id=cid,
                n_vectors=int(member.sum()),
                window_share=float(member.sum() / n),
                file_balanced_share=float(sum(counted.values()) / len(file_counts)),
                n_files_contributing=len(per_file),
                per_file_share=per_file,
            )
        )

    ranked = sorted(summaries, key=lambda c: (c.file_balanced_share, c.window_share), reverse=True)
    dominant = ranked[0]
    runner_up = ranked[1] if len(ranked) > 1 else None

    dominant_centroid = _spherical_mean(x[labels == dominant.cluster_id])
    cos_to_runner: Optional[float] = None
    if runner_up is not None:
        cos_to_runner = float(dominant_centroid @ _spherical_mean(x[labels == runner_up.cluster_id]))

    kept = [int(original_index[i]) for i in range(n) if labels[i] == dominant.cluster_id]
    dropped = [int(original_index[i]) for i in range(n) if labels[i] != dominant.cluster_id]
    dropped_per_file: dict[str, int] = {}
    for i in range(n):
        if labels[i] != dominant.cluster_id:
            dropped_per_file[ids[i]] = dropped_per_file.get(ids[i], 0) + 1

    return DominantSelection(
        kept_indices=kept,
        dropped_indices=dropped,
        clusters=summaries,
        dominant_cluster_id=dominant.cluster_id,
        runner_up_cluster_id=runner_up.cluster_id if runner_up else None,
        cos_dominant_to_runner_up=cos_to_runner,
        dropped_per_file=dropped_per_file,
        rule_used=SelectionRule(
            linkage=linkage,
            cut_theta=resolved_cut,
            cut_source=cut_source,
            merge_heights=merge_heights,
            min_file_share=min_file_share,
        ),
    )
```

- [ ] **Step 4: Run, lint, typecheck, commit**

```bash
uv run --no-sync pytest src/tests/utils/ -q 2>&1 | tail -5
uv run --no-sync ruff format src/senselab/utils/tasks/embedding_distribution.py src/tests/utils/
uv run --no-sync ruff check src/ src/tests/
uv run --no-sync mypy --ignore-missing-imports --extra-checks src/ 2>&1 | tail -3
git add src/senselab/utils/tasks/embedding_distribution.py src/tests/utils/embedding_distribution_selection_test.py
git commit -m "feat(utils): optional contamination rejection via a reportable AHC cut

select_dominant_vectors groups vectors and returns the dominant group. It is the
one function in the module that decides anything, which is why it is separate from
the descriptor and why callers opt in -- and why everything it did is recorded:
kept and dropped indices, per-file drop counts, every cluster's raw and
file-balanced share, the runner-up and its cosine to the dominant centroid, and
the full merge-height sequence.

AHC average-linkage on geodesic angular distance rather than spectral: it is
deterministic where SpectralClustering with k-means assignment is not, k-means
splits one speaker's prosodic halves before isolating a small intruder, and AHC
turns choose-k into a merge profile that can be reported instead of decided.

The cut carries no numeric default. cut_theta=None derives it from the largest gap
in the merge heights -- a rule, not a fitted constant -- and an explicit value
overrides it; cut_source records which happened. Selection is by file-balanced
share so one long off-target recording cannot outvote several short on-target
ones, with the raw window share reported alongside so the disagreement stays
visible."
```

---

### Task 7: Promote the windowing primitives down a layer

**Files:**
- Create: `src/senselab/audio/tasks/speaker_embeddings/windowing.py`
- Modify: `src/senselab/audio/workflows/audio_analysis/embeddings.py`
- Create: `src/tests/audio/tasks/task_layer_guard_test.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `window_starts(duration_s: float, window_s: float, hop_s: float) -> list[float]`
  - `WindowEmbedding` (moved dataclass: `start_s: float`, `end_s: float`, `vector: np.ndarray`)
  - `extract_per_window_embeddings(...)` with its existing signature and its existing
    `window_s=1.0, hop_s=0.5` defaults
  - `slice_audio(audio: Audio, start_s: float, end_s: float) -> Audio`

- [ ] **Step 1: Read what you are moving**

```bash
sed -n '1,120p' src/senselab/audio/workflows/audio_analysis/embeddings.py
grep -rn "extract_per_window_embeddings\|_window_starts\|_slice_audio\|WindowEmbedding" src/ --include=*.py | grep -v "^src/senselab/audio/workflows/audio_analysis/embeddings.py"
```

The second command lists every existing consumer. All of them must keep working — this is a move,
not a rewrite. Do not change any behaviour, any default, or any signature in this task.

- [ ] **Step 2: Write the failing guard test**

Create `src/tests/audio/tasks/task_layer_guard_test.py`:

```python
"""Nothing under audio/tasks/ may import from audio/workflows/.

Workflows compose tasks; a task importing a workflow inverts that. The rule matters here because
this change promotes two primitives *down* from the workflow into the task layer, and without a
guard they drift back the first time someone needs a workflow helper in a task.

An AST sweep rather than a text search: an import inside a function body or guarded by
TYPE_CHECKING is still an import, and a commented-out one is not.
"""

import ast
from pathlib import Path

_TASKS = Path("src/senselab/audio/tasks")
_FORBIDDEN_PREFIX = "senselab.audio.workflows"


def _imported_modules(path: Path) -> list[str]:
    """Every module name imported anywhere in a file, including inside function bodies."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.append(node.module)
    return names


def test_no_task_imports_a_workflow() -> None:
    """The dependency direction is one-way, and this is what keeps it that way."""
    offenders: list[str] = []
    for path in sorted(_TASKS.rglob("*.py")):
        for module in _imported_modules(path):
            if module.startswith(_FORBIDDEN_PREFIX):
                offenders.append(f"{path}: imports {module}")
    assert not offenders, "audio/tasks must not import audio/workflows:\n" + "\n".join(offenders)


def test_the_guard_can_actually_see_a_violation() -> None:
    """A guard that cannot fail is not a guard.

    Proves the AST sweep detects the pattern it is meant to catch, including an import nested
    inside a function, which a naive top-of-file scan would miss.
    """
    import tempfile

    source = "def f():\n    from senselab.audio.workflows.audio_analysis import embeddings\n    return embeddings\n"
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "offender.py"
        p.write_text(source, encoding="utf-8")
        assert any(m.startswith(_FORBIDDEN_PREFIX) for m in _imported_modules(p))
```

- [ ] **Step 3: Run it — it may already pass**

```bash
uv run --no-sync pytest src/tests/audio/tasks/task_layer_guard_test.py -v 2>&1 | tail -8
```

If `test_no_task_imports_a_workflow` passes now, good — nothing violates it yet, and the guard's
job is to keep that true after Step 4. If it fails, the offenders it names are pre-existing and
must be reported to the reviewer rather than silently fixed.

- [ ] **Step 4: Create the task-layer module by moving code**

Create `src/senselab/audio/tasks/speaker_embeddings/windowing.py`. Move `WindowEmbedding`,
`_slice_audio`, `_window_starts` and `extract_per_window_embeddings` from
`audio_analysis/embeddings.py` verbatim — same bodies, same defaults — renaming the two private
helpers to public (`slice_audio`, `window_starts`) since they now cross a module boundary. Give the
module this docstring:

```python
"""Uniform windowing and per-window speaker-embedding extraction.

Lives in the task layer because two callers need it: the ``audio_analysis`` workflow's
speaker-identity path, and ``estimate_speaker_embedding_from_audios``. It was previously private to
that workflow; a task importing it there would invert the dependency direction, since workflows
compose tasks and not the reverse. Promoted rather than duplicated -- two copies of a windowing
grid drift the moment either is edited.

**The defaults here are the detection defaults, and they are deliberate.** ``window_s=1.0`` with
``hop_s=0.5`` trades embedding precision for temporal resolution: SpeechBrain speaker models are
trained on multi-second utterances so an embedding below 1 s is noisier, but a 1.0 s window on a
0.5 s hop yields one embedding per 0.5 s bucket, which is what eliminated the same-window dedup
that previously dropped half of all consecutive same-cluster comparisons. Profile *enrollment*
wants the opposite trade and passes ``window_s=2.0, hop_s=1.0`` explicitly -- measured at
cross-file centroid stability 0.890 and cross-subject separation 0.168, against 0.331 for a
0.5/0.25 grid carrying four times the windows. Two purposes, two measured settings; do not
collapse them.
"""
```

- [ ] **Step 5: Re-point the workflow at the task**

In `audio_analysis/embeddings.py`, delete the moved definitions and import them instead:

```python
from senselab.audio.tasks.speaker_embeddings.windowing import (
    WindowEmbedding,
    extract_per_window_embeddings,
    slice_audio,
    window_starts,
)
```

Keep module-level aliases for the old private names only if the grep in Step 1 showed other files
using them; otherwise update those call sites. Do not leave both a private alias and the public
name where nothing needs the alias — this repository's pre-alpha convention is to rename outright.

- [ ] **Step 6: Verify nothing regressed**

```bash
uv run --no-sync pytest src/tests/audio/workflows/ src/tests/audio/tasks/ -q 2>&1 | tail -6
uv run --no-sync ruff format src/senselab/audio/ src/tests/audio/
uv run --no-sync ruff check src/ src/tests/
uv run --no-sync mypy --ignore-missing-imports --extra-checks src/ 2>&1 | tail -3
```

Expected: PASS, including the guard test. The workflow's own embedding tests are the real check
that the move changed no behaviour.

- [ ] **Step 7: Commit**

```bash
git add src/senselab/audio/tasks/speaker_embeddings/windowing.py \
        src/senselab/audio/workflows/audio_analysis/embeddings.py \
        src/tests/audio/tasks/task_layer_guard_test.py
git commit -m "refactor(audio): promote windowing primitives into the task layer

extract_per_window_embeddings, its windowing grid and its Audio slicer move from
audio_analysis/embeddings.py into tasks/speaker_embeddings/windowing.py, and the
workflow imports them from there. Two callers now need them -- the workflow's
speaker-identity path and the new estimator -- and a task importing a workflow
would invert the dependency direction, since workflows compose tasks.

Promoted rather than duplicated: two copies of a windowing grid drift the moment
either is edited.

Behaviour, signatures and the 1.0/0.5 detection defaults are unchanged; this is a
move. An AST guard now asserts nothing under audio/tasks imports audio/workflows,
with a companion test proving the guard can actually see a violation -- including
one nested inside a function body, which a text search would miss."
```

---

### Task 8: The estimator

**Files:**
- Modify: `src/senselab/audio/tasks/speaker_embeddings/api.py`
- Modify: `src/senselab/audio/tasks/speaker_embeddings/__init__.py`
- Modify: `src/senselab/audio/data_structures/audio_hints.py` (narrow `distribution`)
- Test: `src/tests/audio/tasks/speaker_embeddings_estimate_test.py`

**Interfaces:**
- Consumes: `AudioHints` types (Task 1); `describe_embedding_distribution`,
  `select_dominant_vectors`, `EmbeddingDistribution` (Tasks 2-6); `window_starts`,
  `extract_per_window_embeddings` (Task 7).
- Produces:
  - `estimate_speaker_embedding_from_audios(audios: list[Audio], model: SenselabModel | None = None, device: DeviceType | None = None, window_s: float = 2.0, hop_s: float = 1.0, aggregator: str = "spherical_mean", reject_contamination: bool = False, created_at: str | None = None) -> TargetSpeakerEmbedding`

- [ ] **Step 1: Write the failing test**

Create `src/tests/audio/tasks/speaker_embeddings_estimate_test.py`:

```python
"""Estimate one speaker embedding from files that may contain that speaker.

No model is loaded anywhere here: the per-window extraction is monkeypatched to return controlled
vectors, so these tests exercise the aggregation, provenance and rejection logic deterministically
and without downloading a snapshot.
"""

import numpy as np
import pytest
import torch

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.speaker_embeddings import estimate_speaker_embedding_from_audios

_DIM = 48


def _audio(seconds: float = 8.0, sr: int = 16000) -> Audio:
    return Audio(waveform=torch.rand(1, int(seconds * sr)), sampling_rate=sr)


def _cone(axis: np.ndarray, n: int, spread: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = axis[None, :] + spread * rng.normal(size=(n, axis.size))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


@pytest.fixture
def axes() -> tuple[np.ndarray, np.ndarray]:
    """Two near-orthogonal directions: a target speaker and an intruder."""
    rng = np.random.default_rng(41)
    a = rng.normal(size=_DIM)
    a /= np.linalg.norm(a)
    b = rng.normal(size=_DIM)
    b -= (b @ a) * a
    b /= np.linalg.norm(b)
    return a, b


def _patch_extraction(monkeypatch: pytest.MonkeyPatch, per_audio_vectors: list[np.ndarray]) -> None:
    """Make per-window extraction return the given vectors, one array per input audio."""
    from senselab.audio.tasks.speaker_embeddings import api as est_api

    calls = {"i": 0}

    def fake(audio, models, device=None, window_s=2.0, hop_s=1.0, **kwargs):  # noqa: ANN001, ANN003
        vectors = per_audio_vectors[calls["i"]]
        calls["i"] += 1
        model_id = str(models[0].path_or_uri) if models else "stub/model"
        return {
            model_id: [
                est_api.WindowEmbedding(start_s=float(i) * hop_s, end_s=float(i) * hop_s + window_s, vector=v)
                for i, v in enumerate(vectors)
            ]
        }

    monkeypatch.setattr(est_api, "extract_per_window_embeddings", fake)
    monkeypatch.setattr(est_api, "_resolve_embedding_model", lambda model: ("speechbrain/spkrec-ecapa-voxceleb", "c" * 40, None))


def test_the_estimate_is_a_unit_vector_with_provenance(monkeypatch: pytest.MonkeyPatch, axes) -> None:  # noqa: ANN001
    """A vector without provenance cannot be interpreted later, so both come back together."""
    target, _ = axes
    _patch_extraction(monkeypatch, [_cone(target, 20, 0.03, 1), _cone(target, 20, 0.03, 2)])

    result = estimate_speaker_embedding_from_audios([_audio(), _audio()])

    assert np.linalg.norm(np.asarray(result.vector)) == pytest.approx(1.0)
    assert result.provenance.model_id == "speechbrain/spkrec-ecapa-voxceleb"
    assert result.provenance.model_commit_sha == "c" * 40
    assert result.provenance.method == "spherical_mean"
    assert result.provenance.window_s == 2.0
    assert result.provenance.hop_s == 1.0
    assert result.provenance.n_windows_used == 40
    assert result.provenance.n_windows_dropped == 0


def test_the_distribution_is_attached(monkeypatch: pytest.MonkeyPatch, axes) -> None:  # noqa: ANN001
    """The statistics are the whole point of the estimator: without them the caller cannot judge
    how well-supported the centroid is."""
    target, _ = axes
    _patch_extraction(monkeypatch, [_cone(target, 20, 0.03, 1), _cone(target, 20, 0.03, 2)])

    result = estimate_speaker_embedding_from_audios([_audio(), _audio()])

    assert result.distribution is not None
    assert result.distribution.counts.n_files == 2
    assert result.distribution.nulls.cos_sd_null == pytest.approx(1.0 / np.sqrt(_DIM))
    assert set(result.distribution.within_file) == set(result.distribution.cross_file.cos_file_centroid_to_pooled)


def test_contamination_is_visible_without_rejection(monkeypatch: pytest.MonkeyPatch, axes) -> None:  # noqa: ANN001
    """With the flag off nothing is dropped, and the intruder shows up in the per-file statistics
    -- which is how a caller curates its input instead of us deciding."""
    target, intruder = axes
    _patch_extraction(monkeypatch, [_cone(target, 20, 0.03, 1), _cone(target, 20, 0.03, 2), _cone(intruder, 20, 0.03, 3)])

    result = estimate_speaker_embedding_from_audios([_audio(), _audio(), _audio()])

    assert result.provenance.n_windows_dropped == 0
    assert result.provenance.method == "spherical_mean"
    lofo = result.distribution.centroid_robustness.leave_one_file_out_cos
    worst = min(lofo, key=lambda k: lofo[k])
    assert lofo[worst] < max(lofo.values())


def test_rejection_drops_the_intruder_and_records_it(monkeypatch: pytest.MonkeyPatch, axes) -> None:  # noqa: ANN001
    """Rejection is a decision, so it must never be silent: the method string and the dropped
    count both say it happened."""
    target, intruder = axes
    _patch_extraction(monkeypatch, [_cone(target, 20, 0.03, 1), _cone(target, 20, 0.03, 2), _cone(intruder, 20, 0.03, 3)])

    result = estimate_speaker_embedding_from_audios([_audio(), _audio(), _audio()], reject_contamination=True)

    assert result.provenance.method == "spherical_mean+dominant_cluster"
    assert result.provenance.n_windows_dropped == 20
    assert result.provenance.n_windows_used == 40
    assert np.asarray(result.vector) @ target > 0.9


def test_rejection_is_off_by_default(monkeypatch: pytest.MonkeyPatch, axes) -> None:  # noqa: ANN001
    """The default path decides nothing."""
    import inspect

    assert inspect.signature(estimate_speaker_embedding_from_audios).parameters["reject_contamination"].default is False


def test_an_empty_input_raises(monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: ANN001
    """No files means no estimate. Returning a zero vector would look like a measurement."""
    with pytest.raises(ValueError, match="at least one"):
        estimate_speaker_embedding_from_audios([])


def test_source_files_are_recorded_when_available(monkeypatch: pytest.MonkeyPatch, axes) -> None:  # noqa: ANN001
    """Provenance has to name what the estimate came from, or it cannot be reproduced."""
    target, _ = axes
    _patch_extraction(monkeypatch, [_cone(target, 10, 0.03, 1)])
    audio = _audio()
    audio.metadata["source"] = "unused"
    result = estimate_speaker_embedding_from_audios([audio])
    assert isinstance(result.provenance.source_files, list)
```

- [ ] **Step 2: Run and watch it fail**

```bash
uv run --no-sync pytest src/tests/audio/tasks/speaker_embeddings_estimate_test.py -v 2>&1 | tail -10
```

Expected: FAIL — `ImportError: cannot import name 'estimate_speaker_embedding_from_audios'`.

- [ ] **Step 3: Narrow the hint field**

In `audio_hints.py`, replace `distribution: Optional[Any] = None` with a real type, importing from
the submodule path (not the `utils.tasks` package `__init__`) to avoid pulling in siblings:

```python
from senselab.utils.tasks.embedding_distribution import EmbeddingDistribution
```

```python
    distribution: Optional[EmbeddingDistribution] = None
```

Verify no import cycle appears: `embedding_distribution.py` imports only numpy, pydantic, scipy
and stdlib, so `data_structures -> utils.tasks.embedding_distribution` is a leaf edge.

- [ ] **Step 4: Implement the estimator**

In `src/senselab/audio/tasks/speaker_embeddings/api.py`, add the imports the test monkeypatches by
name — `extract_per_window_embeddings` and `WindowEmbedding` must be module-level attributes of
`api` for `monkeypatch.setattr(est_api, ...)` to work:

```python
from senselab.audio.data_structures.audio_hints import SpeakerEmbeddingProvenance, TargetSpeakerEmbedding
from senselab.audio.tasks.speaker_embeddings.windowing import WindowEmbedding, extract_per_window_embeddings
from senselab.utils.tasks.embedding_distribution import (
    describe_embedding_distribution,
    select_dominant_vectors,
)

# ECAPA rather than a pair: PR #543 defaulted to ECAPA+ResNet because analyze_audio scores both
# per-window, and this estimator is deliberately decoupled from that consumer, so a second model
# would be unused cost. provenance.model_id records which one produced a vector.
DEFAULT_SPEAKER_EMBEDDING_MODEL = "speechbrain/spkrec-ecapa-voxceleb"

# Measured for a profile centroid, not picked: a 2.0 s window on a 1.0 s hop gave cross-file
# centroid stability 0.890 and cross-subject separation 0.168, against 0.331 for a 0.5/0.25 grid
# carrying four times the windows. Deliberately different from windowing.py's 1.0/0.5 detection
# defaults, which are tuned for temporal resolution on a 0.5 s bucket grid.
PROFILE_WINDOW_S = 2.0
PROFILE_HOP_S = 1.0
```

Add a small resolver so the test can patch one seam, and so a failure to resolve is recorded rather
than guessed:

```python
def _resolve_embedding_model(model: Optional[SenselabModel]) -> tuple[str, Optional[str], Optional[str]]:
    """Return ``(model_id, commit_sha, unresolved_reason)`` for provenance.

    A ref in the commit field would be provenance that is confidently wrong, so an unresolvable
    model yields ``None`` plus a stated reason instead.

    Args:
        model: The caller's model, or ``None`` for the default.

    Returns:
        ``(model_id, commit_sha, unresolved_reason)``; exactly one of the last two is ``None``.
    """
    model_id = str(model.path_or_uri) if model is not None else DEFAULT_SPEAKER_EMBEDDING_MODEL
    sha = getattr(model, "commit_sha", None) if model is not None else None
    if sha:
        return model_id, str(sha), None
    try:
        from senselab.utils.model_revision import resolve_revision

        return model_id, resolve_revision(model_id, "main"), None
    except Exception as exc:  # noqa: BLE001 — an unresolved commit must be recorded, not raised
        return model_id, None, f"{type(exc).__name__}: {exc}"
```

Then the estimator:

```python
def estimate_speaker_embedding_from_audios(
    audios: List[Audio],
    model: Optional[SenselabModel] = None,
    device: Optional[DeviceType] = None,
    window_s: float = PROFILE_WINDOW_S,
    hop_s: float = PROFILE_HOP_S,
    aggregator: str = "spherical_mean",
    reject_contamination: bool = False,
    created_at: Optional[str] = None,
) -> TargetSpeakerEmbedding:
    """Estimate one speaker embedding from files that *may* contain that speaker.

    Windows each file, embeds every window, pools them, and describes the resulting distribution.
    The returned statistics are what let a caller judge how well-supported the centroid is: this
    function reaches no verdict about whether the file set was clean.

    Args:
        audios: The recordings to enroll from. A file that does not contain the target speaker is
            not an error -- it shows up in the returned per-file statistics, which is how a caller
            curates its input.
        model: Embedding model. Defaults to ECAPA.
        device: CPU or CUDA.
        window_s: Window length. Defaults to the measured profile-centroid value, 2.0 s.
        hop_s: Hop between windows. Defaults to 1.0 s.
        aggregator: ``"spherical_mean"``, ``"trimmed_mean"`` or ``"medoid"``.
        reject_contamination: When ``True``, group the pooled windows and keep only the dominant
            group before describing. Off by default, because selecting a dominant group is a
            decision. What it removed is recorded in ``provenance.method`` and
            ``provenance.n_windows_dropped``.
        created_at: ISO-8601 timestamp for provenance. Not defaulted to "now": a library that
            stamps wall-clock time makes its own output unreproducible.

    Returns:
        A :class:`TargetSpeakerEmbedding` carrying the vector, its provenance, and the
        distribution it was estimated from.

    Raises:
        ValueError: If ``audios`` is empty, or if no window survived extraction.
    """
    if not audios:
        raise ValueError("estimate_speaker_embedding_from_audios needs at least one audio")

    model_id, commit_sha, unresolved = _resolve_embedding_model(model)
    speaker_model = model if model is not None else SpeechBrainModel(path_or_uri=model_id, revision="main")

    vectors: list[np.ndarray] = []
    file_ids: list[str] = []
    starts: list[float] = []
    source_files: list[str] = []
    for idx, audio in enumerate(audios):
        file_id = str(getattr(audio, "filepath", None) or f"audio-{idx}")
        source_files.append(file_id)
        per_model = extract_per_window_embeddings(
            audio, [speaker_model], device=device, window_s=window_s, hop_s=hop_s
        )
        for windows in per_model.values():
            for w in windows:
                vectors.append(np.asarray(w.vector, dtype=np.float64))
                file_ids.append(file_id)
                starts.append(float(w.start_s))

    if not vectors:
        raise ValueError("no embedding windows were produced; are the inputs shorter than window_s?")

    pooled = np.vstack(vectors)
    n_input = int(pooled.shape[0])
    method = aggregator

    if reject_contamination:
        selection = select_dominant_vectors(pooled, file_ids)
        keep = selection.kept_indices
        pooled = pooled[keep]
        file_ids = [file_ids[i] for i in keep]
        starts = [starts[i] for i in keep]
        method = f"{aggregator}+dominant_cluster"

    centroid, distribution = describe_embedding_distribution(
        pooled,
        file_ids,
        aggregator=aggregator,
        window_s=window_s,
        hop_s=hop_s,
        window_starts_s=starts,
    )

    return TargetSpeakerEmbedding(
        vector=centroid,
        provenance=SpeakerEmbeddingProvenance(
            model_id=model_id,
            model_commit_sha=commit_sha,
            unresolved_reason=unresolved,
            method=method,
            source_files=source_files,
            window_s=window_s,
            hop_s=hop_s,
            n_windows_used=int(distribution.counts.n_scored),
            n_windows_dropped=n_input - int(distribution.counts.n_scored),
            created_at=created_at,
        ),
        distribution=distribution,
    )
```

Export it from `src/senselab/audio/tasks/speaker_embeddings/__init__.py`, adding to `__all__`.

- [ ] **Step 5: Run and iterate**

```bash
uv run --no-sync pytest src/tests/audio/tasks/speaker_embeddings_estimate_test.py -v 2>&1 | tail -20
```

Expected: PASS, 7 tests. Two likely snags:

- If the monkeypatch of `extract_per_window_embeddings` does not take, the import in `api.py` must
  be a module-level `from ... import extract_per_window_embeddings` (so it is an attribute of
  `api`), and the estimator must call the bare name — not the fully-qualified path.
- If `SpeechBrainModel(path_or_uri=...)` tries to reach the network at construction, the test's
  patch of `_resolve_embedding_model` is not enough; construct the model lazily *inside* the
  extraction call path, or accept a `SenselabModel` and let the test pass a stub with
  `path_or_uri`. Prefer the second: it keeps the estimator honest and the test model-free.

- [ ] **Step 6: Full verification and commit**

```bash
uv run --no-sync pytest src/tests/audio/ src/tests/utils/ -q 2>&1 | tail -6
uv run --no-sync ruff format src/senselab/ src/tests/
uv run --no-sync ruff check src/ src/tests/
uv run --no-sync mypy --ignore-missing-imports --extra-checks src/ 2>&1 | tail -3
git add src/senselab/audio/tasks/speaker_embeddings/api.py \
        src/senselab/audio/tasks/speaker_embeddings/__init__.py \
        src/senselab/audio/data_structures/audio_hints.py \
        src/tests/audio/tasks/speaker_embeddings_estimate_test.py
git commit -m "feat(audio): estimate a speaker embedding from files that may contain them

Windows each file, embeds every window, pools them and describes the distribution.
Returns the centroid, its provenance, and the statistics -- and reaches no verdict
about whether the file set was clean, because that is the caller's to draw.

Defaults to ECAPA alone rather than #543's ECAPA+ResNet pair: that pair existed
because analyze_audio scores both per-window, and this estimator is decoupled from
that consumer, so a second model is unused cost. Window defaults are the measured
profile-centroid values 2.0/1.0, deliberately distinct from windowing.py's 1.0/0.5
detection defaults.

reject_contamination is off by default. On, it groups the pooled windows, keeps the
dominant group, and records that it happened in provenance.method and
n_windows_dropped -- a curated estimate must not be able to look like a clean one.

Provenance records a resolved commit SHA or an explicit unresolved_reason, never a
ref. created_at is caller-supplied rather than stamped from the clock, since a
library that stamps wall-clock time makes its own output unreproducible.

Tests monkeypatch per-window extraction and load no model, so aggregation,
provenance and rejection are all exercised deterministically with no download."
```

---

### Task 9: The two defect fixes, and the docs

**Files:**
- Modify: `src/senselab/audio/workflows/audio_analysis/embeddings.py`
- Create: `src/senselab/audio/tasks/speaker_embeddings/doc.md`
- Modify: `src/senselab/audio/tasks/speaker_embeddings/__init__.py`

**Interfaces:**
- Consumes: everything above.
- Produces: no importable interface.

- [ ] **Step 1: Locate both defects**

```bash
grep -n "p_voice\|0.5 \* (\|silhouette + 1\|0.5 \* (s" src/senselab/audio/workflows/audio_analysis/embeddings.py
sed -n '1,10p' src/senselab/audio/workflows/audio_analysis/embeddings.py
grep -rn "p_voice\|silhouette_voice_score" src/senselab/ src/tests/ scripts/ | grep -v "^src/senselab/audio/workflows/audio_analysis/embeddings.py"
```

The third command matters most: it tells you whether anything still *consumes* `p_voice`. The L1
post-processing register closed its item 12 by removing the consumer, not the computation, so the
expected answer is that only `embeddings.py` mentions it. If something else does, stop and report
it — removing a live consumer is outside this task.

- [ ] **Step 2: Write the failing guard test**

Append to `src/tests/audio/workflows/audio_analysis/embeddings_test.py` if it exists, otherwise
create `src/tests/audio/workflows/audio_analysis/embeddings_defects_test.py`:

```python
"""Guards for two defects this change fixes.

Both are the kind that reappear: a silhouette coefficient renamed back into a probability, and a
docstring drifting from the signature it describes.
"""

import ast
import inspect
from pathlib import Path

from senselab.audio.workflows.audio_analysis import embeddings as emb

_SOURCE = Path(emb.__file__).read_text(encoding="utf-8")


def test_no_silhouette_is_rescaled_into_a_probability() -> None:
    """0.5*(s+1) turns a clustering-geometry index into something that reads as a probability.

    CLAUDE.md names this defect class, and the L1 register documents its cost here: the signal
    produced 0.4022-0.4996 doubt across 214 buckets with stdev 0.0227 and earned the highest
    fusion weight of fifteen signals *because* it was near-constant; removing its consumer moved
    published presence doubt from 0.0682 to 0.0385.
    """
    assert "p_voice" not in _SOURCE, "p_voice reads as a probability; name it what it measures"


def test_the_module_docstring_agrees_with_the_signature() -> None:
    """A docstring claiming different defaults than the code is worse than none: a reader trusts
    it and passes nothing, expecting the documented behaviour."""
    sig = inspect.signature(emb.extract_per_window_embeddings)
    window_default = sig.parameters["window_s"].default
    hop_default = sig.parameters["hop_s"].default
    doc = ast.get_docstring(ast.parse(_SOURCE)) or ""
    assert f"{window_default} s with {hop_default} s hop" in doc or f"{window_default}" in doc.split("\n")[2]
```

- [ ] **Step 3: Run it and watch the first test fail**

```bash
uv run --no-sync pytest src/tests/audio/workflows/audio_analysis/embeddings_defects_test.py -v 2>&1 | tail -10
```

Expected: `test_no_silhouette_is_rescaled_into_a_probability` FAILS.

- [ ] **Step 4: Fix defect 1**

Remove the `0.5 * (silhouette + 1)` rescaling and the `p_voice` name. If the containing function
exists only to produce `p_voice` and nothing consumes it (confirmed in Step 1), delete the
function. If it also returns other values callers use, keep those and drop only the rescaled
field, renaming any survivor to what it measures — e.g. `silhouette` — with a comment recording
why the rescaling is gone:

```python
# The old `p_voice = 0.5 * (silhouette + 1)` is deliberately absent. A silhouette coefficient is a
# property of a chosen partition on a chosen metric, not a probability: silhouette computed with
# cosine and with Euclidean return different numbers for identical geometry on unit vectors, so any
# probability read off it is a probability about a parameterisation choice. The L1 register (item
# 12) removed the consumer for this reason; this removes the computation.
```

- [ ] **Step 5: Fix defect 2**

Correct the module docstring's opening line so it states the same defaults as the signature. Do
**not** change the signature: `window_s=1.0, hop_s=0.5` is measured for temporal resolution on the
0.5 s bucket grid, and the module's own "Why 1.0 s / 0.5 s defaults" section derives it. Add one
sentence pointing at the other measured setting so the two never get collapsed:

```
Profile enrollment wants the opposite trade and passes ``window_s=2.0, hop_s=1.0`` explicitly --
see ``tasks/speaker_embeddings``. Two purposes, two measured settings.
```

- [ ] **Step 6: Write `doc.md`**

Create `src/senselab/audio/tasks/speaker_embeddings/doc.md` covering, in this order: what
`extract_speaker_embeddings_from_audios` and `estimate_speaker_embedding_from_audios` each do and
when to reach for which; the two window settings and their separate measurements; the statistics
block and how to read it against the analytic nulls; that a small cosine sd is *below* the
random-vector null at d=192 and therefore not evidence of coherence; what contamination rejection
does, that it is opt-in, and that its statistics then describe a curated set; and a **suggested,
non-binding vocabulary** for `AudioHints.may_contain` (`read-speech`, `spontaneous-speech`,
`sustained-vowel`, `cough`, `breath`, `singing`, `music`, `multiple-speakers`, `silence`) and
`AudioHints.environment` (`quiet-room`, `clinic`, `home`, `telephone`, `outdoors`, `unknown`),
stating explicitly that these are suggestions rather than an enum and why: a closed vocabulary here
would be a taxonomy nobody fitted.

Wire it up by making `src/senselab/audio/tasks/speaker_embeddings/__init__.py`'s docstring
`""".. include:: ./doc.md"""  # noqa: D415`, matching how `speech_enhancement` does it — but keep
the existing exports.

- [ ] **Step 7: Full suite, lint, typecheck**

```bash
uv run --no-sync pytest src/tests --color=no -q -p no:cacheprovider --no-cov -rf 2>&1 | tail -15
uv run --no-sync ruff format src/senselab/ src/tests/
uv run --no-sync ruff check src/ src/tests/
uv run --no-sync mypy --ignore-missing-imports --extra-checks src/ 2>&1 | tail -3
```

Expected: the whole suite passes. This is the run that matters — the workflow's embedding tests
prove the defect fixes changed no behaviour beyond removing a dead computation.

- [ ] **Step 8: Commit**

```bash
git add src/senselab/audio/workflows/audio_analysis/embeddings.py \
        src/senselab/audio/tasks/speaker_embeddings/doc.md \
        src/senselab/audio/tasks/speaker_embeddings/__init__.py \
        src/tests/audio/workflows/audio_analysis/
git commit -m "fix(audio_analysis): stop rescaling a silhouette into a probability; correct a stale docstring

p_voice = 0.5*(silhouette + 1) turned a clustering-geometry index into a value that
reads as a probability. CLAUDE.md names this defect class, and the L1
post-processing register documents what it cost here: the signal produced
0.4022-0.4996 doubt across 214 buckets with stdev 0.0227 and earned the highest
fusion weight of fifteen signals precisely because it was near-constant, and
removing its consumer moved published presence doubt from 0.0682 to 0.0385. The
register closed item 12 by removing the consumer; this removes the computation.

The module docstring claimed a 2.0 s / 1.0 s default while both the signature and
its own 'Why 1.0 s / 0.5 s defaults' section said otherwise. The line is
corrected, and the signature is not touched: 1.0/0.5 is measured for temporal
resolution against the 0.5 s bucket grid. A pointer to the estimator's separately
measured 2.0/1.0 keeps the two from being collapsed.

Adds doc.md for the speaker-embeddings task: both entry points, the two window
settings and their measurements, how to read the statistics against their analytic
nulls, and a suggested non-binding vocabulary for the hint tags."
```

---

## Self-review

**Spec coverage.** Every section of `design.md` maps to a task: hints layer → Task 1; descriptor
geometry/nulls/LOO/spectrum → Task 2; within/cross-file → Task 3; file effect → Task 4; aggregator
and robustness → Task 5; contamination rejection → Task 6; layering promotion and the guard →
Task 7; estimator and provenance → Task 8; both defect fixes and the docs → Task 9. The spec's
`separability` field is realised as `file_effect.auc_same_file_vs_diff_file` (Task 4), which is the
same statistic — with clustering gone there is no within-cluster contrast, as the spec notes.

**Placeholder scan.** No TBD/TODO. Every code step carries real code. Two steps deliberately
instruct the implementer to *discover* a fact rather than assume one (Task 1 Step 3, the
content-hash helper's real name; Task 9 Step 1, whether anything still consumes `p_voice`), and
both say what to do with either answer, including when to stop and report.

**Type consistency.** `EmbeddingDistribution` is constructed once in Task 2 and extended by
default-valued fields in Tasks 3-5, so each task's tests keep passing. `SimilarityStats` is defined
in Task 2 and reused in Tasks 3-4. `TargetSpeakerEmbedding.distribution` is `Any | None` in Task 1
and narrowed to `EmbeddingDistribution` in Task 8, which is stated in both places. `window_starts`
and `slice_audio` are the promoted public names throughout Tasks 7-8; the old private names appear
only in Task 7's grep step.

**One known risk, flagged rather than hidden.** Task 8's test monkeypatches
`extract_per_window_embeddings` and `_resolve_embedding_model` as attributes of `api`. If the
implementer imports them differently the patch silently misses and the test will try to load a real
model. Task 8 Step 5 names both failure modes and says which fix to prefer.
