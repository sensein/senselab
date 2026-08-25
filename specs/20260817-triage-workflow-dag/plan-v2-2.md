# Triage v2 Implementation Plan — branches: SPEECH, VOICE, AIRWAY, REDACT, VERDICT, REPORT

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bring the branches and the folds up to the **v2 specs**. The sibling `plan-v2-1.md` covers
PREPROCESS, TAXONOMY and routing, and fixes the store schema every task here consumes; **read its
§"The v2 store contract" and §"The 33 open keys" before starting any task in this file.** Every open
config key these tasks read is created by sibling T1, so no task here invents a key shape.

**Task order and dependencies:** T4..T7 depend on sibling T1's store schema. T8 depends on sibling T3's
`branch_decision` elements and on T4..T7's verdicts. T9 depends on T8. **T10 is independent** of every
other task and may run first or in parallel.

**Design source of truth:** `specs/20260817-triage-workflow-dag/` — `store.md`, then
`branch-speech.md`, `branch-voice.md`, `branch-airway.md`, `redact.md`, `verdict.md`, `report.md`.

## Global Constraints

Copied verbatim from `plan-v2-1.md`; every one is binding here too.

- **Every Python command runs through `uv run`.** Never bare `python` or `pip`.
- **Never run `pytest -n auto`.** Each xdist worker duplicates ~535 MB of frameworks plus its own model
  weights. Run the directory you changed, serially.
- `uv sync` is subtractive — always pass `--all-extras`.
- Tests live in `src/tests/` mirroring the package, named `*_test.py`. ruff applies to test code, so
  **every test class and function needs a docstring and every test function is annotated `-> None`**.
- Google-style docstrings; line length 120; type hints required (mypy with the pydantic plugin).
- **Rationale does not go in code.** Docstrings say what a thing is and how to call it. Measurements,
  rejected alternatives and the failures that drove a choice go in
  `specs/20260817-triage-workflow-dag/benchmarks/` — except a `derivation:` block in
  `data/config/default.yaml`, which is a config *value* and stays there.
- **No numeric or string tunable appears in production code.** Not as a signature default, not as a
  module-level constant. Every such value is a key in `data/config/default.yaml` with its derivation
  beside it, read through `config.require()` or `config.get()`. Definitional constants are the only
  exemption: `20·log10`, full scale `1.0`, `1e-12` floor clamps, `1000.0` ms-per-second, `1200.0`
  cents-per-octave, and a *vocabulary token* that is a controlled string the store must round-trip
  (`"not_evaluated"`, `"bounded"`, `"unavailable"`, `"not_assessed"`).
- **A value nobody has measured is `null` in the config, and reading it raises.** Sibling T1 creates all
  33 open keys; this file's T5, T7 and T9 create six more (`voice.f0_range_by_population`,
  `voice.f0_range_ratio_max`, `voice.task_duration_ranges`, `redaction.fill`, `redaction.bleep_hz`,
  `report.format`). **Supplying a value for any open key is wrong.** Tests exercise them through
  explicit YAML overrides, which is the intended production mechanism too.
- **Append-only `ProvStore`.** Nothing is modified or deleted; a superseded claim is
  `wasInvalidatedBy`, a refined one `wasDerivedFrom`. Every read of the store goes through
  `nodes/common.py`'s helpers (`find_measurement`, `find_measurements`, `live_entities`,
  `resolve_stream`), which apply the store's shared rule: **an invalidated entity is never read, and of
  the survivors asserting the same thing the latest write wins.** A node that hand-rolls
  `store.entities(...)` without that filter is a defect.
- **Nodes never import sibling nodes**, with exactly one sanctioned exception: the model-identity
  constants re-exported from `nodes/preprocess.py` (`CRISPERWHISPER_ID`, `QWEN_ID`, `AST_ID`,
  `YAMNET_MODEL_URI`). Those are names, not behaviour. Importing a *function* from a sibling node is a
  defect; read the store instead.
- A model load passes a **resolved 40-hex commit SHA, never a ref**.
  `src/tests/utils/revision_pinning_guard_test.py` sweeps the subprocess-worker files for this.
- **TDD, red first.** Every task's step 1 writes the tests and step 2 **runs them and records the
  failure text**. A task whose tests passed before the implementation was written did not test the
  change.
- **Behavioural tests.** A node test seeds a store with the entities its predecessors would have
  written, calls the node, and asserts on the verdict and the entities that came out. Models are mocked
  **at the node module's boundary** (`monkeypatch.setattr(node_module, "diarize_audios", fake)`), never
  deeper; model *constructors* that resolve a commit over the network are reached only through
  module-level factory functions so those can be patched too. **No test loads YAMNet, AST, HeAR,
  CrisperWhisper, Qwen, pyannote, an aligner, SQUIM, an embedder or a separator** — with one stated
  exception, T10, which is a diagnostic whose whole subject is a real model run and which is marked
  `@pytest.mark.large` and excluded from the default suite.
- Run `uv run ruff format` before every commit, then `uv run ruff check` and `uv run mypy` on the paths
  touched.
- **Pre-alpha: rename and replace outright.** The v1 VOICE (residual subtraction), the v1 REDACT
  (re-transcribing verification), the v1 SPEECH target-by-hint identification and the v1 file-verdict
  vocabulary (`Outcome` on the file axis) are **replaced**. Delete the tests that pinned them, and name
  in the deletion commit the v2 spec sentence that superseded each.

## Under-specified points, resolved by this plan

Continuing `plan-v2-1.md`'s `V1..V14`.

| # | point | decision |
| --- | --- | --- |
| V15 | `branch-speech.md` gives `enrollment` its own shape but the tree has `AudioHints.target_speaker` | a new pydantic model `Enrollment` in `src/senselab/audio/workflows/triage/enrollment.py`, passed to `run_triage` and to `speech()` as a keyword argument. `hint.target_speaker` is **no longer read by triage**; when it is set and no enrollment is given, SPEECH flags with `"a target embedding was supplied on the hint; triage identifies the target by enrollment and did not read it"`, so the ignore is never silent |
| V16 | the spec's `provenance: {model_id, revision, task}` versus the tree's `SpeakerEmbeddingProvenance` | `Enrollment.provenance` **is** `SpeakerEmbeddingProvenance` — it already carries `model_id`, a validated 40-hex `model_commit_sha`, `source_files` and `unresolved_reason`. `revision` is `model_commit_sha`; `task` is a new `Enrollment.task: str \| None`. Reusing it means `estimate_speaker_embedding_from_audios` produces an enrollment directly |
| V17 | `branch-speech.md` says unasdiff's sound slot "stands for any background, so the mode is used without conditioning the background on a class" — but `separate_audios` **refuses** `speech_sound` without a `source_classes` entry ("index 0 is 'Hi-hat'") | a real spec/API conflict. `speech.separation_sound_class` is created **null** and named in the derivation. While it is null the unasdiff option cannot run: SPEECH records `separation: "unconditioned_sound_slot_unavailable"` and flags. **Flagged to the owner** — resolving it needs either a defensible FSD class or an unconditioned sound slot in unasdiff |
| V18 | the Glides `ValueError: 'waveform' must be provided as a (channel, time) torch Tensor` | pyannote's `Audio.validate_file` refuses a waveform where `shape[0] > shape[1]` (`pyannote/audio/core/io.py:173`). On a Glides file the consensus collapses to a degenerate `[first word start, last word end]`, `extract_segments` slices `[s:e]` with `s == e`, and a `(1, 0)` tensor trips exactly that guard. The fix is at the source: an interval shorter than one analysis frame is **not diarized**, the branch records `speaker_count: null` with `diarization: "interval_shorter_than_one_frame"` and flags. See T4 step 5 |
| V19 | the disruptions-absence anomaly on wordless files | it is **not** in SPEECH's span scoping: a span nobody measured must not report zero, so per-span absence on a wordless file is correct. The missing half is the file-level reading, which sibling T1 adds as `disruptions_file`. T4 verifies the span scoping and adds nothing — see T4 step 6 |
| V20 | `branch-voice.md`'s "half-frame tolerance" for `min_marks_s` is not in the spec at all; the owner's ruling is | a frame stands for a hop-wide interval **centred on its time**, so a run's duration is `times[last] - times[first] + hop_s`, not `times[last] - times[first]`. The tolerance is one hop — half a hop at each edge — and is an identity of the analysis grid, derived from `phonation_spans.hop_s`, not a magic number |
| V21 | `branch-airway.md`'s "both fall inside the same window" does not say **whose** window | the **HeAR** window, because the HeAR label is what is being contested. A YAMNet label contests only when its window's extent lies inside the extent of a HeAR window whose label set contains the span's label |
| V22 | `redact.md`'s `noise` fill ("speech-shaped noise at the extent's own level") has no measured shaping | `redaction.fill` accepts `"silence"` and `"bleep"`, both implemented. `"noise"` raises `NotImplementedError` naming the measurement it is owed — which fill is least damaging to downstream measurement — rather than shipping an unmeasured spectral shape. `redaction.bleep_hz` is `1000.0` with the derivation "the conventional broadcast censor tone; a presentation choice, not fitted" |
| V23 | `verdict.md`'s `triage` axis is `pass \| flag \| discard`, but the tree folds `Outcome` (`pass \| flag \| fail`) | a new `Triage` enum replaces `Outcome` on the **file** axis. Node verdicts keep `Outcome`. `FileVerdict.triage: Triage`, and `fold_file_verdict` is rewritten around branch authority rather than the v1 contradiction table |
| V24 | `report.md` requires both products on every outcome including an ADMIT refusal, where the store holds one verdict and one stream | REPORT reads whatever is there. On an ADMIT `fail` the summary is one page carrying the file block, the verdict block and the words "nothing was measured"; the JSON carries `verdict`, `provenance` and empty `branches`/`steps`. It never raises for want of a derivative |

---

### Task 4: SPEECH v2 — enrollment, a conditional second diarizer, PII on the consensus, and the Glides fix

**Scope:** `src/senselab/audio/workflows/triage/enrollment.py` (new);
`src/senselab/audio/workflows/triage/nodes/speech.py` (heavily edited);
`src/senselab/audio/workflows/triage/run.py` (forward `enrollment`);
`src/senselab/audio/workflows/triage/__init__.py` (export `Enrollment`);
`src/tests/audio/workflows/triage/nodes/speech_test.py` (rewritten);
`src/tests/audio/workflows/triage/enrollment_test.py` (new).

**Design points this task must not get wrong (from `branch-speech.md`):**

- **It runs no ASR and never re-transcribes.** PREPROCESS produced the consensus with
  `fuse_consensus_words`; this branch reads it and does not re-fuse, re-clean or re-decode. The v1
  `fuse_word_streams` call in this module is **deleted**.
- **It does not read AIRWAY.** No segment is withdrawn for overlapping an airway span. (Already true
  since 8537a83f — step 6 verifies it and adds nothing.)
- **The target speaker is identified by an enrollment**, an embedding estimated across all of a
  subject's provided recordings, not by a per-file target hint.
- **Provenance is required and is model + revision.** An enrollment without both is **refused rather
  than compared**, and the branch flags.
- **The second diarizer runs only when pyannote's count is not 1.** When it is 1, that is the count and
  no second diarizer runs.
- **The count is not compared against a declared count.** `hint.targeted_speaker_count` is not read.
- **Separation is measurement-gated** and does not run while `speech.separation_backend` is null.
- **The PII scan reads the consensus transcript and nothing else** — one scan, one text.
- **Any finding at all sends the branch to REDACT**, whatever the speaker scope; the *flag* is
  speaker-scoped, the *redaction* is not.
- **The non-target axis is measured and reported per span, and `nontarget_speech_s` is null** until all
  three proximity thresholds exist. No span is excluded on this evidence.
- **This branch marks; it removes nothing.**

**Steps:**

- [ ] **Step 1 — write the failing enrollment tests.**

`src/tests/audio/workflows/triage/enrollment_test.py`:

```python
"""The enrollment input: a subject's target vector, with the model and revision behind it."""

import pytest
from pydantic import ValidationError

from senselab.audio.data_structures import SpeakerEmbeddingProvenance
from senselab.audio.workflows.triage.enrollment import Enrollment


def _provenance(**overrides: object) -> SpeakerEmbeddingProvenance:
    """A provenance record naming a model and a resolved commit."""
    fields = {
        "model_id": "speechbrain/spkrec-ecapa-voxceleb",
        "model_commit_sha": "a" * 40,
        "source_files": ["a.wav", "b.wav"],
    }
    fields.update(overrides)
    return SpeakerEmbeddingProvenance(**fields)


class TestTheShape:
    """subject_id, vector, provenance, sources — every recording behind the vector is named."""

    def test_an_enrollment_names_every_recording_behind_it(self) -> None:
        """sources is what makes an enrollment reproducible and a file's own contribution visible."""
        enrollment = Enrollment(subject_id="sub-01", vector=[0.6, 0.8], provenance=_provenance())
        assert enrollment.sources == ["a.wav", "b.wav"]

    def test_the_vector_must_be_non_empty(self) -> None:
        """A zero-length embedding is compared against nothing."""
        with pytest.raises(ValidationError):
            Enrollment(subject_id="sub-01", vector=[], provenance=_provenance())


class TestRefusal:
    """An enrollment that cannot be compared is refused, and the refusal names why."""

    def test_a_missing_commit_is_refused(self) -> None:
        """Two commits of one model are not comparable, so a bare model id is not provenance."""
        enrollment = Enrollment(
            subject_id="sub-01",
            vector=[1.0],
            provenance=_provenance(model_commit_sha=None, unresolved_reason="hub outage"),
        )
        assert "resolved model commit" in (enrollment.refusal_against("speechbrain/spkrec-ecapa-voxceleb") or "")

    def test_a_different_model_is_refused(self) -> None:
        """Embeddings from different models are not comparable at any threshold."""
        enrollment = Enrollment(subject_id="sub-01", vector=[1.0], provenance=_provenance())
        assert "not the probe" in (enrollment.refusal_against("pyannote/embedding") or "")

    def test_a_matching_model_and_commit_is_comparable(self) -> None:
        """The one case that is not a refusal."""
        enrollment = Enrollment(subject_id="sub-01", vector=[1.0], provenance=_provenance())
        assert enrollment.refusal_against("speechbrain/spkrec-ecapa-voxceleb") is None
```

- [ ] **Step 2 — run it; expect FAIL** (`ModuleNotFoundError: ...triage.enrollment`).
  `uv run pytest src/tests/audio/workflows/triage/enrollment_test.py -x -q`

- [ ] **Step 3 — write `enrollment.py`.**

```python
"""The enrollment input: one subject's target-speaker vector, estimated across their recordings.

The target speaker is enrolled, not hinted: a per-file target declaration says which speaker the
protocol intended, while an enrollment says which voice was measured, across every recording the
subject provided. ``branch-speech.md`` §6 is the contract.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field, field_validator

from senselab.audio.data_structures import SpeakerEmbeddingProvenance
from senselab.utils.tasks.embedding_distribution import EmbeddingDistribution


class Enrollment(BaseModel):
    """A speaker embedding enrolled across all of one subject's provided recordings.

    Attributes:
        subject_id: Whose voice this is.
        vector: The embedding, unit-norm.
        provenance: Required. Carries the embedding model and its **resolved** commit; an enrollment
            without both is refused rather than compared, and names every recording that contributed
            in ``source_files``.
        task: The vocal task the enrollment was estimated over, when one was declared.
        distribution: Spread over the contributing windows, when the estimator produced one.
    """

    subject_id: str
    vector: list[float] = Field(min_length=1)
    provenance: SpeakerEmbeddingProvenance
    task: Optional[str] = None
    distribution: Optional[EmbeddingDistribution] = None

    @property
    def sources(self) -> list[str]:
        """Every recording behind the vector.

        Returns:
            The contributing file ids, from the provenance.
        """
        return list(self.provenance.source_files)

    def refusal_against(self, probe_model_id: str) -> str | None:
        """Why this enrollment cannot be compared with a probe from ``probe_model_id``.

        Args:
            probe_model_id: The embedding model the branch will run over the diarized speakers.

        Returns:
            The refusal, in controlled vocabulary, or None when the enrollment is comparable.
        """
        if self.provenance.model_commit_sha is None:
            return "the enrollment carries no resolved model commit; refused rather than compared"
        if self.provenance.model_id != probe_model_id:
            return (
                f"the enrollment's model {self.provenance.model_id} is not the probe {probe_model_id}; "
                "embeddings from different models are not comparable"
            )
        return None

    @field_validator("vector")
    @classmethod
    def _must_be_finite(cls, value: list[float]) -> list[float]:
        """Reject a vector carrying a non-finite component.

        Args:
            value: The candidate vector.

        Returns:
            The vector unchanged.

        Raises:
            ValueError: When any component is NaN or infinite, which no similarity is defined over.
        """
        if any(component != component or component in (float("inf"), float("-inf")) for component in value):
            raise ValueError("every component of an enrollment vector must be finite")
        return value
```

Export it from `src/senselab/audio/workflows/triage/__init__.py` (import and `__all__`).

- [ ] **Step 4 — run it; expect PASS.**
  `uv run pytest src/tests/audio/workflows/triage/enrollment_test.py -x -q`

- [ ] **Step 5 — write the failing SPEECH tests, including the Glides reproduction.**

Replace `src/tests/audio/workflows/triage/nodes/speech_test.py`. The new classes, in full:

```python
class TestItReadsTheConsensusAndReFusesNothing:
    """PREPROCESS produced the consensus; this branch reads it."""

    def test_the_words_come_from_the_consensus_transcript(
        self, store: ProvStore, speech_config: TriageConfig, seeded_store: Callable[..., None], tmp_path: Path
    ) -> None:
        """words_n is the count of consensus word entities, not a re-fusion of the hypotheses."""
        seeded_store(store, tmp_path, words=["hello", "world"])
        result = speech(store, "plain", speech_config, run_dir=tmp_path, enrollment=None)
        assert result.verdict.node == "SPEECH"
        verdict = _verdict_entity(store, "SPEECH")
        assert verdict.attributes["words_n"] == 2

    def test_the_module_cannot_re_fuse(self) -> None:
        """A fusion function reachable from this module is the v1 behaviour the spec deleted."""
        assert not hasattr(speech_module, "fuse_word_streams")
        assert not hasattr(speech_module, "fuse_consensus_words")

    def test_an_event_is_not_a_word(
        self, store: ProvStore, speech_config: TriageConfig, seeded_store: Callable[..., None], tmp_path: Path
    ) -> None:
        """Bracketed and onomatopoeic events count toward no word total and no span extent."""
        seeded_store(store, tmp_path, words=["hello"], events=["[COUGH]", "[BREATH]"])
        speech(store, "plain", speech_config, run_dir=tmp_path, enrollment=None)
        assert _verdict_entity(store, "SPEECH").attributes["words_n"] == 1

    def test_no_consensus_word_fails_and_writes_no_pii_scan(
        self, store: ProvStore, speech_config: TriageConfig, seeded_store: Callable[..., None], tmp_path: Path
    ) -> None:
        """redact.md: a wordless recording has no PII scan, no REDACT verdict and no withheld release."""
        seeded_store(store, tmp_path, words=[])
        result = speech(store, "plain", speech_config, run_dir=tmp_path, enrollment=None)
        assert result.verdict.outcome is Outcome.FAIL
        assert find_measurement(store, "pii_scan") is None


class TestTheSecondDiarizerIsConditional:
    """One speaker is the count; anything else consults a second diarizer and reports disagreement."""

    def test_a_count_of_one_consults_nobody(
        self, store: ProvStore, second_diarizer_config: TriageConfig, seeded_store: Callable[..., None],
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """branch-speech.md: 'No second diarizer runs'."""
        seeded_store(store, tmp_path, words=["hello", "world"])
        calls = _stub_diarizers(monkeypatch, primary_speakers=1, second_speakers=2)
        speech(store, "plain", second_diarizer_config, run_dir=tmp_path, enrollment=None)
        assert calls == ["primary"]
        assert _verdict_entity(store, "SPEECH").attributes["second_diarizer"] == "not_consulted"

    def test_a_count_of_two_consults_the_second(
        self, store: ProvStore, second_diarizer_config: TriageConfig, seeded_store: Callable[..., None],
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The disagreement is reported; it does not replace pyannote's count."""
        seeded_store(store, tmp_path, words=["hello", "world"])
        calls = _stub_diarizers(monkeypatch, primary_speakers=2, second_speakers=3)
        speech(store, "plain", second_diarizer_config, run_dir=tmp_path, enrollment=None)
        assert calls == ["primary", "second"]
        record = _verdict_entity(store, "SPEECH").attributes["second_diarizer"]
        assert record["count"] == 3 and record["agrees"] is False
        assert _verdict_entity(store, "SPEECH").attributes["speaker_count"] == 2

    def test_a_count_of_zero_consults_the_second_too(
        self, store: ProvStore, second_diarizer_config: TriageConfig, seeded_store: Callable[..., None],
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """branch-speech.md: 'the codomain is the counts pyannote can return, and 0 is one of them'."""
        seeded_store(store, tmp_path, words=["hello", "world"])
        calls = _stub_diarizers(monkeypatch, primary_speakers=0, second_speakers=1)
        speech(store, "plain", second_diarizer_config, run_dir=tmp_path, enrollment=None)
        assert calls == ["primary", "second"]

    def test_a_declared_count_is_not_read(
        self, store: ProvStore, speech_config: TriageConfig, seeded_store: Callable[..., None],
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """hint.targeted_speaker_count is the protocol's intent, of unknown provenance; not evidence."""
        seeded_store(store, tmp_path, words=["hello", "world"])
        _stub_diarizers(monkeypatch, primary_speakers=1, second_speakers=1)
        hint = AudioHints(targeted_speaker_count=4)
        result = speech(store, "plain", speech_config, hint, run_dir=tmp_path, enrollment=None)
        assert "4" not in result.verdict.why


class TestTheDegenerateDiarizationInterval:
    """The Glides failure: a (1, 0) crop trips pyannote's (channel, time) guard (V18)."""

    def test_a_zero_length_interval_is_not_diarized(
        self, store: ProvStore, speech_config: TriageConfig, seeded_store: Callable[..., None],
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """One word whose start equals its end collapses the interval; the branch must not call out."""
        seeded_store(store, tmp_path, words=["aaaa"], word_extents=[(1.0, 1.0)])
        calls = _stub_diarizers(monkeypatch, primary_speakers=1, second_speakers=1)
        result = speech(store, "plain", speech_config, run_dir=tmp_path, enrollment=None)
        assert calls == []
        verdict = _verdict_entity(store, "SPEECH")
        assert verdict.attributes["speaker_count"] is None
        assert verdict.attributes["diarization"] == "interval_shorter_than_one_frame"
        assert result.verdict.outcome is Outcome.FLAG

    def test_the_real_pyannote_guard_is_what_this_avoids(self) -> None:
        """A (1, 0) waveform is exactly what pyannote refuses; this pins the mechanism, not the fix."""
        waveform = torch.zeros(1, 0)
        assert waveform.shape[0] > waveform.shape[1]

    def test_a_sub_frame_interval_is_not_diarized_either(
        self, store: ProvStore, speech_config: TriageConfig, seeded_store: Callable[..., None],
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """One sample is not zero samples, and is still not a diarizable interval."""
        seeded_store(store, tmp_path, words=["aaaa"], word_extents=[(1.0, 1.0 + 1 / 16000)])
        calls = _stub_diarizers(monkeypatch, primary_speakers=1, second_speakers=1)
        speech(store, "plain", speech_config, run_dir=tmp_path, enrollment=None)
        assert calls == []


class TestEnrollment:
    """The target is enrolled. An enrollment without provenance is refused rather than compared."""

    def test_no_enrollment_claims_no_identity(
        self, store: ProvStore, speech_config: TriageConfig, seeded_store: Callable[..., None],
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Speakers stay SPEAKER_*, and nothing is called a target."""
        seeded_store(store, tmp_path, words=["hello", "world"])
        _stub_diarizers(monkeypatch, primary_speakers=1, second_speakers=1)
        speech(store, "plain", speech_config, run_dir=tmp_path, enrollment=None)
        assert "target_speaker" not in _verdict_entity(store, "SPEECH").attributes

    def test_an_enrollment_without_a_commit_is_refused(
        self, store: ProvStore, enrollment_config: TriageConfig, seeded_store: Callable[..., None],
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """No embedder runs; the branch flags with the refusal."""
        seeded_store(store, tmp_path, words=["hello", "world"])
        _stub_diarizers(monkeypatch, primary_speakers=1, second_speakers=1)
        embedder = _stub_embedder(monkeypatch)
        result = speech(
            store, "plain", enrollment_config, run_dir=tmp_path, enrollment=_enrollment(commit=None)
        )
        assert embedder == []
        assert result.verdict.outcome is Outcome.FLAG
        assert "resolved model commit" in result.verdict.why

    def test_an_enrollment_from_another_model_is_refused(
        self, store: ProvStore, enrollment_config: TriageConfig, seeded_store: Callable[..., None],
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A similarity between two models' spaces is not a similarity."""
        seeded_store(store, tmp_path, words=["hello", "world"])
        _stub_diarizers(monkeypatch, primary_speakers=1, second_speakers=1)
        result = speech(
            store, "plain", enrollment_config, run_dir=tmp_path, enrollment=_enrollment(model="pyannote/embedding")
        )
        assert "not the probe" in result.verdict.why

    def test_a_null_enrollment_model_key_refuses_before_any_store_write(
        self, store: ProvStore, speech_config: TriageConfig, seeded_store: Callable[..., None],
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """speech.enrollment_model is null on the packaged config; nothing invents a probe."""
        seeded_store(store, tmp_path, words=["hello", "world"])
        _stub_diarizers(monkeypatch, primary_speakers=1, second_speakers=1)
        result = speech(store, "plain", speech_config, run_dir=tmp_path, enrollment=_enrollment())
        assert result.verdict.outcome is Outcome.FLAG
        assert "speech.enrollment_model" in result.verdict.why

    def test_the_enrollment_element_names_every_source(
        self, store: ProvStore, enrollment_config: TriageConfig, seeded_store: Callable[..., None],
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The store carries the enrollment, so a file's own contribution to its target is visible."""
        seeded_store(store, tmp_path, words=["hello", "world"])
        _stub_diarizers(monkeypatch, primary_speakers=1, second_speakers=1)
        _stub_embedder(monkeypatch, similarity=0.99)
        speech(store, "plain", enrollment_config, run_dir=tmp_path, enrollment=_enrollment())
        element = live_entities(store, "enrollment")[0]
        assert element.attributes["subject_id"] == "sub-01"
        assert element.attributes["sources"] == ["a.wav", "b.wav"]
        assert element.attributes["model_commit_sha"] == "a" * 40

    def test_a_hint_target_speaker_is_not_read_and_says_so(
        self, store: ProvStore, speech_config: TriageConfig, seeded_store: Callable[..., None],
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The ignore is never silent (V15)."""
        seeded_store(store, tmp_path, words=["hello", "world"])
        _stub_diarizers(monkeypatch, primary_speakers=1, second_speakers=1)
        hint = AudioHints(target_speaker=_target_speaker_embedding())
        result = speech(store, "plain", speech_config, hint, run_dir=tmp_path, enrollment=None)
        assert "identifies the target by enrollment" in result.verdict.why


class TestSeparationIsMeasurementGated:
    """Neither backend is selected by default, and the choice is a config key."""

    def test_a_null_backend_does_not_separate(
        self, store: ProvStore, speech_config: TriageConfig, seeded_store: Callable[..., None],
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A count of 2 with no ranked backend records the absence rather than picking one."""
        seeded_store(store, tmp_path, words=["hello", "world"])
        _stub_diarizers(monkeypatch, primary_speakers=2, second_speakers=2)
        separator = _stub_separator(monkeypatch)
        speech(store, "plain", speech_config, run_dir=tmp_path, enrollment=None)
        assert separator == []
        assert _verdict_entity(store, "SPEECH").attributes["separation"] == "not_selected"

    def test_mossformer_is_reachable_by_config(
        self, store: ProvStore, seeded_store: Callable[..., None], tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The alternative runs when named, at n_sources 2, and writes one stream per source."""
        config = _override(tmp_path, "speech:\n  separation_backend: MossFormer2_SS_16K\n")
        seeded_store(store, tmp_path, words=["hello", "world"])
        _stub_diarizers(monkeypatch, primary_speakers=2, second_speakers=2)
        separator = _stub_separator(monkeypatch, sources=2)
        speech(store, "plain", config, run_dir=tmp_path, enrollment=None)
        assert separator[0]["model"] == "alibabasglab/MossFormer2_SS_16K"
        assert separator[0]["n_sources"] == 2
        assert len([e for e in live_entities(store, "stream") if e.attributes["name"].startswith("separated")]) == 2

    def test_unasdiff_speech_sound_needs_a_sound_class(
        self, store: ProvStore, seeded_store: Callable[..., None], tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """V17: the spec wants an unconditioned sound slot; the API refuses one. The branch says so."""
        config = _override(tmp_path, "speech:\n  separation_backend: unasdiff\n")
        seeded_store(store, tmp_path, words=["hello", "world"])
        _stub_diarizers(monkeypatch, primary_speakers=2, second_speakers=2)
        separator = _stub_separator(monkeypatch)
        speech(store, "plain", config, run_dir=tmp_path, enrollment=None)
        assert separator == []
        assert (
            _verdict_entity(store, "SPEECH").attributes["separation"]
            == "unconditioned_sound_slot_unavailable"
        )

    def test_unasdiff_runs_in_speech_sound_mode_when_a_class_is_named(
        self, store: ProvStore, seeded_store: Callable[..., None], tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Slot 0 is the speech prior; the sound slot carries the configured class."""
        config = _override(
            tmp_path, "speech:\n  separation_backend: unasdiff\n  separation_sound_class: Applause\n"
        )
        seeded_store(store, tmp_path, words=["hello", "world"])
        _stub_diarizers(monkeypatch, primary_speakers=2, second_speakers=2)
        separator = _stub_separator(monkeypatch, sources=2)
        speech(store, "plain", config, run_dir=tmp_path, enrollment=None)
        assert separator[0]["mode"] == "speech_sound"
        assert separator[0]["source_classes"] == ["Applause"]

    def test_three_speakers_are_reported_not_separated(
        self, store: ProvStore, seeded_store: Callable[..., None], tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """MossFormer fixes n_sources at 2, so a count of 3 is a report, not a wrong decomposition."""
        config = _override(tmp_path, "speech:\n  separation_backend: MossFormer2_SS_16K\n")
        seeded_store(store, tmp_path, words=["hello", "world"])
        _stub_diarizers(monkeypatch, primary_speakers=3, second_speakers=3)
        separator = _stub_separator(monkeypatch)
        result = speech(store, "plain", config, run_dir=tmp_path, enrollment=None)
        assert separator == []
        assert "cannot serve 3" in result.verdict.why


class TestPiiOnTheConsensus:
    """One scan, one text, and the decision is speaker-scoped while the redaction is not."""

    def test_the_scan_reads_the_consensus_transcript_only(
        self, store: ProvStore, speech_config: TriageConfig, seeded_store: Callable[..., None],
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Exactly one text is scanned, and it is the consensus text PREPROCESS wrote."""
        seeded_store(store, tmp_path, words=["my", "name", "is", "alice"])
        _stub_diarizers(monkeypatch, primary_speakers=1, second_speakers=1)
        scanned = _stub_pii(monkeypatch, findings=[("PERSON", "alice")])
        speech(store, "plain", speech_config, run_dir=tmp_path, enrollment=None)
        assert scanned == ["my name is alice"]

    def test_a_finding_carries_category_and_extent_never_text(
        self, store: ProvStore, speech_config: TriageConfig, seeded_store: Callable[..., None],
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The verdict and the element both refuse to carry the matched text."""
        seeded_store(store, tmp_path, words=["my", "name", "is", "alice"])
        _stub_diarizers(monkeypatch, primary_speakers=1, second_speakers=1)
        _stub_pii(monkeypatch, findings=[("PERSON", "alice")])
        result = speech(store, "plain", speech_config, run_dir=tmp_path, enrollment=None)
        finding = live_entities(store, "pii")[0]
        assert finding.attributes["category"] == "PERSON"
        assert finding.extent is not None
        assert "alice" not in str(finding.attributes)
        assert "alice" not in result.verdict.why

    def test_a_finding_marks_the_word_elements(
        self, store: ProvStore, speech_config: TriageConfig, seeded_store: Callable[..., None],
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The store now holds PII, and every artifact must respect the marking."""
        seeded_store(store, tmp_path, words=["my", "name", "is", "alice"])
        _stub_diarizers(monkeypatch, primary_speakers=1, second_speakers=1)
        _stub_pii(monkeypatch, findings=[("PERSON", "alice")])
        speech(store, "plain", speech_config, run_dir=tmp_path, enrollment=None)
        marks = [
            e
            for e in live_entities(store, "assertion")
            if e.attributes.get("verb") == "label" and e.attributes.get("label") == "pii"
        ]
        assert marks and all("alice" not in str(m.attributes) for m in marks)

    def test_a_missing_required_detector_flags(
        self, store: ProvStore, speech_config: TriageConfig, seeded_store: Callable[..., None],
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A detector never attempted is the silent one, and could-not-check is not clean."""
        seeded_store(store, tmp_path, words=["hello"])
        _stub_diarizers(monkeypatch, primary_speakers=1, second_speakers=1)
        _stub_pii(monkeypatch, findings=[], detectors_used=["rules"])
        result = speech(store, "plain", speech_config, run_dir=tmp_path, enrollment=None)
        assert result.verdict.outcome is Outcome.FLAG
        assert find_measurement(store, "pii_scan").attributes["missing"] == ["gliner", "presidio"]

    def test_a_non_target_finding_does_not_flag_but_is_still_a_finding(
        self, store: ProvStore, enrollment_config: TriageConfig, seeded_store: Callable[..., None],
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Flagging asks whether a human is needed; the finding still reaches REDACT."""
        seeded_store(store, tmp_path, words=["my", "name", "is", "alice"], speakers=2)
        _stub_diarizers(monkeypatch, primary_speakers=2, second_speakers=2)
        _stub_embedder(monkeypatch, similarity=0.99, target_label="SPEAKER_00")
        _stub_pii(monkeypatch, findings=[("PERSON", "alice")], at_speaker="SPEAKER_01")
        speech(store, "plain", enrollment_config, run_dir=tmp_path, enrollment=_enrollment())
        verdict = _verdict_entity(store, "SPEECH")
        assert verdict.attributes["pii"]["n"] == 1
        assert not [flag for flag in verdict.attributes["flags"] if "target speaker's speech" in flag]


class TestTheNonTargetAxis:
    """Measured and reported per span; null, not zero, while the thresholds are unmeasured."""

    def test_the_three_legs_are_measured_per_span(
        self, store: ProvStore, speech_config: TriageConfig, seeded_store: Callable[..., None],
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Level, spectral tilt and direct-to-reverberant, on every speech span."""
        seeded_store(store, tmp_path, words=["hello", "world"])
        _stub_diarizers(monkeypatch, primary_speakers=1, second_speakers=1)
        speech(store, "plain", speech_config, run_dir=tmp_path, enrollment=None)
        proximity = find_measurements(store, "proximity")
        assert proximity
        for measurement in proximity:
            assert {"rms_dbfs", "peak_dbfs", "tilt_db_per_octave", "d_to_r_db"} <= set(measurement.attributes)

    def test_nontarget_speech_s_is_null_while_a_threshold_is_unmeasured(
        self, store: ProvStore, speech_config: TriageConfig, seeded_store: Callable[..., None],
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A product that says zero when nobody measured is the failure this row exists to prevent."""
        seeded_store(store, tmp_path, words=["hello", "world"])
        _stub_diarizers(monkeypatch, primary_speakers=1, second_speakers=1)
        speech(store, "plain", speech_config, run_dir=tmp_path, enrollment=None)
        assert _verdict_entity(store, "SPEECH").attributes["nontarget_speech_s"] is None

    def test_no_span_is_excluded_on_this_evidence(
        self, store: ProvStore, speech_config: TriageConfig, seeded_store: Callable[..., None],
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """This branch marks; it removes nothing."""
        seeded_store(store, tmp_path, words=["hello", "world"])
        _stub_diarizers(monkeypatch, primary_speakers=1, second_speakers=1)
        speech(store, "plain", speech_config, run_dir=tmp_path, enrollment=None)
        assert not [e for e in store.entities("span") if store.is_invalidated(e.id)]


class TestQualityAndTheStreamsItNames:
    """SQUIM on plain, disruptions on the original, and every reading names its stream (V19)."""

    def test_disruptions_read_the_original_recording(
        self, store: ProvStore, speech_config: TriageConfig, seeded_store: Callable[..., None],
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Peak normalisation and resampling destroy the plateaus and the crossing rate."""
        seeded_store(store, tmp_path, words=["hello", "world"])
        _stub_diarizers(monkeypatch, primary_speakers=1, second_speakers=1)
        speech(store, "plain", speech_config, run_dir=tmp_path, enrollment=None)
        recording_id = _stream_id(store, "recording")
        for measurement in find_measurements(store, "disruptions"):
            assert measurement.attributes["stream"] == recording_id

    def test_a_wordless_file_has_no_per_span_reading_and_that_is_correct(
        self, store: ProvStore, speech_config: TriageConfig, seeded_store: Callable[..., None], tmp_path: Path
    ) -> None:
        """A span nobody measured must not report zero; the file-level reading is PREPROCESS's."""
        seeded_store(store, tmp_path, words=[], disruptions_file=True)
        speech(store, "plain", speech_config, run_dir=tmp_path, enrollment=None)
        assert find_measurements(store, "disruptions") == []
        assert find_measurement(store, "disruptions_file") is not None


class TestItDoesNotReadAirway:
    """Diarization is a speech-only instrument."""

    def test_an_airway_label_withdraws_no_segment(
        self, store: ProvStore, speech_config: TriageConfig, seeded_store: Callable[..., None],
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The same store with and without AIRWAY's labels yields the same speaker count."""
        seeded_store(store, tmp_path, words=["hello", "world"], airway_labelled=[(0.4, 0.6)])
        _stub_diarizers(monkeypatch, primary_speakers=1, second_speakers=1)
        speech(store, "plain", speech_config, run_dir=tmp_path, enrollment=None)
        assert _verdict_entity(store, "SPEECH").attributes["speaker_count"] == 1
        assert not [e for e in store.entities("speaker") if store.is_invalidated(e.id)]

    def test_the_module_reads_no_airway_activity(self) -> None:
        """Verifying what commit 8537a83f already removed, so a regression is caught here."""
        source = Path(speech_module.__file__).read_text()
        assert "AIRWAY" not in source
```

The helpers `_verdict_entity`, `_stream_id`, `_override`, `_enrollment`,
`_target_speaker_embedding`, `_stub_diarizers`, `_stub_embedder`, `_stub_separator` and `_stub_pii` are
module-private in the same file. `_stub_diarizers` patches `speech_module.diarize_audios` and returns
the mutable call log the assertions read; `_stub_pii` patches `speech_module.scan_for_pii` and returns
the list of texts it was handed. `speech_config`, `second_diarizer_config` and `enrollment_config` are
`conftest.py` fixtures layering, respectively, `speech.word_gap_ms: 500`; that plus
`speech.second_diarizer: pyannote/speaker-diarization-3.1`; and that plus
`speech.enrollment_model: {model_id: speechbrain/spkrec-ecapa-voxceleb, revision: aaaa...}` and
`speech.target_match_cosine: 0.5`. `seeded_store` extends sibling T2's fixture with `word_extents`,
`speakers`, `airway_labelled` and `disruptions_file`.

- [ ] **Step 6 — run them; expect FAIL.**
  `uv run pytest src/tests/audio/workflows/triage/nodes/speech_test.py -x -q`
  Record the first failure verbatim; `test_a_zero_length_interval_is_not_diarized` is the one whose
  pre-fix failure must read `ValueError: 'waveform' must be provided as a (channel, time) torch
  Tensor` when the diarizer stub is replaced by the real call — note that in the commit body.

- [ ] **Step 7 — edit `speech.py`.**

Ordered by the spec's own step numbers.

**Signature.** `speech(store, source, config, hint=None, *, run_dir, enrollment: Enrollment | None = None)`.

**`_required`.** Drops the `hint.target_speaker` branch. Adds, when `enrollment is not None`, a
resolution of `speech.enrollment_model` through `config.require`, which is what makes a null key
refuse before any store write:

```python
    if enrollment is not None:
        model = config.require("speech.enrollment_model")
        values["enrollment_model_id"] = str(model["model_id"])
        values["enrollment_revision"] = str(model["revision"])
        values["target_match_cosine"] = float(config.require("speech.target_match_cosine"))
```

`config.require` raising on a null `speech.enrollment_model` is caught by the caller of `_required`
and turned into the flag `test_a_null_enrollment_model_key_refuses_before_any_store_write` expects:

```python
    try:
        values = _required(config, enrollment)
    except ValueError as error:
        return _flag_before_measuring(store, software, f"{error}")
```

**Step 1 (transcript).** The whole `raw_words` gathering and `fuse_word_streams` call is **deleted**.
Replaced by:

```python
    consensus = find_measurement(store, "consensus_transcript")
    if consensus is None:
        raise LookupError("no consensus_transcript in the store; PREPROCESS has not run")
    words = [store.get_entity(word_id) for word_id in consensus.attributes["word_ids"]]
    words = [word for word in words if not store.is_invalidated(word.id)]
    transcript_text = str(consensus.attributes["text"])
```

The `fabrication` check keeps its energy test but reads a consensus word's `recognizers`, per the
spec's "A word carried by one recognizer alone is not a consensus word":

```python
    single_source = [
        word.id for word in words if len(word.attributes.get("recognizers") or []) == 1
    ]
    if single_source:
        flags.append(f"{len(single_source)} single-recognizer word(s) survive as fabrication candidates")
```

**Step 4 (diarization) — the V18 fix.** After computing `interval`:

```python
    frame_samples = 1
    interval_samples = int(round((interval[1] - interval[0]) * sr))
    if interval_samples <= frame_samples:
        count: int | None = None
        diarization_state = "interval_shorter_than_one_frame"
        speaker_segments = []
        flags.append(
            "the consensus places every word inside one analysis frame; "
            "the diarization interval is not a diarizable signal"
        )
    else:
        (cropped,) = extract_segments([(plain, [interval])])[0]
        [segments] = diarize_audios([cropped], model=diarizer)
        ...
        diarization_state = "diarized"
```

`frame_samples = 1` is definitional, not a threshold: it is the smallest waveform pyannote's own guard
(`shape[0] > shape[1]`, `pyannote/audio/core/io.py:173`) accepts for a mono input, so the boundary is
the library's, not a value this plan chose. Every downstream read of `count` handles `None`: the
second diarizer is not consulted, separation does not run, `speaker_count` is written as `None`, and
attribution marks every word `unassigned`.

**Step 5 (separation).** The `_separation_model()` hard-coded MossFormer is replaced:

```python
    backend = config.get("speech.separation_backend")
    sound_class = config.get("speech.separation_sound_class")
    if count is None or count < 2:
        separation_state: Any = "not_needed"
    elif backend is None:
        separation_state = "not_selected"
    elif count >= 3:
        separation_state = f"count_{count}_exceeds_backend"
        flags.append(f"separation cannot serve {count} speakers; the checkpoints separate exactly 2")
    elif str(backend) == "unasdiff" and sound_class is None:
        separation_state = "unconditioned_sound_slot_unavailable"
        flags.append(
            "unasdiff speech_sound requires a conditioning class for its sound slot and "
            "speech.separation_sound_class is unmeasured"
        )
    elif str(backend) == "unasdiff":
        separated = separate_audios(
            [cropped], model=None, n_sources=2, mode="speech_sound", source_classes=[str(sound_class)]
        )[0]
        separation_state = {"backend": "unasdiff", "mode": "speech_sound", "source_classes": [str(sound_class)]}
    else:
        separator = _clearvoice_model(str(backend))
        separated = separate_audios([cropped], model=separator, n_sources=2)[0]
        separation_state = {"backend": str(backend), "n_sources": 2}
```

**Step 6 (identification).** The `hint.target_speaker` block is deleted in full. Replaced by an
enrollment block that (a) writes the `enrollment` element, (b) refuses on
`enrollment.refusal_against(values["enrollment_model_id"])`, (c) embeds each diarized speaker with
`SpeechBrainModel(path_or_uri=values["enrollment_model_id"], revision=values["enrollment_revision"])`,
(d) writes one `target_match` per speaker naming both embeddings' model and revision, and (e) marks
each speech span `attributed_to` and, where the speaker is not the target, `nontarget`. **No span is
invalidated.**

The `enrollment` element:

```python
        enrollment_id = store.entity(
            prov_type="enrollment",
            extent=None,
            attributes={
                "subject_id": enrollment.subject_id,
                "model_id": enrollment.provenance.model_id,
                "model_commit_sha": enrollment.provenance.model_commit_sha,
                "unresolved_reason": enrollment.provenance.unresolved_reason,
                "task": enrollment.task,
                "method": enrollment.provenance.method,
                "sources": enrollment.sources,
                "n_windows_used": enrollment.provenance.n_windows_used,
                "n_windows_dropped": enrollment.provenance.n_windows_dropped,
                "dimension": len(enrollment.vector),
            },
        )
```

The vector itself is **not** written to the store: it is the caller's input, it is reproducible from
`sources` and `provenance`, and an embedding in an append-only store is one more thing a release has
to reason about. `dimension` is what a reader needs to see the comparison was well-formed.

**Step 7 (PII).** The per-span, per-recognizer `ScriptLine` assembly is **deleted**. Replaced by one
scan over `transcript_text`, with each finding located against the consensus words by the existing
`_locate` (now taking the `word` entities rather than raw dicts). Each `pii` entity gains
`recognizers`, from the consensus word(s) it overlaps, so a finding resting on one recognizer alone is
legible. `_decide_pii` is unchanged in rule.

**Step 9 (the non-target axis), new.** Per speech span, on `plain`:

```python
    def _proximity(segment: np.ndarray, reference_rms_dbfs: float) -> dict[str, float]:
        """The proximity leg's three measures over one span, against the file's own reference level.

        Args:
            segment: The span's samples, mono.
            reference_rms_dbfs: The file's RMS from PREPROCESS's ``level`` measurement.

        Returns:
            ``{rms_dbfs, peak_dbfs, level_over_reference_db, tilt_db_per_octave, d_to_r_db}``.
        """
```

`tilt_db_per_octave` is the least-squares slope of the span's log-magnitude spectrum against
`log2(frequency)`; `d_to_r_db` is `10*log10(peak_energy / (total_energy - peak_energy))` over the
span's autocorrelation, taking the peak lag as direct and the tail as reverberant. Both are written to
a `measurement(name="proximity")` per span, `stream` named. **Neither is compared to anything**: the
three thresholds are null, so `nontarget_speech_s` is written `None` and no span carries a `nontarget`
marking on this evidence.

**The verdict detail** becomes exactly `branch-speech.md`'s product:

```python
    detail: dict[str, Any] = {
        "speaker_count": count,
        "diarization": diarization_state,
        "words_n": len(words),
        "speech_s": speech_s,
        "nontarget_speech_s": nontarget_speech_s,
        "pii": {
            "categories": categories,
            "n": len(findings),
            "scanned_by": sorted(scanned_by),
            "failed": sorted(failures),
            "missing": missing,
        },
        "second_diarizer": second_record,
        "separation": separation_state,
        "flags": flags,
    }
    if target_speaker is not None:
        detail["target_speaker"] = target_speaker
    if enrollment is not None:
        detail["enrollment_id"] = enrollment_id
```

- [ ] **Step 8 — forward `enrollment` through `run.py`.** Sibling T3 already threads the parameter;
  narrow its type from `Any` to `Enrollment | None` and import it.

- [ ] **Step 9 — run them; expect PASS.**
  `uv run pytest src/tests/audio/workflows/triage/nodes/speech_test.py src/tests/audio/workflows/triage/enrollment_test.py -x -q`

- [ ] **Step 10 — lint, type-check.**
  `uv run ruff format src/senselab/audio/workflows/triage src/tests/audio/workflows/triage`
  `uv run ruff check src/senselab/audio/workflows/triage src/tests/audio/workflows/triage`
  `uv run mypy src/senselab/audio/workflows/triage`

- [ ] **Step 11 — commit.**
  `git commit -m "feat(triage/speech): the target is enrolled, and a one-frame interval is not diarized"`

**Interfaces:**

*Consumed:* sibling T1's `consensus_transcript` (its `word_ids`, `event_ids` and `text`), `word` and
`event` entities, `yamnet_windows`, `squim` assertions, `energy_envelope`, `silence`, `level`, `span`
entities; sibling T3's `branch_decision` (read by `run.py`, not by this node);
`diarize_audios(audios, model=..., ...)`, `separate_audios(audios, model=None, n_sources=2, mode=..., source_classes=[...])`,
`extract_speaker_embeddings_from_audios(audios, model=...)`, `scan_for_pii(inputs, detectors=None)`,
`detect_disruptions`, `extract_objective_quality_features_from_audios`, `extract_segments`.

*Produced (the T4→T7 and T4→T8 contract):*
- `speech(store, source, config, hint=None, *, run_dir, enrollment=None) -> NodeResult`.
- `Enrollment` (importable from `senselab.audio.workflows.triage`), with `.sources` and
  `.refusal_against(probe_model_id)`.
- Store: `speech` `span` entities carrying `attributed_to` and `nontarget`; `speaker` entities per
  diarizer; `interval` `diarization_interval`; `stream` `separated_*`; `enrollment`; `target_match`;
  `pii` entities and the `pii_scan` measurement; `proximity`, `squim` and `disruptions` measurements
  per span; the `SPEECH` verdict with the detail above.
- **`pii` entity presence is what `run._speech_found_pii` gates REDACT on** — sibling T3's contract.

**Superseded tests, deleted with the ruling that justifies each:**

| deleted | ruling |
| --- | --- |
| every `speech_test.py` assertion on `fuse_word_streams` being called here | branch-speech.md: "PREPROCESS produced it with `fuse_consensus_words`; this branch reads it and does not re-fuse" |
| every `hint.target_speaker` identification test | branch-speech.md §6: "The target speaker is identified by an embedding enrolled across all of the subject's provided recordings, not by a per-file target hint" |
| the per-span, per-recognizer PII scan tests | branch-speech.md §7: "One scan, one text" |
| the unconditional-second-diarizer test | branch-speech.md §4: a count of 1 consults nobody |
| the "separation runs at count 2" test | branch-speech.md §5: "neither is selected by default, the choice is the config key `speech.separation_backend`, and it ships null" |

---

### Task 5: VOICE v2 — measure PREPROCESS's phonation spans; there is no residual

**Scope:** `src/senselab/audio/workflows/triage/nodes/voice.py` (rewritten);
`src/senselab/audio/workflows/triage/data/config/default.yaml` (three new `voice.*` keys);
`src/tests/audio/workflows/triage/nodes/voice_test.py` (rewritten).

**Design points this task must not get wrong (from `branch-voice.md`):**

- **There is no residual.** VOICE measures the phonation spans PREPROCESS detected. It does **not**
  subtract AIRWAY's labels or SPEECH's spans from an energy track and analyse what is left. The
  `_subtract_intervals`, `_airway_labelled` and `_speech_spans` helpers are **deleted**.
- **The kind is `voice`**, not `voice_no_words`.
- **The subject's production may be voiced, unvoiced or mixed**, and no span is refused for being
  unvoiced. An unvoiced span carries no period marks and is not thereby a failure.
- **Tracks are computed once on the stream and then sliced.** (Already true since 29da3633 — verify,
  keep.)
- **Period marks are a point process, absent outside voiced and mixed spans** — not zero, not
  interpolated.
- **The onset is a period where one exists; the offset is a criterion.** The two edges are not the
  same kind of quantity and the product does not present them as one.
- **`longest_span_s` is a first-class product**, reported with the criterion that closed it, because a
  task measurement a reader has to reassemble from fragments is not recoverable.
- **The F0 range is a property of the declared population.** `voice.f0_range_hz`, overridable through
  `voice.f0_range_by_population`; a configuration whose `f0_max / f0_min` exceeds
  `voice.f0_range_ratio_max` is **refused at load, not run and flagged**.
- **`min_marks_s` gets a half-frame tolerance derived from the hop (V20)** — a frame stands for a
  hop-wide interval centred on its time.

**Steps:**

- [ ] **Step 1 — add the three `voice.*` keys.**

Add to `derivation:` in `data/config/default.yaml`:

```
  voice v2 -- branch-voice.md. voice.f0_range_by_population overrides voice.f0_range_hz per declared
  age and sex; null, owed a fit per population, and a range spanning too wide an interval makes any
  period-doubling test on it vacuous. voice.f0_range_ratio_max is the f0_max / f0_min above which the
  period-doubling check reports nothing because it flags everything; it is null, and a configuration
  exceeding it is REFUSED AT LOAD rather than run and flagged -- a check that fires on every file
  transports no information, and running it anyway would put that non-information into every verdict.
  While the ratio is null no configuration is refused, which is the honest state: nobody has fixed the
  bound. voice.task_duration_ranges is the expected duration range per declared task, against which a
  span outside it flags with the declared range named; null, because no corpus here establishes what a
  maximum-phonation-time task should produce.
```

and the keys, under the existing `voice:` block:

```yaml
  f0_range_by_population: null
  f0_range_ratio_max: null
  task_duration_ranges: null
```

- [ ] **Step 2 — write the failing tests.**

Replace `src/tests/audio/workflows/triage/nodes/voice_test.py`. The classes:

```python
class TestTheSubjectIsPreprocessesSpans:
    """VOICE measures what PREPROCESS detected. Nothing is subtracted from anything."""

    def test_the_spans_are_preprocesses_phonation_spans(
        self, store: ProvStore, voice_config: TriageConfig, seeded_store: Callable[..., None], tmp_path: Path
    ) -> None:
        """spans_n is the count of phonation spans in the store, not of a residual."""
        seeded_store(store, tmp_path, phonation=[(0.0, 1.5, "voiced"), (2.0, 2.8, "voiced")])
        voice(store, "plain", voice_config, run_dir=tmp_path)
        assert _verdict_entity(store, "VOICE").attributes["spans_n"] == 2

    def test_a_speech_span_removes_nothing(
        self, store: ProvStore, voice_config: TriageConfig, seeded_store: Callable[..., None], tmp_path: Path
    ) -> None:
        """branch-voice.md: 'Nothing another branch claimed is removed from this branch's subject'."""
        seeded_store(
            store, tmp_path, phonation=[(0.0, 1.5, "voiced")], speech_spans=[(0.0, 1.5)]
        )
        voice(store, "plain", voice_config, run_dir=tmp_path)
        assert _verdict_entity(store, "VOICE").attributes["spans_n"] == 1

    def test_an_airway_label_removes_nothing(
        self, store: ProvStore, voice_config: TriageConfig, seeded_store: Callable[..., None], tmp_path: Path
    ) -> None:
        """Nothing this branch measures is conditioned on what another branch concluded."""
        seeded_store(
            store, tmp_path, phonation=[(0.0, 1.5, "voiced")], airway_labelled=[(0.0, 1.5)]
        )
        voice(store, "plain", voice_config, run_dir=tmp_path)
        assert _verdict_entity(store, "VOICE").attributes["spans_n"] == 1

    def test_the_module_computes_no_residual(self) -> None:
        """The three residual helpers are deleted, not left unreachable."""
        for name in ("_subtract_intervals", "_airway_labelled", "_speech_spans"):
            assert not hasattr(voice_module, name)

    def test_no_phonation_span_fails(
        self, store: ProvStore, voice_config: TriageConfig, seeded_store: Callable[..., None], tmp_path: Path
    ) -> None:
        """This path is reached only when a hint forced the branch, which routing gates on the same fact."""
        seeded_store(store, tmp_path, phonation=[])
        assert voice(store, "plain", voice_config, run_dir=tmp_path).verdict.outcome is Outcome.FAIL

    def test_the_kind_is_voice(
        self, store: ProvStore, voice_config: TriageConfig, seeded_store: Callable[..., None], tmp_path: Path
    ) -> None:
        """voice_no_words is gone; VERDICT joins branch to kind on this string."""
        seeded_store(store, tmp_path, phonation=[(0.0, 1.5, "voiced")])
        assert voice(store, "plain", voice_config, run_dir=tmp_path).verdict.kind == "voice"


class TestProductionModes:
    """Voiced, unvoiced and mixed are all measured; an unvoiced span is not a failure."""

    def test_an_unvoiced_span_is_measured(
        self, store: ProvStore, voice_config: TriageConfig, seeded_store: Callable[..., None], tmp_path: Path
    ) -> None:
        """A disordered voice sustaining without periodicity is exactly what must be measured."""
        seeded_store(store, tmp_path, phonation=[(0.0, 1.5, "unvoiced")])
        result = voice(store, "plain", voice_config, run_dir=tmp_path)
        assert result.verdict.outcome is not Outcome.FAIL
        assert _verdict_entity(store, "VOICE").attributes["production"]["unvoiced"] == 1

    def test_an_unvoiced_span_carries_no_period_marks(
        self, store: ProvStore, voice_config: TriageConfig, seeded_store: Callable[..., None], tmp_path: Path
    ) -> None:
        """Absent, not zero and not interpolated: its duration, formants and level are its measurement."""
        seeded_store(store, tmp_path, phonation=[(0.0, 1.5, "unvoiced")])
        voice(store, "plain", voice_config, run_dir=tmp_path)
        marks = find_measurements(store, "period_marks")
        assert marks and "n" not in marks[0].attributes
        assert marks[0].attributes["unmeasured"] == "unvoiced_span"

    def test_the_production_counts_are_reported(
        self, store: ProvStore, voice_config: TriageConfig, seeded_store: Callable[..., None], tmp_path: Path
    ) -> None:
        """The verdict's production block is a count per mode, as branch-voice.md's product names it."""
        seeded_store(
            store, tmp_path,
            phonation=[(0.0, 1.0, "voiced"), (2.0, 3.0, "unvoiced"), (4.0, 5.0, "mixed")],
        )
        voice(store, "plain", voice_config, run_dir=tmp_path)
        assert _verdict_entity(store, "VOICE").attributes["production"] == {
            "voiced": 1, "unvoiced": 1, "mixed": 1
        }


class TestMptRecoverableProducts:
    """longest_span_s and its criterion, so a task measurement is not reassembled from fragments."""

    def test_longest_span_s_is_a_first_class_product(
        self, store: ProvStore, voice_config: TriageConfig, seeded_store: Callable[..., None], tmp_path: Path
    ) -> None:
        """The longest span's duration, reported directly."""
        seeded_store(store, tmp_path, phonation=[(0.0, 1.0, "voiced"), (2.0, 5.5, "voiced")])
        voice(store, "plain", voice_config, run_dir=tmp_path)
        assert _verdict_entity(store, "VOICE").attributes["longest_span_s"] == pytest.approx(3.5)

    def test_the_criterion_that_closed_it_travels_with_it(
        self, store: ProvStore, voice_config: TriageConfig, seeded_store: Callable[..., None], tmp_path: Path
    ) -> None:
        """A duration without its offset criterion is not a maximum phonation time."""
        seeded_store(store, tmp_path, phonation=[(0.0, 3.5, "voiced")])
        voice(store, "plain", voice_config, run_dir=tmp_path)
        assert _verdict_entity(store, "VOICE").attributes["longest_span_criterion"] == "f0_stability"

    def test_phonation_s_totals_every_span(
        self, store: ProvStore, voice_config: TriageConfig, seeded_store: Callable[..., None], tmp_path: Path
    ) -> None:
        """The total is over the spans, whatever their production mode."""
        seeded_store(store, tmp_path, phonation=[(0.0, 1.0, "voiced"), (2.0, 2.5, "unvoiced")])
        voice(store, "plain", voice_config, run_dir=tmp_path)
        assert _verdict_entity(store, "VOICE").attributes["phonation_s"] == pytest.approx(1.5)

    def test_a_declared_task_outside_its_range_flags_with_the_range_named(
        self, store: ProvStore, seeded_store: Callable[..., None], tmp_path: Path
    ) -> None:
        """The task conditions how a duration is reported, never whether a span exists."""
        config = _override(
            tmp_path,
            "voice:\n  f0_range_hz: [75, 500]\n  task_duration_ranges: {maximum_phonation_time: [10.0, 40.0]}\n",
        )
        seeded_store(store, tmp_path, phonation=[(0.0, 3.5, "voiced")])
        hint = AudioHints(metadata={"task": "maximum_phonation_time"})
        result = voice(store, "plain", config, hint, run_dir=tmp_path)
        assert result.verdict.outcome is Outcome.FLAG
        assert "10.0" in result.verdict.why and "40.0" in result.verdict.why

    def test_a_null_task_range_leaves_the_row_inert(
        self, store: ProvStore, voice_config: TriageConfig, seeded_store: Callable[..., None], tmp_path: Path
    ) -> None:
        """Nobody derived a range, so no span is out of one."""
        seeded_store(store, tmp_path, phonation=[(0.0, 3.5, "voiced")])
        hint = AudioHints(metadata={"task": "maximum_phonation_time"})
        result = voice(store, "plain", voice_config, hint, run_dir=tmp_path)
        assert result.verdict.outcome is not Outcome.FLAG
        assert _verdict_entity(store, "VOICE").attributes["task_range"] == "not_evaluated"


class TestTheHalfFrameTolerance:
    """A frame stands for a hop-wide interval centred on its time (V20)."""

    def test_a_span_of_exactly_min_marks_s_is_measured(
        self, store: ProvStore, voice_config: TriageConfig, seeded_store: Callable[..., None],
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Without the tolerance this span reads one hop short and its marks are skipped."""
        hop_s = 0.01
        min_marks_s = 3.0 / 75.0
        start, end = 1.0, 1.0 + min_marks_s - hop_s
        seeded_store(store, tmp_path, phonation=[(start, end, "voiced")], hop_s=hop_s)
        calls = _stub_period_marks(monkeypatch, marks=4)
        voice(store, "plain", voice_config, run_dir=tmp_path)
        assert calls, "the frame-edge tolerance is one hop; this span reaches min_marks_s with it"
        assert _verdict_entity(store, "VOICE").attributes["marks_skipped_short_n"] == 0

    def test_a_span_one_hop_shorter_still_is_skipped(
        self, store: ProvStore, voice_config: TriageConfig, seeded_store: Callable[..., None],
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The tolerance is one hop, not an open-ended slack."""
        hop_s = 0.01
        min_marks_s = 3.0 / 75.0
        seeded_store(store, tmp_path, phonation=[(1.0, 1.0 + min_marks_s - 2 * hop_s, "voiced")], hop_s=hop_s)
        calls = _stub_period_marks(monkeypatch, marks=4)
        voice(store, "plain", voice_config, run_dir=tmp_path)
        assert calls == []
        marks = find_measurements(store, "period_marks")
        assert marks[0].attributes["unmeasured"] == "shorter_than_mark_window"

    def test_the_tolerance_is_recorded_as_the_hop_not_a_constant(
        self, store: ProvStore, voice_config: TriageConfig, seeded_store: Callable[..., None], tmp_path: Path
    ) -> None:
        """The activity's parameters must show where the tolerance came from."""
        seeded_store(store, tmp_path, phonation=[(0.0, 1.5, "voiced")], hop_s=0.01)
        voice(store, "plain", voice_config, run_dir=tmp_path)
        analyze = next(a for a in store.activities("VOICE") if a.step == "analyze")
        assert analyze.parameters["frame_edge_tolerance_s"] == pytest.approx(0.01)


class TestTheF0RangeServesAPopulation:
    """The range is declared, overridable per population, and a vacuous ratio is refused at load."""

    def test_a_population_override_replaces_the_range(
        self, store: ProvStore, seeded_store: Callable[..., None], tmp_path: Path
    ) -> None:
        """Age and sex move the range; the hint names which population."""
        config = _override(
            tmp_path,
            "voice:\n  f0_range_hz: [75, 500]\n  f0_range_by_population: {adult_male: [60, 250]}\n",
        )
        seeded_store(store, tmp_path, phonation=[(0.0, 1.5, "voiced")])
        hint = AudioHints(metadata={"population": "adult_male"})
        voice(store, "plain", config, hint, run_dir=tmp_path)
        analyze = next(a for a in store.activities("VOICE") if a.step == "analyze")
        assert analyze.parameters["f0_range_hz"] == [60.0, 250.0]

    def test_a_vacuous_ratio_is_refused_before_the_store_is_touched(
        self, store: ProvStore, seeded_store: Callable[..., None], tmp_path: Path
    ) -> None:
        """A check that flags everything reports nothing, so it is refused rather than run and flagged."""
        config = _override(tmp_path, "voice:\n  f0_range_hz: [50, 800]\n  f0_range_ratio_max: 4.0\n")
        seeded_store(store, tmp_path, phonation=[(0.0, 1.5, "voiced")])
        before = len(store.entities())
        with pytest.raises(ValueError, match="f0_range_ratio_max"):
            voice(store, "plain", config, run_dir=tmp_path)
        assert len(store.entities()) == before

    def test_a_null_ratio_refuses_nothing(
        self, store: ProvStore, voice_config: TriageConfig, seeded_store: Callable[..., None], tmp_path: Path
    ) -> None:
        """Nobody fixed the bound, so no configuration exceeds it."""
        seeded_store(store, tmp_path, phonation=[(0.0, 1.5, "voiced")])
        assert voice(store, "plain", voice_config, run_dir=tmp_path).verdict.outcome is not Outcome.FAIL


class TestEdgesAreNamedApart:
    """The onset is a period where one exists; the offset is always a criterion."""

    def test_a_span_with_marks_has_a_period_onset(
        self, store: ProvStore, voice_config: TriageConfig, seeded_store: Callable[..., None],
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """An observed event, named as one."""
        seeded_store(store, tmp_path, phonation=[(0.0, 1.5, "voiced")])
        _stub_period_marks(monkeypatch, marks=6)
        voice(store, "plain", voice_config, run_dir=tmp_path)
        span = _voice_spans(store)[0]
        assert span.attributes["onset_kind"] == "period"
        assert span.attributes["offset_kind"] == "criterion"

    def test_f0_median_is_reported_only_with_its_stream(
        self, store: ProvStore, voice_config: TriageConfig, seeded_store: Callable[..., None],
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Two F0 values from two streams are two measurements, never one."""
        seeded_store(store, tmp_path, phonation=[(0.0, 1.5, "voiced")])
        _stub_period_marks(monkeypatch, marks=6)
        voice(store, "plain", voice_config, run_dir=tmp_path)
        detail = _verdict_entity(store, "VOICE").attributes
        assert ("f0_median_hz" in detail) == ("f0_stream" in detail)
```

- [ ] **Step 3 — run them; expect FAIL.**
  `uv run pytest src/tests/audio/workflows/triage/nodes/voice_test.py -x -q`

- [ ] **Step 4 — rewrite `voice.py`.**

Delete `_subtract_intervals`, `_airway_labelled`, `_speech_spans`, `_contiguous_true`, `_runs_of_true`
and `_generating_node` — every one served the residual. Delete `KIND = "voice_no_words"`; it becomes
`KIND = "voice"`.

The new `_required`, resolving the range and refusing a vacuous one **before the store is touched**:

```python
def _f0_range(config: TriageConfig, hint: AudioHints | None) -> tuple[float, float]:
    """The F0 search range for this recording's declared population.

    Args:
        config: The triage configuration.
        hint: The caller's hint; ``metadata["population"]`` selects an override.

    Returns:
        ``(f0_min_hz, f0_max_hz)``.

    Raises:
        ValueError: If ``voice.f0_range_hz`` is unmeasured, if the named population has no entry, or
            if ``f0_max / f0_min`` exceeds ``voice.f0_range_ratio_max`` — a period-doubling check over
            a range that wide flags every recording, and a check that fires on everything reports
            nothing, so the configuration is refused rather than run and flagged.
    """
    population = str(hint.metadata.get("population")) if hint is not None and hint.metadata.get("population") else None
    by_population = config.get("voice.f0_range_by_population") or {}
    raw = by_population.get(population) if population is not None else None
    if raw is None:
        raw = config.require("voice.f0_range_hz")
    f0_min_hz, f0_max_hz = float(raw[0]), float(raw[1])
    ratio_max = config.get("voice.f0_range_ratio_max")
    if ratio_max is not None and f0_max_hz / f0_min_hz > float(ratio_max):
        raise ValueError(
            f"voice.f0_range_ratio_max is {float(ratio_max)} and the declared range "
            f"[{f0_min_hz}, {f0_max_hz}] has ratio {f0_max_hz / f0_min_hz:.2f}; the period-doubling "
            "check over that range flags every recording and is refused rather than run"
        )
    return f0_min_hz, f0_max_hz
```

The subject read replaces the residual fold in full:

```python
    spans = [
        entity
        for entity in live_entities(store, "span")
        if entity.attributes.get("family") == _PHONATION_FAMILY and entity.extent is not None
    ]
    spans.sort(key=lambda entity: entity.extent or (0.0, 0.0))
```

The V20 tolerance, computed once and recorded on the activity:

```python
    hop_s = float(config.require("phonation_spans.hop_s"))
    min_marks_s = _MARK_PERIODS / f0_min_hz
    # A frame stands for a hop-wide interval centred on its time, so a span's measurable extent runs
    # from half a hop before its first frame to half a hop after its last: the tolerance is the hop,
    # an identity of the analysis grid.
    frame_edge_tolerance_s = hop_s
```

and, per span:

```python
        measurable_s = (span.extent[1] - span.extent[0]) + frame_edge_tolerance_s
        production = str(span.attributes["production"])
        if production == "unvoiced":
            marks_attributes = {"name": "period_marks", "signal": source, "unmeasured": _UNVOICED_SPAN}
            marks = []
        elif measurable_s < min_marks_s:
            marks_attributes = {"name": "period_marks", "signal": source, "unmeasured": _MARKS_UNMEASURED}
            marks = []
            marks_skipped_short_n += 1
        else:
            marks = period_marks(
                plain, span.extent[0], span.extent[1], f0_min_hz=f0_min_hz, f0_max_hz=f0_max_hz
            )
            marks_attributes = {
                "name": "period_marks",
                "signal": source,
                "n": len(marks),
                "marks": [{"time_s": m.time_s, "period_s": m.period_s, "amplitude": m.amplitude} for m in marks],
            }
```

with `_UNVOICED_SPAN = "unvoiced_span"` beside the existing `_MARKS_UNMEASURED` — two vocabulary
tokens, two different reasons nobody looked, and a reader can tell them apart.

The tracks block is **kept unchanged**: `hnr_track`, `f0_track` and `_rms_track` are already computed
once over the whole stream and sliced by time. Its slices are now indexed by the phonation spans
rather than by residual intervals.

The verdict detail becomes exactly `branch-voice.md`'s product:

```python
    detail: dict[str, Any] = {
        "spans_n": len(spans),
        "phonation_s": phonation_s,
        "longest_span_s": longest_span_s,
        "longest_span_criterion": longest_span_criterion,
        "production": {"voiced": voiced_n, "unvoiced": unvoiced_n, "mixed": mixed_n},
        "ambiguous_spans_n": ambiguous_spans_n,
        "marks_skipped_short_n": marks_skipped_short_n,
        "task_range": task_range,
        "gate_interval": gate_interval,
        "flags": flags,
    }
    if f0_median_hz is not None:
        detail["f0_median_hz"] = f0_median_hz
        detail["f0_stream"] = source
```

`longest_span_s` is `max(duration_s)` over the spans and `longest_span_criterion` is that span's
`offset_criterion` attribute, read from PREPROCESS — the criterion that closed it, travelling with it.
`task_range` is `"not_evaluated"` while `voice.task_duration_ranges` is null, and otherwise
`{task, range, longest_span_s, within}` with a flag naming the declared range when it is not within.

- [ ] **Step 5 — run them; expect PASS.**
  `uv run pytest src/tests/audio/workflows/triage/nodes/voice_test.py -x -q`

- [ ] **Step 6 — lint, type-check, commit.**
  `uv run ruff format src/senselab/audio/workflows/triage src/tests/audio/workflows/triage`
  `uv run ruff check src/senselab/audio/workflows/triage src/tests/audio/workflows/triage`
  `uv run mypy src/senselab/audio/workflows/triage`
  `git commit -m "feat(triage/voice): measure the phonation spans, and drop the residual"`

**Interfaces:**

*Consumed:* sibling T1's `span` entities with `family == "phonation"` and their `duration_s`,
`production`, `offset_criterion`, `f0_median_hz`; `formant_tracks` measurements; `energy_envelope`;
`silence`; `word` and `event` entities; `phonation_spans.hop_s`; `hnr_track`, `f0_track`,
`period_marks`; `common.live_entities`, `find_measurement`.

*Produced (the T5→T8 contract):*
- `voice(store, source, config, hint=None, *, run_dir) -> NodeResult` with `verdict.kind == "voice"`.
- Store: `span` entities for each measured phonation span with `onset_kind`/`offset_kind`;
  `period_marks` measurements per span (carrying `n` and `marks`, **or** `unmeasured` naming
  `"unvoiced_span"` or `"shorter_than_mark_window"`); `voice_tracks`; the `VOICE` verdict with the
  detail above.

**Superseded tests, deleted with the ruling that justifies each:**

| deleted | ruling |
| --- | --- |
| every residual test (`test_a_speech_span_is_subtracted`, `test_an_airway_label_is_subtracted`, `short_intervals_n`) | branch-voice.md: "There is no residual. VOICE measures the phonation spans PREPROCESS detected" |
| every `voice_no_words` assertion | taxonomy.md and branch-voice.md both name the kind `voice` |
| `runs_n` / `voiced_s` verdict-field tests | branch-voice.md's product names `spans_n`, `phonation_s`, `longest_span_s`, `production` |
| the energy-and-periodicity gate tests | branch-voice.md: the gate is PREPROCESS's continuity criterion; this branch opens no span |

---

### Task 6: AIRWAY v2 — HeAR as span confirmation, a near-gate band, and co-located contest only

**Scope:** `src/senselab/audio/workflows/triage/nodes/airway.py` (edited);
`src/tests/audio/workflows/triage/nodes/airway_test.py` (edited). No config change — sibling T1
created `airway.k_db`, `airway.k_db_by_task`, `airway.k_margin_db` and `airway.contest_labels`.

**Design points this task must not get wrong (from `branch-airway.md`):**

- **This branch runs no classifier.** The `detect_health_acoustic_events` and `span_to_hear_buffer`
  calls move out; every window classification it reads was written by PREPROCESS. **HeAR confirms a
  span; it does not find one.**
- **A span is eligible for a HeAR label only if it carries no non-cough/breath transcript.** A span
  overlapping consensus `word` entities is transcribed content and is not offered to HeAR; a span
  overlapping only `event` entities remains eligible.
- **The label is the `labels_of_interest` member confident in the overlapping `hear_windows` —
  membership in the window's set, not a score compared here.**
- **A contest requires co-location in the same HeAR window (V21)**, and the contesting labels are the
  declared `airway.contest_labels`, not all 521.
- **`airway.contest_labels` and `taxonomy.audioset_airway_labels` are disjoint, and the config is
  refused if they intersect.** A label cannot both support and contest the same conclusion.
- **`K` is per task**: `airway.k_db` overrides `spans.k_db.airway`, and `airway.k_db_by_task` overrides
  that per declared task.
- **A labelled span within `airway.k_margin_db` of the gate flags**, with its margin reported.
- **The merge rate is reported**, so a span covering several events is legible as one.
- **AIRWAY concludes about the airway kind and no other.**

**Steps:**

- [ ] **Step 1 — write the failing tests.**

Add to `src/tests/audio/workflows/triage/nodes/airway_test.py`, and delete the classes named in the
superseded table below:

```python
class TestItRunsNoClassifier:
    """Every window classification was written by PREPROCESS."""

    def test_the_module_calls_no_detector(self) -> None:
        """HeAR confirms a span; running it here would make it find one."""
        for name in ("detect_health_acoustic_events", "span_to_hear_buffer", "classify_audios"):
            assert not hasattr(airway_module, name)

    def test_it_writes_no_activity_naming_a_model_agent_it_ran(
        self, store: ProvStore, airway_config: TriageConfig, seeded_store: Callable[..., None], tmp_path: Path
    ) -> None:
        """The HeAR and YAMNet agents it associates with are the ones PREPROCESS's windows carry."""
        seeded_store(store, tmp_path, spans=[(1.0, 1.3, 30.0)], hear_windows=[((0.0, 2.0), ["Cough"])])
        airway(store, "plain", airway_config, run_dir=tmp_path)
        assert {a.step for a in store.activities("AIRWAY")} <= {"classify", "confirm", "lexical"}


class TestHearConfirmsRatherThanFinds:
    """The candidate is the span; HeAR says whether that extent carries cough or breath."""

    def test_a_span_whose_windows_carry_the_label_is_labelled(
        self, store: ProvStore, airway_config: TriageConfig, seeded_store: Callable[..., None], tmp_path: Path
    ) -> None:
        """Membership in the window's set is the evidence; no score is compared here."""
        seeded_store(store, tmp_path, spans=[(1.0, 1.3, 30.0)], hear_windows=[((0.0, 2.0), ["Cough"])])
        airway(store, "plain", airway_config, run_dir=tmp_path)
        assert _verdict_entity(store, "AIRWAY").attributes["by_label"] == {"Cough": 1}

    def test_a_hear_window_without_a_span_labels_nothing(
        self, store: ProvStore, airway_config: TriageConfig, seeded_store: Callable[..., None], tmp_path: Path
    ) -> None:
        """HeAR does not find a span; with no candidate there is nothing to confirm."""
        seeded_store(store, tmp_path, spans=[], hear_windows=[((0.0, 2.0), ["Cough"])])
        result = airway(store, "plain", airway_config, run_dir=tmp_path)
        assert result.verdict.outcome is Outcome.FAIL
        assert _verdict_entity(store, "AIRWAY").attributes["labelled_n"] == 0

    def test_a_transcribed_span_is_not_offered_to_hear(
        self, store: ProvStore, airway_config: TriageConfig, seeded_store: Callable[..., None], tmp_path: Path
    ) -> None:
        """A span overlapping consensus words is transcribed content, not an airway candidate."""
        seeded_store(
            store, tmp_path, spans=[(1.0, 1.3, 30.0)], hear_windows=[((0.0, 2.0), ["Cough"])],
            words=[("hello", (1.0, 1.2))],
        )
        airway(store, "plain", airway_config, run_dir=tmp_path)
        assert _verdict_entity(store, "AIRWAY").attributes["labelled_n"] == 0

    def test_a_span_carrying_only_events_stays_eligible(
        self, store: ProvStore, airway_config: TriageConfig, seeded_store: Callable[..., None], tmp_path: Path
    ) -> None:
        """Bracketed and onomatopoeic events are exactly what this branch is looking for."""
        seeded_store(
            store, tmp_path, spans=[(1.0, 1.3, 30.0)], hear_windows=[((0.0, 2.0), ["Cough"])],
            events=[("[COUGH]", (1.0, 1.2))],
        )
        airway(store, "plain", airway_config, run_dir=tmp_path)
        assert _verdict_entity(store, "AIRWAY").attributes["labelled_n"] == 1

    def test_a_span_whose_windows_carry_no_member_of_interest_is_unlabelled(
        self, store: ProvStore, airway_config: TriageConfig, seeded_store: Callable[..., None], tmp_path: Path
    ) -> None:
        """A span without a label is simply a span without a label assertion."""
        seeded_store(store, tmp_path, spans=[(1.0, 1.3, 30.0)], hear_windows=[((0.0, 2.0), ["Laugh"])])
        airway(store, "plain", airway_config, run_dir=tmp_path)
        assert not [
            e for e in live_entities(store, "assertion") if e.attributes.get("verb") == "label"
        ]


class TestContestRequiresColocation:
    """A label a window away is a different event, not a disagreement about this one (V21)."""

    def test_a_contest_label_in_the_same_hear_window_contests(
        self, store: ProvStore, airway_config: TriageConfig, seeded_store: Callable[..., None], tmp_path: Path
    ) -> None:
        """Both inside the HeAR window whose set carried the label."""
        seeded_store(
            store, tmp_path, spans=[(1.0, 1.3, 30.0)], hear_windows=[((0.0, 2.0), ["Cough"])],
            yamnet_windows=[((0.96, 1.92), ["Speech"])],
        )
        result = airway(store, "plain", airway_config, run_dir=tmp_path)
        assert _verdict_entity(store, "AIRWAY").attributes["contested_n"] == 1
        assert result.verdict.outcome is Outcome.FLAG

    def test_a_contest_label_outside_that_window_does_not(
        self, store: ProvStore, airway_config: TriageConfig, seeded_store: Callable[..., None], tmp_path: Path
    ) -> None:
        """The YAMNet window is outside the HeAR window, so it describes a different event."""
        seeded_store(
            store, tmp_path, spans=[(1.0, 1.3, 30.0)], hear_windows=[((0.0, 2.0), ["Cough"])],
            yamnet_windows=[((2.88, 3.84), ["Speech"])],
        )
        airway(store, "plain", airway_config, run_dir=tmp_path)
        assert _verdict_entity(store, "AIRWAY").attributes["contested_n"] == 0

    def test_a_label_outside_contest_labels_does_not_contest(
        self, store: ProvStore, airway_config: TriageConfig, seeded_store: Callable[..., None], tmp_path: Path
    ) -> None:
        """The eligible set is declared, not all 521."""
        seeded_store(
            store, tmp_path, spans=[(1.0, 1.3, 30.0)], hear_windows=[((0.0, 2.0), ["Cough"])],
            yamnet_windows=[((0.96, 1.92), ["Rain"])],
        )
        airway(store, "plain", airway_config, run_dir=tmp_path)
        assert _verdict_entity(store, "AIRWAY").attributes["contested_n"] == 0

    def test_a_contest_never_relabels(
        self, store: ProvStore, airway_config: TriageConfig, seeded_store: Callable[..., None], tmp_path: Path
    ) -> None:
        """Flag the span; the label stands and the assertion is not invalidated."""
        seeded_store(
            store, tmp_path, spans=[(1.0, 1.3, 30.0)], hear_windows=[((0.0, 2.0), ["Cough"])],
            yamnet_windows=[((0.96, 1.92), ["Speech"])],
        )
        airway(store, "plain", airway_config, run_dir=tmp_path)
        label = next(e for e in live_entities(store, "assertion") if e.attributes.get("verb") == "label")
        assert label.attributes["label"] == "Cough"

    def test_intersecting_label_sets_are_refused_at_load(
        self, store: ProvStore, seeded_store: Callable[..., None], tmp_path: Path
    ) -> None:
        """A label cannot both support and contest the same conclusion."""
        config = _override(tmp_path, "airway:\n  contest_labels: [Speech, Cough]\n  k_db: 18.0\n")
        seeded_store(store, tmp_path, spans=[(1.0, 1.3, 30.0)])
        before = len(store.entities())
        with pytest.raises(ValueError, match="disjoint"):
            airway(store, "plain", config, run_dir=tmp_path)
        assert len(store.entities()) == before


class TestTheGateIsAdjustableAndItsEdgeFlags:
    """K is per task, and a span that only just cleared it is a decision a human should see."""

    def test_airway_k_db_overrides_the_shared_gate(
        self, store: ProvStore, seeded_store: Callable[..., None], tmp_path: Path
    ) -> None:
        """An airway event is level-limited; one value fitted on coughs does not serve quiet breaths."""
        config = _override(tmp_path, "airway:\n  k_db: 12.0\n  k_margin_db: 2.0\n  contest_labels: [Speech]\n")
        seeded_store(store, tmp_path, spans=[(1.0, 1.3, 30.0)], span_k_db=12.0)
        airway(store, "plain", config, run_dir=tmp_path)
        assert _verdict_entity(store, "AIRWAY").attributes["k_db"] == 12.0

    def test_a_declared_task_overrides_it_again(
        self, store: ProvStore, seeded_store: Callable[..., None], tmp_path: Path
    ) -> None:
        """airway.k_db_by_task is the per-task gate."""
        config = _override(
            tmp_path,
            "airway:\n  k_db: 18.0\n  k_db_by_task: {breath: 8.0}\n  k_margin_db: 2.0\n  contest_labels: [Speech]\n",
        )
        seeded_store(store, tmp_path, spans=[(1.0, 1.3, 30.0)], span_k_db=8.0)
        hint = AudioHints(metadata={"task": "breath"})
        airway(store, "plain", config, hint, run_dir=tmp_path)
        assert _verdict_entity(store, "AIRWAY").attributes["k_db"] == 8.0

    def test_a_span_inside_the_margin_flags_with_its_margin(
        self, store: ProvStore, airway_config: TriageConfig, seeded_store: Callable[..., None], tmp_path: Path
    ) -> None:
        """Any span the gate would have kept out under a slightly different setting is visible."""
        seeded_store(store, tmp_path, spans=[(1.0, 1.3, 19.0)], hear_windows=[((0.0, 2.0), ["Cough"])])
        result = airway(store, "plain", airway_config, run_dir=tmp_path)
        assert _verdict_entity(store, "AIRWAY").attributes["near_gate_n"] == 1
        assert result.verdict.outcome is Outcome.FLAG
        label = next(e for e in live_entities(store, "assertion") if e.attributes.get("verb") == "label")
        assert label.attributes["margin_over_k_db"] == pytest.approx(1.0)

    def test_a_null_margin_leaves_the_band_inert(
        self, store: ProvStore, seeded_store: Callable[..., None], tmp_path: Path
    ) -> None:
        """Nobody derived how close is too close, so no span is near the gate."""
        config = _override(tmp_path, "airway:\n  k_db: 18.0\n  contest_labels: [Speech]\n")
        seeded_store(store, tmp_path, spans=[(1.0, 1.3, 19.0)], hear_windows=[((0.0, 2.0), ["Cough"])])
        airway(store, "plain", config, run_dir=tmp_path)
        assert _verdict_entity(store, "AIRWAY").attributes["near_gate_n"] == 0

    def test_the_merge_rate_is_reported(
        self, store: ProvStore, airway_config: TriageConfig, seeded_store: Callable[..., None], tmp_path: Path
    ) -> None:
        """A span covering several events must be legible as one."""
        seeded_store(
            store, tmp_path, spans=[(1.0, 1.9, 30.0)], span_merged=3,
            hear_windows=[((0.0, 2.0), ["Cough"])],
        )
        airway(store, "plain", airway_config, run_dir=tmp_path)
        assert _verdict_entity(store, "AIRWAY").attributes["merged_n"] == 3
```

- [ ] **Step 2 — run them; expect FAIL.**
  `uv run pytest src/tests/audio/workflows/triage/nodes/airway_test.py -x -q`

- [ ] **Step 3 — edit `airway.py`.**

Delete the `detect_health_acoustic_events`, `span_to_hear_buffer`, `HEAR_MODEL_ID`, `HEAR_REVISION`
imports and the whole `buffered`/`sliding` classification block. Delete `_best_of_interest` and
`_max_score`. Replace with reads of PREPROCESS's stored windows:

```python
def _windows_covering(
    store: ProvStore, classifier: str, extent: tuple[float, float]
) -> list[Entity]:
    """Every one of this classifier's stored windows overlapping the extent, oldest first.

    Args:
        store: The provenance store.
        classifier: ``"hear"`` or ``"yamnet"``.
        extent: The span's extent.

    Returns:
        The per-window measurement entities PREPROCESS wrote, filtered to those that overlap.
    """
    return [
        window
        for window in find_measurements(store, f"{classifier}_window")
        if window.extent is not None and window.extent[0] < extent[1] and window.extent[1] > extent[0]
    ]
```

The eligibility check, exactly the spec's rule:

```python
def _is_transcribed(store: ProvStore, extent: tuple[float, float]) -> bool:
    """Whether a consensus word overlaps this span, which makes it transcribed content.

    An ``event`` entity — a bracketed or onomatopoeic non-word — does not make a span transcribed:
    those are the events this branch is looking for.

    Args:
        store: The provenance store.
        extent: The span's extent.

    Returns:
        True when at least one live ``word`` entity overlaps.
    """
    return any(
        word.extent is not None and word.extent[0] < extent[1] and word.extent[1] > extent[0]
        for word in live_entities(store, "word")
    )
```

The label step reads membership, never a score:

```python
    for span in spans:
        extent = span.extent or (0.0, 0.0)
        if _is_transcribed(store, extent):
            continue
        hear_windows = _windows_covering(store, "hear", extent)
        members: dict[str, list[str]] = {}
        for hear_window in hear_windows:
            for label in hear_window.attributes.get("labels") or []:
                if label in labels_of_interest:
                    members.setdefault(str(label), []).append(hear_window.id)
        if not members:
            continue
        for label, window_ids in sorted(members.items()):
            attributes: dict[str, Any] = {
                "verb": "label",
                "label": label,
                "hear_window_ids": window_ids,
                "in_certified_silence": _inside_certified_silence(span, silence_windows),
                "merged_proposals": span.attributes.get("merged_proposals", 1),
            }
            if k_margin_db is not None:
                margin = float(span.attributes["peak_over_floor_db"]) - k_db
                attributes["margin_over_k_db"] = margin
                if margin <= float(k_margin_db):
                    near_gate_n += 1
                    flags.append(f"labelled span at {extent[0]:.2f}s sits {margin:.1f} dB over the gate")
```

**A span may now carry more than one label**, because a HeAR window's product is a set. That is a
change from v1's `max(scores)` and is what "the label is the `labels_of_interest` member confident in
the `hear_windows`" requires; `by_label` counts each.

The contest step implements V21:

```python
        contest_windows: list[tuple[str, str, str]] = []  # (yamnet_window_id, label, hear_window_id)
        for hear_window_id in window_ids:
            hear_extent = store.get_entity(hear_window_id).extent or (0.0, 0.0)
            for yamnet_window in _windows_covering(store, "yamnet", hear_extent):
                inside = (
                    yamnet_window.extent is not None
                    and yamnet_window.extent[0] >= hear_extent[0]
                    and yamnet_window.extent[1] <= hear_extent[1]
                )
                if not inside:
                    continue
                for yamnet_label in yamnet_window.attributes.get("labels") or []:
                    if str(yamnet_label) in confirmation_map.get(label, set()):
                        confirms.append((yamnet_window.id, str(yamnet_label), hear_window_id))
                    elif str(yamnet_label) in contest_labels:
                        contest_windows.append((yamnet_window.id, str(yamnet_label), hear_window_id))
```

with the disjointness check at entry, before any store write:

```python
    contest_labels = {str(label) for label in (config.get("airway.contest_labels") or [])}
    airway_evidence = {str(label) for label in config.require("taxonomy.audioset_airway_labels")}
    overlap = contest_labels & airway_evidence
    if overlap:
        raise ValueError(
            f"airway.contest_labels and taxonomy.audioset_airway_labels must be disjoint; "
            f"{sorted(overlap)} appear in both, so the same label would be airway evidence and a "
            "contest of airway evidence"
        )
```

and the gate resolution:

```python
    task = str(hint.metadata.get("task")) if hint is not None and hint.metadata.get("task") else None
    by_task = config.get("airway.k_db_by_task") or {}
    k_db = float(
        by_task.get(task) if task is not None and by_task.get(task) is not None
        else config.get("airway.k_db", config.require("spans.k_db.airway"))
    )
    k_margin_db = config.get("airway.k_margin_db")
```

The verdict detail becomes exactly `branch-airway.md`'s product:
`{labelled_n, by_label, contested_n, near_gate_n, merged_n, k_db, flags}`.

**PREPROCESS must record the merge count.** `merged_proposals` is added to the `span` entity attributes
in sibling T1's `_spans` block (`propose_spans` already merges; the count is `len(absorbed)`), and
`merged_n` here is the sum over labelled spans. **This is a one-line addition to sibling T1 that T6
depends on; note it in T6's commit body so a reviewer can see the cross-task edit.**

- [ ] **Step 4 — run them; expect PASS.**
  `uv run pytest src/tests/audio/workflows/triage/nodes/airway_test.py -x -q`

- [ ] **Step 5 — lint, type-check, commit.**
  `uv run ruff format src/senselab/audio/workflows/triage src/tests/audio/workflows/triage`
  `uv run ruff check src/senselab/audio/workflows/triage src/tests/audio/workflows/triage`
  `uv run mypy src/senselab/audio/workflows/triage`
  `git commit -m "feat(triage/airway): HeAR confirms a span, and a contest must be co-located"`

**Interfaces:**

*Consumed:* sibling T1's `hear_window` and `yamnet_window` per-window measurements (their `labels`
sets and extents), `span` entities with `peak_over_floor_db`, `k_db` and `merged_proposals`, `word` and
`event` entities, `silence`, `spans_no_contrast`; `common.find_measurements`, `live_entities`.

*Produced (the T6→T8 contract):* `airway(...) -> AirwayResult` with `verdict.kind == "airway"`;
`label`/`confirm`/`contest`/`abstain` assertions over PREPROCESS's spans, each naming the HeAR window
behind it and, for a contest, the window the co-location was found in; the
`airway_labelled_interval`; the `AIRWAY` verdict with
`{labelled_n, by_label, contested_n, near_gate_n, merged_n, k_db, flags}`.

**Superseded tests, deleted with the ruling that justifies each:**

| deleted | ruling |
| --- | --- |
| every `span_to_hear_buffer` / `hear.placement` test | branch-airway.md: "This branch runs no classifier. Every window classification it reads was written by PREPROCESS" |
| the `hear.label_floor` tests | preprocess.md: the threshold is `windows.hear.*` and is applied in PREPROCESS; membership is what this branch reads |
| the "YAMNet coverage over the span" confirmation tests | branch-airway.md: "A contest requires co-location… both fall inside the same window" — coverage over the span is not co-location |
| any test asserting one label per span | branch-airway.md: "The label is the `labels_of_interest` member confident in the `hear_windows`" — a window's product is a set |

---

### Task 7: REDACT v2 — a step of SPEECH that runs no recognizer

**Scope:** `src/senselab/audio/tasks/redaction/api.py` (a `fill` parameter on `apply_redactions`);
`src/senselab/audio/workflows/triage/nodes/redact.py` (rewritten verification and fill);
`src/senselab/audio/workflows/triage/data/config/default.yaml` (`redaction.fill`,
`redaction.bleep_hz`); `src/tests/audio/tasks/redaction/api_test.py` (extended);
`src/tests/audio/workflows/triage/nodes/redact_test.py` (rewritten).

**Design points this task must not get wrong (from `redact.md`):**

- **REDACT runs no recognizer.** Re-transcription draws a second sample from the recognizers, which is
  a different measurement of a different signal, not a check on this one. The `transcribe_audios`
  import is **deleted**.
- **Verification is a re-scan of the redacted consensus text.** The planned redactions are applied to
  the consensus transcript, the same detectors are re-run over the redacted text.
- **It runs only when SPEECH's PII scan found something.** Sibling T3's `run._speech_found_pii` gates
  it; this node additionally refuses an incoherent store (findings but no scan measurement).
- **It redacts every finding, regardless of speaker.** SPEECH flags target-speaker PII; REDACT redacts
  all of it, because a non-target speaker naming the participant is exactly as unsafe.
- **The fill is configurable and the key ships with no default.**
- **`audio_check` is the constant `"bounded"` on every path.** Verification establishes that the
  redacted *text* no longer carries the finding, and nothing about the audio.
- **A surviving finding is remediable exactly once**; one re-planning pass, then `unremediable`.
- **Only a pass produces a released pair; a flag withholds exactly like a fail.**
- **`+` is reserved in a category label**, and no released artifact carries a store element id.

**Steps:**

- [ ] **Step 1 — add the two config keys.**

Add to `derivation:`:

```
  redaction.fill -- redact.md leaves this DEFERRED: which of silence, noise or bleep is least damaging
  to the measurements taken downstream of a released artifact has not been measured, so the key ships
  null and a run must declare the fill it used. silence and bleep are implemented; noise raises rather
  than shipping an unmeasured spectral shape, because "speech-shaped" names a shaping nobody here has
  fitted. redaction.bleep_hz 1000.0 is the conventional broadcast censor tone -- a presentation
  choice, declared rather than defaulted silently, and not fitted.
```

```yaml
redaction:
  padding_ms: null
  fill: null
  bleep_hz: 1000.0
```

- [ ] **Step 2 — write the failing `apply_redactions` tests.**

Add to `src/tests/audio/tasks/redaction/api_test.py`:

```python
class TestTheFill:
    """What is written into a redacted extent, and what is refused."""

    def test_silence_writes_zeros(self) -> None:
        """The historical behaviour, now named rather than implied."""
        audio = _tone(1.0)
        out = apply_redactions(audio, [RedactionExtent(0.2, 0.4, "PERSON")], fill="silence")
        assert float(out.waveform[:, 3200:6400].abs().max()) == 0.0

    def test_bleep_writes_a_tone_at_the_extents_own_level(self) -> None:
        """The extent is masked, not removed, and the level is the extent's own."""
        audio = _tone(1.0, amplitude=0.5)
        out = apply_redactions(audio, [RedactionExtent(0.2, 0.4, "PERSON")], fill="bleep", bleep_hz=1000.0)
        inside = out.waveform[:, 3200:6400]
        assert float(inside.abs().max()) == pytest.approx(0.5, rel=0.05)
        assert float(inside.abs().min()) < 0.05

    def test_noise_is_refused_with_the_measurement_it_is_owed(self) -> None:
        """Shipping an unmeasured spectral shape would be a value nobody fitted (V22)."""
        with pytest.raises(NotImplementedError, match="least damaging"):
            apply_redactions(_tone(1.0), [RedactionExtent(0.2, 0.4, "PERSON")], fill="noise")

    def test_an_unknown_fill_is_refused(self) -> None:
        """A typo must not silently fall back to silence."""
        with pytest.raises(ValueError, match="fill"):
            apply_redactions(_tone(1.0), [RedactionExtent(0.2, 0.4, "PERSON")], fill="beep")

    def test_the_duration_is_preserved_under_every_implemented_fill(self) -> None:
        """A redaction masks; it does not shorten."""
        audio = _tone(1.0)
        for fill in ("silence", "bleep"):
            out = apply_redactions(audio, [RedactionExtent(0.2, 0.4, "PERSON")], fill=fill, bleep_hz=1000.0)
            assert out.waveform.shape == audio.waveform.shape
```

- [ ] **Step 3 — run it; expect FAIL** (`TypeError: apply_redactions() got an unexpected keyword
  argument 'fill'`).
  `uv run pytest src/tests/audio/tasks/redaction/api_test.py -x -q`

- [ ] **Step 4 — add `fill` to `apply_redactions`.**

```python
def apply_redactions(
    audio: Audio,
    extents: Sequence[RedactionExtent],
    *,
    fill: str = "silence",
    bleep_hz: float | None = None,
) -> Audio:
    """Mask every extent with the named fill, preserving duration.

    Args:
        audio: The recording.
        extents: Regions to mask. Pass the output of :func:`plan_redactions`, not raw findings.
        fill: ``"silence"`` writes zeros; ``"bleep"`` writes a sine at ``bleep_hz`` scaled to the
            extent's own peak. Read it from ``redaction.fill``.
        bleep_hz: The bleep's frequency. Required when ``fill`` is ``"bleep"``. Read it from
            ``redaction.bleep_hz``.

    Returns:
        A new ``Audio``. The input is not modified. Each extent's start rounds down to a sample index
        and its end rounds up, both clamped to the recording; an extent that selects no samples is a
        no-op.

    Raises:
        NotImplementedError: If ``fill`` is ``"noise"``. Which fill is least damaging to the
            measurements taken downstream of a released artifact has not been measured, and
            "speech-shaped" names a shaping nobody has fitted.
        ValueError: If ``fill`` names no implemented fill, or if ``"bleep"`` is asked for without
            ``bleep_hz``.
    """
    if fill == "noise":
        raise NotImplementedError(
            "the 'noise' fill is deferred: which fill is least damaging to downstream measurement "
            "has not been measured, and a speech-shaped spectrum nobody fitted is not a default"
        )
    if fill not in ("silence", "bleep"):
        raise ValueError(f"fill must be 'silence' or 'bleep'; got {fill!r}")
    if fill == "bleep" and bleep_hz is None:
        raise ValueError("fill='bleep' needs bleep_hz; read it from redaction.bleep_hz")
    x = np.array(np.asarray(audio.waveform, dtype=np.float32), copy=True)
    if x.ndim == 1:
        x = x[None, :]
    sr = audio.sampling_rate
    n = x.shape[-1]
    for extent in extents:
        lo = max(0, int(extent.start * sr))
        hi = max(lo, min(n, math.ceil(extent.end * sr)))
        if hi <= lo:
            continue
        if fill == "silence":
            x[:, lo:hi] = 0.0
        else:
            level = float(np.abs(x[:, lo:hi]).max())
            t = np.arange(hi - lo, dtype=np.float32) / sr
            x[:, lo:hi] = (level * np.sin(2.0 * np.pi * float(bleep_hz) * t)).astype(np.float32)
    return Audio(waveform=x, sampling_rate=sr)
```

- [ ] **Step 5 — run it; expect PASS.**
  `uv run pytest src/tests/audio/tasks/redaction/api_test.py -x -q`

- [ ] **Step 6 — write the failing REDACT tests.**

Replace `src/tests/audio/workflows/triage/nodes/redact_test.py`'s verification classes:

```python
class TestVerificationDoesNotReTranscribe:
    """A re-decode is a second sample of a different signal, not a check on this one."""

    def test_the_module_cannot_transcribe(self) -> None:
        """The recognizer import is deleted, not left unreachable."""
        assert not hasattr(redact_module, "transcribe_audios")

    def test_verification_re_scans_the_redacted_text(
        self, store: ProvStore, redact_config: TriageConfig, seeded_store: Callable[..., None],
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Exactly one text is re-scanned, and it is the transcript the plan produced."""
        seeded_store(store, tmp_path, words=["my", "name", "is", "alice"], findings=[("PERSON", (3.0, 4.0))])
        scanned = _stub_pii(monkeypatch, findings=[])
        result = redact(store, "recording", redact_config, run_dir=tmp_path, artifacts_dir=tmp_path / "rel")
        assert result.verdict.outcome is Outcome.PASS
        assert scanned == ["my name is [PERSON]"]

    def test_the_verify_activity_names_no_model_agent(
        self, store: ProvStore, redact_config: TriageConfig, seeded_store: Callable[..., None],
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Nothing here runs at a commit, because nothing here runs a model."""
        seeded_store(store, tmp_path, words=["hello", "alice"], findings=[("PERSON", (1.0, 2.0))])
        _stub_pii(monkeypatch, findings=[])
        redact(store, "recording", redact_config, run_dir=tmp_path, artifacts_dir=tmp_path / "rel")
        verify = next(a for a in store.activities("REDACT") if a.step == "verify")
        assert not [
            agent for agent in store.associated_with(verify.id) if store.get_agent(agent).agent_type == "model"
        ]

    def test_the_audio_claim_is_bounded_on_every_path(
        self, store: ProvStore, redact_config: TriageConfig, seeded_store: Callable[..., None],
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A text re-scan cannot answer whether intelligible speech survives outside the extent."""
        seeded_store(store, tmp_path, words=["hello", "alice"], findings=[("PERSON", (1.0, 2.0))])
        for survivors in ([], [("PERSON", "alice")]):
            other = ProvStore(run_id="bounded")
            seeded_store(other, tmp_path, words=["hello", "alice"], findings=[("PERSON", (1.0, 2.0))])
            _stub_pii(monkeypatch, findings=survivors)
            redact(other, "recording", redact_config, run_dir=tmp_path, artifacts_dir=tmp_path / "rel")
            assert _verdict_entity(other, "REDACT").attributes["audio_check"] == "bounded"


class TestRemediationHappensExactlyOnce:
    """A finding the planner placed and the verifier still sees gets one re-planning pass."""

    def test_a_survivor_triggers_one_replan(
        self, store: ProvStore, redact_config: TriageConfig, seeded_store: Callable[..., None],
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The verifier's extent is fed back once, and a clean second scan passes."""
        seeded_store(store, tmp_path, words=["hello", "alice"], findings=[("PERSON", (1.0, 2.0))])
        _stub_pii_sequence(monkeypatch, [[("PERSON", "alice")], []])
        result = redact(store, "recording", redact_config, run_dir=tmp_path, artifacts_dir=tmp_path / "rel")
        assert result.verdict.outcome is Outcome.PASS
        assert _verdict_entity(store, "REDACT").attributes["replanned_n"] == 1

    def test_a_survivor_of_the_replan_is_unremediable(
        self, store: ProvStore, redact_config: TriageConfig, seeded_store: Callable[..., None],
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """An operator must be able to tell this from an ordinary withhold."""
        seeded_store(store, tmp_path, words=["hello", "alice"], findings=[("PERSON", (1.0, 2.0))])
        _stub_pii_sequence(monkeypatch, [[("PERSON", "alice")], [("PERSON", "alice")]])
        result = redact(store, "recording", redact_config, run_dir=tmp_path, artifacts_dir=tmp_path / "rel")
        assert result.verdict.outcome is Outcome.FAIL
        detail = _verdict_entity(store, "REDACT").attributes
        assert detail["unremediable"] == ["PERSON"]
        assert detail["survived"] == ["PERSON"]
        assert result.artifacts == {}


class TestTheFillIsDeclared:
    """A run declares the fill it used, and the verdict records it."""

    def test_a_null_fill_refuses_before_any_store_write(
        self, store: ProvStore, config: TriageConfig, seeded_store: Callable[..., None], tmp_path: Path
    ) -> None:
        """The key ships with no default; two artifacts under different fills are not comparable."""
        seeded_store(store, tmp_path, words=["hello", "alice"], findings=[("PERSON", (1.0, 2.0))])
        before = len(store.entities())
        with pytest.raises(ValueError, match="redaction.fill"):
            redact(store, "recording", config, run_dir=tmp_path, artifacts_dir=tmp_path / "rel")
        assert len(store.entities()) == before

    def test_the_verdict_records_the_fill(
        self, store: ProvStore, redact_config: TriageConfig, seeded_store: Callable[..., None],
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """So two artifacts made under different fills are never compared as one."""
        seeded_store(store, tmp_path, words=["hello", "alice"], findings=[("PERSON", (1.0, 2.0))])
        _stub_pii(monkeypatch, findings=[])
        redact(store, "recording", redact_config, run_dir=tmp_path, artifacts_dir=tmp_path / "rel")
        assert _verdict_entity(store, "REDACT").attributes["fill"] == "silence"

    def test_bleep_is_reachable_by_config(
        self, store: ProvStore, seeded_store: Callable[..., None], tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Both implemented fills are selectable; neither is a default."""
        config = _override(tmp_path, "redaction:\n  padding_ms: 100\n  fill: bleep\n")
        seeded_store(store, tmp_path, words=["hello", "alice"], findings=[("PERSON", (1.0, 2.0))])
        _stub_pii(monkeypatch, findings=[])
        redact(store, "recording", config, run_dir=tmp_path, artifacts_dir=tmp_path / "rel")
        assert _verdict_entity(store, "REDACT").attributes["fill"] == "bleep"


class TestItRedactsEverySpeaker:
    """SPEECH flags target-speaker PII; redaction is about whether an artifact is releasable."""

    def test_a_non_target_finding_is_redacted(
        self, store: ProvStore, redact_config: TriageConfig, seeded_store: Callable[..., None],
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A non-target speaker naming the participant is exactly as unsafe."""
        seeded_store(
            store, tmp_path, words=["hello", "alice"],
            findings=[("PERSON", (1.0, 2.0), "SPEAKER_01")], target_speaker="SPEAKER_00",
        )
        _stub_pii(monkeypatch, findings=[])
        redact(store, "recording", redact_config, run_dir=tmp_path, artifacts_dir=tmp_path / "rel")
        assert _verdict_entity(store, "REDACT").attributes["redactions_n"] == 1
```

- [ ] **Step 7 — run them; expect FAIL.**
  `uv run pytest src/tests/audio/workflows/triage/nodes/redact_test.py -x -q`

- [ ] **Step 8 — edit `redact.py`.**

Delete the `transcribe_audios` import, `_verification_model`, `_declared_recognizers`, `_asr_models`
and the whole `verify_systems`/`expected_systems`/`unverifiable` apparatus. `_verify` becomes:

```python
def _verify(transcript_text: str, required: list[str]) -> _Verification:
    """Re-scan the redacted consensus text with the same detectors.

    No recognizer runs. Re-transcribing would draw a second sample from the recognizers, which is a
    different measurement of a different signal rather than a check on this one, and the claim about
    the audio is bounded either way.

    Args:
        transcript_text: The redacted consensus transcript.
        required: The detector set ``pii.required_detectors`` names.

    Returns:
        What the re-scan established. A finding that survives fails; a re-scan that skipped a
        required detector did not run, which is not a clean result.
    """
    scan = scan_for_pii(transcript_text)
    scan = scan[0] if isinstance(scan, list) else scan
    missing = sorted(set(required) - set(scan.detectors_used) - set(scan.failures))
    if scan.failures or not scan.detectors_used or missing:
        return _Verification(verified=False, survived=[], scan_ran=False, missing=missing)
    survived = sorted({span.category for span in scan.spans})
    return _Verification(verified=not survived, survived=survived, scan_ran=True, missing=[])
```

`_Verification` gains a `missing: list[str]` field.

`_consensus_words` now reads **PREPROCESS's** words, since sibling T1 made PREPROCESS the only author:

```python
def _consensus_words(store: ProvStore) -> list[Entity]:
    """The consensus words, in time order; a word the store places nowhere sorts first.

    Args:
        store: The provenance store.

    Returns:
        The live ``word`` entities PREPROCESS authored, oldest-extent first.
    """
    return sorted(live_entities(store, "word"), key=lambda w: w.extent or (-1.0, -1.0))
```

The main body's verification section becomes one planning pass, one verification, and at most one
re-plan:

```python
    fill = str(config.require("redaction.fill"))
    bleep_hz = config.get("redaction.bleep_hz")
    ...
    planned = plan_redactions(extents, padding_ms=padding_ms)
    transcript_text, unplaced_n = _transcript(words, planned)
    checked = _verify(transcript_text, required_detectors) if not scan_incomplete else _Verification(
        verified=False, survived=[], scan_ran=False, missing=scan_missing
    )
    replanned_n = 0
    unremediable: list[str] = []
    if checked.scan_ran and checked.survived:
        # One re-planning pass: the verifier's own extents are fed back, then the answer stands.
        replanned_n = 1
        extents = extents + [
            RedactionExtent(start=word.extent[0], end=word.extent[1], category=category)
            for category in checked.survived
            for word in words
            if word.extent is not None and _matches_surviving(word, category, planned)
        ]
        planned = plan_redactions(extents, padding_ms=padding_ms)
        transcript_text, unplaced_n = _transcript(words, planned)
        checked = _verify(transcript_text, required_detectors)
        unremediable = list(checked.survived)
    redacted = apply_redactions(recording, planned, fill=fill, bleep_hz=bleep_hz)
```

`_matches_surviving(word, category, planned)` is a small helper returning True for a live `word`
entity that carries a `pii` marking of that category and is **not** already covered by a planned
extent — the re-plan widens what the first pass missed rather than re-planning the same extents.

The outcome ladder loses the `unverifiable` row (there are no recognizers to be unable to re-run) and
gains the incomplete-re-scan row:

```python
    if scan_incomplete:
        outcome, why = Outcome.FAIL, "..."          # unchanged
    elif not checked.scan_ran:
        outcome = Outcome.FLAG
        why = (
            "the re-scan over the redacted text did not cover every required pii detector "
            f"({', '.join(checked.missing) or 'a detector failed'}); an unverified artifact is withheld"
        )
    elif checked.survived:
        outcome = Outcome.FAIL
        why = "verification found pii on the redacted transcript: " + ", ".join(checked.survived)
    else:
        outcome = Outcome.PASS
        why = "every finding redacted; the redacted transcript re-scans clean"
        artifacts = _write_artifacts(redacted, transcript_text, artifacts_dir)
```

Note the change: an incomplete re-scan is a **`flag`**, per `redact.md` ("a re-scan that skipped a
required detector is a `flag`"), where v1 made it a `fail`. On both paths `artifacts` is empty and
`artifacts_withheld` is `true`, so the release axis is unchanged; what changes is which reason
`verdict.md` reads.

The verdict detail becomes exactly `redact.md`'s product:

```python
        detail={
            "redactions_n": len(planned),
            "by_category": dict(Counter(extent.category for extent in planned)),
            "padding_ms": padding_ms,
            "fill": fill,
            "verified": checked.verified,
            "survived": checked.survived,
            "unremediable": unremediable,
            "replanned_n": replanned_n,
            "scan_failed": scan_failed,
            "scan_missing": scan_missing,
            "required_detectors": required_detectors,
            "unplaced_words_n": unplaced_n,
            "audio_check": "bounded",
            "artifacts_withheld": not artifacts,
        },
```

**`not_assessed` for a never-reached REDACT** is not this node's job: sibling T3's runner never calls
it, so no `REDACT` verdict exists, and T8's release fold maps an absent verdict to `not_assessed`.
That is the whole mechanism, and this task adds nothing for it beyond not writing a verdict on a path
it was not called on.

- [ ] **Step 9 — run them; expect PASS.**
  `uv run pytest src/tests/audio/workflows/triage/nodes/redact_test.py -x -q`

- [ ] **Step 10 — lint, type-check, commit.**
  `uv run ruff format src/senselab/audio src/tests/audio`
  `uv run ruff check src/senselab/audio/workflows/triage src/senselab/audio/tasks/redaction src/tests/audio/workflows/triage src/tests/audio/tasks/redaction`
  `uv run mypy src/senselab/audio/workflows/triage src/senselab/audio/tasks/redaction`
  `git commit -m "feat(triage/redact): verify by re-scanning the redacted text, and never re-transcribe"`

**Interfaces:**

*Consumed:* T4's `pii` entities and `pii_scan` measurement; sibling T1's `word` entities and
`consensus_transcript`; `plan_redactions`, `apply_redactions(audio, extents, *, fill, bleep_hz)`,
`scan_for_pii`; `common.live_entities`, `find_measurement`, `resolve_stream`.

*Produced (the T7→T8 contract):* `redact(...) -> RedactResult` with `artifacts` empty on anything but
a pass; the `REDACT` verdict with the detail above; `span` entities of `name="redaction"`. **T8 reads
only the verdict's `outcome`, `survived`, `unremediable` and `artifacts_withheld`.**

**Superseded tests, deleted with the ruling that justifies each:**

| deleted | ruling |
| --- | --- |
| every test that verification re-runs the recognizers at their recorded commits | redact.md: "REDACT runs no recognizer" |
| `verify_systems` / `expected_systems` / `expected_source` / `unverifiable` assertions | redact.md replaces the recognizer-coverage rule with a text re-scan judged by `pii.required_detectors` |
| the test that an incomplete re-scan fails | redact.md: "a re-scan that skipped a required detector is a `flag`" |
| `_consensus_words` reading SPEECH-authored words | preprocess.md: "`word` entities are written here, and only here" |

---

### Task 8: VERDICT v2 — pass/flag/discard, branch authority scoped to the branch's own kind

**Scope:** `src/senselab/audio/workflows/triage/vocabulary.py` (the `Triage` enum, `FileVerdict`,
`fold_file_verdict` rewritten); `src/senselab/audio/workflows/triage/nodes/verdict.py` (rewritten);
`src/senselab/audio/workflows/triage/__init__.py`; `src/tests/audio/workflows/triage/vocabulary_test.py`
(rewritten); `src/tests/audio/workflows/triage/nodes/verdict_test.py` (rewritten).

**Design points this task must not get wrong (from `verdict.md`):**

- **Two axes.** `triage` ∈ `{pass, flag, discard}`, `release` ∈ `{releasable, withheld, not_assessed}`.
  Collapsing them makes a recording with clean measurements and surviving PII look like a measurement
  problem.
- **`discard` has exactly two grounds:** ADMIT failed (unmeasurable), or every kind absent with nothing
  found and no hint claiming otherwise (acoustically empty). **A branch `fail` is not a file
  `discard`.**
- **A hint that claims otherwise turns the second ground into a `flag`, never a `discard`.**
- **Branch authority is scoped to the branch's own kind.** SPEECH refutes neither `airway` nor `voice`.
  A branch's conclusion about its own kind stands in the resolved `kinds` map whatever the
  classification said, and whether the branch passed, flagged or failed.
- **TAXONOMY is reported beside the branches, never over them.** Both `kinds` and `screened` are always
  present, and `agreement` records per kind whether they agree, mismatch or were resolved.
- **A mismatch flags; it never overrides.**
- **A branch that never ran is not a branch that failed** — the `branch_decision` elements are what
  distinguish the two. `will_run: true` with no verdict is a flag naming which of three reasons.
- **Hints are read here**, for branch mismatch only. A hint never resolves a kind and never turns a
  flag into a pass.
- **A REDACT non-pass does not flip triage, and is never invisible.**
- **`ran` is merged, the runner's over the store's.**

**Steps:**

- [ ] **Step 1 — write the failing vocabulary tests.**

Replace `src/tests/audio/workflows/triage/vocabulary_test.py`'s fold tests:

```python
class TestTheTriageVocabulary:
    """pass, flag, discard — three values, and fail is not one of them."""

    def test_the_members_are_exactly_three(self) -> None:
        """verdict.md's triage axis; a branch's `fail` has no counterpart here."""
        assert {member.value for member in Triage} == {"pass", "flag", "discard"}

    def test_a_node_outcome_is_not_a_triage(self) -> None:
        """Outcome stays the node-level vocabulary; the file axis is its own type."""
        assert not isinstance(Outcome.FAIL, Triage)


class TestDiscardIsNarrow:
    """Exactly two grounds, and they carry different reasons."""

    def test_admit_failure_discards_as_unmeasurable(self) -> None:
        """Nothing ran and nothing is claimed about the recording."""
        folded = fold_file_verdict(
            [NodeVerdict("ADMIT", Outcome.FAIL, None, "decode failure")],
            screened={}, branch_decisions={}, ran={}, hint_claims={},
        )
        assert folded.triage is Triage.DISCARD
        assert folded.discard_ground == "unmeasurable"

    def test_all_absent_with_nothing_found_discards_as_acoustically_empty(self) -> None:
        """Measured, and there is nothing of interest in it."""
        folded = fold_file_verdict(
            [NodeVerdict("ADMIT", Outcome.PASS, None, "ok")],
            screened={"speech": "absent", "airway": "absent", "voice": "absent"},
            branch_decisions=_all_skipped(),
            ran={}, hint_claims={},
        )
        assert folded.triage is Triage.DISCARD
        assert folded.discard_ground == "acoustically_empty"

    def test_a_branch_fail_is_not_a_discard(self) -> None:
        """A cough recording has no speech; SPEECH failing is the expected outcome."""
        folded = fold_file_verdict(
            [
                NodeVerdict("ADMIT", Outcome.PASS, None, "ok"),
                NodeVerdict("AIRWAY", Outcome.PASS, "airway", "labelled"),
                NodeVerdict("SPEECH", Outcome.FAIL, "speech", "no consensus word"),
            ],
            screened={"speech": "absent", "airway": "present", "voice": "absent"},
            branch_decisions=_decisions(airway=True, speech=True, voice=False),
            ran={}, hint_claims={},
        )
        assert folded.triage is not Triage.DISCARD

    def test_a_hint_turns_the_empty_ground_into_a_flag(self) -> None:
        """Discarding would delete the evidence that the graph was wrong."""
        folded = fold_file_verdict(
            [NodeVerdict("ADMIT", Outcome.PASS, None, "ok")],
            screened={"speech": "absent", "airway": "absent", "voice": "absent"},
            branch_decisions=_all_skipped(),
            ran={}, hint_claims={"speech": True},
        )
        assert folded.triage is Triage.FLAG


class TestBranchAuthorityIsScoped:
    """A branch is the authority on its own kind and on nothing else."""

    def test_speech_resolves_speech_and_touches_nothing_else(self) -> None:
        """It refutes neither airway nor voice."""
        folded = fold_file_verdict(
            [
                NodeVerdict("ADMIT", Outcome.PASS, None, "ok"),
                NodeVerdict("SPEECH", Outcome.PASS, "speech", "words in the store"),
            ],
            screened={"speech": "uncertain", "airway": "present", "voice": "present"},
            branch_decisions=_decisions(airway=False, speech=True, voice=False),
            ran={}, hint_claims={},
        )
        assert folded.kinds["speech"] == "present"
        assert folded.kinds["airway"] == "present"
        assert folded.kinds["voice"] == "present"

    def test_a_flagged_branch_still_resolves_its_kind(self) -> None:
        """The flag travels beside the resolution and is not a reason to withhold it."""
        folded = fold_file_verdict(
            [
                NodeVerdict("ADMIT", Outcome.PASS, None, "ok"),
                NodeVerdict("VOICE", Outcome.FLAG, "voice", "a declared range is not met"),
            ],
            screened={"speech": "absent", "airway": "absent", "voice": "uncertain"},
            branch_decisions=_decisions(airway=False, speech=False, voice=True),
            ran={}, hint_claims={},
        )
        assert folded.kinds["voice"] == "present"
        assert folded.triage is Triage.FLAG

    def test_a_failed_branch_resolves_its_kind_absent(self) -> None:
        """A branch with no subject is authority for that too."""
        folded = fold_file_verdict(
            [
                NodeVerdict("ADMIT", Outcome.PASS, None, "ok"),
                NodeVerdict("SPEECH", Outcome.FAIL, "speech", "no consensus word"),
                NodeVerdict("AIRWAY", Outcome.PASS, "airway", "labelled"),
            ],
            screened={"speech": "present", "airway": "present", "voice": "absent"},
            branch_decisions=_decisions(airway=True, speech=True, voice=False),
            ran={}, hint_claims={},
        )
        assert folded.kinds["speech"] == "absent"


class TestTaxonomyIsReportedBeside:
    """Both maps are always present, and agreement is checkable by a reader."""

    def test_screened_and_kinds_are_both_present(self) -> None:
        """Keeping both is what makes agreement checkable rather than asserted."""
        folded = fold_file_verdict(
            [
                NodeVerdict("ADMIT", Outcome.PASS, None, "ok"),
                NodeVerdict("SPEECH", Outcome.PASS, "speech", "words"),
            ],
            screened={"speech": "absent", "airway": "absent", "voice": "absent"},
            branch_decisions=_decisions(airway=False, speech=True, voice=False),
            ran={}, hint_claims={},
        )
        assert folded.screened["speech"] == "absent"
        assert folded.kinds["speech"] == "present"

    def test_absent_classified_but_found_is_a_mismatch_and_flags(self) -> None:
        """It flags; it never overrides, and both stay in the product."""
        folded = fold_file_verdict(
            [
                NodeVerdict("ADMIT", Outcome.PASS, None, "ok"),
                NodeVerdict("SPEECH", Outcome.PASS, "speech", "words"),
            ],
            screened={"speech": "absent", "airway": "absent", "voice": "absent"},
            branch_decisions=_decisions(airway=False, speech=True, voice=False),
            ran={}, hint_claims={},
        )
        assert folded.agreement["speech"] == "mismatch"
        assert folded.triage is Triage.FLAG

    def test_present_classified_but_not_found_is_a_mismatch(self) -> None:
        """The other direction of the same row."""
        folded = fold_file_verdict(
            [
                NodeVerdict("ADMIT", Outcome.PASS, None, "ok"),
                NodeVerdict("SPEECH", Outcome.FAIL, "speech", "no consensus word"),
            ],
            screened={"speech": "present", "airway": "absent", "voice": "absent"},
            branch_decisions=_decisions(airway=False, speech=True, voice=False),
            ran={}, hint_claims={},
        )
        assert folded.agreement["speech"] == "mismatch"

    def test_uncertain_classified_is_resolved_not_mismatched(self) -> None:
        """A branch settling an unsettled kind is the design working, not a disagreement."""
        folded = fold_file_verdict(
            [
                NodeVerdict("ADMIT", Outcome.PASS, None, "ok"),
                NodeVerdict("SPEECH", Outcome.PASS, "speech", "words"),
            ],
            screened={"speech": "uncertain", "airway": "absent", "voice": "absent"},
            branch_decisions=_decisions(airway=False, speech=True, voice=False),
            ran={}, hint_claims={},
        )
        assert folded.agreement["speech"] == "resolved"
        assert folded.triage is Triage.PASS


class TestABranchThatNeverRanIsNotOneThatFailed:
    """The branch_decision elements are what distinguish the two."""

    def test_declined_and_unforced_is_expected(self) -> None:
        """The graph declined to look, and said why."""
        folded = fold_file_verdict(
            [
                NodeVerdict("ADMIT", Outcome.PASS, None, "ok"),
                NodeVerdict("AIRWAY", Outcome.PASS, "airway", "labelled"),
            ],
            screened={"speech": "absent", "airway": "present", "voice": "absent"},
            branch_decisions=_decisions(airway=True, speech=False, voice=False),
            ran={}, hint_claims={},
        )
        assert folded.agreement["speech"] == "not_run"
        assert folded.triage is Triage.PASS

    def test_asked_but_silent_flags(self) -> None:
        """will_run true with no verdict is a branch that left no answer."""
        folded = fold_file_verdict(
            [NodeVerdict("ADMIT", Outcome.PASS, None, "ok")],
            screened={"speech": "present", "airway": "absent", "voice": "absent"},
            branch_decisions=_decisions(airway=False, speech=True, voice=False),
            ran={"SPEECH": RunState.ERRORED}, hint_claims={},
        )
        assert folded.triage is Triage.FLAG
        assert any("errored without a verdict" in reason.why for reason in folded.reasons)

    def test_the_three_silent_reasons_are_distinguished(self) -> None:
        """errored, completed-without-a-verdict and never-ran are different findings."""
        for state, phrase in (
            (RunState.ERRORED, "errored without a verdict"),
            (RunState.COMPLETED, "completed without a verdict"),
            (RunState.SKIPPED, "never ran"),
        ):
            folded = fold_file_verdict(
                [NodeVerdict("ADMIT", Outcome.PASS, None, "ok")],
                screened={"speech": "present", "airway": "absent", "voice": "absent"},
                branch_decisions=_decisions(airway=False, speech=True, voice=False),
                ran={"SPEECH": state}, hint_claims={},
            )
            assert any(phrase in reason.why for reason in folded.reasons)


class TestHintsForMismatchOnly:
    """A hint names a mismatch and prevents a discard. It has no other power on this axis."""

    def test_a_hinted_kind_the_branch_did_not_find_flags(self) -> None:
        """The kind, the hint that claimed it, and the branch's conclusion, all named."""
        folded = fold_file_verdict(
            [
                NodeVerdict("ADMIT", Outcome.PASS, None, "ok"),
                NodeVerdict("AIRWAY", Outcome.FAIL, "airway", "no span carries a label"),
            ],
            screened={"speech": "absent", "airway": "absent", "voice": "absent"},
            branch_decisions=_decisions(airway=True, speech=False, voice=False),
            ran={}, hint_claims={"airway": True},
        )
        assert folded.triage is Triage.FLAG
        assert folded.hints["airway"] == "claimed_not_found"

    def test_a_kind_found_that_no_hint_claimed_is_recorded_not_flagged(self) -> None:
        """Recorded; not a flag on its own."""
        folded = fold_file_verdict(
            [
                NodeVerdict("ADMIT", Outcome.PASS, None, "ok"),
                NodeVerdict("AIRWAY", Outcome.PASS, "airway", "labelled"),
            ],
            screened={"speech": "absent", "airway": "present", "voice": "absent"},
            branch_decisions=_decisions(airway=True, speech=False, voice=False),
            ran={}, hint_claims={},
        )
        assert folded.hints["airway"] == "found_unclaimed"
        assert folded.triage is Triage.PASS

    def test_a_hint_never_turns_a_flag_into_a_pass(self) -> None:
        """Its one power is to prevent a discard and to name a mismatch."""
        folded = fold_file_verdict(
            [
                NodeVerdict("ADMIT", Outcome.PASS, None, "ok"),
                NodeVerdict("SPEECH", Outcome.FLAG, "speech", "pii in the target's speech"),
            ],
            screened={"speech": "present", "airway": "absent", "voice": "absent"},
            branch_decisions=_decisions(airway=False, speech=True, voice=False),
            ran={}, hint_claims={"speech": True},
        )
        assert folded.triage is Triage.FLAG


class TestTheReleaseAxis:
    """Only a REDACT pass clears an artifact, and not_assessed is not releasable."""

    def test_no_redact_verdict_is_not_assessed(self) -> None:
        """No speech branch, no words, or no PII found."""
        folded = fold_file_verdict(
            [NodeVerdict("ADMIT", Outcome.PASS, None, "ok")],
            screened={"speech": "absent", "airway": "present", "voice": "absent"},
            branch_decisions=_decisions(airway=True, speech=False, voice=False),
            ran={}, hint_claims={},
        )
        assert folded.release is Release.NOT_ASSESSED

    def test_a_redact_flag_withholds(self) -> None:
        """Unresolved is not cleared."""
        folded = _with_redact(Outcome.FLAG)
        assert folded.release is Release.WITHHELD

    def test_a_redact_fail_withholds(self) -> None:
        """A finding survived verification."""
        assert _with_redact(Outcome.FAIL).release is Release.WITHHELD

    def test_a_redact_pass_is_releasable(self) -> None:
        """For its artifacts only; never for the store."""
        assert _with_redact(Outcome.PASS).release is Release.RELEASABLE


class TestARedactNonPassIsVisibleWithoutFlippingTriage:
    """Triage asks whether a human must look; release asks whether an artifact may be handed on."""

    def test_a_surviving_finding_does_not_move_triage(self) -> None:
        """A release problem is not a measurement problem."""
        folded = _with_redact(Outcome.FAIL, speech=Outcome.PASS, screened_speech="present")
        assert folded.triage is Triage.PASS
        assert folded.release is Release.WITHHELD

    def test_it_appears_in_reasons_regardless(self) -> None:
        """A consumer filtering on triage == pass sees the release axis in the same record."""
        folded = _with_redact(Outcome.FAIL, speech=Outcome.PASS, screened_speech="present")
        assert any(reason.node == "REDACT" for reason in folded.reasons)
```

- [ ] **Step 2 — run them; expect FAIL** (`ImportError: cannot import name 'Triage'`).
  `uv run pytest src/tests/audio/workflows/triage/vocabulary_test.py -x -q`

- [ ] **Step 3 — rewrite the vocabulary.**

`src/senselab/audio/workflows/triage/vocabulary.py`: `KindState`'s `UNDECIDED` becomes `UNCERTAIN`
(value `"uncertain"`), and:

```python
class Triage(Enum):
    """What should happen to this recording. The file axis; a node's ``Outcome`` is not one of these."""

    PASS = "pass"
    FLAG = "flag"
    DISCARD = "discard"


@dataclass(frozen=True)
class BranchDecision:
    """What ``routing`` decided about one branch, as the fold reads it.

    Attributes:
        branch: The branch's name.
        kind: The kind it concludes about.
        will_run: Whether routing selected it.
        kind_state: What TAXONOMY classified, verbatim.
        forced_by_hint: Whether a hint added it.
    """

    branch: str
    kind: str
    will_run: bool
    kind_state: str
    forced_by_hint: bool


@dataclass(frozen=True)
class FileVerdict:
    """The graph's conclusion about one recording, on both axes.

    Attributes:
        triage: What should happen to the recording.
        release: Whether REDACT's artifacts may be handed on. Never describes the store.
        discard_ground: ``"unmeasurable"``, ``"acoustically_empty"`` or None — the two grounds carry
            different reasons and a consumer that cannot tell them apart treats an empty recording as
            a broken one.
        kinds: The resolved state per kind, after branch authority.
        screened: What TAXONOMY classified. Present always, beside ``kinds``.
        agreement: ``agree`` | ``mismatch`` | ``resolved`` | ``not_run`` per kind.
        hints: ``claimed_and_found`` | ``claimed_not_found`` | ``found_unclaimed`` | ``no_claim``.
        reasons: Every contributing verdict, in order — not only the deciding one.
        ran: Whether each node ran.
        branches: The routing decision joined to the branch verdict.
    """

    triage: Triage
    release: Release
    discard_ground: str | None = None
    kinds: dict[str, str] = field(default_factory=dict)
    screened: dict[str, str] = field(default_factory=dict)
    agreement: dict[str, str] = field(default_factory=dict)
    hints: dict[str, str] = field(default_factory=dict)
    reasons: list[NodeVerdict] = field(default_factory=list)
    ran: dict[str, RunState] = field(default_factory=dict)
    branches: dict[str, dict[str, Any]] = field(default_factory=dict)
```

`fold_file_verdict` is rewritten to the signature

```python
def fold_file_verdict(
    node_verdicts: Sequence[NodeVerdict],
    *,
    screened: Mapping[str, str],
    branch_decisions: Mapping[str, BranchDecision],
    ran: Mapping[str, RunState],
    hint_claims: Mapping[str, bool],
) -> FileVerdict:
```

and implements, in order: (1) resolve `kinds` from branch authority — a branch with a verdict about
kind `k` sets `kinds[k]` to `"present"` on `PASS`/`FLAG` and `"absent"` on `FAIL`, and every other kind
keeps `screened[k]`; (2) compute `agreement` per kind from the table in `verdict.md`; (3) compute
`hints` per kind; (4) append the mismatch, silent-branch and hint-mismatch reasons; (5) the triage
ladder — ADMIT fail → `DISCARD`/`unmeasurable`; any `flag` or fired reason → `FLAG`; all-absent with
nothing found and no hint → `DISCARD`/`acoustically_empty`; else `PASS`; (6) the release fold from
REDACT's verdict, with an absent verdict → `NOT_ASSESSED`; (7) REDACT's non-pass appended to `reasons`
on every path, so it is never invisible.

**`speech_type` mismatch is a `hint_claims` entry, not a special case.** `verdict.py` builds
`hint_claims` by mapping `hint.may_contain` and `hint.metadata["speech_type"]` through
`routing.hint_kind_map` — the same map ROUTING used, so a tag that forced a branch is the same tag that
can name a mismatch, and the two cannot disagree about what a tag means.

- [ ] **Step 4 — run the vocabulary tests; expect PASS.**
  `uv run pytest src/tests/audio/workflows/triage/vocabulary_test.py -x -q`

- [ ] **Step 5 — rewrite `verdict.py` to feed the new fold.**

`_kind_predictions` becomes `_screened`, returning `dict[str, str]` verbatim from the `kind` elements
(no `KindState` coercion, no `not_screened` special case — sibling T2 deleted it). A new
`_branch_decisions(store)` reads sibling T3's `branch_decision` entities into `BranchDecision`s under
the store's shared latest-live rule. A new `_hint_claims(config, hint)` builds the claim map. The
`_is_gated` helper is **deleted** — the branch decisions answer that question directly now.

`_GRAPH_ORDER` gains `"routing"` in the same casing `run.GRAPH_ORDER` uses:

```python
_GRAPH_ORDER = ("ADMIT", "PREPROCESS", "TAXONOMY", "routing", "AIRWAY", "SPEECH", "VOICE", "REDACT")
```

The written detail becomes exactly `verdict.md`'s product: `triage`, `release`, `reasons`, `ran`,
`branches`, `kinds`, `screened`, `agreement`, `hints`. The verdict entity's `outcome` attribute
carries `file_verdict.triage.value`, and `write_verdict`'s `outcome` parameter is widened to
`Outcome | Triage` (both are enums with a `.value`; the annotation change is the whole edit).

- [ ] **Step 6 — run the node tests; expect PASS.**
  `uv run pytest src/tests/audio/workflows/triage -x -q`

- [ ] **Step 7 — lint, type-check, commit.**
  `uv run ruff format src/senselab/audio/workflows/triage src/tests/audio/workflows/triage`
  `uv run ruff check src/senselab/audio/workflows/triage src/tests/audio/workflows/triage`
  `uv run mypy src/senselab/audio/workflows/triage`
  `git commit -m "feat(triage/verdict): pass, flag, discard, with each branch authority over its own kind"`

**Interfaces:**

*Consumed:* every node's `verdict` entity; sibling T2's `kind` entities (`state` as a bare string);
sibling T3's `branch_decision` entities and `BRANCH_FOR_KIND`; `routing.hint_kind_map`; the runner's
`ran` mapping.

*Produced (the T8→T9 contract):*
- `Triage`, `BranchDecision`, the rewritten `FileVerdict` and `fold_file_verdict` — all exported from
  `senselab.audio.workflows.triage`.
- `verdict(store, source, config, hint=None, *, run_dir, ran=None) -> VerdictResult` with
  `file_verdict: FileVerdict`.
- One `verdict` entity carrying `verdict.md`'s product verbatim. **T9 reads this entity and nothing
  else to build the summary JSON's `verdict` block.**

**Superseded tests, deleted with the ruling that justifies each:**

| deleted | ruling |
| --- | --- |
| every `fold_file_verdict` test asserting `Outcome.FAIL` on the file axis | verdict.md: the axis is `pass \| flag \| discard` |
| the `_BRANCH_FOR_KIND` / `voice_no_words` mapping tests | taxonomy.md and branch-voice.md name the kind `voice`; routing owns the mapping |
| `test_absent_predicted_and_branch_passed_promotes_the_kind` and the rest of the v1 contradiction table | verdict.md: a branch's conclusion **is** the resolved kind, and the classification's disagreement is recorded as a mismatch rather than resolved by promotion |
| `test_gated` and `_is_gated` | verdict.md: the `branch_decision` rows answer "was this branch asked" directly |

---

### Task 9: REPORT — a per-file summary and a summary JSON, on every file and every outcome

**Scope:** `src/senselab/audio/workflows/triage/nodes/report.py` (new);
`src/senselab/audio/workflows/triage/run.py` (call it last, and place `summary/` under the run root);
`src/senselab/audio/workflows/triage/data/config/default.yaml` (`report.format`);
`src/tests/audio/workflows/triage/nodes/report_test.py` (new).

**Design points this task must not get wrong (from `report.md`):**

- **Every run emits both products, on every file, on every outcome** — including a file ADMIT refused,
  where the report says that and nothing else (V24).
- **It writes no elements and asserts nothing.** A rendering is not evidence.
- **The summary respects the PII marking.** A `word` element the scan marked is rendered redacted, and
  **no matched text appears anywhere in either product.**
- **Neither product is a released artifact.** Both carry element ids, which are a join key back into
  the store, so `summary/` sits beside the store and **not** under `released/`.
- **Provenance is embedded, not referenced**: the config hash *and* the merged mapping, the senselab
  commit, every model agent with its resolved commit or its `unresolved_reason` — **never a bare ref**.
- **Every claim names the store elements behind it**, through `steps.<step>.element_ids`.
- **`report.format` is `pdf` or `png`, declared rather than defaulted silently.**

**Steps:**

- [ ] **Step 1 — add the config key.**

Add to `derivation:`:

```
  report.format -- pdf or png. report.md calls it a presentation choice owed no measurement, but one
  that "must be declared rather than defaulted silently", so the key ships null and REPORT requires it.
  The two forms carry the same claims; the image form exists for a file whose summary fits on one page.
```

```yaml
report:
  format: null
```

- [ ] **Step 2 — write the failing tests.**

`src/tests/audio/workflows/triage/nodes/report_test.py`:

```python
"""REPORT: both products on every file and every outcome, no elements written, no matched text."""

import json
from pathlib import Path
from typing import Callable

import pytest

from senselab.audio.workflows.triage.config import TriageConfig, load_triage_config
from senselab.audio.workflows.triage.nodes.report import report
from senselab.utils.prov_store import ProvStore


def _png(tmp_path: Path) -> TriageConfig:
    """The packaged config with the report format declared as an image."""
    path = tmp_path / "report.yaml"
    path.write_text("report:\n  format: png\n")
    return load_triage_config(path)


class TestBothProductsAlways:
    """One summary and one JSON per file, whatever the graph concluded."""

    def test_a_full_run_emits_both(
        self, store: ProvStore, seeded_store: Callable[..., None], tmp_path: Path
    ) -> None:
        """The ordinary path."""
        seeded_store(store, tmp_path, full=True)
        artifacts = report(store, tmp_path / "summary", _png(tmp_path))
        assert artifacts["summary"].exists() and artifacts["summary"].suffix == ".png"
        assert artifacts["json"].exists()

    def test_an_admit_refusal_emits_both_and_says_nothing_was_measured(
        self, store: ProvStore, seeded_store: Callable[..., None], tmp_path: Path
    ) -> None:
        """A file ADMIT refused gets a report that says that, not an exception (V24)."""
        seeded_store(store, tmp_path, admit_failed=True)
        artifacts = report(store, tmp_path / "summary", _png(tmp_path))
        payload = json.loads(artifacts["json"].read_text())
        assert payload["verdict"]["triage"] == "discard"
        assert payload["branches"] == {}
        assert artifacts["summary"].exists()

    def test_a_null_format_refuses(
        self, store: ProvStore, config: TriageConfig, seeded_store: Callable[..., None], tmp_path: Path
    ) -> None:
        """A presentation choice owed no measurement is still owed a declaration."""
        seeded_store(store, tmp_path, full=True)
        with pytest.raises(ValueError, match="report.format"):
            report(store, tmp_path / "summary", config)

    def test_pdf_is_reachable_by_config(
        self, store: ProvStore, seeded_store: Callable[..., None], tmp_path: Path
    ) -> None:
        """The two forms carry the same claims; the choice does not change the content."""
        seeded_store(store, tmp_path, full=True)
        pdf_config = load_triage_config(_write(tmp_path, "report:\n  format: pdf\n"))
        artifacts = report(store, tmp_path / "summary", pdf_config)
        assert artifacts["summary"].suffix == ".pdf"


class TestItWritesNoElements:
    """A rendering is not evidence."""

    def test_the_store_is_unchanged(
        self, store: ProvStore, seeded_store: Callable[..., None], tmp_path: Path
    ) -> None:
        """No entity, no activity, no agent, no relation."""
        seeded_store(store, tmp_path, full=True)
        before = store.fingerprint()
        report(store, tmp_path / "summary", _png(tmp_path))
        assert store.fingerprint() == before


class TestItRespectsThePiiMarking:
    """No matched text appears anywhere in either product."""

    def test_a_marked_word_is_rendered_redacted_in_the_json(
        self, store: ProvStore, seeded_store: Callable[..., None], tmp_path: Path
    ) -> None:
        """The store holds PII by design; every artifact must respect the marking."""
        seeded_store(store, tmp_path, full=True, marked_words=[("alice", "PERSON")])
        artifacts = report(store, tmp_path / "summary", _png(tmp_path))
        text = artifacts["json"].read_text()
        assert "alice" not in text
        assert "[PERSON]" in text

    def test_an_unmarked_word_is_rendered_verbatim(
        self, store: ProvStore, seeded_store: Callable[..., None], tmp_path: Path
    ) -> None:
        """The marking is what redacts, not a blanket refusal to render words."""
        seeded_store(store, tmp_path, full=True, words=["hello"], marked_words=[])
        artifacts = report(store, tmp_path / "summary", _png(tmp_path))
        assert "hello" in artifacts["json"].read_text()


class TestPlacement:
    """summary/ sits beside the store, never under released/."""

    def test_the_summary_is_not_under_the_release_directory(
        self, tmp_path: Path, wav_writer: Callable[..., Path], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """It carries element ids and marked words' extents, so it inherits the store's sensitivity."""
        _stub_graph(monkeypatch)
        result = run_triage(wav_writer("s.wav", _sine()), tmp_path, _png(tmp_path))
        assert result.summary_dir.parent == result.run_dir.parent
        assert not result.summary_dir.is_relative_to(result.artifacts_dir)


class TestTheProvenanceIsEmbedded:
    """A hash identifies a run; the mapping is what makes it readable without the repository."""

    def test_the_config_hash_and_the_mapping_both_appear(
        self, store: ProvStore, seeded_store: Callable[..., None], tmp_path: Path
    ) -> None:
        """Both, always."""
        seeded_store(store, tmp_path, full=True)
        payload = json.loads(report(store, tmp_path / "summary", _png(tmp_path))["json"].read_text())
        assert payload["provenance"]["config_hash"]
        assert payload["provenance"]["config"]["name"] == "senselab-triage/default"

    def test_every_model_carries_its_resolved_commit_or_a_reason(
        self, store: ProvStore, seeded_store: Callable[..., None], tmp_path: Path
    ) -> None:
        """An agent whose commit could not be resolved appears with its reason, never with a bare ref."""
        seeded_store(store, tmp_path, full=True)
        payload = json.loads(report(store, tmp_path / "summary", _png(tmp_path))["json"].read_text())
        for model in payload["provenance"]["models"]:
            assert model["revision"] is not None or model["unresolved_reason"] is not None
            assert model["revision"] != "main"

    def test_every_step_names_the_elements_behind_it(
        self, store: ProvStore, seeded_store: Callable[..., None], tmp_path: Path
    ) -> None:
        """This is what makes the JSON a view of the store rather than a second copy of it."""
        seeded_store(store, tmp_path, full=True)
        payload = json.loads(report(store, tmp_path / "summary", _png(tmp_path))["json"].read_text())
        assert payload["steps"]
        for entry in payload["steps"].values():
            assert isinstance(entry["element_ids"], list)


class TestTheSummaryLayers:
    """One shared time axis, drawn from the store."""

    def test_the_shared_axis_carries_every_layer_the_store_holds(
        self, store: ProvStore, seeded_store: Callable[..., None], tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Waveform, envelope with floor, spans, phonation spans, the three label lanes, and the branches."""
        panels = _capture_panels(monkeypatch)
        seeded_store(store, tmp_path, full=True)
        report(store, tmp_path / "summary", _png(tmp_path))
        kinds = [panel["type"] for panel in panels[0]]
        assert kinds.count("segments") >= 5
        assert "waveform" in kinds and "features" in kinds

    def test_labelled_and_unlabelled_spans_are_distinguishable(
        self, store: ProvStore, seeded_store: Callable[..., None], tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """branch-airway.md requires it on the shared axis."""
        panels = _capture_panels(monkeypatch)
        seeded_store(store, tmp_path, full=True, airway_labelled=[(1.0, 1.3)], airway_unlabelled=[(2.0, 2.3)])
        report(store, tmp_path / "summary", _png(tmp_path))
        labels = {
            segment["label"]
            for panel in panels[0]
            if panel["type"] == "segments"
            for segment in panel["segments"]
        }
        assert "unlabelled" in labels and "Cough" in labels
```

- [ ] **Step 3 — run them; expect FAIL** (`ModuleNotFoundError: ...nodes.report`).
  `uv run pytest src/tests/audio/workflows/triage/nodes/report_test.py -x -q`

- [ ] **Step 4 — write `report.py`.**

The signature is `report(store, summary_dir, config, *, run_dir=None) -> dict[str, Path]`, returning
`{"summary": ..., "json": ...}`. It reads the whole store and writes nothing to it.

The four routines, each with its own docstring and none holding a number:

```python
def _redacted_text(store: ProvStore, word: Entity) -> str:
    """A word's renderable text: its category placeholder when the scan marked it, else the word.

    The store holds PII by design and the report carries element ids, so the report is not a released
    artifact — but no matched text may appear in it either way.

    Args:
        store: The provenance store.
        word: A ``word`` entity.

    Returns:
        ``"[<CATEGORY>]"`` when a live ``pii`` label assertion is derived from this word, else the
        word's own text.
    """


def _panels(store: ProvStore, run_dir: Path, config: TriageConfig) -> list[dict[str, Any]]:
    """The summary's layers on one shared time axis, drawn from whatever the store holds.

    A layer whose derivative is absent is omitted; nothing raises for want of one, because
    report.md requires a product on every outcome including a file ADMIT refused.

    Args:
        store: The provenance store.
        run_dir: Where sidecar paths resolve against.
        config: The triage configuration, read for the envelope decimation stride only.

    Returns:
        Panel specifications for ``plot_aligned_panels``.
    """


def _steps(store: ProvStore) -> dict[str, dict[str, Any]]:
    """Per-step summary fields, each naming the element ids behind it.

    Args:
        store: The provenance store.

    Returns:
        ``{step: {**verdict detail, "element_ids": [...]}}`` over every node that wrote a verdict.
    """


def _provenance(store: ProvStore, config: TriageConfig, run_id: str) -> dict[str, Any]:
    """The run's provenance, embedded rather than referenced.

    Args:
        store: The provenance store, read for its Agent records.
        config: The triage configuration.
        run_id: The run's id.

    Returns:
        ``{config_hash, config, commit, models, run_id, started, ended}``. Every model agent appears
        with its resolved commit or its ``unresolved_reason`` — never a bare ref.
    """
```

`_panels` **reuses `plot_aligned_panels`** (`senselab.audio.tasks.plotting.plotting`) exactly as
`airway.py` already does, with these panels in order: `waveform`; `features` carrying the envelope and
its floor; one `segments` panel each for PREPROCESS's spans (labelled by `peak_over_floor_db`),
phonation spans (labelled by `member` and `production`), the YAMNet / AST / HeAR label lanes (one
segment per non-empty window, labelled by its joined label set), SPEECH's spans (labelled by
`attributed_to`, with `nontarget` appended), AIRWAY's spans (labelled by their label, or
`"unlabelled"`), VOICE's spans, and REDACT's extents; then `spectrogram`. The `airway.py`
`_render_figure` helper is **deleted** and its AIRWAY-only figure with it: one shared axis per file
replaces one figure per branch, which is what `report.md` asks for.

The side blocks beside the axis are rendered as a text panel carrying, in order: the branch decisions
(`will_run`, `forced_by_hint`, `kind_state`), each branch's conclusion and flags, TAXONOMY's
`screened` beside the resolved `kinds` with the per-kind `agreement`, and the verdict's `triage`,
`release` and every reason — **with REDACT's outcome shown whatever the triage axis says**.

`run.py` gains, after `verdict`:

```python
    summary_dir = layout.root / SUMMARY_SUBDIR
    summary = _attempt_artifacts(lambda: report(store, summary_dir, config, run_dir=layout.run_dir))
```

`SUMMARY_SUBDIR = "summary"`, `RunLayout` gains `summary_dir`, and `TriageRunResult` gains
`summary_dir: Path` and `summary: dict[str, Path]`. A REPORT failure is recorded like any other node's
and does not change the verdict — the store was already written.

- [ ] **Step 5 — run them; expect PASS.**
  `uv run pytest src/tests/audio/workflows/triage/nodes/report_test.py src/tests/audio/workflows/triage/run_test.py -x -q`

- [ ] **Step 6 — lint, type-check, commit.**
  `uv run ruff format src/senselab/audio/workflows/triage src/tests/audio/workflows/triage`
  `uv run ruff check src/senselab/audio/workflows/triage src/tests/audio/workflows/triage`
  `uv run mypy src/senselab/audio/workflows/triage`
  `git commit -m "feat(triage/report): one summary and one JSON per file, on every outcome"`

**Interfaces:**

*Consumed:* the whole store; T8's `verdict` entity; `plot_aligned_panels(audio, panels, title=...)`;
`common.live_entities`, `find_measurement`, `find_measurements`, `resolve_stream`;
`ProvStore.fingerprint()` (used only by the test that pins "writes no elements").

*Produced:* `report(store, summary_dir, config, *, run_dir=None) -> dict[str, Path]`;
`<run_root>/summary/summary.{pdf,png}` and `<run_root>/summary/summary.json`;
`TriageRunResult.summary_dir` and `.summary`.

**Superseded code, deleted with the ruling that justifies it:**

| deleted | ruling |
| --- | --- |
| `airway._render_figure` and `AirwayResult.figure_path` | report.md: one summary per file on one shared axis, drawn after the verdict — a per-branch figure drawn mid-branch cannot carry the branch decisions or the fold |
| `airway_test.py`'s figure assertions | same |

---

### Task 10: The CrisperWhisper-on-CPU empty-output diagnostic — bounded

**Scope:** a diagnostic, then **either** a fix in
`src/senselab/audio/tasks/speech_to_text/crisperwhisper.py` **or** one row in
`specs/20260815-215106-analyze-audio-audit/register.md`. Independent of every other task; may run
first.

**This task has a budget and a stop rule.** It is a bounded investigation, not an open-ended one. The
budget is **one recording, one host, three hypotheses, and the four steps below.** If step 4 does not
identify the mechanism, **stop and write the register row** with what was measured and what was
excluded. Do not widen to a second model, a second host, or a second recording; do not refactor the
backend; do not disable the backend anywhere.

**What is already known, so it is not re-derived:**

- The dtype is already correct on CPU: `transcribe_with_crisperwhisper`
  (`crisperwhisper.py:166-171`) selects `device_str = "cpu"` and `compute_type = "float32"`, and
  float16 only on CUDA. **Hypothesis "CPU dtype" is therefore already half-excluded** and step 2 only
  confirms the value reaches the worker.
- The backend is platform-selected: `ct2` on Linux x86_64, `transformers` everywhere else
  (`crisperwhisper.py:36-66`). A macOS CPU run takes the **transformers** path, which is a different
  decode implementation from the one CI exercises.
- The worker reads `r.text` and `r.words` off the library's result object, falling back through
  `("word", "text")` and `("probability", "confidence", "score", "prob")`. **An empty output could be
  an empty decode or an unreadable result shape**, and the worker cannot currently tell them apart.

**The three hypotheses, in the order step 4 tests them:**

| # | hypothesis | what would confirm it |
| --- | --- | --- |
| H1 | the transformers-backend decode genuinely returns empty text on CPU | the raw result object carries `text == ""` and `words == []` |
| H2 | the decode succeeds and the worker cannot read its shape | the raw result carries text under a name the `_first_attr` list does not cover, or `words` is a non-empty iterable of a type `getattr(w, "start")` fails on |
| H3 | the subprocess is killed or times out and the parent reads the truncated stream as empty | the worker's stderr carries a signal or a timeout, or `parse_subprocess_result` sees a short read |

**Steps:**

- [ ] **Step 1 — reproduce, once, on one file.**

```bash
uv run python -c "
from pathlib import Path
from senselab.audio.data_structures import Audio
from senselab.audio.tasks.speech_to_text.api import transcribe_audios
from senselab.utils.data_structures import DeviceType, HFModel

audio = Audio(filepath=str(Path('src/tests/data_for_testing/audio_48khz_mono_16bits.wav')))
model = HFModel(path_or_uri='nyralabs/CrisperWhisper2.0_turbo', revision='main')
[line] = transcribe_audios([audio], model=model, device=DeviceType.CPU)
print('text:', repr(line.text))
print('chunks:', len(line.chunks or []))
"
```

Record the exact output. **If the text is non-empty, the defect does not reproduce on this host and
this file: stop here and write the register row saying so**, naming the host, the backend
(`transformers` off Linux x86_64), and the file. A defect that does not reproduce is a finding.

- [ ] **Step 2 — confirm what reached the worker.**

Re-run step 1 with `SENSELAB_LOG_LEVEL=DEBUG` (or add a one-line `print` to the payload assembly at
`crisperwhisper.py:190-200` and revert it after), and record the `device`, `compute_type`, `backend`
and `model_id` values in the JSON payload. **Expected:** `cpu`, `float32`, `transformers`, and a local
snapshot directory rather than a repo id. Any deviation from those four is the finding; write it up
and go to step 5.

- [ ] **Step 3 — capture the raw result object, to separate H1 from H2.**

Run the worker's own body directly inside the venv, with one added line that prints the result's
shape before the extraction:

```bash
VENV=$(uv run python -c "from senselab.utils.subprocess_venv import venv_python; print(venv_python('crisperwhisper'))")
"$VENV" -c "
from crisperwhisper import CrisperWhisperModel
import sys
model = CrisperWhisperModel(sys.argv[1], backend='transformers', device='cpu', compute_type='float32')
r = model.transcribe(sys.argv[2], language='en', word_timestamps=True)
print('type:', type(r))
print('attrs:', [a for a in dir(r) if not a.startswith('_')])
print('text:', repr(getattr(r, 'text', None)))
words = getattr(r, 'words', None)
print('words type:', type(words), 'len:', len(list(words)) if words is not None else None)
" "<snapshot-dir>" "<wav>"
```

`<snapshot-dir>` is the path printed by step 2. **`text == "" and words == []` confirms H1;
attributes carrying the transcript under other names confirm H2.**

- [ ] **Step 4 — check H3 only if steps 1-3 leave it open.**

Re-run step 1 and capture the worker's stderr and exit status from `parse_subprocess_result`. A
non-zero status, a signal, or a stderr traceback is H3.

- [ ] **Step 5 — one of two outcomes, and no third.**

**If H2 is confirmed** — the decode worked and the worker could not read it — fix it: extend
`_first_attr`'s name list to cover the observed attribute, **and** make the worker distinguish the two
cases it currently conflates, by adding to its result payload:

```python
        results.append({
            "text": getattr(r, "text", "") or "",
            "language": getattr(r, "language", language),
            "words": words,
            "result_type": type(r).__name__,
            "result_attrs": sorted(a for a in dir(r) if not a.startswith("_")),
        })
```

and have `transcribe_with_crisperwhisper` raise — rather than return an empty `ScriptLine` — when
`text` is empty **and** `result_attrs` carries a name the extraction did not read. An empty transcript
and an unreadable result shape are different findings, and a backend that returns the first for the
second is the absence-versus-zero failure this codebase keeps removing. Add one test at
`src/tests/audio/tasks/speech_to_text/crisperwhisper_test.py` that feeds the worker's *parser* a
recorded result payload of each shape and asserts the two are told apart — **the parser only; no model
loads in the test.**

**If H1 or H3 is confirmed, or nothing is** — write one row in
`specs/20260815-215106-analyze-audio-audit/register.md` naming: the host and OS, the selected backend,
the file, the exact reproduction command, which hypotheses were excluded and by which observation, and
what would settle the remaining one. Assign it the next free `F-*` id. **Do not fix a decode you have
not localised, and do not disable the backend.**

- [ ] **Step 6 — commit.** Either
  `git commit -m "fix(crisperwhisper): an unreadable result shape is not an empty transcript"`
  or
  `git commit -m "register: the CrisperWhisper CPU empty-output finding, with what it excludes"`

**Interfaces:**

*Consumed:* `transcribe_audios(audios, model, device=...)`,
`CrisperWhisperASR.transcribe_with_crisperwhisper`, `venv_python`, `parse_subprocess_result`.

*Produced:* either a narrowed `crisperwhisper.py` result contract (`result_type`, `result_attrs`, and a
raise on the ambiguous case) with one parser test, or one `F-*` register row. **Nothing in this task
changes the triage graph**, and no other task in either plan file depends on it.

---

## What this plan file does not build

- PREPROCESS, TAXONOMY and routing — `plan-v2-1.md`.
- Any fit for any open key. Every one stays `null`, and every test that needs a value supplies an
  override.
- The `noise` redaction fill, which is refused with the measurement it is owed (V22).
- Multi-file orchestration, which lives in `specs/20260817-triage-workflow-dag/nextflow/`.
- Any second pass over a suppressed-foreground stream.

## Self-review

### Spec coverage — every v2 spec section this file owns maps to a task

| spec section | task | where |
| --- | --- | --- |
| branch-speech.md §Signature, §What it reads | T4 | the new `enrollment` keyword; the consensus read replacing the per-recognizer gather |
| branch-speech.md §1 Transcript | T4 | `consensus_transcript` read; `fuse_word_streams` deleted; `TestItReadsTheConsensusAndReFusesNothing` |
| branch-speech.md §2 Speech spans | T4 | `_group_words_into_spans` over consensus words, unchanged in rule |
| branch-speech.md §3 Corroboration | T4 | kept from v1: YAMNet coverage + SQUIM, both floors null → `not_evaluated` |
| branch-speech.md §4 Diarization | T4 | the conditional second diarizer; the declared count not read; the V18 degenerate-interval fix; `TestTheSecondDiarizerIsConditional`, `TestTheDegenerateDiarizationInterval` |
| branch-speech.md §5 Separation | T4 (V17) | `speech.separation_backend` gate, both backends reachable, ≥3 reported; `TestSeparationIsMeasurementGated` |
| branch-speech.md §6 Speaker identification | T4 (V15, V16) | `Enrollment`, the refusal rule, the `enrollment` element, `nontarget` marking; `TestEnrollment` |
| branch-speech.md §7 PII | T4 | one scan over the consensus text; speaker-scoped decision; `TestPiiOnTheConsensus` |
| branch-speech.md §8 Quality | T4 (V19) | SQUIM on plain, disruptions on the original; `TestQualityAndTheStreamsItNames` |
| branch-speech.md §9 The non-target axis | T4 | `_proximity`, the three legs, `nontarget_speech_s` null; `TestTheNonTargetAxis` |
| branch-speech.md §10 REDACT | T4 + sibling T3 | `pii` presence gates `run._speech_found_pii` |
| branch-speech.md §"It does not read AIRWAY" | T4 | `TestItDoesNotReadAirway` — verifies 8537a83f and pins it |
| branch-speech.md §Open derivations (v2), 6 rows | sibling T1 | keys created there; read here |
| branch-voice.md §Signature, §1 The subject | T5 | phonation spans as the subject; residual helpers deleted; `TestTheSubjectIsPreprocessesSpans` |
| branch-voice.md §2 Tracks | T5 | kept from 29da3633: computed once on the stream, sliced |
| branch-voice.md §3 Period marks | T5 (V20) | the half-frame tolerance; `unvoiced_span` vs `shorter_than_mark_window`; `TestTheHalfFrameTolerance`, `TestProductionModes` |
| branch-voice.md §4 Edges | T5 | `onset_kind`/`offset_kind`; `TestEdgesAreNamedApart` |
| branch-voice.md §5 Duration against the task | T5 | `longest_span_s` + its criterion, `task_duration_ranges`; `TestMptRecoverableProducts` |
| branch-voice.md §"The F0 range serves a population" | T5 | `_f0_range`, refusal at load; `TestTheF0RangeServesAPopulation` |
| branch-voice.md §"Two members that are not acoustic classes" | T5 | nothing is labelled; MPT is a duration and loud phonation is a contrast, neither attached |
| branch-voice.md §Open derivations (v2), 4 rows | T5 + sibling T1 | three keys here, `phonation.*` already null |
| branch-airway.md §1 Label each span | T6 | stored `hear_window` membership; eligibility by transcript; `TestHearConfirmsRatherThanFinds` |
| branch-airway.md §2 Confirm or contest | T6 (V21) | co-location in the HeAR window; disjointness refused at load; `TestContestRequiresColocation` |
| branch-airway.md §3 Lexical contamination | T6 | kept from v1, now reading consensus `word` entities and ignoring `event`s |
| branch-airway.md §4 `K` is adjustable | T6 | `airway.k_db`, `k_db_by_task`, `k_margin_db`, `merged_n`; `TestTheGateIsAdjustableAndItsEdgeFlags` |
| branch-airway.md §5 Outcome, §Product | T6 | the verdict detail; the report renders labelled and unlabelled distinguishably (T9) |
| branch-airway.md §Open derivations (v2), 3 rows | sibling T1 | keys created there; read here |
| redact.md §When it runs | sibling T3 + T7 | `run._speech_found_pii`; no verdict on a path never called |
| redact.md §What it redacts | T7 | every finding regardless of speaker; `TestItRedactsEverySpeaker` |
| redact.md §Conservative at the edges | T7 | `plan_redactions(padding_ms=...)`, unchanged |
| redact.md §The fill is configurable | T7 (V22) | `apply_redactions(fill=..., bleep_hz=...)`; `TestTheFillIsDeclared` |
| redact.md §Verification does not re-transcribe | T7 | `_verify` over text only; `TestVerificationDoesNotReTranscribe` |
| redact.md §remediable exactly once | T7 | `replanned_n`, `unremediable`; `TestRemediationHappensExactlyOnce` |
| redact.md §Two exfiltration paths | T7 | kept from v1: `+` reserved, bounds-only error messages |
| redact.md §The store cannot be made releasable, §The source is not destroyed | T7, T9 | no element id in an artifact; `summary/` outside `released/` |
| redact.md §Open derivations (v2), 2 rows | T7 | `redaction.fill` created; `padding_ms` already null |
| verdict.md §Two axes | T8 | `Triage` + `Release`; `TestTheTriageVocabulary` |
| verdict.md §`discard` is narrow | T8 | two grounds, `discard_ground`; `TestDiscardIsNarrow` |
| verdict.md §Branch authority is scoped | T8 | `TestBranchAuthorityIsScoped` |
| verdict.md §TAXONOMY is reported beside | T8 | `kinds` + `screened` + `agreement`; `TestTaxonomyIsReportedBeside` |
| verdict.md §A branch that never ran | T8 | `BranchDecision`; `TestABranchThatNeverRanIsNotOneThatFailed` |
| verdict.md §Hints are read here | T8 | `hint_claims` via `routing.hint_kind_map`; `TestHintsForMismatchOnly` |
| verdict.md §The triage fold, §The release fold | T8 | the two ladders; `TestTheReleaseAxis` |
| verdict.md §A REDACT non-pass does not flip triage | T8 | `TestARedactNonPassIsVisibleWithoutFlippingTriage` |
| verdict.md §Product, §`ran` is merged | T8 | the detail written; the runner's mapping over the store's, unchanged from v1 |
| report.md §Two products | T9 | `TestBothProductsAlways` |
| report.md §The summary | T9 | `_panels` reusing `plot_aligned_panels`; `TestTheSummaryLayers` |
| report.md §The summary JSON | T9 | `_steps`, `_provenance`; `TestTheProvenanceIsEmbedded` |
| report.md §"respects the PII marking" | T9 | `_redacted_text`; `TestItRespectsThePiiMarking` |
| report.md §Placement | T9 | `TestPlacement` |
| report.md §Open derivations (v2), 1 row | T9 | `report.format` |

**No v2 spec section in this file's scope is unassigned.** The sections this file does not own —
`preprocess.md`, `taxonomy.md`, `routing.md`, `store.md` — are covered by `plan-v2-1.md`'s
self-review, which carries the same table for them.

**One spec sentence deliberately not implemented, and why:** `branch-voice.md` §2 says "Where a
separated or enhanced stream exists for the extent, F0 and its derived statistics are recomputed on
it." T5 does **not** implement the recomputation, because `speech.separation_backend` is null, no
separated stream exists on any current run, and building a recomputation path nothing can exercise
would be untested code shipped on a hypothesis. What T5 *does* implement is the invariant the sentence
protects: `f0_median_hz` is reported only with `f0_stream` beside it, pinned by
`test_f0_median_is_reported_only_with_its_stream`, so the day a separated stream exists the two
measurements cannot be conflated. **Flagged to the owner as a deliberate omission.**

### Placeholder scan

Searched for `TBD`, `TODO`, `FIXME`, `XXX`, `...` as an ellipsis-in-code, "add validation", "similar to
task N", "as above", "etc." in a step body, and "handle appropriately".

- **`TBD`/`TODO`/`FIXME`/`XXX`: none.**
- **`...` appears in three places, each deliberate and each named here:** (a) in T4's separation and
  T7's verification code blocks, marking *unchanged surrounding lines* an implementer must not delete
  — the changed lines around them are complete; (b) in T4's step-7 prose where a helper's body is
  described rather than shown; (c) in T10's shell placeholders `<snapshot-dir>` and `<wav>`, which are
  values step 2 produces and step 3 consumes.
- **"similar to task N": none.** Every cross-task dependency names the artifact it needs and where it
  comes from.
- **Four places where a step describes rather than shows, all flagged:**
  1. T4 step 7's step-6 enrollment block — five lettered obligations rather than a code block, because
     the block is 60 lines of store writes whose *shape* is already fixed by the `enrollment` and
     `target_match` schemas given in full.
  2. T4 step 7's `_proximity` — the docstring and the two formulae in prose (`tilt_db_per_octave` is
     the least-squares slope of log-magnitude against `log2(f)`; `d_to_r_db` is
     `10*log10(peak/(total-peak))` over the autocorrelation), which is more precise than a body would
     be and leaves no choice open.
  3. T7's `_matches_surviving` — one sentence stating the rule ("a live `word` carrying a `pii`
     marking of that category and not already covered by a planned extent").
  4. T9's four routines — full signatures and full docstrings, with `_panels`'s panel list enumerated
     in order and its reuse of `plot_aligned_panels` named. The bodies are rendering code with no
     decisions in them.
  **A reviewer should judge these four; everything else in the file is a literal code block.**

### Type-consistency scan

| type | fixed in | every reader agrees |
| --- | --- | --- |
| `Enrollment` | T4 | `run_triage(enrollment=...)`, `speech(..., enrollment=...)`, `Enrollment.refusal_against` — sibling T3 declares the parameter as `Any` and T4 narrows it; **T4 step 8 is that narrowing and must not be skipped** |
| `speech.enrollment_model` shape | `{model_id: str, revision: str}` (a mapping, not a bare string) | T4's `_required` reads both members; the config's derivation says so |
| `speaker_count` | `int \| None` — **None on the degenerate-interval path** | T4 writes; T8 folds only the branch's `Outcome`, so a `None` count never reaches the fold; T9 renders it |
| branch verdict `kind` | `"airway"` / `"speech"` / `"voice"` — lowercase, matching sibling T2's kind names and T3's `BRANCH_FOR_KIND` keys | T5 changes `KIND` to `"voice"`; **this is the join key T8 uses and a stale `voice_no_words` anywhere breaks the fold silently** |
| node name `"routing"` | lowercase | sibling T3's `NODE`, `run.GRAPH_ORDER`, **and T8's `_GRAPH_ORDER`** — T8 step 5 names it explicitly for this reason |
| `Triage` vs `Outcome` | `Triage` on the file axis only; every node still returns `Outcome` | T8's `write_verdict` widening to `Outcome \| Triage` is the only place they meet |
| `KindState.UNCERTAIN` | renamed from `UNDECIDED`, value `"uncertain"` | T8 renames; sibling T2 already writes the string `"uncertain"` — **the enum and the store string agree only after T8, so T8 must land before any code reads `KindState` from a store string** |
| `hint_claims` | `Mapping[str, bool]` keyed by kind, built from `routing.hint_kind_map` | T8 builds it in `verdict.py` and passes it to the pure fold, so the fold takes no config |
| `redaction.fill` | `str`, `"silence"` or `"bleep"` | T7's config read, `apply_redactions(fill=...)`, the verdict's `fill` field, T9's rendering |
| per-window `labels` | `list[str]` | sibling T1 writes; T6 reads membership |
| `phonation` span attributes | `duration_s: float`, `production: str`, `offset_criterion: str` | sibling T1 writes; sibling T2 reads `duration_s`; T5 reads all three |
| `report()` return | `dict[str, Path]` with keys `"summary"`, `"json"` | T9; `run.TriageRunResult.summary` |

Two inconsistencies found and fixed while writing:

1. T7's `_consensus_words` originally still filtered on `generating activity == "SPEECH"`, which
   sibling T1 made empty by making PREPROCESS the only `word` author — REDACT would have released a
   transcript of zero words and reported `unplaced_words_n: 0`, a clean-looking withhold over nothing.
   It now reads `live_entities(store, "word")` and the change is named in T7's superseded table.
2. T8's release fold originally mapped a REDACT `flag` through the same row as a `fail`, which is
   correct for the *axis* but lost the distinction `verdict.md` draws in `reasons`. `TestTheReleaseAxis`
   now pins both rows separately, and T7's outcome ladder was changed from `fail` to `flag` on an
   incomplete re-scan to match `redact.md`'s own words — a behaviour change from v1, called out in
   T7 step 8 rather than made silently.
