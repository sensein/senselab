"""SPEECH node tests. Every model call is faked at the node module; DSP and the store run real."""

import math
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

import numpy as np
import pytest
import torch
import yaml

from senselab.audio.data_structures import (
    Audio,
    AudioHints,
    SpeakerEmbeddingProvenance,
    TargetSpeakerEmbedding,
)
from senselab.audio.workflows.audio_analysis.level import integrated_lufs
from senselab.audio.workflows.triage.config import TriageConfig, load_triage_config
from senselab.audio.workflows.triage.enrollment import Enrollment
from senselab.audio.workflows.triage.nodes import speech as speech_module
from senselab.audio.workflows.triage.nodes.common import find_measurement, find_measurements, live_entities
from senselab.audio.workflows.triage.nodes.speech import speech
from senselab.audio.workflows.triage.vocabulary import Outcome
from senselab.text.tasks.pii_detection.api import PiiScan, PiiSpan, default_detectors
from senselab.utils.data_structures import ScriptLine
from senselab.utils.prov_store import Entity, ProvStore

SR = 16000
ENROLLMENT_MODEL = "speechbrain/spkrec-ecapa-voxceleb"
ENROLLMENT_SHA = "a" * 40

_SEEDER: Optional[Callable[..., None]] = None


class _FakeModel:
    """A model spec stub carrying exactly what the node reads: path_or_uri and commit_sha."""

    def __init__(self, path_or_uri: str, revision: str = "main") -> None:
        """Stub a resolved model.

        Args:
            path_or_uri: The model id.
            revision: What was asked for.
        """
        self.path_or_uri = path_or_uri
        self.revision = revision
        self.commit_sha = "c" * 40


# --------------------------------------------------------------------------------------
# Config builders. Not fixtures: a test that needs a variant builds one inline.
# --------------------------------------------------------------------------------------


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Merge ``overlay`` into ``base``, recursing into mappings.

    Args:
        base: The mapping written into.
        overlay: The mapping layered over it.

    Returns:
        ``base``, merged.
    """
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def _override(tmp_path: Path, yaml_text: str = "") -> TriageConfig:
    """The packaged config with ``speech.word_gap_ms`` supplied and ``yaml_text`` layered over it.

    Args:
        tmp_path: Where the override file is written.
        yaml_text: A partial config, in the production override shape.

    Returns:
        The resolved configuration.
    """
    values: dict[str, Any] = {"speech": {"word_gap_ms": 500}}
    _deep_merge(values, yaml.safe_load(yaml_text) or {})
    path = tmp_path / f"override-{abs(hash(yaml_text)) % 10**10}.yaml"
    path.write_text(yaml.safe_dump(values))
    return load_triage_config(path)


def _speech_config(tmp_path: Path) -> TriageConfig:
    """``speech.word_gap_ms`` and nothing else.

    Args:
        tmp_path: Where the override file is written.

    Returns:
        The configuration.
    """
    return _override(tmp_path)


def _second_diarizer_config(tmp_path: Path) -> TriageConfig:
    """That, plus a ranked second diarizer.

    Args:
        tmp_path: Where the override file is written.

    Returns:
        The configuration.
    """
    return _override(tmp_path, "speech:\n  second_diarizer: pyannote/speaker-diarization-3.1\n")


def _enrollment_config(tmp_path: Path) -> TriageConfig:
    """That, plus the enrollment probe and the match cut.

    Args:
        tmp_path: Where the override file is written.

    Returns:
        The configuration.
    """
    return _override(
        tmp_path,
        "speech:\n"
        "  second_diarizer: pyannote/speaker-diarization-3.1\n"
        "  enrollment_model:\n"
        f"    model_id: {ENROLLMENT_MODEL}\n"
        f"    revision: {ENROLLMENT_SHA}\n"
        "  target_match_cosine: 0.5\n",
    )


@pytest.fixture
def speech_config(tmp_path: Path) -> TriageConfig:
    """The base configuration, as a parameter.

    Args:
        tmp_path: Where the override file is written.

    Returns:
        The configuration.
    """
    return _speech_config(tmp_path)


@pytest.fixture
def second_diarizer_config(tmp_path: Path) -> TriageConfig:
    """The second-diarizer configuration, as a parameter.

    Args:
        tmp_path: Where the override file is written.

    Returns:
        The configuration.
    """
    return _second_diarizer_config(tmp_path)


@pytest.fixture
def enrollment_config(tmp_path: Path) -> TriageConfig:
    """The enrollment configuration, as a parameter.

    Args:
        tmp_path: Where the override file is written.

    Returns:
        The configuration.
    """
    return _enrollment_config(tmp_path)


# --------------------------------------------------------------------------------------
# The store this branch's predecessors leave behind.
# --------------------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _bind_shared_seeder(seed_preprocess_store: Callable[..., None]) -> Iterator[None]:
    """Bind T1's shared seeder for the duration of one test.

    ``_seed_speech_store`` layers SPEECH's own predecessors over it, and is called positionally
    rather than requested as a fixture, so the shared seeder is bound here instead.

    Args:
        seed_preprocess_store: The shared seeder.

    Yields:
        Nothing.
    """
    global _SEEDER
    _SEEDER = seed_preprocess_store
    yield
    _SEEDER = None


def _place(words: list[str], speakers: int, duration_s: float) -> list[tuple[str, tuple[float, float]]]:
    """Lay the words out in ``speakers`` contiguous groups, one group per diarizer segment.

    Args:
        words: The word texts, in order.
        speakers: How many equal parts of the word interval the words are split across.
        duration_s: The stream's duration, which bounds the layout.

    Returns:
        ``[(text, (start, end)), ...]``.
    """
    if not words:
        return []
    first = 0.5
    last = min(duration_s - 0.2, first + 0.4 * len(words) + 0.2)
    total = last - first
    placed: list[tuple[str, tuple[float, float]]] = []
    per_group = [len(range(index, len(words), speakers)) for index in range(speakers)]
    bounds: list[list[int]] = []
    cursor = 0
    for size in per_group:
        bounds.append(list(range(cursor, cursor + size)))
        cursor += size
    slots: dict[int, tuple[float, float]] = {}
    for group, members in enumerate(bounds):
        low = first + total * group / speakers
        high = first + total * (group + 1) / speakers
        slot = (high - low) / max(len(members), 1)
        for offset, index in enumerate(members):
            start = low + offset * slot + slot * 0.05
            slots[index] = (round(start, 4), round(low + (offset + 1) * slot - slot * 0.05, 4))
    for index, text in enumerate(words):
        placed.append((text, slots[index]))
    return placed


def _seed_speech_store(
    store: ProvStore,
    tmp_path: Path,
    *,
    words: Optional[list[str]] = None,
    word_extents: Optional[list[tuple[float, float]]] = None,
    events: Optional[list[str]] = None,
    speakers: int = 1,
    airway_labelled: Optional[list[tuple[float, float]]] = None,
    disruptions_file: bool = False,
    duration_s: float = 5.0,
) -> None:
    """Write what SPEECH's predecessors would have left, over T1's shared seeder.

    Args:
        store: The store to seed.
        tmp_path: The run directory the streams and sidecars go under.
        words: The consensus word texts.
        word_extents: Extents overriding the layout ``speakers`` would have produced.
        events: Bracketed or onomatopoeic non-words.
        speakers: How many equal parts of the word interval the words are laid out across, so a
            diarizer splitting the interval into that many segments attributes each word to one.
        airway_labelled: Extents AIRWAY labelled, each with the PREPROCESS span it hangs off.
        disruptions_file: Whether PREPROCESS's file-level disruption reading is present.
        duration_s: The streams' duration.
    """
    assert _SEEDER is not None, "the shared seeder is bound by the autouse fixture"
    placed: Optional[list[Any]] = None
    if words is not None:
        placed = (
            [(text, extent) for text, extent in zip(words, word_extents)]
            if word_extents is not None
            else list(_place(words, speakers, duration_s))
        )
    _SEEDER(
        store,
        duration_s=duration_s,
        words=placed,
        events=list(events) if events is not None else None,
        disruptions_file=disruptions_file,
    )
    _seed_level(store, tmp_path)
    for extent in airway_labelled or []:
        span_id = store.entity(
            prov_type="span",
            extent=extent,
            attributes={"peak_over_floor_db": 30.0, "k_db": 18.0, "signal": "preemphasised", "merged_proposals": 1},
        )
        airway_act = store.activity(node="AIRWAY", step="classify", parameters={})
        label_id = store.entity(
            prov_type="assertion",
            extent=extent,
            attributes={"verb": "label", "label": "Cough", "score": 0.97},
        )
        store.was_generated_by(label_id, airway_act)
        store.was_derived_from(label_id, span_id)


def _seed_level(store: ProvStore, tmp_path: Path) -> None:
    """Write PREPROCESS's file-level reading, which the proximity leg measures each span against.

    The shared seeder writes no ``level``, and without one every span's level sits over nothing.

    Args:
        store: The store to seed.
        tmp_path: The run directory the plain stream lives under.
    """
    plain = [e for e in live_entities(store, "stream") if e.attributes.get("name") == "plain"][-1]
    samples = Audio(filepath=str(tmp_path / plain.attributes["path"])).waveform.squeeze(0).numpy()
    rate = int(plain.attributes["sampling_rate"])
    activity = store.activity(node="PREPROCESS", step="level", parameters={})
    entity_id = store.entity(
        prov_type="measurement",
        extent=None,
        attributes={
            "name": "level",
            "signal": "plain",
            "peak_dbfs": float(20.0 * np.log10(max(float(np.abs(samples).max()), 1e-12))),
            "rms_dbfs": float(20.0 * np.log10(max(float(np.sqrt(np.mean(samples**2))), 1e-12))),
            "lufs": float(integrated_lufs(samples, rate)),
        },
    )
    store.was_generated_by(entity_id, activity)


# --------------------------------------------------------------------------------------
# Store readers.
# --------------------------------------------------------------------------------------


def _verdict_entity(store: ProvStore, node: str) -> Entity:
    """The latest live verdict entity one node wrote.

    Args:
        store: The provenance store.
        node: The node's name.

    Returns:
        The verdict entity.
    """
    found = [e for e in live_entities(store, "verdict") if e.attributes.get("node") == node]
    assert found, f"no {node} verdict in the store"
    return found[-1]


def _stream_id(store: ProvStore, name: str) -> str:
    """The latest live stream entity's id, by name.

    Args:
        store: The provenance store.
        name: The stream's name.

    Returns:
        The entity id.
    """
    found = [e for e in live_entities(store, "stream") if e.attributes.get("name") == name]
    assert found, f"no stream named {name!r}"
    return found[-1].id


# --------------------------------------------------------------------------------------
# Model stubs. Every one records what production asked for.
# --------------------------------------------------------------------------------------


def _segments(count: int, duration_s: float) -> list[ScriptLine]:
    """``count`` speakers splitting a cropped window into equal turns.

    Args:
        count: How many speakers.
        duration_s: The cropped window's duration.

    Returns:
        The segments, on the cropped clock.
    """
    if count <= 0:
        return []
    step = duration_s / count
    return [
        ScriptLine(speaker=f"SPEAKER_{index:02d}", start=index * step, end=(index + 1) * step) for index in range(count)
    ]


def _stub_diarizers(monkeypatch: pytest.MonkeyPatch, *, primary_speakers: int, second_speakers: int) -> list[str]:
    """Fake both diarizers and return the log of which was consulted.

    Args:
        monkeypatch: The patcher.
        primary_speakers: pyannote's count.
        second_speakers: the configured second diarizer's count.

    Returns:
        The mutable call log, ``["primary", "second"]`` in call order.
    """
    calls: list[str] = []

    def _fake(audios: list[Audio], model: Any = None, **kw: Any) -> list[list[ScriptLine]]:  # noqa: ANN401
        duration_s = audios[0].waveform.shape[-1] / audios[0].sampling_rate
        which = "primary" if "community-1" in str(getattr(model, "path_or_uri", "")) else "second"
        calls.append(which)
        return [_segments(primary_speakers if which == "primary" else second_speakers, duration_s)]

    monkeypatch.setattr(speech_module, "diarize_audios", _fake)
    return calls


def _stub_embedder(
    monkeypatch: pytest.MonkeyPatch, *, similarity: float = 0.99, target_label: str = "SPEAKER_00"
) -> list[dict[str, Any]]:
    """Fake the speaker embedder and return the log of what it was asked to embed.

    The enrollment vector is ``[1, 0]``, so ``target_label``'s probe is placed at ``similarity``
    from it and every other speaker's is placed orthogonal to it.

    Args:
        monkeypatch: The patcher.
        similarity: The cosine the target speaker's probe reaches.
        target_label: Which diarized speaker is the target.

    Returns:
        The mutable call log.
    """
    calls: list[dict[str, Any]] = []

    def _fake(audios: list[Audio], model: Any = None, device: Any = None) -> list[torch.Tensor]:  # noqa: ANN401
        calls.append({"n": len(audios), "model": str(getattr(model, "path_or_uri", ""))})
        target_index = int(target_label.rsplit("_", 1)[-1])
        orthogonal = math.sqrt(max(0.0, 1.0 - similarity**2))
        return [
            torch.tensor([similarity, orthogonal]) if index == target_index else torch.tensor([0.0, 1.0])
            for index in range(len(audios))
        ]

    monkeypatch.setattr(speech_module, "extract_speaker_embeddings_from_audios", _fake)
    return calls


def _stub_separator(monkeypatch: pytest.MonkeyPatch, *, sources: int = 0) -> list[dict[str, Any]]:
    """Fake source separation and return the log of what it was asked for.

    Args:
        monkeypatch: The patcher.
        sources: How many streams the fake returns.

    Returns:
        The mutable call log.
    """
    calls: list[dict[str, Any]] = []

    def _fake(
        audios: list[Audio],
        model: Any = None,  # noqa: ANN401
        n_sources: int = 2,
        mode: str = "speech_sound",
        source_classes: Optional[list[str]] = None,
        **kw: Any,  # noqa: ANN401
    ) -> list[list[Audio]]:
        calls.append(
            {
                "model": None if model is None else str(model.path_or_uri),
                "n_sources": n_sources,
                "mode": mode,
                "source_classes": source_classes,
            }
        )
        out: list[Audio] = []
        for index in range(sources):
            separated = Audio(waveform=audios[0].waveform.clone(), sampling_rate=audios[0].sampling_rate)
            separated.metadata["clearvoice"] = {
                "model": "alibabasglab/MossFormer2_SS_16K",
                "commit": "b" * 40,
                "source_index": index,
                "n_sources": sources,
                "input_norm_scalar": 0.31,
            }
            out.append(separated)
        return [out]

    monkeypatch.setattr(speech_module, "separate_audios", _fake)
    return calls


def _stub_pii(
    monkeypatch: pytest.MonkeyPatch,
    *,
    findings: list[tuple[str, str]],
    detectors_used: Optional[list[str]] = None,
) -> list[str]:
    """Fake the PII scan and return the list of texts it was handed.

    Args:
        monkeypatch: The patcher.
        findings: ``[(category, text), ...]`` the scan reports for every input.
        detectors_used: Which detectors ran; the module's default set when None.

    Returns:
        The mutable log of scanned texts.
    """
    scanned: list[str] = []
    used = list(detectors_used) if detectors_used is not None else default_detectors()

    def _fake(inputs: Any, detectors: Any = None, **kw: Any) -> list[PiiScan]:  # noqa: ANN401
        texts = [inputs] if isinstance(inputs, str) else list(inputs)
        scanned.extend(str(text) for text in texts)
        return [
            PiiScan(
                spans=[
                    PiiSpan(text=text, category=category, source="presidio", asr_model="consensus_transcript")
                    for category, text in findings
                ],
                detectors_used=list(used),
                failures={},
            )
            for _ in texts
        ]

    monkeypatch.setattr(speech_module, "scan_for_pii", _fake)
    return scanned


def _enrollment(*, commit: Optional[str] = ENROLLMENT_SHA, model: str = ENROLLMENT_MODEL) -> Enrollment:
    """One subject's enrollment, comparable unless a field is knocked out.

    Args:
        commit: The resolved commit, or None for an enrollment that cannot be compared.
        model: The embedding model behind the vector.

    Returns:
        The enrollment.
    """
    return Enrollment(
        subject_id="sub-01",
        vector=[1.0, 0.0],
        provenance=SpeakerEmbeddingProvenance(
            model_id=model,
            model_commit_sha=commit,
            unresolved_reason=None if commit is not None else "the estimator recorded none",
            source_files=["a.wav", "b.wav"],
            n_windows_used=12,
            n_windows_dropped=1,
        ),
    )


def _target_speaker_embedding() -> TargetSpeakerEmbedding:
    """The per-file target hint this branch no longer reads.

    Returns:
        A well-formed target embedding.
    """
    return TargetSpeakerEmbedding(
        vector=[1.0, 0.0],
        provenance=SpeakerEmbeddingProvenance(model_id=ENROLLMENT_MODEL, model_commit_sha=ENROLLMENT_SHA),
    )


@pytest.fixture(autouse=True)
def _no_model_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    """One speaker, plausible SQUIM, no PII, and no constructor that would resolve against the Hub.

    Args:
        monkeypatch: The patcher.
    """

    def _one_speaker(audios: list[Audio], model: Any = None, **kw: Any) -> list[list[ScriptLine]]:  # noqa: ANN401
        return [_segments(1, audios[0].waveform.shape[-1] / audios[0].sampling_rate)]

    monkeypatch.setattr(speech_module, "diarize_audios", _one_speaker)
    monkeypatch.setattr(
        speech_module,
        "extract_objective_quality_features_from_audios",
        lambda audios, device=None: [{"stoi": 0.9, "pesq": 3.0, "si_sdr": 18.0} for _ in audios],
    )
    monkeypatch.setattr(
        speech_module, "_diarization_model", lambda: _FakeModel("pyannote/speaker-diarization-community-1")
    )
    monkeypatch.setattr(speech_module, "_second_diarizer_model", lambda model_id: _FakeModel(model_id))
    monkeypatch.setattr(speech_module, "_clearvoice_model", lambda model_id: _FakeModel(model_id))
    monkeypatch.setattr(speech_module, "_embedding_model", lambda model_id, revision: _FakeModel(model_id, revision))
    monkeypatch.setattr(
        speech_module,
        "separate_audios",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("separation must not run")),
    )
    monkeypatch.setattr(
        speech_module,
        "extract_speaker_embeddings_from_audios",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("embedding must not run")),
    )
    monkeypatch.setattr(
        speech_module,
        "scan_for_pii",
        lambda inputs, **kw: [
            PiiScan(spans=[], detectors_used=default_detectors(), failures={})
            for _ in ([inputs] if isinstance(inputs, str) else list(inputs))
        ],
    )


class TestItReadsTheConsensusAndReFusesNothing:
    """PREPROCESS produced the consensus; this branch reads it."""

    def test_the_words_come_from_the_consensus_transcript(
        self, store: ProvStore, speech_config: TriageConfig, tmp_path: Path
    ) -> None:
        """words_n is the count of consensus word entities, not a re-fusion of the hypotheses."""
        _seed_speech_store(store, tmp_path, words=["hello", "world"])
        result = speech(store, "plain", speech_config, run_dir=tmp_path, enrollment=None)
        assert result.verdict.node == "SPEECH"
        verdict = _verdict_entity(store, "SPEECH")
        assert verdict.attributes["words_n"] == 2

    def test_the_module_cannot_re_fuse(self) -> None:
        """A fusion function reachable from this module is the v1 behaviour the spec deleted."""
        assert not hasattr(speech_module, "fuse_word_streams")
        assert not hasattr(speech_module, "fuse_consensus_words")

    def test_an_event_is_not_a_word(self, store: ProvStore, speech_config: TriageConfig, tmp_path: Path) -> None:
        """Bracketed and onomatopoeic events count toward no word total and no span extent."""
        _seed_speech_store(store, tmp_path, words=["hello"], events=["[COUGH]", "[BREATH]"])
        speech(store, "plain", speech_config, run_dir=tmp_path, enrollment=None)
        assert _verdict_entity(store, "SPEECH").attributes["words_n"] == 1

    def test_no_consensus_word_fails_and_writes_no_pii_scan(
        self, store: ProvStore, speech_config: TriageConfig, tmp_path: Path
    ) -> None:
        """redact.md: a wordless recording has no PII scan, no REDACT verdict and no withheld release."""
        _seed_speech_store(store, tmp_path, words=[])
        result = speech(store, "plain", speech_config, run_dir=tmp_path, enrollment=None)
        assert result.verdict.outcome is Outcome.FAIL
        assert find_measurement(store, "pii_scan") is None


class TestTheSecondDiarizerIsConditional:
    """One speaker is the count; anything else consults a second diarizer and reports disagreement."""

    def test_a_count_of_one_consults_nobody(
        self,
        store: ProvStore,
        second_diarizer_config: TriageConfig,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """branch-speech.md: 'No second diarizer runs'."""
        _seed_speech_store(store, tmp_path, words=["hello", "world"])
        calls = _stub_diarizers(monkeypatch, primary_speakers=1, second_speakers=2)
        speech(store, "plain", second_diarizer_config, run_dir=tmp_path, enrollment=None)
        assert calls == ["primary"]
        assert _verdict_entity(store, "SPEECH").attributes["second_diarizer"] == "not_consulted"

    def test_a_count_of_two_consults_the_second(
        self,
        store: ProvStore,
        second_diarizer_config: TriageConfig,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The disagreement is reported; it does not replace pyannote's count."""
        _seed_speech_store(store, tmp_path, words=["hello", "world"])
        calls = _stub_diarizers(monkeypatch, primary_speakers=2, second_speakers=3)
        speech(store, "plain", second_diarizer_config, run_dir=tmp_path, enrollment=None)
        assert calls == ["primary", "second"]
        record = _verdict_entity(store, "SPEECH").attributes["second_diarizer"]
        assert record["count"] == 3 and record["agrees"] is False
        assert _verdict_entity(store, "SPEECH").attributes["speaker_count"] == 2

    def test_a_count_of_zero_consults_the_second_too(
        self,
        store: ProvStore,
        second_diarizer_config: TriageConfig,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """branch-speech.md: 'the codomain is the counts pyannote can return, and 0 is one of them'."""
        _seed_speech_store(store, tmp_path, words=["hello", "world"])
        calls = _stub_diarizers(monkeypatch, primary_speakers=0, second_speakers=1)
        speech(store, "plain", second_diarizer_config, run_dir=tmp_path, enrollment=None)
        assert calls == ["primary", "second"]

    def test_a_declared_count_is_not_read(
        self, store: ProvStore, speech_config: TriageConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """hint.targeted_speaker_count is the protocol's intent, of unknown provenance; not evidence."""
        _seed_speech_store(store, tmp_path, words=["hello", "world"])
        _stub_diarizers(monkeypatch, primary_speakers=1, second_speakers=1)
        hint = AudioHints(targeted_speaker_count=4)
        result = speech(store, "plain", speech_config, hint, run_dir=tmp_path, enrollment=None)
        assert "4" not in result.verdict.why


class TestTheDegenerateIntervalIsAFindingNotACrash:
    """C3: a consensus placing every word at one instant selects no samples to diarize."""

    def test_a_zero_length_interval_is_not_diarized(
        self, store: ProvStore, speech_config: TriageConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The Glides-Low-to-High shape: one word at [0.72, 0.72], and pyannote is never reached."""
        _seed_speech_store(store, tmp_path, words=["Ee"], word_extents=[(0.72, 0.72)])
        calls = _stub_diarizers(monkeypatch, primary_speakers=1, second_speakers=1)
        result = speech(store, "plain", speech_config, run_dir=tmp_path, enrollment=None)
        assert calls == [], "the crop refuses before any model sees a (1, 0) tensor"
        verdict = _verdict_entity(store, "SPEECH")
        assert verdict.attributes["diarization"] == "interval_selects_no_samples"
        assert verdict.attributes["speaker_count"] is None
        assert result.verdict.outcome is Outcome.FLAG

    def test_the_branch_still_writes_the_scan_redact_would_read(
        self, store: ProvStore, speech_config: TriageConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The cluster run lost its PII scan to the crash; a finding costs the branch nothing."""
        _seed_speech_store(store, tmp_path, words=["Ee"], word_extents=[(0.72, 0.72)])
        _stub_diarizers(monkeypatch, primary_speakers=1, second_speakers=1)
        speech(store, "plain", speech_config, run_dir=tmp_path, enrollment=None)
        assert find_measurement(store, "pii_scan") is not None


class TestEnrollment:
    """The target is enrolled. An enrollment without provenance is refused rather than compared."""

    def test_no_enrollment_claims_no_identity(
        self, store: ProvStore, speech_config: TriageConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Speakers stay SPEAKER_*, and nothing is called a target."""
        _seed_speech_store(store, tmp_path, words=["hello", "world"])
        _stub_diarizers(monkeypatch, primary_speakers=1, second_speakers=1)
        speech(store, "plain", speech_config, run_dir=tmp_path, enrollment=None)
        assert "target_speaker" not in _verdict_entity(store, "SPEECH").attributes

    def test_an_enrollment_without_a_commit_is_refused(
        self, store: ProvStore, enrollment_config: TriageConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No embedder runs; the branch flags with the refusal."""
        _seed_speech_store(store, tmp_path, words=["hello", "world"])
        _stub_diarizers(monkeypatch, primary_speakers=1, second_speakers=1)
        embedder = _stub_embedder(monkeypatch)
        result = speech(store, "plain", enrollment_config, run_dir=tmp_path, enrollment=_enrollment(commit=None))
        assert embedder == []
        assert result.verdict.outcome is Outcome.FLAG
        assert "resolved model commit" in result.verdict.why

    def test_an_enrollment_from_another_model_is_refused(
        self, store: ProvStore, enrollment_config: TriageConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A similarity between two models' spaces is not a similarity."""
        _seed_speech_store(store, tmp_path, words=["hello", "world"])
        _stub_diarizers(monkeypatch, primary_speakers=1, second_speakers=1)
        result = speech(
            store,
            "plain",
            enrollment_config,
            run_dir=tmp_path,
            enrollment=_enrollment(model="pyannote/embedding"),
        )
        assert "not the probe" in result.verdict.why

    def test_an_enrollment_at_another_commit_of_the_same_model_is_refused(
        self, store: ProvStore, enrollment_config: TriageConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """branch-speech.md section 6: a matching model id is not enough; the commits must agree."""
        _seed_speech_store(store, tmp_path, words=["hello", "world"])
        _stub_diarizers(monkeypatch, primary_speakers=1, second_speakers=1)
        embedder = _stub_embedder(monkeypatch)
        result = speech(
            store, "plain", enrollment_config, run_dir=tmp_path, enrollment=_enrollment(commit="b" * 40)
        )
        assert embedder == [], "no probe runs against an enrollment it cannot be compared with"
        assert result.verdict.outcome is Outcome.FLAG
        assert "two commits of one model are not comparable" in result.verdict.why
        assert not live_entities(store, "target_match"), "no comparison happened"

    def test_a_null_enrollment_model_key_refuses_before_any_store_write(
        self, store: ProvStore, speech_config: TriageConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """speech.enrollment_model is null on the packaged config; nothing invents a probe."""
        _seed_speech_store(store, tmp_path, words=["hello", "world"])
        _stub_diarizers(monkeypatch, primary_speakers=1, second_speakers=1)
        result = speech(store, "plain", speech_config, run_dir=tmp_path, enrollment=_enrollment())
        assert result.verdict.outcome is Outcome.FLAG
        assert "speech.enrollment_model" in result.verdict.why

    def test_the_enrollment_element_names_every_source(
        self, store: ProvStore, enrollment_config: TriageConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The store carries the enrollment, so a file's own contribution to its target is visible."""
        _seed_speech_store(store, tmp_path, words=["hello", "world"])
        _stub_diarizers(monkeypatch, primary_speakers=1, second_speakers=1)
        _stub_embedder(monkeypatch, similarity=0.99)
        speech(store, "plain", enrollment_config, run_dir=tmp_path, enrollment=_enrollment())
        element = live_entities(store, "enrollment")[0]
        assert element.attributes["subject_id"] == "sub-01"
        assert element.attributes["sources"] == ["a.wav", "b.wav"]
        assert element.attributes["model_commit_sha"] == "a" * 40

    def test_the_probe_is_loaded_at_the_enrolled_commit(
        self, store: ProvStore, enrollment_config: TriageConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A match recorded against an unpinned probe is provenance that is confidently wrong."""
        _seed_speech_store(store, tmp_path, words=["hello", "world"])
        _stub_diarizers(monkeypatch, primary_speakers=1, second_speakers=1)
        _stub_embedder(monkeypatch, similarity=0.99)
        speech(store, "plain", enrollment_config, run_dir=tmp_path, enrollment=_enrollment())
        (match,) = live_entities(store, "target_match")
        assert match.attributes["probe_model"] == ENROLLMENT_MODEL
        assert match.attributes["probe_revision"] == ENROLLMENT_SHA
        assert match.attributes["enrollment_commit"] == ENROLLMENT_SHA

    def test_a_hint_target_speaker_is_not_read_and_says_so(
        self, store: ProvStore, speech_config: TriageConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The ignore is never silent (V15)."""
        _seed_speech_store(store, tmp_path, words=["hello", "world"])
        _stub_diarizers(monkeypatch, primary_speakers=1, second_speakers=1)
        hint = AudioHints(target_speaker=_target_speaker_embedding())
        result = speech(store, "plain", speech_config, hint, run_dir=tmp_path, enrollment=None)
        assert "identifies the target by enrollment" in result.verdict.why


class TestSeparationIsMeasurementGated:
    """Neither backend is selected by default, and the choice is a config key."""

    def test_a_null_backend_does_not_separate(
        self, store: ProvStore, speech_config: TriageConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A count of 2 with no ranked backend records the absence rather than picking one."""
        _seed_speech_store(store, tmp_path, words=["hello", "world"])
        _stub_diarizers(monkeypatch, primary_speakers=2, second_speakers=2)
        separator = _stub_separator(monkeypatch)
        speech(store, "plain", speech_config, run_dir=tmp_path, enrollment=None)
        assert separator == []
        assert _verdict_entity(store, "SPEECH").attributes["separation"] == "not_selected"

    def test_mossformer_is_reachable_by_config(
        self, store: ProvStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The alternative runs when named, at n_sources 2, and writes one stream per source."""
        config = _override(tmp_path, "speech:\n  separation_backend: MossFormer2_SS_16K\n")
        _seed_speech_store(store, tmp_path, words=["hello", "world"])
        _stub_diarizers(monkeypatch, primary_speakers=2, second_speakers=2)
        separator = _stub_separator(monkeypatch, sources=2)
        speech(store, "plain", config, run_dir=tmp_path, enrollment=None)
        assert separator[0]["model"] == "alibabasglab/MossFormer2_SS_16K"
        assert separator[0]["n_sources"] == 2
        assert len([e for e in live_entities(store, "stream") if e.attributes["name"].startswith("separated")]) == 2

    def test_unasdiff_speech_sound_needs_a_sound_class(
        self, store: ProvStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """V17: the spec wants an unconditioned sound slot; the API refuses one. The branch says so."""
        config = _override(tmp_path, "speech:\n  separation_backend: unasdiff\n")
        _seed_speech_store(store, tmp_path, words=["hello", "world"])
        _stub_diarizers(monkeypatch, primary_speakers=2, second_speakers=2)
        separator = _stub_separator(monkeypatch)
        speech(store, "plain", config, run_dir=tmp_path, enrollment=None)
        assert separator == []
        assert _verdict_entity(store, "SPEECH").attributes["separation"] == "unconditioned_sound_slot_unavailable"

    def test_unasdiff_runs_in_speech_sound_mode_when_a_class_is_named(
        self, store: ProvStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Slot 0 is the speech prior; the sound slot carries the configured class."""
        config = _override(tmp_path, "speech:\n  separation_backend: unasdiff\n  separation_sound_class: Applause\n")
        _seed_speech_store(store, tmp_path, words=["hello", "world"])
        _stub_diarizers(monkeypatch, primary_speakers=2, second_speakers=2)
        separator = _stub_separator(monkeypatch, sources=2)
        speech(store, "plain", config, run_dir=tmp_path, enrollment=None)
        assert separator[0]["mode"] == "speech_sound"
        assert separator[0]["source_classes"] == ["Applause"]

    def test_three_speakers_are_reported_not_separated(
        self, store: ProvStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """MossFormer fixes n_sources at 2, so a count of 3 is a report, not a wrong decomposition."""
        config = _override(tmp_path, "speech:\n  separation_backend: MossFormer2_SS_16K\n")
        _seed_speech_store(store, tmp_path, words=["hello", "world"])
        _stub_diarizers(monkeypatch, primary_speakers=3, second_speakers=3)
        separator = _stub_separator(monkeypatch)
        result = speech(store, "plain", config, run_dir=tmp_path, enrollment=None)
        assert separator == []
        assert "cannot serve 3" in result.verdict.why


class TestPiiOnTheConsensus:
    """One scan, one text, and the decision is speaker-scoped while the redaction is not."""

    def test_the_scan_reads_the_consensus_transcript_only(
        self, store: ProvStore, speech_config: TriageConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Exactly one text is scanned, and it is the consensus text PREPROCESS wrote."""
        _seed_speech_store(store, tmp_path, words=["my", "name", "is", "alice"])
        _stub_diarizers(monkeypatch, primary_speakers=1, second_speakers=1)
        scanned = _stub_pii(monkeypatch, findings=[("PERSON", "alice")])
        speech(store, "plain", speech_config, run_dir=tmp_path, enrollment=None)
        assert scanned == ["my name is alice"]

    def test_a_finding_carries_category_and_extent_never_text(
        self, store: ProvStore, speech_config: TriageConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The verdict and the element both refuse to carry the matched text."""
        _seed_speech_store(store, tmp_path, words=["my", "name", "is", "alice"])
        _stub_diarizers(monkeypatch, primary_speakers=1, second_speakers=1)
        _stub_pii(monkeypatch, findings=[("PERSON", "alice")])
        result = speech(store, "plain", speech_config, run_dir=tmp_path, enrollment=None)
        finding = live_entities(store, "pii")[0]
        assert finding.attributes["category"] == "PERSON"
        assert finding.extent is not None
        assert "alice" not in str(finding.attributes)
        assert "alice" not in result.verdict.why

    def test_a_finding_names_the_recognizers_behind_the_words_it_rests_on(
        self, store: ProvStore, speech_config: TriageConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A finding resting on one recognizer alone must be legible as such."""
        _seed_speech_store(store, tmp_path, words=["my", "name", "is", "alice"])
        _stub_diarizers(monkeypatch, primary_speakers=1, second_speakers=1)
        _stub_pii(monkeypatch, findings=[("PERSON", "alice")])
        speech(store, "plain", speech_config, run_dir=tmp_path, enrollment=None)
        finding = live_entities(store, "pii")[0]
        assert len(finding.attributes["recognizers"]) == 2

    def test_a_finding_marks_the_word_elements(
        self, store: ProvStore, speech_config: TriageConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The store now holds PII, and every artifact must respect the marking."""
        _seed_speech_store(store, tmp_path, words=["my", "name", "is", "alice"])
        _stub_diarizers(monkeypatch, primary_speakers=1, second_speakers=1)
        _stub_pii(monkeypatch, findings=[("PERSON", "alice")])
        speech(store, "plain", speech_config, run_dir=tmp_path, enrollment=None)
        marks = [
            e
            for e in live_entities(store, "assertion")
            if e.attributes.get("verb") == "label" and e.attributes.get("label") == "pii"
        ]
        assert marks and all("alice" not in str(mark.attributes) for mark in marks)

    def test_a_missing_required_detector_flags(
        self, store: ProvStore, speech_config: TriageConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A detector never attempted is the silent one, and could-not-check is not clean."""
        _seed_speech_store(store, tmp_path, words=["hello"])
        _stub_diarizers(monkeypatch, primary_speakers=1, second_speakers=1)
        _stub_pii(monkeypatch, findings=[], detectors_used=["rules"])
        result = speech(store, "plain", speech_config, run_dir=tmp_path, enrollment=None)
        assert result.verdict.outcome is Outcome.FLAG
        scan = find_measurement(store, "pii_scan")
        assert scan is not None and scan.attributes["missing"] == ["gliner", "presidio"]

    def test_a_non_target_finding_does_not_flag_but_is_still_a_finding(
        self, store: ProvStore, enrollment_config: TriageConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Flagging asks whether a human is needed; the finding still reaches REDACT.

        The layout puts the first two words in SPEAKER_00's half of the interval and the last two,
        'alice' among them, in SPEAKER_01's — and SPEAKER_00 is the speaker the enrollment matches.
        """
        _seed_speech_store(store, tmp_path, words=["my", "name", "is", "alice"], speakers=2)
        _stub_diarizers(monkeypatch, primary_speakers=2, second_speakers=2)
        _stub_embedder(monkeypatch, similarity=0.99, target_label="SPEAKER_00")
        _stub_pii(monkeypatch, findings=[("PERSON", "alice")])
        speech(store, "plain", enrollment_config, run_dir=tmp_path, enrollment=_enrollment())
        verdict = _verdict_entity(store, "SPEECH")
        assert verdict.attributes["pii"]["n"] == 1
        assert not [flag for flag in verdict.attributes["flags"] if "target speaker's speech" in flag]


class TestTheNonTargetAxis:
    """Measured and reported per span; null, not zero, while the thresholds are unmeasured."""

    def test_the_three_legs_are_measured_per_span(
        self, store: ProvStore, speech_config: TriageConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Level, spectral tilt and direct-to-reverberant, on every speech span."""
        _seed_speech_store(store, tmp_path, words=["hello", "world"])
        _stub_diarizers(monkeypatch, primary_speakers=1, second_speakers=1)
        speech(store, "plain", speech_config, run_dir=tmp_path, enrollment=None)
        proximity = find_measurements(store, "proximity")
        assert proximity
        for measurement in proximity:
            assert {"rms_dbfs", "peak_dbfs", "tilt_db_per_octave", "d_to_r_db"} <= set(measurement.attributes)

    def test_nontarget_speech_s_is_null_while_a_threshold_is_unmeasured(
        self, store: ProvStore, speech_config: TriageConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A product that says zero when nobody measured is the failure this row exists to prevent."""
        _seed_speech_store(store, tmp_path, words=["hello", "world"])
        _stub_diarizers(monkeypatch, primary_speakers=1, second_speakers=1)
        speech(store, "plain", speech_config, run_dir=tmp_path, enrollment=None)
        assert _verdict_entity(store, "SPEECH").attributes["nontarget_speech_s"] is None

    def test_the_product_appears_once_every_threshold_is_supplied(
        self, store: ProvStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The null is a gate, not a hard-coded None: supplying the three cuts folds the legs."""
        config = _override(
            tmp_path,
            "speech:\n  nontarget:\n    level_db: 0.0\n    tilt_db_per_octave: 0.0\n    d_to_r_db: 0.0\n",
        )
        _seed_speech_store(store, tmp_path, words=["hello", "world"])
        _stub_diarizers(monkeypatch, primary_speakers=1, second_speakers=1)
        speech(store, "plain", config, run_dir=tmp_path, enrollment=None)
        assert isinstance(_verdict_entity(store, "SPEECH").attributes["nontarget_speech_s"], float)

    def test_no_span_is_excluded_on_this_evidence(
        self, store: ProvStore, speech_config: TriageConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """This branch marks; it removes nothing."""
        _seed_speech_store(store, tmp_path, words=["hello", "world"])
        _stub_diarizers(monkeypatch, primary_speakers=1, second_speakers=1)
        speech(store, "plain", speech_config, run_dir=tmp_path, enrollment=None)
        assert not [e for e in store.entities("span") if store.is_invalidated(e.id)]


class TestQualityAndTheStreamsItNames:
    """SQUIM on plain, disruptions on the original, and every reading names its stream (V19)."""

    def test_disruptions_read_the_original_recording(
        self, store: ProvStore, speech_config: TriageConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Peak normalisation and resampling destroy the plateaus and the crossing rate."""
        _seed_speech_store(store, tmp_path, words=["hello", "world"])
        _stub_diarizers(monkeypatch, primary_speakers=1, second_speakers=1)
        speech(store, "plain", speech_config, run_dir=tmp_path, enrollment=None)
        recording_id = _stream_id(store, "recording")
        readings = find_measurements(store, "disruptions")
        assert readings
        for measurement in readings:
            assert measurement.attributes["stream"] == recording_id

    def test_a_wordless_file_has_no_per_span_reading_and_that_is_correct(
        self, store: ProvStore, speech_config: TriageConfig, tmp_path: Path
    ) -> None:
        """A span nobody measured must not report zero; the file-level reading is PREPROCESS's."""
        _seed_speech_store(store, tmp_path, words=[], disruptions_file=True)
        speech(store, "plain", speech_config, run_dir=tmp_path, enrollment=None)
        assert find_measurements(store, "disruptions") == []
        assert find_measurement(store, "disruptions_file") is not None


class TestItDoesNotReadAirway:
    """Diarization is a speech-only instrument."""

    def test_an_airway_label_withdraws_no_segment(
        self, store: ProvStore, speech_config: TriageConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The same store with and without AIRWAY's labels yields the same speaker count."""
        _seed_speech_store(store, tmp_path, words=["hello", "world"], airway_labelled=[(0.4, 0.6)])
        _stub_diarizers(monkeypatch, primary_speakers=1, second_speakers=1)
        speech(store, "plain", speech_config, run_dir=tmp_path, enrollment=None)
        assert _verdict_entity(store, "SPEECH").attributes["speaker_count"] == 1
        assert not [e for e in store.entities("speaker") if store.is_invalidated(e.id)]

    def test_the_module_reads_no_airway_activity(self) -> None:
        """Verifying what commit 8537a83f already removed, so a regression is caught here."""
        source = Path(speech_module.__file__).read_text()
        assert "AIRWAY" not in source
