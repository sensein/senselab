"""The SPEECH -> REDACT interlock, run over one store by both nodes.

Everything else in ``redact_test.py`` seeds the marking REDACT reads. This module does not: it runs
the **real** ``speech()`` and then reads the store back with REDACT's own reader, so the two nodes'
agreement about how a PII marking is written is a measurement rather than a convention. The
marking is a live ``assertion`` entity carrying ``verb="label"`` and ``label="pii"``,
``wasDerivedFrom`` the ``word`` it is about; if SPEECH ever writes it another way, REDACT's
re-planning pass widens to nothing and does so **silently** — ``replanned_n`` still reads 1 and every
seeded test still passes. This is the test that fails instead.

It also pins single authorship of ``word`` entities. Words written by two nodes at two offsets once
put a duplicate of a name outside every planned extent and into a released transcript; PREPROCESS is
now the only author, and the fourth assertion below is the regression guard for that.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
import pytest

from senselab.audio.data_structures import Audio
from senselab.audio.workflows.audio_analysis.level import integrated_lufs
from senselab.audio.workflows.triage.config import TriageConfig, load_triage_config
from senselab.audio.workflows.triage.nodes import redact as redact_module
from senselab.audio.workflows.triage.nodes import speech as speech_module
from senselab.audio.workflows.triage.nodes.common import live_entities
from senselab.audio.workflows.triage.nodes.redact import _pii_marked_words, redact
from senselab.audio.workflows.triage.nodes.speech import speech
from senselab.audio.workflows.triage.vocabulary import Outcome
from senselab.text.tasks.pii_detection.api import PiiScan, PiiSpan, default_detectors
from senselab.utils.data_structures import ScriptLine
from senselab.utils.prov_store import ProvStore

SENTINEL = "alicia"  # the name the scan finds; nothing released may carry it
WORDS = ["my", "name", "is", SENTINEL, "and", "i", "live", "here"]
DURATION_S = 5.0


class _FakeModel:
    """A model spec stub carrying exactly what SPEECH reads: path_or_uri and commit_sha."""

    def __init__(self, path_or_uri: str, revision: str = "main") -> None:
        """Stub a resolved model.

        Args:
            path_or_uri: The model id.
            revision: What was asked for.
        """
        self.path_or_uri = path_or_uri
        self.revision = revision
        self.commit_sha = "c" * 40


def _config(tmp_path: Path) -> TriageConfig:
    """The three keys this interlock needs, none of which has a packaged default.

    Args:
        tmp_path: Where the override file is written.

    Returns:
        The resolved configuration.
    """
    path = tmp_path / "interlock-override.yaml"
    path.write_text("redaction:\n  padding_ms: 50\n  fill: silence\n")
    return load_triage_config(path)


def _release(tmp_path: Path) -> Path:
    """A release directory disjoint from the run directory, which is ``tmp_path`` itself.

    Args:
        tmp_path: The run directory.

    Returns:
        The release directory.
    """
    return tmp_path.parent / f"{tmp_path.name}-release"


def _place(words: list[str]) -> list[tuple[str, tuple[float, float]]]:
    """One word per 0.4 s slot, well inside the stream, so every word is its own extent.

    Args:
        words: The word texts, in order.

    Returns:
        ``[(text, (start, end)), ...]``.
    """
    return [(text, (0.5 + 0.4 * index, 0.5 + 0.4 * index + 0.25)) for index, text in enumerate(words)]


def _seed_level(store: ProvStore, tmp_path: Path) -> None:
    """PREPROCESS's file-level reading, which SPEECH's proximity leg measures each span against.

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


@pytest.fixture(autouse=True)
def _no_model_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    """One speaker, plausible SQUIM, and no constructor that would resolve against the Hub.

    The PII scan is left alone here; each fixture stubs the finding it wants.

    Args:
        monkeypatch: The patcher.
    """
    monkeypatch.setattr(
        speech_module,
        "diarize_audios",
        lambda audios, model=None, **kw: [
            [ScriptLine(speaker="SPEAKER_00", start=0.0, end=audios[0].waveform.shape[-1] / audios[0].sampling_rate)]
        ],
    )
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


def _stub_speech_pii(monkeypatch: pytest.MonkeyPatch, findings: list[tuple[str, str]]) -> list[str]:
    """Fake SPEECH's scanner with one fixed answer, recording the texts it was handed.

    Args:
        monkeypatch: The patcher.
        findings: ``(category, matched text)`` pairs the scan reports.

    Returns:
        The mutable log of scanned texts.
    """
    scanned: list[str] = []

    def _fake(inputs: Any, **kw: Any) -> list[PiiScan]:  # noqa: ANN401
        texts = [inputs] if isinstance(inputs, str) else list(inputs)
        scanned.extend(str(text) for text in texts)
        return [
            PiiScan(
                spans=[
                    PiiSpan(text=text, category=category, source="presidio", asr_model="consensus_transcript")
                    for category, text in findings
                ],
                detectors_used=default_detectors(),
                failures={},
            )
            for _ in texts
        ]

    monkeypatch.setattr(speech_module, "scan_for_pii", _fake)
    return scanned


def _stub_redact_pii_clean(monkeypatch: pytest.MonkeyPatch) -> None:
    """REDACT's re-scan finds nothing, so the interlock is about the plan rather than the verdict.

    Args:
        monkeypatch: The patcher.
    """
    monkeypatch.setattr(
        redact_module,
        "scan_for_pii",
        lambda inputs, **kw: PiiScan(spans=[], detectors_used=default_detectors(), failures={}),
    )


@pytest.fixture(name="spoken_store")
def _spoken_store(
    store: ProvStore,
    tmp_path: Path,
    seed_preprocess_store: Callable[..., None],
    monkeypatch: pytest.MonkeyPatch,
) -> ProvStore:
    """PREPROCESS's store with the real SPEECH branch run over it, its scan finding one PERSON.

    Args:
        store: The empty store.
        tmp_path: The run directory.
        seed_preprocess_store: T1's shared seeder.
        monkeypatch: The patcher.

    Returns:
        The store SPEECH left behind.
    """
    seed_preprocess_store(store, duration_s=DURATION_S, words=_place(WORDS))
    _seed_level(store, tmp_path)
    _stub_speech_pii(monkeypatch, [("PERSON", SENTINEL)])
    speech(store, "plain", _config(tmp_path), run_dir=tmp_path)
    return store


class TestTheMarkingSpeechWritesIsTheMarkingRedactReads:
    """One store, written by SPEECH and read back by REDACT's own reader."""

    def test_the_marking_is_non_empty_and_keyed_by_word_entity_id(self, spoken_store: ProvStore) -> None:
        """An empty mapping is how a drifted marking shape fails, and it fails silently."""
        marked = _pii_marked_words(spoken_store)
        assert marked, "REDACT's reader found no pii marking in a store whose scan found a PERSON"
        word_ids = {word.id for word in spoken_store.entities("word")}
        assert set(marked) <= word_ids, "the marking must key on word entities, not on spans or findings"

    def test_the_category_survives_the_handoff(self, spoken_store: ProvStore) -> None:
        """REDACT re-plans by category, so the category is the half of the marking that must match."""
        marked = _pii_marked_words(spoken_store)
        assert "PERSON" in {category for categories in marked.values() for category in categories}

    def test_the_marked_word_is_the_one_the_scan_matched(self, spoken_store: ProvStore) -> None:
        """A marking on the wrong word would widen the re-plan to the wrong extent."""
        marked = _pii_marked_words(spoken_store)
        texts = {
            str(spoken_store.get_entity(word_id).attributes.get("text")): sorted(categories)
            for word_id, categories in marked.items()
        }
        assert texts == {SENTINEL: ["PERSON"]}

    def test_every_word_entity_leads_back_to_one_preprocess_write(self, spoken_store: ProvStore) -> None:
        """Two authors at two offsets once put a duplicate of a name into a released transcript.

        SPEECH has run for real here, and it authored no word: every live ``word`` still leads back
        to the single PREPROCESS activity that wrote them. A second author shows up as a second
        node; a second write site within one node shows up as a second activity id. The step name
        is the seeder's, so it is not asserted — the node and the write count are what F1 was about.
        """
        activities = set()
        for word in live_entities(spoken_store, "word"):
            activity_id = spoken_store.generated_by(word.id)
            assert activity_id is not None, "a word with no generating activity has no author at all"
            activities.add(activity_id)
        assert {spoken_store.get_activity(a).node for a in activities} == {"PREPROCESS"}
        assert len(activities) == 1, "one write site, so no word can be a duplicate of another at an offset"

    def test_speech_authored_no_word_entity_of_its_own(self, spoken_store: ProvStore) -> None:
        """The same claim from the other side, in the form the regression would have taken."""
        speech_activities = {a.id for a in spoken_store.activities("SPEECH")}
        assert not [
            word for word in spoken_store.entities("word") if spoken_store.generated_by(word.id) in speech_activities
        ]

    def test_the_speech_scan_read_the_consensus_text(self, spoken_store: ProvStore) -> None:
        """The control for the fixture: the branch scanned the transcript, not something else."""
        scan = [e for e in live_entities(spoken_store, "measurement") if e.attributes.get("name") == "pii_scan"][-1]
        assert scan.attributes["signal"] == "consensus_transcript"
        assert scan.attributes["failed"] == [] and scan.attributes["missing"] == []


class TestRedactRunsOverSpeechsOwnStore:
    """The end of the interlock: the store SPEECH wrote is the store REDACT releases from."""

    def test_the_released_transcript_carries_no_sentinel(
        self, spoken_store: ProvStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No seeder stands between the two nodes; the plan is made off SPEECH's own findings."""
        _stub_redact_pii_clean(monkeypatch)
        result = redact(
            spoken_store, "recording", _config(tmp_path), run_dir=tmp_path, artifacts_dir=_release(tmp_path)
        )
        assert result.verdict.outcome is Outcome.PASS
        text = result.artifacts["transcript"].read_text()
        assert SENTINEL not in text
        assert text.split() == ["my", "name", "is", "[PERSON]", "and", "i", "live", "here"]

    def test_no_released_artifact_carries_the_sentinel_or_a_store_id(
        self, spoken_store: ProvStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An id indexing both the store and a released artifact is a join key back to the PII."""
        _stub_redact_pii_clean(monkeypatch)
        result = redact(
            spoken_store, "recording", _config(tmp_path), run_dir=tmp_path, artifacts_dir=_release(tmp_path)
        )
        ids = [e.id for e in spoken_store.entities()]
        for path in result.artifacts.values():
            blob = path.read_bytes()
            assert SENTINEL.encode() not in blob
            for entity_id in ids:
                assert entity_id.encode() not in blob
