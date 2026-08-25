"""SPEECH node tests. Every model call is faked at the node module; DSP and the store run real."""

import json
import math
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

import numpy as np
import pytest
import soundfile as sf
import torch
import yaml
from scipy.signal import lfilter

from senselab.audio.data_structures import (
    Audio,
    AudioHints,
    ExpectedSpeech,
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
    yamnet_labels: Optional[list[list[str]]] = None,
    spans: Optional[list[tuple[float, float, float]]] = None,
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
        yamnet_labels: One retained label set per YAMNet window, on the shared seeder's grid.
            ``None`` writes no classification at all.
        spans: PREPROCESS's envelope spans, ``[(start, end, peak_over_floor_db), ...]``, which a
            SPEECH span refines where the two overlap.
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
        yamnet_labels=yamnet_labels,
        spans=spans,
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


_NEAR_EXTENTS = [(0.5, 0.8), (0.9, 1.2)]  # one span, 0.5-1.2 s
_FAR_EXTENTS = [(2.0, 2.4), (2.6, 3.1)]  # one span, 2.0-3.1 s -- a different length, deliberately


def _seed_two_distance_spans(store: ProvStore, tmp_path: Path, duration_s: float = 5.0) -> None:
    """Seed two speech spans a metre apart on every proximity leg, over a real waveform.

    The near span is loud broadband noise: high RMS, a flat spectrum, and an autocorrelation that is
    nearly a delta, so its direct-to-reverberant ratio is high. The far span is the same noise, made
    quiet and run through a one-pole low pass: low RMS, a spectrum falling about 6 dB per octave,
    and a smeared autocorrelation whose tail carries most of the energy, so its ratio is low. Every
    leg puts the far span behind the near one, which is what lets a test pin each comparison's
    direction rather than its threshold.

    Args:
        store: The store to seed.
        tmp_path: The run directory.
        duration_s: The streams' duration.
    """
    _seed_speech_store(
        store,
        tmp_path,
        words=["near", "one", "far", "two"],
        word_extents=[*_NEAR_EXTENTS, *_FAR_EXTENTS],
        duration_s=duration_s,
    )
    plain = [e for e in live_entities(store, "stream") if e.attributes.get("name") == "plain"][-1]
    rate = int(plain.attributes["sampling_rate"])
    samples = np.zeros(int(duration_s * rate), dtype=np.float32)
    rng = np.random.default_rng(0)
    near = slice(int(0.5 * rate), int(1.2 * rate))
    samples[near] = (0.4 * rng.standard_normal(near.stop - near.start)).astype(np.float32)
    far = slice(int(2.0 * rate), int(3.1 * rate))
    low_passed = lfilter([1 - 0.99], [1, -0.99], rng.standard_normal(far.stop - far.start))
    samples[far] = (0.01 * low_passed / np.abs(low_passed).max()).astype(np.float32)
    sf.write(str(tmp_path / plain.attributes["path"]), samples, rate)
    _seed_level(store, tmp_path)


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


class TestTheUnmeasuredKeyAndTheHintThatContradictsTheFile:
    """F8i, F8j: an unmeasured key is a configuration error, and a contradicted hint outranks a fail."""

    def test_the_packaged_config_refuses_and_leaves_the_store_untouched(self, store: ProvStore, tmp_path: Path) -> None:
        """speech.word_gap_ms ships null, so an ordinary run raises before it measures anything.

        This is not a finding about the recording, so it is not demoted to a flag: nobody has
        derived the key, and a branch that guessed one would report a span count it invented.
        """
        _seed_speech_store(store, tmp_path, words=["hello", "world"])
        before = store.fingerprint()
        with pytest.raises(ValueError, match="speech.word_gap_ms"):
            speech(store, "plain", load_triage_config(), run_dir=tmp_path, enrollment=None)
        assert store.fingerprint() == before, "an unmeasured key must leave the store untouched"

    def test_a_hint_asserting_speech_this_branch_did_not_find_flags(
        self, store: ProvStore, speech_config: TriageConfig, tmp_path: Path
    ) -> None:
        """A wordless recording fails; a wordless recording the caller said held speech flags."""
        _seed_speech_store(store, tmp_path, words=[])
        hint = AudioHints(expected_speech=[ExpectedSpeech(text="the rainbow passage")])
        result = speech(store, "plain", speech_config, hint, run_dir=tmp_path, enrollment=None)
        assert result.verdict.outcome is Outcome.FLAG
        assert "a hint asserts speech not found" in result.verdict.why

    def test_a_hint_tag_asserts_speech_the_same_way(
        self, store: ProvStore, speech_config: TriageConfig, tmp_path: Path
    ) -> None:
        """speech.hint_tags is the vocabulary, so a tagged recording contradicts a fail too."""
        _seed_speech_store(store, tmp_path, words=[])
        result = speech(
            store, "plain", speech_config, AudioHints(may_contain=["Read-Speech"]), run_dir=tmp_path, enrollment=None
        )
        assert result.verdict.outcome is Outcome.FLAG

    def test_a_wordless_recording_nobody_claimed_held_speech_simply_fails(
        self, store: ProvStore, speech_config: TriageConfig, tmp_path: Path
    ) -> None:
        """The control: fail means this branch has no subject, and a cough recording is not an error."""
        _seed_speech_store(store, tmp_path, words=[])
        result = speech(store, "plain", speech_config, AudioHints(), run_dir=tmp_path, enrollment=None)
        assert result.verdict.outcome is Outcome.FAIL


class TestTheSpeechFamilyIsAConfigKey:
    """F4: ``taxonomy.speech_labels`` is null and owed the AudioSet speech FAMILY, not one member."""

    def test_a_null_family_makes_the_vote_inert_and_records_that_it_is(
        self, store: ProvStore, speech_config: TriageConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No family is not a family of one: nothing can be disconfirmed against an unmeasured set."""
        _seed_speech_store(store, tmp_path, words=["hello", "world"], yamnet_labels=[["Music"]] * 11)
        _stub_diarizers(monkeypatch, primary_speakers=1, second_speakers=1)
        speech(store, "plain", speech_config, run_dir=tmp_path, enrollment=None)
        spans = [e for e in live_entities(store, "span") if e.attributes.get("family") == "speech"]
        assert spans and all(span.attributes["yamnet_vote"] == "unavailable" for span in spans)
        assert all(span.attributes["yamnet_coverage"] is None for span in spans)
        flags = _verdict_entity(store, "SPEECH").attributes["flags"]
        assert not [flag for flag in flags if "disconfirm" in flag], "an unmeasured family disconfirms nothing"

    def test_the_family_the_config_names_is_the_family_that_votes(
        self, store: ProvStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A window carrying any member of the configured family confirms the span."""
        config = _override(tmp_path, "taxonomy:\n  speech_labels: [Speech, 'Narration, monologue']\n")
        _seed_speech_store(store, tmp_path, words=["hello", "world"], yamnet_labels=[["Narration, monologue"]] * 11)
        _stub_diarizers(monkeypatch, primary_speakers=1, second_speakers=1)
        speech(store, "plain", config, run_dir=tmp_path, enrollment=None)
        spans = [e for e in live_entities(store, "span") if e.attributes.get("family") == "speech"]
        assert spans and all(span.attributes["yamnet_vote"] == "confirm" for span in spans)
        assert all(span.attributes["yamnet_coverage"] == 1.0 for span in spans)

    def test_a_window_outside_the_family_disconfirms_and_flags(
        self, store: ProvStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The other half: coverage below the threshold is a flag carrying the measure (F8l)."""
        config = _override(tmp_path, "taxonomy:\n  speech_labels: [Speech, 'Narration, monologue']\n")
        _seed_speech_store(store, tmp_path, words=["hello", "world"], yamnet_labels=[["Music"]] * 11)
        _stub_diarizers(monkeypatch, primary_speakers=1, second_speakers=1)
        result = speech(store, "plain", config, run_dir=tmp_path, enrollment=None)
        spans = [e for e in live_entities(store, "span") if e.attributes.get("family") == "speech"]
        assert spans and all(span.attributes["yamnet_vote"] == "disconfirm" for span in spans)
        assert all(span.attributes["yamnet_coverage"] == 0.0 for span in spans)
        assert result.verdict.outcome is Outcome.FLAG
        assert [flag for flag in _verdict_entity(store, "SPEECH").attributes["flags"] if "speech coverage" in flag]

    def test_a_span_no_window_overlaps_is_not_evaluated_rather_than_disconfirmed(
        self, store: ProvStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A family exists and the classifier saw nothing here; that is not evidence against."""
        config = _override(tmp_path, "taxonomy:\n  speech_labels: [Speech]\n")
        _seed_speech_store(store, tmp_path, words=["hello", "world"], yamnet_labels=[])
        _stub_diarizers(monkeypatch, primary_speakers=1, second_speakers=1)
        speech(store, "plain", config, run_dir=tmp_path, enrollment=None)
        spans = [e for e in live_entities(store, "span") if e.attributes.get("family") == "speech"]
        assert spans and all(span.attributes["yamnet_vote"] == "not_evaluated" for span in spans)

    def test_the_activity_records_the_family_it_voted_with(
        self, store: ProvStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A vote whose label set is not recorded cannot be re-read against a later family."""
        config = _override(tmp_path, "taxonomy:\n  speech_labels: [Speech, 'Narration, monologue']\n")
        _seed_speech_store(store, tmp_path, words=["hello", "world"], yamnet_labels=[["Speech"]] * 11)
        _stub_diarizers(monkeypatch, primary_speakers=1, second_speakers=1)
        speech(store, "plain", config, run_dir=tmp_path, enrollment=None)
        (corroborate,) = [a for a in store.activities("SPEECH") if a.step == "corroborate"]
        assert corroborate.parameters["speech_labels"] == ["Narration, monologue", "Speech"]


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


class TestTheDiarizersClockIsTheRecordings:
    """F8a: pyannote sees a crop, and every segment it returns has to be put back where it came from."""

    def test_the_cropped_window_is_the_interval_not_the_file(
        self, store: ProvStore, speech_config: TriageConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Restricting the interval is what keeps non-speech events out of the speaker count."""
        seen: dict[str, float] = {}

        def _fake(audios: list[Audio], model: Any = None, **kw: Any) -> list[list[ScriptLine]]:  # noqa: ANN401
            seen["duration_s"] = audios[0].waveform.shape[-1] / audios[0].sampling_rate
            return [_segments(1, seen["duration_s"])]

        _seed_speech_store(store, tmp_path, words=["one", "two"], word_extents=[(2.0, 2.3), (2.4, 2.8)])
        monkeypatch.setattr(speech_module, "diarize_audios", _fake)
        speech(store, "plain", speech_config, run_dir=tmp_path, enrollment=None)
        assert seen["duration_s"] == pytest.approx(0.8, abs=1 / SR), "cropped to the interval, not the file"

    def test_a_segment_is_offset_back_onto_the_recordings_clock(
        self, store: ProvStore, speech_config: TriageConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The diarizer answers on the crop's clock; a stored segment that keeps it is 2 s early."""
        _seed_speech_store(store, tmp_path, words=["one", "two"], word_extents=[(2.0, 2.3), (2.4, 2.8)])
        _stub_diarizers(monkeypatch, primary_speakers=1, second_speakers=1)
        speech(store, "plain", speech_config, run_dir=tmp_path, enrollment=None)
        (segment,) = live_entities(store, "speaker")
        assert segment.extent is not None
        assert segment.extent[0] == pytest.approx(2.0, abs=1 / SR)
        assert segment.extent[1] == pytest.approx(2.8, abs=1 / SR)

    def test_a_word_is_attributed_through_the_offset_segment(
        self, store: ProvStore, speech_config: TriageConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The offset is load-bearing downstream: un-offset segments overlap no word at all."""
        _seed_speech_store(store, tmp_path, words=["one", "two"], word_extents=[(2.0, 2.3), (2.4, 2.8)])
        _stub_diarizers(monkeypatch, primary_speakers=1, second_speakers=1)
        speech(store, "plain", speech_config, run_dir=tmp_path, enrollment=None)
        attributions = [e for e in live_entities(store, "assertion") if e.attributes.get("verb") == "attribute"]
        assert attributions and all(e.attributes["speaker"] == "SPEAKER_00" for e in attributions)
        assert all(e.attributes["note"] is None for e in attributions), "no word is left unassigned"

    def test_a_word_straddling_a_boundary_is_marked_not_assigned(
        self, store: ProvStore, speech_config: TriageConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """F8h: a word overlapping two segments belongs to neither, and the note says which case it is."""
        _seed_speech_store(store, tmp_path, words=["one"], word_extents=[(1.0, 1.4)])
        _stub_diarizers(monkeypatch, primary_speakers=2, second_speakers=2)
        speech(store, "plain", speech_config, run_dir=tmp_path, enrollment=None)
        (attribution,) = [e for e in live_entities(store, "assertion") if e.attributes.get("verb") == "attribute"]
        assert attribution.attributes["speaker"] is None
        assert attribution.attributes["note"] == "straddles"


class TestTheClampTolerance:
    """F8d: one sample period is a numerical identity; a tenth of a second is an inconsistency."""

    def test_a_word_ending_a_hair_past_the_decode_is_clamped_not_a_crash(
        self, store: ProvStore, speech_config: TriageConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The cluster's own case: 92137 samples at 16 kHz, a last word rounded to 5.7586.

        That is 0.6 of a sample past the decode, and ``extract_segments`` raised "End must be <=
        duration of the audio (5.7585625 sec)" on it while the same file ran clean on the Mac.
        """
        duration_s = 92137 / SR
        end = round(duration_s, 4)
        assert end > duration_s, "the fixture must reproduce the overshoot, not merely resemble it"
        _seed_speech_store(
            store, tmp_path, words=["one", "two"], word_extents=[(1.0, 1.3), (5.0, end)], duration_s=duration_s
        )
        _stub_diarizers(monkeypatch, primary_speakers=1, second_speakers=1)
        result = speech(store, "plain", speech_config, run_dir=tmp_path, enrollment=None)
        assert result.verdict.outcome in (Outcome.PASS, Outcome.FLAG)
        (interval,) = [e for e in live_entities(store, "interval") if e.attributes["name"] == "diarization_interval"]
        assert interval.extent is not None
        assert interval.extent[1] == duration_s, "the interval the store records is the one that was diarized"

    def test_a_word_ending_a_tenth_of_a_second_past_the_decode_still_raises(
        self, store: ProvStore, speech_config: TriageConfig, tmp_path: Path
    ) -> None:
        """An extent that far outside the recording is an inconsistency, not rounding."""
        _seed_speech_store(store, tmp_path, words=["one", "two"], word_extents=[(1.0, 1.3), (5.0, 5.6)], duration_s=5.5)
        with pytest.raises(ValueError, match="past the"):
            speech(store, "plain", speech_config, run_dir=tmp_path, enrollment=None)


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
        result = speech(store, "plain", enrollment_config, run_dir=tmp_path, enrollment=_enrollment(commit="b" * 40))
        assert embedder == [], "no probe runs against an enrollment it cannot be compared with"
        assert result.verdict.outcome is Outcome.FLAG
        assert "two commits of one model are not comparable" in result.verdict.why
        assert not live_entities(store, "target_match"), "no comparison happened"

    def test_a_null_enrollment_model_key_refuses_before_the_branch_measures_anything(
        self, store: ProvStore, speech_config: TriageConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """speech.enrollment_model is null on the packaged config; nothing invents a probe.

        The refusal does write: a caller-input problem is a finding about the run, so it gets a
        verdict rather than an exception. What it must not write is any measurement, span or
        finding, because none was taken.
        """
        _seed_speech_store(store, tmp_path, words=["hello", "world"])
        _stub_diarizers(monkeypatch, primary_speakers=1, second_speakers=1)
        result = speech(store, "plain", speech_config, run_dir=tmp_path, enrollment=_enrollment())
        assert result.verdict.outcome is Outcome.FLAG
        assert "speech.enrollment_model" in result.verdict.why
        assert find_measurement(store, "pii_scan") is None
        assert not [e for e in live_entities(store, "span") if e.attributes.get("family") == "speech"]
        assert not live_entities(store, "speaker") and not live_entities(store, "pii")
        assert result.view == (result.verdict_entity_id,), "the view is the refusal and nothing else"

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

    def test_a_name_said_twice_is_found_twice_and_marked_twice(
        self, store: ProvStore, speech_config: TriageConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The scan dedupes by (category, text, source), so one finding can cover two occurrences.

        Locating only the first left the second occurrence unmarked, and REDACT plans off the
        marking: it would have withheld the release unremediably, having found in its own re-scan a
        name no plan covered.
        """
        _seed_speech_store(store, tmp_path, words=["hi", "alice", "bye", "alice"])
        _stub_diarizers(monkeypatch, primary_speakers=1, second_speakers=1)
        _stub_pii(monkeypatch, findings=[("PERSON", "alice")])
        speech(store, "plain", speech_config, run_dir=tmp_path, enrollment=None)
        assert _verdict_entity(store, "SPEECH").attributes["pii"]["n"] == 2
        assert len(live_entities(store, "pii")) == 2
        marked = [
            store.derived_from(e.id)[0] for e in live_entities(store, "assertion") if e.attributes.get("label") == "pii"
        ]
        texts = sorted(str(store.get_entity(word_id).attributes["text"]) for word_id in marked)
        assert texts == ["alice", "alice"], "both occurrences carry the marking"
        assert len({e.extent for e in live_entities(store, "pii")}) == 2, "two distinct extents"

    def test_a_multi_word_finding_said_twice_is_located_at_both_runs(
        self, store: ProvStore, speech_config: TriageConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The runs do not overlap, so a two-word name is four marked words, not three."""
        _seed_speech_store(store, tmp_path, words=["hi", "ada", "lovelace", "and", "ada", "lovelace"])
        _stub_diarizers(monkeypatch, primary_speakers=1, second_speakers=1)
        _stub_pii(monkeypatch, findings=[("PERSON", "ada lovelace")])
        speech(store, "plain", speech_config, run_dir=tmp_path, enrollment=None)
        assert _verdict_entity(store, "SPEECH").attributes["pii"]["n"] == 2
        marks = [e for e in live_entities(store, "assertion") if e.attributes.get("label") == "pii"]
        assert len(marks) == 4

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

    def test_every_leg_is_compared_in_the_direction_the_spec_states(
        self, store: ProvStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Behind the target is quieter, more steeply tilted and less direct — not the reverse.

        The two spans are placed on opposite sides of all three legs by construction, and the cuts
        are then set strictly between the two measured values, so what is pinned is each
        comparison's *sense*, never a threshold. The far span is 1.1 s and the near one 0.7 s, so
        reversing the senses changes the product rather than leaving it alone.
        """
        _stub_diarizers(monkeypatch, primary_speakers=1, second_speakers=1)
        _seed_two_distance_spans(store, tmp_path)
        speech(store, "plain", _override(tmp_path), run_dir=tmp_path, enrollment=None)
        readings = sorted(find_measurements(store, "proximity"), key=lambda m: (m.extent or (0.0, 0.0))[0])
        assert len(readings) == 2, "the fixture must produce exactly the two spans it describes"
        near, far = readings[0].attributes, readings[1].attributes
        for leg in ("level_over_reference_db", "tilt_db_per_octave", "d_to_r_db"):
            assert far[leg] < near[leg], f"the fixture does not separate the spans on {leg}"

        cuts = {
            "level_db": (far["level_over_reference_db"] + near["level_over_reference_db"]) / 2,
            "tilt_db_per_octave": (far["tilt_db_per_octave"] + near["tilt_db_per_octave"]) / 2,
            "d_to_r_db": (far["d_to_r_db"] + near["d_to_r_db"]) / 2,
        }
        config = _override(
            tmp_path,
            "speech:\n  nontarget:\n" + "".join(f"    {leg}: {value!r}\n" for leg, value in cuts.items()),
        )
        second = ProvStore(run_id="second")
        _seed_two_distance_spans(second, tmp_path)
        speech(second, "plain", config, run_dir=tmp_path, enrollment=None)
        assert _verdict_entity(second, "SPEECH").attributes["nontarget_speech_s"] == pytest.approx(1.1, abs=1e-3)

    def test_no_span_is_excluded_on_this_evidence(
        self, store: ProvStore, speech_config: TriageConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """This branch marks; it removes nothing."""
        _seed_speech_store(store, tmp_path, words=["hello", "world"])
        _stub_diarizers(monkeypatch, primary_speakers=1, second_speakers=1)
        speech(store, "plain", speech_config, run_dir=tmp_path, enrollment=None)
        assert not [e for e in store.entities("span") if store.is_invalidated(e.id)]


class TestTheVerdictHangsOffTheStepThatConcluded:
    """store.md: the verdict is wasGeneratedBy the step that concluded, which is the last one."""

    def test_the_concluding_step_is_the_last_speech_step_to_run(
        self, store: ProvStore, speech_config: TriageConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Hanging it off quality said the conclusion was reached before the non-target axis ran.

        The detail carries ``nontarget_speech_s``, which step 9 produces, so a verdict generated by
        step 8 claims a conclusion that predates half its own content.
        """
        _seed_speech_store(store, tmp_path, words=["hello", "world"])
        _stub_diarizers(monkeypatch, primary_speakers=1, second_speakers=1)
        result = speech(store, "plain", speech_config, run_dir=tmp_path, enrollment=None)
        concluding = store.generated_by(result.verdict_entity_id)
        assert concluding is not None
        assert store.get_activity(concluding).step == "proximity"
        opened = [activity.id for activity in store.activities("SPEECH")]
        assert opened[-1] == concluding, "the concluding step is the last one this branch opened"
        assert "nontarget_speech_s" in _verdict_entity(store, "SPEECH").attributes

    def test_the_wordless_verdict_hangs_off_the_step_that_concluded_there(
        self, store: ProvStore, speech_config: TriageConfig, tmp_path: Path
    ) -> None:
        """That path concludes at the transcript, because it is the only step that ran."""
        _seed_speech_store(store, tmp_path, words=[])
        result = speech(store, "plain", speech_config, run_dir=tmp_path, enrollment=None)
        concluding = store.generated_by(result.verdict_entity_id)
        assert concluding is not None and store.get_activity(concluding).step == "transcript"
        assert [activity.id for activity in store.activities("SPEECH")][-1] == concluding


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


class TestSquimIsInertWhileItsFloorsAreNull:
    """F8g: the speech test's floors are null, so awful SQUIM numbers decide nothing."""

    def test_awful_squim_leaves_a_pass_a_pass_and_records_not_evaluated(
        self, store: ProvStore, speech_config: TriageConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A vote taken against a floor nobody derived is a decision dressed as a measurement."""
        monkeypatch.setattr(
            speech_module,
            "extract_objective_quality_features_from_audios",
            lambda audios, device=None: [{"stoi": 0.05, "pesq": 1.0, "si_sdr": -20.0} for _ in audios],
        )
        _seed_speech_store(store, tmp_path, words=["hello", "world"])
        _stub_diarizers(monkeypatch, primary_speakers=1, second_speakers=1)
        result = speech(store, "plain", speech_config, run_dir=tmp_path, enrollment=None)
        assert result.verdict.outcome is Outcome.PASS, "quality is reported, never gating"
        spans = [e for e in live_entities(store, "span") if e.attributes.get("family") == "speech"]
        assert spans and all(span.attributes["squim_vote"] == "not_evaluated" for span in spans)
        readings = find_measurements(store, "squim")
        assert readings and all(reading.attributes["stream"] == _stream_id(store, "plain") for reading in readings)

    def test_a_span_squim_refuses_is_unmeasured_rather_than_padded(
        self, store: ProvStore, speech_config: TriageConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A number invented for a span the instrument would not read is worse than no number."""
        monkeypatch.setattr(
            speech_module,
            "extract_objective_quality_features_from_audios",
            lambda audios, device=None: (_ for _ in ()).throw(RuntimeError("too short")),
        )
        _seed_speech_store(store, tmp_path, words=["hello", "world"])
        _stub_diarizers(monkeypatch, primary_speakers=1, second_speakers=1)
        speech(store, "plain", speech_config, run_dir=tmp_path, enrollment=None)
        readings = find_measurements(store, "squim")
        assert readings and all(reading.attributes["unmeasured"] == "RuntimeError" for reading in readings)


class TestWhatAPiiFailureMayCarry:
    """F8b, F8c: an unresolved speaker is the target's, and a failure message never escapes."""

    def test_a_finding_whose_speaker_cannot_be_resolved_is_treated_as_the_targets(
        self, store: ProvStore, enrollment_config: TriageConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """N12: a word straddling two segments belongs to neither, so it cannot be exempted."""
        _seed_speech_store(store, tmp_path, words=["alice"], word_extents=[(1.0, 1.4)])
        _stub_diarizers(monkeypatch, primary_speakers=2, second_speakers=2)
        _stub_embedder(monkeypatch, similarity=0.99, target_label="SPEAKER_00")
        _stub_pii(monkeypatch, findings=[("PERSON", "alice")])
        result = speech(store, "plain", enrollment_config, run_dir=tmp_path, enrollment=_enrollment())
        verdict = _verdict_entity(store, "SPEECH")
        assert verdict.attributes["target_speaker"] == "SPEAKER_00", "a target is known"
        assert result.verdict.outcome is Outcome.FLAG
        assert [flag for flag in verdict.attributes["flags"] if "cannot be resolved" in flag]

    def test_a_detector_failure_message_never_reaches_the_store_or_the_verdict(
        self, store: ProvStore, speech_config: TriageConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A failure message can quote the scanned text; only the detector and the type escape."""
        sentinel = "LEAKED-PII-TEXT"

        def _fake(inputs: Any, **kw: Any) -> list[PiiScan]:  # noqa: ANN401
            texts = [inputs] if isinstance(inputs, str) else list(inputs)
            return [
                PiiScan(
                    spans=[],
                    detectors_used=["presidio", "rules"],
                    failures={"gliner": f"ValueError: {sentinel}"},
                )
                for _ in texts
            ]

        monkeypatch.setattr(speech_module, "scan_for_pii", _fake)
        _seed_speech_store(store, tmp_path, words=["hello", "world"])
        _stub_diarizers(monkeypatch, primary_speakers=1, second_speakers=1)
        result = speech(store, "plain", speech_config, run_dir=tmp_path, enrollment=None)
        dumped = json.dumps([(e.prov_type, e.attributes) for e in store.entities()], default=str)
        assert sentinel not in dumped, "no entity carries a detector's failure message"
        assert sentinel not in result.verdict.why
        flags = _verdict_entity(store, "SPEECH").attributes["flags"]
        assert [flag for flag in flags if "gliner" in flag and "ValueError" in flag], "detector and type remain"

    def test_a_narrower_required_set_makes_the_same_scan_complete(
        self, store: ProvStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The required set is a config key, so an operator running two detectors can say so."""
        config = _override(tmp_path, "pii:\n  required_detectors: [presidio, rules]\n")
        _seed_speech_store(store, tmp_path, words=["hello", "world"])
        _stub_diarizers(monkeypatch, primary_speakers=1, second_speakers=1)
        _stub_pii(monkeypatch, findings=[], detectors_used=["presidio", "rules"])
        result = speech(store, "plain", config, run_dir=tmp_path, enrollment=None)
        assert result.verdict.outcome is Outcome.PASS
        scan = find_measurement(store, "pii_scan")
        assert scan is not None and scan.attributes["missing"] == []


class TestWhatTheBranchRecordsAboutItsOwnReads:
    """F8e, F8f, F8k: the used edges, the view, and the PREPROCESS span a speech span refines."""

    def test_every_read_is_recorded_with_used(
        self, store: ProvStore, speech_config: TriageConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Not the airway label: nothing in this branch reads one, so no edge may claim it did."""
        _seed_speech_store(
            store,
            tmp_path,
            words=["hello", "world"],
            spans=[(0.4, 1.4, 20.0)],
            airway_labelled=[(4.0, 4.4)],
        )
        _stub_diarizers(monkeypatch, primary_speakers=1, second_speakers=1)
        consensus = find_measurement(store, "consensus_transcript")
        assert consensus is not None
        word_ids = [word.id for word in live_entities(store, "word")]
        level = find_measurement(store, "level")
        assert level is not None
        prior_span = [e for e in live_entities(store, "span") if "peak_over_floor_db" in e.attributes][0]
        label_id = [e for e in live_entities(store, "assertion") if e.attributes.get("label") == "Cough"][0].id
        speech(store, "plain", speech_config, run_dir=tmp_path, enrollment=None)
        used: set[str] = set()
        for activity in store.activities("SPEECH"):
            used.update(store.uses_of(activity.id))
        assert consensus.id in used and set(word_ids) <= used
        assert prior_span.id in used
        assert level.id in used
        assert _stream_id(store, "plain") in used and _stream_id(store, "recording") in used
        assert label_id not in used, "SPEECH reads no airway label; the span it hangs on is read as a span"

    def test_the_view_carries_what_this_branch_authored(
        self, store: ProvStore, speech_config: TriageConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Partial is a view, not a payload: the ids are the consumer's way into the store."""
        _seed_speech_store(store, tmp_path, words=["hello", "world"])
        _stub_diarizers(monkeypatch, primary_speakers=1, second_speakers=1)
        result = speech(store, "plain", speech_config, run_dir=tmp_path, enrollment=None)
        assert result.verdict_entity_id in result.view
        speech_activities = {activity.id for activity in store.activities("SPEECH")}
        for prov_type in ("span", "speaker", "interval"):
            authored = {e.id for e in live_entities(store, prov_type) if store.generated_by(e.id) in speech_activities}
            assert authored and authored <= set(result.view), f"the view omits a {prov_type} this branch wrote"
        scan = find_measurement(store, "pii_scan")
        assert scan is not None and scan.id in result.view

    def test_a_speech_span_refines_the_preprocess_span_it_overlaps(
        self, store: ProvStore, speech_config: TriageConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Any temporal intersection > 0 refines (N10); a span with no words is left alone."""
        _seed_speech_store(store, tmp_path, words=["hello", "world"], spans=[(0.4, 1.4, 20.0), (4.0, 4.5, 25.0)])
        _stub_diarizers(monkeypatch, primary_speakers=1, second_speakers=1)
        overlapping, untouched = [e.id for e in live_entities(store, "span") if "peak_over_floor_db" in e.attributes]
        speech(store, "plain", speech_config, run_dir=tmp_path, enrollment=None)
        speech_spans = [e for e in live_entities(store, "span") if e.attributes.get("family") == "speech"]
        assert speech_spans
        refined = [span for span in speech_spans if overlapping in store.derived_from(span.id)]
        assert len(refined) == 1
        assert not [span for span in speech_spans if untouched in store.derived_from(span.id)]
        assert not store.is_invalidated(untouched), "a span with no words is left alone, not withdrawn"


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
