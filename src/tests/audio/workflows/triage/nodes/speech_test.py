"""SPEECH node tests. Every model call is faked at the node module; DSP and the store run real."""

import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pytest
import torch

from senselab.audio.data_structures import (
    Audio,
    AudioHints,
    ExpectedSpeech,
    SpeakerEmbeddingProvenance,
    TargetSpeakerEmbedding,
)
from senselab.audio.workflows.triage.config import TriageConfig, load_triage_config
from senselab.audio.workflows.triage.nodes import speech as speech_module
from senselab.audio.workflows.triage.nodes.common import find_measurement
from senselab.audio.workflows.triage.vocabulary import Outcome
from senselab.text.tasks.pii_detection.api import PiiScan, PiiSpan, default_detectors, flatten_script_line
from senselab.utils.data_structures import ScriptLine
from senselab.utils.prov_store import ProvStore

SR = 16000
WORDS = [("one", 1.0, 1.3), ("two", 1.4, 1.7)]
WORDS_WITH_EMAIL = [("contact", 1.0, 1.2), ("jane.doe@example.com", 1.25, 1.6)]

SeedSpeechStore = Callable[..., tuple[ProvStore, TriageConfig, Path]]


class _FakeModel:
    """A model spec stub carrying exactly what the node reads: path_or_uri and commit_sha."""

    def __init__(self, path_or_uri: str) -> None:
        """Stub a resolved model."""
        self.path_or_uri = path_or_uri
        self.commit_sha = "a" * 40


def _clean_scan() -> PiiScan:
    """A scan in which every default detector ran and found nothing."""
    return PiiScan(spans=[], detectors_used=["presidio", "gliner", "rules"], failures={})


def _scan_finding(secret: str, category: str) -> Callable[..., list[PiiScan]]:
    """A scan fake reporting one finding wherever the secret appears in the scanned line."""

    def _scan(inputs: list[ScriptLine], **kw: Any) -> list[PiiScan]:  # noqa: ANN401
        scans: list[PiiScan] = []
        for line in inputs:
            if secret.lower() in flatten_script_line(line).lower():
                scans.append(
                    PiiScan(
                        spans=[PiiSpan(text=secret, category=category, source="presidio", asr_model="0")],
                        detectors_used=["presidio", "gliner", "rules"],
                        failures={},
                    )
                )
            else:
                scans.append(_clean_scan())
        return scans

    return _scan


def _two_speaker_fake(audios: list[Audio], model: Any = None, **kw: Any) -> list[list[ScriptLine]]:  # noqa: ANN401
    """Two speakers splitting the diarized window in half, on the cropped clock."""
    dur = audios[0].waveform.shape[-1] / audios[0].sampling_rate
    half = dur / 2.0
    return [
        [
            ScriptLine(speaker="SPEAKER_00", start=0.0, end=half),
            ScriptLine(speaker="SPEAKER_01", start=half, end=dur),
        ]
    ]


def _three_speaker_fake(audios: list[Audio], model: Any = None, **kw: Any) -> list[list[ScriptLine]]:  # noqa: ANN401
    """Three speakers splitting the diarized window in thirds, on the cropped clock."""
    dur = audios[0].waveform.shape[-1] / audios[0].sampling_rate
    third = dur / 3.0
    return [[ScriptLine(speaker=f"SPEAKER_0{i}", start=i * third, end=(i + 1) * third) for i in range(3)]]


def _author(store: ProvStore, entity_id: str) -> str | None:
    """The node that generated an entity, or None when nothing did."""
    activity_id = store.generated_by(entity_id)
    return store.get_activity(activity_id).node if activity_id else None


def _speech_spans(store: ProvStore) -> list:
    """The span entities SPEECH authored."""
    return [e for e in store.entities("span") if _author(store, e.id) == "SPEECH"]


def _consensus_words(store: ProvStore) -> list:
    """The word entities SPEECH authored."""
    return [e for e in store.entities("word") if _author(store, e.id) == "SPEECH"]


@pytest.fixture(autouse=True)
def quiet_models(monkeypatch: pytest.MonkeyPatch) -> None:
    """One speaker, plausible SQUIM, no PII, no separation call, no Hub-resolving constructor."""

    def fake_diarize(audios: list[Audio], model: Any = None, **kw: Any) -> list[list[ScriptLine]]:  # noqa: ANN401
        dur = audios[0].waveform.shape[-1] / audios[0].sampling_rate
        return [[ScriptLine(speaker="SPEAKER_00", start=0.0, end=dur)]]

    monkeypatch.setattr(
        speech_module,
        "extract_objective_quality_features_from_audios",
        lambda audios, device=None: [{"stoi": 0.9, "pesq": 3.0, "si_sdr": 18.0} for _ in audios],
    )
    monkeypatch.setattr(speech_module, "diarize_audios", fake_diarize)
    monkeypatch.setattr(
        speech_module, "_diarization_model", lambda: _FakeModel("pyannote/speaker-diarization-community-1")
    )
    monkeypatch.setattr(speech_module, "_separation_model", lambda: _FakeModel("alibabasglab/MossFormer2_SS_16K"))
    monkeypatch.setattr(speech_module, "_embedding_model", lambda: _FakeModel("speechbrain/spkrec-ecapa-voxceleb"))
    monkeypatch.setattr(speech_module, "_second_diarizer_model", lambda model_id: _FakeModel(model_id))
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
    monkeypatch.setattr(speech_module, "scan_for_pii", lambda inputs, **kw: [_clean_scan() for _ in inputs])


class TestConfigKeys:
    """The Task 5 config additions: present, overridable, and refusing while unmeasured."""

    def test_new_speech_keys_exist_and_the_unmeasured_ones_raise(self) -> None:
        """Null keys are present (overridable) and refuse to be read as values."""
        cfg = load_triage_config()
        assert cfg.get("yamnet.top_k") == 521
        for key in (
            "speech.word_gap_ms",
            "speech.second_diarizer",
            "speech.target_match_cosine",
            "speech.agreement_flag_floor",
            "speech.speech_test_stoi_floor",
        ):
            with pytest.raises(ValueError, match="benchmarks/open.md|no value"):
                cfg.require(key)


def test_packaged_config_refuses_and_the_store_is_untouched(seed_speech_store: SeedSpeechStore) -> None:
    """word_gap_ms is null by design; the node raises at entry, before any store write."""
    store, _, run_dir = seed_speech_store([("hi", 1.0, 1.3)], [("hi", 1.0, 1.3)])
    before = store.fingerprint()
    with pytest.raises(ValueError, match="speech.word_gap_ms"):
        speech_module.speech(store, "plain", load_triage_config(), run_dir=run_dir)
    assert store.fingerprint() == before, "an unmeasured key must leave the store untouched"


def test_no_words_from_either_recognizer_is_a_normal_fail(seed_speech_store: SeedSpeechStore) -> None:
    """Fail means this branch has no subject — a cough recording is not an error."""
    store, cfg, run_dir = seed_speech_store([], [])
    result = speech_module.speech(store, "plain", cfg, run_dir=run_dir)
    assert result.verdict.outcome is Outcome.FAIL
    assert store.entities("verdict"), "the verdict entity is written even on fail"


def test_the_verdict_is_generated_by_the_step_that_concluded(seed_speech_store: SeedSpeechStore) -> None:
    """Walking generated_by from the verdict reaches the last step, not the transcript step.

    Attributing it to ``transcript`` said the conclusion was reached before diarization, PII and
    quality had run, each of which can turn a pass into a flag.
    """
    store, cfg, run_dir = seed_speech_store(WORDS, WORDS)
    result = speech_module.speech(store, "plain", cfg, run_dir=run_dir)
    concluding = store.generated_by(result.verdict_entity_id)
    assert concluding is not None
    assert store.get_activity(concluding).step == "quality"


def test_the_no_words_path_still_writes_the_scan_redact_reads(seed_speech_store: SeedSpeechStore) -> None:
    """REDACT refuses a store with no scan measurement, so a branch with no subject must not leave none.

    An empty scan is what REDACT's incomplete-scan row already reads as withheld, which is the right
    conclusion; absence of the measurement instead made REDACT raise on a recording that simply had
    no speech.
    """
    store, cfg, run_dir = seed_speech_store([], [])
    result = speech_module.speech(store, "plain", cfg, run_dir=run_dir)
    scan = find_measurement(store, "pii_scan")
    assert scan is not None, "the measurement is written even when nothing was scanned"
    assert scan.attributes["scanned_by"] == []
    assert scan.attributes["failed"] == []
    assert result.verdict.outcome is Outcome.FAIL


def test_hint_asserting_speech_not_found_flags(seed_speech_store: SeedSpeechStore) -> None:
    """A hint asserting speech turns the no-words fail into a flag — the contradiction outranks it."""
    store, cfg, run_dir = seed_speech_store([], [])
    hint = AudioHints(expected_speech=[ExpectedSpeech(text="the rainbow passage")])
    result = speech_module.speech(store, "plain", cfg, hint, run_dir=run_dir)
    assert result.verdict.outcome is Outcome.FLAG


def test_spans_come_from_word_timings_and_refine_preprocess_spans(seed_speech_store: SeedSpeechStore) -> None:
    """Two word runs over word_gap_ms apart are two spans.

    An overlapping PREPROCESS span is refined (wasDerivedFrom); one with no words is left alone.
    """
    words = [("one", 1.0, 1.2), ("two", 1.25, 1.5), ("three", 3.0, 3.4)]
    store, cfg, run_dir = seed_speech_store(words, words, airway_label_extent=(4.5, 5.0))
    pre_act = store.activity(node="PREPROCESS", step="spans", parameters={})
    pre_span = store.entity(
        prov_type="span",
        extent=(0.9, 1.6),
        attributes={"peak_over_floor_db": 20.0, "k_db": 18.0, "signal": "preemphasised"},
    )
    store.was_generated_by(pre_span, pre_act)
    speech_module.speech(store, "plain", cfg, run_dir=run_dir)
    speech_spans = _speech_spans(store)
    assert [
        tuple(round(x, 2) for x in (s.extent or (0.0, 0.0)))
        for s in sorted(speech_spans, key=lambda s: s.extent or (0.0, 0.0))
    ] == [(1.0, 1.5), (3.0, 3.4)]
    refined = [s for s in speech_spans if pre_span in store.derived_from(s.id)]
    assert len(refined) == 1, "any temporal intersection > 0 refines (N10); the airway span is untouched"


def test_pyannote_sees_only_the_word_interval_and_segments_are_offset_back(
    seed_speech_store: SeedSpeechStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The diarizer gets [first word start, last word end]; its clock is shifted back to the recording's."""
    seen: dict[str, float] = {}

    def fake_diarize(audios: list[Audio], **kw: Any) -> list[list[ScriptLine]]:  # noqa: ANN401
        seen["dur"] = audios[0].waveform.shape[-1] / audios[0].sampling_rate
        return [[ScriptLine(speaker="SPEAKER_00", start=0.0, end=seen["dur"])]]

    monkeypatch.setattr(speech_module, "diarize_audios", fake_diarize)
    words = [("one", 2.0, 2.3), ("two", 2.4, 2.8)]
    store, cfg, run_dir = seed_speech_store(words, words)
    speech_module.speech(store, "plain", cfg, run_dir=run_dir)
    assert seen["dur"] == pytest.approx(0.8, abs=1 / SR), "cropped to the interval, not the file"
    seg = store.entities("speaker")[0]
    assert seg.extent == pytest.approx((2.0, 2.8)), "offset added back onto the returned segment"


def test_a_segment_overlapping_an_airway_label_survives_because_diarization_is_speech_only(
    seed_speech_store: SeedSpeechStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A diarizer segment is never withdrawn for overlapping an airway event.

    The Story-recall shape: one narrator, words across the whole interval, one Breathe label inside
    it. The one diarizer segment covers the narration and overlaps the label, so under the withdrawal
    rule the file's speaker count read 0 — measured on 10 of the campaign's 28 files — every word went
    unattributed and the unattributed words cascaded into false PII withholds. Diarization is about
    speech, so an airway label carries no authority over a diarizer segment.
    """
    words = [("one", 1.0, 1.4), ("two", 1.6, 2.0), ("three", 4.4, 4.8)]
    store, cfg, run_dir = seed_speech_store(words, words, airway_label_extent=(4.5, 5.0))

    def fake_diarize(audios: list[Audio], **kw: Any) -> list[list[ScriptLine]]:  # noqa: ANN401
        return [[ScriptLine(speaker="SPEAKER_00", start=0.0, end=3.8)]]  # (1.0, 4.8) after the offset

    monkeypatch.setattr(speech_module, "diarize_audios", fake_diarize)
    result = speech_module.speech(store, "plain", cfg, run_dir=run_dir)
    (speaker,) = store.entities("speaker")
    assert not store.is_invalidated(speaker.id), "the airway label does not withdraw the segment"
    verdict = store.get_entity(result.verdict_entity_id)
    assert verdict.attributes["speaker_count"] == 1, "the count is the live segment count"
    spoken = store.entities("word")
    assert spoken and all(w.attributes["speaker"] == "SPEAKER_00" for w in spoken if "recognizer" not in w.attributes)
    assert not any("speaker_note" in w.attributes for w in spoken), "no word is left unattributed"


def test_count_two_separates_and_measurements_record_their_stream(
    seed_speech_store: SeedSpeechStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """MossFormer runs at n_sources=2 with no unasdiff arguments; streams become entities."""
    calls: dict[str, Any] = {}

    def fake_separate(
        audios: list[Audio],
        model: Any = None,  # noqa: ANN401
        n_sources: int = 2,
        device: Any = None,  # noqa: ANN401
        timeout_s: float | None = None,
        **kw: Any,  # noqa: ANN401
    ) -> list[list[Audio]]:
        calls["n_sources"] = n_sources
        calls["unasdiff_args"] = {
            k: v for k, v in kw.items() if k in ("mode", "source_classes", "seed", "diffusion_steps")
        }
        out = []
        for i in range(2):
            a = Audio(waveform=audios[0].waveform, sampling_rate=SR)
            a.metadata["clearvoice"] = {  # the real record's shape, tasks/clearvoice.py:103-112
                "model": "alibabasglab/MossFormer2_SS_16K",
                "commit": "b" * 40,
                "capability": "separation",
                "sampling_rate": SR,
                "source_index": i,
                "n_sources": 2,
                "input_norm_scalar": 0.31,
                "input_norm_applied_to_output": False,
            }
            out.append(a)
        return [out]

    monkeypatch.setattr(speech_module, "separate_audios", fake_separate)
    monkeypatch.setattr(speech_module, "diarize_audios", _two_speaker_fake)
    store, cfg, run_dir = seed_speech_store(WORDS, WORDS)
    result = speech_module.speech(store, "plain", cfg, run_dir=run_dir)
    assert calls["n_sources"] == 2
    assert calls["unasdiff_args"] == {}, "mode/source_classes/seed/diffusion_steps are unasdiff's; the API refuses them"
    streams = [e for e in store.entities("stream") if "source_index" in e.attributes]
    assert {s.attributes["source_index"] for s in streams} == {0, 1}
    assert all("input_norm_scalar" in s.attributes for s in streams), (
        "level died at the worker's normalisation; the un-applied scalar is the record (N28)"
    )
    squims = [
        m
        for m in store.entities("measurement")
        if m.attributes.get("name") == "squim" and _author(store, m.id) == "SPEECH"
    ]
    assert squims and all("stream" in m.attributes for m in squims), "every measurement records its stream (N28)"
    assert result.verdict.outcome is Outcome.FLAG, "count != 1 flags"


def test_count_three_reports_rather_than_separating_wrong(
    seed_speech_store: SeedSpeechStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The checkpoint separates exactly two; >= 3 is reported, separate_audios is never called."""
    monkeypatch.setattr(speech_module, "diarize_audios", _three_speaker_fake)
    # the autouse fake for separate_audios raises AssertionError if called
    store, cfg, run_dir = seed_speech_store(WORDS, WORDS)
    result = speech_module.speech(store, "plain", cfg, run_dir=run_dir)
    assert result.verdict.outcome is Outcome.FLAG
    verdict = store.get_entity(result.verdict_entity_id)
    assert any("separation" in f for f in verdict.attributes["flags"])


def test_second_diarizer_null_records_not_consulted(
    seed_speech_store: SeedSpeechStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """While speech.second_diarizer is null, count != 1 records not_consulted and still flags (N6)."""
    monkeypatch.setattr(speech_module, "diarize_audios", _two_speaker_fake)
    monkeypatch.setattr(speech_module, "separate_audios", lambda *a, **k: [[]])
    store, cfg, run_dir = seed_speech_store(WORDS, WORDS)
    result = speech_module.speech(store, "plain", cfg, run_dir=run_dir)
    verdict = store.get_entity(result.verdict_entity_id)
    assert verdict.attributes["second_diarizer"] == "not_consulted"
    assert result.verdict.outcome is Outcome.FLAG


def test_second_diarizer_consulted_when_configured(
    seed_speech_store: SeedSpeechStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A configured second diarizer is consulted on count != 1 and its disagreement is reported."""

    def fake_diarize(audios: list[Audio], model: Any = None, **kw: Any) -> list[list[ScriptLine]]:  # noqa: ANN401
        if model is not None and "diarizen" in str(model.path_or_uri):
            return _three_speaker_fake(audios)
        return _two_speaker_fake(audios)

    monkeypatch.setattr(speech_module, "diarize_audios", fake_diarize)
    monkeypatch.setattr(speech_module, "separate_audios", lambda *a, **k: [[]])
    store, cfg, run_dir = seed_speech_store(
        WORDS,
        WORDS,
        config_yaml="speech:\n  word_gap_ms: 300\n  second_diarizer: BUT-FIT/diarizen-wavlm-large-s80-md\n",
    )
    result = speech_module.speech(store, "plain", cfg, run_dir=run_dir)
    verdict = store.get_entity(result.verdict_entity_id)
    assert verdict.attributes["second_diarizer"]["count"] == 3
    assert verdict.attributes["second_diarizer"]["agrees"] is False
    assert result.verdict.outcome is Outcome.FLAG


def test_pii_decision_is_speaker_scoped(seed_speech_store: SeedSpeechStore, monkeypatch: pytest.MonkeyPatch) -> None:
    """Target-speaker finding flags; non-target-only does not; no target flags; failure flags."""
    words = [("alice", 1.0, 1.3), ("bob", 2.0, 2.3)]
    target_yaml = "speech:\n  word_gap_ms: 300\n  target_match_cosine: 0.5\n"
    hint = AudioHints(
        target_speaker=TargetSpeakerEmbedding(
            vector=[1.0, 0.0],
            provenance=SpeakerEmbeddingProvenance(
                model_id="speechbrain/spkrec-ecapa-voxceleb", model_commit_sha="a" * 40
            ),
        )
    )

    def fake_diarize(audios: list[Audio], model: Any = None, **kw: Any) -> list[list[ScriptLine]]:  # noqa: ANN401
        return [
            [
                ScriptLine(speaker="SPEAKER_00", start=0.0, end=0.4),
                ScriptLine(speaker="SPEAKER_01", start=0.9, end=1.3),
            ]
        ]

    def fake_embed(audios: list[Audio], model: Any = None, device: Any = None) -> list[torch.Tensor]:  # noqa: ANN401
        return [torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])][: len(audios)]

    def _pii_flags(store: ProvStore, result: Any) -> list[str]:  # noqa: ANN401
        return [f for f in store.get_entity(result.verdict_entity_id).attributes["flags"] if "pii" in f]

    # (a) finding on the target speaker's words -> flag
    monkeypatch.setattr(speech_module, "diarize_audios", fake_diarize)
    monkeypatch.setattr(speech_module, "extract_speaker_embeddings_from_audios", fake_embed)
    monkeypatch.setattr(speech_module, "separate_audios", lambda *a, **k: [[]])
    monkeypatch.setattr(speech_module, "scan_for_pii", _scan_finding("alice", "PERSON"))
    store, cfg, run_dir = seed_speech_store(words, words, config_yaml=target_yaml)
    result = speech_module.speech(store, "plain", cfg, hint, run_dir=run_dir)
    assert store.get_entity(result.verdict_entity_id).attributes.get("target_speaker") == "SPEAKER_00"
    assert _pii_flags(store, result), "a target-speaker finding flags"

    # (b) same finding, attributed to the non-target speaker, target known -> no flag from PII
    monkeypatch.setattr(speech_module, "scan_for_pii", _scan_finding("bob", "PERSON"))
    store, cfg, run_dir = seed_speech_store(words, words, config_yaml=target_yaml)
    result = speech_module.speech(store, "plain", cfg, hint, run_dir=run_dir)
    assert not _pii_flags(store, result), "a non-target finding with a known target does not flag"

    # (c) same finding, no hint at all -> flag ("no speaker to exempt")
    monkeypatch.setattr(speech_module, "scan_for_pii", _scan_finding("alice", "PERSON"))
    store, cfg, run_dir = seed_speech_store(words, words, config_yaml=target_yaml)
    result = speech_module.speech(store, "plain", cfg, run_dir=run_dir)
    assert _pii_flags(store, result), "pii with no known target flags"

    # (d) clean spans but failures={"gliner": ...} -> flag ("could not check")
    monkeypatch.setattr(
        speech_module,
        "scan_for_pii",
        lambda inputs, **kw: [
            PiiScan(spans=[], detectors_used=["presidio", "rules"], failures={"gliner": "load failed"}) for _ in inputs
        ],
    )
    store, cfg, run_dir = seed_speech_store(words, words, config_yaml=target_yaml)
    result = speech_module.speech(store, "plain", cfg, run_dir=run_dir)
    assert any("gliner" in f for f in _pii_flags(store, result)), "could not check is not clean"


def test_a_required_detector_that_never_ran_flags_could_not_check(
    seed_speech_store: SeedSpeechStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A detector neither scanned nor failed is recorded as missing, and the scan is incomplete.

    Locally gliner was never attempted, so ``failed`` was empty and the scan claimed completeness;
    on the cluster the same recording attempted it and recorded the failure. "Complete" must not
    depend on which host ran it.
    """
    monkeypatch.setattr(
        speech_module,
        "scan_for_pii",
        lambda inputs, **kw: [PiiScan(spans=[], detectors_used=["presidio", "rules"], failures={}) for _ in inputs],
    )
    store, cfg, run_dir = seed_speech_store(WORDS, WORDS)
    result = speech_module.speech(store, "plain", cfg, run_dir=run_dir)
    assert result.verdict.outcome is Outcome.FLAG
    flags = store.get_entity(result.verdict_entity_id).attributes["flags"]
    assert any("gliner" in flag and "pii" in flag for flag in flags), "never attempted is not clean"
    scan = find_measurement(store, "pii_scan")
    assert scan is not None
    assert scan.attributes["missing"] == ["gliner"]
    assert scan.attributes["failed"] == [], "never attempted is not the same as attempted and failed"


def test_narrowing_the_required_set_makes_the_same_scan_complete(
    seed_speech_store: SeedSpeechStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The required set is a config key, so an operator running two detectors can say so."""
    monkeypatch.setattr(
        speech_module,
        "scan_for_pii",
        lambda inputs, **kw: [PiiScan(spans=[], detectors_used=["presidio", "rules"], failures={}) for _ in inputs],
    )
    store, cfg, run_dir = seed_speech_store(
        WORDS,
        WORDS,
        config_yaml="speech:\n  word_gap_ms: 300\npii:\n  required_detectors: [presidio, rules]\n",
    )
    result = speech_module.speech(store, "plain", cfg, run_dir=run_dir)
    assert result.verdict.outcome is Outcome.PASS
    scan = find_measurement(store, "pii_scan")
    assert scan is not None
    assert scan.attributes["missing"] == []


def test_the_no_words_path_records_every_required_detector_as_missing(
    seed_speech_store: SeedSpeechStore,
) -> None:
    """Nothing was scanned there, so nothing required was met — the measurement must say so."""
    store, cfg, run_dir = seed_speech_store([], [])
    speech_module.speech(store, "plain", cfg, run_dir=run_dir)
    scan = find_measurement(store, "pii_scan")
    assert scan is not None
    assert scan.attributes["missing"] == default_detectors()


def test_pii_entities_and_verdict_never_carry_matched_text(
    seed_speech_store: SeedSpeechStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Projection, not filtering: no store entity, verdict value or exception carries the match."""
    secret = "jane.doe@example.com"
    monkeypatch.setattr(speech_module, "scan_for_pii", _scan_finding(secret, "EMAIL_ADDRESS"))
    store, cfg, run_dir = seed_speech_store(WORDS_WITH_EMAIL, WORDS_WITH_EMAIL)
    result = speech_module.speech(store, "plain", cfg, run_dir=run_dir)
    dumped = json.dumps([(e.prov_type, e.attributes) for e in store.entities() if e.prov_type != "word"], default=str)
    assert secret not in dumped, "pii/measurement/verdict entities are projections"
    assert secret not in json.dumps(store.get_entity(result.verdict_entity_id).attributes, default=str)
    pii = store.entities("pii")
    assert pii and pii[0].attributes["category"] == "EMAIL_ADDRESS" and pii[0].extent is not None


def test_target_without_commit_is_refused_and_flagged_without_an_embedding_call(
    seed_speech_store: SeedSpeechStore,
) -> None:
    """Embeddings from different models are not comparable; unprovenanced targets are refused."""
    hint = AudioHints(
        target_speaker=TargetSpeakerEmbedding(
            vector=[1.0, 0.0],
            provenance=SpeakerEmbeddingProvenance(
                model_id="speechbrain/spkrec-ecapa-voxceleb",
                model_commit_sha=None,
                unresolved_reason="enrollment vector of unknown commit",
            ),
        )
    )
    store, cfg, run_dir = seed_speech_store(WORDS, WORDS)
    # the autouse fake for extract_speaker_embeddings_from_audios raises AssertionError if called
    result = speech_module.speech(store, "plain", cfg, hint, run_dir=run_dir)
    assert result.verdict.outcome is Outcome.FLAG
    verdict = store.get_entity(result.verdict_entity_id)
    assert any("commit" in f for f in verdict.attributes["flags"]), "refused, with the reason recorded"
    assert not store.entities("target_match"), "no comparison happened"


def test_target_with_provenance_requires_the_null_cosine_key(seed_speech_store: SeedSpeechStore) -> None:
    """A hint carrying a target under a null speech.target_match_cosine raises at entry (N7)."""
    hint = AudioHints(
        target_speaker=TargetSpeakerEmbedding(
            vector=[1.0, 0.0],
            provenance=SpeakerEmbeddingProvenance(
                model_id="speechbrain/spkrec-ecapa-voxceleb", model_commit_sha="a" * 40
            ),
        )
    )
    store, cfg, run_dir = seed_speech_store(WORDS, WORDS)
    before = store.fingerprint()
    with pytest.raises(ValueError, match="target_match_cosine"):
        speech_module.speech(store, "plain", cfg, hint, run_dir=run_dir)
    assert store.fingerprint() == before, "the refusal precedes any store write"


def test_provenanced_target_from_another_model_is_refused_like_a_commitless_one(
    seed_speech_store: SeedSpeechStore,
) -> None:
    """Comparability gates the cut, not provenance alone: a wrong-model target is refused, never read."""
    hint = AudioHints(
        target_speaker=TargetSpeakerEmbedding(
            vector=[1.0, 0.0],
            provenance=SpeakerEmbeddingProvenance(model_id="pyannote/embedding", model_commit_sha="b" * 40),
        )
    )
    store, cfg, run_dir = seed_speech_store(WORDS, WORDS)  # target_match_cosine is null here
    # the autouse fake for extract_speaker_embeddings_from_audios raises AssertionError if called
    result = speech_module.speech(store, "plain", cfg, hint, run_dir=run_dir)
    assert result.verdict.outcome is Outcome.FLAG
    verdict = store.get_entity(result.verdict_entity_id)
    assert any("not comparable" in f for f in verdict.attributes["flags"]), "refused, with the reason recorded"
    assert not store.entities("target_match"), "no comparison happened"


def test_pii_whose_speaker_is_unresolved_flags_when_a_target_is_known(
    seed_speech_store: SeedSpeechStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """N12: a finding on a word straddling two segments is treated as the target's and flags."""
    words = [("alice", 1.0, 1.3), ("bob", 2.0, 2.3)]
    target_yaml = "speech:\n  word_gap_ms: 300\n  target_match_cosine: 0.5\n"
    hint = AudioHints(
        target_speaker=TargetSpeakerEmbedding(
            vector=[1.0, 0.0],
            provenance=SpeakerEmbeddingProvenance(
                model_id="speechbrain/spkrec-ecapa-voxceleb", model_commit_sha="a" * 40
            ),
        )
    )

    def fake_diarize(audios: list[Audio], model: Any = None, **kw: Any) -> list[list[ScriptLine]]:  # noqa: ANN401
        # On the cropped clock the boundary falls inside "alice", so that word straddles both segments.
        return [
            [
                ScriptLine(speaker="SPEAKER_00", start=0.0, end=0.15),
                ScriptLine(speaker="SPEAKER_01", start=0.15, end=1.3),
            ]
        ]

    def fake_embed(audios: list[Audio], model: Any = None, device: Any = None) -> list[torch.Tensor]:  # noqa: ANN401
        return [torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])][: len(audios)]

    monkeypatch.setattr(speech_module, "diarize_audios", fake_diarize)
    monkeypatch.setattr(speech_module, "extract_speaker_embeddings_from_audios", fake_embed)
    monkeypatch.setattr(speech_module, "separate_audios", lambda *a, **k: [[]])
    monkeypatch.setattr(speech_module, "scan_for_pii", _scan_finding("alice", "PERSON"))
    store, cfg, run_dir = seed_speech_store(words, words, config_yaml=target_yaml)
    result = speech_module.speech(store, "plain", cfg, hint, run_dir=run_dir)
    verdict = store.get_entity(result.verdict_entity_id)
    assert verdict.attributes.get("target_speaker") == "SPEAKER_00", "a target is known"
    (straddler,) = [w for w in _consensus_words(store) if w.attributes["text"] == "alice"]
    assert straddler.attributes["speaker"] is None and straddler.attributes["speaker_note"] == "straddles"
    assert result.verdict.outcome is Outcome.FLAG
    assert any("cannot be resolved" in f for f in verdict.attributes["flags"]), "treated as the target's"


def test_a_detector_failure_message_never_reaches_the_store_or_the_verdict(
    seed_speech_store: SeedSpeechStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failure message can quote the scanned text; only the detector and the exception type escape."""
    sentinel = "LEAKED-PII-TEXT"
    monkeypatch.setattr(
        speech_module,
        "scan_for_pii",
        lambda inputs, **kw: [
            PiiScan(spans=[], detectors_used=["presidio"], failures={"gliner": f"ValueError: {sentinel}"})
            for _ in inputs
        ],
    )
    store, cfg, run_dir = seed_speech_store(WORDS, WORDS)
    result = speech_module.speech(store, "plain", cfg, run_dir=run_dir)
    dumped = json.dumps([(e.prov_type, e.attributes) for e in store.entities()], default=str)
    assert sentinel not in dumped, "no entity carries a detector's failure message"
    verdict = store.get_entity(result.verdict_entity_id)
    assert sentinel not in json.dumps(verdict.attributes, default=str)
    assert sentinel not in result.verdict.why
    assert any("gliner" in f and "ValueError" in f for f in verdict.attributes["flags"]), "detector and type remain"


def test_quality_is_reported_never_gating(seed_speech_store: SeedSpeechStore, monkeypatch: pytest.MonkeyPatch) -> None:
    """Terrible SQUIM numbers and real disruptions leave a pass a pass."""
    monkeypatch.setattr(
        speech_module,
        "extract_objective_quality_features_from_audios",
        lambda audios, device=None: [{"stoi": 0.1, "pesq": 1.0, "si_sdr": -10.0} for _ in audios],
    )
    store, cfg, run_dir = seed_speech_store(WORDS, WORDS)
    result = speech_module.speech(store, "plain", cfg, run_dir=run_dir)
    assert result.verdict.outcome is Outcome.PASS
    dis = [m for m in store.entities("measurement") if m.attributes.get("name") == "disruptions"]
    assert dis and all("clipped_runs" in m.attributes for m in dis), "counts and extents, not a score"
    assert all("zero_crossing_rate" in m.attributes for m in dis), "the ZCR reading rides along, ungated"


def test_disruptions_are_measured_on_the_original_recording_not_the_normalised_copy(
    seed_speech_store: SeedSpeechStore,
) -> None:
    """Clipping is a property of the recording, and normalise-then-resample destroys the evidence.

    The campaign read clipped_runs == 0 on every file, four of which peak at exactly 0.0 dBFS. The
    discriminating pair is here: one original carrying a flat plateau at full scale, and the plain
    stream PREPROCESS derived from it, in which the plateau no longer exists. The node must report
    the first, and must name the stream it measured.
    """
    import soundfile as sf

    from senselab.audio.tasks.disruptions import detect_disruptions

    duration_s = 6.0
    store, cfg, run_dir = seed_speech_store(WORDS, WORDS, duration_s=duration_s)
    original = np.zeros(int(duration_s * SR), dtype=np.float32)
    original[int(0.9 * SR) : int(1.8 * SR)] = 1.0  # a flat plateau at full scale across the words
    sf.write(str(run_dir / "streams" / "recording.wav"), original, SR)
    pre = store.activity(node="PREPROCESS", step="capture", parameters={})
    recording_id = store.entity(
        prov_type="stream",
        extent=(0.0, duration_s),
        attributes={"name": "recording", "path": "streams/recording.wav", "sampling_rate": SR, "channels": 1},
    )
    store.was_generated_by(recording_id, pre)

    speech_module.speech(store, "plain", cfg, run_dir=run_dir)
    dis = [m for m in store.entities("measurement") if m.attributes.get("name") == "disruptions"]
    assert dis, "every span carries a disruptions measurement"
    assert any(m.attributes["clipped_runs"] > 0 for m in dis), "the plateau in the original is clipping"
    assert all(m.attributes["stream"] == recording_id for m in dis), "a measurement names the stream it read"

    plain_audio = Audio(filepath=str(run_dir / "streams" / "plain.wav"))
    invisible = detect_disruptions(
        plain_audio,
        1.0,
        1.7,
        clip_headroom=float(cfg.require("disruptions.clip_headroom")),
        min_clip_run=int(cfg.require("disruptions.min_clip_run")),
        min_dropout_ms=float(cfg.require("disruptions.min_dropout_ms")),
        discontinuity_local_factor=float(cfg.require("disruptions.discontinuity_local_factor")),
        discontinuity_window_ms=float(cfg.require("disruptions.discontinuity_window_ms")),
    )
    assert invisible.clipped_runs == 0, "the same clipping is invisible on the derived copy"


def test_squim_vote_is_inert_while_thresholds_are_null(
    seed_speech_store: SeedSpeechStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """squim_vote records not_evaluated and no flag fires on awful SQUIM while the floors are null (N4)."""
    monkeypatch.setattr(
        speech_module,
        "extract_objective_quality_features_from_audios",
        lambda audios, device=None: [{"stoi": 0.05, "pesq": 1.0, "si_sdr": -20.0} for _ in audios],
    )
    store, cfg, run_dir = seed_speech_store(WORDS, WORDS)
    result = speech_module.speech(store, "plain", cfg, run_dir=run_dir)
    assert result.verdict.outcome is Outcome.PASS
    spans = _speech_spans(store)
    assert spans and all(s.attributes["squim_vote"] == "not_evaluated" for s in spans)


def test_fusion_runs_real_and_confidence_is_agreement_not_correctness(
    seed_speech_store: SeedSpeechStore,
) -> None:
    """Two hypotheses disagreeing on one word leave that consensus word less confident than the agreed one."""
    store, cfg, run_dir = seed_speech_store(
        [("one", 1.0, 1.2), ("cat", 1.5, 1.7)], [("one", 1.0, 1.2), ("hat", 1.5, 1.7)]
    )
    speech_module.speech(store, "plain", cfg, run_dir=run_dir)
    words = sorted(_consensus_words(store), key=lambda w: w.extent or (0.0, 0.0))
    assert len(words) == 2
    agreed, disputed = words[0].attributes, words[1].attributes
    assert agreed["confidence"] is not None and disputed["confidence"] is not None
    assert disputed["confidence"] < agreed["confidence"], "agreement bounds confidence from above"


def test_a_word_over_no_energy_is_a_fabrication_candidate_and_flags(
    seed_speech_store: SeedSpeechStore,
) -> None:
    """A word whose extent never clears the local floor gets a fabrication_candidate label and flags (N9)."""
    words = [*WORDS, ("ghost", 5.5, 5.7)]
    store, cfg, run_dir = seed_speech_store(words, words)
    sidecar = run_dir / "derivatives" / "energy_envelope.npz"
    data = np.load(sidecar)
    env = data["envelope_dbfs"].copy()
    env[int(5.5 * SR) : int(5.7 * SR)] = -80.0
    np.savez(sidecar, envelope_dbfs=env, floor_dbfs=data["floor_dbfs"])
    result = speech_module.speech(store, "plain", cfg, run_dir=run_dir)
    assert result.verdict.outcome is Outcome.FLAG
    verdict = store.get_entity(result.verdict_entity_id)
    assert any("fabrication" in f for f in verdict.attributes["flags"])
    labels = [a for a in store.entities("assertion") if a.attributes.get("label") == "fabrication_candidate"]
    assert len(labels) == 1
    assert labels[0].extent == pytest.approx((5.5, 5.7))
    (word_id,) = store.derived_from(labels[0].id)
    assert store.get_entity(word_id).prov_type == "word", "the label hangs off the offending word"


def test_yamnet_disconfirmation_flags(seed_speech_store: SeedSpeechStore) -> None:
    """Speech coverage below the threshold over one span flags, and the span carries the coverage."""
    words = [("one", 1.0, 1.3), ("far", 4.0, 4.3)]
    store, cfg, run_dir = seed_speech_store(words, words)
    sidecar = run_dir / "derivatives" / "yamnet_windows.json"
    windows = json.loads(sidecar.read_text())
    for window in windows:
        if float(window["start"]) < 4.3 and float(window["end"]) > 4.0:
            window["label_scores"] = [{"Speech": 0.1}]
    sidecar.write_text(json.dumps(windows))
    result = speech_module.speech(store, "plain", cfg, run_dir=run_dir)
    assert result.verdict.outcome is Outcome.FLAG
    verdict = store.get_entity(result.verdict_entity_id)
    assert any("yamnet" in f for f in verdict.attributes["flags"])
    doubted = [s for s in _speech_spans(store) if (s.extent or (0.0, 0.0))[0] == pytest.approx(4.0)]
    assert doubted and doubted[0].attributes["yamnet_coverage"] < 0.5
    assert doubted[0].attributes["yamnet_vote"] == "disconfirm"


def test_straddling_word_is_marked_not_assigned(
    seed_speech_store: SeedSpeechStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A word overlapping two diarizer segments is marked, never assigned to either."""
    monkeypatch.setattr(speech_module, "diarize_audios", _two_speaker_fake)
    monkeypatch.setattr(speech_module, "separate_audios", lambda *a, **k: [[]])
    store, cfg, run_dir = seed_speech_store([("one", 1.0, 1.4)], [("one", 1.0, 1.4)])
    speech_module.speech(store, "plain", cfg, run_dir=run_dir)
    (word,) = _consensus_words(store)
    assert word.attributes["speaker"] is None
    assert word.attributes["speaker_note"] == "straddles"


def test_flag_view_includes_every_segment_the_count_was_taken_over(
    seed_speech_store: SeedSpeechStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Partial is a view, not a payload: the flagged result's view carries the contested entities.

    The flag here is a speaker count of three, and what is contested is the segments that produced
    it — including the one overlapping the airway label, which now counts like any other.
    """
    words = [("one", 1.0, 1.4), ("two", 4.4, 4.8)]
    store, cfg, run_dir = seed_speech_store(words, words, airway_label_extent=(4.5, 5.0))

    def fake_diarize(audios: list[Audio], **kw: Any) -> list[list[ScriptLine]]:  # noqa: ANN401
        return [
            [
                ScriptLine(speaker="SPEAKER_00", start=0.0, end=1.5),
                ScriptLine(speaker="SPEAKER_01", start=2.0, end=3.0),
                ScriptLine(speaker="SPEAKER_02", start=3.4, end=3.8),  # overlaps the label after offset
            ]
        ]

    monkeypatch.setattr(speech_module, "diarize_audios", fake_diarize)
    monkeypatch.setattr(speech_module, "separate_audios", lambda *a, **k: [[]])
    result = speech_module.speech(store, "plain", cfg, run_dir=run_dir)
    assert result.verdict.outcome is Outcome.FLAG
    verdict = store.get_entity(result.verdict_entity_id)
    assert verdict.attributes["speaker_count"] == 3, "the label-overlapping segment counts too"
    speakers = [s.id for s in store.entities("speaker")]
    assert len(speakers) == 3 and all(entity_id in result.view for entity_id in speakers)
    assert not any(store.is_invalidated(entity_id) for entity_id in speakers)


def test_every_read_is_recorded_with_used(seed_speech_store: SeedSpeechStore) -> None:
    """The SPEECH activities' used targets include the word, envelope and span entities read.

    Not the airway label: diarization is speech-only, so nothing in this branch reads one.
    """
    words = [("one", 1.0, 1.2), ("two", 1.25, 1.5)]
    store, cfg, run_dir = seed_speech_store(words, words, airway_label_extent=(4.5, 5.0))
    pre_act = store.activity(node="PREPROCESS", step="spans", parameters={})
    pre_span = store.entity(
        prov_type="span",
        extent=(0.9, 1.6),
        attributes={"peak_over_floor_db": 20.0, "k_db": 18.0, "signal": "preemphasised"},
    )
    store.was_generated_by(pre_span, pre_act)
    source_words = [w.id for w in store.entities("word")]
    envelope = next(e.id for e in store.entities("measurement") if e.attributes.get("name") == "energy_envelope")
    label = next(a.id for a in store.entities("assertion") if a.attributes.get("verb") == "label")
    speech_module.speech(store, "plain", cfg, run_dir=run_dir)
    used: set[str] = set()
    for activity in store._activities.values():
        if activity.node == "SPEECH":
            used.update(store.uses_of(activity.id))
    assert set(source_words) <= used
    assert envelope in used
    assert pre_span in used
    assert label not in used, "SPEECH reads no airway label; the span it hangs on is read as a span"


def test_a_word_ending_a_hair_past_the_decode_is_clamped_not_a_crash(
    seed_speech_store: SeedSpeechStore,
) -> None:
    """The diarization interval is bounded by the decode, so a float hair cannot end the branch.

    The duration here is the cluster's own — 92137 samples at 16 kHz — where a last word ending at
    the recording's end is reported by ``fuse_word_streams``, which rounds to 1e-4 s, as 5.7586.
    That is 0.6 of a sample past the decode, and ``extract_segments`` raised "End must be <=
    duration of the audio (5.7585625 sec)" on it while the same file ran clean on the Mac.
    """
    duration_s = 92137 / SR
    words = [("one", 1.0, 1.3), ("two", 5.0, round(duration_s, 4))]
    assert words[1][2] > duration_s, "the fixture must reproduce the overshoot, not merely resemble it"
    store, cfg, run_dir = seed_speech_store(words, words, duration_s=duration_s)
    result = speech_module.speech(store, "plain", cfg, run_dir=run_dir)
    assert result.verdict.outcome in (Outcome.PASS, Outcome.FLAG)
    interval = next(e for e in store.entities("interval") if e.attributes.get("name") == "diarization_interval")
    assert interval.extent is not None
    assert interval.extent[1] == duration_s, "the interval the store records is the one that was diarized"


def test_a_word_ending_a_tenth_of_a_second_past_the_decode_still_raises(
    seed_speech_store: SeedSpeechStore,
) -> None:
    """An extent that far outside the recording is an inconsistency, and must not be silently trimmed."""
    words = [("one", 1.0, 1.3), ("two", 5.0, 5.5)]
    store, cfg, run_dir = seed_speech_store(words, words, duration_s=6.0)
    stray = store.activity(node="PREPROCESS", step="asr:stray", parameters={})
    from senselab.audio.workflows.triage.nodes.preprocess import CRISPERWHISPER_ID

    word_id = store.entity(
        prov_type="word",
        extent=(5.9, 6.1),
        attributes={"text": "three", "score": 0.9, "recognizer": CRISPERWHISPER_ID, "timestamp_source": "native"},
    )
    store.was_generated_by(word_id, stray)
    with pytest.raises(ValueError, match="past the"):
        speech_module.speech(store, "plain", cfg, run_dir=run_dir)
