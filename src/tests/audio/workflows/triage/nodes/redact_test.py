"""REDACT node tests. ASR and the PII scan are faked at the node module; redaction and the store run real."""

import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pytest

from senselab.audio.data_structures import Audio
from senselab.audio.workflows.triage.config import TriageConfig, load_triage_config
from senselab.audio.workflows.triage.nodes import redact as redact_module
from senselab.audio.workflows.triage.nodes.preprocess import CRISPERWHISPER_ID as CW
from senselab.audio.workflows.triage.nodes.preprocess import QWEN_ID as QW
from senselab.audio.workflows.triage.vocabulary import Outcome
from senselab.text.tasks.pii_detection.api import PiiScan, PiiSpan
from senselab.utils.data_structures import ScriptLine
from senselab.utils.prov_store import ProvStore

SR = 16000
EDGE = 0.001

MakeRedactRun = Callable[..., tuple[ProvStore, TriageConfig, Path]]


class _FakeModel:
    """A model spec stub carrying exactly what the node reads: path_or_uri and commit_sha."""

    def __init__(self, path_or_uri: str, commit_sha: str) -> None:
        """Stub a resolved model."""
        self.path_or_uri = path_or_uri
        self.commit_sha = commit_sha


def _clean_scan() -> PiiScan:
    """A scan in which every default detector ran and found nothing."""
    return PiiScan(spans=[], detectors_used=["gliner", "presidio", "rules"], failures={})


def _scan_finding(secret: str, category: str) -> Callable[..., list[PiiScan]]:
    """A scan fake reporting one finding wherever the secret appears in a scanned text."""

    def _scan(inputs: list[str], **kw: Any) -> list[PiiScan]:  # noqa: ANN401
        scans: list[PiiScan] = []
        for text in inputs:
            if secret.lower() in text.lower():
                scans.append(
                    PiiScan(
                        spans=[PiiSpan(text=secret, category=category, source="presidio", asr_model="0")],
                        detectors_used=["gliner", "presidio", "rules"],
                        failures={},
                    )
                )
            else:
                scans.append(_clean_scan())
        return scans

    return _scan


def _failing_scan(failures: dict[str, str]) -> Callable[..., list[PiiScan]]:
    """A scan fake in which a detector did not run — could-not-check, never clean."""

    def _scan(inputs: list[str], **kw: Any) -> list[PiiScan]:  # noqa: ANN401
        return [PiiScan(spans=[], detectors_used=["presidio", "rules"], failures=dict(failures)) for _ in inputs]

    return _scan


def _verdict(store: ProvStore, result: redact_module.RedactResult) -> dict[str, Any]:
    """The verdict entity's attributes — where the node's design-named mapping lives."""
    return store.get_entity(result.verdict_entity_id).attributes


@pytest.fixture(autouse=True)
def _fake_inference(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default fakes: nothing re-transcribed, every scan clean, no HFModel constructed."""
    monkeypatch.setattr(
        redact_module,
        "transcribe_audios",
        lambda audios, model, **kw: [ScriptLine(text="", start=0.0, end=0.0)],
    )
    monkeypatch.setattr(redact_module, "scan_for_pii", lambda inputs, **kw: [_clean_scan() for _ in inputs])
    monkeypatch.setattr(
        redact_module, "_verification_model", lambda model_id, commit_sha: _FakeModel(model_id, commit_sha)
    )


@pytest.fixture(name="make_redact_run")
def _make_redact_run(seed_redact_store: MakeRedactRun) -> MakeRedactRun:
    """The shared seeder, under the name the design plan's tests use."""
    return seed_redact_store


def test_constructible_but_refuses_without_a_padding_override(make_redact_run: MakeRedactRun, tmp_path: Path) -> None:
    """redaction.padding_ms is null by design: the module imports, the call refuses at entry (N3)."""
    store, _, run_dir = make_redact_run(tmp_path, findings=(((1.0, 1.4), "PERSON"),))
    before = store.fingerprint()
    with pytest.raises(ValueError, match="redaction.padding_ms"):
        redact_module.redact(
            store, "recording", load_triage_config(), run_dir=run_dir, artifacts_dir=tmp_path / "release"
        )
    assert store.fingerprint() == before


def test_every_finding_is_redacted_regardless_of_speaker(make_redact_run: MakeRedactRun, tmp_path: Path) -> None:
    """A non-target finding SPEECH did not flag is exactly as unsafe to release."""
    store, cfg, run_dir = make_redact_run(
        tmp_path,
        findings=(((1.0, 1.4), "PERSON", "SPEAKER_00"), ((3.0, 3.5), "LOCATION", "SPEAKER_01")),
    )
    result = redact_module.redact(store, "recording", cfg, run_dir=run_dir, artifacts_dir=tmp_path / "release")
    assert _verdict(store, result)["redactions_n"] == 2
    audio = Audio(filepath=str(result.artifacts["audio"]))
    x = np.asarray(audio.waveform)[0]
    pad = cfg.require("redaction.padding_ms") / 1000.0
    for s, e in ((1.0, 1.4), (3.0, 3.5)):
        assert not x[int((s - pad + EDGE) * SR) : int((e + pad - EDGE) * SR)].any(), "silenced, padded outward"
    assert x[: int(0.5 * SR)].any(), "audio outside the redactions survives"


def test_padded_overlapping_extents_merge_and_categories_join(make_redact_run: MakeRedactRun, tmp_path: Path) -> None:
    """Two findings whose padded extents touch become one redaction.

    An audible sliver between two separate redactions is the failure merging exists to prevent.
    """
    store, cfg, run_dir = make_redact_run(
        tmp_path, findings=(((1.0, 1.2), "PERSON"), ((1.25, 1.5), "LOCATION"))
    )  # the 50 ms override padding makes them overlap
    result = redact_module.redact(store, "recording", cfg, run_dir=run_dir, artifacts_dir=tmp_path / "release")
    assert _verdict(store, result)["redactions_n"] == 1
    assert _verdict(store, result)["by_category"] == {"PERSON+LOCATION": 1}


def test_a_category_containing_plus_is_refused_by_the_node_not_discovered_later(
    make_redact_run: MakeRedactRun, tmp_path: Path
) -> None:
    """+ is reserved for merged categories; a label carrying it would silently decompose (invariant 5)."""
    store, cfg, run_dir = make_redact_run(tmp_path, findings=(((1.0, 1.4), "A+B"),))
    with pytest.raises(ValueError, match="reserved") as err:
        redact_module.redact(store, "recording", cfg, run_dir=run_dir, artifacts_dir=tmp_path / "release")
    assert "A+B" in str(err.value), "the message names the category and bounds only"
    # pii entities carry no text field at all, so the exception cannot quote a match


def test_verification_reruns_both_recognizers_and_a_surviving_finding_fails(
    make_redact_run: MakeRedactRun, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Any finding in the re-scan — survivor or new — withholds the artifact and names categories only."""
    seen_models: list[str] = []

    def fake_transcribe(audios: list[Audio], model: Any, **kw: Any) -> list[ScriptLine]:  # noqa: ANN401
        seen_models.append(str(model.path_or_uri))
        return [ScriptLine(text="jane doe", start=1.0, end=1.4)]

    monkeypatch.setattr(redact_module, "transcribe_audios", fake_transcribe)
    monkeypatch.setattr(redact_module, "scan_for_pii", _scan_finding("jane doe", "PERSON"))
    store, cfg, run_dir = make_redact_run(tmp_path, findings=(((1.0, 1.4), "PERSON"),))
    result = redact_module.redact(store, "recording", cfg, run_dir=run_dir, artifacts_dir=tmp_path / "release")
    assert result.verdict.outcome is Outcome.FAIL
    assert _verdict(store, result)["survived"] == ["PERSON"], "categories, never matched text"
    assert "jane doe" not in json.dumps(_verdict(store, result))
    assert sorted(seen_models) == sorted({CW, QW}), "both recognizers PREPROCESS used (N14)"
    assert result.artifacts == {}, "a failed verification releases nothing"


def test_verification_scans_the_redacted_transcript_alongside_the_audio(
    make_redact_run: MakeRedactRun, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A finding surviving only in the transcript artifact is caught by the same gate."""
    monkeypatch.setattr(redact_module, "scan_for_pii", _scan_finding("world", "LOCATION"))
    store, cfg, run_dir = make_redact_run(tmp_path, findings=(((1.0, 1.4), "PERSON"),))
    result = redact_module.redact(store, "recording", cfg, run_dir=run_dir, artifacts_dir=tmp_path / "release")
    # "world" is a transcript word outside every redaction; the ASR fakes transcribe nothing.
    assert result.verdict.outcome is Outcome.FAIL
    assert _verdict(store, result)["survived"] == ["LOCATION"]
    assert result.artifacts == {}


def test_a_failed_detector_during_verification_withholds(
    make_redact_run: MakeRedactRun, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """could-not-verify is fail(survived=[]) -> withheld, not a pass and not not_assessed (N16)."""
    monkeypatch.setattr(redact_module, "scan_for_pii", _failing_scan({"gliner": "load failed"}))
    store, cfg, run_dir = make_redact_run(tmp_path, findings=(((1.0, 1.4), "PERSON"),))
    result = redact_module.redact(store, "recording", cfg, run_dir=run_dir, artifacts_dir=tmp_path / "release")
    assert result.verdict.outcome is Outcome.FAIL
    assert _verdict(store, result)["survived"] == []
    assert _verdict(store, result)["verified"] is False
    assert result.artifacts == {}


def test_released_artifacts_share_no_element_ids_with_the_store(make_redact_run: MakeRedactRun, tmp_path: Path) -> None:
    """An id indexing both the store and a released artifact is a join key back to the PII."""
    store, cfg, run_dir = make_redact_run(tmp_path, findings=(((1.0, 1.4), "PERSON"),))
    result = redact_module.redact(store, "recording", cfg, run_dir=run_dir, artifacts_dir=tmp_path / "release")
    assert result.artifacts, "a verified run releases both artifacts"
    ids = [e.id for e in store.entities()]
    for path in result.artifacts.values():
        blob = path.read_bytes()
        for eid in ids:
            assert eid.encode() not in blob


def test_the_source_is_not_destroyed_and_the_store_only_grows(make_redact_run: MakeRedactRun, tmp_path: Path) -> None:
    """Redaction writes; deletion is an operator decision with its own authorisation."""
    store, cfg, run_dir = make_redact_run(tmp_path, findings=(((1.0, 1.4), "PERSON"),))
    entities_before = {e.id for e in store.entities()}
    redact_module.redact(store, "recording", cfg, run_dir=run_dir, artifacts_dir=tmp_path / "release")
    assert (run_dir / "streams" / "plain.wav").exists()
    assert entities_before <= {e.id for e in store.entities()}, "append-only: nothing removed"


def test_an_unscanned_store_is_refused_not_certified(make_redact_run: MakeRedactRun, tmp_path: Path) -> None:
    """No pii_scan measurement means 'unchecked', which must not launder into releasable (N15)."""
    store, cfg, run_dir = make_redact_run(tmp_path, findings=(), scanned=False)
    with pytest.raises(ValueError, match="no PII scan"):
        redact_module.redact(store, "recording", cfg, run_dir=run_dir, artifacts_dir=tmp_path / "release")


def test_zero_findings_still_verifies_before_passing(
    make_redact_run: MakeRedactRun, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A clean scan's artifact is verified like any other; verification is part of the node."""
    scanned: list[list[str]] = []

    def counting_scan(inputs: list[str], **kw: Any) -> list[PiiScan]:  # noqa: ANN401
        scanned.append(list(inputs))
        return [_clean_scan() for _ in inputs]

    monkeypatch.setattr(redact_module, "scan_for_pii", counting_scan)
    store, cfg, run_dir = make_redact_run(tmp_path, findings=())
    result = redact_module.redact(store, "recording", cfg, run_dir=run_dir, artifacts_dir=tmp_path / "release")
    assert result.verdict.outcome is Outcome.PASS
    assert _verdict(store, result)["verified"] is True and _verdict(store, result)["redactions_n"] == 0
    assert scanned, "the re-scan ran even with nothing to redact"
    assert result.artifacts.keys() == {"audio", "transcript"}


def test_artifacts_dir_nested_in_run_dir_is_refused(make_redact_run: MakeRedactRun, tmp_path: Path) -> None:
    """The store's directory and the release directory must not be one publish step apart."""
    store, cfg, run_dir = make_redact_run(tmp_path, findings=(((1.0, 1.4), "PERSON"),))
    with pytest.raises(ValueError, match="artifacts_dir"):
        redact_module.redact(store, "recording", cfg, run_dir=run_dir, artifacts_dir=run_dir / "release")


def test_transcript_artifact_replaces_findings_with_category_placeholders(
    make_redact_run: MakeRedactRun, tmp_path: Path
) -> None:
    """Words inside planned extents render as [CATEGORY]; padded-in neighbours go with them.

    The transcript then matches what the audio lost: no timestamps, no ids, no matched text.
    """
    store, cfg, run_dir = make_redact_run(
        tmp_path,
        findings=(((1.0, 1.4), "PERSON"),),
        words=(("my", 0.5, 0.8, "S"), ("name", 0.96, 0.99, "S"), ("jane", 1.0, 1.4, "S"), ("here", 2.0, 2.3, "S")),
    )
    result = redact_module.redact(store, "recording", cfg, run_dir=run_dir, artifacts_dir=tmp_path / "release")
    text = result.artifacts["transcript"].read_text()
    # "name" at 0.96-0.99 falls inside the 50 ms padding around the finding: it goes with the audio.
    assert text.split() == ["my", "[PERSON]", "here"]
    assert "jane" not in text and "0.9" not in text and "1.4" not in text
