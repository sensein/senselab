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
from senselab.text.tasks.pii_detection.api import scan_for_pii as real_scan_for_pii
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
    """A non-target finding SPEECH did not flag is exactly as unsafe to release.

    The store carries a known target, so a speaker-scoped reader has something to scope by: SPEECH
    flagged only SPEAKER_00's finding, and REDACT must still redact SPEAKER_01's.
    """
    store, cfg, run_dir = make_redact_run(
        tmp_path,
        findings=(((1.0, 1.4), "PERSON", "SPEAKER_00"), ((3.0, 3.5), "LOCATION", "SPEAKER_01")),
        target="SPEAKER_00",
    )
    speech_verdict = next(v for v in store.entities("verdict") if v.attributes["node"] == "SPEECH")
    assert speech_verdict.attributes["target_speaker"] == "SPEAKER_00"
    assert speech_verdict.attributes["flags"] == ["pii (PERSON) in the target speaker's speech"], "LOCATION unflagged"
    result = redact_module.redact(store, "recording", cfg, run_dir=run_dir, artifacts_dir=tmp_path / "release")
    assert _verdict(store, result)["redactions_n"] == 2
    assert _verdict(store, result)["by_category"] == {"PERSON": 1, "LOCATION": 1}
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
    seen_models: list[tuple[str, str]] = []

    def fake_transcribe(audios: list[Audio], model: Any, **kw: Any) -> list[ScriptLine]:  # noqa: ANN401
        seen_models.append((str(model.path_or_uri), str(model.commit_sha)))
        return [ScriptLine(text="jane doe", start=1.0, end=1.4)]

    monkeypatch.setattr(redact_module, "transcribe_audios", fake_transcribe)
    monkeypatch.setattr(redact_module, "scan_for_pii", _scan_finding("jane doe", "PERSON"))
    store, cfg, run_dir = make_redact_run(tmp_path, findings=(((1.0, 1.4), "PERSON"),))
    result = redact_module.redact(store, "recording", cfg, run_dir=run_dir, artifacts_dir=tmp_path / "release")
    assert result.verdict.outcome is Outcome.FAIL
    assert _verdict(store, result)["survived"] == ["PERSON"], "categories, never matched text"
    assert "jane doe" not in json.dumps(_verdict(store, result))
    assert sorted(seen_models) == sorted([(CW, "c" * 40), (QW, "d" * 40)]), (
        "both recognizers PREPROCESS used, each at the 40-hex commit the store recorded — never a ref (N14)"
    )
    assert _verdict(store, result)["verify_systems"] == sorted([CW, QW])
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


def test_a_scan_that_never_ran_is_not_read_as_a_clean_scan(
    make_redact_run: MakeRedactRun, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The real scanner's empty-input answer is ``detectors_used=[] failures={}``: nothing ran.

    An empty ``failures`` is not evidence of a scan, so the did-it-run check reads
    ``detectors_used``. Driven through the real ``scan_for_pii``, which spawns no subprocess for
    empty input, so the shape under test is the shipped one.
    """
    monkeypatch.setattr(redact_module, "scan_for_pii", real_scan_for_pii)
    store, cfg, run_dir = make_redact_run(tmp_path, findings=(), words=())
    result = redact_module.redact(store, "recording", cfg, run_dir=run_dir, artifacts_dir=tmp_path / "release")
    assert result.verdict.outcome is Outcome.FAIL
    assert _verdict(store, result)["verified"] is False
    assert _verdict(store, result)["survived"] == []
    assert result.artifacts == {}, "an unverified pair is withheld (N16)"
    assert not (tmp_path / "release").exists(), "nothing was written to the release directory"


def test_a_detector_that_ran_and_found_nothing_still_verifies(
    make_redact_run: MakeRedactRun, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The counterpart of the empty-input case: populated detectors_used, empty spans, is clean."""
    monkeypatch.setattr(
        redact_module,
        "scan_for_pii",
        lambda inputs, **kw: [PiiScan(spans=[], detectors_used=["presidio"], failures={}) for _ in inputs],
    )
    store, cfg, run_dir = make_redact_run(tmp_path, findings=(((1.0, 1.4), "PERSON"),))
    result = redact_module.redact(store, "recording", cfg, run_dir=run_dir, artifacts_dir=tmp_path / "release")
    assert result.verdict.outcome is Outcome.PASS
    assert _verdict(store, result)["verified"] is True


def test_a_store_scan_whose_detector_failed_is_withheld(make_redact_run: MakeRedactRun, tmp_path: Path) -> None:
    """SPEECH writes the pii_scan measurement even when a detector failed; presence is not evidence.

    An empty ``spans`` with a populated ``failed`` means the scan did not happen, and reading it as
    clean is the one outcome worse than not scanning (``branch-speech.md``).
    """
    store, cfg, run_dir = make_redact_run(tmp_path, findings=(((1.0, 1.4), "PERSON"),), scan_failed=("gliner",))
    result = redact_module.redact(store, "recording", cfg, run_dir=run_dir, artifacts_dir=tmp_path / "release")
    assert result.verdict.outcome is Outcome.FAIL
    assert result.artifacts == {}, "no releasable pair from an incomplete scan"
    assert _verdict(store, result)["scan_failed"] == ["gliner"], "detector names, never their messages"
    assert _verdict(store, result)["verified"] is False
    assert _verdict(store, result)["verify_systems"] == []
    assert "gliner" in result.verdict.why


def test_a_store_scan_with_no_detectors_is_withheld(make_redact_run: MakeRedactRun, tmp_path: Path) -> None:
    """An empty scanned_by is "nothing ran", whatever the measurement's presence suggests."""
    store, cfg, run_dir = make_redact_run(tmp_path, findings=(((1.0, 1.4), "PERSON"),), scanned_by=())
    result = redact_module.redact(store, "recording", cfg, run_dir=run_dir, artifacts_dir=tmp_path / "release")
    assert result.verdict.outcome is Outcome.FAIL
    assert result.artifacts == {}
    assert _verdict(store, result)["verified"] is False
    assert _verdict(store, result)["scan_failed"] == []


def test_a_failure_message_from_the_store_scan_never_reaches_the_verdict(
    make_redact_run: MakeRedactRun, tmp_path: Path
) -> None:
    """A detector's failure message may quote the scanned input, so only its name is recorded."""
    store, cfg, run_dir = make_redact_run(tmp_path, findings=(((1.0, 1.4), "PERSON"),))
    scan = next(e for e in store.entities("measurement") if e.attributes.get("name") == "pii_scan")
    scan.attributes["failed"] = {"gliner": "ValueError on 'jane doe'"}
    result = redact_module.redact(store, "recording", cfg, run_dir=run_dir, artifacts_dir=tmp_path / "release")
    assert result.verdict.outcome is Outcome.FAIL
    assert _verdict(store, result)["scan_failed"] == ["gliner"]
    assert "jane doe" not in json.dumps(_verdict(store, result)) and "jane doe" not in result.verdict.why


def test_a_negative_padding_override_is_refused_at_entry(make_redact_run: MakeRedactRun, tmp_path: Path) -> None:
    """A negative margin shrinks each extent instead of widening it, silencing nothing.

    Neither verification channel can catch it: the transcript still renders the placeholder while
    the audio keeps the name, so the check is at entry, before any store write.
    """
    store, cfg, run_dir = make_redact_run(
        tmp_path, findings=(((1.0, 1.4), "PERSON"),), config_yaml="redaction:\n  padding_ms: -300\n"
    )
    before = store.fingerprint()
    with pytest.raises(ValueError, match="redaction.padding_ms"):
        redact_module.redact(store, "recording", cfg, run_dir=run_dir, artifacts_dir=tmp_path / "release")
    assert store.fingerprint() == before
    assert not (tmp_path / "release").exists()


def test_a_fractional_padding_override_is_refused_rather_than_truncated(
    make_redact_run: MakeRedactRun, tmp_path: Path
) -> None:
    """int() would silently truncate 49.9 ms to 49; a fractional margin is a typo, not a value."""
    store, cfg, run_dir = make_redact_run(
        tmp_path, findings=(((1.0, 1.4), "PERSON"),), config_yaml="redaction:\n  padding_ms: 49.9\n"
    )
    before = store.fingerprint()
    with pytest.raises(ValueError, match="redaction.padding_ms"):
        redact_module.redact(store, "recording", cfg, run_dir=run_dir, artifacts_dir=tmp_path / "release")
    assert store.fingerprint() == before


def test_an_int_valued_float_padding_override_is_accepted(make_redact_run: MakeRedactRun, tmp_path: Path) -> None:
    """YAML renders 50.0 as a float; it is the same margin as 50 and is not a typo."""
    store, cfg, run_dir = make_redact_run(
        tmp_path, findings=(((1.0, 1.4), "PERSON"),), config_yaml="redaction:\n  padding_ms: 50.0\n"
    )
    result = redact_module.redact(store, "recording", cfg, run_dir=run_dir, artifacts_dir=tmp_path / "release")
    assert _verdict(store, result)["padding_ms"] == 50
    assert isinstance(_verdict(store, result)["padding_ms"], int)


def test_a_non_numeric_padding_override_is_refused(make_redact_run: MakeRedactRun, tmp_path: Path) -> None:
    """A string margin would reach plan_redactions as a string and divide by 1000 there."""
    store, cfg, run_dir = make_redact_run(
        tmp_path, findings=(((1.0, 1.4), "PERSON"),), config_yaml='redaction:\n  padding_ms: "50"\n'
    )
    with pytest.raises(ValueError, match="redaction.padding_ms"):
        redact_module.redact(store, "recording", cfg, run_dir=run_dir, artifacts_dir=tmp_path / "release")


def test_one_reverifiable_recognizer_flags_rather_than_passes(make_redact_run: MakeRedactRun, tmp_path: Path) -> None:
    """Verification on half the recognizers is a weaker check, so it is a flag and says which ran."""
    store, cfg, run_dir = make_redact_run(tmp_path, findings=(((1.0, 1.4), "PERSON"),), commitless=(QW,))
    result = redact_module.redact(store, "recording", cfg, run_dir=run_dir, artifacts_dir=tmp_path / "release")
    assert result.verdict.outcome is Outcome.FLAG
    assert _verdict(store, result)["verify_systems"] == [CW], "the recognizer that could be re-run, named"
    assert _verdict(store, result)["verified"] is True
    assert QW in result.verdict.why, "the recognizer that could not be re-run is named too"


def test_both_reverifiable_recognizers_pass_rather_than_flag(make_redact_run: MakeRedactRun, tmp_path: Path) -> None:
    """The control for the degraded case: two of two is the undegraded check."""
    store, cfg, run_dir = make_redact_run(tmp_path, findings=(((1.0, 1.4), "PERSON"),))
    result = redact_module.redact(store, "recording", cfg, run_dir=run_dir, artifacts_dir=tmp_path / "release")
    assert result.verdict.outcome is Outcome.PASS
    assert _verdict(store, result)["verify_systems"] == sorted([CW, QW])


def test_zero_reverifiable_recognizers_raise_rather_than_release(
    make_redact_run: MakeRedactRun, tmp_path: Path
) -> None:
    """No recognizer to re-run means no verification at all, which must never release a pair."""
    store, cfg, run_dir = make_redact_run(tmp_path, findings=(((1.0, 1.4), "PERSON"),), commitless=(CW, QW))
    with pytest.raises(ValueError, match="re-verify"):
        redact_module.redact(store, "recording", cfg, run_dir=run_dir, artifacts_dir=tmp_path / "release")
    assert not (tmp_path / "release").exists(), "no artifact may exist when nothing verified"


def test_a_non_finite_padding_override_is_refused_naming_the_key(
    make_redact_run: MakeRedactRun, tmp_path: Path
) -> None:
    """A YAML .inf reaches the same entry check as a negative value and is named the same way."""
    store, cfg, run_dir = make_redact_run(
        tmp_path, findings=(((1.0, 1.4), "PERSON"),), config_yaml="redaction:\n  padding_ms: .inf\n"
    )
    before = store.fingerprint()
    with pytest.raises(ValueError, match="redaction.padding_ms"):
        redact_module.redact(store, "recording", cfg, run_dir=run_dir, artifacts_dir=tmp_path / "release")
    assert store.fingerprint() == before


def test_an_invalidated_finding_is_not_redacted_and_not_derived_from(
    make_redact_run: MakeRedactRun, tmp_path: Path
) -> None:
    """The store's latest-non-invalidated rule applies to findings as it does to streams."""
    store, cfg, run_dir = make_redact_run(tmp_path, findings=(((1.0, 1.4), "PERSON"), ((3.0, 3.5), "LOCATION")))
    withdrawn = next(e for e in store.entities("pii") if e.attributes["category"] == "LOCATION")
    retraction = store.activity(node="SPEECH", step="retract", parameters={})
    store.was_invalidated_by(withdrawn.id, retraction)
    result = redact_module.redact(store, "recording", cfg, run_dir=run_dir, artifacts_dir=tmp_path / "release")
    assert _verdict(store, result)["redactions_n"] == 1
    assert _verdict(store, result)["by_category"] == {"PERSON": 1}
    spans = [e for e in store.entities("span") if e.attributes.get("name") == "redaction"]
    assert all(withdrawn.id not in store.derived_from(span.id) for span in spans)
    audio = Audio(filepath=str(result.artifacts["audio"]))
    x = np.asarray(audio.waveform)[0]
    assert x[int(3.1 * SR) : int(3.4 * SR)].any(), "the withdrawn finding's region is untouched"


def test_a_word_with_no_extent_is_not_released_verbatim(make_redact_run: MakeRedactRun, tmp_path: Path) -> None:
    """Text whose location is unknown overlaps no redaction, so it cannot be shown to be safe."""
    store, cfg, run_dir = make_redact_run(tmp_path, findings=(((1.0, 1.4), "PERSON"),))
    placed = next(w for w in store.entities("word") if w.attributes.get("speaker"))  # SPEECH's, not PREPROCESS's
    speech_act = store.generated_by(placed.id)
    assert speech_act is not None
    floating = store.entity(
        prov_type="word", extent=None, attributes={"text": "unplaceable-sentinel", "speaker": "SPEAKER_00"}
    )
    store.was_generated_by(floating, speech_act)
    result = redact_module.redact(store, "recording", cfg, run_dir=run_dir, artifacts_dir=tmp_path / "release")
    text = result.artifacts["transcript"].read_text()
    assert "unplaceable-sentinel" not in text
    assert "[UNPLACED]" in text
    assert _verdict(store, result)["unplaced_words_n"] == 1


def test_a_placed_transcript_counts_no_unplaced_words(make_redact_run: MakeRedactRun, tmp_path: Path) -> None:
    """The control: every word carrying an extent leaves the count at zero."""
    store, cfg, run_dir = make_redact_run(tmp_path, findings=(((1.0, 1.4), "PERSON"),))
    result = redact_module.redact(store, "recording", cfg, run_dir=run_dir, artifacts_dir=tmp_path / "release")
    assert _verdict(store, result)["unplaced_words_n"] == 0
    assert "[UNPLACED]" not in result.artifacts["transcript"].read_text()


def test_the_store_records_the_three_activities_the_spans_and_every_read(
    make_redact_run: MakeRedactRun, tmp_path: Path
) -> None:
    """One activity per phase, one span entity per planned extent, and a used edge per read element."""
    store, cfg, run_dir = make_redact_run(
        tmp_path, findings=(((1.0, 1.4), "PERSON"), ((3.0, 3.5), "LOCATION")), target="SPEAKER_00"
    )
    findings = {e.attributes["category"]: e.id for e in store.entities("pii")}
    scan = next(e for e in store.entities("measurement") if e.attributes.get("name") == "pii_scan")
    recording = next(e for e in store.entities("stream") if e.attributes.get("name") == "recording")
    speech_words = {w.id for w in store.entities("word") if w.attributes.get("speaker")}
    result = redact_module.redact(store, "recording", cfg, run_dir=run_dir, artifacts_dir=tmp_path / "release")

    activities = [a for a in store._activities.values() if a.node == "REDACT"]
    assert sorted(str(a.step) for a in activities) == ["apply", "plan", "verify"]
    plan_act = next(a for a in activities if a.step == "plan")
    apply_act = next(a for a in activities if a.step == "apply")
    verify_act = next(a for a in activities if a.step == "verify")

    spans = [e for e in store.entities("span") if e.attributes.get("name") == "redaction"]
    assert len(spans) == 2, "one span entity per planned extent"
    for span in spans:
        assert set(span.attributes) == {"name", "category"}, "the span carries a name and a category, nothing else"
        assert store.generated_by(span.id) == plan_act.id
        assert span.id in result.view
    by_category = {str(span.attributes["category"]): span for span in spans}
    for category, finding_id in findings.items():
        assert store.derived_from(by_category[category].id) == [finding_id], "derived from the pii it covers"

    assert set(store.uses_of(plan_act.id)) == {scan.id, *findings.values()}
    apply_used = set(store.uses_of(apply_act.id))
    assert recording.id in apply_used, "the recording stream it redacted"
    assert speech_words <= apply_used, "every consensus word the transcript read"
    assert {span.id for span in spans} <= apply_used

    verdict = store.get_entity(result.verdict_entity_id)
    assert store.generated_by(verdict.id) == verify_act.id
    software = next(a.id for a in store._agents.values() if a.agent_type == "software")
    for activity_id in (plan_act.id, apply_act.id, verify_act.id):
        assert software in store.associated_with(activity_id)
    model_agents = {
        store.get_agent(a).model_id for a in store.associated_with(verify_act.id) if store.get_agent(a).model_id
    }
    assert model_agents == {CW, QW}, "verification is answerable to the recognizers it re-ran"


def test_a_recognizer_whose_asr_died_in_preprocess_still_degrades_the_check(
    make_redact_run: MakeRedactRun, tmp_path: Path
) -> None:
    """A recognizer that wrote no word must not vanish from the expected set.

    Deriving the expected set from words made a dead recognizer indistinguishable from one that
    was never declared, so verification on one of two recognizers read as undegraded and released
    the pair. The expected set is PREPROCESS's declared systems.
    """
    store, cfg, run_dir = make_redact_run(tmp_path, findings=(((1.0, 1.4), "PERSON"),), wordless=(QW,))
    result = redact_module.redact(store, "recording", cfg, run_dir=run_dir, artifacts_dir=tmp_path / "release")
    assert result.verdict.outcome is Outcome.FLAG, "one of two recognizers is a degraded check, never a pass"
    assert _verdict(store, result)["verify_systems"] == [CW]
    assert _verdict(store, result)["expected_source"] == "preprocess"
    assert QW in result.verdict.why, "the recognizer that could not be re-run is named"


def test_the_expected_set_falls_back_to_words_and_says_so(make_redact_run: MakeRedactRun, tmp_path: Path) -> None:
    """A store carrying no PREPROCESS declaration is read from its words, and the verdict records that."""
    store, cfg, run_dir = make_redact_run(tmp_path, findings=(((1.0, 1.4), "PERSON"),), declared=False)
    result = redact_module.redact(store, "recording", cfg, run_dir=run_dir, artifacts_dir=tmp_path / "release")
    assert _verdict(store, result)["expected_source"] == "words"
