"""REDACT node tests. The PII scan is faked at the node module; redaction and the store run real.

Nothing here fakes a recognizer, because REDACT runs none: verification is a re-scan of the redacted
consensus text. The seeder writes PREPROCESS's consensus words and SPEECH's findings, which are the
only two authors REDACT reads.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pytest
import soundfile as sf

from senselab.audio.data_structures import Audio
from senselab.audio.data_structures.audio_hints import AudioHints, ExpectedSpeech
from senselab.audio.workflows.triage.config import TriageConfig, load_triage_config
from senselab.audio.workflows.triage.nodes import redact as redact_module
from senselab.audio.workflows.triage.nodes.common import resolve_stream
from senselab.audio.workflows.triage.nodes.redact import STREAM_NAME, redact
from senselab.audio.workflows.triage.nodes.verdict import verdict as verdict_node
from senselab.audio.workflows.triage.vocabulary import LLM_REDACTION_RESIDUE, Outcome, Release, Triage
from senselab.text.tasks.pii_detection.api import PiiScan, PiiSpan
from senselab.text.tasks.pii_detection.api import scan_for_pii as real_scan_for_pii
from senselab.text.tasks.pii_detection.redaction_review import ReviewFinding, ReviewResult, parse_completion
from senselab.utils.prov_store import Entity, ProvStore
from tests.audio.workflows.triage.nodes.conftest import word_attributes

SR = 16000
EDGE = 0.001
WORD_STRIDE_S = 1.0  # one word per second, so a 50 ms margin never reaches a neighbour
WORD_LENGTH_S = 0.5
ALL_DETECTORS = ("gliner", "presidio", "rules")


def _release(tmp_path: Path) -> Path:
    """A release directory disjoint from the run directory, which is ``tmp_path`` itself."""
    return tmp_path.parent / f"{tmp_path.name}-release"


def _override(tmp_path: Path, yaml_text: str) -> TriageConfig:
    """The production override mechanism: a partial YAML deep-merged over the packaged config."""
    path = tmp_path / f"override-{abs(hash(yaml_text))}.yaml"
    path.write_text(yaml_text)
    return load_triage_config(path)


@pytest.fixture(name="redact_config")
def _redact_config(tmp_path: Path) -> TriageConfig:
    """The two keys every REDACT call needs, neither of which has a packaged default."""
    return _override(tmp_path, "redaction:\n  padding_ms: 50\n  fill: silence\n")


def _verdict_entity(store: ProvStore, node: str) -> Entity:
    """The last verdict entity a node wrote."""
    return [e for e in store.entities("verdict") if e.attributes.get("node") == node][-1]


def _scan(findings: Sequence[tuple[str, str]], detectors_used: Sequence[str], failures: dict[str, str]) -> PiiScan:
    """One scan result from ``(category, text)`` pairs."""
    return PiiScan(
        spans=[PiiSpan(text=text, category=category, source="presidio", asr_model="0") for category, text in findings],
        detectors_used=list(detectors_used),
        failures=dict(failures),
    )


def _stub_pii(
    monkeypatch: pytest.MonkeyPatch,
    *,
    findings: Sequence[tuple[str, str]],
    detectors_used: Sequence[str] = ALL_DETECTORS,
    failures: dict[str, str] | None = None,
) -> list[str]:
    """Replace the node's scanner with one fixed answer, recording every text it was handed."""
    scanned: list[str] = []

    def _fake(inputs: Any, **kw: Any) -> PiiScan:  # noqa: ANN401
        scanned.append(str(inputs))
        return _scan(findings, detectors_used, failures or {})

    monkeypatch.setattr(redact_module, "scan_for_pii", _fake)
    return scanned


def _stub_pii_sequence(monkeypatch: pytest.MonkeyPatch, rounds: Sequence[Sequence[tuple[str, str]]]) -> list[str]:
    """Replace the node's scanner with one answer per call, in order, recording each scanned text."""
    scanned: list[str] = []
    remaining = list(rounds)

    def _fake(inputs: Any, **kw: Any) -> PiiScan:  # noqa: ANN401
        scanned.append(str(inputs))
        assert remaining, "the node scanned more times than the test declared answers for"
        return _scan(remaining.pop(0), ALL_DETECTORS, {})

    monkeypatch.setattr(redact_module, "scan_for_pii", _fake)
    return scanned


def _word_extent(index: int) -> tuple[float, float]:
    """Where the seeder puts the ``index``-th consensus word."""
    return (index * WORD_STRIDE_S, index * WORD_STRIDE_S + WORD_LENGTH_S)


def _seed_redact_store(  # noqa: C901 — one independent block per author, as the store has
    store: ProvStore,
    tmp_path: Path,
    *,
    words: Sequence[str] = ("hello", "alice"),
    findings: Sequence[tuple[Any, ...]] = (),
    extra_marks: Sequence[tuple[str, str]] = (),
    target_speaker: str | None = None,
    scanned: bool = True,
    scanned_by: Sequence[str] = ALL_DETECTORS,
    scan_failed: Sequence[str] = (),
) -> None:
    """Write the store PREPROCESS and SPEECH leave for REDACT, with ``tmp_path`` as the run dir.

    ``findings`` are ``(category, (start, end))`` or ``(category, (start, end), speaker)``. Each
    writes a ``pii`` entity and a ``label``/``pii`` assertion derived from every consensus word it
    overlaps — the store's shared shape for a marking. ``extra_marks`` are ``(word_text, category)``
    markings placed on a word the finding's own extent does not reach, which is the state a
    re-planning pass exists to widen. ``target_speaker`` writes SPEECH's verdict so a speaker-scoped
    reader has something to scope by.
    """
    ends = [_word_extent(i)[1] for i in range(len(words))] + [float(extent[1]) for _c, extent, *_r in findings]
    duration_s = max([5.0, *(end + 1.0 for end in ends)])
    rng = np.random.default_rng(0)
    wave = (0.05 * rng.standard_normal(int(duration_s * SR))).astype(np.float32)
    (tmp_path / "streams").mkdir(parents=True, exist_ok=True)
    sf.write(str(tmp_path / "streams" / "plain.wav"), wave, SR)

    software = store.agent(agent_type="software", version="senselab test-seed")
    pre = store.activity(node="PREPROCESS", step="condition", parameters={})
    store.was_associated_with(pre, software)
    for name in ("recording", "plain"):
        stream_id = store.entity(
            prov_type="stream",
            extent=(0.0, duration_s),
            attributes={"name": name, "path": "streams/plain.wav", "sampling_rate": SR, "channels": 1},
        )
        store.was_generated_by(stream_id, pre)

    consensus = store.activity(node="PREPROCESS", step="consensus", parameters={})
    store.was_associated_with(consensus, software)
    word_ids: list[str] = []
    for index, text in enumerate(words):
        word_id = store.entity(
            prov_type="word",
            extent=_word_extent(index),
            attributes=word_attributes(text, _word_extent(index), index=index),
        )
        store.was_generated_by(word_id, consensus)
        word_ids.append(word_id)
    transcript_id = store.entity(
        prov_type="measurement",
        extent=None,
        attributes={
            "name": "consensus_transcript",
            "signal": "plain",
            "role": "consensus",
            "n_words": len(words),
            "word_ids": word_ids,
            "text": " ".join(words),
        },
    )
    store.was_generated_by(transcript_id, consensus)

    pii_act = store.activity(node="SPEECH", step="pii", parameters={})
    store.was_associated_with(pii_act, software)

    def _mark(category: str, extent: tuple[float, float], covered: Iterable[str]) -> None:
        """One ``label``/``pii`` assertion, derived from each word it is about."""
        mark_id = store.entity(
            prov_type="assertion",
            extent=extent,
            attributes={"verb": "label", "label": "pii", "category": category},
        )
        store.was_generated_by(mark_id, pii_act)
        for word_id in covered:
            store.was_derived_from(mark_id, word_id)

    for category, extent, *_rest in findings:
        bounds = (float(extent[0]), float(extent[1]))
        pii_id = store.entity(
            prov_type="pii",
            extent=bounds,
            attributes={
                "category": category,
                "source": "presidio",
                "detectors_used": list(scanned_by),
                "detectors_failed": list(scan_failed),
            },
        )
        store.was_generated_by(pii_id, pii_act)
        _mark(
            str(category),
            bounds,
            [
                word_ids[i]
                for i in range(len(words))
                if _word_extent(i)[0] < bounds[1] and _word_extent(i)[1] > bounds[0]
            ],
        )
    for text, category in extra_marks:
        index = list(words).index(text)
        _mark(str(category), _word_extent(index), [word_ids[index]])

    if scanned:
        scan_id = store.entity(
            prov_type="measurement",
            extent=None,
            attributes={"name": "pii_scan", "scanned_by": list(scanned_by), "failed": list(scan_failed)},
        )
        store.was_generated_by(scan_id, pii_act)

    if target_speaker is not None:
        flagged = [
            f"pii ({category}) in the target speaker's speech"
            for category, _extent, *rest in findings
            if (rest[0] if rest else target_speaker) == target_speaker
        ]
        verdict_id = store.entity(
            prov_type="verdict",
            extent=None,
            attributes={
                "node": "SPEECH",
                "outcome": "flag" if flagged else "pass",
                "kind": "speech",
                "why": "; ".join(flagged) or "words, spans, speakers and quality are in the store",
                "target_speaker": target_speaker,
                "flags": flagged,
            },
        )
        store.was_generated_by(verdict_id, store.activity(node="SPEECH", step="transcript", parameters={}))


class TestVerificationDoesNotReTranscribe:
    """A re-decode is a second sample of a different signal, not a check on this one."""

    def test_the_module_cannot_transcribe(self) -> None:
        """The recognizer import is deleted, not left unreachable."""
        assert not hasattr(redact_module, "transcribe_audios")

    def test_verification_re_scans_the_redacted_text(
        self,
        store: ProvStore,
        redact_config: TriageConfig,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Exactly one text is re-scanned, and it is the transcript the plan produced."""
        _seed_redact_store(store, tmp_path, words=["my", "name", "is", "alice"], findings=[("PERSON", (3.0, 4.0))])
        scanned = _stub_pii(monkeypatch, findings=[])
        result = redact(store, "recording", redact_config, run_dir=tmp_path, artifacts_dir=_release(tmp_path))
        assert result.verdict.outcome is Outcome.PASS
        assert scanned == ["my name is [PERSON]"]

    def test_the_verify_activity_names_no_model_agent(
        self,
        store: ProvStore,
        redact_config: TriageConfig,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Nothing here runs at a commit, because nothing here runs a model."""
        _seed_redact_store(store, tmp_path, words=["hello", "alice"], findings=[("PERSON", (1.0, 2.0))])
        _stub_pii(monkeypatch, findings=[])
        redact(store, "recording", redact_config, run_dir=tmp_path, artifacts_dir=_release(tmp_path))
        verify = next(a for a in store.activities("REDACT") if a.step == "verify")
        assert not [agent for agent in store.associated_with(verify.id) if store.get_agent(agent).agent_type == "model"]

    def test_the_audio_claim_is_bounded_on_every_path(
        self,
        store: ProvStore,
        redact_config: TriageConfig,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A text re-scan cannot answer whether intelligible speech survives outside the extent."""
        _seed_redact_store(store, tmp_path, words=["hello", "alice"], findings=[("PERSON", (1.0, 2.0))])
        for survivors in ([], [("PERSON", "alice")]):
            other = ProvStore(run_id="bounded")
            _seed_redact_store(other, tmp_path, words=["hello", "alice"], findings=[("PERSON", (1.0, 2.0))])
            _stub_pii(monkeypatch, findings=survivors)
            redact(other, "recording", redact_config, run_dir=tmp_path, artifacts_dir=_release(tmp_path))
            assert _verdict_entity(other, "REDACT").attributes["audio_check"] == "bounded"


class TestRemediationHappensExactlyOnce:
    """A finding the planner placed and the verifier still sees gets one re-planning pass."""

    def test_a_survivor_triggers_one_replan(
        self,
        store: ProvStore,
        redact_config: TriageConfig,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The verifier's extent is fed back once, and a clean second scan passes."""
        _seed_redact_store(store, tmp_path, words=["hello", "alice"], findings=[("PERSON", (1.0, 2.0))])
        _stub_pii_sequence(monkeypatch, [[("PERSON", "alice")], []])
        result = redact(store, "recording", redact_config, run_dir=tmp_path, artifacts_dir=_release(tmp_path))
        assert result.verdict.outcome is Outcome.PASS
        assert _verdict_entity(store, "REDACT").attributes["replanned_n"] == 1

    def test_the_replan_widens_to_a_marked_word_the_first_plan_missed(
        self,
        store: ProvStore,
        redact_config: TriageConfig,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The remediation is a widening, not a re-run of the same extents.

        The finding's extent reaches ``jane`` and stops; ``doe`` carries the same marking a second
        away, so the first pass releases it verbatim and the verifier still sees a PERSON.
        """
        _seed_redact_store(
            store,
            tmp_path,
            words=["jane", "doe", "here"],
            findings=[("PERSON", (0.0, 0.5))],
            extra_marks=[("doe", "PERSON")],
        )
        scanned = _stub_pii_sequence(monkeypatch, [[("PERSON", "doe")], []])
        result = redact(store, "recording", redact_config, run_dir=tmp_path, artifacts_dir=_release(tmp_path))
        assert scanned[0] == "[PERSON] doe here", "the first pass released the second marked word"
        assert scanned[1] == "[PERSON] [PERSON] here", "the re-plan covered it"
        assert result.verdict.outcome is Outcome.PASS
        detail = _verdict_entity(store, "REDACT").attributes
        assert detail["replanned_n"] == 1 and detail["redactions_n"] == 2
        assert detail["unremediable"] == []

    def test_the_replan_does_not_widen_to_a_word_a_planned_extent_already_covers(
        self,
        store: ProvStore,
        redact_config: TriageConfig,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The exclusion half of the clause, which the widening test alone does not reach.

        ``boston`` carries a LOCATION marking and sits **inside** the planned PERSON extent, so the
        surviving LOCATION has nothing to widen to and the plan must come out unchanged. Without the
        exclusion the re-plan would add ``boston``'s own extent, which merges into the PERSON one and
        renames the category — so ``by_category`` is the observable that tells the two apart.
        """
        _seed_redact_store(
            store,
            tmp_path,
            words=["hello", "alicia", "boston"],
            findings=[("PERSON", (1.0, 2.5))],
            extra_marks=[("boston", "LOCATION")],
        )
        scanned = _stub_pii(monkeypatch, findings=[("LOCATION", "boston")])
        result = redact(store, "recording", redact_config, run_dir=tmp_path, artifacts_dir=_release(tmp_path))
        detail = _verdict_entity(store, "REDACT").attributes
        assert detail["by_category"] == {"PERSON": 1}, "a covered word must not be re-planned as its own extent"
        assert detail["redactions_n"] == 1
        assert scanned == ["hello [PERSON]", "hello [PERSON]"], "the re-plan changed nothing to scan"
        assert detail["replanned_n"] == 1 and detail["unremediable"] == ["LOCATION"]
        assert result.verdict.outcome is Outcome.FAIL

    def test_a_failed_and_a_missing_verify_detector_are_reported_apart(
        self,
        store: ProvStore,
        redact_config: TriageConfig,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """'It broke' and 'nobody ran it' are different findings; the second is the silent one (M6)."""
        _seed_redact_store(store, tmp_path, words=["hello", "alice"], findings=[("PERSON", (1.0, 2.0))])
        _stub_pii(monkeypatch, findings=[], detectors_used=["presidio"], failures={"gliner": "OSError: x"})
        result = redact(store, "recording", redact_config, run_dir=tmp_path, artifacts_dir=_release(tmp_path))
        assert result.verdict.outcome is Outcome.FLAG
        detail = _verdict_entity(store, "REDACT").attributes
        assert detail["verify_failed"] == ["gliner"]
        assert detail["verify_missing"] == ["rules"]
        assert detail["scan_failed"] == [] and detail["scan_missing"] == []
        assert "OSError: x" not in str(detail)

    def test_a_survivor_of_the_replan_is_unremediable(
        self,
        store: ProvStore,
        redact_config: TriageConfig,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """An operator must be able to tell this from an ordinary withhold."""
        _seed_redact_store(store, tmp_path, words=["hello", "alice"], findings=[("PERSON", (1.0, 2.0))])
        _stub_pii_sequence(monkeypatch, [[("PERSON", "alice")], [("PERSON", "alice")]])
        result = redact(store, "recording", redact_config, run_dir=tmp_path, artifacts_dir=_release(tmp_path))
        assert result.verdict.outcome is Outcome.FAIL
        detail = _verdict_entity(store, "REDACT").attributes
        assert detail["unremediable"] == ["PERSON"]
        assert detail["survived"] == ["PERSON"]
        assert result.artifacts == {}

    def test_remediation_stops_after_one_pass(
        self,
        store: ProvStore,
        redact_config: TriageConfig,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Exactly two scans, never a third: the answer stands after the single re-plan."""
        _seed_redact_store(store, tmp_path, words=["hello", "alice"], findings=[("PERSON", (1.0, 2.0))])
        scanned = _stub_pii(monkeypatch, findings=[("PERSON", "alice")])
        redact(store, "recording", redact_config, run_dir=tmp_path, artifacts_dir=_release(tmp_path))
        assert len(scanned) == 2

    def test_a_clean_first_scan_never_replans(
        self,
        store: ProvStore,
        redact_config: TriageConfig,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The control: nothing survived, so there is nothing to widen to."""
        _seed_redact_store(store, tmp_path, words=["hello", "alice"], findings=[("PERSON", (1.0, 2.0))])
        scanned = _stub_pii(monkeypatch, findings=[])
        redact(store, "recording", redact_config, run_dir=tmp_path, artifacts_dir=_release(tmp_path))
        assert len(scanned) == 1
        assert _verdict_entity(store, "REDACT").attributes["replanned_n"] == 0


class TestTheFillIsDeclared:
    """A run declares the fill it used, and the verdict records it."""

    def test_a_null_fill_refuses_before_any_store_write(self, store: ProvStore, tmp_path: Path) -> None:
        """An override may null the shipped fill; two artifacts under different fills are not comparable."""
        config = _override(tmp_path, "redaction:\n  padding_ms: 50\n  fill: null\n")
        _seed_redact_store(store, tmp_path, words=["hello", "alice"], findings=[("PERSON", (1.0, 2.0))])
        before = len(store.entities())
        with pytest.raises(ValueError, match="redaction.fill"):
            redact(store, "recording", config, run_dir=tmp_path, artifacts_dir=_release(tmp_path))
        assert len(store.entities()) == before

    def test_the_verdict_records_the_fill(
        self,
        store: ProvStore,
        redact_config: TriageConfig,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """So two artifacts made under different fills are never compared as one."""
        _seed_redact_store(store, tmp_path, words=["hello", "alice"], findings=[("PERSON", (1.0, 2.0))])
        _stub_pii(monkeypatch, findings=[])
        redact(store, "recording", redact_config, run_dir=tmp_path, artifacts_dir=_release(tmp_path))
        assert _verdict_entity(store, "REDACT").attributes["fill"] == "silence"

    def test_bleep_is_reachable_by_config(
        self,
        store: ProvStore,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Both implemented fills are selectable; neither is a default."""
        config = _override(tmp_path, "redaction:\n  padding_ms: 100\n  fill: bleep\n")
        _seed_redact_store(store, tmp_path, words=["hello", "alice"], findings=[("PERSON", (1.0, 2.0))])
        _stub_pii(monkeypatch, findings=[])
        redact(store, "recording", config, run_dir=tmp_path, artifacts_dir=_release(tmp_path))
        assert _verdict_entity(store, "REDACT").attributes["fill"] == "bleep"

    def test_the_bleep_reaches_the_released_audio(
        self,
        store: ProvStore,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A declared bleep must be audible in the artifact, not merely recorded in the verdict."""
        config = _override(tmp_path, "redaction:\n  padding_ms: 50\n  fill: bleep\n")
        _seed_redact_store(store, tmp_path, words=["hello", "alice"], findings=[("PERSON", (1.0, 2.0))])
        _stub_pii(monkeypatch, findings=[])
        result = redact(store, "recording", config, run_dir=tmp_path, artifacts_dir=_release(tmp_path))
        x = np.asarray(Audio(filepath=str(result.artifacts["audio"])).waveform)[0]
        assert x[int(1.2 * SR) : int(1.8 * SR)].any(), "a bleep masks the extent rather than emptying it"


class TestItRedactsEverySpeaker:
    """SPEECH flags target-speaker PII; redaction is about whether an artifact is releasable."""

    def test_a_non_target_finding_is_redacted(
        self,
        store: ProvStore,
        redact_config: TriageConfig,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A non-target speaker naming the participant is exactly as unsafe."""
        _seed_redact_store(
            store,
            tmp_path,
            words=["hello", "alice"],
            findings=[("PERSON", (1.0, 2.0), "SPEAKER_01")],
            target_speaker="SPEAKER_00",
        )
        _stub_pii(monkeypatch, findings=[])
        redact(store, "recording", redact_config, run_dir=tmp_path, artifacts_dir=_release(tmp_path))
        assert _verdict_entity(store, "REDACT").attributes["redactions_n"] == 1

    def test_every_finding_is_redacted_whatever_speech_flagged(
        self,
        store: ProvStore,
        redact_config: TriageConfig,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """SPEECH flagged one of two findings; both are silenced in the released audio."""
        _seed_redact_store(
            store,
            tmp_path,
            words=["hello", "alice", "in", "boston"],
            findings=[("PERSON", (1.0, 1.5), "SPEAKER_00"), ("LOCATION", (3.0, 3.5), "SPEAKER_01")],
            target_speaker="SPEAKER_00",
        )
        speech = _verdict_entity(store, "SPEECH").attributes
        assert speech["flags"] == ["pii (PERSON) in the target speaker's speech"], "LOCATION unflagged"
        _stub_pii(monkeypatch, findings=[])
        result = redact(store, "recording", redact_config, run_dir=tmp_path, artifacts_dir=_release(tmp_path))
        detail = _verdict_entity(store, "REDACT").attributes
        assert detail["redactions_n"] == 2
        assert detail["by_category"] == {"PERSON": 1, "LOCATION": 1}
        x = np.asarray(Audio(filepath=str(result.artifacts["audio"])).waveform)[0]
        pad = 50 / 1000.0
        for start, end in ((1.0, 1.5), (3.0, 3.5)):
            assert not x[int((start - pad + EDGE) * SR) : int((end + pad - EDGE) * SR)].any(), "silenced, padded out"
        assert x[: int(0.4 * SR)].any(), "audio outside the redactions survives"


class TestPlanning:
    """Padding, merging and the reserved category character, at the node's own boundary."""

    def test_padded_overlapping_extents_merge_and_categories_join(
        self,
        store: ProvStore,
        redact_config: TriageConfig,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """An audible sliver between two separate redactions is the failure merging prevents."""
        _seed_redact_store(
            store, tmp_path, words=["hello", "alice"], findings=[("PERSON", (1.0, 1.2)), ("LOCATION", (1.25, 1.5))]
        )
        _stub_pii(monkeypatch, findings=[])
        redact(store, "recording", redact_config, run_dir=tmp_path, artifacts_dir=_release(tmp_path))
        detail = _verdict_entity(store, "REDACT").attributes
        assert detail["redactions_n"] == 1
        assert detail["by_category"] == {"PERSON+LOCATION": 1}

    def test_a_category_containing_plus_is_refused_by_the_node_not_discovered_later(
        self, store: ProvStore, redact_config: TriageConfig, tmp_path: Path
    ) -> None:
        """+ is reserved for merged categories; a label carrying it would silently decompose."""
        _seed_redact_store(store, tmp_path, words=["hello", "alice"], findings=[("A+B", (1.0, 1.4))])
        with pytest.raises(ValueError, match="reserved") as err:
            redact(store, "recording", redact_config, run_dir=tmp_path, artifacts_dir=_release(tmp_path))
        assert "A+B" in str(err.value), "the message names the category and bounds only"

    def test_an_invalidated_finding_is_not_redacted_and_not_derived_from(
        self,
        store: ProvStore,
        redact_config: TriageConfig,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The store's latest-non-invalidated rule applies to findings as it does to streams."""
        _seed_redact_store(
            store,
            tmp_path,
            words=["hello", "alice", "in", "boston"],
            findings=[("PERSON", (1.0, 1.5)), ("LOCATION", (3.0, 3.5))],
        )
        withdrawn = next(e for e in store.entities("pii") if e.attributes["category"] == "LOCATION")
        store.was_invalidated_by(withdrawn.id, store.activity(node="SPEECH", step="retract", parameters={}))
        _stub_pii(monkeypatch, findings=[])
        result = redact(store, "recording", redact_config, run_dir=tmp_path, artifacts_dir=_release(tmp_path))
        detail = _verdict_entity(store, "REDACT").attributes
        assert detail["redactions_n"] == 1 and detail["by_category"] == {"PERSON": 1}
        spans = [e for e in store.entities("span") if e.attributes.get("name") == "redaction"]
        assert all(withdrawn.id not in store.derived_from(span.id) for span in spans)
        x = np.asarray(Audio(filepath=str(result.artifacts["audio"])).waveform)[0]
        assert x[int(3.1 * SR) : int(3.4 * SR)].any(), "the withdrawn finding's region is untouched"


class TestThePaddingIsValidated:
    """padding_ms is a validity check at entry, before any store write."""

    def test_a_null_padding_refuses_before_any_store_write(self, store: ProvStore, tmp_path: Path) -> None:
        """An override may null the shipped margin, and a null margin gets no answer."""
        config = _override(tmp_path, "redaction:\n  padding_ms: null\n  fill: silence\n")
        _seed_redact_store(store, tmp_path, words=["hello", "alice"], findings=[("PERSON", (1.0, 2.0))])
        before = store.fingerprint()
        with pytest.raises(ValueError, match="redaction.padding_ms"):
            redact(store, "recording", config, run_dir=tmp_path, artifacts_dir=_release(tmp_path))
        assert store.fingerprint() == before

    @pytest.mark.parametrize("value", ["-300", "49.9", '"50"', ".inf"])
    def test_an_unusable_padding_override_is_refused_at_entry(
        self, store: ProvStore, tmp_path: Path, value: str
    ) -> None:
        """A negative margin narrows every extent, and neither channel can see the difference.

        A fractional one is a typo ``int()`` would truncate, a string would divide by 1000 inside
        ``plan_redactions``, and ``.inf`` is neither a margin nor a number a plan can use.
        """
        config = _override(tmp_path, f"redaction:\n  padding_ms: {value}\n  fill: silence\n")
        _seed_redact_store(store, tmp_path, words=["hello", "alice"], findings=[("PERSON", (1.0, 2.0))])
        before = store.fingerprint()
        with pytest.raises(ValueError, match="redaction.padding_ms"):
            redact(store, "recording", config, run_dir=tmp_path, artifacts_dir=_release(tmp_path))
        assert store.fingerprint() == before
        assert not _release(tmp_path).exists()

    def test_an_int_valued_float_padding_override_is_accepted(
        self,
        store: ProvStore,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """YAML renders 50.0 as a float; it is the same margin as 50 and is not a typo."""
        config = _override(tmp_path, "redaction:\n  padding_ms: 50.0\n  fill: silence\n")
        _seed_redact_store(store, tmp_path, words=["hello", "alice"], findings=[("PERSON", (1.0, 2.0))])
        _stub_pii(monkeypatch, findings=[])
        redact(store, "recording", config, run_dir=tmp_path, artifacts_dir=_release(tmp_path))
        recorded = _verdict_entity(store, "REDACT").attributes["padding_ms"]
        assert recorded == 50 and isinstance(recorded, int)


class TestTheStoresScanIsEvidenceOrItIsNot:
    """The planning scan's completeness is read from the measurement, never assumed."""

    def test_an_unscanned_store_is_refused_not_certified(
        self, store: ProvStore, redact_config: TriageConfig, tmp_path: Path
    ) -> None:
        """Findings with no scan measurement is an incoherent store, not a clean one (N15)."""
        _seed_redact_store(store, tmp_path, words=["hello", "alice"], findings=[("PERSON", (1.0, 2.0))], scanned=False)
        with pytest.raises(ValueError, match="no PII scan"):
            redact(store, "recording", redact_config, run_dir=tmp_path, artifacts_dir=_release(tmp_path))

    def test_a_store_scan_whose_detector_failed_is_withheld(
        self, store: ProvStore, redact_config: TriageConfig, tmp_path: Path
    ) -> None:
        """An empty ``spans`` with a populated ``failed`` means the scan did not happen."""
        _seed_redact_store(
            store,
            tmp_path,
            words=["hello", "alice"],
            findings=[("PERSON", (1.0, 2.0))],
            scanned_by=["presidio", "rules"],
            scan_failed=["gliner"],
        )
        result = redact(store, "recording", redact_config, run_dir=tmp_path, artifacts_dir=_release(tmp_path))
        assert result.verdict.outcome is Outcome.FAIL
        assert result.artifacts == {}
        detail = _verdict_entity(store, "REDACT").attributes
        assert detail["scan_failed"] == ["gliner"], "detector names, never their messages"
        assert detail["verified"] is False
        assert "gliner" in result.verdict.why

    def test_a_store_scan_with_no_detectors_is_withheld(
        self, store: ProvStore, redact_config: TriageConfig, tmp_path: Path
    ) -> None:
        """An empty ``scanned_by`` is "nothing ran", whatever the measurement's presence suggests."""
        _seed_redact_store(store, tmp_path, words=["hello", "alice"], findings=[("PERSON", (1.0, 2.0))], scanned_by=[])
        result = redact(store, "recording", redact_config, run_dir=tmp_path, artifacts_dir=_release(tmp_path))
        assert result.verdict.outcome is Outcome.FAIL
        assert result.artifacts == {}
        detail = _verdict_entity(store, "REDACT").attributes
        assert detail["verified"] is False and detail["scan_failed"] == []

    def test_a_required_detector_that_was_never_attempted_is_an_incomplete_scan(
        self, store: ProvStore, redact_config: TriageConfig, tmp_path: Path
    ) -> None:
        """A complete scan must not depend on the host that ran it."""
        _seed_redact_store(
            store,
            tmp_path,
            words=["hello", "alice"],
            findings=[("PERSON", (1.0, 2.0))],
            scanned_by=["presidio", "rules"],
        )
        result = redact(store, "recording", redact_config, run_dir=tmp_path, artifacts_dir=_release(tmp_path))
        assert result.verdict.outcome is Outcome.FAIL
        assert result.artifacts == {}
        detail = _verdict_entity(store, "REDACT").attributes
        assert detail["scan_missing"] == ["gliner"]
        assert detail["scan_failed"] == [], "never attempted is not the same as attempted and failed"
        assert "gliner" in result.verdict.why

    def test_narrowing_the_required_set_makes_the_same_scan_complete(
        self,
        store: ProvStore,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The required set is a config key, so an operator running two detectors can say so."""
        config = _override(
            tmp_path,
            "redaction:\n  padding_ms: 50\n  fill: silence\npii:\n  required_detectors: [presidio, rules]\n",
        )
        _seed_redact_store(
            store,
            tmp_path,
            words=["hello", "alice"],
            findings=[("PERSON", (1.0, 2.0))],
            scanned_by=["presidio", "rules"],
        )
        _stub_pii(monkeypatch, findings=[], detectors_used=["presidio", "rules"])
        result = redact(store, "recording", config, run_dir=tmp_path, artifacts_dir=_release(tmp_path))
        assert result.verdict.outcome is Outcome.PASS
        assert _verdict_entity(store, "REDACT").attributes["scan_missing"] == []
        assert result.artifacts.keys() == {"audio", "transcript", "consensus"}

    def test_a_failure_message_from_the_store_scan_never_reaches_the_verdict(
        self, store: ProvStore, redact_config: TriageConfig, tmp_path: Path
    ) -> None:
        """A detector's failure message may quote the scanned input, so only its name is recorded."""
        _seed_redact_store(store, tmp_path, words=["hello", "alice"], findings=[("PERSON", (1.0, 2.0))])
        scan = next(e for e in store.entities("measurement") if e.attributes.get("name") == "pii_scan")
        scan.attributes["failed"] = {"gliner": "ValueError on 'jane doe'"}
        result = redact(store, "recording", redact_config, run_dir=tmp_path, artifacts_dir=_release(tmp_path))
        assert result.verdict.outcome is Outcome.FAIL
        assert _verdict_entity(store, "REDACT").attributes["scan_failed"] == ["gliner"]
        assert "jane doe" not in json.dumps(_verdict_entity(store, "REDACT").attributes)
        assert "jane doe" not in result.verdict.why


class TestOnlyAPassReleases:
    """A flag withholds exactly like a fail, and the verdict says the withholding was deliberate."""

    def test_an_incomplete_re_scan_flags_rather_than_fails(
        self,
        store: ProvStore,
        redact_config: TriageConfig,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """redact.md: a re-scan that skipped a required detector is a flag, not a fail."""
        _seed_redact_store(store, tmp_path, words=["hello", "alice"], findings=[("PERSON", (1.0, 2.0))])
        _stub_pii(monkeypatch, findings=[], detectors_used=["presidio", "rules"])
        result = redact(store, "recording", redact_config, run_dir=tmp_path, artifacts_dir=_release(tmp_path))
        assert result.verdict.outcome is Outcome.FLAG
        assert _verdict_entity(store, "REDACT").attributes["verify_missing"] == ["gliner"]

    def test_a_flag_withholds_the_pair_and_records_it(
        self,
        store: ProvStore,
        redact_config: TriageConfig,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Only a pass produces a released pair; an empty mapping is legible only if the verdict says so."""
        _seed_redact_store(store, tmp_path, words=["hello", "alice"], findings=[("PERSON", (1.0, 2.0))])
        _stub_pii(monkeypatch, findings=[], detectors_used=["presidio", "rules"])
        result = redact(store, "recording", redact_config, run_dir=tmp_path, artifacts_dir=_release(tmp_path))
        assert result.verdict.outcome is Outcome.FLAG
        assert result.artifacts == {}
        assert not _release(tmp_path).exists(), "and writes nothing under the release directory"
        assert _verdict_entity(store, "REDACT").attributes["artifacts_withheld"] is True

    def test_a_scan_that_never_ran_is_not_read_as_a_clean_scan(
        self,
        store: ProvStore,
        redact_config: TriageConfig,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The real scanner's empty-input answer is ``detectors_used=[] failures={}``: nothing ran.

        Driven through the real ``scan_for_pii``, which spawns no subprocess for empty input, so the
        shape under test is the shipped one.
        """
        monkeypatch.setattr(redact_module, "scan_for_pii", real_scan_for_pii)
        _seed_redact_store(store, tmp_path, words=[], findings=[])
        result = redact(store, "recording", redact_config, run_dir=tmp_path, artifacts_dir=_release(tmp_path))
        assert result.verdict.outcome is Outcome.FLAG
        detail = _verdict_entity(store, "REDACT").attributes
        assert detail["verified"] is False and detail["survived"] == []
        assert result.artifacts == {}, "an unverified pair is withheld"
        assert not _release(tmp_path).exists(), "nothing was written to the release directory"

    def test_a_pass_releases_both_artifacts(
        self,
        store: ProvStore,
        redact_config: TriageConfig,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The control for the withholding cases."""
        _seed_redact_store(store, tmp_path, words=["hello", "alice"], findings=[("PERSON", (1.0, 2.0))])
        _stub_pii(monkeypatch, findings=[])
        result = redact(store, "recording", redact_config, run_dir=tmp_path, artifacts_dir=_release(tmp_path))
        assert result.verdict.outcome is Outcome.PASS
        assert result.artifacts.keys() == {"audio", "transcript", "consensus"}
        assert _verdict_entity(store, "REDACT").attributes["artifacts_withheld"] is False

    def test_artifacts_dir_nested_in_run_dir_is_refused(
        self, store: ProvStore, redact_config: TriageConfig, tmp_path: Path
    ) -> None:
        """The store's directory and the release directory must not be one publish step apart."""
        _seed_redact_store(store, tmp_path, words=["hello", "alice"], findings=[("PERSON", (1.0, 2.0))])
        with pytest.raises(ValueError, match="artifacts_dir"):
            redact(store, "recording", redact_config, run_dir=tmp_path, artifacts_dir=tmp_path / "release")

    def test_released_artifacts_share_no_element_ids_with_the_store(
        self,
        store: ProvStore,
        redact_config: TriageConfig,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """An id indexing both the store and a released artifact is a join key back to the PII."""
        _seed_redact_store(store, tmp_path, words=["hello", "alice"], findings=[("PERSON", (1.0, 2.0))])
        _stub_pii(monkeypatch, findings=[])
        result = redact(store, "recording", redact_config, run_dir=tmp_path, artifacts_dir=_release(tmp_path))
        assert result.artifacts, "a verified run releases both artifacts"
        ids = [e.id for e in store.entities()]
        for path in result.artifacts.values():
            blob = path.read_bytes()
            for entity_id in ids:
                assert entity_id.encode() not in blob

    def test_the_source_is_not_destroyed_and_the_store_only_grows(
        self,
        store: ProvStore,
        redact_config: TriageConfig,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Redaction writes; deletion is an operator decision with its own authorisation."""
        _seed_redact_store(store, tmp_path, words=["hello", "alice"], findings=[("PERSON", (1.0, 2.0))])
        _stub_pii(monkeypatch, findings=[])
        before = {e.id for e in store.entities()}
        redact(store, "recording", redact_config, run_dir=tmp_path, artifacts_dir=_release(tmp_path))
        assert (tmp_path / "streams" / "plain.wav").exists()
        assert before <= {e.id for e in store.entities()}, "append-only: nothing removed"


class TestTheTranscriptArtifact:
    """What the released text carries, and what it never carries."""

    def test_findings_render_as_category_placeholders(
        self,
        store: ProvStore,
        redact_config: TriageConfig,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Words inside planned extents render as [CATEGORY]; padded-in neighbours go with them."""
        _seed_redact_store(store, tmp_path, words=["my", "name", "jane", "here"], findings=[("PERSON", (2.0, 2.5))])
        _stub_pii(monkeypatch, findings=[])
        result = redact(store, "recording", redact_config, run_dir=tmp_path, artifacts_dir=_release(tmp_path))
        text = result.artifacts["transcript"].read_text()
        assert text.split() == ["my", "name", "[PERSON]", "here"]
        assert "jane" not in text and "2.0" not in text and "2.5" not in text

    def test_a_word_with_no_extent_is_not_released_verbatim(
        self,
        store: ProvStore,
        redact_config: TriageConfig,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Text whose location is unknown overlaps no redaction, so it cannot be shown to be safe."""
        _seed_redact_store(store, tmp_path, words=["hello", "alice"], findings=[("PERSON", (1.0, 2.0))])
        consensus = next(a for a in store.activities("PREPROCESS") if a.step == "consensus")
        floating = store.entity(
            prov_type="word",
            extent=None,
            attributes={**word_attributes("unplaceable-sentinel", (0.0, 0.0), index=99), "timings": {}},
        )
        store.was_generated_by(floating, consensus.id)
        _stub_pii(monkeypatch, findings=[])
        result = redact(store, "recording", redact_config, run_dir=tmp_path, artifacts_dir=_release(tmp_path))
        text = result.artifacts["transcript"].read_text()
        assert "unplaceable-sentinel" not in text
        assert "[UNPLACED]" in text
        assert _verdict_entity(store, "REDACT").attributes["unplaced_words_n"] == 1

    def test_words_are_released_in_stream_order_not_time_order(
        self,
        store: ProvStore,
        redact_config: TriageConfig,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test 31 (C-7): extents that disagree with the index must not reorder the released text."""
        _seed_redact_store(store, tmp_path, words=["one", "two", "three"], findings=[("PERSON", (0.0, 0.5))])
        by_text = {w.attributes["text"]: w for w in store.entities("word")}
        consensus = next(a for a in store.activities("PREPROCESS") if a.step == "consensus")
        withdraw = store.activity(node="PREPROCESS", step="withdraw", parameters={})
        store.was_invalidated_by(by_text["two"].id, withdraw)
        late = store.entity(prov_type="word", extent=(9.0, 9.5), attributes=word_attributes("two", (9.0, 9.5), index=1))
        store.was_generated_by(late, consensus.id)
        _stub_pii(monkeypatch, findings=[])
        result = redact(store, "recording", redact_config, run_dir=tmp_path, artifacts_dir=_release(tmp_path))
        assert result.artifacts["transcript"].read_text().split() == ["[PERSON]", "two", "three"]

    def test_a_word_whose_derived_extent_misses_its_sources_is_masked_by_the_hull(
        self,
        store: ProvStore,
    ) -> None:
        """The derived extent is a fitted estimate and can fall outside every source's reading.

        Deciding the mask on it would release the name and blank the silence between the two
        placements, so the transcript decides on the hull of the sources' own timings.
        """
        from senselab.audio.tasks.redaction.api import RedactionExtent
        from senselab.audio.workflows.triage.nodes.redact import _render

        pre = store.activity(node="PREPROCESS", step="consensus", parameters={})
        # The sources place "alice" at 1.0-1.2 and 5.4-5.8; the fit derives 3.0-3.2, between them.
        ids = [
            store.entity(
                prov_type="word",
                extent=extent,
                attributes=word_attributes(text, extent, index=index, timings=timings),
            )
            for index, (text, extent, timings) in enumerate(
                [
                    ("one", (0.1, 0.4), {"asr_a": (0.1, 0.4), "asr_b": (0.1, 0.4)}),
                    ("alice", (3.0, 3.2), {"asr_a": (1.0, 1.2), "asr_b": (5.4, 5.8)}),
                ]
            )
        ]
        for word_id in ids:
            store.was_generated_by(word_id, pre)
        words = [store.get_entity(word_id) for word_id in ids]
        planned = [RedactionExtent(start=0.9, end=1.3, category="PERSON")]

        _records, text, unplaced = _render(words, planned)

        assert "alice" not in text, "the name survived because the mask read the fitted extent"
        assert text == "one [PERSON]"
        assert unplaced == 0

    def test_a_placed_transcript_counts_no_unplaced_words(
        self,
        store: ProvStore,
        redact_config: TriageConfig,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The control: every word carrying an extent leaves the count at zero."""
        _seed_redact_store(store, tmp_path, words=["hello", "alice"], findings=[("PERSON", (1.0, 2.0))])
        _stub_pii(monkeypatch, findings=[])
        result = redact(store, "recording", redact_config, run_dir=tmp_path, artifacts_dir=_release(tmp_path))
        assert _verdict_entity(store, "REDACT").attributes["unplaced_words_n"] == 0
        assert "[UNPLACED]" not in result.artifacts["transcript"].read_text()


RAINBOW = "When the sunlight strikes raindrops in the air, they act as a prism and form a rainbow."


def _hint(*prompts: str) -> AudioHints:
    """A hint declaring what the participant was asked to read."""
    return AudioHints(expected_speech=[ExpectedSpeech(text=prompt) for prompt in prompts])


def _exemptions(store: ProvStore) -> list[Entity]:
    """Every ``exempt``/``expected_speech`` assertion the node wrote."""
    return [
        entity
        for entity in store.entities("assertion")
        if entity.attributes.get("verb") == "exempt" and entity.attributes.get("label") == "expected_speech"
    ]


class TestTheStimulusAccountsForACandidate:
    """A PII-shaped token the prompt asked for is not a disclosure — and never a silent one."""

    def test_a_candidate_the_prompt_asked_for_is_not_redacted(
        self,
        store: ProvStore,
        redact_config: TriageConfig,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """``rainbow`` reads as a LOCATION to a detector and as line one of the passage to a reader."""
        _seed_redact_store(store, tmp_path, words=["form", "a", "rainbow"], findings=[("LOCATION", (2.0, 2.5))])
        _stub_pii(monkeypatch, findings=[("LOCATION", "rainbow")])
        result = redact(
            store, "recording", redact_config, _hint(RAINBOW), run_dir=tmp_path, artifacts_dir=_release(tmp_path)
        )
        assert result.verdict.outcome is Outcome.PASS
        assert result.artifacts["transcript"].read_text().split() == ["form", "a", "rainbow"]
        assert _verdict_entity(store, "REDACT").attributes["redactions_n"] == 0

    def test_the_suppression_is_recorded_with_what_accounted_for_it(
        self,
        store: ProvStore,
        redact_config: TriageConfig,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A reader must be able to audit what was *not* redacted, and on whose authority."""
        _seed_redact_store(store, tmp_path, words=["form", "a", "rainbow"], findings=[("LOCATION", (2.0, 2.5))])
        _stub_pii(monkeypatch, findings=[("LOCATION", "rainbow")])
        redact(store, "recording", redact_config, _hint(RAINBOW), run_dir=tmp_path, artifacts_dir=_release(tmp_path))
        [assertion] = _exemptions(store)
        assert assertion.attributes["category"] == "LOCATION"
        assert assertion.attributes["expected_keys"] == ["rainbow"]
        assert assertion.attributes["expected_prompt"] == 0
        assert assertion.attributes["expected_unit_text"] == RAINBOW
        assert assertion.attributes["words_n"] == 1
        assert assertion.extent == (2.0, 2.5)
        finding = next(e for e in store.entities("pii"))
        word = next(e for e in store.entities("word") if e.attributes["text"] == "rainbow")
        assert set(store.derived_from(assertion.id)) == {finding.id, word.id}

    def test_the_verdict_and_a_measurement_both_count_the_suppressions(
        self,
        store: ProvStore,
        redact_config: TriageConfig,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A count with no entity would be unauditable; an entity with no count would be unfindable."""
        _seed_redact_store(store, tmp_path, words=["form", "a", "rainbow"], findings=[("LOCATION", (2.0, 2.5))])
        _stub_pii(monkeypatch, findings=[("LOCATION", "rainbow")])
        redact(store, "recording", redact_config, _hint(RAINBOW), run_dir=tmp_path, artifacts_dir=_release(tmp_path))
        detail = _verdict_entity(store, "REDACT").attributes
        assert detail["expected_exempt_n"] == 1
        assert detail["expected_exempt_by_category"] == {"LOCATION": 1}
        assert detail["expected_survivors"] == ["LOCATION"]
        assert detail["expected_speech_declared"] is True
        measurement = next(
            e for e in store.entities("measurement") if e.attributes.get("name") == "redaction_exemptions"
        )
        assert measurement.attributes["n"] == 1 and measurement.attributes["n_findings"] == 1

    def test_a_candidate_the_prompt_does_not_account_for_is_still_redacted(
        self,
        store: ProvStore,
        redact_config: TriageConfig,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The discrimination. A hint is not a blanket exemption."""
        _seed_redact_store(store, tmp_path, words=["form", "a", "alice"], findings=[("PERSON", (2.0, 2.5))])
        _stub_pii(monkeypatch, findings=[])
        result = redact(
            store, "recording", redact_config, _hint(RAINBOW), run_dir=tmp_path, artifacts_dir=_release(tmp_path)
        )
        assert result.artifacts["transcript"].read_text().split() == ["form", "a", "[PERSON]"]
        assert _exemptions(store) == []
        assert _verdict_entity(store, "REDACT").attributes["expected_exempt_n"] == 0

    def test_one_exempt_and_one_unaccounted_candidate_are_decided_apart(
        self,
        store: ProvStore,
        redact_config: TriageConfig,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The exemption is per finding, not per recording: the name goes, the passage word stays."""
        _seed_redact_store(
            store,
            tmp_path,
            words=["rainbow", "with", "alice"],
            findings=[("LOCATION", (0.0, 0.5)), ("PERSON", (2.0, 2.5))],
        )
        _stub_pii(monkeypatch, findings=[("LOCATION", "rainbow")])
        result = redact(
            store, "recording", redact_config, _hint(RAINBOW), run_dir=tmp_path, artifacts_dir=_release(tmp_path)
        )
        assert result.verdict.outcome is Outcome.PASS
        assert result.artifacts["transcript"].read_text().split() == ["rainbow", "with", "[PERSON]"]
        assert [a.attributes["category"] for a in _exemptions(store)] == ["LOCATION"]

    def test_a_finding_reaching_every_word_is_never_exempt(
        self,
        store: ProvStore,
        redact_config: TriageConfig,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """SPEECH's signature for a finding it could not place; a whole-passage prompt must not absolve it.

        The extent is the exact hull of every word, so the span-coverage condition is satisfied and
        ``form a rainbow`` really is a contiguous run of the passage. Only the whole-stream rule
        stands between that and an exemption, which is what this pins.
        """
        _seed_redact_store(store, tmp_path, words=["form", "a", "rainbow"], findings=[("LOCATION", (0.0, 2.5))])
        _stub_pii(monkeypatch, findings=[])
        result = redact(
            store, "recording", redact_config, _hint(RAINBOW), run_dir=tmp_path, artifacts_dir=_release(tmp_path)
        )
        assert _exemptions(store) == []
        assert result.artifacts["transcript"].read_text().split() == ["[LOCATION]"]

    def test_a_finding_reaching_past_the_words_it_identifies_is_not_exempt(
        self,
        store: ProvStore,
        redact_config: TriageConfig,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The words a finding reaches only account for it when they reconstruct its whole extent.

        SPEECH builds a located finding's extent as the hull of the words it covers, so a covered
        set whose own hulls fall short of that extent means a word was missed — and a missed word is
        one the exemption would be accounting for without having looked at it. The finding here runs
        to 3.5 s while ``rainbow`` stops at 2.5 s, so it is planned rather than exempted.
        """
        _seed_redact_store(store, tmp_path, words=["form", "a", "rainbow"], findings=[("LOCATION", (2.0, 3.5))])
        _stub_pii(monkeypatch, findings=[])
        result = redact(
            store, "recording", redact_config, _hint(RAINBOW), run_dir=tmp_path, artifacts_dir=_release(tmp_path)
        )
        assert _exemptions(store) == []
        assert result.artifacts["transcript"].read_text().split() == ["form", "a", "[LOCATION]"]

    def test_the_covered_words_must_be_a_contiguous_run_of_one_declared_unit(
        self,
        store: ProvStore,
        redact_config: TriageConfig,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Two words the prompt happens to contain apart do not account for them said together."""
        _seed_redact_store(store, tmp_path, words=["hi", "sunlight", "rainbow"], findings=[("PERSON", (1.0, 2.5))])
        _stub_pii(monkeypatch, findings=[])
        result = redact(
            store, "recording", redact_config, _hint(RAINBOW), run_dir=tmp_path, artifacts_dir=_release(tmp_path)
        )
        assert _exemptions(store) == [], "a scattered pair was treated as accounted for"
        assert result.artifacts["transcript"].read_text().split() == ["hi", "[PERSON]"]

    def test_a_contiguous_pair_the_prompt_does_contain_is_exempt(
        self,
        store: ProvStore,
        redact_config: TriageConfig,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The control for the run rule: the same two words in the prompt's own order are accounted for."""
        _seed_redact_store(store, tmp_path, words=["hi", "a", "rainbow"], findings=[("LOCATION", (1.0, 2.5))])
        _stub_pii(monkeypatch, findings=[("LOCATION", "a rainbow")])
        result = redact(
            store, "recording", redact_config, _hint(RAINBOW), run_dir=tmp_path, artifacts_dir=_release(tmp_path)
        )
        assert result.artifacts["transcript"].read_text().split() == ["hi", "a", "rainbow"]
        assert _exemptions(store)[0].attributes["expected_keys"] == ["a", "rainbow"]

    def test_the_comparison_uses_the_branches_own_normaliser(
        self,
        store: ProvStore,
        redact_config: TriageConfig,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Casefold and edge punctuation, on both sides; a second spelling would compare two vocabularies."""
        _seed_redact_store(store, tmp_path, words=["form", "a", "Rainbow."], findings=[("LOCATION", (2.0, 2.5))])
        _stub_pii(monkeypatch, findings=[("LOCATION", "Rainbow")])
        redact(store, "recording", redact_config, _hint(RAINBOW), run_dir=tmp_path, artifacts_dir=_release(tmp_path))
        assert [a.attributes["expected_keys"] for a in _exemptions(store)] == [["rainbow"]]

    def test_the_replan_does_not_widen_onto_an_exempt_word(
        self,
        store: ProvStore,
        redact_config: TriageConfig,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The verifier sees the exempt word again; remediating it would undo the decision silently."""
        _seed_redact_store(
            store,
            tmp_path,
            words=["rainbow", "with", "alice"],
            findings=[("LOCATION", (0.0, 0.5)), ("PERSON", (2.0, 2.5))],
        )
        scanned = _stub_pii_sequence(monkeypatch, [[("LOCATION", "rainbow")], [("LOCATION", "rainbow")]])
        result = redact(
            store, "recording", redact_config, _hint(RAINBOW), run_dir=tmp_path, artifacts_dir=_release(tmp_path)
        )
        assert scanned == ["rainbow with [PERSON]"], "the node re-planned over an accounted-for candidate"
        assert result.verdict.outcome is Outcome.PASS
        assert _verdict_entity(store, "REDACT").attributes["replanned_n"] == 0

    def test_a_replan_over_a_shared_category_widens_past_the_exempt_word(
        self,
        store: ProvStore,
        redact_config: TriageConfig,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The one case where both halves of the exemption are live at once.

        ``rainbow`` and ``alice`` both carry LOCATION. ``alice`` is not accounted for, so LOCATION is
        outstanding and the re-plan runs — and it must widen onto ``alice`` alone. Widening onto both
        would undo the exemption on the very pass that exists to catch what the first plan missed,
        which is the failure a skip that is only exercised here can hide.
        """
        _seed_redact_store(
            store,
            tmp_path,
            words=["rainbow", "with", "alice"],
            findings=[("LOCATION", (0.0, 0.5))],
            extra_marks=[("alice", "LOCATION")],
        )
        scanned = _stub_pii_sequence(monkeypatch, [[("LOCATION", "alice")], []])
        result = redact(
            store, "recording", redact_config, _hint(RAINBOW), run_dir=tmp_path, artifacts_dir=_release(tmp_path)
        )
        assert scanned[0] == "rainbow with alice"
        assert scanned[1] == "rainbow with [LOCATION]", "the re-plan widened onto an accounted-for word"
        assert result.verdict.outcome is Outcome.PASS
        assert result.artifacts["transcript"].read_text().split() == ["rainbow", "with", "[LOCATION]"]
        assert _verdict_entity(store, "REDACT").attributes["replanned_n"] == 1

    def test_a_survivor_no_exempt_word_explains_still_fails(
        self,
        store: ProvStore,
        redact_config: TriageConfig,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The exemption attributes one category to one word; it is not an amnesty on the category."""
        _seed_redact_store(
            store,
            tmp_path,
            words=["rainbow", "with", "alice"],
            findings=[("LOCATION", (0.0, 0.5))],
            extra_marks=[("alice", "PERSON")],
        )
        _stub_pii(monkeypatch, findings=[("PERSON", "alice")])
        result = redact(
            store, "recording", redact_config, _hint(RAINBOW), run_dir=tmp_path, artifacts_dir=_release(tmp_path)
        )
        assert result.verdict.outcome is Outcome.FAIL
        assert _verdict_entity(store, "REDACT").attributes["unremediable"] == ["PERSON"]
        assert result.artifacts == {}


class TestNoHintIsNoChange:
    """With no hint, or a hint declaring no utterance, the node is what it was."""

    @pytest.mark.parametrize("hint", [None, AudioHints(), AudioHints(expected_speech=[])])
    def test_the_released_text_is_unchanged_without_expected_speech(
        self,
        store: ProvStore,
        redact_config: TriageConfig,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        hint: AudioHints | None,
    ) -> None:
        """The passage word is redacted because nothing declared it, which is the old behaviour."""
        _seed_redact_store(store, tmp_path, words=["form", "a", "rainbow"], findings=[("LOCATION", (2.0, 2.5))])
        _stub_pii(monkeypatch, findings=[])
        result = redact(store, "recording", redact_config, hint, run_dir=tmp_path, artifacts_dir=_release(tmp_path))
        assert result.artifacts["transcript"].read_text().split() == ["form", "a", "[LOCATION]"]
        assert _exemptions(store) == []
        detail = _verdict_entity(store, "REDACT").attributes
        assert detail["expected_exempt_n"] == 0 and detail["expected_speech_declared"] is False

    def test_a_survivor_without_a_hint_still_fails(
        self,
        store: ProvStore,
        redact_config: TriageConfig,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The control on the survivor path: no exemption can be manufactured out of no hint."""
        _seed_redact_store(store, tmp_path, words=["form", "a", "rainbow"], findings=[("LOCATION", (2.0, 2.5))])
        _stub_pii(monkeypatch, findings=[("LOCATION", "rainbow")])
        result = redact(store, "recording", redact_config, None, run_dir=tmp_path, artifacts_dir=_release(tmp_path))
        assert result.verdict.outcome is Outcome.FAIL
        assert result.artifacts == {}


class TestTheRedactedConsensusArtifact:
    """The consensus structure survives redaction; only the redacted tokens' surfaces do not."""

    def _consensus(self, result: Any) -> dict[str, Any]:  # noqa: ANN401 — the node's own result type
        """The released consensus artifact, parsed."""
        return dict(json.loads(result.artifacts["consensus"].read_text()))

    def test_a_surviving_word_keeps_its_times_and_its_source_agreement(
        self,
        store: ProvStore,
        redact_config: TriageConfig,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A flat string loses the structure the consensus was built to carry; the JSON keeps it."""
        _seed_redact_store(store, tmp_path, words=["my", "name", "jane", "here"], findings=[("PERSON", (2.0, 2.5))])
        _stub_pii(monkeypatch, findings=[])
        result = redact(store, "recording", redact_config, run_dir=tmp_path, artifacts_dir=_release(tmp_path))
        document = self._consensus(result)
        assert document["schema"] == "senselab.triage.redacted_consensus"
        first = next(record for record in document["records"] if record.get("text") == "my")
        assert first["kind"] == "word"
        assert first["start_s"] == 0.0 and first["end_s"] == 0.5
        assert first["agreement"] == 1.0
        assert first["sources"] and first["timings"] and first["outcome"]

    def test_a_redacted_word_contributes_a_placeholder_record_carrying_no_surface(
        self,
        store: ProvStore,
        redact_config: TriageConfig,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The record says where and what category, never the token, its readings or its variants."""
        _seed_redact_store(store, tmp_path, words=["my", "name", "jane", "here"], findings=[("PERSON", (2.0, 2.5))])
        _stub_pii(monkeypatch, findings=[])
        result = redact(store, "recording", redact_config, run_dir=tmp_path, artifacts_dir=_release(tmp_path))
        document = self._consensus(result)
        assert "jane" not in json.dumps(document), "the redacted surface reached the released structure"
        masked = next(record for record in document["records"] if record["kind"] == "redaction")
        assert masked["category"] == "PERSON"
        assert masked["token"] == "[PERSON]"
        assert "text" not in masked and "readings" not in masked and "variants" not in masked
        assert masked["start_s"] < 2.0 and masked["end_s"] > 2.5, "the record carries the padded extent"

    def test_one_merged_extent_is_one_record_counting_the_words_it_swallowed(
        self,
        store: ProvStore,
        redact_config: TriageConfig,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The structure matches what the audio lost, so a reader can tell 1 word from 2."""
        _seed_redact_store(store, tmp_path, words=["hi", "jane", "doe", "bye"], findings=[("PERSON", (1.0, 2.5))])
        _stub_pii(monkeypatch, findings=[])
        result = redact(store, "recording", redact_config, run_dir=tmp_path, artifacts_dir=_release(tmp_path))
        document = self._consensus(result)
        assert document["n_redactions"] == 1
        assert next(r for r in document["records"] if r["kind"] == "redaction")["words_n"] == 2

    def test_the_flat_text_is_the_records_joined_so_the_two_artifacts_cannot_disagree(
        self,
        store: ProvStore,
        redact_config: TriageConfig,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """One renderer, two artifacts: a mask visible in one and absent from the other is unreachable."""
        _seed_redact_store(store, tmp_path, words=["my", "name", "jane", "here"], findings=[("PERSON", (2.0, 2.5))])
        _stub_pii(monkeypatch, findings=[])
        result = redact(store, "recording", redact_config, run_dir=tmp_path, artifacts_dir=_release(tmp_path))
        document = self._consensus(result)
        rebuilt = " ".join(
            record["text"] if record["kind"] == "word" else record["token"] for record in document["records"]
        )
        assert rebuilt == document["text"]
        assert result.artifacts["transcript"].read_text().strip() == document["text"]

    def test_an_unplaced_word_is_a_record_with_no_surface(
        self,
        store: ProvStore,
        redact_config: TriageConfig,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Text of unknown location cannot be shown to be safe, in the structure as in the string."""
        _seed_redact_store(store, tmp_path, words=["hello", "alice"], findings=[("PERSON", (1.0, 2.0))])
        consensus = next(a for a in store.activities("PREPROCESS") if a.step == "consensus")
        floating = store.entity(
            prov_type="word",
            extent=None,
            attributes={**word_attributes("unplaceable-sentinel", (0.0, 0.0), index=99), "timings": {}},
        )
        store.was_generated_by(floating, consensus.id)
        _stub_pii(monkeypatch, findings=[])
        result = redact(store, "recording", redact_config, run_dir=tmp_path, artifacts_dir=_release(tmp_path))
        document = self._consensus(result)
        assert "unplaceable-sentinel" not in json.dumps(document)
        assert document["n_unplaced"] == 1
        assert next(r for r in document["records"] if r["kind"] == "unplaced")["token"] == "[UNPLACED]"

    def test_a_withheld_run_releases_no_consensus_artifact_either(
        self,
        store: ProvStore,
        redact_config: TriageConfig,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The control: the new artifact is on the same release gate as the other two."""
        _seed_redact_store(store, tmp_path, words=["hello", "alice"], findings=[("PERSON", (1.0, 2.0))])
        _stub_pii(monkeypatch, findings=[("PERSON", "alice")])
        result = redact(store, "recording", redact_config, run_dir=tmp_path, artifacts_dir=_release(tmp_path))
        assert result.artifacts == {}


LLM_ON = "redaction:\n  padding_ms: 50\n  fill: silence\n  llm_check:\n    enabled: true\n"


def _stub_review(monkeypatch: pytest.MonkeyPatch, rounds: Sequence[ReviewResult]) -> list[str]:
    """Replace the node's reviewer with one answer per round, recording every text it was handed."""
    seen: list[str] = []
    remaining = list(rounds)

    def _fake(text: str, **kw: Any) -> ReviewResult:  # noqa: ANN401
        seen.append(text)
        assert remaining, "the node reviewed more times than the test declared answers for"
        return remaining.pop(0)

    monkeypatch.setattr(redact_module, "review_redacted_text", _fake)
    return seen


def _clean(reasoning: str = "Nothing here identifies the speaker.") -> ReviewResult:
    """A round that ran and flagged nothing."""
    return ReviewResult(available=True, reasoning=reasoning, model_id="stub/model", revision="a" * 40)


def _flags(
    *findings: ReviewFinding, reasoning: str = "A date and a street together locate one person."
) -> ReviewResult:
    """A round that ran and flagged something."""
    return ReviewResult(
        available=True,
        reasoning=reasoning,
        findings=list(findings),
        model_id="stub/model",
        revision="a" * 40,
    )


def _reviews(store: ProvStore) -> list[Entity]:
    """Every captured review measurement, in write order."""
    return [e for e in store.entities("measurement") if e.attributes.get("name") == "redaction_llm_review"]


def _annotation(store: ProvStore) -> dict[str, Any]:
    """The re-read's summary, as the annotation measurement VERDICT reads carries it."""
    found = [e for e in store.entities("measurement") if e.attributes.get("name") == "redaction_llm_annotation"]
    assert len(found) == 1, f"expected exactly one annotation, found {len(found)}"
    return dict(found[0].attributes)


class TestTheLlmCheckIsOffUnlessAskedFor:
    """A step that turns itself on would make two hosts disagree with no record of why."""

    def test_the_packaged_config_leaves_it_disabled(self) -> None:
        """The default is off, and the other keys are present so an override may only flip one."""
        cfg = load_triage_config()
        assert cfg.require("redaction.llm_check.enabled") is False
        assert cfg.require("redaction.llm_check.model_id") == "google/gemma-4-31B-it-qat-w4a16-ct"
        assert cfg.require("redaction.llm_check.ref") == "main"
        assert cfg.require("redaction.llm_check.max_iterations") == 3

    def test_a_disabled_check_contacts_nothing_and_says_so(
        self,
        store: ProvStore,
        redact_config: TriageConfig,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Off means no subprocess, no venv build, and a verdict that records the absence as a choice."""
        _seed_redact_store(store, tmp_path, words=["hello", "alice"], findings=[("PERSON", (1.0, 2.0))])
        _stub_pii(monkeypatch, findings=[])
        seen = _stub_review(monkeypatch, [])
        result = redact(store, "recording", redact_config, run_dir=tmp_path, artifacts_dir=_release(tmp_path))
        assert seen == []
        assert result.verdict.outcome is Outcome.PASS
        assert _annotation(store)["status"] == "disabled"
        assert _reviews(store) == []

    def test_it_is_not_run_when_the_detector_path_already_withheld(
        self,
        store: ProvStore,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """It can only withhold, so there is nothing for it to decide over a withheld artifact."""
        config = _override(tmp_path, LLM_ON)
        _seed_redact_store(store, tmp_path, words=["hello", "alice"], findings=[("PERSON", (1.0, 2.0))])
        _stub_pii(monkeypatch, findings=[("PERSON", "alice")])
        seen = _stub_review(monkeypatch, [])
        result = redact(store, "recording", config, run_dir=tmp_path, artifacts_dir=_release(tmp_path))
        assert result.verdict.outcome is Outcome.FAIL
        assert seen == []
        assert _annotation(store)["status"] == "not_run"


class TestTheLlmCheckIterates:
    """It reviews, and a round that flags something reviews again on the masked text."""

    def test_a_clean_first_round_stops_there_and_releases(
        self,
        store: ProvStore,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """One review is the whole loop when nothing is flagged."""
        config = _override(tmp_path, LLM_ON)
        _seed_redact_store(store, tmp_path, words=["hello", "alice"], findings=[("PERSON", (1.0, 2.0))])
        _stub_pii(monkeypatch, findings=[])
        seen = _stub_review(monkeypatch, [_clean()])
        result = redact(store, "recording", config, run_dir=tmp_path, artifacts_dir=_release(tmp_path))
        assert seen == ["hello [PERSON]"], "the reviewer reads the redacted transcript, not the source"
        assert result.verdict.outcome is Outcome.PASS
        check = _annotation(store)
        assert check["status"] == "clean" and check["iterations"] == 1

    def test_a_flagged_round_reviews_again_on_the_masked_text(
        self,
        store: ProvStore,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The reviewer sees the effect of its own concern; that is what the loop is for."""
        config = _override(tmp_path, LLM_ON)
        _seed_redact_store(store, tmp_path, words=["born", "in", "belmont"], findings=[("PERSON", (0.0, 0.5))])
        _stub_pii(monkeypatch, findings=[])
        seen = _stub_review(
            monkeypatch,
            [_flags(ReviewFinding(text="belmont", category="LOCATION", why="a town with one clinic")), _clean()],
        )
        result = redact(store, "recording", config, run_dir=tmp_path, artifacts_dir=_release(tmp_path))
        assert seen == ["[PERSON] in belmont", "[PERSON] in [LLM_LOCATION]"]
        check = _annotation(store)
        assert check["status"] == "flagged", "a concern raised and then masked is still a concern"
        assert result.verdict.outcome is Outcome.PASS, "the reviewer annotates; the detector path decided"
        assert result.artifacts != {}, "an unmeasured model does not withhold the release"

    def test_the_loop_is_bounded_by_the_config_key(
        self,
        store: ProvStore,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A reviewer that flags something every round must stop, and the bound is declared."""
        config = _override(tmp_path, LLM_ON + "    max_iterations: 2\n")
        _seed_redact_store(store, tmp_path, words=["born", "in", "belmont"], findings=[("PERSON", (0.0, 0.5))])
        _stub_pii(monkeypatch, findings=[])
        forever = _flags(ReviewFinding(text="in", category="OTHER", why="still uneasy"))
        seen = _stub_review(monkeypatch, [forever, forever, forever])
        redact(store, "recording", config, run_dir=tmp_path, artifacts_dir=_release(tmp_path))
        assert len(seen) == 2
        check = _annotation(store)
        assert check["status"] == "flagged" and check["iterations"] == 2

    def test_a_named_substring_the_text_does_not_carry_is_left_alone(
        self,
        store: ProvStore,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A model that hallucinates a substring must not make the next round's text a guess."""
        config = _override(tmp_path, LLM_ON)
        _seed_redact_store(store, tmp_path, words=["hello", "alice"], findings=[("PERSON", (1.0, 2.0))])
        _stub_pii(monkeypatch, findings=[])
        seen = _stub_review(
            monkeypatch,
            [_flags(ReviewFinding(text="nowhere-in-the-text", category="OTHER", why="invented")), _clean()],
        )
        redact(store, "recording", config, run_dir=tmp_path, artifacts_dir=_release(tmp_path))
        assert seen == ["hello [PERSON]", "hello [PERSON]"]


class TestTheChainOfThoughtIsCaptured:
    """The reasoning is the product of the step, not a by-product of it."""

    def test_every_round_reasoning_reaches_the_store(
        self,
        store: ProvStore,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """One measurement per round, carrying that round's reasoning verbatim."""
        config = _override(tmp_path, LLM_ON)
        _seed_redact_store(store, tmp_path, words=["born", "in", "belmont"], findings=[("PERSON", (0.0, 0.5))])
        _stub_pii(monkeypatch, findings=[])
        _stub_review(
            monkeypatch,
            [
                _flags(
                    ReviewFinding(text="belmont", category="LOCATION", why="a town with one clinic"),
                    reasoning="The town name narrows the population to a few thousand.",
                ),
                _clean(reasoning="With the town masked, nothing remains."),
            ],
        )
        redact(store, "recording", config, run_dir=tmp_path, artifacts_dir=_release(tmp_path))
        rounds = _reviews(store)
        assert [e.attributes["iteration"] for e in rounds] == [1, 2]
        assert rounds[0].attributes["reasoning"] == "The town name narrows the population to a few thousand."
        assert rounds[1].attributes["reasoning"] == "With the town masked, nothing remains."
        assert rounds[0].attributes["findings"][0]["why"] == "a town with one clinic"

    def test_the_review_activity_names_the_model_at_its_commit(
        self,
        store: ProvStore,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A captured chain of thought nobody can attribute to a commit is not evidence."""
        config = _override(tmp_path, LLM_ON)
        _seed_redact_store(store, tmp_path, words=["hello", "alice"], findings=[("PERSON", (1.0, 2.0))])
        _stub_pii(monkeypatch, findings=[])
        _stub_review(monkeypatch, [_clean()])
        redact(store, "recording", config, run_dir=tmp_path, artifacts_dir=_release(tmp_path))
        activity = next(a for a in store.activities("REDACT") if a.step == "llm_check")
        agents = [store.get_agent(a) for a in store.associated_with(activity.id)]
        model = next(agent for agent in agents if agent.agent_type == "model")
        assert model.commit_sha == "a" * 40
        assert _annotation(store)["revision"] == "a" * 40

    def test_the_report_surfaces_the_reasoning(
        self,
        store: ProvStore,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Captured in the store and unreachable from the report is captured nowhere a reader looks."""
        from senselab.audio.workflows.triage.nodes.report import _llm_annotation, _llm_check_lines, _llm_reviews

        config = _override(tmp_path, LLM_ON)
        _seed_redact_store(store, tmp_path, words=["born", "in", "belmont"], findings=[("PERSON", (0.0, 0.5))])
        _stub_pii(monkeypatch, findings=[])
        _stub_review(
            monkeypatch,
            [
                _flags(
                    ReviewFinding(text="belmont", category="LOCATION", why="a town with one clinic"),
                    reasoning="The town name narrows the population to a few thousand.",
                ),
                _clean(reasoning="With the town masked, nothing remains."),
            ],
        )
        redact(store, "recording", config, run_dir=tmp_path, artifacts_dir=_release(tmp_path))
        text = "\n".join(_llm_check_lines(_llm_annotation(store), _llm_reviews(store)))
        assert "The town name narrows the population to a few thousand." in text
        assert "With the town masked, nothing remains." in text
        assert "concern [LOCATION]: a town with one clinic" in text
        assert "llm check: flagged" in text
        assert f"revision={'a' * 40}" in text, "a chain of thought nobody can attribute to a commit is not evidence"


class TestTheReviewerAnnotatesAndVerdictDecides:
    """Owner, 2026-09-17: "the llm is part of a branch, so it can only annotate (with provenance)"."""

    def test_a_flagged_re_read_releases_and_flags_the_triage_axis(
        self,
        store: ProvStore,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The whole split, end to end: REDACT's own path cleared the artifact, and a human still looks."""
        config = _override(tmp_path, LLM_ON)
        _seed_redact_store(store, tmp_path, words=["born", "in", "belmont"], findings=[("PERSON", (0.0, 0.5))])
        _stub_pii(monkeypatch, findings=[])
        _stub_review(
            monkeypatch,
            [_flags(ReviewFinding(text="belmont", category="LOCATION", why="a town with one clinic")), _clean()],
        )
        result = redact(store, "recording", config, run_dir=tmp_path, artifacts_dir=_release(tmp_path))
        assert result.verdict.outcome is Outcome.PASS and result.artifacts != {}
        folded = verdict_node(store, None, config, run_dir=tmp_path).file_verdict
        assert folded.release is Release.RELEASABLE, "an unmeasured model does not gate a release"
        assert folded.triage is Triage.FLAG, "and the safety signal survives the split"
        assert any(LLM_REDACTION_RESIDUE in reason.why for reason in folded.reasons)
        assert folded.llm_redaction["revision"] == "a" * 40

    def test_a_surviving_finding_still_withholds_with_no_re_read_in_sight(
        self,
        store: ProvStore,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The control on the real redaction path: the change must not weaken it."""
        config = _override(tmp_path, LLM_ON)
        _seed_redact_store(store, tmp_path, words=["hello", "alice"], findings=[("PERSON", (1.0, 2.0))])
        _stub_pii(monkeypatch, findings=[("PERSON", "alice")])
        result = redact(store, "recording", config, run_dir=tmp_path, artifacts_dir=_release(tmp_path))
        assert result.verdict.outcome is Outcome.FAIL and result.artifacts == {}
        folded = verdict_node(store, None, config, run_dir=tmp_path).file_verdict
        assert folded.release is Release.WITHHELD
        assert folded.llm_redaction["status"] == "not_run"


class TestTheLlmCheckDegradesHonestly:
    """A model that could not be reached never reads as a model that found nothing."""

    def test_an_unavailable_model_is_recorded_absent_and_does_not_crash(
        self,
        store: ProvStore,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The node concludes; the verdict names the failure; the detector path's answer stands."""
        config = _override(tmp_path, LLM_ON)
        _seed_redact_store(store, tmp_path, words=["hello", "alice"], findings=[("PERSON", (1.0, 2.0))])
        _stub_pii(monkeypatch, findings=[])
        _stub_review(
            monkeypatch,
            [ReviewResult(available=False, failure="OSError: no such model", model_id="stub/model")],
        )
        result = redact(store, "recording", config, run_dir=tmp_path, artifacts_dir=_release(tmp_path))
        assert result.verdict.outcome is Outcome.PASS
        check = _annotation(store)
        assert check["status"] == "absent" and check["failure"] == "OSError: no such model"
        assert "llm" not in result.verdict.why, "REDACT's why is its detector path's; the absence is the annotation's"

    def test_an_absent_model_is_never_a_silent_pass(
        self,
        store: ProvStore,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The absence is an entity as well as a sentence, so a reader who skips the why still sees it."""
        config = _override(tmp_path, LLM_ON)
        _seed_redact_store(store, tmp_path, words=["hello", "alice"], findings=[("PERSON", (1.0, 2.0))])
        _stub_pii(monkeypatch, findings=[])
        _stub_review(monkeypatch, [ReviewResult(available=False, failure="timeout", model_id="stub/model")])
        redact(store, "recording", config, run_dir=tmp_path, artifacts_dir=_release(tmp_path))
        [round_one] = _reviews(store)
        assert round_one.attributes["available"] is False
        assert round_one.attributes["failure"] == "timeout"
        activity = next(a for a in store.activities("REDACT") if a.step == "llm_check")
        model = next(
            store.get_agent(a) for a in store.associated_with(activity.id) if store.get_agent(a).agent_type == "model"
        )
        assert model.commit_sha is None and model.unresolved_reason == "timeout"

    def test_a_model_lost_after_it_already_flagged_keeps_the_flag(
        self,
        store: ProvStore,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A concern already raised is not withdrawn because the next round could not run."""
        config = _override(tmp_path, LLM_ON)
        _seed_redact_store(store, tmp_path, words=["born", "in", "belmont"], findings=[("PERSON", (0.0, 0.5))])
        _stub_pii(monkeypatch, findings=[])
        _stub_review(
            monkeypatch,
            [
                _flags(ReviewFinding(text="belmont", category="LOCATION", why="a town with one clinic")),
                ReviewResult(available=False, failure="CUDA out of memory", model_id="stub/model"),
            ],
        )
        result = redact(store, "recording", config, run_dir=tmp_path, artifacts_dir=_release(tmp_path))
        assert result.verdict.outcome is Outcome.PASS
        check = _annotation(store)
        assert check["status"] == "flagged" and check["failure"] == "CUDA out of memory"


class TestTheReviewerParsesWhatItIsGiven:
    """The backend's own parsing, with no model anywhere near it."""

    def test_reasoning_survives_a_missing_findings_array(self) -> None:
        """The reasoning is the product; a malformed array must not discard it."""
        reasoning, findings = parse_completion("REASONING: nothing identifying remains.")
        assert reasoning == "nothing identifying remains."
        assert findings == []

    def test_a_findings_array_is_read_and_the_reasoning_kept_apart(self) -> None:
        """Both halves come back, and the headings do not leak into either."""
        reasoning, findings = parse_completion(
            'REASONING: the town narrows it.\nFINDINGS: [{"text": "belmont", "category": "location", '
            '"why": "one clinic"}]'
        )
        assert reasoning == "the town narrows it."
        assert [(f.text, f.category, f.why) for f in findings] == [("belmont", "LOCATION", "one clinic")]

    def test_a_quoted_placeholder_in_the_reasoning_does_not_truncate_it(self) -> None:
        """The reasoning quotes the transcript's own [CATEGORY] tokens; splitting on those loses it."""
        reasoning, findings = parse_completion(
            "REASONING: the [PERSON] token already covers the name, so nothing remains.\nFINDINGS: []"
        )
        assert reasoning == "the [PERSON] token already covers the name, so nothing remains."
        assert findings == []

    def test_an_unparsable_array_yields_no_findings_rather_than_raising(self) -> None:
        """A chatty model is not an unreachable one, and neither is a broken one a clean one."""
        reasoning, findings = parse_completion("REASONING: hmm.\nFINDINGS: [not json at all}")
        assert reasoning == "hmm." and findings == []


class TestTheRedactedStream:
    """The masked audio is a stream in the store, resolvable like plain/enhanced/residual."""

    def test_the_redacted_audio_resolves_as_a_named_stream(
        self,
        store: ProvStore,
        redact_config: TriageConfig,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A consumer downstream of REDACT asks the store for it by name, not the release directory."""
        _seed_redact_store(store, tmp_path, words=["hello", "alice"], findings=[("PERSON", (1.0, 2.0))])
        _stub_pii(monkeypatch, findings=[])
        redact(store, "recording", redact_config, run_dir=tmp_path, artifacts_dir=_release(tmp_path))
        stream_id, audio = resolve_stream(store, tmp_path, STREAM_NAME)
        entity = store.get_entity(stream_id)
        assert entity.attributes["name"] == STREAM_NAME
        assert entity.attributes["fill"] == "silence"
        assert entity.attributes["checksum_sha256"]
        window = audio.waveform[:, int(0.95 * SR) : int(2.05 * SR)]
        assert float(window.abs().max()) == 0.0, "the planned extent is not silent in the store's stream"

    def test_the_stream_is_the_masked_audio_not_the_source(
        self,
        store: ProvStore,
        redact_config: TriageConfig,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The control: the source stream still carries signal where the redacted one carries none."""
        _seed_redact_store(store, tmp_path, words=["hello", "alice"], findings=[("PERSON", (1.0, 2.0))])
        _stub_pii(monkeypatch, findings=[])
        redact(store, "recording", redact_config, run_dir=tmp_path, artifacts_dir=_release(tmp_path))
        _, source = resolve_stream(store, tmp_path, "recording")
        window = source.waveform[:, int(1.1 * SR) : int(1.4 * SR)]
        assert float(window.abs().max()) > 0.0

    def test_the_stream_is_registered_even_when_the_artifacts_are_withheld(
        self,
        store: ProvStore,
        redact_config: TriageConfig,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The run directory is the store side, never the release side; a withheld run still records it."""
        _seed_redact_store(store, tmp_path, words=["hello", "alice"], findings=[("PERSON", (1.0, 2.0))])
        _stub_pii(monkeypatch, findings=[("PERSON", "alice")])
        result = redact(store, "recording", redact_config, run_dir=tmp_path, artifacts_dir=_release(tmp_path))
        assert result.artifacts == {}
        assert not _release(tmp_path).exists() or not list(_release(tmp_path).iterdir())
        stream_id, _ = resolve_stream(store, tmp_path, STREAM_NAME)
        assert store.derived_from(stream_id), "the stream records what it came from"

    def test_the_artifact_file_is_still_written_beside_the_stream(
        self,
        store: ProvStore,
        redact_config: TriageConfig,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The stream is additional to the released file, not a replacement for it."""
        _seed_redact_store(store, tmp_path, words=["hello", "alice"], findings=[("PERSON", (1.0, 2.0))])
        _stub_pii(monkeypatch, findings=[])
        result = redact(store, "recording", redact_config, run_dir=tmp_path, artifacts_dir=_release(tmp_path))
        assert result.artifacts["audio"].exists()
        assert result.artifacts["audio"].parent == _release(tmp_path)


class TestWhatTheStoreRecords:
    """One activity per phase, one span per planned extent, and a used edge per read element."""

    def test_the_four_activities_the_spans_and_every_read(
        self,
        store: ProvStore,
        redact_config: TriageConfig,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The provenance a reader needs to see what REDACT read and what it wrote."""
        _seed_redact_store(
            store,
            tmp_path,
            words=["hello", "alice", "in", "boston"],
            findings=[("PERSON", (1.0, 1.5)), ("LOCATION", (3.0, 3.5))],
        )
        _stub_pii(monkeypatch, findings=[])
        findings = {e.attributes["category"]: e.id for e in store.entities("pii")}
        scan = next(e for e in store.entities("measurement") if e.attributes.get("name") == "pii_scan")
        consensus = next(e for e in store.entities("measurement") if e.attributes.get("name") == "consensus_transcript")
        recording = next(e for e in store.entities("stream") if e.attributes.get("name") == "recording")
        words = {w.id for w in store.entities("word")}
        result = redact(store, "recording", redact_config, run_dir=tmp_path, artifacts_dir=_release(tmp_path))

        activities = store.activities("REDACT")
        assert sorted(str(a.step) for a in activities) == ["apply", "llm_check", "plan", "verify"]
        plan_act = next(a for a in activities if a.step == "plan")
        apply_act = next(a for a in activities if a.step == "apply")
        verify_act = next(a for a in activities if a.step == "verify")

        spans = [e for e in store.entities("span") if e.attributes.get("name") == "redaction"]
        assert len(spans) == 2, "one span entity per planned extent"
        for span in spans:
            assert set(span.attributes) == {"name", "category"}, "a name and a category, nothing else"
            assert store.generated_by(span.id) == plan_act.id
            assert span.id in result.view
        by_category = {str(span.attributes["category"]): span for span in spans}
        for category, finding_id in findings.items():
            assert store.derived_from(by_category[category].id) == [finding_id], "derived from the pii it covers"

        markings = {e.id for e in store.entities("assertion") if e.attributes.get("label") == "pii"}
        assert set(store.uses_of(plan_act.id)) == {scan.id, *findings.values(), *markings}, (
            "the plan read the scan, the findings, and the markings the re-plan would widen to"
        )
        apply_used = set(store.uses_of(apply_act.id))
        assert recording.id in apply_used, "the recording stream it redacted"
        assert words <= apply_used, "every consensus word the transcript read"
        assert {span.id for span in spans} <= apply_used
        verify_used = set(store.uses_of(verify_act.id))
        assert consensus.id in verify_used, "the transcript measurement the verified text came from"
        assert words <= verify_used, "and the words it was rendered from"
        assert {span.id for span in spans} <= verify_used, "and the extents that shaped it"

        verdict = store.get_entity(result.verdict_entity_id)
        assert store.generated_by(verdict.id) == verify_act.id
        associated = [set(store.associated_with(a.id)) for a in (plan_act, apply_act, verify_act)]
        assert associated[0] == associated[1] == associated[2], "one agent answerable for all three steps"
        (software,) = associated[0]
        assert store.get_agent(software).agent_type == "software", "and it is software, not a model"

    def test_the_widen_path_records_what_it_read_and_what_caused_it(
        self,
        store: ProvStore,
        redact_config: TriageConfig,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A re-plan that leaves no edges is a redaction nobody can trace to its cause.

        The plan read the markings, the verification read the transcript it verified, and the span
        the re-plan added exists because one marking said so. Each was an empty relation before.
        """
        _seed_redact_store(
            store,
            tmp_path,
            words=["jane", "doe", "here"],
            findings=[("PERSON", (0.0, 0.5))],
            extra_marks=[("doe", "PERSON")],
        )
        _stub_pii_sequence(monkeypatch, [[("PERSON", "doe")], []])
        doe = next(w for w in store.entities("word") if w.attributes.get("text") == "doe")
        marking = next(e for e in store.entities("assertion") if doe.id in store.derived_from(e.id))
        consensus = next(e for e in store.entities("measurement") if e.attributes.get("name") == "consensus_transcript")
        redact(store, "recording", redact_config, run_dir=tmp_path, artifacts_dir=_release(tmp_path))

        plan_act = next(a for a in store.activities("REDACT") if a.step == "plan")
        verify_act = next(a for a in store.activities("REDACT") if a.step == "verify")
        assert marking.id in store.uses_of(plan_act.id), "the plan consulted the marking it widened to"
        assert consensus.id in store.uses_of(verify_act.id), "verification read the transcript measurement"
        assert {w.id for w in store.entities("word")} <= set(store.uses_of(verify_act.id))

        spans = [e for e in store.entities("span") if e.attributes.get("name") == "redaction"]
        widened = next(
            span for span in spans if span.extent is not None and span.extent[0] < 1.5 and span.extent[1] > 1.0
        )
        assert marking.id in store.derived_from(widened.id), "the widened span names the marking that caused it"
