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
from senselab.audio.workflows.triage.config import TriageConfig, load_triage_config
from senselab.audio.workflows.triage.nodes import redact as redact_module
from senselab.audio.workflows.triage.nodes.redact import redact
from senselab.audio.workflows.triage.vocabulary import Outcome
from senselab.text.tasks.pii_detection.api import PiiScan, PiiSpan
from senselab.text.tasks.pii_detection.api import scan_for_pii as real_scan_for_pii
from senselab.utils.prov_store import Entity, ProvStore

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
            attributes={"text": text, "confidence": 0.9, "coverage": 1.0, "index": index},
        )
        store.was_generated_by(word_id, consensus)
        word_ids.append(word_id)
    transcript_id = store.entity(
        prov_type="measurement",
        extent=None,
        attributes={
            "name": "consensus_transcript",
            "signal": "plain",
            "words": [
                {"text": text, "start": _word_extent(i)[0], "end": _word_extent(i)[1]} for i, text in enumerate(words)
            ],
            "provenance": {"operator": "consensus_words/resample", "n_words": len(words)},
            "word_ids": word_ids,
            "event_ids": [],
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

    def test_a_null_fill_refuses_before_any_store_write(
        self, store: ProvStore, config: TriageConfig, tmp_path: Path
    ) -> None:
        """The key ships with no default; two artifacts under different fills are not comparable."""
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
        """The margin is unmeasured, so a run that does not declare one gets no answer."""
        config = _override(tmp_path, "redaction:\n  fill: silence\n")
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
        assert result.artifacts.keys() == {"audio", "transcript"}

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
        assert result.artifacts.keys() == {"audio", "transcript"}
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
        floating = store.entity(prov_type="word", extent=None, attributes={"text": "unplaceable-sentinel"})
        store.was_generated_by(floating, consensus)
        _stub_pii(monkeypatch, findings=[])
        result = redact(store, "recording", redact_config, run_dir=tmp_path, artifacts_dir=_release(tmp_path))
        text = result.artifacts["transcript"].read_text()
        assert "unplaceable-sentinel" not in text
        assert "[UNPLACED]" in text
        assert _verdict_entity(store, "REDACT").attributes["unplaced_words_n"] == 1

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


class TestWhatTheStoreRecords:
    """One activity per phase, one span per planned extent, and a used edge per read element."""

    def test_the_three_activities_the_spans_and_every_read(
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
        recording = next(e for e in store.entities("stream") if e.attributes.get("name") == "recording")
        words = {w.id for w in store.entities("word")}
        result = redact(store, "recording", redact_config, run_dir=tmp_path, artifacts_dir=_release(tmp_path))

        activities = store.activities("REDACT")
        assert sorted(str(a.step) for a in activities) == ["apply", "plan", "verify"]
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

        assert set(store.uses_of(plan_act.id)) == {scan.id, *findings.values()}
        apply_used = set(store.uses_of(apply_act.id))
        assert recording.id in apply_used, "the recording stream it redacted"
        assert words <= apply_used, "every consensus word the transcript read"
        assert {span.id for span in spans} <= apply_used

        verdict = store.get_entity(result.verdict_entity_id)
        assert store.generated_by(verdict.id) == verify_act.id
        associated = [set(store.associated_with(a.id)) for a in (plan_act, apply_act, verify_act)]
        assert associated[0] == associated[1] == associated[2], "one agent answerable for all three steps"
        (software,) = associated[0]
        assert store.get_agent(software).agent_type == "software", "and it is software, not a model"
