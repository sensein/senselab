"""REDACT node tests. The PII scan is faked at the node module; redaction and the store run real.

Nothing here fakes a recognizer, because REDACT runs none: verification is a re-scan of the redacted
consensus text. The seeder writes PREPROCESS's consensus words and SPEECH's findings, which are the
only two authors REDACT reads.
"""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pytest
import soundfile as sf

from senselab.audio.data_structures import Audio
from senselab.audio.data_structures.audio_hints import AudioHints, ExpectedSpeech
from senselab.audio.workflows.triage.config import TriageConfig, load_triage_config
from senselab.audio.workflows.triage.nodes import redact as redact_module
from senselab.audio.workflows.triage.nodes import verdict as verdict_module
from senselab.audio.workflows.triage.nodes.common import live_entities, resolve_stream
from senselab.audio.workflows.triage.nodes.redact import (
    AGREED_MASKED,
    AGREED_PLACED,
    DETECTOR,
    MASK_PARTLY_UNMASKED,
    MASK_TRIMMED,
    MASK_UNCHANGED,
    MASK_UNMASKED,
    MASKED,
    NEW,
    PII_LEDGER,
    PLACED_SUBSTRING,
    PLACED_WORDS,
    PROPAGATED,
    PROPAGATION_MASK,
    PROPAGATION_RELEASE,
    PROPOSED_BY_REVIEWER,
    RELEASED_BY_KIND,
    RELEASED_FILES,
    RELEASED_NOT_PROPER,
    REVIEWER,
    STREAM_NAME,
    TASK_EVENT_LABEL,
    UNMASKED_BY_APPROVAL,
    UNMASKED_BY_REVIEWER,
    UNMASKED_BY_TRIM,
    MaskPlan,
    mask_plan,
    redact,
    settle_release,
    time_by_kind,
)
from senselab.audio.workflows.triage.vocabulary import (
    FINDINGS_ARE_TASK_CONTENT,
    NO_CONTENT_MASKED,
    REDACTION_LLM_ANNOTATION,
    REVIEWER_CLEARED_RESCAN,
    REVIEWER_UNMASKED_SOME,
    TASK_CONTENT_UNMASKED,
    UNPLACED_CLEARED,
    UNPLACED_OPEN,
    UNPLACED_PLACED,
    UNPLACED_UNREAD,
    Outcome,
)
from senselab.text.tasks.pii_detection.api import PiiScan, PiiSpan
from senselab.text.tasks.pii_detection.api import scan_for_pii as real_scan_for_pii
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
    words: Sequence[str] = ("hello", "Alice"),
    findings: Sequence[tuple[Any, ...]] = (),
    extra_marks: Sequence[tuple[str, str]] = (),
    target_speaker: str | None = None,
    scanned: bool = True,
    scanned_by: Sequence[str] = ALL_DETECTORS,
    scan_failed: Sequence[str] = (),
    timings: dict[int, dict[str, tuple[float, float]]] | None = None,
    residue: Sequence[int] | None = None,
    unplaced: Sequence[tuple[str, str]] = (),
    readings: dict[int, dict[str, str]] | None = None,
    haystacks: dict[int, str] | None = None,
    recording_stem: str = "plain",
) -> None:
    """Write the store PREPROCESS and SPEECH leave for REDACT, with ``tmp_path`` as the run dir.

    ``findings`` are ``(category, (start, end))`` or ``(category, (start, end), speaker)``. Each
    writes a ``pii`` entity and a ``label``/``pii`` assertion derived from every consensus word it
    overlaps — the store's shared shape for a marking. ``extra_marks`` are ``(word_text, category)``
    markings placed on a word the finding's own extent does not reach, which is the state a
    re-planning pass exists to widen. ``target_speaker`` writes SPEECH's verdict so a speaker-scoped
    reader has something to scope by. ``timings`` gives a word, by index, its sources' own timings,
    so its hull can reach past its derived extent. ``residue`` names, by index, the words the scan's
    residue holds; every word when None. ``readings`` gives a word, by index, each source's surface;
    ``haystacks`` gives a finding, by index, the text it was read off. ``recording_stem`` names the
    recording's file, which declares its task.
    """
    ends = [_word_extent(i)[1] for i in range(len(words))] + [float(extent[1]) for _c, extent, *_r in findings]
    duration_s = max([5.0, *(end + 1.0 for end in ends)])
    rng = np.random.default_rng(0)
    wave = (0.05 * rng.standard_normal(int(duration_s * SR))).astype(np.float32)
    (tmp_path / "streams").mkdir(parents=True, exist_ok=True)
    sf.write(str(tmp_path / "streams" / "plain.wav"), wave, SR)
    if recording_stem != "plain":
        sf.write(str(tmp_path / "streams" / f"{recording_stem}.wav"), wave, SR)

    software = store.agent(agent_type="software", version="senselab test-seed")
    pre = store.activity(node="PREPROCESS", step="condition", parameters={})
    store.was_associated_with(pre, software)
    for name in ("recording", "plain"):
        path = f"streams/{recording_stem}.wav" if name == "recording" else "streams/plain.wav"
        stream_id = store.entity(
            prov_type="stream",
            extent=(0.0, duration_s),
            attributes={"name": name, "path": path, "sampling_rate": SR, "channels": 1},
        )
        store.was_generated_by(stream_id, pre)

    consensus = store.activity(node="PREPROCESS", step="consensus", parameters={})
    store.was_associated_with(consensus, software)
    word_ids: list[str] = []
    for index, text in enumerate(words):
        word_id = store.entity(
            prov_type="word",
            extent=_word_extent(index),
            attributes=word_attributes(
                text,
                _word_extent(index),
                index=index,
                timings=(timings or {}).get(index),
                readings=(readings or {}).get(index),
            ),
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

    for position, (category, extent, *_rest) in enumerate(findings):
        bounds = (float(extent[0]), float(extent[1]))
        covered = [
            word_ids[i] for i in range(len(words)) if _word_extent(i)[0] < bounds[1] and _word_extent(i)[1] > bounds[0]
        ]
        pii_id = store.entity(
            prov_type="pii",
            extent=bounds,
            attributes={
                "category": category,
                "source": "presidio",
                "haystack": (haystacks or {}).get(position, "consensus"),
                "word_ids": covered,
                "detectors_used": list(scanned_by),
                "detectors_failed": list(scan_failed),
            },
        )
        store.was_generated_by(pii_id, pii_act)
        _mark(str(category), bounds, covered)
    for text, category in extra_marks:
        index = list(words).index(text)
        _mark(str(category), _word_extent(index), [word_ids[index]])

    if scanned:
        scan_id = store.entity(
            prov_type="measurement",
            extent=None,
            attributes={
                "name": "pii_scan",
                "scanned_by": list(scanned_by),
                "failed": list(scan_failed),
                "residue_word_ids": list(word_ids) if residue is None else [word_ids[i] for i in residue],
                "unplaced_findings": [
                    {"category": category, "text": text, "source": "gliner", "haystack": "consensus"}
                    for category, text in unplaced
                ],
            },
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
        _seed_redact_store(store, tmp_path, words=["my", "name", "is", "Alice"], findings=[("PERSON", (3.0, 4.0))])
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
        _seed_redact_store(store, tmp_path, words=["hello", "Alice"], findings=[("PERSON", (1.0, 2.0))])
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
        _seed_redact_store(store, tmp_path, words=["hello", "Alice"], findings=[("PERSON", (1.0, 2.0))])
        for survivors in ([], [("PERSON", "Alice")]):
            other = ProvStore(run_id="bounded")
            _seed_redact_store(other, tmp_path, words=["hello", "Alice"], findings=[("PERSON", (1.0, 2.0))])
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
        _seed_redact_store(store, tmp_path, words=["hello", "Alice"], findings=[("PERSON", (1.0, 2.0))])
        _stub_pii_sequence(monkeypatch, [[("PERSON", "Alice")], []])
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
        _seed_redact_store(store, tmp_path, words=["hello", "Alice"], findings=[("PERSON", (1.0, 2.0))])
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
        _seed_redact_store(store, tmp_path, words=["hello", "Alice"], findings=[("PERSON", (1.0, 2.0))])
        _stub_pii_sequence(monkeypatch, [[("PERSON", "Alice")], [("PERSON", "Alice")]])
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
        _seed_redact_store(store, tmp_path, words=["hello", "Alice"], findings=[("PERSON", (1.0, 2.0))])
        scanned = _stub_pii(monkeypatch, findings=[("PERSON", "Alice")])
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
        _seed_redact_store(store, tmp_path, words=["hello", "Alice"], findings=[("PERSON", (1.0, 2.0))])
        scanned = _stub_pii(monkeypatch, findings=[])
        redact(store, "recording", redact_config, run_dir=tmp_path, artifacts_dir=_release(tmp_path))
        assert len(scanned) == 1
        assert _verdict_entity(store, "REDACT").attributes["replanned_n"] == 0


class TestAPlaceholderIsNotASurvivor:
    """A detector that reads the placeholder the plan wrote has found the redaction, not a survivor."""

    @pytest.mark.parametrize("echo", ["[PERSON]", "PERSON", "[PERSON].", " [PERSON] "])
    def test_a_placeholder_echoed_back_passes_without_a_replan(
        self,
        echo: str,
        store: ProvStore,
        redact_config: TriageConfig,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """With or without its brackets, the placeholder alone is the mask."""
        _seed_redact_store(store, tmp_path, words=["hello", "Alice"], findings=[("PERSON", (1.0, 2.0))])
        scanned = _stub_pii(monkeypatch, findings=[("PERSON", echo)])
        result = redact(store, "recording", redact_config, run_dir=tmp_path, artifacts_dir=_release(tmp_path))
        assert result.verdict.outcome is Outcome.PASS
        assert scanned == ["hello [PERSON]"]
        assert _verdict_entity(store, "REDACT").attributes["replanned_n"] == 0

    @pytest.mark.parametrize("echo", ["[NAME+PERSON]", "NAME+PERSON", "NAME", "PERSON"])
    def test_a_merged_placeholder_and_each_category_it_joins_are_the_mask(
        self,
        echo: str,
        store: ProvStore,
        redact_config: TriageConfig,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Two findings on one word merge into one placeholder, and either category may come back."""
        _seed_redact_store(
            store, tmp_path, words=["hello", "Alice"], findings=[("PERSON", (1.0, 2.0)), ("NAME", (1.0, 2.0))]
        )
        scanned = _stub_pii(monkeypatch, findings=[("PERSON", echo)])
        result = redact(store, "recording", redact_config, run_dir=tmp_path, artifacts_dir=_release(tmp_path))
        assert len(scanned) == 1
        assert "+" in scanned[0]
        assert result.verdict.outcome is Outcome.PASS

    def test_a_span_that_reaches_past_the_placeholder_still_survives(
        self,
        store: ProvStore,
        redact_config: TriageConfig,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Words the recording said, beside the placeholder, are still read as a finding."""
        _seed_redact_store(
            store, tmp_path, words=["he", "walks", "daily", "twice"], findings=[("DATE_TIME", (2.0, 3.0))]
        )
        _stub_pii(monkeypatch, findings=[("DATE_TIME", "[DATE_TIME] twice")])
        result = redact(store, "recording", redact_config, run_dir=tmp_path, artifacts_dir=_release(tmp_path))
        assert result.verdict.outcome is Outcome.FAIL
        assert _verdict_entity(store, "REDACT").attributes["unremediable"] == ["DATE_TIME"]

    def test_the_word_a_placeholder_replaced_still_survives(
        self,
        store: ProvStore,
        redact_config: TriageConfig,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The control: a surviving surface is not a placeholder, whatever the scanner was handed."""
        _seed_redact_store(store, tmp_path, words=["hello", "Alice"], findings=[("PERSON", (1.0, 2.0))])
        _stub_pii(monkeypatch, findings=[("PERSON", "Alice")])
        result = redact(store, "recording", redact_config, run_dir=tmp_path, artifacts_dir=_release(tmp_path))
        assert result.verdict.outcome is Outcome.FAIL


def _settle(store: ProvStore, release: str, ground: str | None, tmp_path: Path) -> dict[str, Path]:
    """Settle the release directory the way every driver does, under a silence fill."""
    return settle_release(
        store, release, ground, run_dir=tmp_path, artifacts_dir=_release(tmp_path), bleep_hz=None, fill="silence"
    )


class TestTheFoldSettlesTheReleaseOfAFail:
    """REDACT writes a copy only on a pass; where the fold releases a fail, the copy comes from the store."""

    def _failed(self, store: ProvStore, config: TriageConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        """A REDACT fail whose re-scan still reads the name, and its (empty) release directory."""
        _seed_redact_store(store, tmp_path, words=["hello", "Alice"], findings=[("PERSON", (1.0, 2.0))])
        _stub_pii(monkeypatch, findings=[("PERSON", "Alice")])
        result = redact(store, "recording", config, run_dir=tmp_path, artifacts_dir=_release(tmp_path))
        assert result.verdict.outcome is Outcome.FAIL and result.artifacts == {}
        return _release(tmp_path)

    def test_a_released_fail_gets_the_masked_copy(
        self,
        store: ProvStore,
        redact_config: TriageConfig,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The triple a pass would have written, over the plan REDACT left in the store."""
        released = self._failed(store, redact_config, tmp_path, monkeypatch)
        written = _settle(store, "redacted", REVIEWER_CLEARED_RESCAN, tmp_path)
        assert sorted(path.name for path in written.values()) == sorted(RELEASED_FILES)
        assert (released / "transcript.txt").read_text() == "hello [PERSON]\n"
        assert "Alice" not in (released / "consensus.json").read_text()

    def test_a_withheld_fail_leaves_no_copy(
        self,
        store: ProvStore,
        redact_config: TriageConfig,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A fold that withholds again removes a copy an earlier fold released."""
        released = self._failed(store, redact_config, tmp_path, monkeypatch)
        _settle(store, "redacted", REVIEWER_CLEARED_RESCAN, tmp_path)
        assert _settle(store, "withheld", None, tmp_path) == {}
        assert not any((released / name).exists() for name in RELEASED_FILES)

    def test_a_withheld_pass_leaves_no_copy(
        self,
        store: ProvStore,
        redact_config: TriageConfig,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The fold, not REDACT, decides what the directory holds; a withholding empties REDACT's own copy."""
        _seed_redact_store(store, tmp_path, words=["hello", "Alice"], findings=[("PERSON", (1.0, 2.0))])
        _stub_pii(monkeypatch, findings=[])
        result = redact(store, "recording", redact_config, run_dir=tmp_path, artifacts_dir=_release(tmp_path))
        assert result.verdict.outcome is Outcome.PASS
        assert all((_release(tmp_path) / name).exists() for name in RELEASED_FILES)
        assert _settle(store, "withheld", None, tmp_path) == {}
        assert not any((_release(tmp_path) / name).exists() for name in RELEASED_FILES)

    def test_a_pass_released_as_planned_is_rewritten_identically(
        self,
        store: ProvStore,
        redact_config: TriageConfig,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Re-settling REDACT's own release reproduces it, so no earlier fold's copy can linger."""
        _seed_redact_store(store, tmp_path, words=["hello", "Alice"], findings=[("PERSON", (1.0, 2.0))])
        _stub_pii(monkeypatch, findings=[])
        redact(store, "recording", redact_config, run_dir=tmp_path, artifacts_dir=_release(tmp_path))
        before = (_release(tmp_path) / "transcript.txt").read_text()
        _settle(store, "redacted", None, tmp_path)
        assert (_release(tmp_path) / "transcript.txt").read_text() == before == "hello [PERSON]\n"


_RESET_WORDS = ("i", "met", "Alice", "in", "Brooklyn", "today")
_RESET_FINDINGS = [("PERSON", _word_extent(2)), ("LOCATION", _word_extent(4))]


FULL_REVIEW_INPUTS = {
    "version": 1,
    "prompt_version": 10,
    "task": True,
    "consensus_transcript": True,
    "asr_readings": True,
    "recognisers": ["asr_crisperwhisper", "asr_qwen"],
    "columns_n": 1,
}
"""A reading given the task, the consensus transcript and both recognisers' readings."""


def _annotate(
    store: ProvStore,
    proposal: Sequence[dict[str, str]],
    *,
    original: str = "clean",
    redaction: str = "",
    inputs: Mapping[str, Any] | None = None,
) -> None:
    """REVIEW's annotation, carrying one proposal and, where given, the inputs its reading recorded."""
    agent = store.agent(agent_type="software", version="senselab test-review")
    activity = store.activity(node="REVIEW", step="read", parameters={})
    store.was_associated_with(activity, agent)
    annotation = store.entity(
        prov_type="measurement",
        extent=None,
        attributes={
            "name": REDACTION_LLM_ANNOTATION,
            "status": "flagged",
            "original": original,
            "redaction": redaction,
            "proposal": [dict(entry) for entry in proposal],
            **({"review_inputs": dict(inputs)} if inputs is not None else {}),
        },
    )
    store.was_generated_by(annotation, activity)


def _release_entry(text: str, category: str = "OTHER") -> dict[str, str]:
    """One ``release`` entry of a proposal."""
    return {"text": text, "action": "release", "category": category, "why": ""}


def _place_release(text: str, reason: str = "public_landmark_or_general_knowledge") -> dict[str, str]:
    """One ``release`` entry letting a place through, with the reason policy v8 requires."""
    return {"text": text, "action": "release", "category": "LOCATION", "why": "", "place_reason": reason}


def _redact_entry(text: str, category: str = "PERSON") -> dict[str, str]:
    """One ``redact`` entry of a proposal."""
    return {"text": text, "action": "redact", "category": category, "why": ""}


def _plan(store: ProvStore, *, applies: bool = True, approvals: Sequence[str] = ()) -> MaskPlan:
    """The word-level plan under a 50 ms margin, as VERDICT computes it."""
    return mask_plan(store, reviewer_applies=applies, padding_ms=50, name_approvals=approvals)


def _write_ledger(store: ProvStore, plan: MaskPlan, release: str, ground: str | None) -> None:
    """The ledger VERDICT writes, so the release is settled from it the way every driver settles it."""
    agent = store.agent(agent_type="software", version="senselab test-verdict")
    activity = store.activity(node="VERDICT", step=None, parameters={})
    store.was_associated_with(activity, agent)
    ledger = store.entity(
        prov_type="measurement", extent=None, attributes=plan.record(release=release, release_ground=ground)
    )
    store.was_generated_by(ledger, activity)


def _states(plan: MaskPlan) -> dict[str, str]:
    """Each covered word's surface to its state."""
    return {word.text: word.state for mask in plan.masks for word in mask.words}


def _silent(path: Path, start: float, end: float) -> bool:
    """Whether the released audio is all zeros over an interval."""
    wave = Audio(filepath=str(path)).waveform.numpy()[0]
    return float(np.abs(wave[int(start * SR) : int(end * SR)]).max()) == 0.0


class TestTheWordLevelMaskRule:
    """Owner, 2026-09-27: the reviewer unmasks exactly the words it names, and no mask keeps a non-content word."""

    def _passed(
        self,
        store: ProvStore,
        config: TriageConfig,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        findings: Sequence[tuple[Any, ...]] = tuple(_RESET_FINDINGS),
    ) -> None:
        """A REDACT pass over "i met Alice in Brooklyn today"."""
        _seed_redact_store(store, tmp_path, words=list(_RESET_WORDS), findings=findings)
        _stub_pii(monkeypatch, findings=[])
        result = redact(store, "recording", config, run_dir=tmp_path, artifacts_dir=_release(tmp_path))
        assert result.verdict.outcome is Outcome.PASS

    def test_no_release_entry_keeps_every_content_mask(
        self, store: ProvStore, redact_config: TriageConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without a reading, and with one that releases nothing, masks over content words stand as planned."""
        self._passed(store, redact_config, tmp_path, monkeypatch)
        plan = _plan(store)
        assert not plan.changed and plan.final == plan.planned
        _annotate(store, [])
        assert [mask.outcome for mask in _plan(store).masks] == [MASK_UNCHANGED, MASK_UNCHANGED]

    def test_a_released_name_takes_its_lower_case_words_with_it(
        self, store: ProvStore, redact_config: TriageConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Productive vocabulary: "Gladiator fighter" tagged PERSON with "Gladiator" released frees "fighter"."""
        span = (_word_extent(1)[0], _word_extent(2)[1])
        _seed_redact_store(
            store, tmp_path, words=["a", "Gladiator", "fighter", "from", "rome"], findings=[("PERSON", span)]
        )
        _stub_pii(monkeypatch, findings=[])
        result = redact(store, "recording", redact_config, run_dir=tmp_path, artifacts_dir=_release(tmp_path))
        assert result.verdict.outcome is Outcome.PASS
        _annotate(store, [_release_entry("Gladiator", "PERSON")])
        proposed = {word.text: word for mask in _plan(store).masks for word in mask.words}
        assert (proposed["Gladiator"].state, proposed["fighter"].state) == (MASKED, MASKED)
        plan = _plan(store, approvals=("Gladiator",))
        words = {word.text: word for mask in plan.masks for word in mask.words}
        assert words["Gladiator"].state == UNMASKED_BY_APPROVAL
        assert words["fighter"].state == UNMASKED_BY_REVIEWER and words["fighter"].with_head
        assert not words["Gladiator"].with_head

    def test_a_lower_case_word_a_redact_entry_quotes_stays_with_its_mask(
        self, store: ProvStore, redact_config: TriageConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The head release never frees a word the same reading asks to hide."""
        span = (_word_extent(1)[0], _word_extent(2)[1])
        _seed_redact_store(
            store, tmp_path, words=["a", "Gladiator", "fighter", "from", "rome"], findings=[("PERSON", span)]
        )
        _stub_pii(monkeypatch, findings=[])
        redact(store, "recording", redact_config, run_dir=tmp_path, artifacts_dir=_release(tmp_path))
        _annotate(store, [_release_entry("Gladiator", "PERSON"), _redact_entry("fighter", "PERSON")])
        words = {word.text: word for mask in _plan(store).masks for word in mask.words}
        assert not words["fighter"].with_head and words["fighter"].state != UNMASKED_BY_REVIEWER

    def test_a_named_word_is_unmasked_and_the_copy_keeps_the_rest(
        self, store: ProvStore, redact_config: TriageConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Releasing the place keeps the name; the re-masked copy shows the place and hides the name."""
        self._passed(store, redact_config, tmp_path, monkeypatch)
        _annotate(store, [_place_release("in Brooklyn,")])
        plan = _plan(store)
        assert [mask.outcome for mask in plan.masks] == [MASK_UNCHANGED, MASK_UNMASKED]
        assert _states(plan)["Brooklyn"] == UNMASKED_BY_REVIEWER
        _write_ledger(store, plan, "redacted", REVIEWER_UNMASKED_SOME)
        _settle(store, "redacted", REVIEWER_UNMASKED_SOME, tmp_path)
        released = _release(tmp_path)
        assert (released / "transcript.txt").read_text() == "i met [PERSON] in Brooklyn today\n"
        assert _silent(released / "audio.wav", 2.1, 2.4)
        assert not _silent(released / "audio.wav", 4.1, 4.4)

    def test_part_of_a_mask_is_unmasked_and_the_mask_is_split(
        self, store: ProvStore, redact_config: TriageConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """One mask over "Alice in Brooklyn": the named place goes, the function word goes, the name stays."""
        self._passed(store, redact_config, tmp_path, monkeypatch, findings=[("PERSON", (2.0, 4.5))])
        _annotate(
            store,
            [{**_release_entry("Brooklyn", "PERSON"), "relabel": "other_non_person"}],
            original="carries_pii",
        )
        plan = _plan(store)
        (mask,) = plan.masks
        assert mask.outcome == MASK_PARTLY_UNMASKED
        assert _states(plan) == {"Alice": MASKED, "in": UNMASKED_BY_TRIM, "Brooklyn": UNMASKED_BY_REVIEWER}
        (extent,) = plan.final
        assert extent.start <= 2.0 and extent.end >= 2.5
        assert extent.end <= 3.0, "the padding stops where the unmasked neighbour begins"
        _write_ledger(store, plan, "redacted", REVIEWER_UNMASKED_SOME)
        _settle(store, "redacted", REVIEWER_UNMASKED_SOME, tmp_path)
        released = _release(tmp_path)
        assert (released / "transcript.txt").read_text() == "i met [PERSON] in Brooklyn today\n"
        assert _silent(released / "audio.wav", 2.05, 2.45)
        assert not _silent(released / "audio.wav", 3.1, 3.4)
        assert not _silent(released / "audio.wav", 4.1, 4.4)

    def test_a_mask_keeps_no_function_word_with_or_without_a_reading(
        self, store: ProvStore, redact_config: TriageConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A mask over "Alice in" keeps only the name, whether the padding or the finding caught "in"."""
        self._passed(store, redact_config, tmp_path, monkeypatch, findings=[("PERSON", (2.0, 3.5))])
        plan = _plan(store, applies=False)
        assert [mask.outcome for mask in plan.masks] == [MASK_TRIMMED]
        assert _states(plan) == {"Alice": MASKED, "in": UNMASKED_BY_TRIM}

    def test_a_word_only_the_padding_reached_is_not_under_the_mask(
        self, store: ProvStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A 600 ms margin reaches "met" and "in"; the mask is the finding's own word and nothing else."""
        config = _override(tmp_path, "redaction:\n  padding_ms: 600\n  fill: silence\n")
        self._passed(store, config, tmp_path, monkeypatch, findings=[("PERSON", _word_extent(2))])
        plan = mask_plan(store, reviewer_applies=False, padding_ms=600)
        (mask,) = plan.masks
        assert mask.outcome == MASK_UNCHANGED
        assert {word.text: (word.state, word.finding) for word in mask.words} == {"Alice": (MASKED, True)}
        assert plan.changed, "REDACT's padded extent reached the neighbours; the released copy does not"
        (extent,) = plan.final
        assert (extent.start, extent.end) == (1.5, 3.0), "the margin stops at each unmasked neighbour"

    def test_a_name_spelled_as_a_function_word_is_never_trimmed(
        self, store: ProvStore, redact_config: TriageConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A "Will" marked PERSON keeps its mask through the trim; only a human's approval unmasks it."""
        _seed_redact_store(store, tmp_path, words=["i", "met", "Will", "today"], findings=[("PERSON", _word_extent(2))])
        _stub_pii(monkeypatch, findings=[])
        redact(store, "recording", redact_config, run_dir=tmp_path, artifacts_dir=_release(tmp_path))
        unprotected = mask_plan(store, reviewer_applies=False, padding_ms=50)
        assert _states(unprotected)["Will"] == UNMASKED_BY_TRIM
        protected = mask_plan(store, reviewer_applies=False, padding_ms=50, protected_categories=("PERSON", "NAME"))
        assert _states(protected)["Will"] == MASKED
        (word,) = [word for mask in protected.masks for word in mask.words if word.text == "Will"]
        assert (word.categories, word.proper, word.locked) == (("PERSON",), True, "person")
        _annotate(store, [_release_entry("will", "PERSON")])
        proposed = mask_plan(store, reviewer_applies=True, padding_ms=50, protected_categories=("PERSON", "NAME"))
        assert _states(proposed)["Will"] == MASKED
        assert proposed.name_release_proposed == ("will",)
        approved = mask_plan(
            store,
            reviewer_applies=True,
            padding_ms=50,
            protected_categories=("PERSON", "NAME"),
            name_approvals=("Will",),
        )
        assert _states(approved)["Will"] == UNMASKED_BY_APPROVAL

    def test_a_function_word_a_name_finding_spans_is_still_trimmed(
        self, store: ProvStore, redact_config: TriageConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A PERSON finding over "and the" does not make them names; lower case, they are trimmed."""
        _seed_redact_store(store, tmp_path, words=["Alice", "and", "the", "dog"], findings=[("PERSON", (0.0, 2.5))])
        _stub_pii(monkeypatch, findings=[])
        redact(store, "recording", redact_config, run_dir=tmp_path, artifacts_dir=_release(tmp_path))
        plan = mask_plan(store, reviewer_applies=False, padding_ms=50, protected_categories=("PERSON", "NAME"))
        assert _states(plan)["and"] == UNMASKED_BY_TRIM and _states(plan)["the"] == UNMASKED_BY_TRIM
        assert _states(plan)["Alice"] == MASKED

    def test_a_lone_capitalised_function_word_tagged_person_is_trimmed(
        self, store: ProvStore, redact_config: TriageConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Owner, 2026-10-01: a lone "She" a detector tagged PERSON is no name; the trim releases it."""
        _seed_redact_store(store, tmp_path, words=["and", "She", "said", "so"], findings=[("PERSON", _word_extent(1))])
        _stub_pii(monkeypatch, findings=[])
        redact(store, "recording", redact_config, run_dir=tmp_path, artifacts_dir=_release(tmp_path))
        plan = mask_plan(store, reviewer_applies=False, padding_ms=50, protected_categories=("PERSON", "NAME"))
        assert _states(plan)["She"] == UNMASKED_BY_TRIM

    def test_a_function_word_inside_a_multi_word_name_stays_masked(
        self, store: ProvStore, redact_config: TriageConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The word "The" in "The Green Mile" is part of the name beside "Green" and "Mile"; it stays masked."""
        _seed_redact_store(
            store, tmp_path, words=["i", "watched", "The", "Green", "Mile"], findings=[("PERSON", (2.0, 5.0))]
        )
        _stub_pii(monkeypatch, findings=[])
        redact(store, "recording", redact_config, run_dir=tmp_path, artifacts_dir=_release(tmp_path))
        plan = mask_plan(store, reviewer_applies=False, padding_ms=50, protected_categories=("PERSON", "NAME"))
        states = _states(plan)
        assert states["The"] == MASKED and states["Green"] == MASKED and states["Mile"] == MASKED

    def test_an_abbreviated_place_in_capitals_stays_masked(
        self, store: ProvStore, redact_config: TriageConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The token "LA" is the Spanish article in lower case; in capitals a detector's place is an abbreviation."""
        _seed_redact_store(
            store, tmp_path, words=["we", "moved", "to", "LA,"], findings=[("LOCATION", _word_extent(3))]
        )
        _stub_pii(monkeypatch, findings=[])
        redact(store, "recording", redact_config, run_dir=tmp_path, artifacts_dir=_release(tmp_path))
        plan = mask_plan(store, reviewer_applies=False, padding_ms=50, protected_categories=("LOCATION", "LOC"))
        assert _states(plan)["LA,"] == MASKED

    def test_los_angeles_stays_masked_as_a_place(
        self, store: ProvStore, redact_config: TriageConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A two-word place keeps both words under the place protection."""
        _seed_redact_store(
            store, tmp_path, words=["we", "lived", "in", "Los", "Angeles"], findings=[("LOCATION", (3.0, 5.0))]
        )
        _stub_pii(monkeypatch, findings=[])
        redact(store, "recording", redact_config, run_dir=tmp_path, artifacts_dir=_release(tmp_path))
        plan = mask_plan(store, reviewer_applies=False, padding_ms=50, protected_categories=("LOCATION", "LOC"))
        states = _states(plan)
        assert states["Los"] == MASKED and states["Angeles"] == MASKED

    def test_the_packaged_config_protects_person_and_name(self) -> None:
        """The shipped key names the person and the place categories."""
        from senselab.audio.workflows.triage.vocabulary import FoldPolicy

        assert FoldPolicy.from_config(load_triage_config()).trim_protected_categories == (
            "PERSON",
            "NAME",
            "LOCATION",
            "LOC",
        )

    def test_a_trimmed_word_is_released_where_the_trim_leaves_the_extent_as_planned(
        self, store: ProvStore, redact_config: TriageConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """r6's "back in [DATE_TIME]": the trim releases "in" though its hull reaches the mask.

        One source times "in" into the padded mask and the re-cut extent equals REDACT's own; the
        released text still shows "in".
        """
        _seed_redact_store(
            store,
            tmp_path,
            words=["back", "in", "twenty", "twenty", "is"],
            findings=[("DATE_TIME", (2.0, 3.5))],
            timings={1: {"asr_crisperwhisper": (1.0, 1.5), "asr_qwen": (1.0, 1.97)}},
        )
        _stub_pii(monkeypatch, findings=[])
        redact(store, "recording", redact_config, run_dir=tmp_path, artifacts_dir=_release(tmp_path))
        plan = _plan(store, applies=False)
        assert [mask.outcome for mask in plan.masks] == [MASK_UNCHANGED]
        assert "in" not in _states(plan), "a word only a timing hull reaches is not the finding's"
        _write_ledger(store, plan, "redacted", None)
        _settle(store, "redacted", None, tmp_path)
        released = _release(tmp_path)
        assert (released / "transcript.txt").read_text() == "back in [DATE_TIME] is\n"
        assert '"words_n": 2' in (released / "consensus.json").read_text()
        assert _silent(released / "audio.wav", 2.1, 3.4)
        assert not _silent(released / "audio.wav", 1.1, 1.4)

    def test_a_finding_bridging_task_words_masks_only_the_residue_either_side(
        self, store: ProvStore, redact_config: TriageConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A read passage's residue "Maria ... Smith" around stimulus words: two masks, the passage audible."""
        _seed_redact_store(
            store,
            tmp_path,
            words=["the", "caterpillar", "Maria", "ate", "leaves", "Smith"],
            findings=[("PERSON", (2.0, 5.5))],
            residue=[2, 5],
        )
        _stub_pii(monkeypatch, findings=[])
        redact(store, "recording", redact_config, run_dir=tmp_path, artifacts_dir=_release(tmp_path))
        plan = _plan(store, applies=False)
        (mask,) = plan.masks
        assert (mask.outcome, mask.task_words_n) == (MASK_UNCHANGED, 2)
        assert _states(plan) == {"Maria": MASKED, "Smith": MASKED}, "no task word is listed under a mask"
        assert len(plan.final) == 2
        _write_ledger(store, plan, "redacted", None)
        _settle(store, "redacted", None, tmp_path)
        released = _release(tmp_path)
        assert (released / "transcript.txt").read_text() == "the caterpillar [PERSON] ate leaves [PERSON]\n"
        assert _silent(released / "audio.wav", 2.1, 2.4) and _silent(released / "audio.wav", 5.1, 5.4)
        assert not _silent(released / "audio.wav", 3.1, 3.4)
        assert not _silent(released / "audio.wav", 4.1, 4.4)

    def test_a_mask_over_task_words_only_disappears(
        self, store: ProvStore, redact_config: TriageConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A finding placed on the passage's own words masks nothing: the scan never read them."""
        _seed_redact_store(
            store,
            tmp_path,
            words=["the", "caterpillar", "ate", "Maria"],
            findings=[("PERSON", _word_extent(1))],
            residue=[3],
        )
        _stub_pii(monkeypatch, findings=[])
        redact(store, "recording", redact_config, run_dir=tmp_path, artifacts_dir=_release(tmp_path))
        plan = _plan(store, applies=False)
        (mask,) = plan.masks
        assert (mask.outcome, mask.words, plan.final, mask.task_words_n) == (MASK_UNMASKED, (), [], 1)

    def test_a_finding_over_the_whole_transcript_masks_only_residue_content(
        self, store: ProvStore, redact_config: TriageConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """SPEECH's fail-safe for an unlocated finding covers every word; only the residue's content stays masked."""
        _seed_redact_store(
            store,
            tmp_path,
            words=["the", "caterpillar", "ate", "Maria", "and", "leaves"],
            findings=[("PERSON", (0.0, 5.5))],
            residue=[3, 4],
        )
        _stub_pii(monkeypatch, findings=[])
        redact(store, "recording", redact_config, run_dir=tmp_path, artifacts_dir=_release(tmp_path))
        plan = _plan(store, applies=False)
        assert _states(plan) == {"Maria": MASKED, "and": UNMASKED_BY_TRIM}
        (extent,) = plan.final
        assert extent.start >= 2.5 and extent.end <= 4.0, "the mask stops at the task words either side"

    def test_a_mask_over_function_words_only_disappears(
        self, store: ProvStore, redact_config: TriageConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No content word left under it, the mask is gone and nothing stands."""
        self._passed(store, redact_config, tmp_path, monkeypatch, findings=[("PERSON", _word_extent(3))])
        plan = _plan(store, applies=False)
        assert [mask.outcome for mask in plan.masks] == [MASK_UNMASKED]
        assert plan.final == []

    def test_a_quote_unmasks_every_place_it_occurs(
        self, store: ProvStore, redact_config: TriageConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The judgement is about the words, so each occurrence the masks hide is unmasked."""
        _seed_redact_store(
            store,
            tmp_path,
            words=["cinderella", "went", "and", "cinderella", "danced"],
            findings=[("PERSON", _word_extent(0)), ("PERSON", _word_extent(3))],
        )
        _stub_pii(monkeypatch, findings=[])
        redact(store, "recording", redact_config, run_dir=tmp_path, artifacts_dir=_release(tmp_path))
        _annotate(store, [_release_entry("Cinderella", "PERSON")])
        assert _plan(store).final == []

    def test_a_quote_naming_a_different_phrase_leaves_that_mask(
        self, store: ProvStore, redact_config: TriageConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """r6's hoarse card: the quoted phrase goes, and "hoarse" -- the term the reviewer judged -- goes everywhere."""
        _seed_redact_store(
            store,
            tmp_path,
            words=["i", "am", "very", "Hoarse", "my", "Voice", "is", "very", "Hoarse"],
            findings=[("MISC", (2.0, 3.5)), ("MISC", (5.0, 8.5))],
        )
        _stub_pii(monkeypatch, findings=[])
        redact(store, "recording", redact_config, run_dir=tmp_path, artifacts_dir=_release(tmp_path))
        _annotate(store, [_release_entry("am very hoarse", "OTHER")])
        plan = _plan(store)
        assert [mask.outcome for mask in plan.masks] == [MASK_UNMASKED, MASK_PARTLY_UNMASKED]
        kept = {word.text for mask in plan.masks for word in mask.words if word.state == MASKED}
        assert kept == {"Voice"}
        (second,) = [word for word in plan.masks[1].words if word.text == "Hoarse"]
        assert second.propagated and not second.named

    def test_a_lower_case_word_a_detector_marked_is_released_not_proper(
        self, store: ProvStore, redact_config: TriageConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """v8, the r12 page: "doing brisk" a role rule marked, and a lone "man", leave their masks in any language."""
        _seed_redact_store(
            store,
            tmp_path,
            words=["i", "keep", "doing", "brisk", "walks", "with", "a", "man", "daily"],
            findings=[("MISC", (2.0, 3.5)), ("PERSON", _word_extent(7))],
        )
        _stub_pii(monkeypatch, findings=[])
        redact(store, "recording", redact_config, run_dir=tmp_path, artifacts_dir=_release(tmp_path))
        plan = _plan(store, applies=False)
        states = _states(plan)
        assert (states["doing"], states["brisk"], states["man"]) == (RELEASED_NOT_PROPER,) * 3
        assert plan.final == []
        assert plan.record(release="x", release_ground=None)["counts"]["released_not_proper_n"] == 3

    def test_an_unplaceable_quote_unmasks_nothing(
        self, store: ProvStore, redact_config: TriageConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A quote the residue does not contain, a placeholder, or part of a word, keeps every mask."""
        self._passed(store, redact_config, tmp_path, monkeypatch)
        _annotate(store, [_release_entry("[PERSON]"), _release_entry("brook"), _release_entry("bob")])
        plan = _plan(store)
        assert (plan.changed, len(plan.release_unplaced)) == (False, 3)

    def test_a_quote_naming_only_unmasked_words_is_off_mask(
        self, store: ProvStore, redact_config: TriageConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Releasing a word nothing masked changes nothing and is recorded as such."""
        self._passed(store, redact_config, tmp_path, monkeypatch)
        _annotate(store, [_release_entry("today")])
        plan = _plan(store)
        assert (plan.changed, plan.release_off_mask) == (False, ("today",))

    def test_with_the_key_off_named_words_stay_masked_and_say_so(
        self, store: ProvStore, redact_config: TriageConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``verdict.llm_reset_redactions`` off: nothing is unmasked, and the ledger still shows what was named."""
        self._passed(store, redact_config, tmp_path, monkeypatch)
        _annotate(store, [_release_entry("Brooklyn"), _redact_entry("today", "DATE_TIME")])
        plan = _plan(store, applies=False)
        Brooklyn = next(word for mask in plan.masks for word in mask.words if word.text == "Brooklyn")
        assert (Brooklyn.state, Brooklyn.named) == (MASKED, True)
        assert not plan.reviewer_applied

    def test_an_identifying_original_is_unmasked_where_the_reviewer_says(
        self, store: ProvStore, redact_config: TriageConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Owner, 2026-09-27: named unmasks always apply, over an original read as carrying PII too."""
        self._passed(store, redact_config, tmp_path, monkeypatch)
        _annotate(
            store,
            [{**_release_entry("Alice"), "relabel": "work_title"}, _place_release("Brooklyn", "fictional")],
            original="carries_pii",
        )
        plan = _plan(store)
        assert plan.reviewer_applied
        assert plan.final == []
        assert set(_states(plan).values()) == {UNMASKED_BY_REVIEWER}

    def test_redact_entries_are_placed_on_words_and_a_condition_is_recorded_apart(
        self, store: ProvStore, redact_config: TriageConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Each proposal lists the words it names; one already masked says so; a condition is no proposal."""
        self._passed(store, redact_config, tmp_path, monkeypatch)
        _annotate(
            store,
            [
                _redact_entry("met", "CONDITION"),
                _redact_entry("Alice", "PERSON"),
                _redact_entry("Brooklyn tod", "LOCATION"),
                _redact_entry("nowhere", "LOCATION"),
            ],
            original="carries_pii",
        )
        plan = _plan(store, applies=False)
        placed = {span.text: (span.placed, span.texts, bool(span.masked_ids)) for span in plan.proposals}
        assert placed == {
            "Alice": (PLACED_WORDS, ("Alice",), True),
            "Brooklyn tod": (PLACED_SUBSTRING, ("Brooklyn", "today"), True),
            "nowhere": ("", (), False),
        }
        (condition,) = plan.conditions
        assert (condition.text, condition.texts, condition.condition_kind) == ("met", ("met",), "other")
        assert plan.condition_indices == (0,) and 0 in plan.agreed

    def test_the_ledger_record_counts_every_state(
        self, store: ProvStore, redact_config: TriageConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The record VERDICT writes carries the masks, the final masks, the proposals and the counts."""
        self._passed(store, redact_config, tmp_path, monkeypatch, findings=[("PERSON", (2.0, 4.5))])
        _annotate(
            store,
            [{**_release_entry("Brooklyn", "PERSON"), "relabel": "other_non_person"}],
            original="carries_pii",
        )
        record = _plan(store).record(release="redacted", release_ground=REVIEWER_UNMASKED_SOME)
        assert record["name"] == PII_LEDGER
        assert record["counts"]["masked_n"] == 1
        assert record["counts"]["unmasked_by_reviewer_n"] == 1
        assert record["counts"]["unmasked_by_trim_n"] == 1
        assert record["counts"]["partly_unmasked_n"] == 1
        assert [entry["word_ids"] for entry in record["final_masks"]] and len(record["final_masks"]) == 1
        assert record["categories"]["unmasked_by_reviewer"] == ["PERSON"]


class TestTheFillIsDeclared:
    """A run declares the fill it used, and the verdict records it."""

    def test_a_null_fill_refuses_before_any_store_write(self, store: ProvStore, tmp_path: Path) -> None:
        """An override may null the shipped fill; two artifacts under different fills are not comparable."""
        config = _override(tmp_path, "redaction:\n  padding_ms: 50\n  fill: null\n")
        _seed_redact_store(store, tmp_path, words=["hello", "Alice"], findings=[("PERSON", (1.0, 2.0))])
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
        _seed_redact_store(store, tmp_path, words=["hello", "Alice"], findings=[("PERSON", (1.0, 2.0))])
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
        _seed_redact_store(store, tmp_path, words=["hello", "Alice"], findings=[("PERSON", (1.0, 2.0))])
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
        _seed_redact_store(store, tmp_path, words=["hello", "Alice"], findings=[("PERSON", (1.0, 2.0))])
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
            words=["hello", "Alice"],
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
            words=["hello", "Alice", "in", "boston"],
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
            store, tmp_path, words=["hello", "Alice"], findings=[("PERSON", (1.0, 1.2)), ("LOCATION", (1.25, 1.5))]
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
        _seed_redact_store(store, tmp_path, words=["hello", "Alice"], findings=[("A+B", (1.0, 1.4))])
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
            words=["hello", "Alice", "in", "boston"],
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
        _seed_redact_store(store, tmp_path, words=["hello", "Alice"], findings=[("PERSON", (1.0, 2.0))])
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
        _seed_redact_store(store, tmp_path, words=["hello", "Alice"], findings=[("PERSON", (1.0, 2.0))])
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
        _seed_redact_store(store, tmp_path, words=["hello", "Alice"], findings=[("PERSON", (1.0, 2.0))])
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
        _seed_redact_store(store, tmp_path, words=["hello", "Alice"], findings=[("PERSON", (1.0, 2.0))], scanned=False)
        with pytest.raises(ValueError, match="no PII scan"):
            redact(store, "recording", redact_config, run_dir=tmp_path, artifacts_dir=_release(tmp_path))

    def test_a_store_scan_whose_detector_failed_is_withheld(
        self, store: ProvStore, redact_config: TriageConfig, tmp_path: Path
    ) -> None:
        """An empty ``spans`` with a populated ``failed`` means the scan did not happen."""
        _seed_redact_store(
            store,
            tmp_path,
            words=["hello", "Alice"],
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
        _seed_redact_store(store, tmp_path, words=["hello", "Alice"], findings=[("PERSON", (1.0, 2.0))], scanned_by=[])
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
            words=["hello", "Alice"],
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
            words=["hello", "Alice"],
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
        _seed_redact_store(store, tmp_path, words=["hello", "Alice"], findings=[("PERSON", (1.0, 2.0))])
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
        _seed_redact_store(store, tmp_path, words=["hello", "Alice"], findings=[("PERSON", (1.0, 2.0))])
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
        _seed_redact_store(store, tmp_path, words=["hello", "Alice"], findings=[("PERSON", (1.0, 2.0))])
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
        _seed_redact_store(store, tmp_path, words=["hello", "Alice"], findings=[("PERSON", (1.0, 2.0))])
        _stub_pii(monkeypatch, findings=[])
        result = redact(store, "recording", redact_config, run_dir=tmp_path, artifacts_dir=_release(tmp_path))
        assert result.verdict.outcome is Outcome.PASS
        assert result.artifacts.keys() == {"audio", "transcript", "consensus"}
        assert _verdict_entity(store, "REDACT").attributes["artifacts_withheld"] is False

    def test_artifacts_dir_nested_in_run_dir_is_refused(
        self, store: ProvStore, redact_config: TriageConfig, tmp_path: Path
    ) -> None:
        """The store's directory and the release directory must not be one publish step apart."""
        _seed_redact_store(store, tmp_path, words=["hello", "Alice"], findings=[("PERSON", (1.0, 2.0))])
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
        _seed_redact_store(store, tmp_path, words=["hello", "Alice"], findings=[("PERSON", (1.0, 2.0))])
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
        _seed_redact_store(store, tmp_path, words=["hello", "Alice"], findings=[("PERSON", (1.0, 2.0))])
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
        _seed_redact_store(store, tmp_path, words=["hello", "Alice"], findings=[("PERSON", (1.0, 2.0))])
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
        # The sources place "Alice" at 1.0-1.2 and 5.4-5.8; the fit derives 3.0-3.2, between them.
        ids = [
            store.entity(
                prov_type="word",
                extent=extent,
                attributes=word_attributes(text, extent, index=index, timings=timings),
            )
            for index, (text, extent, timings) in enumerate(
                [
                    ("one", (0.1, 0.4), {"asr_a": (0.1, 0.4), "asr_b": (0.1, 0.4)}),
                    ("Alice", (3.0, 3.2), {"asr_a": (1.0, 1.2), "asr_b": (5.4, 5.8)}),
                ]
            )
        ]
        for word_id in ids:
            store.was_generated_by(word_id, pre)
        words = [store.get_entity(word_id) for word_id in ids]
        planned = [RedactionExtent(start=0.9, end=1.3, category="PERSON")]

        _records, text, unplaced = _render(words, planned)

        assert "Alice" not in text, "the name survived because the mask read the fitted extent"
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
        _seed_redact_store(store, tmp_path, words=["hello", "Alice"], findings=[("PERSON", (1.0, 2.0))])
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


class TestTheTaskSOwnCastAccountsForACandidate:
    """Owner, 2026-09-23: "analyze the task to determine expected names".

    `cinderella-story` declares no `stimulus_text` -- the source is a physical storybook -- so all
    5,833 of its findings read `in_stimulus: null` and none was ever exempted. 80.54% of them sit
    within one edit of the cast the task itself puts in the speaker's mouth. See
    ``specs/20260923-pii-near-match-and-expected-names/near-match-and-expected-names.md``.
    """

    def test_a_declared_cast_name_is_not_redacted(
        self,
        store: ProvStore,
        redact_config: TriageConfig,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The name the instruction asked for is not a disclosure, with no prompt text in sight."""
        _seed_redact_store(store, tmp_path, words=["and", "then", "cinderella"], findings=[("PERSON", (2.0, 2.5))])
        _stub_pii(monkeypatch, findings=[("PERSON", "cinderella")])
        result = redact(
            store,
            "recording",
            redact_config,
            AudioHints(),
            run_dir=tmp_path,
            artifacts_dir=_release(tmp_path),
            task_family="cinderella-story",
        )
        assert result.verdict.outcome is Outcome.PASS
        assert _verdict_entity(store, "REDACT").attributes["redactions_n"] == 0

    def test_the_exemption_names_the_declaration_that_accounted_for_it(
        self,
        store: ProvStore,
        redact_config: TriageConfig,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A reader must be able to audit what was not redacted, and on whose authority."""
        _seed_redact_store(store, tmp_path, words=["and", "then", "cinderella"], findings=[("PERSON", (2.0, 2.5))])
        _stub_pii(monkeypatch, findings=[("PERSON", "cinderella")])
        redact(
            store,
            "recording",
            redact_config,
            AudioHints(),
            run_dir=tmp_path,
            artifacts_dir=_release(tmp_path),
            task_family="cinderella-story",
        )
        [assertion] = _exemptions(store)
        assert assertion.attributes["expected_keys"] == ["cinderella"]

    def test_a_respelled_cast_name_is_not_redacted_either(
        self,
        store: ProvStore,
        redact_config: TriageConfig,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The cast is matched under the same fitted bound as a declared stimulus."""
        _seed_redact_store(store, tmp_path, words=["and", "then", "cindarela"], findings=[("PERSON", (2.0, 2.5))])
        _stub_pii(monkeypatch, findings=[("PERSON", "cindarela")])
        result = redact(
            store,
            "recording",
            redact_config,
            AudioHints(),
            run_dir=tmp_path,
            artifacts_dir=_release(tmp_path),
            task_family="cinderella-story",
        )
        assert result.verdict.outcome is Outcome.PASS
        assert _verdict_entity(store, "REDACT").attributes["redactions_n"] == 0

    def test_a_name_outside_the_cast_is_still_redacted(
        self,
        store: ProvStore,
        redact_config: TriageConfig,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A retelling that names the speaker's own sister is exactly what must survive."""
        _seed_redact_store(store, tmp_path, words=["and", "then", "springfield"], findings=[("PERSON", (2.0, 2.5))])
        _stub_pii(monkeypatch, findings=[("PERSON", "springfield")])
        redact(
            store,
            "recording",
            redact_config,
            AudioHints(),
            run_dir=tmp_path,
            artifacts_dir=_release(tmp_path),
            task_family="cinderella-story",
        )
        assert _verdict_entity(store, "REDACT").attributes["redactions_n"] == 1
        assert _exemptions(store) == []

    def test_a_family_that_declares_no_cast_exempts_nothing(
        self,
        store: ProvStore,
        redact_config: TriageConfig,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """`picture-description` has an image for a stimulus; no list can cover it and none is used."""
        _seed_redact_store(store, tmp_path, words=["and", "then", "cinderella"], findings=[("PERSON", (2.0, 2.5))])
        _stub_pii(monkeypatch, findings=[("PERSON", "cinderella")])
        redact(
            store,
            "recording",
            redact_config,
            AudioHints(),
            run_dir=tmp_path,
            artifacts_dir=_release(tmp_path),
            task_family="picture-description",
        )
        assert _verdict_entity(store, "REDACT").attributes["redactions_n"] == 1
        assert _verdict_entity(store, "REDACT").attributes["expected_speech_declared"] is False


class TestANearSpellingOfAStimulusWordIsStillThatWord:
    """Owner, 2026-09-23: "expected words/near spelling mismatches don't trigger redaction".

    The bound is fitted in
    ``specs/20260923-pii-near-match-and-expected-names/near-match-and-expected-names.md``.
    """

    def test_a_stimulus_word_the_recogniser_respelled_is_still_exempt(
        self,
        store: ProvStore,
        redact_config: TriageConfig,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """One dropped character does not make the task's own word a disclosure."""
        _seed_redact_store(store, tmp_path, words=["form", "a", "rainbo"], findings=[("LOCATION", (2.0, 2.5))])
        _stub_pii(monkeypatch, findings=[("LOCATION", "rainbo")])
        result = redact(
            store, "recording", redact_config, _hint(RAINBOW), run_dir=tmp_path, artifacts_dir=_release(tmp_path)
        )
        assert result.verdict.outcome is Outcome.PASS
        assert _verdict_entity(store, "REDACT").attributes["redactions_n"] == 0

    def test_the_exemption_records_the_prompt_s_own_spelling_not_the_transcript_s(
        self,
        store: ProvStore,
        redact_config: TriageConfig,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """What accounted for it is the stimulus; the transcript is what needed accounting for."""
        _seed_redact_store(store, tmp_path, words=["form", "a", "rainbo"], findings=[("LOCATION", (2.0, 2.5))])
        _stub_pii(monkeypatch, findings=[("LOCATION", "rainbo")])
        redact(store, "recording", redact_config, _hint(RAINBOW), run_dir=tmp_path, artifacts_dir=_release(tmp_path))
        [assertion] = _exemptions(store)
        assert assertion.attributes["expected_keys"] == ["rainbow"]

    def test_a_short_word_one_edit_from_a_stimulus_word_is_still_redacted(
        self,
        store: ProvStore,
        redact_config: TriageConfig,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Under five characters one edit is a different word, and the fit says so."""
        _seed_redact_store(store, tmp_path, words=["they", "act", "aa"], findings=[("PERSON", (2.0, 2.5))])
        _stub_pii(monkeypatch, findings=[("PERSON", "aa")])
        redact(store, "recording", redact_config, _hint(RAINBOW), run_dir=tmp_path, artifacts_dir=_release(tmp_path))
        assert _verdict_entity(store, "REDACT").attributes["redactions_n"] == 1
        assert _exemptions(store) == []

    def test_a_name_that_is_not_near_any_stimulus_word_is_still_redacted(
        self,
        store: ProvStore,
        redact_config: TriageConfig,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The control: a tolerance that admits a genuinely different name is worse than none."""
        _seed_redact_store(store, tmp_path, words=["form", "a", "springfield"], findings=[("LOCATION", (2.0, 2.5))])
        _stub_pii(monkeypatch, findings=[("LOCATION", "springfield")])
        redact(store, "recording", redact_config, _hint(RAINBOW), run_dir=tmp_path, artifacts_dir=_release(tmp_path))
        assert _verdict_entity(store, "REDACT").attributes["redactions_n"] == 1
        assert _exemptions(store) == []


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

    def test_a_pass_that_masked_nothing_releases_the_original_and_no_copy(
        self,
        store: ProvStore,
        redact_config: TriageConfig,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Every finding exempt: the fold releases the original, and REDACT's identical copy goes.

        Measured on r6: 539 recordings, story-recall and productive-vocabulary above all, were
        released "with redaction" under a copy byte-for-byte the original.
        """
        _seed_redact_store(store, tmp_path, words=["form", "a", "rainbow"], findings=[("LOCATION", (2.0, 2.5))])
        _stub_pii(monkeypatch, findings=[("LOCATION", "rainbow")])
        redact(store, "recording", redact_config, _hint(RAINBOW), run_dir=tmp_path, artifacts_dir=_release(tmp_path))
        assert all((_release(tmp_path) / name).exists() for name in RELEASED_FILES)
        result = verdict_module.verdict(store, None, redact_config, run_dir=tmp_path)
        folded = result.file_verdict
        assert folded.release is not None
        assert folded.release is not None
        assert (folded.release.value, folded.release_ground) == ("as_is", FINDINGS_ARE_TASK_CONTENT)
        ledger = store.get_entity(result.ledger_entity_id)
        assert ledger.attributes["release_ground"] == FINDINGS_ARE_TASK_CONTENT
        assert _settle(store, folded.release.value, folded.release_ground, tmp_path) == {}
        assert not any((_release(tmp_path) / name).exists() for name in RELEASED_FILES)

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
        _seed_redact_store(store, tmp_path, words=["form", "a", "Alice"], findings=[("PERSON", (2.0, 2.5))])
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
            words=["rainbow", "with", "Alice"],
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

    def test_a_finding_placed_on_a_word_the_stimulus_lacks_is_not_exempt(
        self,
        store: ProvStore,
        redact_config: TriageConfig,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The exemption reads the words SPEECH placed the finding on, all of them.

        "my" is not in the rainbow passage, so a finding placed on "my rainbow" is not accounted for
        by the stimulus, however much of it is.
        """
        _seed_redact_store(store, tmp_path, words=["form", "my", "rainbow"], findings=[("LOCATION", (1.0, 2.5))])
        _stub_pii(monkeypatch, findings=[])
        redact(store, "recording", redact_config, _hint(RAINBOW), run_dir=tmp_path, artifacts_dir=_release(tmp_path))
        assert _exemptions(store) == []

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
            words=["rainbow", "with", "Alice"],
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

        ``rainbow`` and ``Alice`` both carry LOCATION. ``Alice`` is not accounted for, so LOCATION is
        outstanding and the re-plan runs — and it must widen onto ``Alice`` alone. Widening onto both
        would undo the exemption on the very pass that exists to catch what the first plan missed,
        which is the failure a skip that is only exercised here can hide.
        """
        _seed_redact_store(
            store,
            tmp_path,
            words=["rainbow", "with", "Alice"],
            findings=[("LOCATION", (0.0, 0.5))],
            extra_marks=[("Alice", "LOCATION")],
        )
        scanned = _stub_pii_sequence(monkeypatch, [[("LOCATION", "Alice")], []])
        result = redact(
            store, "recording", redact_config, _hint(RAINBOW), run_dir=tmp_path, artifacts_dir=_release(tmp_path)
        )
        assert scanned[0] == "rainbow with Alice"
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
            words=["rainbow", "with", "Alice"],
            findings=[("LOCATION", (0.0, 0.5))],
            extra_marks=[("Alice", "PERSON")],
        )
        _stub_pii(monkeypatch, findings=[("PERSON", "Alice")])
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
        _seed_redact_store(store, tmp_path, words=["hello", "Alice"], findings=[("PERSON", (1.0, 2.0))])
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
        _seed_redact_store(store, tmp_path, words=["hello", "Alice"], findings=[("PERSON", (1.0, 2.0))])
        _stub_pii(monkeypatch, findings=[("PERSON", "Alice")])
        result = redact(store, "recording", redact_config, run_dir=tmp_path, artifacts_dir=_release(tmp_path))
        assert result.artifacts == {}


class TestThePackagedConfigLeavesTheReviewerOff:
    """A step that turns itself on would make two hosts disagree with no record of why."""

    def test_the_packaged_config_leaves_it_disabled(self) -> None:
        """The default is off, and the other keys are present so an override may only flip one."""
        cfg = load_triage_config()
        assert cfg.require("redaction.llm_check.enabled") is False
        assert cfg.require("redaction.llm_check.model_id") == "google/gemma-4-31B-it-qat-w4a16-ct"
        assert cfg.require("redaction.llm_check.ref") == "main"
        assert cfg.require("redaction.llm_check.max_iterations") == 3


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
        _seed_redact_store(store, tmp_path, words=["hello", "Alice"], findings=[("PERSON", (1.0, 2.0))])
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
        _seed_redact_store(store, tmp_path, words=["hello", "Alice"], findings=[("PERSON", (1.0, 2.0))])
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
        _seed_redact_store(store, tmp_path, words=["hello", "Alice"], findings=[("PERSON", (1.0, 2.0))])
        _stub_pii(monkeypatch, findings=[("PERSON", "Alice")])
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
        _seed_redact_store(store, tmp_path, words=["hello", "Alice"], findings=[("PERSON", (1.0, 2.0))])
        _stub_pii(monkeypatch, findings=[])
        result = redact(store, "recording", redact_config, run_dir=tmp_path, artifacts_dir=_release(tmp_path))
        assert result.artifacts["audio"].exists()
        assert result.artifacts["audio"].parent == _release(tmp_path)


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
            words=["hello", "Alice", "in", "boston"],
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


class TestTheReScanDoesNotReadBracketedTokens:
    """SPEECH stopped scanning them; the verification re-scan must stop too, or nothing releases.

    See ``specs/20260922-brackets-are-not-speech/design.md``.
    """

    def test_the_rescan_is_handed_no_bracketed_token(
        self,
        store: ProvStore,
        redact_config: TriageConfig,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A filler token survives into the released transcript and must not be re-scanned."""
        _seed_redact_store(
            store, tmp_path, words=["[UH]", "my", "name", "is", "Alice"], findings=[("PERSON", (4.0, 4.5))]
        )
        scanned = _stub_pii(monkeypatch, findings=[])
        redact(store, "recording", redact_config, run_dir=tmp_path, artifacts_dir=_release(tmp_path))
        assert scanned == ["my name is [PERSON]"]

    def test_the_released_transcript_still_carries_it(
        self,
        store: ProvStore,
        redact_config: TriageConfig,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Dropping it from the scan is not dropping it from the transcript."""
        _seed_redact_store(
            store, tmp_path, words=["[UH]", "my", "name", "is", "Alice"], findings=[("PERSON", (4.0, 4.5))]
        )
        _stub_pii(monkeypatch, findings=[])
        release = _release(tmp_path)
        result = redact(store, "recording", redact_config, run_dir=tmp_path, artifacts_dir=release)
        assert result.verdict.outcome is Outcome.PASS
        assert (release / "transcript.txt").read_text().strip() == "[UH] my name is [PERSON]"

    def test_a_survivor_outside_the_brackets_still_fails_the_release(
        self,
        store: ProvStore,
        redact_config: TriageConfig,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The guard against over-narrowing: a real survivor is still a fail."""
        _seed_redact_store(
            store, tmp_path, words=["[UH]", "my", "name", "is", "Alice"], findings=[("PERSON", (4.0, 4.5))]
        )
        _stub_pii(monkeypatch, findings=[("PERSON", "Alice")])
        result = redact(store, "recording", redact_config, run_dir=tmp_path, artifacts_dir=_release(tmp_path))
        assert result.verdict.outcome is not Outcome.PASS


class TestAMaskIsAFindingsOwnWords:
    """Owner, 2026-09-27: a mask is one finding's own words, never the whole transcript or a chain of them."""

    def _seed(
        self,
        store: ProvStore,
        config: TriageConfig,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        words: Sequence[str],
        located: Sequence[tuple[str, tuple[float, float]]],
        unplaced: Sequence[str] = (),
    ) -> None:
        """A REDACT pass over ``words``; each ``unplaced`` category a finding SPEECH could not place."""
        _seed_redact_store(
            store,
            tmp_path,
            words=list(words),
            findings=list(located),
            unplaced=[(category, "Alan") for category in unplaced],
        )
        _stub_pii(monkeypatch, findings=[])
        redact(store, "recording", config, run_dir=tmp_path, artifacts_dir=_release(tmp_path))

    def test_an_unplaced_finding_masks_nothing(
        self, store: ProvStore, redact_config: TriageConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """r6's story-recall card: an unlocated PERSON no longer masks the examiner's instructions."""
        words = ["okay", "you", "were", "given", "the", "test", "grandpa", "ninety"]
        self._seed(store, redact_config, tmp_path, monkeypatch, words, [("DATE_TIME", _word_extent(7))], ["PERSON"])
        plan = _plan(store, applies=False)
        assert _states(plan) == {"ninety": MASKED}
        assert [(f.category, f.state) for f in plan.unplaced] == [("PERSON", UNPLACED_UNREAD)]
        assert [mask.planned.category for mask in plan.masks] == ["DATE_TIME"]

    def test_a_reviewer_entry_of_its_family_places_it(
        self, store: ProvStore, redact_config: TriageConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """r6's open-response card: "Alan" places the unlocated PERSON; "Wisconsin" agrees with its mask."""
        words = ["my", "brother", "alan", "lives", "in", "wisconsin"]
        self._seed(store, redact_config, tmp_path, monkeypatch, words, [("LOCATION", _word_extent(5))], ["PERSON"])
        _annotate(
            store,
            [_redact_entry("Alan", "PERSON"), _redact_entry("Wisconsin", "LOCATION")],
            original="carries_pii",
        )
        plan = _plan(store)
        assert [span.agreement for span in plan.proposals] == [AGREED_PLACED, AGREED_MASKED]
        assert plan.agreed == frozenset({0, 1})
        assert [(f.category, f.state) for f in plan.unplaced] == [("PERSON", UNPLACED_PLACED)]
        assert _states(plan) == {"alan": MASKED, "wisconsin": MASKED}
        assert sorted(mask.source for mask in plan.masks) == [DETECTOR, REVIEWER]

    def test_a_clean_reading_clears_an_unplaced_finding_and_a_silent_one_leaves_it_open(
        self, store: ProvStore, redact_config: TriageConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Read clean: cleared. Read as identifying but naming nothing of its family: open for review."""
        words = ["grandpa", "is", "ninety"]
        self._seed(store, redact_config, tmp_path, monkeypatch, words, [("DATE_TIME", _word_extent(2))], ["PERSON"])
        _annotate(store, [], original="clean")
        assert [f.state for f in _plan(store).unplaced] == [UNPLACED_CLEARED]
        _annotate(store, [_release_entry("ninety", "DATE_TIME")], original="carries_pii")
        assert [f.state for f in _plan(store).unplaced] == [UNPLACED_OPEN]

    def test_a_new_word_is_a_new_proposal(
        self, store: ProvStore, redact_config: TriageConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A redact entry naming a content word no mask hides is the reviewer proposing to hide more."""
        words = ["i", "met", "Alice", "in", "Brooklyn"]
        self._seed(store, redact_config, tmp_path, monkeypatch, words, [("PERSON", _word_extent(2))])
        _annotate(store, [_redact_entry("Alice", "PERSON"), _redact_entry("Brooklyn", "LOCATION")])
        plan = _plan(store)
        assert [span.agreement for span in plan.proposals] == [AGREED_MASKED, NEW]
        assert plan.agreed == frozenset({0})

    def test_a_proposals_function_words_are_not_proposed(
        self, store: ProvStore, redact_config: TriageConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A redact quote's function words are neither its ``content_ids`` nor counted as proposed."""
        words = ["i", "had", "a", "cyst", "in", "my", "neck"]
        self._seed(store, redact_config, tmp_path, monkeypatch, words, [])
        _annotate(store, [_redact_entry("a cyst in my neck", "OTHER")])
        plan = _plan(store)
        (span,) = plan.proposals
        texts = dict(zip(span.word_ids, span.texts))
        assert sorted(texts[i] for i in span.content_ids) == ["cyst", "neck"]
        assert plan.count(PROPOSED_BY_REVIEWER) == 2

    def test_identical_findings_are_one_mask_labelled_by_family(
        self, store: ProvStore, redact_config: TriageConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Two detectors' LOCATION and LOC on "wisconsin" are one mask, labelled LOCATION, carrying both."""
        words = ["lives", "in", "wisconsin"]
        self._seed(
            store,
            redact_config,
            tmp_path,
            monkeypatch,
            words,
            [("LOCATION", _word_extent(2)), ("LOC", _word_extent(2))],
        )
        plan = _plan(store, applies=False)
        (mask,) = plan.masks
        assert (mask.planned.category, mask.categories) == ("LOCATION", ("LOC", "LOCATION"))
        assert "+" in plan.planned[0].category, "REDACT's own extent carries the joined label"
        assert [extent.category for extent in plan.final] == ["LOCATION"]
        assert not plan.changed, "a relabelled extent silences the same audio"


class TestTheReviewerJudgesATermNotAnInstance:
    """Owner, 2026-09-27: a released finding term is released at every occurrence, unless redacted there."""

    def _seed(self, store: ProvStore, config: TriageConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Three PERSON findings on "prince" in "the prince danced the prince smiled the prince"."""
        words = ["the", "Prince", "danced", "the", "Prince", "smiled", "the", "Prince"]
        _seed_redact_store(store, tmp_path, words=words, findings=[("PERSON", _word_extent(i)) for i in (1, 4, 7)])
        _stub_pii(monkeypatch, findings=[])
        redact(store, "recording", config, run_dir=tmp_path, artifacts_dir=_release(tmp_path))

    def test_one_quoted_occurrence_releases_every_one(
        self, store: ProvStore, redact_config: TriageConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """r6's Cinderella card: "the Prince danced", relabelled no person, releases the other two as well."""
        self._seed(store, redact_config, tmp_path, monkeypatch)
        _annotate(store, [{**_release_entry("the Prince danced", "PERSON"), "relabel": "other_non_person"}])
        plan = _plan(store)
        princes = [word for mask in plan.masks for word in mask.words]
        assert [word.state for word in princes] == [UNMASKED_BY_REVIEWER] * 3
        assert [(word.named, word.propagated) for word in princes] == [(True, False), (False, True), (False, True)]

    def test_a_redact_entry_on_the_term_stops_it_spreading(
        self, store: ProvStore, redact_config: TriageConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A reading without its inputs recorded: the name kept masked wins, so every occurrence is masked."""
        self._seed(store, redact_config, tmp_path, monkeypatch)
        _annotate(
            store,
            [
                {**_release_entry("the Prince danced", "PERSON"), "relabel": "other_non_person"},
                _redact_entry("Prince smiled", "PERSON"),
            ],
        )
        plan = _plan(store)
        princes = [word for mask in plan.masks for word in mask.words]
        assert [word.state for word in princes] == [MASKED, MASKED, MASKED]
        assert [word.propagation for word in princes] == ["mask", "", ""]
        assert not plan.reviewer_precedence

    def test_a_release_given_every_input_keeps_the_quoted_prince_released(
        self, store: ProvStore, redact_config: TriageConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Owner, 2026-10-09: with the task, the consensus and both readings, the release outranks the kept name."""
        self._seed(store, redact_config, tmp_path, monkeypatch)
        _annotate(
            store,
            [
                {**_release_entry("the Prince danced", "PERSON"), "relabel": "other_non_person"},
                _redact_entry("Prince smiled", "PERSON"),
            ],
            inputs=FULL_REVIEW_INPUTS,
        )
        plan = _plan(store)
        princes = [word for mask in plan.masks for word in mask.words]
        assert [word.state for word in princes] == [UNMASKED_BY_REVIEWER, MASKED, MASKED]
        assert [word.propagation for word in princes] == ["", "", ""]
        assert plan.reviewer_precedence
        assert plan.record(release="x", release_ground=None)["reviewer_precedence"] is True

    def test_a_reading_missing_one_input_does_not_outrank_the_kept_name(
        self, store: ProvStore, redact_config: TriageConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The same reading without the recognisers' readings: the mask wins."""
        self._seed(store, redact_config, tmp_path, monkeypatch)
        _annotate(
            store,
            [
                {**_release_entry("the Prince danced", "PERSON"), "relabel": "other_non_person"},
                _redact_entry("Prince smiled", "PERSON"),
            ],
            inputs={**FULL_REVIEW_INPUTS, "asr_readings": False},
        )
        plan = _plan(store)
        assert [word.state for mask in plan.masks for word in mask.words] == [MASKED, MASKED, MASKED]
        assert not plan.reviewer_precedence


class TestAReleaseWithEveryInputOutranksADetectorSName:
    """Owner, 2026-10-09: a release of one occurrence releases every occurrence a detector kept as a name."""

    def _seed(self, store: ProvStore, config: TriageConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """PERSON findings on a lower-case "prince" and, later, a capitalised "Prince" the person lock keeps."""
        words = ["the", "prince", "danced", "then", "Prince", "smiled"]
        _seed_redact_store(store, tmp_path, words=words, findings=[("PERSON", _word_extent(i)) for i in (1, 4)])
        _stub_pii(monkeypatch, findings=[])
        redact(store, "recording", config, run_dir=tmp_path, artifacts_dir=_release(tmp_path))

    def test_every_occurrence_is_released(
        self, store: ProvStore, redact_config: TriageConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The reviewer releases "the prince danced": the locked "Prince" goes too, and nothing re-masks it."""
        self._seed(store, redact_config, tmp_path, monkeypatch)
        _annotate(store, [_release_entry("the prince danced", "PERSON")], inputs=FULL_REVIEW_INPUTS)
        plan = _plan(store)
        assert MASKED not in {word.state for mask in plan.masks for word in mask.words}
        assert plan.final == []

    def test_without_every_input_the_kept_name_masks_every_occurrence(
        self, store: ProvStore, redact_config: TriageConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The same release from a reading that recorded no inputs: both stay masked."""
        self._seed(store, redact_config, tmp_path, monkeypatch)
        _annotate(store, [_release_entry("the prince danced", "PERSON")])
        plan = _plan(store)
        assert [word.state for mask in plan.masks for word in mask.words] == [MASKED, MASKED]


class TestAFunctionWordIsNotATerm:
    """Term propagation carries content words; a released "the" is the trim's, not a term's."""

    def test_the_second_prince_is_released_and_its_article_is_not_propagated(
        self, store: ProvStore, redact_config: TriageConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Two PERSON findings over "the prince"; the reviewer releases the first."""
        words = ["the", "Prince", "sang", "the", "Prince"]
        _seed_redact_store(
            store,
            tmp_path,
            words=words,
            findings=[
                ("PERSON", (_word_extent(0)[0], _word_extent(1)[1])),
                ("PERSON", (_word_extent(3)[0], _word_extent(4)[1])),
            ],
        )
        _stub_pii(monkeypatch, findings=[])
        redact(store, "recording", redact_config, run_dir=tmp_path, artifacts_dir=_release(tmp_path))
        _annotate(store, [{**_release_entry("the Prince sang", "PERSON"), "relabel": "other_non_person"}])
        second = _plan(store).masks[1]
        flags = {word.text: word.propagated for word in second.words}
        assert flags == {"the": False, "Prince": True}
        assert [word.state for word in second.words] == [UNMASKED_BY_TRIM, UNMASKED_BY_REVIEWER]


def test_an_extent_reaching_no_word_is_labelled_by_its_family(
    store: ProvStore, redact_config: TriageConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An audio-only REDACT extent keeps its bounds and reads one category, never REDACT's joined one."""
    from senselab.audio.workflows.triage.nodes.redact import REDACTION_SPAN

    _seed_redact_store(store, tmp_path, words=["hello", "there"], findings=[])
    agent = store.agent(agent_type="software", version="senselab test-redact")
    activity = store.activity(node="REDACT", step="plan", parameters={})
    store.was_associated_with(activity, agent)
    span = store.entity(
        prov_type="span", extent=(3.2, 3.4), attributes={"name": REDACTION_SPAN, "category": "PERSON+NAME"}
    )
    store.was_generated_by(span, activity)
    plan = _plan(store, applies=False)
    (mask,) = plan.masks
    assert (mask.planned.category, mask.categories, mask.words) == ("PERSON", ("PERSON", "NAME"), ())
    assert [(extent.start, extent.end, extent.category) for extent in plan.final] == [(3.2, 3.4, "PERSON")]
    assert not plan.changed


class TestTaskWordsKindCutAndAReadingThatNamesNothing:
    """Owner, 2026-09-28: the task's own words are never masked; a name is its proper nouns."""

    def test_a_task_lexicon_word_is_never_masked(
        self, store: ProvStore, redact_config: TriageConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """r7's Cinderella card: a PERSON on "Cinderella" and a DATE_TIME on "midnight" leave no mask."""
        from senselab.audio.workflows.triage.task_lexicon import TaskLexicon

        words = ["then", "cinderella's", "coach", "left", "at", "midnight"]
        _seed_redact_store(
            store, tmp_path, words=words, findings=[("PERSON", _word_extent(1)), ("DATE_TIME", _word_extent(5))]
        )
        _stub_pii(monkeypatch, findings=[])
        redact(store, "recording", redact_config, run_dir=tmp_path, artifacts_dir=_release(tmp_path))
        lexicon = TaskLexicon("cinderella-story", (("cinderella",), ("midnight",)))
        plan = mask_plan(store, reviewer_applies=False, padding_ms=50, lexicon=lexicon)
        assert plan.final == []
        assert len(plan.task_lexicon_ids) == 2
        assert not plan.task_content_only, "a lexical task's declared words are not task events"
        assert plan.record(release="x", release_ground=None)["counts"]["task_lexicon_words_n"] == 2

    def test_a_word_not_in_proper_form_under_a_place_is_cut(
        self, store: ProvStore, redact_config: TriageConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """r7's free-speech-v2 card: LOC "Florida Like"; "like" is cut, and policy v7 keeps the state masked."""
        words = ["i", "went", "in", "Florida", "like", "every", "summer", "i", "like", "it"]
        _seed_redact_store(
            store,
            tmp_path,
            words=words,
            findings=[
                ("LOCATION", _word_extent(3)),
                ("LOC", (_word_extent(3)[0], _word_extent(4)[1])),
            ],
        )
        _stub_pii(monkeypatch, findings=[])
        redact(store, "recording", redact_config, run_dir=tmp_path, artifacts_dir=_release(tmp_path))
        _annotate(store, [_release_entry("in Florida", "LOCATION")])
        plan = _plan(store)
        words_of = {word.text: word for mask in plan.masks for word in mask.words}
        assert (words_of["Florida"].state, words_of["Florida"].locked) == (MASKED, "place")
        assert (words_of["like"].state, words_of["like"].kind_cut) == (UNMASKED_BY_TRIM, True)
        assert "summer" not in words_of, "a season is released, so the policy places no mask on it"
        assert [(mask.source, mask.planned.category) for mask in plan.masks if mask.final] == [
            (DETECTOR, "LOCATION"),
        ]

    def test_a_name_part_the_transcript_uses_nowhere_else_is_not_cut(
        self, store: ProvStore, redact_config: TriageConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """r7 replay: "ben Fletcher" -- a lower-case first name is still a name, so the cut leaves it."""
        words = ["my", "friend", "ben", "Fletcher", "called"]
        _seed_redact_store(
            store, tmp_path, words=words, findings=[("PERSON", (_word_extent(2)[0], _word_extent(3)[1]))]
        )
        _stub_pii(monkeypatch, findings=[])
        redact(store, "recording", redact_config, run_dir=tmp_path, artifacts_dir=_release(tmp_path))
        plan = _plan(store, applies=False)
        assert _states(plan) == {"ben": MASKED, "Fletcher": MASKED}

    def test_a_finding_with_no_capital_letter_is_released_not_proper(
        self, store: ProvStore, redact_config: TriageConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """v8: no capital under the mask, so it is no proper noun; the name cut is not what releases it."""
        words = ["my", "friend", "joan", "didion", "called"]
        _seed_redact_store(
            store, tmp_path, words=words, findings=[("PERSON", (_word_extent(2)[0], _word_extent(3)[1]))]
        )
        _stub_pii(monkeypatch, findings=[])
        redact(store, "recording", redact_config, run_dir=tmp_path, artifacts_dir=_release(tmp_path))
        plan = _plan(store, applies=False)
        assert _states(plan) == {"joan": RELEASED_NOT_PROPER, "didion": RELEASED_NOT_PROPER}
        assert not any(word.kind_cut for mask in plan.masks for word in mask.words)

    def test_a_flagged_reading_with_no_entries_is_recorded_and_moves_no_mask(
        self, store: ProvStore, redact_config: TriageConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A judgment without words: REDACT's masks stand and the ledger says why."""
        _seed_redact_store(store, tmp_path, words=["hello", "Alice"], findings=[("PERSON", _word_extent(1))])
        _stub_pii(monkeypatch, findings=[])
        redact(store, "recording", redact_config, run_dir=tmp_path, artifacts_dir=_release(tmp_path))
        _annotate(store, [], original="clean", redaction="incomplete")
        plan = _plan(store)
        assert plan.named_no_words is True
        assert _states(plan) == {"Alice": MASKED}
        assert plan.record(release="x", release_ground=None)["reviewer_named_no_words"] is True

    def test_the_ledger_records_safe_harbor_and_each_release_reason(
        self, store: ProvStore, redact_config: TriageConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Every mask names its Safe Harbor identifier; every release entry keeps its reason."""
        _seed_redact_store(
            store, tmp_path, words=["i", "grew", "up", "in", "Florida"], findings=[("LOCATION", _word_extent(4))]
        )
        _stub_pii(monkeypatch, findings=[])
        redact(store, "recording", redact_config, run_dir=tmp_path, artifacts_dir=_release(tmp_path))
        entry = {**_release_entry("Florida", "LOCATION"), "why": "a state, and nothing else here narrows it"}
        _annotate(store, [entry])
        record = _plan(store).record(release="x", release_ground=None)
        assert [mask["safe_harbor"] for mask in record["masks"]] == [["B"]]
        assert record["releases"] == [
            {
                "text": "Florida",
                "category": "LOCATION",
                "safe_harbor": "B",
                "why": "a state, and nothing else here narrows it",
                "relabel": "",
                "place_reason": "",
                "placed": True,
            }
        ]


@pytest.mark.parametrize(
    ("texts", "released"),
    [
        (["this", "morning"], {0, 1}),
        (["the", "past", "couple", "of", "weeks"], {0, 1, 2, 3, 4}),
        (["two", "years"], {0, 1}),
        (["a", "week"], {0, 1}),
        (["years"], {0}),
        (["3", "o'clock"], {0, 1}),
        (["3pm"], {0}),
        (["last", "year"], {0, 1}),
        (["the", "week"], set()),
        (["March", "3rd"], set()),
        (["last", "Tuesday"], {0, 1}),
        (["Tuesday", "morning"], {0, 1}),
        (["93", "years", "old"], set()),
        (["yesterday", "evening"], {0, 1}),
        (["two", "years", "in", "Boston"], {0, 1, 2}),
        (["2-3", "weeks", "ago"], {0, 1, 2}),
        (["this", "summer"], {0, 1}),
        (["en", "invierno"], {0, 1}),
        (["in", "2021"], set()),
        (["hace", "dos", "semanas"], {0, 1, 2}),
    ],
)
def test_time_by_kind_releases_times_of_day_and_durations_only(texts: list[str], released: set[int]) -> None:
    """A time of day, a weekday, a season, a relative or a length of time is released; a date or an age is not."""
    assert time_by_kind(texts) == released


class TestRedactionPolicyV7:
    """Owner, 2026-10-03: the fold masks and releases by the redaction policy's rules, whatever the reviewer says."""

    def _plan(
        self,
        store: ProvStore,
        tmp_path: Path,
        words: Sequence[str],
        located: Sequence[tuple[str, Sequence[int]]] = (),
        *,
        proposal: Sequence[dict[str, str]] = (),
        approvals: Sequence[str] = (),
        language: str | None = None,
        protected: Sequence[str] = ("PERSON", "NAME", "LOCATION", "LOC"),
    ) -> MaskPlan:
        """The plan over ``words``, a finding per ``(category, word indices)``, and a reading proposing ``proposal``."""
        findings = [
            (category, (_word_extent(indices[0])[0], _word_extent(indices[-1])[1])) for category, indices in located
        ]
        _seed_redact_store(store, tmp_path, words=list(words), findings=findings)
        if proposal:
            _annotate(store, proposal)
        return mask_plan(
            store,
            reviewer_applies=True,
            padding_ms=50,
            protected_categories=protected,
            name_approvals=approvals,
            language=language,
        )

    def test_a_year_is_masked_whatever_the_reviewer_says(self, store: ProvStore, tmp_path: Path) -> None:
        """Case: "I had COVID in 2021": the year goes, and a release does not free it."""
        plan = self._plan(
            store,
            tmp_path,
            ["i", "had", "covid", "in", "2021", "and"],
            [("DATE_TIME", [4])],
            proposal=[_release_entry("in 2021", "DATE_TIME")],
        )
        (word,) = [word for mask in plan.masks for word in mask.words if word.text == "2021"]
        assert (word.state, word.locked, word.named) == (MASKED, "date", True)

    def test_holidays_and_months_no_detector_found_are_masked_by_the_policy(
        self, store: ProvStore, tmp_path: Path
    ) -> None:
        """Case: "Halloween" and "October" get a policy mask each; the season beside them gets none (v9)."""
        plan = self._plan(store, tmp_path, ["every", "summer", "and", "on", "Halloween", "in", "October", "we", "go"])
        policy = [mask for mask in plan.masks if mask.source == "policy"]
        assert [[word.text for word in mask.words] for mask in policy] == [["Halloween"], ["October"]]
        assert all(word.state == MASKED and word.locked == "date" for mask in policy for word in mask.words)
        assert plan.policy_masks_n == 2

    def test_a_season_a_detector_tagged_is_released_and_a_month_year_or_holiday_is_not(
        self, store: ProvStore, tmp_path: Path
    ) -> None:
        """Owner, 2026-10-04: "redact absolute years and months etc and leave seasons and days of the week"."""
        words = ["last", "summer", "en", "invierno", "on", "Monday", "in", "October", "2021", "for", "Halloween"]
        plan = self._plan(
            store,
            tmp_path,
            words,
            [("DATE_TIME", [0, 1]), ("DATE_TIME", [3]), ("DATE_TIME", [5]), ("DATE_TIME", [7, 8]), ("DATE_TIME", [10])],
            proposal=[_release_entry("October 2021", "DATE_TIME"), _release_entry("Halloween", "DATE_TIME")],
        )
        states = _states(plan)
        assert states["summer"] == RELEASED_BY_KIND and states["invierno"] == RELEASED_BY_KIND
        assert states["Monday"] == RELEASED_BY_KIND
        assert states["October"] == MASKED and states["2021"] == MASKED and states["Halloween"] == MASKED

    def test_a_weekday_and_relative_times_are_released_by_kind(self, store: ProvStore, tmp_path: Path) -> None:
        """Case: "on Monday", "2-3 weeks ago" and "last year" are shown whatever the reviewer said."""
        words = ["on", "Monday", "and", "2-3", "weeks", "ago", "and", "last", "year"]
        plan = self._plan(store, tmp_path, words, [("DATE_TIME", [1]), ("DATE_TIME", [3, 4, 5]), ("DATE_TIME", [7, 8])])
        states = _states(plan)
        shown = ("Monday", "2-3", "weeks", "ago", "last", "year")
        assert {text: states[text] for text in shown} == dict.fromkeys(shown, RELEASED_BY_KIND)
        assert plan.final == []

    def test_an_age_is_masked(self, store: ProvStore, tmp_path: Path) -> None:
        """Case: "I'm 73." with no finding, and "thirty-six years old" released by the reviewer: both stay masked."""
        plan = self._plan(
            store,
            tmp_path,
            ["i'm", "73.", "i", "was", "thirty-six", "years", "old", "then"],
            [("DATE_TIME", [4, 5, 6])],
            proposal=[_release_entry("thirty-six years old", "DATE_TIME")],
        )
        states = _states(plan)
        assert (states["73."], states["thirty-six"], states["years"], states["old"]) == (MASKED,) * 4
        locks = {word.text: word.locked for mask in plan.masks for word in mask.words}
        assert (locks["73."], locks["thirty-six"], locks["old"]) == ("age", "age", "age")

    def test_a_person_name_needs_a_humans_approval(self, store: ProvStore, tmp_path: Path) -> None:
        """Case: "Ray Bradbury": the reviewer's release is recorded and does not unmask; an approval does."""
        words = ["i", "read", "Ray", "Bradbury", "books"]
        proposal = [{**_release_entry("Ray Bradbury", "PERSON"), "why": "an author"}]
        plan = self._plan(store, tmp_path, words, [("PERSON", [2, 3])], proposal=proposal)
        assert _states(plan) == {"Ray": MASKED, "Bradbury": MASKED}
        assert plan.name_release_proposed == ("Ray Bradbury",) and plan.person_names_masked == 2
        approved = mask_plan(
            store,
            reviewer_applies=True,
            padding_ms=50,
            protected_categories=("PERSON",),
            name_approvals=("Ray Bradbury",),
        )
        assert _states(approved) == {"Ray": UNMASKED_BY_APPROVAL, "Bradbury": UNMASKED_BY_APPROVAL}
        assert approved.person_names_masked == 0

    def test_a_kinship_word_is_never_masked(self, store: ProvStore, tmp_path: Path) -> None:
        """Case: "my brother John": the relationship is shown, the name is not; "Mom" too."""
        plan = self._plan(
            store, tmp_path, ["my", "brother", "John", "and", "Mom", "came"], [("PERSON", [1, 2]), ("PERSON", [4])]
        )
        states = _states(plan)
        assert (states["brother"], states["John"], states["Mom"]) == (RELEASED_BY_KIND, MASKED, RELEASED_BY_KIND)
        assert plan.record(release="x", release_ground=None)["counts"]["released_kinship_n"] == 2

    def test_a_state_is_masked_and_a_country_is_released(self, store: ProvStore, tmp_path: Path) -> None:
        """Case: "Florida" stays masked though the reviewer releases it; "Mexico" is released by kind."""
        plan = self._plan(
            store,
            tmp_path,
            ["we", "moved", "from", "Mexico", "to", "Florida", "then"],
            [("LOCATION", [3]), ("LOCATION", [5])],
            proposal=[{**_release_entry("Florida", "LOCATION"), "why": "a state"}],
        )
        words_of = {word.text: word for mask in plan.masks for word in mask.words}
        assert (words_of["Florida"].state, words_of["Florida"].locked) == (MASKED, "place")
        assert (words_of["Mexico"].state, words_of["Mexico"].kind) == (RELEASED_BY_KIND, "country")

    def test_a_country_the_reviewer_asks_to_hide_is_masked_without_withholding(
        self, store: ProvStore, tmp_path: Path
    ) -> None:
        """A triangulating country: the reviewer's ``redact`` entry masks it and proposes nothing further."""
        from senselab.audio.workflows.triage.nodes.redact import COUNTRY_MASKED

        plan = self._plan(
            store,
            tmp_path,
            ["the", "only", "doctor", "from", "Mexico", "here"],
            [("LOCATION", [4])],
            proposal=[_redact_entry("Mexico", "LOCATION")],
        )
        assert _states(plan)["Mexico"] == MASKED
        assert [span.agreement for span in plan.proposals] == [COUNTRY_MASKED] and plan.agreed == {0}

    def test_a_country_inside_an_organisations_name_stays_masked(self, store: ProvStore, tmp_path: Path) -> None:
        """sub-09f16959: "United States Marine Corps" tagged LOCATION is not a country; the reviewer cannot free it."""
        plan = self._plan(
            store,
            tmp_path,
            ["he", "is", "in", "the", "United", "States", "Marine", "Corps", "now"],
            [("LOCATION", [4, 5, 6, 7])],
            proposal=[{**_release_entry("United States Marine Corps", "LOCATION"), "why": "a country"}],
        )
        states = _states(plan)
        assert [states[text] for text in ("United", "States", "Marine", "Corps")] == [MASKED] * 4

    def test_a_specific_organisation_is_masked_and_a_generic_one_released(
        self, store: ProvStore, tmp_path: Path
    ) -> None:
        """Case: "USF voice center": the reviewer releases the generic words, the proper name stays."""
        plan = self._plan(
            store,
            tmp_path,
            ["at", "the", "USF", "voice", "center", "today"],
            [("ORGANIZATION", [2, 3, 4])],
            proposal=[{**_release_entry("voice center", "ORGANIZATION"), "why": "a generic description"}],
            protected=("PERSON", "NAME", "LOCATION", "LOC", "ORGANIZATION"),
        )
        states = _states(plan)
        assert (states["USF"], states["voice"], states["center"]) == (
            MASKED,
            UNMASKED_BY_REVIEWER,
            UNMASKED_BY_REVIEWER,
        )

    def test_spanish_words_an_english_model_mistook_for_names_are_released(
        self, store: ProvStore, tmp_path: Path
    ) -> None:
        """The r10 Spanish card: "mi estado de ánimo" tagged PERSON; a month and a name stay masked."""
        words = ["Pensando", "en", "mi", "estado", "de", "ánimo,", "mi", "familia", "y", "María", "en", "octubre."]
        plan = self._plan(
            store,
            tmp_path,
            words,
            [("PERSON", [0]), ("PERSON", [3, 4, 5]), ("PERSON", [7]), ("PERSON", [9])],
            language="es",
        )
        states = _states(plan)
        assert states["estado"] == UNMASKED_BY_TRIM and states["ánimo,"] == UNMASKED_BY_TRIM
        assert states["Pensando"] == UNMASKED_BY_TRIM and states["de"] == UNMASKED_BY_TRIM
        assert states["familia"] == RELEASED_BY_KIND
        assert states["María"] == MASKED
        assert states["octubre."] == MASKED and plan.language == "es"

    def test_the_same_tags_in_english_are_released_not_proper(self, store: ProvStore, tmp_path: Path) -> None:
        """v8 holds in every language: a PERSON tag with no capital letter is no proper noun."""
        plan = self._plan(store, tmp_path, ["my", "friend", "joan", "didion", "called"], [("PERSON", [2, 3])])
        assert _states(plan) == {"joan": RELEASED_NOT_PROPER, "didion": RELEASED_NOT_PROPER}

    def test_a_policy_mask_is_released_where_redact_never_ran(self, store: ProvStore, tmp_path: Path) -> None:
        """No finding, no REDACT: the fold's policy mask still makes the released copy, audio and text."""
        plan = self._plan(store, tmp_path, ["we", "go", "every", "October", "home"])
        assert plan.policy_masks_n == 1
        _write_ledger(store, plan, "redacted", None)
        written = settle_release(
            store,
            "redacted",
            None,
            run_dir=tmp_path,
            artifacts_dir=_release(tmp_path),
            bleep_hz=None,
            fill="silence",
        )
        assert set(written)
        released = _release(tmp_path)
        assert (released / "transcript.txt").read_text() == "we go every [DATE_TIME] home\n"
        assert _silent(released / "audio.wav", 3.1, 3.4)
        assert not _silent(released / "audio.wav", 4.1, 4.4)

    def test_a_weekday_or_relative_day_tagged_as_a_name_is_released(self, store: ProvStore, tmp_path: Path) -> None:
        """An English model tagging "Hoy" or "Monday" PERSON names nobody: released by kind, in any family."""
        plan = self._plan(
            store,
            tmp_path,
            ["y", "Hoy", "fui", "y", "el", "Monday", "también"],
            [("PERSON", [1]), ("PERSON", [5])],
            language="es",
        )
        assert _states(plan) == {"Hoy": RELEASED_BY_KIND, "Monday": RELEASED_BY_KIND}


class TestRedactionPolicyV8:
    """Owner, 2026-10-04: no mask on a word nobody writes as a proper noun, nor on task content or a void place."""

    def _plan(
        self,
        store: ProvStore,
        tmp_path: Path,
        words: Sequence[str],
        located: Sequence[tuple[str, Sequence[int]]] = (),
        *,
        proposal: Sequence[dict[str, str]] = (),
        approvals: Sequence[str] = (),
        task_text: Sequence[str] = (),
    ) -> MaskPlan:
        """The plan over ``words``, a finding per ``(category, word indices)``, a reading and the task's texts."""
        findings = [
            (category, (_word_extent(indices[0])[0], _word_extent(indices[-1])[1])) for category, indices in located
        ]
        _seed_redact_store(store, tmp_path, words=list(words), findings=findings)
        if proposal:
            _annotate(store, proposal)
        return mask_plan(
            store,
            reviewer_applies=True,
            padding_ms=50,
            protected_categories=("PERSON", "NAME", "LOCATION", "LOC", "ORGANIZATION"),
            name_approvals=approvals,
            task_text=task_text,
        )

    def test_a_proper_name_still_waits_for_approval(self, store: ProvStore, tmp_path: Path) -> None:
        """Case: "John" is written as a proper noun and stays masked, pending a human; "said" is released."""
        plan = self._plan(store, tmp_path, ["then", "John", "said", "hi"], [("PERSON", [1]), ("PERSON", [2])])
        states = _states(plan)
        assert (states["John"], states["said"]) == (MASKED, RELEASED_NOT_PROPER)
        assert plan.person_names_masked == 1

    def test_a_state_stays_masked_under_v8(self, store: ProvStore, tmp_path: Path) -> None:
        """Case: "Florida" and a lower-case "wisconsin" are states; neither leaves its mask."""
        plan = self._plan(
            store, tmp_path, ["from", "Florida", "and", "wisconsin", "too"], [("LOCATION", [1]), ("LOCATION", [3])]
        )
        assert _states(plan) == {"Florida": MASKED, "wisconsin": MASKED}

    def test_a_lower_case_word_the_reviewer_asks_to_hide_keeps_its_mask(self, store: ProvStore, tmp_path: Path) -> None:
        """A reviewer ``redact`` entry on a lower-case word is the reviewer's call: the word stays masked."""
        plan = self._plan(
            store,
            tmp_path,
            ["i", "was", "a", "falconer", "there"],
            [("MISC", [3])],
            proposal=[_redact_entry("falconer", "OTHER")],
        )
        assert _states(plan) == {"falconer": MASKED}

    def test_a_number_a_detector_marked_keeps_its_mask(self, store: ProvStore, tmp_path: Path) -> None:
        """A number word or digits may be part of an identifier or an age: never released for its case."""
        plan = self._plan(store, tmp_path, ["call", "five", "five", "5", "now"], [("PHONE_NUMBER", [1, 2, 3])])
        assert set(_states(plan).values()) == {MASKED}

    def test_a_decade_or_an_ordinal_a_detector_marked_keeps_its_mask(self, store: ProvStore, tmp_path: Path) -> None:
        """r12 page: "fifties" and "twenty-second" are no proper nouns, but they write an age or a date."""
        plan = self._plan(
            store,
            tmp_path,
            ["back", "then", "fifties", "and", "the", "twenty-second", "too"],
            [("DATE_TIME", [2]), ("DATE_TIME", [5])],
        )
        assert _states(plan) == {"fifties": MASKED, "twenty-second": MASKED}

    def test_a_historical_place_is_released_with_its_reason(self, store: ProvStore, tmp_path: Path) -> None:
        """Case: "Roman Empire" released as historical leaves its mask, and the ledger counts the reason."""
        plan = self._plan(
            store,
            tmp_path,
            ["a", "fighter", "in", "the", "Roman", "Empire", "long", "ago"],
            [("LOCATION", [4, 5])],
            proposal=[_place_release("Roman Empire", "historical")],
        )
        assert _states(plan) == {"Roman": UNMASKED_BY_REVIEWER, "Empire": UNMASKED_BY_REVIEWER}
        assert plan.record(release="x", release_ground=None)["counts"]["place_reason_historical_n"] == 1

    def test_a_place_released_without_a_reason_stays_masked(self, store: ProvStore, tmp_path: Path) -> None:
        """The same release without ``place_reason`` lifts nothing."""
        plan = self._plan(
            store,
            tmp_path,
            ["a", "fighter", "in", "the", "Roman", "Empire", "long", "ago"],
            [("LOCATION", [4, 5])],
            proposal=[_release_entry("Roman Empire", "LOCATION")],
        )
        assert _states(plan) == {"Roman": MASKED, "Empire": MASKED}

    def test_a_state_released_for_any_other_reason_stays_masked(self, store: ProvStore, tmp_path: Path) -> None:
        """Case: "the Civil War in Virginia" released as fictional: a state goes only as historical."""
        plan = self._plan(
            store,
            tmp_path,
            ["the", "Civil", "War", "in", "Virginia", "was", "long"],
            [("LOCATION", [4])],
            proposal=[_place_release("Virginia", "fictional")],
        )
        assert _states(plan) == {"Virginia": MASKED}

    def test_a_state_released_as_historical_leaves_its_mask(self, store: ProvStore, tmp_path: Path) -> None:
        """The historical reason is the one that frees a state."""
        plan = self._plan(
            store,
            tmp_path,
            ["the", "Civil", "War", "in", "Virginia", "was", "long"],
            [("LOCATION", [4])],
            proposal=[_place_release("Virginia", "historical")],
        )
        assert _states(plan) == {"Virginia": UNMASKED_BY_REVIEWER}

    def test_a_work_title_tagged_person_is_released_when_relabelled(self, store: ProvStore, tmp_path: Path) -> None:
        """Case: "Star Wars" tagged PERSON, relabelled work_title, leaves its mask and asks no human."""
        plan = self._plan(
            store,
            tmp_path,
            ["i", "love", "Star", "Wars", "movies"],
            [("PERSON", [2, 3])],
            proposal=[{**_release_entry("Star Wars", "PERSON"), "relabel": "work_title", "why": "a film"}],
        )
        assert _states(plan) == {"Star": UNMASKED_BY_REVIEWER, "Wars": UNMASKED_BY_REVIEWER}
        assert plan.name_release_proposed == () and plan.person_names_masked == 0
        assert plan.record(release="x", release_ground=None)["counts"]["relabel_work_title_n"] == 1

    def test_a_relabel_outside_the_set_releases_nothing(self, store: ProvStore, tmp_path: Path) -> None:
        """An unknown relabel is no relabel: the name stays masked and waits for a human."""
        plan = self._plan(
            store,
            tmp_path,
            ["i", "love", "Star", "Wars", "movies"],
            [("PERSON", [2, 3])],
            proposal=[{**_release_entry("Star Wars", "PERSON"), "relabel": "movie", "why": "a film"}],
        )
        assert _states(plan) == {"Star": MASKED, "Wars": MASKED}
        assert plan.name_release_proposed == ("Star Wars",)

    def test_a_name_relabelled_a_place_follows_the_place_rule(self, store: ProvStore, tmp_path: Path) -> None:
        """A PERSON tag on "Tampa" relabelled place stays masked, as a place, without a place reason."""
        plan = self._plan(
            store,
            tmp_path,
            ["we", "went", "to", "Tampa", "often"],
            [("PERSON", [3])],
            proposal=[{**_release_entry("Tampa", "PERSON"), "relabel": "place"}],
        )
        (word,) = [word for mask in plan.masks for word in mask.words]
        assert (word.state, word.locked) == (MASKED, "place")

    def test_the_target_word_is_never_masked(self, store: ProvStore, tmp_path: Path) -> None:
        """Productive vocabulary: "Gladiators" tagged PERSON is the target word "gladiator", inflected."""
        plan = self._plan(
            store, tmp_path, ["Gladiators", "fought", "in", "arenas"], [("PERSON", [0])], task_text=("gladiator",)
        )
        assert plan.final == [] and plan.person_names_masked == 0
        assert len(plan.task_text_ids) == 1
        assert not plan.task_content_only, "a lexical task's target word is not a task event"

    def test_a_season_the_question_names_is_task_content(self, store: ProvStore, tmp_path: Path) -> None:
        """free-speech-v2-1 asks about seasons: "summer" is the question's own word, "July" is the speaker's."""
        plan = self._plan(
            store,
            tmp_path,
            ["my", "favorite", "is", "summer", "in", "July"],
            task_text=("What is your favorite season (winter, fall, summer, or spring) and why?",),
        )
        policy = [mask for mask in plan.masks if mask.source == "policy"]
        assert [[word.text for word in mask.words] for mask in policy] == [["July"]]

    def test_a_personal_place_inside_a_definition_stays_masked(self, store: ProvStore, tmp_path: Path) -> None:
        """Case: "Roman Empire" released as task content; "Tampa", where the speaker's brother is, kept."""
        plan = self._plan(
            store,
            tmp_path,
            ["a", "fighter", "of", "the", "Roman", "Empire", "like", "my", "brother", "in", "Tampa"],
            [("LOCATION", [4, 5]), ("LOCATION", [10])],
            proposal=[_place_release("Roman Empire", "task_content")],
            task_text=("gladiator",),
        )
        states = _states(plan)
        assert (states["Roman"], states["Empire"], states["Tampa"]) == (
            UNMASKED_BY_REVIEWER,
            UNMASKED_BY_REVIEWER,
            MASKED,
        )


def _all_words(plan: MaskPlan) -> dict[str, list[Any]]:
    """Each surface to the words of that surface across every mask, in mask order."""
    out: dict[str, list[Any]] = {}
    for mask in plan.masks:
        for word in mask.words:
            out.setdefault(word.text, []).append(word)
    return out


class TestADecisionHoldsForEveryOccurrenceOfAToken:
    """Owner, 2026-10-08: a token decided a name, or released, is decided so at every occurrence."""

    def _redacted(
        self,
        store: ProvStore,
        config: TriageConfig,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        words: Sequence[str],
        findings: Sequence[tuple[Any, ...]],
        **seed: Any,  # noqa: ANN401
    ) -> None:
        """REDACT over a seeded store, its re-scan clean."""
        _seed_redact_store(store, tmp_path, words=list(words), findings=findings, **seed)
        _stub_pii(monkeypatch, findings=[])
        redact(store, "recording", config, run_dir=tmp_path, artifacts_dir=_release(tmp_path))

    def test_a_name_kept_masked_masks_every_occurrence(
        self, store: ProvStore, redact_config: TriageConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A name masked once ("Alice") masks the "alice" no detector marked, as a mask of its own."""
        words = ["i", "met", "Alice", "and", "alice", "left"]
        self._redacted(store, redact_config, tmp_path, monkeypatch, words, [("PERSON", _word_extent(2))])
        plan = _plan(store)
        assert [word.state for word in _all_words(plan)["alice"]] == [MASKED]
        (spread,) = [mask for mask in plan.masks if mask.source == PROPAGATED]
        (word,) = spread.words
        finding = live_entities(store, "pii")[0]
        assert (word.propagated, word.propagation, word.source_findings) == (True, PROPAGATION_MASK, (finding.id,))
        assert len(plan.final) == 2 and plan.changed and plan.propagated_masked_n == 1
        record = plan.record(release="redacted", release_ground=None)
        assert record["counts"]["propagated_masked_n"] == 1
        (written,) = [w for m in record["masks"] if m["source"] == PROPAGATED for w in m["words"]]
        assert (written["propagated"], written["source_findings"]) == (True, [finding.id])

    def test_a_name_masked_once_masks_an_occurrence_a_finding_released(
        self, store: ProvStore, redact_config: TriageConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The lower-case "barakat" the not-proper rule would release stays masked beside a masked "Barakat"."""
        words = ["Barakat", "came", "and", "barakat", "went"]
        self._redacted(
            store,
            redact_config,
            tmp_path,
            monkeypatch,
            words,
            [("PERSON", _word_extent(0)), ("PERSON", _word_extent(3))],
        )
        states = {text: [word.state for word in found] for text, found in _all_words(_plan(store)).items()}
        assert states == {"Barakat": [MASKED], "barakat": [MASKED]}

    def test_a_name_another_recogniser_heard_is_masked_there_too(
        self, store: ProvStore, redact_config: TriageConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Qwen read "what" as "Barakat": the name masked elsewhere is masked under that word as well."""
        words = ["my", "friend", "Barakat", "said", "what"]
        self._redacted(
            store,
            redact_config,
            tmp_path,
            monkeypatch,
            words,
            [("PERSON", _word_extent(2))],
            readings={4: {"asr_crisperwhisper": "what", "asr_qwen": "Barakat"}},
        )
        plan = _plan(store)
        assert [word.state for word in _all_words(plan)["what"]] == [MASKED]
        assert any(extent.start <= _word_extent(4)[0] and extent.end >= _word_extent(4)[1] for extent in plan.final)

    def test_a_finding_read_off_one_recogniser_spreads_by_that_recogniser_s_token(
        self, store: ProvStore, redact_config: TriageConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A PERSON read off Qwen's "Barakat" under the consensus "Time" spreads "barakat", never "time"."""
        words = ["Time", "and", "time", "and", "what"]
        self._redacted(
            store,
            redact_config,
            tmp_path,
            monkeypatch,
            words,
            [("PERSON", _word_extent(0))],
            readings={
                0: {"asr_crisperwhisper": "Time", "asr_qwen": "Barakat"},
                4: {"asr_crisperwhisper": "what", "asr_qwen": "barakat"},
            },
            haystacks={0: "asr_qwen"},
        )
        found = _all_words(_plan(store))
        assert [word.state for word in found["what"]] == [MASKED]
        assert "time" not in found

    def test_a_released_token_is_released_wherever_a_recogniser_read_it(
        self, store: ProvStore, redact_config: TriageConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The reviewer's "the Prince danced" releases the "Prance" Qwen read as "Prince"."""
        words = ["the", "Prince", "danced", "the", "Prance"]
        self._redacted(
            store,
            redact_config,
            tmp_path,
            monkeypatch,
            words,
            [("PERSON", _word_extent(1)), ("PERSON", _word_extent(4))],
            readings={4: {"asr_crisperwhisper": "Prance", "asr_qwen": "Prince"}},
        )
        _annotate(store, [{**_release_entry("the Prince danced", "PERSON"), "relabel": "other_non_person"}])
        plan = _plan(store)
        found = _all_words(plan)
        assert [word.state for word in found["Prince"]] == [UNMASKED_BY_REVIEWER]
        (prance,) = found["Prance"]
        assert (prance.state, prance.propagation) == (UNMASKED_BY_REVIEWER, PROPAGATION_RELEASE)
        assert prance.propagated_from == (found["Prince"][0].word_id,)
        assert plan.final == []


_DDK_STEM = "sub-x_ses-y_task-diadochokinesis-v2-buttercup"


def _ddk_reading(store: ProvStore, events: Sequence[tuple[float, float]], decision: str = "present") -> None:
    """SPEECH's syllable-task reading, carrying its events."""
    agent = store.agent(agent_type="software", version="senselab test-ddk")
    activity = store.activity(node="SPEECH", step="ddk", parameters={})
    store.was_associated_with(activity, agent)
    reading = store.entity(
        prov_type="measurement",
        extent=None,
        attributes={
            "name": "ddk_task_reading",
            "decision": decision,
            "unit": "cycle",
            "reading": {
                "evidence": {
                    "decision": decision,
                    "events": [{"start_s": start, "end_s": end} for start, end in events],
                }
            },
        },
    )
    store.was_generated_by(reading, activity)


class TestTheTaskSOwnEventsAreNeverMasked:
    """Owner, 2026-10-08: a word lying on the task's own events is task content, never PII and never masked."""

    _WORDS = ("What", "are", "the", "time?", "like")
    _READINGS = {
        0: {"asr_crisperwhisper": "What", "asr_qwen": "Barakat"},
        3: {"asr_crisperwhisper": "time?", "asr_qwen": "barakat"},
    }
    _EVENTS = ((0.0, 0.6), (2.9, 3.6))

    def _seed(self, store: ProvStore, tmp_path: Path) -> None:
        """The buttercup card: Qwen's "barakat" under the consensus "time?", tagged PERSON off Qwen."""
        _seed_redact_store(
            store,
            tmp_path,
            words=list(self._WORDS),
            findings=[("PERSON", _word_extent(3))],
            readings=self._READINGS,
            haystacks={0: "asr_qwen"},
            recording_stem=_DDK_STEM,
        )

    def test_redact_plans_no_mask_over_the_task_s_events(
        self, store: ProvStore, redact_config: TriageConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """REDACT exempts the finding as the task's own content, records why, and masks nothing."""
        self._seed(store, tmp_path)
        _ddk_reading(store, self._EVENTS)
        _stub_pii(monkeypatch, findings=[])
        result = redact(
            store,
            "recording",
            redact_config,
            run_dir=tmp_path,
            artifacts_dir=_release(tmp_path),
            task_family="diadochokinesis-v2-buttercup",
        )
        assert result.verdict.outcome is Outcome.PASS
        assert _verdict_entity(store, "REDACT").attributes["redactions_n"] == 0
        assert _verdict_entity(store, "REDACT").attributes["task_event_exempt_n"] == 1
        (exempt,) = [
            e
            for e in store.entities("assertion")
            if e.attributes.get("verb") == "exempt" and e.attributes.get("label") == TASK_EVENT_LABEL
        ]
        assert exempt.attributes["events_n"] == 2
        folded = verdict_module.verdict(store, None, redact_config, run_dir=tmp_path).file_verdict
        assert folded.release is not None
        assert (folded.release.value, folded.release_ground) == ("as_is", FINDINGS_ARE_TASK_CONTENT)

    def test_a_re_fold_releases_the_original_over_a_mask_redact_already_planned(
        self, store: ProvStore, redact_config: TriageConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A store REDACT masked before the rule: the fold masks nothing and says the task's content is why."""
        self._seed(store, tmp_path)
        _stub_pii(monkeypatch, findings=[])
        redact(store, "recording", redact_config, run_dir=tmp_path, artifacts_dir=_release(tmp_path))
        assert _verdict_entity(store, "REDACT").attributes["redactions_n"] == 1
        _ddk_reading(store, self._EVENTS)
        plan = _plan(store)
        assert plan.final == [] and plan.task_content_only
        assert plan.task_event_ids == tuple(live_entities(store, "pii")[0].attributes["word_ids"])
        folded = verdict_module.verdict(store, None, redact_config, run_dir=tmp_path).file_verdict
        assert folded.release is not None
        assert (folded.release.value, folded.release_ground) == ("as_is", TASK_CONTENT_UNMASKED)

    def test_the_token_read_on_the_events_is_released_off_them_too(
        self, store: ProvStore, redact_config: TriageConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The "barakat" past the last event, tagged too, is the same misreading of the task."""
        _seed_redact_store(
            store,
            tmp_path,
            words=[*self._WORDS, "barakat"],
            findings=[("PERSON", _word_extent(3)), ("PERSON", _word_extent(5))],
            readings=self._READINGS,
            haystacks={0: "asr_qwen"},
            recording_stem=_DDK_STEM,
        )
        _ddk_reading(store, self._EVENTS)
        _stub_pii(monkeypatch, findings=[])
        redact(store, "recording", redact_config, run_dir=tmp_path, artifacts_dir=_release(tmp_path))
        plan = _plan(store)
        assert plan.final == [] and len(plan.task_event_ids) == 2

    def test_a_reading_that_read_another_activity_releases_nothing(
        self, store: ProvStore, redact_config: TriageConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Events of a reading decided absent are not the task's: the mask stands."""
        self._seed(store, tmp_path)
        _stub_pii(monkeypatch, findings=[])
        redact(store, "recording", redact_config, run_dir=tmp_path, artifacts_dir=_release(tmp_path))
        _ddk_reading(store, self._EVENTS, decision="absent")
        assert _plan(store, applies=False).task_event_ids == ()

    def test_a_lexical_family_has_no_task_events(
        self, store: ProvStore, redact_config: TriageConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A free-speech recording's words are never task content by time, whatever reading the store holds."""
        _seed_redact_store(
            store,
            tmp_path,
            words=list(self._WORDS),
            findings=[("PERSON", _word_extent(3))],
            recording_stem="sub-x_ses-y_task-free-speech-1",
        )
        _ddk_reading(store, self._EVENTS)
        _stub_pii(monkeypatch, findings=[])
        redact(store, "recording", redact_config, run_dir=tmp_path, artifacts_dir=_release(tmp_path))
        assert _verdict_entity(store, "REDACT").attributes["task_event_exempt_n"] == 0


class TestTheTaskContentGroundIsTheEventsAlone:
    """Owner, 2026-10-09: ``task_content_unmasked`` is for findings on a non-lexical task's events only."""

    def test_a_lexical_task_s_declared_word_keeps_its_previous_ground(
        self, store: ProvStore, redact_config: TriageConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A Cinderella retelling REDACT masked on "cinderella's": no mask stands, on ``no_content_masked``."""
        _seed_redact_store(
            store,
            tmp_path,
            words=["then", "cinderella's", "coach", "left"],
            findings=[("PERSON", _word_extent(1))],
            recording_stem="sub-x_ses-y_task-cinderella-story",
        )
        _stub_pii(monkeypatch, findings=[])
        redact(store, "recording", redact_config, run_dir=tmp_path, artifacts_dir=_release(tmp_path))
        assert _verdict_entity(store, "REDACT").attributes["redactions_n"] == 1
        folded = verdict_module.verdict(store, None, redact_config, run_dir=tmp_path).file_verdict
        assert folded.release is not None
        assert (folded.release.value, folded.release_ground) == ("as_is", NO_CONTENT_MASKED)


class TestAMultiWordNameSpreadsAsARun:
    """A kept name of several words is matched as the run it is, never word by word."""

    def test_new_york_masks_new_york_and_never_a_lone_new(
        self, store: ProvStore, redact_config: TriageConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The kept "New York" masks "new york" later, and leaves "a new car" alone."""
        words = ["in", "New", "York", "a", "new", "car", "in", "new", "york"]
        _seed_redact_store(
            store, tmp_path, words=words, findings=[("LOCATION", (_word_extent(1)[0], _word_extent(2)[1]))]
        )
        _stub_pii(monkeypatch, findings=[])
        redact(store, "recording", redact_config, run_dir=tmp_path, artifacts_dir=_release(tmp_path))
        plan = _plan(store)
        assert sorted(_masked_indices(store, plan)) == [1, 2, 7, 8]


def _masked_indices(store: ProvStore, plan: MaskPlan) -> list[int]:
    """The stream positions of the words the plan keeps masked."""
    kept = {word.word_id for mask in plan.masks for word in mask.words if word.state == MASKED}
    return [int(word.attributes["index"]) for word in store.entities("word") if word.id in kept]
