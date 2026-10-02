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
from typing import Any, Iterable, Sequence

import numpy as np
import pytest
import soundfile as sf

from senselab.audio.data_structures import Audio
from senselab.audio.data_structures.audio_hints import AudioHints, ExpectedSpeech
from senselab.audio.workflows.triage.config import TriageConfig, load_triage_config
from senselab.audio.workflows.triage.nodes import redact as redact_module
from senselab.audio.workflows.triage.nodes import verdict as verdict_module
from senselab.audio.workflows.triage.nodes.common import resolve_stream
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
    PROPOSED_BY_REVIEWER,
    RELEASED_FILES,
    REVIEWER,
    STREAM_NAME,
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
    REDACTION_LLM_ANNOTATION,
    REVIEWER_CLEARED_RESCAN,
    REVIEWER_UNMASKED_SOME,
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
    words: Sequence[str] = ("hello", "alice"),
    findings: Sequence[tuple[Any, ...]] = (),
    extra_marks: Sequence[tuple[str, str]] = (),
    target_speaker: str | None = None,
    scanned: bool = True,
    scanned_by: Sequence[str] = ALL_DETECTORS,
    scan_failed: Sequence[str] = (),
    timings: dict[int, dict[str, tuple[float, float]]] | None = None,
    residue: Sequence[int] | None = None,
    unplaced: Sequence[tuple[str, str]] = (),
) -> None:
    """Write the store PREPROCESS and SPEECH leave for REDACT, with ``tmp_path`` as the run dir.

    ``findings`` are ``(category, (start, end))`` or ``(category, (start, end), speaker)``. Each
    writes a ``pii`` entity and a ``label``/``pii`` assertion derived from every consensus word it
    overlaps — the store's shared shape for a marking. ``extra_marks`` are ``(word_text, category)``
    markings placed on a word the finding's own extent does not reach, which is the state a
    re-planning pass exists to widen. ``target_speaker`` writes SPEECH's verdict so a speaker-scoped
    reader has something to scope by. ``timings`` gives a word, by index, its sources' own timings,
    so its hull can reach past its derived extent. ``residue`` names, by index, the words the scan's
    residue holds; every word when None.
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
            attributes=word_attributes(text, _word_extent(index), index=index, timings=(timings or {}).get(index)),
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
        covered = [
            word_ids[i] for i in range(len(words)) if _word_extent(i)[0] < bounds[1] and _word_extent(i)[1] > bounds[0]
        ]
        pii_id = store.entity(
            prov_type="pii",
            extent=bounds,
            attributes={
                "category": category,
                "source": "presidio",
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
        _seed_redact_store(store, tmp_path, words=["hello", "alice"], findings=[("PERSON", (1.0, 2.0))])
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
            store, tmp_path, words=["hello", "alice"], findings=[("PERSON", (1.0, 2.0)), ("NAME", (1.0, 2.0))]
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
        _seed_redact_store(store, tmp_path, words=["hello", "alice"], findings=[("PERSON", (1.0, 2.0))])
        _stub_pii(monkeypatch, findings=[("PERSON", "alice")])
        result = redact(store, "recording", redact_config, run_dir=tmp_path, artifacts_dir=_release(tmp_path))
        assert result.verdict.outcome is Outcome.FAIL


def _settle(store: ProvStore, release: str, ground: str | None, tmp_path: Path) -> dict[str, Path]:
    """Settle the release directory the way every driver does, under a silence fill."""
    return settle_release(store, release, ground, run_dir=tmp_path, artifacts_dir=_release(tmp_path), bleep_hz=None)


class TestTheFoldSettlesTheReleaseOfAFail:
    """REDACT writes a copy only on a pass; where the fold releases a fail, the copy comes from the store."""

    def _failed(self, store: ProvStore, config: TriageConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        """A REDACT fail whose re-scan still reads the name, and its (empty) release directory."""
        _seed_redact_store(store, tmp_path, words=["hello", "alice"], findings=[("PERSON", (1.0, 2.0))])
        _stub_pii(monkeypatch, findings=[("PERSON", "alice")])
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
        written = _settle(store, "release_with_redaction", REVIEWER_CLEARED_RESCAN, tmp_path)
        assert sorted(path.name for path in written.values()) == sorted(RELEASED_FILES)
        assert (released / "transcript.txt").read_text() == "hello [PERSON]\n"
        assert "alice" not in (released / "consensus.json").read_text()

    def test_a_withheld_fail_leaves_no_copy(
        self,
        store: ProvStore,
        redact_config: TriageConfig,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A fold that withholds again removes a copy an earlier fold released."""
        released = self._failed(store, redact_config, tmp_path, monkeypatch)
        _settle(store, "release_with_redaction", REVIEWER_CLEARED_RESCAN, tmp_path)
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
        _seed_redact_store(store, tmp_path, words=["hello", "alice"], findings=[("PERSON", (1.0, 2.0))])
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
        _seed_redact_store(store, tmp_path, words=["hello", "alice"], findings=[("PERSON", (1.0, 2.0))])
        _stub_pii(monkeypatch, findings=[])
        redact(store, "recording", redact_config, run_dir=tmp_path, artifacts_dir=_release(tmp_path))
        before = (_release(tmp_path) / "transcript.txt").read_text()
        _settle(store, "release_with_redaction", None, tmp_path)
        assert (_release(tmp_path) / "transcript.txt").read_text() == before == "hello [PERSON]\n"


_RESET_WORDS = ("i", "met", "alice", "in", "brooklyn", "today")
_RESET_FINDINGS = [("PERSON", _word_extent(2)), ("LOCATION", _word_extent(4))]


def _annotate(
    store: ProvStore, proposal: Sequence[dict[str, str]], *, original: str = "clean", redaction: str = ""
) -> None:
    """REVIEW's annotation, carrying one proposal."""
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
        },
    )
    store.was_generated_by(annotation, activity)


def _release_entry(text: str, category: str = "OTHER") -> dict[str, str]:
    """One ``release`` entry of a proposal."""
    return {"text": text, "action": "release", "category": category, "why": ""}


def _redact_entry(text: str, category: str = "PERSON") -> dict[str, str]:
    """One ``redact`` entry of a proposal."""
    return {"text": text, "action": "redact", "category": category, "why": ""}


def _plan(store: ProvStore, *, applies: bool = True, review: Sequence[str] = ()) -> MaskPlan:
    """The word-level plan under a 50 ms margin, as VERDICT computes it."""
    return mask_plan(store, reviewer_applies=applies, padding_ms=50, human_review_categories=review)


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
        """A REDACT pass over "i met alice in brooklyn today"."""
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
        plan = _plan(store)
        words = {word.text: word for mask in plan.masks for word in mask.words}
        assert words["Gladiator"].state == UNMASKED_BY_REVIEWER
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
        _annotate(store, [_release_entry("in Brooklyn,", "LOCATION")])
        plan = _plan(store)
        assert [mask.outcome for mask in plan.masks] == [MASK_UNCHANGED, MASK_UNMASKED]
        assert _states(plan)["brooklyn"] == UNMASKED_BY_REVIEWER
        _write_ledger(store, plan, "release_with_redaction", REVIEWER_UNMASKED_SOME)
        _settle(store, "release_with_redaction", REVIEWER_UNMASKED_SOME, tmp_path)
        released = _release(tmp_path)
        assert (released / "transcript.txt").read_text() == "i met [PERSON] in brooklyn today\n"
        assert _silent(released / "audio.wav", 2.1, 2.4)
        assert not _silent(released / "audio.wav", 4.1, 4.4)

    def test_part_of_a_mask_is_unmasked_and_the_mask_is_split(
        self, store: ProvStore, redact_config: TriageConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """One mask over "alice in brooklyn": the named place goes, the function word goes, the name stays."""
        self._passed(store, redact_config, tmp_path, monkeypatch, findings=[("PERSON", (2.0, 4.5))])
        _annotate(store, [_release_entry("brooklyn", "LOCATION")], original="carries_pii")
        plan = _plan(store)
        (mask,) = plan.masks
        assert mask.outcome == MASK_PARTLY_UNMASKED
        assert _states(plan) == {"alice": MASKED, "in": UNMASKED_BY_TRIM, "brooklyn": UNMASKED_BY_REVIEWER}
        (extent,) = plan.final
        assert extent.start <= 2.0 and extent.end >= 2.5
        assert extent.end <= 3.0, "the padding stops where the unmasked neighbour begins"
        _write_ledger(store, plan, "release_with_redaction", REVIEWER_UNMASKED_SOME)
        _settle(store, "release_with_redaction", REVIEWER_UNMASKED_SOME, tmp_path)
        released = _release(tmp_path)
        assert (released / "transcript.txt").read_text() == "i met [PERSON] in brooklyn today\n"
        assert _silent(released / "audio.wav", 2.05, 2.45)
        assert not _silent(released / "audio.wav", 3.1, 3.4)
        assert not _silent(released / "audio.wav", 4.1, 4.4)

    def test_a_mask_keeps_no_function_word_with_or_without_a_reading(
        self, store: ProvStore, redact_config: TriageConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A mask over "alice in" keeps only the name, whether the padding or the finding caught "in"."""
        self._passed(store, redact_config, tmp_path, monkeypatch, findings=[("PERSON", (2.0, 3.5))])
        plan = _plan(store, applies=False)
        assert [mask.outcome for mask in plan.masks] == [MASK_TRIMMED]
        assert _states(plan) == {"alice": MASKED, "in": UNMASKED_BY_TRIM}

    def test_a_word_only_the_padding_reached_is_not_under_the_mask(
        self, store: ProvStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A 600 ms margin reaches "met" and "in"; the mask is the finding's own word and nothing else."""
        config = _override(tmp_path, "redaction:\n  padding_ms: 600\n  fill: silence\n")
        self._passed(store, config, tmp_path, monkeypatch, findings=[("PERSON", _word_extent(2))])
        plan = mask_plan(store, reviewer_applies=False, padding_ms=600)
        (mask,) = plan.masks
        assert mask.outcome == MASK_UNCHANGED
        assert {word.text: (word.state, word.finding) for word in mask.words} == {"alice": (MASKED, True)}
        assert plan.changed, "REDACT's padded extent reached the neighbours; the released copy does not"
        (extent,) = plan.final
        assert (extent.start, extent.end) == (1.5, 3.0), "the margin stops at each unmasked neighbour"

    def test_a_name_spelled_as_a_function_word_is_never_trimmed(
        self, store: ProvStore, redact_config: TriageConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A "May" marked PERSON keeps its mask through the trim; only a reviewer quote naming it unmasks it."""
        _seed_redact_store(store, tmp_path, words=["i", "met", "May", "today"], findings=[("PERSON", _word_extent(2))])
        _stub_pii(monkeypatch, findings=[])
        redact(store, "recording", redact_config, run_dir=tmp_path, artifacts_dir=_release(tmp_path))
        unprotected = mask_plan(store, reviewer_applies=False, padding_ms=50)
        assert _states(unprotected)["May"] == UNMASKED_BY_TRIM
        protected = mask_plan(store, reviewer_applies=False, padding_ms=50, protected_categories=("PERSON", "NAME"))
        assert _states(protected)["May"] == MASKED
        (word,) = [word for mask in protected.masks for word in mask.words if word.text == "May"]
        assert (word.categories, word.proper) == (("PERSON",), True)
        _annotate(store, [_release_entry("may")])
        released = mask_plan(store, reviewer_applies=True, padding_ms=50, protected_categories=("PERSON", "NAME"))
        assert _states(released)["May"] == UNMASKED_BY_REVIEWER

    def test_a_function_word_a_name_finding_spans_is_still_trimmed(
        self, store: ProvStore, redact_config: TriageConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A PERSON finding over "and the" does not make them names; lower case, they are trimmed."""
        _seed_redact_store(store, tmp_path, words=["alice", "and", "the", "dog"], findings=[("PERSON", (0.0, 2.5))])
        _stub_pii(monkeypatch, findings=[])
        redact(store, "recording", redact_config, run_dir=tmp_path, artifacts_dir=_release(tmp_path))
        plan = mask_plan(store, reviewer_applies=False, padding_ms=50, protected_categories=("PERSON", "NAME"))
        assert _states(plan)["and"] == UNMASKED_BY_TRIM and _states(plan)["the"] == UNMASKED_BY_TRIM
        assert _states(plan)["alice"] == MASKED

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
        _write_ledger(store, plan, "release_with_redaction", None)
        _settle(store, "release_with_redaction", None, tmp_path)
        released = _release(tmp_path)
        assert (released / "transcript.txt").read_text() == "back in [DATE_TIME] is\n"
        assert '"words_n": 2' in (released / "consensus.json").read_text()
        assert _silent(released / "audio.wav", 2.1, 3.4)
        assert not _silent(released / "audio.wav", 1.1, 1.4)

    def test_a_finding_bridging_task_words_masks_only_the_residue_either_side(
        self, store: ProvStore, redact_config: TriageConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A read passage's residue "maria ... smith" around stimulus words: two masks, the passage audible."""
        _seed_redact_store(
            store,
            tmp_path,
            words=["the", "caterpillar", "maria", "ate", "leaves", "smith"],
            findings=[("PERSON", (2.0, 5.5))],
            residue=[2, 5],
        )
        _stub_pii(monkeypatch, findings=[])
        redact(store, "recording", redact_config, run_dir=tmp_path, artifacts_dir=_release(tmp_path))
        plan = _plan(store, applies=False)
        (mask,) = plan.masks
        assert (mask.outcome, mask.task_words_n) == (MASK_UNCHANGED, 2)
        assert _states(plan) == {"maria": MASKED, "smith": MASKED}, "no task word is listed under a mask"
        assert len(plan.final) == 2
        _write_ledger(store, plan, "release_with_redaction", None)
        _settle(store, "release_with_redaction", None, tmp_path)
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
            words=["the", "caterpillar", "ate", "maria"],
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
            words=["the", "caterpillar", "ate", "maria", "and", "leaves"],
            findings=[("PERSON", (0.0, 5.5))],
            residue=[3, 4],
        )
        _stub_pii(monkeypatch, findings=[])
        redact(store, "recording", redact_config, run_dir=tmp_path, artifacts_dir=_release(tmp_path))
        plan = _plan(store, applies=False)
        assert _states(plan) == {"maria": MASKED, "and": UNMASKED_BY_TRIM}
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
            words=["i", "am", "very", "hoarse", "my", "voice", "is", "very", "hoarse"],
            findings=[("MISC", (2.0, 3.5)), ("MISC", (5.0, 8.5))],
        )
        _stub_pii(monkeypatch, findings=[])
        redact(store, "recording", redact_config, run_dir=tmp_path, artifacts_dir=_release(tmp_path))
        _annotate(store, [_release_entry("am very hoarse", "OTHER")])
        plan = _plan(store)
        assert [mask.outcome for mask in plan.masks] == [MASK_UNMASKED, MASK_PARTLY_UNMASKED]
        kept = {word.text for mask in plan.masks for word in mask.words if word.state == MASKED}
        assert kept == {"voice"}
        (second,) = [word for word in plan.masks[1].words if word.text == "hoarse"]
        assert second.propagated and not second.named

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
        _annotate(store, [_release_entry("brooklyn"), _redact_entry("today", "DATE_TIME")])
        plan = _plan(store, applies=False)
        brooklyn = next(word for mask in plan.masks for word in mask.words if word.text == "brooklyn")
        assert (brooklyn.state, brooklyn.named) == (MASKED, True)
        assert not plan.reviewer_applied

    def test_an_identifying_original_is_unmasked_where_the_reviewer_says(
        self, store: ProvStore, redact_config: TriageConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Owner, 2026-09-27: named unmasks always apply, over an original read as carrying PII too."""
        self._passed(store, redact_config, tmp_path, monkeypatch)
        _annotate(store, [_release_entry("alice"), _release_entry("brooklyn")], original="carries_pii")
        plan = _plan(store)
        assert plan.reviewer_applied
        assert plan.final == []
        assert set(_states(plan).values()) == {UNMASKED_BY_REVIEWER}

    def test_redact_entries_are_placed_on_words_and_marked_for_review(
        self, store: ProvStore, redact_config: TriageConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Each proposal lists the words it names; one already masked says so; a condition is marked."""
        self._passed(store, redact_config, tmp_path, monkeypatch)
        _annotate(
            store,
            [
                _redact_entry("met", "CONDITION"),
                _redact_entry("Alice", "PERSON"),
                _redact_entry("brooklyn tod", "LOCATION"),
                _redact_entry("nowhere", "LOCATION"),
            ],
            original="carries_pii",
        )
        plan = _plan(store, applies=False, review=("CONDITION",))
        placed = {
            span.text: (span.placed, span.texts, bool(span.masked_ids), span.human_review) for span in plan.proposals
        }
        assert placed["met"] == (PLACED_WORDS, ("met",), False, True)
        assert placed["Alice"] == (PLACED_WORDS, ("alice",), True, False)
        assert placed["brooklyn tod"] == (PLACED_SUBSTRING, ("brooklyn", "today"), True, False)
        assert placed["nowhere"] == ("", (), False, False)

    def test_a_condition_the_task_itself_names_is_held_for_no_one(
        self, store: ProvStore, redact_config: TriageConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A health condition every word of which is the task's own vocabulary is task content, not a review."""
        from senselab.audio.workflows.triage.nodes.redact import NEW, TASK_CONTENT
        from senselab.audio.workflows.triage.task_lexicon import TaskLexicon

        self._passed(store, redact_config, tmp_path, monkeypatch)
        _annotate(
            store, [_redact_entry("met", "CONDITION"), _redact_entry("Alice", "CONDITION")], original="carries_pii"
        )
        plan = mask_plan(
            store,
            reviewer_applies=True,
            padding_ms=50,
            human_review_categories=("CONDITION",),
            lexicon=TaskLexicon(None, (("met",),)),
        )
        by_text = {span.text: (span.agreement, span.human_review) for span in plan.proposals}
        assert by_text["met"] == (TASK_CONTENT, False)
        assert by_text["Alice"][0] != TASK_CONTENT
        assert not any(span.human_review and span.agreement == NEW and span.text == "met" for span in plan.proposals)

    def test_the_ledger_record_counts_every_state(
        self, store: ProvStore, redact_config: TriageConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The record VERDICT writes carries the masks, the final masks, the proposals and the counts."""
        self._passed(store, redact_config, tmp_path, monkeypatch, findings=[("PERSON", (2.0, 4.5))])
        _annotate(store, [_release_entry("brooklyn", "LOCATION")], original="carries_pii")
        record = _plan(store).record(release="release_with_redaction", release_ground=REVIEWER_UNMASKED_SOME)
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
        assert (folded.release.value, folded.release_ground) == ("release_without_redaction", FINDINGS_ARE_TASK_CONTENT)
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
            store, tmp_path, words=["[UH]", "my", "name", "is", "alice"], findings=[("PERSON", (4.0, 4.5))]
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
            store, tmp_path, words=["[UH]", "my", "name", "is", "alice"], findings=[("PERSON", (4.0, 4.5))]
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
            store, tmp_path, words=["[UH]", "my", "name", "is", "alice"], findings=[("PERSON", (4.0, 4.5))]
        )
        _stub_pii(monkeypatch, findings=[("PERSON", "alice")])
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
        words = ["i", "met", "alice", "in", "brooklyn"]
        self._seed(store, redact_config, tmp_path, monkeypatch, words, [("PERSON", _word_extent(2))])
        _annotate(store, [_redact_entry("alice", "PERSON"), _redact_entry("brooklyn", "LOCATION")])
        plan = _plan(store)
        assert [span.agreement for span in plan.proposals] == [AGREED_MASKED, NEW]
        assert plan.agreed == frozenset({0})

    def test_a_proposals_function_words_are_not_proposed(
        self, store: ProvStore, redact_config: TriageConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A redact quote's function words are neither its ``content_ids`` nor counted as proposed."""
        words = ["i", "had", "a", "cyst", "in", "my", "neck"]
        self._seed(store, redact_config, tmp_path, monkeypatch, words, [])
        _annotate(store, [_redact_entry("a cyst in my neck", "CONDITION")])
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
        words = ["the", "prince", "danced", "the", "prince", "smiled", "the", "prince"]
        _seed_redact_store(store, tmp_path, words=words, findings=[("PERSON", _word_extent(i)) for i in (1, 4, 7)])
        _stub_pii(monkeypatch, findings=[])
        redact(store, "recording", config, run_dir=tmp_path, artifacts_dir=_release(tmp_path))

    def test_one_quoted_occurrence_releases_every_one(
        self, store: ProvStore, redact_config: TriageConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """r6's Cinderella card: "the prince danced" releases the other two princes as well."""
        self._seed(store, redact_config, tmp_path, monkeypatch)
        _annotate(store, [_release_entry("the prince danced", "PERSON")])
        plan = _plan(store)
        princes = [word for mask in plan.masks for word in mask.words]
        assert [word.state for word in princes] == [UNMASKED_BY_REVIEWER] * 3
        assert [(word.named, word.propagated) for word in princes] == [(True, False), (False, True), (False, True)]

    def test_a_redact_entry_on_the_term_stops_it_spreading(
        self, store: ProvStore, redact_config: TriageConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Where the reviewer also asks for the term hidden, only the quoted occurrence is released."""
        self._seed(store, redact_config, tmp_path, monkeypatch)
        _annotate(store, [_release_entry("the prince danced", "PERSON"), _redact_entry("prince smiled", "PERSON")])
        plan = _plan(store)
        states = [word.state for mask in plan.masks for word in mask.words]
        assert states == [UNMASKED_BY_REVIEWER, MASKED, MASKED]


class TestAFunctionWordIsNotATerm:
    """Term propagation carries content words; a released "the" is the trim's, not a term's."""

    def test_the_second_prince_is_released_and_its_article_is_not_propagated(
        self, store: ProvStore, redact_config: TriageConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Two PERSON findings over "the prince"; the reviewer releases the first."""
        words = ["the", "prince", "sang", "the", "prince"]
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
        _annotate(store, [_release_entry("the prince sang", "PERSON")])
        second = _plan(store).masks[1]
        flags = {word.text: word.propagated for word in second.words}
        assert flags == {"the": False, "prince": True}
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
        assert plan.record(release="x", release_ground=None)["counts"]["task_lexicon_words_n"] == 2

    def test_a_word_not_in_proper_form_under_a_place_is_cut(
        self, store: ProvStore, redact_config: TriageConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """r7's free-speech-v2 card: LOC "Florida Like"; the reviewer releases "in Florida"; nothing stays."""
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
        assert words_of["Florida"].state == UNMASKED_BY_REVIEWER
        assert (words_of["like"].state, words_of["like"].kind_cut) == (UNMASKED_BY_TRIM, True)
        assert plan.final == []

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

    def test_a_lower_case_name_keeps_its_mask(
        self, store: ProvStore, redact_config: TriageConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No proper noun under the mask: the recognizer may have lower-cased the name, so nothing is cut."""
        words = ["my", "friend", "joan", "didion", "called"]
        _seed_redact_store(
            store, tmp_path, words=words, findings=[("PERSON", (_word_extent(2)[0], _word_extent(3)[1]))]
        )
        _stub_pii(monkeypatch, findings=[])
        redact(store, "recording", redact_config, run_dir=tmp_path, artifacts_dir=_release(tmp_path))
        plan = _plan(store, applies=False)
        assert _states(plan) == {"joan": MASKED, "didion": MASKED}
        assert not any(word.kind_cut for mask in plan.masks for word in mask.words)

    def test_a_flagged_reading_with_no_entries_is_recorded_and_moves_no_mask(
        self, store: ProvStore, redact_config: TriageConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A judgment without words: REDACT's masks stand and the ledger says why."""
        _seed_redact_store(store, tmp_path, words=["hello", "alice"], findings=[("PERSON", _word_extent(1))])
        _stub_pii(monkeypatch, findings=[])
        redact(store, "recording", redact_config, run_dir=tmp_path, artifacts_dir=_release(tmp_path))
        _annotate(store, [], original="clean", redaction="incomplete")
        plan = _plan(store)
        assert plan.named_no_words is True
        assert _states(plan) == {"alice": MASKED}
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
        (["last", "year"], set()),
        (["the", "week"], set()),
        (["March", "3rd"], set()),
        (["last", "Tuesday"], set()),
        (["Tuesday", "morning"], set()),
        (["93", "years", "old"], set()),
        (["yesterday", "evening"], set()),
        (["two", "years", "in", "Boston"], {0, 1}),
    ],
)
def test_time_by_kind_releases_times_of_day_and_durations_only(texts: list[str], released: set[int]) -> None:
    """A time of day or a quantified length of time is released; a date, a weekday or an age is not."""
    assert time_by_kind(texts) == released
