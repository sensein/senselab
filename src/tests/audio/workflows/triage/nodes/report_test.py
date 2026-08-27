"""REPORT: both products on every file and every outcome, no elements written, no matched text."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pytest
import soundfile as sf

from senselab.audio.data_structures import Audio, AudioHints
from senselab.audio.workflows.triage.config import TriageConfig, load_triage_config
from senselab.audio.workflows.triage.nodes.common import software_agent, write_verdict
from senselab.audio.workflows.triage.nodes.report import ReportRenderError, report
from senselab.audio.workflows.triage.nodes.verdict import verdict as fold_verdict
from senselab.audio.workflows.triage.vocabulary import Outcome
from senselab.utils.prov_store import ProvStore

_SHA = "0" * 39 + "1"
_DURATION_S = 6.0
_RATE = 16000
_UNLABELLED_LABEL = "unlabelled"


def _write(tmp_path: Path, body: str) -> Path:
    """A partial config override on disk."""
    path = tmp_path / "override.yaml"
    path.write_text(body)
    return path


def _override(tmp_path: Path, body: str) -> TriageConfig:
    """The packaged config with one partial override merged over it."""
    return load_triage_config(_write(tmp_path, body))


def _png(tmp_path: Path) -> TriageConfig:
    """The packaged config with the report format declared as an image."""
    path = tmp_path / "report.yaml"
    path.write_text("report:\n  format: png\n")
    return load_triage_config(path)


def _capture_panels(monkeypatch: pytest.MonkeyPatch) -> list[list[dict[str, Any]]]:
    """Record the panel specifications every render is handed, without changing what it draws."""
    from senselab.audio.workflows.triage.nodes import report as report_module

    captured: list[list[dict[str, Any]]] = []
    real = report_module.plot_aligned_panels

    def _spy(audio: Any, panels: list[dict[str, Any]], **kwargs: Any) -> Any:  # noqa: ANN401
        captured.append([dict(panel) for panel in panels])
        return real(audio, panels, **kwargs)

    monkeypatch.setattr(report_module, "plot_aligned_panels", _spy)
    return captured


def _capture_titles(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record the title every aligned render is handed, without changing what it draws."""
    from senselab.audio.workflows.triage.nodes import report as report_module

    captured: list[str] = []
    real = report_module.plot_aligned_panels

    def _spy(audio: Any, panels: list[dict[str, Any]], **kwargs: Any) -> Any:  # noqa: ANN401
        captured.append(str(kwargs.get("title") or ""))
        return real(audio, panels, **kwargs)

    monkeypatch.setattr(report_module, "plot_aligned_panels", _spy)
    return captured


def _capture_headers(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, str]]:
    """Record the decision header supplied to the time-aligned page."""
    from senselab.audio.workflows.triage.nodes import report as report_module

    captured: list[dict[str, str]] = []
    real = report_module.plot_aligned_panels

    def _spy(audio: Any, panels: list[dict[str, Any]], **kwargs: Any) -> Any:  # noqa: ANN401
        captured.append({str(key): str(value) for key, value in (kwargs.get("header") or {}).items()})
        return real(audio, panels, **kwargs)

    monkeypatch.setattr(report_module, "plot_aligned_panels", _spy)
    return captured


def _stub_the_drawing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the aligned-panel render with a bare figure, to time the store-reading path alone."""
    from matplotlib import pyplot

    from senselab.audio.workflows.triage.nodes import report as report_module

    monkeypatch.setattr(report_module, "plot_aligned_panels", lambda audio, panels, **kw: pyplot.figure())


def _seed_report_store(  # noqa: C901, D417 — one independent block per node, as the graph itself has
    store: ProvStore,
    tmp_path: Path,
    *,
    full: bool = False,
    admit_failed: bool = False,
    words: Sequence[str] = ("the", "quick", "brown", "fox"),
    marked_words: Sequence[tuple[str, str]] = (),
    airway_labelled: Sequence[tuple[float, float]] = ((1.0, 1.3),),
    airway_unlabelled: Sequence[tuple[float, float]] = ((2.0, 2.3),),
    absent: dict[str, str] | None = None,
    classifiers: Sequence[tuple[str, float, str]] | None = None,
    yamnet_labels: Sequence[str] | None = None,
    skip_voice: bool = False,
    foreign_span_label: str | None = None,
    scan: str = "complete",
) -> None:
    """Write what a completed graph would have left behind, so REPORT has a store to read.

    Args:
        store: The store to seed.
        tmp_path: Where the plain stream and the envelope sidecar are written.
        full: Whether to seed every node's output. False seeds ADMIT and nothing else.
        admit_failed: Whether ADMIT refused the file, which is the outcome REPORT must still speak to.
        words: The consensus words, rendered verbatim unless a mark covers them.
        marked_words: ``(text, category)`` words the PII scan marked, appended to ``words``.
        airway_labelled: Extents AIRWAY put a label of interest on.
        airway_unlabelled: Extents AIRWAY looked at and did not label.
        absent: What PREPROCESS records as uncomputed, ``{derivative: ExceptionClass}``.
    classifiers: ``(name, grid_s, label)`` per window classifier; ``()`` writes none at all,
            which is the state a null threshold leaves behind. None writes the three defaults.
        yamnet_labels: One extra label per YAMNet window, for a category list long enough to truncate.
        skip_voice: Whether VOICE writes an activity and no verdict — the store's own reading of a
            node that raised.
        foreign_span_label: A ``label`` assertion over an envelope span generated by a node other
            than AIRWAY, which must not reach the airway lane.
        scan: ``"complete"`` writes SPEECH's ``pii_scan`` with every detector attempted,
            ``"incomplete"`` writes one with a failed detector, and ``"absent"`` writes none —
            which is what routing declining SPEECH leaves behind.
    """
    config = load_triage_config()
    software = software_agent(store)
    activity = store.activity(
        node="ADMIT", step=None, parameters={"audio_file": str(tmp_path / ("refused.wav" if admit_failed else "r.wav"))}
    )
    store.was_associated_with(activity, software)

    def _entity(prov_type: str, extent: tuple[float, float] | None, attributes: dict[str, Any]) -> str:
        """One seeded entity, generated and attributed like the node that would have written it."""
        entity_id = store.entity(prov_type=prov_type, extent=extent, attributes=attributes)  # type: ignore[arg-type]
        store.was_generated_by(entity_id, activity)
        store.was_attributed_to(entity_id, software)
        return entity_id

    if admit_failed:
        write_verdict(
            store, activity, software, node="ADMIT", outcome=Outcome.FAIL, kind=None, why="decode failure", detail={}
        )
        fold_verdict(store, None, config, None, run_dir=tmp_path)
        return

    (tmp_path / "streams").mkdir(parents=True, exist_ok=True)
    (tmp_path / "derivatives").mkdir(parents=True, exist_ok=True)
    samples = np.linspace(-0.4, 0.4, int(_DURATION_S * _RATE), dtype=np.float32)
    sf.write(str(tmp_path / "streams" / "plain.wav"), samples, _RATE)
    for name in ("recording", "plain"):
        _entity(
            "stream",
            (0.0, _DURATION_S),
            {"name": name, "path": "streams/plain.wav", "sampling_rate": _RATE, "channels": 1},
        )

    write_verdict(store, activity, software, node="ADMIT", outcome=Outcome.PASS, kind=None, why="it decodes", detail={})
    if not full:
        fold_verdict(store, None, config, None, run_dir=tmp_path)
        return

    store.agent(agent_type="model", model_id="openai/whisper-large-v3", commit_sha=_SHA)
    store.agent(agent_type="model", model_id="https://tfhub.dev/google/yamnet/1", unresolved_reason="TF-Hub URL pin")

    envelope = np.linspace(-60.0, -10.0, int(_DURATION_S * _RATE), dtype=np.float64)
    np.savez(tmp_path / "derivatives" / "energy_envelope.npz", envelope_dbfs=envelope, floor_dbfs=envelope - 20.0)
    preprocess = store.activity(node="PREPROCESS", step="envelope", parameters={})
    store.was_associated_with(preprocess, software)
    _entity(
        "measurement",
        None,
        {
            "name": "energy_envelope",
            "signal": "preemphasised",
            "path": "derivatives/energy_envelope.npz",
            "sampling_rate": _RATE,
        },
    )
    envelope_spans = [*airway_labelled, *airway_unlabelled]
    span_ids = {
        tuple(extent): _entity(
            "span",
            (extent[0], extent[1]),
            {"peak_over_floor_db": 24.0 + index, "k_db": 18.0, "signal": "preemphasised", "merged_proposals": 1},
        )
        for index, extent in enumerate(envelope_spans)
    }
    _entity(
        "span",
        (3.0, 3.8),
        {
            "family": "phonation",
            "member": "sustained",
            "duration_s": 0.8,
            "production": "voiced",
            "voiced_fraction": 1.0,
            "offset_criterion": "f0_stability",
            "signal": "preemphasised",
            "hop_s": 0.01,
        },
    )
    declared = (
        (("yamnet", 0.96, "Speech"), ("ast", 10.24, "Cough"), ("hear", 2.0, "Breathe"))
        if classifiers is None
        else tuple(classifiers)
    )
    for classifier, grid, label in declared:
        start = 0.0
        index = 0
        while start < _DURATION_S:
            extra = [yamnet_labels[index % len(yamnet_labels)]] if classifier == "yamnet" and yamnet_labels else []
            _entity(
                "measurement",
                (start, min(start + grid, _DURATION_S)),
                {
                    "name": f"{classifier}_window",
                    "classifier": classifier,
                    "signal": "plain",
                    "labels": [label, *extra],
                    "scores": {name: 0.9 for name in (label, *extra)},
                },
            )
            start += grid
            index += 1
        _entity(
            "measurement",
            None,
            {
                "name": f"{classifier}_windows",
                "classifier": classifier,
                "signal": "plain",
                "labels": [label],
                "windows_by_label": {label: ["seeded"]},
                "n_windows": 1,
                "win_length_s": grid,
                "hop_s": grid,
            },
        )
    write_verdict(
        store,
        preprocess,
        software,
        node="PREPROCESS",
        outcome=Outcome.PASS,
        kind=None,
        why="conditioning complete",
        detail={"absent": dict(absent or {}), "derivatives": {}},
    )

    fold = store.activity(node="TAXONOMY", step="fold", parameters={})
    store.was_associated_with(fold, software)
    kind_lines = {
        "speech": {
            "acoustic": {"state": "present", "evidence": 1, "unit": "windows", "floor": 1},
            "lexical": {"state": "present", "evidence": len(words), "unit": "words", "floor": 1},
        },
        "airway": {
            "health_acoustic": {"state": "present", "evidence": 1, "unit": "windows", "floor": 1},
            "acoustic": {"state": "present", "evidence": 1, "unit": "windows", "floor": 1},
        },
        "voice": {"phonation": {"state": "present", "evidence": 0.8, "unit": "seconds", "floor": 0.5}},
    }
    for kind, state in (("speech", "present"), ("airway", "present"), ("voice", "present")):
        _entity("kind", None, {"kind": kind, "state": state, "lines": kind_lines[kind], "stream": "plain"})
    write_verdict(
        store,
        fold,
        software,
        node="TAXONOMY",
        outcome=Outcome.PASS,
        kind=None,
        why="every kind is present",
        detail={"kinds": {"speech": "present", "airway": "present", "voice": "present"}},
    )

    route = store.activity(node="routing", step=None, parameters={})
    store.was_associated_with(route, software)
    for branch, kind in (("AIRWAY", "airway"), ("SPEECH", "speech"), ("VOICE", "voice")):
        _entity(
            "branch_decision",
            None,
            {
                "branch": branch,
                "kind": kind,
                "will_run": True,
                "kind_state": "present",
                "raw_state": "present",
                "forced_by_hint": False,
                "hint_tags": [],
                "unmapped_tags": [],
                "bad_map_values": {},
                "why": "kind_present",
                "stream": "plain",
            },
        )
    write_verdict(
        store,
        route,
        software,
        node="routing",
        outcome=Outcome.PASS,
        kind=None,
        why="runs: AIRWAY, SPEECH, VOICE",
        detail={"runs": ["AIRWAY", "SPEECH", "VOICE"], "skipped": [], "forced": [], "empty_set": False},
    )

    classify = store.activity(node="AIRWAY", step="classify", parameters={})
    store.was_associated_with(classify, software)
    for extent in airway_labelled:
        label_id = store.entity(
            prov_type="assertion",
            extent=(extent[0], extent[1]),
            attributes={"verb": "label", "label": "Cough", "hear_window_ids": [], "merged_proposals": 1},
        )
        store.was_generated_by(label_id, classify)
        store.was_attributed_to(label_id, software)
        store.was_derived_from(label_id, span_ids[tuple(extent)])
    if foreign_span_label is not None and airway_unlabelled:
        elsewhere = store.activity(node="TAXONOMY", step="label", parameters={})
        store.was_associated_with(elsewhere, software)
        target = span_ids[tuple(airway_unlabelled[0])]
        foreign = store.entity(
            prov_type="assertion",
            extent=(airway_unlabelled[0][0], airway_unlabelled[0][1]),
            attributes={"verb": "label", "label": foreign_span_label},
        )
        store.was_generated_by(foreign, elsewhere)
        store.was_attributed_to(foreign, software)
        store.was_derived_from(foreign, target)
    write_verdict(
        store,
        classify,
        software,
        node="AIRWAY",
        outcome=Outcome.PASS,
        kind="airway",
        why="a span carries a label of interest",
        detail={"labelled_n": len(airway_labelled), "by_label": {"Cough": len(airway_labelled)}, "flags": []},
    )

    consensus = store.activity(node="PREPROCESS", step="consensus", parameters={})
    store.was_associated_with(consensus, software)
    every_word = [(text, None) for text in words] + [(text, category) for text, category in marked_words]
    word_ids: list[str] = []
    for index, (text, category) in enumerate(every_word):
        extent = (0.2 * index, 0.2 * index + 0.15)
        word_id = _entity(
            "word",
            extent,
            {
                "text": text,
                "confidence": 0.9,
                "existence_confidence": 0.88,
                "temporal_confidence": 0.86,
                "coverage": 1.0,
                "recognizers": ["crisperwhisper", "qwen"],
                "timing_sources": ["native", "bundled_aligner"],
                "index": index,
            },
        )
        word_ids.append(word_id)
        if category is not None:
            mark_id = _entity("assertion", extent, {"verb": "label", "label": "pii", "category": category})
            store.was_derived_from(mark_id, word_id)
    _entity(
        "measurement",
        None,
        {
            "name": "consensus_transcript",
            "signal": "plain",
            "words": [{"text": text} for text, _ in every_word],
            "word_ids": word_ids,
            "event_ids": [],
            "text": " ".join(text for text, _ in every_word),
        },
    )

    speech = store.activity(node="SPEECH", step="corroborate", parameters={})
    store.was_associated_with(speech, software)
    _entity(
        "span",
        (0.0, 1.0),
        {"family": "speech", "words_n": len(every_word), "attributed_to": "SPEAKER_00", "nontarget": False},
    )
    for _, category in marked_words:
        _entity(
            "pii",
            (0.0, 1.0),
            {"category": category, "source": "gliner", "occurrence": 0, "occurrences_n": 1, "recognizers": []},
        )
    if scan != "absent":
        _entity(
            "measurement",
            None,
            {
                "name": "pii_scan",
                "signal": "consensus_transcript",
                "scanned_by": ["gliner", "presidio", "rules"],
                "failed": ["presidio"] if scan == "incomplete" else [],
                "missing": [],
            },
        )
    write_verdict(
        store,
        speech,
        software,
        node="SPEECH",
        outcome=Outcome.PASS,
        kind="speech",
        why="words, spans, speakers and quality are in the store",
        detail={
            "speaker_count": 1,
            "words_n": len(every_word),
            "pii": {"categories": sorted({category for _, category in marked_words}), "n": len(marked_words)},
            "flags": [],
        },
    )

    voice = store.activity(node="VOICE", step=None, parameters={})
    store.was_associated_with(voice, software)
    if skip_voice:
        plan_only = store.activity(node="REDACT", step="plan", parameters={})
        store.was_associated_with(plan_only, software)
        fold_verdict(store, None, config, None, run_dir=tmp_path)
        return
    _entity(
        "span",
        (3.0, 3.8),
        {
            "family": "phonation",
            "member": "sustained",
            "production": "voiced",
            "duration_s": 0.8,
            "onset_kind": "period",
            "offset_kind": "criterion",
            "offset_criterion": "f0_stability",
            "marks_n": 12,
        },
    )
    write_verdict(
        store,
        voice,
        software,
        node="VOICE",
        outcome=Outcome.PASS,
        kind="voice",
        why="phonation spans measured; nothing contested",
        detail={"spans_n": 1, "phonation_s": 0.8, "gate_interval": "unmeasured", "flags": []},
    )

    plan = store.activity(node="REDACT", step="plan", parameters={})
    store.was_associated_with(plan, software)
    for _, category in marked_words:
        _entity("span", (0.0, 1.0), {"name": "redaction", "category": category})
    write_verdict(
        store,
        plan,
        software,
        node="REDACT",
        outcome=Outcome.PASS if marked_words else Outcome.FLAG,
        kind=None,
        why="every finding redacted" if marked_words else "no finding to redact",
        detail={"redactions_n": len(marked_words), "by_category": {}, "artifacts_withheld": not marked_words},
    )
    fold_verdict(store, None, config, None, run_dir=tmp_path)


class TestBothProductsAlways:
    """One summary and one JSON per file, whatever the graph concluded."""

    def test_a_full_run_emits_both(self, store: ProvStore, tmp_path: Path) -> None:
        """The ordinary path."""
        _seed_report_store(store, tmp_path, full=True)
        artifacts = report(store, tmp_path / "summary", _png(tmp_path))
        assert artifacts["summary"].exists() and artifacts["summary"].suffix == ".png"
        assert artifacts["json"].exists()

    def test_an_admit_refusal_emits_both_and_says_nothing_was_measured(self, store: ProvStore, tmp_path: Path) -> None:
        """A file ADMIT refused gets a report that says that, not an exception (V24)."""
        _seed_report_store(store, tmp_path, admit_failed=True)
        artifacts = report(store, tmp_path / "summary", _png(tmp_path))
        payload = json.loads(artifacts["json"].read_text())
        assert payload["verdict"]["triage"] == "discard"
        assert payload["branches"] == {}
        assert artifacts["summary"].exists()

    def test_a_store_holding_nothing_at_all_still_emits_both(self, store: ProvStore, tmp_path: Path) -> None:
        """Not even ADMIT concluded, so there is no verdict to read; the products are owed anyway."""
        artifacts = report(store, tmp_path / "summary", _png(tmp_path))
        payload = json.loads(artifacts["json"].read_text())
        assert payload["verdict"]["triage"] is None
        assert artifacts["summary"].exists()

    def test_the_packaged_config_emits_a_report(self, store: ProvStore, config: TriageConfig, tmp_path: Path) -> None:
        """The one unconditional product must be reachable with no override at all (I4)."""
        _seed_report_store(store, tmp_path, full=True)
        artifacts = report(store, tmp_path / "summary", config)
        assert artifacts["summary"].suffix == ".pdf"
        assert artifacts["json"].exists()

    def test_an_unknown_format_refuses(self, store: ProvStore, tmp_path: Path) -> None:
        """A typo must not fall through to a silent default."""
        _seed_report_store(store, tmp_path, full=True)
        with pytest.raises(ValueError, match="report.format"):
            report(store, tmp_path / "summary", _override(tmp_path, "report:\n  format: jpeg\n"))

    def test_png_is_reachable_by_config(self, store: ProvStore, tmp_path: Path) -> None:
        """The two forms carry the same claims; the choice does not change the content."""
        _seed_report_store(store, tmp_path, full=True)
        artifacts = report(store, tmp_path / "summary", _png(tmp_path))
        assert artifacts["summary"].suffix == ".png"

    def test_the_two_forms_carry_the_same_json(self, store: ProvStore, tmp_path: Path) -> None:
        """``report.format`` is a presentation choice and changes no claim."""
        _seed_report_store(store, tmp_path, full=True)
        as_png = json.loads(report(store, tmp_path / "png", _png(tmp_path))["json"].read_text())
        pdf_config = load_triage_config(_write(tmp_path, "report:\n  format: pdf\n"))
        as_pdf = json.loads(report(store, tmp_path / "pdf", pdf_config)["json"].read_text())
        assert as_png["steps"] == as_pdf["steps"]
        assert as_png["verdict"] == as_pdf["verdict"]
        assert as_png["transcript"] == as_pdf["transcript"]


class TestItWritesNoElements:
    """A rendering is not evidence."""

    def test_the_store_is_unchanged(self, store: ProvStore, tmp_path: Path) -> None:
        """No entity, no activity, no agent, no relation."""
        _seed_report_store(store, tmp_path, full=True)
        before = store.fingerprint()
        report(store, tmp_path / "summary", _png(tmp_path))
        assert store.fingerprint() == before


class TestTheStructuredJsonCompanion:
    """The JSON is the same report model as the PDF, with a stable machine interface."""

    def test_pdf_and_versioned_json_agree_on_the_file_decision(self, store: ProvStore, tmp_path: Path) -> None:
        """Both sibling artifacts exist and the decision fields are one source of truth."""
        _seed_report_store(store, tmp_path, full=True, marked_words=[("alice", "PERSON")])
        pdf_config = load_triage_config(_write(tmp_path, "report:\n  format: pdf\n"))
        artifacts = report(store, tmp_path / "summary", pdf_config)
        payload = json.loads(artifacts["json"].read_text())
        assert artifacts["summary"].exists() and artifacts["json"].exists()
        assert payload["schema_version"] == "triage-summary/v2"
        assert payload["decisions"]["file_triage"] == payload["verdict"]["triage"]
        assert payload["decisions"]["release"] == payload["verdict"]["release"]
        assert payload["artifacts"]["summary"]["path"] == artifacts["summary"].name
        assert payload["artifacts"]["json"]["path"] == artifacts["json"].name

    def test_json_exposes_context_routing_and_timed_evidence(self, store: ProvStore, tmp_path: Path) -> None:
        """A consumer can audit a branch without parsing PDF prose."""
        _seed_report_store(store, tmp_path, full=True)
        payload = json.loads(report(store, tmp_path / "summary", _png(tmp_path))["json"].read_text())
        assert {"recording", "screening", "routing", "evidence"} <= set(payload)
        assert payload["recording"]["run_label"]
        assert payload["routing"]["SPEECH"]["verdict"] is not None
        speech_items = payload["evidence"]["branches"]["SPEECH"]
        assert any(item["timing"] and item["provenance"]["node"] for item in speech_items)
        token = payload["evidence"]["consensus_transcript_tokens"][0]
        assert token["timing_authority"] == "consensus"
        assert token["confidence"] == 0.9
        assert token["existence_confidence"] == 0.88 and token["temporal_confidence"] == 0.86
        assert token["coverage"] == 1.0
        assert token["recognizers"] == ["crisperwhisper", "qwen"]
        assert token["timing_sources"] == ["native", "bundled_aligner"]
        assert all("entity_id" in item for item in payload["evidence"]["consensus_transcript_tokens"])
        assert all("entity_id" in item for item in payload["evidence"]["redacted_transcript_tokens"])


class TestItSeparatesConsensusAndRedactedTranscript:
    """The audit transcript and the redacted release representation stay visibly distinct."""

    def test_a_marked_word_is_preserved_in_consensus_and_replaced_in_redacted_json(
        self, store: ProvStore, tmp_path: Path
    ) -> None:
        """A placeholder is a REDACT result, never a claim about what ASR emitted."""
        _seed_report_store(store, tmp_path, full=True, marked_words=[("alice", "PERSON")])
        artifacts = report(store, tmp_path / "summary", _png(tmp_path))
        payload = json.loads(artifacts["json"].read_text())
        assert payload["transcript"]["text"].endswith("alice")
        assert payload["transcript"]["redacted_text"].endswith("[PERSON]")
        assert payload["evidence"]["consensus_transcript_tokens"][-1]["text"] == "alice"
        assert payload["evidence"]["redacted_transcript_tokens"][-1]["text"] == "[PERSON]"

    def test_an_unmarked_transcript_does_not_draw_a_duplicate_redacted_lane(
        self, store: ProvStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A second identical token lane would add clutter without communicating a redaction."""
        panels = _capture_panels(monkeypatch)
        _seed_report_store(store, tmp_path, full=True, words=["hello"], marked_words=[])
        report(store, tmp_path / "summary", _png(tmp_path))
        assert not any(panel.get("report_lane") == "redacted" for panel in panels[0])

    def test_the_marked_word_uses_distinct_consensus_and_redacted_lanes(
        self, store: ProvStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Readers can distinguish a PII replacement from the word ASR and consensus supplied."""
        panels = _capture_panels(monkeypatch)
        _seed_report_store(store, tmp_path, full=True, marked_words=[("alice", "PERSON")])
        report(store, tmp_path / "summary", _png(tmp_path))
        consensus = next(panel for panel in panels[0] if panel.get("report_lane") == "words")
        redacted = next(panel for panel in panels[0] if panel.get("report_lane") == "redacted")
        assert [token["text"] for token in consensus["tokens"]][-1] == "alice"
        assert [token["text"] for token in redacted["tokens"]][-1] == "[PERSON]"


class TestTheProvenanceIsEmbedded:
    """A hash identifies a run; the mapping is what makes it readable without the repository."""

    def test_the_config_hash_and_the_mapping_both_appear(self, store: ProvStore, tmp_path: Path) -> None:
        """Both, always."""
        _seed_report_store(store, tmp_path, full=True)
        payload = json.loads(report(store, tmp_path / "summary", _png(tmp_path))["json"].read_text())
        assert payload["provenance"]["config_hash"]
        assert payload["provenance"]["config"]["name"] == "senselab-triage/default"

    def test_every_model_carries_its_resolved_commit_or_a_reason(self, store: ProvStore, tmp_path: Path) -> None:
        """An agent whose commit could not be resolved appears with its reason, never with a bare ref."""
        _seed_report_store(store, tmp_path, full=True)
        payload = json.loads(report(store, tmp_path / "summary", _png(tmp_path))["json"].read_text())
        assert payload["provenance"]["models"]
        for model in payload["provenance"]["models"]:
            assert model["revision"] is not None or model["unresolved_reason"] is not None
            assert model["revision"] != "main"

    def test_the_senselab_commit_is_a_sha_or_a_reason(self, store: ProvStore, tmp_path: Path) -> None:
        """The run's own commit obeys the rule its models do."""
        _seed_report_store(store, tmp_path, full=True)
        payload = json.loads(report(store, tmp_path / "summary", _png(tmp_path))["json"].read_text())
        provenance = payload["provenance"]
        assert provenance["commit"] is not None or provenance["commit_unresolved_reason"] is not None
        assert provenance["commit"] != "main"

    def test_every_step_names_the_elements_behind_it(self, store: ProvStore, tmp_path: Path) -> None:
        """This is what makes the JSON a view of the store rather than a second copy of it."""
        _seed_report_store(store, tmp_path, full=True)
        payload = json.loads(report(store, tmp_path / "summary", _png(tmp_path))["json"].read_text())
        assert payload["steps"]
        for entry in payload["steps"].values():
            assert isinstance(entry["element_ids"], list)
            assert entry["element_ids"]

    def test_the_named_elements_are_in_the_store(self, store: ProvStore, tmp_path: Path) -> None:
        """An id that joins back to nothing is not a join key."""
        _seed_report_store(store, tmp_path, full=True)
        payload = json.loads(report(store, tmp_path / "summary", _png(tmp_path))["json"].read_text())
        named = [element for entry in payload["steps"].values() for element in entry["element_ids"]]
        assert named
        for element in named:
            assert store.get_entity(element).id == element


class TestTheSummaryLayers:
    """One shared time axis, drawn from the store."""

    def test_the_shared_axis_carries_every_layer_the_store_holds(
        self, store: ProvStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Waveform, envelope with floor, spans, phonation, probability rasters and branch evidence."""
        panels = _capture_panels(monkeypatch)
        _seed_report_store(store, tmp_path, full=True)
        report(store, tmp_path / "summary", _png(tmp_path))
        kinds = [panel["type"] for panel in panels[0]]
        assert kinds.count("segments") >= 4
        assert kinds.count("score_raster") == 2
        assert "waveform" in kinds
        assert panels[0][0]["twin"]["data"]

    def test_the_header_leads_with_decisions_and_routing(
        self, store: ProvStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A reader sees the file decision before inspecting the supporting lanes."""
        headers = _capture_headers(monkeypatch)
        _seed_report_store(store, tmp_path, full=True)
        report(store, tmp_path / "summary", _png(tmp_path))
        header = headers[0]
        assert header["decision_label"] == "PRIMARY FILE DECISION"
        assert "TRIAGE:" in header["decision"] and "RELEASE:" in header["decision"]
        assert header["evidence_label"] == "LEADING DECISION EVIDENCE"
        assert header["context_label"].endswith("(context only)")
        assert header["support_label"].endswith("(report-only summary)")

    def test_the_spectrogram_and_the_blocks_are_both_drawn(
        self, store: ProvStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A reviewer judging the output needs the time-frequency picture and the judgment beside it."""
        panels = _capture_panels(monkeypatch)
        _seed_report_store(store, tmp_path, full=True)
        report(store, tmp_path / "summary", _png(tmp_path))
        kinds = [panel["type"] for panel in panels[0]]
        assert "spectrogram" in kinds
        assert kinds[-1] == "text"

    def test_the_blocks_carry_the_decision_the_transcript_and_the_categories(
        self, store: ProvStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The content a reviewer judges the run by, on the page rather than only in the JSON."""
        panels = _capture_panels(monkeypatch)
        _seed_report_store(store, tmp_path, full=True, words=["hello", "world"])
        report(store, tmp_path / "summary", _png(tmp_path))
        blocks = "\n".join(panels[0][-1]["lines"])
        assert "triage:" in blocks and "release:" in blocks
        assert "hello world" in blocks
        assert "Speech" in blocks
        assert "AIRWAY" in blocks

    def test_coarse_ast_labels_are_summarized_instead_of_drawn_as_a_timeline(
        self, store: ProvStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """One 10.24-second AST context is a coarse label summary, not a time-local event lane."""
        panels = _capture_panels(monkeypatch)
        _seed_report_store(store, tmp_path, full=True)
        artifacts = report(store, tmp_path / "summary", _png(tmp_path))
        assert "ast labels" not in {panel.get("name") for panel in panels[0]}
        payload = json.loads(artifacts["json"].read_text())
        assert payload["evidence"]["label_presentations"]["ast"] == {
            "mode": "summary_only",
            "hop_s": 10.24,
            "window_length_s": 10.24,
            "reason": "coarse_window_hop",
        }
        blocks = "\n".join(panels[0][-1]["lines"])
        assert "ast: summary only (10.24 s window, 10.24 s hop) Cough (1)" in blocks

    def test_ast_with_a_dense_historical_grid_remains_a_timeline(
        self, store: ProvStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The presentation follows the stored grid, preserving honest rendering of old runs."""
        panels = _capture_panels(monkeypatch)
        _seed_report_store(
            store,
            tmp_path,
            full=True,
            classifiers=(("yamnet", 0.96, "Speech"), ("ast", 2.0, "Cough"), ("hear", 2.0, "Breathe")),
        )
        artifacts = report(store, tmp_path / "summary", _png(tmp_path))
        assert "ast labels" in {panel.get("name") for panel in panels[0]}
        payload = json.loads(artifacts["json"].read_text())
        assert payload["evidence"]["label_presentations"]["ast"]["mode"] == "timeline"

    def test_classifier_rasters_and_json_keep_labels_paired_with_probabilities(
        self, store: ProvStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Comma-separated classes alone discard the magnitude that made each one relevant."""
        panels = _capture_panels(monkeypatch)
        _seed_report_store(store, tmp_path, full=True)

        payload = json.loads(report(store, tmp_path / "summary", _png(tmp_path))["json"].read_text())

        raster = next(panel for panel in panels[0] if panel.get("name") == "yamnet labels")
        assert raster["type"] == "score_raster"
        assert raster["rows"] == ["Speech"]
        assert raster["windows"][0]["scores"] == {"Speech": 0.9}
        assert payload["evidence"]["classifier_windows"]["yamnet"][0]["label_scores"] == {"Speech": 0.9}

    def test_labelled_and_unlabelled_spans_are_distinguishable(
        self, store: ProvStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """branch-airway.md requires it on the shared axis."""
        panels = _capture_panels(monkeypatch)
        _seed_report_store(store, tmp_path, full=True, airway_labelled=[(1.0, 1.3)], airway_unlabelled=[(2.0, 2.3)])
        report(store, tmp_path / "summary", _png(tmp_path))
        labels = {
            segment["label"] for panel in panels[0] if panel.get("name") == "airway" for segment in panel["segments"]
        }
        assert "unlabelled" in labels and "Cough" in labels

    def test_a_layer_the_store_does_not_hold_is_simply_absent(
        self, store: ProvStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Nothing raises for want of a derivative; a report is owed on every outcome."""
        panels = _capture_panels(monkeypatch)
        _seed_report_store(store, tmp_path)
        report(store, tmp_path / "summary", _png(tmp_path))
        kinds = [panel["type"] for panel in panels[0]]
        assert "waveform" in kinds
        assert "features" not in kinds


class TestItReadsTheStoreOnceRatherThanPerWord:
    """The index behind the PII rule and the airway labels is built once, not per element."""

    def test_the_assertion_index_is_built_once(
        self, store: ProvStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Two readers ask the same reverse question; one pass over the assertions answers both."""
        from senselab.audio.workflows.triage.nodes import report as report_module

        calls: list[int] = []
        real = report_module._assertions_by_source

        def _counted(store_: ProvStore) -> dict[str, list[Any]]:
            calls.append(1)
            return real(store_)

        monkeypatch.setattr(report_module, "_assertions_by_source", _counted)
        _seed_report_store(store, tmp_path, full=True, marked_words=[("alice", "PERSON")])
        report(store, tmp_path / "summary", _png(tmp_path))
        assert len(calls) == 1

    def test_a_long_transcript_renders_in_bounded_time(
        self, store: ProvStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """800 words and 800 markings; per-word rescanning made this quadratic and then some."""
        _stub_the_drawing(monkeypatch)
        _seed_report_store(
            store,
            tmp_path,
            full=True,
            words=[],
            marked_words=[(f"name{index}", "PERSON") for index in range(800)],
        )
        started = time.monotonic()
        artifacts = report(store, tmp_path / "summary", _png(tmp_path))
        elapsed = time.monotonic() - started
        assert json.loads(artifacts["json"].read_text())["transcript"]["words_n"] == 800
        assert elapsed < 15.0, f"reading the store took {elapsed:.1f}s; the per-word rescan is back"


class TestARenderFailureKeepsTheJson:
    """The JSON is complete before the picture is drawn, and must not go down with it."""

    def test_the_json_survives_and_the_error_names_the_cause(
        self, store: ProvStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A consumer reading many files loses nothing because one page could not be drawn."""
        from senselab.audio.workflows.triage.nodes import report as report_module

        def _raise(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
            raise RuntimeError("the canvas is gone")

        monkeypatch.setattr(report_module, "_render", _raise)
        _seed_report_store(store, tmp_path, full=True)
        with pytest.raises(ReportRenderError) as caught:
            report(store, tmp_path / "summary", _png(tmp_path))
        artifacts = caught.value.artifacts
        assert sorted(artifacts) == ["json"]
        assert json.loads(artifacts["json"].read_text())["verdict"]["triage"]
        assert "the canvas is gone" in str(caught.value)


class TestTheAbsentLanesAreOnThePage:
    """A lane not drawn must never read as a measured absence."""

    def test_an_unfitted_derivative_is_named_apart_from_an_errored_one(
        self, store: ProvStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A null config key and a crash are different facts about the run."""
        panels = _capture_panels(monkeypatch)
        _seed_report_store(
            store,
            tmp_path,
            full=True,
            absent={"phonation_spans": "ValueError", "gammatone": "AttributeError", "silence": "LookupError"},
        )
        report(store, tmp_path / "summary", _png(tmp_path))
        blocks = "\n".join(panels[0][-1]["lines"])
        assert "ABSENT" in blocks
        assert "unfitted (a config key it reads is null): phonation_spans [ValueError]" in blocks
        assert "unavailable (a derivative it reads is absent): silence [LookupError]" in blocks
        assert "errored: gammatone [AttributeError]" in blocks

    def test_the_recorded_message_is_rendered_beside_the_reading(
        self, store: ProvStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """PREPROCESS records which key was null; a page that showed only the class would drop it."""
        panels = _capture_panels(monkeypatch)
        _seed_report_store(
            store,
            tmp_path,
            full=True,
            absent={"phonation_spans": "ValueError: phonation_spans.silence_floor_db is null"},
        )
        report(store, tmp_path / "summary", _png(tmp_path))
        blocks = "\n".join(panels[0][-1]["lines"])
        assert "unfitted (a config key it reads is null)" in blocks
        assert "phonation_spans [ValueError: phonation_spans.silence_floor_db is null]" in blocks

    def test_a_lane_absence_carries_the_message_too(
        self, store: ProvStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The lane line is where a reader looks first, so the attribution must reach it."""
        panels = _capture_panels(monkeypatch)
        _seed_report_store(
            store,
            tmp_path,
            full=True,
            absent={"yamnet_windows": "ValueError: yamnet.window_s is null"},
            classifiers=(),
        )
        report(store, tmp_path / "summary", _png(tmp_path))
        blocks = "\n".join(panels[0][-1]["lines"])
        assert "lane not drawn — yamnet labels: PREPROCESS/yamnet_windows unfitted" in blocks
        assert "yamnet.window_s is null" in blocks


class TestTheHintReadingIsOnThePage:
    """``verdict.hints`` reached the JSON and not the page, so a reader saw the flag and not its cause."""

    def test_an_unclaimed_kind_reads_found_unclaimed(
        self, store: ProvStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With nothing declared, every kind the graph found is found and unclaimed, and says so."""
        panels = _capture_panels(monkeypatch)
        _seed_report_store(store, tmp_path, full=True)
        report(store, tmp_path / "summary", _png(tmp_path))
        blocks = "\n".join(panels[0][-1]["lines"])
        assert "airway: screened=present resolved=present agreement=agree hint=found_unclaimed" in blocks

    def test_a_declared_kind_no_branch_found_reads_claimed_not_found(
        self, store: ProvStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The reason line names the mismatch; the kind line must agree with it rather than omit it."""
        panels = _capture_panels(monkeypatch)
        _seed_report_store(store, tmp_path, full=True)
        software = software_agent(store)
        supersede = store.activity(node="routing", step="declare", parameters={})
        store.was_associated_with(supersede, software)
        decision = store.entity(
            prov_type="branch_decision",
            extent=None,
            attributes={
                "branch": "AIRWAY",
                "kind": "airway",
                "will_run": True,
                "kind_state": "present",
                "raw_state": "present",
                "forced_by_hint": False,
                "hint_tags": ["cough"],
                "unmapped_tags": [],
                "bad_map_values": {},
                "why": "kind_present",
                "stream": "plain",
            },
        )
        store.was_generated_by(decision, supersede)
        refuted = store.activity(node="AIRWAY", step="reclassify", parameters={})
        store.was_associated_with(refuted, software)
        write_verdict(
            store,
            refuted,
            software,
            node="AIRWAY",
            outcome=Outcome.FAIL,
            kind="airway",
            why="spans exist but none carries a label of interest",
            detail={},
        )
        fold_verdict(store, None, load_triage_config(), AudioHints(may_contain=["cough"]), run_dir=tmp_path)

        report(store, tmp_path / "summary", _png(tmp_path))
        blocks = "\n".join(panels[0][-1]["lines"])
        assert "airway: screened=present resolved=absent agreement=mismatch hint=claimed_not_found" in blocks

    def test_a_lane_the_page_did_not_draw_is_named_with_its_reason(
        self, store: ProvStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Under the packaged config the label lanes are missing; the page must say why."""
        panels = _capture_panels(monkeypatch)
        _seed_report_store(store, tmp_path, full=True, absent={"yamnet_windows": "ValueError"}, classifiers=())
        report(store, tmp_path / "summary", _png(tmp_path))
        blocks = "\n".join(panels[0][-1]["lines"])
        assert "lane not drawn — yamnet labels: PREPROCESS/yamnet_windows unfitted" in blocks

    def test_a_page_that_drew_every_lane_says_so(
        self, store: ProvStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Silence about absences is indistinguishable from having none."""
        panels = _capture_panels(monkeypatch)
        _seed_report_store(store, tmp_path, full=True, marked_words=[("alice", "PERSON")])
        report(store, tmp_path / "summary", _png(tmp_path))
        blocks = "\n".join(panels[0][-1]["lines"])
        assert "every declared lane was drawn" in blocks
        assert "PREPROCESS reports no absent derivative" in blocks


class TestThePageCarriesItsOwnProvenanceAndRunState:
    """A reviewer judging one page must not have to open the JSON to know where it came from."""

    def test_the_provenance_line_names_the_config_the_commit_and_the_models(
        self, store: ProvStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The full model list stays in the JSON; the identity belongs on the page."""
        panels = _capture_panels(monkeypatch)
        config = _png(tmp_path)
        _seed_report_store(store, tmp_path, full=True)
        report(store, tmp_path / "summary", config)
        blocks = "\n".join(panels[0][-1]["lines"])
        assert f"config {config.config_hash}" in blocks
        assert "senselab " in blocks
        assert "models: 1 at a resolved commit, 1 with a reason" in blocks

    def test_the_ran_line_tells_an_errored_node_from_a_silent_one(
        self, store: ProvStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The distinction the whole graph is built to preserve must reach the page."""
        panels = _capture_panels(monkeypatch)
        _seed_report_store(store, tmp_path, full=True, skip_voice=True)
        report(store, tmp_path / "summary", _png(tmp_path))
        blocks = "\n".join(panels[0][-1]["lines"])
        assert "ran: ADMIT:completed" in blocks
        assert "VOICE:errored" in blocks

    def test_the_branch_block_carries_its_measurements(
        self, store: ProvStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A branch conclusion without its numbers cannot be judged, only believed."""
        panels = _capture_panels(monkeypatch)
        _seed_report_store(store, tmp_path, full=True, marked_words=[("alice", "PERSON")])
        report(store, tmp_path / "summary", _png(tmp_path))
        blocks = "\n".join(panels[0][-1]["lines"])
        assert "speaker_count=1" in blocks
        assert "phonation_s=0.8" in blocks and "0.79999" not in blocks
        assert "words_n=5" in blocks
        assert "labelled_n=1" in blocks
        assert "redactions_n=1" in blocks

    def test_a_truncated_category_list_says_how_many_it_left_out(
        self, store: ProvStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Six of six and six of forty must not render identically."""
        panels = _capture_panels(monkeypatch)
        _seed_report_store(store, tmp_path, full=True, yamnet_labels=[f"L{index}" for index in range(9)])
        report(store, tmp_path / "summary", _png(tmp_path))
        blocks = "\n".join(panels[0][-1]["lines"])
        assert "yamnet (top 6 of " in blocks
        assert "top 6 of 6" not in blocks
        assert "ast:" in blocks


class TestTheRefusalPage:
    """The file ADMIT refused is the one that most needs a legible page."""

    def test_it_names_the_file_and_shows_unknowns_as_a_dash(self, store: ProvStore, tmp_path: Path) -> None:
        """A refusal page that cannot name the file, or that prints "None", is of no use."""
        _seed_report_store(store, tmp_path, admit_failed=True)
        artifacts = report(store, tmp_path / "summary", _png(tmp_path))
        payload = json.loads(artifacts["json"].read_text())
        assert payload["file"]["path"] is not None and payload["file"]["path"].endswith("refused.wav")
        assert payload["file"]["duration_s"] is None

    def test_the_page_itself_carries_the_name_the_dash_and_the_reason(
        self, store: ProvStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Everything the reviewer needs, with no Python repr leaking through."""
        drawn: list[list[str]] = []
        from senselab.audio.workflows.triage.nodes import report as report_module

        real = report_module._blocks

        def _spy(document: dict[str, Any], lanes: set[str]) -> list[str]:
            lines = real(document, lanes)
            drawn.append(lines)
            return lines

        monkeypatch.setattr(report_module, "_blocks", _spy)
        _seed_report_store(store, tmp_path, admit_failed=True)
        report(store, tmp_path / "summary", _png(tmp_path))
        blocks = "\n".join(drawn[0])
        assert "refused.wav" in blocks
        assert "duration: — s" in blocks
        assert "decode failure" in blocks
        assert ": None" not in blocks


class TestOnlyAirwayLabelsTheAirwayLane:
    """A label assertion is read as AIRWAY's only when AIRWAY generated it."""

    def test_a_label_from_another_node_does_not_reach_the_lane(
        self, store: ProvStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The lane answers "what did AIRWAY conclude", not "what has anyone said over this span"."""
        panels = _capture_panels(monkeypatch)
        _seed_report_store(store, tmp_path, full=True, foreign_span_label="Applause")
        report(store, tmp_path / "summary", _png(tmp_path))
        labels = {
            segment["label"] for panel in panels[0] if panel.get("name") == "airway" for segment in panel["segments"]
        }
        assert "Applause" not in labels
        assert "Cough" in labels and _UNLABELLED_LABEL in labels


class TestElementIdsNameLiveEvidenceOnly:
    """A join key to a withdrawn element credits a claim to evidence that no longer stands."""

    def test_an_invalidated_entity_is_not_cited(self, store: ProvStore, tmp_path: Path) -> None:
        """The store's shared read rule applies to the report's citations too."""
        _seed_report_store(store, tmp_path, full=True)
        withdrawn = [entity for entity in store.entities("span") if entity.attributes.get("family") == "speech"][0]
        store.was_invalidated_by(withdrawn.id, store.activities("SPEECH")[0].id)
        payload = json.loads(report(store, tmp_path / "summary", _png(tmp_path))["json"].read_text())
        cited = [element for entry in payload["steps"].values() for element in entry["element_ids"]]
        assert withdrawn.id not in cited


class TestAnUnscannedTranscriptIsNotACleanOne:
    """The marking is what redacts, so no marking is only reassuring if something looked."""

    def test_a_transcript_nobody_scanned_keeps_consensus_and_marks_the_redacted_view(
        self, store: ProvStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A missing PII scan changes only the redacted representation, never consensus ASR evidence."""
        panels = _capture_panels(monkeypatch)
        _seed_report_store(store, tmp_path, full=True, words=["hello", "world"], scan="absent")
        artifacts = report(store, tmp_path / "summary", _png(tmp_path))
        payload = json.loads(artifacts["json"].read_text())
        assert payload["transcript"]["text"] == "hello world"
        assert {token["text"] for token in payload["evidence"]["redacted_transcript_tokens"]} == {"[unscanned]"}
        blocks = "\n".join(panels[0][-1]["lines"])
        assert "CONSENSUS TRANSCRIPT" in blocks
        assert "no pii scan is in the store" in blocks

    def test_an_incomplete_scan_marks_the_redacted_view_and_names_the_detector(
        self, store: ProvStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A scan that lost a detector leaves consensus intact and marks the redacted form as unscanned."""
        panels = _capture_panels(monkeypatch)
        _seed_report_store(store, tmp_path, full=True, words=["hello"], scan="incomplete")
        payload = json.loads(report(store, tmp_path / "summary", _png(tmp_path))["json"].read_text())
        assert payload["transcript"]["text"] == "hello"
        assert payload["transcript"]["redacted_text"] == "[unscanned]"
        blocks = "\n".join(panels[0][-1]["lines"])
        assert "detectors failed: presidio" in blocks

    def test_a_complete_scan_renders_the_words_it_cleared(self, store: ProvStore, tmp_path: Path) -> None:
        """Withholding everything unconditionally would make the marking pointless."""
        _seed_report_store(store, tmp_path, full=True, words=["hello"], marked_words=[("alice", "PERSON")])
        text = report(store, tmp_path / "summary", _png(tmp_path))["json"].read_text()
        assert "hello" in text
        assert "[PERSON]" in text
        assert "alice" in text
        assert "[unscanned]" not in text


class TestTheTitleIsShort:
    """The title carries the decision at a glance; the run id is provenance and belongs in the block."""

    _RUN_ID = "sub-1f4ea26f_ses-D987B8B0_task-Respiration-and-cough-(v2)-Breath_20260825-123640"

    def test_it_names_the_task_token_the_date_and_the_decision(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The four things a reader needs before anything else, and nothing else."""
        titles = _capture_titles(monkeypatch)
        store = ProvStore(run_id=self._RUN_ID)
        _seed_report_store(store, tmp_path, full=True)
        report(store, tmp_path / "summary", _png(tmp_path))
        assert titles[0].startswith("task-Respiration-and-cough-(v2)-Breath · 2026-08-25")
        assert "triage:" in titles[0] and "release:" in titles[0]

    def test_the_full_run_id_is_not_in_the_title(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """An unwrapped 70-character id across the top of the page is the objection this answers."""
        titles = _capture_titles(monkeypatch)
        store = ProvStore(run_id=self._RUN_ID)
        _seed_report_store(store, tmp_path, full=True)
        report(store, tmp_path / "summary", _png(tmp_path))
        assert self._RUN_ID not in titles[0]

    def test_the_full_run_id_and_the_path_are_in_the_block(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Dropping the id from the title must not drop it from the page."""
        panels = _capture_panels(monkeypatch)
        store = ProvStore(run_id=self._RUN_ID)
        _seed_report_store(store, tmp_path, full=True)
        report(store, tmp_path / "summary", _png(tmp_path))
        blocks = "\n".join(panels[0][-1]["lines"])
        assert self._RUN_ID in blocks
        assert "file: streams/plain.wav" in blocks

    def test_a_run_id_with_no_task_token_still_titles(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Not every corpus is BIDS-named; the title falls back to the stem rather than to nothing."""
        titles = _capture_titles(monkeypatch)
        store = ProvStore(run_id="recording-42_20260825-123640")
        _seed_report_store(store, tmp_path, full=True)
        report(store, tmp_path / "summary", _png(tmp_path))
        assert titles[0].startswith("recording-42 · 2026-08-25")

    def test_a_long_line_is_wrapped_rather_than_run_off_the_page(self, tmp_path: Path) -> None:
        """A path wider than the block is text nobody can read; every block line is wrapped."""
        from senselab.audio.workflows.triage.nodes.report import _BLOCK_COLUMNS, _wrapped

        wrapped = _wrapped(["  file: " + "x" * 400])
        assert len(wrapped) > 1
        assert max(len(line) for line in wrapped) <= _BLOCK_COLUMNS

    def test_wrapping_keeps_the_blank_lines_that_separate_the_blocks(self) -> None:
        """An empty string wraps to nothing, and the blocks would run together if the pass let it."""
        from senselab.audio.workflows.triage.nodes.report import _wrapped

        assert _wrapped(["BRANCHES", "", "TAXONOMY"]) == ["BRANCHES", "", "TAXONOMY"]


def _pdf_pages(path: Path) -> int:
    """How many pages a rendered PDF carries, counted from its own page objects."""
    return len(re.findall(rb"/Type\s*/Page[^s]", path.read_bytes()))


def _pdf_letter_landscape_pages(path: Path) -> int:
    """How many PDF pages declare the US Letter landscape media box, in points."""
    return len(re.findall(rb"/MediaBox\s*\[\s*0\s+0\s+792\s+612\s*\]", path.read_bytes()))


class TestThePdfPagination:
    """The PDF keeps evidence legible, then gives its decision prose a dedicated Letter page."""

    def test_the_packaged_format_is_the_pdf(self, config: TriageConfig) -> None:
        """The form the packaged config ships is the paginated document, not the tall image."""
        assert config.require("report.format") == "pdf"

    def test_a_full_run_renders_exactly_two_pages(self, store: ProvStore, tmp_path: Path) -> None:
        """Page one is the aligned panels; page two is the concise decision record."""
        _seed_report_store(store, tmp_path, full=True)
        pdf_config = load_triage_config(_write(tmp_path, "report:\n  format: pdf\n"))
        artifacts = report(store, tmp_path / "summary", pdf_config)
        assert artifacts["summary"].suffix == ".pdf"
        assert _pdf_pages(artifacts["summary"]) == 2
        assert _pdf_letter_landscape_pages(artifacts["summary"]) == 2

    def test_a_refusal_renders_exactly_two_pages_too(self, store: ProvStore, tmp_path: Path) -> None:
        """A file with no readable stream has no axis to draw, and still owes both pages."""
        _seed_report_store(store, tmp_path, admit_failed=True)
        pdf_config = load_triage_config(_write(tmp_path, "report:\n  format: pdf\n"))
        artifacts = report(store, tmp_path / "summary", pdf_config)
        assert _pdf_pages(artifacts["summary"]) == 2
        assert _pdf_letter_landscape_pages(artifacts["summary"]) == 2

    def test_a_long_recording_gets_one_ten_second_evidence_page_per_interval(
        self, store: ProvStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Long timelines are paginated rather than compressed into unreadable labels."""
        from matplotlib import pyplot

        from senselab.audio.workflows.triage.nodes import report as report_module

        windows: list[tuple[float, float]] = []

        def _plot(*_args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
            windows.append(tuple(kwargs["time_limits"]))
            return pyplot.figure()

        _seed_report_store(store, tmp_path, full=True)
        long_audio = Audio(waveform=np.zeros((1, 25 * _RATE), dtype=np.float32), sampling_rate=_RATE)
        monkeypatch.setattr(report_module, "_stream", lambda *_args: long_audio)
        monkeypatch.setattr(report_module, "plot_aligned_panels", _plot)
        pdf_config = load_triage_config(_write(tmp_path, "report:\n  format: pdf\n"))

        artifacts = report(store, tmp_path / "summary", pdf_config)

        assert windows == [(0.0, 10.0), (10.0, 20.0), (20.0, 25.0)]
        assert _pdf_pages(artifacts["summary"]) == 4

    def test_the_first_page_carries_no_text_block(
        self, store: ProvStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The blocks are page two; leaving them on page one is what made the image 32 inches tall."""
        panels = _capture_panels(monkeypatch)
        _seed_report_store(store, tmp_path, full=True)
        pdf_config = load_triage_config(_write(tmp_path, "report:\n  format: pdf\n"))
        report(store, tmp_path / "summary", pdf_config)
        assert "text" not in [panel["type"] for panel in panels[0]]

    def test_the_second_page_carries_every_block(
        self, store: ProvStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The decision page contains the reader-facing findings while JSON retains the full audit."""
        from senselab.audio.workflows.triage.nodes import report as report_module

        drawn: list[list[str]] = []
        real = report_module._text_figure

        def _spy(lines: list[str], title: str, **kwargs: Any) -> Any:  # noqa: ANN401
            drawn.append(list(lines))
            return real(lines, title, **kwargs)

        monkeypatch.setattr(report_module, "_text_figure", _spy)
        _seed_report_store(store, tmp_path, full=True, words=["hello", "world"])
        pdf_config = load_triage_config(_write(tmp_path, "report:\n  format: pdf\n"))
        report(store, tmp_path / "summary", pdf_config)
        blocks = "\n".join(drawn[-1])
        assert "DECISION SUMMARY" in blocks and "SCREENING AND ROUTING" in blocks and "hello world" in blocks
        assert "TAXONOMY DECISION PATH" in blocks and "lexical consensus decides" in blocks
        assert "ANALYTIC RECORD" in blocks and "summary.json" in blocks

    def test_the_png_stays_one_image_with_the_blocks_on_it(
        self, store: ProvStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The image form is unchanged: one uncut canvas, blocks included."""
        panels = _capture_panels(monkeypatch)
        _seed_report_store(store, tmp_path, full=True)
        artifacts = report(store, tmp_path / "summary", _png(tmp_path))
        assert artifacts["summary"].suffix == ".png"
        assert panels[0][-1]["type"] == "text"


class TestTheWaveformAndTheEnvelopeShareOneRow:
    """Two scales over one signal, not two rows a reader has to register by eye."""

    def test_the_envelope_has_no_row_of_its_own(
        self, store: ProvStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The separate features row is gone; nothing else moved."""
        panels = _capture_panels(monkeypatch)
        _seed_report_store(store, tmp_path, full=True)
        report(store, tmp_path / "summary", _png(tmp_path))
        assert "features" not in [panel["type"] for panel in panels[0]]
        assert [panel for panel in panels[0] if panel.get("name") == "envelope"] == []

    def test_the_waveform_row_carries_the_envelope_and_its_floor(
        self, store: ProvStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Both curves reach the twin axis, and the axis says what its scale is."""
        panels = _capture_panels(monkeypatch)
        _seed_report_store(store, tmp_path, full=True)
        report(store, tmp_path / "summary", _png(tmp_path))
        twin = panels[0][0]["twin"]
        assert panels[0][0]["type"] == "waveform"
        assert [curve[2] for curve in twin["data"]] == ["envelope dBFS", "floor dBFS"]
        assert "dBFS" in twin["name"]

    def test_a_store_with_no_envelope_leaves_the_waveform_bare(
        self, store: ProvStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An absent derivative is an absent curve, not a twin axis with nothing on it."""
        panels = _capture_panels(monkeypatch)
        _seed_report_store(store, tmp_path)
        report(store, tmp_path / "summary", _png(tmp_path))
        assert panels[0][0]["type"] == "waveform"
        assert "twin" not in panels[0][0]

    def test_the_envelope_lane_still_reads_as_drawn(
        self, store: ProvStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Sharing a row is not being absent; the ABSENT block must not claim it was skipped."""
        panels = _capture_panels(monkeypatch)
        _seed_report_store(store, tmp_path, full=True)
        report(store, tmp_path / "summary", _png(tmp_path))
        blocks = "\n".join(panels[0][-1]["lines"])
        assert "lane not drawn — envelope" not in blocks


class TestTheSpansAreAnOverlayNotALane:
    """A span is a stretch of the waveform; it belongs over the waveform, not in a row beneath it."""

    def test_the_spans_row_is_gone(self, store: ProvStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """One row fewer, and the one it merged into is the one it was measured from."""
        panels = _capture_panels(monkeypatch)
        _seed_report_store(store, tmp_path, full=True)
        report(store, tmp_path / "summary", _png(tmp_path))
        assert [panel for panel in panels[0] if panel.get("name") == "spans (dB over floor)"] == []

    def test_the_waveform_row_carries_the_spans_and_their_decibels(
        self, store: ProvStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The dB over floor is the number the span exists to report; it must survive the move."""
        panels = _capture_panels(monkeypatch)
        _seed_report_store(store, tmp_path, full=True)
        report(store, tmp_path / "summary", _png(tmp_path))
        overlay = panels[0][0]["spans"]
        assert overlay["name"] == "spans (dB over floor)"
        assert overlay["segments"]
        assert all(segment["label"].endswith(" dB") for segment in overlay["segments"])

    def test_the_drawn_figure_keeps_the_right_hand_scale_compact(self, store: ProvStore, tmp_path: Path) -> None:
        """Span labels stay on their spans; the shared scale needs only its unit."""
        from senselab.audio.workflows.triage.nodes import report as report_module

        drawn: list[Any] = []
        real = report_module.plot_aligned_panels

        def _spy(audio: Any, panels: list[dict[str, Any]], **kwargs: Any) -> Any:  # noqa: ANN401
            figure = real(audio, panels, **kwargs)
            drawn.append(figure)
            return figure

        with pytest.MonkeyPatch.context() as patched:
            patched.setattr(report_module, "plot_aligned_panels", _spy)
            _seed_report_store(store, tmp_path, full=True)
            report(store, tmp_path / "summary", _png(tmp_path))
        labels = [axis.get_ylabel() for axis in drawn[0].axes]
        assert "dBFS" in labels

    def test_the_spans_lane_still_reads_as_drawn(
        self, store: ProvStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Becoming an overlay is not becoming absent."""
        panels = _capture_panels(monkeypatch)
        _seed_report_store(store, tmp_path, full=True, marked_words=[("alice", "PERSON")])
        report(store, tmp_path / "summary", _png(tmp_path))
        blocks = "\n".join(panels[0][-1]["lines"])
        assert "lane not drawn — spans (dB over floor)" not in blocks
        assert "every declared lane was drawn" in blocks


class TestTheEnvelopePanelsScaleIsTheSignals:
    """The dB panel's y-scale must come from measured values, never from a clamp's own value."""

    @staticmethod
    def _undershooting_envelope(tmp_path: Path) -> np.ndarray:
        """PREPROCESS's own envelope over a burst with a sharp offset, written to the run's sidecar.

        The zero-phase lowpass undershoots at the offset, which is where a clamped floor used to be
        fabricated. This is the real producer feeding the real panel.
        """
        from senselab.audio.data_structures import Audio
        from senselab.audio.tasks.envelope import hilbert_envelope_dbfs, rolling_floor_dbfs

        grid = np.arange(int(_DURATION_S * _RATE)) / _RATE
        samples = np.zeros_like(grid)
        voiced = (grid >= 1.0) & (grid < 3.0)
        samples[voiced] = 0.6 * np.sin(2 * np.pi * 220.0 * grid[voiced])
        audio = Audio(waveform=samples.astype(np.float32)[None, :], sampling_rate=_RATE)
        envelope = hilbert_envelope_dbfs(audio, lowpass_hz=40.0, filter_order=4)
        floor = rolling_floor_dbfs(envelope, _RATE, window_s=1.0, percentile=10.0, eval_grid_s=0.1)
        np.savez(tmp_path / "derivatives" / "energy_envelope.npz", envelope_dbfs=envelope, floor_dbfs=floor)
        return envelope

    def test_the_twin_axis_stays_inside_the_measured_range(self, store: ProvStore, tmp_path: Path) -> None:
        """15 fabricated -240 dBFS spikes stretched the axis to -250 and squashed -50..-90 flat."""
        from senselab.audio.workflows.triage.nodes import report as report_module

        drawn: list[Any] = []
        real = report_module.plot_aligned_panels

        def _spy(audio: Any, panels: list[dict[str, Any]], **kwargs: Any) -> Any:  # noqa: ANN401
            figure = real(audio, panels, **kwargs)
            drawn.append(figure)
            return figure

        with pytest.MonkeyPatch.context() as patched:
            patched.setattr(report_module, "plot_aligned_panels", _spy)
            _seed_report_store(store, tmp_path, full=True)
            envelope = self._undershooting_envelope(tmp_path)
            report(store, tmp_path / "summary", _png(tmp_path))
        twin = [axis for axis in drawn[0].axes if "dBFS" in axis.get_ylabel()]
        assert twin, "the envelope's twin axis must be on the figure"
        low, high = twin[0].get_ylim()
        assert high < 20.0, "a 0.6-amplitude tone reads near -4 dBFS"
        assert low > -120.0, f"the axis floor {low:.0f} dBFS is below anything this signal can measure"
        assert high - low < 150.0, "the informative band must not be squashed into a corner"
        assert envelope.size == int(_DURATION_S * _RATE)

    def test_no_curve_carries_the_clamps_value(self, store: ProvStore, tmp_path: Path) -> None:
        """Whatever the axis does, -240 dBFS must not reach the panel as a datum."""
        panels: list[list[dict[str, Any]]] = []
        with pytest.MonkeyPatch.context() as patched:
            captured = _capture_panels(patched)
            _seed_report_store(store, tmp_path, full=True)
            self._undershooting_envelope(tmp_path)
            report(store, tmp_path / "summary", _png(tmp_path))
            panels = captured
        values = np.concatenate([np.asarray(curve[1], dtype=float) for curve in panels[0][0]["twin"]["data"]])
        assert not np.any(values <= -240.0)


class TestTheWordsLaneFollowsTheConsensusStyle:
    """A word's text belongs on its own bar, as the analyze_audio consensus row draws it."""

    @staticmethod
    def _render(store: ProvStore, tmp_path: Path, **seed: Any) -> Any:  # noqa: ANN401
        """Seed the store, draw the real figure, and hand back the figure and the panels."""
        from senselab.audio.workflows.triage.nodes import report as report_module

        drawn: list[Any] = []
        captured: list[list[dict[str, Any]]] = []
        real = report_module.plot_aligned_panels

        def _spy(audio: Any, panels: list[dict[str, Any]], **kwargs: Any) -> Any:  # noqa: ANN401
            captured.append([dict(panel) for panel in panels])
            figure = real(audio, panels, **kwargs)
            drawn.append(figure)
            return figure

        with pytest.MonkeyPatch.context() as patched:
            patched.setattr(report_module, "plot_aligned_panels", _spy)
            _seed_report_store(store, tmp_path, full=True, **seed)
            report(store, tmp_path / "summary", _png(tmp_path))
        return drawn[0], captured[0]

    @staticmethod
    def _words_axis(figure: Any, panels: list[dict[str, Any]]) -> Any:  # noqa: ANN401
        """The axis the words lane was drawn on, found by the lane's position in the stack."""
        index = [position for position, panel in enumerate(panels) if panel.get("report_lane") == "words"]
        assert index, "the words lane must be on the page"
        return figure.axes[index[0]]

    def test_the_words_lane_is_a_token_lane(self, store: ProvStore, tmp_path: Path) -> None:
        """A generic segments lane turns each word's text into a y-tick label."""
        _, panels = self._render(store, tmp_path, words=["hello", "world"])
        lane = [panel for panel in panels if panel.get("report_lane") == "words"]
        assert [panel["type"] for panel in lane] == ["tokens"]
        assert lane[0]["name"] == "consensus ASR"
        assert [token["text"] for token in lane[0]["tokens"]] == ["hello", "world"]
        assert all(token["color"].startswith("#") for token in lane[0]["tokens"])

    def test_every_word_is_drawn_on_the_bar_and_in_a_small_cycling_lane(self, store: ProvStore, tmp_path: Path) -> None:
        """The reader sees timed artists; compact lane ids never become token labels."""
        figure, panels = self._render(store, tmp_path, words=["hello", "world"])
        axis = self._words_axis(figure, panels)
        assert [text.get_text() for text in axis.texts] == ["hello", "world"]
        assert [tick.get_text() for tick in axis.get_yticklabels()] == []

    def test_short_timed_consensus_words_remain_visible(self, store: ProvStore, tmp_path: Path) -> None:
        """The timing bar stays short, but its cycling row gives every normal word readable label space."""
        figure, panels = self._render(
            store,
            tmp_path,
            words=["one", "two", "three", "four"],
            marked_words=[("five", "PERSON")],
        )
        axis = self._words_axis(figure, panels)
        figure.canvas.draw()
        assert {text.get_text() for text in axis.texts if text.get_visible()} == {
            "one",
            "two",
            "three",
            "four",
            "five",
        }

    def test_forty_words_do_not_become_forty_ticks(self, store: ProvStore, tmp_path: Path) -> None:
        """The rendered failure: 40+ overlapping tick labels beside unlabelled coloured dashes."""
        figure, panels = self._render(store, tmp_path, words=[f"word{index}" for index in range(40)])
        axis = self._words_axis(figure, panels)
        assert 0 < len(axis.patches) < 40, "off-page words are not artists on this timeline"
        assert [tick.get_text() for tick in axis.get_yticklabels()] == []

    def test_a_marked_word_stays_in_consensus_and_is_replaced_in_the_redacted_lane(
        self, store: ProvStore, tmp_path: Path
    ) -> None:
        """The two lanes make clear that a replacement is not an ASR token."""
        figure, panels = self._render(store, tmp_path, words=["hello"], marked_words=[("alice", "PERSON")])
        drawn = [text.get_text() for text in self._words_axis(figure, panels).texts]
        redacted_index = next(index for index, panel in enumerate(panels) if panel.get("report_lane") == "redacted")
        redacted = [text.get_text() for text in figure.axes[redacted_index].texts]
        assert "alice" in drawn
        assert "[PERSON]" in redacted

    def test_an_unscanned_transcript_keeps_consensus_and_withholds_the_redacted_lane(
        self, store: ProvStore, tmp_path: Path
    ) -> None:
        """The ASR evidence remains visible while the release representation declares its uncertainty."""
        figure, panels = self._render(store, tmp_path, words=["hello", "world"], scan="absent")
        drawn = [text.get_text() for text in self._words_axis(figure, panels).texts]
        redacted_index = next(index for index, panel in enumerate(panels) if panel.get("report_lane") == "redacted")
        redacted = [text.get_text() for text in figure.axes[redacted_index].texts]
        assert drawn == ["hello", "world"]
        assert redacted == ["[unscanned]", "[unscanned]"]

    @staticmethod
    def _placed(axis: Any) -> list[Any]:  # noqa: ANN401
        """The labels the saved page actually drew, the fit having been decided against its renderer."""
        return [text for text in axis.texts if text.get_visible()]

    def test_a_marked_word_is_drawn_verbatim_only_in_the_consensus_lane(self, store: ProvStore, tmp_path: Path) -> None:
        """The fit decides legibility, while the lane decides whether the text is raw or redacted."""
        figure, panels = self._render(store, tmp_path, words=["hello"], marked_words=[("alice", "PERSON")])
        drawn = {text.get_text() for text in self._placed(self._words_axis(figure, panels))}
        assert drawn <= {"hello", "alice"}

    def test_an_unscanned_page_keeps_the_consensus_words(self, store: ProvStore, tmp_path: Path) -> None:
        """ASR evidence is not silently rewritten because the independent PII scan failed."""
        figure, panels = self._render(store, tmp_path, words=["hello", "world"], scan="absent")
        drawn = {text.get_text() for text in self._placed(self._words_axis(figure, panels))}
        assert drawn <= {"hello", "world"}

    def test_no_two_words_collide_on_the_rendered_page(self, store: ProvStore, tmp_path: Path) -> None:
        """The defect as the reader met it: adjacent labels ran into one another's glyphs."""
        prose = (
            "grandfather remembered everything about wandering along riverbanks collecting interesting pebbles".split()
        )
        figure, panels = self._render(store, tmp_path, words=[prose[index % len(prose)] for index in range(40)])
        axis = self._words_axis(figure, panels)
        renderer = figure.canvas.get_renderer()
        extents = [text.get_window_extent(renderer) for text in self._placed(axis)]
        assert extents, "the page must still carry words"
        collisions = [
            (one, two) for index, one in enumerate(extents) for two in extents[index + 1 :] if one.overlaps(two)
        ]
        assert collisions == []

    def test_words_cycle_through_three_inspectable_rows(
        self, store: ProvStore, tmp_path: Path
    ) -> None:
        """Row cycling makes dense consensus timing inspectable without a label pileup."""
        prose = (
            "grandfather remembered everything about wandering along riverbanks collecting interesting pebbles".split()
        )
        figure, panels = self._render(store, tmp_path, words=[prose[index % len(prose)] for index in range(40)])
        axis = self._words_axis(figure, panels)
        assert len({round(float(patch.get_y()), 6) for patch in axis.patches}) == 3
        assert self._placed(axis)

    def test_the_staggered_page_keeps_the_consensus_words(self, store: ProvStore, tmp_path: Path) -> None:
        """More rows widen label slots but do not turn consensus evidence into placeholders."""
        figure, panels = self._render(
            store,
            tmp_path,
            words=[f"word{index}" for index in range(40)],
            marked_words=[("alice", "PERSON"), ("bob", "PERSON")],
        )
        drawn = {text.get_text() for text in self._placed(self._words_axis(figure, panels))}
        assert drawn <= {f"word{index}" for index in range(40)} | {"alice", "bob"}
