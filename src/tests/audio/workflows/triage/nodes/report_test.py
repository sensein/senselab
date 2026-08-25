"""REPORT: both products on every file and every outcome, no elements written, no matched text."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pytest
import soundfile as sf

from senselab.audio.workflows.triage.config import TriageConfig, load_triage_config
from senselab.audio.workflows.triage.nodes.common import software_agent, write_verdict
from senselab.audio.workflows.triage.nodes.report import report
from senselab.audio.workflows.triage.nodes.verdict import verdict as fold_verdict
from senselab.audio.workflows.triage.vocabulary import Outcome
from senselab.utils.prov_store import ProvStore

_SHA = "0" * 39 + "1"
_DURATION_S = 6.0
_RATE = 16000


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


def _seed_report_store(  # noqa: C901 — one independent block per node, as the graph itself has
    store: ProvStore,
    tmp_path: Path,
    *,
    full: bool = False,
    admit_failed: bool = False,
    words: Sequence[str] = ("the", "quick", "brown", "fox"),
    marked_words: Sequence[tuple[str, str]] = (),
    airway_labelled: Sequence[tuple[float, float]] = ((1.0, 1.3),),
    airway_unlabelled: Sequence[tuple[float, float]] = ((2.0, 2.3),),
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
    """
    config = load_triage_config()
    software = software_agent(store)
    activity = store.activity(node="ADMIT", step=None, parameters={})
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
    for classifier, grid, label in (("yamnet", 0.96, "Speech"), ("ast", 2.0, "Cough"), ("hear", 2.0, "Breathe")):
        start = 0.0
        while start < _DURATION_S:
            _entity(
                "measurement",
                (start, min(start + grid, _DURATION_S)),
                {
                    "name": f"{classifier}_window",
                    "classifier": classifier,
                    "signal": "plain",
                    "labels": [label],
                    "scores": {label: 0.9},
                },
            )
            start += grid
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
        detail={"absent": {}, "derivatives": {}},
    )

    fold = store.activity(node="TAXONOMY", step="fold", parameters={})
    store.was_associated_with(fold, software)
    for kind, state in (("speech", "present"), ("airway", "present"), ("voice", "present")):
        _entity("kind", None, {"kind": kind, "state": state, "lines": {}, "stream": "plain"})
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
        label_id = _entity(
            "assertion",
            (extent[0], extent[1]),
            {"verb": "label", "label": "Cough", "hear_window_ids": [], "merged_proposals": 1},
        )
        store.was_derived_from(label_id, span_ids[tuple(extent)])
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
        word_id = _entity("word", extent, {"text": text, "confidence": 0.9, "recognizers": [], "index": index})
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
        assert artifacts["summary"].suffix == ".png"
        assert artifacts["json"].exists()

    def test_an_unknown_format_refuses(self, store: ProvStore, tmp_path: Path) -> None:
        """A typo must not fall through to a silent default."""
        _seed_report_store(store, tmp_path, full=True)
        with pytest.raises(ValueError, match="report.format"):
            report(store, tmp_path / "summary", _override(tmp_path, "report:\n  format: jpeg\n"))

    def test_pdf_is_reachable_by_config(self, store: ProvStore, tmp_path: Path) -> None:
        """The two forms carry the same claims; the choice does not change the content."""
        _seed_report_store(store, tmp_path, full=True)
        pdf_config = load_triage_config(_write(tmp_path, "report:\n  format: pdf\n"))
        artifacts = report(store, tmp_path / "summary", pdf_config)
        assert artifacts["summary"].suffix == ".pdf"

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


class TestItRespectsThePiiMarking:
    """No matched text appears anywhere in either product."""

    def test_a_marked_word_is_rendered_redacted_in_the_json(self, store: ProvStore, tmp_path: Path) -> None:
        """The store holds PII by design; every artifact must respect the marking."""
        _seed_report_store(store, tmp_path, full=True, marked_words=[("alice", "PERSON")])
        artifacts = report(store, tmp_path / "summary", _png(tmp_path))
        text = artifacts["json"].read_text()
        assert "alice" not in text
        assert "[PERSON]" in text

    def test_an_unmarked_word_is_rendered_verbatim(self, store: ProvStore, tmp_path: Path) -> None:
        """The marking is what redacts, not a blanket refusal to render words."""
        _seed_report_store(store, tmp_path, full=True, words=["hello"], marked_words=[])
        artifacts = report(store, tmp_path / "summary", _png(tmp_path))
        assert "hello" in artifacts["json"].read_text()

    def test_the_marked_word_never_reaches_the_drawn_lanes(
        self, store: ProvStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The summary is redacted by the same rule as the JSON, not only the JSON."""
        panels = _capture_panels(monkeypatch)
        _seed_report_store(store, tmp_path, full=True, marked_words=[("alice", "PERSON")])
        report(store, tmp_path / "summary", _png(tmp_path))
        drawn = json.dumps(panels[0], default=str)
        assert "alice" not in drawn
        assert "[PERSON]" in drawn


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
        """Waveform, envelope with floor, spans, phonation spans, the three label lanes, the branches."""
        panels = _capture_panels(monkeypatch)
        _seed_report_store(store, tmp_path, full=True)
        report(store, tmp_path / "summary", _png(tmp_path))
        kinds = [panel["type"] for panel in panels[0]]
        assert kinds.count("segments") >= 5
        assert "waveform" in kinds and "features" in kinds

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

    def test_labelled_and_unlabelled_spans_are_distinguishable(
        self, store: ProvStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """branch-airway.md requires it on the shared axis."""
        panels = _capture_panels(monkeypatch)
        _seed_report_store(store, tmp_path, full=True, airway_labelled=[(1.0, 1.3)], airway_unlabelled=[(2.0, 2.3)])
        report(store, tmp_path / "summary", _png(tmp_path))
        labels = {
            segment["label"] for panel in panels[0] if panel["type"] == "segments" for segment in panel["segments"]
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
