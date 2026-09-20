"""Behavioural tests for the corpus fold over a tree of triage decisions.

What it counts is fixed by :meth:`FileVerdict.record`, so the tests build real folds and assert the
counts, rather than asserting against a hand-written decision shape that could drift from it.
"""

from __future__ import annotations

import json
from pathlib import Path

from senselab.audio.workflows.triage.corpus_report import (
    UNREAD,
    aggregate,
    decisions,
    render_markdown,
)
from senselab.audio.workflows.triage.vocabulary import FileVerdict, NodeVerdict, Outcome, Release, RunState, Triage


def _verdict(
    triage: Triage = Triage.PASS,
    family: str | None = "syllable",
    **kwargs: object,
) -> FileVerdict:
    """Build one fold, defaulting everything the tests do not vary."""
    return FileVerdict(triage=triage, release=Release.NOT_ASSESSED, declared_family=family, **kwargs)  # type: ignore[arg-type]


def _write_row(root: Path, stem: str, verdict: FileVerdict | None, **extra: object) -> Path:
    """Write one driver row carrying a decision, as the corpus driver does."""
    path = root / stem[:2] / f"{stem}.row.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"stem": stem, "decision": verdict.record() if verdict is not None else None, **extra}
    path.write_text(json.dumps(payload))
    return path


class TestReading:
    """A decision is read from whichever record the tree holds."""

    def test_reads_driver_rows_and_runner_logs_alike(self, tmp_path: Path) -> None:
        """Both names carry the same key, so a tree of either answers the same questions."""
        _write_row(tmp_path, "sub-a", _verdict())
        log = tmp_path / "sub-b" / "run.json"
        log.parent.mkdir(parents=True)
        log.write_text(json.dumps({"source": "/data/sub-b.wav", "decision": _verdict().record()}))
        read = list(decisions(tmp_path))
        assert sorted(name for name, _, _ in read) == ["sub-a", "sub-b"]
        assert all(decision is not None for _, decision, _ in read)

    def test_a_recording_with_both_a_row_and_a_log_is_counted_once(self, tmp_path: Path) -> None:
        """A run tree holds both; counting both doubles every total in the report."""
        _write_row(tmp_path, "sub-a", _verdict(), duration_s=12.0)
        log = tmp_path / "out" / "sub-a" / "run.json"
        log.parent.mkdir(parents=True)
        log.write_text(json.dumps({"source": "/data/sub-a.wav", "decision": _verdict().record()}))
        report = aggregate(decisions(tmp_path))
        assert report.files == 1
        assert report.triage == {"pass": 1}

    def test_the_row_is_preferred_over_the_log_for_the_same_recording(self, tmp_path: Path) -> None:
        """Only the row carries the driver's own fields, the header duration among them."""
        _write_row(tmp_path, "sub-a", _verdict(), duration_s=0.5)
        log = tmp_path / "out" / "sub-a" / "run.json"
        log.parent.mkdir(parents=True)
        log.write_text(json.dumps({"stem": "sub-a", "decision": _verdict().record()}))
        report = aggregate(decisions(tmp_path))
        assert report.durations == {"0-1s": 1}

    def test_a_row_without_a_decision_is_read_and_counted_unread(self, tmp_path: Path) -> None:
        """A run that raised before VERDICT folded is a file of the corpus, not a file skipped."""
        _write_row(tmp_path, "sub-a", None, ok=False, error="RuntimeError: boom")
        report = aggregate(decisions(tmp_path))
        assert report.files == 1
        assert report.unread == 1
        assert report.triage == {UNREAD: 1}

    def test_unparseable_records_do_not_stop_the_fold(self, tmp_path: Path) -> None:
        """One truncated row must not cost the other 62,577 their counts."""
        _write_row(tmp_path, "sub-a", _verdict())
        broken = tmp_path / "zz" / "sub-z.row.json"
        broken.parent.mkdir(parents=True)
        broken.write_text("{not json")
        report = aggregate(decisions(tmp_path))
        assert report.files == 2
        assert report.unread == 1
        assert report.triage["pass"] == 1


class TestCounting:
    """Every decision point of the fold reaches a count."""

    def test_counts_the_two_axes_and_the_discard_ground(self, tmp_path: Path) -> None:
        """A discard is countable by its ground, so an empty recording is not read as a broken one."""
        _write_row(tmp_path, "sub-a", _verdict(Triage.DISCARD, discard_ground="acoustically_empty"))
        _write_row(tmp_path, "sub-b", _verdict(Triage.DISCARD, discard_ground="unmeasurable"))
        _write_row(tmp_path, "sub-c", _verdict())
        report = aggregate(decisions(tmp_path))
        assert report.triage == {"discard": 2, "pass": 1}
        assert report.release == {"not_assessed": 3}
        assert report.discard_ground == {"acoustically_empty": 1, "unmeasurable": 1}

    def test_counts_conformance_per_node_and_per_declared_family(self, tmp_path: Path) -> None:
        """The corpus question is per family: a False on a family the branch does not target is not one."""
        _write_row(tmp_path, "sub-a", _verdict(conformance={"SPEECH": False}, family="syllable"))
        _write_row(tmp_path, "sub-b", _verdict(conformance={"SPEECH": True}, family="syllable"))
        _write_row(tmp_path, "sub-c", _verdict(conformance={"SPEECH": "UNDETERMINED"}, family="free"))
        report = aggregate(decisions(tmp_path))
        assert report.conformance["SPEECH"] == {"False": 1, "True": 1, "UNDETERMINED": 1}
        assert report.conformance_by_family["syllable"]["SPEECH"] == {"False": 1, "True": 1}
        assert report.conformance_by_family["free"]["SPEECH"] == {"UNDETERMINED": 1}

    def test_counts_deviation_types_and_unmeasured_paths_per_node(self, tmp_path: Path) -> None:
        """A config path nobody measured is a defect signal, and is counted as one."""
        _write_row(
            tmp_path,
            "sub-a",
            _verdict(deviations={"SPEECH": ["truncation", "rate"]}, unmeasured={"SPEECH": ["speech.response_min_s"]}),
        )
        _write_row(tmp_path, "sub-b", _verdict(deviations={"SPEECH": ["truncation"]}))
        report = aggregate(decisions(tmp_path))
        assert report.deviations["SPEECH"] == {"truncation": 2, "rate": 1}
        assert report.unmeasured["SPEECH"] == {"speech.response_min_s": 1}

    def test_counts_critical_absences_by_branch_and_gate(self, tmp_path: Path) -> None:
        """Any entry is a run that reached no branch at all, so it is named rather than left to be noticed."""
        absence = {"SPEECH": {"speech.response_min_s": "no measurement wrote it"}}
        _write_row(tmp_path, "sub-a", _verdict(Triage.FLAG, critical_absences=absence))
        report = aggregate(decisions(tmp_path))
        assert report.critical_absences == {"SPEECH": {"speech.response_min_s": 1}}

    def test_flagged_files_are_attributed_to_their_declared_family(self, tmp_path: Path) -> None:
        """Which families flag is the corpus question; a bare flag count cannot answer it."""
        _write_row(tmp_path, "sub-a", _verdict(Triage.FLAG, family="syllable"))
        _write_row(tmp_path, "sub-b", _verdict(Triage.PASS, family="syllable"))
        _write_row(tmp_path, "sub-c", _verdict(Triage.FLAG, family=None))
        report = aggregate(decisions(tmp_path))
        assert report.families == {"syllable": 2, "undeclared": 1}
        assert report.flagged_families == {"syllable": 1, "undeclared": 1}

    def test_counts_routes_findings_and_run_states_per_branch(self, tmp_path: Path) -> None:
        """Routing and finding are separate questions and are counted separately."""
        _write_row(
            tmp_path,
            "sub-a",
            _verdict(
                routes={"SPEECH": "ROUTED", "VOICE": "DECLINED"},
                findings={"SPEECH": "present", "VOICE": "uncertain"},
                ran={"SPEECH": RunState.COMPLETED, "VOICE": RunState.SKIPPED},
            ),
        )
        report = aggregate(decisions(tmp_path))
        assert report.routes == {"SPEECH": {"ROUTED": 1}, "VOICE": {"DECLINED": 1}}
        assert report.findings == {"SPEECH": {"present": 1}, "VOICE": {"uncertain": 1}}
        assert report.ran == {"SPEECH": {"completed": 1}, "VOICE": {"skipped": 1}}

    def test_counts_every_contributing_reason_not_only_the_deciding_one(self, tmp_path: Path) -> None:
        """The fold carries them all, so the corpus can say what else nearly decided."""
        reasons = [
            NodeVerdict(node="QUALITY", outcome=Outcome.FLAG, kind="clip", why="clipped"),
            NodeVerdict(node="SPEECH", outcome=Outcome.PASS, kind=None, why="ok"),
        ]
        _write_row(tmp_path, "sub-a", _verdict(Triage.FLAG, reasons=reasons))
        report = aggregate(decisions(tmp_path))
        assert report.reasons == {"QUALITY|flag|clip": 1, "SPEECH|pass|None": 1}

    def test_counts_llm_redaction_status_and_its_flagged_categories(self, tmp_path: Path) -> None:
        """A list value is counted per member, so the categories are countable across the corpus."""
        _write_row(tmp_path, "sub-a", _verdict(llm_redaction={"status": "ran", "flagged": ["name", "date"]}))
        _write_row(tmp_path, "sub-b", _verdict(llm_redaction={"status": "ran", "flagged": ["name"]}))
        report = aggregate(decisions(tmp_path))
        assert report.llm_redaction["status"] == {"ran": 2}
        assert report.llm_redaction["flagged"] == {"name": 2, "date": 1}

    def test_node_errors_are_counted_from_the_record_not_the_decision(self, tmp_path: Path) -> None:
        """A node that raised leaves its mark on the row, whether or not VERDICT still folded."""
        _write_row(tmp_path, "sub-a", _verdict(), errors={"REPORT": "ReportRenderError: no axis"})
        report = aggregate(decisions(tmp_path))
        assert report.errored == {"REPORT": 1}
        assert report.unread == 0


class TestRendering:
    """The report reads as a decision record, with denominators on every share."""

    def test_every_count_carries_its_share_of_the_corpus(self, tmp_path: Path) -> None:
        """A count without a denominator is not a corpus finding."""
        _write_row(tmp_path, "sub-a", _verdict(Triage.FLAG))
        _write_row(tmp_path, "sub-b", _verdict())
        rendered = render_markdown(aggregate(decisions(tmp_path)), tmp_path)
        assert "2 recordings read" in rendered
        assert "| `flag` | 1 | 50.00% |" in rendered

    def test_an_empty_tree_renders_without_dividing_by_zero(self, tmp_path: Path) -> None:
        """A run that produced nothing must still produce a readable report."""
        rendered = render_markdown(aggregate(decisions(tmp_path)), tmp_path)
        assert "No decisions were read." in rendered


class TestDuration:
    """How long the recording is, counted beside what the graph decided about it."""

    def test_buckets_run_from_the_shortest_edge_upward(self, tmp_path: Path) -> None:
        """An unusually short recording is a different finding from a task attempted and failed."""
        for stem, seconds in (("sub-a", 0.4), ("sub-b", 2.0), ("sub-c", 45.0), ("sub-d", 300.0)):
            _write_row(tmp_path, stem, _verdict(), duration_s=seconds)
        report = aggregate(decisions(tmp_path))
        assert list(report.durations) == ["0-1s", "1-3s", "30-60s", ">=60s"]

    def test_a_header_that_gave_no_duration_is_counted_last_not_dropped(self, tmp_path: Path) -> None:
        """A recording nobody could time is still a recording of the corpus."""
        _write_row(tmp_path, "sub-a", _verdict(), duration_s=None)
        _write_row(tmp_path, "sub-b", _verdict(), duration_s=12.0)
        report = aggregate(decisions(tmp_path))
        assert list(report.durations)[-1] == "unknown"
        assert report.durations["unknown"] == 1

    def test_triage_is_cross_tabbed_against_duration(self, tmp_path: Path) -> None:
        """The corpus question is whether the short recordings are the ones that flag."""
        _write_row(tmp_path, "sub-a", _verdict(Triage.FLAG), duration_s=0.5)
        _write_row(tmp_path, "sub-b", _verdict(Triage.FLAG), duration_s=0.7)
        _write_row(tmp_path, "sub-c", _verdict(Triage.PASS), duration_s=20.0)
        report = aggregate(decisions(tmp_path))
        assert report.triage_by_duration["0-1s"] == {"flag": 2}
        assert report.triage_by_duration["10-30s"] == {"pass": 1}

    def test_a_row_without_a_decision_still_reaches_its_duration_bucket(self, tmp_path: Path) -> None:
        """A run that raised is counted where its duration puts it, not nowhere."""
        _write_row(tmp_path, "sub-a", None, duration_s=0.2)
        report = aggregate(decisions(tmp_path))
        assert report.durations["0-1s"] == 1
        assert report.triage_by_duration["0-1s"] == {UNREAD: 1}


class TestWhyARecordingDidNotPass:
    """A flag count that cannot say why is not a decision record."""

    def test_the_family_is_taken_off_a_ground_so_one_ground_is_one_count(self, tmp_path: Path) -> None:
        """The same conclusion on two families is one ground, not two."""
        from senselab.audio.workflows.triage.corpus_report import reason_ground

        first = "VOICE reported that what the instruction asked for did not happen on maximum-phonation-time"
        second = "VOICE reported that what the instruction asked for did not happen on prolonged-vowel"
        assert reason_ground(first) == reason_ground(second)
        assert reason_ground(first).endswith("did not happen")

    def test_a_ground_naming_no_family_is_left_whole(self) -> None:
        """Most grounds carry no family, and must not be truncated at an incidental preposition."""
        from senselab.audio.workflows.triage.corpus_report import reason_ground

        ground = "mismatch: routing routed VOICE, it found no subject"
        assert reason_ground(ground) == ground

    def test_grounds_are_counted_per_node_and_exclude_passes(self, tmp_path: Path) -> None:
        """A passing verdict is not a reason a recording did not pass."""
        reasons = [
            NodeVerdict(node="VOICE", outcome=Outcome.FLAG, kind="voice", why="routed VOICE, it found no subject"),
            NodeVerdict(node="VOICE", outcome=Outcome.FLAG, kind="voice", why="did not happen on prolonged-vowel"),
            NodeVerdict(node="SPEECH", outcome=Outcome.PASS, kind=None, why="ran and reported"),
        ]
        _write_row(tmp_path, "sub-a", _verdict(Triage.FLAG, reasons=reasons))
        report = aggregate(decisions(tmp_path))
        assert report.grounds["VOICE"] == {"routed VOICE, it found no subject": 1, "did not happen": 1}
        assert "SPEECH" not in report.grounds

    def test_the_same_ground_across_recordings_accumulates(self, tmp_path: Path) -> None:
        """This is the number that says whether one defect is the corpus's flag rate."""
        for index in range(3):
            reasons = [NodeVerdict(node="VOICE", outcome=Outcome.FLAG, kind="voice", why="it found no subject")]
            _write_row(tmp_path, f"sub-{index}", _verdict(Triage.FLAG, reasons=reasons))
        report = aggregate(decisions(tmp_path))
        assert report.grounds == {"VOICE": {"it found no subject": 3}}
