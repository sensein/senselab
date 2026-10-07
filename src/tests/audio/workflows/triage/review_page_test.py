"""Tests for the triage review page: spectrogram packing, overlays, records, the index and the page."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf

from senselab.audio.workflows.triage import review_page
from senselab.audio.workflows.triage.recording_vectors import read_store
from senselab.audio.workflows.triage.review_page import page as page_module
from senselab.audio.workflows.triage.review_page.records import overlays
from senselab.audio.workflows.triage.review_page.spectrogram import settings
from senselab.audio.workflows.triage.vocabulary import (
    ROUTED,
    BranchDecision,
    NodeVerdict,
    Outcome,
    TaskEvidence,
    fold_file_verdict,
)

RATE = 16000
STEM = "sub-0123abcd_ses-01_task-respiration-and-cough-breath"


def _fold() -> dict[str, Any]:
    """A pass over a breath task, its record as VERDICT stores it."""
    inputs = {"floor_db": 24.0, "local_db": 20.0, "entangled": False, "snr_low_db": 10.0, "snr_high_db": 16.0}
    folded = fold_file_verdict(
        [NodeVerdict("ADMIT", Outcome.PASS, None, "ok")],
        branch_decisions={
            "AIRWAY": BranchDecision(branch="AIRWAY", will_run=False, route_state=ROUTED, forced_by_declaration=False),
            "SPEECH": BranchDecision(
                branch="SPEECH", will_run=False, route_state="declined", forced_by_declaration=False
            ),
        },
        ran={},
        hint_claims={},
        route_state=ROUTED,
        declared_family="respiration-and-cough-breath",
        task=TaskEvidence(
            owning_branches=("AIRWAY",),
            duration_s=4.0,
            minimum_duration_s=1.0,
            required_event="breath",
            breath_mode="sustained",
            breath_decision="present",
            breath_review=False,
            breath_train_breaths=2,
            breath_reading={"evidence": {"inputs": inputs, "rhythm": {"hz": 0.25, "prominence_db": 9.0}}},
        ),
    )
    return {"node": "VERDICT", **folded.record()}


def _tone(seconds: float, hz: float) -> np.ndarray:
    t = np.arange(int(seconds * RATE)) / RATE
    return (0.3 * np.sin(2 * np.pi * hz * t)).astype(np.float32)


def _run_root(root: Path) -> Path:
    """A finished run directory: a fold, streams on disk, a breath reading and a background reading."""
    run_root = root / "sub-0123abcd" / "ses-01" / f"{STEM}_20261007-120000"
    (run_root / "run" / "streams").mkdir(parents=True)
    (run_root / "released").mkdir()
    sf.write(run_root / "run" / "streams" / "plain.flac", _tone(4.0, 1000.0), RATE)
    sf.write(run_root / "released" / "audio.flac", _tone(1.0, 500.0), RATE)

    def entity(eid: str, prov: str, attributes: dict[str, Any], extent: list[float] | None = None) -> dict[str, Any]:
        return {"record": "entity", "id": eid, "prov_type": prov, "extent": extent, "attributes": attributes}

    lines = [
        entity("rec", "stream", {"name": "recording", "path": "streams/recording.flac"}, [0.0, 4.0]),
        entity("plain", "stream", {"name": "plain", "path": "streams/plain.flac"}, [0.0, 4.0]),
        entity("ext", "span", {"role": "task_extent"}, [0.5, 3.5]),
        entity(
            "br",
            "measurement",
            {"name": "airway_breath_reading", "reading": {"train": {"bursts_s": [[1.0, 0.8, 1.3], [2.5, 2.2, 2.9]]}}},
        ),
        entity(
            "bg",
            "measurement",
            {
                "name": "background_model",
                "regions": [{"start_s": 0.7, "end_s": 3.0}],
                "impulses": [{"peak_s": 3.3, "start_s": 3.3, "end_s": 3.31}],
                "faults": {"shutoff": [], "dropout": [[3.6, 3.7]], "discontinuity": [], "clip": []},
            },
        ),
        entity("fold", "verdict", _fold()),
        {"record": "activity", "node": "VERDICT", "step": None, "parameters": {"config_hash": "c0ffee"}},
    ]
    (run_root / "run" / "store.jsonl").write_text("\n".join(json.dumps(line) for line in lines) + "\n")
    return run_root


def test_levels_pack_two_bits_a_cell_and_unpack_to_the_same_grid() -> None:
    """Packing is lossless over the level grid, and 16 x 64 cells take 256 bytes."""
    spec = settings()["spectrogram"]
    rng = np.random.default_rng(0)
    levels = rng.integers(0, 4, size=(spec["frames"], spec["bands"]), dtype=np.uint8)
    text = review_page.pack(levels)
    assert len(text) == 344
    assert np.array_equal(review_page.unpack(text, spec["frames"], spec["bands"]), levels)


def test_a_tone_is_loudest_in_its_own_band_and_silence_is_level_zero() -> None:
    """The grid places a 1 kHz tone in the band that holds 1 kHz, at the top level."""
    spec = settings()["spectrogram"]
    noise = 1e-4 * np.random.default_rng(1).standard_normal(3 * RATE).astype(np.float32)
    levels = review_page.quantised_levels(_tone(3.0, 1000.0) + noise, RATE)
    assert levels.shape == (spec["frames"], spec["bands"])
    edges = np.geomspace(spec["f_min_hz"], spec["f_max_hz"], spec["bands"] + 1)
    band = int(np.searchsorted(edges, 1000.0) - 1)
    assert int(np.median(levels[:, band])) == spec["levels"] - 1
    assert not review_page.quantised_levels(np.zeros(RATE, dtype=np.float32), RATE).any()


def test_overlays_read_events_activity_and_issues(tmp_path: Path) -> None:
    """Breath bursts are events, background regions activity, impulses and faults issues."""
    view = read_store(_run_root(tmp_path) / "run" / "store.jsonl")
    drawn = overlays(view)
    assert drawn["events"] == [[0.8, 1.3], [2.2, 2.9]]
    assert drawn["activity"] == [[0.7, 3.0]]
    assert [3.3, 3.31, "impulse"] in drawn["issues"] and [3.6, 3.7, "dropout"] in drawn["issues"]


def test_a_record_carries_decision_evidence_streams_and_a_spectrogram(tmp_path: Path) -> None:
    """The record reads the fold through the decision tables and adds what the page draws."""
    run_root = _run_root(tmp_path)
    record = review_page.review_record(run_root, tmp_path, speech=lambda _: {"html": "never asked"})
    assert record is not None
    assert (record["verdict"], record["release"], record["branch"]) == ("pass", "as_is", "AIRWAY")
    assert record["extent"] == [0.5, 3.5] and record["duration_s"] == 4.0
    decisive = {item["name"] for item in record["evidence"] if item["decisive"]}
    assert "breath_event_db_over_local" in decisive
    assert any(item["name"] == "breath_rhythm_hz" and not item["decisive"] for item in record["evidence"])
    assert record["spec_stream"] == "plain" and len(record["spec"]) == 344
    assert record["streams"]["plain"].endswith("run/streams/plain.flac")
    assert record["streams"]["released"].endswith("released/audio.flac")
    assert record["speech"] is None


def test_the_page_inlines_its_index_and_writes_side_files(tmp_path: Path) -> None:
    """index.html carries the index and every part; the side files hand their records to the page."""
    record = review_page.review_record(_run_root(tmp_path / "corpus"), tmp_path / "corpus")
    assert record is not None
    other = {
        **record,
        "stem": "sub-ffffffff_ses-01_task-harvard-sentences",
        "participant": "sub-ffffffff",
        "verdict": "review",
        "reason": "weak_events",
        "evidence": [],
        "speech": {"html": "a <mark>Name</mark>"},
    }
    written = review_page.write_page([other, record], tmp_path / "page", title="t", audio_base="../", mark_style=".m{}")
    assert written["recordings"] == 2 and written["shards"] == 1
    html_text = (tmp_path / "page" / "index.html").read_text()
    assert "/*@" not in html_text and ".m{}" in html_text
    index = json.loads(re.search(r"var REVIEW_INDEX = (\{.*?\});</script>", html_text, re.S).group(1))  # type: ignore[union-attr]
    assert index["cols"]["stem"][0] == record["stem"]
    assert index["cols"]["text"][1] == "a name"
    assert index["evidence"]["breath_event_db_over_local"]["kind"] == "numeric"
    shard = (tmp_path / "page" / "data" / "shard-0000.js").read_text()
    assert shard.startswith("ReviewPage.shard(0,") and "</" not in shard.replace("<\\/", "")


def test_every_part_the_shell_inlines_is_listed() -> None:
    """The shell and PARTS agree, so a page never ships a marker."""
    named = set(page_module.INLINE_PATTERN.findall(page_module.SHELL.read_text()))
    assert named == set(page_module.PARTS)
    assert all(path.is_file() for path in page_module.PARTS.values())


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not on PATH; the JavaScript suite cannot run")
def test_the_page_script_suite_passes() -> None:
    """The review page's JavaScript suite: index decoding, columns, axes, search, unpacking, export."""
    suite = Path(__file__).parent / "review_page" / "review.test.mjs"
    result = subprocess.run(["node", "--test", str(suite)], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
