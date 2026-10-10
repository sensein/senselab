"""The session background read off the residual of the session's speech tasks, and the recording's quality."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile

from senselab.audio.workflows.triage.background_model import measure_background
from senselab.audio.workflows.triage.nodes.background import (
    SESSION_FLOOR,
    SPEECH_RESIDUAL,
    write_background,
    write_own_floor,
    write_session_floor,
    write_speech_residual,
)
from senselab.audio.workflows.triage.nodes.common import find_measurement
from senselab.audio.workflows.triage.nodes.quality import speech_quality_record
from senselab.audio.workflows.triage.session_background import (
    NO_FOREGROUND,
    NO_SPEECH_TASK,
    SPEECH_RESIDUAL_SOURCE,
    background_floor_for,
    session_background_of,
    session_background_parameters,
    speech_residual_of,
)
from senselab.audio.workflows.triage.vocabulary import _speech_quality_items
from senselab.utils.prov_store import ProvStore

RATE = 16000
SPEECH = [(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)]


def _noise(seconds: float, db: float, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(int(seconds * RATE)) * 10 ** (db / 20.0)


def _speech(seconds: float, db: float) -> np.ndarray:
    t = np.arange(int(seconds * RATE)) / RATE
    tone = sum(np.sin(2 * np.pi * k * 140.0 * t) / k for k in range(1, 25))
    tone = tone / np.sqrt(np.mean(tone**2)) * 10 ** (db / 20.0)
    gate = np.zeros_like(t)
    for start, end in SPEECH:
        gate[(t >= start) & (t < end)] = 1.0
    return tone * gate


def _task(room_db: float, seed: int, speech_db: float = -25.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    enhanced = _speech(7.0, speech_db)
    residual = _noise(7.0, room_db, seed)
    return enhanced + residual, enhanced, residual


def _reading(room_db: float, seed: int, family: str, *, words: int = 6, **kw: Any) -> dict[str, Any]:  # noqa: ANN401
    plain, enhanced, residual = _task(room_db, seed, **kw)
    return {
        "stem": f"sub-a_ses-1_task-{family}-{seed}",
        **speech_residual_of(
            (plain, RATE),
            (enhanced, RATE),
            (residual, RATE),
            speech=SPEECH if words else [],
            lexical_words_n=words,
            enhanced_speech_max=0.9 if words else 0.0,
            family=family,
        ),
    }


def test_a_speech_task_reads_its_known_residual_as_the_background() -> None:
    """White noise at -60 dB under speech at -25 dB: the residual reads near -60 dB and the foreground stands over it."""
    reading = _reading(-60.0, 1, "rainbow-passage")
    assert reading["member"] and reading["not_applicable"] is None
    assert reading["residual_db"]["whole"] == pytest.approx(-60.4, abs=1.0)
    assert reading["residual_db"]["non_speech"] == pytest.approx(-60.4, abs=1.0)
    assert reading["foreground_minus_residual_db"]["whole"] == pytest.approx(35.0, abs=3.0)
    assert reading["plain_minus_residual_db"]["non_speech"] == pytest.approx(0.0, abs=0.5)
    assert reading["plain_minus_residual_db"]["speech"] > 20.0
    assert reading["speech_s"] == pytest.approx(3.0, abs=0.1)


def test_a_session_of_speech_tasks_takes_their_median_residual() -> None:
    """Three speech tasks at -62, -60 and -50 dB and a breath task at -20 dB: the session reads -60 dB from the three."""
    readings = [
        _reading(-62.0, 1, "rainbow-passage"),
        _reading(-60.0, 2, "free-speech"),
        _reading(-50.0, 3, "cape-v-sentences"),
        _reading(-20.0, 4, "breath-sounds"),
    ]
    session = session_background_of(readings)
    assert session["source"] == SPEECH_RESIDUAL_SOURCE and session["fallback"] is None
    assert session["candidates_n"] == 3 and session["members_n"] == 3
    assert "sub-a_ses-1_task-breath-sounds-4" not in session["members"]
    assert session["level_db"] == pytest.approx(-60.4, abs=1.0)
    assert session["spread_db"]["members_iqr"] == pytest.approx(6.0, abs=1.5)
    assert session["quality"]["foreground_minus_residual_db"]["whole"] == pytest.approx(35.0, abs=3.0)
    assert len(session["band_db"]) == len(session["band_edges_hz"]) - 1


def test_a_session_without_a_speech_task_falls_back_to_the_session_floor() -> None:
    """Only airway and voice tasks: no session background, and BACKGROUND keeps the session floor and says why."""
    readings = [_reading(-60.0, 1, "breath-sounds"), _reading(-60.0, 2, "prolonged-vowel")]
    session = session_background_of(readings)
    assert session["source"] is None and session["fallback"] == NO_SPEECH_TASK and session["band_db"] is None
    assert session["quality"] == {"not_applicable": NO_FOREGROUND}
    on = {**session_background_parameters(), "decides": True}
    band, reference = background_floor_for("breath-sounds", session, [], on)
    assert band is None and reference["used"] == "session_floor" and reference["fallback"] == NO_SPEECH_TASK


def test_an_empty_enhanced_stream_has_no_foreground() -> None:
    """No words and no speech on the enhanced stream: the quality is not applicable, never a number."""
    residual = _noise(7.0, -60.0, 5)
    reading = speech_residual_of(
        (residual, RATE),
        (np.zeros_like(residual), RATE),
        (residual, RATE),
        speech=[],
        lexical_words_n=0,
        enhanced_speech_max=0.02,
        family="rainbow-passage",
    )
    assert not reading["foreground"] and not reading["member"]
    assert reading["not_applicable"] == NO_FOREGROUND
    assert reading["foreground_db"] is None
    assert set(reading["foreground_minus_residual_db"].values()) == {None}
    assert reading["plain_minus_residual_db"]["speech"] is None
    assert reading["plain_minus_residual_db"]["whole"] == pytest.approx(0.0, abs=0.1)
    session = session_background_of([{**reading, "stem": "x"}])
    assert session["fallback"] == NO_SPEECH_TASK
    items = {
        i.name: i.value
        for i in _speech_quality_items({"speech_quality": speech_quality_record(reading, session)})
    }
    assert items["speech_quality.foreground_minus_residual_db.whole"] == NO_FOREGROUND
    assert items["speech_quality.plain_minus_residual_db.speech"] == NO_FOREGROUND
    assert items["speech_quality.plain_minus_residual_db.whole"] == pytest.approx(0.0, abs=0.1)
    assert items["speech_quality.session_foreground_minus_residual_db"] == NO_FOREGROUND


def test_the_switch_decides_which_floor_a_non_speech_task_reads() -> None:
    """Off, every task keeps the session floor; on, an airway task reads the session background and a speech task does not."""
    session = session_background_of([_reading(-60.0, 1, "rainbow-passage")])
    edges = session["band_edges_hz"]
    off_band, off = background_floor_for("breath-sounds", session, edges)
    assert off_band is None and off == {"decides": False, "kind": "airway", "used": "session_floor", "fallback": None}
    on = {**session_background_parameters(), "decides": True}
    band, reference = background_floor_for("breath-sounds", session, edges, on)
    assert band is not None and reference["used"] == SPEECH_RESIDUAL_SOURCE
    assert background_floor_for("rainbow-passage", session, edges, on)[0] is None
    plain = _noise(6.0, -40.0, 3)
    reading = measure_background((plain, RATE), residual=None, recording=None, clips=(), background_db=band)
    assert reading.floor.source == SPEECH_RESIDUAL_SOURCE
    assert np.array_equal(reading.floor.band_db, band)


def _store(tmp_path: Path, stem: str, room_db: float, seed: int) -> ProvStore:
    plain, enhanced, residual = _task(room_db, seed)
    store = ProvStore(run_id=stem)
    for name, samples in (("recording", plain), ("plain", plain), ("enhanced", enhanced), ("residual", residual)):
        path = tmp_path / f"{stem}_{name}.wav" if name != "recording" else tmp_path / f"{stem}.wav"
        soundfile.write(path, samples.astype(np.float32), RATE, subtype="FLOAT")
        store.entity(prov_type="stream", extent=None, attributes={"name": name, "path": str(path)})
    windows = [
        {"start": a, "end": b, "label_scores": [{"Speech": 0.9}]} for a, b in SPEECH
    ] + [{"start": 6.5, "end": 7.0, "label_scores": [{"Silence": 0.9}]}]
    (tmp_path / f"{stem}_enhanced_yamnet.json").write_text(json.dumps(windows))
    store.entity(
        prov_type="measurement",
        extent=None,
        attributes={"name": "enhanced_yamnet_scores", "path": f"{stem}_enhanced_yamnet.json"},
    )
    return store


def test_session_writes_the_reading_and_background_reads_the_reference(tmp_path: Path) -> None:
    """SESSION writes each recording's reading and the session background; BACKGROUND records which floor it used."""
    speech = _store(tmp_path, "sub-a_ses-1_task-rainbow-passage", -60.0, 1)
    breath = _store(tmp_path, "sub-a_ses-1_task-breath-sounds", -60.0, 2)
    readings = [write_speech_residual(store, tmp_path) for store in (speech, breath)]
    assert readings[0]["family"] == "rainbow-passage" and readings[1]["kind"] == "airway"
    for store in (speech, breath):
        write_own_floor(store, tmp_path)
    for store in (speech, breath):
        write_session_floor(store, (), session="sub-a_ses-1", background_members=readings)
        write_background(store, tmp_path, clips=())
    found = find_measurement(breath, SESSION_FLOOR)
    assert found is not None
    background = found.attributes["session_background"]
    assert background["members"] == ["sub-a_ses-1_task-rainbow-passage"]
    model = find_measurement(breath, "background_model")
    assert model is not None and model.attributes["floor_reference"]["used"] == "session_floor"
    assert model.attributes["floor_reference"]["decides"] is False
    assert find_measurement(speech, SPEECH_RESIDUAL) is not None
