"""The recording-level background split around the session floor: own floor, SESSION, BACKGROUND."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile

from senselab.audio.workflows.triage.background_model import (
    BACKGROUND_MODEL,
    measure_background,
    own_floor_of,
    session_floor_db,
)
from senselab.audio.workflows.triage.nodes.background import (
    FLOOR_FROM_RECORDING,
    FLOOR_FROM_SESSION,
    SESSION_FLOOR,
    session_attributes,
    session_key,
    write_background,
    write_own_floor,
    write_session_floor,
)
from senselab.audio.workflows.triage.nodes.common import find_measurement
from senselab.audio.workflows.triage.task_events import generic_view, generic_view_of
from senselab.utils.prov_store import ProvStore

RATE = 16000


def _noise(seconds: float, db: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(int(seconds * RATE)) * 10 ** (db / 20.0)


def _vowel(seconds: float, db: float, f0: float = 150.0) -> np.ndarray:
    t = np.arange(int(seconds * RATE)) / RATE
    tone = sum(np.sin(2 * np.pi * k * f0 * t) / k for k in range(1, 20) if k * f0 < RATE / 2)
    return tone / np.max(np.abs(tone)) * 10 ** (db / 20.0)


def _breaths(seconds: float, room_db: float, seed: int) -> np.ndarray:
    x = _noise(seconds, room_db, seed)
    for start in np.arange(1.0, seconds - 1.0, 2.0):
        n = int(0.6 * RATE)
        burst = _noise(0.6, room_db + 25.0, seed + int(start * 10)) * np.hanning(n)
        x[int(start * RATE) : int(start * RATE) + n] += burst
    return x


def test_the_session_is_the_bids_participant_and_session() -> None:
    """The key is read off the stem; a stem naming no session has none."""
    assert session_key("sub-0a1b_ses-77AA_task-breath_20260920") == "sub-0a1b_ses-77AA"
    assert session_key("recording") is None


def test_a_recording_that_captured_only_the_room_stands_far_under_its_session() -> None:
    """In a quiet session, a room-only recording keeps its own floor and reads its level against the session's."""
    members = [own_floor_of((_breaths(8.0, -60.0, seed), RATE), None) for seed in (1, 2, 3)]
    room = own_floor_of((_noise(8.0, -60.0, 9), RATE), None)
    session = session_floor_db([*members, room])
    assert session["band_db"] is not None and session["used_n"] == 4
    record = session_attributes(room, [*members, room])
    assert record["floor_source"] == FLOOR_FROM_SESSION
    assert record["level_rel_db"] is not None and record["level_rel_db"] <= -10.0
    assert abs(record["floor_rel_db"]) < 3.0


def test_a_task_that_fills_the_file_takes_its_floor_from_its_session() -> None:
    """A vowel with no quiet frames is read against its session's floor and stays active throughout."""
    members = [own_floor_of((_breaths(8.0, -70.0, seed), RATE), None) for seed in (1, 2, 3)]
    vowel = _vowel(6.0, -20.0) + _noise(6.0, -70.0, 5)
    own = own_floor_of((vowel, RATE), None)
    session = session_floor_db([*members, own])
    quiet = np.median([m["own_db"] for m in members], axis=0)
    assert float(np.median(np.asarray(session["band_db"]) - quiet)) < 3.0
    reading = measure_background(
        (vowel, RATE), residual=None, recording=None, clips=(), session_db=np.asarray(session["band_db"])
    )
    assert reading.floor.source == "session"
    assert reading.active_s >= 5.5


def test_a_session_floor_equal_to_the_recordings_own_changes_nothing() -> None:
    """Where the session floor is the recording's own, the reading is the one with no session at all."""
    x = _breaths(6.0, -60.0, 4)
    without = measure_background((x, RATE), residual=None, recording=None, clips=())
    with_own = measure_background((x, RATE), residual=None, recording=None, clips=(), session_db=without.floor.own_db)
    assert np.array_equal(without.floor.band_db, with_own.floor.band_db)
    assert without.regions == with_own.regions and without.impulses == with_own.impulses


def test_too_few_quiet_members_leave_no_session_floor() -> None:
    """Under the minimum of usable members the session has no floor, and the recording keeps its own."""
    members = [own_floor_of((_breaths(8.0, -60.0, seed), RATE), None) for seed in (1, 2)]
    assert session_floor_db(members)["band_db"] is None
    assert session_attributes(members[0], members)["floor_source"] == FLOOR_FROM_RECORDING


def _store_with_streams(tmp_path: Path, plain: np.ndarray, residual: np.ndarray | None) -> ProvStore:
    store = ProvStore(run_id="background-test")
    for name, samples in (("plain", plain), ("residual", residual)):
        if samples is None:
            continue
        path = tmp_path / f"{name}.wav"
        soundfile.write(path, samples.astype(np.float32), RATE, subtype="FLOAT")
        store.entity(prov_type="stream", extent=None, attributes={"name": name, "path": str(path)})
    return store


def test_the_branches_read_the_background_bit_for_bit_as_they_would_recompute_it(tmp_path: Path) -> None:
    """A single-file run's stored view is the view a branch would have read off the streams itself."""
    plain = _breaths(6.0, -60.0, 7)
    residual = _noise(6.0, -70.0, 8)
    store = _store_with_streams(tmp_path, plain, residual)
    write_own_floor(store, tmp_path)
    write_session_floor(store, (), session=None)
    session = find_measurement(store, SESSION_FLOOR)
    assert session is not None and session.attributes["floor_source"] == FLOOR_FROM_RECORDING
    write_background(store, tmp_path, clips=())
    stored = generic_view_of(store, tmp_path)
    plain_read, _ = soundfile.read(tmp_path / "plain.wav", dtype="float32")
    residual_read, _ = soundfile.read(tmp_path / "residual.wav", dtype="float32")
    recomputed = generic_view((plain_read, RATE), (residual_read, RATE))
    assert stored is not None
    assert np.array_equal(stored.frames.band_db, recomputed.frames.band_db)
    assert np.array_equal(stored.floor.band_db, recomputed.floor.band_db)
    assert stored.regions == recomputed.regions and stored.impulses == recomputed.impulses
    assert stored.broadband_floor_db == recomputed.broadband_floor_db


def test_without_a_session_floor_the_background_is_not_read(tmp_path: Path) -> None:
    """BACKGROUND names the session floor it lacked; the branches then read no background at all."""
    store = _store_with_streams(tmp_path, _breaths(4.0, -60.0, 2), None)
    write_background(store, tmp_path, clips=())
    found = find_measurement(store, BACKGROUND_MODEL)
    assert found is not None and found.attributes["missing"] == [SESSION_FLOOR]
    assert generic_view_of(store, tmp_path) is None


def test_a_store_without_a_plain_stream_records_what_it_lacked(tmp_path: Path) -> None:
    """The own floor names the absent input rather than going unwritten, and SESSION names the floor."""
    store = ProvStore(run_id="no-plain")
    write_own_floor(store, tmp_path)
    write_session_floor(store, (), session=None)
    own, session = find_measurement(store, "background_floor"), find_measurement(store, SESSION_FLOOR)
    assert own is not None and own.attributes["missing"] == ["plain"]
    assert session is not None and session.attributes["missing"] == ["background_floor"]
