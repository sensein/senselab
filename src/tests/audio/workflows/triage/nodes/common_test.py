"""The shared node helpers: verdict-key shadowing, and the latest-non-invalidated read rule."""

from pathlib import Path
from typing import Callable

import numpy as np
import pytest

from senselab.audio.workflows.triage.nodes.common import find_measurement, resolve_stream, software_agent, write_verdict
from senselab.audio.workflows.triage.vocabulary import Outcome
from senselab.utils.prov_store import ProvStore


class TestWriteVerdict:
    """The reserved verdict keys cannot be shadowed by detail."""

    def test_a_reserved_key_in_detail_raises_naming_it(self, store: ProvStore) -> None:
        """A detail dict carrying a reserved key is a ValueError naming the offender, not a silent overwrite."""
        activity_id = store.activity(node="TEST", step=None, parameters={})
        agent_id = software_agent(store)
        with pytest.raises(ValueError, match="kind"):
            write_verdict(
                store,
                activity_id,
                agent_id,
                node="TEST",
                outcome=Outcome.PASS,
                kind=None,
                why="testing",
                detail={"kind": "airway"},
            )

    def test_a_key_that_merely_resembles_a_reserved_one_is_allowed(self, store: ProvStore) -> None:
        """`kinds` (plural) is not reserved and lands in the stored attributes."""
        activity_id = store.activity(node="TEST", step=None, parameters={})
        agent_id = software_agent(store)
        entity_id, _ = write_verdict(
            store,
            activity_id,
            agent_id,
            node="TEST",
            outcome=Outcome.PASS,
            kind=None,
            why="testing",
            detail={"kinds": {"speech": "present"}},
        )
        assert store.get_entity(entity_id).attributes["kinds"] == {"speech": "present"}


class TestFindMeasurement:
    """find_measurement returns the latest non-invalidated match, or None."""

    def test_returns_none_when_nothing_carries_the_name(self, store: ProvStore) -> None:
        """An empty store yields None, not an error."""
        assert find_measurement(store, "hnr") is None

    def test_returns_the_entity_carrying_the_name(self, store: ProvStore) -> None:
        """A single matching measurement is found by its name attribute."""
        entity_id = store.entity(prov_type="measurement", extent=None, attributes={"name": "hnr", "value": 1.0})
        found = find_measurement(store, "hnr")
        assert found is not None
        assert found.id == entity_id

    def test_skips_an_invalidated_measurement(self, store: ProvStore) -> None:
        """When the latest match is invalidated, the earlier surviving one wins."""
        first = store.entity(prov_type="measurement", extent=None, attributes={"name": "hnr", "value": 1.0})
        second = store.entity(prov_type="measurement", extent=None, attributes={"name": "hnr", "value": 2.0})
        activity_id = store.activity(node="TEST", step=None, parameters={})
        store.was_invalidated_by(second, activity_id)
        found = find_measurement(store, "hnr")
        assert found is not None
        assert found.id == first


class TestResolveStream:
    """resolve_stream reads the latest non-invalidated stream and resolves relative sidecars."""

    def test_resolves_a_relative_sidecar_path_against_run_dir(
        self, store: ProvStore, tmp_path: Path, wav_writer: Callable[..., Path]
    ) -> None:
        """A relative stored path is joined onto run_dir before loading."""
        wav_writer("sidecar.wav", np.zeros(16000, dtype=np.float32))
        entity_id = store.entity(
            prov_type="stream", extent=(0.0, 1.0), attributes={"name": "recording", "path": "sidecar.wav"}
        )
        found_id, audio = resolve_stream(store, tmp_path, "recording")
        assert found_id == entity_id
        assert audio.waveform.shape[-1] == 16000

    def test_the_latest_stream_wins(self, store: ProvStore, tmp_path: Path, wav_writer: Callable[..., Path]) -> None:
        """Two live streams under one name resolve to the later write."""
        wav_writer("first.wav", np.zeros(16000, dtype=np.float32))
        wav_writer("second.wav", np.zeros(8000, dtype=np.float32))
        store.entity(prov_type="stream", extent=(0.0, 1.0), attributes={"name": "recording", "path": "first.wav"})
        second = store.entity(
            prov_type="stream", extent=(0.0, 0.5), attributes={"name": "recording", "path": "second.wav"}
        )
        found_id, audio = resolve_stream(store, tmp_path, "recording")
        assert found_id == second
        assert audio.waveform.shape[-1] == 8000

    def test_skips_an_invalidated_stream(
        self, store: ProvStore, tmp_path: Path, wav_writer: Callable[..., Path]
    ) -> None:
        """An invalidated stream is no longer read as what it was; the surviving one is returned."""
        wav_writer("first.wav", np.zeros(16000, dtype=np.float32))
        wav_writer("second.wav", np.zeros(8000, dtype=np.float32))
        first = store.entity(
            prov_type="stream", extent=(0.0, 1.0), attributes={"name": "recording", "path": "first.wav"}
        )
        second = store.entity(
            prov_type="stream", extent=(0.0, 0.5), attributes={"name": "recording", "path": "second.wav"}
        )
        activity_id = store.activity(node="TEST", step=None, parameters={})
        store.was_invalidated_by(first, activity_id)
        found_id, _ = resolve_stream(store, tmp_path, "recording")
        assert found_id == second
