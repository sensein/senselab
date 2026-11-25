"""Triage's entity types are registered with the provenance store, the historical ones among them (plan step 6c)."""

from __future__ import annotations

from pathlib import Path

import pytest

from senselab.audio.workflows.triage.prov_types import TRIAGE_PROV_TYPES
from senselab.utils.prov_store import ProvStore, registered_prov_types


def test_loading_triage_registers_every_type_it_writes() -> None:
    """Importing any part of triage loads the package, which registers its types."""
    assert set(TRIAGE_PROV_TYPES) <= registered_prov_types()


@pytest.mark.parametrize("prov_type", TRIAGE_PROV_TYPES)
def test_every_triage_type_round_trips(prov_type: str, tmp_path: Path) -> None:
    """The reader's vocabulary is what makes a finished run openable, so none of it may be dropped.

    ``kind`` is the case this guards: TAXONOMY's evidence fold wrote it into every store of the 62,578-recording
    corpus until 2026-09-13, nothing writes it now, and dropping it made ``read_jsonl`` refuse every one of those
    stores.
    """
    store = ProvStore(run_id="r")
    entity = store.entity(prov_type=prov_type, extent=None, attributes={"state": "uncertain"})
    path = tmp_path / "prov.jsonl"
    store.write_jsonl(path)
    assert ProvStore.read_jsonl(path).get_entity(entity).prov_type == prov_type


def test_the_historical_kind_stays_registered() -> None:
    """The writer's vocabulary may shrink; the reader's may not."""
    assert "kind" in TRIAGE_PROV_TYPES
