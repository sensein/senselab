"""The result cache: keys that change with input or process, lossless arrays, one writer, one origin."""

from __future__ import annotations

from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import pytest

from senselab.utils.tasks import cached_inference as ci

_BASE: dict[str, Any] = {
    "input_signature": "a" * 64,
    "process": "clearvoice",
    "model_id": "org/model",
    "commit_sha": "b" * 40,
    "params": {"capability": "speech_separation", "sampling_rate": 16000},
}


def test_the_key_is_stable_for_the_same_input_and_process() -> None:
    """Two computations of the same key agree."""
    assert ci.result_cache_key(**_BASE) == ci.result_cache_key(**_BASE)


@pytest.mark.parametrize(
    "field, value",
    [
        ("input_signature", "c" * 64),
        ("model_id", "org/other"),
        ("commit_sha", "d" * 40),
        ("params", {"capability": "speech_separation", "sampling_rate": 48000}),
    ],
)
def test_a_change_in_input_or_process_changes_the_key(field: str, value: object) -> None:
    """Input content, model, resolved commit and parameters are each in the key."""
    assert ci.result_cache_key(**{**_BASE, field: value}) != ci.result_cache_key(**_BASE)


def test_a_process_version_bump_changes_the_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bumping a process's behaviour version orphans its entries."""
    before = ci.result_cache_key(**_BASE)
    monkeypatch.setattr(
        ci, "RESULT_PROCESS_VERSIONS", MappingProxyType({**ci.RESULT_PROCESS_VERSIONS, "clearvoice": 99})
    )
    assert ci.result_cache_key(**_BASE) != before


def test_the_senselab_version_is_not_in_the_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """A repository commit that leaves the process unchanged must not orphan its entries."""
    before = ci.result_cache_key(**_BASE)
    monkeypatch.setattr(ci, "senselab_version", lambda: "9.9.9.dev1")
    assert ci.result_cache_key(**_BASE) == before


def test_an_undeclared_process_is_refused() -> None:
    """A process with no behaviour version cannot be keyed."""
    with pytest.raises(KeyError):
        ci.result_cache_key(**{**_BASE, "process": "nothing"})


def test_a_stored_result_and_its_arrays_come_back_exactly() -> None:
    """Arrays are stored losslessly, float32 included."""
    key = ci.result_cache_key(**_BASE)
    wave = np.linspace(-1, 1, 1001, dtype=np.float32).reshape(1, -1)
    assert ci.result_store(
        key, {"n": 1}, process="clearvoice", model_id="org/model", commit_sha="b" * 40, arrays={"s": wave}
    )
    entry = ci.result_lookup(key)
    assert entry is not None
    assert entry["result"] == {"n": 1}
    assert entry["arrays"]["s"].dtype == np.float32
    assert np.array_equal(entry["arrays"]["s"], wave)
    assert entry["provenance"]["commit_sha"] == "b" * 40
    assert entry["origin"] is None


def test_the_first_writer_wins() -> None:
    """A second store of a held key writes nothing."""
    key = ci.result_cache_key(**_BASE)
    assert ci.result_store(key, {"n": 1}, process="clearvoice", model_id=None, commit_sha=None)
    assert not ci.result_store(key, {"n": 2}, process="clearvoice", model_id=None, commit_sha=None)
    entry = ci.result_lookup(key)
    assert entry is not None and entry["result"] == {"n": 1}


def test_the_origin_is_recorded_once() -> None:
    """The first caller to name where an entry was computed is the one a later hit reports."""
    key = ci.result_cache_key(**_BASE)
    ci.result_store(key, {"n": 1}, process="clearvoice", model_id=None, commit_sha=None)
    assert ci.annotate_result_origin(key, {"run": "r1", "activity": "act-1"})
    assert not ci.annotate_result_origin(key, {"run": "r2", "activity": "act-2"})
    entry = ci.result_lookup(key)
    assert entry is not None and entry["origin"] == {"run": "r1", "activity": "act-1"}


def test_no_origin_without_an_entry() -> None:
    """An origin cannot be attached to a key nothing was stored under."""
    assert not ci.annotate_result_origin(ci.result_cache_key(**_BASE), {"run": "r1"})


def test_a_corrupt_entry_is_a_miss() -> None:
    """An unreadable result file reads as absent, not as an error."""
    key = ci.result_cache_key(**_BASE)
    ci.result_store(key, {"n": 1}, process="clearvoice", model_id=None, commit_sha=None)
    cache_dir = ci.result_cache_dir()
    assert cache_dir is not None
    (cache_dir / key[:2] / key / "result.json").write_text("{not json", encoding="utf-8")
    assert ci.result_lookup(key) is None


def test_switched_off_the_cache_neither_reads_nor_writes(monkeypatch: pytest.MonkeyPatch) -> None:
    """``SENSELAB_RESULT_CACHE=off`` disables both directions."""
    monkeypatch.setenv("SENSELAB_RESULT_CACHE", "off")
    key = ci.result_cache_key(**_BASE)
    assert ci.result_cache_dir() is None
    assert not ci.result_store(key, {"n": 1}, process="clearvoice", model_id=None, commit_sha=None)
    assert ci.result_lookup(key) is None


def test_entries_are_scoped_by_schema_version(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A schema bump reads a fresh directory instead of wiping one other jobs may be reading."""
    monkeypatch.setenv("SENSELAB_RESULT_CACHE", str(tmp_path))
    assert ci.result_cache_dir() == tmp_path / f"schema-{ci.CACHE_SCHEMA_VERSION}"
    monkeypatch.setattr(ci, "CACHE_SCHEMA_VERSION", ci.CACHE_SCHEMA_VERSION + 1)
    assert ci.result_cache_dir() == tmp_path / f"schema-{ci.CACHE_SCHEMA_VERSION}"


def test_unset_the_cache_lives_under_the_senselab_cache_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """With no override the cache follows ``SENSELAB_CACHE``."""
    monkeypatch.delenv("SENSELAB_RESULT_CACHE")
    monkeypatch.setenv("SENSELAB_CACHE", str(tmp_path))
    assert ci.result_cache_dir() == tmp_path / "results" / f"schema-{ci.CACHE_SCHEMA_VERSION}"
