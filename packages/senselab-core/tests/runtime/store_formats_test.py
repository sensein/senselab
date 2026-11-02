"""The result store's format gate: later minors read, another version's entries skipped, retired ones refused."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from planted_objects_fixture import entry_dir, rewrite_json, store_context, store_key
from senselab_core.kernel.identity import ReusePolicy
from senselab_core.runtime.store import (
    QUARANTINE_DIR,
    STORE_MIGRATION_STATEMENT,
    ResultStore,
    StoreFormatRetired,
    StoreFormatUnknown,
)

SKEWS: dict[str, tuple[str, Callable[[dict[str, Any]], None]]] = {
    "later_major_result": ("result.json", lambda r: r.update(format="result@3.0")),
    "later_major_context": ("context.json", lambda r: r.update(format="execution-context@4.0")),
    "later_minor_context_field": ("context.json", lambda r: r.update(format="execution-context@3.4", added=1)),
}


@pytest.mark.logic
@pytest.mark.parametrize("kind", sorted(SKEWS))
def test_another_versions_entry_is_skipped_not_quarantined(tmp_path: Path, kind: str) -> None:
    """An entry in another version's format is version skew: not served, never moved, and put leaves it."""
    s, k, c = ResultStore(tmp_path), store_key(), store_context()
    s.put(k, {"y": 1}, c)
    entry = entry_dir(tmp_path, k, c)
    member, change = SKEWS[kind]
    rewrite_json(entry / member, change)
    assert s.get(k, c, ReusePolicy.TOLERANT) is None
    with pytest.raises(StoreFormatUnknown):
        s.put(k, {"y": 1}, c)
    assert entry.is_dir() and not (tmp_path / QUARANTINE_DIR).exists()


@pytest.mark.logic
def test_a_later_minor_is_read(tmp_path: Path) -> None:
    """A later minor of each member is read; a result's fields unknown to this version are ignored."""
    s, k, c = ResultStore(tmp_path), store_key(), store_context()
    s.put(k, {"y": 1}, c)
    entry = entry_dir(tmp_path, k, c)
    rewrite_json(entry / "result.json", lambda r: r.update(format="result@2.3", added=1))
    rewrite_json(entry / "context.json", lambda r: r.update(format="execution-context@3.4"))
    got = s.get(k, c, ReusePolicy.EXACT)
    assert got is not None and got.outputs == {"y": 1}


@pytest.mark.logic
@pytest.mark.parametrize("member", ["result.json", "context.json"])
def test_a_retired_entry_raises_and_stays(tmp_path: Path, member: str) -> None:
    """A member in a retired format raises `StoreFormatRetired` naming the migration; the entry is never moved."""
    s, k, c = ResultStore(tmp_path), store_key(), store_context()
    s.put(k, {"y": 1}, c)
    entry = entry_dir(tmp_path, k, c)
    retired = {"result.json": "result@1.0", "context.json": "execution-context@1.0"}[member]
    rewrite_json(entry / member, lambda r: r.update(format=retired))
    with pytest.raises(StoreFormatRetired, match=STORE_MIGRATION_STATEMENT) as info:
        s.get(k, c, ReusePolicy.EXACT)
    assert info.value.found == retired
    assert entry.is_dir() and not (tmp_path / QUARANTINE_DIR).exists()


@pytest.mark.logic
def test_a_retired_sibling_does_not_hide_a_readable_entry(tmp_path: Path) -> None:
    """Under one key, a readable entry is served past a retired one, which stays where it is."""
    s, k = ResultStore(tmp_path), store_key()
    linux, mac = store_context(platform="linux"), store_context(platform="macos")
    s.put(k, {"y": 1}, linux)
    s.put(k, {"y": 2}, mac)
    retired = entry_dir(tmp_path, k, mac)
    rewrite_json(retired / "result.json", lambda r: r.update(format="result@1.0"))
    got = s.get(k, mac, ReusePolicy.TOLERANT)
    assert got is not None and got.outputs == {"y": 1}
    assert retired.is_dir() and not (tmp_path / QUARANTINE_DIR).exists()
