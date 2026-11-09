"""The result store's format gate: later minors read, another version's entries skipped, retired ones refused."""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from planted_objects_fixture import entry_dir, rewrite_json, store_context, store_key
from senselab_core.kernel.identity import ReusePolicy
from senselab_core.runtime.hardware import host_token
from senselab_core.runtime.store import (
    CONTEXT_FORMAT,
    QUARANTINE_DIR,
    RESULT_FORMAT,
    STORE_DIR,
    STORE_MIGRATION_STATEMENT,
    ResultStore,
    StoreFormatRetired,
    StoreFormatUnknown,
)

CONTEXT_3_0_FIELDS = set(
    "env_lock platform wheel_set numerics decoder driver host protocol model_source env_license".split()
)
"""The fields of `execution-context@3.0` as of 3bf258e3: a reader of it requires every one and refuses any other."""

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
def test_an_old_context_with_a_host_field_is_still_read(tmp_path: Path) -> None:
    """A `execution-context@3.0` member carrying `host` is served as written (an entry is never rewritten)."""
    s, k, c = ResultStore(tmp_path), store_key(), store_context()
    s.put(k, {"y": 1}, c)
    entry = entry_dir(tmp_path, k, c)
    rewrite_json(entry / "context.json", lambda r: r.update(format="execution-context@3.0", host="node1"))
    before = (entry / "context.json").read_bytes()
    got = s.get(k, c, ReusePolicy.EXACT)
    assert got is not None and got.outputs == {"y": 1} and got.context.host == "node1"
    assert (entry / "context.json").read_bytes() == before


@pytest.mark.logic
def test_a_written_context_is_one_a_3_0_reader_accepts(tmp_path: Path) -> None:
    """A written context holds exactly the fields of `execution-context@3.0`, `host` among them as a host token."""
    s, k = ResultStore(tmp_path), store_key()
    c = store_context(host=host_token())
    s.put(k, {"y": 1}, c)
    record = json.loads((entry_dir(tmp_path, k, c) / "context.json").read_bytes())
    assert record.pop("format") == CONTEXT_FORMAT == "execution-context@3.1"
    assert set(record) == CONTEXT_3_0_FIELDS and record["host"] == host_token()


@pytest.mark.logic
def test_a_retired_entry_raises_and_stays(tmp_path: Path) -> None:
    """A retired context member raises `StoreFormatRetired` naming the migration; the entry is never moved."""
    s, k, c = ResultStore(tmp_path), store_key(), store_context()
    s.put(k, {"y": 1}, c)
    entry = entry_dir(tmp_path, k, c)
    retired = "execution-context@1.0"
    rewrite_json(entry / "context.json", lambda r: r.update(format=retired))
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
    rewrite_json(retired / "context.json", lambda r: r.update(format="execution-context@1.0"))
    got = s.get(k, mac, ReusePolicy.TOLERANT)
    assert got is not None and got.outputs == {"y": 1}
    assert retired.is_dir() and not (tmp_path / QUARANTINE_DIR).exists()


@pytest.mark.logic
def test_a_stored_execution_context_2_is_refused_as_retired(tmp_path: Path) -> None:
    """An entry whose context is `execution-context@2.0`, written before the env license was context, is refused."""
    s, k, c = ResultStore(tmp_path), store_key(), store_context()
    s.put(k, {"y": 1}, c)
    entry = entry_dir(tmp_path, k, c)
    rewrite_json(entry / "context.json", lambda r: r.update(format="execution-context@2.0"))
    for policy in ReusePolicy:
        with pytest.raises(StoreFormatRetired, match=STORE_MIGRATION_STATEMENT) as info:
            s.get(k, c, policy)
        assert info.value.found == "execution-context@2.0"
    assert entry.is_dir() and not (tmp_path / QUARANTINE_DIR).exists()


@pytest.mark.contract
def test_a_store_dir_holds_one_member_format_pair() -> None:
    """`STORE_DIR` and the result and context members' majors change as one unit (ADR-006, 2026-11-08)."""
    majors = (STORE_DIR, RESULT_FORMAT.rsplit(".", 1)[0], CONTEXT_FORMAT.rsplit(".", 1)[0])
    assert majors == ("store@5", "result@2", "execution-context@3"), (
        "one member pair per STORE_DIR major: a major bump of result or context bumps STORE_DIR; update this pin"
    )
