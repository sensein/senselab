"""The senselab store identity and lease: a checksum with pydra's absence suffix, and the profile's lease bounds."""

import hashlib
import os
from pathlib import Path

import pytest
from pool_stack_fixture import relabel_store
from senselab_core.kernel.identity import LogicalKey
from senselab_core.runtime.execution import portable, warm
from senselab_core.runtime.store import key_record
from senselab_core.testing.pools import POOL_PROFILE
from senselab_worker.canonical import jcs_dumps

KEY = LogicalKey(
    task="t.score",
    task_version="1",
    backend="score",
    backend_version="1",
    model=None,
    params={},
    backend_params={},
    inputs={},
)
TRIPLES = [("values", 1, "silent"), ("values", 0, "silent")]
FORM = [["values", {"$int": "0"}, "silent"], ["values", {"$int": "1"}, "silent"]]
"""`TRIPLES` as a key's `absent` holds them: sorted, each index an `$int`."""
SUFFIX = "-absent-" + hashlib.sha256(jcs_dumps(FORM)).hexdigest()
"""pydra's absence suffix of `TRIPLES`: the hex sha256 of their canonical JSON form."""


@pytest.mark.logic
def test_a_checksum_with_an_absence_suffix_is_the_key_digest_plus_the_suffix_of_its_absent() -> None:
    """A key with `absent` matches its digest plus that suffix and nothing else; a key without one, its digest."""
    identity, digest, record = portable.SenselabIdentity(), KEY.digest(), key_record(KEY)
    assert isinstance(record, dict)
    suffixed = {**record, "absent": FORM}
    assert identity.check_key(digest + SUFFIX, suffixed)
    assert identity.check_key(digest, record)
    refused = [
        (digest, suffixed),
        (digest + SUFFIX, record),
        (digest + SUFFIX, {**record, "absent": FORM[::-1]}),
        (digest, {**record, "absent": []}),
        (digest + SUFFIX, {**record, "absent": "values"}),
        (digest + SUFFIX, {**record, "task_version": "2", "absent": FORM}),
    ]
    assert [identity.check_key(c, k) for c, k in refused] == [False] * len(refused)


@pytest.mark.logic
def test_a_job_with_absent_values_keys_its_entry_by_its_logical_key_and_their_triples() -> None:
    """The entry's key is the logical key's fields, plus `absent` only when pydra recorded absence triples."""
    assert portable.entry_key(KEY, []) == key_record(KEY)
    keyed = portable.entry_key(KEY, TRIPLES)
    assert keyed == {**key_record(KEY), "absent": FORM}  # type: ignore[dict-item]
    assert portable.SenselabIdentity().check_key(KEY.digest() + SUFFIX, keyed)


@pytest.mark.logic
@pytest.mark.parametrize("name", ["senselab-identity@3", "senselab-identity@0", "pydra-identity@2", "other"])
def test_a_store_of_any_other_identity_is_refused_and_left_as_it_was(tmp_path: Path, name: str) -> None:
    """Only a `senselab-identity@1` header is upgraded; a newer, older or foreign scheme is a layout mismatch."""
    mismatch = pytest.importorskip("pydra.engine.portable.layout").StoreLayoutMismatch
    portable.open_store(tmp_path / "store", tmp_path / "work")
    header = relabel_store(tmp_path / "store", name)
    written = header.read_bytes()
    with pytest.raises(mismatch, match="identity"):
        portable.open_store(tmp_path / "store", tmp_path / "work")
    assert header.read_bytes() == written


def _previous_store(tmp_path: Path) -> Path:
    """Make a store at `tmp_path/store` whose header names `senselab-identity@1`; return its header."""
    portable.open_store(tmp_path / "store", tmp_path / "work")
    return relabel_store(tmp_path / "store", "senselab-identity@1")


@pytest.mark.logic
@pytest.mark.parametrize("window", ["senselab-identity@2", "senselab-identity@1"])
def test_an_opener_racing_another_processs_upgrade_opens_the_upgraded_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, window: str
) -> None:
    """Another process upgrades the header while this one opens: after its @2 open was refused, or as its @1 open reads.

    Either way the open follows the header it finds and ends on the upgraded store, never a layout mismatch.
    """
    store_module = pytest.importorskip("pydra.engine.portable.store")
    header = _previous_store(tmp_path)
    real = store_module.open_layout
    raced: list[str] = []

    def racing(handle: object, expected: object, *args: object, **kwargs: object) -> object:
        configured = expected.identity["name"]  # type: ignore[attr-defined]
        if not raced and configured == window:
            raced.append(configured)
            if window == "senselab-identity@1":
                portable.upgrade_header(tmp_path / "store")
            else:
                try:
                    return real(handle, expected, *args, **kwargs)
                finally:
                    portable.upgrade_header(tmp_path / "store")
        return real(handle, expected, *args, **kwargs)

    monkeypatch.setattr(store_module, "open_layout", racing)
    store = portable.open_store(tmp_path / "store", tmp_path / "work")
    assert raced == [window] and store.identity.name == "senselab-identity@2"
    assert b"senselab-identity@2" in header.read_bytes()


@pytest.mark.logic
def test_an_upgraded_header_keeps_its_mode_whatever_the_umask(tmp_path: Path) -> None:
    """The header's mode is set through its descriptor, so an upgrader's umask never narrows it for the group."""
    header = _previous_store(tmp_path)
    header.chmod(0o444)
    previous = os.umask(0o077)
    try:
        portable.open_store(tmp_path / "store", tmp_path / "work")
    finally:
        os.umask(previous)
    assert b"senselab-identity@2" in header.read_bytes()
    assert header.stat().st_mode & 0o777 == 0o444


@pytest.mark.logic
def test_a_refused_upgrade_serves_the_previous_store_unchanged_and_says_so(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A root that looks writable but refuses the header's rename is read under @1, header unchanged, logged once."""
    header = _previous_store(tmp_path)
    written, names = header.read_bytes(), sorted(p.name for p in header.parent.iterdir())
    replace = os.replace

    def refusing(source: str, target: str, **dirs: int) -> None:
        if str(target).endswith("pydra-store.json"):
            raise PermissionError(13, "refused", str(target))
        replace(source, target, **dirs)

    monkeypatch.setattr(os, "replace", refusing)
    store = portable.open_store(tmp_path / "store", tmp_path / "work")
    assert store.identity.name == "senselab-identity@1" and header.read_bytes() == written
    assert [r.levelname for r in caplog.records if "not upgraded" in r.getMessage()] == ["WARNING"]
    assert sorted(p.name for p in header.parent.iterdir()) == names


@pytest.mark.logic
def test_the_store_leases_are_bounded_by_the_profiles_lease_section(tmp_path: Path) -> None:
    """A session's store takes, beats and waits on its job leases as the profile's `lease` says, as its pool does."""
    bounds = warm.lease_settings(POOL_PROFILE)
    lease = POOL_PROFILE.lease
    assert (bounds.stale_after, bounds.renew_fraction, bounds.attribute_delay) == (
        lease.stale_after_s,
        lease.renew_fraction,
        lease.attribute_delay_s,
    )
    assert bounds.timeout == max(lease.wait_s, lease.stale_after_s + lease.attribute_delay_s)
    assert warm.settings(POOL_PROFILE).lease == bounds
    store = portable.open_store(tmp_path / "store", tmp_path / "work", bounds)
    assert store.lease_settings == bounds and store.lease_settings.holder == "anonymous"
