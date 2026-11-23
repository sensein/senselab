"""The senselab store identity: a checksum with pydra's absence suffix."""

import hashlib
from types import SimpleNamespace

import pytest
from senselab_core.kernel.identity import LogicalKey
from senselab_core.runtime.execution import portable
from senselab_core.runtime.execution.run import RESULTS_DIR
from senselab_core.runtime.store import key_record
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
def test_a_job_with_absent_values_keys_its_entry_by_its_logical_key_and_their_triples(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The entry's key is the logical key's fields, plus `absent` only when pydra recorded absence triples."""
    monkeypatch.setattr(portable, "doc_of", lambda task: task)
    task, identity = SimpleNamespace(key=KEY), portable.SenselabIdentity()
    assert identity.key(SimpleNamespace(task=task, absent=[])) == key_record(KEY)
    keyed = identity.key(SimpleNamespace(task=task, absent=sorted(TRIPLES)))
    assert keyed == {**key_record(KEY), "absent": FORM}  # type: ignore[dict-item]
    assert identity.check_key(KEY.digest() + SUFFIX, keyed)


@pytest.mark.logic
def test_the_default_result_store_is_named_by_its_identity() -> None:
    """A new store identity reads no root an earlier one wrote: the default root carries its name."""
    assert RESULTS_DIR.endswith(portable.SenselabIdentity.name)
