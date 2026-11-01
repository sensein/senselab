"""Test support for provenance: a store as the writer would have written it with one entity's attributes changed."""

import json
import tempfile
from pathlib import Path

from senselab_worker.canonical import jcs_dumps

from senselab_core.runtime.provenance import ProvStore, revised


def with_entity_attributes(store: ProvStore, record_id: str, **attributes: object) -> ProvStore:
    """Return a copy of `store` in which one entity's attributes carry `attributes`.

    The entity is minted again under the store's run id with the changed attributes, so its id changes; its
    line is replaced in place and every relation naming the old id names the new one. Every other record,
    every relation and the order of all of them are kept, and the copy is an ordinary `provenance@2.0` store.
    An id held inside another record's attributes or parameters is not rewritten.

    Raises:
        KeyError: if the store holds no entity `record_id`.
    """
    entity = store.get_entity(record_id)
    minter = ProvStore(run_id=store.run_id)
    new_id = minter.entity(
        prov_type=entity.prov_type, extent=entity.extent, attributes=revised(entity.attributes, **attributes)
    )
    replacement = jcs_dumps(minter.get_entity(new_id).as_record())
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "store.jsonl"
        store.write_jsonl(path)
        lines = []
        for line in path.read_bytes().splitlines():
            held = json.loads(line)
            if held.get("record") == "relation" and record_id in (held["source"], held["target"]):
                held.update({k: new_id for k in ("source", "target") if held[k] == record_id})
                line = jcs_dumps(held)
            lines.append(replacement if held.get("id") == record_id else line)
        path.write_bytes(b"".join(line + b"\n" for line in lines))
        return ProvStore.read_jsonl(path, run_id=store.run_id)
