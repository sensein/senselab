"""Test support for provenance: a store as a hand edit of its file would leave it."""

import json
import tempfile
from pathlib import Path

from senselab_worker.canonical import jcs_dumps

from senselab_core.runtime.provenance import UNVERIFIED_FORMAT, ProvStore, revised


def hand_edited(store: ProvStore, record_id: str, **attributes: object) -> ProvStore:
    """Return a copy of `store` in which one entity's attributes carry `attributes`, as a hand-edited file would.

    The store is written, the entity's line is replaced by `revised` of it (its id kept, its `run_id` null),
    the file is labelled `provenance@1.1`, and the file is read back under the store's run id, so every other
    record, every relation and the order of all of them are kept, and the edited record is unverified.

    Raises:
        KeyError: if the store holds no entity `record_id`.
    """
    entity = store.get_entity(record_id)
    edited = revised(entity, attributes=revised(entity.attributes, **attributes))
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "store.jsonl"
        store.write_jsonl(path)
        lines = [jcs_dumps({"format": UNVERIFIED_FORMAT, "record": "format"})]
        for line in path.read_bytes().splitlines()[1:]:
            held = json.loads(line)
            lines.append(jcs_dumps(edited.as_record()) if held.get("id") == record_id else line)
        path.write_bytes(b"".join(line + b"\n" for line in lines))
        return ProvStore.read_jsonl(path, run_id=store.run_id)
