"""Where a branch's typed findings land in the report's 4-item printed evidence cap.

Replicates _branch_evidence's item set and ordering for SPEECH and VOICE. Read-only, stdlib only.
"""
import json
import os
import sys
from collections import Counter
from multiprocessing import Pool

ROOT = sys.argv[1]
OUT = sys.argv[2]
CAP = 4
FAMILY = {"SPEECH": "speech", "VOICE": "voice"}


def sweep_one(path):
    activities = {}
    entities = {}
    generated = {}
    invalidated = set()
    try:
        with open(path) as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                kind = rec.get("record")
                if kind == "activity":
                    activities[rec["id"]] = rec.get("node")
                elif kind == "entity":
                    entities[rec["id"]] = rec
                elif kind == "relation":
                    if rec["relation"] == "wasGeneratedBy":
                        generated[rec["source"]] = rec["target"]
                    elif rec["relation"] == "wasInvalidatedBy":
                        invalidated.add(rec["source"])
    except Exception as err:  # noqa: BLE001
        return {"error": "%s: %s" % (type(err).__name__, err)}

    out = {}
    for branch in ("SPEECH", "VOICE"):
        items = {}
        for entity_id, rec in entities.items():
            if entity_id in invalidated:
                continue
            prov_type = rec.get("prov_type")
            attrs = rec.get("attributes") or {}
            activity_id = generated.get(entity_id)
            node = activities.get(activity_id) if activity_id else None
            own = node == branch and prov_type in ("span", "measurement", "assertion")
            if own and prov_type == "assertion" and attrs.get("verb") not in ("deviate", "contest"):
                own = False
            source = prov_type == "span" and attrs.get("family") == FAMILY[branch]
            if not own and not source:
                continue
            extent = rec.get("extent")
            start = float(extent[0]) if extent else -1.0
            items[entity_id] = (start, entity_id, prov_type == "assertion")
        ordered = sorted(items.values())
        typed_ranks = [index for index, item in enumerate(ordered) if item[2]]
        if not typed_ranks:
            continue
        out[branch] = {
            "n_items": len(ordered),
            "n_typed": len(typed_ranks),
            "visible": sum(1 for rank in typed_ranks if rank < CAP),
        }
    return out


def find_stores(root):
    found = []
    for dirpath, dirnames, filenames in os.walk(root):
        if "store.jsonl" in filenames:
            found.append(os.path.join(dirpath, "store.jsonl"))
            dirnames[:] = []
    return found


def main():
    stores = find_stores(ROOT)
    tally = Counter()
    errors = Counter()
    pool = Pool(int(os.environ.get("SWEEP_PROCS", "32")))
    try:
        for result in pool.imap_unordered(sweep_one, stores, chunksize=32):
            if "error" in result:
                errors[result["error"]] += 1
                continue
            if not result:
                continue
            tally["recordings_with_typed"] += 1
            any_visible = False
            for branch, row in result.items():
                tally["%s_recordings_with_typed" % branch] += 1
                tally["%s_typed_total" % branch] += row["n_typed"]
                tally["%s_typed_visible" % branch] += row["visible"]
                if row["visible"]:
                    tally["%s_recordings_with_one_visible" % branch] += 1
                    any_visible = True
                else:
                    tally["%s_recordings_with_none_visible" % branch] += 1
            if any_visible:
                tally["recordings_with_one_visible"] += 1
            else:
                tally["recordings_with_none_visible"] += 1
    finally:
        pool.close()
        pool.join()
    summary = {"root": ROOT, "cap": CAP, "tally": dict(tally), "parse_errors": dict(errors)}
    with open(OUT, "w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
