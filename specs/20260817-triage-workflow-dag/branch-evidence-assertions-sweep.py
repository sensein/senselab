"""Count typed deviations (assertions) per branch across the finished triage corpus.

Read-only. Stdlib only. Writes nothing under the run tree.
"""
import json
import os
import sys
from collections import Counter
from multiprocessing import Pool

ROOT = sys.argv[1]
OUT = sys.argv[2]


def sweep_one(path):
    activities = {}
    entities = []
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
                    if rec.get("prov_type") == "assertion":
                        entities.append((rec["id"], rec.get("attributes") or {}, rec.get("extent")))
                elif kind == "relation":
                    if rec["relation"] == "wasGeneratedBy":
                        generated[rec["source"]] = rec["target"]
                    elif rec["relation"] == "wasInvalidatedBy":
                        invalidated.add(rec["source"])
    except Exception as err:  # noqa: BLE001
        return {"error": "%s: %s" % (type(err).__name__, err)}

    per_branch_type = Counter()
    per_branch = Counter()
    branches_present = set()
    extent_present = Counter()
    for entity_id, attrs, extent in entities:
        if entity_id in invalidated:
            continue
        activity_id = generated.get(entity_id)
        if activity_id is None:
            continue
        branch = activities.get(activity_id)
        if branch not in ("AIRWAY", "SPEECH", "VOICE", "REDACT"):
            continue
        verb = attrs.get("verb")
        name = attrs.get("deviation_type") or attrs.get("claim") or "<none>"
        per_branch_type[(branch, verb, name)] += 1
        per_branch[branch] += 1
        if verb in ("deviate", "contest"):
            branches_present.add(branch)
        if extent:
            extent_present[branch] += 1
    return {
        "per_branch_type": {"|".join([str(p) for p in k]): v for k, v in per_branch_type.items()},
        "per_branch": dict(per_branch),
        "branches_present": sorted(branches_present),
        "extent_present": dict(extent_present),
    }


def find_stores(root):
    found = []
    for dirpath, dirnames, filenames in os.walk(root):
        if "store.jsonl" in filenames:
            found.append(os.path.join(dirpath, "store.jsonl"))
            dirnames[:] = []
    return found


def main():
    stores = find_stores(ROOT)
    total = len(stores)
    per_branch_type = Counter()
    per_branch_assertions = Counter()
    extent_present = Counter()
    recordings_with = Counter()
    recordings_with_speech_or_voice = 0
    recordings_with_any_assertion = 0
    errors = Counter()
    pool = Pool(int(os.environ.get("SWEEP_PROCS", "16")))
    try:
        for result in pool.imap_unordered(sweep_one, stores, chunksize=32):
            if "error" in result:
                errors[result["error"]] += 1
                continue
            for key, value in result["per_branch_type"].items():
                per_branch_type[key] += value
            for branch, value in result["per_branch"].items():
                per_branch_assertions[branch] += value
            for branch, value in result["extent_present"].items():
                extent_present[branch] += value
            present = set(result["branches_present"])
            for branch in present:
                recordings_with[branch] += 1
            if present:
                recordings_with_any_assertion += 1
            if present & {"SPEECH", "VOICE"}:
                recordings_with_speech_or_voice += 1
    finally:
        pool.close()
        pool.join()
    summary = {
        "note": "branches_present counts only verb deviate/contest (typed findings)",
        "root": ROOT,
        "recordings_total": total,
        "recordings_with_any_assertion": recordings_with_any_assertion,
        "recordings_with_speech_or_voice_assertion": recordings_with_speech_or_voice,
        "recordings_with_assertion_by_branch": dict(recordings_with),
        "assertions_by_branch": dict(per_branch_assertions),
        "assertions_with_extent_by_branch": dict(extent_present),
        "assertions_by_branch_verb_name": dict(per_branch_type),
        "parse_errors": dict(errors),
    }
    with open(OUT, "w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
