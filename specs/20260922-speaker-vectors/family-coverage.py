"""What excluding a family costs in coverage, over the whole replayed corpus.

Reads the task-extent census ``census_shard.py`` wrote -- one JSON object per recording, carrying
each extent's ``family`` and its span -- and counts, for each candidate family set, how many
subjects and recordings still supply an extent above the refusal floor. The separation harness
(``family-split.py``) measures quality on the 1,513 subjects that carried four or more extents;
this measures what the other subjects lose.

Usage: ``family_coverage.py <census_dir> [min_extent_s]``
"""

import glob
import json
import sys

FAMILY_SETS = {
    "speech": ("speech",),
    "speech+voice": ("speech", "voice"),
    "all-three": ("speech", "voice", "airway"),
}


def main() -> None:
    """Count subjects and recordings retained under each family set."""
    root = sys.argv[1]
    floor = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0

    by_subject: dict = {}
    recordings = 0
    rec_families: dict = {}
    for path in sorted(glob.glob(root + "/shard-*.jsonl")):
        with open(path) as fh:
            for line in fh:
                record = json.loads(line)
                if not record.get("has_store"):
                    continue
                recordings += 1
                subject = record["subject"]
                held = by_subject.setdefault(subject, set())
                present = set()
                for extent in record.get("extents") or []:
                    if extent["end_s"] - extent["start_s"] < floor:
                        continue
                    held.add(extent["family"])
                    present.add(extent["family"])
                for name, allowed in FAMILY_SETS.items():
                    if present & set(allowed):
                        rec_families[name] = rec_families.get(name, 0) + 1

    subjects = len(by_subject)
    print(f"census: {recordings} recordings with a store, {subjects} subjects, floor {floor} s")
    hdr = (
        f"{'family set':>14} {'subjects kept':>14} {'% of subjects':>14} "
        f"{'recordings kept':>16} {'% of recordings':>16}"
    )
    print(hdr)
    print("-" * len(hdr))
    for name, allowed in FAMILY_SETS.items():
        kept = sum(1 for held in by_subject.values() if held & set(allowed))
        rkept = rec_families.get(name, 0)
        print(
            f"{name:>14} {kept:14d} {100 * kept / max(subjects, 1):13.2f}% "
            f"{rkept:16d} {100 * rkept / max(recordings, 1):15.2f}%"
        )

    speech_only = {s for s, held in by_subject.items() if "speech" in held}
    any_family = {s for s, held in by_subject.items() if held}
    print(f"\nsubjects with an extent but none in `speech`: {len(any_family - speech_only)}")
    lost = [held for s, held in by_subject.items() if held and "speech" not in held]
    composition: dict = {}
    for held in lost:
        key = "+".join(sorted(held))
        composition[key] = composition.get(key, 0) + 1
    print(f"  their family sets: {dict(sorted(composition.items(), key=lambda kv: -kv[1]))}")


if __name__ == "__main__":
    main()
