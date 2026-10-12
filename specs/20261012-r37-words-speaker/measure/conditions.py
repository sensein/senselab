"""Each participant's recording condition from pass 1's rows, by COHORT's own ``condition_summary``.

    python conditions.py "out/cond.*.jsonl" conditions.json
"""

import glob
import json
import sys

from senselab.audio.workflows.triage.cohort_fold import hot_or_clipped
from senselab.audio.workflows.triage.cohort_stage import RecordingFacts, condition_summary

by: dict[str, list[RecordingFacts]] = {}
for path in glob.glob(sys.argv[1]):
    for line in open(path):
        r = json.loads(line)
        if "error" in r:
            continue
        by.setdefault(r["participant"], []).append(
            RecordingFacts(
                stem=r["stem"],
                session=None,
                family="",
                lexical_task=False,
                duration_s=None,
                task_extent=None,
                rms_dbfs=r["rms_dbfs"],
                clipped_s=r["clipped_s"],
                snr_db=r.get("snr_db"),
            )
        )
out = {p: condition_summary(members) for p, members in by.items()}
json.dump(out, open(sys.argv[2], "w"))
hot = [p for p, s in out.items() if hot_or_clipped({"condition": {"participant": s}})]
print(len(out), "participants;", len(hot), "hot_or_clipped;", sum(len(by[p]) for p in hot), "recordings")
for key in ("SUBJ08", "0482bc77", "168c5e68"):
    print(key, [hot_or_clipped({"condition": {"participant": s}}) for p, s in out.items() if key in p])
