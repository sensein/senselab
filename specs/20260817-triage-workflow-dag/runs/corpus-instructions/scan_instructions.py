"""Read-only reduction over the b2ai adult BIDS tree.

Emits a JSON summary: per-family instruction multisets, per-task-id instruction
multisets, the two-grain disagreement table, and family-declaration coverage.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from multiprocessing import Pool

_TASK = re.compile(r"_task-(?P<task>[^_]+)$")
_TRAILING_INDEX = re.compile(r"(?:-\d+)+$")


def task_id_of(stem: str) -> str:
    """The sanitized task id in a BIDS stem, or ``"unknown"``.

    Args:
        stem: A BIDS stem such as ``sub-a_ses-b_task-prolonged-vowel``.

    Returns:
        The lowercased ``task-`` token.
    """
    m = _TASK.search(stem)
    return m.group("task").lower() if m else "unknown"


def task_family(task_id: str) -> str:
    """The family a task id collapses into.

    Args:
        task_id: A sanitized task id.

    Returns:
        The id with every trailing numeric segment removed.
    """
    return _TRAILING_INDEX.sub("", task_id)


def scan_subject(subdir: str) -> dict:
    """Reduce one subject directory to its counters.

    Args:
        subdir: Absolute path to a ``sub-*`` directory.

    Returns:
        A mapping of counter name to plain dicts, mergeable by :func:`merge`.
    """
    rec_by_family_instr = defaultdict(Counter)
    rec_by_taskid_instr = defaultdict(Counter)
    rec_by_family_stim = defaultdict(Counter)
    rec_by_famlang_instr = defaultdict(Counter)
    lang_by_family = defaultdict(Counter)
    dur_by_family = defaultdict(Counter)
    taskid_counts = Counter()
    family_counts = Counter()
    wav_by_taskid = Counter()
    grain = Counter()
    grain_by_family = defaultdict(Counter)
    at_instr_by_name = defaultdict(Counter)
    at_stim_by_name = defaultdict(Counter)
    speech_type_by_family = defaultdict(Counter)
    link_miss = 0
    n_at = 0

    for root, _dirs, files in os.walk(subdir):
        if os.path.basename(root) != "audio":
            continue
        atasks = {}
        recs = []
        for fn in files:
            if fn.endswith(".wav"):
                wav_by_taskid[task_id_of(fn[:-4])] += 1
            elif fn.endswith("_acoustictask-metadata.json"):
                p = os.path.join(root, fn)
                try:
                    with open(p) as fh:
                        d = json.load(fh)
                except Exception:
                    continue
                n_at += 1
                name = task_id_of(fn[: -len("_acoustictask-metadata.json")])
                instr = (d.get("instructions") or "").strip()
                stim = (d.get("stimulus_text") or "").strip()
                at_instr_by_name[name][instr] += 1
                at_stim_by_name[name][stim] += 1
                aid = d.get("acoustic_task_id")
                if aid:
                    atasks[aid] = (instr, stim)
            elif fn.endswith("_recording-metadata.json"):
                recs.append((root, fn))

        for root2, fn in recs:
            p = os.path.join(root2, fn)
            try:
                with open(p) as fh:
                    d = json.load(fh)
            except Exception:
                continue
            stem = fn[: -len("_recording-metadata.json")]
            tid = task_id_of(stem)
            fam = task_family(tid)
            taskid_counts[tid] += 1
            family_counts[fam] += 1
            instr = (d.get("instructions") or "").strip()
            stim = (d.get("stimulus_text") or "").strip()
            rec_by_family_instr[fam][instr] += 1
            rec_by_taskid_instr[tid][instr] += 1
            rec_by_family_stim[fam][stim] += 1
            speech_type_by_family[fam][str(d.get("speech_type"))] += 1
            lang = str(d.get("language") or "?")
            lang_by_family[fam][lang] += 1
            rec_by_famlang_instr[fam + "\t" + lang][instr] += 1
            dur = d.get("recording_duration")
            try:
                dur_by_family[fam][str(int(float(dur)))] += 1
            except (TypeError, ValueError):
                dur_by_family[fam]["?"] += 1

            aid = d.get("recording_acoustic_task_id")
            if aid not in atasks:
                link_miss += 1
                continue
            a_instr, a_stim = atasks[aid]
            if stim == a_stim:
                b = "stim_same_both_empty" if not stim else "stim_same_both_text"
            elif not a_stim:
                b = "stim_at_empty_rec_text"
            elif not stim:
                b = "stim_rec_empty_at_text"
            else:
                b = "stim_both_text_differ"
            grain[b] += 1
            grain_by_family[fam][b] += 1
            ib = "instr_same" if instr == a_instr else "instr_differ"
            grain[ib] += 1
            grain_by_family[fam][ib] += 1
            grain["linked"] += 1
            same_stim = b in ("stim_same_both_empty", "stim_same_both_text")
            if not same_stim:
                grain["stim_differ_any"] += 1
            if ib == "instr_differ" or not same_stim:
                grain["either_differs"] += 1

    return {
        "rec_by_family_instr": {k: dict(v) for k, v in rec_by_family_instr.items()},
        "rec_by_famlang_instr": {k: dict(v) for k, v in rec_by_famlang_instr.items()},
        "lang_by_family": {k: dict(v) for k, v in lang_by_family.items()},
        "dur_by_family": {k: dict(v) for k, v in dur_by_family.items()},
        "rec_by_taskid_instr": {k: dict(v) for k, v in rec_by_taskid_instr.items()},
        "rec_by_family_stim": {k: dict(v) for k, v in rec_by_family_stim.items()},
        "taskid_counts": dict(taskid_counts),
        "family_counts": dict(family_counts),
        "wav_by_taskid": dict(wav_by_taskid),
        "grain": dict(grain),
        "grain_by_family": {k: dict(v) for k, v in grain_by_family.items()},
        "at_instr_by_name": {k: dict(v) for k, v in at_instr_by_name.items()},
        "at_stim_by_name": {k: dict(v) for k, v in at_stim_by_name.items()},
        "speech_type_by_family": {k: dict(v) for k, v in speech_type_by_family.items()},
        "link_miss": link_miss,
        "n_acoustictask": n_at,
    }


def merge(dst: dict, src: dict) -> None:
    """Accumulate one subject's counters into the running total.

    Args:
        dst: The running total, mutated in place.
        src: One :func:`scan_subject` result.
    """
    nested = (
        "rec_by_family_instr",
        "rec_by_taskid_instr",
        "rec_by_family_stim",
        "rec_by_famlang_instr",
        "lang_by_family",
        "dur_by_family",
        "grain_by_family",
        "at_instr_by_name",
        "at_stim_by_name",
        "speech_type_by_family",
    )
    for key in nested:
        d = dst.setdefault(key, {})
        for k, sub in src[key].items():
            t = d.setdefault(k, {})
            for k2, n in sub.items():
                t[k2] = t.get(k2, 0) + n
    for key in ("taskid_counts", "family_counts", "wav_by_taskid", "grain"):
        d = dst.setdefault(key, {})
        for k, n in src[key].items():
            d[k] = d.get(k, 0) + n
    dst["link_miss"] = dst.get("link_miss", 0) + src["link_miss"]
    dst["n_acoustictask"] = dst.get("n_acoustictask", 0) + src["n_acoustictask"]


def main() -> int:
    """Walk every subject and write the merged counters as JSON.

    Returns:
        A process exit status.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("out")
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    subs = sorted(os.path.join(args.root, d) for d in os.listdir(args.root) if d.startswith("sub-"))
    if args.limit:
        subs = subs[: args.limit]
    print(f"{len(subs)} subjects, {args.procs} procs", file=sys.stderr, flush=True)

    total: dict = {}
    done = 0
    with Pool(args.procs) as pool:
        for res in pool.imap_unordered(scan_subject, subs, chunksize=4):
            merge(total, res)
            done += 1
            if done % 200 == 0:
                print(f"  {done}/{len(subs)}", file=sys.stderr, flush=True)
    total["n_subjects"] = len(subs)
    with open(args.out, "w") as fh:
        json.dump(total, fh)
    print("wrote", args.out, file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
