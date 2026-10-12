"""Summarise the r37 words/speaker measurement (fold.py's rows) into results.json. No transcript text."""

import collections
import glob
import json
import statistics
import sys

from senselab.audio.workflows.triage.decision import reason_of
from senselab.audio.workflows.triage.routing_analysis.families import (
    AIRWAY_ELICITING,
    SYLLABLE_REPETITION,
    VOICE_ELICITING,
)
from senselab.audio.workflows.triage.task_speech import item_set_family, narrative_family

rows = [json.loads(line) for f in sorted(glob.glob(sys.argv[1])) for line in open(f)]
labels = json.load(open(sys.argv[2]))
GATE = "gate:dominant_speaker_share_min"
R: dict = {"rows": len(rows)}
R["errors"] = collections.Counter(k for r in rows for k in r if k.startswith("error"))
ok = [r for r in rows if "A" in r and "B" in r and r.get("r36")]
R["ok"] = len(ok)


def group(family):
    if family in VOICE_ELICITING:
        return "voice"
    if family in AIRWAY_ELICITING:
        return "airway"
    if family in SYLLABLE_REPETITION:
        return "ddk"
    if item_set_family(family):
        return "item_set"
    if narrative_family(family):
        return "narrative"
    return "ordered"


def half(r):
    return int(r["participant"][4:6], 16) % 2


def os_keys(d):
    return [k for k in d.get("grounds") or () if reason_of(k) == "other_speaker"]


def has_os(d):
    return "other_speaker" in (d.get("reasons") or ())


def axes(d):
    return f"{d.get('triage')}/{d.get('release')}"


def count_by(pred, rs=ok):
    c = collections.Counter(group(r["family"]) for r in rs if pred(r))
    c["total"] = sum(c.values())
    return dict(c)


def halves(pred, rs=ok):
    return [sum(1 for r in rs if half(r) == h and pred(r)) for h in (0, 1)]


# transitions
R["transitions_r36_A"] = collections.Counter(
    f"{axes(r['r36'])} -> {axes(r['A'])}" for r in ok if axes(r["r36"]) != axes(r["A"])
).most_common(25)
R["transitions_A_B"] = collections.Counter(
    f"{axes(r['A'])} -> {axes(r['B'])}" for r in ok if axes(r["A"]) != axes(r["B"])
).most_common(25)
R["verdict_counts"] = {k: dict(collections.Counter(r[k]["triage"] for r in ok)) for k in ("r36", "A", "B")}
R["release_counts"] = {k: dict(collections.Counter(str(r[k]["release"]) for r in ok)) for k in ("r36", "A", "B")}

# B1
b1 = [r for r in ok if r.get("B1")]
R["B1"] = {
    "recordings_with_disagreement": sum(1 for r in b1 if r["B1"]["disagreed_n"]),
    "words_disagreed": sum(r["B1"]["disagreed_n"] for r in b1),
    "recordings_retimed": sum(1 for r in b1 if r["B1"]["retimed_n"]),
    "words_retimed": sum(r["B1"]["retimed_n"] for r in b1),
    "recordings_unlocated": sum(1 for r in b1 if r["B1"]["unlocated_n"]),
    "words_unlocated": sum(r["B1"]["unlocated_n"] for r in b1),
    "won_by_source": dict(
        sum((collections.Counter(r["B1"]["won_by_source"] or {}) for r in b1), collections.Counter())
    ),
    "voice_held_phonation_recordings": sum(1 for r in b1 if r["B1"].get("held_phonation")),
}
narrowed = [r for r in ok if r["B"]["masked_s"] < r["A"]["masked_s"] - 0.01]
widened = [r for r in ok if r["B"]["masked_s"] > r["A"]["masked_s"] + 0.01]
R["B1"]["masks_narrowed"] = len(narrowed)
R["B1"]["masks_widened"] = len(widened)
R["B1"]["masked_s_removed_median"] = (
    round(statistics.median(r["A"]["masked_s"] - r["B"]["masked_s"] for r in narrowed), 3) if narrowed else None
)
R["B1"]["masked_s_removed_total"] = round(sum(r["A"]["masked_s"] - r["B"]["masked_s"] for r in narrowed), 1)
R["B1"]["release_to_withheld_with_unlocated"] = sum(
    1
    for r in ok
    if r["B"]["release"] == "withheld" and r["A"]["release"] != "withheld" and (r.get("B1") or {}).get("unlocated_n")
)
R["B1"]["release_reason_changes_A_B"] = collections.Counter(
    f"{r['A']['release_reason']} -> {r['B']['release_reason']}"
    for r in ok
    if r["A"]["release_reason"] != r["B"]["release_reason"]
).most_common(12)
R["B1"]["by_half_retimed_recordings"] = halves(lambda r: (r.get("B1") or {}).get("retimed_n"))
R["B3"] = {
    "recordings_post_task_words": sum(1 for r in ok if (r.get("B3") or {}).get("post_task_n")),
    "post_task_words": sum((r.get("B3") or {}).get("post_task_n") or 0 for r in ok),
    "speech_words_increased": sum(
        1 for r in ok if r.get("B3") and (r["B3"]["words_n_new"] or 0) > (r["B3"]["words_n_old"] or 0)
    ),
}

# C1
fires = lambda d: GATE in (d.get("grounds") or ())  # noqa: E731
down = lambda d: "speaker_gate_recording_condition" in (d.get("annotations") or ())  # noqa: E731
R["C1"] = {
    "r36_fires": count_by(lambda r: fires(r["r36"])),
    "A_fires": count_by(lambda r: fires(r["A"])),
    "B_fires": count_by(lambda r: fires(r["B"])),
    "B_fires_uncorroborated_kept_in_review": sum(1 for r in ok if fires(r["B"]) and not r.get("signals")),
    "B_fires_corroborated": sum(1 for r in ok if fires(r["B"]) and r.get("signals")),
    "downgraded_to_annotation": sum(1 for r in ok if down(r["B"])),
    "fires_stopped": sum(1 for r in ok if fires(r["r36"]) and not fires(r["B"]) and not down(r["B"])),
    "fires_new": sum(1 for r in ok if fires(r["B"]) and not fires(r["r36"])),
    "by_half_r36_B_fires": [halves(lambda r: fires(r["r36"])), halves(lambda r: fires(r["B"]))],
    "recomputed_share_matches_r36": sum(
        1
        for r in ok
        for c in r.get("C1") or ()
        if c["old"] is not None and c["old_recomputed"] is not None and abs(c["old"] - c["old_recomputed"]) < 1e-3
    ),
    "share_readings": sum(1 for r in ok for c in r.get("C1") or () if c["old"] is not None),
    "hot_or_clipped_recordings": sum(1 for r in ok if r.get("hot_or_clipped")),
    "fires_stopped_corroborated": sum(1 for r in ok if fires(r["r36"]) and not fires(r["B"]) and r.get("signals")),
}

# C2
so_alone = lambda d: os_keys(d) == ["second_opinion:other_voice"]  # noqa: E731
R["C2"] = {
    "other_voice_to_off_task_A": sum(1 for r in ok if "second_opinion:other_voice_participant" in r["A"]["grounds"]),
    "other_speaker_lost_by_C2_A": sum(
        1
        for r in ok
        if "second_opinion:other_voice_participant" in r["A"]["grounds"] and has_os(r["r36"]) and not has_os(r["A"])
    ),
    "r36_other_speaker_on_second_opinion_alone": count_by(lambda r: so_alone(r["r36"])),
    "r36_so_alone_cohort_matched": sum(1 for r in ok if so_alone(r["r36"]) and r["readers"]["participant_matched"]),
}

# G1 / G2 / G3
R["G1"] = {
    "off_task_aside_A": count_by(lambda r: "off_task_aside" in r["A"]["annotations"]),
    "reviewer_off_task_flag_r36_to_aside": sum(
        1 for r in ok if "reviewer_off_task_speech" in r["r36"]["grounds"] and "off_task_aside" in r["A"]["annotations"]
    ),
    "aside_release_r36_A": collections.Counter(
        f"{r['r36']['release']} -> {r['A']['release']}" for r in ok if "off_task_aside" in r["A"]["annotations"]
    ).most_common(),
    "aside_verdict_r36_A": collections.Counter(
        f"{r['r36']['triage']} -> {r['A']['triage']}" for r in ok if "off_task_aside" in r["A"]["annotations"]
    ).most_common(),
    "identifier_disagreement_r36": sum(
        1 for r in ok if "second_opinion:policy_identifier_present" in r["r36"]["grounds"]
    ),
    "identifier_disagreement_A": sum(1 for r in ok if "second_opinion:policy_identifier_present" in r["A"]["grounds"]),
}
nf = [r for r in ok if r["r36"].get("release_reason") == "second_opinion_not_free"]
R["G2"] = {
    "r36_withheld_second_opinion_not_free": len(nf),
    "A_still_withheld_corroborated": sum(1 for r in nf if r["A"]["release_reason"] == "second_opinion_not_free"),
    "A_release_kept": collections.Counter(
        f"{r['A']['triage']}/{r['A']['release']} ({r['A']['release_reason']})"
        for r in nf
        if r["A"]["release_reason"] != "second_opinion_not_free"
    ).most_common(8),
    "A_not_free_flag": sum(1 for r in ok if "second_opinion:masked_text_free_of_identifiers" in r["A"]["grounds"]),
    "by_half_released": halves(lambda r: r in nf and r["A"]["release_reason"] != "second_opinion_not_free", nf),
}
R["G3"] = count_by(lambda r: "speech_before_task" in r["A"]["annotations"])

# other_speaker by family group
R["other_speaker_by_group"] = {k: count_by(lambda r, k=k: has_os(r[k])) for k in ("r36", "A", "B")}
R["other_speaker_by_half"] = {k: halves(lambda r, k=k: has_os(r[k])) for k in ("r36", "A", "B")}

# C3
items = [
    i
    for r in ok
    for i in r["B"]["speaker_items"]
    if reason_of(i["name"]) == "other_speaker" and i["effect"] in ("review", "annotation")
]
R["C3"] = {
    "other_speaker_items": len(items),
    "with_source": sum(1 for i in items if i["source"]),
    "with_model": sum(1 for i in items if i["model"]),
    "with_spans": sum(1 for i in items if i["spans_n"]),
    "without_model_by_name": collections.Counter(i["name"] for i in items if not i["model"]).most_common(8),
    "without_spans_by_name": collections.Counter(i["name"] for i in items if not i["spans_n"]).most_common(8),
}

# labels
by_stem = {}
for r in ok:
    by_stem.setdefault(r["stem"].rsplit("_", 1)[0], r)
agree = []
for stem, expect in labels.items():
    r = by_stem.get(stem)
    if r is None:
        agree.append([stem[4:12], expect, "missing"])
        continue
    b = r["B"]
    checks = {
        "no_other_speaker": not has_os(b),
        "other_speaker_kept": has_os(b),
        "off_task_not_other": not has_os(b) and "off_task_speech" in b["reasons"],
        "cleared": b["triage"] == "pass" and b["release"] == "as_is",
        "release_kept": b["release"] != "withheld",
        "withheld_not_pass": b["release"] == "withheld" and b["triage"] != "pass",
        "pass_not_withheld": b["triage"] == "pass" and b["release"] != "withheld",
        "speech_before_task": "speech_before_task" in b["annotations"],
        "masks_narrowed": b["masked_s"] < r["A"]["masked_s"] - 0.01,
        "review": b["triage"] == "review",
    }
    agree.append(
        [
            stem[4:12] + " " + r["family"],
            expect,
            checks[expect],
            f"{axes(r['r36'])}:{r['r36']['reason']} -> {axes(b)}:{b['reason']}",
        ]
    )
R["labels"] = agree
R["labels_met"] = f"{sum(1 for a in agree if a[2] is True)}/{len(agree)}"
json.dump(R, open(sys.argv[3], "w"), indent=1, default=str)
print(json.dumps(R, indent=1, default=str))
