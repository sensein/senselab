"""Pass 2 of the r37 words/speaker measurement: the r36 decision against two in-memory re-folds, read-only.

    python fold.py MANIFEST K N CONDITIONS.json HINTS_DIR CONFIG.yaml OUT_PREFIX

Per run, three readings of the same r36 store, none written back:

- ``r36``: the decision the store holds.
- ``A``: VERDICT at this commit over the store as it stands, the speaker gate fed the r36 overlap-inclusive share
  (so the gate fires as in r36). Moves the fold-level rules only (C2, C3, G1-G3, and whatever else the branch holds).
- ``B``: as A, after B1 (word timing arbitration), B3 (the task-speech reading re-read), C1 (the exclusive share)
  and the participant's recording condition (from pass 1) are written into the in-memory store.

Writes one JSON line per run. No transcript text: counts, keys, spans and probabilities only.
"""

import json
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from senselab.audio.workflows.triage.cohort_fold import (
    COHORT_READING,
    hot_or_clipped,
    nonmatch_spans,
    participant_matched,
    pool_block,
)
from senselab.audio.workflows.triage.config import load_triage_config
from senselab.audio.workflows.triage.extend import load_hint_builder, read_store, run_root_of, source_of
from senselab.audio.workflows.triage.live_evidence import declared_task, recording_stem
from senselab.audio.workflows.triage.nodes.common import (
    find_measurement,
    find_measurements,
    live_entities,
    software_agent,
    write_measurement,
)
from senselab.audio.workflows.triage.nodes.redact import fold_mask_plan
from senselab.audio.workflows.triage.nodes.verdict import separated_speakers_n, verdict
from senselab.audio.workflows.triage.speaker_share import speakers_within
from senselab.audio.workflows.triage.task_speech import TASK_SPEECH_READING, reads_task_speech, write_task_speech
from senselab.audio.workflows.triage.vocabulary import (
    REDACTION_LLM_ANNOTATION,
    FoldPolicy,
    opinion_reads_other_voice,
    release_ground_key,
    reviewer_may_unmask,
    second_voice_signals,
)
from senselab.audio.workflows.triage.word_arbitration import WORD_ARBITRATION, arbitrate_words

OLD_SHARE = "extent_dominant_speaker_share"
NEW_SHARE = "extent_dominant_exclusive_share"
STATE: dict[str, Any] = {}


def _init(conditions_path: str, hints: str, config_path: str) -> None:
    STATE["conditions"] = json.load(open(conditions_path))
    STATE["hint"] = load_hint_builder(Path(hints))
    STATE["config"] = load_triage_config(Path(config_path))
    STATE["exclusive_min_s"] = float(STATE["config"].require("speech.secondary_exclusive_min_s"))


def _activity(store: Any, step: str) -> tuple[str, str]:  # noqa: ANN401
    software = software_agent(store)
    activity = store.activity(node="R37_MEASURE", step=step, parameters={})
    store.was_associated_with(activity, software)
    return activity, software


def _replace(store: Any, old: Any, name: str, attributes: dict, *, extent: Any = None) -> str:  # noqa: ANN401
    activity, software = _activity(store, name)
    signal = str(old.attributes.get("signal") or "plain") if old is not None else "plain"
    new = write_measurement(store, activity, software, name=name, signal=signal, attributes=attributes, extent=extent)
    if old is not None and old.id != new:
        store.was_invalidated_by(old.id, activity)
    return new


def _shares(store: Any, *, exclusive: bool) -> list[dict]:  # noqa: ANN401
    """Write the gate's reading per task extent: the r36 share (A) or the exclusive share (B)."""
    out = []
    activity, software = _activity(store, NEW_SHARE)
    for measurement in find_measurements(store, OLD_SHARE):
        if measurement.extent is None:
            continue
        extent = (float(measurement.extent[0]), float(measurement.extent[1]))
        diarizer, signal = measurement.attributes.get("diarizer"), measurement.attributes.get("signal")
        segments = [
            (sp.id, str(sp.attributes.get("speaker")), (float(sp.extent[0]), float(sp.extent[1])))
            for sp in live_entities(store, "speaker")
            if sp.extent is not None
            and sp.attributes.get("diarizer") == diarizer
            and sp.attributes.get("signal") == signal
        ]
        reading = speakers_within(extent, segments, exclusive_min_s=STATE["exclusive_min_s"])
        old = measurement.attributes.get("value")
        value = reading["exclusive_share"] if exclusive else old
        write_measurement(
            store,
            activity,
            software,
            name=NEW_SHARE,
            signal=str(signal or "enhanced"),
            attributes={
                "value": value,
                "diarizer": diarizer,
                "secondary_exclusive_spans": reading["secondary_exclusive_spans"],
                "secondary_exclusive_s": reading["secondary_exclusive_s"],
            },
            extent=extent,
        )
        out.append(
            {
                "extent": [round(extent[0], 3), round(extent[1], 3)],
                "old": None if old is None else round(float(old), 4),
                "old_recomputed": None if reading["share"] is None else round(reading["share"], 4),
                "exclusive": None if reading["exclusive_share"] is None else round(reading["exclusive_share"], 4),
                "exclusive_s": reading["secondary_exclusive_s"],
                "secondary_s": reading["secondary_s"],
            }
        )
    return out


def _remap_findings(store: Any, before: list) -> int:  # noqa: ANN401
    """Point the r36 PII findings at the arbitrated words, as a replay's SPEECH would place them afresh."""
    transcript = find_measurement(store, "consensus_transcript")
    after = [] if transcript is None else list(transcript.attributes.get("word_ids") or ())
    if len(after) != len(before):
        return -1
    mapping = {str(o): str(n) for o, n in zip(before, after) if o != n}
    if not mapping:
        return 0
    activity, software = _activity(store, "pii_remap")
    moved = 0
    for finding in live_entities(store, "pii"):
        ids = [str(i) for i in finding.attributes.get("word_ids") or ()]
        if not any(i in mapping for i in ids):
            continue
        attributes = {**finding.attributes, "word_ids": [mapping.get(i, i) for i in ids]}
        new = store.entity(prov_type="pii", extent=finding.extent, attributes=attributes)
        store.was_generated_by(new, activity)
        store.was_attributed_to(new, software)
        store.was_invalidated_by(finding.id, activity)
        moved += 1
    for scan in find_measurements(store, "pii_scan"):
        ids = [str(i) for i in scan.attributes.get("residue_word_ids") or ()]
        if any(i in mapping for i in ids):
            attributes = {k: v for k, v in scan.attributes.items() if k not in ("name", "signal")}
            attributes["residue_word_ids"] = [mapping.get(i, i) for i in ids]
            _replace(store, scan, "pii_scan", attributes)
    return moved


def _condition(store: Any, participant: str) -> dict | None:  # noqa: ANN401
    cohort = find_measurement(store, COHORT_READING)
    summary = STATE["conditions"].get(participant)
    if cohort is None or summary is None:
        return None
    attributes = {k: v for k, v in cohort.attributes.items() if k not in ("name", "signal")}
    attributes["condition"] = {**dict(attributes.get("condition") or {}), "participant": summary}
    _replace(store, cohort, COHORT_READING, attributes)
    return summary


def _masks(store: Any, hint: Any) -> dict:  # noqa: ANN401
    config = STATE["config"]
    annotation = find_measurement(store, REDACTION_LLM_ANNOTATION)
    reading = {} if annotation is None else dict(annotation.attributes)
    policy = FoldPolicy.from_config(config)
    plan = fold_mask_plan(
        store, config, hint, reviewer_applies=policy.llm_reset_redactions and reviewer_may_unmask(reading)
    )
    spans = sorted((float(e.start), float(e.end)) for e in plan.final)
    merged: list[list[float]] = []
    for a, b in spans:
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return {
        "masks_n": len(plan.final),
        "pii_masks_n": plan.pii_masks_final_n,
        "masked_s": round(sum(b - a for a, b in merged), 3),
        "speech_masked_words_n": plan.non_task_speech_masked_n,
        "speech_untimed_n": len(plan.non_task_speech_untimed),
    }


def _fold(store: Any, hint: Any, run_root: Path) -> dict:  # noqa: ANN401
    result = verdict(store, None, STATE["config"], hint, run_dir=run_root / "run")
    fv = result.file_verdict
    speaker = [
        {
            "name": e.name,
            "effect": e.effect,
            "decisive": e.decisive,
            "source": e.source,
            "model": e.model,
            "spans_n": len(e.spans),
            "value": e.value if isinstance(e.value, (int, float, bool, str)) or e.value is None else str(e.value),
            "threshold": e.threshold if isinstance(e.threshold, (int, float, bool)) else None,
        }
        for e in fv.evidence
        if e.source is not None
    ]
    return {
        "triage": fv.triage.value,
        "release": None if fv.release is None else fv.release.value,
        "release_reason": release_ground_key(fv.release_ground),
        "reason": fv.reason,
        "reasons": list(fv.reason_keys),
        "grounds": list(fv.ground_keys),
        "annotations": list(fv.annotation_keys),
        "speaker_items": speaker,
        **_masks(store, hint),
    }


def _r36(store: Any) -> dict:  # noqa: ANN401
    from senselab.audio.workflows.triage.nodes.common import find_verdict

    held = find_verdict(store, "VERDICT")
    if held is None:
        return {}
    a = held.attributes
    return {
        "triage": a.get("triage"),
        "release": a.get("release"),
        "release_reason": a.get("release_ground_key"),
        "reason": a.get("reason"),
        "reasons": list(a.get("reason_keys") or ()),
        "grounds": list(a.get("ground_keys") or ()),
        "annotations": list(a.get("annotation_keys") or ()),
    }


def one(row: dict) -> dict:
    """One run, three readings."""
    out: dict[str, Any] = {"stem": row.get("stem")}
    try:
        run_root = run_root_of(Path(row["enhanced"]))
        source = Path(str(row["source"])) if row.get("source") else source_of(run_root)
        hint = STATE["hint"](source)[0]
        base = read_store(run_root)
        stem = recording_stem(base) or run_root.name
        family = declared_task(stem)[1]
        participant = run_root.name.split("_")[0]
        out.update(stem=run_root.name, participant=participant, family=family, r36=_r36(base))
        annotation = find_measurement(base, REDACTION_LLM_ANNOTATION)
        opinion = find_measurement(base, "second_opinion_answers")
        cohort = find_measurement(base, COHORT_READING)
        reading = {} if annotation is None else dict(annotation.attributes)
        held = {} if opinion is None else dict(opinion.attributes)
        cohort_attrs = None if cohort is None else dict(cohort.attributes)
        p = (held.get("probabilities") or {}).get("other_voice")
        out["readers"] = {
            "reviewer_speakers": reading.get("speakers"),
            "reviewer_other_kind": reading.get("other_speaker"),
            "reviewer_off_task": reading.get("off_task_speech"),
            "so_other_voice_p": None if p is None or isinstance(p, dict) else round(float(p), 4),
            "so_status": held.get("status"),
            "participant_matched": participant_matched(cohort_attrs),
            "cohort_nonmatch": bool(nonmatch_spans(cohort_attrs)),
            "target_status": pool_block(cohort_attrs, "target").get("status"),
            "target_match": pool_block(cohort_attrs, "target").get("match"),
            "target_pooled_s": pool_block(cohort_attrs, "target").get("pooled_s"),
            "separated_speakers_n": separated_speakers_n(base),
        }
    except Exception as error:  # noqa: BLE001
        out["error"] = "setup: " + repr(error)[:300]
        return out
    try:
        _shares(base, exclusive=False)
        out["A"] = _fold(base, hint, run_root)
    except Exception:  # noqa: BLE001
        out["error_A"] = traceback.format_exc()[-600:]
    try:
        store = read_store(run_root)
        transcript = find_measurement(store, "consensus_transcript")
        before = [] if transcript is None else list(transcript.attributes.get("word_ids") or ())
        arbitrate_words(store, run_dir=run_root / "run", family=family)
        out["B1_findings_remapped"] = _remap_findings(store, before)
        arbitration = find_measurement(store, WORD_ARBITRATION)
        out["B1"] = (
            None
            if arbitration is None
            else {
                k: arbitration.attributes.get(k)
                for k in (
                    "disagreed_n",
                    "retimed_n",
                    "unlocated_n",
                    "moved_n",
                    "won_by_source",
                    "activity_s",
                    "held_phonation",
                )
            }
        )
        if reads_task_speech(family):
            old = find_measurement(store, TASK_SPEECH_READING)
            activity, software = _activity(store, TASK_SPEECH_READING)
            new = write_task_speech(store, activity, software, family=family, config=STATE["config"], hint=hint)
            if old is not None and new is not None and old.id != new:
                store.was_invalidated_by(old.id, activity)
            fresh = find_measurement(store, TASK_SPEECH_READING)
            out["B3"] = {
                "words_n_old": None if old is None else old.attributes.get("words_n"),
                "words_n_new": None if fresh is None else fresh.attributes.get("words_n"),
                "post_task_n": None if fresh is None else fresh.attributes.get("post_task_n"),
            }
        out["C1"] = _shares(store, exclusive=True)
        out["condition"] = _condition(store, participant)
        cohort_b = find_measurement(store, COHORT_READING)
        cohort_b_attrs = None if cohort_b is None else dict(cohort_b.attributes)
        out["hot_or_clipped"] = hot_or_clipped(cohort_b_attrs)
        others = [dict(o) for o in reading.get("other_speakers") or () if isinstance(o, dict)]
        heard = reading.get("speakers") == "more_than_one" or (reading.get("speakers") == "unclear" and bool(others))
        out["signals"] = second_voice_signals(
            reviewer=bool(heard) or reading.get("other_speaker") in ("assistant", "background", "unclear"),
            second_opinion=opinion_reads_other_voice(held, 0.8, cohort_b_attrs),
            cohort=cohort_b_attrs,
            separated_speakers_n=separated_speakers_n(store),
        )
        out["B"] = _fold(store, hint, run_root)
    except Exception:  # noqa: BLE001
        out["error_B"] = traceback.format_exc()[-600:]
    return out


if __name__ == "__main__":
    manifest, k, n, conditions, hints, config, prefix = sys.argv[1:8]
    rows = [json.loads(line) for line in open(manifest) if line.strip()]
    only = os.environ.get("R37_ONLY")
    if only:
        wanted = {line.strip() for line in open(only) if line.strip()}
        rows = [r for r in rows if any(r["stem"].startswith(w) for w in wanted)]
    mine = rows[int(k) :: int(n)]
    workers = int(os.environ.get("R37_WORKERS", "8"))
    with (
        open(f"{prefix}.{int(k):03d}.jsonl", "w") as out,
        ProcessPoolExecutor(workers, initializer=_init, initargs=(conditions, hints, config)) as pool,
    ):
        for record in pool.map(one, mine, chunksize=4):
            out.write(json.dumps(record, default=str) + "\n")
            out.flush()
