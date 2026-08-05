"""Verify a completed analyze_audio run against the grid-unification contract, as measurements.

    uv run python scripts/verify_grid_unification.py artifacts/analyze_audio/<run_dir> [...]
    uv run python scripts/verify_grid_unification.py artifacts/analyze_audio/*   # every run

Checks, from `specs/20260728-221507-per-speaker-identity-scene/next-grid-unification-and-cli-config.md`:

2. **Every axis reports the same row count, the same `(window, hop)`, and the same bucket keys.** The
   directive, stated as a check. Keys are compared as sequences rather than inferred from the counts
   matching — four axes can agree on a count and still share no spans.
3. **Cross-axis coupling moves something**: rounds are not byte-identical, which is the symptom
   `fuse.project_axis_onto` was written for and could not exhibit while the axes shared no keys.
4. **The asr axis reflects its evidence.** Zero uncertainty is a legitimate reading — an axis reads
   zero when the algorithms agree, and cleanly-extracted audio should produce exactly that. A *flat*
   axis is therefore only a failure when its inputs disagreed, so this re-folds the run's own cached
   ASR outcomes and fails only on zero-despite-disagreement.
5. **`final/transcript.json`** words and confidences, printed with a digest so two runs can be
   compared without diffing the file.

Plus: the run config's identity must appear in `final/summary.json` and `L2/disagreements.json` — a
run whose configuration cannot be named cannot be reproduced.

Exits non-zero if any check fails, so it can gate a run.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


def _recognizers_unanimous(run_dir: Path) -> object:
    """``True`` when every fused word has ``existence_confidence == 1.0``; the value set otherwise.

    Re-folds the run's own cached ASR outcomes through the same call the harvest makes, so the answer
    comes from the pipeline rather than from a fixture. ``None`` when it cannot be determined.
    """
    try:
        from senselab.audio.tasks.speech_to_text_ensemble import fuse_word_streams, iter_word_leaves
        from senselab.audio.workflows.audio_analysis.adaptive.interventions import (
            load_alignments_matched,
            load_outcomes_dir,
        )
        from senselab.audio.workflows.audio_analysis.asr import _as_plain, aligned_columns, phoneme_similarity
        from senselab.audio.workflows.audio_analysis.harvesters import resolve_asr_result

        asr_by_model = dict(load_outcomes_dir(run_dir, "raw", "asr"))
        align = load_alignments_matched(run_dir, "raw", asr_by_model)
        resolved = {m: resolve_asr_result(b, align.get(m)) for m, b in asr_by_model.items() if b.get("status") == "ok"}
        streams = {m: w for m, r in resolved.items() if (w := iter_word_leaves(_as_plain(r)))}
        fused = fuse_word_streams(
            streams,
            slot_overlap=0.3,
            slot_mid_tol_s=0.15,
            text_similarity=phoneme_similarity,
            columns=aligned_columns(streams),
        )
        values = {
            round(float(w["existence_confidence"]), 6) for w in fused if w.get("existence_confidence") is not None
        }
        return True if values == {1.0} else sorted(values)
    except Exception as exc:  # noqa: BLE001 — a diagnostic must not fail the verification
        return f"undetermined: {exc!r}"


def check(run_dir: Path) -> list[str]:
    """Run every check against one run directory; return the human-readable failures."""
    failures: list[str] = []
    print(f"\n{'=' * 78}\n{run_dir.name}\n{'=' * 78}")

    # ── 2. All four axes report the same row count and the same (window, hop) ──
    rounds = sorted(
        (p for p in (run_dir / "L2").glob("round*") if (p / "uncertainty").is_dir()),
        key=lambda p: int(p.name.replace("round", "")),
    )
    if not rounds:
        rounds = sorted(
            (p for p in (run_dir / "L2" / "round").glob("*") if (p / "estimates").is_dir()),
            key=lambda p: int(p.name),
        )
    if not rounds:
        return [f"{run_dir}: no L2 round directories found"]

    print(f"rounds present: {[r.name for r in rounds]}")
    per_round: dict[str, dict[str, pd.DataFrame]] = {}
    for rd in rounds:
        sub = rd / "uncertainty" if (rd / "uncertainty").is_dir() else rd / "estimates"
        frames = {p.stem: pd.read_parquet(p) for p in sorted(sub.glob("*.parquet"))}
        per_round[rd.name] = frames

    last = rounds[-1].name
    frames = per_round[last]
    print(f"\n[2] axes in {last}: {sorted(frames)}")
    counts = {axis: len(df) for axis, df in frames.items()}
    grids: dict[str, tuple[float, float]] = {}
    for axis, df in frames.items():
        if df.empty:
            grids[axis] = (float("nan"), float("nan"))
            continue
        win = round(float(df.iloc[0]["end"]) - float(df.iloc[0]["start"]), 6)
        hop = round(float(df.iloc[1]["start"]) - float(df.iloc[0]["start"]), 6) if len(df) > 1 else win
        grids[axis] = (win, hop)
    for axis in sorted(frames):
        print(f"    {axis:20s} rows={counts[axis]:6d}  window={grids[axis][0]}  hop={grids[axis][1]}")
    if len(set(counts.values())) != 1:
        failures.append(f"[2] row counts differ: {counts}")
    if len(set(grids.values())) != 1:
        failures.append(f"[2] (window, hop) differ: {grids}")

    keysets = {
        axis: [(round(float(r.start), 6), round(float(r.end), 6)) for r in df.itertuples()]
        for axis, df in frames.items()
    }
    reference = keysets[sorted(keysets)[0]]
    for axis, spans in sorted(keysets.items()):
        if spans != reference:
            failures.append(f"[2] {axis} does not share bucket keys with the others")
    if not failures:
        print("    → same count, same grid, same keys ✓")

    # ── 3. Cross-axis coupling moves something: rounds are not byte-identical ──
    print("\n[3] round-to-round movement (the symptom project_axis_onto was written for):")
    if len(rounds) < 2:
        print(f"    only one round ({last}) — cannot compare; run with rounds.max_rounds > 1")
    else:
        moved = False
        for a, b in zip([r.name for r in rounds], [r.name for r in rounds][1:]):
            for axis in sorted(set(per_round[a]) & set(per_round[b])):
                left, right = per_round[a][axis], per_round[b][axis]
                cols = [c for c in ("uncertainty", "triage_score", "confidence") if c in left and c in right]
                if len(left) != len(right):
                    print(f"    {axis}: {a}->{b} row count changed {len(left)} -> {len(right)}")
                    moved = True
                    continue
                for col in cols:
                    delta = (left[col].fillna(-1) - right[col].fillna(-1)).abs()
                    n_changed = int((delta > 1e-12).sum())
                    if n_changed:
                        print(f"    {axis}.{col}: {a}->{b} {n_changed} buckets moved, max |Δ|={delta.max():.6f}")
                        moved = True
        if not moved:
            failures.append("[3] every round is byte-identical to the last — coupling still does nothing")
        else:
            print("    → rounds are not byte-identical ✓")

    # ── 4. The asr axis reflects its evidence ──
    print("\n[4] asr axis distribution (the grid must not flatten evidence that disagrees):")
    asr = frames.get("asr")
    if asr is None or asr.empty:
        failures.append("[4] no asr axis rows at all")
    else:
        for col in ("uncertainty", "triage_score"):
            if col not in asr:
                continue
            vals = asr[col].dropna()
            distinct = sorted({round(float(v), 4) for v in vals})
            print(
                f"    {col:14s} n={len(vals):5d}/{len(asr)}  mean={vals.mean():.4f}  "
                f"min={vals.min():.4f}  max={vals.max():.4f}  distinct={len(distinct)}"
            )
            if col == "uncertainty":
                if len(vals) == 0:
                    failures.append("[4] asr uncertainty is entirely null")
                elif len(distinct) < 2:
                    # **Zero is a legitimate answer**: an axis reads zero when the algorithms agree,
                    # and cleanly-extracted audio should produce exactly that. What would be a defect
                    # is zero while the *inputs* disagree — the fold not reaching the axis. Only that
                    # second case is a failure, so the two are told apart by measuring the input.
                    unanimous = _recognizers_unanimous(run_dir)
                    if unanimous is True:
                        print(
                            f"    {col} is 0 everywhere because the recognizers are unanimous on "
                            "every word (existence_confidence == 1.0 throughout) — nothing is in "
                            "doubt about what was said, which is the correct reading ✓"
                        )
                    else:
                        failures.append(
                            f"[4] asr {col} is flat at {distinct} while the fused words' "
                            f"existence_confidence varies ({unanimous!r}) — the fold is not reaching it"
                        )
        n_null = int(asr["uncertainty"].isna().sum())
        print(f"    unmeasured buckets (no word reached): {n_null}/{len(asr)}")
        sig_col = "contributing_signals"
        if sig_col in asr:
            names = {s for row in asr[sig_col].dropna() for s in list(row)}
            print(f"    contributing signals: {sorted(names)}")
            if not names:
                failures.append("[4] asr axis has zero contributing signals (the shape-mismatch failure)")

    # ── 5. final/transcript.json words + confidences ──
    print("\n[5] final/transcript.json:")
    tpath = run_dir / "final" / "transcript.json"
    if not tpath.exists():
        failures.append("[5] final/transcript.json missing")
    else:
        doc = json.loads(tpath.read_text())
        words = doc.get("words") or []
        texts = [w.get("text") for w in words]
        confs = [round(float(w.get("confidence") or 0.0), 6) for w in words]
        digest = json.dumps({"texts": texts, "confidences": confs}, sort_keys=True)
        import hashlib

        print(f"    n_words={len(words)}  digest={hashlib.sha256(digest.encode()).hexdigest()[:16]}")
        print(f"    text: {' '.join(str(t) for t in texts[:24])}{' ...' if len(texts) > 24 else ''}")
        if confs:
            print(f"    confidence mean={sum(confs) / len(confs):.4f} min={min(confs):.4f} max={max(confs):.4f}")

    # ── the config identity travels ──
    print("\n[config] identity in provenance:")
    spath = run_dir / "final" / "summary.json"
    if spath.exists():
        summary = json.loads(spath.read_text())
        rc = summary.get("run_config")
        print(f"    final/summary.json run_config = {rc}")
        if not rc or not rc.get("config_hash"):
            failures.append("[config] final/summary.json carries no run_config identity")
    dis = run_dir / "L2" / "disagreements.json"
    if dis.exists():
        cfg = (json.loads(dis.read_text()).get("config") or {}).get("run_config")
        print(f"    disagreements.json run_config = {cfg}")
        if not cfg:
            failures.append("[config] disagreements.json carries no run_config identity")

    return failures


def main() -> int:
    """Check each run directory named on the command line."""
    all_failures: list[str] = []
    for arg in sys.argv[1:]:
        all_failures.extend(check(Path(arg)))
    print(f"\n{'=' * 78}")
    if all_failures:
        print("FAILURES:")
        for f in all_failures:
            print(f"  ✗ {f}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
