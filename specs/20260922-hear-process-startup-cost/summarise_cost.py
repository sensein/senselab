"""Summarise the HeAR cost probe: what is fixed per invocation and what scales with the audio."""

import json
import statistics
import sys
from collections import defaultdict


def fit(xs: list, ys: list) -> tuple:
    """Least-squares intercept, slope and r2 of ``ys`` on ``xs``."""
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx if sxx else 0.0
    intercept = my - slope * mx
    ss_tot = sum((y - my) ** 2 for y in ys)
    ss_res = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys))
    return intercept, slope, (1 - ss_res / ss_tot) if ss_tot else 0.0


def main() -> None:
    """Print the parent-side staging cost, the phase table and the two fits."""
    rows = [json.loads(line) for line in open(sys.argv[1]) if line.strip()]
    by = defaultdict(list)
    for row in rows:
        by[row["trial"]].append(row)

    print("=== parent-side staging, per invocation of the one-shot path")
    for row in by["parent_staging"]:
        print(f"  rep {row['rep']}: ensure_venv {row['ensure_venv_s']:.3f} s  stage_snapshot {row['stage_snapshot_s']:.3f} s")

    for trial in ("single", "encoder"):
        entries = by.get(trial) or []
        if not entries:
            continue
        print(f"\n=== {trial}: one audio per invocation")
        print(f"{'audio s':>9} {'windows':>8} {'wall':>7} {'import':>8} {'load':>7} {'sig':>6} {'read':>7} {'infer':>8} {'w1':>7}")
        per_duration = defaultdict(list)
        for row in entries:
            per_duration[row["seconds"]].append(row)
        xs, ys, is_, ps = [], [], [], []
        for seconds in sorted(per_duration):
            group = per_duration[seconds]
            med = lambda f: statistics.median(f(r) for r in group)  # noqa: E731
            job = lambda r: r["timing"]["per_job"][0]  # noqa: E731
            print(
                f"{seconds:9.0f} {job(group[0])['n_windows']:8d} {med(lambda r: r['wall_s']):7.2f}"
                f" {med(lambda r: r['timing']['import_s']):8.2f}"
                f" {med(lambda r: r['timing']['saved_model_load_s']):7.2f}"
                f" {med(lambda r: r['timing']['signature_s']):6.2f}"
                f" {med(lambda r: job(r)['read_s']):7.3f}"
                f" {med(lambda r: job(r)['infer_s']):8.3f}"
                f" {med(lambda r: job(r)['first_window_s'] or 0.0):7.3f}"
            )
            for row in group:
                xs.append(seconds)
                ys.append(row["wall_s"])
                is_.append(job(row)["infer_s"])
                ps.append(job(row)["n_windows"])
        a, b, r2 = fit(xs, ys)
        print(f"  wall_s   = {a:.3f} + {b:.5f} * audio_seconds   (n={len(xs)}, r2={r2:.3f})")
        a, b, r2 = fit(xs, is_)
        print(f"  infer_s  = {a:.3f} + {b:.5f} * audio_seconds   (n={len(xs)}, r2={r2:.3f})")
        a, b, r2 = fit(ps, is_)
        print(f"  infer_s  = {a:.3f} + {b:.5f} * n_windows       (n={len(ps)}, r2={r2:.3f})")

    if by.get("batch"):
        print("\n=== batch: n audios of 21 s inside one invocation")
        xs, ys = [], []
        per_n = defaultdict(list)
        for row in by["batch"]:
            per_n[row["n"]].append(row["wall_s"])
            xs.append(row["n"])
            ys.append(row["wall_s"])
        for n in sorted(per_n):
            print(f"  n={n:3d}  wall median {statistics.median(per_n[n]):6.2f} s")
        a, b, r2 = fit(xs, ys)
        print(f"  wall_s = {a:.3f} + {b:.4f} * n_audios_of_21s   (n={len(xs)}, r2={r2:.3f})")

    if by.get("shipped_run_hear"):
        print("\n=== the shipped run_hear, end to end")
        per_d = defaultdict(list)
        for row in by["shipped_run_hear"]:
            per_d[(round(row["seconds"], 1), row["n_windows"])].append(row["wall_s"])
        for key in sorted(per_d):
            print(f"  {key[0]:6.1f} s, {key[1]:3d} windows: median {statistics.median(per_d[key]):6.2f} s")
        xs = [k[0] for k in per_d for _ in per_d[k]]
        ys = [v for k in per_d for v in per_d[k]]
        a, b, r2 = fit(xs, ys)
        print(f"  wall_s = {a:.3f} + {b:.5f} * audio_seconds   (n={len(xs)}, r2={r2:.3f})")


if __name__ == "__main__":
    main()
