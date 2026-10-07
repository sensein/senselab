"""Summarise the YAMNet cost probe: the fixed intercept and the per-second slope."""

import json
import statistics
import sys


def fit(xs: list, ys: list) -> tuple:
    """Ordinary least squares; returns (intercept, slope, r2)."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx
    intercept = my - slope * mx
    ss_res = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys))
    ss_tot = sum((y - my) ** 2 for y in ys)
    return intercept, slope, 1.0 - ss_res / ss_tot if ss_tot else float("nan")


def main() -> None:
    """Print the per-phase table and the two regressions."""
    rows = [json.loads(line) for line in open(sys.argv[1]) if line.strip()]
    single = [r for r in rows if r["trial"] == "single"]
    batch = [r for r in rows if r["trial"] == "batch"]

    print("=== single: one audio per invocation")
    print(f"{'dur_s':>7} {'n':>3} {'wall':>7} {'import':>7} {'hubload':>8} {'classmap':>9} {'infer':>7} {'post':>6} {'read':>6}")
    by_d: dict = {}
    for r in single:
        by_d.setdefault(r["seconds"], []).append(r)
    for d in sorted(by_d):
        rs = by_d[d]

        def med(fn: object) -> float:
            return statistics.median([fn(r) for r in rs])  # type: ignore[operator]

        print(
            f"{d:7.1f} {len(rs):3d} {med(lambda r: r['wall_s']):7.2f} "
            f"{med(lambda r: r['timing']['import_s']):7.2f} "
            f"{med(lambda r: r['timing']['hub_load_s']):8.2f} "
            f"{med(lambda r: r['timing']['classmap_s']):9.2f} "
            f"{med(lambda r: r['timing']['per_audio'][0]['infer_s']):7.2f} "
            f"{med(lambda r: r['timing']['per_audio'][0]['post_s']):6.2f} "
            f"{med(lambda r: r['timing']['per_audio'][0]['read_s']):6.2f}"
        )

    xs = [r["seconds"] for r in single]
    ys = [r["wall_s"] for r in single]
    a, b, r2 = fit(xs, ys)
    print(f"\nwall_s = {a:.3f} + {b:.4f} * audio_seconds   (r2={r2:.3f}, n={len(xs)})")
    ys2 = [r["timing"]["per_audio"][0]["infer_s"] + r["timing"]["per_audio"][0]["post_s"] for r in single]
    a2, b2, r22 = fit(xs, ys2)
    print(f"infer+post_s = {a2:.3f} + {b2:.4f} * audio_seconds   (r2={r22:.3f})")
    print(f"fixed share at the corpus median (7.3 s): {100.0 * a / (a + b * 7.3):.1f}%")
    print(f"fixed share at 21 s:                      {100.0 * a / (a + b * 21.0):.1f}%")

    if batch:
        print("\n=== batch: N copies of a 21 s audio in ONE invocation")
        print(f"{'n':>3} {'wall':>7} {'per_audio':>10}")
        by_n: dict = {}
        for r in batch:
            by_n.setdefault(r["n"], []).append(r)
        for n in sorted(by_n):
            w = statistics.median([r["wall_s"] for r in by_n[n]])
            print(f"{n:3d} {w:7.2f} {w / n:10.2f}")
        xs3 = [r["n"] for r in batch]
        ys3 = [r["wall_s"] for r in batch]
        a3, b3, r23 = fit(xs3, ys3)
        print(f"\nwall_s = {a3:.3f} + {b3:.4f} * n_audios_of_21s   (r2={r23:.3f})")


if __name__ == "__main__":
    main()
