"""Print one ``probe_second_voice.py`` row as a table. Counts and seconds only."""

from __future__ import annotations

import json
import sys


def show(row: dict) -> None:
    """Print the baseline verdict, the cluster verdict and the split-cause diagnostics."""
    p = row.get("pyannote", {})
    e = row.get("embedding", {})
    print(f"AUDIO {row.get('audio')}  duration_s={row.get('duration_s')}")
    if "error" in p:
        print("  PYANNOTE error:", p["error"])
    else:
        print(
            f"  PYANNOTE n_speakers={p.get('n_speakers')} n_segments={p.get('n_segments')} "
            f"speech_s={round(p.get('speech_s', 0), 2)} wall_s={round(p.get('wall_s', 0), 1)}"
        )
        for s in p.get("segments", []):
            print("     ", s)
    if "error" in e:
        print("  EMBEDDING error:", e["error"])
        return
    print(
        f"  EMBEDDING windows_total={e.get('n_windows_total')} windows_speech={e.get('n_windows_speech')} "
        f"wall_s={round(e.get('wall_s', 0), 1)}"
    )
    a = e.get("ahc", {})
    rule = a.get("rule", {})
    mh = rule.get("merge_heights", [])
    print(
        f"  AHC n_clusters={a.get('n_clusters')} cos_dominant_to_runner_up={a.get('cos_dominant_to_runner_up')} "
        f"n_dropped={a.get('n_dropped')}"
    )
    print(
        f"  rule cut_theta={rule.get('cut_theta')} source={rule.get('cut_source')} "
        f"gap_margin={rule.get('gap_significance_margin')} ({rule.get('gap_significance_margin_source')})"
    )
    print("  merge_heights tail:", [round(x, 4) for x in mh[-8:]])
    for c in a.get("clusters", []):
        print("     cluster", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in c.items()})
    diag = a.get("per_cluster_diagnostics")
    if diag:
        print("  split-cause diagnostics:", json.dumps(diag, indent=2))
    d = e.get("distribution", {})
    for k in ("similarity", "spectrum", "nulls"):
        print(f"  {k}:", json.dumps(d.get(k)))
    print("  spectral:", json.dumps(e.get("spectral")))


def main() -> int:
    """Print every row of every file named on the command line."""
    for path in sys.argv[1:]:
        with open(path) as fh:
            for line in fh:
                if line.strip():
                    show(json.loads(line))
                    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
