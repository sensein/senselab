"""Print one ``probe_backends.py`` row per backend, with its segments and per-label seconds.

Scores each backend's two-way labelling against a supplied ground-truth boundary when one is
given: the fraction of attributed speech it places on the correct side.
"""

from __future__ import annotations

import argparse
import json
from typing import Any


def main() -> int:
    """Print every backend row, optionally scored against a boundary in seconds."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("rows", nargs="+")
    ap.add_argument("--truth", nargs=2, type=float, default=None, help="ground-truth second-voice span, seconds")
    ap.add_argument("--only-audio", default=None)
    args = ap.parse_args()

    for path in args.rows:
        with open(path) as fh:
            for line in fh:
                if not line.strip():
                    continue
                r: dict[str, Any] = json.loads(line)
                if args.only_audio and args.only_audio not in str(r.get("audio", "")):
                    continue
                if r.get("status") != "ok":
                    print(f"{r.get('backend'):12s} FAILED  {str(r.get('error'))[:200]}")
                    continue
                per: dict[str, float] = {}
                for s, e, lab in r.get("segments", []):
                    per[lab] = per.get(lab, 0.0) + (e - s)
                print(
                    f"{r.get('backend'):12s} n_speakers={r.get('n_speakers')} n_seg={r.get('n_segments'):3d} "
                    f"speech_s={r.get('speech_s')} overlap_s={r.get('overlap_s')} wall={r.get('wall_s')}"
                )
                print("             per_label_s=" + json.dumps({k: round(v, 2) for k, v in sorted(per.items())}))
                if r.get("n_speakers", 0) >= 2:
                    for s, e, lab in r.get("segments", []):
                        print(f"               [{s:7.2f}, {e:7.2f}] {lab}")
                    if args.truth:
                        lo, hi = args.truth
                        # The label that best covers the known second-voice span, scored as
                        # recall of that span and precision of that label. A backend that
                        # peels off a one-second fragment scores high precision and near-zero
                        # recall, which is the distinction a purity number hides.
                        # The minority label is the backend's claim about the second voice:
                        # the participant holds the bulk of every recording here.
                        lab = min(per, key=lambda k: per[k])
                        cov = sum(max(0.0, min(e, hi) - max(s, lo)) for s, e, lb in r.get("segments", []) if lb == lab)
                        print(
                            f"             vs truth [{lo}, {hi}]: minority label {lab} = {per[lab]:.2f}s, "
                            f"covers {cov:.2f}s  recall={cov / (hi - lo):.3f} "
                            f"precision={cov / per[lab] if per[lab] else 0:.3f}"
                        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
