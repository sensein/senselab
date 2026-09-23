"""Sweep the incumbent pipeline's VBx clustering threshold and record what each value counts.

``pyannote/speaker-diarization-community-1`` ships ``params.clustering.threshold: 0.6``.
The senselab wrapper does not expose it, so this script drives ``pyannote.audio`` directly —
the same checkpoint at the same resolved commit, one parameter changed. That makes "retune
the incumbent" a candidate the comparison can price alongside a different model.

Counts, seconds and labels only.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

PYANNOTE = "pyannote/speaker-diarization-community-1"


def build(threshold: float, sha: str, device: str) -> Any:  # noqa: ANN401
    """The shipped pipeline with one clustering threshold substituted."""
    import torch
    from pyannote.audio import Pipeline

    pipe = Pipeline.from_pretrained(PYANNOTE, revision=sha)
    pipe.instantiate(
        {
            **pipe.parameters(instantiated=True),
            "clustering": {**pipe.parameters(instantiated=True)["clustering"], "threshold": threshold},
        }
    )
    if device != "cpu":
        pipe.to(torch.device(device))
    return pipe


def run(pipe: Any, path: Path) -> dict[str, Any]:  # noqa: ANN401
    """One pipeline call, summarised by label."""
    t0 = time.time()
    ann = pipe(str(path))
    per: dict[str, float] = {}
    n = 0
    for seg, _, lab in ann.itertracks(yield_label=True):
        per[str(lab)] = per.get(str(lab), 0.0) + float(seg.duration)
        n += 1
    return {
        "wall_s": round(time.time() - t0, 2),
        "n_segments": n,
        "n_speakers": len(per),
        "per_speaker_s": {k: round(v, 3) for k, v in sorted(per.items())},
        "boundaries": [
            [round(float(s.start), 3), round(float(s.end), 3), str(lab)]
            for s, _, lab in ann.itertracks(yield_label=True)
        ],
    }


def main() -> int:
    """Run every audio file at every threshold and write one row per pair."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("audio", nargs="+")
    ap.add_argument("--thresholds", nargs="+", type=float, default=[0.6, 0.5, 0.45, 0.4, 0.35, 0.3])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from senselab.utils.model_revision import resolve_revision

    sha = resolve_revision(PYANNOTE, "main")
    with open(args.out, "a") as fh:
        for th in args.thresholds:
            try:
                pipe = build(th, sha, args.device)
            except Exception as exc:  # noqa: BLE001
                print(f"threshold {th}: build failed {exc!r}", file=sys.stderr)
                continue
            for p in args.audio:
                try:
                    row = run(pipe, Path(p))
                except Exception:  # noqa: BLE001
                    row = {"error": traceback.format_exc()[-400:]}
                row.update({"threshold": th, "audio": str(p), "name": Path(p).name, "revision": sha})
                fh.write(json.dumps(row) + "\n")
                fh.flush()
                print(
                    f"th={th} {Path(p).name[:56]} n_speakers={row.get('n_speakers')} per={row.get('per_speaker_s')}",
                    file=sys.stderr,
                    flush=True,
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
