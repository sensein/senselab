"""Choose a calibration sample in both directions from precomputed per-recording speaker counts.

Two arms, deterministic under ``--seed``:

* ``multi`` — recordings the incumbent diarizer already called two-speaker. These are the
  positives a candidate must keep finding.
* ``single`` — sustained-vowel and diadochokinesis recordings, where a second voice is
  near-impossible by task construction. These are the negatives, and the false-positive
  rate on them is what decides whether a candidate can replace the incumbent.

Reads the slice rows written by the earlier ``extend_diarization.py`` campaign, which carry
``stem`` and ``enhanced_n_speakers`` per recording. Emits stems and paths only.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any

# Tasks whose protocol is one person producing a non-linguistic sound to a prompt. A second
# voice in one of these is a recording fault, not a protocol feature.
SINGLE_SPEAKER_TASKS = ("task-prolonged-vowel", "task-maximum-phonation-time", "task-diadochokinesis", "task-glides")

# Connected-speech tasks the incumbent calls one speaker. These are the negatives that
# actually matter: they are the same length and the same kind of signal as the recordings a
# detector would fire on, so a false positive here is a false positive in production. The
# sustained-vowel arm above is too short to window and too unlike speech to generalise from.
SINGLE_SPEECH_TASKS = (
    "task-caterpillar",
    "task-harvard",
    "task-picture-description",
    "task-story-recall",
    "task-free-speech",
    "task-rainbow",
    "task-cinderella",
)
MIN_SPEECH_S = 20.0

_TASK = re.compile(r"_(task-[^_]+)")


def task_of(stem: str) -> str:
    """The ``task-...`` token of a stem, or the empty string."""
    m = _TASK.search(stem)
    return m.group(1) if m else ""


def main() -> int:
    """Write one JSON row per chosen recording, tagged with the arm it belongs to."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("slices", help="directory of *.jsonl rows from the earlier diarization campaign")
    ap.add_argument("--corpus", required=True, help="run tree to resolve each stem's audio against")
    ap.add_argument("--n-multi", type=int, default=20)
    ap.add_argument("--n-single", type=int, default=20)
    ap.add_argument("--n-speech", type=int, default=40, help="long connected-speech single-speaker negatives")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    pools: dict[str, list[dict[str, Any]]] = {"multi": [], "single": [], "speech_single": []}
    for p in sorted(Path(args.slices).rglob("*.jsonl")):
        with open(p) as fh:
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                stem = r.get("stem")
                n = r.get("enhanced_n_speakers")
                if not stem or n is None:
                    continue
                task = task_of(stem)
                speech_s = float(r.get("enhanced_speech_s") or 0.0)
                base = {"stem": stem, "task": task, "prior_n_speakers": n, "prior_speech_s": round(speech_s, 2)}
                if n >= 2 and speech_s >= MIN_SPEECH_S:
                    pools["multi"].append({**base, "arm": "multi"})
                elif n == 1 and task.startswith(SINGLE_SPEAKER_TASKS):
                    pools["single"].append({**base, "arm": "single"})
                elif n == 1 and task.startswith(SINGLE_SPEECH_TASKS) and speech_s >= MIN_SPEECH_S:
                    pools["speech_single"].append({**base, "arm": "speech_single"})

    rng = random.Random(args.seed)
    for v in pools.values():
        rng.shuffle(v)
    chosen = pools["multi"][: args.n_multi] + pools["single"][: args.n_single] + pools["speech_single"][: args.n_speech]
    print("pool: " + " ".join(f"{k}={len(v)}" for k, v in pools.items()) + f"; chose {len(chosen)}")

    corpus = Path(args.corpus)
    written = 0
    with open(args.out, "w") as fh:
        for row in chosen:
            stem = row["stem"]
            sub = stem.split("_")[0]
            ses = stem.split("_")[1]
            hits = sorted((corpus / sub / ses).glob(f"{stem}_*"))
            if not hits:
                row["audio"] = None
                row["note"] = "no run directory in corpus"
            else:
                row["run_dir"] = str(hits[0])
                plain = hits[0] / "run" / "streams" / "plain.flac"
                row["audio"] = str(plain) if plain.exists() else None
            fh.write(json.dumps(row) + "\n")
            written += 1
    resolved = sum(1 for r in chosen if r.get("audio"))
    print(f"wrote {written} rows, {resolved} with resolvable plain audio, to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
