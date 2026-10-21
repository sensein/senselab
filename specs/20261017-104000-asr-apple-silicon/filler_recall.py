"""Filler recall/precision of each recognizer against AMI references.

Usage: uv run python specs/20261017-104000-asr-apple-silicon/filler_recall.py CALIB_DIR REFS_DIR

A reference filler (UM, UH) counts as found when the recognizer emits a filler word (bracketed or
plain um/uh/er/...) whose midpoint lies inside the same reference utterance, matched one-to-one
within the utterance. A hypothesis filler outside every utterance holding a reference filler counts
against precision.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from senselab.utils.data_structures import ScriptLine

REF_FILLERS = {"UM", "UH", "UHM", "ER", "ERM"}
HYP_FILLERS = {"um", "uh", "uhm", "umm", "er", "erm"}


def hyp_is_filler(text: str) -> bool:
    """Whether a recognizer word is a hesitation filler (``[UM]``, ``um,``, ``Uh``)."""
    t = text.strip().strip(".,!?;:").lower().strip("[]")
    return t in HYP_FILLERS


def main() -> None:
    """Print per-file filler counts, recall and precision."""
    calib, refs_dir = Path(sys.argv[1]), Path(sys.argv[2])
    report = {}
    for path in sorted(calib.glob("*.json")):
        if path.name == "timings.jsonl":
            continue
        stem, tag = path.name.split(".")[:2]
        refs = json.loads((refs_dir / f"{'_'.join(stem.split('_')[:2])}_refs.json").read_text())
        line = ScriptLine.model_validate_json(path.read_text())
        hyp = [((c.start or 0) + (c.end or 0)) / 2 for c in (line.chunks or []) if hyp_is_filler(c.text or "")]
        found = 0
        n_ref = 0
        used = set()
        for r in refs:
            k = sum(w in REF_FILLERS for w in r["text"].split())
            if not k:
                continue
            n_ref += k
            inside = [i for i, m in enumerate(hyp) if r["start"] - 0.2 <= m <= r["end"] + 0.2 and i not in used]
            take = inside[:k]
            used.update(take)
            found += len(take)
        report[f"{stem}.{tag}"] = {
            "ref_fillers": n_ref,
            "hyp_fillers": len(hyp),
            "matched": found,
            "recall": found / n_ref if n_ref else None,
            "precision": found / len(hyp) if hyp else None,
        }
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
