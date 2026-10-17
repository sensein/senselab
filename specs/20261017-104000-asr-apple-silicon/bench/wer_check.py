# ruff: noqa  -- benchmark harness run inside the subprocess venvs, not part of the package.
"""WER of bench outputs against AMI references over [offset, offset+dur)."""

import json, sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[3] / "src"))
from senselab.audio.tasks.speech_to_text_evaluation.accuracy_estimate import _normalise, _wer

refs = json.load(open(sys.argv[1]))
off = float(sys.argv[2])
dur = float(sys.argv[3])
ref = []
for r in refs:
    m = (r["start"] + r["end"]) / 2
    if off <= m < off + dur:
        ref.extend(_normalise(r["text"], True))
for f in sys.argv[4:]:
    d = json.load(open(f))
    hyp = []
    for w in d["words"]:
        if w[1] is not None and (w[1] + w[2]) / 2 <= dur + 1:
            hyp.extend(_normalise(w[0], True))
    print(f, "ref", len(ref), "hyp", len(hyp), "WER %.3f" % _wer(ref, hyp))
