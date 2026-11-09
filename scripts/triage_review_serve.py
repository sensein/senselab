"""Serve a triage review page and the audio it plays, including raw recordings outside the corpus.

Moved into the package as ``senselab.audio.workflows.triage.drivers.triage_review_serve``.
This file keeps ``python scripts/triage_review_serve.py`` working; from an installed senselab, run
``python -m senselab.audio.workflows.triage.drivers.triage_review_serve``.
"""

from senselab.audio.workflows.triage.drivers.triage_review_serve import main

if __name__ == "__main__":
    raise SystemExit(main())
