"""Build a local HTML review page for the free-response transcripts in a triage corpus.

Moved into the package as ``senselab.audio.workflows.triage.drivers.free_speech_review_page``.
This file keeps ``python scripts/free_speech_review_page.py`` working; from an installed senselab, run
``python -m senselab.audio.workflows.triage.drivers.free_speech_review_page``.
"""

from senselab.audio.workflows.triage.drivers.free_speech_review_page import main

if __name__ == "__main__":
    raise SystemExit(main())
