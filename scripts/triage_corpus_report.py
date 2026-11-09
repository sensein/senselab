"""Fold a tree of triage runs into one corpus decision report.

Moved into the package as ``senselab.audio.workflows.triage.drivers.triage_corpus_report``.
This file keeps ``python scripts/triage_corpus_report.py`` working; from an installed senselab, run
``python -m senselab.audio.workflows.triage.drivers.triage_corpus_report``.
"""

from senselab.audio.workflows.triage.drivers.triage_corpus_report import main

if __name__ == "__main__":
    raise SystemExit(main())
