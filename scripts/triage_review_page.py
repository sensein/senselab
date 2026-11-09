"""Build the triage review page: every recording's decision and evidence, faceted, for a reviewer.

Moved into the package as ``senselab.audio.workflows.triage.drivers.triage_review_page``.
This file keeps ``python scripts/triage_review_page.py`` working; from an installed senselab, run
``python -m senselab.audio.workflows.triage.drivers.triage_review_page``.
"""

from senselab.audio.workflows.triage.drivers.triage_review_page import main

if __name__ == "__main__":
    raise SystemExit(main())
