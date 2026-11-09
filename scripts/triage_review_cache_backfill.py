"""Seed the reviewer's result cache from finished runs whose readings were made with the current prompt.

Moved into the package as ``senselab.audio.workflows.triage.drivers.triage_review_cache_backfill``.
This file keeps ``python scripts/triage_review_cache_backfill.py`` working; from an installed senselab, run
``python -m senselab.audio.workflows.triage.drivers.triage_review_cache_backfill``.
"""

from senselab.audio.workflows.triage.drivers.triage_review_cache_backfill import main

if __name__ == "__main__":
    raise SystemExit(main())
