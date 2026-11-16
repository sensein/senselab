"""Re-render the summaries for a tree of completed triage runs, from their stores.

Moved into the package as ``senselab.audio.workflows.triage.drivers.triage_rerender``.
This file keeps ``python scripts/triage_rerender.py`` working; from an installed senselab, run
``python -m senselab.audio.workflows.triage.drivers.triage_rerender``.
"""

from senselab.audio.workflows.triage.drivers.triage_rerender import main

if __name__ == "__main__":
    raise SystemExit(main())
