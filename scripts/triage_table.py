"""Write ``triage.tsv``, one flat row per recording, and its column dictionary beside it.

Moved into the package as ``senselab.audio.workflows.triage.drivers.triage_table``.
This file keeps ``python scripts/triage_table.py`` working; from an installed senselab, run
``python -m senselab.audio.workflows.triage.drivers.triage_table``.
"""

from senselab.audio.workflows.triage.drivers.triage_table import main

if __name__ == "__main__":
    raise SystemExit(main())
