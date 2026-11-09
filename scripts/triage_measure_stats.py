"""Report how every measurement the triage graph writes is distributed over a corpus.

Moved into the package as ``senselab.audio.workflows.triage.drivers.triage_measure_stats``.
This file keeps ``python scripts/triage_measure_stats.py`` working; from an installed senselab, run
``python -m senselab.audio.workflows.triage.drivers.triage_measure_stats``.
"""

from senselab.audio.workflows.triage.drivers.triage_measure_stats import main

if __name__ == "__main__":
    raise SystemExit(main())
