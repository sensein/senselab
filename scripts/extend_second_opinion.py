"""Run SECOND_OPINION over a finished triage corpus, in place, without replaying the graph.

Moved into the package as ``senselab.audio.workflows.triage.drivers.extend_second_opinion``.
This file keeps ``python scripts/extend_second_opinion.py`` working; from an installed senselab, run
``python -m senselab.audio.workflows.triage.drivers.extend_second_opinion``.
"""

from senselab.audio.workflows.triage.drivers.extend_second_opinion import main

if __name__ == "__main__":
    raise SystemExit(main())
