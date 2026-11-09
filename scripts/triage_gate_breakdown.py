"""Break a recording-vectors parquet down by task and by gate.

Moved into the package as ``senselab.audio.workflows.triage.drivers.triage_gate_breakdown``.
This file keeps ``python scripts/triage_gate_breakdown.py`` working; from an installed senselab, run
``python -m senselab.audio.workflows.triage.drivers.triage_gate_breakdown``.
"""

from senselab.audio.workflows.triage.drivers.triage_gate_breakdown import main

if __name__ == "__main__":
    raise SystemExit(main())
