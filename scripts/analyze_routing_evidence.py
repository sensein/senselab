"""Score candidate TAXONOMY routing detectors over a completed triage run.

Moved into the package as ``senselab.audio.workflows.triage.drivers.analyze_routing_evidence``.
This file keeps ``python scripts/analyze_routing_evidence.py`` working; from an installed senselab, run
``python -m senselab.audio.workflows.triage.drivers.analyze_routing_evidence``.
"""

from senselab.audio.workflows.triage.drivers.analyze_routing_evidence import main

if __name__ == "__main__":
    raise SystemExit(main())
