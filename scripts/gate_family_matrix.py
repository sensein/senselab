"""Where each task family routes, the family x ruleset-gate matrix behind it, and the heatmap.

Moved into the package as ``senselab.audio.workflows.triage.drivers.gate_family_matrix``.
This file keeps ``python scripts/gate_family_matrix.py`` working; from an installed senselab, run
``python -m senselab.audio.workflows.triage.drivers.gate_family_matrix``.
"""

from senselab.audio.workflows.triage.drivers.gate_family_matrix import main

if __name__ == "__main__":
    raise SystemExit(main())
