"""Build the single-file viewer for recording_vectors.parquet.

Moved into the package as ``senselab.audio.workflows.triage.drivers.triage_vectors_viewer``.
This file keeps ``python scripts/triage_vectors_viewer.py`` working; from an installed senselab, run
``python -m senselab.audio.workflows.triage.drivers.triage_vectors_viewer``.
"""

from senselab.audio.workflows.triage.drivers.triage_vectors_viewer import main

if __name__ == "__main__":
    raise SystemExit(main())
