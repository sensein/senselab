"""Extend finished triage runs with the clip-amplitude measurement QUALITY reads, in place.

Moved into the package as ``senselab.audio.workflows.triage.drivers.extend_clip_amplitudes``.
This file keeps ``python scripts/extend_clip_amplitudes.py`` working; from an installed senselab, run
``python -m senselab.audio.workflows.triage.drivers.extend_clip_amplitudes``.
"""

from senselab.audio.workflows.triage.drivers.extend_clip_amplitudes import main

if __name__ == "__main__":
    raise SystemExit(main())
