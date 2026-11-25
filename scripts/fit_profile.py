"""Fit a dataset profile's corpus statistics from a pilot run's corpus table, into the profile's ``fitted.yaml``.

Moved into the package as ``senselab.audio.workflows.triage.drivers.fit_profile``.
This file keeps ``python scripts/fit_profile.py`` working; from an installed senselab, run
``python -m senselab.audio.workflows.triage.drivers.fit_profile``.
"""

from senselab.audio.workflows.triage.drivers.fit_profile import main

if __name__ == "__main__":
    raise SystemExit(main())
