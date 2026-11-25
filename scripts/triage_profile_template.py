"""Write a new dataset's triage profile as a template: every value it must supply blank, beside what it means.

Moved into the package as ``senselab.audio.workflows.triage.drivers.triage_profile_template``.
This file keeps ``python scripts/triage_profile_template.py`` working; from an installed senselab, run
``python -m senselab.audio.workflows.triage.drivers.triage_profile_template``.
"""

from senselab.audio.workflows.triage.drivers.triage_profile_template import main

if __name__ == "__main__":
    raise SystemExit(main())
