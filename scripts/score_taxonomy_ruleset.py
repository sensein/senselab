"""Score the family taxonomy ruleset over a completed triage run's extracted features.

Moved into the package as ``senselab.audio.workflows.triage.drivers.score_taxonomy_ruleset``.
This file keeps ``python scripts/score_taxonomy_ruleset.py`` working; from an installed senselab, run
``python -m senselab.audio.workflows.triage.drivers.score_taxonomy_ruleset``.
"""

from senselab.audio.workflows.triage.drivers.score_taxonomy_ruleset import main

if __name__ == "__main__":
    raise SystemExit(main())
