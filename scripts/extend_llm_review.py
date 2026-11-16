"""Run REVIEW over a finished triage corpus, in place, without replaying the graph.

Moved into the package as ``senselab.audio.workflows.triage.drivers.extend_llm_review``.
This file keeps ``python scripts/extend_llm_review.py`` working; from an installed senselab, run
``python -m senselab.audio.workflows.triage.drivers.extend_llm_review``.
"""

from senselab.audio.workflows.triage.drivers.extend_llm_review import main

if __name__ == "__main__":
    raise SystemExit(main())
