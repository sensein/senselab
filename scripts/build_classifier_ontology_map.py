"""Generate the classifier-ontology profile the triage workflow's airway branch corroborates with.

Moved into the package as ``senselab.audio.workflows.triage.drivers.build_classifier_ontology_map``.
This file keeps ``python scripts/build_classifier_ontology_map.py`` working; from an installed senselab, run
``python -m senselab.audio.workflows.triage.drivers.build_classifier_ontology_map``.
"""

from senselab.audio.workflows.triage.drivers.build_classifier_ontology_map import main

if __name__ == "__main__":
    raise SystemExit(main())
