"""Reproduce F-7: adaptive/interventions.py's module docstring lists "Still deferred:
P2_fine_posteriors" -- but _p2_trigger/_p2_guard/_p2_execute are fully defined and
registered in the RULES table with id "P2_fine_posteriors".

No model load, no network: pure source/module inspection.

Run from the repository root:
    uv run --no-sync python specs/20260815-215106-analyze-audio-audit/repro/F-7.py
"""

import inspect
import sys

import senselab.audio.workflows.audio_analysis.adaptive.interventions as interventions

module_doc = inspect.getdoc(interventions) or ""
normalized = " ".join(module_doc.split())
claimed = "Still deferred: ``P2_fine_posteriors``"
assert claimed in normalized, "interventions.py module docstring wording changed"

rule_ids = {rule.get("id") for rule in interventions.RULES}
implemented_funcs = {
    "_p2_trigger": hasattr(interventions, "_p2_trigger"),
    "_p2_guard": hasattr(interventions, "_p2_guard"),
    "_p2_execute": hasattr(interventions, "_p2_execute"),
}
is_registered = "P2_fine_posteriors" in rule_ids
registered_entry = next((r for r in interventions.RULES if r.get("id") == "P2_fine_posteriors"), None)

print(f"claimed (module docstring): 'Still deferred: P2_fine_posteriors'")
print(f"actual  (functions defined): {implemented_funcs}")
print(f"actual  (RULES table ids): 'P2_fine_posteriors' in RULES = {is_registered}")
if registered_entry is not None:
    wired = {k: (v.__name__ if callable(v) else v) for k, v in registered_entry.items()
             if k in ("id", "trigger", "guard", "execute")}
    print(f"actual  (RULES entry): {wired}")

if all(implemented_funcs.values()) and is_registered:
    print("DEFECT REPRODUCED: docstring claims P2_fine_posteriors is 'still deferred', but "
          "_p2_trigger/_p2_guard/_p2_execute are all defined and the rule is registered live "
          "in RULES.")
    sys.exit(0)
else:
    print("NOT REPRODUCED: P2_fine_posteriors is actually deferred/unimplemented.")
    sys.exit(1)
