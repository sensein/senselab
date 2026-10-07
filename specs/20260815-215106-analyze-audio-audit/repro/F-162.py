"""Reproduction for F-162 (raised-by C-1, verdict: SURVIVED-CORRECTED).

Claim (corrected mechanism, per verdicts/refutation.md):
`default.yaml`'s `linking.asr_slot_overlap` / `linking.asr_slot_mid_tol_s` never reach the fold
performed by `fuse_consensus_words`, on *every* production call path -- not merely at the one
named call site (`compute.py:433`), but also on the U1 live-re-ASR intervention route
(`adaptive/interventions.py:595` -> `_reharvest_asr` -> `asr.harvest_asr_votes`), because
`harvest_asr_votes` has no `policy` parameter to forward one through in the first place.

This script:
  1. Calls `fuse_consensus_words` exactly as `compute.harvest_pass` does (positionally, with no
     `policy=`), while a policy object carrying non-default slot parameters is in scope -- exactly
     mirroring compute.py:433, where `speech_presence_policy` is bound but not threaded through.
     Shows the fold's own recorded provenance reports the hardcoded 0.3/0.15 regardless.
  2. Inspects `harvest_asr_votes`'s signature (the function `_reharvest_asr` calls for the U1
     escalation route) and shows it has no `policy` parameter at all -- so even a caller that
     *wanted* to thread the config value through the U1 route structurally cannot.

Must be run from the repository root (audio_analysis/types.py shadows stdlib `types` otherwise).
Loads no model, downloads nothing -- pure data-structure fusion over synthetic word lists.
"""

from __future__ import annotations

import inspect
import sys
from types import SimpleNamespace

from senselab.audio.workflows.audio_analysis.asr import fuse_consensus_words, harvest_asr_votes


def main() -> int:
    # Synthetic two-model word streams -- two recognizers each producing a few timed words.
    # Structure matches what iter_word_leaves reads directly (text/start/end leaves).
    asr_resolved = {
        "model_a": [
            {"text": "hello", "start": 0.10, "end": 0.40},
            {"text": "world", "start": 0.50, "end": 0.90},
        ],
        "model_b": [
            {"text": "hello", "start": 0.12, "end": 0.42},
            {"text": "world", "start": 0.55, "end": 0.95},
        ],
    }

    # A policy object carrying config values that differ sharply from the hardcoded defaults --
    # exactly what a user setting linking.asr_slot_overlap / linking.asr_slot_mid_tol_s in
    # default.yaml would produce. It is bound in scope, exactly as `speech_presence_policy` is
    # bound in `compute.harvest_pass` three lines before the real call site.
    configured_policy = SimpleNamespace(asr_slot_overlap=0.95, asr_slot_mid_tol_s=0.001)

    # ---- Step 1: the real call, verbatim as compute.py:433 makes it ----
    # `consensus_fold = fuse_consensus_words(asr_resolved)` -- no policy= argument, even though
    # `configured_policy` (standing in for `speech_presence_policy`) is bound in this scope.
    _fused_words, provenance = fuse_consensus_words(asr_resolved)

    observed_slot_overlap = provenance.get("slot_overlap")
    observed_slot_mid_tol_s = provenance.get("slot_mid_tol_s")

    expected_slot_overlap = configured_policy.asr_slot_overlap
    expected_slot_mid_tol_s = configured_policy.asr_slot_mid_tol_s

    print("=== Step 1: compute.py:433 call-site path ===")
    print(f"configured (in-scope) policy.asr_slot_overlap    = {expected_slot_overlap}")
    print(f"configured (in-scope) policy.asr_slot_mid_tol_s  = {expected_slot_mid_tol_s}")
    print(f"provenance['slot_overlap']    (used value) = {observed_slot_overlap}")
    print(f"provenance['slot_mid_tol_s']  (used value) = {observed_slot_mid_tol_s}")

    site1_defect = (
        observed_slot_overlap == 0.3
        and observed_slot_mid_tol_s == 0.15
        and observed_slot_overlap != expected_slot_overlap
        and observed_slot_mid_tol_s != expected_slot_mid_tol_s
    )

    # ---- Step 2: the U1 escalation path, harvest_asr_votes, has no policy param at all ----
    # `_reharvest_asr` (interventions.py) calls `harvest_asr_votes(pass_summary=..., grid=...,
    # alignment_by_model=...)` with no `fused=` and no way to pass a policy -- confirm the
    # signature structurally forecloses threading config through this route too.
    sig = inspect.signature(harvest_asr_votes)
    has_policy_param = "policy" in sig.parameters

    print()
    print("=== Step 2: U1 route (adaptive/interventions.py:_reharvest_asr -> harvest_asr_votes) ===")
    print(f"harvest_asr_votes signature: {sig}")
    print(f"'policy' in signature.parameters: {has_policy_param}")

    site2_defect = not has_policy_param

    if site1_defect and site2_defect:
        print()
        print("DEFECT REPRODUCED: fuse_consensus_words used hardcoded slot_overlap=0.3, "
              f"slot_mid_tol_s=0.15 despite configured policy.asr_slot_overlap="
              f"{expected_slot_overlap}, policy.asr_slot_mid_tol_s={expected_slot_mid_tol_s} "
              "being in scope but not threaded through (compute.py:433's exact call shape); "
              "AND harvest_asr_votes (the U1 live-re-ASR route's harvester) has no `policy` "
              "parameter at all, so the config is unreachable on that path too, not merely "
              "'unreachable elsewhere' as originally filed.")
        return 0

    print("Could not reproduce the defect (values matched configuration, or a policy param exists).")
    return 1


if __name__ == "__main__":
    sys.exit(main())
