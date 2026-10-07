"""Reproduction for F-172 (raised-by D-9, verdict: SURVIVED).

Claim: `global_summary.py`'s `compute_run_global_summary` scores any recording with 2+ detected
speakers as maximally violating the "single speaker" claim (`single_speaker_uncertainty = 1.0`),
with no task-type branch -- so a caregiver-mediated pediatric elicitation recording (where a
co-occurring adult is *correct*, not a defect) is scored identically to a solo-paradigm recording
that unexpectedly contains a second speaker. The function's only population/expectation knob is
`expects_speech: bool`, and that governs only the `n_speakers == 0` branch (lines ~292-296);
there is no analogous parameter for the `n_speakers >= 2` branch (line ~306, `else: ... = 1.0`).

This script calls the real `compute_run_global_summary` with `n_speakers=2` (surfaced via a
synthetic diarization block reporting `"n_speakers": 2`) under BOTH `expects_speech=True` and
`expects_speech=False`, and shows `single_speaker_uncertainty` is 1.0 in both cases -- i.e. the
one knob the function exposes cannot express "two speakers were expected here."

Must be run from the repository root. Loads no model, downloads nothing -- pure Python over
synthetic dicts (empty fused axes / ASR / PII inputs, since none of those are needed to reach the
`n_speakers` branch).
"""

from __future__ import annotations

import sys

from senselab.audio.workflows.audio_analysis.global_summary import compute_run_global_summary


def _run(expects_speech: bool) -> float | None:
    pass_summary = {
        "duration_s": 5.0,
        "diarization": {
            "by_model": {
                # A caregiver+child pass: the diarizer legitimately reports 2 speakers.
                "pyannote/speaker-diarization-community-1": {"status": "ok", "n_speakers": 2},
            }
        },
    }
    result = compute_run_global_summary(
        fused_axes={},
        passes={"raw": pass_summary},
        asr_resolved_by_pass={"raw": {}},
        pii_reports={},
        expects_speech=expects_speech,
    )
    return result["single_speaker"]["uncertainty"]


def main() -> int:
    unc_expects_true = _run(expects_speech=True)
    unc_expects_false = _run(expects_speech=False)

    print(f"n_speakers = 2 (caregiver + child), expects_speech=True  -> single_speaker_uncertainty = {unc_expects_true}")
    print(f"n_speakers = 2 (caregiver + child), expects_speech=False -> single_speaker_uncertainty = {unc_expects_false}")
    print("expects_speech only branches on n_speakers == 0 (source lines ~292-296); there is no")
    print("parameter that branches on n_speakers >= 2, so both calls above land on the same bare")
    print("`else: single_speaker_uncertainty = 1.0` (source line ~306), the maximal-violation value.")

    if unc_expects_true == 1.0 and unc_expects_false == 1.0:
        print()
        print(
            "DEFECT REPRODUCED: single_speaker_uncertainty = 1.0 (maximal violation) for a "
            "recording with n_speakers=2, unconditionally -- the correct value for a caregiver-"
            "mediated pediatric elicitation task (co-occurring adult is BY DESIGN, should read low "
            "or None) is indistinguishable from the correct value for a solo-paradigm recording "
            "that unexpectedly contains a second speaker (a real violation). The only exposed knob, "
            "expects_speech, cannot change this because it only gates the n_speakers==0 branch, not "
            "the n_speakers>=2 branch."
        )
        return 0

    print("Could not reproduce the defect (values were not both 1.0).")
    return 1


if __name__ == "__main__":
    sys.exit(main())
