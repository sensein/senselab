"""SC-006 regression: ``--speaker-profile`` is scoped, not invasive.

Runs ``analyze_audio`` twice on the same clip — once without a profile, once
with one — and asserts that supplying a profile touches **only** the
speaker-identity / target-quality outputs, leaving everything else
byte-identical (after normalizing run-to-run provenance like timestamps and the
script wrapper hash).

**The real invariant (post option-C integration):** the *no-profile* run is the
baseline, and supplying a profile is allowed to change only:

- the per-pass ``speaker_profile.json`` sidecar (added),
- ``profile_*`` sub-signal keys + the ``single_speaker`` / ``quality`` headline
  uncertainties under those claims (FR-020 / FR-010),
- the **per-pass identity-axis aggregated uncertainty** and its
  ``contributing_models`` / ``speaker_profile/*`` votes — because the profile is
  now a *real reference-based identity voter* ("is this the enrolled subject?"),
  not a decorative side-signal, and
- ``disagreements.json`` ordering (it ranks on the identity uncertainty above).

Everything else — presence/utterance row data, the ``raw_vs_enhanced`` deltas,
AST/YAMNet/features/PII/ASR, the Label Studio bundle — must be byte-identical.
So the feature stays **opt-in and zero-impact when off**, and when on it
perturbs *only* the speaker-related signals it is designed to refine. (Earlier
this test asserted the profile was *purely additive*; option C intentionally
lets it move the identity uncertainty, so the guarantee is narrowed to the
above — the no-profile path is still exactly unchanged.)

This is a slow integration test: it invokes the real ``analyze_audio`` CLI
twice. It skips the heaviest stages (diarization/ASR/alignment, enhancement) for
speed while still exercising embeddings + presence + identity + the profile
path.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytest.importorskip("soundfile")

from senselab.audio.workflows.speaker_profile import constants as C  # noqa: E402
from senselab.audio.workflows.speaker_profile.io import save_profile  # noqa: E402
from senselab.audio.workflows.speaker_profile.types import (  # noqa: E402
    ClusterStats,
    ProfileParams,
    SpeakerProfile,
)

_REPO = Path(__file__).resolve().parents[5]
_CLIP = _REPO / "src/tests/data_for_testing/synthetic/sub-A-confident/ses-1/harvard-00.flac"
# Embedding dims per default model, so the synthetic centroid actually scores.
_DIMS = {C.ECAPA_MODEL_ID: 192, C.RESNET_MODEL_ID: 256}

# Provenance / timing fields that legitimately differ run-to-run.
_VOLATILE = {
    "elapsed_s",
    "timestamp_utc",
    "wrapper_version_hash",
    "cache_key",
    "wrapper_hash",
    "built_at",
    "generated_at",
    "run_dir",
    "output_dir",
}


def _make_profile(path: Path) -> None:
    """Write a minimal but valid two-model profile (random unit centroids)."""
    rng = np.random.default_rng(0)
    centroids: dict[str, list[float]] = {}
    band: dict[str, tuple[float, float]] = {}
    for model, dim in _DIMS.items():
        v = rng.standard_normal(dim)
        v = v / np.linalg.norm(v)
        centroids[model] = [float(x) for x in v]
        band[model] = (0.30, 0.70)
    profile = SpeakerProfile(
        subject_id="sub-regression",
        centroids=centroids,
        confidence="ok",
        aggregate_speech_seconds=40.0,
        dominant_cluster=ClusterStats(n_windows=40, speech_seconds=40.0, silhouette=0.3, share=1.0),
        runner_up_cluster=None,
        calibration_band=band,
        sources=[],  # empty → analyzed file never matches → no leave-one-out
        params=ProfileParams(
            embedding_models=list(_DIMS),
            profile_window_s=C.PROFILE_WINDOW_S,
            profile_hop_s=C.PROFILE_HOP_S,
            detect_window_s=C.DETECT_WINDOW_S,
            detect_hop_s=C.DETECT_HOP_S,
            min_confident_speech_s=C.MIN_CONFIDENT_SPEECH_S,
            target_confident_speech_s=C.TARGET_CONFIDENT_SPEECH_S,
            ambiguity_share_ratio=C.AMBIGUITY_SHARE_RATIO,
        ),
        provenance={},
    )
    save_profile(profile, path)


def _run_analyze(clip: Path, out_dir: Path, *, profile: Path | None) -> Path:
    """Invoke the analyze_audio CLI; return the created run directory."""
    cmd = [
        sys.executable,
        str(_REPO / "scripts/analyze_audio.py"),
        str(clip),
        "--output-dir",
        str(out_dir),
        "--no-enhancement",
        "--skip",
        "diarization",
        "asr",
        "alignment",
        "--no-cache",
        "--device",
        "cpu",
    ]
    if profile is not None:
        cmd += ["--speaker-profile", str(profile)]
    subprocess.run(cmd, cwd=_REPO, check=True, capture_output=True, text=True)
    runs = sorted(out_dir.glob("*/"))
    assert runs, f"no run dir produced in {out_dir}"
    return runs[0]


def _strip(obj: Any) -> Any:  # noqa: ANN401
    """Recursively drop volatile fields and any ``profile_*`` / ``speaker_profile/*`` keys."""
    if isinstance(obj, dict):
        return {
            k: _strip(v)
            for k, v in obj.items()
            if k not in _VOLATILE and not k.startswith("profile_") and not str(k).startswith("speaker_profile/")
        }
    if isinstance(obj, list):
        return [_strip(x) for x in obj]
    return obj


def _norm_json_file(path: Path) -> Any:  # noqa: ANN401
    return _strip(json.loads(path.read_text()))


def _norm_summary(path: Path) -> Any:  # noqa: ANN401
    """Normalize summary.json, additionally dropping the *intended* profile folds.

    The profile legitimately moves the ``single_speaker`` / ``quality`` headline
    uncertainties (and the derived ``combined_uncertainty`` / ``best_pass``) —
    those are FR-020 / FR-010 outputs, not regressions. Everything else in the
    summary must be unchanged, which is what this comparison checks.
    """
    d = _strip(json.loads(path.read_text()))
    gu = d.get("global_uncertainty")
    if isinstance(gu, dict):
        gu.pop("combined_uncertainty", None)
        gu.pop("best_pass", None)
        for ps in (gu.get("by_pass") or {}).values():
            if not isinstance(ps, dict):
                continue
            ps.pop("combined_uncertainty", None)
            for claim in ("single_speaker", "quality"):
                c = ps.get(claim)
                if isinstance(c, dict):
                    c.pop("uncertainty", None)
    return d


def _norm_parquet(path: Path, *, relax_identity: bool = False) -> Any:  # noqa: ANN401
    import pyarrow.parquet as pq

    table = pq.read_table(path).to_pydict()
    # model_votes is a JSON-encoded string column; strip profile additions inside it.
    if "model_votes" in table:
        table["model_votes"] = [json.dumps(_strip(json.loads(s))) if s else s for s in table["model_votes"]]
    if relax_identity:
        # Option C: the profile is now a real reference-based identity voter, so a
        # per-pass identity bucket's aggregated uncertainty (and its contributing
        # model list) legitimately changes when a profile is supplied. Drop those
        # profile-affected columns; the rest of the identity parquet (bucket
        # bounds, comparison_status, intensity_weight, non-profile votes) must
        # still match byte-for-byte.
        for col in ("aggregated_uncertainty", "raw_aggregated_uncertainty", "contributing_models"):
            table.pop(col, None)
    return _strip(table)


def test_speaker_profile_is_additive(tmp_path: Path) -> None:
    """Supplying a profile changes only speaker-identity/quality outputs; all else byte-identical."""
    assert _CLIP.exists(), f"missing fixture clip {_CLIP}"
    profile_path = tmp_path / "profile.json"
    _make_profile(profile_path)

    base = _run_analyze(_CLIP, tmp_path / "base", profile=None)
    prof = _run_analyze(_CLIP, tmp_path / "prof", profile=profile_path)

    base_files = {p.relative_to(base) for p in base.rglob("*") if p.is_file()}
    prof_files = {p.relative_to(prof) for p in prof.rglob("*") if p.is_file()}

    # The profiled run only ADDS speaker_profile.json sidecars.
    added = prof_files - base_files
    assert added, "profiled run added no speaker_profile.json sidecar"
    assert all(p.name == "speaker_profile.json" for p in added), f"unexpected added files: {added}"
    assert not (base_files - prof_files), f"profiled run dropped files: {base_files - prof_files}"

    # Every shared file is identical once volatile + profile additions (and the
    # intended single_speaker/quality folds in summary.json) are stripped.
    for rel in sorted(base_files & prof_files):
        if rel.name.endswith(".parquet"):
            # Per-pass identity parquet: relax the profile-affected uncertainty
            # columns (option C — profile is a real identity voter). Every other
            # parquet (presence, utterance, raw_vs_enhanced/*) must be byte-identical.
            relax = rel.stem == "identity" and "raw_vs_enhanced" not in rel.parts
            assert _norm_parquet(base / rel, relax_identity=relax) == _norm_parquet(prof / rel, relax_identity=relax), (
                f"parquet differs: {rel}"
            )
        elif rel.name == "disagreements.json":
            # Identity-derived ranking: legitimately reorders/reweights when the
            # profile feeds the identity axis. Not a regression (option C).
            continue
        elif rel.name == "summary.json":
            assert _norm_summary(base / rel) == _norm_summary(prof / rel), "summary.json differs beyond intended folds"
        elif rel.suffix == ".json":
            assert _norm_json_file(base / rel) == _norm_json_file(prof / rel), f"json differs: {rel}"
        elif rel.suffix == ".xml":
            assert (base / rel).read_text() == (prof / rel).read_text(), f"xml differs: {rel}"

    # The intended additions are actually present (test isn't vacuously passing).
    prof_summary = json.loads((prof / "summary.json").read_text())
    raw_claims = prof_summary["global_uncertainty"]["by_pass"]["raw_16k"]
    assert any(k.startswith("profile_") for k in raw_claims["single_speaker"]), "single_speaker profile_* missing"
    assert any(k.startswith("profile_") for k in raw_claims["quality"]), "quality profile_* missing"
