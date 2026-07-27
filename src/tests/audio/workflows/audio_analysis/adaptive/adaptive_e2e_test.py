"""Environment-gated e2e checks for the adaptive loop (tasks.md T037/T038/T039).

These need real artifacts, so they skip unless the relevant env vars point at data:

- ``SENSELAB_ADAPTIVE_E2E_RUN_DIR``: a completed analyze_audio run dir →
  **T037 determinism** (two loop runs must produce byte-identical decision logs;
  SC-004) and basic final-output invariants.
- ``SENSELAB_GOLDEN_RUN_DIR`` + ``SENSELAB_CANDIDATE_RUN_DIR``: two analyze_audio
  run dirs → **T038 golden compat** (SC-005): the pre-existing artifact set
  (9 uncertainty parquets by value; per-task JSON result payloads) must match.
  Produce the candidate with ``--max-rounds 1``-equivalent settings
  (``--enhancement always``) on the same audio/cache as the golden run.
- ``SENSELAB_DEGRADATION_SUITE_DIR``: output of ``scripts/make_degradation_suite.py``
  with per-variant adaptive runs at ``<suite>/<variant>_run/`` → **T039 SC-001**:
  injected spans must be proposed as regions and either improved or explained.

Run e.g.::

    SENSELAB_ADAPTIVE_E2E_RUN_DIR=artifacts/e2e_runs/<run> uv run pytest \
        src/tests/audio/workflows/audio_analysis/adaptive/adaptive_e2e_test.py -v
"""

import json
import os
from pathlib import Path

import pytest

_RUN_DIR = os.environ.get("SENSELAB_ADAPTIVE_E2E_RUN_DIR")
_GOLDEN = os.environ.get("SENSELAB_GOLDEN_RUN_DIR")
_CANDIDATE = os.environ.get("SENSELAB_CANDIDATE_RUN_DIR")
_SUITE = os.environ.get("SENSELAB_DEGRADATION_SUITE_DIR")


@pytest.mark.skipif(not _RUN_DIR, reason="SENSELAB_ADAPTIVE_E2E_RUN_DIR not set")
def test_t037_determinism_byte_identical(tmp_path: Path) -> None:
    """SC-004: identical inputs → byte-identical iterations.json + convergence.json."""
    from senselab.audio.workflows.audio_analysis.adaptive.loop import run_adaptive_loop

    run_dir = Path(_RUN_DIR or "")
    cache_dir = run_dir.parent.parent / "analyze_audio_cache"
    # Hermetic: determinism must not depend on live models or network-gated
    # backends — pin cache-replay-only behavior (live re-ASR + overlap
    # detection off, DSP audio loader, no U3 bundle download) so the test is
    # fast and environment-independent.
    policy_override = tmp_path / "policy.yaml"
    policy_override.write_text(
        'fusion:\n  consensus_alignment: "off"\n'
        "audio_io_backend: dsp\n"
        "rules:\n  U1_region_reasr: {enabled: false}\n  I4_overlap_detection: {enabled: false}\n"
    )
    outs = []
    for name in ("a", "b"):
        out = tmp_path / name
        run_adaptive_loop(
            run_dir,
            cache_dir=cache_dir if cache_dir.is_dir() else None,
            out_dir=out,
            policy_path=policy_override,
        )
        outs.append(out)
    for fname in ("final/iterations.json", "final/convergence.json"):
        a = json.dumps(json.loads((outs[0] / fname).read_text()), sort_keys=True)
        b = json.dumps(json.loads((outs[1] / fname).read_text()), sort_keys=True)
        # elapsed_s is wall-clock provenance; strip before comparing.
        a_doc, b_doc = json.loads(a), json.loads(b)
        for doc in (a_doc, b_doc):
            doc.pop("elapsed_s", None)
        assert a_doc == b_doc, f"{fname} differs between identical runs"

    transcript = json.loads((outs[0] / "final" / "transcript.json").read_text())
    words = transcript["words"]
    assert words == sorted(words, key=lambda w: (w["start"], w["end"]))
    assert all(0.0 <= w["confidence"] <= 1.0 for w in words)


@pytest.mark.skipif(not (_GOLDEN and _CANDIDATE), reason="SENSELAB_GOLDEN/CANDIDATE_RUN_DIR not set")
def test_t038_golden_compat_preexisting_artifacts() -> None:
    """SC-005: candidate run reproduces the golden run's pre-existing artifact values."""
    import pandas as pd

    golden, candidate = Path(_GOLDEN or ""), Path(_CANDIDATE or "")
    checked = 0
    for pq in sorted(golden.rglob("uncertainty/*.parquet")):
        rel = pq.relative_to(golden)
        cand = candidate / rel
        assert cand.exists(), f"candidate missing {rel}"
        g = pd.read_parquet(pq).reset_index(drop=True)
        c = pd.read_parquet(cand).reset_index(drop=True)
        pd.testing.assert_frame_equal(g, c, check_exact=False, rtol=0, atol=1e-12)
        checked += 1
    assert checked >= 9, f"expected >=9 uncertainty parquets, found {checked}"

    for task_json in sorted(golden.rglob("asr/*.json")) + sorted(golden.rglob("diarization/*.json")):
        rel = task_json.relative_to(golden)
        cand = candidate / rel
        assert cand.exists(), f"candidate missing {rel}"
        g = json.loads(task_json.read_text())
        c = json.loads(cand.read_text())
        assert g.get("result") == c.get("result"), f"result payload differs: {rel}"


@pytest.mark.skipif(not _SUITE, reason="SENSELAB_DEGRADATION_SUITE_DIR not set")
def test_t039_injected_spans_attacked_or_explained() -> None:
    """SC-001: each injected span is proposed as a region and improved or explained."""
    suite = Path(_SUITE or "")
    manifest = json.loads((suite / "manifest.json").read_text())
    hits = total = 0
    for name, variant in (manifest.get("variants") or {}).items():
        out_dir = suite / f"{name}_run"
        if not (out_dir / "final" / "convergence.json").exists():
            continue  # variant not analyzed yet — not a failure of the loop itself
        total += 1
        span = variant["injected_span"]
        regions = []
        for rd in sorted((out_dir / "rounds").iterdir()):
            f = rd / "regions.json"
            if f.exists():
                regions.extend(json.loads(f.read_text()))
        overlapping = [r for r in regions if r["core_start"] < span[1] and r["core_end"] > span[0]]
        conv = json.loads((out_dir / "final" / "convergence.json").read_text())
        explained = any(r["start"] < span[1] and r["end"] > span[0] for r in conv.get("irreducible_regions") or [])
        it = json.loads((out_dir / "final" / "iterations.json").read_text())
        improved = any(
            e.get("status") == "fired"
            and any((d or {}).get("delta", 0) < -0.05 for d in (e.get("delta") or {}).values())
            for e in it["entries"]
        )
        if overlapping and (improved or explained):
            hits += 1
    assert total > 0, "no analyzed variants found under the suite dir"
    assert hits / total >= 0.7, f"SC-001: only {hits}/{total} injected spans attacked-or-explained"
