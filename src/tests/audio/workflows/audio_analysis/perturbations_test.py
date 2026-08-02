"""The perturbation set is open, and nothing downstream may assume how big it is (D-17).

Two assumptions were spelled into the code rather than declared: *exactly two*, and *the name
carries the transform*. These are the tests that would have caught each — written against the
property ("a third needs no code edit") rather than against the two names that were wrong,
because a test naming ``raw_16k`` and ``enhanced_16k`` is the same mistake in the test suite.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from senselab.audio.workflows.audio_analysis import layout
from senselab.audio.workflows.audio_analysis.io import write_signal_parquet
from senselab.audio.workflows.audio_analysis.perturbations import (
    IDENTITY_NAME,
    TRANSFORMS,
    Perturbation,
    identity,
    read_measurements,
    read_register,
    speech_enhancement,
    write_register,
)
from senselab.audio.workflows.audio_analysis.types import SignalResult, SignalRow

WORKFLOW_DIR = Path(__file__).resolve().parents[5] / "src" / "senselab" / "audio" / "workflows" / "audio_analysis"
SCRIPTS_DIR = Path(__file__).resolve().parents[5] / "scripts"


def _third() -> Perturbation:
    """A perturbation nothing in the codebase has ever seen."""
    return speech_enhancement("some-lab/other-enhancer", name="other")


# ── the register ─────────────────────────────────────────────────────────────


def test_a_third_perturbation_round_trips_through_the_register(tmp_path: Path) -> None:
    """Three go in, three come out — the register does not count, it lists."""
    declared = [identity(), speech_enhancement("speechbrain/sepformer-wham16k-enhancement"), _third()]
    write_register(tmp_path, declared, source_audio="/tmp/x.wav")

    assert read_register(tmp_path) == tuple(declared)
    payload = json.loads((tmp_path / "L1" / "perturbations.json").read_text())
    assert payload["source_audio"] == "/tmp/x.wav"
    assert [e["name"] for e in payload["perturbations"]] == ["raw", "enhanced", "other"]


def test_the_transform_is_declared_beside_the_name_not_inferred_from_it() -> None:
    """A perturbation named for its model still declares what it did to the recording."""
    assert _third().transform == "speech_enhanced"
    assert not _third().is_identity
    assert identity().is_identity


def test_an_undeclared_transform_is_rejected_at_construction() -> None:
    """A typo cannot reach provenance: it fails where it was written."""
    with pytest.raises(ValueError, match="declare it in TRANSFORMS"):
        Perturbation(name="x", transform="band_limited")


def test_the_measurements_travel_with_the_declaration(tmp_path: Path) -> None:
    """What running a perturbation produced sits beside what it *is*, in one file.

    ``L1/passes.json`` held the measurements and nothing else, and was rewritten by the run's
    last stage — a back-edge from the deliverable to the file defining L1's inputs.
    """
    write_register(
        tmp_path,
        [identity(), _third()],
        measured={"raw": {"duration_s": 4.0, "audio_signature": "a" * 64}, "other": {"status": "failed"}},
    )
    measured = read_measurements(tmp_path)
    assert measured["raw"]["duration_s"] == 4.0
    assert measured["other"] == {"status": "failed"}


def test_a_missing_register_is_reported_as_absence_rather_than_raising(tmp_path: Path) -> None:
    """A consumer that needs a perturbation reports *that*, not a parse error."""
    assert read_register(tmp_path) == ()
    assert read_measurements(tmp_path) == {}


# ── the layout ───────────────────────────────────────────────────────────────


def test_the_identity_has_its_own_directory_and_every_other_shares_one(tmp_path: Path) -> None:
    """The identity is not one transform among many — it is what the others transform."""
    assert layout.perturbation_dir(tmp_path, IDENTITY_NAME) == tmp_path / "L1" / "raw"
    assert layout.perturbation_dir(tmp_path, "other") == tmp_path / "L1" / "perturbation" / "other"


def test_signals_accumulate_across_perturbations_in_one_file(tmp_path: Path) -> None:
    """``L1/signals/<signal>.parquet`` is what the signal said under *every* perturbation."""
    import pyarrow.parquet as pq

    results = [
        SignalResult(perturbation=name, signal="brouhaha_snr_db", rows=[SignalRow(start=0.0, end=0.5, signal="s")])
        for name in ("raw", "enhanced", "other")
    ]
    dest = write_signal_parquet(results, layout.signals_dir(tmp_path) / "brouhaha_snr_db.parquet")

    table = pq.read_table(dest)
    assert table.column("perturbation").to_pylist() == ["enhanced", "other", "raw"]
    assert dest.parent == tmp_path / "L1" / "signals"


def test_one_file_per_signal_is_enforced_rather_than_assumed(tmp_path: Path) -> None:
    """The filename is the artifact's identity, so a mixture is refused, not silently written."""
    mixed = [
        SignalResult(perturbation="raw", signal="a"),
        SignalResult(perturbation="raw", signal="b"),
    ]
    with pytest.raises(ValueError, match="one file per signal"):
        write_signal_parquet(mixed, tmp_path / "L1" / "signals" / "a.parquet")


# ── nothing counts the perturbations, and nothing spells the old names ───────


def _pipeline_sources() -> list[Path]:
    return sorted(WORKFLOW_DIR.rglob("*.py")) + [SCRIPTS_DIR / "analyze_audio.py", SCRIPTS_DIR / "adaptive_loop.py"]


def test_no_pipeline_module_spells_the_two_pass_names() -> None:
    """``raw_16k`` / ``enhanced_16k`` were load-bearing strings in eleven modules.

    The sampling rate is a property of the pipeline, applied to every perturbation equally, so it
    never belonged in a perturbation's name — and a module that compares against a literal name
    is a module a third perturbation has to be added to.
    """
    offenders = [
        f"{path.name}:{n}"
        for path in _pipeline_sources()
        if path.exists() and path.name != "perturbations.py"
        for n, line in enumerate(path.read_text().splitlines(), start=1)
        if "raw_16k" in line or "enhanced_16k" in line
    ]
    assert offenders == [], f"the two-pass vocabulary survives in {offenders}"


def test_only_the_identity_is_named_in_the_pipeline_and_only_through_the_constant() -> None:
    """Comparing a perturbation to ``"enhanced"`` is the two-name assumption in a new spelling.

    The identity is different: it is the one member every run has, so naming it is legitimate —
    but through ``IDENTITY_NAME``, so the string exists once.
    """
    offenders: list[str] = []
    for path in _pipeline_sources():
        if not path.exists() or path.name in {"perturbations.py", "contracts.py"}:
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, ast.Compare):
                continue
            operands = [node.left, *node.comparators]
            for operand in operands:
                if isinstance(operand, ast.Constant) and operand.value in {"raw", "enhanced"}:
                    offenders.append(f"{path.name}:{operand.lineno}")
    assert offenders == [], f"a perturbation name compared as a literal in {offenders}"


def test_the_variant_vocabulary_is_the_transform_vocabulary() -> None:
    """One list, not two that have to agree.

    ``StageContext`` validated ``variant`` against its own copy of three strings. A perturbation
    could therefore declare a transform the context rejected, or the reverse.
    """
    from senselab.audio.workflows.audio_analysis.stage_context import _VARIANT_NAMES

    assert set(_VARIANT_NAMES) == set(TRANSFORMS)


def test_regenerating_a_perturbation_dispatches_on_its_transform(tmp_path: Path) -> None:
    """``get_stream_wav`` must not know any perturbation's name but the identity's.

    It branched on ``raw_16k`` / ``enhanced_16k``; a third perturbation was an edit here. Now the
    run's register says what each one *is*, and an unregistered name is refused rather than
    silently treated as unknown-but-plausible.
    """
    from senselab.audio.workflows.audio_analysis.adaptive.audio_io import _declared_transform

    ctx = {"perturbations": [p.to_json() for p in (identity(), _third())]}
    assert _declared_transform(ctx, "raw") == "unmodified"
    assert _declared_transform(ctx, "other") == "speech_enhanced"
    assert _declared_transform(ctx, "never_declared") is None
    # With no register at all the identity still resolves: a live-audio caller that never ran L1
    # still has the recording, and refusing it would make the loop depend on an artifact it does
    # not need.
    assert _declared_transform({}, IDENTITY_NAME) == "unmodified"
    assert _declared_transform({}, "enhanced") is None
