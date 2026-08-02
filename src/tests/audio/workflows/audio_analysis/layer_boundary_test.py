"""Guards for the four boundaries whose absence let every defect in this feature survive.

Each of these was found by inspecting a real run's outputs, and none by a test — because each
failure was *silent*. An absent directory returns ``{}``, a glob matches nothing, a projection
matches zero keys, a field read by the wrong name returns ``None``. Nothing raised in any of
them. These tests make each one raise.

The four:

1. **Nothing under ``L1/`` is named for an axis.** An axis is a fold across signals *and* across
   passes, so it can be neither produced by one pass nor stored under one. (register item 25)
2. **No pipeline module reads a path under ``final/``.** A deliverable something reads is an
   intermediate wearing the wrong name. (register item 26)
3. **A threshold-derived value carries the policy that produced it.** L2's one-line test.
4. **A field is not read by a name that does not exist on the rows being read.** Three live bugs
   in this feature were exactly this, and each read as "the measurement was absent".
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any

import pytest

WORKFLOW_DIR = Path(__file__).resolve().parents[4] / "senselab" / "audio" / "workflows" / "audio_analysis"
SCRIPTS_DIR = Path(__file__).resolve().parents[4].parent / "scripts"
AXIS_NAMES = ("speech_presence", "speaker", "asr", "uncertainty")


def _pipeline_sources() -> list[Path]:
    """Every module that participates in producing a run, plus the two CLI drivers."""
    files = sorted(WORKFLOW_DIR.rglob("*.py"))
    files += [SCRIPTS_DIR / "analyze_audio.py", SCRIPTS_DIR / "adaptive_loop.py"]
    return [f for f in files if f.exists()]


# ── 1. Nothing under L1/ is named for an axis ────────────────────────────────


def test_no_writer_puts_an_axis_under_l1(tmp_path: Path) -> None:
    """A run's ``L1/`` tree must contain no path segment naming an axis.

    Exercised on a real write rather than by reading source, so a future writer that constructs
    the path some other way is caught too.
    """
    from senselab.audio.workflows.audio_analysis.io import write_signal_parquet, write_signal_stability
    from senselab.audio.workflows.audio_analysis.layout import evidence_dir, pass_dir, stability_dir
    from senselab.audio.workflows.audio_analysis.types import SignalResult, SignalRow

    write_signal_parquet(
        SignalResult(
            pass_label="raw_16k",
            signal="diar_pyannote",
            rows=[SignalRow(start=0.0, end=0.5, signal="diar_pyannote", measurement={"covered_fraction": 1.0})],
        ),
        pass_dir(tmp_path, "raw_16k") / "signals" / "diar_pyannote.parquet",
    )
    write_signal_stability(
        [
            {
                "start": 0.0,
                "end": 0.5,
                "signal": "diar_pyannote",
                "pass_a": "raw_16k",
                "pass_b": "enhanced_16k",
                "abs_delta": 0.1,
                "n_passes_present": 2,
            }
        ],
        stability_dir(tmp_path) / "diar_pyannote.parquet",
    )

    offenders = [
        str(p.relative_to(evidence_dir(tmp_path)))
        for p in evidence_dir(tmp_path).rglob("*")
        if any(part == axis or part.startswith(f"{axis}.") for part in p.parts for axis in AXIS_NAMES)
    ]
    assert not offenders, f"L1/ must not name an axis; found {offenders}"


def test_signal_row_carries_no_axis_and_no_fold() -> None:
    """The L1 row type has no axis field and no cross-signal reduction."""
    from senselab.audio.workflows.audio_analysis.types import SignalRow

    fields = set(SignalRow.__dataclass_fields__)
    assert "axis" not in fields
    forbidden = {
        "within_pass_uncertainty",
        "raw_within_pass_uncertainty",
        "intensity_weight",
        "speech_presence_confidence",
        "speech_presence_uncertainty",
        "token_entropy",
        "scene_quality_coupling",
        "signal_uncertainty",
        "contributing_models",
    }
    assert not (fields & forbidden), f"L1 row must carry no fold; found {sorted(fields & forbidden)}"


def test_fused_axis_has_no_pass_index() -> None:
    """An axis aggregates across passes, so it cannot be indexed by one."""
    import typing

    from senselab.audio.workflows.audio_analysis import types as workflow_types

    assert "pass_label" not in workflow_types.FusedAxis.__dataclass_fields__
    assert set(typing.get_args(workflow_types.PassLabel)) == {
        "raw_16k",
        "enhanced_16k",
    }, "raw_vs_enhanced is not a pass"


def test_the_name_within_pass_uncertainty_is_gone_from_the_workflow_layer() -> None:
    """A per-pass axis is a contradiction, and the name taught the vocabulary that hid it.

    The adaptive subsystem still carries it as the belief store's own per-bucket column. That is
    a *per-(stream, axis, bucket)* value the store computes for itself, not something L1 hands it,
    and collapsing its stream index is tracked as the remainder of register item 27. Nothing in
    the workflow layer may name it.
    """
    workflow = [f for f in WORKFLOW_DIR.glob("*.py")] + [SCRIPTS_DIR / "analyze_audio.py"]
    offenders = [
        f"{f.name}:{n}"
        for f in workflow
        if f.exists()
        for n, line in enumerate(f.read_text().splitlines(), start=1)
        if "within_pass_uncertainty" in line
    ]
    assert not offenders, f"within_pass_uncertainty must not survive in the workflow layer: {offenders}"


# ── 2. No pipeline module reads a path under final/ ──────────────────────────

_FINAL_READ = re.compile(
    r"""final_dir\([^)]*\)\s*/\s*["'][\w.]+["']\s*\)?\s*\.\s*(read_text|read_bytes|open|exists)"""
    r"""|read_parquet\(\s*final""",
)


def test_no_pipeline_module_reads_under_final() -> None:
    """``final/`` holds the answer; a stage that reads it is treating a deliverable as state.

    The evaluator is exempt by design — it scores the deliverable, so it is a consumer of the
    answer rather than a stage that builds it.
    """
    exempt = {"evaluate.py", "layer_boundary_test.py"}
    offenders: list[str] = []
    for path in _pipeline_sources():
        if path.name in exempt:
            continue
        for lineno, line in enumerate(path.read_text().splitlines(), start=1):
            if _FINAL_READ.search(line):
                offenders.append(f"{path.name}:{lineno}: {line.strip()}")
    assert not offenders, "pipeline stages must not read final/:\n" + "\n".join(offenders)


def test_run_summary_deliverable_carries_no_l1_evidence(tmp_path: Path) -> None:
    """``final/summary.json`` must not inline what already exists under ``L1/``."""
    import json
    import sys

    sys.path.insert(0, str(SCRIPTS_DIR))
    try:
        from analyze_audio import _write_run_summary  # type: ignore[import-not-found]
    finally:
        sys.path.pop(0)

    summaries: dict[str, Any] = {
        "input_audio": "/tmp/x.wav",
        "passes": {"raw_16k": {"duration_s": 4.0, "audio_signature": "a" * 64, "features": {"huge": [0] * 1000}}},
        "global_uncertainty": {"combined_uncertainty": 0.2},
    }
    _write_run_summary(tmp_path, summaries)

    deliverable = json.loads((tmp_path / "final" / "summary.json").read_text())
    assert "passes" not in deliverable
    assert deliverable["global_uncertainty"]["combined_uncertainty"] == 0.2

    index = json.loads((tmp_path / "L1" / "passes.json").read_text())
    assert index["passes"]["raw_16k"] == {"duration_s": 4.0, "audio_signature": "a" * 64}
    assert index["input_audio"] == "/tmp/x.wav"


# ── 3. A threshold-derived value names the policy that produced it ───────────


def test_label_bins_travel_with_the_policy_that_produced_them() -> None:
    """A track that says "high" is a thresholded value, so the thresholds ride with it."""
    from senselab.audio.workflows.audio_analysis.labelstudio import attach_uncertainty_tracks_to_ls
    from senselab.audio.workflows.audio_analysis.types import FusedAxis

    tasks = [{"data": {"pass": "raw_16k"}, "predictions": [{"result": []}]}]
    out, _ = attach_uncertainty_tracks_to_ls(
        ls_tasks=tasks,
        ls_config="<View></View>",
        fused_axes={"speaker": FusedAxis(axis="speaker", rows=[{"start": 0.0, "end": 0.5, "uncertainty": 0.9}])},
    )
    policy = out[0]["data"]["uncertainty_bin_policy"]
    assert {"policy", "low_threshold", "high_threshold"} <= set(policy)


def test_link_records_the_presence_policy_on_every_pass() -> None:
    """Every threshold that turned a measurement into a belief is named in the provenance."""
    from senselab.audio.workflows.audio_analysis.votes import PassHarvest, link_pass

    linked = link_pass(
        PassHarvest(
            pass_label="raw_16k",
            speech_presence_evidence=[{"start": 0.0, "end": 0.5, "evidence": {"m": {"covered_fraction": 1.0}}}],
        ),
        params={},
    )
    assert linked.provenance["speech_presence_policy"], "the link's thresholds must be recorded"
    for result in linked.signal_results.values():
        assert result.provenance["speech_presence_policy"]


def test_scene_coupling_records_its_weights_and_pre_coupling_value() -> None:
    """A multiplier applied at L2 must be re-decidable from the row it changed."""
    from senselab.audio.workflows.audio_analysis.compute import _apply_scene_coupling
    from senselab.audio.workflows.audio_analysis.types import FusedAxis

    axes = {
        "speech_presence": FusedAxis(axis="speech_presence", rows=[{"start": 0.0, "end": 1.0, "quality_snr": 1.0}]),
        "asr": FusedAxis(axis="asr", rows=[{"start": 0.0, "end": 1.0, "triage_score": 0.4}]),
    }
    _apply_scene_coupling(axes, {})
    row = axes["asr"].rows[0]
    assert row["triage_score_pre_coupling"] == pytest.approx(0.4)
    assert row["scene_quality_coupling"] > 1.0
    assert axes["asr"].provenance["asr_scene_coupling"]["weights"]


# ── 4. No field is read by a name that does not exist on the rows read ───────

_FUSED_AXIS_FIELDS = {
    "start",
    "end",
    "uncertainty",
    "epistemic_uncertainty",
    "confidence",
    "variability",
    "triage_score",
    "contributing_signals",
    "contributing_passes",
    "signal_weights",
    "weight_basis",
    "round",
}


def test_fuse_axis_emits_exactly_the_fields_its_consumers_read() -> None:
    """The fused row's column set is the contract every downstream reader is written against.

    Pinned because the failure mode is silent: ``row.get("speech_presence_confidence")`` on a row
    that has no such key returns ``None`` and takes a fallback, which is how
    ``final/speech_presence.parquet`` came to carry ``p_voice`` under a column name promising the
    calibrated quantity.
    """
    from senselab.audio.workflows.audio_analysis.fuse import fuse_axis

    rows = fuse_axis(
        {"raw_16k": [{"start": 0.0, "end": 0.5, "votes": {"m": {"value": 0.3}}}]},
        weights={"m": 1.0},
    )
    assert rows and set(rows[0]) == _FUSED_AXIS_FIELDS


def test_reliability_compares_a_field_the_evidence_actually_has() -> None:
    """Stability must be measured on the same quantity the fold consumes.

    The previous implementation matched a fixed tuple of vote field names, none of which the
    presence harvest emits — so ``signal_stability(axis="speech_presence")`` returned ``{}`` on
    every real run, every presence signal kept weight 1.0, and the factor silently never applied.
    It floored correctly, which is exactly why nobody noticed.
    """
    from senselab.audio.workflows.audio_analysis.reliability import signal_stability
    from senselab.audio.workflows.audio_analysis.speech_presence_link import votes_for_harvest
    from senselab.audio.workflows.audio_analysis.votes import PassHarvest

    def _harvest(label: str, covered: float) -> PassHarvest:
        return PassHarvest(
            pass_label=label,
            speech_presence_evidence=[
                {"start": 0.0, "end": 0.5, "evidence": {"diar_a": {"covered_fraction": covered}}}
            ],
        )

    harvests = {"raw_16k": _harvest("raw_16k", 1.0), "enhanced_16k": _harvest("enhanced_16k", 0.0)}
    buckets = {label: votes_for_harvest(h) for label, h in harvests.items()}
    instability = signal_stability(harvests, axis="speech_presence", buckets_by_pass=buckets)
    assert "diar_a" in instability, "a signal that flips between passes must be measured as unstable"
    assert instability["diar_a"] > 0.0


def test_belief_store_meta_columns_are_measurements_only() -> None:
    """The store carries measurements about a bucket, never a fold or an L2 decision."""
    from senselab.audio.workflows.audio_analysis.adaptive.belief import _META_COLUMNS

    forbidden = {
        "within_pass_uncertainty",
        "raw_within_pass_uncertainty",
        "speech_presence_confidence",
        "speech_presence_uncertainty",
        "comparison_status",
        "intensity_weight",
        "scene_quality_coupling",
    }
    assert not (set(_META_COLUMNS) & forbidden)


def test_module_docstrings_do_not_promise_a_reader_that_does_not_exist() -> None:
    """``layout.stability_dir`` must describe what the files under it are actually for.

    Its docstring asserted the cross-pass deltas "feed L2's weights". They fed nothing: the
    weights came from an in-memory computation, and the parquets had no reader in src/ or
    scripts/. A docstring documenting an intent the code does not implement is the same class of
    defect as a glob that matches nothing.
    """
    from senselab.audio.workflows.audio_analysis import layout

    doc = layout.stability_dir.__doc__ or ""
    assert "signal" in doc.lower(), "stability is a property of a signal, and the docstring must say so"
    tree = ast.parse(Path(layout.__file__).read_text())
    module_doc = ast.get_docstring(tree) or ""
    # The module docstring may explain what `raw_vs_enhanced` was and why it is gone; it may not
    # describe it as something the layout still contains.
    assert "uncertainty parquets record what one pass alone" not in module_doc
