"""Guards for the boundaries whose absence let every defect in this feature survive.

Each of these was found by inspecting a real run's outputs, and none by a test — because each
failure was *silent*. An absent directory returns ``{}``, a glob matches nothing, a projection
matches zero keys, a field read by the wrong name returns ``None``. Nothing raised in any of
them. These tests make each one raise.

Three of the guards read a **real run directory**, not a tree this file builds. A guard that
walks its own fixture only ever proves that the fixture is consistent with itself; the defects
this feature shipped were all in what the pipeline actually wrote. When no run is available the
guards :func:`pytest.skip` with the reason, so "did not run" stays distinguishable from "found
nothing" — which is the failure signature the whole file exists to remove.

The rules:

1. **Nothing under ``L1/`` carries a fold.** An axis is an aggregate across signals *and* across
   passes, so it can be neither produced by one pass nor stored under one. Keyed on **shape**, not
   on names: the violation is a parquet carrying an ``axis`` column or a column whose value is an
   aggregate across signals. ``L1/<pass>/asr/<model>.json`` merely *shares* a name with an axis —
   it is one model's raw transcript, and no name list can tell the two apart. (register item 25)
2. **No pipeline module reads a path under ``final/``.** Checked by resolving aliases in the AST:
   ``final = final_dir(out_dir)`` followed by ``final / "transcript.json"`` is the same violation
   as writing it on one line, and a regex sees only the second. (register item 26)
3. **No artifact is keyed by both a pass and an axis.** A vote may be per-pass — a signal measured
   on a pass is a real per-pass measurement — but an axis may not, because a pass is an input
   dimension to the fold and never an index on its output. (register item 27)
4. **A threshold-derived value carries the policy that produced it.** L2's one-line test.
5. **A field is not read by a name that does not exist on the rows being read.** Three live bugs
   in this feature were exactly this, and each read as "the measurement was absent".
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, Iterator

import pytest

REPO_ROOT = Path(__file__).resolve().parents[5]
WORKFLOW_DIR = REPO_ROOT / "src" / "senselab" / "audio" / "workflows" / "audio_analysis"
SCRIPTS_DIR = REPO_ROOT / "scripts"
RUNS_DIR = REPO_ROOT / "artifacts" / "analyze_audio"


def _pipeline_sources() -> list[Path]:
    """Every module that participates in producing a run, plus the two CLI drivers."""
    files = sorted(WORKFLOW_DIR.rglob("*.py"))
    files += [SCRIPTS_DIR / "analyze_audio.py", SCRIPTS_DIR / "adaptive_loop.py"]
    return [f for f in files if f.exists()]


# ── the real run these guards are read against ───────────────────────────────


@pytest.fixture(scope="session")
def real_run_dir() -> Path:
    """The newest completed run under ``artifacts/analyze_audio/``.

    A run counts as complete when it has both an ``L1/`` and an ``L2/`` directory — the two the
    layout guards are about. Skips rather than passing when there is none, because a layout guard
    with nothing to walk reports exactly what a layout guard that found no violation reports.
    """
    candidates = (
        [p for p in RUNS_DIR.iterdir() if p.is_dir() and (p / "L1").is_dir() and (p / "L2").is_dir()]
        if RUNS_DIR.is_dir()
        else []
    )
    if not candidates:
        pytest.skip(
            f"no completed run under {RUNS_DIR} (need one with L1/ and L2/). "
            "These guards read what the pipeline actually wrote; with no run they would pass "
            "vacuously. Produce one with: uv run python scripts/analyze_audio.py <audio>"
        )
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _columns(path: Path) -> set[str]:
    """The column names of a parquet, read from its schema alone."""
    import pyarrow.parquet as pq

    return set(pq.read_schema(path).names)


#: Columns whose value is an aggregate across signals. A row carrying one of these has had a fold
#: applied to it; ``axis`` says so outright. Named, rather than inferred, because the whole point
#: is that the set is auditable — a new fold column is added here or it is not guarded.
FOLD_COLUMNS = frozenset(
    {
        "uncertainty",
        "epistemic_uncertainty",
        "triage_score",
        "within_pass_uncertainty",
        "contributing_signals",
    }
)

#: A row attributed to one source or one signal is a measurement or a vote, not a fold — even when
#: it carries an ``axis`` column saying which fold it will feed.
PER_SOURCE_COLUMNS = frozenset({"source", "signal"})

#: The ways an artifact can be indexed by a pass. ``elected_stream`` is one of them: naming the
#: pass whose reading was *taken as* the axis's is a per-pass axis with the index moved into the
#: value. ``contributing_passes`` and ``folded_from`` are not — they say which passes fed the fold,
#: which is provenance about an input dimension rather than a selection among outputs.
PASS_COLUMNS = frozenset({"stream", "pass_label", "pass", "elected_stream"})


def _fold_columns(columns: set[str]) -> list[str]:
    """Which of a parquet's columns make it a fold. Empty when it is a measurement or a vote."""
    if columns & PER_SOURCE_COLUMNS:
        return []
    return sorted((columns & FOLD_COLUMNS) | ({"axis"} & columns))


def _pass_labels(run_dir: Path) -> set[str]:
    """The pass names this run actually used, taken from ``L1/`` rather than assumed."""
    return {p.name for p in (run_dir / "L1").iterdir() if p.is_dir() and p.name != "stability"}


# ── 1. Nothing under L1/ carries a fold ──────────────────────────────────────


def test_nothing_under_l1_carries_a_fold(real_run_dir: Path) -> None:
    """``L1/`` is evidence: each signal's own measurement, in that signal's own units.

    Keyed on the shape of what was written, not on a list of axis names. A name list is guarded
    only against the axes that existed when it was written, so an axis added later is unguarded by
    construction — which is how ``L1/<pass>/background_mask.parquet`` came to sit under a pass
    carrying a per-region ``uncertainty`` folded from every presence signal and thresholded by a
    detection-margin profile.
    """
    offenders = [
        f"L1/{p.relative_to(real_run_dir / 'L1')}: {_fold_columns(_columns(p))}"
        for p in sorted((real_run_dir / "L1").rglob("*.parquet"))
        if _fold_columns(_columns(p))
    ]
    assert not offenders, f"{real_run_dir.name}: L1 stores measurements, never a fold across signals:\n" + "\n".join(
        offenders
    )


def test_the_l1_guard_discriminates_by_shape_and_not_by_name(real_run_dir: Path) -> None:
    """``L1/<pass>/asr/<model>.json`` is named for an axis and is not a violation.

    It is one model's raw transcript — evidence, at that model's own resolution, in that model's
    own units. Only shape separates it from a fold that happens to be stored under a different
    name, so this test asserts the discrimination is actually exercised: if no such file exists,
    rule 1 above has never had to make the distinction and its passing means less than it looks.
    """
    named_for_an_axis = sorted((real_run_dir / "L1").rglob("asr/*.json"))
    assert named_for_an_axis, "no per-model transcript under L1/<pass>/asr/ — the shape rule is untested here"
    signals = sorted((real_run_dir / "L1").rglob("signals/*.parquet"))
    assert signals, "no per-signal measurement under L1/<pass>/signals/"
    assert not any(_fold_columns(_columns(p)) for p in signals)


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
    """A per-pass axis is a contradiction, and the name taught the vocabulary that hid it."""
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

#: Path methods that read. ``mkdir`` and ``write_text`` are absent on purpose: a stage may *write*
#: its deliverable into ``final/``; what it may not do is take one back out as input.
_READ_METHODS = frozenset({"read_text", "read_bytes", "open", "glob", "rglob", "iterdir", "is_file", "is_dir"})

#: Free/attribute calls that read a path passed as an argument.
_READ_FUNCTIONS = frozenset({"open", "read_parquet", "read_json", "read_csv", "read_table", "load"})


def _final_rooted(node: ast.AST, roots: set[str]) -> bool:
    """Is this expression a path under ``final/``?

    True for a ``final_dir(...)`` call, for anything divided out of one, for ``<x> / "final"``,
    and for any local name bound to one of those.
    """
    if isinstance(node, ast.Call):
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
        return name == "final_dir"
    if isinstance(node, ast.Name):
        return node.id in roots
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        if _final_rooted(node.left, roots):
            return True
        right = node.right
        if isinstance(right, ast.Constant) and right.value == "final":
            return True
        if isinstance(right, ast.Name) and right.id == "FINAL_DIR":
            return True
        if isinstance(right, ast.Attribute) and right.attr == "FINAL_DIR":
            return True
    return False


def _own_nodes(scope: ast.AST) -> Iterator[ast.AST]:
    """Every node belonging to this scope, without descending into a nested one."""
    for child in ast.iter_child_nodes(scope):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            continue
        yield child
        yield from _own_nodes(child)


def _nested_scopes(scope: ast.AST) -> Iterator[ast.AST]:
    """The scopes defined directly inside this one."""
    for child in ast.iter_child_nodes(scope):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            yield child
        else:
            yield from _nested_scopes(child)


def _scope_reads(scope: ast.AST, inherited: frozenset[str], path: Path, lines: list[str]) -> list[str]:
    """Reads of a ``final/`` path in this scope and the scopes inside it."""
    roots = set(inherited)
    assignments = [n for n in _own_nodes(scope) if isinstance(n, (ast.Assign, ast.AnnAssign, ast.NamedExpr))]
    # A fixpoint, because aliases chain: ``final = final_dir(d)`` then ``tasks = final / "t.json"``.
    for _ in range(len(assignments) + 1):
        before = len(roots)
        for node in assignments:
            value = node.value
            if value is None or not _final_rooted(value, roots):
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            roots.update(t.id for t in targets if isinstance(t, ast.Name))
        if len(roots) == before:
            break

    offenders: list[str] = []
    for call in (n for n in _own_nodes(scope) if isinstance(n, ast.Call)):
        func = call.func
        hit = False
        if isinstance(func, ast.Attribute) and func.attr in _READ_METHODS and _final_rooted(func.value, roots):
            hit = True
        elif isinstance(func, ast.Attribute) and func.attr in _READ_FUNCTIONS:
            hit = any(_final_rooted(a, roots) for a in call.args)
        elif isinstance(func, ast.Name) and func.id in _READ_FUNCTIONS:
            hit = any(_final_rooted(a, roots) for a in call.args)
        if hit:
            offenders.append(f"{path.name}:{call.lineno}: {lines[call.lineno - 1].strip()}")

    for child in _nested_scopes(scope):
        offenders += _scope_reads(child, frozenset(roots), path, lines)
    return offenders


def test_no_pipeline_module_reads_under_final(real_run_dir: Path) -> None:
    """``final/`` holds the answer; a stage that reads it is treating a deliverable as state.

    Resolved through the AST rather than matched as text. The regex this replaces required the
    read to be attached to the ``final_dir(...)`` call in one expression, so binding the directory
    to a name first — which is what every real caller does — walked straight past it, and the
    guard passed while three live reads sat in the tree.

    The evaluator is exempt by design: it scores the deliverable, so it is a consumer of the
    answer rather than a stage that builds it.
    """
    exempt = {"evaluate.py", Path(__file__).name}
    offenders: list[str] = []
    for path in _pipeline_sources():
        if path.name in exempt:
            continue
        source = path.read_text()
        offenders += _scope_reads(ast.parse(source), frozenset(), path, source.splitlines())
    assert not offenders, "pipeline stages must not read final/:\n" + "\n".join(offenders)


def test_the_final_read_guard_sees_through_an_alias() -> None:
    """The rule is about the path, not about how many expressions it took to name it.

    Both forms below are the same violation; the previous regex caught only the first. Asserted
    directly so the guard's discriminating power is itself tested, rather than inferred from the
    tree happening to be clean.
    """
    inline = "def f(d):\n    return json.loads((final_dir(d) / 'transcript.json').read_text())\n"
    aliased = (
        "def f(d):\n    final = final_dir(d)\n    p = final / 'transcript.json'\n    return json.loads(p.read_text())\n"
    )
    writing = (
        "def f(d):\n"
        "    final = final_dir(d)\n"
        "    final.mkdir(parents=True, exist_ok=True)\n"
        "    (final / 'x.json').write_text('{}')\n"
    )
    for src in (inline, aliased):
        assert _scope_reads(ast.parse(src), frozenset(), Path("probe.py"), src.splitlines())
    assert not _scope_reads(ast.parse(writing), frozenset(), Path("probe.py"), writing.splitlines())


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


# ── 3. No artifact is keyed by both a pass and an axis ───────────────────────


def test_no_artifact_is_keyed_by_both_a_pass_and_an_axis(real_run_dir: Path) -> None:
    """An axis IS an aggregator — across signals and across passes alike.

    A pass is an input dimension to the fold, never an index on its output, so a per-pass axis is
    a category error rather than a redundancy. It is detectable by shape: rows carrying a fold
    *and* a pass index. Votes are exempt and stay per-pass, which is what makes perturbation
    stability computable at all — a vote carries a ``source``, and that is what tells the two
    apart without consulting a filename.
    """
    labels = _pass_labels(real_run_dir)
    offenders: list[str] = []
    for path in sorted(real_run_dir.rglob("*.parquet")):
        columns = _columns(path)
        folds = _fold_columns(columns)
        if not folds:
            continue
        relative = path.relative_to(real_run_dir)
        keys = sorted(columns & PASS_COLUMNS) + [part for part in relative.parts if part in labels]
        if keys:
            offenders.append(f"{relative}: fold {folds} keyed by pass {keys}")
    assert not offenders, (
        f"{real_run_dir.name}: an axis is a fold across passes, so it cannot be indexed by one:\n"
        + "\n".join(offenders)
    )


def test_a_fold_has_one_row_per_bucket_not_one_per_pass(real_run_dir: Path) -> None:
    """The arithmetic consequence of rule 3, checked so the rule cannot be satisfied on paper.

    Dropping a ``stream`` column while still emitting one row per (pass, bucket) leaves the
    category error intact and merely unlabelled: the file would then carry two rows claiming the
    same bucket. Every fold's ``(start, end)`` pairs must therefore be unique.
    """
    import pyarrow.parquet as pq

    offenders: list[str] = []
    for path in sorted(real_run_dir.rglob("*.parquet")):
        columns = _columns(path)
        if not _fold_columns(columns) or not {"start", "end"} <= columns:
            continue
        table = pq.read_table(path, columns=["start", "end"])
        buckets = list(zip(table.column("start").to_pylist(), table.column("end").to_pylist()))
        if len(buckets) != len(set(buckets)):
            offenders.append(f"{path.relative_to(real_run_dir)}: {len(buckets)} rows over {len(set(buckets))} buckets")
    assert not offenders, f"{real_run_dir.name}: a fold has one row per bucket:\n" + "\n".join(offenders)


def test_no_belief_api_takes_a_pass_to_produce_an_axis() -> None:
    """Rule 3 applied to the API, because an artifact guard can only see the last step.

    The writer can collapse two per-pass rows into one at the moment it writes, and every
    artifact guard above then passes while the loop itself still holds — and reasons over — one
    axis value per pass: regions proposed per (pass, axis), convergence marked per (pass, axis),
    a fusion stream elected by comparing one pass's axis against another's. The collapse is then
    a *presentation* of the category error rather than its absence, and every reader that reaches
    past the parquet still sees two answers.

    So the shape is asserted where it is decided: an axis-producing call may not take a pass.
    Vote-level calls are exempt and must stay per-pass — that is what makes perturbation
    stability computable — so they are listed here by name rather than by omission.
    """
    import inspect

    from senselab.audio.workflows.audio_analysis.adaptive import regions as regions_module
    from senselab.audio.workflows.audio_analysis.adaptive.belief import BeliefState, VoteStore

    axis_producing = [
        VoteStore.reaggregate_bucket,
        VoteStore.buckets,
        BeliefState.axis_rows,
        BeliefState.uncertainty_mass,
        BeliefState.update_buckets,
        BeliefState.from_store,
        regions_module.propose_regions,
    ]
    offenders = [
        f"{fn.__qualname__}({name})"
        for fn in axis_producing
        for name in inspect.signature(fn).parameters
        if name in {"stream", "streams", "passes", "pass_label"}
    ]
    assert not offenders, (
        "an axis is a fold across passes, so nothing that produces one may be indexed by a pass:\n"
        + "\n".join(offenders)
    )


# ── 4. A threshold-derived value names the policy that produced it ───────────


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


# ── 5. No field is read by a name that does not exist on the rows read ───────

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
