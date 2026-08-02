"""Conformance of the pipeline to its declared stage contracts (D-17).

Three rounds of guards were written against the violation last found, and each missed the next
instance of the same class. This file checks the opposite direction: every stage declares what
it may read and what it may write in
:mod:`senselab.audio.workflows.audio_analysis.contracts`, and these tests fail when code or an
artifact tree steps outside the declaration.

**Every rule here has been seen to fail.** Prior rounds shipped guards that had never been
observed failing, and three of them turned out not to. So each rule below is paired with a test
that *constructs* the violation, asserts it is caught, removes it, and asserts the guard then
goes quiet — the second half matters as much as the first, because a guard that fires on
everything is as uninformative as one that fires on nothing.

The four rule families:

1. **The DAG is acyclic.** Edges are derived by matching one stage's declared reads against
   another's declared writes, so `final/` being read by the pipeline and a round reading a
   sibling are the same finding — a cycle — rather than two separately-discovered defects.
2. **Static conformance.** The AST of every pipeline module, the whole package including
   ``adaptive/`` and both CLI drivers, with local aliases resolved to a fixpoint.
3. **Artifact conformance.** A real run's tree, which catches what static analysis cannot: a
   writer reached through a helper, or a file nobody meant to emit.
4. **The register does not rot.** A registered deviation that stops matching fails, so closing a
   violation requires deleting its entry.

``layer_boundary_test.py`` keeps its rules 4 and 5 — a threshold-derived value naming the policy
that produced it, and a field read by a name the rows actually have. Neither is a path rule and
D-17 does not replace them. Its rules 1-3 are superseded and go with the restructure.
"""

from __future__ import annotations

import graphlib
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from senselab.audio.workflows.audio_analysis.contracts import (
    DAG_STAGES,
    KNOWN_DEVIATIONS,
    MODULE_STAGE,
    STAGE_CONTRACTS,
    Artifact,
    StageContract,
    artifact_violations,
    check_source,
    dag_edges,
    dead_static_deviations,
    matches,
    overlap,
    pipeline_sources,
    static_violations,
    topological_order,
    unrolled_contracts,
    unwaived,
    unwaived_artifacts,
)

REPO_ROOT = Path(__file__).resolve().parents[5]
RUNS_DIR = REPO_ROOT / "artifacts" / "analyze_audio"

#: A module standing in for one under each contract, so a proof case can be written against a
#: stage without depending on which real file happens to be assigned to it.
PROBE = "probe.py"


# ── the pattern language ─────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("path", "pattern", "expected"),
    [
        ("L1/signals/brouhaha.parquet", "L1/signals/**", True),
        ("L1/signals", "L1/signals/**", True),
        ("L1/raw/signals/brouhaha.parquet", "L1/signals/**", False),
        ("L2/round/0/estimates/asr.parquet", "L2/round/*/estimates/*.parquet", True),
        ("L2/round0/uncertainty/asr.parquet", "L2/round/*/estimates/*.parquet", False),
        ("final/timeline.png", "final/timeline.png", True),
        ("L1/perturbation/gain6/asr/whisper.json", "L1/perturbation/*/**", True),
    ],
)
def test_a_concrete_path_matches_only_the_pattern_that_names_it(path: str, pattern: str, expected: bool) -> None:
    """``*`` is one segment and ``**`` is any number, including none."""
    assert matches(path, pattern) is expected


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        ("L2/round/*/**", "L2/round/1/estimates/asr.parquet", True),
        ("L2/round/0/**", "L2/round/1/**", False),
        ("L1/*/features/*.parquet", "L1/perturbation/*/**", True),
        ("final/**", "L2/round/*/**", False),
        ("L1/signals/**", "L1/signals/x.parquet", True),
    ],
)
def test_two_patterns_overlap_when_some_path_could_fall_under_both(left: str, right: str, expected: bool) -> None:
    """Overlap is what turns declarations into edges, and neither side is concrete until a run."""
    assert overlap(left, right) is expected


# ── the declaration is well formed ───────────────────────────────────────────


def test_every_module_named_in_the_stage_map_exists() -> None:
    """A stale entry is worse than a missing one: it declares a permission nothing needs."""
    missing = [module for module in MODULE_STAGE if not (REPO_ROOT / module).is_file()]
    assert not missing, f"MODULE_STAGE names files that do not exist: {missing}"


def test_the_scan_covers_the_whole_package_including_adaptive() -> None:
    """A previous guard globbed one directory and could not see half the round logic."""
    scanned = {p.relative_to(REPO_ROOT).as_posix() for p in pipeline_sources(REPO_ROOT)}
    assert "src/senselab/audio/workflows/audio_analysis/adaptive/loop.py" in scanned
    assert "src/senselab/audio/workflows/audio_analysis/adaptive/interventions.py" in scanned
    assert "scripts/analyze_audio.py" in scanned
    assert "scripts/adaptive_loop.py" in scanned


def test_an_unlisted_module_declares_nothing() -> None:
    """Permission defaults to none — that is what declaring the permitted buys."""
    source = 'def f(run_dir):\n    return (run_dir / "L2" / "x.json").read_text()\n'
    assert check_source("src/senselab/audio/workflows/audio_analysis/statistics.py", source)


def test_no_two_stages_claim_the_same_artifact() -> None:
    """Two producers for one path makes "which stage produces this" unanswerable."""
    claims: list[tuple[str, str]] = [
        (stage, artifact.pattern) for stage in DAG_STAGES for artifact in STAGE_CONTRACTS[stage].instantiate().writes
    ]
    collisions = [
        f"{a_stage}:{a_pattern} vs {b_stage}:{b_pattern}"
        for index, (a_stage, a_pattern) in enumerate(claims)
        for b_stage, b_pattern in claims[index + 1 :]
        if a_stage != b_stage and overlap(a_pattern, b_pattern)
    ]
    assert not collisions, "each artifact has exactly one producing stage:\n" + "\n".join(collisions)


def test_every_deviation_says_why_and_names_a_real_module() -> None:
    """A registered deviation without a reason is a silenced test."""
    for deviation in KNOWN_DEVIATIONS:
        assert deviation.why.strip(), f"{deviation.module} {deviation.op} {deviation.pattern} has no reason"
        if deviation.module:
            assert (REPO_ROOT / deviation.module).is_file(), deviation.module


# ── 1. the DAG is acyclic ────────────────────────────────────────────────────


@pytest.mark.parametrize("n_rounds", [1, 2, 3, 10])
def test_the_pipeline_dag_is_acyclic(n_rounds: int) -> None:
    """L1 -> round 0 -> ... -> round n-1 -> final -> eval, unrolled and topologically sorted."""
    order = topological_order(unrolled_contracts(n_rounds))
    assert order[0] == "L1"
    assert order[-1] == "EVAL"
    assert order.index("FINAL") == len(order) - 2


def test_the_dag_edges_are_derived_from_the_declarations_and_not_restated() -> None:
    """Nothing else defines the graph, so a contract change moves the graph with it."""
    edges = set(dag_edges(unrolled_contracts(3)))
    assert ("L1", "L2_ROUND[0]") in edges, "every round reads L1/signals/"
    assert ("L2_ROUND[0]", "L2_ROUND[1]") in edges, "a round reads its predecessor"
    assert ("L2_ROUND[2]", "FINAL") in edges, "final extracts the last round"
    assert ("FINAL", "EVAL") in edges, "the evaluator scores the deliverable"
    assert ("L2_ROUND[1]", "L2_ROUND[0]") not in edges, "a round never reads its successor"
    assert not any(producer == consumer for producer, consumer in edges), "no stage reads itself"


def test_the_acyclicity_check_fails_on_a_round_that_reads_its_own_output() -> None:
    """The proof that rule 1 can fail.

    "A round reads a sibling updated within the same round" and "the pipeline reads final/" are
    the same defect — a cycle edge — and this is what the check does when handed one. Written as
    a contract rather than by editing the real declaration so the failure is reproducible without
    breaking the tree.
    """
    same_round = StageContract(
        stage="L2_ROUND",
        why="a round reading its own estimates",
        reads=("L2/round/{n}/estimates/**",),
        writes=(Artifact("L2/round/{n}/estimates/*.parquet", "the axes", key=("axis", "bucket")),),
    )
    cyclic = (same_round.instantiate(0),)
    assert ("L2_ROUND[0]", "L2_ROUND[0]") in dag_edges(cyclic)
    with pytest.raises(graphlib.CycleError):
        topological_order(cyclic)

    # ...and with the same-round read removed, the identical stage sorts.
    previous_round = StageContract(
        stage="L2_ROUND",
        why="a round reading its predecessor",
        reads=("L2/round/{prev}/estimates/**",),
        writes=same_round.writes,
    )
    acyclic = (previous_round.instantiate(0), previous_round.instantiate(1))
    assert topological_order(acyclic) == ("L2_ROUND[0]", "L2_ROUND[1]")


# ── 2. static conformance ────────────────────────────────────────────────────


def test_a_read_of_final_reached_through_a_local_alias_is_flagged() -> None:
    """The rule is about the path, not about how many expressions it took to name it.

    The regex this replaces required the read to be attached to the ``final_dir(...)`` call in
    one expression, so binding the directory to a name first — which is what every real caller
    does — walked straight past it.
    """
    aliased = (
        "def f(run_dir):\n"
        "    final = final_dir(run_dir)\n"
        "    path = final / 'transcript.json'\n"
        "    return json.loads(path.read_text())\n"
    )
    findings = check_source(PROBE, aliased, STAGE_CONTRACTS["L2_ROUND"])
    assert [(f.op, f.pattern) for f in findings] == [("read", "final/transcript.json")]

    # Remove the read and the same three lines of aliasing are silent.
    without = "def f(run_dir):\n    final = final_dir(run_dir)\n    path = final / 'transcript.json'\n    return path\n"
    assert check_source(PROBE, without, STAGE_CONTRACTS["L2_ROUND"]) == []


def test_a_read_of_final_reached_through_exists_is_flagged() -> None:
    """An existence probe that decides what a stage does next is a read.

    ``exists`` was absent from the previous guard's method list, which is why
    ``analyze_audio.py`` branches on ``final/labelstudio_tasks.json`` under a guard that
    reported the rule held.
    """
    probing = (
        "def f(run_dir):\n"
        "    if (final_dir(run_dir) / 'labelstudio_tasks.json').exists():\n"
        "        return 'final'\n"
        "    return 'L2'\n"
    )
    findings = check_source(PROBE, probing, STAGE_CONTRACTS["L2_ROUND"])
    assert [(f.op, f.pattern) for f in findings] == [("read", "final/labelstudio_tasks.json")]

    assert check_source(PROBE, "def f(run_dir):\n    return 'L2'\n", STAGE_CONTRACTS["L2_ROUND"]) == []


@pytest.mark.parametrize("method", ["stat", "is_file", "is_dir", "glob", "iterdir", "open", "read_text"])
def test_every_named_read_is_a_read(method: str) -> None:
    """The read vocabulary the rule was specified with, each one exercised."""
    source = f"def f(run_dir):\n    return (final_dir(run_dir) / 'transcript.json').{method}()\n"
    findings = check_source(PROBE, source, STAGE_CONTRACTS["L2_ROUND"])
    assert [f.op for f in findings] == ["read"], method


def test_json_loads_over_read_text_is_a_read_and_read_parquet_is_a_read() -> None:
    """The two spellings the pipeline actually uses."""
    source = (
        "def f(run_dir):\n"
        "    a = json.loads((final_dir(run_dir) / 'transcript.json').read_text())\n"
        "    b = pd.read_parquet(final_dir(run_dir) / 'speech_presence.parquet')\n"
        "    return a, b\n"
    )
    findings = check_source(PROBE, source, STAGE_CONTRACTS["L2_ROUND"])
    assert sorted(f.pattern for f in findings) == ["final/speech_presence.parquet", "final/transcript.json"]
    assert {f.op for f in findings} == {"read"}


def test_an_l1_stage_writing_into_l2_is_flagged() -> None:
    """The cycle edge that lets round 0 depend on a file only an L1 node can create.

    ``stage_background_mask`` runs inside ``run_pass`` and writes ``L2/background_mask.parquet``;
    the driver reads it back as the mask axis's only evidence.
    """
    writing = "def f(ctx, mask):\n    write_background_mask(mask, belief_dir(ctx.run_dir))\n"
    findings = check_source(PROBE, writing, STAGE_CONTRACTS["L1"])
    assert [(f.op, f.pattern) for f in findings] == [("write", "L2")]

    # The same call against its own stage's tree is fine.
    own = "def f(ctx, rows):\n    write_signal_parquet(rows, evidence_dir(ctx.run_dir) / 'signals' / 'x.parquet')\n"
    assert check_source(PROBE, own, STAGE_CONTRACTS["L1"]) == []


def test_l2_reading_l1_outside_signals_is_flagged() -> None:
    """``L1/signals/`` is the only thing L2 reads from L1.

    ``perturbation_dir`` denotes two places — the identity's directory and any other
    perturbation's — so one such read is flagged twice, once per place it could have reached
    into. An access conforms only when *every* path it could name is permitted.
    """
    reaching = (
        "def f(run_dir, stream):\n    return sorted((perturbation_dir(run_dir, stream) / 'asr').glob('*.json'))\n"
    )
    findings = check_source(PROBE, reaching, STAGE_CONTRACTS["L2_ROUND"])
    assert [(f.op, f.pattern) for f in findings] == [
        ("read", "L1/raw/asr/*.json"),
        ("read", "L1/perturbation/*/asr/*.json"),
    ]

    permitted = "def f(run_dir):\n    return sorted(signals_dir(run_dir).glob('*.parquet'))\n"
    assert check_source(PROBE, permitted, STAGE_CONTRACTS["L2_ROUND"]) == []


def test_an_l1_stage_writing_into_its_perturbation_directory_conforms() -> None:
    """Either branch of ``perturbation_dir`` is L1's own tree, and neither is a violation.

    The disjunction has to cut both ways or it is not a check: if reaching into a perturbation
    directory from L2 fails on both branches, writing into one from L1 has to pass on both.
    """
    writing = "def f(run_dir, name):\n    (perturbation_dir(run_dir, name) / 'asr' / 'whisper.json').write_text('{}')\n"
    assert check_source(PROBE, writing, STAGE_CONTRACTS["L1"]) == []


def test_the_resolver_chases_an_alias_chain_to_a_fixpoint() -> None:
    """Aliases chain, and a single pass sees the first binding and not the second."""
    chained = (
        "def f(run_dir):\n"
        "    belief = belief_dir(run_dir)\n"
        "    rounds = belief / 'rounds'\n"
        "    first = rounds / '1'\n"
        "    return (first / 'summary.json').read_text()\n"
    )
    findings = check_source(PROBE, chained, STAGE_CONTRACTS["FINAL"])
    assert [(f.op, f.pattern) for f in findings] == [("read", "L2/rounds/1/summary.json")]


def test_a_hand_built_path_is_resolved_even_without_the_layout_helper() -> None:
    """``Path(run_dir) / "L2" / "round0" / "votes"`` is a third spelling of one location."""
    hand_built = (
        "def f(run_dir):\n"
        "    votes_dir = Path(run_dir) / 'L2' / 'round0' / 'votes'\n"
        "    return sorted(votes_dir.glob('*.parquet'))\n"
    )
    findings = check_source(PROBE, hand_built, STAGE_CONTRACTS["L1"])
    assert [(f.op, f.pattern) for f in findings] == [("read", "L2/round0/votes/*.parquet")]


def test_opening_for_writing_is_a_write_and_opening_for_reading_is_a_read() -> None:
    """Mode decides, and read is the default as it is in Python."""
    writing = "def f(run_dir):\n    with open(final_dir(run_dir) / 'summary.md', 'w') as fh:\n        fh.write('x')\n"
    reading = "def f(run_dir):\n    with open(final_dir(run_dir) / 'summary.md') as fh:\n        return fh.read()\n"
    assert [f.op for f in check_source(PROBE, writing, STAGE_CONTRACTS["L1"])] == ["write"]
    assert [f.op for f in check_source(PROBE, reading, STAGE_CONTRACTS["L1"])] == ["read"]


def test_making_a_stage_its_own_directory_is_not_a_write_but_making_anothers_is() -> None:
    """Five dead ``final.mkdir(...)`` calls sit in modules that write nothing there."""
    own = "def f(run_dir):\n    evidence_dir(run_dir).mkdir(parents=True, exist_ok=True)\n"
    other = "def f(run_dir):\n    final_dir(run_dir).mkdir(parents=True, exist_ok=True)\n"
    assert check_source(PROBE, own, STAGE_CONTRACTS["L1"]) == []
    assert [(f.op, f.pattern) for f in check_source(PROBE, other, STAGE_CONTRACTS["L1"])] == [("write", "final")]


def test_a_stage_reads_and_writes_its_own_declared_outputs_without_complaint() -> None:
    """The negative control: a guard that fires on everything says nothing."""
    conformant = (
        "def f(run_dir):\n"
        "    signals = evidence_dir(run_dir) / 'signals'\n"
        "    signals.mkdir(parents=True, exist_ok=True)\n"
        "    (signals / 'brouhaha.parquet').write_bytes(b'')\n"
        "    return sorted(signals.glob('*.parquet'))\n"
    )
    assert check_source(PROBE, conformant, STAGE_CONTRACTS["L1"]) == []


def test_the_pipeline_conforms_or_the_violation_is_in_the_register() -> None:
    """Rule 2, applied to the tree.

    A new violation fails here. An old one is in ``KNOWN_DEVIATIONS`` with the D-17 clause it
    breaks and what closes it — which makes the register the restructure's worklist rather than
    an exemption list.
    """
    offenders = unwaived(static_violations(REPO_ROOT))
    assert not offenders, (
        "these reads/writes are outside the declaring stage's contract and are not registered:\n"
        + "\n".join(str(f) for f in offenders)
    )


def test_no_registered_deviation_has_gone_stale() -> None:
    """Closing a violation requires deleting its entry, or the register rots into an exemption."""
    dead = dead_static_deviations(static_violations(REPO_ROOT))
    assert not dead, "these registered deviations no longer match anything — delete them:\n" + "\n".join(
        f"{d.module} {d.op} {d.pattern}" for d in dead
    )


# ── 3. artifact conformance ──────────────────────────────────────────────────


def _write_table(path: Path, columns: dict[str, list[object]]) -> None:
    """A one-row parquet with the given columns, for the artifact proofs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({name: pa.array(values) for name, values in columns.items()}), path)


def _conformant_run(root: Path) -> None:
    """The smallest tree that satisfies every stage's declaration."""
    (root / "L1").mkdir(parents=True, exist_ok=True)
    (root / "L1" / "perturbations.json").write_text("{}")
    (root / "L1" / "raw" / "asr").mkdir(parents=True, exist_ok=True)
    (root / "L1" / "raw" / "asr" / "whisper.json").write_text("{}")
    _write_table(
        root / "L1" / "signals" / "brouhaha_snr_db.parquet",
        {"perturbation": ["raw"], "signal": ["brouhaha_snr_db"], "start": [0.0], "end": [0.5]},
    )
    _write_table(
        root / "L2" / "round" / "0" / "estimates" / "speaker.parquet",
        {"start": [0.0], "end": [0.5], "uncertainty": [0.3], "contributing_passes": [["raw"]]},
    )
    (root / "L2" / "round" / "0" / "summary.json").write_text("{}")
    (root / "final").mkdir(parents=True, exist_ok=True)
    (root / "final" / "transcript.json").write_text("{}")


def test_a_conformant_tree_is_clean(tmp_path: Path) -> None:
    """The negative control for rule 3."""
    _conformant_run(tmp_path)
    assert artifact_violations(tmp_path) == []


def test_a_file_no_stage_declared_is_flagged(tmp_path: Path) -> None:
    """What static analysis cannot see: a writer reached through a helper, or a stray emission."""
    _conformant_run(tmp_path)
    stray = tmp_path / "triage.json"
    stray.write_text("{}")
    assert artifact_violations(tmp_path) == ["triage.json: written by no declared stage output"]

    stray.unlink()
    assert artifact_violations(tmp_path) == []


def test_a_per_pass_axis_table_is_flagged(tmp_path: Path) -> None:
    """An axis is an aggregator across signals **and** across perturbations.

    The hard case, and the one three previous artifact rules all passed on: the table sits at a
    path some stage genuinely declares, and only its key gives it away. Both halves of the
    category error are reported — a fold under a key that has no ``axis``, and a perturbation
    index on something that aggregates over perturbations.
    """
    _conformant_run(tmp_path)
    _write_table(
        tmp_path / "L1" / "signals" / "speaker.parquet",
        {
            "perturbation": ["raw"],
            "signal": ["speaker"],
            "start": [0.0],
            "end": [0.5],
            "axis": ["speaker"],
            "uncertainty": [0.7],
        },
    )
    problems = artifact_violations(tmp_path)
    assert any("carries ['axis']" in p for p in problems), problems
    assert any("carries fold column(s) ['uncertainty']" in p for p in problems), problems

    (tmp_path / "L1" / "signals" / "speaker.parquet").unlink()
    assert artifact_violations(tmp_path) == []


def test_an_l1_table_keyed_by_two_perturbations_is_flagged(tmp_path: Path) -> None:
    """``L1/stability/`` in the shape that survived being re-keyed by signal.

    A per-signal instability parquet has no ``axis`` column and no fold-across-signals column,
    so it *looks* like a measurement. What gives it away is its keyspace: a row carrying
    ``pass_a`` and ``pass_b`` relates two perturbations, and relating two is a fold.
    """
    _conformant_run(tmp_path)
    _write_table(
        tmp_path / "L1" / "signals" / "stability.parquet",
        {"signal": ["diar_a"], "start": [0.0], "end": [0.5], "pass_a": ["raw"], "pass_b": ["enhanced"]},
    )
    problems = artifact_violations(tmp_path)
    assert any("names more than one" in p and "perturbation" in p for p in problems), problems

    (tmp_path / "L1" / "signals" / "stability.parquet").unlink()
    assert artifact_violations(tmp_path) == []


def test_an_l1_table_keyed_by_no_perturbation_is_flagged(tmp_path: Path) -> None:
    """The other half of the same rule, and the one a column blacklist cannot express.

    ``L1/signals/`` accumulates across raw and every perturbation, so a row that does not say
    which one it came from has folded them — the file is a run-level statistic wearing a
    measurement's shape.
    """
    _conformant_run(tmp_path)
    _write_table(
        tmp_path / "L1" / "signals" / "instability.parquet",
        {"signal": ["diar_a"], "start": [0.0], "end": [0.5], "abs_delta": [0.4]},
    )
    problems = artifact_violations(tmp_path)
    assert any("a row cannot say which one it came from" in p and "perturbation" in p for p in problems), problems

    (tmp_path / "L1" / "signals" / "instability.parquet").unlink()
    assert artifact_violations(tmp_path) == []


def test_cross_perturbation_evaluation_stored_under_l1_is_flagged(tmp_path: Path) -> None:
    """``L1/stability/`` is not a declared L1 output at all — the path alone settles it."""
    _conformant_run(tmp_path)
    _write_table(
        tmp_path / "L1" / "stability" / "diar_a.parquet",
        {"signal": ["diar_a"], "start": [0.0], "end": [0.5], "pass_a": ["raw"], "pass_b": ["enhanced"]},
    )
    (tmp_path / "L1" / "stability" / "signals.json").write_text("{}")
    problems = artifact_violations(tmp_path)
    assert "L1/stability/diar_a.parquet: written by no declared stage output" in problems
    assert "L1/stability/signals.json: written by no declared stage output" in problems

    for path in sorted((tmp_path / "L1" / "stability").iterdir()):
        path.unlink()
    assert artifact_violations(tmp_path) == []


def test_a_fold_indexed_by_a_perturbation_is_flagged(tmp_path: Path) -> None:
    """An axis is a fold *over* perturbations, so it cannot be indexed by one."""
    _conformant_run(tmp_path)
    _write_table(
        tmp_path / "L2" / "round" / "0" / "estimates" / "asr.parquet",
        {"start": [0.0], "end": [0.5], "uncertainty": [0.4], "stream": ["raw"]},
    )
    problems = artifact_violations(tmp_path)
    assert any("is not indexed by perturbation" in p and "['stream']" in p for p in problems), problems

    _write_table(
        tmp_path / "L2" / "round" / "0" / "estimates" / "asr.parquet",
        {"start": [0.0], "end": [0.5], "uncertainty": [0.4], "contributing_passes": [["raw"]]},
    )
    assert artifact_violations(tmp_path) == [], "contributing_passes is provenance about an input, not an index"


def _current_run_tree(root: Path) -> None:
    """The tree a completed run leaves today, as recorded from ``clip18s_20260802-155406``.

    Reproduced rather than assumed, so the artifact half of the register is exercised on every
    machine — including the ones with no ``artifacts/analyze_audio/`` to walk. It is not a
    substitute for :func:`test_a_real_run_conforms_or_the_violation_is_in_the_register`, which
    reads what the pipeline actually wrote; it is what keeps the register honest in between.
    """
    (root / "triage.json").write_text("{}")
    (root / "L1").mkdir(parents=True, exist_ok=True)
    (root / "L1" / "perturbations.json").write_text("{}")
    (root / "L1" / "signals.png").write_bytes(b"")
    (root / "L1" / "timeline.png").write_bytes(b"")
    # One file per signal, accumulating across every perturbation — L2's only input from L1.
    _write_table(
        root / "L1" / "signals" / "brouhaha_snr_db.parquet",
        {
            "perturbation": ["raw", "enhanced"],
            "start": [0.0, 0.0],
            "end": [0.5, 0.5],
            "signal": ["brouhaha_snr_db"] * 2,
        },
    )
    for label, pert_dir in (("raw", root / "L1" / "raw"), ("enhanced", root / "L1" / "perturbation" / "enhanced")):
        for name in ("ast.json", "yamnet.json", "features.json", "ppgs.json", "scene_agreement.json", "pii.json"):
            pert_dir.mkdir(parents=True, exist_ok=True)
            (pert_dir / name).write_text("{}")
        for task, model in (("asr", "whisper"), ("alignment", "whisper"), ("diarization", "pyannote")):
            (pert_dir / task).mkdir(parents=True, exist_ok=True)
            (pert_dir / task / f"{model}.json").write_text("{}")
        (pert_dir / "embeddings").mkdir(parents=True, exist_ok=True)
        (pert_dir / "embeddings" / "ecapa.json").write_text("{}")
        _write_table(pert_dir / "features" / "opensmile.parquet", {"start": [0.0], "end": [0.5]})
        _write_table(pert_dir / "noise_floor.parquet", {"band_hz": [125.0]})
        _write_table(pert_dir / "background_sources.parquet", {"band_hz": [125.0]})
    belief = root / "L2"
    belief.mkdir(parents=True, exist_ok=True)
    _write_table(belief / "background_mask.parquet", {"start": [0.0], "end": [0.5], "uncertainty": [0.2]})
    for name in (
        "background_mask.json",
        "disagreements.json",
        "labelstudio_tasks.json",
        "labelstudio_config.xml",
        "speakers.json",
        "rounds.json",
        "convergence.json",
        "iterations.json",
    ):
        (belief / name).write_text("{}")
    _write_table(belief / "per_speaker_presence.parquet", {"start": [0.0], "end": [0.5], "speaker": ["S0"]})
    _write_table(belief / "speech_presence.parquet", {"start": [0.0], "end": [0.5], "round": [1]})
    # One round tree, 0-based, shared by both producers. Round 0 is fusion's baseline, which the
    # adaptive loop adopts rather than renumbering as its own "round 1".
    for index in (0, 1, 2):
        for axis in ("speech_presence", "speaker", "asr", "background_mask"):
            _write_table(
                belief / "round" / str(index) / "estimates" / f"{axis}.parquet",
                {"start": [0.0], "end": [0.5], "uncertainty": [0.3], "round": [index]},
            )
        (belief / "round" / str(index) / "timeline.png").write_bytes(b"")
        (belief / "round" / str(index) / "summary.json").write_text("{}")
    derivatives = belief / "round" / "0" / "derivatives"
    for axis in ("speech_presence", "speaker", "asr"):
        _write_table(
            derivatives / "votes" / f"{axis}.parquet",
            {"start": [0.0], "end": [0.5], "source": ["diar_a"], "stream": ["raw"]},
        )
    # Cross-perturbation stability: a round derivative, keyed by signal, carrying two
    # perturbations per row — which is what a fold looks like and what no L1 artifact may be.
    _write_table(
        derivatives / "stability" / "diar_a.parquet",
        {"start": [0.0], "end": [0.5], "signal": ["diar_a"], "pass_a": ["raw"], "pass_b": ["enhanced"]},
    )
    for index in (1, 2):
        regions = belief / "round" / str(index) / "derivatives" / "regions.json"
        regions.parent.mkdir(parents=True, exist_ok=True)
        regions.write_text("[]")
    final = root / "final"
    final.mkdir(parents=True, exist_ok=True)
    for name in (
        "summary.json",
        "run_summary.json",
        "summary.md",
        "transcript.json",
        "diarization.json",
        "diarization.rttm",
        "disagreements_resolved.json",
        "labelstudio_tasks.json",
        "labelstudio_config.xml",
    ):
        (final / name).write_text("{}")
    (final / "timeline.png").write_bytes(b"")


def test_the_current_run_tree_is_flagged_and_fully_accounted_for(tmp_path: Path) -> None:
    """The proof that rule 3 fires on the tree as it stands, and that the register covers it.

    Both halves matter. Unregistered, the current layout produces a long list of undeclared
    artifacts — the restructure's worklist. Registered, the list is empty, so the next artifact
    nobody meant to emit is visible against a quiet background instead of buried in it.
    """
    _current_run_tree(tmp_path)
    raw = artifact_violations(tmp_path)
    # What is left is one class: per-round quantities flattened to the ``L2/`` root with no round
    # to belong to, plus the triage decision at the run root. Steps 1 and 2 took this list from
    # 34 to 12 by *declaring* the perturbation tree and the round tree, not by waiving them.
    assert len(raw) >= 10, "the L2 root is not D-17's, and the guard must say so"
    assert all(p.startswith(("L2/", "triage.json")) for p in raw), raw
    assert any(p.startswith("triage.json") for p in raw)
    assert any(p.startswith("L2/background_mask") for p in raw)

    # What step 1 closed: the perturbation tree, its register, and the accumulated signal files
    # are declared, so they no longer appear. ``L1/stability/`` is gone from the tree entirely —
    # a cross-perturbation fold is a round derivative, and its run-level mean is on every fused
    # row's ``weight_basis`` rather than in a second file.
    assert not any(p.startswith("L1/stability/") for p in raw)
    assert not any(p.startswith("L1/perturbations.json") for p in raw)
    assert not any(p.startswith("L1/signals/") for p in raw)
    assert not any(p.startswith("L1/raw/asr") for p in raw)

    # What step 2 closed: one round tree, so a round's timeline and summary are declared and the
    # second tree (``L2/rounds/<N>/``, 1-based) does not exist to be flagged.
    assert not any(p.startswith("L2/round/1/timeline.png") for p in raw)
    assert not any(p.startswith("L2/round/2/summary.json") for p in raw)
    assert not any(p.startswith("L2/round/0/derivatives/") for p in raw)
    assert not any(p.startswith("L2/rounds/") for p in raw)

    assert unwaived_artifacts(raw) == []


@pytest.fixture(scope="session")
def real_run_dir() -> Path:
    """The newest completed run under ``artifacts/analyze_audio/``.

    Skips rather than passing when there is none: an artifact guard with nothing to walk reports
    exactly what an artifact guard that found no violation reports, and "did not run" has to stay
    distinguishable from "found nothing".
    """
    candidates = (
        [p for p in RUNS_DIR.iterdir() if p.is_dir() and (p / "L1").is_dir() and (p / "L2").is_dir()]
        if RUNS_DIR.is_dir()
        else []
    )
    if not candidates:
        pytest.skip(
            f"no completed run under {RUNS_DIR} (need one with L1/ and L2/). "
            "Produce one with: uv run python scripts/analyze_audio.py <audio>"
        )
    return max(candidates, key=lambda p: p.stat().st_mtime)


def test_a_real_run_conforms_or_the_violation_is_in_the_register(real_run_dir: Path) -> None:
    """Rule 3, applied to what the pipeline actually wrote.

    Read against a real run rather than a tree this file builds: a guard that walks its own
    fixture only proves the fixture is consistent with itself, and every defect this feature
    shipped was in what the pipeline actually produced.
    """
    offenders = unwaived_artifacts(artifact_violations(real_run_dir))
    assert not offenders, f"{real_run_dir.name}: artifacts no stage declared, and not registered:\n" + "\n".join(
        offenders
    )
