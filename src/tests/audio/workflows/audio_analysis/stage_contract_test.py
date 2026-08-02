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
    TABULAR_SUFFIXES,
    Artifact,
    StageContract,
    artifact_violations,
    check_source,
    dag_edges,
    dead_artifact_deviations,
    dead_static_deviations,
    declared_artifacts,
    folding_stages,
    matches,
    overlap,
    pipeline_sources,
    static_violations,
    structural_vocabulary,
    topological_order,
    unproduced_declarations,
    unrolled_contracts,
    unwaived,
    unwaived_artifacts,
    unwaived_unproduced,
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


# ── 0. the declaration cannot be written broad ───────────────────────────────
#
# The first defeat is not a bug in a check, it is a check that never applied. A ``**`` carrying
# ``key=None`` reports nothing beneath it, so the guard's output on a tree full of violations is
# byte-identical to its output on a clean one. These rules make that declaration impossible to
# write rather than merely inadvisable, and they run at construction: a rule enforced only here
# would still let any caller that imports the module without running this file build the hole.


def test_a_broad_pattern_with_no_content_rule_is_refused_at_construction() -> None:
    """The proof that rule 0 can fail, on the declaration that was actually in the tree.

    ``L2/round/{n}/derivatives/**`` carried ``key=None`` for three steps of the restructure. Under
    it, ``derivatives/estimates/speaker.parquet`` — a per-perturbation axis table, the one shape
    D-16 says cannot exist — produced no finding, because with no key there is no rule to break.
    """
    with pytest.raises(ValueError, match="applies no content rule"):
        Artifact("L2/round/{n}/derivatives/**", "anything at all", suffixes=(".parquet",))

    # ...and the same pattern with a key and a pinned file kind is a declaration, not a hole.
    bounded = Artifact(
        "L2/round/{n}/derivatives/**",
        "anything at all",
        key=("axis", "bucket"),
        suffixes=(".parquet",),
    )
    assert bounded.permitted_suffixes() == frozenset({".parquet"})


def test_a_broad_pattern_must_pin_the_file_kinds_it_admits() -> None:
    """``**`` names no extension, so an unpinned one admits every format the repo can write."""
    with pytest.raises(ValueError, match="must declare suffixes"):
        Artifact("L1/raw/**", "the identity's model outputs", key=("perturbation", "bucket"))

    pinned = Artifact("L1/raw/**", "the identity's model outputs", key=("perturbation", "bucket"), suffixes=(".json",))
    assert pinned.permitted_suffixes() == frozenset({".json"})


def test_a_key_naming_every_dimension_prohibits_nothing() -> None:
    """Breadth of location is paid for with narrowness of content, or it is not paid for."""
    with pytest.raises(ValueError, match="prohibits none"):
        Artifact(
            "L1/raw/**",
            "the identity's model outputs",
            key=("perturbation", "axis", "signal", "bucket", "speaker", "round"),
            suffixes=(".json",),
        )


def test_every_declared_artifact_pins_the_file_kinds_it_admits() -> None:
    """Applied to the declaration itself: a pattern that names no extension and declares none."""
    unpinned = [artifact.pattern for _stage, artifact in declared_artifacts() if not artifact.permitted_suffixes()]
    assert not unpinned, f"these declarations admit any file kind: {unpinned}"


def test_only_a_stage_that_decides_may_declare_a_fold() -> None:
    """Relating two values of an input dimension is a decision, so a measuring stage may not.

    ``folded`` is what lets ``derivatives/stability/`` carry ``pass_a`` beside ``pass_b``. The
    same licence at L1 would be a stage measuring and folding in one breath, which is the
    boundary the whole layering rests on.
    """
    assert "L1" not in folding_stages(), folding_stages()


def test_the_structural_vocabulary_is_derived_from_the_declaration_and_not_listed() -> None:
    """Reserved names nobody has to remember to extend: they *are* the declaration's own words."""
    vocabulary = structural_vocabulary()
    assert vocabulary["final"] == frozenset({0}), "final/ is a root, and only a root"
    assert vocabulary["estimates"] == frozenset({3}), "estimates/ sits under L2/round/<n>/"
    assert "whisper.json" not in vocabulary, "a filename is not structure"
    assert "asr" not in vocabulary, "a tool's own directory name is not the declaration's"


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


def test_a_path_bound_by_tuple_unpacking_is_visible() -> None:
    """The proof that rule 2 can fail, on a line that is in the tree.

    ``adaptive/plot.py:347`` binds both per-speaker deliverables in one statement and then uses
    them on four lines. The resolver recorded ``ast.Name`` targets only, so the binding produced
    nothing and every use afterwards was silent — four reads of the L2 root from the stage that
    is supposed to only extract, under a guard reporting that the rule held.
    """
    live = (
        "def f(out_dir):\n"
        "    belief = belief_dir(out_dir)\n"
        "    speakers_path, presence_path = belief / 'speakers.json', belief / 'per_speaker_presence.parquet'\n"
        "    if not speakers_path.exists() or not presence_path.exists():\n"
        "        return None\n"
        "    return _json.loads(speakers_path.read_text()), pd.read_parquet(presence_path)\n"
    )
    findings = check_source(PROBE, live, STAGE_CONTRACTS["FINAL"])
    assert sorted({f.pattern for f in findings}) == ["L2/per_speaker_presence.parquet", "L2/speakers.json"]
    assert {f.op for f in findings} == {"read"}

    # Bind the same two paths and use neither: the binding itself is not the finding.
    unused = (
        "def f(out_dir):\n"
        "    belief = belief_dir(out_dir)\n"
        "    speakers_path, presence_path = belief / 'speakers.json', belief / 'per_speaker_presence.parquet'\n"
        "    return speakers_path, presence_path\n"
    )
    assert check_source(PROBE, unused, STAGE_CONTRACTS["FINAL"]) == []


@pytest.mark.parametrize(
    ("form", "source"),
    [
        (
            "tuple target",
            "def f(d):\n    b = belief_dir(d)\n    x, y = b / 'speakers.json', b / 'rounds.json'\n"
            "    x.write_text('{}')\n    y.write_text('{}')\n",
        ),
        (
            "list target",
            "def f(d):\n    b = belief_dir(d)\n    [x, y] = [b / 'speakers.json', b / 'rounds.json']\n"
            "    x.write_text('{}')\n    y.write_text('{}')\n",
        ),
        (
            "starred target",
            "def f(d):\n    b = belief_dir(d)\n    head, *rest = [b / 'speakers.json', b / 'rounds.json']\n"
            "    head.write_text('{}')\n    for other in rest:\n        other.write_text('{}')\n",
        ),
        (
            "augmented assignment",
            "def f(d):\n    p = belief_dir(d)\n    p /= 'speakers.json'\n    p.write_text('{}')\n"
            "    q = belief_dir(d)\n    q /= 'rounds.json'\n    q.write_text('{}')\n",
        ),
        (
            "walrus",
            "def f(d):\n    b = belief_dir(d)\n    if (p := b / 'speakers.json').exists():\n"
            "        p.write_text('{}')\n    (b / 'rounds.json').write_text('{}')\n",
        ),
        (
            "for target",
            "def f(d):\n    b = belief_dir(d)\n    for p in (b / 'speakers.json', b / 'rounds.json'):\n"
            "        p.write_text('{}')\n",
        ),
        (
            "comprehension target",
            "def f(d):\n    b = belief_dir(d)\n"
            "    return [p.read_text() for p in [b / 'speakers.json', b / 'rounds.json']]\n",
        ),
    ],
)
def test_every_binding_form_that_names_a_path_is_visible(form: str, source: str) -> None:
    """Assignment is one of seven binding forms, and six of them used to bind invisibly.

    Not exotic syntax: a tuple assignment, a ``for``, an in-place ``/=``, a walrus and a
    comprehension are ordinary lines. What made them a defeat is that the guard's silence on them
    is the same silence it reports for a conformant module.
    """
    findings = check_source(PROBE, source, STAGE_CONTRACTS["L1"])
    assert sorted({f.pattern for f in findings}) == ["L2/rounds.json", "L2/speakers.json"], form


def test_an_in_place_extension_reports_the_artifact_and_not_its_directory() -> None:
    """``p /= "speakers.json"`` extends a path; read as a rebinding it names the wrong thing.

    A finding that names ``L2`` where the write was to ``L2/speakers.json`` is not a smaller
    version of the truth — it is what a register entry then waives, and the entry would waive
    every write to that directory.
    """
    source = "def f(d):\n    p = belief_dir(d)\n    p /= 'speakers.json'\n    p.write_text('{}')\n"
    assert [(f.op, f.pattern) for f in check_source(PROBE, source, STAGE_CONTRACTS["L1"])] == [
        ("write", "L2/speakers.json")
    ]


def test_a_for_over_a_glob_binds_the_paths_the_glob_yields() -> None:
    """A ``for`` target is only visible if the iterable is, and a glob is how runs are walked.

    Two findings for one loop, and both are the truth: the ``glob`` is itself a read of that
    directory, and the loop body reads each file it yielded. What matters is that the second one
    exists at all — the loop variable used to resolve to nothing, so a body of any size was free.
    """
    source = (
        "def f(d):\n"
        "    for path in (belief_dir(d) / 'round').glob('*/estimates/*.parquet'):\n"
        "        path.read_text()\n"
    )
    findings = check_source(PROBE, source, STAGE_CONTRACTS["L1"])
    assert {(f.op, f.pattern) for f in findings} == {("read", "L2/round/*/estimates/*.parquet")}
    assert sorted(f.via for f in findings) == ["glob", "read_text"]


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


@pytest.mark.parametrize(
    ("relative", "swallowed"),
    [
        ("L1/raw/final/transcript.json", "final"),
        ("L1/raw/estimates/asr.parquet", "estimates"),
        ("L1/signals/round/asr.parquet", "round"),
    ],
)
def test_a_broad_pattern_does_not_admit_the_trees_own_vocabulary_out_of_place(
    tmp_path: Path, relative: str, swallowed: str
) -> None:
    """The second half of rule 0, and the half no per-artifact rule could express.

    ``L1/raw/**`` has to stay open — whichever directories a tool reports by, it reports by, and
    nobody can enumerate them in advance. What it must not do is admit another stage's *shape*:
    a ``final/`` or an ``estimates/`` under a perturbation directory is L2's layout smuggled into
    L1's open tree, and both used to conform. The vocabulary is derived from every declaration at
    once, so it grows with the tree rather than being a list somebody maintains.
    """
    _conformant_run(tmp_path)
    path = tmp_path / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".json":
        path.write_text("{}")
    else:
        _write_table(path, {"perturbation": ["raw"], "signal": ["x"], "start": [0.0], "end": [0.5]})
    problems = artifact_violations(tmp_path)
    assert any(f"'{swallowed}' is the tree's own name" in p for p in problems), problems

    path.unlink()
    assert artifact_violations(tmp_path) == []


def test_a_tool_directory_the_declaration_never_mentions_is_admitted(tmp_path: Path) -> None:
    """The negative control for the vocabulary rule: openness is the point of a ``**``."""
    _conformant_run(tmp_path)
    (tmp_path / "L1" / "raw" / "diarization").mkdir(parents=True, exist_ok=True)
    (tmp_path / "L1" / "raw" / "diarization" / "pyannote.json").write_text("{}")
    assert artifact_violations(tmp_path) == []


@pytest.mark.parametrize("suffix", [".csv", ".feather"])
def test_the_same_rows_in_another_tabular_format_do_not_escape_the_content_rules(tmp_path: Path, suffix: str) -> None:
    """The proof that rule 3's content half can fail, and how cheaply it used to.

    ``_table_columns`` returned "not a table" for every suffix but ``.parquet``, so the key rules
    were opt-in: the identical per-perturbation axis table written as ``speaker.csv`` conformed.
    Two closures apply here at once — the declaration pins the kinds ``L1/signals/**`` admits, and
    the reader now covers every format the repo can write, so neither alone has to be perfect.
    """
    _conformant_run(tmp_path)
    path = tmp_path / "L1" / "signals" / f"speaker{suffix}"
    if suffix == ".csv":
        path.write_text("axis,stream,uncertainty\nspeaker,raw,0.7\n")
    else:
        import pyarrow.feather as feather

        feather.write_feather(pa.table({"axis": ["speaker"], "stream": ["raw"], "uncertainty": [0.7]}), path)
    problems = artifact_violations(tmp_path)
    assert any("is not a permitted file kind here" in p for p in problems), problems

    path.unlink()
    assert artifact_violations(tmp_path) == []


def test_a_json_list_of_records_is_read_as_a_table(tmp_path: Path) -> None:
    """JSON records are a table the repo writes, so the key rules have to reach them.

    ``L1/raw/`` permits JSON by declaration — that is where each tool's own outcome lands — and
    the escape was to put the axis fold in one. An object stays a document; a list of objects is
    rows, and rows have a key.
    """
    _conformant_run(tmp_path)
    path = tmp_path / "L1" / "raw" / "scene.json"
    path.write_text('[{"axis": "speaker", "signal": "diar_a", "uncertainty": 0.7}]')
    problems = artifact_violations(tmp_path)
    assert any("carries ['axis']" in p for p in problems), problems
    assert any("carries fold column(s) ['uncertainty']" in p for p in problems), problems

    path.write_text('{"axis": "speaker", "uncertainty": 0.7}')
    assert artifact_violations(tmp_path) == [], "a JSON object is a document, not a table"


def test_an_empty_record_list_has_no_row_to_contradict_a_key(tmp_path: Path) -> None:
    """``regions.json`` is empty on a round that proposed nothing, and empty is not wrong."""
    _conformant_run(tmp_path)
    (tmp_path / "L2" / "round" / "0" / "derivatives").mkdir(parents=True, exist_ok=True)
    (tmp_path / "L2" / "round" / "0" / "derivatives" / "regions.json").write_text("[]")
    assert artifact_violations(tmp_path) == []


def test_a_file_the_guard_cannot_read_is_a_finding_and_not_a_pass(tmp_path: Path) -> None:
    """Unreadable and conformant were the same outcome, and one of them is a lie."""
    _conformant_run(tmp_path)
    broken = tmp_path / "L1" / "signals" / "brouhaha_snr_db.parquet"
    intact = broken.read_bytes()
    broken.write_bytes(b"not a parquet file")
    problems = artifact_violations(tmp_path)
    assert any("could not be read" in p and "nothing about it has been checked" in p for p in problems), problems

    broken.write_bytes(intact)
    assert artifact_violations(tmp_path) == []


@pytest.mark.parametrize("suffix", sorted(TABULAR_SUFFIXES))
def test_every_tabular_format_the_repo_can_write_is_one_the_guard_can_read(tmp_path: Path, suffix: str) -> None:
    """The set is closed by declaration, so it has to be closed by the reader too."""
    from senselab.audio.workflows.audio_analysis.contracts import _table_columns

    path = tmp_path / f"table{suffix}"
    if suffix == ".parquet":
        _write_table(path, {"axis": ["speaker"], "start": [0.0]})
    elif suffix in {".feather", ".arrow"}:
        import pyarrow.feather as feather

        feather.write_feather(pa.table({"axis": ["speaker"], "start": [0.0]}), path)
    elif suffix in {".csv", ".tsv"}:
        path.write_text("axis\tstart\nspeaker\t0.0\n" if suffix == ".tsv" else "axis,start\nspeaker,0.0\n")
    elif suffix in {".jsonl", ".ndjson"}:
        path.write_text('{"axis": "speaker", "start": 0.0}\n')
    else:
        path.write_text('[{"axis": "speaker", "start": 0.0}]')
    assert _table_columns(path) == frozenset({"axis", "start"}), suffix


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
    """The tree a completed run leaves today, as recorded from ``verify_clip_18s_20260802-182800``.

    Reproduced rather than assumed, so the artifact half of the register is exercised on every
    machine — including the ones with no ``artifacts/analyze_audio/`` to walk. It is not a
    substitute for :func:`test_a_real_run_conforms_or_the_violation_is_in_the_register`, which
    reads what the pipeline actually wrote; it is what keeps the register honest in between.

    Recorded, which is the whole of its value: the previous version wrote every round's
    ``summary.json`` and ``timeline.png`` and all four axes in every round, which is what the
    declaration says a round owes and *not* what the pipeline produces. A fixture that records the
    intent rather than the output is a fixture that cannot fail, and under it the four defects
    this tree carries — the fourth axis stopping mid-loop, two schemas under one artifact name, no
    round producing the full set, and ``final/`` computing rather than extracting — were all
    invisible.
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
    # One round tree, 0-based, shared by both producers — and that is where the sharing stops.
    # Rounds 0-2 come from ``fuse.write_final_uncertainty`` and rounds 3-4 from the adaptive
    # loop's belief store, so the run has two producers writing one declared artifact:
    #
    #   - the fuse rounds carry ``axis``/``signal_weights``/``weight_basis``/``coupled_from``;
    #   - the loop rounds carry ``status``/``p_voice``/``aleatoric_floor``/``attenuation`` and no
    #     ``axis`` at all, so a reader cannot tell from a round which producer wrote it;
    #   - the fourth axis has estimates in 0-2 and none in 3-4;
    #   - only the loop writes ``summary.json`` (for its baseline and its own rounds) and only
    #     fuse writes ``timeline.png``.
    for index in (0, 1, 2):
        for axis in ("speech_presence", "speaker", "asr", "background_mask"):
            _write_table(
                belief / "round" / str(index) / "estimates" / f"{axis}.parquet",
                {
                    "start": [0.0],
                    "end": [0.5],
                    "axis": [axis],
                    "uncertainty": [0.3],
                    "round": [index],
                    "signal_weights": ["{}"],
                    "weight_basis": ["{}"],
                },
            )
        (belief / "round" / str(index) / "timeline.png").write_bytes(b"")
    for index in (3, 4):
        for axis in ("speech_presence", "speaker", "asr"):
            _write_table(
                belief / "round" / str(index) / "estimates" / f"{axis}.parquet",
                {
                    "start": [0.0],
                    "end": [0.5],
                    "uncertainty": [0.3],
                    "round": [index],
                    "status": ["open"],
                    "p_voice": [0.9],
                    "aleatoric_floor": [0.1],
                },
            )
    # The loop's baseline is the last round fusion wrote, so round 2 gets a summary and the two
    # rounds before it get none.
    for index in (2, 3, 4):
        (belief / "round" / str(index) / "summary.json").write_text("{}")
    derivatives = belief / "round" / "0" / "derivatives"
    for axis in ("speech_presence", "speaker", "asr", "background_mask"):
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
    for index in (3, 4):
        regions = belief / "round" / str(index) / "derivatives" / "regions.json"
        regions.parent.mkdir(parents=True, exist_ok=True)
        regions.write_text("[]")
        # The votes a round added, in the store's own spelling: an interval written as
        # ``bucket_start``/``bucket_end`` is the same dimension as ``start``/``end``, which is a
        # synonym the guard had to be taught rather than a second key.
        _write_table(
            regions.parent / "votes_added.parquet",
            {
                "axis": ["speaker"],
                "bucket_start": [0.0],
                "bucket_end": [0.5],
                "source": ["diar_a"],
                "stream": ["raw"],
                "round": [index],
                "evidence_weight": [1.0],
            },
        )
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


# ── 4. a declaration nothing satisfies ───────────────────────────────────────
#
# The mirror of rule 3, and the direction nothing used to look. Every content rule passes on a
# file that is not there, so a declaration nobody produces is invisible by construction — and it
# is what let a 26-file fragment be judged a completed run.


def test_a_complete_run_produces_every_declared_artifact_or_the_gap_is_registered(tmp_path: Path) -> None:
    """Rule 4, against the recorded complete tree.

    Against a real run this question cannot be asked — a partial run makes every declaration
    unproduced and the answer says nothing. Against the tree a completed run actually leaves it
    is exact, and what it reports is the restructure's other half: seven of ``final/``'s declared
    deliverables are written to the L2 root or not at all, so ``final/`` carries no converged
    axis, no per-speaker output and no account of how the run reached its answer.
    """
    _current_run_tree(tmp_path)
    missing = unwaived_unproduced(unproduced_declarations(tmp_path))
    assert not missing, "these declared outputs nothing produces, and no register entry says why:\n" + "\n".join(
        missing
    )


def test_a_declaration_nothing_produces_is_a_finding(tmp_path: Path) -> None:
    """The proof that rule 4 can fail, written against a declaration this test owns.

    Constructed rather than taken from the real declaration so the failure is reproducible
    without a registered gap standing in for it.
    """
    _current_run_tree(tmp_path)
    invented = Artifact("final/spectrogram.png", "a deliverable nobody writes")
    assert unproduced_declarations(tmp_path, declared=[("FINAL", invented)]) == [
        "final/spectrogram.png: declared by FINAL (a deliverable nobody writes) "
        "and the run produced nothing matching it"
    ]

    (tmp_path / "final" / "spectrogram.png").write_bytes(b"")
    assert unproduced_declarations(tmp_path, declared=[("FINAL", invented)]) == []


def test_no_registered_artifact_deviation_has_gone_stale(tmp_path: Path) -> None:
    """The artifact register decays exactly as the static one does, and now says so.

    It was exempted on the grounds that run trees legitimately differ. They do — which is why
    this is asked of the recorded tree rather than of whatever run is on the machine. Under the
    exemption, ``L1/perturbation/*/**`` outlived its defect by two steps of the restructure.
    """
    _current_run_tree(tmp_path)
    problems = [*artifact_violations(tmp_path), *unproduced_declarations(tmp_path)]
    dead = dead_artifact_deviations(problems)
    assert not dead, "these registered deviations no longer match anything — delete them:\n" + "\n".join(
        f"{d.op} {d.pattern}" for d in dead
    )


def _incomplete(run: Path) -> list[str]:
    """Declared outputs this run produced nothing for, ignoring the registered gaps."""
    return unwaived_unproduced(unproduced_declarations(run))


@pytest.fixture(scope="session")
def real_run_dir() -> Path:
    """The newest **complete** run under ``artifacts/analyze_audio/``.

    Completeness is judged against the declaration, by the same computation that reports a
    declaration nothing satisfies. The previous criterion was "the directory has an ``L1/`` and
    an ``L2/``", which a 26-file partial run satisfies with no ``L1/signals/``, no ``L2/round/``
    and no ``L1/perturbations.json`` — and against that fragment every rule below passes, because
    a rule about a file cannot fail on a file that is not there.

    Skips rather than passing when there is none, and names what was missing: "did not run" has
    to stay distinguishable from "found nothing", and "ran against a fragment" from both.
    """
    candidates = [p for p in RUNS_DIR.iterdir() if p.is_dir()] if RUNS_DIR.is_dir() else []
    if not candidates:
        pytest.skip(f"no run under {RUNS_DIR}. Produce one with: uv run python scripts/analyze_audio.py <audio>")
    complete = [run for run in candidates if not _incomplete(run)]
    if not complete:
        newest = max(candidates, key=lambda p: p.stat().st_mtime)
        pytest.skip(
            f"the newest run {newest.name} is incomplete — it produced nothing for:\n" + "\n".join(_incomplete(newest))
        )
    return max(complete, key=lambda p: p.stat().st_mtime)


def test_the_completeness_criterion_rejects_a_fragment(tmp_path: Path) -> None:
    """The proof that rule 4's fixture can fail, on the fragment that was accepted.

    An ``L1/`` and an ``L2/`` and nothing else inside them: the shape the previous criterion took
    for a completed run, and the shape against which the whole artifact half of this file
    reported clean.
    """
    (tmp_path / "L1" / "raw").mkdir(parents=True, exist_ok=True)
    (tmp_path / "L1" / "raw" / "ast.json").write_text("{}")
    (tmp_path / "L2").mkdir(parents=True, exist_ok=True)
    (tmp_path / "L2" / "rounds.json").write_text("{}")
    assert artifact_violations(tmp_path) == ["L2/rounds.json: written by no declared stage output"]

    missing = _incomplete(tmp_path)
    assert any(p.startswith("L1/signals/**") for p in missing), missing
    assert any(p.startswith("L1/perturbations.json") for p in missing), missing
    assert any(p.startswith("L2/round/*/estimates/*.parquet") for p in missing), missing

    # ...and the recorded complete tree is not a fragment, or the criterion says nothing.
    complete = tmp_path / "complete"
    complete.mkdir()
    _current_run_tree(complete)
    assert _incomplete(complete) == []


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
