"""Capability-passing I/O: a stage can only name what it is allowed to name (D-18).

Four generations of guard were defeated, each by a mechanism its author had not enumerated — a name
list that omitted the fourth axis, a regex an alias slipped past, a glob blind to ``adaptive/``, a
``**`` with ``key=None`` permitting anything. The pattern is not carelessness: **inspecting an
undecidable property after the fact cannot terminate.** ``_PathResolver`` tries to evaluate path
*expressions* to a declared pattern and cannot see a path handed to a helper as a parameter.

The reframing is that there is nothing to resolve. Paths are **derived from keys**, so a stage that
holds a ``StageIO`` cannot construct a path at all — it can only present a key and be told yes or no.
The capability is over *key kinds and rounds*, which is a finite predicate, not a string-matching
problem.

Two properties follow and are tested here: a stage cannot write outside its own directory (there is
no method that would let it), and the read/write predicates make the DAG acyclic by construction
rather than by a graph built from pattern overlap.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from senselab.audio.workflows.audio_analysis.keys import (
    DerivativeKey,
    EstimateKey,
    Operator,
    Route,
    SignalKey,
)
from senselab.audio.workflows.audio_analysis.stage_io import (
    STAGE_ORDER,
    Artifact,
    ReportKey,
    Stage,
    StageIO,
    UnauthorizedArtifact,
)

RUN = Path("/runs/x")
SNR = SignalKey(target="snr", producer="pyannote/brouhaha", route=Route())


def _derivative(round_: int) -> DerivativeKey:
    return DerivativeKey(target="snr", operator=Operator("resample", "mean"), sources=(SNR,), round=round_)


# ── the write capability is a single directory, and it is not a check ──


def test_l1_may_write_a_signal() -> None:
    """The one stage that produces measurements."""
    io = StageIO.for_stage(Stage.L1, run_dir=RUN)
    assert io.path_for(SNR, ".parquet") == RUN / "L1/signals/snr/pyannote__brouhaha/direct/unmodified.parquet"


def test_l1_may_not_write_a_derivative() -> None:
    """A fold or projection at L1 is the violation the whole layering exists to prevent."""
    io = StageIO.for_stage(Stage.L1, run_dir=RUN)
    with pytest.raises(UnauthorizedArtifact, match="L1"):
        io.path_for(_derivative(0), ".parquet")


def test_derive_may_not_write_a_signal() -> None:
    """L1 measures; L2 decides. A derivative stage inventing a measurement inverts that."""
    io = StageIO.for_stage(Stage.DERIVE, round=0, run_dir=RUN)
    with pytest.raises(UnauthorizedArtifact, match="signal"):
        io.path_for(SNR, ".parquet")


def test_derive_may_not_write_another_round_s_derivative() -> None:
    """Round 1 writing round 0's directory is how two round trees with different bases arose."""
    io = StageIO.for_stage(Stage.DERIVE, round=1, run_dir=RUN)
    assert io.path_for(_derivative(1), ".parquet")
    with pytest.raises(UnauthorizedArtifact, match="round"):
        io.path_for(_derivative(0), ".parquet")


def test_derive_may_not_write_an_estimate() -> None:
    """Derivatives are values; estimates are doubt about a family of values. Different stages."""
    io = StageIO.for_stage(Stage.DERIVE, round=0, run_dir=RUN)
    with pytest.raises(UnauthorizedArtifact, match="estimate"):
        io.path_for(EstimateKey(axis="speaker", round=0), ".parquet")


def test_estimate_writes_only_its_own_round_s_axes() -> None:
    """Its own round's, not a neighbour's — the same containment as derive."""
    io = StageIO.for_stage(Stage.ESTIMATE, round=2, run_dir=RUN)
    assert io.path_for(EstimateKey(axis="speaker", round=2)) == RUN / "L2/round/2/estimates/speaker.parquet"
    with pytest.raises(UnauthorizedArtifact, match="round"):
        io.path_for(EstimateKey(axis="speaker", round=1))


def test_there_is_no_method_that_takes_a_path() -> None:
    """The guarantee is an absence of capability, not a check that can be bypassed.

    Every previous guard could be defeated by handing a path to a helper. Here a stage never holds a
    path it did not receive back from ``path_for``, so there is no expression to resolve.
    """
    assert not any(
        name for name in dir(StageIO) if not name.startswith("_") and name in {"open", "write_path", "resolve"}
    )


# ── the read capability is generated from the round, not enumerated ────


def test_derive_reads_l1_and_every_strictly_earlier_round() -> None:
    """The pool is monotone (D-22 correction), so the read set grows with n rather than being n-1."""
    io = StageIO.for_stage(Stage.DERIVE, round=3, run_dir=RUN)
    assert io.may_read(SNR)
    assert io.may_read(_derivative(0))
    assert io.may_read(_derivative(2))
    assert io.may_read(EstimateKey(axis="asr", round=2))


def test_derive_may_not_read_its_own_round() -> None:
    """Reading what this stage is in the middle of writing is the sideways edge the DAG forbids."""
    io = StageIO.for_stage(Stage.DERIVE, round=1, run_dir=RUN)
    assert not io.may_read(_derivative(1))
    assert not io.may_read(EstimateKey(axis="asr", round=1))


def test_estimate_reads_its_own_round_s_derivatives_and_that_is_the_one_intra_round_edge() -> None:
    """``derive`` before ``estimate`` is exactly this edge, and it is why a round is two nodes.

    As one node ``L2_ROUND`` reads and writes the same directory and is trivially its own
    predecessor, which is what made the ordering uncheckable.
    """
    io = StageIO.for_stage(Stage.ESTIMATE, round=1, run_dir=RUN)
    assert io.may_read(_derivative(1))
    assert not io.may_read(EstimateKey(axis="asr", round=1)), "its own output"


def test_an_axis_may_read_an_l1_signal_directly() -> None:
    """The D-22 correction: derivatives are add-on signals, not a layer axes must be funnelled through.

    The ASR axis triggers off L1 transcripts, which are the finest-grained thing in the run —
    requiring a projection would hand it a coarsened copy of evidence already available in full.
    """
    io = StageIO.for_stage(Stage.ESTIMATE, round=0, run_dir=RUN)
    assert io.may_read(SignalKey(target="transcript", producer="Qwen/Qwen3-ASR-1.7B", route=Route()))


def test_l1_reads_nothing_inside_the_run() -> None:
    """It measures the audio. A signal derived from another signal is a derivative."""
    io = StageIO.for_stage(Stage.L1, run_dir=RUN)
    assert not io.may_read(SNR)
    assert not io.may_read(_derivative(0))


def test_final_is_read_by_nothing_and_computes_nothing() -> None:
    """A deliverable something reads is an intermediate wearing the wrong name.

    ``final/summary.json`` carrying 4.8 MB of L1 evidence that the pipeline read back is the case in
    hand.
    """
    for stage in STAGE_ORDER:
        io = StageIO.for_stage(stage, round=3 if stage.is_round_scoped else None, run_dir=RUN)
        assert not io.may_read(ReportKey(name="summary.json", round=None)), f"{stage} reads final/"


# ── acyclicity, provable from the predicates ──────────────────────────


def _candidate_keys(rounds: int) -> list[Artifact]:
    """One artifact of every kind, at every round — the population the acyclicity check ranges over."""
    keys: list[Artifact] = [SNR]
    for index in range(rounds):
        keys.append(_derivative(index))
        keys.append(EstimateKey(axis="speaker", round=index))
        keys.append(ReportKey(name="timeline.png", round=index))
    keys.append(ReportKey(name="summary.json", round=None))
    return keys


def _io(stage: Stage, index: int, rounds: int) -> StageIO:
    """A capability for ``stage``, with a round only if it is round-scoped."""
    return StageIO.for_stage(
        stage,
        run_dir=RUN,
        round=index if stage.is_round_scoped else None,
        last_round=rounds - 1 if stage is Stage.FINAL else None,
    )


def test_no_stage_reads_what_a_later_stage_writes() -> None:
    """Acyclicity by construction, checked exhaustively over the stage/round product.

    This replaces a graph built from *pattern overlap*, where ``pass_dir(run_dir, stream) / "asr"``
    resolving to ``L1/*/asr`` intersected the ``signals`` in ``L1/signals/**`` and silently permitted
    every ``adaptive/`` read of a per-perturbation directory.
    """
    rounds = 3
    nodes = [(stage, index) for index in range(rounds) for stage in STAGE_ORDER if stage.is_round_scoped or index == 0]

    # Unscoped stages sit at the ends: L1 before every round, FINAL after all of them.
    def rank(node: tuple[Stage, int]) -> tuple[int, int]:
        stage, index = node
        if stage is Stage.L1:
            return (-1, 0)
        if stage is Stage.FINAL:
            return (rounds, 0)
        return (index, STAGE_ORDER.index(stage))

    position = {node: i for i, node in enumerate(sorted(nodes, key=rank))}
    candidates = _candidate_keys(rounds)
    written = {node: [k for k in candidates if _io(*node, rounds).may_write(k)] for node in nodes}
    checked = 0
    for reader in nodes:
        io = _io(*reader, rounds)
        for writer, keys in written.items():
            if writer == reader:
                continue
            if any(io.may_read(k) for k in keys):
                assert position[writer] < position[reader], f"{reader} reads what {writer} writes"
                checked += 1
    assert checked > 0, "the test proved nothing if no stage read any other stage's output"


def test_the_intra_round_order_is_declared_once() -> None:
    """Two orderings of derive/estimate would let a reader disagree with a writer about the DAG."""
    assert STAGE_ORDER.index(Stage.DERIVE) < STAGE_ORDER.index(Stage.ESTIMATE)
    assert STAGE_ORDER.index(Stage.ESTIMATE) < STAGE_ORDER.index(Stage.REPORT)


# ── round-scoped stages need a round, and unscoped ones must not take one ──


def test_a_round_scoped_stage_without_a_round_is_refused() -> None:
    """Defaulting to 0 would make a stage silently write round 0 from anywhere in the loop."""
    with pytest.raises(ValueError, match="round"):
        StageIO.for_stage(Stage.DERIVE, run_dir=RUN)


def test_an_unscoped_stage_given_a_round_is_refused() -> None:
    """L1 has no round. Accepting one would imply a per-round L1, which is the re-entry confusion."""
    with pytest.raises(ValueError, match="round"):
        StageIO.for_stage(Stage.L1, round=0, run_dir=RUN)


# ── the schema travels with the artifact (D-17's key rule) ─────────────


def test_a_fold_s_required_columns_are_reported_beside_its_path() -> None:
    """A key dimension the path does not supply must appear as a column.

    Otherwise a stability over one route is indistinguishable from one over five, which is the
    absent-vs-empty confusion at the schema level.
    """
    fold = DerivativeKey(
        target="stability",
        operator=Operator("mean_abs_delta"),
        sources=(SNR, SignalKey("snr", "pyannote/brouhaha", Route(perturbation="enhanced"))),
        round=0,
    )
    io = StageIO.for_stage(Stage.DERIVE, round=0, run_dir=RUN)
    assert io.required_columns(fold) == ("contributing_producers", "contributing_routes")
    assert io.required_columns(_derivative(0)) == (), "a projection's path supplies both"
