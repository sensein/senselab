"""One artifact name, one shape — asserted of the two producers, not of a recorded tree.

``L2/round/<n>/estimates/<axis>.parquet`` is written by ``fuse.write_final_uncertainty`` for the
rounds it folds and by the adaptive loop's belief store for the rounds it iterates. The tree guard
in ``stage_contract_test.py`` catches a divergence *after* a run has produced one; these tests
catch it at the two writers, which is where it can be fixed without re-running anything.

The divergence was real and shipped: fusion's rows carried ``axis``/``signal_weights``/
``weight_basis`` and the loop's carried ``status``/``p_voice``/``aleatoric_floor`` and no ``axis``
at all, so a reader plotting one axis across the trajectory got different columns on either side
of a boundary the path does not mention. Both key rules passed on both, because both are genuinely
keyed ``(axis, bucket)`` and what differed was below the key.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from senselab.audio.workflows.audio_analysis.axes import AXIS_NAMES
from senselab.audio.workflows.audio_analysis.estimates import ESTIMATE_COLUMNS, estimate_frame


def test_the_two_producers_write_the_same_columns(tmp_path: Path) -> None:
    """The proof at the source: both writers go through one declaration, so both agree.

    Written as a comparison of the two *files* rather than of the two code paths, because that is
    what a consumer sees — and because a test that merely asserted each writer calls
    ``estimate_frame`` could be satisfied by a writer that then added a column of its own.
    """
    pytest.importorskip("pandas")
    import pandas as pd

    from senselab.audio.workflows.audio_analysis.adaptive.belief import BeliefState, Vote, VoteStore
    from senselab.audio.workflows.audio_analysis.adaptive.loop import _write_round_belief
    from senselab.audio.workflows.audio_analysis.fuse import write_final_uncertainty
    from senselab.audio.workflows.audio_analysis.layout import estimates_dir
    from senselab.audio.workflows.audio_analysis.votes import PassHarvest

    harvest = PassHarvest(
        perturbation="raw",
        speaker_votes=[{"start": 0.0, "end": 0.5, "votes": {"diar_a": {"same_label_uncertainty": 0.4}}}],
    )
    write_final_uncertainty(tmp_path, harvests={"raw": harvest}, weights_by_axis={}, aggregator="min")

    store = VoteStore()
    store.add_vote(
        Vote(
            axis="speaker",
            bucket=(0.0, 0.5),
            source="diar_a",
            stream="raw",
            scope="file",
            round=0,
            payload={"same_label_uncertainty": 0.4},
        )
    )
    _write_round_belief(tmp_path, 9, BeliefState.from_store(store, aggregator="min", round_index=1))

    fused = pd.read_parquet(estimates_dir(tmp_path, 0) / "speaker.parquet")
    looped = pd.read_parquet(estimates_dir(tmp_path, 9) / "speaker.parquet")
    assert list(fused.columns) == list(ESTIMATE_COLUMNS)
    assert list(looped.columns) == list(fused.columns), (
        "one artifact name, two shapes — a reader cannot tell which producer wrote a round"
    )


def test_a_column_no_declaration_names_is_refused() -> None:
    """A producer that grows a column has to grow the schema, where the other producer sees it.

    Without this the union is a snapshot: the next column either writer adds re-opens the split,
    and the tree guard would not report it until a run had already been written.
    """
    with pytest.raises(ValueError, match="written by no declaration"):
        estimate_frame("speaker", [{"start": 0.0, "end": 0.5, "invented_by_one_writer": 1.0}], round_index=0)


def test_an_axis_with_nothing_to_say_still_has_a_shape() -> None:
    """An axis with nothing to say still has a shape, because absent is not zero.

    An axis with no rows used to be skipped, so its file was absent — and absent is how the fourth
    axis's estimates stopped after round 2 while the convergence report called it settled. The
    empty table carries every declared column, which says the first of the two.
    """
    frame = estimate_frame("background_mask", [], round_index=0)
    assert list(frame.columns) == list(ESTIMATE_COLUMNS)
    assert len(frame) == 0


def test_the_axis_column_comes_from_the_filename_not_the_caller() -> None:
    """The file is named for its axis, so the column and the name must not be able to disagree."""
    frame = estimate_frame("asr", [{"start": 0.0, "end": 0.5, "axis": "speaker", "uncertainty": 0.2}], round_index=0)
    assert list(frame["axis"]) == ["asr"]


def test_the_round_column_comes_from_the_directory_not_the_caller() -> None:
    """And the round, for the same reason: the path fixes it, so a caller cannot decide it.

    A caller *did*. The adaptive producer passed each row's last-refolded round through as
    ``round``, so ``L2/round/4/estimates/speech_presence.parquet`` held rows claiming rounds 1, 3
    and 4 — and the round-4 extraction in ``final/`` inherited the claim. The fact the loop was
    spending the column on survives beside it, under a name that says what it is.
    """
    frame = estimate_frame(
        "asr",
        [{"start": 0.0, "end": 0.5, "round": 1, "last_refolded_round": 1}],
        round_index=4,
    )
    assert list(frame["round"]) == [4]
    assert list(frame["last_refolded_round"]) == [1]


def test_every_active_axis_is_writable_under_this_schema() -> None:
    """A schema that only fits three axes is the three-axis bug in another place."""
    for axis in AXIS_NAMES:
        assert list(estimate_frame(axis, [{"start": 0.0, "end": 0.5}], round_index=0)["axis"]) == [axis]


def test_the_writers_produce_what_the_declaration_says_a_round_and_final_owe(tmp_path: Path) -> None:
    """Driven through the real writers, then checked against the real declaration.

    The recorded fixture in ``stage_contract_test.py`` is a transcription of a run, so it can drift
    from what the code emits — that is exactly how it came to claim every round wrote a summary and
    a timeline and all four axes when none of the three was true. This drives
    ``write_final_uncertainty`` and ``run_adaptive_loop`` over synthetic harvests and asks the
    conformance guard about the tree they actually leave, which is the one thing a fixture cannot
    tell you.

    Deliberately *not* a substitute for the recorded fixture or for the real-run test: this tree
    has one perturbation, no models and no L1, so it exercises the round and ``final/`` writers and
    nothing else. What it proves is that those writers satisfy their own declaration.

    **The fusion half has to fold more than one round, and one of its axes has to converge before
    the rest.** This assertion existed and passed while the run on disk violated it, for exactly
    that reason: at ``max_rounds=1`` fusion writes round 0 and nothing else, so every round in the
    tree came from the adaptive producer — which writes all four axes unconditionally — and the
    path where an axis stops being re-folded was never reached. ``diarization_by_model`` is what
    reaches it: it gives ``_speaker_assignment`` a binding, so C2 holds for ``speaker`` and blocks
    for the other three, and ``speaker`` converges with rounds still to run.
    """
    pytest.importorskip("pandas")
    import pandas as pd

    from senselab.audio.workflows.audio_analysis.adaptive.loop import run_adaptive_loop
    from senselab.audio.workflows.audio_analysis.contracts import (
        STAGE_CONTRACTS,
        artifact_violations,
        enumerated_members,
        matches,
    )
    from senselab.audio.workflows.audio_analysis.fuse import write_final_uncertainty
    from senselab.audio.workflows.audio_analysis.layout import rounds_present
    from senselab.audio.workflows.audio_analysis.votes import PassHarvest

    harvest = PassHarvest(
        perturbation="raw",
        speech_presence_evidence=[
            {"start": 0.0, "end": 0.5, "evidence": {"m1": {"covered_fraction": 1.0}}},
            {"start": 0.5, "end": 1.0, "evidence": {"m1": {"covered_fraction": 0.0}}},
        ],
        speaker_votes=[
            {
                "start": 0.0,
                "end": 0.5,
                "votes": {"diar_a": {"same_label_uncertainty": 0.4, "cluster_ids": {"SPEAKER_00": "C0"}}},
            },
            {
                "start": 0.5,
                "end": 1.0,
                "votes": {"diar_a": {"same_label_uncertainty": 0.4, "cluster_ids": {"SPEAKER_01": "C1"}}},
            },
        ],
        asr_votes=[{"start": 0.0, "end": 1.0, "votes": {"a": {"text": "hi"}}}],
        # The mask axis's evidence rides on the harvest with the other three, per bucket. It used to
        # be handed to the loop separately as one vote per mask *region*, so the two producers of
        # this tree disagreed about the fourth axis by three orders of magnitude in bucket count.
        background_mask_evidence=[
            {"start": 0.0, "end": 0.5, "task_type": "speech", "votes": {"speech": {"same_label_uncertainty": 0.2}}},
            {"start": 0.5, "end": 1.0, "task_type": "speech", "votes": {"speech": {"same_label_uncertainty": 0.6}}},
        ],
        # What lets one axis settle before the others: a declared-capacity diarizer's spans give
        # C2 something to be stable about, and C2 is a claim about the speaker axis alone.
        diarization_by_model={
            "pyannote/speaker-diarization-community-1": {
                "status": "ok",
                "result": [
                    [
                        {"start": 0.0, "end": 0.5, "speaker": "SPEAKER_00"},
                        {"start": 0.5, "end": 1.0, "speaker": "SPEAKER_01"},
                    ]
                ],
            }
        },
        grids={"asr": {"win_length": 1.0, "hop_length": 1.0}},
    )
    # Regions, not buckets: this is what regional trust withdraws over, which is the one thing a
    # region is still the right unit for.
    mask = [{"start": 0.0, "end": 1.0, "state": "target_free", "confidence": 0.8}]
    fusion_rounds = 4
    written = write_final_uncertainty(
        tmp_path,
        harvests={"raw": harvest},
        weights_by_axis={},
        aggregator="min",
        mask_regions=mask,
        max_rounds=fusion_rounds,
    )
    run_adaptive_loop(
        tmp_path,
        harvests={"raw": harvest},
        summary={"passes": {"raw": {"duration_s": 1.0, "audio_signature": "a" * 64}}},
        max_rounds=3,
        aggregator="min",
    )

    # The precondition, asserted rather than assumed: without an axis that stops early this test
    # cannot fail, and a test that cannot fail is how the gap got here.
    stopped_early = {
        axis
        for axis, entries in written["round_logs"].items()
        if max(int(e["round"]) for e in entries) < fusion_rounds - 1
    }
    assert stopped_early, (
        "no fusion axis stopped before the last round, so the carry-forward path this test exists for was not exercised"
    )

    rounds = rounds_present(tmp_path)
    assert len(rounds) >= 2, "a loop that wrote one round proves nothing about the second producer"

    # Every round owes its belief, its account and its view — for every active axis.
    for index in rounds:
        base = tmp_path / "L2" / "round" / str(index)
        assert (base / "summary.json").is_file(), f"round {index} left no account of itself"
        assert (base / "timeline.png").is_file(), f"round {index} left no view of itself"
        for axis in AXIS_NAMES:
            estimate = base / "estimates" / f"{axis}.parquet"
            assert estimate.is_file(), (
                f"round {index} has no belief about {axis} — and an absent file says "
                "'never asked', not 'nothing to say'"
            )
            # And it is *this* round's belief. A row claiming another round sends anything that
            # derives a path from the column to a different fold's numbers.
            frame = pd.read_parquet(estimate)
            assert set(frame["round"].dropna().astype(int)) <= {index}, (
                f"round {index}'s {axis} estimate carries rounds {sorted(set(frame['round'].dropna().astype(int)))}"
            )

    # An axis carried forward keeps saying which round produced its numbers, or "settled in round 1"
    # and "re-folded to the same value" are the same row.
    for axis in stopped_early:
        last = max(int(e["round"]) for e in written["round_logs"][axis])
        carried = pd.read_parquet(tmp_path / "L2" / "round" / str(fusion_rounds - 1) / "estimates" / f"{axis}.parquet")
        assert set(carried["last_refolded_round"].dropna().astype(int)) == {last}

    # final/ carries the extraction, one file per active axis, plus the run's own account.
    for axis in AXIS_NAMES:
        assert (tmp_path / "final" / "estimates" / f"{axis}.parquet").is_file(), axis
    assert (tmp_path / "final" / "decisions.json").is_file()

    # And the declaration agrees: nothing here is keyed against its artifact, and the enumerated
    # instances of every L2_ROUND artifact that owes one are all present.
    assert artifact_violations(tmp_path) == []
    produced = [p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*") if p.is_file()]
    members = enumerated_members(tmp_path)
    for artifact in STAGE_CONTRACTS["L2_ROUND"].instantiate().writes:
        for instance in artifact.instances(members):
            assert any(matches(relative, instance) for relative in produced), instance
