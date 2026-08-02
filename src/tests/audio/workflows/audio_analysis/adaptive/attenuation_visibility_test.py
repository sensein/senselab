"""Attenuation has to be legible in the artifacts, or "the claim stays visible" is not a property.

The store keeps an attenuated vote in the fold and records why, and ``reaggregate_bucket`` returns
both the withdrawn weights and the measurement behind them. None of it reached a file. The round
belief parquet wrote ``n_sources`` — a count that is *identical* before and after an attenuation,
because attenuation is the mechanism that deliberately does not change who contributed — and the
final presence rows wrote neither.

So a bucket whose only speech claim had been discounted to the floor was indistinguishable, in
every artifact a consumer reads, from one where every source agreed. The withdrawal survived only
in memory: the loop acted on evidence it never published, and an analyst asking "why is this span
uncertain / why did this speaker thin out here" had nothing to read.

These tests fix the property at the file boundary, since that is where it failed — not at the
store, which was already correct.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from senselab.audio.workflows.audio_analysis.adaptive.belief import BeliefState, Vote, VoteStore, bucket_key
from senselab.audio.workflows.audio_analysis.adaptive.policy import load_policy
from senselab.audio.workflows.audio_analysis.floors import MIN_EVIDENCE_WEIGHT
from senselab.audio.workflows.audio_analysis.speech_presence_link import directed_presence_vote

pytest.importorskip("pandas")

STREAM = "raw_16k"
CLAIMANT = "openai/whisper-large-v3"
QUIET = bucket_key(0.0, 0.5)  # the frame voter reports near-silence; the ASR claims speech
LOUD = bucket_key(0.5, 1.0)  # everyone agrees — nothing is withdrawn here
CORROBORATION = 0.02


def _vote(source: str, bucket: tuple[float, float], p_speech: float, **extra: Any) -> Vote:  # noqa: ANN401
    return Vote(
        axis="speech_presence",
        bucket=bucket,
        source=source,
        stream=STREAM,
        scope="file",
        round=1,
        payload={**directed_presence_vote(p_speech), **extra},
    )


def _attenuated_store() -> VoteStore:
    """One bucket where a lone ASR claim was discounted, one where nothing was."""
    store = VoteStore()
    store.add_vote(_vote("frame_brouhaha_vad", QUIET, CORROBORATION, frame_mean=CORROBORATION))
    store.add_vote(_vote(CLAIMANT, QUIET, 0.9, word_overlap_s=0.4))
    store.add_vote(_vote("frame_brouhaha_vad", LOUD, 0.95, frame_mean=0.95))
    store.add_vote(_vote(CLAIMANT, LOUD, 0.9, word_overlap_s=0.4))
    store.attenuate_source_in_bucket(
        STREAM,
        QUIET,
        CLAIMANT,
        corroboration=CORROBORATION,
        evidence_sources=["frame_brouhaha_vad"],
        reason="uncorroborated_speech_claim",
        round_idx=2,
        measured_on=("speech_presence", QUIET),
    )
    return store


def _rows_by_bucket(frame: Any) -> dict[tuple[float, float], dict[str, Any]]:  # noqa: ANN401
    return {bucket_key(r["start"], r["end"]): dict(r) for _, r in frame.iterrows()}


def _written_round_belief(tmp_path: Path) -> dict[tuple[float, float], dict[str, Any]]:
    import pandas as pd

    from senselab.audio.workflows.audio_analysis.adaptive.loop import _write_round_belief

    store = _attenuated_store()
    state = BeliefState.from_store(store, aggregator="min")
    _write_round_belief(tmp_path, state)
    return _rows_by_bucket(pd.read_parquet(tmp_path / "belief" / "speech_presence.parquet"))


def _written_final_presence(tmp_path: Path) -> dict[tuple[float, float], dict[str, Any]]:
    import pandas as pd

    from senselab.audio.workflows.audio_analysis.adaptive.fusion import build_final_outputs
    from senselab.audio.workflows.audio_analysis.layout import belief_dir

    store = _attenuated_store()
    state = BeliefState.from_store(store, aggregator="min")
    build_final_outputs(
        out_dir=tmp_path,
        words=[{"text": "maybe", "start": 0.1, "end": 0.4, "confidence": 0.9, "corroboration": 0.95}],
        store=store,
        state=state,
        stream=STREAM,
        policy=load_policy(),
        generated_from_round=2,
        corroboration_provenance={"evidence_pool": ["frame_brouhaha_vad"]},
    )
    return _rows_by_bucket(pd.read_parquet(belief_dir(tmp_path) / "speech_presence.parquet"))


@pytest.mark.parametrize("written", [_written_round_belief, _written_final_presence], ids=["round", "final"])
def test_an_attenuated_bucket_is_distinguishable_from_an_unattenuated_one(
    written: Any,  # noqa: ANN401
    tmp_path: Path,
) -> None:
    """The minimum claim: read the file, tell the two apart.

    ``n_sources`` could not do this. Attenuation keeps the source contributing — that is its whole
    point — so the count is the same on both rows.
    """
    rows = written(tmp_path)
    assert rows[QUIET]["n_attenuated_sources"] == 1
    assert rows[LOUD]["n_attenuated_sources"] == 0
    assert json.loads(rows[LOUD]["attenuated_sources"]) == {}


@pytest.mark.parametrize("written", [_written_round_belief, _written_final_presence], ids=["round", "final"])
def test_the_artifact_names_the_source_and_the_weight_it_was_left_with(
    written: Any,  # noqa: ANN401
    tmp_path: Path,
) -> None:
    """Which source, and by how much. A flag alone cannot be appealed against."""
    weights = json.loads(written(tmp_path)[QUIET]["attenuated_sources"])
    assert weights[CLAIMANT] == pytest.approx(MIN_EVIDENCE_WEIGHT)


@pytest.mark.parametrize("written", [_written_round_belief, _written_final_presence], ids=["round", "final"])
def test_the_artifact_carries_the_measurement_that_produced_the_withdrawal(
    written: Any,  # noqa: ANN401
    tmp_path: Path,
) -> None:
    """The corroboration, its pool, where it was taken, and the floor that stopped it.

    Without these the weight is an assertion. With them a reader can re-derive it — and can
    disagree with the threshold without re-running a model, which is the whole reason every
    threshold in this system is named.
    """
    detail = json.loads(written(tmp_path)[QUIET]["attenuation"])
    assert [d["source"] for d in detail] == [CLAIMANT]
    factor = detail[0]
    assert factor["evidence_weight"] == pytest.approx(MIN_EVIDENCE_WEIGHT)
    assert factor["corroboration"] == pytest.approx(CORROBORATION)
    assert factor["evidence_sources"] == ["frame_brouhaha_vad"]
    assert factor["reason"] == "uncorroborated_speech_claim"
    assert factor["measured_on"] == {"axis": "speech_presence", "bucket": [QUIET[0], QUIET[1]]}
    assert factor["floor"] == pytest.approx(MIN_EVIDENCE_WEIGHT)
    assert factor["round"] == 2


def test_every_withdrawal_is_listed_not_only_the_last(tmp_path: Path) -> None:
    """Two rules may each have something to say about one vote; an overwrite hides the first.

    The store already appends its factors. The artifact has to keep them appended, or the audit
    trail is truncated exactly where it gets interesting.
    """
    import pandas as pd

    from senselab.audio.workflows.audio_analysis.adaptive.loop import _write_round_belief

    store = _attenuated_store()
    store.attenuate_source_in_bucket(
        STREAM,
        QUIET,
        CLAIMANT,
        corroboration=0.4,
        evidence_sources=["frame_brouhaha_vad"],
        reason="another_rule_with_something_to_say",
        round_idx=3,
        measured_on=("speech_presence", QUIET),
    )
    state = BeliefState.from_store(store, aggregator="min")
    _write_round_belief(tmp_path, state)
    rows = _rows_by_bucket(pd.read_parquet(tmp_path / "belief" / "speech_presence.parquet"))
    reasons = [f["reason"] for f in json.loads(rows[QUIET]["attenuation"])]
    assert reasons == ["uncorroborated_speech_claim", "another_rule_with_something_to_say"]


def test_a_later_round_republishes_the_attenuation_it_inherited(tmp_path: Path) -> None:
    """Re-aggregating one bucket must not drop the columns off the rest of the run.

    ``update_buckets`` refreshes only the touched rows, so anything it forgets to carry over
    silently disappears from every later round's file.
    """
    import pandas as pd

    from senselab.audio.workflows.audio_analysis.adaptive.loop import _write_round_belief

    store = _attenuated_store()
    state = BeliefState.from_store(store, aggregator="min")
    state.update_buckets(store, "speech_presence", {QUIET}, round_idx=3)
    _write_round_belief(tmp_path, state)
    rows = _rows_by_bucket(pd.read_parquet(tmp_path / "belief" / "speech_presence.parquet"))
    assert rows[QUIET]["n_attenuated_sources"] == 1
    assert json.loads(rows[QUIET]["attenuation"])[0]["corroboration"] == pytest.approx(CORROBORATION)
