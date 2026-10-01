"""SECOND_OPINION: the decision model reads what the reviewer reads, and its answer is a measurement.

``specs/20261001-nimble-second-opinion/design.md`` is the design.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, Mapping, Sequence

from senselab.audio.data_structures import AudioHints
from senselab.audio.workflows.triage.config import TriageConfig, load_triage_config
from senselab.audio.workflows.triage.nodes.second_opinion import (
    ABSENT,
    DISABLED,
    NODE,
    NOTHING_TO_READ,
    OK,
    opinion_cache_key,
    pin_of,
    second_opinion,
)
from senselab.audio.workflows.triage.vocabulary import NIMBLE_OPINION, PII_SCAN
from senselab.utils.prov_store import ProvStore
from tests.audio.workflows.triage.nodes.conftest import word_attributes

ON = "second_opinion:\n  enabled: true\n"


def _config(tmp_path: Path, text: str = ON) -> TriageConfig:
    path = tmp_path / "override.yaml"
    path.write_text(text, encoding="utf-8")
    return load_triage_config(path)


def _store(words: Sequence[str] = ("my", "name", "is", "alicia")) -> ProvStore:
    """Consensus words and SPEECH's scan of all of them: the residue REVIEW reads."""
    store = ProvStore(run_id="second-opinion")
    software = store.agent(agent_type="software", version="senselab test-seed")
    activity = store.activity(node="PREPROCESS", step="consensus", parameters={})
    store.was_associated_with(activity, software)
    word_ids = []
    for index, text in enumerate(words):
        extent = (float(index), float(index) + 0.5)
        word_id = store.entity(prov_type="word", extent=extent, attributes=word_attributes(text, extent, index=index))
        store.was_generated_by(word_id, activity)
        word_ids.append(word_id)
    transcript = store.entity(
        prov_type="measurement",
        extent=None,
        attributes={
            "name": "consensus_transcript",
            "signal": "plain",
            "role": "consensus",
            "n_words": len(words),
            "word_ids": word_ids,
            "text": " ".join(words),
        },
    )
    store.was_generated_by(transcript, activity)
    scan = store.entity(
        prov_type="measurement",
        extent=None,
        attributes={"name": PII_SCAN, "signal": "consensus_transcript", "findings_n": 0, "residue_word_ids": word_ids},
    )
    store.was_generated_by(scan, activity)
    return store


def _answers(other: float = 0.9) -> dict[str, Any]:
    return {
        "other_voice": {"choice": "more_than_one", "probabilities": {"one": 1 - other, "more_than_one": other}},
        "instructions_spoken": {"noul": 0.02},
        "named_diagnosis": {"noul": 0.01},
        "safe_harbor_identifier_present": {"noul": 0.97},
    }


class _Ask:
    def __init__(self, answers: Mapping[str, Any] | Exception) -> None:
        self.answers = answers
        self.states: list[Mapping[str, Any]] = []

    def __call__(self, state: Mapping[str, Any], questions: Any) -> Mapping[str, Any]:  # noqa: ANN401
        self.states.append(state)
        if isinstance(self.answers, Exception):
            raise self.answers
        return self.answers


def _opinion(store: ProvStore, measurement_id: str) -> dict[str, Any]:
    return dict(store.get_entity(measurement_id).attributes)


class TestItAsksAndRecords:
    """One measurement on every path, with the model identified by its blob digest."""

    def test_an_answer_lands_as_probabilities_with_its_pin(self, tmp_path: Path) -> None:
        """The good path."""
        store, ask = _store(), _Ask(_answers())
        config = _config(tmp_path)
        outcome = second_opinion(store, config, AudioHints(task="free-speech"), ask)
        held = _opinion(store, outcome.measurement_id)
        assert outcome.status == OK and held["name"] == NIMBLE_OPINION
        assert held["probabilities"]["other_voice"] == 0.9
        assert held["probabilities"]["safe_harbor_identifier_present"] == 0.97
        assert held["blob_digest"] == pin_of(config).blob_digest
        assert "alicia" in ask.states[0]["transcript"]
        assert store.get_activity(store.generated_by(outcome.measurement_id) or "").node == NODE

    def test_the_model_agent_carries_the_digest_not_a_fake_commit(self, tmp_path: Path) -> None:
        """An Ollama blob is not a git commit; the agent says so and records the digest as its version."""
        store = _store()
        config = _config(tmp_path)
        second_opinion(store, config, None, _Ask(_answers()))
        models = [agent for agent in store.agents("model")]
        assert len(models) == 1
        assert models[0].model_id.endswith(pin_of(config).blob_digest)
        assert models[0].version == pin_of(config).blob_digest

    def test_a_second_ask_is_served_from_the_cache(self, tmp_path: Path) -> None:
        """Same text, same context, same pin: the model is not asked again."""
        config = _config(tmp_path)
        first = _Ask(_answers())
        second_opinion(_store(), config, None, first)
        again = _Ask(RuntimeError("must not be asked"))
        store = _store()
        outcome = second_opinion(store, config, None, again)
        assert outcome.status == OK and not again.states
        assert _opinion(store, outcome.measurement_id)["result_cache"]["hit"] is True

    def test_the_key_moves_with_the_text_the_context_the_blob_and_the_seed(self, tmp_path: Path) -> None:
        """Each of the four is in the key."""
        pin = pin_of(_config(tmp_path))
        base = opinion_cache_key("hello", {"task": "a"}, pin, 7)
        moved_pin = dataclasses.replace(pin, blob_digest="sha256:" + "0" * 64)
        assert (
            len(
                {
                    base,
                    opinion_cache_key("hello there", {"task": "a"}, pin, 7),
                    opinion_cache_key("hello", {"task": "b"}, pin, 7),
                    opinion_cache_key("hello", {"task": "a"}, moved_pin, 7),
                    opinion_cache_key("hello", {"task": "a"}, pin, 8),
                }
            )
            == 5
        )


class TestTheOtherPaths:
    """Off, nothing to read, and asked without an answer are each recorded, and none asks the model."""

    def test_the_packaged_config_is_disabled_and_asks_nothing(self, tmp_path: Path) -> None:
        """No GPU and no store in a default run."""
        store, ask = _store(), _Ask(_answers())
        outcome = second_opinion(store, load_triage_config(), None, ask)
        assert outcome.status == DISABLED and not ask.states

    def test_an_empty_transcript_asks_nothing(self, tmp_path: Path) -> None:
        """No residue, nothing to read."""
        store, ask = _store(words=()), _Ask(_answers())
        outcome = second_opinion(store, _config(tmp_path), None, ask)
        assert outcome.status == NOTHING_TO_READ and not ask.states

    def test_no_server_is_absent_with_a_failure(self, tmp_path: Path) -> None:
        """ask=None on a cache miss."""
        store = _store()
        outcome = second_opinion(store, _config(tmp_path), None, None)
        assert outcome.status == ABSENT
        assert _opinion(store, outcome.measurement_id)["failure"]

    def test_a_malformed_answer_is_absent_and_is_not_cached(self, tmp_path: Path) -> None:
        """A missing question is a failure, never stored as a reading."""
        config = _config(tmp_path)
        broken = _answers()
        del broken["named_diagnosis"]
        store = _store()
        outcome = second_opinion(store, config, None, _Ask(broken))
        assert outcome.status == ABSENT and "named_diagnosis" in _opinion(store, outcome.measurement_id)["failure"]
        retried = _Ask(_answers())
        assert second_opinion(_store(), config, None, retried).status == OK and retried.states
