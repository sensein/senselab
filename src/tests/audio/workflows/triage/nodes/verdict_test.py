"""VERDICT reads the store into the fold and records the result. Nothing here loads a model."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import pytest

from senselab.audio.data_structures import AudioHints
from senselab.audio.workflows.triage.config import TriageConfig, load_triage_config
from senselab.audio.workflows.triage.nodes import verdict as verdict_module
from senselab.audio.workflows.triage.nodes.common import software_agent, write_verdict
from senselab.audio.workflows.triage.nodes.routing import routing
from senselab.audio.workflows.triage.run import GRAPH_ORDER
from senselab.audio.workflows.triage.vocabulary import UNREAD_DECLARATION, Outcome, Release, RunState, Triage
from senselab.utils.prov_store import Entity, ProvStore

BASE: tuple[tuple[str, Outcome, str | None], ...] = (
    ("ADMIT", Outcome.PASS, None),
    ("TAXONOMY", Outcome.PASS, None),
    ("AIRWAY", Outcome.PASS, "airway"),
    ("SPEECH", Outcome.PASS, "speech"),
)
KINDS = {"airway": "present", "speech": "present", "voice": "absent"}


def _hint_config(tmp_path: Path) -> TriageConfig:
    """The packaged config with a hint map supplied, the same one ROUTING is tested against.

    Args:
        tmp_path: The test's temporary directory, holding the override file.

    Returns:
        The merged configuration.
    """
    path = tmp_path / "hints.yaml"
    path.write_text("routing:\n  hint_kind_map:\n    cough: airway\n    read-speech: speech\n")
    return load_triage_config(path)


@pytest.fixture
def make_verdict_store(tmp_path: Path) -> Callable[..., ProvStore]:
    """A builder seeding node verdicts and kinds, then letting the real ROUTING decide over them."""

    def _make(
        *,
        node_verdicts: Sequence[tuple[str, Outcome, str | None]] = (),
        kinds: Mapping[str, str] | None = None,
        route: bool = True,
        config: TriageConfig | None = None,
        hint: AudioHints | None = None,
    ) -> ProvStore:
        store = ProvStore(run_id="verdict-test")
        agent = software_agent(store)
        taxonomy = store.activity(node="TAXONOMY", step="seed-kinds", parameters={})
        store.was_associated_with(taxonomy, agent)
        for kind_name, state in (kinds or {}).items():
            kind_id = store.entity(
                prov_type="kind",
                extent=None,
                attributes={"kind": kind_name, "state": state, "lines": {}, "stream": "plain"},
            )
            store.was_generated_by(kind_id, taxonomy)
        if route:
            routing(store, None, config or load_triage_config(), hint, run_dir=tmp_path)
        for node, outcome, kind in node_verdicts:
            activity = store.activity(node=node, step="seed", parameters={})
            store.was_associated_with(activity, agent)
            write_verdict(
                store,
                activity,
                agent,
                node=node,
                outcome=outcome,
                kind=kind,
                why=f"{node} concluded {outcome.value}",
                detail={},
            )
        return store

    return _make


def _file_verdict_entity(store: ProvStore) -> Entity:
    """The verdict entity whose node attribute is VERDICT — the file verdict, not a node's.

    Args:
        store: The provenance store.

    Returns:
        The single file verdict entity.
    """
    found = [e for e in store.entities("verdict") if e.attributes["node"] == "VERDICT"]
    assert len(found) == 1, f"expected exactly one file verdict, found {len(found)}"
    return found[0]


class TestTheTriageAxisIsWired:
    """The three values reach the store, and each carries the ground the fold gave it."""

    def test_a_branch_fail_against_an_absent_kind_is_a_file_pass(
        self, make_verdict_store: Callable[..., ProvStore], config: TriageConfig, tmp_path: Path
    ) -> None:
        """A cough recording: airway present and found, speech absent and not looked for."""
        store = make_verdict_store(
            node_verdicts=[("ADMIT", Outcome.PASS, None), ("AIRWAY", Outcome.PASS, "airway")],
            kinds={"airway": "present", "speech": "absent", "voice": "absent"},
        )
        result = verdict_module.verdict(store, None, config, run_dir=tmp_path)
        assert result.file_verdict.triage is Triage.PASS
        assert result.file_verdict.kinds == {"airway": "present", "speech": "absent", "voice": "absent"}

    def test_an_admit_failure_discards_as_unmeasurable(
        self, make_verdict_store: Callable[..., ProvStore], config: TriageConfig, tmp_path: Path
    ) -> None:
        """Nothing ran, so nothing is claimed about the recording."""
        store = make_verdict_store(node_verdicts=[("ADMIT", Outcome.FAIL, None)], kinds={}, route=False)
        result = verdict_module.verdict(store, None, config, run_dir=tmp_path)
        assert result.file_verdict.triage is Triage.DISCARD
        assert result.file_verdict.discard_ground == "unmeasurable"

    def test_every_kind_absent_discards_as_acoustically_empty(
        self, make_verdict_store: Callable[..., ProvStore], config: TriageConfig, tmp_path: Path
    ) -> None:
        """ROUTING declined every branch and recorded it; the fold reads that off the decisions."""
        store = make_verdict_store(
            node_verdicts=[("ADMIT", Outcome.PASS, None), ("TAXONOMY", Outcome.FAIL, None)],
            kinds={"airway": "absent", "speech": "absent", "voice": "absent"},
        )
        result = verdict_module.verdict(store, None, config, run_dir=tmp_path)
        assert result.file_verdict.triage is Triage.DISCARD
        assert result.file_verdict.discard_ground == "acoustically_empty"

    def test_the_outcome_attribute_carries_the_triage_value(
        self, make_verdict_store: Callable[..., ProvStore], config: TriageConfig, tmp_path: Path
    ) -> None:
        """``discard`` is not an ``Outcome``, and the entity must still record it."""
        store = make_verdict_store(node_verdicts=[("ADMIT", Outcome.FAIL, None)], kinds={}, route=False)
        verdict_module.verdict(store, None, config, run_dir=tmp_path)
        entity = _file_verdict_entity(store)
        assert entity.attributes["outcome"] == "discard"
        assert entity.attributes["triage"] == "discard"
        assert entity.attributes["discard_ground"] == "unmeasurable"


class TestTheClassificationIsReadVerbatim:
    """A kind state is a string here; the fold reads it and never coerces it."""

    def test_uncertain_folds_without_raising(
        self, make_verdict_store: Callable[..., ProvStore], config: TriageConfig, tmp_path: Path
    ) -> None:
        """TAXONOMY writes ``uncertain`` on a real run, which the vocabulary once had no member for."""
        store = make_verdict_store(
            node_verdicts=[("ADMIT", Outcome.PASS, None), ("SPEECH", Outcome.PASS, "speech")],
            kinds={"airway": "absent", "speech": "uncertain", "voice": "absent"},
        )
        result = verdict_module.verdict(store, None, config, run_dir=tmp_path)
        assert result.file_verdict.screened["speech"] == "uncertain"
        assert result.file_verdict.kinds["speech"] == "present"
        assert result.file_verdict.agreement["speech"] == "resolved"

    def test_a_state_nobody_can_read_is_reported_not_raised(
        self, make_verdict_store: Callable[..., ProvStore], config: TriageConfig, tmp_path: Path
    ) -> None:
        """The classification is reported beside the branches; refusing to fold it hides the branch too."""
        store = make_verdict_store(
            node_verdicts=[("ADMIT", Outcome.PASS, None), ("AIRWAY", Outcome.PASS, "airway")],
            kinds={"airway": "maybe", "speech": "absent", "voice": "absent"},
        )
        result = verdict_module.verdict(store, None, config, run_dir=tmp_path)
        assert result.file_verdict.screened["airway"] == "maybe"
        assert result.file_verdict.kinds["airway"] == "present"

    def test_the_classification_is_never_rewritten_in_the_store(
        self, make_verdict_store: Callable[..., ProvStore], config: TriageConfig, tmp_path: Path
    ) -> None:
        """A branch resolving its kind leaves TAXONOMY's element exactly as it was."""
        store = make_verdict_store(
            node_verdicts=[("ADMIT", Outcome.PASS, None), ("SPEECH", Outcome.PASS, "speech")],
            kinds={"airway": "absent", "speech": "absent", "voice": "absent"},
        )
        result = verdict_module.verdict(store, None, config, run_dir=tmp_path)
        assert result.file_verdict.agreement["speech"] == "mismatch"
        assert result.file_verdict.triage is Triage.FLAG
        assert [e.attributes["state"] for e in store.entities("kind") if e.attributes["kind"] == "speech"] == ["absent"]

    def test_the_latest_live_classification_wins(
        self, make_verdict_store: Callable[..., ProvStore], config: TriageConfig, tmp_path: Path
    ) -> None:
        """Two elements for one kind are one assertion, the later; the store's shared rule."""
        store = make_verdict_store(node_verdicts=[("ADMIT", Outcome.PASS, None)], kinds={"speech": "absent"})
        taxonomy = store.activity(node="TAXONOMY", step="revise", parameters={})
        revised = store.entity(
            prov_type="kind", extent=None, attributes={"kind": "speech", "state": "present", "lines": {}}
        )
        store.was_generated_by(revised, taxonomy)
        result = verdict_module.verdict(store, None, config, run_dir=tmp_path)
        assert result.file_verdict.screened["speech"] == "present"


class TestAnUnreadableNodeVerdictDoesNotKillTheFold:
    """One node writing something no reader can act on must not cost the whole file verdict."""

    def test_an_alien_outcome_flags_and_names_the_node_and_the_value(
        self, make_verdict_store: Callable[..., ProvStore], config: TriageConfig, tmp_path: Path
    ) -> None:
        """The union on ``write_verdict``'s outcome means a node can write a triage value by mistake."""
        store = make_verdict_store(
            node_verdicts=[("ADMIT", Outcome.PASS, None), ("AIRWAY", Outcome.PASS, "airway")],
            kinds={"airway": "present", "speech": "present", "voice": "absent"},
        )
        activity = store.activity(node="SPEECH", step="seed", parameters={})
        agent = software_agent(store)
        store.was_associated_with(activity, agent)
        alien = store.entity(
            prov_type="verdict",
            extent=None,
            attributes={"node": "SPEECH", "outcome": "discard", "kind": "speech", "why": "a node erred"},
        )
        store.was_generated_by(alien, activity)

        result = verdict_module.verdict(store, None, config, run_dir=tmp_path)
        assert result.file_verdict.triage is Triage.FLAG
        assert any(
            "SPEECH" in reason.why and "'discard'" in reason.why for reason in result.file_verdict.reasons
        ), "the offending node and the value it wrote are both named"

    def test_the_unreadable_verdict_resolves_no_kind_and_the_fold_still_completes(
        self, make_verdict_store: Callable[..., ProvStore], config: TriageConfig, tmp_path: Path
    ) -> None:
        """Every other node's conclusion survives, and the kind that node screened stays unanswered."""
        store = make_verdict_store(
            node_verdicts=[("ADMIT", Outcome.PASS, None), ("AIRWAY", Outcome.PASS, "airway")],
            kinds={"airway": "present", "speech": "present", "voice": "absent"},
        )
        activity = store.activity(node="SPEECH", step="seed", parameters={})
        store.was_associated_with(activity, software_agent(store))
        alien = store.entity(
            prov_type="verdict",
            extent=None,
            attributes={"node": "SPEECH", "outcome": "banana", "kind": "speech", "why": "a node erred"},
        )
        store.was_generated_by(alien, activity)

        result = verdict_module.verdict(store, None, config, run_dir=tmp_path)
        assert result.file_verdict.kinds["airway"] == "present"
        assert result.file_verdict.kinds["speech"] == "present", "the classification stands where no branch answered"
        assert result.file_verdict.agreement["speech"] == "not_run"
        assert _file_verdict_entity(store).attributes["triage"] == "flag"


class TestTheBranchDecisionsAreRead:
    """Which branch was asked is a store fact now, not a guess from the classification."""

    def test_a_declined_branch_is_expected_and_does_not_flag(
        self, make_verdict_store: Callable[..., ProvStore], config: TriageConfig, tmp_path: Path
    ) -> None:
        """ROUTING declined SPEECH and said why; a missing verdict there is the design working."""
        store = make_verdict_store(
            node_verdicts=[("ADMIT", Outcome.PASS, None), ("AIRWAY", Outcome.PASS, "airway")],
            kinds={"airway": "present", "speech": "absent", "voice": "absent"},
        )
        result = verdict_module.verdict(store, None, config, run_dir=tmp_path)
        assert result.file_verdict.agreement["speech"] == "not_run"
        assert result.file_verdict.triage is Triage.PASS

    def test_an_asked_branch_that_left_no_verdict_flags(
        self, make_verdict_store: Callable[..., ProvStore], config: TriageConfig, tmp_path: Path
    ) -> None:
        """ROUTING selected SPEECH and nothing came back; the reason names which silence it was."""
        store = make_verdict_store(
            node_verdicts=[("ADMIT", Outcome.PASS, None), ("AIRWAY", Outcome.PASS, "airway")],
            kinds={"airway": "present", "speech": "present", "voice": "absent"},
        )
        result = verdict_module.verdict(store, None, config, run_dir=tmp_path, ran={"SPEECH": RunState.ERRORED})
        assert result.file_verdict.triage is Triage.FLAG
        assert any("errored without a verdict" in reason.why for reason in result.file_verdict.reasons)

    def test_the_branches_map_joins_the_decision_to_the_verdict(
        self, make_verdict_store: Callable[..., ProvStore], config: TriageConfig, tmp_path: Path
    ) -> None:
        """A skipped branch carries the reason it was skipped, in the same record as the one that ran."""
        store = make_verdict_store(
            node_verdicts=[("ADMIT", Outcome.PASS, None), ("AIRWAY", Outcome.PASS, "airway")],
            kinds={"airway": "present", "speech": "absent", "voice": "absent"},
        )
        result = verdict_module.verdict(store, None, config, run_dir=tmp_path)
        branches = result.file_verdict.branches
        assert branches["AIRWAY"] == {
            "will_run": True,
            "forced_by_hint": False,
            "kind_state": "present",
            "verdict": "pass",
        }
        assert branches["SPEECH"]["will_run"] is False
        assert branches["SPEECH"]["verdict"] is None

    def test_a_run_with_no_routing_element_reads_no_branch_as_asked(
        self, make_verdict_store: Callable[..., ProvStore], config: TriageConfig, tmp_path: Path
    ) -> None:
        """ROUTING itself never ran, so no branch was asked and none is owed an answer."""
        store = make_verdict_store(
            node_verdicts=[("ADMIT", Outcome.PASS, None)],
            kinds={"airway": "present", "speech": "present", "voice": "absent"},
            route=False,
        )
        result = verdict_module.verdict(store, None, config, run_dir=tmp_path)
        assert result.file_verdict.branches == {}
        assert result.file_verdict.triage is Triage.PASS


class TestHintsAreReadThroughRoutingsMap:
    """The tag that forces a branch is the tag that can name a mismatch; one map, not two."""

    def test_a_declared_kind_no_branch_found_flags(
        self, make_verdict_store: Callable[..., ProvStore], tmp_path: Path
    ) -> None:
        """The declaration claimed a cough and AIRWAY found no labelled span."""
        hint = AudioHints(may_contain=["cough"])
        hint_config = _hint_config(tmp_path)
        store = make_verdict_store(
            node_verdicts=[("ADMIT", Outcome.PASS, None), ("AIRWAY", Outcome.FAIL, "airway")],
            kinds={"airway": "absent", "speech": "absent", "voice": "absent"},
            config=hint_config,
            hint=hint,
        )
        result = verdict_module.verdict(store, None, hint_config, hint, run_dir=tmp_path)
        assert result.file_verdict.hints["airway"] == "claimed_not_found"
        assert result.file_verdict.triage is Triage.FLAG

    def test_a_speech_type_value_is_a_claim_like_any_tag(
        self, make_verdict_store: Callable[..., ProvStore], tmp_path: Path
    ) -> None:
        """``speech_type`` goes through the same map, so the two cannot disagree about a tag."""
        hint = AudioHints(metadata={"speech_type": "read-speech"})
        hint_config = _hint_config(tmp_path)
        store = make_verdict_store(
            node_verdicts=[("ADMIT", Outcome.PASS, None), ("SPEECH", Outcome.FAIL, "speech")],
            kinds={"airway": "absent", "speech": "present", "voice": "absent"},
            config=hint_config,
            hint=hint,
        )
        result = verdict_module.verdict(store, None, hint_config, hint, run_dir=tmp_path)
        assert result.file_verdict.hints["speech"] == "claimed_not_found"

    def test_a_tag_the_map_does_not_cover_claims_nothing(
        self, make_verdict_store: Callable[..., ProvStore], config: TriageConfig, tmp_path: Path
    ) -> None:
        """The packaged map is null, so no tag reaches a kind and nothing is claimed."""
        hint = AudioHints(may_contain=["cough"])
        store = make_verdict_store(
            node_verdicts=[("ADMIT", Outcome.PASS, None), ("AIRWAY", Outcome.PASS, "airway")],
            kinds={"airway": "present", "speech": "absent", "voice": "absent"},
            hint=hint,
        )
        result = verdict_module.verdict(store, None, config, hint, run_dir=tmp_path)
        assert result.file_verdict.hints["airway"] == "found_unclaimed"
        assert result.file_verdict.triage is Triage.PASS

    def test_the_claim_is_read_off_the_decision_not_re_derived_from_the_config(
        self, make_verdict_store: Callable[..., ProvStore], config: TriageConfig, tmp_path: Path
    ) -> None:
        """ROUTING read the declaration with a map; VERDICT is handed one without it and must agree.

        The two nodes resolving the same tag independently is the divergence this reading removes:
        the claim is ROUTING's record of what it made of the hint, not a second opinion about it.
        """
        hint = AudioHints(may_contain=["cough"])
        store = make_verdict_store(
            node_verdicts=[("ADMIT", Outcome.PASS, None), ("AIRWAY", Outcome.FAIL, "airway")],
            kinds={"airway": "absent", "speech": "absent", "voice": "absent"},
            config=_hint_config(tmp_path),
            hint=hint,
        )
        result = verdict_module.verdict(store, None, config, hint, run_dir=tmp_path)
        assert result.file_verdict.hints["airway"] == "claimed_not_found"
        assert result.file_verdict.triage is Triage.FLAG

    def test_a_declaration_no_decision_survived_to_read_is_named_not_dropped(
        self, make_verdict_store: Callable[..., ProvStore], config: TriageConfig, tmp_path: Path
    ) -> None:
        """ROUTING errored, so what the declaration claimed is unknown; reading it as no claim is silent."""
        hint = AudioHints(may_contain=["cough"])
        store = make_verdict_store(
            node_verdicts=[("ADMIT", Outcome.PASS, None)],
            kinds={"airway": "absent", "speech": "absent", "voice": "absent"},
            route=False,
            hint=hint,
        )
        result = verdict_module.verdict(store, None, config, hint, run_dir=tmp_path)
        assert result.file_verdict.hints == {}
        assert result.file_verdict.triage is Triage.FLAG
        assert any(reason.why == UNREAD_DECLARATION for reason in result.file_verdict.reasons)
        assert _file_verdict_entity(store).attributes["hints"] == {}

    def test_no_declaration_and_no_decision_claims_nothing_without_flagging(
        self, make_verdict_store: Callable[..., ProvStore], config: TriageConfig, tmp_path: Path
    ) -> None:
        """With no hint there is nothing to have lost, so the empty claim map is the honest one."""
        store = make_verdict_store(
            node_verdicts=[("ADMIT", Outcome.PASS, None), ("AIRWAY", Outcome.PASS, "airway")],
            kinds={"airway": "present", "speech": "absent", "voice": "absent"},
            route=False,
        )
        result = verdict_module.verdict(store, None, config, run_dir=tmp_path)
        assert result.file_verdict.hints["airway"] == "found_unclaimed"
        assert result.file_verdict.triage is Triage.PASS

    def test_a_declaration_prevents_the_empty_discard(
        self, make_verdict_store: Callable[..., ProvStore], tmp_path: Path
    ) -> None:
        """Discarding a file the declaration says had a cough would delete the graph's own error."""
        hint = AudioHints(may_contain=["cough"])
        hint_config = _hint_config(tmp_path)
        store = make_verdict_store(
            node_verdicts=[("ADMIT", Outcome.PASS, None)],
            kinds={"airway": "absent", "speech": "absent", "voice": "absent"},
            config=hint_config,
            hint=hint,
        )
        result = verdict_module.verdict(store, None, hint_config, hint, run_dir=tmp_path)
        assert result.file_verdict.triage is Triage.FLAG
        assert result.file_verdict.discard_ground is None


class TestTheReleaseAxis:
    """REDACT's verdict, and nothing else, decides whether an artifact may be handed on."""

    def test_no_redact_verdict_is_not_assessed(
        self, make_verdict_store: Callable[..., ProvStore], config: TriageConfig, tmp_path: Path
    ) -> None:
        """A recording with no scan is unexamined, which must not read as cleared."""
        store = make_verdict_store(node_verdicts=BASE, kinds=KINDS)
        result = verdict_module.verdict(store, None, config, run_dir=tmp_path)
        assert result.file_verdict.release is Release.NOT_ASSESSED

    def test_a_fail_withholds_and_a_pass_releases(
        self, make_verdict_store: Callable[..., ProvStore], config: TriageConfig, tmp_path: Path
    ) -> None:
        """The two ends of the mapping, and the attribute the store records them in."""
        withheld = make_verdict_store(node_verdicts=[*BASE, ("REDACT", Outcome.FAIL, None)], kinds=KINDS)
        released = make_verdict_store(node_verdicts=[*BASE, ("REDACT", Outcome.PASS, None)], kinds=KINDS)
        assert verdict_module.verdict(withheld, None, config, run_dir=tmp_path).file_verdict.release is Release.WITHHELD
        result = verdict_module.verdict(released, None, config, run_dir=tmp_path)
        assert result.file_verdict.release is Release.RELEASABLE
        assert _file_verdict_entity(released).attributes["release"] == "releasable"

    def test_a_surviving_finding_does_not_move_the_triage_axis(
        self, make_verdict_store: Callable[..., ProvStore], config: TriageConfig, tmp_path: Path
    ) -> None:
        """A release problem is not a measurement problem, and it is in the same record regardless."""
        store = make_verdict_store(node_verdicts=[*BASE, ("REDACT", Outcome.FAIL, None)], kinds=KINDS)
        result = verdict_module.verdict(store, None, config, run_dir=tmp_path)
        assert result.file_verdict.triage is Triage.PASS
        assert result.file_verdict.release is Release.WITHHELD
        assert any(reason.node == "REDACT" for reason in result.file_verdict.reasons)

    def test_the_later_redact_verdict_governs(
        self, make_verdict_store: Callable[..., ProvStore], config: TriageConfig, tmp_path: Path
    ) -> None:
        """A repaired REDACT run wrote fail then pass; the release axis must read the repair."""
        store = make_verdict_store(
            node_verdicts=[*BASE, ("REDACT", Outcome.FAIL, None), ("REDACT", Outcome.PASS, None)], kinds=KINDS
        )
        result = verdict_module.verdict(store, None, config, run_dir=tmp_path)
        assert result.file_verdict.release is Release.RELEASABLE
        assert [r.outcome for r in result.file_verdict.reasons if r.node == "REDACT"] == [Outcome.PASS]


class TestWhatTheStoreRecords:
    """The written detail is verdict.md's product, and every id it folded is used."""

    def test_the_detail_is_the_product(
        self, make_verdict_store: Callable[..., ProvStore], config: TriageConfig, tmp_path: Path
    ) -> None:
        """T9 reads this entity and nothing else; a missing key there is a re-derivation."""
        store = make_verdict_store(node_verdicts=BASE, kinds=KINDS)
        result = verdict_module.verdict(store, None, config, run_dir=tmp_path)
        attributes = _file_verdict_entity(store).attributes
        assert {
            "triage",
            "release",
            "discard_ground",
            "reasons",
            "ran",
            "branches",
            "kinds",
            "screened",
            "agreement",
            "hints",
        } <= attributes.keys()
        assert attributes["kinds"] == result.file_verdict.kinds
        assert attributes["screened"] == result.file_verdict.screened
        assert attributes["agreement"] == result.file_verdict.agreement
        assert attributes["hints"] == result.file_verdict.hints
        assert attributes["branches"] == result.file_verdict.branches
        assert attributes["ran"] == {node: state.value for node, state in result.file_verdict.ran.items()}

    def test_reasons_carry_every_contribution(
        self, make_verdict_store: Callable[..., ProvStore], config: TriageConfig, tmp_path: Path
    ) -> None:
        """A flag naming one cause hides the others; every node's verdict appears in reasons."""
        store = make_verdict_store(
            node_verdicts=[
                ("ADMIT", Outcome.PASS, None),
                ("PREPROCESS", Outcome.PASS, None),
                ("TAXONOMY", Outcome.PASS, None),
                ("AIRWAY", Outcome.FAIL, "airway"),
                ("SPEECH", Outcome.PASS, "speech"),
            ],
            kinds=KINDS,
        )
        result = verdict_module.verdict(store, None, config, run_dir=tmp_path)
        assert {"ADMIT", "PREPROCESS", "TAXONOMY", "AIRWAY", "SPEECH"} <= {r.node for r in result.file_verdict.reasons}
        stored: list[dict[str, Any]] = _file_verdict_entity(store).attributes["reasons"]
        assert [r["node"] for r in stored] == [r.node for r in result.file_verdict.reasons]
        assert [r["outcome"] for r in stored] == [r.outcome.value for r in result.file_verdict.reasons]
        assert [r["why"] for r in stored] == [r.why for r in result.file_verdict.reasons]

    def test_every_folded_id_is_used_and_the_view_leads_with_the_file_verdict(
        self, make_verdict_store: Callable[..., ProvStore], config: TriageConfig, tmp_path: Path
    ) -> None:
        """Used edges to every node verdict, kind and branch decision; view = [file id, *folded ids]."""
        store = make_verdict_store(node_verdicts=BASE, kinds=KINDS)
        folded_ids = (
            {e.id for e in store.entities("verdict")}
            | {e.id for e in store.entities("kind")}
            | {e.id for e in store.entities("branch_decision")}
        )
        result = verdict_module.verdict(store, None, config, run_dir=tmp_path)

        file_entity = _file_verdict_entity(store)
        assert result.verdict_entity_id == file_entity.id
        assert result.view[0] == file_entity.id
        assert set(result.view[1:]) == folded_ids
        assert len(result.view) == len(set(result.view))

        activity_id = store.generated_by(file_entity.id)
        assert activity_id is not None
        activity = store.get_activity(activity_id)
        assert activity.node == "VERDICT"
        assert activity.parameters["config_hash"] == config.config_hash
        assert folded_ids <= set(store.uses_of(activity_id))
        agent_ids = store.associated_with(activity_id)
        assert agent_ids, "the software agent runs the fold"
        assert store.get_agent(agent_ids[0]).agent_type == "software"

    def test_the_file_verdict_is_not_folded_back_into_itself(
        self, make_verdict_store: Callable[..., ProvStore], config: TriageConfig, tmp_path: Path
    ) -> None:
        """Running twice over one store must not read the first file verdict as a node's."""
        store = make_verdict_store(node_verdicts=BASE, kinds=KINDS)
        first = verdict_module.verdict(store, None, config, run_dir=tmp_path)
        second = verdict_module.verdict(store, None, config, run_dir=tmp_path)
        assert second.file_verdict == first.file_verdict
        assert "VERDICT" not in {r.node for r in second.file_verdict.reasons}

    def test_an_invalidated_node_verdict_does_not_vote(
        self, make_verdict_store: Callable[..., ProvStore], config: TriageConfig, tmp_path: Path
    ) -> None:
        """An invalidated verdict is not a verdict; the branch that wrote it is owed an answer again."""
        store = make_verdict_store(
            node_verdicts=[("ADMIT", Outcome.PASS, None), ("SPEECH", Outcome.PASS, "speech")],
            kinds={"airway": "absent", "speech": "present", "voice": "absent"},
        )
        speech = next(e for e in store.entities("verdict") if e.attributes["node"] == "SPEECH")
        store.was_invalidated_by(speech.id, store.activity(node="SPEECH", step="withdraw", parameters={}))
        result = verdict_module.verdict(store, None, config, run_dir=tmp_path)
        assert "SPEECH" not in {r.node for r in result.file_verdict.reasons if r.outcome is not Outcome.FLAG}
        assert result.file_verdict.triage is Triage.FLAG
        assert speech.id not in result.view

    def test_a_superseded_verdict_is_replaced_not_added(
        self, make_verdict_store: Callable[..., ProvStore], config: TriageConfig, tmp_path: Path
    ) -> None:
        """Two verdicts from one node are one contribution, the latest."""
        store = make_verdict_store(
            node_verdicts=[
                ("ADMIT", Outcome.PASS, None),
                ("SPEECH", Outcome.FAIL, "speech"),
                ("SPEECH", Outcome.PASS, "speech"),
            ],
            kinds={"airway": "absent", "speech": "present", "voice": "absent"},
        )
        result = verdict_module.verdict(store, None, config, run_dir=tmp_path)
        speech_reasons = [
            r for r in result.file_verdict.reasons if r.node == "SPEECH" and r.outcome is not Outcome.FLAG
        ]
        assert len(speech_reasons) == 1
        assert speech_reasons[0].outcome is Outcome.PASS
        assert result.file_verdict.kinds["speech"] == "present"


class TestGraphOrderAndRan:
    """The node order is the runner's, and ``ran`` is merged, the runner's over the store's."""

    def test_routing_is_in_the_graph_order_the_runner_uses(self) -> None:
        """The casing is ``run.GRAPH_ORDER``'s: a name that sorts as unknown reports no run state."""
        assert "routing" in verdict_module._GRAPH_ORDER
        assert set(verdict_module._GRAPH_ORDER) <= set(GRAPH_ORDER)

    def test_node_verdicts_are_folded_in_graph_order(
        self, make_verdict_store: Callable[..., ProvStore], config: TriageConfig, tmp_path: Path
    ) -> None:
        """Seeded out of order, the reasons still read in graph order; unknown nodes come last."""
        store = make_verdict_store(
            node_verdicts=[
                ("REDACT", Outcome.PASS, None),
                ("SOMETHING_ELSE", Outcome.PASS, None),
                ("ADMIT", Outcome.PASS, None),
                ("TAXONOMY", Outcome.PASS, None),
            ],
            kinds={"airway": "absent", "speech": "absent", "voice": "absent"},
        )
        result = verdict_module.verdict(store, None, config, run_dir=tmp_path)
        seeded = [r.node for r in result.file_verdict.reasons if r.node != "VERDICT"]
        assert seeded == ["ADMIT", "TAXONOMY", "routing", "REDACT", "SOMETHING_ELSE"]

    def test_ran_is_derived_when_omitted_and_the_runners_wins_where_it_speaks(
        self, make_verdict_store: Callable[..., ProvStore], config: TriageConfig, tmp_path: Path
    ) -> None:
        """A verdict is completed, an activity without one is errored, neither is skipped."""
        store = make_verdict_store(node_verdicts=BASE, kinds=KINDS)
        derived = verdict_module.verdict(store, None, config, run_dir=tmp_path)
        assert derived.file_verdict.ran["ADMIT"] is RunState.COMPLETED
        assert derived.file_verdict.ran["routing"] is RunState.COMPLETED
        assert derived.file_verdict.ran["REDACT"] is RunState.SKIPPED

        supplied = verdict_module.verdict(
            make_verdict_store(node_verdicts=BASE, kinds=KINDS),
            None,
            config,
            run_dir=tmp_path,
            ran={"REDACT": RunState.ERRORED},
        )
        assert supplied.file_verdict.ran["REDACT"] is RunState.ERRORED
        assert supplied.file_verdict.ran["ADMIT"] is RunState.COMPLETED

    def test_a_node_that_ran_and_left_no_verdict_is_errored_not_skipped(
        self, make_verdict_store: Callable[..., ProvStore], config: TriageConfig, tmp_path: Path
    ) -> None:
        """An activity with no verdict is the raising node's signature; neither is never having run."""
        store = make_verdict_store(
            node_verdicts=[("ADMIT", Outcome.PASS, None), ("TAXONOMY", Outcome.PASS, None)],
            kinds={"airway": "absent", "speech": "present", "voice": "absent"},
        )
        store.was_associated_with(
            store.activity(node="SPEECH", step="transcript", parameters={}), software_agent(store)
        )
        result = verdict_module.verdict(store, None, config, run_dir=tmp_path)
        assert result.file_verdict.ran["SPEECH"] is RunState.ERRORED
        assert result.file_verdict.ran["VOICE"] is RunState.SKIPPED
        assert any("errored without a verdict" in r.why for r in result.file_verdict.reasons)
        assert _file_verdict_entity(store).attributes["ran"]["SPEECH"] == "errored"


class TestTheFoldIsWiredNotReimplemented:
    """The node maps store facts onto the fold's inputs and does not decide anything itself."""

    def test_the_fold_is_called_and_no_table_lives_here(self) -> None:
        """The kind-to-branch table is ROUTING's, and the gate question is answered by its elements."""
        source = inspect.getsource(verdict_module)
        assert "fold_file_verdict(" in source
        assert "_BRANCH_FOR_KIND" not in source
        assert "_is_gated" not in source

    def test_no_sibling_node_is_imported(self) -> None:
        """A node calling into another node's module couples two nodes outside the store."""
        source = inspect.getsource(verdict_module)
        assert "workflows.triage.nodes.routing" not in source
