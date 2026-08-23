"""VERDICT wires the store into the existing fold. Nothing here loads a model."""

import inspect
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import pytest

from senselab.audio.workflows.triage.config import TriageConfig
from senselab.audio.workflows.triage.nodes import verdict as verdict_module
from senselab.audio.workflows.triage.nodes.common import software_agent, write_verdict
from senselab.audio.workflows.triage.vocabulary import KindState, Outcome, Release, RunState
from senselab.utils.prov_store import Entity, ProvStore

BASE: tuple[tuple[str, Outcome, str | None], ...] = (
    ("ADMIT", Outcome.PASS, None),
    ("TAXONOMY", Outcome.PASS, None),
    ("AIRWAY", Outcome.PASS, "airway"),
    ("SPEECH", Outcome.PASS, "speech"),
)
KINDS = {"airway": "present", "speech": "present"}


@pytest.fixture
def make_verdict_store() -> Callable[..., ProvStore]:
    """A builder writing node-verdict and kind entities directly; no branch node runs."""

    def _make(
        *,
        node_verdicts: Sequence[tuple[str, Outcome, str | None]] = (),
        kinds: Mapping[str, str] | None = None,
    ) -> ProvStore:
        store = ProvStore(run_id="verdict-test")
        agent = software_agent(store)
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
        taxonomy = store.activity(node="TAXONOMY", step="seed-kinds", parameters={})
        store.was_associated_with(taxonomy, agent)
        for kind_name, state in (kinds or {}).items():
            kind_id = store.entity(
                prov_type="kind", extent=None, attributes={"kind": kind_name, "state": state, "families": {}}
            )
            store.was_generated_by(kind_id, taxonomy)
        return store

    return _make


def _file_verdict_entity(store: ProvStore) -> Entity:
    """The verdict entity whose node attribute is VERDICT — the file verdict, not a node's."""
    found = [e for e in store.entities("verdict") if e.attributes["node"] == "VERDICT"]
    assert len(found) == 1, f"expected exactly one file verdict, found {len(found)}"
    return found[0]


def test_a_branch_fail_against_an_absent_kind_is_a_file_pass(
    make_verdict_store: Callable[..., ProvStore], config: TriageConfig, tmp_path: Path
) -> None:
    """A cough recording: airway present+pass, speech absent+fail, voice absent-by-resolution."""
    store = make_verdict_store(
        node_verdicts=[
            ("ADMIT", Outcome.PASS, None),
            ("TAXONOMY", Outcome.PASS, None),
            ("AIRWAY", Outcome.PASS, "airway"),
            ("SPEECH", Outcome.FAIL, "speech"),
            ("VOICE", Outcome.FAIL, "voice_no_words"),
        ],
        kinds={"airway": "present", "speech": "absent", "voice_no_words": "not_screened"},
    )
    result = verdict_module.verdict(store, None, config, run_dir=tmp_path)
    assert result.file_verdict.triage is Outcome.PASS, "a branch fail is not a file fail"
    assert result.file_verdict.kinds["voice_no_words"] is KindState.ABSENT, (
        "not_screened maps to UNDECIDED (N27), which VOICE's fail resolves to absent"
    )


def test_admit_fail_and_every_kind_absent_are_distinct_fails(
    make_verdict_store: Callable[..., ProvStore], config: TriageConfig, tmp_path: Path
) -> None:
    """Could-not-measure and measured-and-empty carry different reasons, in different shapes."""
    broken = verdict_module.verdict(
        make_verdict_store(node_verdicts=[("ADMIT", Outcome.FAIL, None)], kinds={}), None, config, run_dir=tmp_path
    )
    empty = verdict_module.verdict(
        make_verdict_store(
            node_verdicts=[("ADMIT", Outcome.PASS, None), ("TAXONOMY", Outcome.FAIL, None)],
            kinds={"airway": "absent", "speech": "absent", "voice_no_words": "absent"},
        ),
        None,
        config,
        run_dir=tmp_path,
    )
    assert broken.file_verdict.triage is empty.file_verdict.triage is Outcome.FAIL
    assert broken.file_verdict.reasons[0].node == "ADMIT"
    assert any("every kind is absent" in r.why for r in empty.file_verdict.reasons)
    assert not any("every kind is absent" in r.why for r in broken.file_verdict.reasons)


def test_release_mapping_and_not_assessed_is_not_releasable(
    make_verdict_store: Callable[..., ProvStore], config: TriageConfig, tmp_path: Path
) -> None:
    """No REDACT verdict -> NOT_ASSESSED; fail -> WITHHELD; pass -> RELEASABLE. Never a default."""
    none_ran = verdict_module.verdict(
        make_verdict_store(node_verdicts=BASE, kinds=KINDS), None, config, run_dir=tmp_path
    )
    assert none_ran.file_verdict.release is Release.NOT_ASSESSED
    assert none_ran.file_verdict.release is not Release.RELEASABLE, (
        "a recording with no speech has no scan; unexamined must not read as cleared"
    )
    withheld = verdict_module.verdict(
        make_verdict_store(node_verdicts=[*BASE, ("REDACT", Outcome.FAIL, None)], kinds=KINDS),
        None,
        config,
        run_dir=tmp_path,
    )
    assert withheld.file_verdict.release is Release.WITHHELD
    released = verdict_module.verdict(
        make_verdict_store(node_verdicts=[*BASE, ("REDACT", Outcome.PASS, None)], kinds=KINDS),
        None,
        config,
        run_dir=tmp_path,
    )
    assert released.file_verdict.release is Release.RELEASABLE


def test_the_store_never_records_the_release_as_a_pass(
    make_verdict_store: Callable[..., ProvStore], config: TriageConfig, tmp_path: Path
) -> None:
    """The release axis lives in its own attribute; the outcome attribute stays the triage axis."""
    store = make_verdict_store(node_verdicts=[*BASE, ("REDACT", Outcome.PASS, None)], kinds=KINDS)
    result = verdict_module.verdict(store, None, config, run_dir=tmp_path)
    entity = _file_verdict_entity(store)
    assert entity.attributes["outcome"] == result.file_verdict.triage.value
    assert entity.attributes["release"] == Release.RELEASABLE.value
    assert entity.attributes["triage"] == result.file_verdict.triage.value


def test_contradiction_wiring_resolves_the_kind_and_flags(
    make_verdict_store: Callable[..., ProvStore], config: TriageConfig, tmp_path: Path
) -> None:
    """Absent-predicted kind whose branch passed -> flag, kind resolved present, both visible."""
    store = make_verdict_store(
        node_verdicts=[("ADMIT", Outcome.PASS, None), ("SPEECH", Outcome.PASS, "speech")],
        kinds={"speech": "absent"},
    )
    result = verdict_module.verdict(store, None, config, run_dir=tmp_path)
    assert result.file_verdict.triage is Outcome.FLAG
    assert result.file_verdict.kinds["speech"] is KindState.PRESENT
    kind_entities = store.entities("kind")
    assert kind_entities[0].attributes["state"] == "absent", (
        "TAXONOMY's assertion stays in the store; the resolution is this node's, and both remain"
    )


def test_a_present_kind_whose_branch_never_ran_flags(
    make_verdict_store: Callable[..., ProvStore], config: TriageConfig, tmp_path: Path
) -> None:
    """The absence of evidence, on a kind the graph was asked about, is a gap a human sees."""
    result = verdict_module.verdict(
        make_verdict_store(node_verdicts=[("ADMIT", Outcome.PASS, None)], kinds={"airway": "present"}),
        None,
        config,
        run_dir=tmp_path,
        ran={"ADMIT": RunState.COMPLETED, "AIRWAY": RunState.SKIPPED},
    )
    assert result.file_verdict.triage is Outcome.FLAG
    assert result.file_verdict.ran["AIRWAY"] is RunState.SKIPPED, "the caller's ran is recorded, not re-derived"


def test_ran_is_derived_when_omitted_and_cannot_see_errored(
    make_verdict_store: Callable[..., ProvStore], config: TriageConfig, tmp_path: Path
) -> None:
    """Verdict entity -> completed, none -> skipped; the docstring states the errored blindness (N26)."""
    result = verdict_module.verdict(make_verdict_store(node_verdicts=BASE, kinds=KINDS), None, config, run_dir=tmp_path)
    assert result.file_verdict.ran["ADMIT"] is RunState.COMPLETED
    assert result.file_verdict.ran["REDACT"] is RunState.SKIPPED
    assert RunState.ERRORED not in result.file_verdict.ran.values()
    assert "errored" in (verdict_module.verdict.__doc__ or ""), (
        "the derived fallback's blindness to errored (N26) is stated where the caller reads it"
    )


def test_gated_run_is_marked(
    make_verdict_store: Callable[..., ProvStore], config: TriageConfig, tmp_path: Path
) -> None:
    """An absent kind with no branch verdict marks gated: the contradiction check did not happen."""
    store = make_verdict_store(node_verdicts=[("ADMIT", Outcome.PASS, None)], kinds={"speech": "absent"})
    verdict_module.verdict(store, None, config, run_dir=tmp_path)
    file_entity = _file_verdict_entity(store)
    assert file_entity.attributes["gated"] is True


def test_an_ungated_run_is_not_marked_gated(
    make_verdict_store: Callable[..., ProvStore], config: TriageConfig, tmp_path: Path
) -> None:
    """Every kind that was screened had its branch report; nothing was skipped over."""
    store = make_verdict_store(node_verdicts=BASE, kinds=KINDS)
    verdict_module.verdict(store, None, config, run_dir=tmp_path)
    assert _file_verdict_entity(store).attributes["gated"] is False


def test_no_kind_entities_records_an_unscreened_run(
    make_verdict_store: Callable[..., ProvStore], config: TriageConfig, tmp_path: Path
) -> None:
    """TAXONOMY absent means no predictions to fold against, which the entity says out loud."""
    store = make_verdict_store(node_verdicts=[("ADMIT", Outcome.PASS, None)], kinds={})
    result = verdict_module.verdict(store, None, config, run_dir=tmp_path)
    assert result.file_verdict.kinds == {}
    assert _file_verdict_entity(store).attributes["screened"] is False
    assert result.file_verdict.triage is Outcome.PASS, "no kinds is not every kind absent"


def test_the_fold_is_wired_not_reimplemented() -> None:
    """The node's module calls fold_file_verdict; no second fold lives here."""
    src = inspect.getsource(verdict_module)
    assert "fold_file_verdict(" in src
    assert "_BRANCH_FOR_KIND" not in src, "the kind->branch table stays in vocabulary.py"


def test_every_folded_id_is_used_and_the_view_leads_with_the_file_verdict(
    make_verdict_store: Callable[..., ProvStore], config: TriageConfig, tmp_path: Path
) -> None:
    """Used edges to every node-verdict and kind entity; view = [file id, *folded ids]."""
    store = make_verdict_store(node_verdicts=BASE, kinds=KINDS)
    node_verdict_ids = {e.id for e in store.entities("verdict")}
    kind_ids = {e.id for e in store.entities("kind")}
    result = verdict_module.verdict(store, None, config, run_dir=tmp_path)

    file_entity = _file_verdict_entity(store)
    assert result.verdict_entity_id == file_entity.id
    assert result.view[0] == file_entity.id
    assert set(result.view[1:]) == node_verdict_ids | kind_ids
    assert len(result.view) == len(set(result.view))

    activity_id = store.generated_by(file_entity.id)
    assert activity_id is not None
    activity = store.get_activity(activity_id)
    assert activity.node == "VERDICT"
    assert activity.parameters["config_hash"] == config.config_hash
    assert node_verdict_ids | kind_ids <= set(store.uses_of(activity_id))
    agent_ids = store.associated_with(activity_id)
    assert agent_ids, "the software agent runs the fold"
    assert store.get_agent(agent_ids[0]).agent_type == "software"


def test_reasons_carry_every_contribution(
    make_verdict_store: Callable[..., ProvStore], config: TriageConfig, tmp_path: Path
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
        kinds={"airway": "present", "speech": "present"},
    )
    result = verdict_module.verdict(store, None, config, run_dir=tmp_path)
    assert result.file_verdict.triage is Outcome.FLAG
    assert {"ADMIT", "PREPROCESS", "TAXONOMY", "AIRWAY", "SPEECH"} <= {r.node for r in result.file_verdict.reasons}
    assert sum(1 for r in result.file_verdict.reasons if r.outcome is Outcome.FLAG) == 1

    stored: list[dict[str, Any]] = _file_verdict_entity(store).attributes["reasons"]
    assert [r["node"] for r in stored] == [r.node for r in result.file_verdict.reasons]
    assert [r["outcome"] for r in stored] == [r.outcome.value for r in result.file_verdict.reasons]
    assert [r["why"] for r in stored] == [r.why for r in result.file_verdict.reasons]


def test_node_verdicts_are_folded_in_graph_order(
    make_verdict_store: Callable[..., ProvStore], config: TriageConfig, tmp_path: Path
) -> None:
    """Seeded out of order, the reasons still read in graph order; unknown nodes come last."""
    store = make_verdict_store(
        node_verdicts=[
            ("REDACT", Outcome.PASS, None),
            ("SOMETHING_ELSE", Outcome.PASS, None),
            ("ADMIT", Outcome.PASS, None),
            ("TAXONOMY", Outcome.PASS, None),
        ],
        kinds={},
    )
    result = verdict_module.verdict(store, None, config, run_dir=tmp_path)
    seeded = [r.node for r in result.file_verdict.reasons]
    assert seeded == ["ADMIT", "TAXONOMY", "REDACT", "SOMETHING_ELSE"]


def test_the_file_verdict_is_not_folded_back_into_itself(
    make_verdict_store: Callable[..., ProvStore], config: TriageConfig, tmp_path: Path
) -> None:
    """Running twice over one store must not read the first file verdict as a node's."""
    store = make_verdict_store(node_verdicts=BASE, kinds=KINDS)
    first = verdict_module.verdict(store, None, config, run_dir=tmp_path)
    second = verdict_module.verdict(store, None, config, run_dir=tmp_path)
    assert second.file_verdict == first.file_verdict
    assert "VERDICT" not in {r.node for r in second.file_verdict.reasons}


def test_the_callers_ran_is_used_where_it_disagrees_with_the_store(
    make_verdict_store: Callable[..., ProvStore], config: TriageConfig, tmp_path: Path
) -> None:
    """Only the runner knows a branch ran and errored; the store would call that never having run."""
    store = make_verdict_store(node_verdicts=[("ADMIT", Outcome.PASS, None)], kinds={"airway": "present"})
    supplied = verdict_module.verdict(
        store,
        None,
        config,
        run_dir=tmp_path,
        ran={"ADMIT": RunState.COMPLETED, "AIRWAY": RunState.ERRORED, "SPEECH": RunState.COMPLETED},
    )
    derived = verdict_module.verdict(
        make_verdict_store(node_verdicts=[("ADMIT", Outcome.PASS, None)], kinds={"airway": "present"}),
        None,
        config,
        run_dir=tmp_path,
    )
    assert supplied.file_verdict.ran["AIRWAY"] is RunState.ERRORED
    assert derived.file_verdict.ran["AIRWAY"] is RunState.SKIPPED
    assert supplied.file_verdict.ran.keys() == {"ADMIT", "AIRWAY", "SPEECH"}, (
        "the caller's ran is recorded as given, not widened to the graph"
    )
    assert _file_verdict_entity(store).attributes["ran"]["AIRWAY"] == RunState.ERRORED.value


def test_a_completed_branch_with_no_verdict_reads_differently_from_one_that_never_ran(
    make_verdict_store: Callable[..., ProvStore], config: TriageConfig, tmp_path: Path
) -> None:
    """Both flag, but the reason distinguishes a silent completion from a branch the runner skipped."""
    completed = verdict_module.verdict(
        make_verdict_store(node_verdicts=[("ADMIT", Outcome.PASS, None)], kinds={"airway": "present"}),
        None,
        config,
        run_dir=tmp_path,
        ran={"ADMIT": RunState.COMPLETED, "AIRWAY": RunState.COMPLETED},
    )
    skipped = verdict_module.verdict(
        make_verdict_store(node_verdicts=[("ADMIT", Outcome.PASS, None)], kinds={"airway": "present"}),
        None,
        config,
        run_dir=tmp_path,
    )
    assert completed.file_verdict.triage is skipped.file_verdict.triage is Outcome.FLAG
    assert any("completed without a verdict" in r.why for r in completed.file_verdict.reasons)
    assert any("never ran" in r.why for r in skipped.file_verdict.reasons)
