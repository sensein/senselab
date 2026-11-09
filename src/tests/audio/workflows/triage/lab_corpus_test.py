"""The lab corpus: a made-up dataset, one invented task per kind of task triage reads (plan step 8c).

Its profile (``profiles/lab_corpus``) is written from senselab's template, builds on the reference profile and takes
its corpus statistics from a ``fitted.yaml``. Every recording has a plain name and declares its identity in its
hint, so these tests run with no file, task or value from the fixture profile. The kinds ``effort`` and
``per_sentence`` are left out: no branch has a matcher for them. A second word-list task takes its category from
each recording's hint. Past the start check, ADMIT and the gates: SPEECH's matcher per kind, the declared words
REDACT and SPEECH hold a transcript against, VERDICT's task evidence and fold over prepared readings, a task
described inline in the hint, and a run with stages off.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np
import pytest
import soundfile
import yaml  # type: ignore[import-untyped]

from senselab.audio.data_structures import AudioHints, ExpectedSpeech
from senselab.audio.workflows.triage.breath_pattern import BreathPattern
from senselab.audio.workflows.triage.config import TriageConfig, load_triage_config
from senselab.audio.workflows.triage.cough_pattern import CoughPattern
from senselab.audio.workflows.triage.dataset_profile import (
    deactivate_profile,
    missing_dataset_values,
    unfitted_values,
)
from senselab.audio.workflows.triage.ddk_task import CYCLE, SYLLABLE
from senselab.audio.workflows.triage.identity import recorded_identity
from senselab.audio.workflows.triage.nodes import verdict as verdict_module
from senselab.audio.workflows.triage.nodes.admit import admit
from senselab.audio.workflows.triage.nodes.airway_task import (
    BREATH_READING,
    COUGH_READING,
    breath_attributes,
    cough_attributes,
)
from senselab.audio.workflows.triage.nodes.branches import (
    CATEGORY_FROM_HINT,
    EXPECTATIONS,
    UNDETERMINED,
    Expectation,
    Finding,
    Pattern,
    Result,
    branch_params,
    declared_carrier,
    dispatch,
    item_category_of,
    mode_of,
)
from senselab.audio.workflows.triage.nodes.common import software_agent, write_measurement
from senselab.audio.workflows.triage.nodes.gates import conformance_gate_names, load_gate_bounds
from senselab.audio.workflows.triage.nodes.redact import _expected_units
from senselab.audio.workflows.triage.nodes.speech import align_speech, declared_content, detect_speech
from senselab.audio.workflows.triage.run import run_triage
from senselab.audio.workflows.triage.task_catalogue import (
    FROM_HINT_DESCRIPTION,
    FROM_TASK_LIST,
    recorded_task,
    task_list,
)
from senselab.audio.workflows.triage.task_events import TaskEvent
from senselab.audio.workflows.triage.task_events import TaskEvidence as EventEvidence
from senselab.audio.workflows.triage.vocabulary import (
    BREATH_SUSTAINED,
    COUGH_COUNTED,
    EMPTY,
    TASK,
    TOO_SHORT_FOR_TASK,
    BranchDecision,
    BranchReport,
    NodeVerdict,
    Outcome,
    RunState,
    Triage,
    fold_file_verdict,
)
from senselab.audio.workflows.triage.workflow import OFF_IN_WORKFLOW
from senselab.utils.prov_store import ProvStore
from tests.audio.workflows.triage.conftest import declared_identity_attributes, hint_for, record_task
from tests.audio.workflows.triage.nodes.conftest import gated_conformance
from tests.audio.workflows.triage.run_test import graph  # noqa: F401 -- the runner's fake graph, used as a fixture
from tests.conftest import FIXTURE_PROFILE

LAB_CORPUS = Path(__file__).parent / "profiles" / "lab_corpus"
"""The made-up dataset's profile."""

KINDS = {
    "hold-a-tone": ("VOICE", "sustained"),
    "slide-up": ("VOICE", "glide"),
    "read-the-card": ("SPEECH", "ordered_tokens"),
    "tell-a-story": ("SPEECH", "free_response"),
    "name-some-fruit": ("SPEECH", "item_list"),
    "name-from-the-card": ("SPEECH", "item_list"),
    "pa-pa-pa": ("SPEECH", "syllable_train"),
    "pa-ta-ka": ("SPEECH", "syllable_sequence"),
    "three-coughs": ("AIRWAY", "event_series"),
    "cough-then-breathe": ("AIRWAY", "event_alternation"),
    "quiet-breathing": ("AIRWAY", "sound_coverage"),
}
"""Each invented task, its branch and its kind (the task list's ``expects.pattern``).

``name-from-the-card`` is the second word-list task: its entry lists categories and fixes none, so each recording's
hint declares which one it asks for."""


@pytest.fixture
def lab_corpus(without_triage_profile: None) -> Iterator[TriageConfig]:
    """The lab corpus's profile, active for the test, and no profile active after it."""
    config = load_triage_config(profile=LAB_CORPUS)
    yield config
    deactivate_profile()


def _admitted(config: TriageConfig, wav: Path, task: str) -> ProvStore:
    """A plainly named recording through the real ADMIT node, its identity declared in its hint."""
    store = ProvStore(run_id=wav.stem)
    t = np.arange(16000) / 16000
    soundfile.write(wav, (0.5 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32), 16000)
    result = admit(store, wav, config, hint_for("p07", "s2", task), run_dir=wav.parent)
    assert result.verdict.outcome is Outcome.PASS
    return store


def test_its_profile_is_all_a_run_needs(lab_corpus: TriageConfig) -> None:
    """The start check finds nothing missing and nothing unfitted; the task list is the lab's ten tasks."""
    assert missing_dataset_values(lab_corpus.values) == []
    assert unfitted_values() == []
    assert set(task_list()) == set(KINDS)


def test_it_reads_no_file_of_the_fixture_profile(lab_corpus: TriageConfig) -> None:
    """No task of the fixture's is the lab's, and the active profile chain does not include the fixture."""
    fixture_tasks = set((yaml.safe_load((FIXTURE_PROFILE / "tasks.yaml").read_text()) or {})["tasks"])
    assert not fixture_tasks & set(task_list())
    assert lab_corpus.profile is not None
    assert all(layer.root != FIXTURE_PROFILE.resolve() for layer in lab_corpus.profile.chain())
    fixture_files = {path.read_bytes() for path in FIXTURE_PROFILE.rglob("*") if path.is_file()}
    copied = [
        str(path.relative_to(LAB_CORPUS))
        for path in LAB_CORPUS.rglob("*")
        if path.is_file() and path.read_bytes() in fixture_files
    ]
    assert copied == [], f"copied from the fixture: {copied}"


@pytest.mark.parametrize("task", sorted(KINDS))
def test_each_kind_is_admitted_with_its_declared_identity(lab_corpus: TriageConfig, tmp_path: Path, task: str) -> None:
    """ADMIT records the hint's participant, session and task from a plain file name."""
    store = _admitted(lab_corpus, tmp_path / "rec-0001.wav", task)
    identity = recorded_identity(store)
    assert identity is not None and (identity.participant, identity.session, identity.task_id) == ("p07", "s2", task)
    recorded = recorded_task(store)
    assert recorded is not None and (recorded.name, recorded.source, recorded.branch) == (
        task,
        FROM_TASK_LIST,
        KINDS[task][0],
    )


@pytest.mark.parametrize("task", sorted(KINDS))
def test_each_kind_is_read_by_its_branchs_matcher(lab_corpus: TriageConfig, tmp_path: Path, task: str) -> None:
    """The owning branch aligns the recording with the task's matcher; the other branches detect."""
    store = _admitted(lab_corpus, tmp_path / "rec-0002.wav", task)
    owner, kind = KINDS[task]
    assert EXPECTATIONS[owner][task].pattern.value == kind
    params = branch_params(lab_corpus)
    for branch in ("AIRWAY", "SPEECH", "VOICE"):
        ran: list[str] = []
        dispatch(
            branch,
            store,
            params,
            hint_for("p07", "s2", task),
            align=lambda family, *_: ran.append(f"align {family}") or Result([], []),
            detect=lambda *_: ran.append("detect") or Result([], []),
        )
        assert ran == ([f"align {task}"] if branch == owner else ["detect"]), branch


@pytest.mark.parametrize("task", sorted(KINDS))
def test_each_kinds_gates_resolve(lab_corpus: TriageConfig, task: str) -> None:
    """Every gate the task's kind reads as its conformance has a bound, from the reference profile's layers."""
    owner, _ = KINDS[task]
    pattern = EXPECTATIONS[owner][task].pattern
    bounds = load_gate_bounds(lab_corpus, pattern, task)
    missing = [name for name in conformance_gate_names(pattern) if bounds.bounds.get(name) is None]
    assert missing == [], f"{task}: no bound for {missing}"


# --------------------------------------------------------------------- prepared readings

CARD = "Turn the card over. Then read it."
"""What ``read-the-card``'s recordings are asked to read, as their hint declares it."""


def _seeded(task: str, *, duration_s: float = 20.0, **recorded: Any) -> ProvStore:  # noqa: ANN401
    """A plainly named recording whose identity and task are recorded as ADMIT records them.

    Args:
        task: The task the hint names.
        duration_s: The recording's duration.
        **recorded: What else the hint declares (``item_category``, an inline ``description``).

    Returns:
        The store.
    """
    store = ProvStore(run_id="lab-corpus-test")
    store.entity(
        prov_type="stream",
        extent=(0.0, duration_s),
        attributes={
            "name": "recording",
            "path": "rec-0003.wav",
            "sampling_rate": 16000,
            "channels": 1,
            **declared_identity_attributes("p07", "s2", task),
        },
    )
    record_task(store, task, **recorded)
    return store


def _said(store: ProvStore, words: list[str], *, start_s: float = 1.0) -> None:
    """Write the consensus words, one a second from ``start_s``, and the measurement naming them."""
    ids = [
        store.entity(
            prov_type="word",
            extent=(start_s + index, start_s + index + 0.5),
            attributes={"text": text, "index": index, "bracketed": False, "agreement": 1.0, "variants": []},
        )
        for index, text in enumerate(words)
    ]
    store.entity(
        prov_type="measurement",
        extent=None,
        attributes={
            "name": "consensus_transcript",
            "signal": "plain",
            "text": " ".join(words),
            "word_ids": ids,
            "sources": [{"name": "whisper"}],
        },
    )


def _found(result: Result, kind: str, name: str) -> list[Finding]:
    """Every finding of one kind and name a mode returned."""
    return [finding for finding in result.deviations if finding.kind == kind and finding.name == name]


def _speech(config: TriageConfig, store: ProvStore, hint: AudioHints | None) -> Result:
    """SPEECH's mode decision over the store, with the real matchers."""
    return dispatch("SPEECH", store, branch_params(config), hint, align=align_speech, detect=detect_speech)


def _absent_instruments(result: Result) -> list[tuple[str, Any]]:
    """The measures a mode could not take, with the instrument each lacked."""
    return [
        (finding.name, finding.evidence.get("unavailable"))
        for finding in result.deviations
        if finding.kind == "measure" and finding.evidence.get("unavailable")
    ]


class TestSpeechReadsEachKindWithItsMatcher:
    """SPEECH's in-family matcher, per kind, over a prepared transcript."""

    def test_a_read_card_is_matched_token_by_token(self, lab_corpus: TriageConfig) -> None:
        """Every declared word read: no substitution, no omission, a task extent, and the gates pass."""
        store = _seeded("read-the-card")
        _said(store, CARD.split())
        store.entity(
            prov_type="measurement",
            extent=None,
            attributes={"name": "stimulus_alignment", "signal": "plain", "path": "derivatives/stimulus_alignment.npz"},
        )
        hint = hint_for("p07", "s2", "read-the-card", expected_speech=[ExpectedSpeech(text=CARD)])
        result = _speech(lab_corpus, store, hint)
        assert _found(result, "deviation", "stimulus_mismatch") == []
        assert _found(result, "deviation", "omission") == []
        assert "task_extent" in [proposal.role for proposal in result.components]
        assert gated_conformance(result, Pattern.ORDERED_TOKENS, settings=lab_corpus) is True

    def test_a_story_is_read_as_a_free_response(self, lab_corpus: TriageConfig) -> None:
        """The free-response body measures the speech rate over every word the participant said."""
        store = _seeded("tell-a-story")
        _said(store, [f"word{index}" for index in range(9)])
        for start in (1.0, 4.0, 7.0):
            store.entity(
                prov_type="span",
                extent=(start, start + 2.6),
                attributes={"signal": "plain", "measure": "asr", "merged_proposals": 1},
            )
        result = _speech(lab_corpus, store, hint_for("p07", "s2", "tell-a-story"))
        [rate] = _found(result, "measure", "speech_rate_from_consensus_words_per_s")
        assert rate.evidence["support_words"] == 9

    def test_a_fixed_category_counts_its_own_items(self, lab_corpus: TriageConfig) -> None:
        """``name-some-fruit`` fixes its category; its word list admits two of the three items."""
        store = _seeded("name-some-fruit")
        _said(store, ["apple", "pear", "chair"])
        result = _speech(lab_corpus, store, hint_for("p07", "s2", "name-some-fruit"))
        assert _found(result, "count", "item_category")[0].evidence["found"] == "Fruit"
        assert _found(result, "count", "category_items")[0].evidence["found"] == 2

    @pytest.mark.parametrize(
        ("category", "items"), [("Fruit", ["apple", "pear", "chair"]), ("Letters", ["a", "b", "apple"])]
    )
    def test_a_category_the_hint_declares_is_the_one_counted(
        self, lab_corpus: TriageConfig, category: str, items: list[str]
    ) -> None:
        """A task whose entry lists categories and fixes none reads the recording's own, with no repetition rule."""
        hint = hint_for("p07", "s2", "name-from-the-card", item_category=category)
        assert item_category_of("name-from-the-card", hint) == (category, CATEGORY_FROM_HINT)
        store = _seeded("name-from-the-card", item_category=category)
        _said(store, items)
        result = _speech(lab_corpus, store, hint)
        assert _found(result, "count", "item_category")[0].evidence["found"] == category
        assert _found(result, "count", "item_category_from")[0].evidence["found"] == CATEGORY_FROM_HINT
        assert _found(result, "count", "category_items")[0].evidence["found"] == 2

    @pytest.mark.parametrize("task", ["pa-pa-pa", "pa-ta-ka"])
    def test_a_syllable_task_takes_the_syllable_body(self, lab_corpus: TriageConfig, task: str) -> None:
        """With no envelope the syllable body names the instrument it lacked, and the gates cannot answer."""
        store = _seeded(task)
        _said(store, [])
        result = _speech(lab_corpus, store, hint_for("p07", "s2", task))
        assert ("ddk_syllable_rate_from_envelope_modulation_hz", "energy_envelope") in _absent_instruments(result)
        assert gated_conformance(result, EXPECTATIONS["SPEECH"][task].pattern, settings=lab_corpus) == UNDETERMINED


class TestTheDeclaredWordsATranscriptIsHeldAgainst:
    """What REDACT and SPEECH take a faithful performance to contain: the prompts, and the carrier a task declares."""

    @pytest.mark.parametrize(("task", "carrier"), [("pa-pa-pa", "pa"), ("pa-ta-ka", "pataka")])
    def test_a_syllable_task_names_the_carrier_its_entry_declares(
        self, lab_corpus: TriageConfig, task: str, carrier: str
    ) -> None:
        """``pa-ta-ka`` spells "pataka": its name's last part ("ka") is not what a transcript of it holds."""
        assert declared_carrier(task) == carrier
        assert declared_content(None, task) == (carrier,)
        assert _expected_units(None, task, terminators=".") == [(-1, carrier, [carrier])]

    def test_a_task_that_is_not_syllables_declares_no_carrier(self, lab_corpus: TriageConfig) -> None:
        """A word-list task's name is not a carrier either."""
        assert declared_carrier("name-some-fruit") is None
        assert declared_content(None, "name-some-fruit") is None

    def test_the_card_is_split_into_its_sentences(self, lab_corpus: TriageConfig) -> None:
        """REDACT places a covered run inside one declared sentence of the card."""
        hint = hint_for("p07", "s2", "read-the-card", expected_speech=[ExpectedSpeech(text=CARD)])
        units = _expected_units(hint, "read-the-card", terminators=".")
        assert [(prompt, text) for prompt, text, _ in units] == [(0, "Turn the card over."), (0, "Then read it.")]


def _reading(store: ProvStore, name: str, attributes: dict[str, Any]) -> None:
    """Write one of AIRWAY's task readings, as AIRWAY writes it."""
    activity = store.activity(node="AIRWAY", step="align", parameters={})
    write_measurement(store, activity, software_agent(store), name=name, signal="plain", attributes=attributes)


def _breaths(phases: int) -> EventEvidence:
    """The task layer's evidence of ``phases`` breath phases, decided present."""
    events = tuple(TaskEvent(float(i), float(i) + 0.5, 15.0) for i in range(phases))
    return EventEvidence(events, events, None, (0.0, float(phases)), "present", "clear")


class TestVerdictReadsEachKind:
    """VERDICT's task evidence names the owning branch and the reading each kind is decided on."""

    @pytest.mark.parametrize(
        ("task", "field", "mode"),
        [
            ("hold-a-tone", "voice_mode", "sustained"),
            ("slide-up", "voice_mode", "glide"),
            ("pa-pa-pa", "ddk_mode", SYLLABLE),
            ("pa-ta-ka", "ddk_mode", CYCLE),
        ],
    )
    def test_a_voice_or_syllable_task_is_read_in_its_own_mode(
        self, lab_corpus: TriageConfig, task: str, field: str, mode: str
    ) -> None:
        """The mode is the task list's pattern; a branch that never reported lacks nothing."""
        evidence = verdict_module._task_evidence(_seeded(task), task)
        assert evidence.owning_branches == (KINDS[task][0],)
        assert getattr(evidence, field) == mode
        assert evidence.owner_absent_inputs == ()

    @pytest.mark.parametrize("task", ["three-coughs", "cough-then-breathe"])
    def test_a_cough_task_is_decided_on_airways_cough_reading(self, lab_corpus: TriageConfig, task: str) -> None:
        """The counted cough mode, the instructed count from the task list, and the onsets AIRWAY wrote."""
        store = _seeded(task)
        _reading(store, COUGH_READING, cough_attributes(CoughPattern(onsets_s=(1.0, 2.0, 3.0)), 3))
        evidence = verdict_module._task_evidence(store, task)
        assert (evidence.owning_branches, evidence.event_kind) == (("AIRWAY",), "cough")
        assert evidence.cough_mode == COUGH_COUNTED
        assert evidence.instructed_count == EXPECTATIONS["AIRWAY"][task].required_count.value  # type: ignore[union-attr]
        assert evidence.cough_onsets_n == 3

    def test_a_breathing_task_is_decided_on_airways_breath_reading(self, lab_corpus: TriageConfig) -> None:
        """Quiet breathing asks for a stretch of breathing, so it is read in the sustained mode."""
        store = _seeded("quiet-breathing")
        read = BreathPattern(pattern="alternating_breaths", events_n=6, evidence=_breaths(6))
        _reading(store, BREATH_READING, breath_attributes(read))
        evidence = verdict_module._task_evidence(store, "quiet-breathing")
        assert evidence.breath_mode == BREATH_SUSTAINED
        assert evidence.breath_decision == "present"

    def test_a_recording_shorter_than_its_task_discards(self, lab_corpus: TriageConfig) -> None:
        """The fold reads the evidence: a 0.4 s cough recording is under the lab's 1 s minimum."""
        store = _seeded("three-coughs", duration_s=0.4)
        _reading(store, COUGH_READING, cough_attributes(CoughPattern(onsets_s=(0.1,)), 3))
        evidence = verdict_module._task_evidence(store, "three-coughs")
        assert (evidence.duration_s, evidence.minimum_duration_s) == (0.4, 1.0)
        folded = fold_file_verdict(
            [NodeVerdict("ADMIT", Outcome.PASS, None, "ok")],
            branch_reports=[BranchReport(node="AIRWAY", kind="airway", conformance=UNDETERMINED, conformance_of=TASK)],
            spans_by_node={"AIRWAY": 1},
            branch_decisions={
                branch: BranchDecision(
                    branch=branch,
                    will_run=branch == "AIRWAY",
                    route_state="declined",
                    forced_by_declaration=branch == "AIRWAY",
                    declared=branch == "AIRWAY",
                )
                for branch in ("AIRWAY", "SPEECH", "VOICE")
            },
            ran={"AIRWAY": RunState.COMPLETED},
            hint_claims={"AIRWAY": True},
            route_state=EMPTY,
            declared_family="three-coughs",
            task=evidence,
        )
        assert folded.triage is Triage.DISCARD
        assert folded.discard_ground == TOO_SHORT_FOR_TASK


class TestATaskDescribedInline:
    """A hint may describe its task itself; every stage reads ADMIT's record of it as it reads a listed task."""

    DESCRIPTION = {"branch": "VOICE", "pattern": "sustained", "declared_duration_s": 4.0}
    """A held note the lab's task list does not name."""

    def test_its_own_branch_aligns_it_and_the_others_detect(self, lab_corpus: TriageConfig) -> None:
        """The recorded branch and description decide the mode, though the task list has no entry for it."""
        store = _seeded("hum-softly", description=self.DESCRIPTION)
        recorded = recorded_task(store)
        assert recorded is not None and (recorded.source, recorded.branch) == (FROM_HINT_DESCRIPTION, "VOICE")
        assert "hum-softly" not in task_list()
        assert mode_of("VOICE", store) == ("align", "hum-softly")
        assert mode_of("SPEECH", store) == ("detect", "hum-softly")
        ran: list[str] = []
        dispatch(
            "VOICE",
            store,
            branch_params(lab_corpus),
            None,
            align=lambda family, *_: ran.append(f"align {family}") or Result([], []),
            detect=lambda *_: ran.append("detect") or Result([], []),
        )
        assert ran == ["align hum-softly"]

    def test_verdict_reads_its_recorded_description(self, lab_corpus: TriageConfig) -> None:
        """The gates and the task evidence take the owning branch and pattern from ADMIT's record."""
        store = _seeded("hum-softly", description=self.DESCRIPTION)
        assert verdict_module.declared_expectation(store, "hum-softly") == (
            "VOICE",
            Expectation(pattern=Pattern.SUSTAINED, declared_duration_s=4.0),
        )
        evidence = verdict_module._task_evidence(store, "hum-softly")
        assert (evidence.owning_branches, evidence.voice_mode) == (("VOICE",), "sustained")


def _workflow(config: TriageConfig, tmp_path: Path, *off: str) -> TriageConfig:
    """The lab corpus's configuration with ``off`` left out of its workflow."""
    override = tmp_path / "workflow.yaml"
    stages = [stage for stage in config.values["workflow"]["stages"] if stage not in off]
    override.write_text(yaml.safe_dump({"workflow": {"stages": stages}}))
    return load_triage_config(override, profile=LAB_CORPUS)


class TestARunWithStagesOff:
    """The lab corpus runs with stages its workflow leaves off (plan step 11), over the runner's fake graph."""

    def test_a_branch_its_tasks_need_cannot_be_left_off(self, lab_corpus: TriageConfig, tmp_path: Path) -> None:
        """VOICE off: the start check names each of the lab's voice tasks."""
        missing = missing_dataset_values(_workflow(lab_corpus, tmp_path, "VOICE").values)
        for task in ("hold-a-tone", "slide-up"):
            assert any(item.startswith(f"tasks.yaml task {task}: its branch VOICE is") for item in missing), task

    def test_quality_and_cohort_off_are_not_called_and_the_file_is_folded(
        self,
        lab_corpus: TriageConfig,
        graph: Callable[..., list[str]],  # noqa: F811 -- the fixture imported above
        tmp_path: Path,
    ) -> None:
        """Every other stage runs; the two are noted off, and the absent quality join sends nothing to review."""
        config = _workflow(lab_corpus, tmp_path, "QUALITY", "COHORT")
        assert missing_dataset_values(config.values) == []
        calls = graph(routed=(), declared=("SPEECH",), family="read-the-card")
        result = run_triage(tmp_path / "rec-0004.wav", tmp_path / "out", config, hint_for("p07", "s2", "read-the-card"))
        assert "QUALITY" not in calls and "COHORT" not in calls and "VERDICT" in calls
        for stage in ("QUALITY", "COHORT"):
            assert result.ran[stage] is RunState.SKIPPED
            assert result.nodes[stage].note == OFF_IN_WORKFLOW
        assert result.ran["SPEECH"] is RunState.COMPLETED
        assert result.file_verdict is not None
        assert not any("quality_join" in reason.why for reason in result.file_verdict.reasons)
