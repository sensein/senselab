"""A dataset's task list, and the task a recording's hint declares."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pytest
import soundfile
import yaml

from senselab.audio.data_structures import AudioHints
from senselab.audio.workflows.triage import task_catalogue
from senselab.audio.workflows.triage.config import TriageConfig, load_triage_config
from senselab.audio.workflows.triage.dataset_profile import (
    DatasetDataMissing,
    deactivate_profile,
    missing_dataset_values,
)
from senselab.audio.workflows.triage.identity import identity_from_stem, identity_of
from senselab.audio.workflows.triage.identity_check import TASK_NOT_DESCRIBED, IdentityRecord, check_identities
from senselab.audio.workflows.triage.live_evidence import declared_task_of
from senselab.audio.workflows.triage.nodes import branches
from senselab.audio.workflows.triage.nodes.admit import admit
from senselab.audio.workflows.triage.nodes.branches import (
    AIRWAY_EXPECTATIONS,
    EXPECTATIONS,
    SPEECH_EXPECTATIONS,
    VOICE_EXPECTATIONS,
    VOICE_EXPECTATIONS_PENDING_DECLARATION,
    Expectation,
    Result,
    branch_params,
    dispatch,
    mode_of,
)
from senselab.audio.workflows.triage.nodes.gates import Pattern
from senselab.audio.workflows.triage.profile_template import values_to_supply, write_template
from senselab.audio.workflows.triage.task_catalogue import (
    DECLARED_TASK,
    DEFINES_A_CUE_WORD,
    FROM_HINT_DESCRIPTION,
    FROM_TASK_LIST,
    KINDS,
    MODEL_SPEAKER_ALLOWED,
    NO_TASK,
    NOT_IN_TASK_LIST,
    QUICK_BREATHS,
    QUIET_TO_THE_CLASSIFIER,
    SPANS_THE_RECORDING,
    TASK_EXPECTS_DIGEST,
    TASKS_BY_KIND,
    TRAITS,
    generates_items,
    has_own_words,
    has_trait,
    is_kind,
    kinds_of,
    read_task_list,
    recorded_task,
    resolve_task,
    task_attributes,
    task_list,
    voice_reading_text,
)
from senselab.audio.workflows.triage.vocabulary import Outcome
from senselab.utils.prov_store import ProvStore

B2AI_PROFILE = Path(__file__).parent / "profiles" / "bridge2ai_voice_adult"

# The task sets senselab wrote out by hand for B2AI before step 4 (routing_analysis/families.py at 6fc1c8011).
# The task list's kinds must reproduce them exactly.
B2AI_KINDS: dict[str, frozenset[str]] = {
    "speech": frozenset(
        {
            "animal-fluency",
            "cape-v-sentences",
            "cape-v-sentences-v2",
            "caterpillar-passage",
            "cinderella-story",
            "diadochokinesis-buttercup",
            "diadochokinesis-ka",
            "diadochokinesis-pa",
            "diadochokinesis-pataka",
            "diadochokinesis-ta",
            "diadochokinesis-v2-buttercup",
            "diadochokinesis-v2-kuh",
            "diadochokinesis-v2-puh",
            "diadochokinesis-v2-puhtuhkuh",
            "diadochokinesis-v2-tuh",
            "free-speech",
            "free-speech-v2",
            "harvard-sentences-list",
            "loudness",
            "loudness-v2",
            "open-response-questions",
            "picture-description",
            "picture-description-option1",
            "picture-description-option2",
            "productive-vocabulary",
            "rainbow-passage",
            "random-item-generation",
            "random-item-generation-v2",
            "story-recall",
            "story-recall-v2",
            "word-color-stroop",
        }
    ),
    "lexical_speech": frozenset(
        {
            "animal-fluency",
            "cape-v-sentences",
            "cape-v-sentences-v2",
            "caterpillar-passage",
            "cinderella-story",
            "free-speech",
            "free-speech-v2",
            "harvard-sentences-list",
            "loudness",
            "loudness-v2",
            "open-response-questions",
            "picture-description",
            "picture-description-option1",
            "picture-description-option2",
            "productive-vocabulary",
            "rainbow-passage",
            "random-item-generation",
            "random-item-generation-v2",
            "story-recall",
            "story-recall-v2",
            "word-color-stroop",
        }
    ),
    "syllable_repetition": frozenset(
        {
            "diadochokinesis-buttercup",
            "diadochokinesis-ka",
            "diadochokinesis-pa",
            "diadochokinesis-pataka",
            "diadochokinesis-ta",
            "diadochokinesis-v2-buttercup",
            "diadochokinesis-v2-kuh",
            "diadochokinesis-v2-puh",
            "diadochokinesis-v2-puhtuhkuh",
            "diadochokinesis-v2-tuh",
        }
    ),
    "voice": frozenset(
        {
            "glides-high-to-low",
            "glides-low-to-high",
            "high-to-low",
            "maximum-phonation-time",
            "maximum-phonation-time-v2",
            "prolonged-vowel",
        }
    ),
    "glide": frozenset(
        {
            "glides-high-to-low",
            "glides-low-to-high",
            "high-to-low",
        }
    ),
    "sustained": frozenset(
        {
            "maximum-phonation-time",
            "maximum-phonation-time-v2",
            "prolonged-vowel",
        }
    ),
    "airway": frozenset(
        {
            "breath-sounds",
            "respiration-and-cough-breath",
            "respiration-and-cough-cough",
            "respiration-and-cough-fivebreaths",
            "respiration-and-cough-threequickbreaths",
            "respiration-and-cough-v2-breath",
            "respiration-and-cough-v2-hardcough",
            "respiration-and-cough-v2-threebreaths",
            "respiration-and-cough-v2-threebreathsmouth",
            "respiration-and-cough-v2-threebreathsnose",
            "voluntary-cough",
        }
    ),
    "cough": frozenset(
        {
            "respiration-and-cough-cough",
            "respiration-and-cough-v2-hardcough",
            "voluntary-cough",
        }
    ),
    "breath": frozenset(
        {
            "breath-sounds",
            "respiration-and-cough-breath",
            "respiration-and-cough-fivebreaths",
            "respiration-and-cough-threequickbreaths",
            "respiration-and-cough-v2-breath",
            "respiration-and-cough-v2-threebreaths",
            "respiration-and-cough-v2-threebreathsmouth",
            "respiration-and-cough-v2-threebreathsnose",
        }
    ),
}


_WORDS = (
    "version: 1\ntasks:\n  a:\n    branch: SPEECH\n    expects: {{pattern: item_list}}\n"
    "    categories:\n      C: {category}\n"
)

ONE_TASK = """
version: 1
tasks:
  read-aloud:
    branch: SPEECH
    expects: {pattern: ordered_tokens, token_source: stimulus_text}
"""


class TestTheTaskList:
    """A profile's ``tasks.yaml``: one entry per task, its branch and what it asks for."""

    def test_an_entry_fills_every_field_it_leaves_out_with_the_default(self) -> None:
        """Only the fields that differ from the defaults are written."""
        (entry,) = read_task_list(ONE_TASK, "t").values()
        assert entry.branch == "SPEECH"
        assert entry.expects == Expectation(pattern=Pattern.ORDERED_TOKENS, token_source="stimulus_text")

    @pytest.mark.parametrize(
        ("text", "message"),
        [
            ("version: 2\ntasks: {}\n", "version"),
            ("version: 1\ntasks:\n  a: {branch: COHORT, expects: {pattern: sustained}}\n", "branch"),
            ("version: 1\ntasks:\n  a: {branch: VOICE, expects: {pattern: hummed}}\n", "hummed"),
            ("version: 1\ntasks:\n  a: {branch: VOICE, expects: {pattern: sustained, pitch: 3}}\n", "pitch"),
            ("version: 1\ntasks:\n  a: {branch: VOICE, expects: {}}\n", "pattern"),
            ("version: 1\ntasks:\n  a: {branch: VOICE, expects: {pattern: sustained}, route: nose}\n", "route"),
            ("version: 1\nanchors: {}\ntasks: {}\n", "anchors"),
            (_WORDS.format(category="{rule: colour}"), "colour"),
            (_WORDS.format(category="{rule: list}"), "words file"),
            (_WORDS.format(category="{rule: letters, words: a.yaml}"), "words file"),
            (_WORDS.format(category="{rule: initial_letter, letter: tt}"), "one letter"),
            (_WORDS.format(category="{rule: letters, number_words: [one]}"), "number_words"),
            (_WORDS.format(category="{rule: letters, repetition_allowed: sometimes}"), "true or false"),
            (_WORDS.format(category="{rule: letters, language: en}"), "language"),
            ("version: 1\ntasks:\n  a: {branch: VOICE, expects: {pattern: sustained}, traits: [loud]}\n", "traits"),
        ],
    )
    def test_a_malformed_list_is_refused(self, text: str, message: str) -> None:
        """A wrong version, branch, pattern, field or key is refused, naming what is wrong."""
        with pytest.raises(ValueError, match=message):
            read_task_list(text, "t")

    def test_the_b2ai_profile_supplies_every_task_senselab_applied(self) -> None:
        """48 tasks over three branches, each in exactly one, and the four unused voice readings."""
        tasks = task_list()
        assert len(tasks) == 48
        assert len(SPEECH_EXPECTATIONS) + len(VOICE_EXPECTATIONS) + len(AIRWAY_EXPECTATIONS) == 48
        assert {name for table in EXPECTATIONS.values() for name in table} == set(tasks)
        assert sorted(VOICE_EXPECTATIONS_PENDING_DECLARATION) == sorted(
            ["cape-v-sentences", "cape-v-sentences-v2", "loudness", "loudness-v2"]
        )
        assert all(name in SPEECH_EXPECTATIONS for name in VOICE_EXPECTATIONS_PENDING_DECLARATION)

    def test_a_task_names_one_of_the_branches_triage_has(self) -> None:
        """The task list's branches are the graph's (kept apart only so neither module imports the other)."""
        assert task_catalogue.BRANCHES == branches.BRANCHES

    def test_shared_anchors_sit_under_x_keys(self) -> None:
        """A YAML anchor shared by several entries is kept under an ``x-`` key, which the list ignores."""
        text = (
            "version: 1\nx-c: &c {Letters: {rule: letters, repetition_allowed: true}}\ntasks:\n"
            "  a: {branch: SPEECH, expects: {pattern: item_list}, categories: *c}\n"
            "  b: {branch: SPEECH, expects: {pattern: item_list}, categories: *c, lexicon: task_lexicon/b.yaml}\n"
        )
        tasks = read_task_list(text, "t")
        assert tasks["a"].categories == tasks["b"].categories
        assert tasks["a"].categories["Letters"].repetition_allowed
        assert (tasks["a"].lexicon, tasks["b"].lexicon) == (None, "task_lexicon/b.yaml")

    @pytest.mark.usefixtures("without_triage_profile")
    def test_senselab_ships_no_task(self) -> None:
        """Without a profile there is no task list to read."""
        with pytest.raises(DatasetDataMissing):
            len(SPEECH_EXPECTATIONS)


class TestKinds:
    """Which tasks are speech, voice, airway and their sub-kinds, worked out from each task's entry."""

    @pytest.mark.parametrize("kind", KINDS)
    def test_the_b2ai_task_list_reproduces_the_hand_written_sets(self, kind: str) -> None:
        """Each kind's tasks are the ones senselab named by hand before."""
        assert TASKS_BY_KIND[kind] == B2AI_KINDS[kind]

    def test_a_task_has_its_branch_and_one_sub_kind(self) -> None:
        """Airway carries a breath/cough sub-kind, voice a glide/sustained one, speech a lexical/syllable one."""
        assert kinds_of("respiration-and-cough-fivebreaths") == frozenset({"airway", "breath"})
        assert kinds_of("voluntary-cough") == frozenset({"airway", "cough"})
        assert kinds_of("maximum-phonation-time") == frozenset({"voice", "sustained"})
        assert kinds_of("harvard-sentences-list") == frozenset({"speech", "lexical_speech"})
        assert kinds_of("diadochokinesis-ka") == frozenset({"speech", "syllable_repetition"})
        assert kinds_of("audio-check") == frozenset()
        assert kinds_of(None) == frozenset()

    def test_an_unknown_kind_is_refused(self) -> None:
        """A configuration naming a kind triage does not know is a mistake, not an empty set."""
        with pytest.raises(KeyError):
            is_kind("voluntary-cough", "coughing")
        with pytest.raises(KeyError):
            TASKS_BY_KIND["coughing"]

    @pytest.mark.usefixtures("without_triage_profile")
    def test_naming_a_kind_needs_no_profile(self) -> None:
        """A configuration is checked against the kinds before any task list is read."""
        assert "airway" in TASKS_BY_KIND
        assert "coughing" not in TASKS_BY_KIND
        assert tuple(TASKS_BY_KIND) == KINDS


class TestTraits:
    """What an entry says about its task beyond its description (plan step 5a)."""

    def test_the_b2ai_entries_carry_the_five_traits_senselab_listed_by_hand(self) -> None:
        """Each trait is on exactly the tasks the deleted senselab and profile lists named."""
        named = {trait: {name for name in task_list() if has_trait(name, trait)} for trait in TRAITS}
        assert named == {
            DEFINES_A_CUE_WORD: {"productive-vocabulary"},
            MODEL_SPEAKER_ALLOWED: {"harvard-sentences-list", "cape-v-sentences", "cape-v-sentences-v2"},
            QUIET_TO_THE_CLASSIFIER: {"respiration-and-cough-breath", "respiration-and-cough-v2-breath"},
            QUICK_BREATHS: {"respiration-and-cough-threequickbreaths", "respiration-and-cough-v2-threebreaths"},
            SPANS_THE_RECORDING: {"diadochokinesis-buttercup", "diadochokinesis-v2-buttercup"},
        }

    def test_an_unknown_trait_is_refused(self) -> None:
        """A trait triage does not know is a mistake, not a no."""
        with pytest.raises(KeyError):
            has_trait("productive-vocabulary", "defines_a_word")

    def test_the_lists_the_task_list_now_works_out_equal_the_deleted_ones(self) -> None:
        """The eight lists B2AI's profile held before step 5a, task for task."""
        listed_words = {"cinderella-story", "animal-fluency", "random-item-generation", "random-item-generation-v2"}
        assert {name for name in task_list() if has_own_words(name)} == listed_words
        assert {name for name in task_list() if generates_items(name)} == listed_words - {"cinderella-story"}
        assert {name for name in task_list() if voice_reading_text(name)} == {"loudness", "loudness-v2"}
        assert TASKS_BY_KIND["cough"] == {
            "respiration-and-cough-cough",
            "respiration-and-cough-v2-hardcough",
            "voluntary-cough",
        }
        assert len(TASKS_BY_KIND["breath"]) == 8


class TestTheTaskAHintDeclares:
    """A hint names its task exactly, or describes it inline; nothing is read off a file name."""

    def test_a_named_task_resolves_to_its_entry(self) -> None:
        """The branch and the description are the task list's."""
        resolved = resolve_task(AudioHints(task="rainbow-passage", task_take=2, item_category=None))
        assert (resolved.source, resolved.branch, resolved.take) == (FROM_TASK_LIST, "SPEECH", 2)
        assert resolved.expects == SPEECH_EXPECTATIONS["rainbow-passage"]

    def test_a_task_the_list_lacks_or_no_task_has_no_description(self) -> None:
        """Both take the default treatment: no branch, no expectation."""
        missing = resolve_task(AudioHints(task="audio-check"))
        assert (missing.name, missing.source, missing.branch, missing.expects) == (
            "audio-check",
            NOT_IN_TASK_LIST,
            None,
            None,
        )
        for hint in (None, AudioHints()):
            assert resolve_task(hint).source == NO_TASK

    def test_an_inline_description_needs_no_task_list(self) -> None:
        """A dataset with no task list describes each task in its hint, branch included."""
        hint = AudioHints(task="hum", task_description={"branch": "VOICE", "pattern": "sustained"})
        resolved = resolve_task(hint)
        assert (resolved.source, resolved.branch) == (FROM_HINT_DESCRIPTION, "VOICE")
        assert resolved.expects == Expectation(pattern=Pattern.SUSTAINED)

    @pytest.mark.parametrize(
        "description", [{"pattern": "sustained"}, {"branch": "VOICE"}, {"branch": "VOICE", "pattern": "x"}]
    )
    def test_an_inline_description_without_a_branch_or_a_valid_pattern_is_refused(
        self, description: dict[str, Any]
    ) -> None:
        """A recording cannot be routed on a description it does not complete."""
        with pytest.raises(ValueError):
            resolve_task(AudioHints(task_description=description))

    def test_admit_records_the_resolved_task_for_every_later_reader(self) -> None:
        """The record reads back as the task resolved; a changed description changes its digest."""
        store = ProvStore(run_id="r")
        hint = AudioHints(task="respiration-and-cough-fivebreaths", task_variant={"route": "mouth"}, task_take=2)
        attributes = task_attributes(hint)
        store.entity(
            prov_type="stream", extent=(0.0, 1.0), attributes={"name": "recording", "path": "/x.wav", **attributes}
        )
        assert recorded_task(store) == resolve_task(hint)
        assert attributes[DECLARED_TASK] == "respiration-and-cough-fivebreaths"
        other = task_attributes(AudioHints(task="respiration-and-cough-threequickbreaths"))
        assert other[TASK_EXPECTS_DIGEST] != attributes[TASK_EXPECTS_DIGEST]

    def test_a_store_from_before_the_record_has_no_recorded_task(self) -> None:
        """Nothing is guessed for it."""
        store = ProvStore(run_id="old")
        store.entity(prov_type="stream", extent=(0.0, 1.0), attributes={"name": "recording", "path": "/x.wav"})
        assert recorded_task(store) is None


# ------------------------------------------------------------------ a dataset that is not B2AI

LAB_TASKS = """
version: 1
tasks:
  hum-a-note:
    branch: VOICE
    expects: {pattern: sustained}
  read-the-card:
    branch: SPEECH
    expects: {pattern: ordered_tokens, token_source: stimulus_text}
"""
"""A made-up lab corpus: two tasks under its own names, none of them B2AI's."""


def _set(document: dict, path: list[str], value: Any) -> None:  # noqa: ANN401 -- a YAML value
    *parents, last = path
    for part in parents:
        document = document.setdefault(part, {})
    document[last] = value


def _b2ai_value(relative: str, path: list[str]) -> Any:  # noqa: ANN401 -- a YAML value
    document = yaml.safe_load(
        (B2AI_PROFILE / ("config.yaml" if relative == "config/default.yaml" else relative)).read_text()
    )
    for part in path:
        document = document[part]
    return document


def _lab_profile(root: Path) -> Path:
    """The lab corpus's profile, written from senselab's template and filled in by hand (plan step 5, check 4).

    The template extends senselab's reference profile, so it asks only for the lab's own values. The lab writes its
    own task list; its policy and statistics are B2AI's here, a stand-in for the choices a real dataset makes and the
    statistics it fits (step 10). No entry names a B2AI task.

    Args:
        root: Where the profile directory goes.

    Returns:
        The profile directory.
    """
    write_template(root, name="lab-corpus")
    (root / "tasks.yaml").write_text(LAB_TASKS)
    (root / "airway_event_tokens.yaml").write_text("version: 3\ncough: [cough]\n")
    for name in ("discard_contested.yaml", "time_release.yaml"):
        shutil.copy(B2AI_PROFILE / name, root / name)
    for relative, entries in values_to_supply().items():
        target = root / ("config.yaml" if relative == "config/default.yaml" else relative)
        document = yaml.safe_load(target.read_text()) or {}
        for entry in entries:
            path = [str(part) for part in entry["path"]]
            value = ["t", "d"] if path[-1] == "alveolar" else _b2ai_value(relative, path)
            _set(document, path, value)
        target.write_text(yaml.safe_dump(document))
    return root


@pytest.fixture
def lab_corpus(without_triage_profile: None, tmp_path: Path) -> Iterator[TriageConfig]:
    """The lab corpus's profile, active for the test, and no profile active after it."""
    config = load_triage_config(profile=_lab_profile(tmp_path / "lab-profile"))
    yield config
    deactivate_profile()


def _admitted(config: TriageConfig, wav: Path, hint: AudioHints) -> ProvStore:
    """A recording through the real ADMIT node."""
    store = ProvStore(run_id=wav.stem)
    t = np.arange(16000) / 16000
    soundfile.write(wav, (0.5 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32), 16000)
    result = admit(store, wav, config, hint, run_dir=wav.parent)
    assert result.verdict.outcome is Outcome.PASS
    return store


class TestADatasetThatIsNotB2AI:
    """A made-up dataset, its own task names, no BIDS-like file names: triaged from its own task list.

    Plan step 4, check 6.
    """

    def test_its_profile_is_all_a_run_needs(self, lab_corpus: TriageConfig) -> None:
        """Written from the template and filled in, it names no B2AI task and the run's start check finds nothing."""
        assert missing_dataset_values(lab_corpus.values) == []
        assert set(task_list()) == {"hum-a-note", "read-the-card"}

    def test_its_tasks_are_grouped_from_its_own_entries(self, lab_corpus: TriageConfig) -> None:
        """The kinds are worked out from the two entries; no B2AI task is in any of them."""
        assert TASKS_BY_KIND["voice"] == TASKS_BY_KIND["sustained"] == {"hum-a-note"}
        assert TASKS_BY_KIND["speech"] == TASKS_BY_KIND["lexical_speech"] == {"read-the-card"}
        assert TASKS_BY_KIND["airway"] == TASKS_BY_KIND["syllable_repetition"] == frozenset()
        assert set(VOICE_EXPECTATIONS) == {"hum-a-note"}

    @pytest.mark.parametrize(
        ("task", "aligned_by"),
        [("hum-a-note", "VOICE"), ("read-the-card", "SPEECH")],
    )
    def test_each_recording_is_read_by_its_tasks_branch(
        self, lab_corpus: TriageConfig, tmp_path: Path, task: str, aligned_by: str
    ) -> None:
        """ADMIT records the hint's task from a plain file name; its branch aligns it and the others detect."""
        store = _admitted(lab_corpus, tmp_path / "clip_0007.wav", AudioHints(task=task))
        recorded = recorded_task(store)
        assert recorded is not None and (recorded.name, recorded.source, recorded.branch) == (
            task,
            FROM_TASK_LIST,
            aligned_by,
        )
        assert declared_task_of(store) == (task, task)
        params = branch_params(lab_corpus)
        for branch in ("AIRWAY", "SPEECH", "VOICE"):
            ran: list[str] = []
            dispatch(
                branch,
                store,
                params,
                AudioHints(task=task),
                align=lambda family, *_: ran.append(f"align {family}") or Result([], []),
                detect=lambda *_: ran.append("detect") or Result([], []),
            )
            assert ran == ([f"align {task}"] if branch == aligned_by else ["detect"]), branch

    def test_a_task_its_list_lacks_is_read_by_no_branch(self, lab_corpus: TriageConfig, tmp_path: Path) -> None:
        """B2AI's own task name means nothing to this dataset: every branch detects."""
        store = _admitted(lab_corpus, tmp_path / "clip_0008.wav", AudioHints(task="prolonged-vowel"))
        recorded = recorded_task(store)
        assert recorded is not None and (recorded.source, recorded.branch) == (NOT_IN_TASK_LIST, None)
        assert mode_of("VOICE", store) == ("detect", "prolonged-vowel")

    def test_its_identity_check_resolves_against_its_list(self, lab_corpus: TriageConfig) -> None:
        """Hints that declare a participant, a session and a listed task pass; an unlisted task is reported."""
        records = []
        for stem, task in (("clip_0001", "hum-a-note"), ("clip_0002", "read-the-card"), ("clip_0003", "whistle")):
            hint = AudioHints(task=task, metadata={"participant_id": "p01", "session_id": "visit1", "task_token": stem})
            resolved = resolve_task(hint)
            records.append(
                IdentityRecord(
                    stem,
                    f"/lab/{stem}.wav",
                    identity_of(stem, hint),
                    identity_from_stem(stem),
                    task=resolved.name,
                    task_source=resolved.source,
                )
            )
        report = check_identities(records, session_scope="participant")
        assert not report.refused
        assert [(finding.check, finding.key) for finding in report.findings] == [(TASK_NOT_DESCRIBED, "whistle")]
