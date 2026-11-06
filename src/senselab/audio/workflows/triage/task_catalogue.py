"""A dataset's task list: what each of its tasks asks a recording to contain.

senselab knows no task by name. A dataset profile supplies ``tasks.yaml``: one entry per task, under the name the
dataset's hints declare, naming the branch that reads the task and describing what it asks for as an
:class:`~senselab.audio.workflows.triage.nodes.branches.Expectation` (a field left out takes its default)::

    version: 1
    tasks:
      rainbow-passage:
        branch: SPEECH
        expects: {pattern: ordered_tokens, token_source: stimulus_text, connected: true}

An entry may also carry ``unused_voice_reading``: a VOICE description a dataset wrote for a SPEECH task and that no
stage applies, kept with the task it describes; and, on a syllable task, ``carrier``: the text the repeated syllables
spell (``pa``, ``buttercup``), which a transcript of a faithful performance contains.

A recording's hint names its task exactly (``AudioHints.task``), or describes it inline
(``AudioHints.task_description``) for a dataset with no task list. ADMIT records what it resolved
(:func:`task_attributes`) on the ``recording`` stream and its activity, so every later stage reads the record
(:func:`recorded_task`), never the file name.

See ``specs/20261022-triage-dataset-profile/plan.md``, step 4.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING, Any, Iterator, Mapping

import yaml

from senselab.audio.data_structures import AudioHints
from senselab.audio.workflows.triage.dataset_profile import data_path, profile_cache
from senselab.audio.workflows.triage.nodes.gates import Pattern

if TYPE_CHECKING:
    from senselab.audio.workflows.triage.nodes.branches import Expectation
    from senselab.utils.prov_store import ProvStore

TASKS_FILE = "tasks.yaml"
"""The profile data file the task list is read from."""

TASKS_VERSION = 1
"""The task list format this module reads."""

_ENTRY_KEYS = frozenset(
    {"branch", "expects", "unused_voice_reading", "voice_reading_text", "lexicon", "categories", "traits", "carrier"}
)

DEFINES_A_CUE_WORD = "defines_a_cue_word"
MODEL_SPEAKER_ALLOWED = "model_speaker_allowed"
QUIET_TO_THE_CLASSIFIER = "quiet_to_the_classifier"
QUICK_BREATHS = "quick_breaths"
SPANS_THE_RECORDING = "spans_the_recording"
TRAITS = (DEFINES_A_CUE_WORD, MODEL_SPEAKER_ALLOWED, QUIET_TO_THE_CLASSIFIER, QUICK_BREATHS, SPANS_THE_RECORDING)
"""What an entry may say about its task beyond its description (plan step 5a):

* ``defines_a_cue_word``: the stimulus is a cue word the participant defines, so a word reading as it is task content.
* ``model_speaker_allowed``: the instructions let someone say the item first; the speaker gate's flag says so.
* ``quiet_to_the_classifier``: the sound is too quiet for the sound classifier to hear, so its branch not finding the
  declared sound is no flag.
* ``quick_breaths``: breaths close together; the breath train is read at the quick spacing.
* ``spans_the_recording``: the task's own word fills the recording, so its syllable train is never trimmed.
"""

LIST_RULE = "list"
LETTERS_RULE = "letters"
NUMBERS_RULE = "numbers"
INITIAL_LETTER_RULE = "initial_letter"
CATEGORY_RULES = (LIST_RULE, LETTERS_RULE, NUMBERS_RULE, INITIAL_LETTER_RULE)
"""The kinds of rule a word-generation category is: a fixed word list, single letters, numbers (digits and the
category's number words), or the words that begin with one letter."""

_CATEGORY_KEYS = frozenset({"rule", "words", "letter", "number_words", "repetition_allowed"})

SPEECH_KIND = "speech"
LEXICAL_SPEECH_KIND = "lexical_speech"
SYLLABLE_REPETITION_KIND = "syllable_repetition"
VOICE_KIND = "voice"
GLIDE_KIND = "glide"
SUSTAINED_KIND = "sustained"
AIRWAY_KIND = "airway"
COUGH_KIND = "cough"
BREATH_KIND = "breath"
KINDS = (
    SPEECH_KIND,
    LEXICAL_SPEECH_KIND,
    SYLLABLE_REPETITION_KIND,
    VOICE_KIND,
    GLIDE_KIND,
    SUSTAINED_KIND,
    AIRWAY_KIND,
    COUGH_KIND,
    BREATH_KIND,
)
"""The kinds of task triage knows, each worked out from a task's branch and description (:func:`kinds_of_entry`)."""

SYLLABLE_PATTERNS = frozenset({Pattern.SYLLABLE_TRAIN, Pattern.SYLLABLE_SEQUENCE})
"""The patterns that make a SPEECH task syllable repetition rather than lexical speech."""

BRANCHES = ("AIRWAY", "SPEECH", "VOICE")
"""The branches a task may name, as :data:`~senselab.audio.workflows.triage.nodes.branches.BRANCHES` lists them."""


def _expectation() -> type["Expectation"]:
    """The ``Expectation`` class, imported when first needed.

    The branches module reads this one's tables lazily, and this one is imported by modules the branches module's
    own imports load (the routing analysis), so neither imports the other at load time.

    Returns:
        :class:`~senselab.audio.workflows.triage.nodes.branches.Expectation`.
    """
    from senselab.audio.workflows.triage.nodes.branches import Expectation  # noqa: PLC0415

    return Expectation


@dataclass(frozen=True)
class ItemCategory:
    """One category a word-generation task may ask for ("name as many animals as you can").

    Attributes:
        name: The category, as the recording's hint declares it (``AudioHints.item_category``).
        rule: Which of :data:`CATEGORY_RULES` decides whether a word belongs to it.
        words: For a ``list`` category, the profile data file holding its items (``items:``); else None.
        letter: For an ``initial_letter`` category, the letter its words begin with; else None.
        number_words: For a ``numbers`` category, the spelled numbers it admits beside the digits.
        repetition_allowed: Whether saying an item twice is part of the task rather than a mistake.
    """

    name: str
    rule: str
    words: str | None = None
    letter: str | None = None
    number_words: frozenset[str] = frozenset()
    repetition_allowed: bool = False


def _item_category(name: str, entry: Any, where: str) -> ItemCategory:  # noqa: ANN401 -- parsed YAML
    """Parse one entry of a task's ``categories``.

    Args:
        name: The category's name.
        entry: Its mapping.
        where: The task, for messages.

    Returns:
        The category.

    Raises:
        ValueError: On an unknown key or rule, or a rule missing what it needs.
    """
    where = f"{where} category {name!r}"
    if not isinstance(entry, Mapping):
        raise ValueError(f"{where} is not a mapping")
    unknown = sorted(set(entry) - _CATEGORY_KEYS)
    if unknown:
        raise ValueError(f"{where}: unknown keys {unknown}")
    rule = entry.get("rule")
    if rule not in CATEGORY_RULES:
        raise ValueError(f"{where}: rule {rule!r} is not one of {', '.join(CATEGORY_RULES)}")
    words, letter, numbers = entry.get("words"), entry.get("letter"), entry.get("number_words")
    if (rule == LIST_RULE) != isinstance(words, str):
        raise ValueError(f"{where}: a list category, and only one, names its words file")
    if (rule == INITIAL_LETTER_RULE) != (isinstance(letter, str) and len(letter) == 1 and letter.isalpha()):
        raise ValueError(f"{where}: an initial_letter category, and only one, names one letter")
    if numbers is not None and (rule != NUMBERS_RULE or not isinstance(numbers, list)):
        raise ValueError(f"{where}: number_words is a list, on a numbers category only")
    allowed = entry.get("repetition_allowed", False)
    if not isinstance(allowed, bool):
        raise ValueError(f"{where}: repetition_allowed is true or false")
    return ItemCategory(
        name=name,
        rule=str(rule),
        words=words,
        letter=letter.lower() if isinstance(letter, str) else None,
        number_words=frozenset(str(word).lower() for word in numbers or ()),
        repetition_allowed=allowed,
    )


@dataclass(frozen=True)
class TaskEntry:
    """One task of a dataset's task list.

    Attributes:
        name: The task's name, as the dataset's hints declare it.
        branch: The branch that reads it, one of
            :data:`~senselab.audio.workflows.triage.nodes.branches.BRANCHES`.
        expects: What the task asks a recording to contain.
        unused_voice_reading: A VOICE description of a SPEECH task that no stage applies, or None.
        voice_reading_text: For a word task also read for its voice (a loudness task: say "hey", then shout it),
            the words the voice is read over; None for every other task.
        lexicon: The profile data file listing the task's own words and phrases (``task_lexicon/...``), or None.
        categories: For a word-generation task, the categories a recording may ask for, by name; empty otherwise.
        traits: What the entry says about its task beyond its description, each one of :data:`TRAITS`.
        carrier: For a syllable task, the text its syllables spell, as a transcript would hold it; None otherwise.
    """

    name: str
    branch: str
    expects: Expectation
    unused_voice_reading: Expectation | None = None
    voice_reading_text: tuple[str, ...] | None = None
    lexicon: str | None = None
    categories: Mapping[str, ItemCategory] = field(default_factory=dict)
    traits: frozenset[str] = frozenset()
    carrier: str | None = None


def expectation_of(description: Mapping[str, Any], where: str) -> Expectation:
    """An :class:`Expectation` from a task list's (or a hint's) description, defaults filled in.

    Args:
        description: The description: ``pattern`` and any other :class:`Expectation` fields.
        where: What the description belongs to, for the message.

    Returns:
        The expectation.

    Raises:
        ValueError: If the description names no pattern, an unknown pattern or an unknown field.
    """
    if not isinstance(description, Mapping) or "pattern" not in description:
        raise ValueError(f"{where}: a task description is a mapping naming its pattern")
    unknown = sorted(set(description) - {known.name for known in fields(_expectation())})
    if unknown:
        raise ValueError(f"{where}: unknown task description fields {unknown}")
    try:
        pattern = Pattern(description["pattern"])
    except ValueError as error:
        raise ValueError(f"{where}: {error}") from error
    expectation = _expectation()
    return expectation.from_mapping({**expectation(pattern=pattern).as_mapping(), **description})


def read_task_list(text: str, source: str) -> dict[str, TaskEntry]:
    """Parse a task list.

    Args:
        text: The YAML text.
        source: Where it was read from, for messages.

    Returns:
        The entries by name, in file order.

    Raises:
        ValueError: On another format version, an unknown branch or key, or a description :func:`expectation_of`
            refuses.
    """
    data = yaml.safe_load(text) or {}
    if data.get("version") != TASKS_VERSION:
        raise ValueError(f"{source}: version {data.get('version')!r}, expected {TASKS_VERSION}")
    stray = sorted(str(key) for key in data if key not in ("version", "tasks") and not str(key).startswith("x-"))
    if stray:
        raise ValueError(f"{source}: unknown top-level keys {stray} (shared YAML anchors go under an x- key)")
    out: dict[str, TaskEntry] = {}
    for name, entry in (data.get("tasks") or {}).items():
        where = f"{source}: task {name!r}"
        if not isinstance(entry, Mapping):
            raise ValueError(f"{where} is not a mapping")
        unknown = sorted(set(entry) - _ENTRY_KEYS)
        if unknown:
            raise ValueError(f"{where}: unknown keys {unknown}")
        if entry.get("branch") not in BRANCHES:
            raise ValueError(f"{where}: branch {entry.get('branch')!r} is not one of {', '.join(BRANCHES)}")
        voice = entry.get("unused_voice_reading")
        text = entry.get("voice_reading_text")
        if text is not None and (not isinstance(text, list) or not all(isinstance(word, str) for word in text)):
            raise ValueError(f"{where}: voice_reading_text is a list of words")
        lexicon = entry.get("lexicon")
        if lexicon is not None and not isinstance(lexicon, str):
            raise ValueError(f"{where}: lexicon is the path of a profile data file")
        categories = entry.get("categories") or {}
        if not isinstance(categories, Mapping):
            raise ValueError(f"{where}: categories is a mapping of category name to its rule")
        traits = entry.get("traits") or []
        if not isinstance(traits, list) or not set(traits) <= set(TRAITS):
            raise ValueError(f"{where}: traits is a list of {', '.join(TRAITS)}")
        expects = expectation_of(entry.get("expects") or {}, where)
        carrier = entry.get("carrier")
        if carrier is not None and (
            not isinstance(carrier, str) or not carrier.strip() or expects.pattern not in SYLLABLE_PATTERNS
        ):
            raise ValueError(f"{where}: carrier is the text a syllable task's syllables spell, on a syllable task only")
        out[str(name)] = TaskEntry(
            name=str(name),
            branch=str(entry["branch"]),
            expects=expects,
            unused_voice_reading=None if voice is None else expectation_of(voice, f"{where} unused_voice_reading"),
            voice_reading_text=None if text is None else tuple(text),
            lexicon=lexicon,
            categories={str(key): _item_category(str(key), value, where) for key, value in categories.items()},
            traits=frozenset(str(trait) for trait in traits),
            carrier=carrier,
        )
    return out


@profile_cache
def task_list() -> dict[str, TaskEntry]:
    """The active dataset profile's task list.

    Returns:
        The entries by name, in file order.

    Raises:
        DatasetDataMissing: If no active profile supplies ``tasks.yaml``.
    """
    path = data_path(TASKS_FILE)
    return read_task_list(path.read_text(encoding="utf-8"), str(path))


@profile_cache
def branch_rows(branch: str) -> dict[str, Expectation]:
    """One branch's tasks and what each asks for.

    Args:
        branch: One of :data:`~senselab.audio.workflows.triage.nodes.branches.BRANCHES`.

    Returns:
        Task name to expectation, in the task list's order.
    """
    return {name: entry.expects for name, entry in task_list().items() if entry.branch == branch}


def kinds_of_entry(branch: str | None, expects: Expectation | None) -> frozenset[str]:
    """The kinds a task is, worked out from the branch that reads it and what it asks for.

    Speech, voice and airway are the branch. A SPEECH task is syllable repetition where its pattern is a syllable
    train or sequence, and lexical speech otherwise; a VOICE task is a glide where its pattern is the glide, and
    sustained otherwise; an AIRWAY task is a cough or a breath as its ``label_set`` names.

    Args:
        branch: The branch, or None where nothing describes the task.
        expects: What the task asks for, or None.

    Returns:
        The task's kinds (:data:`KINDS`); empty where nothing describes it.
    """
    if branch is None or expects is None:
        return frozenset()
    if branch == "SPEECH":
        sub = SYLLABLE_REPETITION_KIND if expects.pattern in SYLLABLE_PATTERNS else LEXICAL_SPEECH_KIND
        return frozenset({SPEECH_KIND, sub})
    if branch == "VOICE":
        return frozenset({VOICE_KIND, GLIDE_KIND if expects.pattern is Pattern.GLIDE else SUSTAINED_KIND})
    if branch == "AIRWAY":
        sound = {COUGH_KIND, BREATH_KIND} & {str(expects.label_set)}
        return frozenset({AIRWAY_KIND, *sound})
    return frozenset()


def kinds_of(task: str | None) -> frozenset[str]:
    """The kinds of a task the active profile's task list names (:func:`kinds_of_entry`).

    Args:
        task: The task's name, or None.

    Returns:
        Its kinds; empty for a task the list lacks, or None.
    """
    entry = task_list().get(task) if task else None
    return kinds_of_entry(entry.branch, entry.expects) if entry is not None else frozenset()


@profile_cache
def tasks_of_kind(kind: str) -> frozenset[str]:
    """Every task of the active profile's task list that is of a kind.

    Args:
        kind: One of :data:`KINDS`.

    Returns:
        The tasks' names.

    Raises:
        KeyError: If ``kind`` is not one of :data:`KINDS`.
    """
    if kind not in KINDS:
        raise KeyError(f"{kind!r} is not a kind of task triage knows: {', '.join(KINDS)}")
    return frozenset(name for name in task_list() if kind in kinds_of(name))


class _TasksByKind(Mapping[str, frozenset[str]]):
    """Each kind's tasks in the active profile's task list (:func:`tasks_of_kind`), read when looked up.

    Membership is the kind's, not the profile's: ``"airway" in TASKS_BY_KIND`` holds with no profile active, so a
    configuration naming a kind is checked without one.
    """

    def __getitem__(self, kind: str) -> frozenset[str]:
        return tasks_of_kind(kind)

    def __contains__(self, kind: object) -> bool:
        return kind in KINDS

    def __iter__(self) -> Iterator[str]:
        return iter(KINDS)

    def __len__(self) -> int:
        return len(KINDS)


TASKS_BY_KIND: Mapping[str, frozenset[str]] = _TasksByKind()
"""Each of :data:`KINDS` to its tasks; the named task sets a configuration file refers to by kind."""


def is_kind(task: str | None, kind: str) -> bool:
    """Whether the task list makes a task of a kind.

    Args:
        task: The task's name, or None.
        kind: One of :data:`KINDS`.

    Returns:
        True where the task is in the list and of that kind.

    Raises:
        KeyError: If ``kind`` is not one of :data:`KINDS`.
    """
    if kind not in KINDS:
        raise KeyError(f"{kind!r} is not a kind of task triage knows: {', '.join(KINDS)}")
    return kind in kinds_of(task)


def voice_reading_text(task: str | None) -> tuple[str, ...] | None:
    """The words a word task is also read for its voice over (:attr:`TaskEntry.voice_reading_text`).

    Args:
        task: The task's name, or None.

    Returns:
        The words, or None for a task not read so.
    """
    entry = task_list().get(task) if task else None
    return entry.voice_reading_text if entry is not None else None


def task_carrier(task: str | None) -> str | None:
    """The text a syllable task's entry says its syllables spell (:attr:`TaskEntry.carrier`).

    Args:
        task: The task's name, or None.

    Returns:
        The carrier as the entry spells it; None where the entry declares none or the task is not in the list.
    """
    entry = task_list().get(task) if task else None
    return entry.carrier if entry is not None else None


def item_category_rule(task: str | None, category: str | None) -> ItemCategory | None:
    """The rule a word-generation task's entry gives one of its categories.

    Args:
        task: The task's name, or None.
        category: The category the recording's hint declares, or None.

    Returns:
        The entry's category of that name; None where the task is not in the list or does not list the category.
    """
    entry = task_list().get(task) if task else None
    return entry.categories.get(category) if entry is not None and category else None


def has_trait(task: str | None, trait: str) -> bool:
    """Whether a task's entry says a trait of it (:data:`TRAITS`).

    Args:
        task: The task's name, or None.
        trait: One of :data:`TRAITS`.

    Returns:
        True where the task is in the list and its entry names the trait.

    Raises:
        KeyError: If ``trait`` is not one of :data:`TRAITS`.
    """
    if trait not in TRAITS:
        raise KeyError(f"{trait!r} is not a task trait triage knows: {', '.join(TRAITS)}")
    entry = task_list().get(task) if task else None
    return entry is not None and trait in entry.traits


def has_own_words(task: str | None) -> bool:
    """Whether a task's entry names its own words: a lexicon file, or word-generation categories.

    Args:
        task: The task's name, or None.

    Returns:
        True where the entry names either.
    """
    entry = task_list().get(task) if task else None
    return entry is not None and (entry.lexicon is not None or bool(entry.categories))


def generates_items(task: str | None) -> bool:
    """Whether a task asks for items of a category (a word-generation task): its entry lists categories.

    Args:
        task: The task's name, or None.

    Returns:
        True where the entry lists any category.
    """
    entry = task_list().get(task) if task else None
    return entry is not None and bool(entry.categories)


def task_lexicon_file(task: str | None) -> str | None:
    """The profile data file a task's entry names as its own words (:attr:`TaskEntry.lexicon`).

    Args:
        task: The task's name, or None.

    Returns:
        The path under the profile's data, or None where the task names none or is not in the list.
    """
    entry = task_list().get(task) if task else None
    return entry.lexicon if entry is not None else None


@profile_cache
def unused_voice_rows() -> dict[str, Expectation]:
    """The VOICE descriptions entries carry and no stage applies.

    Returns:
        Task name to that description.
    """
    return {
        name: entry.unused_voice_reading
        for name, entry in task_list().items()
        if entry.unused_voice_reading is not None
    }


DECLARED_TASK = "declared_task"
TASK_FROM = "task_from"
TASK_BRANCH = "task_branch"
TASK_EXPECTS = "task_expects"
TASK_EXPECTS_DIGEST = "task_expects_digest"
TASK_VARIANT = "task_variant"
TASK_TAKE = "task_take"
ITEM_CATEGORY = "item_category"
"""The ``recording`` stream attributes ADMIT records the resolved task under."""

FROM_TASK_LIST = "task_list"
FROM_HINT_DESCRIPTION = "hint_description"
NOT_IN_TASK_LIST = "not_in_task_list"
NO_TASK = "none"
"""Where a recording's task description came from: the task list, the hint, neither for a named task the list
lacks, or no task declared."""


@dataclass(frozen=True)
class ResolvedTask:
    """What a recording was asked to contain, as ADMIT resolved it from its hint.

    Attributes:
        name: The task's name, or None where the hint names none.
        source: :data:`FROM_TASK_LIST`, :data:`FROM_HINT_DESCRIPTION`, :data:`NOT_IN_TASK_LIST` or :data:`NO_TASK`.
        branch: The branch that reads it, or None where nothing describes the task.
        expects: What it asks for, or None where nothing describes the task.
        variant: The task's declared variants.
        take: Which take, or None.
        item_category: The word-generation category declared for the recording, or None.
    """

    name: str | None
    source: str
    branch: str | None
    expects: Expectation | None
    variant: Mapping[str, str]
    take: int | None
    item_category: str | None

    def as_attributes(self) -> dict[str, Any]:
        """The attributes ADMIT writes onto the ``recording`` stream and its activity."""
        expects = self.expects.as_mapping() if self.expects is not None else None
        return {
            DECLARED_TASK: self.name,
            TASK_FROM: self.source,
            TASK_BRANCH: self.branch,
            TASK_EXPECTS: expects,
            TASK_EXPECTS_DIGEST: _digest(expects) if expects is not None else None,
            TASK_VARIANT: dict(self.variant),
            TASK_TAKE: self.take,
            ITEM_CATEGORY: self.item_category,
        }


def _digest(value: Any) -> str:  # noqa: ANN401 -- any JSON value
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def resolve_task(hint: AudioHints | None) -> ResolvedTask:
    """The task a hint declares, and what it asks for.

    An inline ``task_description`` describes the task itself (its branch under ``branch``); otherwise ``task``
    names an entry of the active profile's task list. Nothing is derived from a file name.

    Args:
        hint: The recording's hint, or None.

    Returns:
        The resolved task; one whose description is unknown carries no branch and no expectation.

    Raises:
        ValueError: If an inline description names no known branch or is not a valid description.
    """
    if hint is None:
        return ResolvedTask(None, NO_TASK, None, None, {}, None, None)
    variant, take, category = dict(hint.task_variant), hint.task_take, hint.item_category
    if hint.task_description is not None:
        description = dict(hint.task_description)
        branch = description.pop("branch", None)
        if branch not in BRANCHES:
            raise ValueError(f"the hint's task description names branch {branch!r}, not one of {', '.join(BRANCHES)}")
        expects = expectation_of(description, "the hint's task description")
        return ResolvedTask(hint.task, FROM_HINT_DESCRIPTION, str(branch), expects, variant, take, category)
    if hint.task is None:
        return ResolvedTask(None, NO_TASK, None, None, variant, take, category)
    entry = task_list().get(hint.task)
    if entry is None:
        return ResolvedTask(hint.task, NOT_IN_TASK_LIST, None, None, variant, take, category)
    return ResolvedTask(hint.task, FROM_TASK_LIST, entry.branch, entry.expects, variant, take, category)


def task_attributes(hint: AudioHints | None) -> dict[str, Any]:
    """What ADMIT records about the recording's task (:meth:`ResolvedTask.as_attributes`).

    Args:
        hint: The recording's hint, or None.

    Returns:
        The attributes.
    """
    return resolve_task(hint).as_attributes()


def recorded_task(store: "ProvStore") -> ResolvedTask | None:
    """The task ADMIT recorded, or None where the store predates the record.

    Args:
        store: The provenance store.

    Returns:
        The recorded task.
    """
    from senselab.audio.workflows.triage.identity import admit_record  # noqa: PLC0415

    record = admit_record(store)
    return task_from_attributes(record[0]) if record is not None else None


def task_from_attributes(attributes: Mapping[str, Any]) -> ResolvedTask | None:
    """The task ADMIT's record holds: a ``recording`` stream's attributes, or ADMIT's activity parameters.

    Args:
        attributes: The record (:func:`~senselab.audio.workflows.triage.identity.admit_choice`).

    Returns:
        The recorded task, or None where the record predates it.
    """
    if TASK_FROM not in attributes:
        return None
    expects = attributes.get(TASK_EXPECTS)
    return ResolvedTask(
        name=attributes.get(DECLARED_TASK),
        source=str(attributes[TASK_FROM]),
        branch=attributes.get(TASK_BRANCH),
        expects=_expectation().from_mapping(dict(expects)) if isinstance(expects, Mapping) else None,
        variant=dict(attributes.get(TASK_VARIANT) or {}),
        take=attributes.get(TASK_TAKE),
        item_category=attributes.get(ITEM_CATEGORY),
    )
