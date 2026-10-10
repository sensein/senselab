"""Speech outside the task: the lexical consensus words a recording holds that are not the task's own content.

Two kinds of task have a defined content, so speech outside it can be read:

- An airway or voice family asks for a sound, not words. A lexical consensus word in such a recording -- not
  bracketed, not a vocalisation (:func:`~senselab.audio.workflows.triage.residue.is_non_lexical`) -- that is
  neither the voice family's declared task text nor a recogniser's misreading of the task's own events
  (:func:`~senselab.audio.workflows.triage.task_content.task_content_ids`) is speech outside the task.
- An item-set family (fluency, item generation) asks for a list of open-vocabulary items. Its lexical words are
  split into utterances at pauses (:func:`utterances`) and each utterance is an item or a phrase
  (:func:`is_phrase`). An item utterance's words are the task's own content whatever they spell; a phrase's
  words, less any member of the declared category, are speech outside the task. The share of utterances that
  are phrases is ``phrase_share``.

It is read over the whole file. The owning branch stores the reading; VERDICT reviews on it and the fold's mask
plan masks it. The bounds are in ``data/task_speech.yaml``; the design is
``specs/20261007-task-events-in-background/design.md`` ("Speech outside the task", "Phrases instead of items").
"""

from __future__ import annotations

import functools
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from senselab.audio.workflows.triage.nodes.branches import AIRWAY_EXPECTATIONS, VOICE_EXPECTATIONS
from senselab.audio.workflows.triage.nodes.common import consensus_words, word_hull, write_measurement
from senselab.audio.workflows.triage.residue import is_function_word, is_non_lexical, token_key
from senselab.audio.workflows.triage.task_content import task_content_ids, task_content_parameters, task_events
from senselab.utils.prov_store import Entity, ProvStore

if TYPE_CHECKING:
    from senselab.audio.data_structures import AudioHints
    from senselab.audio.workflows.triage.config import TriageConfig

TASK_SPEECH_PATH = Path(__file__).parent / "data" / "task_speech.yaml"

TASK_SPEECH_READING = "task_speech_reading"
"""The measurement holding a task's reading of the speech outside it."""


@functools.cache
def task_speech_parameters() -> dict[str, Any]:
    """The parameters of ``data/task_speech.yaml``.

    Returns:
        The parsed mapping.
    """
    return dict(yaml.safe_load(TASK_SPEECH_PATH.read_text()) or {})


def non_lexical_family(family: str | None) -> bool:
    """Whether a declared family asks for a sound rather than words.

    Args:
        family: The declared family, or None.

    Returns:
        True for an airway or voice family.
    """
    return bool(family) and (family in AIRWAY_EXPECTATIONS or family in VOICE_EXPECTATIONS)


def item_set_family(family: str | None) -> bool:
    """Whether a declared family elicits open-vocabulary items (``task_content.yaml`` ``open_vocabulary_families``).

    Args:
        family: The declared family, or None.

    Returns:
        True for a fluency or item-generation family.
    """
    return bool(family) and family in set(task_content_parameters().get("open_vocabulary_families") or ())


def reads_task_speech(family: str | None) -> bool:
    """Whether a family's speech outside the task is read.

    Args:
        family: The declared family, or None.

    Returns:
        True for a non-lexical or an item-set family.
    """
    return non_lexical_family(family) or item_set_family(family)


def words_min_for(family: str | None) -> int | None:
    """The speech words outside the task at or over which a non-lexical recording is reviewed and masked.

    Args:
        family: The declared family, or None.

    Returns:
        ``words_min`` for a non-lexical family, None otherwise.
    """
    return int(task_speech_parameters()["words_min"]) if non_lexical_family(family) else None


def phrase_share_bound(family: str | None) -> float | None:
    """The ``phrase_share`` at or over which an item-set recording is reviewed and its phrases masked.

    Args:
        family: The declared family, or None.

    Returns:
        ``phrase_share_review`` for an item-set family, None otherwise.
    """
    return float(task_speech_parameters()["phrase_share_review"]) if item_set_family(family) else None


@dataclass(frozen=True)
class TaskSpeech:
    """The lexical words of a recording that are not the task's own content.

    Attributes:
        lexical_n: The lexical consensus words, over the whole file.
        task_text_ids: Those that are the voice family's declared task text.
        task_content_ids: Those that are the task's own content: a misreading of its events, an item utterance's
            word, or a category member inside a phrase.
        word_ids: The rest, the speech outside the task, in stream order.
        untimed_ids: Those of them with no usable timing, which no mask can place.
        runs: Each run of adjacent timed speech words: ``(start, end, words)``.
        item_extent: The item utterances' hull, for an item-set family where one was read; None otherwise.
        utterances: An item-set family's utterances, ``(start, end, words, phrase)``, in stream order; times
            are None for an utterance of untimed words only.
        phrase_words_n: The words of its phrase utterances.
    """

    lexical_n: int = 0
    task_text_ids: tuple[str, ...] = ()
    task_content_ids: tuple[str, ...] = ()
    word_ids: tuple[str, ...] = ()
    untimed_ids: tuple[str, ...] = ()
    runs: tuple[tuple[float, float, int], ...] = ()
    item_extent: tuple[float, float] | None = None
    utterances: tuple[tuple[float | None, float | None, int, bool], ...] = ()
    phrase_words_n: int = 0

    @property
    def words_n(self) -> int:
        """The speech words outside the task."""
        return len(self.word_ids)

    @property
    def off_task_fraction(self) -> float | None:
        """The speech words outside the task over every lexical word; None with no lexical word."""
        return round(self.words_n / self.lexical_n, 3) if self.lexical_n else None

    @property
    def phrase_utterances_n(self) -> int:
        """The utterances that are phrases."""
        return sum(1 for utterance in self.utterances if utterance[3])

    @property
    def phrase_share(self) -> float | None:
        """The phrase utterances over every utterance; None with no utterance."""
        return round(self.phrase_utterances_n / len(self.utterances), 3) if self.utterances else None

    @property
    def phrase_word_share(self) -> float | None:
        """The phrase utterances' words over every lexical word; None with no utterance."""
        return round(self.phrase_words_n / self.lexical_n, 3) if self.utterances and self.lexical_n else None

    def record(self) -> dict[str, Any]:
        """The reading, for the store: counts, times and word ids, never a word's text.

        Returns:
            JSON-ready attributes.
        """
        return {
            "words_n": self.words_n,
            "untimed_n": len(self.untimed_ids),
            "lexical_n": self.lexical_n,
            "off_task_fraction": self.off_task_fraction,
            "task_text_n": len(self.task_text_ids),
            "task_content_n": len(self.task_content_ids),
            "runs_n": len(self.runs),
            "runs": [[round(a, 3), round(b, 3), n] for a, b, n in self.runs],
            "item_extent": None if self.item_extent is None else [round(v, 3) for v in self.item_extent],
            "utterances_n": len(self.utterances),
            "phrase_utterances_n": self.phrase_utterances_n,
            "phrase_words_n": self.phrase_words_n,
            "phrase_share": self.phrase_share,
            "phrase_word_share": self.phrase_word_share,
            "utterances": [
                [None if a is None else round(a, 3), None if b is None else round(b, 3), n, phrase]
                for a, b, n, phrase in self.utterances
            ],
            "word_ids": list(self.word_ids),
            "untimed_ids": list(self.untimed_ids),
            "task_content_ids": list(self.task_content_ids),
        }


def timed(word: Entity) -> bool:
    """Whether a word carries timing a mask can be placed on.

    Args:
        word: A consensus word.

    Returns:
        True where its timing hull has a positive length.
    """
    start, end = word_hull(word)
    return end > start


def task_text_ids(words: Sequence[Entity], family: str | None, task_text: Sequence[str]) -> set[str]:
    """The words of a voice family's own declared text: its instructions, stimulus and expected tokens.

    Args:
        words: The consensus words, in stream order.
        family: The declared family.
        task_text: The task's texts as the recording declares them
            (:func:`~senselab.audio.workflows.triage.nodes.redact.task_texts`).

    Returns:
        The ids of the words matching them, by stem or as a number word (``1`` is ``one``); empty for any
        family other than a voice one.
    """
    if not family or family not in VOICE_EXPECTATIONS:
        return set()
    from senselab.audio.workflows.triage.nodes.redact import task_text_positions  # noqa: PLC0415

    expectation = VOICE_EXPECTATIONS[family]
    texts = [*task_text, *(expectation.tokens or ())]
    keys = {token_key(piece) for text in texts for piece in str(text).split()} - {""}
    surfaces = [str(word.attributes.get("text") or "") for word in words]
    by_stem = task_text_positions(surfaces, texts)
    return {
        word.id for position, word in enumerate(words) if position in by_stem or token_key(surfaces[position]) in keys
    }


def _timing(word: Entity) -> tuple[float, float] | None:
    """A word's consensus timing (its derived extent), or its hull where it has none; None where neither has length."""
    span = (float(word.extent[0]), float(word.extent[1])) if word.extent is not None else word_hull(word)
    return span if span[1] > span[0] else None


def utterances(words: Sequence[Entity], *, gap_s: float) -> list[list[Entity]]:
    """The words split into utterances at pauses.

    Args:
        words: The lexical words, in stream order.
        gap_s: A pause at least this long between one word's consensus timing and the next one's starts an
            utterance.

    Returns:
        The utterances, in stream order. A word with no usable timing stays in the utterance it falls in.
    """
    groups: list[list[Entity]] = []
    last_end: float | None = None
    for word in words:
        span = _timing(word)
        if span is None:
            if groups:
                groups[-1].append(word)
            else:
                groups.append([word])
            continue
        start, end = span
        if groups and (last_end is None or start - last_end < gap_s):
            groups[-1].append(word)
            last_end = end if last_end is None else max(last_end, end)
        else:
            groups.append([word])
            last_end = end
    return groups


def is_phrase(words: Sequence[Entity], *, words_min: int) -> bool:
    """Whether an utterance is a phrase rather than an item: long enough, and holding a closed-class word.

    Args:
        words: The utterance's lexical words.
        words_min: The fewest words a phrase holds.

    Returns:
        True where it holds at least ``words_min`` words and one of them is a closed-class word
        (:func:`~senselab.audio.workflows.triage.residue.is_function_word`).
    """
    return len(words) >= words_min and any(is_function_word(str(w.attributes.get("text") or "")) for w in words)


def item_utterances(words: Sequence[Entity]) -> list[list[Entity]]:
    """An item list's item utterances, under ``data/task_speech.yaml``.

    Args:
        words: The lexical words, in stream order.

    Returns:
        Every utterance that is not a phrase; every utterance where all are phrases.
    """
    p = task_speech_parameters()
    groups = utterances(words, gap_s=float(p["utterance_gap_s"]))
    items = [group for group in groups if not is_phrase(group, words_min=int(p["phrase_words_min"]))]
    return items or groups


def _hull(words: Sequence[Entity]) -> tuple[float, float] | None:
    spans = [word_hull(w) for w in words if timed(w)]
    return (min(a for a, _ in spans), max(b for _, b in spans)) if spans else None


def _speech_runs(words: Sequence[Entity], speech: Sequence[Entity], lexical_ids: set[str]) -> list[list[Entity]]:
    order = {word.id: position for position, word in enumerate(words)}
    runs: list[list[Entity]] = []
    for word in (w for w in speech if timed(w)):
        if runs:
            between = words[order[runs[-1][-1].id] + 1 : order[word.id]]
            if not any(other.id in lexical_ids for other in between):
                runs[-1].append(word)
                continue
        runs.append([word])
    return runs


def task_speech_of(
    store: ProvStore,
    family: str | None,
    *,
    task_text: Sequence[str] = (),
    lexicon_ids: Collection[str] = frozenset(),
    member_ids: Collection[str] = frozenset(),
) -> TaskSpeech:
    """The speech outside the task, read over the whole file.

    Args:
        store: The provenance store, read for the consensus words and the task readings' events.
        family: The declared family.
        task_text: The task's declared texts, exempt in a voice family.
        lexicon_ids: The words of the task's own lexicon.
        member_ids: The words that are members of an item-set family's declared category, never masked inside a
            phrase.

    Returns:
        The reading; empty for a family whose speech outside the task is not read.
    """
    if not reads_task_speech(family):
        return TaskSpeech()
    words = consensus_words(store)
    vocal = non_lexical_family(family)
    lexical = [
        word
        for word in words
        if not word.attributes.get("bracketed")
        and not word.attributes.get("degenerate")
        and not word.attributes.get("unconfirmed")
        and not is_non_lexical(str(word.attributes.get("text") or ""), vocal_task=vocal)
    ]
    item_extent: tuple[float, float] | None = None
    said: tuple[tuple[float | None, float | None, int, bool], ...] = ()
    phrase_words_n = 0
    if vocal:
        text_ids = task_text_ids(words, family, task_text)
        content = task_content_ids(words, task_events(store, family), family=family, lexicon_ids=lexicon_ids)
    else:
        p = task_speech_parameters()
        text_ids = set()
        groups = utterances(lexical, gap_s=float(p["utterance_gap_s"]))
        phrases = [is_phrase(group, words_min=int(p["phrase_words_min"])) for group in groups]
        items = [word for group, phrase in zip(groups, phrases) if not phrase for word in group]
        content = {word.id for word in items} | {
            word.id for group, phrase in zip(groups, phrases) if phrase for word in group if word.id in member_ids
        }
        phrase_words_n = sum(len(group) for group, phrase in zip(groups, phrases) if phrase)
        item_extent = _hull(items)
        said = tuple((*(_hull(group) or (None, None)), len(group), phrase) for group, phrase in zip(groups, phrases))
    speech = [word for word in lexical if word.id not in text_ids and word.id not in content]
    runs_of_speech = _speech_runs(words, speech, {word.id for word in lexical})
    return TaskSpeech(
        lexical_n=len(lexical),
        task_text_ids=tuple(word.id for word in lexical if word.id in text_ids),
        task_content_ids=tuple(word.id for word in lexical if word.id in content and word.id not in text_ids),
        word_ids=tuple(word.id for word in speech),
        untimed_ids=tuple(word.id for word in speech if not timed(word)),
        runs=tuple(
            (min(word_hull(w)[0] for w in run), max(word_hull(w)[1] for w in run), len(run)) for run in runs_of_speech
        ),
        item_extent=item_extent,
        utterances=said,
        phrase_words_n=phrase_words_n,
    )


def category_member_ids(words: Sequence[Entity], family: str | None, hint: AudioHints | None) -> set[str]:
    """The words that are members of an item-set family's declared category.

    Args:
        words: The consensus words, in stream order.
        family: The declared family.
        hint: What the recording was declared to contain, read for a category named per recording.

    Returns:
        The member words' ids; empty for any other family or where the instructions name no category.
    """
    if not item_set_family(family):
        return set()
    from senselab.audio.workflows.triage.nodes.branches import item_category  # noqa: PLC0415
    from senselab.audio.workflows.triage.task_lexicon import category_members  # noqa: PLC0415

    positions = category_members(item_category(str(family), hint), [str(w.attributes.get("text") or "") for w in words])
    return {words[i].id for i in positions}


def write_task_speech(
    store: ProvStore,
    activity: str,
    software: str,
    *,
    family: str,
    config: TriageConfig,
    hint: AudioHints | None,
) -> str | None:
    """Write a task's reading of the speech outside it (:func:`task_speech_of`).

    Args:
        store: The provenance store.
        activity: The owning branch's activity.
        software: The software agent.
        family: The declared family.
        config: The triage configuration, read for the family's task lexicon.
        hint: What the recording was declared to contain, read for the task's texts and category.

    Returns:
        The measurement's id; None for a family whose speech outside the task is not read.
    """
    if not reads_task_speech(family):
        return None
    from senselab.audio.workflows.triage.nodes.redact import task_texts  # noqa: PLC0415
    from senselab.audio.workflows.triage.task_lexicon import task_lexicon  # noqa: PLC0415

    words = consensus_words(store)
    positions = task_lexicon(config, family, hint).positions([str(w.attributes.get("text") or "") for w in words])
    reading = task_speech_of(
        store,
        family,
        task_text=task_texts(hint),
        lexicon_ids={words[i].id for i in positions},
        member_ids=category_member_ids(words, family, hint),
    )
    return write_measurement(
        store, activity, software, name=TASK_SPEECH_READING, signal="plain", attributes=reading.record()
    )
