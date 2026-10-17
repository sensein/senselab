"""Thresholds and lexicons for conversation analysis, read from the packaged ``data/`` profiles."""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, FrozenSet, Tuple

import yaml
from pydantic import BaseModel, ConfigDict, Field

DATA_DIR = Path(__file__).parent / "data"
DEFAULTS_PATH = DATA_DIR / "defaults.yaml"
"""Packaged thresholds, each with its derivation."""
LEXICONS_PATH = DATA_DIR / "lexicons.yaml"
"""Packaged English lexicons."""

Phrase = Tuple[str, ...]


class _Yaml12BoolLoader(yaml.SafeLoader):
    """PyYAML's safe loader with YAML 1.2 booleans: only true/false, so ``yes``/``no``/``on`` stay strings."""


_Yaml12BoolLoader.yaml_implicit_resolvers = {
    key: [r for r in resolvers if r[0] != "tag:yaml.org,2002:bool"]
    for key, resolvers in yaml.SafeLoader.yaml_implicit_resolvers.items()
}
_Yaml12BoolLoader.add_implicit_resolver(
    "tag:yaml.org,2002:bool", re.compile(r"^(?:true|True|TRUE|false|False|FALSE)$"), list("tTfF")
)


def _read_yaml(path: Path) -> Dict[str, Any]:
    return dict(yaml.load(path.read_text(), Loader=_Yaml12BoolLoader) or {})  # noqa: S506


def _strings(values: Any) -> list[str]:  # noqa: ANN401
    """The entries of a lexicon list, refusing anything YAML did not read as a string."""
    out = list(values or ())
    bad = [v for v in out if not isinstance(v, str)]
    if bad:
        raise ValueError(f"lexicon entries must be strings (quote YAML booleans such as yes/no/on): {bad}")
    return [v.lower() for v in out]


def _phrases(values: Any) -> FrozenSet[Phrase]:  # noqa: ANN401
    return frozenset(tuple(v.split()) for v in _strings(values))


class Lexicons(BaseModel):
    """Word lists used by the rule layers. Every entry is a tuple of lower-case tokens.

    Attributes:
        backchannel: Entries a lexically backchannel-like IPU is composed of.
        uh_fillers: Filled pauses of the "uh" family.
        um_fillers: Filled pauses of the "um" family.
        other_fillers: Filled pauses outside both families.
        repetition_exempt: Tokens whose immediate repeats are not disfluencies.
        discourse_markers: Discourse markers.
        hedges: Hedges.
        wh_words: Wh-question heads.
        auxiliaries: Yes-no question heads.
        lead_skip: Tokens skipped before head-word rules.
        directive_phrases: Phrases that make a segment a directive.
        imperative_verbs: Base verbs that make a segment-initial imperative.
        thanking: Thanking phrases.
        apology: Apology phrases.
        greeting_start: Greeting openers.
        closing: Closing phrases.
        agreement_phrases: Phrases signalling agreement anywhere in a segment.
        agreement_tokens: Tokens that alone constitute agreement after a question.
        disagreement_phrases: Phrases signalling disagreement at segment start.
        politeness: Politeness-strategy lexicons, by strategy name.
        pronouns: Pronoun lexicons keyed ``i``, ``we``, ``you``.
        stopwords: Function words excluded from lexical alignment.
    """

    model_config = ConfigDict(frozen=True)

    backchannel: FrozenSet[Phrase]
    uh_fillers: FrozenSet[str]
    um_fillers: FrozenSet[str]
    other_fillers: FrozenSet[str]
    repetition_exempt: FrozenSet[str]
    discourse_markers: FrozenSet[Phrase]
    hedges: FrozenSet[Phrase]
    wh_words: FrozenSet[str]
    auxiliaries: FrozenSet[str]
    lead_skip: FrozenSet[str]
    directive_phrases: FrozenSet[Phrase]
    imperative_verbs: FrozenSet[str]
    thanking: FrozenSet[Phrase]
    apology: FrozenSet[Phrase]
    greeting_start: FrozenSet[Phrase]
    closing: FrozenSet[Phrase]
    agreement_phrases: FrozenSet[Phrase]
    agreement_tokens: FrozenSet[Phrase]
    disagreement_phrases: FrozenSet[Phrase]
    politeness: Dict[str, FrozenSet[Phrase]]
    pronouns: Dict[str, FrozenSet[str]]
    stopwords: FrozenSet[str]

    @property
    def fillers(self) -> FrozenSet[str]:
        """Every filled-pause form."""
        return self.uh_fillers | self.um_fillers | self.other_fillers


@lru_cache(maxsize=1)
def load_default_lexicons() -> Lexicons:
    """Read the packaged ``data/lexicons.yaml``.

    Returns:
        The packaged lexicons.
    """
    raw = _read_yaml(LEXICONS_PATH)

    def words(values: Any) -> FrozenSet[str]:  # noqa: ANN401
        return frozenset(_strings(values))

    fillers = raw["fillers"]
    return Lexicons(
        backchannel=_phrases(raw["backchannel"]),
        uh_fillers=words(fillers["uh_family"]),
        um_fillers=words(fillers["um_family"]),
        other_fillers=words(fillers["other_family"]),
        repetition_exempt=words(raw["repetition_exempt"]),
        discourse_markers=_phrases(raw["discourse_markers"]),
        hedges=_phrases(raw["hedges"]),
        wh_words=words(raw["wh_words"]),
        auxiliaries=words(raw["auxiliaries"]),
        lead_skip=words(raw["lead_skip"]),
        directive_phrases=_phrases(raw["directive_phrases"]),
        imperative_verbs=words(raw["imperative_verbs"]),
        thanking=_phrases(raw["thanking"]),
        apology=_phrases(raw["apology"]),
        greeting_start=_phrases(raw["greeting_start"]),
        closing=_phrases(raw["closing"]),
        agreement_phrases=_phrases(raw["agreement_phrases"]),
        agreement_tokens=_phrases(raw["agreement_tokens"]),
        disagreement_phrases=_phrases(raw["disagreement_phrases"]),
        politeness={k: _phrases(v) for k, v in raw["politeness"].items()},
        pronouns={k: words(v) for k, v in raw["pronouns"].items()},
        stopwords=words(raw["stopwords"]),
    )


@lru_cache(maxsize=1)
def _default_thresholds() -> Dict[str, float]:
    raw = _read_yaml(DEFAULTS_PATH)
    return {key: entry["value"] for key, entry in raw["thresholds"].items()}


def _threshold(name: str) -> Any:  # noqa: ANN401
    return Field(default_factory=lambda: _default_thresholds()[name])


class ConversationAnalysisConfig(BaseModel):
    """Every threshold and lexicon the analysis uses. Defaults come from ``data/defaults.yaml``.

    Attributes:
        ipu_gap_s: Intra-speaker silence below which words join one inter-pausal unit (IPU).
        no_gap_no_overlap_s: ``|FTO|`` below which a floor transfer is no-gap-no-overlap.
        backchannel_max_duration_s: Longest IPU that can be a backchannel.
        backchannel_max_words: Most words a backchannel IPU can hold.
        backchannel_floor_window_s: Another speaker must speak within this long after a backchannel
            ends (or still be speaking) for it not to have taken the floor.
        terminal_overlap_window_s: An overlap starting within this long of the end of the current
            speaker's IPU is terminal (non-competitive).
        interruption_yield_window_s: An interrupted speaker who stops within this long of the overlap
            onset, outlasted by the interrupter, has yielded (successful interruption).
        silent_pause_min_s: Shortest silence counted as a pause before/after a filler.
        prolongation_z: Per-character duration z-score above which a word is a prolongation.
        prolongation_min_words: Fewest timed non-filler words a speaker needs for prolongation z-scores.
        repetition_max_ngram: Longest token n-gram checked for immediate repetition.
        lexicons: Word lists for the rule layers.
    """

    model_config = ConfigDict(frozen=True)

    ipu_gap_s: float = _threshold("ipu_gap_s")
    no_gap_no_overlap_s: float = _threshold("no_gap_no_overlap_s")
    backchannel_max_duration_s: float = _threshold("backchannel_max_duration_s")
    backchannel_max_words: int = _threshold("backchannel_max_words")
    backchannel_floor_window_s: float = _threshold("backchannel_floor_window_s")
    terminal_overlap_window_s: float = _threshold("terminal_overlap_window_s")
    interruption_yield_window_s: float = _threshold("interruption_yield_window_s")
    silent_pause_min_s: float = _threshold("silent_pause_min_s")
    prolongation_z: float = _threshold("prolongation_z")
    prolongation_min_words: int = _threshold("prolongation_min_words")
    repetition_max_ngram: int = _threshold("repetition_max_ngram")
    lexicons: Lexicons = Field(default_factory=load_default_lexicons)
