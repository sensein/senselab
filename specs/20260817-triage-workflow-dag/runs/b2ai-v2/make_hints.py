#!/usr/bin/env python3
"""Emit one triage hint file per b2ai recording, from the task token in its filename.

    # from a directory of wavs (what a cluster job does)
    uv run python specs/20260817-triage-workflow-dag/runs/b2ai-v2/make_hints.py \
        --dir /orcd/data/satra/002/.../sub-<uuid>/ --out hints/

    # from a listing
    ls sub-*/**/*.wav | uv run python .../make_hints.py --list - --out hints/

    # reproduce the 28-file set this file carries inline, and check the rules against v1
    uv run python .../make_hints.py --b2ai-28 --out hints/
    uv run python .../make_hints.py --selftest

A hint is what the acquisition protocol DECLARED, never a measurement. Each output is an
``AudioHints`` mapping — ``may_contain`` and ``metadata`` — which ``scripts/triage_audio.py --hint``
reads directly. ``targeted_speaker_count`` is deliberately never written: it is an intent, the v2
fold reads nothing from it, and a number nobody counted does not belong in a file the graph reads.

Two tag paths reach ROUTING, and both go through ``routing.hint_kind_map`` in the config override:
the ``may_contain`` tags, and ``metadata.speech_type``, which ``routing._declared_tags`` appends to
the tag list. So every tag written here is checked against that map before anything is written, and
an unmapped tag is a hard failure rather than a hint that reads as declared while forcing nothing.

The rules are the b2ai task vocabulary, not a fit. Where the ledger's summary of the rules and the
v1 hint table disagree, the v1 table governs and ``--selftest`` is what proves the rules reproduce
it; see ``README.md``, "The one rule that differs from the ledger's summary".
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Callable, Iterable, NamedTuple, Sequence

import yaml  # type: ignore[import-untyped]

HERE = Path(__file__).resolve().parent
DEFAULT_OVERRIDE = HERE / "override.yaml"

REGISTRY = "b2aiprep@ebc2a14e task_registry v1"
"""The registry string the v1 hint table stamped on every file it covered. Copied, not derived."""


class Rule(NamedTuple):
    """One task-token rule.

    Attributes:
        name: What the rule is called, for ``--explain`` and for error messages.
        matches: Whether this rule claims a task token.
        tags: The ``may_contain`` tags, in the order the v1 table wrote them.
        speech_type: The ``metadata.speech_type`` value, which ROUTING reads as a tag too.
        task_id: The registry's task id, when the v1 table recorded one for this rule's tokens.
    """

    name: str
    matches: Callable[[str], bool]
    tags: tuple[str, ...]
    speech_type: str
    task_id: str | None


def _ends_with_cough(token: str) -> bool:
    """Whether the token's TRAILING element is a cough, not merely the protocol's name.

    Args:
        token: The task token, e.g. ``Respiration-and-cough-Cough-1``.

    Returns:
        True for ``...-Cough``, ``...-Cough-2`` and ``...-HardCough``, False for
        ``Respiration-and-cough-Breath-1``. This is the whole reason the rule is a trailing match:
        every respiration file carries "cough" in the protocol's own name, so a substring test tags
        five breath files as coughs. The capital C is what preserves that discrimination -- the
        final hyphen-segment must END in ``Cough``, so ``HardCough`` is claimed while the lowercase
        ``cough`` inside the protocol prefix never is.
    """
    return re.search(r"(?:^|-)[A-Za-z]*Cough(?:-\d+)?$", token) is not None


def _contains_breath(token: str) -> bool:
    """Whether the token names a breath task.

    Args:
        token: The task token.

    Returns:
        True for ``Breath-1``, ``FiveBreaths-3``, ``ThreeQuickBreaths-2``.
    """
    return "breath" in token.casefold()


def _is(*tokens: str) -> Callable[[str], bool]:
    """A matcher for an exact token, ignoring a trailing repetition index.

    Args:
        tokens: The task tokens this rule claims.

    Returns:
        A predicate that strips a trailing ``-<n>`` and compares casefolded.
    """
    wanted = {token.casefold() for token in tokens}

    def _match(token: str) -> bool:
        return re.sub(r"-\d+$", "", token).casefold() in wanted

    return _match


def _starts(prefix: str) -> Callable[[str], bool]:
    """A matcher for a token prefix.

    Args:
        prefix: The prefix, compared casefolded.

    Returns:
        A predicate.
    """
    return lambda token: token.casefold().startswith(prefix.casefold())


def _contains(needle: str) -> Callable[[str], bool]:
    """A matcher for a substring.

    Args:
        needle: The substring, compared casefolded.

    Returns:
        A predicate.
    """
    return lambda token: needle.casefold() in token.casefold()


RULES: tuple[Rule, ...] = (
    # ORDER IS LOAD-BEARING for the first three: every respiration-and-cough file carries both
    # "cough" and the protocol prefix, so the trailing-Cough test must run before the breath test,
    # and both must run before anything that would match the protocol name.
    Rule("trailing cough", _ends_with_cough, ("cough", "airway"), "non-lexical", "adult.respiration-and-cough.v2"),
    Rule("breath", _contains_breath, ("breathe", "airway"), "non-lexical", "adult.respiration-and-cough.v2"),
    Rule(
        "diadochokinesis buttercup",
        lambda token: _starts("Diadochokinesis")(token) and token.casefold().endswith("buttercup"),
        ("speech", "voice"),
        "non-lexical",
        "adult.diadochokinesis.v2",
    ),
    Rule("diadochokinesis", _starts("Diadochokinesis"), ("voice",), "non-lexical", "adult.diadochokinesis.v2"),
    Rule(
        "prolonged vowel",
        _contains("vowel"),
        ("sustained-vowel", "phonation", "voice"),
        "non-lexical",
        "adult.prolonged-vowel",
    ),
    Rule(
        "maximum phonation time",
        _starts("Maximum-phonation-time"),
        ("sustained-vowel", "phonation", "voice"),
        "non-lexical",
        "adult.maximum-phonation-time.v2",
    ),
    Rule("glides", _starts("Glides"), ("phonation", "voice"), "non-lexical", "adult.glides"),
    Rule("loudness", _starts("Loudness"), ("phonation", "voice"), "non-lexical", "adult.loudness.v2"),
    # Read tasks. `Passage` covers Rainbow and any other passage the protocol carries; the explicit
    # tokens are here so a passage that is not named "-Passage" still resolves.
    Rule("rainbow passage", _is("Rainbow-Passage"), ("read-speech", "speech"), "read", "adult.rainbow-passage"),
    Rule("passage (read)", _contains("Passage"), ("read-speech", "speech"), "read", None),
    Rule("reading (read)", _starts("Reading"), ("read-speech", "speech"), "read", None),
    Rule("harvard sentences (read)", _starts("Harvard-Sentences"), ("read-speech", "speech"), "read", None),
    Rule("cape-v sentences (read)", _starts("Cape-V-sentences"), ("read-speech", "speech"), "read", None),
    # Elicited and recalled connected speech. Every one of these is [speech] and nothing else: the
    # v1 table gives them no voice tag, and adding one would force VOICE on connected speech.
    Rule("free speech", _starts("Free-speech"), ("speech",), "elicited", "adult.free-speech.v2"),
    Rule("picture description", _starts("Picture-description"), ("speech",), "elicited", "adult.picture-description"),
    Rule("story recall", _starts("Story-recall"), ("speech",), "recall", "adult.story-recall.v2"),
    Rule("cinderella story", _starts("Cinderella-Story"), ("speech",), "recall", None),
    Rule("productive vocabulary", _starts("Productive-Vocabulary"), ("speech",), "elicited", None),
    Rule("random item generation", _starts("Random-Item-Generation"), ("speech",), "elicited", None),
    Rule("word-colour stroop", _starts("Word-color-Stroop"), ("speech",), "elicited", None),
)
"""The task-token rules, first match wins.

Every entry is the b2ai protocol's own vocabulary read off the task token; none of it is fitted.
``task_id`` is ``None`` for a token the v1 hint table never covered — the registry id is a fact
about the registry, and inventing one for a task nobody recorded would put a fabricated provenance
string into every artifact that reads the hint.
"""


B2AI_28: dict[str, tuple[tuple[str, ...], str, str]] = {
    "Diadochokinesis-KA": (("voice",), "non-lexical", "adult.diadochokinesis.v2"),
    "Diadochokinesis-PA": (("voice",), "non-lexical", "adult.diadochokinesis.v2"),
    "Diadochokinesis-Pataka": (("voice",), "non-lexical", "adult.diadochokinesis.v2"),
    "Diadochokinesis-TA": (("voice",), "non-lexical", "adult.diadochokinesis.v2"),
    "Diadochokinesis-buttercup": (("speech", "voice"), "non-lexical", "adult.diadochokinesis.v2"),
    "Free-speech-1": (("speech",), "elicited", "adult.free-speech.v2"),
    "Free-speech-2": (("speech",), "elicited", "adult.free-speech.v2"),
    "Free-speech-3": (("speech",), "elicited", "adult.free-speech.v2"),
    "Glides-High-to-Low": (("phonation", "voice"), "non-lexical", "adult.glides"),
    "Glides-Low-to-High": (("phonation", "voice"), "non-lexical", "adult.glides"),
    "Loudness": (("phonation", "voice"), "non-lexical", "adult.loudness.v2"),
    "Maximum-phonation-time-1": (
        ("sustained-vowel", "phonation", "voice"),
        "non-lexical",
        "adult.maximum-phonation-time.v2",
    ),
    "Maximum-phonation-time-2": (
        ("sustained-vowel", "phonation", "voice"),
        "non-lexical",
        "adult.maximum-phonation-time.v2",
    ),
    "Maximum-phonation-time-3": (
        ("sustained-vowel", "phonation", "voice"),
        "non-lexical",
        "adult.maximum-phonation-time.v2",
    ),
    "Picture-description": (("speech",), "elicited", "adult.picture-description"),
    "Prolonged-vowel": (("sustained-vowel", "phonation", "voice"), "non-lexical", "adult.prolonged-vowel"),
    "Rainbow-Passage": (("read-speech", "speech"), "read", "adult.rainbow-passage"),
    "Respiration-and-cough-Breath-1": (("breathe", "airway"), "non-lexical", "adult.respiration-and-cough.v2"),
    "Respiration-and-cough-Breath-2": (("breathe", "airway"), "non-lexical", "adult.respiration-and-cough.v2"),
    "Respiration-and-cough-Cough-1": (("cough", "airway"), "non-lexical", "adult.respiration-and-cough.v2"),
    "Respiration-and-cough-Cough-2": (("cough", "airway"), "non-lexical", "adult.respiration-and-cough.v2"),
    "Respiration-and-cough-FiveBreaths-1": (("breathe", "airway"), "non-lexical", "adult.respiration-and-cough.v2"),
    "Respiration-and-cough-FiveBreaths-2": (("breathe", "airway"), "non-lexical", "adult.respiration-and-cough.v2"),
    "Respiration-and-cough-FiveBreaths-3": (("breathe", "airway"), "non-lexical", "adult.respiration-and-cough.v2"),
    "Respiration-and-cough-FiveBreaths-4": (("breathe", "airway"), "non-lexical", "adult.respiration-and-cough.v2"),
    "Respiration-and-cough-ThreeQuickBreaths-1": (
        ("breathe", "airway"),
        "non-lexical",
        "adult.respiration-and-cough.v2",
    ),
    "Respiration-and-cough-ThreeQuickBreaths-2": (
        ("breathe", "airway"),
        "non-lexical",
        "adult.respiration-and-cough.v2",
    ),
    "Story-recall": (("speech",), "recall", "adult.story-recall.v2"),
}
"""The v1 hint table for subject ``17cee767``, verbatim: task token to (tags, speech_type, task_id).

Kept inline so ``--selftest`` can prove :data:`RULES` reproduces the campaign that already ran, and
so the 28-file set can be regenerated without reaching the cluster.
"""

V2_47: dict[str, tuple[tuple[str, ...], str]] = {
    "Cape-V-sentences-(v2)-1": (("read-speech", "speech"), "read"),
    "Cape-V-sentences-(v2)-2": (("read-speech", "speech"), "read"),
    "Cape-V-sentences-(v2)-3": (("read-speech", "speech"), "read"),
    "Cape-V-sentences-(v2)-4": (("read-speech", "speech"), "read"),
    "Cape-V-sentences-(v2)-5": (("read-speech", "speech"), "read"),
    "Cape-V-sentences-(v2)-6": (("read-speech", "speech"), "read"),
    "Caterpillar-Passage": (("read-speech", "speech"), "read"),
    "Diadochokinesis-(v2)-buttercup": (("speech", "voice"), "non-lexical"),
    "Diadochokinesis-(v2)-kuh": (("voice",), "non-lexical"),
    "Diadochokinesis-(v2)-puh": (("voice",), "non-lexical"),
    "Diadochokinesis-(v2)-puhtuhkuh": (("voice",), "non-lexical"),
    "Diadochokinesis-(v2)-tuh": (("voice",), "non-lexical"),
    "Free-speech-(v2)-1": (("speech",), "elicited"),
    "Free-speech-(v2)-2": (("speech",), "elicited"),
    "Free-speech-(v2)-3": (("speech",), "elicited"),
    "Glides-High-to-Low": (("phonation", "voice"), "non-lexical"),
    "Glides-Low-to-High": (("phonation", "voice"), "non-lexical"),
    "Harvard-Sentences-List-49-1": (("read-speech", "speech"), "read"),
    "Harvard-Sentences-List-49-10": (("read-speech", "speech"), "read"),
    "Harvard-Sentences-List-49-2": (("read-speech", "speech"), "read"),
    "Harvard-Sentences-List-49-3": (("read-speech", "speech"), "read"),
    "Harvard-Sentences-List-49-4": (("read-speech", "speech"), "read"),
    "Harvard-Sentences-List-49-5": (("read-speech", "speech"), "read"),
    "Harvard-Sentences-List-49-6": (("read-speech", "speech"), "read"),
    "Harvard-Sentences-List-49-7": (("read-speech", "speech"), "read"),
    "Harvard-Sentences-List-49-8": (("read-speech", "speech"), "read"),
    "Harvard-Sentences-List-49-9": (("read-speech", "speech"), "read"),
    "Harvard-Sentences-List-52-1": (("read-speech", "speech"), "read"),
    "Harvard-Sentences-List-52-10": (("read-speech", "speech"), "read"),
    "Harvard-Sentences-List-52-2": (("read-speech", "speech"), "read"),
    "Harvard-Sentences-List-52-3": (("read-speech", "speech"), "read"),
    "Harvard-Sentences-List-52-4": (("read-speech", "speech"), "read"),
    "Harvard-Sentences-List-52-5": (("read-speech", "speech"), "read"),
    "Harvard-Sentences-List-52-6": (("read-speech", "speech"), "read"),
    "Harvard-Sentences-List-52-7": (("read-speech", "speech"), "read"),
    "Harvard-Sentences-List-52-8": (("read-speech", "speech"), "read"),
    "Harvard-Sentences-List-52-9": (("read-speech", "speech"), "read"),
    "Loudness-(v2)": (("phonation", "voice"), "non-lexical"),
    "Maximum-phonation-time-(v2)-1": (("sustained-vowel", "phonation", "voice"), "non-lexical"),
    "Picture-description-option2": (("speech",), "elicited"),
    "Prolonged-vowel": (("sustained-vowel", "phonation", "voice"), "non-lexical"),
    "Respiration-and-cough-(v2)-Breath": (("breathe", "airway"), "non-lexical"),
    "Respiration-and-cough-(v2)-HardCough": (("cough", "airway"), "non-lexical"),
    "Respiration-and-cough-(v2)-ThreeBreaths": (("breathe", "airway"), "non-lexical"),
    "Respiration-and-cough-(v2)-ThreeBreathsMouth": (("breathe", "airway"), "non-lexical"),
    "Respiration-and-cough-(v2)-ThreeBreathsNose": (("breathe", "airway"), "non-lexical"),
    "Story-recall-(v2)": (("speech",), "recall"),
}
"""Subject ``1f4ea26f``'s 47 task tokens: the (v2) protocol's own vocabulary, token to (tags,
speech_type). Unlike :data:`B2AI_28` this is not a transcription of a table that ran -- it is the
ruling made when the (v2) session was first screened, pinned here so a later edit to RULES cannot
silently reclassify a file. No task_id: the v2 registry strings were not recorded.
"""

COUGH_DISCRIMINATION: tuple[tuple[str, bool], ...] = (
    ("Respiration-and-cough-Cough-1", True),
    ("Respiration-and-cough-Cough-2", True),
    ("Respiration-and-cough-(v2)-HardCough", True),
    ("Respiration-and-cough-Breath-1", False),
    ("Respiration-and-cough-(v2)-Breath", False),
    ("Respiration-and-cough-(v2)-ThreeBreathsMouth", False),
    ("Respiration-and-cough-FiveBreaths-3", False),
    ("Respiration-and-cough", False),
)
"""The cough matcher's boundary, both directions. The trailing segment must END in capital-C
``Cough`` -- so ``HardCough`` is a cough, while the lowercase ``cough`` every respiration file
carries in the protocol's own name is not. Both limbs are pinned: a substring test would tag every
breath file as a cough, and a bare ``-Cough$`` test would miss ``HardCough``.
"""

B2AI_28_SUBJECT = "sub-17cee767-1864-457a-b2ec-446a058a81f8"
B2AI_28_SESSION = "ses-DA790C5A-93FF-432F-A5B6-418C19A4F2BA"


def task_token(filename: str) -> str:
    """The task token in a BIDS-style b2ai filename.

    Args:
        filename: The recording's name, e.g. ``sub-X_ses-Y_task-Rainbow-Passage.wav``.

    Returns:
        The token after ``task-``, without the extension.

    Raises:
        ValueError: If the name carries no ``task-`` element. Guessing here would attach a hint to
            the wrong recording, which is worse than refusing the file.
    """
    stem = Path(filename).name
    for suffix in (".wav", ".flac", ".mp3", ".ogg", ".m4a"):
        if stem.casefold().endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    if "task-" not in stem:
        raise ValueError(f"{filename!r} carries no 'task-' element; a hint cannot be derived from it")
    return stem.split("task-", 1)[1]


def resolve(token: str) -> Rule:
    """The first rule claiming a task token.

    Args:
        token: The task token.

    Returns:
        The rule.

    Raises:
        ValueError: If no rule claims it. An unrecognised task must fail loudly: an empty
            ``may_contain`` reads as "the protocol declared nothing", which is a claim, and it
            forces no branch — so a silently unhinted file is a file this campaign screened with
            less evidence than it had.
    """
    for rule in RULES:
        if rule.matches(token):
            return rule
    raise ValueError(
        f"no rule claims task token {token!r}. Add a Rule to RULES with the tags the protocol "
        f"declares, or re-run with --allow-unknown to write an empty hint and record the token as "
        f"unmapped. Known rules: {', '.join(rule.name for rule in RULES)}"
    )


def hint_for(token: str, rule: Rule) -> dict[str, object]:
    """The AudioHints mapping for one recording.

    Args:
        token: The task token, recorded as a fact about the filename.
        rule: The rule that claimed it.

    Returns:
        The mapping ``scripts/triage_audio.py --hint`` reads. No ``targeted_speaker_count``.
    """
    metadata: dict[str, object] = {"task_token": token, "speech_type": rule.speech_type}
    if rule.task_id is not None:
        metadata["task_id"] = rule.task_id
        metadata["registry"] = REGISTRY
    return {"may_contain": list(rule.tags), "metadata": metadata}


def unknown_hint(token: str) -> dict[str, object]:
    """The mapping written for an unclaimed token under ``--allow-unknown``.

    Args:
        token: The task token.

    Returns:
        A hint declaring nothing, and saying so in its own metadata.
    """
    return {"may_contain": [], "metadata": {"task_token": token, "unmapped_task": True}}


def load_kind_map(override: Path) -> dict[str, str]:
    """The campaign's ``routing.hint_kind_map``, read from the override YAML.

    Args:
        override: The campaign override.

    Returns:
        Tag to kind, casefolded on the key, exactly as ROUTING folds it.

    Raises:
        ValueError: If the override carries no ``routing.hint_kind_map``. Without it the cross-check
            below cannot run, and a hint whose tags reach no kind is a hint that forces nothing.
    """
    values = yaml.safe_load(override.read_text()) or {}
    kind_map = ((values.get("routing") or {}).get("hint_kind_map")) or {}
    if not kind_map:
        raise ValueError(f"{override} carries no routing.hint_kind_map; every tag would be unmapped")
    return {str(tag).casefold(): str(kind) for tag, kind in kind_map.items()}


def check_tags(hint: dict[str, object], kind_map: dict[str, str], where: str) -> None:
    """Refuse a hint carrying a tag the campaign's kind map does not route.

    Args:
        hint: The mapping about to be written.
        kind_map: :func:`load_kind_map`'s result.
        where: The filename, for the message.

    Raises:
        ValueError: If any ``may_contain`` tag is absent from the map. ``metadata.speech_type`` is
            deliberately NOT required to be present: ``non-lexical`` covers DDK, glides, breath and
            cough alike, so the override maps no kind to it on purpose and ROUTING records it as
            unmapped, which is the honest state rather than a defect.
    """
    declared = hint.get("may_contain")
    tags = [str(tag) for tag in declared] if isinstance(declared, list) else []
    missing = [tag for tag in tags if tag.casefold() not in kind_map]
    if missing:
        raise ValueError(
            f"{where}: tag(s) {', '.join(missing)} are not keys of routing.hint_kind_map. The hint "
            "would read as declared while forcing no branch. Add them to the override's "
            "hint_kind_map, or change the rule."
        )


def _filenames(args: argparse.Namespace) -> list[str]:
    """Every recording named on the command line, in a listing, or found in a directory.

    Args:
        args: The parsed command line.

    Returns:
        The filenames, sorted and de-duplicated by basename.

    Raises:
        ValueError: If no source of filenames was given.
    """
    names: list[str] = list(args.files)
    if args.b2ai_28:
        names += [f"{B2AI_28_SUBJECT}_{B2AI_28_SESSION}_task-{token}.wav" for token in B2AI_28]
    if args.list is not None:
        text = sys.stdin.read() if str(args.list) == "-" else Path(args.list).read_text()
        names += [line.strip() for line in text.splitlines() if line.strip()]
    if args.dir is not None:
        names += [str(path) for path in sorted(Path(args.dir).rglob("*.wav"))]
    if not names:
        raise ValueError("nothing to do: give filenames, --list, --dir or --b2ai-28")
    seen: dict[str, str] = {}
    for name in names:
        seen.setdefault(Path(name).name, name)
    return [seen[key] for key in sorted(seen)]


def selftest() -> int:
    """Check that :data:`RULES` reproduces the v1 hint table and the (v2) rulings.

    Covers three things: the 28 v1 rows verbatim, subject ``1f4ea26f``'s 47 (v2) tokens, and the
    cough matcher's boundary in both directions.

    Returns:
        0 when every token's tags, speech_type and task_id match, 1 otherwise.
    """
    failures: list[str] = []
    for token, (tags, speech_type, task_id) in B2AI_28.items():
        rule = resolve(token)
        got = (rule.tags, rule.speech_type, rule.task_id)
        if got != (tags, speech_type, task_id):
            failures.append(f"  {token}: rule {rule.name!r} gave {got}, v1 recorded {(tags, speech_type, task_id)}")
    kind_map = load_kind_map(DEFAULT_OVERRIDE)
    for token in B2AI_28:
        try:
            check_tags(hint_for(token, resolve(token)), kind_map, token)
        except ValueError as error:
            failures.append(f"  {error}")
    for token, (tags, speech_type) in V2_47.items():
        try:
            rule = resolve(token)
        except ValueError as error:
            failures.append(f"  {error}")
            continue
        if (rule.tags, rule.speech_type) != (tags, speech_type):
            failures.append(
                f"  {token}: rule {rule.name!r} gave {(rule.tags, rule.speech_type)}, ruled {(tags, speech_type)}"
            )
        else:
            try:
                check_tags(hint_for(token, rule), kind_map, token)
            except ValueError as error:
                failures.append(f"  {error}")
    for token, expected in COUGH_DISCRIMINATION:
        if _ends_with_cough(token) is not expected:
            failures.append(f"  cough matcher: {token!r} gave {not expected}, expected {expected}")
    for rule in RULES:
        for tag in rule.tags:
            if tag.casefold() not in kind_map:
                failures.append(f"  rule {rule.name!r} emits {tag!r}, absent from routing.hint_kind_map")
    if failures:
        print(f"selftest FAILED ({len(failures)}):", file=sys.stderr)
        print("\n".join(failures), file=sys.stderr)
        return 1
    print(
        f"selftest ok: {len(B2AI_28)} v1 rows reproduced; {len(V2_47)} (v2) tokens match their ruling; "
        f"{len(COUGH_DISCRIMINATION)} cough-boundary cases pinned; every rule tag is a hint_kind_map key"
    )
    return 0


def emit(names: Sequence[str], out: Path, kind_map: dict[str, str], allow_unknown: bool) -> list[str]:
    """Write one hint file per recording.

    Args:
        names: The recordings.
        out: The output directory, created if absent.
        kind_map: The campaign's tag-to-kind map, for the cross-check.
        allow_unknown: Whether an unclaimed task token writes an empty hint instead of raising.

    Returns:
        The task tokens that no rule claimed.

    Raises:
        ValueError: Propagated from :func:`resolve` and :func:`check_tags`.
    """
    out.mkdir(parents=True, exist_ok=True)
    unknown: list[str] = []
    for name in names:
        token = task_token(name)
        try:
            hint = hint_for(token, resolve(token))
        except ValueError:
            if not allow_unknown:
                raise
            unknown.append(token)
            hint = unknown_hint(token)
        check_tags(hint, kind_map, Path(name).name)
        (out / f"{Path(name).name.rsplit('.', 1)[0]}.json").write_text(json.dumps(hint, indent=1) + "\n")
    return unknown


def build_parser() -> argparse.ArgumentParser:
    """The CLI.

    Returns:
        The parser.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="*", help="Recording filenames or paths.")
    parser.add_argument("--dir", type=Path, default=None, help="Directory to scan recursively for *.wav.")
    parser.add_argument("--list", default=None, help="File holding one recording name per line, or '-' for stdin.")
    parser.add_argument("--b2ai-28", action="store_true", help="Add the 28 filenames this file carries inline.")
    parser.add_argument("--out", type=Path, default=None, help="Directory the hint JSONs are written to.")
    parser.add_argument(
        "--override",
        type=Path,
        default=DEFAULT_OVERRIDE,
        help="Campaign override, read for routing.hint_kind_map (default: the one beside this script).",
    )
    parser.add_argument(
        "--allow-unknown",
        action="store_true",
        help="Write an empty hint for a task token no rule claims, instead of refusing.",
    )
    parser.add_argument("--selftest", action="store_true", help="Check the rules against the inline v1 table and exit.")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    """Generate the hint files.

    Args:
        argv: The command line, or None to read ``sys.argv``.

    Returns:
        0 on success, 2 when the arguments could not be resolved, 1 when the selftest failed.
    """
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    if args.selftest:
        return selftest()
    if args.out is None:
        print("ERROR: --out is required unless --selftest", file=sys.stderr)
        return 2
    try:
        kind_map = load_kind_map(args.override)
        names = _filenames(args)
        unknown = emit(names, args.out, kind_map, args.allow_unknown)
    except (OSError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(f"wrote {len(names)} hint files to {args.out}")
    if unknown:
        print(f"WARNING: {len(unknown)} unmapped task token(s), hinted with nothing: {sorted(set(unknown))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
