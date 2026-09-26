"""Measure the three PII-redaction artefacts over a completed triage corpus. Read-only.

Emits counts, categories and distance histograms only. No transcript token and no detected surface
appears in the output, so the result can be committed.

Three reductions, one pass over each run directory:

1. **The withhold census** — every REDACT verdict's outcome and which branch of its ``why`` chain
   fired, per family, so the population the LLM reviewer would newly see can be sized and named.

2. **The near-match fit.** ``derivatives/stimulus_alignment.npz`` holds an independent instrument:
   the stimulus aligner has already decided, per consensus word, whether it sits on a declared
   stimulus position (``realised``/``substituted``) or nowhere in the stimulus (``unexpected``).
   That judgment is the oracle; REDACT's exact-match exemption rule is the instrument being fitted
   against it.

   * positives — a live ``pii`` finding every one of whose covered words the aligner placed on a
     stimulus position. The stimulus accounts for it; exact matching may still redact it.
   * negatives — a live ``pii`` finding every one of whose covered words the aligner placed
     nowhere in the stimulus, *plus* every finding in the corpus evaluated against a donor
     stimulus it has nothing to do with (:data:`DONOR`). Admitting either is a false admit.

   For each finding the script reports the smallest tolerance at which REDACT's own predicate —
   the covered keys occurring as a contiguous run inside one declared unit — would admit it, under
   a grid of length-gated rules.

3. **The expected-names census** — per family, the live ``pii`` findings, their ``in_stimulus``
   tri-state, and for the nominated family how near each finding falls to that family's declared
   cast.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from senselab.audio.workflows.audio_analysis.harmonize import normalise_token

# The Cinderella retelling's cast, as task metadata: the proper nouns a faithful performance of
# `cinderella-story` is expected to contain. Not derived from any recording.
CINDERELLA_CAST = (
    "cinderella",
    "cinderellas",
    "ella",
    "cinder",
    "prince",
    "princess",
    "charming",
    "godmother",
    "fairy",
    "stepmother",
    "stepmothers",
    "stepsister",
    "stepsisters",
    "anastasia",
    "drizella",
    "tremaine",
    "gus",
    "jaq",
    "lucifer",
    "bruno",
    "major",
    "duke",
    "king",
    "queen",
)

# The Rainbow Passage (Fairbanks, 1960), a public clinical stimulus and itself a task in this
# corpus. Used only as a *donor*: a haystack a finding has nothing to do with, so that any
# near-match admission against it is unambiguously a false admit.
DONOR = (
    "When the sunlight strikes raindrops in the air, they act as a prism and form a rainbow. "
    "The rainbow is a division of white light into many beautiful colors. These take the shape "
    "of a long round arch, with its path high above, and its two ends apparently beyond the "
    "horizon. There is, according to legend, a boiling pot of gold at one end. People look, but "
    "no one ever finds it. When a man looks for something beyond his reach, his friends say he "
    "is looking for the pot of gold at the end of the rainbow."
)

MAX_DISTANCE = 8
_TASK = re.compile(r"_task-(?P<task>[^_]+)_")
_TRAILING_INDEX = re.compile(r"(?:-\d+)+$")

# The rule grid. `(l1, l2)` reads: a token pair of normalised length below `l1` must match exactly,
# one of length below `l2` may differ by one edit, one at or above `l2` may differ by two. `l1=99`
# is today's shipped rule, exact matching everywhere.
RULES: tuple[tuple[int, int], ...] = tuple(
    (l1, l2) for l1 in (3, 4, 5, 6, 7, 8, 99) for l2 in (6, 7, 8, 9, 10, 12, 99) if l2 >= l1
)


def rule_id(rule: tuple[int, int]) -> str:
    """A rule's name in the output tables.

    Args:
        rule: The ``(l1, l2)`` pair.

    Returns:
        The name.
    """
    return f"l1={rule[0]},l2={rule[1]}"


def tolerance(rule: tuple[int, int], length: int) -> int:
    """How many edits a token pair of this length may differ by under this rule.

    Args:
        rule: The ``(l1, l2)`` pair.
        length: The longer of the two normalised tokens.

    Returns:
        0, 1 or 2.
    """
    l1, l2 = rule
    if length < l1:
        return 0
    if length < l2:
        return 1
    return 2


def levenshtein(a: str, b: str, *, cap: int = MAX_DISTANCE) -> int:
    """Edit distance between two strings, saturating at ``cap``.

    Args:
        a: One string.
        b: The other.
        cap: The value returned once the distance is known to reach it.

    Returns:
        The Levenshtein distance, or ``cap`` when it is at least ``cap``.
    """
    if a == b:
        return 0
    if cap <= 0 or abs(len(a) - len(b)) >= cap:
        return cap
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb)))
        if min(current) >= cap:
            return cap
        previous = current
    return min(previous[-1], cap)


def admits(rule: tuple[int, int], keys: Sequence[str], units: Sequence[Sequence[str]]) -> bool:
    """Whether REDACT's exemption predicate, widened by ``rule``, would admit these keys.

    The predicate is the shipped one: the covered words' keys must occur in order and contiguously
    inside one declared unit. Only the per-token comparison is widened.

    Args:
        rule: The ``(l1, l2)`` pair.
        keys: The covered words' normalised keys, in stream order.
        units: Each declared unit's normalised tokens, in order.

    Returns:
        Whether some unit carries the run.
    """
    if not keys:
        return False
    for tokens in units:
        if len(keys) > len(tokens):
            continue
        for offset in range(len(tokens) - len(keys) + 1):
            window = tokens[offset : offset + len(keys)]
            if all(
                levenshtein(key, token, cap=tolerance(rule, max(len(key), len(token))) + 1)
                <= tolerance(rule, max(len(key), len(token)))
                for key, token in zip(keys, window)
            ):
                return True
    return False


def donor_units() -> list[list[str]]:
    """The donor stimulus, split into sentence units and normalised the way a declared one is.

    Returns:
        One list of keys per sentence.
    """
    units = []
    for sentence in re.split(r"[.!?]", DONOR):
        tokens = [normalise_token(token) for token in sentence.split()]
        kept = [token for token in tokens if token]
        if kept:
            units.append(kept)
    return units


def _family(run: Path) -> str:
    """The declared task family, from the BIDS stem the run directory is named after.

    Args:
        run: The run directory.

    Returns:
        The family, or ``"unknown"``.
    """
    match = _TASK.search(f"{run.name}_")
    return "unknown" if match is None else _TRAILING_INDEX.sub("", match.group("task").lower())


def _ground(why: str) -> str:
    """Which branch of REDACT's ``why`` chain a verdict's prose came from.

    Args:
        why: The verdict's ``why``.

    Returns:
        A short, closed label.
    """
    if why.startswith("the store's pii scan is incomplete"):
        return "scan_incomplete"
    if why.startswith("the re-scan over the redacted text is incomplete"):
        return "verify_incomplete"
    if why.startswith("verification found pii on the redacted transcript"):
        return "outstanding"
    if why.startswith("every finding redacted except"):
        return "passed_with_exemptions"
    if why.startswith("every finding redacted"):
        return "passed"
    return "other"


def _invalidated(records: list[dict[str, Any]]) -> set[str]:
    """The ids the store invalidated.

    Args:
        records: The store's records.

    Returns:
        The invalidated ids.
    """
    return {
        str(record.get("source"))
        for record in records
        if record.get("kind") == "relation" and record.get("relation") == "wasInvalidatedBy"
    }


def _alignment(run: Path) -> dict[str, np.ndarray] | None:
    """The stimulus-alignment sidecar, when this run wrote one.

    Args:
        run: The run directory.

    Returns:
        The npz's arrays, or None.
    """
    path = run / "run" / "derivatives" / "stimulus_alignment.npz"
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def measure(run: Path) -> dict[str, Any] | None:  # noqa: C901 — one pass, several independent tallies
    """Reduce one run directory to counters.

    Args:
        run: A completed triage run directory.

    Returns:
        The counters, or None when the run holds no store.
    """
    store_path = run / "run" / "store.jsonl"
    if not store_path.exists():
        return None
    records = [json.loads(line) for line in store_path.open()]
    dead = _invalidated(records)

    redact_outcome: str | None = None
    redact_ground: str | None = None
    exempt_n = 0
    llm_status: str | None = None
    findings: list[dict[str, Any]] = []
    words: dict[int, str] = {}
    word_extent: dict[int, tuple[float, float]] = {}
    scan_ran: bool | None = None
    for record in records:
        if record.get("id") in dead:
            continue
        attributes = record.get("attributes") or {}
        prov_type = record.get("prov_type")
        if prov_type == "verdict" and attributes.get("node") == "REDACT":
            redact_outcome = str(attributes.get("outcome"))
            redact_ground = _ground(str(attributes.get("why") or ""))
            exempt_n = int(attributes.get("expected_exempt_n") or 0)
        elif prov_type == "measurement" and attributes.get("name") == "redaction_llm_annotation":
            llm_status = str(attributes.get("status"))
        elif prov_type == "measurement" and attributes.get("name") == "pii_scan":
            if attributes.get("scanned") is False:
                scan_ran = False
            elif scan_ran is None:
                scan_ran = True
        elif prov_type == "pii":
            extent = record.get("extent")
            findings.append(
                {
                    "category": str(attributes.get("category")),
                    "in_stimulus": attributes.get("in_stimulus"),
                    "extent": None if extent is None else (float(extent[0]), float(extent[1])),
                }
            )
        elif prov_type == "word":
            index = int(attributes["index"])
            words[index] = str(attributes.get("text") or "")
            extent = record.get("extent")
            if extent is not None:
                word_extent[index] = (float(extent[0]), float(extent[1]))

    out: dict[str, Any] = {
        "family": _family(run),
        "redact_outcome": redact_outcome,
        "redact_ground": redact_ground,
        "exempt_n": exempt_n,
        "llm_status": llm_status,
        "scan_ran": scan_ran,
        "n_findings": len(findings),
        "in_stimulus": Counter("null" if f["in_stimulus"] is None else str(bool(f["in_stimulus"])) for f in findings),
        "finding_categories": Counter(f["category"] for f in findings),
        "substitution_distance": Counter(),
        "class_totals": Counter(),
        "admitted": Counter(),
        "donor_admitted": Counter(),
        "cast_distance": Counter(),
        "n_expected_tokens": 0,
        "has_alignment": False,
    }

    donor = donor_units()

    def covered_keys(finding: dict[str, Any]) -> list[str]:
        if finding["extent"] is None:
            return []
        start, end = finding["extent"]
        order = sorted(i for i, (s, e) in word_extent.items() if s < end and start < e)
        return [normalise_token(words.get(i, "")) for i in order]

    for finding in findings:
        keys = covered_keys(finding)
        if not keys or not all(keys):
            continue
        out["class_totals"]["donor_eligible"] += 1
        out["class_totals"][f"donor_eligible:{finding['category']}"] += 1
        for rule in RULES:
            if admits(rule, keys, donor):
                out["donor_admitted"][rule_id(rule)] += 1
                out["donor_admitted"][(rule_id(rule), finding["category"])] += 1

    data = _alignment(run)
    if data is not None:
        out["has_alignment"] = True
        order = np.argsort(data["expected_index"])
        keys_all = [normalise_token(str(k)) for k in data["expected_key"][order].tolist()]
        reads = [normalise_token(str(r)) for r in data["expected_read"][order].tolist()]
        realisation = [str(r) for r in data["realisation"][order].tolist()]
        word_index = [int(i) for i in data["expected_word_index"][order].tolist()]
        unit_of = [int(u) for u in data["expected_unit"][order].tolist()]
        out["n_expected_tokens"] = len(keys_all)
        units: dict[int, list[str]] = defaultdict(list)
        for unit, key in zip(unit_of, keys_all):
            if key:
                units[unit].append(key)
        declared = [tokens for _, tokens in sorted(units.items())]
        for key, read, state in zip(keys_all, reads, realisation):
            if state == "substituted" and key and read:
                out["substitution_distance"][(min(len(key), 20), levenshtein(key, read))] += 1
        on_stimulus = {i for i, state in zip(word_index, realisation) if state in ("realised", "substituted") and i >= 0}
        off_stimulus = {int(i) for i in data["unexpected_word_index"].tolist()}
        for finding in findings:
            if finding["extent"] is None:
                continue
            start, end = finding["extent"]
            order_i = sorted(i for i, (s, e) in word_extent.items() if s < end and start < e)
            keys = [normalise_token(words.get(i, "")) for i in order_i]
            if not keys or not all(keys):
                continue
            if all(i in on_stimulus for i in order_i):
                klass = "on_stimulus"
            elif all(i in off_stimulus for i in order_i):
                klass = "off_stimulus"
            else:
                klass = "mixed"
            out["class_totals"][klass] += 1
            for rule in RULES:
                if admits(rule, keys, declared):
                    out["admitted"][(klass, rule_id(rule))] += 1

    if out["family"].startswith("cinderella"):
        for finding in findings:
            keys = covered_keys(finding)
            if not keys or not all(keys):
                continue
            best = MAX_DISTANCE
            for key in keys:
                for name in CINDERELLA_CAST:
                    best = min(best, levenshtein(key, name, cap=best if best else 1))
            out["cast_distance"][(finding["category"], len(keys), best)] += 1

    return out


def _safe(run: Path) -> dict[str, Any] | None:
    """:func:`measure`, with a failure reduced to None.

    Args:
        run: A run directory.

    Returns:
        The counters, or None.
    """
    try:
        return measure(run)
    except Exception as error:  # noqa: BLE001 — one bad run must not end the reduction
        print(f"skip {run.name}: {type(error).__name__}: {error}", file=sys.stderr)
        return None


def _flatten(counter: Counter) -> dict[str, int]:
    """A counter with tuple keys, as JSON-safe strings.

    Args:
        counter: The counter.

    Returns:
        The flattened mapping.
    """
    return {
        "|".join(str(part) for part in key) if isinstance(key, tuple) else str(key): value
        for key, value in sorted(counter.items(), key=lambda item: str(item[0]))
    }


def _merge(results: Iterable[dict[str, Any] | None]) -> dict[str, Any]:
    """Fold the per-run counters into the corpus tables.

    Args:
        results: One entry per run.

    Returns:
        The corpus tables, JSON-ready.
    """
    per_family: dict[str, Counter] = defaultdict(Counter)
    in_stimulus_by_family: dict[str, Counter] = defaultdict(Counter)
    ground_by_family: dict[str, Counter] = defaultdict(Counter)
    substitution: Counter = Counter()
    class_totals: Counter = Counter()
    admitted: Counter = Counter()
    donor_admitted: Counter = Counter()
    cast: Counter = Counter()
    categories: Counter = Counter()
    n_runs = 0
    for result in results:
        if result is None:
            continue
        n_runs += 1
        family = str(result["family"])
        counter = per_family[family]
        counter["runs"] += 1
        counter["findings"] += int(result["n_findings"])
        counter["exempt"] += int(result["exempt_n"])
        counter["expected_tokens"] += int(result["n_expected_tokens"])
        counter["with_alignment"] += 1 if result["has_alignment"] else 0
        if result["redact_outcome"] is not None:
            counter["redact_runs"] += 1
            counter[f"redact_{result['redact_outcome']}"] += 1
            ground_by_family[family][str(result["redact_ground"])] += 1
        if result["llm_status"] is not None:
            counter[f"llm_{result['llm_status']}"] += 1
        if result["scan_ran"] is not None:
            counter[f"scan_{result['scan_ran']}"] += 1
        in_stimulus_by_family[family].update(result["in_stimulus"])
        categories.update(result["finding_categories"])
        substitution.update(result["substitution_distance"])
        class_totals.update(result["class_totals"])
        admitted.update(result["admitted"])
        donor_admitted.update(result["donor_admitted"])
        cast.update(result["cast_distance"])

    return {
        "n_runs": n_runs,
        "rules": [rule_id(rule) for rule in RULES],
        "per_family": {family: dict(counter) for family, counter in sorted(per_family.items())},
        "in_stimulus_by_family": {family: dict(counter) for family, counter in sorted(in_stimulus_by_family.items())},
        "redact_ground_by_family": {family: dict(counter) for family, counter in sorted(ground_by_family.items())},
        "finding_categories": dict(categories),
        "substitution_len_distance": _flatten(substitution),
        "finding_class_totals": dict(class_totals),
        "admitted_by_class_rule": _flatten(admitted),
        "donor_admitted_by_rule": _flatten(donor_admitted),
        "cinderella_cast_distance": _flatten(cast),
    }


def main() -> None:
    """Run the reduction and write the JSON tables."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()
    runs = [path.parent.parent for path in args.corpus.glob("sub-*/ses-*/*/run/store.jsonl")]
    print(f"{len(runs)} runs", file=sys.stderr)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(_safe, runs, chunksize=4))
    args.out.write_text(json.dumps(_merge(results), indent=2, sort_keys=True))
    print(f"wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
