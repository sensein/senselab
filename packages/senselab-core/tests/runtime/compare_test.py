"""Per-type output comparison: hash, exact and `isclose` within named thresholds, with a field-level diff."""

import itertools
import json
import math

import numpy
import pytest
from senselab_core.kernel.results import Intervals
from senselab_core.kernel.wire import WIRE_FORMATS, wire_form
from senselab_core.runtime.audio_io import READER
from senselab_core.runtime.compare import (
    COMPARISONS,
    FLATTEN_VERSION,
    Fields,
    Threshold,
    Thresholds,
    best_assignment,
    compare,
    compare_fields,
    flatten,
    index_free,
    relabel,
)
from senselab_core.testing.values import samples_of

ATOL = 0.5
NAMES = sorted({r.threshold for t in COMPARISONS.values() for _, r in t.fields if r.threshold is not None})
THRESHOLDS = Thresholds(by_name={n: Threshold(rtol=0.0, atol=ATOL) for n in NAMES})


def _type_key(tag: str) -> str:
    return tag.split(".", 1)[0]


def _leaves(tag: str) -> Fields:
    return flatten({"out": wire_form(samples_of(WIRE_FORMATS[tag])[0], READER)})


@pytest.mark.logic
def test_every_wire_format_declares_a_comparison() -> None:
    """Every kernel type a port may carry has a comparison, and every leaf of its samples is declared."""
    assert {_type_key(t) for t in WIRE_FORMATS} == set(COMPARISONS)
    for tag in WIRE_FORMATS:
        for sample in samples_of(WIRE_FORMATS[tag]):
            fields = flatten(wire_form(sample, READER))
            undeclared = [p for p in fields if COMPARISONS[_type_key(tag)].rule(p) is None]
            assert not undeclared, (tag, undeclared)


def _perturbed(leaf: list[float | None], delta: float) -> list[float | None]:
    if not leaf:
        return [1.0]
    first = leaf[0]
    return [1.0 if first is None else first + delta, *leaf[1:]]


@pytest.mark.logic
@pytest.mark.parametrize("tag", sorted(WIRE_FORMATS))
def test_each_field_compares_by_its_declared_rule(tag: str) -> None:
    """Changing one leaf of a kernel value differs at exactly that path, judged by its type's rule and threshold.

    A text leaf differs whatever its rule; a number within `atol` agrees under `close` and differs under `exact`;
    beyond it, both differ and the difference names the threshold and its bounds.
    """
    kind = COMPARISONS[_type_key(tag)]
    labels = {f"out.{kind.match_labels[0]}"} if kind.match_labels else set()
    reference = _leaves(tag)
    assert compare_fields(reference, dict(reference), THRESHOLDS).agree
    for path, leaf in reference.items():
        if path == "out.format" or index_free(path).split("[")[0] in labels:
            continue
        rule = kind.rule(path.removeprefix("out."))
        assert rule is not None, path
        if isinstance(leaf, str):
            found = compare_fields(reference, {**reference, path: json.dumps("changed")}, THRESHOLDS)
            assert [(d.path, d.rule) for d in found.differences] == [(path, rule.kind)], path
            continue
        scale = max([1.0, *(abs(x) for x in leaf if x is not None)])
        if scale < 1e6 and leaf and leaf[0] is not None:
            within = compare_fields(reference, {**reference, path: _perturbed(leaf, ATOL / 2)}, THRESHOLDS)
            assert within.agree is (rule.kind == "close"), path
        beyond = compare_fields(reference, {**reference, path: _perturbed(leaf, scale * 1e-3 + 2 * ATOL)}, THRESHOLDS)
        [difference] = beyond.differences
        assert (difference.path, difference.rule, difference.threshold) == (path, rule.kind, rule.threshold)
        if rule.threshold is not None:
            assert difference.bound == THRESHOLDS.by_name[rule.threshold]
            assert difference.detail


@pytest.mark.logic
def test_equal_hashes_agree_and_the_digest_names_the_thresholds() -> None:
    """Equal outputs agree with no difference; every comparison records the digest of the thresholds that judged it."""
    outputs = {"embedding": wire_form(samples_of(WIRE_FORMATS["embedding@1.0"])[0], READER)}
    same = compare(outputs, dict(outputs), THRESHOLDS)
    assert same.agree and same.differences == () and same.thresholds == THRESHOLDS.canonical_digest()
    vector = numpy.asarray(outputs["embedding"]["vector"]) + 10.0
    other = compare(outputs, {"embedding": {**outputs["embedding"], "vector": vector}}, THRESHOLDS)
    assert not other.agree and other.paths == ("embedding.vector",)
    assert other.thresholds == THRESHOLDS.canonical_digest()


@pytest.mark.logic
def test_a_close_threshold_is_relative_and_absolute() -> None:
    """`isclose`: a difference agrees within `atol + rtol * |expected|`, and beyond it differs."""
    expected: Fields = {"e.format": '"embedding@1.0"', "e.space": '"s"', "e.vector": [100.0, 0.0]}
    loose = Thresholds(by_name={**THRESHOLDS.by_name, "embedding": Threshold(rtol=0.01, atol=0.0)})
    assert compare_fields(expected, {**expected, "e.vector": [100.9, 0.0]}, loose).agree
    assert not compare_fields(expected, {**expected, "e.vector": [101.1, 0.0]}, loose).agree
    assert not compare_fields(expected, {**expected, "e.vector": [100.0, 1e-9]}, loose).agree


@pytest.mark.logic
@pytest.mark.parametrize(
    ("a", "b", "agree"),
    [
        ([math.inf, 1.0], [math.inf, 1.0], True),
        ([math.inf], [-math.inf], False),
        ([math.inf], [1.0], False),
        ([None, 1.0], [None, 1.0], True),
        ([None], [1.0], False),
        ([1.0], [1.0, 1.0], False),
    ],
)
def test_close_on_non_finite_absent_and_shape(a: list[float | None], b: list[float | None], agree: bool) -> None:  # noqa: FBT001
    """An equal infinity agrees; any other non-finite pair, a None against a number or a length change differs."""
    expected: Fields = {"e.format": '"embedding@1.0"', "e.vector": a}
    assert compare_fields(expected, {**expected, "e.vector": b}, THRESHOLDS).agree is agree


@pytest.mark.logic
def test_a_leaf_on_one_side_or_undeclared_is_a_difference() -> None:
    """A missing or added leaf is a difference, as is a leaf its type declares no rule for; plain leaves are exact."""
    expected: Fields = {"e.format": '"embedding@1.0"', "e.vector": [1.0], "e.extra": [1.0], "plain": [1.0]}
    found = compare_fields(expected, {**expected, "e.extra": [2.0], "added": '"x"', "plain": [1.1]}, THRESHOLDS)
    assert {d.path: d.rule for d in found.differences} == {
        "added": "missing",
        "e.extra": "undeclared",
        "plain": "exact",
    }


def _turns(labels: list[str], ends: list[int] | None = None) -> Fields:
    """The flattened wire form of an `Intervals` output on port `spans`, one 10 ns turn per label by default."""
    starts = [10 * n for n in range(len(labels))]
    ends = ends or [s + 10 for s in starts]
    spans = Intervals(start_ns=numpy.array(starts), end_ns=numpy.array(ends), label=numpy.array(labels), clock_id="c")
    return flatten({"spans": wire_form(spans, READER)})


@pytest.mark.logic
def test_intervals_compare_under_the_best_label_map() -> None:
    """Labels are matched one to one by shared time (P2-89); a renamed speaker agrees, a merged one differs."""
    ref = _turns(["S0", "S1"], ends=[10, 15])
    assert relabel(ref, _turns(["S1", "S0"], ends=[10, 15])) == ref
    assert compare_fields(ref, _turns(["B", "A"], ends=[10, 15]), THRESHOLDS).agree
    assert compare_fields(ref, _turns(["S0", "S0"], ends=[10, 15]), THRESHOLDS).paths == ("spans.label[1]",)


@pytest.mark.logic
def test_one_label_of_2000_differing_disagrees() -> None:
    """A long label array differing at one index is a difference at that index."""
    labels = ["A"] * 1000 + ["B"] * 1000
    changed = [*labels]
    changed[1500] = "A"
    ref = _turns(labels)
    assert compare_fields(ref, _turns(changed), THRESHOLDS).paths == ("spans.label[1500]",)
    assert compare_fields(ref, _turns(labels), THRESHOLDS).agree


@pytest.mark.logic
def test_best_assignment_is_optimal() -> None:
    """The assignment of largest total weight, against every permutation of a rectangular matrix."""
    rng = numpy.random.default_rng(0)
    for n, m in ((1, 1), (3, 3), (2, 4), (4, 2), (5, 5)):
        weights = rng.integers(0, 10, size=(n, m)).astype(float).tolist()
        pairs = best_assignment(weights)
        assert len(pairs) == min(n, m) and len({r for r, _ in pairs}) == len({c for _, c in pairs}) == min(n, m)
        best = max(
            sum(weights[r][c] for r, c in zip(rows, cols, strict=False))
            for rows in itertools.permutations(range(n), min(n, m))
            for cols in itertools.permutations(range(m), min(n, m))
        )
        assert sum(weights[r][c] for r, c in pairs) == best
    assert best_assignment([]) == []


@pytest.mark.logic
def test_flatten2_leaves() -> None:
    """Numbers, None and NaN as float lists (absent as None); a string array element by element, like a list."""
    assert FLATTEN_VERSION == "flatten@2"
    assert flatten({"span": None, "x": float("nan"), "v": [1, None]}) == {"span": [None], "v": [1.0, None], "x": [None]}
    flat = flatten({"label": numpy.array(["A"] * 1000 + ["B"])})
    assert len(flat) == 1001 and flat["label[1000]"] == '"B"'
    assert flatten({"m": numpy.zeros((2, 2))}) == {"m": [0.0, 0.0, 0.0, 0.0]}
    assert index_free("t.segments[12].words[3].text") == "t.segments[*].words[*].text"
