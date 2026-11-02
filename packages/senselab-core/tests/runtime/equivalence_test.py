"""Equivalence classes (`equivalence@3.0`), read by the runtime, and the verify evidence they are regenerated from."""

import itertools
from collections.abc import Iterable
from pathlib import Path

import numpy
import pytest
import yaml
from senselab_core.kernel.identity import ExecutionContext, acceptable, member_key
from senselab_core.kernel.results import Intervals
from senselab_core.kernel.wire import wire_form
from senselab_core.runtime.audio_io import READER
from senselab_core.runtime.equivalence import (
    EQUIVALENCE,
    HEADER,
    EquivalenceEvidence,
    EquivalenceFormatRetired,
    EvidenceSide,
    load_equivalence,
    parse_equivalence,
    render,
    sampled,
    split,
)
from senselab_core.runtime.store import CorruptEntryError, ResultStore
from senselab_core.runtime.tolerance import (
    Fields,
    ToleranceProfile,
    best_assignment,
    flatten,
    index_free,
    relabel_turns,
    within,
)


@pytest.mark.logic
def test_packaged_equivalence_is_the_data_file_and_every_class_pairs_contexts() -> None:
    """`EQUIVALENCE` is the table read from `runtime/data/equivalence.yaml`; each class holds two or more members."""
    assert EQUIVALENCE == load_equivalence()
    for classes in EQUIVALENCE.values():
        assert all(len(members) >= 2 for members in classes.values())


@pytest.mark.logic
def test_a_store_defaults_to_the_packaged_table(tmp_path: object) -> None:
    """`ResultStore(equivalence=)` defaults to the table the runtime loaded."""
    assert ResultStore(tmp_path).equivalence is EQUIVALENCE  # type: ignore[arg-type]


def _doc(backends: str, fmt: str = "equivalence@3.0") -> str:
    return f"format: {fmt}\nbackends:\n{backends}"


CPU, CUDA, MPS = ({"device_kind": d, "precision": "default"} for d in ("cpu", "cuda", "mps"))
L1, L2 = "lock-1", "lock-2"
ONE_CLASS = """  b:
    c:
      members:
      - {numerics: {device_kind: cpu, precision: default}, env_lock: lock-1}
      - {numerics: {device_kind: cuda, precision: default}, env_lock: lock-1}
      derivation: measured
"""


def key(numerics: dict[str, str], env_lock: str = L1) -> str:
    """Return the member `numerics` is in `env_lock`."""
    return member_key(numerics, env_lock)


@pytest.mark.logic
def test_a_class_lists_members() -> None:
    """A class is keyed by its members, each a numerics class in an env lock, read as `member_key`."""
    assert parse_equivalence(_doc(ONE_CLASS)) == {"b": {"c": frozenset({key(CPU), key(CUDA)})}}


@pytest.mark.logic
@pytest.mark.parametrize("fmt", ["equivalence@1", "equivalence@2.0"])
def test_retired_formats_refused(fmt: str) -> None:
    """An `equivalence@1` or `@2.0` table, whose members carry no env lock, is refused naming its migration."""
    with pytest.raises(EquivalenceFormatRetired, match="regenerate_equivalence.py"):
        parse_equivalence(_doc("  b: {}\n", fmt))


@pytest.mark.logic
@pytest.mark.parametrize(
    "member", ["{numerics: {device_kind: cpu}}", "{numerics: {device_kind: cpu}, env_lock: ''}", "{device_kind: cpu}"]
)
def test_a_member_without_an_env_lock_refused(member: str) -> None:
    """A member is exactly a numerics class and a non-empty env lock."""
    body = f"  b:\n    c:\n      members: [{member}, {{numerics: {{device_kind: mps}}, env_lock: l}}]\n"
    with pytest.raises(ValueError, match="env_lock"):
        parse_equivalence(_doc(body + "      derivation: measured\n"))


@pytest.mark.logic
@pytest.mark.parametrize(
    "backends",
    [
        ONE_CLASS + "  b: {}\n",
        ONE_CLASS + ONE_CLASS.split("\n", 1)[1],
        ONE_CLASS + "    d:\n      members: [{numerics: {device_kind: cuda, precision: default}, env_lock: lock-1},"
        " {numerics: {device_kind: mps}, env_lock: lock-1}]\n      derivation: measured\n",
        "  b:\n    c:\n      members: [{numerics: {device_kind: cpu}, env_lock: l}, {numerics: {device_kind: cpu},"
        " env_lock: l}, {numerics: {device_kind: mps}, env_lock: l}]\n      derivation: measured\n",
        "  b:\n    c:\n      members: [{numerics: {device_kind: cpu, device_kind: mps}, env_lock: l},"
        " {numerics: {device_kind: cuda}, env_lock: l}]\n      derivation: measured\n",
    ],
    ids=["backend", "class", "member-in-two-classes", "member-twice", "numerics-field"],
)
def test_equivalence_duplicate_key_refused(backends: str) -> None:
    """A repeated backend, class or mapping key, or a member listed twice in one backend, is refused."""
    with pytest.raises(ValueError, match="duplicate keys"):
        parse_equivalence(_doc(backends))


@pytest.mark.logic
@pytest.mark.parametrize("derivation", ['""', "'  '", "null", None])
def test_empty_derivation_refused(derivation: str | None) -> None:
    """A class with an empty, blank, null or missing derivation is refused."""
    body = ONE_CLASS.replace(
        "      derivation: measured\n", "" if derivation is None else f"      derivation: {derivation}\n"
    )
    with pytest.raises(ValueError, match="derivation"):
        parse_equivalence(_doc(body))


def _evidence(  # noqa: PLR0913 - one comparison's sides
    backend: str,
    a: dict[str, str],
    b: dict[str, str],
    *,
    agree: bool = True,
    key: str = "1",
    locks: tuple[str, str] = (L1, L1),
) -> EquivalenceEvidence:
    return EquivalenceEvidence(
        task="t",
        backend=backend,
        key=key * 64,
        stored=EvidenceSide(fingerprint="f" * 64, numerics=a, env_lock=locks[0]),
        current=EvidenceSide(fingerprint="f" * 64, numerics=b, env_lock=locks[1]),
        mismatched=() if agree else ("value",),
        tolerance="p" * 64,
        run="slurm-1",
    )


@pytest.mark.logic
def test_regenerated_equivalence_matches_evidence(tmp_path: Path) -> None:
    """A class is a clique of agreeing pairs, largest first; a disagreeing pair is never in one class."""
    TPU, NPU = {"device_kind": "tpu"}, {"device_kind": "npu"}  # noqa: N806 - numerics classes, as the constants above
    store = ResultStore(tmp_path)
    for e in (
        _evidence("clique", CPU, CUDA),
        _evidence("clique", CUDA, MPS, key="2"),
        _evidence("clique", MPS, CPU, key="3"),
        _evidence("refuted", CPU, CUDA),
        _evidence("refuted", CUDA, CPU, agree=False),
        _evidence("chained", CPU, CUDA),
        _evidence("chained", CUDA, MPS),
        _evidence("chained", CPU, MPS, agree=False),
        _evidence("chained", TPU, NPU),
    ):
        store.add_evidence(e)
    store.add_evidence(_evidence("clique", CPU, CUDA))
    records = store.evidence()
    assert len(records) == 9
    text = render(records)
    assert text.startswith(HEADER)
    assert parse_equivalence(text) == {
        "clique": {"class-0": frozenset({key(CPU), key(CUDA), key(MPS)})},
        "chained": {"class-0": frozenset({key(CPU), key(CUDA)}), "class-1": frozenset({key(TPU), key(NPU)})},
    }
    derivation = yaml.safe_load(text)["backends"]["clique"]["class-0"]["derivation"]
    assert "3 agreements and 0 disagreements over 3 keys" in derivation and "slurm-1" in derivation


def _context(numerics: dict[str, str], env_lock: str = L1) -> ExecutionContext:
    return ExecutionContext(
        env_lock=env_lock,
        platform="p",
        wheel_set="w",
        numerics=numerics,
        decoder=None,
        driver={},
        host="h",
        protocol="p",
        model_source=None,
    )


@pytest.mark.logic
def test_an_unmeasured_pair_is_never_joined() -> None:
    """A~B and B~C with A-C never measured: A and C are in no one class, so A is not served for C."""
    table = parse_equivalence(render([_evidence("b", CPU, CUDA), _evidence("b", CUDA, MPS, key="2")]))
    assert table == {"b": {"class-0": frozenset({key(CPU), key(CUDA)})}}
    assert not acceptable(_context(CPU), _context(MPS), "equivalent", backend="b", equivalence=table)
    assert not acceptable(_context(CUDA), _context(MPS), "equivalent", backend="b", equivalence=table)
    assert acceptable(_context(CPU), _context(CUDA), "equivalent", backend="b", equivalence=table)


@pytest.mark.logic
def test_a_class_joins_two_env_locks_only_on_evidence_for_that_pair() -> None:
    """Agreement within each of two env locks never joins them; only a comparison across the pair does."""
    within_each = [_evidence("b", CPU, CUDA), _evidence("b", CPU, CUDA, key="2", locks=(L2, L2))]
    table = parse_equivalence(render(within_each))
    assert table == {
        "b": {"class-0": frozenset({key(CPU), key(CUDA)}), "class-1": frozenset({key(CPU, L2), key(CUDA, L2)})}
    }
    assert not acceptable(_context(CPU), _context(CPU, L2), "equivalent", backend="b", equivalence=table)
    across = parse_equivalence(render([_evidence("b", CPU, CPU, key="3", locks=(L1, L2))]))
    assert across == {"b": {"class-0": frozenset({key(CPU), key(CPU, L2)})}}
    assert acceptable(_context(CPU), _context(CPU, L2), "equivalent", backend="b", equivalence=across)
    assert yaml.safe_load(render(within_each))["backends"]["b"]["class-1"]["members"][0]["env_lock"] == L2


@pytest.mark.logic
def test_evidence_records_both_env_locks(tmp_path: Path) -> None:
    """An evidence record names each side's env lock, and a disagreement across locks splits only that pair."""
    store = ResultStore(tmp_path, equivalence={"b": {"c": frozenset({key(CPU), key(CPU, L2), key(CUDA)})}})
    store.add_evidence(_evidence("b", CPU, CPU, agree=False, locks=(L1, L2)))
    [record] = store.evidence()
    assert (record.stored.env_lock, record.current.env_lock) == (L1, L2)
    assert store.equivalence == {"b": {}}


@pytest.mark.logic
def test_evidence_record_named_by_its_digest(tmp_path: Path) -> None:
    """A record whose bytes no longer digest to its name is refused, not read."""
    store = ResultStore(tmp_path)
    store.add_evidence(_evidence("b", CPU, CUDA))
    [path] = (tmp_path / "equivalence-evidence@2").iterdir()
    path.chmod(0o600)
    path.write_text(path.read_text().replace("slurm-1", "slurm-2"))
    with pytest.raises(CorruptEntryError):
        store.evidence()


@pytest.mark.logic
def test_split_takes_both_members_out() -> None:
    """`split` leaves the rest of the class while two or more remain, and other backends untouched."""
    a, b, c, d = "a", "b", "c", "d"
    table = {"x": {"k": frozenset({a, b, c, d})}, "y": {"k": frozenset({a, b})}}
    assert split(table, "x", a, b) == {"x": {"k": frozenset({c, d})}, "y": {"k": frozenset({a, b})}}
    assert split(split(table, "x", a, b), "x", c, d) == {"x": {}, "y": {"k": frozenset({a, b})}}


@pytest.mark.logic
def test_sampling_is_a_fixed_fraction() -> None:
    """The same hit is always or never sampled; about `k` of hits are; 0 samples none and 1 all."""
    ctx = [_context({"n": str(i)}) for i in range(400)]
    hits = [(str(i), ctx[i], ctx[(i + 1) % 400]) for i in range(400)]
    assert [sampled(0.1, *h) for h in hits] == [sampled(0.1, *h) for h in hits]
    assert 20 <= sum(sampled(0.1, *h) for h in hits) <= 60
    assert not any(sampled(0.0, *h) for h in hits) and all(sampled(1.0, *h) for h in hits)


@pytest.mark.logic
def test_a_disagreement_split_persists_in_the_store(tmp_path: Path) -> None:
    """A disagreeing record splits the store's class, and a new store over the same root applies it too."""
    table = {"b": {"c": frozenset({key(CPU), key(CUDA), key(MPS)})}}
    store = ResultStore(tmp_path, equivalence=table)
    store.add_evidence(_evidence("b", CPU, CUDA))
    assert store.equivalence == table
    store.add_evidence(_evidence("b", CUDA, MPS, agree=False))
    assert store.equivalence == {"b": {}}
    again = ResultStore(tmp_path, equivalence=table)
    assert again.equivalence == {"b": {}}
    assert not acceptable(_context(CPU), _context(MPS), "equivalent", backend="b", equivalence=again.equivalence)
    assert ResultStore(tmp_path / "other", equivalence=table).equivalence == table


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


def _turns(labels: list[str], ends: list[int] | None = None) -> Fields:
    """The flattened wire form of a real `Intervals` output on port `spans`, one 10 ns turn per label by default."""
    n = len(labels)
    spans = Intervals(
        start_ns=[10 * i for i in range(n)],
        end_ns=ends or [10 * i + 10 for i in range(n)],
        label=numpy.array(labels),
        clock_id="c",
    )
    return flatten({"spans": wire_form(spans, READER)})


def _exact_profile(fields: Iterable[str]) -> ToleranceProfile:
    return ToleranceProfile.model_validate(
        {
            "task": "t",
            "backend": "b",
            "derivation": {"shas": [], "jobs": [], "split": "calibration", "items": [], "contexts": []},
            "fields": [
                {"path": p, "kind": "exact", "spreads": [], "tolerance": 0.0} for p in {*map(index_free, fields)}
            ],
            "findings": [],
        }
    )


@pytest.mark.logic
def test_within_compares_intervals_under_the_best_label_map() -> None:
    """On a real `Intervals`, under any port: swapped labels agree, a merged speaker does not."""
    ref = _turns(["A", "B"], ends=[10, 15])
    profile = _exact_profile(ref)
    assert "spans.label[1]" in ref and relabel_turns(ref, _turns(["S1", "S0"], ends=[10, 15])) == ref
    assert within(ref, _turns(["S1", "S0"], ends=[10, 15]), profile) == {}
    assert list(within(ref, _turns(["S0", "S0"], ends=[10, 15]), profile)) == ["spans.label[1]"]


@pytest.mark.logic
def test_one_label_of_2000_differing_disagrees() -> None:
    """A long label array differing at one index is a disagreement, never hidden by a truncated form."""
    labels = ["A"] * 1000 + ["B"] * 1000
    changed = [*labels]
    changed[1500] = "A"
    ref = _turns(labels)
    assert list(within(ref, _turns(changed), _exact_profile(ref))) == ["spans.label[1500]"]
    assert within(ref, _turns(labels), _exact_profile(ref)) == {}
