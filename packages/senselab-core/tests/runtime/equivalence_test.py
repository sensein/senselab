"""Equivalence classes (`equivalence@2.0`), read by the runtime, and the verify evidence they are regenerated from."""

from pathlib import Path

import pytest
import yaml
from senselab_core.kernel.identity import ExecutionContext, numerics_key
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


@pytest.mark.logic
def test_no_measured_equivalence_is_packaged() -> None:
    """The packaged table, read from `runtime/data/equivalence.yaml`, holds no class until evidence is regenerated."""
    assert dict(load_equivalence()) == {}
    assert dict(EQUIVALENCE) == {}


@pytest.mark.logic
def test_a_store_defaults_to_the_packaged_table(tmp_path: object) -> None:
    """`ResultStore(equivalence=)` defaults to the table the runtime loaded."""
    assert ResultStore(tmp_path).equivalence is EQUIVALENCE  # type: ignore[arg-type]


def _doc(backends: str, fmt: str = "equivalence@2.0") -> str:
    return f"format: {fmt}\nbackends:\n{backends}"


CPU, CUDA, MPS = ({"device_kind": d, "precision": "default"} for d in ("cpu", "cuda", "mps"))
ONE_CLASS = """  b:
    c:
      numerics: [{device_kind: cpu, precision: default}, {device_kind: cuda, precision: default}]
      derivation: measured
"""


@pytest.mark.logic
def test_a_class_lists_numerics_classes() -> None:
    """A class is keyed by its numerics classes, each read as `numerics_key`."""
    assert parse_equivalence(_doc(ONE_CLASS)) == {"b": {"c": frozenset({numerics_key(CPU), numerics_key(CUDA)})}}


@pytest.mark.logic
def test_equivalence1_refused() -> None:
    """An `equivalence@1` table is refused with a typed error naming its migration."""
    with pytest.raises(EquivalenceFormatRetired, match="regenerate_equivalence.py"):
        parse_equivalence(_doc("  b: {}\n", "equivalence@1"))


@pytest.mark.logic
@pytest.mark.parametrize(
    "backends",
    [
        ONE_CLASS + "  b: {}\n",
        ONE_CLASS + ONE_CLASS.split("\n", 1)[1],
        ONE_CLASS + "    d:\n      numerics: [{device_kind: cuda, precision: default}, {device_kind: mps}]\n"
        "      derivation: measured\n",
        "  b:\n    c:\n      numerics: [{device_kind: cpu}, {device_kind: cpu}, {device_kind: mps}]\n"
        "      derivation: measured\n",
        "  b:\n    c:\n      numerics: [{device_kind: cpu, device_kind: mps}, {device_kind: cuda}]\n"
        "      derivation: measured\n",
    ],
    ids=["backend", "class", "numerics-in-two-classes", "numerics-twice", "numerics-field"],
)
def test_equivalence_duplicate_key_refused(backends: str) -> None:
    """A repeated backend, class or mapping key, or a numerics class listed twice in one backend, is refused."""
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


def _evidence(
    backend: str, a: dict[str, str], b: dict[str, str], *, agree: bool = True, key: str = "1"
) -> EquivalenceEvidence:
    return EquivalenceEvidence(
        task="t",
        backend=backend,
        key=key * 64,
        stored=EvidenceSide(fingerprint="f" * 64, numerics=a),
        current=EvidenceSide(fingerprint="f" * 64, numerics=b),
        mismatched=() if agree else ("value",),
        tolerance="p" * 64,
        run="slurm-1",
    )


@pytest.mark.logic
def test_regenerated_equivalence_matches_evidence(tmp_path: Path) -> None:
    """Agreements join numerics classes into a class; a disagreement inside one takes both its members out."""
    TPU, NPU = {"device_kind": "tpu"}, {"device_kind": "npu"}  # noqa: N806 - numerics classes, as the constants above
    store = ResultStore(tmp_path)
    for e in (
        _evidence("joined", CPU, CUDA),
        _evidence("joined", CUDA, MPS, key="2"),
        _evidence("refuted", CPU, CUDA),
        _evidence("refuted", CUDA, CPU, agree=False),
        _evidence("chained", CPU, CUDA),
        _evidence("chained", CUDA, MPS),
        _evidence("chained", CPU, MPS, agree=False),
        _evidence("chained", TPU, NPU),
    ):
        store.add_evidence(e)
    store.add_evidence(_evidence("joined", CPU, CUDA))
    records = store.evidence()
    assert len(records) == 8
    text = render(records)
    assert text.startswith(HEADER)
    table = parse_equivalence(text)
    key = numerics_key
    assert table == {
        "joined": {"class-0": frozenset({key(CPU), key(CUDA), key(MPS)})},
        "chained": {"class-0": frozenset({key(TPU), key(NPU)})},
    }
    derivation = yaml.safe_load(text)["backends"]["joined"]["class-0"]["derivation"]
    assert "2 agreements and 0 disagreements over 2 keys" in derivation and "slurm-1" in derivation


@pytest.mark.logic
def test_evidence_record_named_by_its_digest(tmp_path: Path) -> None:
    """A record whose bytes no longer digest to its name is refused, not read."""
    store = ResultStore(tmp_path)
    store.add_evidence(_evidence("b", CPU, CUDA))
    [path] = (tmp_path / "equivalence-evidence@1").iterdir()
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
    ctx = [
        ExecutionContext(
            env_lock="l",
            platform="p",
            wheel_set="w",
            numerics={"n": str(i)},
            decoder=None,
            driver={},
            host="h",
            protocol="p",
            model_source=None,
        )
        for i in range(400)
    ]
    hits = [(str(i), ctx[i], ctx[(i + 1) % 400]) for i in range(400)]
    assert [sampled(0.1, *h) for h in hits] == [sampled(0.1, *h) for h in hits]
    assert 20 <= sum(sampled(0.1, *h) for h in hits) <= 60
    assert not any(sampled(0.0, *h) for h in hits) and all(sampled(1.0, *h) for h in hits)
