"""Generated value contracts over every `Frozen` type, driven by a registry of samples.

`register_samples(cls, factory, invalid_updates=...)` records how to build values of a type;
`value_cases()` returns `(id, case)` pairs with ids `<Type>::<contract>`, where a case is a
no-argument callable that raises `AssertionError` (or the error it provokes) when the type breaks
the contract. `frozen_types()` lists every concrete `Frozen` subclass defined under
`senselab_core.kernel` and `senselab_core.runtime`; each needs samples.

Contracts per type:
- `value_arrays_immutable`: no ndarray reachable from a sample or from any copy of it (pickle,
  `copy`, `deepcopy`, `model_copy`, `model_validate(model_dump())`), nor any array on its `.base`
  chain, can be made writeable.
- `value_owns_bytes`: writing the arrays a value was built from leaves its `canonical_digest()`
  and every `read()` of its groups unchanged.
- `arrays_storable`: every held array round-trips through `canonical_npy_bytes`.
- `eq_hash_consistent`: a value equals its deep copy with the same hash; samples with different
  digests compare unequal; neither raises.
- `copy_revalidates`: each registered invalid update raises `ValidationError` through `model_copy`.
- `construct_refused`: `model_construct` raises `TypeError`.
- `copy_refuses_unknown_fields`: `model_copy(update=)` naming a field the type lacks raises `ValueError`.
- `identity_derived` (types with an `identity` field not in `ATTESTED_IDENTITIES`): a forged
  identity is refused by validation.
- `identity_form` (types in `ATTESTED_IDENTITIES`): an identity in the form of another source (a
  sha256 for a hub model, a commit for a url or local model) or a ref name is refused.
"""

import copy
import functools
import importlib
import io
import pickle
import pkgutil
import tempfile
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy
import soundfile
from pydantic import ValidationError
from senselab_worker.canonical import canonical_npy_bytes

import senselab_core.kernel
import senselab_core.runtime
from senselab_core.kernel._base import Frozen, value_digest
from senselab_core.kernel.declare import BackendSpec, TaskParams, TaskSpec
from senselab_core.kernel.formats import Formatted
from senselab_core.kernel.identity import ExecutionContext, LogicalKey, ModelIdentity, ModelRef, ResolvedModel
from senselab_core.kernel.license import License
from senselab_core.kernel.results import Annotation, Embedding, Events, Intervals
from senselab_core.kernel.signals import (
    Audio,
    ChannelInfo,
    IrregularGroup,
    Recording,
    RegularGroup,
    SourceStamp,
)
from senselab_core.kernel.time import Span, TimeAnchor, TimeMap
from senselab_core.runtime.fetch import DataFile, DataManifest

Case = Callable[[], None]
Factory = Callable[[], Sequence[Frozen]]

SCANNED_PACKAGES = ("senselab_core.kernel", "senselab_core.runtime")
"""The packages whose `Frozen` types must have samples."""

ATTESTED_IDENTITIES: Mapping[type[Frozen], str] = {
    ResolvedModel: "names weights the value does not hold; checked against the weights at load",
    ModelIdentity: "names weights the value does not hold; checked against the weights at load",
}
"""Types whose `identity` field is a claim checked elsewhere, not derived from the value, and where."""

FORGED_IDENTITIES = ("f" * 64, "content:" + "f" * 64)
"""Identities `identity_derived` supplies in place of the derived one."""

_SAMPLES: dict[type[Frozen], tuple[Factory, tuple[Mapping[str, object], ...]]] = {}


def register_samples(
    cls: type[Frozen], factory: Factory, *, invalid_updates: Sequence[Mapping[str, object]] = ()
) -> None:
    """Register `factory`, which returns samples of `cls`, and updates that must fail validation.

    Samples should include at least one multi-element array per array field and, for
    `eq_hash_consistent`, at least two values with different digests.
    """
    _SAMPLES[cls] = (factory, tuple(invalid_updates))


def registered_types() -> set[type[Frozen]]:
    """Return the types with registered samples."""
    return set(_SAMPLES)


def frozen_types() -> list[type[Frozen]]:
    """Return every `Frozen` subclass defined under `SCANNED_PACKAGES`, after importing all their modules."""
    for package in (senselab_core.kernel, senselab_core.runtime):
        for module in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
            importlib.import_module(module.name)
    found: list[type[Frozen]] = []
    pending = list(Frozen.__subclasses__())
    while pending:
        t = pending.pop()
        pending.extend(t.__subclasses__())
        if t.__module__.startswith(SCANNED_PACKAGES) and t not in found:
            found.append(t)
    return sorted(found, key=lambda t: f"{t.__module__}.{t.__qualname__}")


def value_cases() -> list[tuple[str, Case]]:
    """Return the generated value contract cases of every registered type, ids `<Type>::<contract>`."""
    cases: list[tuple[str, Case]] = []
    for cls in sorted(_SAMPLES, key=lambda t: f"{t.__module__}.{t.__qualname__}"):
        factory, invalid = _SAMPLES[cls]
        contracts: dict[str, Case] = {
            "value_arrays_immutable": functools.partial(_arrays_immutable, factory),
            "value_owns_bytes": functools.partial(_owns_bytes, factory),
            "arrays_storable": functools.partial(_arrays_storable, factory),
            "eq_hash_consistent": functools.partial(_eq_hash_consistent, factory),
            "copy_revalidates": functools.partial(_copy_revalidates, factory, invalid),
            "construct_refused": functools.partial(_construct_refused, cls),
            "copy_refuses_unknown_fields": functools.partial(_copy_refuses_unknown_fields, factory),
        }
        if cls in ATTESTED_IDENTITIES:
            contracts["identity_form"] = functools.partial(_identity_form, factory)
        elif "identity" in cls.model_fields:
            contracts["identity_derived"] = functools.partial(_identity_derived, factory)
        cases.extend((f"{cls.__qualname__}::{name}", case) for name, case in contracts.items())
    return cases


def _samples(factory: Factory) -> Sequence[Frozen]:
    samples = factory()
    assert samples, "a sample factory returned no samples"
    return samples


def _copies(v: Frozen) -> Iterator[tuple[str, Frozen]]:
    yield "original", v
    yield "pickle", pickle.loads(pickle.dumps(v))
    yield "copy", copy.copy(v)
    yield "deepcopy", copy.deepcopy(v)
    yield "model_copy", v.model_copy()
    yield "model_copy(deep)", v.model_copy(deep=True)
    yield "model_validate(model_dump)", type(v).model_validate(v.model_dump())


def _reachable_arrays(v: object) -> Iterator[numpy.ndarray]:
    if isinstance(v, numpy.ndarray):
        yield v
    elif isinstance(v, Frozen):
        for name in type(v).model_fields:
            yield from _reachable_arrays(getattr(v, name))
    elif isinstance(v, Mapping):
        for x in v.values():
            yield from _reachable_arrays(x)
    elif isinstance(v, (list, tuple)):
        for x in v:
            yield from _reachable_arrays(x)


def _base_chain(a: numpy.ndarray) -> Iterator[numpy.ndarray]:
    node: object = a
    while isinstance(node, numpy.ndarray):
        yield node
        node = node.base


def _arrays_immutable(factory: Factory) -> None:
    for sample in _samples(factory):
        for how, v in _copies(sample):
            for a in _reachable_arrays(v):
                for link in _base_chain(a):
                    try:
                        link.flags.writeable = True
                    except ValueError:
                        continue
                    raise AssertionError(f"{type(v).__qualname__} via {how}: a held array can be made writeable")


def _writeable_dump(v: object, arrays: list[numpy.ndarray]) -> object:
    if isinstance(v, numpy.ndarray):
        w = numpy.array(v, copy=True)
        arrays.append(w)
        return w
    if isinstance(v, dict):
        return {k: _writeable_dump(x, arrays) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return type(v)(_writeable_dump(x, arrays) for x in v)
    return v


def _reads(v: object) -> list[numpy.ndarray]:
    if isinstance(v, RegularGroup):
        return [numpy.array(v.read(), copy=True)]
    if isinstance(v, IrregularGroup):
        return []
    if isinstance(v, Frozen):
        return [r for name in type(v).model_fields for r in _reads(getattr(v, name))]
    if isinstance(v, Mapping):
        return [r for x in v.values() for r in _reads(x)]
    if isinstance(v, (list, tuple)):
        return [r for x in v for r in _reads(x)]
    return []


def _scribble(a: numpy.ndarray) -> None:
    if a.size == 0:
        return
    flat = a.reshape(-1)
    if a.dtype.kind == "U":
        flat[...] = "~"
    elif a.dtype.kind == "b":
        flat[...] = ~flat
    else:
        flat[...] = flat + 1


def _fresh_digest(v: Frozen) -> str:
    """Return the canonical digest recomputed from the held fields, bypassing the cached one."""
    t = type(v)
    return value_digest(f"{t.__module__}.{t.__qualname__}", v.model_dump())


def _owns_bytes(factory: Factory) -> None:
    for sample in _samples(factory):
        given: list[numpy.ndarray] = []
        v = type(sample).model_validate(_writeable_dump(sample.model_dump(), given))
        digest, reads = _fresh_digest(v), _reads(v)
        for a in given:
            _scribble(a)
        assert _fresh_digest(v) == digest == v.canonical_digest(), (
            f"{type(v).__qualname__}: writing the input changed the digest"
        )
        after = _reads(v)
        assert len(after) == len(reads) and all(numpy.array_equal(x, y) for x, y in zip(after, reads)), (
            f"{type(v).__qualname__}: writing the input changed what read() returns"
        )


def _arrays_storable(factory: Factory) -> None:
    for sample in _samples(factory):
        for a in _reachable_arrays(sample):
            back = numpy.load(io.BytesIO(canonical_npy_bytes(a)), allow_pickle=False)
            assert back.dtype == a.dtype and numpy.array_equal(back, a), f"{type(sample).__qualname__}: {a.dtype}"


def _eq_hash_consistent(factory: Factory) -> None:
    samples = _samples(factory)
    for v in samples:
        twin = copy.deepcopy(v)
        assert v == twin and hash(v) == hash(twin), f"{type(v).__qualname__}: a deep copy differs"
        assert not (v != twin)
    digests = [v.canonical_digest() for v in samples]
    assert len(set(digests)) > 1 or len(samples) == 1, "samples need two values with different digests"
    for i, a in enumerate(samples):
        for b in samples[i + 1 :]:
            if a.canonical_digest() != b.canonical_digest():
                assert a != b and not (a == b), f"{type(a).__qualname__}: values with different digests are equal"


def _copy_revalidates(factory: Factory, invalid: Sequence[Mapping[str, object]]) -> None:
    assert invalid, "register at least one invalid update"
    for v in _samples(factory):
        for update in invalid:
            try:
                v.model_copy(update=dict(update))
            except ValidationError:
                continue
            raise AssertionError(f"{type(v).__qualname__}.model_copy(update={update!r}) was not refused")


def _construct_refused(cls: type[Frozen]) -> None:
    try:
        cls.model_construct()
    except TypeError:
        return
    raise AssertionError(f"{cls.__qualname__}.model_construct() built a value without validation")


def _copy_refuses_unknown_fields(factory: Factory) -> None:
    for v in _samples(factory):
        try:
            v.model_copy(update={"no_such_field": 1})
        except ValueError:
            continue
        raise AssertionError(f"{type(v).__qualname__}.model_copy accepted an unknown field")


def _source_of(v: Frozen) -> str:
    source = getattr(v, "source", None)
    if source is None:
        source = getattr(getattr(v, "ref"), "source")
    assert isinstance(source, str)
    return source


def _identity_form(factory: Factory) -> None:
    for v in _samples(factory):
        other_form = "a" * 64 if _source_of(v) == "hf" else "a" * 40
        for wrong in (other_form, "main", "A" * 64):
            try:
                type(v).model_validate({**v.model_dump(), "identity": wrong})
            except ValidationError:
                continue
            raise AssertionError(f"{type(v).__qualname__} accepted the identity {wrong!r} for a {_source_of(v)} model")


def _identity_derived(factory: Factory) -> None:
    for v in _samples(factory):
        for forged in FORGED_IDENTITIES:
            try:
                type(v).model_validate({**v.model_dump(), "identity": forged})
            except ValidationError:
                continue
            raise AssertionError(f"{type(v).__qualname__} accepted the forged identity {forged!r}")


# Samples of the kernel and runtime types.

_RATE = 8_000
_COMMIT = "0" * 40
_SHA = "a" * 64


_HELD: list[tempfile.TemporaryDirectory[str]] = []


@functools.cache
def _wav_dir() -> Path:
    holder = tempfile.TemporaryDirectory(prefix="senselab-values-")
    _HELD.append(holder)
    return Path(holder.name)


def _wav(name: str, channels: int) -> Path:
    path = _wav_dir() / f"{name}.wav"
    if not path.exists():
        k = numpy.arange(64)
        data = numpy.stack([(k % 10 + c) / 100 for c in range(channels)], axis=1).astype(numpy.float32)
        soundfile.write(path, data, _RATE, subtype="FLOAT")
    return path


def _channels(n: int, kind: str = "ecg", units: str = "mV") -> tuple[ChannelInfo, ...]:
    return tuple(ChannelInfo(name=f"c{i}", type=kind, units=units) for i in range(n))


def _data_group(seed: int = 0) -> RegularGroup:
    return RegularGroup(channels=_channels(2), sampling_rate=250.0, data=numpy.arange(8.0).reshape(2, 4) + seed)


def _irregular(seed: int = 0) -> IrregularGroup:
    return IrregularGroup(
        channels=_channels(1, "hr", "bpm"),
        timestamps_ns=numpy.array([0, 7, 30]),
        data=numpy.array([[60.0, 61, 62 + seed]]),
    )


def _anchor() -> TimeAnchor:
    return TimeAnchor(clock_id="dev", absolute_start_ns=1_700_000_000_000_000_000)


def _license(gated: bool = False) -> License:
    return License(
        expression="MIT",
        commercial_use="yes",
        copyleft="none",
        attribution=True,
        gated=gated,
        source="pep639",
        verified={"date": "2026-10-30", "how": "manual"},
    )


class SampleParams(TaskParams):
    """The params model of the `TaskSpec` samples."""

    n: int = 1


def _context(host: str = "host") -> ExecutionContext:
    return ExecutionContext(
        env_lock="lock",
        platform="linux-x86_64",
        wheel_set="wheels",
        numerics={"class": "fp32"},
        decoder=None,
        driver={"cuda": "12.8"},
        host=host,
        protocol="2",
    )


def _key(task: str = "t") -> LogicalKey:
    return LogicalKey(
        task=task,
        task_version="1",
        backend="b",
        backend_version="1",
        model=ModelIdentity(source="hf", id="org/m", identity=_COMMIT),
        params={"n": [1, 2], "s": "x"},
        inputs={"audio": "content:" + _SHA},
    )


def _data_file(path: str = "a/b.wav") -> DataFile:
    return DataFile(path=path, url=f"https://example.org/{path}", sha256=_SHA, mirrors=("https://m.example.org/x",))


def _manifest(version: str = "1") -> DataManifest:
    return DataManifest(
        format="data-manifest@1.0",
        name="set",
        version=version,
        license=_license(),
        terms="cite",
        files=(_data_file(), _data_file("c.wav")),
    )


def _register_kernel_samples() -> None:
    ns = numpy.array([0, 10, 20])
    labels = numpy.array(["a", "b", "c"])
    register_samples(
        TimeAnchor,
        lambda: [_anchor(), TimeAnchor(clock_id="other", absolute_start_ns=None)],
        invalid_updates=[{"clock_id": ""}, {"absolute_start_ns": 2**63}],
    )
    register_samples(
        TimeMap,
        lambda: [
            TimeMap(source="a", target="b", offset_ns=2**62, drift_ppm=-999_999.5),
            TimeMap(source="a", target="b", offset_ns=0),
        ],
        invalid_updates=[{"drift_ppm": -3e6}, {"drift_ppm": 1e6}, {"uncertainty_ns": -1}, {"format": "other@1.0"}],
    )
    register_samples(
        Span,
        lambda: [Span(start_ns=0, end_ns=2**63 - 1, clock_id="c"), Span(start_ns=5, end_ns=5, clock_id="c")],
        invalid_updates=[{"end_ns": -1}, {"end_ns": 2**200}, {"start_ns": 1.5}],
    )
    register_samples(
        ChannelInfo,
        lambda: [ChannelInfo(name="a", type="ecg", units="mV"), ChannelInfo(name="b", type="audio", units="FS")],
        invalid_updates=[{"name": ""}, {"units": ""}],
    )
    register_samples(
        SourceStamp,
        lambda: [SourceStamp.of(_wav("mono", 1)), SourceStamp.of(_wav("stereo", 2))],
        invalid_updates=[{"size": -1}, {"sampling_rate": True}, {"channels": 0}],
    )
    register_samples(
        RegularGroup,
        lambda: [
            _data_group(),
            _data_group(1),
            RegularGroup(channels=_channels(2, "audio", "FS"), sampling_rate=_RATE, source=_wav("stereo", 2)),
        ],
        invalid_updates=[
            {"sampling_rate": -5},
            {"sampling_rate": True},
            {"channels": ()},
            {"data": numpy.zeros((3, 4))},
            {"data": numpy.zeros((2, 4), dtype=numpy.complex64)},
        ],
    )
    register_samples(
        IrregularGroup,
        lambda: [_irregular(), _irregular(1)],
        invalid_updates=[{"timestamps_ns": numpy.array([3, 2, 1])}, {"data": numpy.ones((1, 2))}],
    )
    register_samples(
        Recording,
        lambda: [
            Recording(groups={"ecg": _data_group(), "hr": _irregular()}, anchor=_anchor()),
            Recording(groups={"ecg": _data_group(1)}, anchor=_anchor()),
            Recording(
                groups={
                    "mic": RegularGroup(
                        channels=_channels(1, "audio", "FS"), sampling_rate=_RATE, source=_wav("mono", 1)
                    )
                },
                anchor=_anchor(),
            ),
        ],
        invalid_updates=[{"groups": {}}, {"identity": "content:" + "f" * 64}],
    )
    register_samples(
        Audio,
        lambda: [
            Audio.from_array(numpy.array([[0.25, -0.5, 0.75], [0.0, 0.5, -1.0]], dtype=numpy.float32), _RATE),
            Audio.from_array(numpy.array([1, -2, 3], dtype=numpy.int16), 16_000),
            Audio.from_path(_wav("stereo", 2)),
        ],
        invalid_updates=[{"groups": {"ecg": _data_group()}}, {"identity": "content:" + "f" * 64}],
    )
    span = Span(start_ns=0, end_ns=10, clock_id="c")
    register_samples(
        Annotation,
        lambda: [
            Annotation(span=span, label="speech", value=None, score=0.9, producer="model"),
            Annotation(span=span, label=None, value=1.5, score=None, producer="human", uncertainty=0.1),
        ],
        invalid_updates=[{"label": None, "value": None}, {"uncertainty": -1.0}, {"score": float("nan")}],
    )
    register_samples(
        Embedding,
        lambda: [Embedding(vector=numpy.array([1.0, 2.0, 3.0]), space="s"), Embedding(vector=[1, 2, 4], space="s")],
        invalid_updates=[
            {"vector": numpy.array([numpy.nan, 1.0])},
            {"vector": numpy.zeros(2, dtype=numpy.complex128)},
            {"vector": []},
        ],
    )
    register_samples(
        Intervals,
        lambda: [
            Intervals(start_ns=ns, end_ns=ns + 5, label=labels, clock_id="c"),
            Intervals(start_ns=ns, end_ns=ns + 5, label=numpy.array([1, 2, 3]), clock_id="c"),
        ],
        invalid_updates=[{"end_ns": ns - 1}, {"label": numpy.array(["a"])}, {"label": numpy.array([1, None])}],
    )
    register_samples(
        Events,
        lambda: [
            Events(time_ns=ns, label=labels, value=numpy.array([0.5, 1.5, 2.5]), clock_id="c"),
            Events(time_ns=ns, label=labels, value=None, clock_id="c"),
        ],
        invalid_updates=[{"label": numpy.array(["a"])}, {"value": numpy.array([True, False, True])}],
    )
    register_samples(
        Formatted,
        lambda: [Formatted(format="thing@1.0"), Formatted(format="thing@2.1")],
        invalid_updates=[{"format": "thing"}, {"format": "Thing@1.0"}],
    )
    register_samples(
        ModelRef,
        lambda: [
            ModelRef.parse("org/m@main"),
            ModelRef.parse("local:/models/m"),
            ModelRef.parse(f"url:https://x.org/w#sha256={_SHA}"),
        ],
        invalid_updates=[{"id": ""}, {"source": "s3"}],
    )
    register_samples(
        ResolvedModel,
        lambda: [
            ResolvedModel(ref=ModelRef.parse("org/m@main"), identity=_COMMIT, license=_license()),
            ResolvedModel(ref=ModelRef.parse("local:/models/m"), identity=_SHA),
        ],
        invalid_updates=[{"identity": "main"}],
    )
    register_samples(
        ModelIdentity,
        lambda: [
            ModelIdentity(source="hf", id="org/m", identity=_COMMIT),
            ModelIdentity(source="local", id="/m", identity=_SHA),
        ],
        invalid_updates=[{"identity": "main"}],
    )
    register_samples(
        LogicalKey,
        lambda: [_key(), _key("u")],
        invalid_updates=[{"model": "main"}, {"inputs": {"a": "/home/x.wav"}}, {"params": {"x": float("nan")}}],
    )
    register_samples(
        ExecutionContext, lambda: [_context(), _context("other")], invalid_updates=[{"env_lock": ""}, {"numerics": 1}]
    )
    register_samples(
        License,
        lambda: [_license(), _license(gated=True)],
        invalid_updates=[{"verified": {"date": "x"}}, {"gated": "no"}],
    )
    register_samples(
        TaskSpec,
        lambda: [
            TaskSpec(
                name="t",
                version="1",
                kind="measure",
                inputs={"a": "Audio"},
                outputs={"e": "Embedding"},
                params=SampleParams,
            ),
            TaskSpec(
                name="u",
                version="1",
                kind="transform",
                inputs={"a": "Audio"},
                outputs={"a": "Audio"},
                params=SampleParams,
            ),
        ],
        invalid_updates=[{"kind": "other"}, {"name": ""}],
    )
    register_samples(
        BackendSpec,
        lambda: [
            BackendSpec(task="t", name="b", version="1", stack="fake", env="e", devices=("cpu",), license=_license()),
            BackendSpec(
                task="t",
                name="c",
                version="1",
                stack="fake",
                env="e",
                devices=("cpu", "cuda"),
                license=_license(),
                output_affecting=frozenset({"chunk_grid", "batch_formation"}),
            ),
        ],
        invalid_updates=[{"devices": ()}, {"output_affecting": frozenset({"bogus"})}],
    )


def _register_runtime_samples() -> None:
    register_samples(
        DataFile,
        lambda: [_data_file(), _data_file("c.wav")],
        invalid_updates=[{"path": "../x"}, {"sha256": "x"}],
    )
    register_samples(
        DataManifest,
        lambda: [_manifest(), _manifest("2")],
        invalid_updates=[{"files": (_data_file(), _data_file())}, {"name": "a/b"}],
    )


_register_kernel_samples()
_register_runtime_samples()
