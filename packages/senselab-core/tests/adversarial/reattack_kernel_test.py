"""Regressions of the second adversarial round on the kernel and provenance: each rebuilds its finding's probe."""

import copy
import json
import os
import types
from pathlib import Path, PurePosixPath

import numpy
import pytest
import soundfile
from pydantic import ValidationError
from senselab_core.kernel import signals
from senselab_core.kernel._base import FrozenMapping
from senselab_core.kernel.identity import EQUIVALENCE, ExecutionContext, LogicalKey, ModelRef, acceptable
from senselab_core.kernel.license import License
from senselab_core.kernel.results import Events, Intervals
from senselab_core.kernel.signals import Audio, Recording, RegularGroup, StaleSource
from senselab_core.kernel.time import Span, TimeMap
from senselab_core.runtime.provenance import ProvStore, UnrecordableValue, normalised, revised
from senselab_worker.canonical import jcs_dumps

RATE = 8_000


def _wav(path: Path, seed: int = 0, frames: int = RATE) -> Path:
    samples = numpy.random.default_rng(seed).integers(-2000, 2000, size=frames, dtype=numpy.int16)
    soundfile.write(path, samples, RATE, subtype="PCM_16")
    return path


def _key(**params: object) -> LogicalKey:
    return LogicalKey(
        task="t", task_version="1", backend="b", backend_version="1", model=None, params=params, inputs={}
    )


@pytest.mark.logic
def test_xk1_attributes_cannot_forge_an_identity_or_a_stamp(tmp_path: Path) -> None:
    """`from_attributes=True` is refused, so neither a forged identity nor a forged stamp is accepted."""
    audio = Audio.from_array(numpy.zeros(16, dtype=numpy.float32), RATE)
    forged = types.SimpleNamespace(groups=dict(audio.groups), anchor=audio.anchor, identity="content:" + "0" * 64)
    for cls in (Audio, Recording):
        with pytest.raises(ValidationError):
            cls.model_validate(forged, from_attributes=True)
    path = _wav(tmp_path / "a.wav")
    group = Audio.from_path(path).groups["audio"]
    assert isinstance(group, RegularGroup) and group.source_stamp is not None
    stamp = group.source_stamp.model_copy(update={"frames": 10**9})
    held = types.SimpleNamespace(
        channels=group.channels, sampling_rate=RATE, data=None, source=path, source_stamp=stamp
    )
    with pytest.raises(ValidationError):
        RegularGroup.model_validate(held, from_attributes=True)


@pytest.mark.logic
def test_xk2_a_same_size_same_mtime_rewrite_is_stale(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A rewrite keeping size, mtime, inode and device is refused on read and on a copy in a fresh process."""
    path = _wav(tmp_path / "a.wav")
    audio = Audio.from_path(path)
    group = audio.groups["audio"]
    assert isinstance(group, RegularGroup)
    st = os.stat(path)
    with soundfile.SoundFile(path, "r+") as f:
        f.seek(10)
        f.write(numpy.full(100, 0.25, dtype=numpy.float32))
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns))
    after = os.stat(path)
    assert (after.st_size, after.st_mtime_ns, after.st_ino) == (st.st_size, st.st_mtime_ns, st.st_ino)
    with pytest.raises(StaleSource):
        group.read()
    monkeypatch.setattr(signals, "_VERIFIED", signals.Memo(1))
    with pytest.raises(StaleSource):
        copy.deepcopy(audio)


@pytest.mark.logic
def test_xk3_a_frozen_mapping_is_initialised_once() -> None:
    """Re-calling `__init__` on a held mapping raises and leaves the value and its digest as they were."""
    key = _key(a=1)
    before = key.digest()
    for mapping in (key.params, key.inputs):
        assert isinstance(mapping, FrozenMapping)
        with pytest.raises(TypeError):
            mapping.__init__({"a": 5})  # type: ignore[misc]
    assert key.params == {"a": 1} and key.digest() == before == LogicalKey.model_validate(key.model_dump()).digest()


@pytest.mark.logic
def test_xk4_config_private_state_and_dict_cannot_change_a_value() -> None:
    """Flipping `model_config`, writing a private attribute, `__setstate__` and `__dict__` all raise."""
    span = Span(start_ns=0, end_ns=10, clock_id="c")
    other = Span(start_ns=0, end_ns=20, clock_id="c")
    with pytest.raises(TypeError):
        Span.model_config["frozen"] = False  # type: ignore[index]
    with pytest.raises(ValidationError):
        span.end_ns = -100  # type: ignore[misc]
    with pytest.raises(ValidationError):
        span._canonical = other.canonical_digest()  # type: ignore[attr-defined]
    with pytest.raises(TypeError):
        span.__pydantic_private__["_canonical"] = other.canonical_digest()  # type: ignore[index]
    with pytest.raises(TypeError):
        span.__setstate__({"__dict__": {"start_ns": 0, "end_ns": -1, "clock_id": "c"}})
    with pytest.raises(TypeError):
        span.__dict__["end_ns"] = -1
    assert span.end_ns == 10 and span != other and hash(span) != hash(other)
    key = _key()
    with pytest.raises(TypeError):
        key.__pydantic_private__["_digest"] = "f" * 64  # type: ignore[index]
    blank = object.__new__(Span)
    with pytest.raises(ValidationError):
        blank.__setstate__({"__dict__": {"start_ns": 5, "end_ns": 1, "clock_id": "c"}})


@pytest.mark.logic
def test_xk5_recorded_content_cannot_drift_from_its_id(tmp_path: Path) -> None:
    """Mutating what was passed in, or what `get_entity` returns, cannot change a recorded entity."""
    store = ProvStore(run_id="r")
    attributes: dict[str, object] = {"ev": {"n": [1, 2, 3]}}
    extent = [0.0, 1.0]
    eid = store.entity(prov_type="span", extent=extent, attributes=attributes)  # type: ignore[arg-type]
    attributes["ev"]["n"].append(4)  # type: ignore[index]
    extent[1] = 99.0
    entity = store.get_entity(eid)
    assert entity.attributes == {"ev": {"n": [1, 2, 3]}} and entity.extent == (0.0, 1.0)
    for write in (
        lambda: entity.attributes.__setitem__("x", 1),  # type: ignore[index]
        lambda: entity.attributes["ev"].__setitem__("x", 1),
        lambda: entity.attributes["ev"]["n"].append(4),
    ):
        with pytest.raises((TypeError, AttributeError)):
            write()
    assert entity.minted_id() == eid
    act = store.get_activity(store.activity(node="N", step=None, parameters={"p": [1]}))
    with pytest.raises(TypeError):
        act.parameters["p"] = 2  # type: ignore[index]
    timed = ProvStore(run_id="r")
    timed.activity(node="N", step=None, parameters={"p": [1]}, started="2026-01-01T00:00:00Z")
    one, other = ProvStore.merge([store, timed]), ProvStore.merge([timed, store])
    assert one.records() == other.records() and one.get_activity(act.id).started == "2026-01-01T00:00:00Z"
    path = tmp_path / "rival.jsonl"
    rival = revised(entity, attributes={"ev": {"n": [9]}})
    lines = [{"format": "provenance@1.1", "record": "format"}, rival.as_record()]
    path.write_bytes(b"".join(jcs_dumps(line) + b"\n" for line in lines))
    with pytest.raises(ValueError, match="two different records"):
        ProvStore.merge([store, ProvStore.read_jsonl(path)])


@pytest.mark.logic
def test_xk6_jsonl_is_lossless_and_ids_recompute(tmp_path: Path) -> None:
    """Arrays, paths, non-finite floats, enums, sets and bytes survive a write and read; ids recompute."""
    import enum

    class E(enum.Enum):
        A = 1

    one, two = numpy.arange(2000), numpy.arange(2000)
    two[1000] = -1
    store = ProvStore(run_id="r")
    for array in (one, two):
        store.entity(
            prov_type="measurement",
            extent=None,
            attributes={
                "a": array,
                "p": PurePosixPath("x/y"),
                "t": (1, 2),
                "f": [float("nan"), float("-inf")],
                "e": E.A,
                "s": {1, 2},
                "b": b"ab",
            },
        )
    path = tmp_path / "prov.jsonl"
    store.write_jsonl(path)
    lines = path.read_bytes().splitlines()
    assert all(json.loads(line) is not None for line in lines) and b"NaN" not in b"".join(lines).replace(b'"NaN"', b"")
    back = ProvStore.read_jsonl(path)
    assert back.records() == store.records() and len(back.entities()) == 2
    assert all(record.minted_id() == record.id for record in back.records())
    again = tmp_path / "again.jsonl"
    back.write_jsonl(again)
    assert again.read_bytes() == path.read_bytes()


@pytest.mark.logic
def test_xk6_unverified_records_never_pass_as_verified(tmp_path: Path) -> None:
    """A 2.0 line without a run_id is refused; a provenance@1 store is written back as 1.1, never as 2.0."""
    store = ProvStore(run_id="r")
    eid = store.entity(prov_type="span", extent=(0.0, 1.0), attributes={"a": 1})
    edited = revised(store.get_entity(eid), attributes={"a": 2})
    assert edited.id == eid and not edited.verified
    path = tmp_path / "edited.jsonl"
    path.write_bytes(
        b"".join(
            jcs_dumps(line) + b"\n" for line in [{"format": "provenance@2.0", "record": "format"}, edited.as_record()]
        )
    )
    with pytest.raises(ValueError, match="no run_id"):
        ProvStore.read_jsonl(path)
    legacy = tmp_path / "old.jsonl"
    legacy.write_text(
        '{"record": "entity", "id": "span-0123456789abcdef", "prov_type": "span", "extent": [0, 1], "attributes": {}}\n'
    )
    back = ProvStore.read_jsonl(legacy)
    assert not back.verified
    back.entity(prov_type="span", extent=None, attributes={"new": True})
    written = tmp_path / "written.jsonl"
    back.write_jsonl(written)
    assert written.read_text().splitlines()[0] == '{"format": "provenance@1.1", "record": "format"}'
    again = ProvStore.read_jsonl(written)
    assert again.records() == back.records() and not again.verified
    for bad in ("[NaN, 1.0]", "[0, 1e400]"):
        legacy.write_text(
            f'{{"record": "entity", "id": "span-0123456789abcdef", "prov_type": "span", "extent": {bad}, '
            '"attributes": {}}\n'
        )
        with pytest.raises(ValueError, match="finite"):
            ProvStore.read_jsonl(legacy)


@pytest.mark.logic
def test_xk7_write_refuses_what_read_refuses() -> None:
    """Unknown types and kinds, non-mapping attributes and malformed extents are refused when written."""
    store = ProvStore(run_id="r")
    with pytest.raises(ValueError):
        store.entity(prov_type="bogus", extent=None, attributes={})  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        store.agent(agent_type="bogus", version="1")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        store.environment(kind="bogus", label="l", python_version="3")  # type: ignore[arg-type]
    with pytest.raises(UnrecordableValue):
        store.entity(prov_type="span", extent=None, attributes=[1])  # type: ignore[arg-type]
    for extent in (("a", 1.0), (0.0, float("nan")), (2.0, 1.0), (0.0,)):
        with pytest.raises(ValueError):
            store.entity(prov_type="span", extent=extent, attributes={})  # type: ignore[arg-type]


@pytest.mark.logic
def test_xk8_a_ten_million_element_array_normalises_as_bytes_not_elements() -> None:
    """Arrays normalise as canonical `.npy` bytes, so memory stays a small multiple of the data, not of its elements."""
    import tracemalloc

    a = numpy.arange(10_000_000, dtype=numpy.float64)
    tracemalloc.start()
    try:
        token = normalised(a)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert isinstance(token, dict) and list(token) == ["$nd"]
    assert peak < 4 * a.nbytes


@pytest.mark.logic
def test_xk9_normalisation_is_total_and_typed() -> None:
    """An integer too long to write, and empty arrays of refused dtypes, raise `UnrecordableValue`."""
    with pytest.raises(UnrecordableValue):
        normalised(10**5000)
    for dtype in ("c16", "M8[ns]", "S3", "O", [("a", "i4"), ("b", "i4")]):
        with pytest.raises(UnrecordableValue):
            normalised(numpy.zeros(0, dtype=dtype))


@pytest.mark.logic
def test_xk10_deep_or_cyclic_params_raise_a_validation_error() -> None:
    """Params nested 5000 deep, or holding themselves, raise `ValidationError`, not `RecursionError`."""
    deep: object = 1
    for _ in range(5000):
        deep = [deep]
    loop: dict[str, object] = {}
    loop["self"] = loop
    for params in ({"a": deep}, loop):
        with pytest.raises(ValidationError):
            _key(**params)


@pytest.mark.logic
def test_xk11_labels_equal_by_value_are_one_value() -> None:
    """Label arrays of one set of strings are equal whatever width they were built at."""
    ns = numpy.array([0])
    by_width = [
        Intervals(start_ns=ns, end_ns=ns + 1, label=numpy.array(["a"], dtype=dtype), clock_id="c")
        for dtype in ("U1", "U3", "U9")
    ]
    assert by_width[0] == by_width[1] == by_width[2]
    assert Events(time_ns=ns, label=numpy.array(["a\x00"]), value=None, clock_id="c") == Events(
        time_ns=ns, label=numpy.array(["a"]), value=None, clock_id="c"
    )


@pytest.mark.logic
def test_xk12_small_loosenesses_are_refused() -> None:
    """Non-finite audio, lax floats, unknown fields, int64 overflow, unparseable refs, week dates, open classes."""
    for samples in ([0.0, float("nan")], [0.0, float("inf")], numpy.array([0.0, 1e300])):
        with pytest.raises(ValueError):
            Audio.from_array(numpy.asarray(samples), RATE)
    for drift in ("5", True):
        with pytest.raises(ValidationError):
            TimeMap(source="a", target="b", offset_ns=0, drift_ppm=drift)  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        Span.model_validate({"start_ns": 0, "end_ns": 1, "clock_id": "c", "extra": 1})
    with pytest.raises(OverflowError):
        TimeMap(source="a", target="b", offset_ns=2**63 - 1).map(10)
    with pytest.raises(ValidationError):
        ModelRef(source="url", id="http://h/x#sha256=", ref="")
    with pytest.raises(ValidationError):
        License(
            expression="MIT",
            commercial_use="yes",
            copyleft="none",
            attribution=True,
            gated=False,
            source="s",
            verified={"date": "2026-W01-1", "how": "manual"},
        )
    with pytest.raises(TypeError):
        EQUIVALENCE["b"] = {"c": frozenset()}  # type: ignore[index]
    context = ExecutionContext(
        env_lock="l", platform="p", wheel_set="w", numerics={}, decoder=None, driver={}, host="h", protocol="1"
    )
    assert acceptable(context, context, "equivalent", backend="b")


@pytest.mark.logic
def test_xk1_a_nested_unsealed_or_subclass_instance_is_validated_again(tmp_path: Path) -> None:
    """An instance built without validation, or of a subclass with other validators, cannot carry a forged identity."""
    from pydantic import BaseModel

    path = _wav(tmp_path / "a.wav")
    real = Audio.from_path(path)
    group = real.groups["audio"]
    assert isinstance(group, RegularGroup) and group.source_stamp is not None
    forged_stamp = BaseModel.model_construct.__func__(  # type: ignore[attr-defined]
        type(group.source_stamp), **{**dict(group.source_stamp.__dict__), "content": "0" * 64}
    )
    planted = BaseModel.model_construct.__func__(  # type: ignore[attr-defined]
        RegularGroup, **{**dict(group.__dict__), "source_stamp": forged_stamp}
    )
    with pytest.raises((ValidationError, StaleSource)):
        Audio(groups={"audio": planted}, anchor=real.anchor)

    class EvilAudio(Audio):
        @classmethod
        def identity_of(cls, groups: object) -> str:  # type: ignore[override]
            return "content:" + "0" * 64

    evil = EvilAudio(groups=dict(real.groups), anchor=real.anchor)  # type: ignore[call-arg]
    assert evil.identity == "content:" + "0" * 64
    with pytest.raises(ValidationError):
        Audio.model_validate(evil)
    unsealed = BaseModel.model_construct.__func__(Span, start_ns=5, end_ns=1, clock_id="c")  # type: ignore[attr-defined]
    with pytest.raises(ValidationError):
        Span.model_validate(unsealed)


@pytest.mark.logic
def test_audio_refuses_nan_of_either_sign_and_events_hold_one_nan() -> None:
    """Audio refuses NaN whatever its sign or payload; an event value holding NaN has one digest across them."""
    nan = numpy.array([0x7FC00000, 0], dtype="<u4").view("<f4")
    negative_nan = numpy.array([0xFFC00001, 0], dtype="<u4").view("<f4")
    for samples in (nan, negative_nan):
        with pytest.raises(ValueError, match="NaN"):
            Audio.from_array(samples, RATE)
    ns = numpy.array([0, 1])
    a, b = (Events(time_ns=ns, label=numpy.array(["a", "b"]), value=v, clock_id="c") for v in (nan, negative_nan))
    assert a == b and a.canonical_digest() == b.canonical_digest()


@pytest.mark.logic
def test_a_float_wav_holding_nan_is_refused_on_decode(tmp_path: Path) -> None:
    """A file-backed audio's samples are checked as decoded, as held samples are when built."""
    path = tmp_path / "nan.wav"
    samples = numpy.zeros(64, dtype=numpy.float32)
    samples[10] = numpy.nan
    soundfile.write(path, samples, RATE, subtype="FLOAT")
    group = Audio.from_path(path).groups["audio"]
    assert isinstance(group, RegularGroup)
    with pytest.raises(ValueError, match="NaN"):
        group.read()
    assert group.read(0, 10 * 10**9 // RATE).shape == (1, 10)


@pytest.mark.logic
def test_an_irregular_group_holds_nan_as_the_one_missing_marker() -> None:
    """NaN of any sign or payload in an irregular group is held as the quiet NaN without payload; inf is refused."""
    from senselab_core.kernel.signals import ChannelInfo, IrregularGroup

    channel = (ChannelInfo(name="x", type="pose", units="px"),)
    nans = numpy.array([[1.0, 0.0, 0.0]], dtype=numpy.float64)
    nans.view("<u8")[0, 1] = 0xFFF8000000000001
    nans.view("<u8")[0, 2] = 0x7FF8000000000000
    group = IrregularGroup(channels=channel, timestamps_ns=numpy.array([0, 5, 9]), data=nans)
    bits = group.data.view("<u8")[0]
    assert bits[1] == bits[2] == 0x7FF8000000000000
    with pytest.raises(ValidationError):
        IrregularGroup(channels=channel, timestamps_ns=numpy.array([0, 5]), data=numpy.array([[1.0, numpy.inf]]))
