"""Identity v2: model references, logical keys, execution context and reuse policy (ADR-002, P-38)."""

import copy
import math
import pickle
import unicodedata

import numpy
import pytest
import senselab_core.kernel.identity as identity
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError
from senselab_core.kernel.identity import (
    DEFAULT_REUSE_POLICY,
    ExecutionContext,
    LogicalKey,
    ModelIdentity,
    ModelRef,
    ResolvedModel,
    ReusePolicy,
    UnpinnedModel,
    acceptable,
)
from senselab_core.kernel.license import License
from senselab_worker.digest import identity_digest

COMMIT = "41f01f3fe87f28c78e2fbf8b568835947dd65ed9"
SHA256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
CONTENT = "content:" + SHA256
OUTPUT = "output:" + SHA256
PROFILE = "profile:" + SHA256
MODEL = ModelIdentity(source="hf", id="org/name", identity=COMMIT)

BASE: dict[str, object] = {
    "task": "t",
    "task_version": "1",
    "backend": "b",
    "backend_version": "1",
    "model": None,
    "params": {"x": 1},
    "inputs": {},
}


def _key(**overrides: object) -> LogicalKey:
    return LogicalKey.model_validate({**BASE, **overrides})


def _ctx(**overrides: object) -> ExecutionContext:
    fields: dict[str, object] = {
        "env_lock": "a" * 64,
        "platform": "linux-x86_64",
        "wheel_set": "b" * 64,
        "numerics": {"device": "cpu", "arch": "x86_64", "precision": "fp32"},
        "decoder": "soundfile-0.12.1",
        "driver": {},
        "host": "node1",
        "protocol": "worker-c0ffee/wire-1",
    }
    fields.update(overrides)
    return ExecutionContext.model_validate(fields)


# Logical key


@pytest.mark.contract
def test_key_ignores_param_order_and_unicode_form() -> None:
    """Params are canonical JSON: member order and Unicode normal form do not change the digest."""
    a = _key(params={"x": 1, "y": "é"})
    b = _key(params={"y": unicodedata.normalize("NFD", "é"), "x": 1})
    assert a.digest() == b.digest()


@pytest.mark.contract
@pytest.mark.parametrize(
    "field,value",
    [
        ("task", "t2"),
        ("task_version", "2"),
        ("backend", "b2"),
        ("backend_version", "2"),
        ("model", MODEL),
        ("params", {"x": 2}),
        ("inputs", {"a": CONTENT}),
        ("profile_slices", {"p": PROFILE}),
        ("batch_formation", "bucket-1s"),
        ("chunk_grid", "30s"),
        ("schema_version", 2),
    ],
)
def test_key_changes_with_each_identity_field(field: str, value: object) -> None:
    """Every identity field reaches the digest."""
    assert _key(**{field: value}).digest() != _key().digest()


@pytest.mark.contract
def test_context_is_not_in_key() -> None:
    """The key holds the ADR-002 logical fields and nothing from the execution context."""
    assert set(LogicalKey.model_fields) == {
        "schema_version",
        "task",
        "task_version",
        "backend",
        "backend_version",
        "model",
        "params",
        "inputs",
        "profile_slices",
        "batch_formation",
        "chunk_grid",
    }
    assert not set(LogicalKey.model_fields) & set(ExecutionContext.model_fields)


@pytest.mark.contract
def test_digest_is_the_tagged_digest_of_the_fields() -> None:
    """The digest is reproducible from the fields alone, by any implementation of JCS and sha256."""
    k = _key(model=MODEL, params={"beam": 5, "lang": "en", "t": [0.5, None]}, inputs={"audio": CONTENT})
    expected = {
        "schema_version": 1,
        "task": "t",
        "task_version": "1",
        "backend": "b",
        "backend_version": "1",
        "model": {"source": "hf", "id": "org/name", "identity": COMMIT},
        "params": {"beam": 5, "lang": "en", "t": [0.5, None]},
        "inputs": {"audio": CONTENT},
        "profile_slices": {},
        "batch_formation": None,
        "chunk_grid": None,
    }
    assert k.model_dump() == expected
    assert k.digest() == identity_digest("logical-key@3", expected)
    assert len(k.digest()) == 64


@pytest.mark.logic
@pytest.mark.parametrize(
    "params",
    [
        {"x": math.nan},
        {"x": math.inf},
        {"x": 2**53},
        {"x": object()},
        {"x": b"bytes"},
        {"x": {1: "int key"}},
        {"x": numpy.zeros(2)},
        {"é": 1, unicodedata.normalize("NFD", "é"): 2},
    ],
)
def test_key_refuses_params_that_are_not_canonical_json(params: dict[str, object]) -> None:
    """A non-JSON param fails at construction, not later at digest time."""
    with pytest.raises(ValidationError):
        _key(params=params)


@pytest.mark.logic
def test_key_is_deeply_immutable_and_pickles() -> None:
    """Fields, nested params and mappings cannot change; the key pickles with the same digest."""
    k = _key(params={"opts": {"n": [1, 2]}}, inputs={"a": CONTENT})
    with pytest.raises(ValidationError):
        k.task = "u"  # type: ignore[misc]
    with pytest.raises(TypeError):
        k.params["opts"]["n"] = 3  # type: ignore[index]
    with pytest.raises(TypeError):
        k.inputs["b"] = CONTENT  # type: ignore[index]
    assert isinstance(k.params["opts"]["n"], tuple)
    back = pickle.loads(pickle.dumps(k))
    assert back == k and back.digest() == k.digest()
    assert LogicalKey.model_validate_json(k.model_dump_json()).digest() == k.digest()
    assert copy.deepcopy(k).digest() == k.digest() and copy.copy(k).digest() == k.digest()


@pytest.mark.logic
def test_model_copy_revalidates_and_redigests() -> None:
    """A copy with an update is a new key: validated, and digested over its own fields."""
    k = _key()
    assert k.model_copy(update={"task": "u"}).digest() == _key(task="u").digest()
    with pytest.raises(ValidationError):
        k.model_copy(update={"model": "main"})


@pytest.mark.logic
@pytest.mark.parametrize(
    "bad",
    [
        {"model": "main"},
        {"model": COMMIT},
        {"model": {"source": "hf", "id": "org/name", "identity": "main"}},
        {"model": {"source": "local", "id": "/m", "identity": COMMIT}},
        {"inputs": {"a": "/home/x/a.wav"}},
        {"inputs": {"a": "f" * 64}},
        {"inputs": {"a": "content:" + "F" * 64}},
        {"inputs": {"a": "content:" + "f" * 63}},
        {"inputs": {"a": PROFILE}},
        {"profile_slices": {"p": "f" * 64}},
        {"profile_slices": {"p": CONTENT}},
    ],
)
def test_a5_key_refuses_untyped_identities(bad: dict[str, object]) -> None:
    """The probe: a ref as the model and a path or bare hex as an input; every identity field is typed."""
    with pytest.raises(ValidationError):
        LogicalKey(**{**BASE, **bad})  # type: ignore[arg-type]


@pytest.mark.logic
@pytest.mark.parametrize("identity_", [CONTENT, OUTPUT])
def test_key_accepts_content_and_output_inputs(identity_: str) -> None:
    """An input is named by source content or by an upstream output."""
    assert _key(inputs={"a": identity_}).inputs["a"] == identity_


@pytest.mark.logic
def test_key_identity_requires_pinned_model() -> None:
    """Only a pinned model has a key identity; an unpinned one raises `UnpinnedModel`, a `ValueError`."""
    pinned = ResolvedModel(ref=ModelRef.parse("org/name@main"), identity=COMMIT)
    assert pinned.key_identity() == MODEL
    unpinned = ResolvedModel(
        ref=ModelRef.parse("url:https://x.org/m"), identity="https://x.org/m", unresolved_reason="no ETag"
    )
    with pytest.raises(UnpinnedModel):
        unpinned.key_identity()
    assert issubclass(UnpinnedModel, ValueError)


@pytest.mark.logic
def test_model_identity_applies_the_pinning_rule() -> None:
    """A model identity built directly is held to the rule `ResolvedModel` applies to a pinned model."""
    assert ModelIdentity(source="local", id="/m", identity=SHA256).identity == SHA256
    for source, ident in [("hf", "main"), ("hf", SHA256), ("url", COMMIT), ("local", "x")]:
        with pytest.raises(ValidationError):
            ModelIdentity(source=source, id="org/name", identity=ident)  # type: ignore[arg-type]


_PARAM_VALUES = st.recursive(
    st.none()
    | st.booleans()
    | st.integers()
    | st.floats()
    | st.sampled_from([1e20, -1e300, 2.0**53, 2**53, 1e-7])
    | st.text(),
    lambda children: st.lists(children, max_size=3) | st.dictionaries(st.text(max_size=3), children, max_size=3),
    max_leaves=10,
)


@pytest.mark.logic
@given(params=st.dictionaries(st.text(max_size=3), _PARAM_VALUES, max_size=4))
def test_a24_constructible_key_is_digestible(params: dict[str, object]) -> None:
    """Any params, large integral floats included: a key either fails validation or digests."""
    try:
        key = _key(params=params)
    except ValidationError:
        return
    assert len(key.digest()) == 64
    assert LogicalKey.model_validate_json(key.model_dump_json()).digest() == key.digest()


@pytest.mark.logic
def test_a24_large_float_param_is_kept_and_digested() -> None:
    """The probe: `params={"x": 1e20}` validated but could not be digested."""
    key = _key(params={"x": 1e20})
    assert key.params["x"] == 1e20 and len(key.digest()) == 64
    assert key.digest() == _key(params={"x": 10.0**20}).digest()


@pytest.mark.logic
def test_key_accepts_numpy_scalars_as_their_python_values() -> None:
    """A numpy scalar param keys like the Python number it holds."""
    assert _key(params={"x": numpy.int64(1)}).digest() == _key(params={"x": 1}).digest()


@pytest.mark.logic
@pytest.mark.parametrize("schema_version", [0, 1.0, True, "1"])
def test_key_schema_version_is_a_positive_integer(schema_version: object) -> None:
    """The schema version is a strict positive integer."""
    with pytest.raises(ValidationError):
        _key(schema_version=schema_version)


# Execution context and reuse policy


@pytest.mark.logic
def test_fingerprint_excludes_host_and_protocol() -> None:
    """Host and protocol do not reach the fingerprint; every other context field does."""
    base = _ctx()
    assert _ctx(host="node2").fingerprint() == base.fingerprint()
    assert _ctx(protocol="worker-beef/wire-2").fingerprint() == base.fingerprint()
    for field, value in [
        ("env_lock", "c" * 64),
        ("platform", "macosx-arm64"),
        ("wheel_set", "d" * 64),
        ("numerics", {"device": "cuda", "arch": "sm90", "precision": "bf16"}),
        ("decoder", None),
        ("driver", {"cuda": "12.9"}),
    ]:
        assert _ctx(**{field: value}).fingerprint() != base.fingerprint(), field


@pytest.mark.logic
def test_context_is_immutable_and_round_trips() -> None:
    """The context and its mappings cannot change; it pickles and survives JSON with its fingerprint."""
    c = _ctx(driver={"cuda": "12.9"})
    with pytest.raises(TypeError):
        c.driver["cuda"] = "13.0"  # type: ignore[index]
    assert pickle.loads(pickle.dumps(c)) == c
    assert ExecutionContext.model_validate_json(c.model_dump_json()).fingerprint() == c.fingerprint()


@pytest.mark.logic
def test_default_policy_is_equivalent() -> None:
    """The owner's default reuse policy is `equivalent`."""
    assert DEFAULT_REUSE_POLICY is ReusePolicy.EQUIVALENT
    assert [p.value for p in ReusePolicy] == ["exact", "equivalent", "tolerant"]


@pytest.mark.logic
def test_equivalent_without_measurement_is_exact() -> None:
    """With no measured equivalence class, `equivalent` reuses only the same context; `tolerant` any."""
    c1, c2 = _ctx(platform="linux"), _ctx(platform="macos")
    assert acceptable(c1, c1, ReusePolicy.EQUIVALENT, backend="b") and not acceptable(
        c1, c2, ReusePolicy.EQUIVALENT, backend="b"
    )
    assert acceptable(c1, c2, ReusePolicy.TOLERANT, backend="b")
    assert acceptable(c1, c1, ReusePolicy.EXACT, backend="b") and not acceptable(c1, c2, ReusePolicy.EXACT, backend="b")


@pytest.mark.logic
@pytest.mark.parametrize("policy", ["exact", "bogus", "EXACT", "", 1, None, True])
def test_a23_acceptable_coerces_policy(policy: object) -> None:
    """A policy value is coerced with `ReusePolicy(policy)`: `"exact"` is exact, anything else raises."""
    c1, c2 = _ctx(platform="linux"), _ctx(platform="macos")
    if policy == "exact":
        assert acceptable(c1, c1, "exact", backend="b") and not acceptable(c1, c2, "exact", backend="b")
        return
    with pytest.raises(ValueError):
        acceptable(c1, c2, policy, backend="b")  # type: ignore[arg-type]


@pytest.mark.logic
@pytest.mark.parametrize("policy", list(ReusePolicy))
def test_every_policy_member_is_dispatched(policy: ReusePolicy) -> None:
    """Each member and its value give the same answer for the same and for different contexts."""
    c1, c2 = _ctx(platform="linux"), _ctx(platform="macos")
    for a, b in [(c1, c1), (c1, c2)]:
        assert acceptable(a, b, policy, backend="b") == acceptable(a, b, policy.value, backend="b")
    assert acceptable(c1, c1, policy, backend="b")


@pytest.mark.logic
def test_fingerprint_is_the_context_tagged_digest() -> None:
    """The fingerprint is `identity_digest("context@3", fields without host and protocol)`."""
    c = _ctx()
    assert c.fingerprint() == identity_digest("context@3", c.model_dump(exclude={"host", "protocol"}))


@pytest.mark.logic
def test_another_host_is_the_same_context() -> None:
    """A result from another host with an identical context is reused even under `exact`."""
    assert acceptable(_ctx(host="laptop"), _ctx(host="node7"), ReusePolicy.EXACT, backend="b")


@pytest.mark.logic
def test_another_protocol_is_the_same_context() -> None:
    """A result recorded under another worker protocol with an identical context is reused under `exact`."""
    stored, current = _ctx(protocol="worker-c0ffee/wire-1"), _ctx(protocol="worker-beef/wire-2")
    assert acceptable(stored, current, ReusePolicy.EXACT, backend="b")
    assert stored.protocol != current.protocol


@pytest.mark.logic
def test_equivalent_reuses_within_a_measured_class(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two contexts in one of the backend's measured classes are equivalent; `exact` still refuses them."""
    c1, c2, c3 = _ctx(platform="linux"), _ctx(platform="macos"), _ctx(platform="windows")
    classes = {"cpu-fp32": frozenset({c1.fingerprint(), c2.fingerprint()})}
    monkeypatch.setattr(identity, "EQUIVALENCE", {"b": classes})
    assert acceptable(c1, c2, ReusePolicy.EQUIVALENT, backend="b") and acceptable(
        c2, c1, ReusePolicy.EQUIVALENT, backend="b"
    )
    assert not acceptable(c1, c3, ReusePolicy.EQUIVALENT, backend="b")
    assert not acceptable(c1, c2, ReusePolicy.EXACT, backend="b")


@pytest.mark.logic
def test_equivalence_is_scoped_to_its_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """A class measured for backend A does not make the same contexts equivalent for backend B."""
    c1, c2 = _ctx(platform="linux"), _ctx(platform="macos")
    monkeypatch.setattr(identity, "EQUIVALENCE", {"A": {"cpu-fp32": frozenset({c1.fingerprint(), c2.fingerprint()})}})
    assert acceptable(c1, c2, ReusePolicy.EQUIVALENT, backend="A")
    assert not acceptable(c1, c2, ReusePolicy.EQUIVALENT, backend="B")


@pytest.mark.logic
def test_phase_1_has_no_measured_equivalence() -> None:
    """The packaged equivalence table is empty in Phase 1."""
    assert dict(identity.EQUIVALENCE) == {}


# Model references


@pytest.mark.logic
def test_modelref_parse() -> None:
    """`org/name@rev` is a hub reference."""
    assert ModelRef.parse("openai/whisper-large-v3-turbo@main") == ModelRef(
        source="hf", id="openai/whisper-large-v3-turbo", ref="main"
    )


@pytest.mark.logic
@pytest.mark.parametrize(
    "text,expected",
    [
        ("gpt2@" + COMMIT, ModelRef(source="hf", id="gpt2", ref=COMMIT)),
        ("org/name@refs/pr/1", ModelRef(source="hf", id="org/name", ref="refs/pr/1")),
        (
            "url:https://example.org/w/model.onnx",
            ModelRef(source="url", id="https://example.org/w/model.onnx", ref=""),
        ),
        (
            "url:https://user@example.org/model.onnx",
            ModelRef(source="url", id="https://user@example.org/model.onnx", ref=""),
        ),
        ("local:/models/yunet", ModelRef(source="local", id="/models/yunet", ref="")),
        ("local:weights/v1@x", ModelRef(source="local", id="weights/v1@x", ref="")),
    ],
)
def test_modelref_parse_sources(text: str, expected: ModelRef) -> None:
    """`url:` and `local:` take the rest of the string verbatim as the id and have no ref; `str` inverts parse."""
    assert ModelRef.parse(text) == expected
    assert str(expected) == text


@pytest.mark.logic
@pytest.mark.parametrize(
    "text",
    [
        "openai/whisper",
        "openai/whisper@",
        "@main",
        "a/b/c@main",
        "org/na me@main",
        "org/name@ma in",
        "url:",
        "url:example.org/model",
        "url:https://",
        "local:",
        "s3:bucket/model@main",
        "",
    ],
)
def test_modelref_parse_refuses_malformed(text: str) -> None:
    """A hub reference needs an explicit `@ref`; `url:` needs an absolute URL; `local:` needs a path."""
    with pytest.raises(ValueError):
        ModelRef.parse(text)


@pytest.mark.logic
def test_modelref_validates_direct_construction() -> None:
    """The same rules hold when the fields are given directly."""
    with pytest.raises(ValidationError):
        ModelRef(source="hf", id="org/name", ref="")
    with pytest.raises(ValidationError):
        ModelRef(source="local", id="/m", ref="main")


# Resolved models


@pytest.mark.logic
def test_resolved_model_pins_by_source() -> None:
    """A pinned hub model's identity is a 40-hex commit; a pinned url or local model's is a sha256."""
    hf = ModelRef.parse("org/name@main")
    local = ModelRef.parse("local:/m")
    assert ResolvedModel(ref=hf, identity=COMMIT).pinned
    assert ResolvedModel(ref=local, identity=SHA256).pinned
    with pytest.raises(ValidationError):
        ResolvedModel(ref=hf, identity="main")
    with pytest.raises(ValidationError):
        ResolvedModel(ref=local, identity=COMMIT)


@pytest.mark.logic
def test_unresolved_model_carries_its_reason() -> None:
    """An unpinnable source resolves with a reason and is not pinned; an empty reason is refused."""
    unresolved = ResolvedModel(
        ref=ModelRef.parse("url:https://x.org/m"), identity="https://x.org/m", unresolved_reason="no ETag"
    )
    assert not unresolved.pinned
    with pytest.raises(ValidationError):
        ResolvedModel(ref=ModelRef.parse("url:https://x.org/m"), identity="x", unresolved_reason="")


@pytest.mark.logic
def test_resolved_model_carries_license_and_pickles() -> None:
    """A resolved model may carry its license record; the whole pickles and round-trips through JSON."""
    lic = License(
        expression="MIT",
        commercial_use="yes",
        copyleft="none",
        attribution=True,
        gated=False,
        source="org/name LICENSE at " + COMMIT,
        verified={"date": "2026-10-30", "how": "hf-card"},
    )
    m = ResolvedModel(ref=ModelRef.parse("org/name@main"), identity=COMMIT, license=lic)
    assert pickle.loads(pickle.dumps(m)) == m
    assert ResolvedModel.model_validate_json(m.model_dump_json()) == m
