"""`build_key`: the one key builder over a call, its placement and the execution context."""

from dataclasses import replace
from typing import Any

import pytest
from senselab_core.kernel.declare import BackendParams, BackendSpec, Env, TaskParams, TaskSpec
from senselab_core.kernel.identity import ExecutionContext
from senselab_core.kernel.plan import slice_identity
from senselab_core.runtime.keys import Call, MissingProfileSlice, Placement, build_key, profile_slices
from senselab_core.testing.contracts import BASE_CONTEXT

TASK = TaskSpec(name="t", version="1", kind="transform", inputs={"audio": "Audio"}, outputs={}, params=TaskParams)
CALL = Call(task=TASK, params=TaskParams(), inputs={"audio": "content:" + "a" * 64})
CONTEXT = ExecutionContext.model_validate(BASE_CONTEXT)


class _Beam(BackendParams):
    beam: int = 1


def _placement(**backend: object) -> Placement:
    spec = BackendSpec(
        task="t",
        name="b",
        version="1",
        stack="fake",
        env=Env.host(),
        devices=("cpu",),
        batch_invariant=False,
        worker="senselab_worker.fake",
        load_stub={},
        params=_Beam,
        **backend,
    )
    return Placement(backend=spec, backend_params=_Beam(), device="cpu", batch_formation="sorted:8", chunk_grid="30s")


@pytest.mark.logic
def test_output_affecting_enters_key_only_when_declared() -> None:
    """A placement's batch formation and chunk grid reach the key only when the backend declares them."""
    undeclared = _placement()
    key = build_key(CALL, undeclared, CONTEXT)
    assert (key.batch_formation, key.chunk_grid) == (None, None)
    other = replace(undeclared, batch_formation="sorted:16", chunk_grid="10s")
    assert build_key(CALL, other, CONTEXT).digest() == key.digest()
    declared = _placement(output_affecting=("batch_formation", "chunk_grid"))
    key = build_key(CALL, declared, CONTEXT)
    assert (key.batch_formation, key.chunk_grid) == ("sorted:8", "30s")
    assert build_key(CALL, replace(declared, batch_formation="other"), CONTEXT).digest() != key.digest()
    assert build_key(CALL, replace(declared, chunk_grid="other"), CONTEXT).digest() != key.digest()


@pytest.mark.logic
def test_backend_version_enters_key() -> None:
    """The backend's name, version and params reach the key."""
    placement = _placement()
    key = build_key(CALL, placement, CONTEXT)
    assert (key.backend, key.backend_version, dict(key.backend_params)) == ("b", "1", {"beam": 1})
    bumped = replace(placement, backend=placement.backend.model_copy(update={"version": "2"}))
    assert build_key(CALL, bumped, CONTEXT).digest() != key.digest()
    tuned = replace(placement, backend_params=_Beam(beam=4))
    assert build_key(CALL, tuned, CONTEXT).digest() != key.digest()


@pytest.mark.logic
def test_key_ignores_context_through_builder() -> None:
    """Two execution contexts and two devices give one key; a context of another type is refused."""
    placement = _placement()
    key = build_key(CALL, placement, CONTEXT)
    other = ExecutionContext.model_validate({**BASE_CONTEXT, "platform": "macos-arm64"})
    assert build_key(CALL, placement, other).digest() == key.digest()
    assert build_key(CALL, replace(placement, device="cuda"), other).digest() == key.digest()
    with pytest.raises(TypeError, match="ExecutionContext"):
        build_key(CALL, placement, BASE_CONTEXT)  # type: ignore[arg-type]


@pytest.mark.logic
def test_declared_slices_reach_the_key_by_identity() -> None:
    """A task's declared slices are cut by dotted path, keyed by identity, and enter the key; nothing else is read."""
    reads = TASK.model_copy(update={"profile_slices": ("run.agg.name", "dataset.lexicon")})
    cut = profile_slices(reads, {"agg": {"name": "mean"}, "x": 1}, {"lexicon": ["a"], "y": 2})
    assert cut == {"run.agg.name": slice_identity("mean"), "dataset.lexicon": slice_identity(["a"])}
    assert profile_slices(reads, {"agg": {"name": "mean"}}, {"lexicon": ["a"]}) == cut
    key = build_key(replace(CALL, profile_slices=cut), _placement(), CONTEXT)
    assert dict(key.profile_slices) == cut
    with pytest.raises(MissingProfileSlice, match="no dataset profile"):
        profile_slices(reads, {"agg": {"name": "mean"}})
    with pytest.raises(MissingProfileSlice, match="'name'"):
        profile_slices(reads, {"agg": {}}, {"lexicon": []})


@pytest.mark.logic
@pytest.mark.parametrize(
    "field,value",
    [
        ("device", "cuda"),
        ("env_root", "/elsewhere"),
        ("why", "another reason"),
        ("resource_class", "gpu:h100"),
        ("batch_size", 16),
        ("partition", "${GPU_PARTITION}"),
    ],
)
def test_nonkey_placement_fields_never_in_key(field: str, value: Any) -> None:  # noqa: ANN401 - a field value
    """Varying a placement field `build_key` does not read leaves the key, with and without output-affecting ones."""
    for declared in ((), ("batch_formation", "chunk_grid")):
        base = _placement(output_affecting=declared)
        varied = replace(base, **{field: value})
        assert build_key(CALL, varied, CONTEXT) == build_key(CALL, base, CONTEXT)
