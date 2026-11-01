"""`build_key`: the one key builder over a call, its placement and the execution context."""

from dataclasses import replace

import pytest
from senselab_core.kernel.declare import BackendParams, BackendSpec, Env, TaskParams, TaskSpec
from senselab_core.kernel.identity import ExecutionContext
from senselab_core.runtime.keys import Call, Placement, build_key
from senselab_core.testing.contracts import BASE_CONTEXT

TASK = TaskSpec(name="t", version="1", kind="measure", inputs={"audio": "Audio"}, outputs={}, params=TaskParams)
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
    other = ExecutionContext.model_validate({**BASE_CONTEXT, "platform": "macos-arm64", "host": "elsewhere"})
    assert build_key(CALL, placement, other).digest() == key.digest()
    assert build_key(CALL, replace(placement, device="cuda"), other).digest() == key.digest()
    with pytest.raises(TypeError, match="ExecutionContext"):
        build_key(CALL, placement, BASE_CONTEXT)  # type: ignore[arg-type]
