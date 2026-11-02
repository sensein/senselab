"""The `vllm` backend of `pii_spans`: the engine over an attested `VllmServer`, selected by the profile alone."""

import stat
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy
import pytest
from senselab._api.packaged import CATALOGUE
from senselab.text.tasks.pii_spans import (
    engine,
    transformers_backend,  # noqa: F401 - declares the backend
    vllm_backend,  # noqa: F401 - declares the backend
)
from senselab.text.tasks.pii_spans import vllm as worker
from senselab_core.kernel.declare import REGISTRY, Registry
from senselab_core.kernel.identity import ModelRef, ResolvedModel
from senselab_core.kernel.results import TextSpans
from senselab_core.runtime.call import CallFailed, Packaged, TaskCall, run_call, use_packaged
from senselab_core.runtime.envs import EnvHandle
from senselab_core.runtime.hardware import GpuInfo, HardwareInfo
from senselab_core.runtime.profiles import ExecutionProfile, resolve_execution
from senselab_core.runtime.store import ResultStore
from senselab_core.runtime.vllm_server import CONFIG_TAG, AttestationMismatch, VllmServer, server_args
from senselab_core.testing.contracts import contract_cases
from senselab_core.testing.lock_licenses import packaged_lock_records
from senselab_worker.digest import identity_digest
from senselab_worker.faults import FaultClass

LOCK_RECORDS = packaged_lock_records("senselab")

FAKE = Path(__file__).parents[5] / "senselab-core/tests/runtime/fake_vllm_fixture.py"
ENTRY = CATALOGUE.entry(engine.MODEL_ID)
CONFIG = vllm_backend.server_config()
GPU_HOST = HardwareInfo(
    cpu_count=1,
    memory_bytes=1,
    gpus=(GpuInfo(name="H100", memory_bytes=80 * 2**30, driver=None, index=0, uuid="GPU-0", compute_capability="9.0"),),
    apple_silicon=False,
    platform="linux-x86_64",
)
ANSWER = '{"spans": [{"quote": "555-0100", "category": "CONTACT", "score": 0.9}]}'
TEXT = "Call me on 555-0100 today."


def _profile(tmp_path: Path, *stacks: str, name: str = "laptop") -> ExecutionProfile:
    return ExecutionProfile.load(
        name, {"stack_preferences": {"pii_spans": list(stacks)}, "env_roots": {"server": str(tmp_path / "servers")}}
    )


def _spans(outputs: dict[str, object]) -> TextSpans:
    spans = outputs["spans"]
    assert isinstance(spans, TextSpans)
    return spans


def _make_env(root: Path, name: str) -> EnvHandle:
    """A server env whose `bin/vllm` is the fake server and whose lock pins vllm 0.31.0."""
    exe = root / name / "bin" / "vllm"
    exe.parent.mkdir(parents=True)
    exe.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{FAKE}" "$@"\n')
    exe.chmod(exe.stat().st_mode | stat.S_IXUSR)
    lock = root / "vllm-server.txt"
    lock.write_text("vllm==0.31.0 \\\n    --hash=sha256:00\n")
    return EnvHandle(
        python=str(exe.with_name("python")),
        python_version="3.12",
        lock_sha="l" * 8,
        wheel_set="w" * 8,
        torch_index=None,
        lock=lock,
    )


def _starts(root: Path) -> int:
    return len(list((root / "fake").glob("*.json")))


@pytest.fixture
def fake_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> EnvHandle:
    """A server env on the fake server, replying `ANSWER`; the fake records each start in `<tmp>/fake`."""
    (tmp_path / "fake").mkdir()
    monkeypatch.setenv("FAKE_VLLM_DIR", str(tmp_path / "fake"))
    monkeypatch.setenv("FAKE_VLLM_CONTENT", ANSWER)
    return _make_env(tmp_path, "env")


@pytest.fixture
def served(fake_env: EnvHandle) -> Iterator[Registry]:
    """The packaged catalogue and backends with `vllm` placeable on this host (CPU only) and its env the fake."""
    registry = Registry()
    spec = REGISTRY.spec("pii_spans")
    registry.add_task(spec)
    for backend in REGISTRY.backends("pii_spans"):
        registry.add_backend(backend.model_copy(update={"devices": ("cpu",)}) if backend.name == "vllm" else backend)
    opened = [fake_env]
    previous = use_packaged(Packaged(CATALOGUE, lambda _b, _p: opened[-1], registry))
    registry.opened = opened  # type: ignore[attr-defined]
    yield registry
    use_packaged(previous)


def _call() -> TaskCall:
    return _call_of(TEXT)


def _server(env: EnvHandle, tmp_path: Path) -> VllmServer:
    model = ResolvedModel(ref=ModelRef.parse(f"{ENTRY.id}@{ENTRY.revision}"), identity=ENTRY.revision)
    return VllmServer(
        env,
        model,
        CONFIG.settings,
        start_lock_dir=tmp_path / "other-locks",
        startup_timeout_s=30,
        start_margin_s=1,
    )


@pytest.mark.contract
def test_backend_serves_the_catalogue_model_through_a_server_env() -> None:
    """One backend `vllm` of stack `vllm` over the catalogue's model, in `Env.server`, with its engine configuration."""
    spec = next(b for b in REGISTRY.backends("pii_spans") if b.name == "vllm")
    assert (spec.stack, spec.models, spec.devices) == ("vllm", (engine.MODEL_ID,), ("cuda",))
    assert spec.env.kind == "server" and spec.env.name == "vllm-server" and spec.batch_invariant
    assert spec.server is not None and spec.server.startup_timeout_s == CONFIG.startup_timeout_s
    assert server_args(spec.server.settings)
    assert list(ENTRY.backends) == ["transformers", "vllm"] and list(ENTRY.stacks) == ["transformers", "vllm"]


@pytest.mark.contract
def test_stack_switch_by_profile(tmp_path: Path) -> None:
    """The profile's stack preference alone moves `pii_spans` between backends; the task and its outputs stay."""
    spec = REGISTRY.spec("pii_spans")
    chosen = {}
    for stacks in (("transformers", "vllm"), ("vllm", "transformers"), ()):
        placed = resolve_execution(spec, ENTRY, _profile(tmp_path, *stacks), GPU_HOST, REGISTRY, CATALOGUE)
        chosen[stacks] = (placed.backend.name, placed.backend.env.kind)
    assert chosen[("transformers", "vllm")] == ("transformers", "isolated")
    assert chosen[("vllm", "transformers")] == ("vllm", "server")
    assert chosen[()] == ("transformers", "isolated")
    assert {b.task for b in REGISTRY.backends("pii_spans")} == {spec.name}


@pytest.mark.logic
def test_run_call_serves_spans_through_the_server_and_records_its_attestation(
    served: Registry,  # noqa: ARG001 - the fixture's effect
    fake_env: EnvHandle,
    tmp_path: Path,
) -> None:
    """A `vllm` call returns the answer's spans, and its context carries what an identical server attests."""
    profile = _profile(tmp_path, "vllm")
    first = run_call(_call(), execution=profile, store=ResultStore(tmp_path / "results"))
    spans = _spans(first.outputs)
    assert [TEXT[a:b] for a, b in zip(spans.start, spans.end, strict=True)] == ["555-0100"]
    assert spans.label.tolist() == ["CONTACT"] and spans.score is not None and numpy.allclose(spans.score, [0.9])
    assert first.placement.backend.name == "vllm" and first.reused is None
    with _server(fake_env, tmp_path) as server:
        assert server.attestation is not None
        assert first.context.driver == {**fake_env.context_fields()["driver"], **server.attestation.driver()}
        assert server.attestation.config_digest == identity_digest(CONFIG_TAG, server_args(CONFIG.settings))
    assert first.context.driver["model_identity"] == ENTRY.revision and first.context.driver["engine"] == "vllm"
    second = run_call(_call(), execution=profile, store=ResultStore(tmp_path / "results"))
    assert second.reused is not None and second.context == first.context


@pytest.mark.logic
def test_run_call_fails_the_item_when_the_answer_is_not_spans(
    served: Registry,  # noqa: ARG001 - the fixture's effect
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A server answer that is not the span object asked for is the item's content fault."""
    monkeypatch.setenv("FAKE_VLLM_CONTENT", "I cannot help with that.")
    with pytest.raises(CallFailed) as raised:
        run_call(_call(), execution=_profile(tmp_path, "vllm"), store=ResultStore(tmp_path / "results"))
    assert raised.value.error["fault"] is FaultClass.CONTENT


@pytest.mark.logic
def test_load_takes_a_pinned_model_with_a_server_or_the_stub() -> None:
    """A model without its running server, a ref, an unknown key and both stub and model are refused."""
    model = {"source": "hf", "id": engine.MODEL_ID, "identity": ENTRY.revision}
    for init in (
        {"model": model},
        {"model": {**model, "identity": "main"}, "server": object()},
        {"stub": True, "bogus": 1},
        {"stub": True, "model": model, "server": object()},
        {},
    ):
        with pytest.raises(ValueError):
            worker.load(init)
    assert isinstance(worker.load({"stub": True}).model, engine.StubModel)
    assert isinstance(worker.load({"model": model, "server": object()}).model, worker.ServerModel)


@pytest.mark.contract
@pytest.mark.parametrize(
    "case",
    [
        pytest.param(c, id=name)
        for name, c in contract_cases(REGISTRY, lock_records=LOCK_RECORDS)
        if name.startswith(("pii_spans/vllm::", "vllm::"))
    ],
)
def test_contract(case: Any) -> None:  # noqa: ANN401 - a no-argument contract case
    """The backend satisfies the generated contract, its worker running on its load stub."""
    case()


@pytest.mark.limits
@pytest.mark.gpu
@pytest.mark.model
@pytest.mark.network
def test_pii_spans_vllm_on_the_h100_finds_spans_and_bounds_edge_inputs(tmp_path: Path) -> None:
    """Through `run_call` on the orcd profile the served pinned model covers a name and a number, attested.

    Empty, non-BMP and long texts each give spans inside the text or a typed fault.
    """
    profile = ExecutionProfile.load("orcd", {"stack_preferences": {"pii_spans": ["vllm"]}})
    store = ResultStore(tmp_path / "results")

    def run(text: str) -> Any:  # noqa: ANN401
        return run_call(_call_of(text), execution=profile, store=store)

    text = "My name is Margaret Hollis and you can reach me on 617-555-0142."
    result = run(text)
    spans = _spans(result.outputs)
    covered = {text[a:b] for a, b in zip(spans.start, spans.end, strict=True)}
    assert any("Hollis" in c for c in covered) and any("555-0142" in c for c in covered), covered
    assert result.placement.backend.name == "vllm" and result.context.driver["model_identity"] == ENTRY.revision
    for content in ("", "😀😀 I am Dana Wu 😀", "The meeting ran long. " * 400 + "Call Dana Wu on 617-555-0142."):
        try:
            out = _spans(run(content).outputs)
        except CallFailed as failed:
            assert failed.error["fault"] in (FaultClass.CONTENT, FaultClass.RESOURCE), failed.error
        else:
            assert out.length == len(content)
            assert all(0 <= a <= b <= out.length for a, b in zip(out.start, out.end, strict=True))


def _call_of(text: str) -> TaskCall:
    return TaskCall.model_validate({"task": "pii_spans", "inputs": {"text": text}, "params": {}})


@pytest.mark.logic
def test_a_store_hit_starts_no_server_and_a_mismatching_attestation_refuses(
    served: Registry, fake_env: EnvHandle, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The context holds the declared attestation, so a hit needs no server; a started server must attest it."""
    profile, store = _profile(tmp_path, "vllm"), ResultStore(tmp_path / "results")
    run_call(_call(), execution=profile, store=store)
    assert _starts(tmp_path) == 1
    (tmp_path / "fake2").mkdir()
    monkeypatch.setenv("FAKE_VLLM_DIR", str(tmp_path / "fake2"))
    served.opened.append(_make_env(tmp_path, "env2"))  # type: ignore[attr-defined]
    assert run_call(_call(), execution=profile, store=store).reused is not None
    assert not list((tmp_path / "fake2").glob("*.json"))
    monkeypatch.setenv("FAKE_VLLM_VERSION", "9.9.9")
    served.opened.append(_make_env(tmp_path, "env3"))  # type: ignore[attr-defined]
    with pytest.raises(AttestationMismatch, match="9.9.9"):
        run_call(_call_of("Call 555-0199."), execution=profile, store=store)
