"""Services: a registry entry under a renewed lease, consumer leases deciding idle exit, and per-reply attestation."""

import errno
import json
import platform
import threading
import urllib.error
import uuid
from collections.abc import Callable, Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest
import senselab_core.runtime.fake_backend as fake
from lease_clock_fixture import ManualClock, age
from planted_objects_fixture import MIRROR_INDEX, identity_leaks
from senselab_core.kernel.declare import BackendSpec, Env, ServerConfig
from senselab_core.kernel.identity import Attestation, ModelRef, ResolvedModel
from senselab_core.runtime.envs import EnvHandle
from senselab_core.runtime.execution.executor import FaultRecord
from senselab_core.runtime.execution.providers import Loader, Model, SenselabWorker
from senselab_core.runtime.execution.services import ServiceEntry, ServiceRegistry, service_registry
from senselab_core.runtime.hardware import host_token
from senselab_core.runtime.locks import LeaseTimeout
from senselab_core.runtime.profiles import ExecutionProfile
from senselab_core.runtime.teardown import run_all
from senselab_core.runtime.vllm_server import ServerRejected, ServiceUnavailable, StaleAttestation, chat, server_fault
from senselab_core.runtime.worker_client import WorkerClient
from senselab_core.testing.interpreter import source_tree_python
from syscall_faults_fixture import refuse_call

pytestmark = pytest.mark.logic

SHA = "a" * 40
MODEL = ResolvedModel(ref=ModelRef.parse(f"org/m@{SHA}"), identity=SHA)
IDENTITY = Attestation(engine_version="0.31.0", served_model="org/m", model_identity=SHA, config_digest="c" * 64)
ENV = EnvHandle(source_tree_python(), platform.python_version(), "1" * 64, "2" * 64, MIRROR_INDEX)
BACKEND = fake.FAKE.model_copy(update={"name": "served", "env": Env.server("s"), "server": ServerConfig(settings={})})
REQUEST_S = 600.0
PROFILE = ExecutionProfile.load(
    "laptop",
    {
        "devices": ["cpu"],
        "resource_classes": {"svc:pii": {"device": "cpu", "max_workers": 1}},
        "pool": {"poll_s": 0.05, "first_poll_s": 0.01},
        "lease": {"stale_after_s": 0.4, "renew_fraction": 0.2, "attribute_delay_s": 0},
        "services": {"idle_exit_s": 0.15},
    },
)
CHAT = [{"role": "user", "content": "hi"}]


class StandIn:
    """A stdlib HTTP server answering chat requests as `reply_model`; a `workers.Server` attesting `identity`.

    With `drop` it closes each connection unanswered, as a server dying mid-request; with `status` it refuses it.
    """

    def __init__(self, identity: Attestation) -> None:
        """Listen on a free loopback port."""
        self.label, self.identity, self.reply_model = identity.served_model, identity, identity.served_model
        self.drop, self.status = False, 200
        stand_in = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: object) -> None:
                """Log nothing."""

            def do_POST(self) -> None:  # noqa: N802 - the http.server hook name
                self.rfile.read(int(self.headers["Content-Length"]))
                if stand_in.drop:
                    self.close_connection = True
                    return
                body = json.dumps({"model": stand_in.reply_model, "choices": [{"message": {"content": "ok"}}]})
                self.send_response(stand_in.status)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body.encode())

        self._http = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self._http.serve_forever, daemon=True).start()
        self.url: str | None = f"http://127.0.0.1:{self._http.server_address[1]}"

    @property
    def attestation(self) -> Attestation | None:
        """The identity while listening."""
        return self.identity if self.url is not None else None

    def context_fields(self) -> dict[str, Any]:
        """No fields."""
        return {}

    def stop(self) -> None:
        """Stop listening."""
        if self.url is not None:
            self._http.shutdown()
            self._http.server_close()
            self.url = None


@pytest.fixture
def registry(tmp_path: Path) -> ServiceRegistry:
    """A registry under `tmp_path` with the profile's short lease."""
    return ServiceRegistry(tmp_path / "registry", lease=PROFILE.lease)


@pytest.fixture
def starts() -> Iterator[list[StandIn]]:
    """The stand-in servers started by a test, stopped after it."""
    started: list[StandIn] = []
    yield started
    for server in started:
        server.stop()


def test_service_registered_under_renewed_lease(tmp_path: Path, lease_clock: Callable[..., ManualClock]) -> None:
    """An entry `{identity, endpoint}` stays live past its stale bound while its lease beats, and not once it stops."""
    clock = lease_clock()
    registry = ServiceRegistry(tmp_path / "registry", lease=PROFILE.lease, clock=clock)
    service = registry.register(IDENTITY, "http://127.0.0.1:1")
    try:
        beat_s = PROFILE.lease.renew_fraction * PROFILE.lease.stale_after_s
        clock.settle()
        for _ in range(round(3 * PROFILE.lease.stale_after_s / beat_s)):
            age(registry.root, beat_s)
            clock.advance(beat_s)
            clock.settle()
        assert registry.lookup(IDENTITY) == ServiceEntry(identity=IDENTITY, endpoint="http://127.0.0.1:1")
        assert service.state.beats >= 2 and not service.state.lost
        written = json.loads((registry.root / f"{registry.name(IDENTITY)}.json").read_text())
        assert set(written) == {"format", "identity", "endpoint"}
        with pytest.raises(LeaseTimeout):
            registry.register(IDENTITY, "http://127.0.0.1:2")
    finally:
        service.close()
    assert registry.lookup(IDENTITY) is None
    name = registry.name(IDENTITY)
    (registry.root / f"{name}.json").write_text(
        ServiceEntry(identity=IDENTITY, endpoint="http://dead").model_dump_json()
    )
    token = registry.root / f"{name}.lease" / uuid.uuid4().hex
    token.mkdir(parents=True)
    age(registry.root, 10 * PROFILE.lease.stale_after_s)
    assert registry.lookup(IDENTITY) is None
    registry.register(IDENTITY, "http://127.0.0.1:3").close()


def test_attestation_checked_per_reply(registry: ServiceRegistry, starts: list[StandIn]) -> None:
    """A reply must name the attested model of a live entry; another model, a withdrawn or reconfigured entry fail."""
    starts.append(StandIn(IDENTITY))
    service = registry.register(IDENTITY, str(starts[0].url))
    consumer = registry.consume(IDENTITY)
    try:
        assert consumer.chat(CHAT, sampling={}).attestation == IDENTITY
        starts[0].reply_model = "org/other"
        with pytest.raises(StaleAttestation, match="another model"):
            consumer.chat(CHAT, sampling={})
        starts[0].reply_model = IDENTITY.served_model
        service.close()
        with pytest.raises(StaleAttestation, match="not registered"):
            consumer.chat(CHAT, sampling={})
        reconfigured = IDENTITY.model_copy(update={"config_digest": "d" * 64})
        starts.append(StandIn(reconfigured))
        service = registry.register(reconfigured, str(starts[1].url))
        with pytest.raises(StaleAttestation, match="not registered"):
            consumer.chat(CHAT, sampling={})
        service.close()
        starts.append(StandIn(IDENTITY))
        service = registry.register(IDENTITY, str(starts[2].url))
        assert consumer.url == starts[2].url and consumer.chat(CHAT, sampling={}).attestation == IDENTITY
    finally:
        consumer.stop()
        service.close()


SETTINGS: dict[str, Any] = {
    "max_model_len": 4096,
    "max_num_seqs": 4,
    "gpu_memory_utilization": 0.9,
    "kv_cache_dtype": "auto",
    "generation_config": "vllm",
    "limit_mm_per_prompt": {"image": 0},
    "enable_prefix_caching": False,
    "flashinfer_sampler": False,
}


def test_a_dead_consumer_counts_until_its_lease_is_stale(registry: ServiceRegistry) -> None:
    """A consumer that died without releasing counts as live until its lease is stale, then is removed."""
    consumers = registry.root / f"{registry.name(IDENTITY)}.consumers"
    token = consumers / f"{uuid.uuid4().hex}.lease" / uuid.uuid4().hex
    token.mkdir(parents=True)
    assert registry.live_consumers(IDENTITY) == 1
    age(consumers, 10 * PROFILE.lease.stale_after_s)
    assert registry.live_consumers(IDENTITY) == 0 and list(consumers.iterdir()) == []


def test_a_reservation_withdraws_a_dead_holders_entry(registry: ServiceRegistry) -> None:
    """Taking over a dead holder's registration removes its entry before the new one is published."""
    name = registry.name(IDENTITY)
    registry.root.mkdir(parents=True)
    (registry.root / f"{name}.json").write_text(
        ServiceEntry(identity=IDENTITY, endpoint="http://dead").model_dump_json()
    )
    token = registry.root / f"{name}.lease" / uuid.uuid4().hex
    token.mkdir(parents=True)
    age(registry.root, 10 * PROFILE.lease.stale_after_s)
    service = registry.reserve(IDENTITY)
    try:
        assert registry.lookup(IDENTITY) is None and not (registry.root / f"{name}.json").exists()
        service.publish("http://127.0.0.1:4")
        assert registry.lookup(IDENTITY) == ServiceEntry(identity=IDENTITY, endpoint="http://127.0.0.1:4")
    finally:
        service.close()


def test_a_server_dying_mid_request_is_unavailable(registry: ServiceRegistry, starts: list[StandIn]) -> None:
    """A connection the server drops is `ServiceUnavailable`; a request it refuses is a `ServerRejected`."""
    starts.append(StandIn(IDENTITY))
    service = registry.register(IDENTITY, str(starts[0].url))
    consumer = registry.consume(IDENTITY)
    try:
        starts[0].drop = True
        with pytest.raises(ServiceUnavailable, match="did not answer"):
            consumer.chat(CHAT, sampling={})
        starts[0].drop, starts[0].status = False, 400
        with pytest.raises(ServerRejected):
            consumer.chat(CHAT, sampling={})
    finally:
        consumer.stop()
        service.close()


def _fault_of(call: Callable[[], Any]) -> tuple[FaultRecord, str]:
    """Return the record and the type of what `call` raises."""
    with pytest.raises(Exception) as raised:  # noqa: PT011, B017 - whatever the boundary raises, typed
        call()
    return FaultRecord.from_exception(raised.value), type(raised.value).__name__


@pytest.mark.parametrize(
    ("status", "fault", "gap"),
    [
        (400, "content", False),
        (404, "content", False),
        (422, "content", False),
        (408, "infrastructure", False),
        (429, "infrastructure", False),
        (500, "infrastructure", False),
        (502, "infrastructure", False),
        (503, "infrastructure", False),
        (507, "resource", False),
    ],
)
def test_a_server_status_is_typed_the_same_on_both_paths(
    registry: ServiceRegistry, starts: list[StandIn], status: int, fault: str, gap: bool
) -> None:
    """The direct call and the consumer lease give the same record for each status; a typed fault is no gap."""
    starts.append(StandIn(IDENTITY))
    service = registry.register(IDENTITY, str(starts[0].url))
    consumer = registry.consume(IDENTITY)
    starts[0].status = status
    try:
        direct = _fault_of(
            lambda: chat(str(starts[0].url), IDENTITY.served_model, CHAT, sampling={}, timeout_s=REQUEST_S)
        )
        leased = _fault_of(lambda: consumer.chat(CHAT, sampling={}))
    finally:
        consumer.stop()
        service.close()
    assert direct == leased
    assert (direct[0].fault, direct[0].gap, direct[0].charged) == (fault, gap, fault == "content")


def test_an_http_reply_naming_memory_is_a_resource_fault_whatever_its_status() -> None:
    """A 5xx whose reason names memory is a `ServerOutOfMemory`; the same status without it is not."""
    named = urllib.error.HTTPError("http://x", 500, "CUDA out of memory", None, None)  # type: ignore[arg-type]
    plain = urllib.error.HTTPError("http://x", 500, "boom", None, None)  # type: ignore[arg-type]
    assert FaultRecord.from_exception(server_fault(named)).fault == "resource"
    assert FaultRecord.from_exception(server_fault(plain)).fault == "infrastructure"
    refused = urllib.error.HTTPError("http://x", 400, "prompt says oom: out of memory", None, None)  # type: ignore[arg-type]
    assert FaultRecord.from_exception(server_fault(refused)).fault == "content"


@pytest.mark.parametrize(
    "error", [TimeoutError("timed out"), ConnectionRefusedError(), ConnectionResetError(), urllib.error.URLError("x")]
)
def test_a_transport_error_is_service_unavailable(error: Exception) -> None:
    """A timeout, a refused connection and a reset are an infrastructure `ServiceUnavailable`, not a resource fault."""
    mapped = server_fault(error)
    assert isinstance(mapped, ServiceUnavailable)
    assert FaultRecord.from_exception(mapped).fault == "infrastructure"


def test_a_closed_port_is_service_unavailable_on_the_direct_call(
    registry: ServiceRegistry, starts: list[StandIn]
) -> None:
    """A server that is gone is an infrastructure `ServiceUnavailable`, as the consumer lease gives."""
    starts.append(StandIn(IDENTITY))
    url = str(starts[0].url)
    starts[0].stop()
    [record, name] = _fault_of(lambda: chat(url, IDENTITY.served_model, CHAT, sampling={}, timeout_s=REQUEST_S))
    assert (name, record.fault) == ("ServiceUnavailable", "infrastructure")


FAULT = OSError(errno.EIO, "storage fault")
"""The storage fault a teardown step meets: `EIO`, which is not one of the errors that mean a resource is gone."""


def _scaler_threads() -> list[str]:
    return [t.name for t in threading.enumerate() if t.name.startswith("senselab-")]


def _nothing_left(registry: ServiceRegistry, before: list[str]) -> None:
    """The service is unregistered, its lease free for another holder, and no `senselab-*` thread outlived it."""
    assert registry.lookup(IDENTITY) is None
    registry.reserve(IDENTITY).close()
    assert _scaler_threads() == before


@pytest.mark.parametrize(
    ("function", "prefix"),
    [("mkdir", "proof-"), ("unlink", "svc-")],
    ids=["proof", "withdraw"],
)
def test_a_fault_while_withdrawing_the_entry_still_releases_the_lease(
    registry: ServiceRegistry, monkeypatch: pytest.MonkeyPatch, function: str, prefix: str
) -> None:
    """`close` meets a storage fault at the lease proof or at the entry's removal: the lease is still released."""
    before = _scaler_threads()
    service = registry.register(IDENTITY, "http://127.0.0.1:1")
    with monkeypatch.context() as patch:
        refused = refuse_call(patch, function, prefix, FAULT)
        with pytest.raises(OSError, match="storage fault"):
            service.close()
    assert refused
    _nothing_left(registry, before)
    service.close()


def test_every_teardown_step_runs_and_the_first_error_is_raised(registry: ServiceRegistry) -> None:
    """Every step runs, the first error is raised, and a later error is attached to it, not lost."""
    ran: list[str] = []

    def failing(name: str) -> Callable[[], None]:
        def step() -> None:
            ran.append(name)
            raise OSError(errno.EIO, name)

        return step

    with pytest.raises(OSError, match="first") as raised:
        run_all(failing("first"), lambda: ran.append("middle"), failing("last"))
    assert ran == ["first", "middle", "last"]
    assert any("last" in note for note in raised.value.__notes__)


def test_an_interrupt_in_one_teardown_step_still_runs_the_rest() -> None:
    """A `KeyboardInterrupt` in the first step does not skip the later ones, and it is the error raised."""
    ran: list[str] = []

    def interrupted() -> None:
        ran.append("first")
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt) as raised:
        run_all(interrupted, lambda: ran.append("second"), lambda: 1 / 0)
    assert ran == ["first", "second"] and any("ZeroDivisionError" in n for n in raised.value.__notes__)


def test_a_fault_releasing_the_lease_is_raised_after_the_renewer_ended(
    registry: ServiceRegistry, monkeypatch: pytest.MonkeyPatch
) -> None:
    """EIO at the release of the registration lease: `close` raises, and the lease's renewer thread has ended."""
    service = registry.register(IDENTITY, "http://127.0.0.1:1")
    with monkeypatch.context() as patch:
        refused = refuse_call(patch, "rename", "gone-", FAULT)
        with pytest.raises(OSError, match="storage fault"):
            service.close()
    assert refused
    assert [t.name for t in threading.enumerate() if t.name.startswith("lease-renewer")] == []


class Recording(StandIn):
    """A stand-in server that records whether its entry was still registered when `stop` ran, and may fail."""

    def __init__(self, identity: Attestation, registry: ServiceRegistry, calls: list[str], fails: bool = False) -> None:
        """Listen as `identity`; `calls` records each stop, which raises when `fails`."""
        super().__init__(identity)
        self._registry, self._calls, self._fails = registry, calls, fails

    def stop(self) -> None:
        """Record the stop and whether the entry still stood; stop listening, then raise when it is to fail."""
        self._calls.append(f"stop:{self.label}:registered={self._registry.lookup(self.identity) is not None}")
        super().stop()
        if self._fails:
            raise FAULT


class _BrokenClient(WorkerClient):
    """A pooled worker process whose `close` raises a storage fault."""

    def __init__(self) -> None:
        """Hold no process."""
        self.closed = False

    def close(self) -> None:
        """Record the call and raise."""
        self.closed = True
        raise FAULT


STARTED: list["StandIn"] = []
"""The stand-in servers `_start` started, across the providers of one test."""


def _start(backend: BackendSpec, env: EnvHandle, model: ResolvedModel, profile: ExecutionProfile) -> "StandIn":  # noqa: ARG001 - a ServerStart
    STARTED.append(StandIn(IDENTITY))
    return STARTED[-1]


@pytest.fixture
def hosted(tmp_path: Path) -> Iterator[tuple[ExecutionProfile, ServiceRegistry]]:
    """The profile with its server root under `tmp_path`, and the registry the providers on device `0` use."""
    profile = PROFILE.model_copy(update={"env_roots": {**PROFILE.env_roots, "server": str(tmp_path / "servers")}})
    STARTED.clear()
    yield profile, service_registry(profile)
    for server in STARTED:
        server.stop()


def _loaded(profile: ExecutionProfile) -> Model:
    provider = SenselabWorker(_start)
    backend = BACKEND.model_copy(update={"worker": "backends.served_backend_fixture"})
    return provider.load(provider.add(Loader(backend, ENV, MODEL, "cpu", profile, IDENTITY)), None)


def test_pool_workers_of_one_identity_share_one_registered_server(
    hosted: tuple[ExecutionProfile, ServiceRegistry],
) -> None:
    """The first worker's load starts and registers the server; the next ones consume it under leases of their own."""
    profile, registry = hosted
    models = [_loaded(profile) for _ in range(3)]
    try:
        assert len(STARTED) == 1 and registry.lookup(IDENTITY) is not None
        assert registry.live_consumers(IDENTITY) == 2
        models[1].close()
        assert registry.live_consumers(IDENTITY) == 1
    finally:
        for model in reversed(models):
            model.close()
    assert registry.lookup(IDENTITY) is None and STARTED[0].url is None


def test_a_service_record_holds_no_host_name(tmp_path: Path, hosted: tuple[ExecutionProfile, ServiceRegistry]) -> None:
    """A providing and a consuming load on the host-keyed registry leave no host name, account or home path there."""
    profile, registry = hosted
    models = [_loaded(profile), _loaded(profile)]
    try:
        assert registry.root.relative_to(tmp_path / "servers").parts[1] == host_token()
        assert identity_leaks(tmp_path / "servers", tmp_path) == []
    finally:
        for model in reversed(models):
            model.close()
