"""Services: a registry entry under a renewed lease, consumer leases deciding idle exit, and per-reply attestation."""

import json
import platform
import threading
import urllib.error
import uuid
from collections.abc import Callable, Iterator
from dataclasses import replace
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
from senselab_core.runtime.execution.pool_worker import IDLE_POLLS, PoolWorker
from senselab_core.runtime.execution.queue import RUN_QUEUES
from senselab_core.runtime.execution.services import ConsumerLease, ServiceEntry, ServiceRegistry, service_registry
from senselab_core.runtime.execution.workers import WorkerSet, expected_of
from senselab_core.runtime.hardware import host_token
from senselab_core.runtime.keys import Call, Placement, call_key
from senselab_core.runtime.locks import LeaseTimeout
from senselab_core.runtime.profiles import ExecutionProfile
from senselab_core.runtime.store import ResultStore
from senselab_core.runtime.vllm_server import ServerRejected, ServiceUnavailable, StaleAttestation, chat, server_fault
from senselab_core.testing.interpreter import source_tree_python

pytestmark = pytest.mark.logic

SHA = "a" * 40
MODEL = ResolvedModel(ref=ModelRef.parse(f"org/m@{SHA}"), identity=SHA)
IDENTITY = Attestation(engine_version="0.31.0", served_model="org/m", model_identity=SHA, config_digest="c" * 64)
ENV = EnvHandle(source_tree_python(), platform.python_version(), "1" * 64, "2" * 64, MIRROR_INDEX)
BACKEND = fake.FAKE.model_copy(
    update={"name": "served", "env": Env.server("s"), "server": ServerConfig(settings={}, startup_timeout_s=1.0)}
)
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


def _provider(registry: ServiceRegistry, starts: list[StandIn], identity: Attestation = IDENTITY) -> WorkerSet:
    def start(backend: BackendSpec, env: EnvHandle, model: ResolvedModel, profile: ExecutionProfile) -> StandIn:  # noqa: ARG001 - a ServerStart
        starts.append(StandIn(identity))
        return starts[-1]

    return WorkerSet(1, start_server=start, registry=registry, provide=True)


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


def test_a_service_record_holds_no_host_name(tmp_path: Path, starts: list[StandIn]) -> None:
    """A provider and a consumer on the host-keyed registry leave no host name, account or home path under its root."""
    profile = PROFILE.model_copy(update={"env_roots": {**PROFILE.env_roots, "server": str(tmp_path / "servers")}})
    registry = service_registry(profile, {"CUDA_VISIBLE_DEVICES": "0"})
    provider, consumer = _provider(registry, starts), WorkerSet(1, registry=registry)
    try:
        provider.server(BACKEND, ENV, MODEL, PROFILE, IDENTITY)
        consumer.server(BACKEND, ENV, MODEL, PROFILE, IDENTITY)
        assert registry.root.relative_to(tmp_path / "servers").parts[1] == host_token()
        assert identity_leaks(tmp_path / "servers", tmp_path) == []
    finally:
        consumer.close()
        provider.close()


def test_two_consumers_one_server(registry: ServiceRegistry, starts: list[StandIn]) -> None:
    """Two consumers of one identity share the one registered server, each under its own lease."""
    provider = _provider(registry, starts)
    consumers = [WorkerSet(1, registry=registry) for _ in range(2)]
    try:
        provider.server(BACKEND, ENV, MODEL, PROFILE, IDENTITY)
        served = [c.server(BACKEND, ENV, MODEL, PROFILE, IDENTITY) for c in consumers]
        assert all(isinstance(s, ConsumerLease) for s in served) and len(starts) == 1
        assert {s.url for s in served} == {starts[0].url}
        assert all(s.chat(CHAT, sampling={}).attestation == IDENTITY for s in served)  # type: ignore[attr-defined]
        assert registry.live_consumers(IDENTITY) == 2
        consumers[0].close()
        assert registry.live_consumers(IDENTITY) == 1
    finally:
        for workers in (*consumers, provider):
            workers.close()
    assert registry.lookup(IDENTITY) is None and starts[0].url is None


def test_workers_of_one_identity_start_one_server(registry: ServiceRegistry, starts: list[StandIn]) -> None:
    """Providing and consuming sets of one identity start one server: every set after the first consumes it."""
    sets = [_provider(registry, starts) for _ in range(3)] + [WorkerSet(1, registry=registry) for _ in range(2)]
    try:
        served = [workers.server(BACKEND, ENV, MODEL, PROFILE, IDENTITY) for workers in sets]
        assert len(starts) == 1 and served[0] is starts[0]
        assert all(isinstance(s, ConsumerLease) and s.url == starts[0].url for s in served[1:])
        assert registry.live_consumers(IDENTITY) == len(sets) - 1
    finally:
        for workers in reversed(sets):
            workers.close()


def test_each_device_set_registers_its_own_server(tmp_path: Path, starts: list[StandIn]) -> None:
    """Providers on two devices start two servers, two on one device one; a consumer binds to its card's."""
    profile = PROFILE.model_copy(update={"env_roots": {**PROFILE.env_roots, "server": str(tmp_path / "servers")}})
    on = {d: service_registry(profile, {"CUDA_VISIBLE_DEVICES": d}) for d in ("0", "1")}
    sets = [_provider(on["0"], starts), _provider(on["1"], starts), _provider(on["0"], starts)]
    consumer = WorkerSet(1, registry=on["1"])
    try:
        served = [w.server(BACKEND, ENV, MODEL, PROFILE, IDENTITY) for w in (*sets, consumer)]
        assert len(starts) == 2 and on["0"].root != on["1"].root
        assert served[2].url == starts[0].url and served[3].url == starts[1].url
    finally:
        for workers in (consumer, *reversed(sets)):
            workers.close()


def test_service_exits_after_idle_consumers(
    tmp_path: Path, starts: list[StandIn], lease_clock: Callable[..., ManualClock]
) -> None:
    """A `svc` worker with an empty queue stays while a consumer lease is live and exits `idle_exit_s` after it ends."""
    clock = lease_clock()
    registry = ServiceRegistry(tmp_path / "registry", lease=PROFILE.lease, clock=clock)
    provider = _provider(registry, starts)
    provider.server(BACKEND, ENV, MODEL, PROFILE, IDENTITY)
    consumer = registry.consume(IDENTITY)
    idle_s, poll_s = PROFILE.services.idle_exit_s, PROFILE.pool.poll_s
    held_s = 4 * idle_s
    assert held_s > IDLE_POLLS * poll_s
    store = ResultStore(tmp_path / "store")
    worker = PoolWorker(
        tmp_path / "run" / RUN_QUEUES, ["svc:pii"], profile=PROFILE, store=store, workers=provider, clock=clock
    )
    exits: list[int] = []
    thread = threading.Thread(target=lambda: exits.append(worker.serve()), daemon=True)
    thread.start()
    try:
        for _ in range(round(held_s / poll_s)):
            clock.settle(parked=3)
            clock.advance(poll_s)
        clock.settle(parked=3)
        assert thread.is_alive()
        consumer.stop()
        left = clock.now()
        clock.drive(thread, poll_s)
        assert exits == [0]
        assert idle_s - poll_s <= clock.now() - left <= idle_s + 2 * poll_s
    finally:
        worker.halt()
        thread.join(clock.hang_s)
        consumer.stop()
        provider.close()
    assert worker.served == [] and registry.lookup(IDENTITY) is None


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


def test_the_key_takes_the_attestation_of_the_service_entry(
    tmp_path: Path, registry: ServiceRegistry, starts: list[StandIn]
) -> None:
    """A call's key takes the entry's attestation: two configs give two keys, a restart of one config the same key."""
    lock = tmp_path / "vllm-server.txt"
    lock.write_text("vllm==0.31.0\n")
    call = Call(task=fake.ECHO, params=fake.EchoParams(), inputs={"payload": "content:" + "a" * 64}, model=MODEL)
    keys = []
    for seqs in (4, 8, 4):
        backend = BACKEND.model_copy(
            update={"server": ServerConfig(settings={**SETTINGS, "max_num_seqs": seqs}, startup_timeout_s=1.0)}
        )
        expected = expected_of(backend, lock, MODEL)
        provider, consumer = _provider(registry, starts, expected), WorkerSet(1, registry=registry)
        try:
            provider.server(backend, ENV, MODEL, PROFILE, expected)
            served = consumer.server(backend, ENV, MODEL, PROFILE, expected)
            assert isinstance(served, ConsumerLease) and served.attestation == expected
            placement = Placement(backend=backend, backend_params=backend.params(), device="cpu")
            keys.append(call_key(replace(call, service=served.attestation), placement).digest())
            with pytest.raises(ValueError, match="is a server backend"):
                call_key(call, placement)
        finally:
            consumer.close()
            provider.close()
    assert keys[0] != keys[1] and keys[0] == keys[2] and len(starts) == 3


def test_a_dead_consumer_counts_until_its_lease_is_stale(registry: ServiceRegistry) -> None:
    """A consumer that died without releasing counts as live until its lease is stale, then is removed."""
    consumers = registry.root / f"{registry.name(IDENTITY)}.consumers"
    token = consumers / f"{uuid.uuid4().hex}.lease" / uuid.uuid4().hex
    token.mkdir(parents=True)
    assert registry.live_consumers(IDENTITY) == 1
    age(consumers, 10 * PROFILE.lease.stale_after_s)
    assert registry.live_consumers(IDENTITY) == 0 and list(consumers.iterdir()) == []


def test_two_providers_racing_start_one_server(registry: ServiceRegistry, tier_bound: float) -> None:
    """Two providing sets of one identity: one reserves, starts and publishes; the other starts nothing."""
    starts: list[StandIn] = []
    barrier, refused = threading.Barrier(2), threading.Event()

    def start(backend: BackendSpec, env: EnvHandle, model: ResolvedModel, profile: ExecutionProfile) -> StandIn:  # noqa: ARG001 - a ServerStart
        assert refused.wait(tier_bound)
        starts.append(StandIn(IDENTITY))
        return starts[-1]

    providers = [WorkerSet(1, start_server=start, registry=registry, provide=True) for _ in range(2)]
    outcomes: list[object] = []

    def provide(workers: WorkerSet) -> None:
        barrier.wait()
        try:
            outcomes.append(workers.server(BACKEND, ENV, MODEL, PROFILE, IDENTITY))
        except LeaseTimeout as busy:
            outcomes.append(busy)
            refused.set()

    threads = [threading.Thread(target=provide, args=(p,)) for p in providers]
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert len(starts) == 1 and sum(isinstance(o, LeaseTimeout) for o in outcomes) == 1
        assert registry.lookup(IDENTITY) == ServiceEntry(identity=IDENTITY, endpoint=str(starts[0].url))
    finally:
        for workers in providers:
            workers.close()
        for server in starts:
            server.stop()


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
        direct = _fault_of(lambda: chat(str(starts[0].url), IDENTITY.served_model, CHAT, sampling={}))
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
    [record, name] = _fault_of(lambda: chat(url, IDENTITY.served_model, CHAT, sampling={}))
    assert (name, record.fault) == ("ServiceUnavailable", "infrastructure")
