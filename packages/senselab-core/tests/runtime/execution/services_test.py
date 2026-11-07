"""Services: a registry entry under a renewed lease, consumer leases deciding idle exit, and per-reply attestation."""

import json
import os
import platform
import threading
import time
import uuid
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest
import senselab_core.runtime.fake_backend as fake
from senselab_core.kernel.declare import BackendSpec, Env, ServerConfig
from senselab_core.kernel.identity import ModelRef, ResolvedModel
from senselab_core.runtime.envs import EnvHandle
from senselab_core.runtime.execution.pool_worker import IDLE_POLLS, PoolWorker
from senselab_core.runtime.execution.queue import RUN_QUEUES
from senselab_core.runtime.execution.services import ConsumerLease, ServiceEntry, ServiceRegistry
from senselab_core.runtime.execution.workers import WorkerSet
from senselab_core.runtime.locks import LeaseTimeout
from senselab_core.runtime.profiles import ExecutionProfile
from senselab_core.runtime.store import ResultStore
from senselab_core.runtime.vllm_server import Attestation, StaleAttestation
from senselab_core.testing.interpreter import source_tree_python

pytestmark = pytest.mark.logic

SHA = "a" * 40
MODEL = ResolvedModel(ref=ModelRef.parse(f"org/m@{SHA}"), identity=SHA)
IDENTITY = Attestation(engine_version="0.31.0", served_model="org/m", model_identity=SHA, config_digest="c" * 64)
ENV = EnvHandle(source_tree_python(), platform.python_version(), "1" * 64, "2" * 64, None)
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
    """A stdlib HTTP server answering chat requests as `reply_model`; a `workers.Server` attesting `identity`."""

    def __init__(self, identity: Attestation) -> None:
        """Listen on a free loopback port."""
        self.label, self.identity, self.reply_model = identity.served_model, identity, identity.served_model
        stand_in = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: object) -> None:
                """Log nothing."""

            def do_POST(self) -> None:  # noqa: N802 - the http.server hook name
                self.rfile.read(int(self.headers["Content-Length"]))
                body = json.dumps({"model": stand_in.reply_model, "choices": [{"message": {"content": "ok"}}]})
                self.send_response(200)
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


def _provider(registry: ServiceRegistry, starts: list[StandIn]) -> WorkerSet:
    def start(backend: BackendSpec, env: EnvHandle, model: ResolvedModel, profile: ExecutionProfile) -> StandIn:  # noqa: ARG001 - a ServerStart
        starts.append(StandIn(IDENTITY))
        return starts[-1]

    return WorkerSet(1, start_server=start, registry=registry, provide=True)


def test_service_registered_under_renewed_lease(registry: ServiceRegistry) -> None:
    """An entry `{identity, endpoint}` stays live past its stale bound while its lease beats, and not once it stops."""
    service = registry.register(IDENTITY, "http://127.0.0.1:1")
    try:
        time.sleep(3 * PROFILE.lease.stale_after_s)
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
    past = time.time() - 10 * PROFILE.lease.stale_after_s
    for path in (token, token.parent):
        os.utime(path, (past, past))
    assert registry.lookup(IDENTITY) is None
    registry.register(IDENTITY, "http://127.0.0.1:3").close()


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


def test_service_exits_after_idle_consumers(tmp_path: Path, registry: ServiceRegistry, starts: list[StandIn]) -> None:
    """A `svc` worker with an empty queue stays while a consumer lease is live and exits `idle_exit_s` after it ends."""
    provider = _provider(registry, starts)
    provider.server(BACKEND, ENV, MODEL, PROFILE, IDENTITY)
    consumer = registry.consume(IDENTITY)
    held_s = 4 * PROFILE.services.idle_exit_s
    assert held_s > IDLE_POLLS * PROFILE.pool.poll_s
    timer = threading.Timer(held_s, consumer.stop)
    store = ResultStore(tmp_path / "store")
    worker = PoolWorker(tmp_path / "run" / RUN_QUEUES, ["svc:pii"], profile=PROFILE, store=store, workers=provider)
    began = time.monotonic()
    timer.start()
    try:
        assert worker.serve() == 0
        elapsed = time.monotonic() - began
    finally:
        timer.cancel()
        consumer.stop()
        provider.close()
    idle_s, poll_s = PROFILE.services.idle_exit_s, PROFILE.pool.poll_s
    assert held_s + idle_s - poll_s <= elapsed < held_s + idle_s + 2
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
