"""One registered vLLM server of the catalogue's PII model consumed by two worker sets (ORCD, H100)."""

from dataclasses import replace
from pathlib import Path

import pytest
import senselab_core.runtime.fake_backend as fake
from senselab_core.kernel.declare import BackendSpec, Env, ServerConfig
from senselab_core.kernel.identity import ModelRef, ResolvedModel
from senselab_core.runtime.catalogue import Catalogue
from senselab_core.runtime.config import load_yaml
from senselab_core.runtime.envs import EnvHandle, ensure_env
from senselab_core.runtime.execution.services import ConsumerLease, ServiceRegistry
from senselab_core.runtime.execution.workers import Server, WorkerSet, expected_of, start_vllm
from senselab_core.runtime.keys import Call, Placement, call_key
from senselab_core.runtime.profiles import ExecutionProfile
from senselab_core.runtime.vllm_server import StaleAttestation

pytestmark = [pytest.mark.limits, pytest.mark.gpu, pytest.mark.model, pytest.mark.network]

DATA = Path(__file__).parents[5] / "packages/senselab/src/senselab/data"
STARTUP_TIMEOUT_S = 1800.0
CHAT = [{"role": "user", "content": "Say ok."}]
SAMPLING = {"max_tokens": 8, "temperature": 0}


def _backend(settings: dict[str, object]) -> BackendSpec:
    config = ServerConfig(settings=settings, startup_timeout_s=STARTUP_TIMEOUT_S)
    return fake.FAKE.model_copy(update={"name": "pii-vllm", "env": Env.server("vllm-server"), "server": config})


@pytest.mark.timeout(4 * STARTUP_TIMEOUT_S)
def test_two_plans_share_one_pii_server_and_a_new_config_is_another_service(tmp_path: Path) -> None:
    """Two consumers share one started server; restarted with a new config it attests anew, changing the call's key."""
    profile = ExecutionProfile.load("orcd")
    env = ensure_env(
        Env.server("vllm-server"),
        DATA / "env_locks/vllm-server.txt",
        profile.env_root("server"),
        settings=profile.envs,
        lease=profile.lease,
    )
    entry = Catalogue.load(DATA / "catalogue.yaml").default("pii_spans")
    model = ResolvedModel(ref=ModelRef(source="hf", id=entry.id, ref=entry.revision), identity=entry.revision)
    settings = load_yaml(DATA / "server_settings/pii_spans_vllm.yaml", "pii_spans_vllm.yaml")["settings"]
    registry = ServiceRegistry(tmp_path / "registry", lease=profile.lease)
    starts: list[int] = []

    def start(backend: BackendSpec, env: EnvHandle, model: ResolvedModel, profile: ExecutionProfile) -> Server:
        starts.append(1)
        return start_vllm(backend, env, model, profile)

    call = Call(task=fake.ECHO, params=fake.EchoParams(), inputs={"payload": "content:" + "a" * 64}, model=model)
    attested, keys, plans = [], [], [WorkerSet(1, registry=registry) for _ in range(4)]
    try:
        for round_plans, seqs in ((plans[:2], settings["max_num_seqs"]), (plans[2:], settings["max_num_seqs"] * 2)):
            backend = _backend({**settings, "max_num_seqs": seqs})
            expected = expected_of(backend, env.lock, model)
            provider = WorkerSet(1, start_server=start, registry=registry, provide=True)
            try:
                provider.server(backend, env, model, profile, expected)
                served = [p.server(backend, env, model, profile, expected) for p in round_plans]
                assert all(isinstance(s, ConsumerLease) for s in served) and registry.live_consumers(expected) == 2
                replies = [s.chat(CHAT, sampling=SAMPLING) for s in served]  # type: ignore[attr-defined]
                assert all(r.attestation == expected and r.body["choices"][0]["message"]["content"] for r in replies)
                attested.append((expected, served[0]))
                placement = Placement(backend=backend, backend_params=backend.params(), device="cuda")
                keys.append({call_key(replace(call, service=s.attestation), placement).digest() for s in served})
                if len(attested) == 2:
                    with pytest.raises(StaleAttestation):
                        attested[0][1].chat(CHAT, sampling=SAMPLING)  # type: ignore[attr-defined]
            finally:
                provider.close()
    finally:
        for workers in plans:
            workers.close()
    assert len(starts) == 2 and attested[0][0].config_digest != attested[1][0].config_digest
    assert all(len(k) == 1 for k in keys) and keys[0] != keys[1]
