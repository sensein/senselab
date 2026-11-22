"""One registered vLLM server of the catalogue's PII model consumed by two pool workers' loads (ORCD, H100)."""

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pytest
import senselab_core.runtime.fake_backend as fake
from senselab_core.kernel.declare import BackendSpec, Env, ServerConfig
from senselab_core.kernel.identity import ModelRef, ResolvedModel
from senselab_core.runtime.catalogue import Catalogue
from senselab_core.runtime.config import load_yaml
from senselab_core.runtime.envs import EnvHandle, ensure_env
from senselab_core.runtime.execution.providers import Loader, server_of
from senselab_core.runtime.execution.services import ConsumerLease, service_registry
from senselab_core.runtime.execution.workers import Server, expected_of, start_vllm
from senselab_core.runtime.keys import Call, Placement, call_key
from senselab_core.runtime.profiles import ExecutionProfile
from senselab_core.runtime.teardown import run_all
from senselab_core.runtime.vllm_server import StaleAttestation

pytestmark = [pytest.mark.limits, pytest.mark.gpu, pytest.mark.model, pytest.mark.network]

DATA = Path(__file__).parents[5] / "packages/senselab/src/senselab/data"
STARTUP_TIMEOUT_S = 1800.0
CHAT = [{"role": "user", "content": "Say ok."}]
SAMPLING = {"max_tokens": 8, "temperature": 0}


def _backend(settings: dict[str, object]) -> BackendSpec:
    config = ServerConfig(settings=settings)
    return fake.FAKE.model_copy(update={"name": "pii-vllm", "env": Env.server("vllm-server"), "server": config})


@pytest.mark.timeout(4 * STARTUP_TIMEOUT_S)
def test_two_plans_share_one_pii_server_and_a_new_config_is_another_service(
    tmp_path: Path, site_profile: Callable[..., ExecutionProfile]
) -> None:
    """Two consumers share one started server; restarted with a new config it attests anew, changing the call's key."""
    profile = site_profile()
    profile = profile.model_copy(update={"env_roots": {**profile.env_roots, "server": str(tmp_path / "servers")}})
    env = ensure_env(
        Env.server("vllm-server"),
        DATA / "env_locks/vllm-server.txt",
        profile.env_root("server"),
        settings=profile.envs,
        lease=profile.lease,
        device="cuda",
    )
    entry = Catalogue.load(DATA / "catalogue.yaml").default("pii_spans")
    model = ResolvedModel(ref=ModelRef(source="hf", id=entry.id, ref=entry.revision), identity=entry.revision)
    settings = load_yaml(DATA / "server_settings/pii_spans_vllm.yaml", "pii_spans_vllm.yaml")["settings"]
    registry = service_registry(profile)
    starts: list[int] = []

    def start(backend: BackendSpec, env: EnvHandle, model: ResolvedModel, profile: ExecutionProfile) -> Server:
        starts.append(1)
        return start_vllm(backend, env, model, profile)

    call = Call(task=fake.ECHO, params=fake.EchoParams(), inputs={"payload": "content:" + "a" * 64}, model=model)
    attested, keys, stops = [], [], []
    try:
        for seqs in (settings["max_num_seqs"], settings["max_num_seqs"] * 2):
            backend = _backend({**settings, "max_num_seqs": seqs})
            expected = expected_of(backend, env.lock, model)
            loader = Loader(backend, env, model, "cuda", profile, expected)
            provided = server_of(loader, start)
            try:
                consumed = [server_of(loader, start) for _ in range(2)]
                stops += [stop for _, own in consumed for stop in own]
                served = [server for server, _ in consumed]
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
                run_all(*provided[1])
    finally:
        run_all(*stops)
    assert len(starts) == 2 and attested[0][0].config_digest != attested[1][0].config_digest
    assert all(len(k) == 1 for k in keys) and keys[0] != keys[1]
