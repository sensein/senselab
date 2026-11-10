"""The `vllm-server` env serving the catalogue's PII model on a GPU, attested per config (ORCD, H100)."""

from pathlib import Path

import pytest
from senselab_core.kernel.declare import Env
from senselab_core.kernel.identity import ModelRef, ResolvedModel
from senselab_core.runtime.catalogue import Catalogue
from senselab_core.runtime.envs import ensure_env
from senselab_core.runtime.profiles import ExecutionProfile
from senselab_core.runtime.vllm_server import VllmServer

CLUSTER = (
    next(p for p in Path(__file__).resolve().parents if p.name == "packages")
    / "senselab-core/tests/data/profiles/cluster.yaml"
)

pytestmark = [pytest.mark.limits, pytest.mark.gpu, pytest.mark.model, pytest.mark.network]

DATA = Path(__file__).parents[4] / "packages/senselab/src/senselab/data"
STARTUP_TIMEOUT_S = 1800.0
SETTINGS = {
    "max_model_len": 4096,
    "max_num_seqs": 2,
    "gpu_memory_utilization": 0.9,
    "kv_cache_dtype": "auto",
    "generation_config": "vllm",
    "limit_mm_per_prompt": {"image": 0},
    "enable_prefix_caching": False,
    "flashinfer_sampler": False,
}


@pytest.mark.timeout(3 * STARTUP_TIMEOUT_S)
def test_vllm_server_attests_the_pii_model_and_a_restart_with_new_config_changes_it() -> None:
    """The server loads the catalogue commit, every reply carries its attestation, and a new config re-attests."""
    profile = ExecutionProfile.load(CLUSTER)
    env = ensure_env(
        Env.server("vllm-server"),
        DATA / "env_locks/vllm-server.txt",
        profile.env_root("server"),
        settings=profile.envs,
        lease=profile.lease,
    )
    entry = Catalogue.load(DATA / "catalogue.yaml").default("pii_spans")
    model = ResolvedModel(ref=ModelRef(source="hf", id=entry.id, ref=entry.revision), identity=entry.revision)
    digests = []
    for seqs in (2, 4):
        settings = {**SETTINGS, "max_num_seqs": seqs}
        with VllmServer(
            env,
            model,
            settings,
            start_lock_dir=profile.env_root("server") / "locks",
            startup_timeout_s=STARTUP_TIMEOUT_S,
            start_margin_s=profile.envs.server_start_margin_s,
            lease=profile.lease,
            log_dir=profile.env_root("server") / "logs",
        ) as server:
            reply = server.chat([{"role": "user", "content": "Say ok."}], sampling={"max_tokens": 8, "temperature": 0})
            assert reply.attestation == server.attestation
            assert (reply.attestation.engine_version, reply.attestation.model_identity) == ("0.31.0", entry.revision)
            assert reply.body["choices"][0]["message"]["content"]
            digests.append(reply.attestation.config_digest)
    assert digests[0] != digests[1]
