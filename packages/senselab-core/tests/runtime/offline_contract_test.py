"""An offline execution profile reaches the network through no component: each refuses with `Offline` instead."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.contract

_CHILD = """
import hashlib, json, sys
from pathlib import Path
from senselab_core.kernel.declare import Env
from senselab_core.runtime.envs import HostCuda, Uv, ensure_env
from senselab_core.runtime.fetch import DataManifest, fetch
from senselab_core.runtime.network import Offline
from senselab_core.runtime.profiles import ExecutionProfile
from senselab_core.runtime.revision import Resolver

events = []
reaching = {"socket.connect", "socket.getaddrinfo", "socket.gethostbyname", "socket.gethostbyaddr", "socket.sendto"}
sys.addaudithook(lambda event, args: events.append(event) if event in reaching else None)
tmp, fake_uv, lock = (Path(a) for a in sys.argv[1:])
profile = ExecutionProfile.load("laptop", {"network": "offline"})
license = {"expression": "MIT", "commercial_use": "yes", "copyleft": "none", "attribution": True, "gated": False,
           "source": "https://example.org/license", "verified": {"date": "2026-10-01", "how": "manual"}}
data = b"x"
manifest = DataManifest.model_validate({"format": "data-manifest@2.0", "name": "demo", "version": "1",
    "license": license, "terms": "t", "files": [{"path": "a.wav", "url": "https://example.org/a.wav",
    "sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}]})
steps = {
    "resolve": lambda: Resolver(cache_dir=tmp / "hf", network=profile.network).lookup("org/model", "main"),
    "fetch": lambda: fetch(manifest, cache_root=tmp / "data", network=profile.network),
    "build": lambda: ensure_env(Env.isolated("demo"), lock, tmp / "envs", settings=profile.envs,
        uv=Uv((sys.executable, str(fake_uv))), host_cuda=HostCuda(None, "none"), network=profile.network),
}
refused = {}
for name, step in steps.items():
    try:
        step()
        refused[name] = "ran"
    except Offline:
        refused[name] = "Offline"
    except Exception as exc:
        refused[name] = type(exc).__name__
print(json.dumps({"refused": refused, "sockets": events}))
"""


def test_an_offline_profile_opens_no_socket_and_runs_no_install(tmp_path: Path) -> None:
    """Under an offline profile the resolver, fetch and an env build each raise `Offline`, with no socket or uv."""
    lock = tmp_path / "demo.txt"
    lock.write_text("# senselab subprocess-venv lock: demo\n# python: 3.12\n# torch: none\n# torch-index: none\n")
    log = tmp_path / "uv.log"
    env = {**os.environ, "FAKE_UV_LOG": str(log), "PYTHONPATH": os.pathsep.join(p for p in sys.path if p)}
    fake_uv = Path(__file__).with_name("fake_uv_fixture.py")
    run = subprocess.run(  # noqa: S603 - this interpreter
        [sys.executable, "-c", _CHILD, str(tmp_path), str(fake_uv), str(lock)],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
        check=True,
    )
    report = json.loads(run.stdout)
    assert report == {"refused": {"resolve": "Offline", "fetch": "Offline", "build": "Offline"}, "sockets": []}
    assert not log.exists()
