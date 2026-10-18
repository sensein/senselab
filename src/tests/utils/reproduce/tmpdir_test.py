"""A short ``TMPDIR`` inside the sandbox: AF_UNIX sockets bind even under a long results root.

Every site is synthetic, under ``tmp_path``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from senselab.utils.reproduce import run, sandbox
from tests.utils.reproduce.run_test import World, _py

needs_userns = pytest.mark.skipif(not sandbox.userns_available(), reason="unprivileged user namespaces are unavailable")

SOCKET = """\
import json, os, socket, sys
path = os.path.join(os.environ["TMPDIR"], "probe.sock")
s = socket.socket(socket.AF_UNIX)
s.bind(path)
s.close()
info = {"tmpdir": os.environ["TMPDIR"], "cuda": os.environ["CUDA_CACHE_PATH"], "home": os.environ["HOME"]}
open(sys.argv[1] + "/env.json", "w").write(json.dumps(info))
open(sys.argv[1] + "/result.json", "w").write(json.dumps({"value": 2.0}))
"""

FD_PASSING = """\
import json, multiprocessing, os, sys
from multiprocessing import reduction

def child(handle, queue):
    fd = handle.detach()
    os.lseek(fd, 0, 0)
    queue.put(os.read(fd, 5).decode())

context = multiprocessing.get_context("fork")
path = os.path.join(os.environ["TMPDIR"], "data.txt")
open(path, "w").write("hello")
handle = reduction.DupFd(os.open(path, os.O_RDONLY))
queue = context.Queue()
worker = context.Process(target=child, args=(handle, queue))
worker.start()
got = queue.get(timeout=60)
worker.join(60)
assert got == "hello" and worker.exitcode == 0, (got, worker.exitcode)
open(sys.argv[1] + "/result.json", "w").write(json.dumps({"value": 2.0}))
"""


@pytest.fixture
def world(tmp_path: Path) -> World:
    """A site whose results root is longer than 150 characters, so its tmp paths cannot hold a socket."""
    deep = tmp_path / ("r" * 60) / ("s" * 60)
    deep.mkdir(parents=True)
    site = World(deep)
    assert len(str(site.profile().results_root)) > 150
    return site


@needs_userns
def test_a_socket_binds_under_tmpdir_with_a_long_results_root(world: World) -> None:
    """The step's ``TMPDIR`` (and the paths under it) is short inside the sandbox, so ``bind`` succeeds."""
    manifest = world.manifest(_py(world, SOCKET))
    world.prefetch(manifest)
    outcome = world.run(manifest)["prep"]
    assert outcome.ok, outcome.codes
    assert outcome.out is not None
    seen = json.loads((outcome.out / run.WORK / "env.json").read_text())
    assert seen["tmpdir"] in ("/tmp", sandbox.SHORT_TMP)
    assert seen["cuda"] == seen["tmpdir"] + "/cuda-cache" and seen["home"] == seen["tmpdir"] + "/home"


@needs_userns
def test_fd_passing_through_the_resource_sharer_completes(world: World) -> None:
    """A ``DupFd`` handed to a forked child goes through the resource sharer's AF_UNIX listener and arrives."""
    manifest = world.manifest(_py(world, FD_PASSING))
    world.prefetch(manifest)
    outcome = world.run(manifest)["prep"]
    assert outcome.ok, outcome.codes
