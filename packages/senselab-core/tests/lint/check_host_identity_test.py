"""The host-identity check: no 2.x source or benchmark reads the host name or the account name."""

import importlib.util
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.logic

_SCRIPT = Path(__file__).resolve().parents[4] / "scripts" / "check_host_identity.py"
_spec = importlib.util.spec_from_file_location("check_host_identity", _SCRIPT)
assert _spec and _spec.loader
hi = importlib.util.module_from_spec(_spec)
sys.modules["check_host_identity"] = hi
_spec.loader.exec_module(hi)


@pytest.mark.parametrize(
    ("line", "found"),
    [
        ("socket.gethostname()", "socket.gethostname"),
        ("platform.node()", "platform.node"),
        ("getpass.getuser()", "getpass.getuser"),
        ("os.uname().nodename", "uname().nodename"),
        ("uname().nodename", "uname().nodename"),
        ("socket.getfqdn()", "socket.getfqdn"),
        ("platform.uname().node", "platform.uname().node"),
        ("os.getlogin()", "os.getlogin"),
        ("pwd.getpwuid(0).pw_name", "pwd.getpwuid"),
        ("s.gethostname()\nimport socket as s", "socket.gethostname"),
        ("gethostname()\nfrom socket import gethostname", "socket.gethostname"),
        ("me()\nfrom getpass import getuser as me", "getpass.getuser"),
        ("uname().node\nfrom platform import uname", "platform.uname().node"),
    ],
)
def test_every_way_to_read_the_name_is_found(tmp_path: Path, line: str, found: str) -> None:
    """A read of the host or account name is reported with its line."""
    source = tmp_path / "m.py"
    source.write_text(f"x = {line}\n")
    assert hi.findings([source], root=tmp_path) == [f"m.py:1: {found}"]


def test_a_host_token_and_a_platform_read_pass(tmp_path: Path) -> None:
    """Reading the platform or the token is not reading a name."""
    source = tmp_path / "m.py"
    source.write_text("from os import path\nx = platform.system() + host_token() + os.getuid() + path.node()\n")
    assert hi.findings([source], root=tmp_path) == []


def test_the_tree_reads_no_name_outside_the_allowlist() -> None:
    """The 2.x packages' source and the benchmarks read no host or account name, bar `ALLOWED`."""
    assert hi.findings(hi.default_paths()) == []
