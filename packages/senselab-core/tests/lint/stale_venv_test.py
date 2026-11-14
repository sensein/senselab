"""The stale-venv marker: a run in a venv synced from another uv.lock says so."""

import hashlib
import importlib.util
from pathlib import Path

import pytest

pytestmark = pytest.mark.logic

_spec = importlib.util.spec_from_file_location("packages_conftest", Path(__file__).resolve().parents[3] / "conftest.py")
assert _spec and _spec.loader
hooks = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hooks)


@pytest.fixture
def lock(tmp_path: Path) -> Path:
    """A uv.lock."""
    path = tmp_path / "uv.lock"
    path.write_text("version = 1\n")
    return path


def test_a_mismatched_digest_shows_the_marker(tmp_path: Path, lock: Path) -> None:
    """A venv whose recorded digest is another lock's prints STALE VENV."""
    (tmp_path / hooks.LOCK_DIGEST).write_text("0" * 64 + "\n")
    assert hooks.venv_status(tmp_path, lock).startswith("STALE VENV")


def test_a_matching_digest_does_not(tmp_path: Path, lock: Path) -> None:
    """A venv synced from this lock is not stale."""
    (tmp_path / hooks.LOCK_DIGEST).write_text(hashlib.sha256(lock.read_bytes()).hexdigest() + "\n")
    assert not hooks.venv_status(tmp_path, lock).startswith("STALE VENV")


def test_an_unrecorded_digest_is_reported_as_unknown(tmp_path: Path, lock: Path) -> None:
    """A venv with no recorded digest is reported as unrecorded, not as stale or current."""
    assert hooks.venv_status(tmp_path, lock).startswith("venv lock digest: unrecorded")
