"""Tests for the shared uv and lock primitives in ``senselab.utils.uv_tools``."""

from __future__ import annotations

import stat
from pathlib import Path

import pytest

from senselab.utils import uv_tools
from senselab.utils.uv_tools import find_uv, is_torch_family, normalize_name, requirements_sha256


def _executable(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\n")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def test_find_uv_prefers_the_env_override_then_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The override wins over PATH, PATH over the home install locations, and a bad override is refused."""
    pinned = _executable(tmp_path / "pinned" / "uv")
    on_path = _executable(tmp_path / "bin" / "uv")
    home = tmp_path / "home"
    in_home = _executable(home / ".local" / "bin" / "uv")
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    monkeypatch.setenv("PATH", str(on_path.parent))

    monkeypatch.setenv(uv_tools.UV_ENV, str(pinned))
    assert find_uv() == str(pinned)

    monkeypatch.delenv(uv_tools.UV_ENV)
    assert find_uv() == str(on_path)

    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    assert find_uv() == str(in_home)

    in_home.unlink()
    with pytest.raises(FileNotFoundError):
        find_uv()

    for bad in ("uv", str(tmp_path / "missing" / "uv")):
        monkeypatch.setenv(uv_tools.UV_ENV, bad)
        with pytest.raises(FileNotFoundError, match=uv_tools.UV_ENV):
            find_uv()
    not_executable = tmp_path / "plain" / "uv"
    not_executable.parent.mkdir()
    not_executable.write_text("")
    monkeypatch.setenv(uv_tools.UV_ENV, str(not_executable))
    with pytest.raises(FileNotFoundError, match=uv_tools.UV_ENV):
        find_uv()


def test_normalize_name_matches_pep503() -> None:
    """Runs of ``-``, ``_`` and ``.`` collapse to one ``-`` and the name is lowercased."""
    cases = {
        "Foo": "foo",
        "foo_bar": "foo-bar",
        "Foo.Bar": "foo-bar",
        "foo--_.bar": "foo-bar",
        "zope.interface": "zope-interface",
        "tensorflow_hub": "tensorflow-hub",
    }
    assert {name: normalize_name(name) for name in cases} == cases


def test_requirements_sha256_is_order_insensitive_and_header_sensitive() -> None:
    """Line order does not change the digest; every header field and every line does."""
    fields = {"python": "3.11", "max_cuda": [12, 8]}
    base = requirements_sha256(["a==1", "b>=2"], header_fields=fields)
    assert base == requirements_sha256(["b>=2", "a==1"], header_fields=fields)
    assert base == requirements_sha256(["a==1", "b>=2"], header_fields=dict(reversed(list(fields.items()))))
    assert base != requirements_sha256(["a==1", "b>=3"], header_fields=fields)
    assert base != requirements_sha256(["a==1", "b>=2"], header_fields={**fields, "python": "3.12"})
    assert base != requirements_sha256(["a==1", "b>=2"], header_fields={**fields, "max_cuda": None})
    assert base != requirements_sha256(["a==1", "b>=2"], header_fields={**fields, "extra": "x"})
    assert len(base) == 64 and int(base, 16) >= 0
    with pytest.raises(ValueError, match="requirements"):
        requirements_sha256(["a==1"], header_fields={"requirements": ["b"]})


def test_torch_family_names() -> None:
    """torch, torchvision, torchaudio and the CUDA-variant names are in the family; roots can be narrowed."""
    family = ["torch", "torchaudio", "torchvision", "Torch", "nvidia-cublas-cu12", "triton", "pytorch-triton"]
    others = ["numpy", "torchmetrics", "pytorch-lightning", "pynvml"]
    assert [name for name in family if not is_torch_family(name)] == []
    assert [name for name in others if is_torch_family(name)] == []
    assert is_torch_family("nvidia_cublas_cu12")
    assert not is_torch_family("torchvision", roots=frozenset({"torch", "torchaudio"}))
    assert is_torch_family("nvidia-nccl-cu12", roots=frozenset({"torch"}))
