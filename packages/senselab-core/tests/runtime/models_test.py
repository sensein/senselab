"""Tests for model resolution and the weights manifest."""

import hashlib
from pathlib import Path

import pytest
from senselab_core.kernel.identity import ModelRef
from senselab_core.runtime import models
from senselab_core.runtime.models import WEIGHTS, WEIGHTS_FORMAT, WeightsMismatch, resolve, verify_weights
from senselab_worker.hashing import weights_identity


@pytest.mark.logic
def test_weights_manifest_holds_the_former_constants() -> None:
    """The five former literals are manifest entries."""
    assert WEIGHTS["yunet"] == "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
    assert WEIGHTS["maai-cpc"] == "d0f474de968c7243c20f43dee1626681618f2f150d3e638e82b79e22563b17f1"
    assert set(WEIGHTS) == {"mediapipe", "yunet", "sface", "s3fd", "maai-cpc"}
    assert WEIGHTS_FORMAT == "weights@1.0"
    with pytest.raises(TypeError):
        WEIGHTS["yunet"] = "x"  # type: ignore[index]


@pytest.mark.logic
def test_verify_weights_accepts_matching_and_rejects_other_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A file matching its entry passes; any other bytes raise a typed error naming both digests."""
    good = tmp_path / "good.bin"
    good.write_bytes(b"detector")
    monkeypatch.setattr(models, "WEIGHTS", {"t": hashlib.sha256(b"detector").hexdigest()})
    verify_weights("t", good)
    bad = tmp_path / "bad.bin"
    bad.write_bytes(b"other")
    with pytest.raises(WeightsMismatch) as exc:
        verify_weights("t", bad)
    assert exc.value.name == "t"
    assert exc.value.expected == hashlib.sha256(b"detector").hexdigest()
    assert exc.value.actual == hashlib.sha256(b"other").hexdigest()
    assert exc.value.path == bad


@pytest.mark.logic
def test_verify_weights_unknown_name(tmp_path: Path) -> None:
    """An unregistered name is a KeyError, not a pass."""
    f = tmp_path / "f"
    f.write_bytes(b"x")
    with pytest.raises(KeyError):
        verify_weights("no-such-weights", f)


@pytest.mark.logic
def test_resolve_local_uses_weights_identity(tmp_path: Path) -> None:
    """A local directory resolves to the weights identity of its files; a file to its own."""
    (tmp_path / "sub").mkdir()
    (tmp_path / "a.bin").write_bytes(b"a")
    (tmp_path / "sub" / "b.bin").write_bytes(b"b")
    r = resolve(ModelRef.parse(f"local:{tmp_path}"))
    assert r.pinned
    assert r.identity == weights_identity([tmp_path / "a.bin", tmp_path / "sub" / "b.bin"], root=tmp_path)
    one = resolve(ModelRef.parse(f"local:{tmp_path / 'a.bin'}"))
    assert one.identity == weights_identity([tmp_path / "a.bin"], root=tmp_path)


@pytest.mark.logic
def test_resolve_unpinnable_sources_say_why(tmp_path: Path) -> None:
    """A missing local path, an empty directory and a url are unresolved with a reason."""
    for text in (f"local:{tmp_path / 'missing'}", f"local:{tmp_path}", "url:https://example.org/w.bin"):
        r = resolve(ModelRef.parse(text))
        assert not r.pinned and r.unresolved_reason


@pytest.mark.logic
def test_resolve_hf_delegates_to_resolve_revision(monkeypatch: pytest.MonkeyPatch) -> None:
    """An hf ref's identity is whatever resolve_revision returns, with the token passed through."""
    seen: dict[str, object] = {}

    def fake(repo: str, ref: str, *, token: str | None = None) -> str:
        seen.update(repo=repo, ref=ref, token=token)
        return "a" * 40

    monkeypatch.setattr("senselab_core.runtime.models.resolve_revision", fake)
    r = resolve(ModelRef.parse("org/m@main"), token="t")
    assert r.identity == "a" * 40 and seen == {"repo": "org/m", "ref": "main", "token": "t"}


@pytest.mark.network
def test_resolve_hf_returns_commit() -> None:
    """A live hub ref resolves to a 40-hex commit."""
    r = resolve(ModelRef.parse("hf-internal-testing/tiny-random-bert@main"))
    assert r.pinned and len(r.identity) == 40
