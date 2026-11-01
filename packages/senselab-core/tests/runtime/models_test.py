"""Tests for model resolution and the weights manifest."""

import hashlib
from pathlib import Path

import pytest
from senselab_core.kernel.identity import LogicalKey, ModelRef
from senselab_core.runtime.models import (
    WEIGHTS,
    WEIGHTS_FORMAT,
    ModelResolutionError,
    WeightsManifestError,
    WeightsMismatch,
    parse_weights,
    resolve,
    verify_model,
    verify_weights,
)
from senselab_core.runtime.revision import Resolver
from senselab_worker.digest import file_digest
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
def test_verify_weights_accepts_matching_and_rejects_other_bytes(tmp_path: Path) -> None:
    """A file matching its entry passes; any other bytes raise a typed error naming both digests."""
    good = tmp_path / "good.bin"
    good.write_bytes(b"detector")
    weights = parse_weights(f"format: weights@1.0\nweights:\n  t: {hashlib.sha256(b'detector').hexdigest()}\n")
    verify_weights("t", good, weights=weights)
    bad = tmp_path / "bad.bin"
    bad.write_bytes(b"other")
    with pytest.raises(WeightsMismatch) as exc:
        verify_weights("t", bad, weights=weights)
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
    assert one.identity == file_digest("file@1", tmp_path / "a.bin")


def _model_key(text: str) -> LogicalKey:
    model = resolve(ModelRef.parse(text)).key_identity()
    return LogicalKey(task="t", task_version="1", backend="b", backend_version="1", model=model, params={}, inputs={})


@pytest.mark.contract
def test_a_local_or_url_model_is_keyed_by_its_weights_not_where_they_live(tmp_path: Path) -> None:
    """The same weights at two paths share a key, different weights do not, and no path or URL enters a key."""
    for name, weights in [("home-alice-m", b"w"), ("scratch-bob-m", b"w"), ("other", b"v")]:
        (tmp_path / name / "cfg").mkdir(parents=True)
        (tmp_path / name / "w.bin").write_bytes(weights)
        (tmp_path / name / "cfg" / "c.json").write_bytes(b"{}")
    a, b, other = (_model_key(f"local:{tmp_path / n}") for n in ("home-alice-m", "scratch-bob-m", "other"))
    assert a.digest() == b.digest()
    assert a.digest() != other.digest()
    file_a, file_b = (_model_key(f"local:{tmp_path / n / 'w.bin'}") for n in ("home-alice-m", "scratch-bob-m"))
    (tmp_path / "renamed.pt").write_bytes(b"w")
    assert file_a.digest() == file_b.digest() == _model_key(f"local:{tmp_path / 'renamed.pt'}").digest()
    sha = file_digest("file@1", tmp_path / "renamed.pt")
    mirror_1, mirror_2 = (_model_key(f"url:https://{h}/u/alice/w.bin#sha256={sha}") for h in ("a.org", "b.net"))
    assert mirror_1.digest() == mirror_2.digest()
    for key in (a, b, file_a, file_b, mirror_1):
        dumped = repr(key.model_dump())
        assert str(tmp_path) not in dumped and "alice" not in dumped and "bob" not in dumped
        assert key.model is not None and key.model.id is None


@pytest.mark.contract
def test_the_same_weights_by_url_and_by_local_share_one_key(tmp_path: Path) -> None:
    """A weight file or directory reached by `url:` and by `local:` gives one key, which names neither source."""
    (tmp_path / "m" / "cfg").mkdir(parents=True)
    (tmp_path / "m" / "w.bin").write_bytes(b"weights")
    (tmp_path / "m" / "cfg" / "c.json").write_bytes(b"{}")
    file_sha = file_digest("file@1", tmp_path / "m" / "w.bin")
    dir_sha = weights_identity([tmp_path / "m" / "w.bin", tmp_path / "m" / "cfg" / "c.json"], root=tmp_path / "m")
    for path, sha in [(tmp_path / "m" / "w.bin", file_sha), (tmp_path / "m", dir_sha)]:
        local, url = _model_key(f"local:{path}"), _model_key(f"url:https://a.org/w#sha256={sha}")
        assert local.digest() == url.digest()
        assert local.model_dump()["model"] == {"id": None, "identity": sha}
    one_file = _model_key(f"url:https://a.org/w#sha256={file_sha}")
    assert one_file.digest() != _model_key(f"local:{tmp_path / 'm'}").digest()


@pytest.mark.logic
def test_resolve_local_errors_are_typed(tmp_path: Path) -> None:
    """A missing local path and an empty directory raise rather than resolving softly."""
    with pytest.raises(ModelResolutionError):
        resolve(ModelRef.parse(f"local:{tmp_path / 'missing'}"))
    with pytest.raises(ModelResolutionError):
        resolve(ModelRef.parse(f"local:{tmp_path}"))


@pytest.mark.logic
def test_url_ref_parses_the_declared_hash() -> None:
    """`url:<url>#sha256=<hex>` carries the hash and round-trips through str."""
    digest = "ab" * 32
    ref = ModelRef.parse(f"url:https://example.org/w.bin#sha256={digest}")
    assert (ref.source, ref.id, ref.ref) == ("url", "https://example.org/w.bin", digest)
    assert str(ref) == f"url:https://example.org/w.bin#sha256={digest}"
    with pytest.raises(ValueError):
        ModelRef.parse("url:https://example.org/w.bin#sha256=zz")


@pytest.mark.logic
def test_resolve_url_identity_is_the_declared_hash() -> None:
    """A url ref resolves to its declared hash with no download."""
    digest = "cd" * 32
    r = resolve(ModelRef.parse(f"url:https://example.org/w.bin#sha256={digest}"))
    assert r.pinned and r.identity == digest


@pytest.mark.logic
def test_resolve_url_without_hash_is_an_error() -> None:
    """A url ref with no declared hash is refused, never unresolved."""
    with pytest.raises(ValueError, match="sha256"):
        resolve(ModelRef.parse("url:https://example.org/w.bin"))


@pytest.mark.logic
def test_verify_model_checks_downloaded_bytes(tmp_path: Path) -> None:
    """A single file is checked by its sha256, a directory by weights_identity."""
    f = tmp_path / "w.bin"
    f.write_bytes(b"weights")
    model = resolve(ModelRef.parse(f"url:https://example.org/w.bin#sha256={hashlib.sha256(b'weights').hexdigest()}"))
    verify_model(model, f)
    f.write_bytes(b"other")
    with pytest.raises(WeightsMismatch):
        verify_model(model, f)
    d = tmp_path / "set"
    d.mkdir()
    (d / "a").write_bytes(b"a")
    ident = weights_identity([d / "a"], root=d)
    verify_model(resolve(ModelRef.parse(f"url:https://example.org/s#sha256={ident}")), d)


@pytest.mark.logic
@pytest.mark.parametrize(
    "text",
    [
        "format: weights@1.0\n",
        "format: weights@1.0\nweights: [a]\n",
        "format: weights@1.0\nweights:\n  bad: 12\n",
        "format: other\nweights: {}\n",
        "- not a mapping\n",
    ],
)
def test_malformed_manifest_raises_one_typed_error(text: str) -> None:
    """Every malformed manifest shape raises WeightsManifestError."""
    with pytest.raises(WeightsManifestError):
        parse_weights(text)


@pytest.mark.logic
def test_resolve_hf_binds_the_commit_its_resolver_names(tmp_path: Path) -> None:
    """An hf ref's identity is the commit its resolver returns, with the token passed through."""
    seen: dict[str, object] = {}

    def hub(repo: str, ref: str, token: str | None) -> str:
        seen.update(repo=repo, ref=ref, token=token)
        return "a" * 40

    r = resolve(
        ModelRef.parse("org/m@main"),
        token="t",
        resolver=Resolver(cache_dir=tmp_path / "hf-hub", hub=hub, cache_root=tmp_path / "senselab"),
    )
    assert r.identity == "a" * 40 and seen == {"repo": "org/m", "ref": "main", "token": "t"}


@pytest.mark.env
@pytest.mark.network
def test_resolve_hf_returns_commit() -> None:
    """A live hub ref resolves to a 40-hex commit."""
    r = resolve(ModelRef.parse("hf-internal-testing/tiny-random-bert@main"))
    assert r.pinned and len(r.identity) == 40
