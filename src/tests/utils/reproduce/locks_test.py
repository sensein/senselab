"""Tests for the lock header, the integrity check and freeze parsing."""

from __future__ import annotations

import datetime
import shutil
from pathlib import Path
from typing import Any

import pytest

from senselab.utils.reproduce.locks import (
    HEADER_MARKER,
    ReproduceLockError,
    check_integrity,
    file_sha256,
    parse_freeze,
    parse_header,
    requirements_sha256,
    write_header,
)
from senselab.utils.reproduce.spec import EnvSpec

DATA = Path(__file__).parent / "data"
ENV_DIR = "envs/a"
EXCLUDE_NEWER = datetime.date(2026, 7, 1)
# numpy 2.4.5, cp311 manylinux_2_27/2_28 x86_64 wheel, from the PyPI JSON API.
NUMPY_245_SHA = "07ce7e74da92d7c71b5df157b9758bcdd53d7fea10602154de3afd2b3ddc34dd"
REQUIREMENTS = ["certifi", "idna", "numpy", "packaging", "six"]


def _header_sha(python: str = "3.11") -> str:
    return requirements_sha256(REQUIREMENTS, python, "12.8", EXCLUDE_NEWER, None, None, ["opencv-python"])


def _copy(base: Path, fixture: str, name: str) -> str:
    rel = f"{ENV_DIR}/{name}"
    shutil.copyfile(DATA / fixture, base / rel)
    return rel


def _write(base: Path, rel: str, text: str) -> None:
    (base / rel).write_text(text)


def _lock(base: Path, rel: str) -> dict[str, str]:
    return {"lock": rel, "lock_sha256": file_sha256(base / rel)}


def _build(base: Path, source: str = "requirements.in", header_python: str = "3.11") -> dict[str, Any]:
    """Lay out a valid env under ``base`` and return its manifest ``env`` mapping."""
    (base / ENV_DIR).mkdir(parents=True)
    torch = _copy(base, "torch_small.lock", "torch.lock")
    body = f"{ENV_DIR}/body.lock"
    body_text = (DATA / "body_small.lock").read_text()
    if source == "requirements.in":
        body_text = write_header(body_text, {"requirements-sha256": _header_sha(header_python), "python": "3.11"})
    _write(base, body, body_text)
    test = _copy(base, "test_small.lock", "test.lock")
    build = _copy(base, "body_small.lock", "build.lock")
    env: dict[str, Any] = {
        "python": "3.11",
        "python_exact": "3.11.9",
        "torch": {"index_url": "https://download.pytorch.org/whl/cu128", **_lock(base, torch)},
        "body": {**_lock(base, body), "source": source},
        "sdist_allow": [
            {
                "name": "python-speech-features",
                "url": "https://files.example/python_speech_features-0.6.tar.gz",
                "sha256": "a0aebf746464bc929dc3162cb369d7ff967c398c5120ddf5fb40a65f01b92b11",
                "build_env": _lock(base, build),
            }
        ],
        "exclude_newer": EXCLUDE_NEWER.isoformat(),
        "extra_no_emit": ["opencv-python"],
        "test": _lock(base, test),
        "max_cuda": "12.8",
    }
    if source == "requirements.in":
        req = _copy(base, "requirements_small.in", "requirements.in")
        first = _copy(base, "first_resolve_small.txt", "first_resolve.txt")
        env["body"].update(requirements_in=req, requirements_in_sha256=file_sha256(base / req))
        env["first_resolve"] = {"file": first, "sha256": file_sha256(base / first)}
    else:
        env["body"]["export_cmd"] = ["uv", "export", "--frozen", "--no-emit-project", "--no-dev"]
    return env


def _rehash(base: Path, env: dict[str, Any]) -> EnvSpec:
    """Refresh every manifest sha256 from the files, as a tamperer who also edits the manifest would."""
    for ref in (env["torch"], env["body"], env["test"], *(s["build_env"] for s in env["sdist_allow"])):
        ref["lock_sha256"] = file_sha256(base / ref["lock"])
    if "first_resolve" in env:
        env["first_resolve"]["sha256"] = file_sha256(base / env["first_resolve"]["file"])
    return EnvSpec.model_validate(env)


def _uv_lock(base: Path) -> Path:
    path = base / "repo" / "uv.lock"
    path.parent.mkdir()
    shutil.copyfile(DATA / "uv_small.lock", path)
    return path


def _replace(base: Path, rel: str, old: str, new: str) -> None:
    text = (base / rel).read_text()
    assert old in text
    _write(base, rel, text.replace(old, new))


def _numpy_block(text: str) -> str:
    lines = text.splitlines(keepends=True)
    start = next(i for i, line in enumerate(lines) if line.startswith("numpy=="))
    end = next(i for i in range(start + 1, len(lines)) if not lines[i].startswith(" "))
    return "".join(lines[start:end])


def test_valid_requirements_in_env_passes(tmp_path: Path) -> None:
    """Files match the manifest, the header recomputes, overlays agree and torch matches the first resolve."""
    check_integrity(EnvSpec.model_validate(_build(tmp_path)), tmp_path)


def test_valid_uv_export_env_passes(tmp_path: Path) -> None:
    """A uv-export env checks its torch lock against the repository ``uv.lock`` entries."""
    check_integrity(EnvSpec.model_validate(_build(tmp_path, "uv-export")), tmp_path, uv_lock=_uv_lock(tmp_path))


def test_numpy_repinned_to_a_real_release_is_refused(tmp_path: Path) -> None:
    """A body repinned to numpy 2.4.5 with its real hash passes ``--require-hashes``; the manifest sha catches it."""
    env = EnvSpec.model_validate(_build(tmp_path))
    body = tmp_path / env.body.lock
    text = body.read_text()
    repinned = f"numpy==2.4.5 \\\n    --hash=sha256:{NUMPY_245_SHA}\n"
    body.write_text(text.replace(_numpy_block(text), repinned))
    assert "numpy==2.4.6" not in body.read_text()
    with pytest.raises(ReproduceLockError, match=r"body\.lock.*sha256"):
        check_integrity(env, tmp_path)


@pytest.mark.parametrize("field", ["torch", "test", "build_env", "first_resolve", "requirements_in"])
def test_any_hashed_file_changed_is_refused(tmp_path: Path, field: str) -> None:
    """Every lock, overlay, build env, first resolve and sidecar is checked against its manifest sha256."""
    raw = _build(tmp_path)
    rel = {
        "torch": raw["torch"]["lock"],
        "test": raw["test"]["lock"],
        "build_env": raw["sdist_allow"][0]["build_env"]["lock"],
        "first_resolve": raw["first_resolve"]["file"],
        "requirements_in": raw["body"]["requirements_in"],
    }[field]
    with (tmp_path / rel).open("a") as handle:
        handle.write("\n")
    with pytest.raises(ReproduceLockError, match="sha256"):
        check_integrity(EnvSpec.model_validate(raw), tmp_path)


def test_header_recompute_mismatch_is_refused(tmp_path: Path) -> None:
    """A header compiled for other inputs fails even when every file sha256 matches the manifest."""
    raw = _build(tmp_path, header_python="3.12")
    with pytest.raises(ReproduceLockError, match="requirements-sha256"):
        check_integrity(EnvSpec.model_validate(raw), tmp_path)


def test_body_without_header_is_refused(tmp_path: Path) -> None:
    """A requirements.in body must carry the reproduce header."""
    raw = _build(tmp_path)
    _write(tmp_path, raw["body"]["lock"], (DATA / "body_small.lock").read_text())
    with pytest.raises(ReproduceLockError, match=HEADER_MARKER):
        check_integrity(_rehash(tmp_path, raw), tmp_path)


def test_overlay_pin_mismatch_is_refused(tmp_path: Path) -> None:
    """A name pinned in an overlay and a base lock must have the identical pin and hashes."""
    raw = _build(tmp_path)
    _replace(tmp_path, raw["test"]["lock"], "packaging==24.2", "packaging==24.1")
    with pytest.raises(ReproduceLockError, match="packaging"):
        check_integrity(_rehash(tmp_path, raw), tmp_path)


def test_overlay_hash_mismatch_is_refused(tmp_path: Path) -> None:
    """The same version with a different hash set is a mismatch."""
    raw = _build(tmp_path)
    _replace(
        tmp_path,
        raw["test"]["lock"],
        " \\\n    --hash=sha256:c228a6dc5e932d346bc5739379109d49e8853dd8223571c7c5b55260edc0b97f",
        "",
    )
    with pytest.raises(ReproduceLockError, match="packaging"):
        check_integrity(_rehash(tmp_path, raw), tmp_path)


def test_torch_mismatch_against_first_resolve_is_refused(tmp_path: Path) -> None:
    """The torch-family pins must equal those the body was resolved against."""
    raw = _build(tmp_path)
    _replace(tmp_path, raw["first_resolve"]["file"], "triton==3.4.0", "triton==3.3.1")
    with pytest.raises(ReproduceLockError, match="triton"):
        check_integrity(_rehash(tmp_path, raw), tmp_path)


def test_torch_name_missing_from_first_resolve_is_refused(tmp_path: Path) -> None:
    """A torch-family name the first resolve never saw is a mismatch."""
    raw = _build(tmp_path)
    _replace(tmp_path, raw["first_resolve"]["file"], "torch==2.8.0+cu128\n", "")
    with pytest.raises(ReproduceLockError, match="torch"):
        check_integrity(_rehash(tmp_path, raw), tmp_path)


def test_torch_version_mismatch_against_uv_lock_is_refused(tmp_path: Path) -> None:
    """For uv-export bodies the torch lock must match the repository ``uv.lock`` versions."""
    raw = _build(tmp_path, "uv-export")
    uv_lock = _uv_lock(tmp_path)
    text = uv_lock.read_text()
    uv_lock.write_text(text.replace('name = "triton"\nversion = "3.4.0"', 'name = "triton"\nversion = "3.3.1"'))
    with pytest.raises(ReproduceLockError, match="triton"):
        check_integrity(EnvSpec.model_validate(raw), tmp_path, uv_lock=uv_lock)


def test_torch_hash_mismatch_against_uv_lock_is_refused(tmp_path: Path) -> None:
    """A torch-lock wheel whose hash is not among the ``uv.lock`` wheels for that name is a mismatch."""
    raw = _build(tmp_path, "uv-export")
    _replace(
        tmp_path,
        raw["torch"]["lock"],
        "5b17e2001cc0d751a5bc2c6ec6d26ad95913324a4adb86788c944f8ce9ba441f",
        NUMPY_245_SHA,
    )
    with pytest.raises(ReproduceLockError, match="nvidia-nvtx-cu12"):
        check_integrity(_rehash(tmp_path, raw), tmp_path, uv_lock=_uv_lock(tmp_path))


def test_uv_export_torch_env_needs_the_repository_lock(tmp_path: Path) -> None:
    """Without the repository ``uv.lock`` the torch check cannot run, so it refuses."""
    with pytest.raises(ReproduceLockError, match=r"uv\.lock"):
        check_integrity(EnvSpec.model_validate(_build(tmp_path, "uv-export")), tmp_path)


def test_header_round_trip() -> None:
    """``write_header`` output parses back to the same fields, and the body is kept."""
    body = (DATA / "body_small.lock").read_text()
    text = write_header(body, {"requirements-sha256": "ab" * 32, "python": "3.11"})
    assert text.splitlines()[0] == HEADER_MARKER
    assert text.endswith(body)
    assert parse_header(text) == {"requirements-sha256": "ab" * 32, "python": "3.11"}


def test_parse_header_without_marker_is_refused() -> None:
    """A lock with uv's own header, or none, has no reproduce header."""
    with pytest.raises(ReproduceLockError, match=HEADER_MARKER):
        parse_header("# requirements-sha256: x\nsix==1.17.0\n")


def test_write_header_refuses_multiline_values() -> None:
    """A value with a newline would inject lines into the lock."""
    with pytest.raises(ValueError):
        write_header("", {"python": "3.11\nsix==1.0"})


def test_requirements_sha256_covers_every_input() -> None:
    """Each input changes the digest; requirement and no-emit order do not."""
    base = (REQUIREMENTS, "3.11", "12.8", EXCLUDE_NEWER, None, None, ["a", "b"])
    digest = requirements_sha256(*base)
    assert requirements_sha256(list(reversed(REQUIREMENTS)), *base[1:6], ["b", "a"]) == digest
    variants = [
        (REQUIREMENTS[:-1], *base[1:]),
        (base[0], "3.12", *base[2:]),
        (*base[:2], None, *base[3:]),
        (*base[:3], datetime.date(2026, 7, 2), *base[4:]),
        (*base[:4], "cd" * 32, *base[5:]),
        (*base[:5], "cd" * 32, base[6]),
        (*base[:6], ["a"]),
    ]
    assert len({requirements_sha256(*v) for v in variants} | {digest}) == len(variants) + 1


def test_parse_freeze_handles_pins_and_urls() -> None:
    """``==``, ``@ file://`` and ``@ https://`` lines parse; names normalise; ``+cuXXX`` is dropped."""
    assert parse_freeze((DATA / "freeze_mixed.txt").read_text()) == {
        "annotated-doc": "0.0.5",
        "numpy": "2.4.6",
        "python-speech-features": "0.6",
        "torch": "2.8.0",
        "torchaudio": "2.8.0",
    }


@pytest.mark.parametrize(
    "line",
    ["-e file:///tmp/src/pkg", "pkg @ file:///tmp/src/pkg", "pkg @ git+https://example.org/pkg.git@abc", "pkg>=1"],
)
def test_parse_freeze_refuses_unpinned_lines(line: str) -> None:
    """Editable installs, source trees, VCS URLs and ranges are not a pinned installed set."""
    with pytest.raises(ReproduceLockError):
        parse_freeze(f"six==1.17.0\n{line}\n")
