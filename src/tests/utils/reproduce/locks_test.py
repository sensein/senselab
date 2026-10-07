"""Tests for the lock header, the integrity check and freeze parsing."""

from __future__ import annotations

import copy
import datetime
import shutil
from pathlib import Path
from typing import Any

import pytest

from senselab.utils.reproduce.locks import (
    HEADER_MARKER,
    ReproduceLockError,
    check_integrity,
    entry_version,
    file_sha256,
    lock_entries,
    parse_freeze,
    parse_header,
    requirements_digest_for,
    requirements_sha256,
    verify_files,
    versions_agree,
    write_header,
)
from senselab.utils.reproduce.spec import EnvSpec, FileRef

DATA = Path(__file__).parent / "data"
ENV_DIR = "envs/a"
EXCLUDE_NEWER = datetime.date(2026, 7, 1)
PLACEHOLDER_SHA = "00" * 32
# numpy 2.4.5, cp311 manylinux_2_27/2_28 x86_64 wheel, from the PyPI JSON API.
NUMPY_245_SHA = "07ce7e74da92d7c71b5df157b9758bcdd53d7fea10602154de3afd2b3ddc34dd"
# nvidia-cublas-cu12 12.8.4.1 wheel, as pinned in the repository uv.lock the torch fixture came from.
CUBLAS_SHA = "8ac4e771d5a348c551b2a426eda6193c19aa630236b418086020df5ba9667142"
REQUIREMENTS = ["certifi", "idna", "numpy", "packaging", "six"]


def _copy(base: Path, fixture: str, name: str) -> str:
    rel = f"{ENV_DIR}/{name}"
    shutil.copyfile(DATA / fixture, base / rel)
    return rel


def _write(base: Path, rel: str, text: str) -> None:
    (base / rel).write_text(text)


def _lock(base: Path, rel: str) -> dict[str, str]:
    return {"lock": rel, "lock_sha256": file_sha256(base / rel)}


def _file(base: Path, rel: str) -> dict[str, str]:
    return {"file": rel, "sha256": file_sha256(base / rel)}


def _build(
    base: Path,
    source: str = "requirements.in",
    torch_fixture: str = "torch_small.lock",
    header_python: str | None = None,
) -> dict[str, Any]:
    """Lay out a valid env under ``base`` and return its manifest ``env`` mapping."""
    (base / ENV_DIR).mkdir(parents=True)
    torch = _copy(base, torch_fixture, "torch.lock")
    body = f"{ENV_DIR}/body.lock"
    _write(base, body, (DATA / "body_small.lock").read_text())
    build = _copy(base, "body_small.lock", "build.lock")
    patch = _copy(base, "sdist_small.diff", "pkg.diff")
    env: dict[str, Any] = {
        "python": "3.11",
        "python_exact": "3.11.9",
        "torch": {"index_url": "https://download.pytorch.org/whl/cu128", **_lock(base, torch)},
        "body": {**_lock(base, body), "source": source},
        "sdist_allow": [
            {
                "name": "python_speech_features",
                "url": "https://files.example/python_speech_features-0.6.tar.gz",
                "sha256": "a0aebf746464bc929dc3162cb369d7ff967c398c5120ddf5fb40a65f01b92b11",
                "patches": [_file(base, patch)],
                "build_env": _lock(base, build),
            }
        ],
        "exclude_newer": EXCLUDE_NEWER.isoformat(),
        "extra_no_emit": ["opencv-python"],
        "wheels": _lock(base, _copy(base, "wheels_small.lock", "wheels.lock")),
        "test": _lock(base, _copy(base, "test_small.lock", "test.lock")),
        "notebook_tools": _lock(base, _copy(base, "test_small.lock", "notebook_tools.lock")),
        "max_cuda": "12.8",
    }
    if source == "requirements.in":
        req = _copy(base, "requirements_small.in", "requirements.in")
        config = _copy(base, "uv_config_small.toml", "uv.toml")
        overrides = _copy(base, "overrides_small.txt", "overrides.txt")
        env["body"].update(
            requirements_in=req,
            requirements_in_sha256=file_sha256(base / req),
            config_file=config,
            config_file_sha256=file_sha256(base / config),
            overrides=overrides,
            overrides_sha256=file_sha256(base / overrides),
        )
        env["first_resolve"] = _file(base, _copy(base, "first_resolve_small.txt", "first_resolve.txt"))
        digest_env = copy.deepcopy(env)
        digest_env["python"] = header_python or env["python"]
        digest = requirements_digest_for(EnvSpec.model_validate(digest_env), base)
        _write(base, body, write_header((base / body).read_text(), {"requirements-sha256": digest}))
        env["body"]["lock_sha256"] = file_sha256(base / body)
    else:
        env["body"]["export_cmd"] = ["uv", "export", "--frozen", "--no-emit-project", "--no-dev"]
    return env


def _rehash(base: Path, env: dict[str, Any]) -> EnvSpec:
    """Refresh the lock, resolve and patch sha256s from the files, as a tamperer who edits the manifest would."""
    refs = [env[k] for k in ("torch", "body", "wheels", "test", "notebook_tools") if env.get(k)]
    for ref in [*refs, *(s["build_env"] for s in env["sdist_allow"])]:
        ref["lock_sha256"] = file_sha256(base / ref["lock"])
    files = [*(p for s in env["sdist_allow"] for p in s["patches"])]
    if env.get("first_resolve"):
        files.append(env["first_resolve"])
    for ref in files:
        ref["sha256"] = file_sha256(base / ref["file"])
    return EnvSpec.model_validate(env)


def _uv_lock(base: Path) -> Path:
    path = base / "repo" / "uv.lock"
    path.parent.mkdir()
    shutil.copyfile(DATA / "uv_small.lock", path)
    return path


def _replace(path: Path, old: str, new: str) -> None:
    text = path.read_text()
    assert old in text
    path.write_text(text.replace(old, new))


def _append(path: Path, text: str) -> None:
    with path.open("a") as handle:
        handle.write(text)


def _numpy_block(text: str) -> str:
    lines = text.splitlines(keepends=True)
    start = next(i for i, line in enumerate(lines) if line.startswith("numpy=="))
    end = next(i for i in range(start + 1, len(lines)) if not lines[i].startswith(" "))
    return "".join(lines[start:end])


# Valid envs


def test_valid_requirements_in_env_passes_and_returns_verified_shas(tmp_path: Path) -> None:
    """Every hashed file is verified, and the mapping names each one with the sha256 read."""
    raw = _build(tmp_path)
    verified = check_integrity(EnvSpec.model_validate(raw), tmp_path)
    expected = {
        raw["torch"]["lock"],
        raw["body"]["lock"],
        raw["wheels"]["lock"],
        raw["test"]["lock"],
        raw["notebook_tools"]["lock"],
        raw["sdist_allow"][0]["build_env"]["lock"],
        raw["sdist_allow"][0]["patches"][0]["file"],
        raw["body"]["requirements_in"],
        raw["body"]["config_file"],
        raw["body"]["overrides"],
        raw["first_resolve"]["file"],
    }
    assert set(verified) == expected
    assert all(verified[rel] == file_sha256(tmp_path / rel) for rel in expected)


def test_index_compiled_torch_lock_passes_against_first_resolve(tmp_path: Path) -> None:
    """An index-compiled torch lock, with its ``--index-url`` line, checks against the first resolve."""
    raw = _build(tmp_path, torch_fixture="torch_index_small.lock")
    assert (tmp_path / raw["torch"]["lock"]).read_text().startswith("--index-url ")
    check_integrity(EnvSpec.model_validate(raw), tmp_path)


def test_valid_uv_export_env_passes(tmp_path: Path) -> None:
    """A uv-export env checks its torch lock against the repository ``uv.lock`` entries."""
    check_integrity(EnvSpec.model_validate(_build(tmp_path, "uv-export")), tmp_path, uv_lock=_uv_lock(tmp_path))


# Manifest sha256s


def test_numpy_repinned_to_a_real_release_is_refused(tmp_path: Path) -> None:
    """A body repinned to numpy 2.4.5 with its real hash passes ``--require-hashes``; the manifest sha catches it."""
    env = EnvSpec.model_validate(_build(tmp_path))
    body = tmp_path / env.body.lock
    text = body.read_text()
    body.write_text(text.replace(_numpy_block(text), f"numpy==2.4.5 \\\n    --hash=sha256:{NUMPY_245_SHA}\n"))
    assert "numpy==2.4.6" not in body.read_text()
    with pytest.raises(ReproduceLockError, match=r"body lock .*body\.lock: sha256"):
        check_integrity(env, tmp_path)


CHANGED_FILES = {
    "torch": (lambda e: e["torch"]["lock"], r"torch lock .*: sha256"),
    "wheels": (lambda e: e["wheels"]["lock"], r"wheels lock .*: sha256"),
    "test": (lambda e: e["test"]["lock"], r"test lock .*: sha256"),
    "notebook_tools": (lambda e: e["notebook_tools"]["lock"], r"notebook_tools lock .*: sha256"),
    "build_env": (lambda e: e["sdist_allow"][0]["build_env"]["lock"], r"build_env lock of .*: sha256"),
    "sdist_patch": (lambda e: e["sdist_allow"][0]["patches"][0]["file"], r"patch of .*: sha256"),
    "requirements_in": (lambda e: e["body"]["requirements_in"], r"requirements\.in .*: sha256"),
    "config_file": (lambda e: e["body"]["config_file"], r"config file .*: sha256"),
    "overrides": (lambda e: e["body"]["overrides"], r"overrides .*: sha256"),
    "first_resolve": (lambda e: e["first_resolve"]["file"], r"first_resolve .*: sha256"),
}


@pytest.mark.parametrize("field", sorted(CHANGED_FILES))
def test_any_hashed_file_changed_is_refused(tmp_path: Path, field: str) -> None:
    """Each hashed input is checked by its own sha256 check; a trailing newline changes nothing else."""
    raw = _build(tmp_path)
    rel, message = CHANGED_FILES[field]
    _append(tmp_path / rel(raw), "\n")
    with pytest.raises(ReproduceLockError, match=message):
        check_integrity(EnvSpec.model_validate(raw), tmp_path)


def test_path_escaping_through_a_symlink_is_refused(tmp_path: Path) -> None:
    """A relative manifest path that resolves outside the manifest repository is refused."""
    base = tmp_path / "manifests"
    base.mkdir()
    raw = _build(base)
    outside = tmp_path / "elsewhere.lock"
    shutil.copyfile(base / raw["wheels"]["lock"], outside)
    (base / raw["wheels"]["lock"]).unlink()
    (base / raw["wheels"]["lock"]).symlink_to(outside)
    with pytest.raises(ReproduceLockError, match="resolves outside"):
        check_integrity(EnvSpec.model_validate(raw), base)


def test_verify_files_checks_patches(tmp_path: Path) -> None:
    """Step patches are verified by the same helper and reported with their sha256."""
    patch = tmp_path / "0001.diff"
    shutil.copyfile(DATA / "sdist_small.diff", patch)
    ref = FileRef(file="0001.diff", sha256=file_sha256(patch))
    assert verify_files([ref], tmp_path) == {"0001.diff": file_sha256(patch)}
    _append(patch, "+tampered\n")
    with pytest.raises(ReproduceLockError, match=r"0001\.diff: sha256"):
        verify_files([ref], tmp_path)


# Header


def test_header_recompute_mismatch_is_refused(tmp_path: Path) -> None:
    """A header compiled for other inputs fails even when every file sha256 matches the manifest."""
    raw = _build(tmp_path, header_python="3.12")
    with pytest.raises(ReproduceLockError, match="header requirements-sha256 .* != recomputed"):
        check_integrity(EnvSpec.model_validate(raw), tmp_path)


def test_body_without_header_is_refused(tmp_path: Path) -> None:
    """A requirements.in body must carry the reproduce header."""
    raw = _build(tmp_path)
    _write(tmp_path, raw["body"]["lock"], (DATA / "body_small.lock").read_text())
    with pytest.raises(ReproduceLockError, match=HEADER_MARKER):
        check_integrity(_rehash(tmp_path, raw), tmp_path)


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
    with pytest.raises(ValueError, match="bad header field"):
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


def test_requirements_digest_normalises_lines(tmp_path: Path) -> None:
    """Comments, continuations and inner whitespace do not change the digest."""
    raw = _build(tmp_path)
    env = EnvSpec.model_validate(raw)
    digest = requirements_digest_for(env, tmp_path)
    _write(
        tmp_path,
        raw["body"]["requirements_in"],
        "# body\ncertifi  # roots\nidna\nnumpy\npackaging\n\n   six\n",
    )
    assert requirements_digest_for(env, tmp_path) == digest
    _write(tmp_path, raw["body"]["requirements_in"], "numpy>=2,  <3\n")
    other = requirements_digest_for(env, tmp_path)
    _write(tmp_path, raw["body"]["requirements_in"], "numpy>=2, <3\n")
    assert requirements_digest_for(env, tmp_path) == other
    _write(tmp_path, raw["body"]["requirements_in"], "numpy>=2, \\\n    <3\n")
    assert requirements_digest_for(env, tmp_path) == other


def test_requirements_digest_refuses_duplicate_lines(tmp_path: Path) -> None:
    """A repeated requirement line is refused rather than silently collapsed."""
    raw = _build(tmp_path)
    _write(tmp_path, raw["body"]["requirements_in"], "six\nnumpy\nsix  # again\n")
    with pytest.raises(ReproduceLockError, match=r"repeats requirement lines: \['six'\]"):
        requirements_digest_for(EnvSpec.model_validate(raw), tmp_path)


def test_requirements_digest_needs_a_requirements_body(tmp_path: Path) -> None:
    """A uv-export body has no requirements digest."""
    with pytest.raises(ReproduceLockError, match="only a requirements.in body"):
        requirements_digest_for(EnvSpec.model_validate(_build(tmp_path, "uv-export")), tmp_path)


@pytest.mark.parametrize(
    "line",
    ["-r more.in", "--requirement more.in", "-c constraints.txt", "--constraint=c.txt", "-e ./pkg", "--editable ."],
)
def test_requirements_digest_refuses_file_and_editable_options(tmp_path: Path, line: str) -> None:
    """A line that pulls in another file or a source tree is refused: what it names is not hashed."""
    raw = _build(tmp_path)
    _write(tmp_path, raw["body"]["requirements_in"], f"six\n{line}\n")
    with pytest.raises(ReproduceLockError, match="option line refused"):
        requirements_digest_for(EnvSpec.model_validate(raw), tmp_path)


def test_requirements_digest_keeps_index_options(tmp_path: Path) -> None:
    """Index options are part of the digested lines, so changing one changes the digest."""
    raw = _build(tmp_path)
    env = EnvSpec.model_validate(raw)
    _write(tmp_path, raw["body"]["requirements_in"], "--index-url https://a.example/simple\nsix\n")
    digest = requirements_digest_for(env, tmp_path)
    _write(tmp_path, raw["body"]["requirements_in"], "--index-url https://b.example/simple\nsix\n")
    assert requirements_digest_for(env, tmp_path) != digest


def test_check_integrity_reads_each_file_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin and header checks run on the bytes that were hashed: no file is opened a second time."""
    raw = _build(tmp_path)
    reads: list[Path] = []
    real_bytes, real_text = Path.read_bytes, Path.read_text

    def read_bytes(self: Path) -> bytes:
        reads.append(self.resolve())
        return real_bytes(self)

    def read_text(self: Path, *args: Any, **kwargs: Any) -> str:  # noqa: ANN401
        reads.append(self.resolve())
        return real_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_bytes", read_bytes)
    monkeypatch.setattr(Path, "read_text", read_text)
    check_integrity(EnvSpec.model_validate(raw), tmp_path)
    assert reads
    assert sorted({p for p in reads if reads.count(p) > 1}) == []


# Overlays


def test_overlay_pin_mismatch_is_refused(tmp_path: Path) -> None:
    """A name pinned in an overlay and a base lock must have the identical pin."""
    raw = _build(tmp_path)
    _replace(tmp_path / raw["test"]["lock"], "packaging==24.2", "packaging==24.1")
    with pytest.raises(ReproduceLockError, match=r"test overlay refused: \['packaging: 24\.1 differs"):
        check_integrity(_rehash(tmp_path, raw), tmp_path)


def test_overlay_hash_mismatch_is_refused(tmp_path: Path) -> None:
    """The same version with a different hash set is a mismatch, in either overlay."""
    raw = _build(tmp_path)
    _replace(
        tmp_path / raw["notebook_tools"]["lock"],
        " \\\n    --hash=sha256:c228a6dc5e932d346bc5739379109d49e8853dd8223571c7c5b55260edc0b97f",
        "",
    )
    with pytest.raises(ReproduceLockError, match=r"notebook_tools overlay refused: \['packaging: 24\.2 differs"):
        check_integrity(_rehash(tmp_path, raw), tmp_path)


def test_overlay_marker_mismatch_is_refused(tmp_path: Path) -> None:
    """The same pin and hashes under a different marker is a mismatch."""
    raw = _build(tmp_path)
    _replace(tmp_path / raw["test"]["lock"], "packaging==24.2 \\", "packaging==24.2 ; sys_platform == 'linux' \\")
    with pytest.raises(ReproduceLockError, match="packaging: 24.2 differs"):
        check_integrity(_rehash(tmp_path, raw), tmp_path)


def test_overlay_against_marker_variants_fails_closed(tmp_path: Path) -> None:
    """When a base lock pins a name more than once, any overlay pin of that name is refused."""
    raw = _build(tmp_path)
    body = tmp_path / raw["body"]["lock"]
    block = next(b for b in body.read_text().split("\n") if b.startswith("packaging=="))
    _append(body, block.replace("packaging==24.2", "packaging==24.1 ; python_version < '3.8'") + "\n")
    with pytest.raises(ReproduceLockError, match="packaging: pinned in more than one entry"):
        check_integrity(_rehash(tmp_path, raw), tmp_path)


@pytest.mark.parametrize(
    "line, message",
    [
        ("python-speech-features==0.6", "python-speech-features: held out"),
        ("opencv-python==4.10.0.84", "opencv-python: held out"),
        (f"nvidia-cublas-cu12==12.8.4.1 --hash=sha256:{CUBLAS_SHA}", "nvidia-cublas-cu12: torch-family name absent"),
        ("torchvision==0.23.0", "torchvision: torch-family name absent"),
    ],
)
def test_overlay_may_not_pin_held_out_or_unlocked_torch_names(tmp_path: Path, line: str, message: str) -> None:
    """Overlays cannot add sdist, extra_no_emit or torch-family names the base locks leave to other stages."""
    raw = _build(tmp_path)
    _append(tmp_path / raw["test"]["lock"], line + "\n")
    with pytest.raises(ReproduceLockError, match=message):
        check_integrity(_rehash(tmp_path, raw), tmp_path)


# Torch consistency


def test_torch_mismatch_against_first_resolve_is_refused(tmp_path: Path) -> None:
    """The torch-family pins must equal those the body was resolved against."""
    raw = _build(tmp_path)
    _replace(tmp_path / raw["first_resolve"]["file"], "triton==3.4.0", "triton==3.3.1")
    with pytest.raises(ReproduceLockError, match=r"triton: torch\.lock 3\.4\.0, reference \['3\.3\.1'\]"):
        check_integrity(_rehash(tmp_path, raw), tmp_path)


def test_torch_name_missing_from_first_resolve_is_refused(tmp_path: Path) -> None:
    """A torch-family name the first resolve never saw is a mismatch."""
    raw = _build(tmp_path)
    _replace(tmp_path / raw["first_resolve"]["file"], "torch==2.8.0+cu128\n", "")
    with pytest.raises(ReproduceLockError, match=r"torch: torch\.lock 2\.8\.0\+cu128, reference \[\]"):
        check_integrity(_rehash(tmp_path, raw), tmp_path)


def test_torch_family_name_missing_from_torch_lock_is_refused(tmp_path: Path) -> None:
    """A torch-family name in the first resolve but not in torch.lock is a mismatch."""
    raw = _build(tmp_path)
    _append(tmp_path / raw["first_resolve"]["file"], "torchaudio==2.8.0+cu128\n")
    with pytest.raises(ReproduceLockError, match="torchaudio: in the reference, absent from torch.lock"):
        check_integrity(_rehash(tmp_path, raw), tmp_path)


def test_torch_cuda_variant_mismatch_is_refused(tmp_path: Path) -> None:
    """When both sides carry a local tag they must match: cu126 is not cu128."""
    raw = _build(tmp_path, torch_fixture="torch_index_small.lock")
    _replace(tmp_path / raw["first_resolve"]["file"], "torch==2.8.0+cu128", "torch==2.8.0+cu126")
    with pytest.raises(ReproduceLockError, match=r"torch: torch\.lock 2\.8\.0\+cu128, reference \['2\.8\.0\+cu126'\]"):
        check_integrity(_rehash(tmp_path, raw), tmp_path)


def test_torch_local_tag_on_one_side_only_agrees(tmp_path: Path) -> None:
    """A resolve that spells torch without its local tag still agrees."""
    raw = _build(tmp_path)
    _replace(tmp_path / raw["first_resolve"]["file"], "torch==2.8.0+cu128", "torch==2.8.0")
    check_integrity(_rehash(tmp_path, raw), tmp_path)
    assert versions_agree("2.8.0+cu128", "2.8.0")
    assert not versions_agree("2.8.0+cu128", "2.8.0+cu126")
    assert not versions_agree("2.8.0", "2.8.1+cu128")


def test_torch_version_mismatch_against_uv_lock_is_refused(tmp_path: Path) -> None:
    """For uv-export bodies the torch lock must match the repository ``uv.lock`` versions."""
    raw = _build(tmp_path, "uv-export")
    uv_lock = _uv_lock(tmp_path)
    _replace(uv_lock, 'name = "triton"\nversion = "3.4.0"', 'name = "triton"\nversion = "3.3.1"')
    with pytest.raises(ReproduceLockError, match=r"triton: torch\.lock 3\.4\.0, reference \['3\.3\.1'\]"):
        check_integrity(EnvSpec.model_validate(raw), tmp_path, uv_lock=uv_lock)


def test_torch_hash_mismatch_against_uv_lock_is_refused(tmp_path: Path) -> None:
    """A torch-lock wheel whose hash is not among the ``uv.lock`` wheels for that name is a mismatch."""
    raw = _build(tmp_path, "uv-export")
    _replace(
        tmp_path / raw["torch"]["lock"],
        "5b17e2001cc0d751a5bc2c6ec6d26ad95913324a4adb86788c944f8ce9ba441f",
        NUMPY_245_SHA,
    )
    with pytest.raises(ReproduceLockError, match=r"nvidia-nvtx-cu12: torch\.lock 12\.8\.90"):
        check_integrity(_rehash(tmp_path, raw), tmp_path, uv_lock=_uv_lock(tmp_path))


def test_unhashed_torch_entry_is_refused(tmp_path: Path) -> None:
    """Every torch-lock entry needs a hash: an empty set is trivially a subset of the uv.lock hashes."""
    raw = _build(tmp_path, "uv-export")
    _replace(
        tmp_path / raw["torch"]["lock"],
        " \\\n    --hash=sha256:5b17e2001cc0d751a5bc2c6ec6d26ad95913324a4adb86788c944f8ce9ba441f",
        "",
    )
    with pytest.raises(ReproduceLockError, match="nvidia-nvtx-cu12: no --hash in torch.lock"):
        check_integrity(_rehash(tmp_path, raw), tmp_path, uv_lock=_uv_lock(tmp_path))


CUBLAS_URL = "https://files.example/nvidia_cublas_cu12-12.8.4.1-py3-none-manylinux_2_27_x86_64.whl"
CUBLAS_PACKAGE = f"""
[[package]]
name = "nvidia-cublas-cu12"
version = "12.8.4.1"
source = {{ registry = "https://pypi.org/simple" }}
wheels = [
    {{ url = "{CUBLAS_URL}", hash = "sha256:{CUBLAS_SHA}" }},
]
"""


def _closure_uv_lock(base: Path, torch_needs_cublas: bool = False) -> Path:
    path = base / "repo" / "uv.lock"
    path.parent.mkdir()
    text = (DATA / "uv_closure_small.lock").read_text()
    if torch_needs_cublas:
        nvtx = '    { name = "nvidia-nvtx-cu12", marker'
        text = text.replace(nvtx, '    { name = "nvidia-cublas-cu12" },\n' + nvtx, 1) + CUBLAS_PACKAGE
    path.write_text(text)
    return path


def test_cuda_names_outside_the_torch_closure_are_not_required(tmp_path: Path) -> None:
    """A CUDA-named package that only a non-torch dependency pulls in is not part of the torch family."""
    raw = _build(tmp_path, "uv-export")
    uv_lock = _closure_uv_lock(tmp_path)
    assert 'name = "cuda-bindings"' in uv_lock.read_text()
    check_integrity(EnvSpec.model_validate(raw), tmp_path, uv_lock=uv_lock)


def test_torch_dependency_missing_from_torch_lock_is_refused(tmp_path: Path) -> None:
    """An ``nvidia-*`` package torch depends on in ``uv.lock`` must be in torch.lock."""
    raw = _build(tmp_path, "uv-export")
    uv_lock = _closure_uv_lock(tmp_path, torch_needs_cublas=True)
    with pytest.raises(ReproduceLockError, match="nvidia-cublas-cu12: in the reference, absent from torch.lock"):
        check_integrity(EnvSpec.model_validate(raw), tmp_path, uv_lock=uv_lock)


def test_torch_root_missing_from_uv_lock_closure_is_refused(tmp_path: Path) -> None:
    """A torch.lock entry outside the ``uv.lock`` torch closure has no reference version."""
    raw = _build(tmp_path, "uv-export")
    uv_lock = _closure_uv_lock(tmp_path)
    _replace(uv_lock, '    { name = "triton", marker', '    { name = "not-triton", marker')
    with pytest.raises(ReproduceLockError, match=r"triton: torch\.lock 3\.4\.0, reference \[\]"):
        check_integrity(EnvSpec.model_validate(raw), tmp_path, uv_lock=uv_lock)


@pytest.mark.parametrize(
    "torch_needs_cublas, refused",
    [(False, False), (True, True)],
)
def test_overlay_torch_rule_uses_the_uv_lock_closure(tmp_path: Path, torch_needs_cublas: bool, refused: bool) -> None:
    """With a repository ``uv.lock``, an overlay may pin a CUDA name torch does not depend on, not one it does."""
    raw = _build(tmp_path, "uv-export")
    uv_lock = _closure_uv_lock(tmp_path, torch_needs_cublas)
    _append(tmp_path / raw["test"]["lock"], f"nvidia-cublas-cu12==12.8.4.1 --hash=sha256:{CUBLAS_SHA}\n")
    env = _rehash(tmp_path, raw)
    if refused:
        with pytest.raises(ReproduceLockError, match="nvidia-cublas-cu12: torch-family name absent"):
            check_integrity(env, tmp_path, uv_lock=uv_lock)
    else:
        check_integrity(env, tmp_path, uv_lock=uv_lock)


TRITON_LINE = "triton==3.4.0 --hash=sha256:7b70f5e6a41e52e48cfc087436c8a28c17ff98db369447bcaff3b887a3ab4467\n"
CUBLAS_LINE = f"nvidia-cublas-cu12==12.8.4.1 --hash=sha256:{CUBLAS_SHA}\n"
CUDA_BINDINGS_LINE = (
    "cuda-bindings==13.3.1 --hash=sha256:04436a9364059c84b8f9636f359eccda1cf814341f5b670c71d80d2f79dbc708\n"
)


@pytest.mark.parametrize("line", [TRITON_LINE, CUBLAS_LINE])
def test_body_may_not_pin_torch_names(tmp_path: Path, line: str) -> None:
    """A body entry named in torch.lock, or torch-family by name against a first resolve, is refused."""
    raw = _build(tmp_path)
    _append(tmp_path / raw["body"]["lock"], line)
    with pytest.raises(ReproduceLockError, match="pins torch.lock or torch-family names"):
        check_integrity(_rehash(tmp_path, raw), tmp_path)


@pytest.mark.parametrize("line, refused", [(TRITON_LINE, True), (CUDA_BINDINGS_LINE, False)])
def test_uv_export_body_torch_rule_uses_the_uv_lock_closure(tmp_path: Path, line: str, refused: bool) -> None:
    """Against ``uv.lock`` a body may carry a CUDA name outside the torch closure, never one inside it."""
    raw = _build(tmp_path, "uv-export")
    uv_lock = _closure_uv_lock(tmp_path)
    _append(tmp_path / raw["body"]["lock"], line)
    env = _rehash(tmp_path, raw)
    if refused:
        with pytest.raises(ReproduceLockError, match=r"pins torch.lock or torch-family names: \['triton'\]"):
            check_integrity(env, tmp_path, uv_lock=uv_lock)
    else:
        check_integrity(env, tmp_path, uv_lock=uv_lock)


def test_uv_export_torch_env_needs_the_repository_lock(tmp_path: Path) -> None:
    """Without the repository ``uv.lock`` the torch check cannot run, so it refuses."""
    with pytest.raises(ReproduceLockError, match=r"needs the repository uv\.lock"):
        check_integrity(EnvSpec.model_validate(_build(tmp_path, "uv-export")), tmp_path)


# Parsing


def test_lock_entries_skip_index_options_and_keep_markers() -> None:
    """Index and find-links lines are skipped; markers are kept as normalised text."""
    text = (
        "--index-url https://download.example/whl/cu128\n"
        "--extra-index-url https://pypi.example/simple\n"
        "--find-links https://wheels.example/\n"
        "six==1.17.0 ;  python_version >= '3.8' \\\n    --hash=sha256:" + "ab" * 32 + "\n"
    )
    (entry,) = lock_entries(text)
    assert (entry.name, entry.pin, entry.marker, entry.hashes) == (
        "six",
        "1.17.0",
        "python_version >= '3.8'",
        frozenset({"ab" * 32}),
    )


@pytest.mark.parametrize(
    "line",
    ["-e file:///tmp/src/pkg", "-r other.txt", "-c constraints.txt", "--no-binary :all:", "pkg===1.0", "pkg>=1"],
)
def test_lock_entries_refuse_other_lines(line: str) -> None:
    """Editable installs, includes, other options, arbitrary-equality and ranges are refused."""
    with pytest.raises(ReproduceLockError, match="refused|not a pinned requirement"):
        lock_entries(f"six==1.17.0\n{line}\n")


def test_url_pin_must_name_its_own_project() -> None:
    """A URL pin whose wheel belongs to another project is refused."""
    url = "https://download-r2.pytorch.org/whl/cu128/torchaudio-2.8.0%2Bcu128-cp311-cp311-manylinux_2_28_x86_64.whl"
    (entry,) = lock_entries(f"torch @ {url}\n")
    with pytest.raises(ReproduceLockError, match="torch is pinned to a wheel of torchaudio"):
        entry_version(entry)


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
    "line, message",
    [
        ("-e file:///tmp/src/pkg", "option line refused"),
        ("pkg @ file:///tmp/src/pkg", "not pinned to a wheel"),
        ("pkg @ git+https://example.org/pkg.git@abc", "not pinned to a wheel"),
        ("pkg>=1", "not a pinned requirement"),
    ],
)
def test_parse_freeze_refuses_unpinned_lines(line: str, message: str) -> None:
    """Editable installs, source trees, VCS URLs and ranges are not a pinned installed set."""
    with pytest.raises(ReproduceLockError, match=message):
        parse_freeze(f"six==1.17.0\n{line}\n")
