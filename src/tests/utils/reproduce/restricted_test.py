"""Tests for restricted steps: the declared flag, the restricted results root, masking and the CLI exposure rule.

Every site is synthetic, under ``tmp_path``. The canary ``CANARY-7f3a`` stands for restricted content: it lives
under ``data/restricted`` and must never reach a public tree or a command's output.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path
from typing import Any

import pytest
import yaml

from senselab.utils.reproduce import cache, cli, inputs, restricted, run, sandbox, slurm
from senselab.utils.reproduce.spec import Manifest, Profile, StepSpec
from tests.utils.reproduce.run_test import World, _git
from tests.utils.reproduce.slurm_test import FakeSbatch
from tests.utils.reproduce.sources_fixture import register_input, verify_input

CANARY = "CANARY-7f3a"
needs_userns = pytest.mark.skipif(not sandbox.userns_available(), reason="unprivileged user namespaces are unavailable")

SCRIPT = """\
import json, os, pathlib, sys
mode, out, src = sys.argv[1], pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3])
if mode == "copy":
    text = src.read_text()
    (out / "copy.txt").write_text(text)
    value = float(len(text))
elif mode == "list":
    names = sorted(os.listdir(src))
    (out / "names.txt").write_text("\\n".join(names))
    value = float(len(names))
elif mode == "prep":
    value = float(src.read_text().strip())
else:
    value = 2 * json.loads(src.read_text())["value"]
(out / "result.json").write_text(json.dumps({"value": value}))
"""

HOOK = """\
#!/bin/sh
if /usr/bin/grep -q CANARY-7f3a "$1"; then exit 1; fi
exit 0
"""


class Site(World):
    """A `World` with a restricted dir holding the canary, a restricted results root and a fixture scan hook."""

    def __init__(self, tmp: Path) -> None:
        """Build the site under ``tmp``; the step script gains ``copy`` and ``list`` modes."""
        super().__init__(tmp)
        (self.repo / "step.py").write_text(SCRIPT)
        _git(self.repo, "add", "-A")
        _git(self.repo, "commit", "-q", "-m", "restricted modes")
        self.commit = _git(self.repo, "rev-parse", "HEAD").strip()
        self.restricted_dir = self.data / "restricted"
        (self.restricted_dir / "canary").mkdir(parents=True)
        (self.restricted_dir / "canary" / "in.txt").write_text(CANARY + "\n")
        self.hook = tmp / "hook.sh"
        self.hook.write_text(HOOK)
        self.hook.chmod(0o755)
        (tmp / "deny.txt").write_text("FORBIDDEN-[0-9]+\n")
        self.restricted_dirs: list[Path] = [self.restricted_dir]

    @property
    def restricted_root(self) -> Path:
        """The restricted results root, under the first restricted dir."""
        return self.restricted_dirs[0] / "reruns" / "results"

    def profile(self) -> Profile:
        """The `World` profile with the restricted fields set."""
        data = super().profile().model_dump(mode="json")
        data |= {
            "restricted_dirs": [str(p) for p in self.restricted_dirs],
            "restricted_results_root": str(self.restricted_root),
            "released_metrics_root": str(self.data / "reruns" / "restricted-metrics"),
            "scan_hook": [str(self.hook)],
            "rw_under_roots": [str(self.restricted_dirs[0] / "reruns"), str(self.data / "reruns")],
        }
        return Profile.model_validate(data)

    def register_canary(self) -> None:
        """Register ``data/restricted/canary`` as the restricted input ``canary``."""
        self.registered["canary"] = register_input(
            self.base,
            {"data": self.data},
            "canary",
            root="data",
            relpath="restricted/canary",
            restricted=True,
            category="restricted-derivative",
        )

    def files(self, manifest: Manifest) -> tuple[Path, Path]:
        """The profile file and the manifest file."""
        profile = self.tmp / "profile.yaml"
        profile.write_text(yaml.safe_dump(self.profile().model_dump(mode="json")))
        path = self.base / "demo.yaml"
        path.write_text(yaml.safe_dump(manifest.model_dump(mode="json"), sort_keys=False))
        return profile, path


@pytest.fixture
def site(tmp_path: Path) -> Site:
    """A fresh synthetic site with a restricted dir."""
    return Site(tmp_path)


def _canary_step(site: Site, name: str = "A", **change: Any) -> dict[str, Any]:  # noqa: ANN401
    return site.step(name, "copy", "{input:canary}/in.txt", lock_inputs=["canary"], **change)


def _up_step(site: Site, name: str, up: str, **change: Any) -> dict[str, Any]:  # noqa: ANN401
    return site.step(
        name,
        "up",
        f"{{up:{up}}}/result.json",
        needs=[up],
        inputs=[{"root": f"up:{up}", "patterns": ["result.json"]}],
        **change,
    )


def _steps(*steps: dict[str, Any]) -> list[StepSpec]:
    return [StepSpec.model_validate(s) for s in steps]


def _registry(site: Site) -> dict[str, inputs.RegisteredInput]:
    return {i: inputs.read_registry_entry(site.base, i)[0] for i in site.registered}


def _plain(site: Site) -> Profile:
    """The site's profile without the restricted fields."""
    data = site.profile().model_dump(mode="json")
    return Profile.model_validate({k: v for k, v in data.items() if k not in restricted.PROFILE_FIELDS})


def _files_with(top: Path, text: str) -> list[str]:
    """Regular files under ``top`` (symlinks not followed) whose bytes contain ``text``, as ``grep -rl`` lists them."""
    found = []
    for dirpath, _, names in os.walk(top, followlinks=False):
        for name in names:
            path = os.path.join(dirpath, name)
            if stat.S_ISREG(os.lstat(path).st_mode) and text.encode() in Path(path).read_bytes():
                found.append(path)
    return found


# --- A7: the declared flag ----------------------------------------------------------------------------------


def test_a_step_on_a_restricted_input_must_declare_restricted(site: Site) -> None:
    """A public step on a restricted input, or on a data-root pattern that reaches a restricted dir, is refused."""
    site.register_canary()
    registry, profile = _registry(site), site.profile()
    with pytest.raises(ValueError, match=r"E_SPEC_RESTRICTED: \['A'\]"):
        restricted.check_restricted(_steps(_canary_step(site)), registry, profile)
    restricted.check_restricted(_steps(_canary_step(site, restricted=True)), registry, profile)

    (site.data / "alias").symlink_to("restricted")
    reaching = ["restricted/canary/in.txt", "restricted/*/in.txt", "**/*.txt", "*/canary/*", "alias/canary/in.txt"]
    for pattern in reaching:
        step = site.step("P", "prep", "{root:data}/in/a.txt", inputs=[{"root": "data", "patterns": [pattern]}])
        with pytest.raises(ValueError, match="E_SPEC_RESTRICTED"):
            restricted.check_restricted(_steps(step), registry, profile)
        restricted.check_restricted(_steps({**step, "restricted": True}), registry, profile)
    for pattern in ("in/a.txt", "in/*.txt", "*.txt", "[ab]*/x.txt"):
        step = site.step("P", "prep", "{root:data}/in/a.txt", inputs=[{"root": "data", "patterns": [pattern]}])
        restricted.check_restricted(_steps(step), registry, profile)

    with pytest.raises(ValueError, match="E_SPEC_RESTRICTED"):
        restricted.check_restricted(_steps(_canary_step(site, restricted=True)), registry, _plain(site))

    manifest = site.manifest(_canary_step(site))
    site.prefetch(manifest)
    with pytest.raises(ValueError, match="E_SPEC_RESTRICTED"):
        run.plan_steps([manifest], site.config())
    with pytest.raises(ValueError, match="E_SPEC_RESTRICTED"):
        run.run_steps([manifest], site.config())
    assert not (site.tmp / "results").exists() and not site.restricted_root.exists()
    profile_file, path = site.files(manifest)
    lint = ["lint", str(path), "--profile", str(profile_file), "--base-dir", str(site.base)]
    assert cli.main(lint) == cli.EXIT_FAILED


def test_a_public_step_with_up_on_a_restricted_step_is_rejected_at_load(
    site: Site, capsys: pytest.CaptureFixture[str]
) -> None:
    """SC7: a public step with ``up:`` (``needs``) on a restricted step is refused before anything runs."""
    site.register_canary()
    a = _canary_step(site, restricted=True)
    b = _up_step(site, "B", "A")
    with pytest.raises(ValueError, match=r"E_SPEC_RESTRICTED: \['B'\]"):
        restricted.check_restricted(_steps(a, b), _registry(site), site.profile())
    restricted.check_restricted(_steps(a, {**b, "restricted": True}), _registry(site), site.profile())

    manifest = site.manifest(a, b)
    site.prefetch(manifest)
    with pytest.raises(ValueError, match="E_SPEC_RESTRICTED"):
        run.run_steps([manifest], site.config())
    with pytest.raises(ValueError, match="E_SPEC_RESTRICTED"):
        run.plan_steps([manifest], site.config())
    assert not site.restricted_root.exists() and not (site.tmp / "results").exists()
    profile_file, path = site.files(manifest)
    capsys.readouterr()
    assert cli.main(["lint", str(path), "--profile", str(profile_file), "--base-dir", str(site.base)]) == 1
    out = capsys.readouterr().out
    assert f"{path}:B: E_SPEC_RESTRICTED" in out and ":A: E_SPEC_RESTRICTED" not in out
    common = ["--profile", str(profile_file), "--base-dir", str(site.base), "--cache", str(site.cache)]
    assert cli.main(["run", str(path), *common, "--dry-run"]) == cli.EXIT_FAILED
    captured = capsys.readouterr()
    assert captured.out == "" and "E_SPEC_RESTRICTED" in captured.err
    with pytest.raises(ValueError, match="E_SPEC_RESTRICTED"):
        slurm.submit([manifest], _submit_config(site, manifest), runner=FakeSbatch())


@needs_userns
def test_restricted_steps_write_only_under_the_restricted_root(site: Site) -> None:
    """SC7: restricted A copies the canary into ``{out}``, restricted B reads ``up:A``; no public tree holds it."""
    site.register_canary()
    verify_input(site.base, site.profile(), "canary")
    manifest = site.manifest(_canary_step(site, restricted=True), _up_step(site, "B", "A", restricted=True))
    site.prefetch(manifest)
    outcomes = site.run(manifest)
    assert all(o.ok and o.status == "executed" for o in outcomes.values()), {n: o.codes for n, o in outcomes.items()}
    assert all(o.out is not None and o.out.parent == site.restricted_root for o in outcomes.values())
    assert (site.restricted_root / cache.CLAIMS).is_dir() and not (site.tmp / "results" / cache.CLAIMS).exists()
    a_out = outcomes["A"].out
    assert a_out is not None and CANARY in (a_out / run.WORK / "copy.txt").read_text()
    public = [site.tmp / "results", site.tmp / "runs", site.data / "reruns"]
    assert [f for top in public for f in _files_with(top, CANARY)] == []
    assert not (site.tmp / "results").exists() or list((site.tmp / "results").iterdir()) == []
    rows = run.read_run_record(cache.run_dir(site.tmp / "runs", site.config().date, "demo") / run.RUN_JSON)
    assert {r["step"]: r["restricted"] for r in rows} == {"A": True, "B": True}

    second = site.run(manifest)
    assert {o.status for o in second.values()} == {"skipped"}


@needs_userns
def test_up_resolves_each_upstream_in_its_own_root(site: Site) -> None:
    """A public upstream is found under the public root, a restricted one under the restricted root."""
    site.register_canary()
    verify_input(site.base, site.profile(), "canary")
    p0 = site.step("P0", "prep", "{root:data}/in/a.txt")
    r1 = _up_step(site, "R1", "P0", restricted=True)
    r2 = _up_step(site, "R2", "R1", restricted=True)
    manifest = site.manifest(p0, r1, r2)
    site.prefetch(manifest)
    first = site.run(manifest)
    assert all(o.ok for o in first.values()), {n: o.codes for n, o in first.items()}
    assert first["P0"].out is not None and first["P0"].out.parent == site.tmp / "results"
    assert first["R1"].out is not None and first["R1"].out.parent == site.restricted_root
    r2_out = first["R2"].out
    assert r2_out is not None and r2_out.parent == site.restricted_root
    assert '"value": 8.0' in (r2_out / run.WORK / "result.json").read_text()

    (only,) = run.run_steps([manifest], site.config(), only=["R2"])
    assert (only.status, only.key, only.out) == ("skipped", first["R2"].key, r2_out)
    upstream = {"P0": first["P0"]}
    resolved = run.resolve_step(StepSpec.model_validate(r1), "demo", site.config(), upstream)
    assert (resolved.status, resolved.out) == ("skipped", first["R1"].out)


@needs_userns
def test_restricted_dirs_are_masked_for_public_steps(site: Site) -> None:
    """A public step listing ``{root:data}/restricted`` sees an empty dir, with or without a restricted step in the set.

    A restricted step sees the dir's entries. A restricted dir named through a symlink is masked at its target.
    """
    site.register_canary()
    verify_input(site.base, site.profile(), "canary")
    listing = site.step("L", "list", "{root:data}/restricted")
    seeing = site.step("S", "list", "{root:data}/restricted", restricted=True)

    def names(*steps: dict[str, Any]) -> dict[str, str]:
        manifest = site.manifest(*steps)
        site.prefetch(manifest)
        outcomes = site.run(manifest)
        assert all(o.ok for o in outcomes.values()), {n: o.codes for n, o in outcomes.items()}
        listed = {n: o.out / run.WORK / "names.txt" for n, o in outcomes.items() if o.out is not None}
        return {n: path.read_text() for n, path in listed.items() if path.exists()}

    assert names(listing)["L"] == ""
    both = names(listing, seeing, _canary_step(site, restricted=True))
    assert both["L"] == "" and "canary" in both["S"].split("\n")

    (site.data / "rlink").symlink_to("restricted")
    site.restricted_dirs = [site.data / "rlink"]
    assert names(site.step("L2", "list", "{root:data}/restricted"))["L2"] == ""


def _submit_config(site: Site, manifest: Manifest) -> slurm.SubmitConfig:
    profile, path = site.files(manifest)
    return slurm.SubmitConfig(
        profile=site.profile(),
        profile_path=profile,
        manifest_paths=[path],
        base_dir=site.base,
        cache=site.cache,
        date=site.config().date,
        exports={},
        driver_python="/usr/bin/python3",
        manifest_sha256s=["0" * 64],
    )


@needs_userns
def test_cli_output_for_restricted_steps_is_codes_only(
    site: Site, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """CLI output over the canary set names no path under the data root and never the canary.

    The commands are ``verify-inputs``, ``run``, ``run --dry-run``, ``status``, ``lint`` and ``submit`` (fake runner).
    """
    site.register_canary()
    manifest = site.manifest(_canary_step(site, restricted=True), _up_step(site, "B", "A", restricted=True))
    site.prefetch(manifest)
    profile_file, path = site.files(manifest)
    common = ["--profile", str(profile_file), "--base-dir", str(site.base)]
    with_cache = [*common, "--cache", str(site.cache)]
    date = site.config().date.isoformat()
    captured: list[str] = []

    def call(*argv: str) -> int:
        code = cli.main(list(argv))
        out = capsys.readouterr()
        captured.extend([out.out, out.err])
        return code

    assert call("verify-inputs", *common, "canary") == cli.EXIT_OK
    assert captured[-2].startswith("canary\tok\t")
    assert call("run", str(path), *with_cache, "--date", date) == cli.EXIT_OK
    assert call("run", str(path), *with_cache, "--date", date, "--dry-run") == cli.EXIT_OK
    assert call("status", str(cache.run_dir(site.tmp / "runs", site.config().date, "demo"))) == cli.EXIT_OK
    assert call("lint", str(path), *common) == cli.EXIT_OK
    monkeypatch.setattr(slurm, "default_runner", FakeSbatch())
    monkeypatch.setenv("UV_CACHE_DIR", str(site.tmp / "uv-cache"))
    monkeypatch.setenv("UV_PYTHON_INSTALL_DIR", str(site.tmp / "uv-python"))
    assert call("submit", str(path), *with_cache, "--date", date) == cli.EXIT_OK
    text = "".join(captured)
    assert "A\t" in text and "B\t" in text
    assert CANARY not in text
    for top in {str(site.data), os.path.realpath(site.data)}:
        assert top not in text


def test_profile_validators_for_the_restricted_fields(site: Site) -> None:
    """The four restricted fields go together; the restricted root is restricted and writable; releases are public."""
    good = site.profile().model_dump(mode="json")
    Profile.model_validate(good)
    data = str(site.data)
    bad = [
        {k: v for k, v in good.items() if k != "scan_hook"},
        {**good, "restricted_results_root": f"{data}/reruns/results"},
        {**good, "rw_under_roots": [f"{data}/reruns"]},
        {**good, "released_metrics_root": f"{data}/restricted/reruns/released"},
        {**good, "results_root": f"{data}/restricted/reruns/public"},
    ]
    for case in bad:
        with pytest.raises(ValueError, match="E_PROFILE_RESTRICTED"):
            Profile.model_validate(case)
    _plain(site)
