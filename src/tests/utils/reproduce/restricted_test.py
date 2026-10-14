"""Tests for restricted steps: the declared flag, the restricted results root, masking and the CLI exposure rule.

Every site is synthetic, under ``tmp_path``. The canary ``CANARY-7f3a`` stands for restricted content: it lives
under ``data/restricted`` and must never reach a public tree or a command's output.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import stat
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import pytest
import yaml

from senselab.utils.reproduce import build, cache, cli, env, inputs, restricted, run, sandbox, slurm
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
        (tmp / "deny.txt").write_text("FORBIDDEN-[0-9]+\n")

    @property
    def restricted_root(self) -> Path:
        """The restricted results root, under the first restricted dir."""
        return self.restricted_dirs[0] / "reruns" / "results"

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

    (site.restricted_dir / "unflagged").mkdir()
    (site.restricted_dir / "unflagged" / "u.txt").write_text("1\n")
    site.registered["unflagged"] = register_input(
        site.base, {"data": site.data}, "unflagged", root="data", relpath="restricted/unflagged"
    )
    unflagged = site.step("U", "prep", "{input:unflagged}/u.txt", lock_inputs=["unflagged"])
    with pytest.raises(ValueError, match=r"E_SPEC_RESTRICTED: \['U'\]"):
        restricted.check_restricted(_steps(unflagged), _registry(site), profile)
    del site.registered["unflagged"]

    (site.data / "in" / "link.txt").symlink_to("../restricted/canary/in.txt")
    linked = site.step("Q", "prep", "{root:data}/in/a.txt", inputs=[{"root": "data", "patterns": ["in/*.txt"]}])
    restricted.check_restricted(_steps(linked), registry, profile)
    linked_manifest = site.manifest(linked)
    site.prefetch(linked_manifest)
    (plan,) = run.plan_steps([linked_manifest], site.config())
    assert plan.status == "blocked" and plan.codes == ("E_SPEC_RESTRICTED",)
    (outcome,) = run.run_steps([linked_manifest], site.config())
    assert (outcome.status, outcome.codes, outcome.key) == ("failed", ("E_SPEC_RESTRICTED",), None)
    (site.data / "in" / "link.txt").unlink()

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
        {**good, "restricted_results_root": f"{data}/restricted"},
        {**good, "job_tmp_root": f"{data}/restricted/tmp"},
        {**good, "runs_root": str(site.tmp / "runs-link" / "runs")},
    ]
    (site.tmp / "runs-link").symlink_to(site.restricted_dir)
    for case in bad:
        with pytest.raises(ValueError, match="E_PROFILE_RESTRICTED"):
            Profile.model_validate(case)
    _plain(site)
    entry = inputs.RegisteredInput.model_validate(
        {
            "id": "x",
            "root": "data",
            "relpath": "restricted/x",
            "manifest": {"root": "data", "relpath": "m.tsv", "sha256": "0" * 64},
            "rung": "a",
            "rows": "none",
            "restricted": True,
            "licence": "CC0-1.0",
            "category": "restricted-derivative",
        }
    )
    assert inputs.verify_dir(entry, site.profile()).is_relative_to(site.restricted_dir)


def test_every_run_record_row_carries_the_restricted_flag(site: Site) -> None:
    """Placeholder, settled and refused rows carry ``restricted`` as run rows do."""
    site.register_canary()
    a, b = _canary_step(site, restricted=True), site.step("P", "prep", "{root:data}/in/a.txt")
    manifest = site.manifest(a, b)
    site.prefetch(manifest)
    config = site.config()
    runs = site.tmp / "runs"
    record = cache.run_dir(runs, config.date, "demo") / run.RUN_JSON

    def flags() -> dict[str, Any]:
        return {r["step"]: r.get("restricted") for r in run.read_run_record(record)}

    run.reserve_run_record(runs, config.date, "demo", ["A", "P"], "sub-1", {"A": True, "P": False})
    assert flags() == {"A": True, "P": False}
    run.finalize_run_records([manifest], config, submission="sub-1")
    assert flags() == {"A": True, "P": False}
    assert {r["status"] for r in run.read_run_record(record)} == {"never-ran"}
    record.unlink()
    profile_file, path = site.files(manifest)
    argv = ["run", str(path), "--profile", str(profile_file), "--base-dir", str(site.base), "--cache", str(site.cache)]
    run.update_run_record(runs, config.date, "demo", [run.never_ran_row("A"), run.never_ran_row("P")], order=["A"])
    cli.main([*argv, "--date", config.date.isoformat(), "--pin", "0" * 64, "--step", "A"])
    assert flags() == {"A": True, "P": None}


# --- A8: release copy, scan hook, closure purge -------------------------------------------------------------

LABEL_ADAPTER = """\
import json, pathlib, sys
out, dest = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
value = json.loads((out / "result.json").read_text())["value"]
entry = {"name": "f_stat", "label": "CANARY-7f3a", "value": value, "numdf": 1, "dendf": 10}
dest.write_text(json.dumps({"headline": [entry]}))
"""


def _released(site: Site, key: str | None) -> Path:
    assert key is not None
    return site.data / "reruns" / "restricted-metrics" / key


def _run_one(site: Site, step: dict[str, Any]) -> run.StepOutcome:
    manifest = site.manifest(step)
    site.prefetch(manifest)
    (outcome,) = run.run_steps([manifest], site.config())
    return outcome


def _summary(outcome: run.StepOutcome) -> dict[str, Any]:
    assert outcome.out is not None
    data: dict[str, Any] = json.loads((outcome.out / cache.SUMMARY).read_text())
    return data


def _verified_canary(site: Site) -> None:
    site.register_canary()
    verify_input(site.base, site.profile(), "canary")


def _hit_step(site: Site, name: str = "H") -> dict[str, Any]:
    (site.base / "adapter_label.py").write_text(LABEL_ADAPTER)
    schema = {"kind": "statistic", "headline": ["f_stat"], "headline_labels": [CANARY]}
    return _canary_step(site, name, restricted=True, adapter="adapter_label.py", metrics_schema=schema)


@needs_userns
def test_released_metrics_appear_after_a_clean_scan(site: Site) -> None:
    """SC7: a done restricted step's clean ``metrics.json`` and ``summary.json`` are copied to the release root."""
    _verified_canary(site)
    outcome = _run_one(site, _canary_step(site, restricted=True))
    assert outcome.ok and outcome.out is not None, outcome.codes
    summary = _summary(outcome)
    assert summary["released"] is True and "W_RELEASE_REFUSED" not in summary["codes"]
    released = _released(site, outcome.key)
    for name in (run.METRICS, cache.SUMMARY):
        assert (released / name).read_bytes() == (outcome.out / name).read_bytes()
        assert stat.S_IMODE(os.stat(released / name).st_mode) == 0o660
    assert (outcome.out / run.LOGS / restricted.SCAN_LOG).is_file()
    public = _run_one(site, site.step("P", "prep", "{root:data}/in/a.txt"))
    assert _summary(public)["released"] is None


@needs_userns
def test_a_scan_hit_withholds_the_release(site: Site) -> None:
    """A headline label carrying the canary reaches ``metrics.json``: the hook refuses, nothing is released, exit 0."""
    _verified_canary(site)
    outcome = _run_one(site, _hit_step(site))
    assert outcome.out is not None and CANARY in (outcome.out / run.METRICS).read_text()
    assert outcome.ok and "W_RELEASE_REFUSED" in outcome.codes
    summary = _summary(outcome)
    assert summary["released"] is False and summary["exit"] == 0
    assert not _released(site, outcome.key).exists()


@needs_userns
def test_a_missing_hook_executable_fails_closed(
    site: Site, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A hook gone after the load check releases nothing; a hook missing at load is ``E_SCAN_HOOK_MISSING``."""
    _verified_canary(site)
    step = _canary_step(site, restricted=True)
    site.hook.rename(site.tmp / "hook.away")
    with monkeypatch.context() as patched:
        patched.setattr(restricted, "hook_missing", lambda profile: False)
        outcome = _run_one(site, step)
    assert outcome.ok and "W_RELEASE_REFUSED" in outcome.codes and _summary(outcome)["released"] is False
    assert not _released(site, outcome.key).exists()

    manifest = site.manifest(_canary_step(site, restricted=True, replicate=1))
    (plan,) = run.plan_steps([manifest], site.config())
    assert plan.status == "blocked" and "E_SCAN_HOOK_MISSING" in plan.codes
    site.prefetch(manifest)
    (failed,) = run.run_steps([manifest], site.config())
    assert failed.status == "failed" and failed.codes == ("E_SCAN_HOOK_MISSING",)
    profile_file, path = site.files(manifest)
    capsys.readouterr()
    assert cli.main(["lint", str(path), "--profile", str(profile_file), "--base-dir", str(site.base)]) == 1
    assert f"{profile_file}: scan-hook-missing" in capsys.readouterr().out
    site.hook.write_text("#!/bin/sh\nexit 0\n")
    site.hook.chmod(0o644)
    assert restricted.hook_missing(site.profile())


@needs_userns
def test_a_hook_error_or_timeout_fails_closed(site: Site, monkeypatch: pytest.MonkeyPatch) -> None:
    """A hook exiting 2, or running past ``SCAN_TIMEOUT_S``, refuses the release."""
    _verified_canary(site)
    for body, timeout in (("exit 2", 300.0), ("/usr/bin/sleep 5\nexit 0", 1.0)):
        site.hook.write_text(f"#!/bin/sh\n{body}\n")
        monkeypatch.setattr(restricted, "SCAN_TIMEOUT_S", timeout)
        outcome = _run_one(site, _canary_step(site, restricted=True, replicate=int(timeout)))
        summary = _summary(outcome)
        assert outcome.ok and "W_RELEASE_REFUSED" in summary["codes"] and summary["released"] is False, body
        assert not _released(site, outcome.key).exists()
    assert not restricted.run_scan_hook(site.profile(), site.hook)


@needs_userns
def test_extracted_values_are_scanned(site: Site) -> None:
    """Extracted values the hook refuses fail the step with ``E_EXTRACT``; so does a profile with no hook."""
    from tests.utils.reproduce.run_test import EXTRACT, EXTRACT_ADAPTER

    (site.base / "adapter_extract.py").write_text(EXTRACT_ADAPTER.replace("EXTRACTED", repr({"matched": "yes"})))
    extract = [EXTRACT[0]]
    step = site.step("X", "prep", "{root:data}/in/a.txt", adapter="adapter_extract.py", extract=extract)
    assert _run_one(site, step).ok
    site.hook.write_text('#!/bin/sh\nif /usr/bin/grep -q \'"yes"\' "$1"; then exit 1; fi\nexit 0\n')
    refused = _run_one(site, {**step, "replicate": 1})
    assert "E_EXTRACT" in refused.codes and not refused.ok and _summary(refused)["extracted"] is None
    site.restricted_fields = False
    plain = _run_one(site, {**step, "replicate": 2})
    assert "E_EXTRACT" in plain.codes and not plain.ok


def _flip_input(site: Site, relpath: str = "pub/flip") -> None:
    tree = site.data / relpath
    tree.mkdir(parents=True, exist_ok=True)
    (tree / "f.txt").write_text("3\n")
    site.registered["flip"] = register_input(site.base, {"data": site.data}, "flip", root="data", relpath=relpath)
    verify_input(site.base, site.profile(), "flip")


def _chain(site: Site) -> Manifest:
    a = site.step("A", "prep", "{input:flip}/f.txt", lock_inputs=["flip"])
    unrelated = site.step("U", "prep", "{root:data}/in/a.txt")
    manifest = site.manifest(a, _up_step(site, "B", "A"), _up_step(site, "C", "B"), unrelated)
    site.prefetch(manifest)
    return manifest


def _verify_line(site: Site, capsys: pytest.CaptureFixture[str], input_id: str) -> str:
    profile_file = site.tmp / "profile.yaml"
    profile_file.write_text(yaml.safe_dump(site.profile().model_dump(mode="json")))
    capsys.readouterr()
    cli.main(["verify-inputs", "--profile", str(profile_file), "--base-dir", str(site.base), input_id])
    out: str = capsys.readouterr().out
    return out.strip()


def _reflag(site: Site, input_id: str, **change: Any) -> None:  # noqa: ANN401
    path = site.base / inputs.REGISTRY_DIR / f"{input_id}.yaml"
    path.write_text(yaml.safe_dump({**yaml.safe_load(path.read_text()), **change}, sort_keys=False))


@needs_userns
def test_public_summaries_record_the_input_closure(site: Site) -> None:
    """Each public summary records the registered inputs of the step and of all its upstream steps."""
    _flip_input(site)
    outcomes = site.run(_chain(site))
    assert all(o.ok for o in outcomes.values()), {n: o.codes for n, o in outcomes.items()}
    closures = {n: _summary(o)["input_closure"] for n, o in outcomes.items()}
    assert closures == {"A": ["flip"], "B": ["flip"], "C": ["flip"], "U": []}
    _verified_canary(site)
    assert _summary(_run_one(site, _canary_step(site, restricted=True)))["input_closure"] is None


@needs_userns
def test_reclassifying_an_input_purges_its_whole_public_closure(site: Site, capsys: pytest.CaptureFixture[str]) -> None:
    """Flip ``flip`` to restricted and verify: ``purged=3``; A, B, C and their run links are gone; U stays."""
    _flip_input(site)
    outcomes = site.run(_chain(site))
    assert all(o.ok for o in outcomes.values())
    tree = site.data / "pub" / "flip" / "f.txt"
    st = os.stat(tree)
    os.utime(tree, ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000_000))
    public_line = _verify_line(site, capsys, "flip")
    assert public_line.startswith("flip\tok\t") and "purged" not in public_line and "mtime_only=1" in public_line
    public_records = site.tmp / inputs.VERIFY_DIR
    assert {"flip.json", f"flip{inputs.OVERLAY_SUFFIX}"} <= {p.name for p in public_records.iterdir()}
    _reflag(site, "flip", restricted=True, category="restricted-derivative")
    line = _verify_line(site, capsys, "flip")
    assert line.startswith("flip\tok\t") and line.endswith(" purged=3"), line
    assert not {"flip.json", f"flip{inputs.OVERLAY_SUFFIX}"} & {p.name for p in public_records.iterdir()}
    for name in ("A", "B", "C"):
        out = outcomes[name].out
        assert out is not None and not out.exists()
        assert not (cache.run_dir(site.tmp / "runs", site.config().date, "demo") / name).is_symlink()
    assert outcomes["U"].out is not None and cache.is_done(outcomes["U"].out)
    assert _verify_line(site, capsys, "flip").endswith(" purged=0")


@needs_userns
def test_moving_an_input_under_a_restricted_dir_purges_too(site: Site, capsys: pytest.CaptureFixture[str]) -> None:
    """A tree moved under a restricted dir purges its closure; left unflagged it is ``E_INPUT_REGISTRY`` too."""
    _flip_input(site)
    outcomes = site.run(_chain(site))
    assert all(o.ok for o in outcomes.values())
    (site.data / "pub" / "flip").rename(site.restricted_dir / "flip")
    register_input(site.base, {"data": site.data}, "flip", root="data", relpath="restricted/flip")
    assert _verify_line(site, capsys, "flip") == "flip\tE_INPUT_REGISTRY"
    moved = inputs.read_registry_entry(site.base, "flip")[0]
    now = datetime.datetime.now(datetime.UTC)
    assert inputs.accepted(moved, site.profile(), now=now) == (["E_INPUT_REGISTRY"], None)
    assert inputs.registry_codes(site.base, "flip", site.profile()) == ["E_INPUT_REGISTRY"]
    assert all(not (o.out is not None and o.out.exists()) for n, o in outcomes.items() if n != "U")
    assert outcomes["U"].out is not None and outcomes["U"].out.exists()
    register_input(
        site.base,
        {"data": site.data},
        "flip",
        root="data",
        relpath="restricted/flip",
        restricted=True,
        category="restricted-derivative",
    )
    assert _verify_line(site, capsys, "flip").endswith(" purged=0")


@needs_userns
def test_canary_grep_over_public_trees_is_zero(site: Site, capsys: pytest.CaptureFixture[str]) -> None:
    """After a release, a refused release and a purge, no public tree holds the canary."""
    _verified_canary(site)
    released = _run_one(site, _canary_step(site, restricted=True))
    hit = _run_one(site, _hit_step(site))
    assert released.ok and hit.ok and "W_RELEASE_REFUSED" in hit.codes
    _flip_input(site)
    site.run(_chain(site))
    _reflag(site, "flip", restricted=True, category="restricted-derivative")
    assert _verify_line(site, capsys, "flip").endswith(" purged=3")
    assert _released(site, released.key).is_dir()
    public = [site.tmp / "results", site.tmp / "runs", site.tmp / "slurm", site.data / "reruns"]
    assert [f for top in public for f in _files_with(top, CANARY)] == []


def _key_dir(root: Path, closure: list[str] | None, name: str) -> Path:
    out = root / hashlib.sha256(name.encode()).hexdigest()
    (out / run.WORK).mkdir(parents=True)
    (out / run.WORK / "x.txt").write_text("1\n")
    if closure is not None:
        (out / cache.SUMMARY).write_text(json.dumps({"exit": 0, "input_closure": closure}))
    return out


def test_purge_waits_for_a_claim_and_removes_incomplete_results(site: Site, monkeypatch: pytest.MonkeyPatch) -> None:
    """A result with no summary is purged under its claim: a run in flight is waited for, a dead one removed.

    A held claim does not stop the purge of the other results; the purge then raises ``E_CLAIM_HELD``.
    """
    profile = site.profile()
    root = profile.results_root
    root.mkdir(parents=True)
    other = _key_dir(root, ["other"], "other")
    dead = _key_dir(root, None, "dead")
    flying = _key_dir(root, None, "flying")
    started = threading.Event()

    def in_flight() -> None:
        with cache.claim(root, flying.name):
            started.set()
            time.sleep(1.0)
            (flying / cache.SUMMARY).write_text(json.dumps({"exit": 0, "input_closure": ["flip"]}))

    worker = threading.Thread(target=in_flight)
    worker.start()
    started.wait(10)
    assert restricted.purge_closure(profile, "flip") == 2
    worker.join()
    assert not dead.exists() and not flying.exists() and other.exists()

    held, free = _key_dir(root, ["flip"], "held"), _key_dir(root, ["flip"], "free")
    release = threading.Event()

    def holder() -> None:
        with cache.claim(root, held.name):
            started.set()
            release.wait(10)

    started.clear()
    worker = threading.Thread(target=holder)
    worker.start()
    started.wait(10)
    monkeypatch.setattr(restricted, "PURGE_CLAIM_TIMEOUT_S", 0.5)
    with pytest.raises(cache.CacheError, match="E_CLAIM_HELD"):
        restricted.purge_closure(profile, "flip")
    release.set()
    worker.join()
    assert held.exists() and not free.exists() and other.exists()


def test_a_release_leaves_nothing_stale(site: Site) -> None:
    """A refused release removes an earlier copy of the key; a symlinked release dir is refused untouched."""
    profile = site.profile()
    result = site.tmp / "r"
    result.mkdir()
    (result / run.METRICS).write_text('{"headline": []}\n')
    key = "ab" * 32
    assert restricted.release(result, key, profile, {"exit": 0, "released": True})
    target = site.data / "reruns" / "restricted-metrics" / key
    assert sorted(p.name for p in target.iterdir()) == ["metrics.json", "summary.json"]
    (result / run.METRICS).write_text(f'{{"label": "{CANARY}"}}\n')
    assert not restricted.release(result, key, profile, {"exit": 0, "released": True})
    assert not target.exists()
    elsewhere = site.tmp / "elsewhere"
    elsewhere.mkdir()
    target.symlink_to(elsewhere)
    (result / run.METRICS).write_text('{"headline": []}\n')
    assert not restricted.release(result, key, profile, {"exit": 0, "released": True})
    assert target.is_symlink() and list(elsewhere.iterdir()) == []
    (result / "linked.json").write_text("{}\n")
    target.unlink()
    (result / run.METRICS).unlink()
    (result / run.METRICS).symlink_to("linked.json")
    assert not restricted.release(result, "cd" * 32, profile, {"exit": 0, "released": True})


@needs_userns
def test_a_failed_summary_write_or_a_restricted_closure_withdraws_the_result(
    site: Site, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A release whose summary write fails is withdrawn; a step whose closure turned restricted mid-run fails.

    Lint reports ``scan-hook-missing`` for a step with ``extract`` on a profile with no hook.
    """
    _verified_canary(site)
    real = cache.write_summary
    calls: list[int] = []

    def failing(out: Path, summary: Any) -> Path:  # noqa: ANN401
        calls.append(1)
        if len(calls) == 1:
            raise OSError("disk full")
        return real(out, summary)

    with monkeypatch.context() as patched:
        patched.setattr(cache, "write_summary", failing)
        outcome = _run_one(site, _canary_step(site, restricted=True))
    assert "E_DRIVER" in outcome.codes and not _released(site, outcome.key).exists()

    _flip_input(site)
    outcomes = site.run(_chain(site))
    assert all(o.ok for o in outcomes.values())
    _reflag(site, "flip", restricted=True, category="restricted-derivative")
    late = StepSpec.model_validate(_up_step(site, "D", "C"))
    after = run.run_step(late, "demo", site.config(), {"C": outcomes["C"]}, {})
    assert after.status == "executed" and "E_SPEC_RESTRICTED" in after.codes and not after.ok
    assert after.out is not None and not (after.out / run.WORK).exists()

    (site.base / "adapter_extract.py").write_text("")
    step = site.step(
        "X", "prep", "{root:data}/in/a.txt", adapter="adapter_extract.py", extract=[{"name": "m", "kind": "yesno"}]
    )
    site.restricted_fields = False
    manifest = site.manifest(step)
    profile_file, path = site.files(manifest)
    capsys.readouterr()
    assert cli.main(["lint", str(path), "--profile", str(profile_file), "--base-dir", str(site.base)]) == 1
    assert f"{path}:X: scan-hook-missing" in capsys.readouterr().out


@needs_userns
def test_build_and_env_sandboxes_never_see_restricted_dirs(site: Site, monkeypatch: pytest.MonkeyPatch) -> None:
    """A build-time command reading a file under a restricted dir fails; with no masks set it is refused outright.

    The entry points holding a profile (``build-one``, ``prefetch_env``) set the masks from it.
    """
    secret = site.restricted_dir / "canary" / "in.txt"
    tool = site.tmp / "fake-uv"
    tool.write_text(f"#!/bin/sh\nexec /usr/bin/cat {secret}\n")
    tool.chmod(0o755)
    work = site.tmp / "build-work"
    work.mkdir()

    def build_run(masks: list[Path]) -> subprocess.CompletedProcess[str]:
        with sandbox.tool_masks(masks):
            argv = build.wheel_build_argv(str(tool), Path(sys.executable), work, work, venv=site.cache, writable=[work])
        return subprocess.run(argv, capture_output=True, text=True, check=False, timeout=60)

    site.cache.mkdir(exist_ok=True)
    unmasked = build_run([])
    assert CANARY in unmasked.stdout
    masked = build_run(site.restricted_dirs)
    assert masked.returncode != 0 and CANARY not in masked.stdout + masked.stderr

    monkeypatch.setattr(sandbox, "_TOOL_MASKS", None)
    with pytest.raises(ValueError, match="E_SANDBOX_NO_MASKS"):
        build.wheel_build_argv(str(tool), Path(sys.executable), work, work, venv=site.cache, writable=[work])
    seen: list[Any] = []

    def record(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        seen.append(sandbox.active_tool_masks())
        raise RuntimeError("stop")

    monkeypatch.setattr(env, "_prefetch_env", record)
    with pytest.raises(RuntimeError):
        env.prefetch_env(
            _steps(site.step("P", "prep", "x"))[0].env,
            site.base,
            site.cache,
            analysis="demo",
            wheelhouse=site.cache / "w",
            profile=site.profile(),
        )
    assert seen == [tuple(str(p) for p in site.restricted_dirs)]
    assert sandbox.active_tool_masks() is None
