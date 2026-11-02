"""The license gate: harvest from metadata, SPDX expressions against named sets, bundled libraries, `on_missing`."""

import importlib.util
import sys
from email.message import Message
from pathlib import Path
from typing import Any, Callable

import pytest
import yaml

pytestmark = pytest.mark.logic

_SCRIPT = Path(__file__).resolve().parents[4] / "scripts" / "check_licenses.py"
_spec = importlib.util.spec_from_file_location("check_licenses", _SCRIPT)
assert _spec and _spec.loader
lic = importlib.util.module_from_spec(_spec)
sys.modules["check_licenses"] = lic
_spec.loader.exec_module(lic)

POLICY = lic.load_policy()


class FakeDist:
    """The slice of `importlib.metadata.Distribution` the gate reads."""

    def __init__(
        self,
        name: str,
        fields: dict[str, str | list[str]] | None = None,
        files: list[str] | None = None,
    ) -> None:
        """A distribution called `name` whose metadata holds `fields` (a list is a repeated field)."""
        self.metadata = Message()
        self.metadata["Name"] = name
        for key, value in (fields or {}).items():
            for item in value if isinstance(value, list) else [value]:
                self.metadata[key] = item
        self.version = "1.0"
        self.files = files


def _read(**fields: str | list[str]) -> tuple[str, str] | None:
    """`(expression, read from)` of a distribution with these metadata fields, or None."""
    record = lic.read_license(FakeDist("x", {k.replace("_", "-"): v for k, v in fields.items()}), POLICY)
    return (record.expression, record.how) if record else None


def _dist(**fields: str | list[str]) -> FakeDist:
    return FakeDist("x", {k.replace("_", "-"): v for k, v in fields.items()})


def test_the_policy_file_loads_with_a_set_per_distribution() -> None:
    """The three distributions each allow a union of named sets; the default for a missing record is to raise."""
    assert set(POLICY.distributions) == {"senselab", "senselab-core", "senselab-worker", "env-lock"}
    assert POLICY.on_missing == "raise"
    assert POLICY.distributions["senselab-worker"] < POLICY.distributions["senselab-core"]
    assert POLICY.bundled_allow["senselab-core"] == POLICY.distributions["senselab-core"]
    assert POLICY.bundled_allow["senselab-worker"] > POLICY.distributions["senselab-worker"]


def _broken(tmp_path: Path, name: str, mutate: Callable[[dict[str, Any]], object]) -> Path:
    raw = yaml.safe_load(_SCRIPT.parent.joinpath("data", "license_policy.yaml").read_text())
    mutate(raw)
    path = tmp_path / f"{name}.yaml"
    path.write_text(yaml.safe_dump(raw))
    return path


@pytest.mark.parametrize(
    ("name", "mutate"),
    [
        ("derivation", lambda raw: raw["sets"]["permissive"].pop("derivation")),
        ("unknown_set", lambda raw: raw["distributions"]["senselab"].update(allow=["nonesuch"])),
        ("on_missing", lambda raw: raw.update(on_missing="ignore")),
        ("date", lambda raw: raw["manual"][0]["verified"].pop("date")),
        ("bundled_source", lambda raw: raw["bundled_libraries"]["soundfile"]["sndfile"].pop("source")),
        ("format", lambda raw: raw.update(format="license-policy@0")),
    ],
)
def test_a_policy_with_a_missing_field_is_refused(
    tmp_path: Path, name: str, mutate: Callable[[dict[str, Any]], object]
) -> None:
    """A missing derivation, source or date, an unknown set name, a bad `on_missing` or a wrong format refuses."""
    with pytest.raises(ValueError):
        lic.load_policy(_broken(tmp_path, name, mutate))


@pytest.mark.parametrize(
    ("expression", "allowed", "ok"),
    [
        ("MIT", {"mit"}, True),
        ("MIT", {"isc"}, False),
        ("MIT AND BSD-3-Clause", {"mit"}, False),
        ("MIT AND BSD-3-Clause", {"mit", "bsd-3-clause"}, True),
        ("MIT OR GPL-3.0-only", {"mit"}, True),
        ("(MIT OR GPL-3.0-only) AND ISC", {"mit"}, False),
        ("(MIT OR GPL-3.0-only) AND ISC", {"mit", "isc"}, True),
        ("GPL-3.0-or-later WITH GCC-exception-3.1", {"gpl-3.0-or-later"}, False),
        ("GPL-3.0-or-later WITH GCC-exception-3.1", {"gpl-3.0-or-later with gcc-exception-3.1"}, True),
        ("A OR B AND C", {"a"}, True),
        ("A OR B AND C", {"b"}, False),
    ],
)
def test_an_expression_is_checked_against_the_allowed_ids(expression: str, allowed: set[str], ok: bool) -> None:
    """`AND` needs both sides, `OR` either, `AND` binds tighter, and `WITH` is part of the id."""
    assert lic.allows(lic.parse_expression(expression), frozenset(allowed)) is ok


@pytest.mark.parametrize("bad", ["", "MIT AND", "(MIT", "MIT)", "AND MIT", "MIT WITH", "MIT MIT"])
def test_a_malformed_expression_is_refused(bad: str) -> None:
    """An expression that does not parse raises rather than reading as allowed."""
    with pytest.raises(ValueError):
        lic.parse_expression(bad)


def test_the_license_is_read_in_order_pep639_field_classifier() -> None:
    """PEP 639 beats the field, the field beats the classifier; a long or unknown field falls through."""
    mit = "License :: OSI Approved :: MIT License"
    assert _read(License_Expression="MIT", License="Apache-2.0") == ("MIT", "pep639")
    assert _read(License="BSD 3-Clause License", Classifier=[mit]) == ("BSD-3-Clause", "license-field")
    long = "# LICENSE\n\nThis project is licensed under the terms of..."
    assert _read(License=long, Classifier=["License :: OSI Approved :: ISC License (ISCL)"]) == ("ISC", "classifier")
    assert _read(License="Some Bespoke Terms", Classifier=[mit]) == ("MIT", "classifier")
    assert _read(License="Some Bespoke Terms") is None
    assert _read() is None


def test_an_expression_of_unknown_ids_is_found_not_allowed() -> None:
    """A PEP 639 expression naming an id in no set is a finding, not a pass."""
    closure = {"x": {"senselab-core"}}
    found = lic.check(closure, POLICY, {"x": _dist(License_Expression="Proprietary-EULA")})
    assert [f.problem for f in found] == ["not allowed for senselab-core"]


def test_the_strictest_distribution_that_needs_a_package_decides() -> None:
    """MPL-2.0 passes for core and fails for the worker; a package both need must pass for both."""
    mpl = _dist(License_Expression="MPL-2.0")
    assert lic.check({"x": {"senselab-core"}}, POLICY, {"x": mpl})[0].problem == ""
    assert lic.check({"x": {"senselab-core", "senselab-worker"}}, POLICY, {"x": mpl})[0].problem == (
        "not allowed for senselab-worker"
    )


def test_a_missing_record_raises_by_default_and_warns_when_relaxed() -> None:
    """No record is a failing finding under `raise`; under `warn` it is printed but not a failure."""
    found = lic.check({"x": {"senselab-core"}}, POLICY, {"x": _dist()})
    assert [f.problem for f in found] == ["no license record"]
    assert lic.failures(found, POLICY) == found
    relaxed = POLICY._replace(on_missing="warn")
    assert lic.failures(found, relaxed) == []
    denied = lic.check({"x": {"senselab-core"}}, relaxed, {"x": _dist(License_Expression="GPL-3.0-only")})
    assert lic.failures(denied, relaxed) == denied


def test_a_manual_record_is_the_fallback_and_never_overrides_metadata() -> None:
    """A package not installed uses its manual record; an installed one uses its metadata."""
    closure = {"colorama": {"senselab-core"}}
    assert lic.check(closure, POLICY, {})[0].expression == "BSD-3-Clause"
    installed = {"colorama": _dist(License_Expression="GPL-3.0-only")}
    assert lic.check(closure, POLICY, installed)[0].problem == "not allowed for senselab-core"


@pytest.mark.parametrize(
    ("path", "name"),
    [
        ("_soundfile_data/libsndfile_arm64.dylib", "sndfile"),
        ("_soundfile_data/libsndfile_x86_64.so", "sndfile"),
        ("numpy.libs/libscipy_openblas64_-8fb3d286.so", "scipy_openblas64"),
        ("numpy.libs/libgfortran-040039e1-0352e75f.so.5.0.0", "gfortran"),
        ("numpy.libs/libquadmath-96973f99-934c22de.so.0.0.0", "quadmath"),
        ("cffi.libs/libffi-3a37023a.so.8.1.0", "ffi"),
        ("numpy.libs/libgcc_s-a3a07e5c.so.1", "gcc_s"),
    ],
)
def test_a_bundled_library_is_named_without_prefix_version_hash_or_architecture(path: str, name: str) -> None:
    """The names the policy keys on come from the wheel's file names, on Linux and macOS."""
    assert lic.library_name(path) == name
    package = {"_soundfile_data": "soundfile", "numpy.libs": "numpy", "cffi.libs": "cffi"}[path.partition("/")[0]]
    assert name in POLICY.bundled[package]


def test_bundled_libraries_exclude_python_extension_modules() -> None:
    """A compiled extension is the package's own code; a shared library beside it is bundled."""
    files = [
        "x/_core.cpython-312-darwin.so",
        "x/_ext.abi3.so",
        "x/_w.cp312-win_amd64.pyd",
        "x.libs/libffi-0a1b2c3d.so.8",
    ]
    assert lic.bundled_libraries(FakeDist("x", files=files)) == {"ffi": "x.libs/libffi-0a1b2c3d.so.8"}


def test_a_bundled_library_with_no_record_is_a_finding_and_a_copyleft_one_follows_the_distribution() -> None:
    """An unlisted library fails; libsndfile (LGPL) passes for core and the worker.

    A distribution with no `allow_bundled` is held to its `allow`.
    """
    unlisted = FakeDist("x", {"License-Expression": "MIT"}, ["x.libs/libmystery-0a1b2c3d.so"])
    found = lic.check({"x": {"senselab-core"}}, POLICY, {"x": unlisted})
    assert [f.problem for f in found] == ["", "no license record"]
    sndfile = FakeDist("soundfile", {"License-Expression": "MIT"}, ["_soundfile_data/libsndfile_arm64.dylib"])
    assert [f.problem for f in lic.check({"soundfile": {"senselab-core"}}, POLICY, {"soundfile": sndfile})] == ["", ""]
    assert lic.check({"soundfile": {"senselab-worker"}}, POLICY, {"soundfile": sndfile})[1].problem == ""
    strict = POLICY._replace(
        bundled_allow={**POLICY.bundled_allow, "senselab-worker": POLICY.distributions["senselab-worker"]}
    )
    found = lic.check({"soundfile": {"senselab-worker"}}, strict, {"soundfile": sndfile})
    assert found[1].problem == "not allowed for senselab-worker"


_DECLARED = [(package, lib) for package, libraries in sorted(POLICY.bundled.items()) for lib in sorted(libraries)]


@pytest.mark.parametrize(("package", "lib"), _DECLARED)
@pytest.mark.parametrize("impostor", ["x", "mystery-wheel"])
def test_a_bundled_record_holds_only_for_the_package_that_declares_it(package: str, lib: str, impostor: str) -> None:
    """Each declared library passes under its own package and is a missing record under any other, whatever its name."""
    file = f"{package}.libs/lib{lib}-0a1b2c3d.so.1"
    for owner, expected in [(package, ""), (impostor, "no license record")]:
        dist = FakeDist(owner, {"License-Expression": "MIT"}, [file.replace(package, owner, 1)])
        found = lic.check({owner: {"senselab-core"}}, POLICY, {owner: dist})
        assert [f.problem for f in found] == ["", expected], (owner, found)


def test_lgpl_fails_as_a_package_under_the_worker_and_passes_as_a_bundled_library() -> None:
    """The worker's bundled allowance never widens what a Python package may be."""
    lgpl_package = _dist(License_Expression="LGPL-3.0-or-later")
    found = lic.check({"x": {"senselab-worker"}}, POLICY, {"x": lgpl_package})
    assert [f.problem for f in found] == ["not allowed for senselab-worker"]
    bundles = FakeDist("soundfile", {"License-Expression": "MIT"}, ["_soundfile_data/libsndfile_arm64.dylib"])
    assert [f.problem for f in lic.check({"soundfile": {"senselab-worker"}}, POLICY, {"soundfile": bundles})] == [
        "",
        "",
    ]


def test_the_locked_closure_is_transitive_and_records_who_needs_each_package(tmp_path: Path) -> None:
    """A dependency of a dependency is in the closure, tagged with every root that reaches it."""
    lock = tmp_path / "uv.lock"
    lock.write_text(
        '[[package]]\nname = "senselab"\ndependencies = [{ name = "senselab-core" }]\n'
        '[[package]]\nname = "senselab-core"\ndependencies = [{ name = "a" }, { name = "senselab-worker" }]\n'
        '[[package]]\nname = "senselab-worker"\ndependencies = [{ name = "c" }]\n'
        '[[package]]\nname = "a"\ndependencies = [{ name = "b" }]\n'
        '[[package]]\nname = "b"\n[[package]]\nname = "c"\n[[package]]\nname = "dev-only"\n'
    )
    closure = lic.locked_closure(lock)
    assert "dev-only" not in closure
    assert closure["b"] == {"senselab", "senselab-core"}
    assert closure["c"] == {"senselab", "senselab-core", "senselab-worker"}


_LOCK = """\
# senselab env lock: toy
alpha==1.0 \\
    --hash=sha256:aa
    # via -r requirements.in
beta==2.1 ; sys_platform == 'win32' \\
    --hash=sha256:bb
Gamma_Pkg==0.3
"""


def test_lock_pins_are_read_with_names_normalised_and_markers_dropped(tmp_path: Path) -> None:
    """Each ``name==version`` line is a pin; comments, hashes and markers are not."""
    path = tmp_path / "toy.txt"
    path.write_text(_LOCK)
    assert lic.lock_pins(path.read_text()) == {"alpha": "1.0", "beta": "2.1", "gamma-pkg": "0.3"}


def test_a_lock_is_read_from_the_installed_metadata_when_the_version_matches_else_fetched() -> None:
    """A matching installed distribution is used; another version or none is fetched; no release is a gap."""
    installed = {"alpha": FakeDist("alpha", {"License-Expression": "MIT"}), "beta": FakeDist("beta")}
    fetched: list[tuple[str, str]] = []

    def fetch(name: str, version: str) -> FakeDist | None:
        fetched.append((name, version))
        return None if name == "gamma-pkg" else FakeDist(name, {"License-Expression": "ISC"})

    pins = {"alpha": "1.0", "beta": "2.1", "gamma-pkg": "0.3"}
    dists = lic.pinned_dists(pins, installed, fetch)
    assert fetched == [("beta", "2.1"), ("gamma-pkg", "0.3")]
    assert sorted(dists) == ["alpha", "beta"]
    found = lic.check({n: {lic.INSTALLER} for n in pins}, POLICY, dists)
    assert [(f.subject, f.expression, f.problem) for f in found] == [
        ("alpha", "MIT", ""),
        ("beta", "ISC", ""),
        ("gamma-pkg", "", "no license record"),
    ]


def test_a_pypi_release_reads_like_installed_metadata() -> None:
    """The release JSON's expression, field and classifiers give the same record the wheel's METADATA would."""
    info: dict[str, Any] = {"name": "x", "version": "1", "license_expression": None, "license": "BSD 3-Clause License"}
    assert lic.read_license(lic.PypiDist(info), POLICY).how == "license-field"
    info = {"name": "x", "version": "1", "license_expression": "MIT", "license": "other", "classifiers": []}
    assert lic.read_license(lic.PypiDist(info), POLICY) == lic.Record("MIT", "pep639", "License-Expression")


def test_env_records_keep_the_package_findings_that_hold_a_record_with_how_it_was_read() -> None:
    """A PEP 639 read stays `pep639`, any other read is `manual`; a bundled library or a gap has no entry."""
    dists = {
        "a": FakeDist("a", {"License-Expression": "MIT"}),
        "b": FakeDist("b", {"License": "ISC License"}),
        "c": FakeDist("c"),
        "colorama": FakeDist("colorama"),
    }
    found = lic.check({n: {lic.INSTALLER} for n in dists}, POLICY, dists)
    records = lic.env_records(found, "2026-11-02")
    assert sorted(records) == ["a", "b", "colorama"]
    assert records["a"]["verified"] == {"date": "2026-11-02", "how": "pep639"}
    assert records["b"]["verified"] == {"date": "2026-11-02", "how": "manual"}
    assert records["colorama"]["verified"] == {"date": "2026-11-01", "how": "manual"}


def test_env_records_replace_one_env_and_keep_the_others(tmp_path: Path) -> None:
    """Writing an env replaces its entries and leaves the other envs and the format tag."""
    path = tmp_path / "env_licenses.yaml"
    lic.write_env_records(path, "one", {"a": {"expression": "MIT"}})
    lic.write_env_records(path, "two", {"b": {"expression": "ISC"}})
    lic.write_env_records(path, "one", {"c": {"expression": "MIT"}})
    raw = yaml.safe_load(path.read_text())
    assert raw["format"] == lic.ENV_LICENSES_FORMAT
    assert {k: sorted(v) for k, v in raw["envs"].items()} == {"one": ["c"], "two": ["b"]}


@pytest.mark.parametrize("expression", ["LicenseRef-NVIDIA-Proprietary", "MIT-CMU", "CNRI-Python"])
def test_the_env_lock_policy_allows_what_a_shipped_distribution_does_not(expression: str) -> None:
    """Vendor and notice-only licenses of backend env packages pass for `env-lock` and fail for every shipped one."""
    dist = _dist(License_Expression=expression)
    assert lic.check({"x": {"env-lock"}}, POLICY, {"x": dist})[0].problem == ""
    for shipped in lic.DISTRIBUTIONS:
        assert lic.check({"x": {shipped}}, POLICY, {"x": dist})[0].problem == f"not allowed for {shipped}"
