"""The runtime catalogue and the SPDX-derived license records, over a toy catalogue."""

from pathlib import Path

import pytest
from pydantic import ValidationError
from senselab_core.kernel.license import License, MissingLicense
from senselab_core.runtime.catalogue import Catalogue, CatalogueFormatRetired, UnknownModel
from senselab_core.runtime.fake_backend import FAKE
from senselab_core.runtime.licenses import (
    LicensePolicy,
    LicenseRefused,
    allows,
    backend_license,
    derive_flags,
    ids_of,
    license_record,
    parse_expression,
)

SHA = "a" * 40
DIGEST = "b" * 64


def _entry(id: str = "org/m", revision: str = SHA, license: str = "{expression: MIT}", extra: str = "") -> str:
    return f"""  - id: {id}
    revision: {revision}
    tasks: [t]
    backends: [b]
    stacks: [s]
    license:
      source: card
      verified: {{date: 2026-11-01, how: hf-card}}
      gated: false
      {license[1:-1].replace(", ", chr(10) + "      ")}
    {extra}
"""


def _text(*entries: str) -> str:
    return "format: catalogue@2.0\nentries:\n" + "".join(entries)


@pytest.mark.logic
@pytest.mark.parametrize(
    "expression,flags",
    [
        ("MIT", ("yes", "none", True)),
        ("CC-BY-NC-4.0", ("no", "none", True)),
        ("GPL-3.0-only", ("yes", "strong", True)),
        ("AGPL-3.0-or-later", ("yes", "network", True)),
        ("MIT AND CC-BY-NC-4.0", ("no", "none", True)),
        ("MIT OR GPL-3.0-only", ("yes", "none", True)),
        ("CC0-1.0 OR MIT", ("yes", "none", False)),
        ("(MIT AND MPL-2.0) OR GPL-2.0-only", ("yes", "weak", True)),
        ("GPL-3.0-or-later WITH GCC-exception-3.1", ("yes", "none", True)),
    ],
)
def test_flags_derived_from_spdx(expression: str, flags: tuple[str, str, bool]) -> None:
    """Flags follow the parsed expression: AND takes the stricter side, OR the more permissive."""
    assert tuple(derive_flags(expression)) == flags
    assert (
        lic := license_record(expression, source="s", verified={"date": "2026-11-01", "how": "manual"}, gated=False)
    )
    assert (lic.commercial_use, lic.copyleft, lic.attribution) == flags


@pytest.mark.logic
@pytest.mark.parametrize(
    "bad", ["mit OR ((", "", "MIT AND", "(MIT", "MIT MIT", "Not-A-License", "MIT OR Not-A-License"]
)
def test_malformed_expression_refused(bad: str) -> None:
    """An unparseable expression and an id outside the vendored table are refused."""
    with pytest.raises(ValueError):
        derive_flags(bad)


@pytest.mark.logic
def test_a_free_flag_that_contradicts_the_expression_is_refused() -> None:
    """A flag written beside the expression must equal the derived one; `how` is a closed set."""
    verified = {"date": "2026-11-01", "how": "hf-card"}
    for flags in ({"copyleft": "none"}, {"commercial_use": "yes"}, {"attribution": False}):
        expression = {"copyleft": "GPL-3.0-only", "commercial_use": "CC-BY-NC-4.0", "attribution": "MIT"}[
            next(iter(flags))
        ]
        with pytest.raises(ValueError, match="contradict"):
            license_record(expression, source="s", verified=verified, gated=False, flags=flags)
    license_record("MIT", source="s", verified=verified, gated=False, flags={"copyleft": "none"})
    with pytest.raises(ValidationError):
        license_record("MIT", source="s", verified={"date": "2026-11-01", "how": "guess"}, gated=False)


@pytest.mark.logic
def test_expression_helpers() -> None:
    """The moved parser reads ids and evaluates AND/OR against an allowed set."""
    tree = parse_expression("MIT OR (GPL-2.0-only AND Apache-2.0)")
    assert ids_of(tree) == ["MIT", "GPL-2.0-only", "Apache-2.0"]
    assert allows(tree, frozenset({"mit"})) and not allows(tree, frozenset({"apache-2.0"}))


@pytest.mark.logic
@pytest.mark.parametrize("revision", ["main", "v1.0", "A" * 40, "a" * 39, "a" * 41, "a" * 63])
def test_catalogue_refuses_ref_revision(revision: str) -> None:
    """A revision is a 40-hex commit or a 64-hex weights digest; a ref or a malformed hash is refused."""
    with pytest.raises(ValidationError, match="never a ref"):
        Catalogue.load(_text(_entry(revision=revision)))
    assert Catalogue.load(_text(_entry(revision=DIGEST))).entry("org/m").revision == DIGEST


@pytest.mark.logic
def test_catalogue_lookup_defaults_and_sources(tmp_path: Path) -> None:
    """`entry`, `default` and `sources` over a toy catalogue read from text and from a path."""
    text = _text(
        _entry("org/a", extra="default_for: [t]"),
        _entry("other/b", revision=DIGEST),
        _entry("local-name", license="{expression: Apache-2.0}"),
    )
    path = tmp_path / "c.yaml"
    path.write_text(text)
    for catalogue in (Catalogue.load(text), Catalogue.load(path)):
        assert catalogue.default("t").id == "org/a"
        assert catalogue.entry("other/b").revision == DIGEST
        assert catalogue.sources() == ("org", "other")
        with pytest.raises(UnknownModel):
            catalogue.entry("org/none")
        with pytest.raises(UnknownModel):
            catalogue.default("nothing")


@pytest.mark.logic
@pytest.mark.parametrize(
    "text",
    [
        _text(_entry(), _entry()),
        _text(_entry("a/x", extra="default_for: [t]"), _entry("a/y", extra="default_for: [t]")),
        _text(_entry(extra="default_for: [other]")),
        _text(_entry(license="{expression: GPL-3.0-only, copyleft: none}")),
        _text(_entry(extra="resources: {a100: {}}")),
        "format: catalogue@3.0\nentries: []\n",
        "format: catalogue@2.0\nentries:\n  - id: org/m\n",
    ],
    ids=["repeat", "two-defaults", "default-not-in-tasks", "contradicting-flag", "resources", "major", "incomplete"],
)
def test_catalogue_refuses_bad_files(text: str) -> None:
    """Repeated ids, two defaults, a default outside the tasks, a contradicting flag, inline resources, a major."""
    with pytest.raises((ValidationError, ValueError)):
        Catalogue.load(text)


def _lic(expression: str, how: str = "pep639", date: str = "2026-11-01", gated: bool = False) -> License:
    return license_record(expression, source="lock", verified={"date": date, "how": how}, gated=gated)


@pytest.mark.logic
def test_backend_license_from_lock_records() -> None:
    """The code license is the AND of the lock's packages, flags derived, dated by the latest record."""
    got = backend_license(
        FAKE,
        {"a": _lic("MIT", date="2026-10-01"), "b": _lic("MPL-2.0", date="2026-11-01"), "c": _lic("MIT", gated=True)},
    )
    assert got.expression == "(MIT) AND (MPL-2.0)"
    assert (got.commercial_use, got.copyleft, got.gated) == ("yes", "weak", True)
    assert dict(got.verified) == {"date": "2026-11-01", "how": "pep639"}
    assert backend_license(FAKE, {"a": _lic("MIT"), "b": _lic("MIT", how="manual")}).verified["how"] == "manual"
    assert backend_license(FAKE, {"a": _lic("MIT")}).expression == "MIT"


@pytest.mark.logic
def test_a_license_policy_is_the_users_allowed_and_refused_ids() -> None:
    """A policy accepts what its allowed, unrefused ids satisfy (case ignored) and refuses a missing record."""
    either, both = _lic("MIT OR GPL-3.0-only"), _lic("MIT AND MPL-2.0")
    refuse_gpl, only_mit = LicensePolicy(refuse=frozenset({"gpl-3.0-only"})), LicensePolicy(allow=frozenset({"MIT"}))
    assert refuse_gpl.refusal("x", either) is None and refuse_gpl.refusal("x", both) is None
    assert only_mit.refusal("x", either) is None
    assert "MIT AND MPL-2.0" in str(only_mit.refusal("x", both))
    with pytest.raises(LicenseRefused, match="x is licensed 'MIT OR GPL-3.0-only'"):
        LicensePolicy(refuse=frozenset({"MIT", "GPL-3.0-only"})).check("x", either)
    assert LicensePolicy().refusal("x", None) == "x has no license record"


@pytest.mark.logic
def test_backend_license_refuses_missing_and_empty_records() -> None:
    """A package without a record raises `MissingLicense`; no packages at all raise `ValueError`."""
    with pytest.raises(MissingLicense, match="package b"):
        backend_license(FAKE, {"a": _lic("MIT"), "b": None})
    with pytest.raises(ValueError, match="no lock records"):
        backend_license(FAKE, {})


@pytest.mark.logic
def test_backend_entry_is_the_first_model_else_the_task_default() -> None:
    """A backend runs its first declared model, else its task's catalogue default, else no model."""
    catalogue = Catalogue.load(_text(_entry("org/declared"), _entry("org/default", extra="default_for: [t]")))
    backend = FAKE.model_copy(update={"task": "t"})
    declared = backend.model_copy(update={"models": ("org/declared",)})
    assert catalogue.backend_entry(declared) == catalogue.entry("org/declared")
    assert catalogue.backend_entry(backend) == catalogue.entry("org/default")
    assert Catalogue.load(_text(_entry())).backend_entry(backend) is None
    with pytest.raises(UnknownModel):
        Catalogue.load(_text(_entry())).backend_entry(backend.model_copy(update={"models": ("org/absent",)}))


@pytest.mark.logic
def test_a_catalogue_file_holds_no_resource_tables() -> None:
    """Measures are not the catalogue's: a file that names `resource_tables` is refused, not read."""
    with pytest.raises(ValueError, match="resource_tables"):
        Catalogue.load(_text(_entry()) + "resource_tables: []\n")


@pytest.mark.logic
def test_catalogue1_refused_as_retired() -> None:
    """A `catalogue@1.0` document is refused as retired, naming the rewrite, never read as corrupt."""
    with pytest.raises(CatalogueFormatRetired, match="catalogue@2.0"):
        Catalogue.load(_text(_entry(extra="resources: {}")).replace("catalogue@2.0", "catalogue@1.0"))
