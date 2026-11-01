"""License record (P-23)."""

import pickle

import pytest
from pydantic import ValidationError
from senselab_core.kernel.license import License, MissingLicense, require_license


def _record(**overrides: object) -> License:
    fields: dict[str, object] = {
        "expression": "CC-BY-NC-4.0",
        "commercial_use": "no",
        "copyleft": "none",
        "attribution": True,
        "gated": False,
        "source": "openai/whisper-large-v3-turbo README.md at 41f01f3fe87f28c78e2fbf8b568835947dd65ed9",
        "verified": {"date": "2026-10-30", "how": "hf-card"},
    }
    fields.update(overrides)
    return License.model_validate(fields)


@pytest.mark.logic
def test_record_holds_the_design_fields() -> None:
    """The record carries the SPDX expression, derived flags, source and verification."""
    lic = _record()
    assert (lic.expression, lic.commercial_use, lic.copyleft, lic.attribution, lic.gated) == (
        "CC-BY-NC-4.0",
        "no",
        "none",
        True,
        False,
    )
    assert dict(lic.verified) == {"date": "2026-10-30", "how": "hf-card"}


@pytest.mark.logic
@pytest.mark.parametrize(
    "field,value",
    [
        ("commercial_use", "maybe"),
        ("copyleft", "viral"),
        ("expression", ""),
        ("source", ""),
        ("attribution", "yes"),
        ("gated", 1),
        ("verified", {"how": "manual"}),
        ("verified", {"date": "2026-10-30"}),
        ("verified", {"date": "30/10/2026", "how": "manual"}),
        ("verified", {"date": "2026-10-30", "how": ""}),
    ],
)
def test_record_refuses_values_outside_the_schema(field: str, value: object) -> None:
    """Flags take only their listed values, booleans are strict, and `verified` names a date and a method."""
    with pytest.raises(ValidationError):
        _record(**{field: value})


@pytest.mark.logic
def test_record_is_immutable_and_round_trips() -> None:
    """Fields and `verified` cannot be changed; the record pickles and survives a JSON round trip."""
    lic = _record()
    with pytest.raises(ValidationError):
        lic.gated = True  # type: ignore[misc]
    with pytest.raises(TypeError):
        lic.verified["how"] = "manual"  # type: ignore[index]
    assert pickle.loads(pickle.dumps(lic)) == lic
    assert License.model_validate_json(lic.model_dump_json()) == lic


@pytest.mark.logic
def test_missing_license_raises() -> None:
    """A missing record raises by default, naming its subject; a present one is returned."""
    lic = _record()
    assert require_license(lic, "openai/whisper") is lic
    with pytest.raises(MissingLicense, match="openai/whisper"):
        require_license(None, "openai/whisper")
