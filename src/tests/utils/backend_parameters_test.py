"""The parameter pathway: declared from a signature, unknown keys refused, effective set recorded."""

from __future__ import annotations

from typing import Optional

import pytest

from senselab.utils.backend_parameters import (
    PARAMETER_RECORD_KEY,
    ParameterRecord,
    declared_parameters,
    record_parameters_on,
    resolve_backend_parameters,
)


def _backend(
    audios: list,
    model: object = None,
    device: object = None,
    variant: str = "default",
    seed: int = 0,
    timeout_s: Optional[float] = None,
) -> list:
    """Stand-in backend with two tunables, a device, and a required-positional first argument."""
    return audios


def _no_tunables(audios: list, model: object = None, device: object = None) -> list:
    return audios


def _kwargs_backend(audios: list, **kwargs: object) -> list:
    return audios


class _WithMetadata:
    def __init__(self) -> None:
        self.metadata: dict = {}


def test_declared_parameters_reads_the_signature_and_drops_dispatcher_arguments() -> None:
    """The declared set comes from the callable, so it cannot drift from the implementation."""
    assert declared_parameters(_backend) == {"variant": "default", "seed": 0, "timeout_s": None}
    assert declared_parameters(_no_tunables) == {}


def test_a_kwargs_backend_declares_nothing_rather_than_everything() -> None:
    """``**kwargs`` is not "anything goes": that would restore the silent-typo failure."""
    assert declared_parameters(_kwargs_backend) == {}
    with pytest.raises(ValueError, match="no tunable parameters"):
        resolve_backend_parameters(_kwargs_backend, {"variant": "x"}, backend_name="kw")


def test_a_declared_key_is_forwarded_and_only_that_key() -> None:
    """Only the caller's explicit keys travel, so a backend default stays the backend's to change."""
    kwargs, record = resolve_backend_parameters(_backend, {"variant": "other"}, backend_name="stub")
    assert kwargs == {"variant": "other"}
    assert record.explicit == ("variant",)


def test_an_unknown_key_raises_and_names_the_near_miss() -> None:
    """The failure mode is a typo, so the message must name the parameter the caller meant."""
    with pytest.raises(ValueError) as exc:
        resolve_backend_parameters(_backend, {"variantt": "other"}, backend_name="stub")
    message = str(exc.value)
    assert "'variantt'" in message
    assert "did you mean 'variant'" in message
    assert "seed" in message, "the message must enumerate the declared parameters"


def test_a_key_the_wrong_backend_declares_raises() -> None:
    """Validation is against the *selected* backend; a DriftSE key with SpeechBrain must not pass."""
    with pytest.raises(ValueError, match="no tunable parameters"):
        resolve_backend_parameters(_no_tunables, {"variant": "x"}, backend_name="other")


def test_a_dispatcher_argument_cannot_be_smuggled_through_the_mapping() -> None:
    """``device`` has one source. Two would mean the loser is silently discarded."""
    with pytest.raises(ValueError, match="argument of the task function itself"):
        resolve_backend_parameters(_backend, {"device": "cuda"}, backend_name="stub")


def test_a_non_mapping_or_non_string_key_is_a_type_error() -> None:
    """A list of pairs would otherwise unpack into something plausible."""
    with pytest.raises(TypeError, match="must be a mapping"):
        resolve_backend_parameters(_backend, [("variant", "x")], backend_name="stub")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="keyed by parameter name"):
        resolve_backend_parameters(_backend, {1: "x"}, backend_name="stub")  # type: ignore[dict-item]


def test_the_record_carries_defaults_as_well_as_explicit_values() -> None:
    """Recording only the explicit values cannot answer what actually ran."""
    _, record = resolve_backend_parameters(_backend, {"seed": 7}, backend_name="stub")
    assert record.effective == {"variant": "default", "seed": 7, "timeout_s": None}
    assert record.explicit == ("seed",)


def test_none_means_backend_defaults_and_still_produces_a_record() -> None:
    """A caller who passes nothing still gets provenance for what the defaults were."""
    kwargs, record = resolve_backend_parameters(_backend, None, backend_name="stub")
    assert kwargs == {}
    assert record.effective["variant"] == "default"
    assert record.explicit == ()


def test_unrepresentable_values_are_rendered_rather_than_dropped() -> None:
    """A record has to be serialisable, but it must not silently omit a parameter."""
    _, record = resolve_backend_parameters(_backend, {"variant": object()}, backend_name="stub")
    assert isinstance(record.effective["variant"], str)
    assert "object object at" in record.effective["variant"]


def test_the_record_lands_on_every_returned_object_including_nested_lists() -> None:
    """A separator returns sources per input; each one must carry the record."""
    record = ParameterRecord(backend="stub", effective={"variant": "x"}, explicit=("variant",))
    nested = [[_WithMetadata(), _WithMetadata()], [_WithMetadata()]]
    record_parameters_on(nested, record)
    for group in nested:
        for item in group:
            assert item.metadata[PARAMETER_RECORD_KEY] == {
                "backend": "stub",
                "parameters": {"variant": "x"},
                "explicit": ["variant"],
            }


def test_stamping_an_object_without_metadata_is_a_no_op() -> None:
    """Duck typing must not turn a plain return value into an AttributeError."""
    record = ParameterRecord(backend="stub")
    record_parameters_on([1, "two", None], record)  # must not raise


def test_a_required_parameter_with_no_default_must_be_supplied() -> None:
    """A backend can declare something mandatory; the dispatcher must not invent a value."""

    def needs_one(audios: list, model: object = None, device: object = None, *, required: int) -> list:
        return audios

    with pytest.raises(ValueError, match="requires parameter"):
        resolve_backend_parameters(needs_one, None, backend_name="stub")
    kwargs, _ = resolve_backend_parameters(needs_one, {"required": 3}, backend_name="stub")
    assert kwargs == {"required": 3}
