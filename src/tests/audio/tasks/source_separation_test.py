"""unasdiff source separation — API contract and class-space handling."""

import pytest

from senselab.audio.tasks.source_separation import unasdiff
from senselab.audio.tasks.source_separation.api import resolve_source_classes


def test_class_map_has_41_classes_in_50_slots() -> None:
    """The prior's 50-wide embedding has only 41 trained rows.

    Passing an index in 41..49 would condition on an untrained embedding row and
    produce plausible-looking noise rather than an error.
    """
    doc = unasdiff.load_fsd_class_map_document()
    assert len(doc["classes"]) == 41
    assert max(doc["classes"].values()) == 40
    assert doc["num_embedding_slots"] == 50


def test_resolve_source_classes_maps_names_to_indices() -> None:
    """Names resolve to the same indices the raw class map carries."""
    assert resolve_source_classes(["Applause", "Cello"]) == [
        unasdiff.load_fsd_class_map_document()["classes"]["Applause"],
        unasdiff.load_fsd_class_map_document()["classes"]["Cello"],
    ]


def test_an_unknown_class_raises_and_names_the_valid_options() -> None:
    """An unmapped name must raise, not fall back to a class index.

    Silently falling back to index 0 would condition the prior on 'Hi-hat' while
    reporting the caller's own label — separation would be wrong and the output
    would claim otherwise.
    """
    with pytest.raises(ValueError) as exc:
        resolve_source_classes(["Helicopter"])
    assert "Helicopter" in str(exc.value)
    assert "Applause" in str(exc.value), "the error must enumerate the valid classes"


def test_upstream_is_pinned_to_a_full_commit_sha() -> None:
    """The upstream clone target is a 40-hex commit, never a mutable ref."""
    assert len(unasdiff._UNASDIFF_COMMIT) == 40
    assert all(c in "0123456789abcdef" for c in unasdiff._UNASDIFF_COMMIT)


def test_flash_attn_is_not_required() -> None:
    """flash-attn is absent from the venv's pinned requirements.

    atten_unet.py sets use_flash=False on ImportError and branches to a manual
    softmax attention, so the venv can omit a package that is slow and fragile to
    build. Verified against upstream, not assumed.
    """
    named = {r.split(">=")[0].split("==")[0].strip().lower() for r in unasdiff._UNASDIFF_REQUIREMENTS}
    assert "flash-attn" not in named and "flash_attn" not in named


def test_torch_is_pinned_for_cuda_routing() -> None:
    """Torch and torchaudio are named explicitly so ensure_venv's CUDA routing fires."""
    named = {r.split(">=")[0].split("==")[0].strip().lower() for r in unasdiff._UNASDIFF_REQUIREMENTS}
    assert "torch" in named and "torchaudio" in named
