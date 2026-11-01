"""The packaged catalogue: it loads, every Phase 2 model is pinned, and every license derives from its expression."""

from importlib import resources

import pytest
from senselab_core.runtime.catalogue import Catalogue

PHASE_2_MODELS = (
    "openai/whisper-large-v3-turbo",
    "nyralabs/CrisperWhisper2.0_turbo",
    "pyannote/speaker-diarization-community-1",
    "pyannote/segmentation-3.0",
    "speechbrain/spkrec-ecapa-voxceleb",
    "google/gemma-4-31B-it-qat-w4a16-ct",
)


def _packaged() -> Catalogue:
    return Catalogue.load(resources.files("senselab").joinpath("data", "catalogue.yaml").read_text("utf-8"))


@pytest.mark.contract
def test_packaged_catalogue_loads() -> None:
    """Every Phase 2 model is an entry pinned by identity, with a default per task and an allowlist of its hubs."""
    catalogue = _packaged()
    assert {e.id for e in catalogue.entries} == set(PHASE_2_MODELS)
    for task in ("speech_to_text", "diarize", "speaker_embeddings", "pii_review"):
        assert catalogue.default(task).id in PHASE_2_MODELS
    assert catalogue.entry("openai/whisper-large-v3-turbo").license.commercial_use == "yes"
    assert catalogue.entry("nyralabs/CrisperWhisper2.0_turbo").license.commercial_use == "no"
    assert catalogue.entry("pyannote/segmentation-3.0").license.gated
    assert catalogue.sources() == ("google", "nyralabs", "openai", "pyannote", "speechbrain")
