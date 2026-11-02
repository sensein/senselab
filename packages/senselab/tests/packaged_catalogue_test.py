"""The packaged catalogue: it loads, every Phase 2 model is pinned, and every license derives from its expression."""

from importlib import resources

import pytest
from senselab._api import generated  # noqa: F401 - importing it declares every backend
from senselab_core.kernel.declare import REGISTRY
from senselab_core.runtime.catalogue import Catalogue

PHASE_2_MODELS = (
    "openai/whisper-large-v3-turbo",
    "nyralabs/CrisperWhisper2.0_turbo",
    "pyannote/speaker-diarization-community-1",
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
    for task in ("speech_to_text", "diarize", "speaker_embeddings", "pii_spans"):
        assert catalogue.default(task).id in PHASE_2_MODELS
    assert catalogue.entry("openai/whisper-large-v3-turbo").license.commercial_use == "yes"
    assert catalogue.entry("nyralabs/CrisperWhisper2.0_turbo").license.commercial_use == "no"
    assert catalogue.entry("pyannote/speaker-diarization-community-1").license.gated
    assert catalogue.sources() == ("google", "nyralabs", "openai", "pyannote", "speechbrain")


@pytest.mark.contract
def test_every_declared_backend_of_an_entry_serves_it() -> None:
    """A catalogue entry names no backend that declares models and omits the entry."""
    for entry in _packaged().entries:
        for task in entry.tasks:
            declared = (
                {b.name: b for b in REGISTRY.backends(task)} if task in {t.name for t in REGISTRY.tasks()} else {}
            )
            for name in entry.backends:
                if name in declared and declared[name].models:
                    assert entry.id in declared[name].models, (entry.id, name)
