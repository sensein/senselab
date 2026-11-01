"""Tests for HF models and functions."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest
from huggingface_hub import HfApi

from senselab.utils.data_structures import HFModel, check_hf_repo_exists
from senselab.utils.data_structures.model import get_huggingface_token


@pytest.mark.skip(reason="torchaudio is a core dependency and always installed; missing-dep path cannot be tested")
def test_check_torchaudio_model_init() -> None:
    """Test torchaudio model initialization."""
    with pytest.raises(ModuleNotFoundError):
        from senselab.utils.data_structures.model import TorchAudioModel

        TorchAudioModel(path_or_uri="torchaudio_model", revision="main")


def test_check_hf_repo_exists_true() -> None:
    """Test HF repo exists."""
    with patch("senselab.utils.dependencies.ensure_hf_model", return_value="abc123"):
        assert check_hf_repo_exists("valid_repo") is True


def test_check_hf_repo_exists_false() -> None:
    """A definitive not-found error means the repo does not exist (False).

    A generic/transient error (e.g. a 429) is deliberately NOT reported as
    missing — check_hf_repo_exists re-raises it — so this uses the concrete
    RepositoryNotFoundError that signals a genuinely absent repo.
    """
    from huggingface_hub.errors import RepositoryNotFoundError

    not_found = RepositoryNotFoundError("not found", response=MagicMock(status_code=404, headers={}))
    with patch("senselab.utils.dependencies.ensure_hf_model", side_effect=not_found):
        assert check_hf_repo_exists("invalid_repo") is False


def test_hfmodel_valid_hf_repo_check(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test valid HFModel repo check."""
    sha = "a" * 40
    monkeypatch.setattr("senselab.utils.data_structures.model.check_hf_repo_exists", lambda **kw: True)
    monkeypatch.setattr("senselab.utils.model_revision.resolve_revision", lambda *a, **k: sha)
    model: HFModel = HFModel(path_or_uri="valid_repo")
    assert model.revision == "main"
    assert model.commit_sha == sha


def test_hf_model_records_the_resolved_commit_sha(monkeypatch: pytest.MonkeyPatch) -> None:
    """Revision keeps the ref asked for; commit_sha carries what it resolved to."""
    sha = "c" * 40
    monkeypatch.setattr("senselab.utils.data_structures.model.check_hf_repo_exists", lambda **kw: True)
    monkeypatch.setattr("senselab.utils.model_revision.resolve_revision", lambda *a, **k: sha)

    model: HFModel = HFModel(path_or_uri="org/model", revision="main")
    assert model.revision == "main", "the requested ref must survive"
    assert model.commit_sha == sha, "the resolved commit must be recorded"


def test_hfmodel_invalid_hf_repo_check() -> None:
    """Test invalid HFModel repo check."""
    with patch("senselab.utils.data_structures.model.check_hf_repo_exists", return_value=False):
        with pytest.raises(ValueError):
            HFModel(path_or_uri="invalid/repo")


def test_hfmodel_wraps_a_gated_repo_error_as_validationerror(monkeypatch: pytest.MonkeyPatch) -> None:
    """A Hub error that ``check_hf_repo_exists`` re-raises must also arrive as ValidationError.

    ``check_hf_repo_exists`` answers False only for a genuinely absent repo; ``GatedRepoError`` and
    transient errors it re-raises on purpose. Both are ``OSError`` subclasses, so before this was
    wrapped they escaped the ``revision`` *field* validator unwrapped -- the same defect finding #5
    fixed one validator further down, on a path that runs first.

    The message must name the original type, because the wrap is what destroys it: pydantic gives
    the ValidationError no ``__cause__``, so "gated" and "Hub outage" are otherwise
    indistinguishable to a caller.
    """
    from huggingface_hub.errors import GatedRepoError
    from pydantic import ValidationError

    # A fresh cache, not `.clear()`: `_hf_cache` is a ClassVar shared by every test in this
    # process, and an earlier test constructing the same repo id leaves a True in it that would
    # make this validator skip the call under test entirely.
    monkeypatch.setattr(HFModel, "_hf_cache", {})

    def _gated(**_kw: object) -> bool:
        raise GatedRepoError(
            "403 gated",
            response=httpx.Response(403, request=httpx.Request("GET", "https://huggingface.co/org/gated")),
        )

    monkeypatch.setattr("senselab.utils.data_structures.model.check_hf_repo_exists", _gated)
    with pytest.raises(ValidationError) as caught:
        HFModel(path_or_uri="org/gated", revision="main")
    assert "GatedRepoError" in str(caught.value), "the erased type must survive in the message"
    assert caught.value.errors()[0]["ctx"]["error"].__cause__.__class__ is GatedRepoError, (
        "the original exception must stay reachable through ctx, which is what `from exc` buys"
    )


def test_hfmodel_wraps_a_resolution_failure_as_validationerror(monkeypatch: pytest.MonkeyPatch) -> None:
    """A commit-resolution failure surfaces as ValidationError, not a bare RuntimeError.

    Finding #5 of the #550 review: ``_resolve_commit_sha`` runs in a ``model_validator`` and let
    ``RevisionResolutionError`` (a ``RuntimeError``) escape. Pydantic only converts
    ``ValueError``/``AssertionError`` into ``ValidationError``, so every caller that catches
    ``ValidationError``/``ValueError`` around ``HFModel(...)`` saw an unhandled crash instead.
    """
    from pydantic import ValidationError

    from senselab.utils.model_revision import RevisionResolutionError

    monkeypatch.setattr("senselab.utils.data_structures.model.check_hf_repo_exists", lambda **kw: True)

    def _boom(*a: object, **k: object) -> str:
        raise RevisionResolutionError("cannot resolve org/model@main")

    monkeypatch.setattr("senselab.utils.model_revision.resolve_revision", _boom)
    with pytest.raises(ValidationError):
        HFModel(path_or_uri="org/model", revision="main")


def test_get_huggingface_token_from_env_file_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test loading a Hugging Face token from an explicit `.env` file path."""
    for env_var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        monkeypatch.delenv(env_var, raising=False)

    env_file = tmp_path / "hf.env"
    env_file.write_text("HF_TOKEN=hf_from_file\n")

    assert get_huggingface_token(env_file) == "hf_from_file"


def test_get_huggingface_token_from_local_dotenv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test loading a Hugging Face token from a local `.env` file in the cwd."""
    for env_var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        monkeypatch.delenv(env_var, raising=False)

    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("HUGGING_FACE_HUB_TOKEN=hf_from_local_dotenv\n")

    assert get_huggingface_token() == "hf_from_local_dotenv"


def test_get_huggingface_token_prefers_environment_over_dotenv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test environment variables take precedence over `.env` values."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("HF_TOKEN=hf_from_file\n")
    monkeypatch.setenv("HF_TOKEN", "hf_from_env")

    assert get_huggingface_token() == "hf_from_env"


@patch("senselab.utils.model_revision.resolve_revision", return_value="c" * 40)
@patch("senselab.utils.dependencies.ensure_hf_model", return_value="abc123")
def test_hfmodel_caches_hf_repo_check(mock_ensure: MagicMock, mock_resolve: MagicMock) -> None:
    """Test that we successfully cache HF repo checks and only make the check once."""
    _ = HFModel(path_or_uri="unique_repo_name_1")
    assert mock_ensure.call_count == 1

    _ = HFModel(path_or_uri="unique_repo_name_1")
    # Second instantiation should use in-memory _hf_cache, not call ensure again
    assert mock_ensure.call_count == 1

    _ = HFModel(path_or_uri="unique_repo_name_2")
    assert mock_ensure.call_count == 2


# ── model_for_task / safe_model_id (T051 consolidation) ───────────────


@pytest.fixture
def _offline_model_construction(monkeypatch: pytest.MonkeyPatch) -> None:
    """Let ``model_for_task`` build models without reaching the Hub.

    These tests assert *routing* — which class an id maps to — so any network at
    all is incidental. Unmocked, each construction runs the ``revision``
    validator into ``ensure_hf_model``, which downloads the entire snapshot: this
    pair alone would pull Sortformer and Whisper on every cold run, and an
    earlier revision of the diarization tests pulled 20 GB this way. Both the
    existence check and the commit resolution are stubbed, independently, so the
    tests never depend on a warm cache — verified under ``HF_HUB_OFFLINE=1`` with
    an empty ``HF_HUB_CACHE``.
    """
    monkeypatch.setattr("senselab.utils.data_structures.model.check_hf_repo_exists", lambda **kw: True)
    monkeypatch.setattr("senselab.utils.model_revision.resolve_revision", lambda *a, **k: "e" * 40)


def test_model_for_task_routes_diarization_by_prefix(_offline_model_construction: None) -> None:
    """Sortformer ids are HF-hosted; every other diarizer is pyannote."""
    from senselab.utils.data_structures import model_for_task
    from senselab.utils.data_structures.model import HFModel, PyannoteAudioModel

    assert isinstance(model_for_task("nvidia/diar_sortformer_4spk-v1", task="diarization"), HFModel)
    assert isinstance(model_for_task("pyannote/speaker-diarization-3.1", task="diarization"), PyannoteAudioModel)


def test_model_for_task_routes_remaining_tasks(_offline_model_construction: None) -> None:
    """ASR → HF; embeddings and enhancement → SpeechBrain."""
    from senselab.utils.data_structures import model_for_task
    from senselab.utils.data_structures.model import HFModel, SpeechBrainModel

    assert isinstance(model_for_task("openai/whisper-tiny", task="asr"), HFModel)
    assert isinstance(model_for_task("speechbrain/spkrec-ecapa-voxceleb", task="embeddings"), SpeechBrainModel)
    assert isinstance(model_for_task("speechbrain/sepformer-wham16k-enhancement", task="enhancement"), SpeechBrainModel)


def test_model_for_task_routes_driftse_enhancement_by_prefix(_offline_model_construction: None) -> None:
    """A ``LIANGXU123/DriftSE`` id is HF-hosted; every other enhancement id is SpeechBrain."""
    from senselab.utils.data_structures import model_for_task
    from senselab.utils.data_structures.model import HFModel, SpeechBrainModel

    assert isinstance(model_for_task("LIANGXU123/DriftSE", task="enhancement"), HFModel)
    assert isinstance(model_for_task("speechbrain/sepformer-wham16k-enhancement", task="enhancement"), SpeechBrainModel)


def test_model_for_task_rejects_unknown_task() -> None:
    """An unrecognized task must fail loudly, not default to a provider."""
    import pytest as _pytest

    from senselab.utils.data_structures import model_for_task

    with _pytest.raises(ValueError, match="unknown task"):
        model_for_task("some/model", task="not-a-task")


def test_safe_model_id_on_real_default_model_ids() -> None:
    """The two pre-consolidation implementations agreed on every shipped default.

    Pinned so the merge of the char-wise (script) and collapsing (labelstudio)
    variants is demonstrably behavior-neutral for real inputs.
    """
    from senselab.utils.data_structures import safe_model_id

    assert safe_model_id("openai/whisper-large-v3-turbo") == "openai_whisper_large_v3_turbo"
    assert safe_model_id("nvidia/diar_sortformer_4spk-v1") == "nvidia_diar_sortformer_4spk_v1"
    assert safe_model_id("MIT/ast-finetuned-audioset-10-10-0.4593") == "MIT_ast_finetuned_audioset_10_10_0_4593"
    assert safe_model_id("speechbrain/spkrec-ecapa-voxceleb") == "speechbrain_spkrec_ecapa_voxceleb"


def test_safe_model_id_collapses_runs_and_strips_edges() -> None:
    """Where the two old variants DIVERGED: runs collapse to one underscore.

    The char-wise variant produced "a__b" for "a--b"; this documents that the
    collapsing form won, so filenames and LS track names can never disagree.
    """
    from senselab.utils.data_structures import safe_model_id

    assert safe_model_id("a--b") == "a_b"
    assert safe_model_id("/leading/and/trailing/") == "leading_and_trailing"
    assert safe_model_id("plain_name") == "plain_name"  # idempotent on safe input
    assert safe_model_id("///") == "model"  # never empty — LS from_name must be valid
