"""Tests for HF models and functions."""

from pathlib import Path
from unittest.mock import MagicMock, patch

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

    model = HFModel(path_or_uri="org/model", revision="main")
    assert model.revision == "main", "the requested ref must survive"
    assert model.commit_sha == sha, "the resolved commit must be recorded"


def test_hfmodel_invalid_hf_repo_check() -> None:
    """Test invalid HFModel repo check."""
    with patch("senselab.utils.data_structures.model.check_hf_repo_exists", return_value=False):
        with pytest.raises(ValueError):
            HFModel(path_or_uri="invalid/repo")


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


def test_model_for_task_routes_diarization_by_prefix() -> None:
    """Sortformer ids are HF-hosted; every other diarizer is pyannote."""
    from senselab.utils.data_structures import model_for_task
    from senselab.utils.data_structures.model import HFModel, PyannoteAudioModel

    assert isinstance(model_for_task("nvidia/diar_sortformer_4spk-v1", task="diarization"), HFModel)
    assert isinstance(model_for_task("pyannote/speaker-diarization-3.1", task="diarization"), PyannoteAudioModel)


def test_model_for_task_routes_remaining_tasks() -> None:
    """ASR → HF; embeddings and enhancement → SpeechBrain."""
    from senselab.utils.data_structures import model_for_task
    from senselab.utils.data_structures.model import HFModel, SpeechBrainModel

    assert isinstance(model_for_task("openai/whisper-tiny", task="asr"), HFModel)
    assert isinstance(model_for_task("speechbrain/spkrec-ecapa-voxceleb", task="embeddings"), SpeechBrainModel)
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
