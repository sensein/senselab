"""This module implements some utilities for the model class."""

import logging
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import ClassVar, Dict, Generic, Optional, Tuple, TypeVar, Union

import requests
import torch
from dotenv import dotenv_values, find_dotenv
from huggingface_hub import HfApi
from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError, RevisionNotFoundError
from huggingface_hub.hf_api import ModelInfo
from pydantic import BaseModel, Field, PrivateAttr, ValidationInfo, field_validator, model_validator
from typing_extensions import Annotated

from senselab.utils.dependencies import torchaudio_available

TORCHAUDIO_AVAILABLE = torchaudio_available()
if TORCHAUDIO_AVAILABLE:
    import torchaudio

logger = logging.getLogger("senselab")

# Define the TypeVar for provider types
PROVIDER_T = TypeVar("PROVIDER_T")


class SenselabModel(BaseModel, Generic[PROVIDER_T]):
    """Base configuration for SenselabModel class."""

    path_or_uri: Union[str, Path]
    revision: Optional[str] = None

    @field_validator("path_or_uri", mode="before")
    def validate_path_or_uri(cls, value: Union[str, Path]) -> Union[str, Path]:
        """Validate the path_or_uri.

        This check is only for files and not for remote resources.
        It does check if the path_or_uri is not empty and if it is an
        existing file.
        """
        if not value:
            raise ValueError("path_or_uri cannot be empty")

        if isinstance(value, Path) and not os.path.isfile(value):
            raise ValueError("path_or_uri is not an existing file")

        # If the value is a string and looks like an existing file path,
        # convert it to a Path object
        if isinstance(value, str) and os.path.isfile(value):
            value = Path(value)
            if not is_torch_model(value):
                raise ValueError("path_or_uri does not point to a valid torch model")

        return value


class HFModel(SenselabModel[PROVIDER_T]):
    """HuggingFace model.

    Note: For some HuggingFace models, HF_TOKEN may be required for access.
    """

    revision: Annotated[str, Field(validate_default=True)] = "main"
    commit_sha: Optional[str] = None
    """The immutable 40-hex commit this run pins to, resolved at construction.

    Distinct from ``revision``, which records what was *asked for*. Keeping both
    lets provenance distinguish "pinned to abc123" from "tracked main, which
    resolved to abc123" -- drift is only diagnosable when those are tellable apart.
    """
    info: Optional[ModelInfo] = None
    _hf_cache: ClassVar[Dict[Tuple[str, str], bool]] = {}

    @field_validator("revision")
    def validate_hf_model_id(cls, value: str, info: ValidationInfo) -> Union[str, Path]:
        """Validate the path_or_uri.

        This check is only for remote resources and not for files.
        It checks if the specified Hugging Face model ID and revision exist
        in the remote Hub.
        """
        path_or_uri = info.data["path_or_uri"]
        if not isinstance(path_or_uri, Path):
            if (str(path_or_uri), value) not in cls._hf_cache:
                cls._hf_cache[(str(path_or_uri), value)] = check_hf_repo_exists(
                    repo_id=str(path_or_uri), revision=value, repo_type="model"
                )

            if not cls._hf_cache[(str(path_or_uri), value)]:
                raise ValueError(
                    f"The huggingface model: path_or_uri ({path_or_uri}) or "
                    f"specified revision ({value}) cannot be found.\n"
                    "Please check the model ID and revision. If the model is "
                    "private or restricted access, make sure you have access "
                    "to it and have exported your huggingface token in your "
                    "environment variables."
                )
        return value

    @model_validator(mode="after")
    def _resolve_commit_sha(self) -> "HFModel":
        """Pin this model to an immutable commit, once, at construction.

        Skipped for local paths, which have no Hub revision. The resolution is one
        the constructor already performs -- ``check_hf_repo_exists`` calls
        ``ensure_hf_model``, which computes this SHA and discards it -- so this
        adds no network call and no download.

        Plain assignment, not ``object.__setattr__``: ``HFModel`` sets neither
        ``frozen`` nor ``validate_assignment`` in its config, so pydantic's normal
        ``__setattr__`` just stores the value -- no frozen-field bypass is needed,
        and no assignment-time re-validation loop to worry about either.
        """
        if isinstance(self.path_or_uri, Path) or self.commit_sha is not None:
            return self
        from senselab.utils.model_revision import RevisionResolutionError, resolve_revision

        # resolve_revision raises RevisionResolutionError, a RuntimeError. Pydantic only converts
        # ValueError/AssertionError raised inside a validator into ValidationError, so a RuntimeError
        # would escape unhandled — and every caller that catches ValidationError/ValueError around
        # HFModel(...) (the pattern the `revision` field validator above establishes) would see an
        # unrelated crash instead. Re-raise as ValueError so it wraps consistently.
        try:
            self.commit_sha = resolve_revision(str(self.path_or_uri), self.revision)
        except RevisionResolutionError as exc:
            raise ValueError(str(exc)) from exc
        return self

    def get_model_info(self) -> ModelInfo:
        """Gets the model info using the HuggingFace API and saves it as a property."""
        if isinstance(self.path_or_uri, Path):
            raise ValueError("Model info is only available for remote resources and not for files.")
        if not self.info:
            api = HfApi(token=get_huggingface_token())
            self.info = api.model_info(repo_id=self.path_or_uri, revision=self.revision)
        return self.info


class SpeechBrainModel(HFModel[PROVIDER_T]):
    """SpeechBrain model."""

    pass


class PyannoteAudioModel(HFModel[PROVIDER_T]):
    """PyannoteAudioModel model."""

    pass


class SentenceTransformersModel(HFModel[PROVIDER_T]):
    """SentenceTransformersModel model."""

    pass


class CoquiTTSModel(SenselabModel[PROVIDER_T]):
    """CoquiTTSModel model."""

    _scope: Optional[str] = None

    @field_validator("path_or_uri", mode="before")
    def validate_path_or_uri(cls, value: Union[str, Path]) -> Union[str, Path]:
        """Validate the path_or_uri.

        Coqui TTS runs in an isolated subprocess venv.  Validation queries
        the model list via the subprocess venv (installs it on first call).
        """
        if not isinstance(value, Path):
            from senselab.audio.tasks.voice_cloning.coqui import list_coqui_models

            model_ids = list_coqui_models()
            if value not in model_ids:
                raise ValueError(f"Model {value} not found. Available models: {model_ids}")
            cls._scope = value.split("/")[0]

        return value


class TorchModel(SenselabModel[PROVIDER_T]):
    """Generic torch model."""

    revision: Annotated[str, Field(validate_default=True)] = "main"

    @field_validator("revision", mode="before")
    def validate_torch_model_id(cls, value: str, info: ValidationInfo) -> Union[str, Path]:
        """Validate the path_or_uri.

        This check is only for remote resources and not for files.
        It checks if the specified torch model ID and revision exist in the
        remote Hub.
        """
        path_or_uri = info.data["path_or_uri"]
        if not isinstance(path_or_uri, Path):
            if not check_github_repo_exists(repo_id=str(path_or_uri), branch=value):
                raise ValueError("path_or_uri or specified revision is not a valid github repo")
        return value


class TorchAudioModel(SenselabModel[PROVIDER_T]):
    """TorchAudio model."""

    revision: Annotated[str, Field(validate_default=True)] = "main"

    @field_validator("revision", mode="before")
    def validate_torchaudio_model_id(cls, value: str, info: ValidationInfo) -> Union[str, Path]:
        """Validate the path_or_uri for torchaudio models.

        This check is only for remote resources and not for files.
        It checks if the specified torchaudio model ID exists.
        """
        path_or_uri = info.data["path_or_uri"]
        if not isinstance(path_or_uri, Path):
            if not check_torchaudio_model_exists(model_id=str(path_or_uri)):
                raise ValueError("path_or_uri is not a valid torchaudio model")
        return value


def check_torchaudio_model_exists(model_id: str) -> bool:
    """Private function to check if a torchaudio model exists."""
    if not TORCHAUDIO_AVAILABLE:
        raise ModuleNotFoundError(
            "`torchaudio` is not installed. Please install senselab audio dependencies using `pip install senselab`."
        )

    try:
        _ = getattr(torchaudio.pipelines, model_id)
        return True
    except AttributeError:
        return False


def is_torch_model(file_path: Path) -> bool:
    """Check if a file is a torch model."""
    try:
        _ = torch.load(file_path)
        return True
    except Exception:
        return False


def get_huggingface_token(env_file_path: Optional[Union[str, Path]] = None) -> Optional[str]:
    """Return a Hugging Face token from the environment or a local `.env` file."""
    token_env_vars = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN")
    for env_var in token_env_vars:
        token = os.getenv(env_var)
        if token:
            return token

    dotenv_path = Path(env_file_path).expanduser() if env_file_path is not None else Path(find_dotenv(usecwd=True))
    if not dotenv_path.is_file():
        return None

    dotenv_values_dict = dotenv_values(dotenv_path)
    for env_var in token_env_vars:
        token = dotenv_values_dict.get(env_var)
        if token:
            return str(token)
    return None


def check_hf_repo_exists(repo_id: str, revision: str = "main", repo_type: str = "model") -> bool:
    """Check if a Hugging Face repository exists.

    For models, uses :func:`ensure_hf_model` which coordinates across
    processes via file locking and caches results on the shared filesystem.
    """
    if repo_type == "model":
        from senselab.utils.dependencies import ensure_hf_model

        try:
            ensure_hf_model(repo_id, revision)
            return True
        except GatedRepoError:
            # Repo exists but requires auth — surface it; do NOT report as "missing".
            raise
        except (RepositoryNotFoundError, RevisionNotFoundError):
            # Genuinely absent — this is the only case that means "does not exist".
            return False
        except Exception as exc:
            # Transient (rate-limit / network) errors must not masquerade as
            # "not found" — that silently turns a throttled real model into a
            # confusing "missing". Log and re-raise so the caller sees the real,
            # retryable error.
            logger.warning(
                "Could not verify HF model %s@%s (transient error; surfacing rather than reporting missing): %s",
                repo_id,
                revision,
                exc,
            )
            raise

    # Non-model repos (rare): direct API check
    api = HfApi(token=get_huggingface_token())
    try:
        api.list_repo_commits(repo_id=repo_id, revision=revision, repo_type=repo_type)
        return True
    except (RepositoryNotFoundError, RevisionNotFoundError):
        return False


@lru_cache(maxsize=128)
def check_github_repo_exists(repo_id: str, branch: str = "main") -> bool:
    """Checks if a GitHub repository exists with caching and authentication."""
    url = f"https://api.github.com/repos/{repo_id}/branches/{branch}"
    token = os.getenv("GITHUB_TOKEN") or None

    headers = {}
    if token:
        headers = {"Authorization": f"token {token}"}

    response = requests.get(url, headers=headers, timeout=10)

    if response.status_code == 200:
        return True
    elif response.status_code == 404:
        return False
    elif response.status_code == 403:  # Handle rate limit exceeded
        print("GitHub API rate limit exceeded. Please try again later.")
        return False
    else:
        response.raise_for_status()
        return False


# Rebuild the model classes to ensure proper generic type resolution
SenselabModel.model_rebuild()
HFModel.model_rebuild()
SpeechBrainModel.model_rebuild()
PyannoteAudioModel.model_rebuild()
SentenceTransformersModel.model_rebuild()
CoquiTTSModel.model_rebuild()
TorchModel.model_rebuild()
TorchAudioModel.model_rebuild()


def model_for_task(model_id: str, *, task: str) -> SenselabModel:
    """Wrap a model id in the right `SenselabModel` subclass for a given task.

    Lives here rather than in a workflow module because it is pure model-id →
    provider-class routing, and it has consumers in the analysis stages, the CLI
    script, and the enhancement path.

    Args:
        model_id: HuggingFace-style model identifier.
        task: One of ``"diarization"``, ``"asr"``, ``"embeddings"``, ``"enhancement"``,
            ``"separation"``.

    Returns:
        The provider-specific `SenselabModel` subclass instance.

    Raises:
        ValueError: If ``task`` is not recognized.

    Note:
        The diarization branch duplicates ``diarize_audios``'s internal dispatch:
        five separate prefix conditions (Sortformer, VibeVoice-ASR-HF, USC-SAIL
        child-adult, MOSS-Transcribe-Diarize, DiariZen — everything else falls
        through to Pyannote) each independently repeated here and in
        ``speaker_diarization/api.py``'s ``elif`` chain. That duplication is
        pre-existing and worth collapsing into one source of truth eventually;
        the two tables must be kept in sync by hand until then. The enhancement
        branch below duplicates ``enhance_audios``'s ``sensein/driftse`` prefix
        check the same way, for the same reason: importing the literal from
        ``audio.tasks.speech_enhancement.api`` here would make ``utils`` depend
        on ``audio``, inverting the package layering. The separation branch has
        only one backend (unasdiff) and always returns ``HFModel`` unconditionally;
        it exists for parity with the other tasks' model-id → class routing, not
        because there is a second backend to dispatch away from.

    Example:
        >>> model_for_task("openai/whisper-tiny", task="asr").path_or_uri
        'openai/whisper-tiny'
    """
    if task == "diarization":
        if (
            model_id.startswith("nvidia/diar_sortformer")
            or model_id.startswith("microsoft/VibeVoice-ASR")
            or model_id.startswith("AlexXu811/whisper-child-adult")
            or model_id.startswith("OpenMOSS-Team/MOSS-Transcribe-Diarize")
            or model_id.startswith("BUT-FIT/diarizen")
        ):
            return HFModel(path_or_uri=model_id)
        return PyannoteAudioModel(path_or_uri=model_id)
    if task == "asr":
        return HFModel(path_or_uri=model_id)
    if task == "embeddings":
        return SpeechBrainModel(path_or_uri=model_id)
    if task == "enhancement":
        if model_id.startswith("sensein/driftse"):
            return HFModel(path_or_uri=model_id)
        return SpeechBrainModel(path_or_uri=model_id)
    if task == "separation":
        return HFModel(path_or_uri=model_id)
    raise ValueError(f"unknown task: {task}")


def safe_model_id(model_id: str) -> str:
    """Sanitize a model id for use in filenames and Label Studio ``from_name`` values.

    Collapses every run of non-alphanumeric/underscore characters to a single
    ``_``, strips leading/trailing separators, and never returns an empty string.

    This consolidates two implementations that had silently diverged — a
    character-wise variant in the CLI script and this collapsing variant in
    ``labelstudio.py``. They agreed on every model id in the current defaults
    (none contains adjacent non-alphanumerics), so unifying on the more robust
    collapsing form is behavior-neutral today and prevents a filename-vs-track
    name divergence later.

    Args:
        model_id: Raw model identifier.

    Returns:
        A filesystem- and Label-Studio-safe token.

    Example:
        >>> safe_model_id("openai/whisper-large-v3-turbo")
        'openai_whisper_large_v3_turbo'
        >>> safe_model_id("a--b")
        'a_b'
        >>> safe_model_id("///")
        'model'
    """
    return re.sub(r"[^A-Za-z0-9_]+", "_", model_id).strip("_") or "model"
