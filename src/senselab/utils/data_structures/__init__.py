"""This module provides the APIs of the senselab utils data structures."""

from .dataset import Participant, SenselabDataset, Session  # noqa: F401
from .device import DeviceType, _select_device_and_dtype  # noqa: F401
from .file import File, from_strings_to_files, get_common_directory  # noqa: F401
from .language import Language  # noqa: F401
from .logging import logger  # noqa: F401
from .model import (  # noqa: F401
    HFModel,
    PyannoteAudioModel,
    SenselabModel,
    SentenceTransformersModel,
    SpeechBrainModel,
    TorchModel,
    check_hf_repo_exists,
)

# Conditionally import TorchAudioModel only if torchaudio is available
try:
    from .model import TorchAudioModel
except ImportError:
    pass

# Conditionally import CoquiTTSModel only if coqui-tts is available
try:
    from .model import CoquiTTSModel
except ImportError:
    pass

from .pydra_helpers import *  # noqa: F403
from .script_line import ScriptLine  # noqa: F401
