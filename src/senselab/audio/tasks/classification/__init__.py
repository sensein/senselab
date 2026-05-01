""".. include:: ./doc.md"""  # noqa: D415

from .api import classify_audios, scene_results_to_segments  # noqa: F401
from .speech_emotion_recognition import classify_emotions_from_speech  # noqa: F401
from .yamnet import YAMNetClassifier  # noqa: F401

__all__ = ["classify_audios", "classify_emotions_from_speech", "scene_results_to_segments", "YAMNetClassifier"]
