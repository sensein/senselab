""".. include:: ./doc.md"""  # noqa: D415

from senselab.video.tasks.scene_description.aggregation import (
    CategoricalSummary,
    ChangePoint,
    NumericSummary,
    VideoSceneSummary,
    aggregate,
)
from senselab.video.tasks.scene_description.api import (
    FrameRecord,
    VideoSceneReport,
    describe_video_scene,
    describe_video_scenes,
)
from senselab.video.tasks.scene_description.backends import (
    DEFAULT_MLX_MODEL,
    DEFAULT_TRANSFORMERS_MODEL,
    Generation,
    GenerationRequest,
    MLXVLMBackend,
    TransformersVLMBackend,
    VLMBackend,
    default_backend,
)
from senselab.video.tasks.scene_description.cv_metrics import FrameMetrics, frame_metrics
from senselab.video.tasks.scene_description.describe import FrameDescription, describe_frames
from senselab.video.tasks.scene_description.sampling import SampledFrame, iter_sampled_frames, sample_frames
from senselab.video.tasks.scene_description.schema import SceneDescription, TileDescription, parse_scene_json

__all__ = [
    "CategoricalSummary",
    "ChangePoint",
    "DEFAULT_MLX_MODEL",
    "DEFAULT_TRANSFORMERS_MODEL",
    "FrameDescription",
    "FrameMetrics",
    "FrameRecord",
    "Generation",
    "GenerationRequest",
    "MLXVLMBackend",
    "NumericSummary",
    "SampledFrame",
    "SceneDescription",
    "TileDescription",
    "TransformersVLMBackend",
    "VLMBackend",
    "VideoSceneReport",
    "VideoSceneSummary",
    "aggregate",
    "default_backend",
    "describe_frames",
    "describe_video_scene",
    "describe_video_scenes",
    "frame_metrics",
    "iter_sampled_frames",
    "parse_scene_json",
    "sample_frames",
]
