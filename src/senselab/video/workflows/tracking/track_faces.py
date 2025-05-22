"""Function for tracking faces in a video."""

from typing import Dict, List, Optional, Union

from senselab.video.data_structures.video import Video

from .face_tracker import FaceTracker


def track_faces(
    video_input: Union[str, Video],
    out_video_path: Optional[str] = None,
    out_results_path: Optional[str] = None,
    results_format: str = "json",
    use_gpu: bool = True,
    use_embedding_averaging: bool = False,
    model_name: str = "VGG-Face",
    detector_backend: str = "retinaface",
    distance_metric: str = "cosine",
    threshold: float = 0.5,
) -> Dict[int, List[Dict]]:
    """Track faces in a video and return a mapping from face_id to a list of detections.

    Args:
        video_input (str or Video): Path to video file or Video object.
        out_video_path (str, optional): Path to save annotated video.
        out_results_path (str, optional): Path to save results (JSON/CSV).
        results_format (str): 'json' or 'csv'.
        use_gpu (bool): Whether to use GPU (default True).
        use_embedding_averaging (bool): Average embeddings for each individual (default False).
        model_name (str): DeepFace model name (default 'VGG-Face').
        detector_backend (str): DeepFace detector backend (default 'retinaface').
        distance_metric (str): Embedding distance metric (default 'cosine').
        threshold (float): Distance threshold for matching (default 0.5).

    Returns:
        Dict[int, List[Dict]]: Mapping from face_id to list of detections (frame, bbox, confidence).

    Example:
        >>> from senselab.video.workflows.tracking.track_faces import track_faces
        >>> results = track_faces('my_video.mp4')
        >>> print(results)
    """
    tracker = FaceTracker(
        model_name=model_name,
        detector_backend=detector_backend,
        distance_metric=distance_metric,
        threshold=threshold,
        use_gpu=use_gpu,
        use_embedding_averaging=use_embedding_averaging,
    )

    _, _, results = tracker.track_video(
        video_input=video_input,
        out_video_path=out_video_path,
        out_results_path=out_results_path,
        results_format=results_format,
    )

    return results
