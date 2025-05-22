"""This module contains a face tracking workflow using DeepFace."""

import csv
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from tqdm import tqdm

from senselab.video.data_structures.video import Video


def set_device(use_gpu: bool) -> None:
    """Set device for DeepFace (CPU or GPU). Must be called before DeepFaceAnalysis import."""
    if not use_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def _to_numpy(frame: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    """Convert a torch tensor or PIL image to a numpy array (H, W, C, BGR)."""
    if isinstance(frame, torch.Tensor):
        arr = frame.cpu().numpy()
        if arr.shape[-1] == 3:
            return arr.astype(np.uint8)
        elif arr.shape[0] == 3:
            return np.transpose(arr, (1, 2, 0)).astype(np.uint8)
    return np.array(frame)


def _draw_box_id(
    frame: np.ndarray,
    bbox: Tuple[int, int, int, int],
    face_id: int,
    color: tuple = (0, 255, 0),
    confidence: Optional[float] = None,
) -> None:
    """Draw bounding box and ID on frame."""
    y, x2, y2, x = bbox
    cv2.rectangle(frame, (x, y), (x2, y2), color, 2)
    label = f"ID:{face_id}"
    if confidence is not None:
        label += f" ({confidence:.2f})"
    cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def _save_results(results: Dict[int, List[Dict]], out_path: Union[str, Path], fmt: str = "json") -> None:
    """Save tracking results to JSON or CSV."""
    out_path = str(out_path)
    if fmt == "json":
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
    elif fmt == "csv":
        # Flatten dict for CSV
        flat_results = []
        for face_id, detections in results.items():
            for det in detections:
                row = {"face_id": face_id, **det}
                flat_results.append(row)
        keys = flat_results[0].keys() if flat_results else []
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(flat_results)
    else:
        raise ValueError(f"Unsupported format: {fmt}")


def _embedding_distance(
    e1: Union[np.ndarray, List[float]], e2: Union[np.ndarray, List[float]], metric: str = "cosine"
) -> float:
    """Compute distance between two embeddings."""
    if metric == "cosine":
        e1 = np.asarray(e1)
        e2 = np.asarray(e2)
        return 1 - np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2))
    else:
        return float(np.linalg.norm(np.asarray(e1) - np.asarray(e2)))


class FaceTracker:
    """Simple face tracker using DeepFaceAnalysis embeddings and nearest neighbor matching.

    Assigns consistent IDs to faces (individuals) across frames.

    Attributes:
        model_name: DeepFace model name
        detector_backend: DeepFace detector backend
        distance_metric: Embedding distance metric
        threshold: Distance threshold for matching
        use_gpu: Whether to use GPU (default True)
        use_embedding_averaging: If True, average embeddings for each individual (default False)

    """

    def __init__(
        self,
        model_name: str = "VGG-Face",
        detector_backend: str = "retinaface",
        distance_metric: str = "cosine",
        threshold: float = 0.5,
        use_gpu: bool = True,
        use_embedding_averaging: bool = False,
    ) -> None:
        """Initialize the FaceTracker class."""
        set_device(use_gpu)
        from senselab.video.tasks.face_analysis.deepface_utils import DeepFaceAnalysis

        self.deepface = DeepFaceAnalysis(
            model_name=model_name,
            detector_backend=detector_backend,
            distance_metric=distance_metric,
            enforce_detection=False,
        )
        self.distance_metric = distance_metric
        self.threshold = threshold
        self.next_id = 0
        self.individuals = []  # type: List[Dict]
        self.use_embedding_averaging = use_embedding_averaging

    def _match(self, embedding: Union[np.ndarray, List[float]], frame_idx: int) -> Tuple[int, float]:
        """Find the closest individual for the embedding, or assign new ID.

        If use_embedding_averaging is True, update individual's embedding as running average.
        """
        min_dist = float("inf")
        min_id = -1
        closest_individual = None
        # Find closest individual
        for individual in self.individuals:
            dist = _embedding_distance(embedding, individual["embedding"], self.distance_metric)
            if dist < min_dist:
                min_dist = dist
                closest_individual = individual
        # Update individual if close enough
        if min_dist < self.threshold and closest_individual is not None:
            min_id = int(closest_individual["id"])
            if self.use_embedding_averaging:
                count = closest_individual.get("count", 1)
                closest_individual["embedding"] = (closest_individual["embedding"] * count + np.array(embedding)) / (
                    count + 1
                )
                closest_individual["count"] = count + 1
            else:
                closest_individual["embedding"] = embedding
            closest_individual["last_seen_frame"] = frame_idx
            return min_id, min_dist
        else:
            # New face/individual
            new_id = self.next_id
            self.individuals.append(
                {
                    "id": new_id,
                    "embedding": np.array(embedding),
                    "last_seen_frame": frame_idx,
                    "count": 1,
                }
            )
            self.next_id += 1
            return new_id, float("inf")

    def track_video(
        self,
        video_input: Union[str, Video],
        out_video_path: Optional[Union[str, Path]] = None,
        out_results_path: Optional[Union[str, Path]] = None,
        results_format: str = "json",
    ) -> Tuple[Optional[Union[str, Path]], Optional[Union[str, Path]], Dict[int, List[Dict]]]:
        """Track faces in a video, save annotated video and results.

        Args:
            video_input: Path to video file or Video object
            out_video_path: Path to save annotated video (optional)
            out_results_path: Path to save results (JSON/CSV)
            results_format: 'json' or 'csv'
        Returns:
            Path(s) to output files, and results dictionary
        """
        # Load video
        if isinstance(video_input, Video):
            frames = video_input.frames
            frame_rate = video_input.frame_rate
        elif isinstance(video_input, str):
            video_obj = Video(filepath=video_input)
            frames = video_obj.frames
            frame_rate = video_obj.frame_rate
        else:
            raise ValueError(f"Unsupported video input type: {type(video_input)}")  # type: ignore[unreachable]

        n_frames = frames.shape[0]
        height, width = frames.shape[1:3]

        # Prepare video writer if needed
        writer = None
        if out_video_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(out_video_path), fourcc, frame_rate, (width, height))

        results: Dict[int, List[Dict]] = defaultdict(list)
        for idx in tqdm(range(n_frames), desc="Tracking faces"):
            frame = _to_numpy(frames[idx])
            try:
                faces = self.deepface.extract_face_embeddings(frame)
            except Exception as e:
                print(f"Error extracting faces in frame {idx}: {e}")
                faces = []
            for face in faces:
                bbox = face.get("facial_area", {})
                x, y, w, h = (
                    bbox.get("x", 0),
                    bbox.get("y", 0),
                    bbox.get("w", 0),
                    bbox.get("h", 0),
                )
                box = (y, x + w, y + h, x)
                embedding = face.get("embedding", [])
                confidence = face.get("face_confidence", 1.0)
                if embedding:
                    face_id, _ = self._match(embedding, idx)
                else:
                    face_id = -1
                # Draw
                if writer:
                    _draw_box_id(frame, box, face_id, confidence=confidence)
                # Save result
                results[face_id].append(
                    {
                        "frame": idx,
                        "bbox": box,
                        "confidence": confidence,
                    }
                )
            if writer:
                # Convert to BGR if needed
                if frame.shape[2] == 3 and np.max(frame) > 1.0:
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                else:
                    frame_bgr = frame
                writer.write(frame_bgr)
        if writer:
            writer.release()
        results_dict: Dict[int, List[Dict]] = dict(results)
        if out_results_path:
            _save_results(results_dict, out_results_path, fmt=results_format)
        return out_video_path, out_results_path, results_dict
