"""Module for testing Pose Estimation data structures."""

import numpy as np

from senselab.video.data_structures.pose import (
    ImagePose,
    IndividualPose,
    MediaPipePoseLandmark,
    PoseModel,
    YOLOPoseLandmark,
)


def test_mediapipe_pose_landmark() -> None:
    """Tests the MediaPipePoseLandmark structure."""
    landmark = MediaPipePoseLandmark(x=0.5, y=0.4, z=0.3, visibility=0.8)
    assert landmark.as_list() == [0.5, 0.4, 0.3, 0.8], "MediaPipePoseLandmark as_list() method failed."


def test_yolo_pose_landmark() -> None:
    """Tests the YOLOPoseLandmark structure."""
    landmark = YOLOPoseLandmark(x=0.7, y=0.6, confidence=0.9)
    assert landmark.as_list() == [0.7, 0.6, 0.9], "YOLOPoseLandmark as_list() method failed."


def test_individual_pose() -> None:
    """Tests the IndividualPose structure."""
    normalized_landmarks = {
        "Nose": MediaPipePoseLandmark(x=0.1, y=0.2, z=0.3, visibility=0.9),
        "Left Eye": MediaPipePoseLandmark(x=0.15, y=0.25, z=0.35, visibility=0.85),
    }
    individual_pose = IndividualPose(
        individual_index=0,
        normalized_landmarks=normalized_landmarks,
        world_landmarks=None,
    )
    assert individual_pose.individual_index == 0, "IndividualPose index mismatch."
    assert len(individual_pose.normalized_landmarks) == 2, "Incorrect number of landmarks."
    assert individual_pose.get_landmark_coordinates("Nose").as_list() == [
        0.1,
        0.2,
        0.3,
        0.9,
    ], "Failed to retrieve coordinates."


def test_image_pose() -> None:
    """Tests the ImagePose structure."""
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    individuals = [
        IndividualPose(
            individual_index=0,
            normalized_landmarks={
                "Nose": MediaPipePoseLandmark(x=0.1, y=0.2, z=0.3, visibility=0.9),
            },
        ),
        IndividualPose(
            individual_index=0,
            normalized_landmarks={
                "Ear": MediaPipePoseLandmark(x=0.4, y=0.3, z=0.2, visibility=0.7),
            },
            world_landmarks={
                "Ear": MediaPipePoseLandmark(x=54, y=34, z=20, visibility=0.7),
            },
        ),
    ]
    image_pose = ImagePose(model=PoseModel.MEDIAPIPE, image=image, individuals=individuals)
    assert image_pose.model == PoseModel.MEDIAPIPE, "Model type mismatch."
    assert len(image_pose.individuals) == 2, "Incorrect number of individuals."
    assert image_pose.get_individual(0) == individuals[0], "Failed to retrieve individual pose."
