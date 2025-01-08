"""This module provides utilities for pose estimation."""

import os

import requests

MODEL_PATH = "src/senselab/video/tasks/pose_estimation/models"

YOLO_KEYPOINT_MAPPING = {
    0: "Nose",
    1: "Left Eye",
    2: "Right Eye",
    3: "Left Ear",
    4: "Right Ear",
    5: "Left Shoulder",
    6: "Right Shoulder",
    7: "Left Elbow",
    8: "Right Elbow",
    9: "Left Wrist",
    10: "Right Wrist",
    11: "Left Hip",
    12: "Right Hip",
    13: "Left Knee",
    14: "Right Knee",
    15: "Left Ankle",
    16: "Right Ankle",
}

MEDIAPIPE_KEYPOINT_MAPPING = {
    0: "nose",
    1: "left_eye_inner",
    2: "left_eye",
    3: "left_eye_outer",
    4: "right_eye_inner",
    5: "right_eye",
    6: "right_eye_outer",
    7: "left_ear",
    8: "right_ear",
    9: "mouth_left",
    10: "mouth_right",
    11: "left_shoulder",
    12: "right_shoulder",
    13: "left_elbow",
    14: "right_elbow",
    15: "left_wrist",
    16: "right_wrist",
    17: "left_pinky",
    18: "right_pinky",
    19: "left_index",
    20: "right_index",
    21: "left_thumb",
    22: "right_thumb",
    23: "left_hip",
    24: "right_hip",
    25: "left_knee",
    26: "right_knee",
    27: "left_ankle",
    28: "right_ankle",
    29: "left_heel",
    30: "right_heel",
    31: "left_foot_index",
    32: "right_foot_index",
}

MODELS = {
    "mediapipe": {
        "lite": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
        "full": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task",
        "heavy": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task",
    },
    "yolo": {
        "8n": "yolov8n-pose.pt",
        "8s": "yolov8s-pose.pt",
        "8m": "yolov8m-pose.pt",
        "8l": "yolov8l-pose.pt",
        "8x": "yolov8x-pose.pt",
        "11n": "yolo11n-pose.pt",
        "11s": "yolo11s-pose.pt",
        "11m": "yolo11m-pose.pt",
        "11l": "yolo11l-pose.pt",
        "11x": "yolo11x-pose.pt",
    },
}


def get_model(model: str, model_type: str) -> str:
    """Retrieve the model file or name, depending on the model type.

    Args:
        model (str): The model category ('mediapipe' or 'yolo').
        model_type (str): The specific model type (e.g., 'lite', 'full', '8n').

    Returns:
        str: The local path to the MediaPipe model file or the YOLO model name.

    Raises:
        ValueError: If an invalid model type is provided.
    """
    if model not in MODELS:
        raise ValueError(f"Invalid model category '{model}'. Choose from {list(MODELS.keys())}.")

    if model_type not in MODELS[model]:
        raise ValueError(f"Invalid {model} model type '{model_type}'. Choose from {list(MODELS[model].keys())}.")

    model_url = MODELS[model][model_type]
    cache_dir = os.path.join(MODEL_PATH, model)
    os.makedirs(cache_dir, exist_ok=True)

    if model == "mediapipe":
        # For MediaPipe, download the model file if necessary
        model_filename = os.path.basename(model_url)
        model_path = os.path.join(cache_dir, model_filename)
        download_model(model_url, model_path)
        return model_path

    elif model == "yolo":
        # For YOLO, just return the model path
        model_path = os.path.join(cache_dir, model_url)
        return model_path

    return ""


def download_model(url: str, save_path: str) -> None:
    """Download a model file from a URL if it doesn't already exist locally.

    Args:
        url (str): The URL of the model file.
        save_path (str): The local path to save the model file.
    """
    if not os.path.exists(save_path):
        print(f"Downloading model from {url}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Model saved to {save_path}")
    else:
        print(f"Model already exists at {save_path}")
