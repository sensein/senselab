"""This module provides utilities for pose estimation."""

import os

import requests

# Path where models are stored
MODEL_PATH = "src/senselab/video/tasks/pose_estimation/models"

# Senselab Pose Estimation keypoint mapping
SENSELAB_KEYPOINT_MAPPING = {
    0: "Nose",
    1: "Left Eye Inner",
    2: "Left Eye",
    3: "Left Eye Outer",
    4: "Right Eye Inner",
    5: "Right Eye",
    6: "Right Eye Outer",
    7: "Left Ear",
    8: "Right Ear",
    9: "Mouth Left",
    10: "Mouth Right",
    11: "Left Shoulder",
    12: "Right Shoulder",
    13: "Left Elbow",
    14: "Right Elbow",
    15: "Left Wrist",
    16: "Right Wrist",
    17: "Left Pinky",
    18: "Right Pinky",
    19: "Left Index",
    20: "Right Index",
    21: "Left Thumb",
    22: "Right Thumb",
    23: "Left Hip",
    24: "Right Hip",
    25: "Left Knee",
    26: "Right Knee",
    27: "Left Ankle",
    28: "Right Ankle",
    29: "Left Heel",
    30: "Right Heel",
    31: "Left Foot Index",
    32: "Right Foot Index",
}

YOLO_KEYPOINT_MAPPING = {
    0: SENSELAB_KEYPOINT_MAPPING[0],  # Nose
    1: SENSELAB_KEYPOINT_MAPPING[2],  # Left Eye
    2: SENSELAB_KEYPOINT_MAPPING[5],  # Right Eye
    3: SENSELAB_KEYPOINT_MAPPING[7],  # Left Ear
    4: SENSELAB_KEYPOINT_MAPPING[8],  # Right Ear
    5: SENSELAB_KEYPOINT_MAPPING[11],  # Left Shoulder
    6: SENSELAB_KEYPOINT_MAPPING[12],  # Right Shoulder
    7: SENSELAB_KEYPOINT_MAPPING[13],  # Left Elbow
    8: SENSELAB_KEYPOINT_MAPPING[14],  # Right Elbow
    9: SENSELAB_KEYPOINT_MAPPING[15],  # Left Wrist
    10: SENSELAB_KEYPOINT_MAPPING[16],  # Right Wrist
    11: SENSELAB_KEYPOINT_MAPPING[23],  # Left Hip
    12: SENSELAB_KEYPOINT_MAPPING[24],  # Right Hip
    13: SENSELAB_KEYPOINT_MAPPING[25],  # Left Knee
    14: SENSELAB_KEYPOINT_MAPPING[26],  # Right Knee
    15: SENSELAB_KEYPOINT_MAPPING[27],  # Left Ankle
    16: SENSELAB_KEYPOINT_MAPPING[28],  # Right Ankle
}

MEDIAPIPE_KEYPOINT_MAPPING = {
    0: SENSELAB_KEYPOINT_MAPPING[0],  # Nose
    1: SENSELAB_KEYPOINT_MAPPING[1],  # Left Eye Inner
    2: SENSELAB_KEYPOINT_MAPPING[2],  # Left Eye
    3: SENSELAB_KEYPOINT_MAPPING[3],  # Left Eye Outer
    4: SENSELAB_KEYPOINT_MAPPING[4],  # Right Eye Inner
    5: SENSELAB_KEYPOINT_MAPPING[5],  # Right Eye
    6: SENSELAB_KEYPOINT_MAPPING[6],  # Right Eye Outer
    7: SENSELAB_KEYPOINT_MAPPING[7],  # Left Ear
    8: SENSELAB_KEYPOINT_MAPPING[8],  # Right Ear
    9: SENSELAB_KEYPOINT_MAPPING[9],  # Mouth Left
    10: SENSELAB_KEYPOINT_MAPPING[10],  # Mouth Right
    11: SENSELAB_KEYPOINT_MAPPING[11],  # Left Shoulder
    12: SENSELAB_KEYPOINT_MAPPING[12],  # Right Shoulder
    13: SENSELAB_KEYPOINT_MAPPING[13],  # Left Elbow
    14: SENSELAB_KEYPOINT_MAPPING[14],  # Right Elbow
    15: SENSELAB_KEYPOINT_MAPPING[15],  # Left Wrist
    16: SENSELAB_KEYPOINT_MAPPING[16],  # Right Wrist
    17: SENSELAB_KEYPOINT_MAPPING[17],  # Left Pinky
    18: SENSELAB_KEYPOINT_MAPPING[18],  # Right Pinky
    19: SENSELAB_KEYPOINT_MAPPING[19],  # Left Index
    20: SENSELAB_KEYPOINT_MAPPING[20],  # Right Index
    21: SENSELAB_KEYPOINT_MAPPING[21],  # Left Thumb
    22: SENSELAB_KEYPOINT_MAPPING[22],  # Right Thumb
    23: SENSELAB_KEYPOINT_MAPPING[23],  # Left Hip
    24: SENSELAB_KEYPOINT_MAPPING[24],  # Right Hip
    25: SENSELAB_KEYPOINT_MAPPING[25],  # Left Knee
    26: SENSELAB_KEYPOINT_MAPPING[26],  # Right Knee
    27: SENSELAB_KEYPOINT_MAPPING[27],  # Left Ankle
    28: SENSELAB_KEYPOINT_MAPPING[28],  # Right Ankle
    29: SENSELAB_KEYPOINT_MAPPING[29],  # Left Heel
    30: SENSELAB_KEYPOINT_MAPPING[30],  # Right Heel
    31: SENSELAB_KEYPOINT_MAPPING[31],  # Left Foot Index
    32: SENSELAB_KEYPOINT_MAPPING[32],  # Right Foot Index
}

# Available models
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
