# Pose Estimation

[![Tutorial](https://img.shields.io/badge/Tutorial-Click%20Here-blue?style=for-the-badge)](https://github.com/sensein/senselab/blob/main/tutorials/video/pose_estimation.ipynb)


## Task Overview

Pose estimation is the process of detecting and tracking key points on a human body or other objects in images or videos. These key points represent joints, limbs, or other significant regions of interest. Pose estimation is widely used in applications such as motion analysis, sports performance tracking, gesture recognition, and augmented reality.

`senselab` supports pose estimation using **MediaPipe** and **YOLO**, offering models with varying accuracy, speed, and computational requirements.

---

## Models

### [MediaPipe](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker)
MediaPipe provides three pose estimation models:
- **Lite**: Optimized for mobile devices with low latency requirements.
- **Full**: Balanced between accuracy and efficiency, suitable for most applications.
- **Heavy**: High-accuracy model designed for tasks where precision is critical but latency is less of a concern.

These models detect 33 key points across the body, including joints, eyes, ears, and the nose.

### YOLO
YOLO-based pose estimation models are efficient and capable of detecting key points in real-time. Supported variants include:
- **[YOLOv8](https://docs.ultralytics.com/models/yolov8/)** and **[YOLOv11](https://docs.ultralytics.com/models/yolo11/#what-tasks-can-yolo11-models-perform)** families, with increasing model sizes (e.g., `8n`, `8s`, `11n`, `11l`) to balance speed and accuracy.

These models detect 17 key points, including joints like shoulders, elbows, knees, and ankles.


## Evaluation

### Metrics
-  **Percentage of Correct Parts (PCP)**: Evaluates limb detection accuracy. A limb is considered correct if the predicted key points are within half the limb’s length from the true points.
- **Percentage of Correct Keypoints (PCK)**: Considers a key point correct if the distance between the true and predicted points is within a threshold (e.g., 0.2 times the person’s head bone length).
- **Percentage of Detected Joints (PDJ)**: Evaluates joints as correct if the true and predicted points are within a fraction of the torso’s diameter.
- **Object Keypoint Similarity (OKS)**: Measures the normalized distance between true and predicted key points, scaled by the person’s size. Computes the mean Average Precision (mAP) for all key points in the frame.


### Benchmark Datasets
- **[COCO Keypoints](https://cocodataset.org/#keypoints-2020)**: Annotated key points for human poses in diverse scenes.
- **[MPII Human Pose](http://human-pose.mpi-inf.mpg.de/)**: Dataset focused on human pose estimation.
- **[Leeds Sports Pose Extended](https://github.com/axelcarlier/lsp)**: 10,000 sports images with up to 14 human joint annotations.
