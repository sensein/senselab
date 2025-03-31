# Face Analysis

[![Tutorial](https://img.shields.io/badge/Tutorial-Click%20Here-blue?style=for-the-badge)](https://github.com/sensein/senselab/blob/main/tutorials/video/face_analysis.ipynb)

## Task Overview
Face analysis encompasses various tasks, including **face recognition**, **face verification**, **attribute analysis**, and **embedding extraction**. These tasks allow for identifying individuals, verifying identities, analyzing facial attributes (age, gender, emotion, race), and extracting unique facial embeddings useful for various downstream tasks.

`senselab` integrates **DeepFace**, a widely-used open-source library for face analysis, providing robust and versatile face recognition and analysis capabilities.

---

## Models

### [DeepFace](https://github.com/serengil/deepface)
DeepFace supports several face recognition models:
- **VGG-Face**
- **Facenet and Facenet512**
- **OpenFace**
- **DeepID**
- **Dlib**
- **ArcFace**
- **SFace**
- **GhostFaceNet**

Face detection backends include **OpenCV**, **RetinaFace**, **MTCNN**, **SSD**, **Dlib**, **MediaPipe**, and various YOLO variants (`yolov8`, `yolov11`).

## Evaluation

### Metrics
- When working with biometrics systems, one typically checks for:
-   **False Accept Rate (FAR)**: Probability that the system incorrectly accepts an unauthorized individual.
-   **False Reject Rate (FRR)**: Probability that the system incorrectly rejects an authorized individual.
-   **True Accept Rate (TAR)**: Probability that the system correctly accepts an authorized individual. Often used alongside FAR to show trade-offs.
-   **Equal Error Rate (EER)**: The threshold at which FAR equals FRR, used to evaluate overall system accuracy.
- **Cosine Similarity and Euclidean Distance**: Common metrics for evaluating embedding similarity. Lower distance (or higher similarity) indicates faces of the same person.

### Benchmark Datasets
- **[Labeled Faces in the Wild (LFW)](https://www.kaggle.com/datasets/jessicali9530/lfw-dataset)**: Standard benchmark for face verification tasks.

## Learn More:
- [DeepFace GitHub Repository](https://github.com/serengil/deepface)
