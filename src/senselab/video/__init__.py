"""This module contains implementation for video processing."""

try:
    from deepface import DeepFace

    # This is to prevent deepface from braking tensorflow
    # (see https://github.com/serengil/deepface/issues/1390:
    # Importing tensorflow before deepface breaks it)
except ImportError:
    pass
