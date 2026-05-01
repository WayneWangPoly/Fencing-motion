"""Video IO helpers."""

from __future__ import annotations

import os
from typing import List, Tuple

import cv2
import numpy as np


Frame = np.ndarray


def read_video_frames(video_path: str) -> Tuple[List[Frame], float]:
    """Load all frames and FPS from an input video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Unable to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames: List[Frame] = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()
    if not frames:
        raise ValueError("No frames found in video.")

    return frames, float(fps)


def ensure_output_dirs(base_output_dir: str) -> Tuple[str, str]:
    """Create output directories and return path tuple.

    Returns: (annotated_frames_dir, output_dir)
    """
    os.makedirs(base_output_dir, exist_ok=True)
    annotated = os.path.join(base_output_dir, "annotated_frames")
    os.makedirs(annotated, exist_ok=True)
    return annotated, base_output_dir
