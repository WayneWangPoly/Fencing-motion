from __future__ import annotations

import urllib.request
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision


MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)

MODEL_PATH = Path("models") / "pose_landmarker_lite.task"


LANDMARK_INDEXES = {
    "nose": 0,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
}


class PoseExtractor:
    """
    Pose extractor using the current MediaPipe Tasks PoseLandmarker API.

    This replaces the deprecated legacy MediaPipe Pose API.
    The first version supports one clearly visible fencer in side-view video.
    """

    def __init__(self, model_path: str | Path = MODEL_PATH) -> None:
        self.model_path = Path(model_path)
        self._ensure_model()

        options = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(self.model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.landmarker = vision.PoseLandmarker.create_from_options(options)

    def _ensure_model(self) -> None:
        self.model_path.parent.mkdir(parents=True, exist_ok=True)

        if self.model_path.exists():
            return

        print(f"Downloading MediaPipe pose model to {self.model_path}...")
        urllib.request.urlretrieve(MODEL_URL, self.model_path)

    def extract_frame_landmarks(
        self,
        frame_bgr: np.ndarray,
        timestamp_ms: int,
    ) -> dict[str, dict[str, float]] | None:
        """
        Extract selected pose landmarks from one OpenCV BGR frame.

        Returns:
            A dictionary of named landmarks, or None if no pose is detected.
        """
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=frame_rgb,
        )

        result = self.landmarker.detect_for_video(mp_image, timestamp_ms)

        if not result.pose_landmarks:
            return None

        landmarks = result.pose_landmarks[0]
        output: dict[str, dict[str, float]] = {}

        for name, index in LANDMARK_INDEXES.items():
            if index >= len(landmarks):
                continue

            lm = landmarks[index]
            output[name] = {
                "x": float(lm.x),
                "y": float(lm.y),
                "z": float(lm.z),
                "visibility": float(getattr(lm, "visibility", 1.0)),
            }

        return output

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        timestamp_ms: int | None = None,
    ) -> tuple[dict[str, dict[str, float]] | None, np.ndarray]:
        """
        Compatibility method for the original analyze_lunge.py flow.

        Returns:
            landmarks, overlay_frame
        """
        if timestamp_ms is None:
            # Fallback timestamp for old callers.
            # MediaPipe VIDEO mode requires monotonically increasing timestamps,
            # so we maintain an internal counter when the caller does not pass one.
            if not hasattr(self, "_frame_counter"):
                self._frame_counter = 0

            timestamp_ms = int((self._frame_counter / 30.0) * 1000)
            self._frame_counter += 1

        landmarks = self.extract_frame_landmarks(frame_bgr, timestamp_ms)
        overlay = self.draw_landmarks(frame_bgr, landmarks)

        return landmarks, overlay
    


    def extract(self, frames: list[np.ndarray], fps: float) -> list[dict[str, Any] | None]:
        """
        Extract landmarks for all frames.

        This method is designed to match the previous project interface.
        """
        results: list[dict[str, Any] | None] = []

        if fps <= 0:
            fps = 30.0

        for frame_index, frame in enumerate(frames):
            timestamp_ms = int((frame_index / fps) * 1000)
            landmarks = self.extract_frame_landmarks(frame, timestamp_ms)
            results.append(landmarks)

        return results

    def extract_landmarks(
        self,
        frames: list[np.ndarray],
        fps: float,
    ) -> list[dict[str, Any] | None]:
        """
        Compatibility alias for possible existing caller names.
        """
        return self.extract(frames, fps)

    def draw_landmarks(
        self,
        frame_bgr: np.ndarray,
        landmarks: dict[str, dict[str, float]] | None,
    ) -> np.ndarray:
        """
        Draw a simple skeleton overlay without using deprecated MediaPipe drawing utilities.
        """
        output = frame_bgr.copy()

        if not landmarks:
            return output

        height, width = output.shape[:2]

        def point(name: str) -> tuple[int, int] | None:
            item = landmarks.get(name)
            if not item:
                return None
            return int(item["x"] * width), int(item["y"] * height)

        connections = [
            ("left_shoulder", "right_shoulder"),
            ("left_shoulder", "left_elbow"),
            ("left_elbow", "left_wrist"),
            ("right_shoulder", "right_elbow"),
            ("right_elbow", "right_wrist"),
            ("left_shoulder", "left_hip"),
            ("right_shoulder", "right_hip"),
            ("left_hip", "right_hip"),
            ("left_hip", "left_knee"),
            ("left_knee", "left_ankle"),
            ("right_hip", "right_knee"),
            ("right_knee", "right_ankle"),
        ]

        for a, b in connections:
            pa = point(a)
            pb = point(b)
            if pa and pb:
                cv2.line(output, pa, pb, (0, 255, 0), 2)

        for name in LANDMARK_INDEXES:
            p = point(name)
            if p:
                cv2.circle(output, p, 4, (0, 0, 255), -1)

        return output

    def close(self) -> None:
        self.landmarker.close()
