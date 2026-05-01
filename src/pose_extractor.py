"""MediaPipe Pose extraction using Tasks API (PoseLandmarker)."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple
from urllib.request import urlretrieve

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.framework.formats import landmark_pb2

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)
MODEL_PATH = Path("models/pose_landmarker_lite.task")

Point = Tuple[float, float]
LandmarkValue = Dict[str, float]
LandmarkMap = Dict[str, LandmarkValue]


class PoseExtractor:
    """Wrapper around MediaPipe Tasks PoseLandmarker.

    Limitations:
    - Landmark names are a subset used by current analysis.
    - Missing detections for a frame return None.
    """

    def __init__(self, model_path: Path = MODEL_PATH) -> None:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        if not model_path.exists():
            urlretrieve(MODEL_URL, model_path)

        BaseOptions = mp.tasks.BaseOptions
        PoseLandmarker = mp.tasks.vision.PoseLandmarker
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=VisionRunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.landmarker = PoseLandmarker.create_from_options(options)

        self._names = {
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

        self._pose_connections = tuple(mp.solutions.pose.POSE_CONNECTIONS)

    def _extract_named(self, landmarks, width: int, height: int) -> LandmarkMap:
        extracted: LandmarkMap = {}
        for name, idx in self._names.items():
            lm = landmarks[idx]
            extracted[name] = {
                "x": lm.x * width,
                "y": lm.y * height,
                "z": lm.z,
                "visibility": getattr(lm, "visibility", 1.0),
            }
        return extracted

    def _draw_landmarks(self, overlay: np.ndarray, landmarks) -> None:
        proto = landmark_pb2.NormalizedLandmarkList()
        proto.landmark.extend(
            landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z) for lm in landmarks
        )
        mp.solutions.drawing_utils.draw_landmarks(
            overlay,
            proto,
            self._pose_connections,
        )

    def process_frame(
        self, frame: np.ndarray, timestamp_ms: int
    ) -> Tuple[Optional[LandmarkMap], np.ndarray]:
        """Process one BGR frame and return landmark map and overlay.

        VIDEO running mode requires monotonically increasing timestamp_ms.
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.landmarker.detect_for_video(mp_image, timestamp_ms)
        overlay = frame.copy()

        if result.pose_landmarks:
            first_pose = result.pose_landmarks[0]
            self._draw_landmarks(overlay, first_pose)
            h, w = frame.shape[:2]
            return self._extract_named(first_pose, w, h), overlay
        return None, overlay

    def close(self) -> None:
        self.landmarker.close()
