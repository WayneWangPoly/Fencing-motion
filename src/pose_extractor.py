"""MediaPipe Pose extraction and debug rendering."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

Point = Tuple[float, float]
LandmarkMap = Dict[str, Point]


class PoseExtractor:
    """Wrapper around MediaPipe Pose.

    Limitations:
    - Landmark names are a subset used by current analysis.
    - Missing detections for a frame return None.
    """

    def __init__(self) -> None:
        self.mp_pose = mp.solutions.pose
        self.mp_draw = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def _extract_named(self, landmarks, width: int, height: int) -> LandmarkMap:
        idx = self.mp_pose.PoseLandmark

        names = {
            "nose": idx.NOSE,
            "left_shoulder": idx.LEFT_SHOULDER,
            "right_shoulder": idx.RIGHT_SHOULDER,
            "left_elbow": idx.LEFT_ELBOW,
            "right_elbow": idx.RIGHT_ELBOW,
            "left_wrist": idx.LEFT_WRIST,
            "right_wrist": idx.RIGHT_WRIST,
            "left_hip": idx.LEFT_HIP,
            "right_hip": idx.RIGHT_HIP,
            "left_knee": idx.LEFT_KNEE,
            "right_knee": idx.RIGHT_KNEE,
            "left_ankle": idx.LEFT_ANKLE,
            "right_ankle": idx.RIGHT_ANKLE,
        }

        extracted: LandmarkMap = {}
        for name, lm_enum in names.items():
            lm = landmarks[lm_enum.value]
            extracted[name] = (lm.x * width, lm.y * height)
        return extracted

    def process_frame(self, frame: np.ndarray) -> Tuple[Optional[LandmarkMap], np.ndarray]:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.pose.process(rgb)
        overlay = frame.copy()

        if result.pose_landmarks:
            self.mp_draw.draw_landmarks(
                overlay,
                result.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
            )
            h, w = frame.shape[:2]
            return self._extract_named(result.pose_landmarks.landmark, w, h), overlay
        return None, overlay

    def close(self) -> None:
        self.pose.close()
