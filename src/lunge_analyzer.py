"""Core lunge analysis logic for v1 prototype."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .geometry import calculate_angle


def get_coord(landmark, axis: str) -> float:
    """
    Support both landmark formats:
    - dict: {"x": ..., "y": ..., "z": ..., "visibility": ...}
    - list/tuple: [x, y, z, visibility]
    """
    if isinstance(landmark, dict):
        return float(landmark[axis])

    axis_index = {
        "x": 0,
        "y": 1,
        "z": 2,
        "visibility": 3,
    }
    return float(landmark[axis_index[axis]])


def moving_average(values: List[float], window: int = 5) -> List[float]:
    """
    Simple moving average smoothing.
    """
    if not values:
        return []

    if window <= 1:
        return values[:]

    half = window // 2
    smoothed: List[float] = []

    for i in range(len(values)):
        start = max(0, i - half)
        end = min(len(values), i + half + 1)
        smoothed.append(sum(values[start:end]) / (end - start))

    return smoothed


def velocities(values: List[float]) -> List[float]:
    """
    Per-frame absolute velocity from normalized coordinate sequence.
    velocity[i] = abs(values[i] - values[i - 1])
    velocity[0] = 0
    """
    if not values:
        return []

    result = [0.0]
    for i in range(1, len(values)):
        result.append(abs(values[i] - values[i - 1]))

    return result


def first_frame_above_threshold(
    values: List[float],
    start: int,
    end: int,
    threshold: float,
) -> int:
    """
    Find first frame in [start, end] where per-frame movement exceeds threshold.
    """
    if not values:
        return -1

    start = max(1, start)
    end = min(end, len(values) - 1)

    for i in range(start, end + 1):
        if abs(values[i] - values[i - 1]) >= threshold:
            return i

    return -1


def detect_lunge_segment_from_velocity(
    ankle_velocity: List[float],
) -> Dict[str, Any]:
    """
    Detect likely lunge segment by finding the strongest ankle movement peak.

    This is better than treating the first movement or first stop as the lunge.
    """
    if len(ankle_velocity) < 5:
        return {
            "success": False,
            "peak_frame": -1,
            "segment_start": -1,
            "segment_end": -1,
            "max_velocity": 0.0,
            "high_threshold": 0.0,
            "low_threshold": 0.0,
        }

    max_velocity = max(ankle_velocity)
    peak_frame = ankle_velocity.index(max_velocity)

    high_threshold = max(0.003, max_velocity * 0.35)
    low_threshold = max(0.0015, max_velocity * 0.15)

    if max_velocity < high_threshold:
        return {
            "success": False,
            "peak_frame": peak_frame,
            "segment_start": -1,
            "segment_end": -1,
            "max_velocity": max_velocity,
            "high_threshold": high_threshold,
            "low_threshold": low_threshold,
        }

    # Search backward from peak until movement becomes quiet.
    segment_start = peak_frame
    while segment_start > 1 and ankle_velocity[segment_start] > low_threshold:
        segment_start -= 1

    # Search forward from peak until movement becomes quiet.
    segment_end = peak_frame
    while segment_end < len(ankle_velocity) - 1 and ankle_velocity[segment_end] > low_threshold:
        segment_end += 1

    # Make sure the segment is not too tiny.
    if segment_end - segment_start < 3:
        return {
            "success": False,
            "peak_frame": peak_frame,
            "segment_start": segment_start,
            "segment_end": segment_end,
            "max_velocity": max_velocity,
            "high_threshold": high_threshold,
            "low_threshold": low_threshold,
        }

    return {
        "success": True,
        "peak_frame": peak_frame,
        "segment_start": segment_start,
        "segment_end": segment_end,
        "max_velocity": max_velocity,
        "high_threshold": high_threshold,
        "low_threshold": low_threshold,
    }


class LungeAnalyzer:
    """Analyze sequence of pose landmarks.

    Assumptions:
    - Side-view clip with one fencer and visible body joints.
    - Front side inferred by larger wrist horizontal excursion.
    """

    def analyze(self, landmark_frames: List[Optional[Dict[str, Any]]]) -> Dict[str, Any]:
        valid = [lm for lm in landmark_frames if lm is not None]

        if len(valid) < 5:
            return {
                "error": "Not enough valid pose frames for analysis.",
                "metrics": {
                    "number_of_valid_frames": len(valid),
                },
                "issues": ["Pose detection insufficient; improve framing/lighting."],
                "correction_plan": ["Record again from side view with whole body visible."],
            }

        left_wrist_x = [get_coord(lm["left_wrist"], "x") for lm in valid]
        right_wrist_x = [get_coord(lm["right_wrist"], "x") for lm in valid]

        left_disp = max(left_wrist_x) - min(left_wrist_x)
        right_disp = max(right_wrist_x) - min(right_wrist_x)
        front_side = "left" if left_disp > right_disp else "right"

        wrist_x_raw = [get_coord(lm[f"{front_side}_wrist"], "x") for lm in valid]
        ankle_x_raw = [get_coord(lm[f"{front_side}_ankle"], "x") for lm in valid]
        nose_y = [get_coord(lm["nose"], "y") for lm in valid]

        wrist_x = moving_average(wrist_x_raw, window=5)
        ankle_x = moving_average(ankle_x_raw, window=5)

        wrist_velocity = velocities(wrist_x)
        ankle_velocity = velocities(ankle_x)

        max_wrist_velocity = max(wrist_velocity) if wrist_velocity else 0.0
        max_ankle_velocity = max(ankle_velocity) if ankle_velocity else 0.0

        segment = detect_lunge_segment_from_velocity(ankle_velocity)

        if not segment["success"]:
            metrics = {
                "front_side_estimate": front_side,
                "number_of_valid_frames": len(valid),
                "wrist_start_frame": -1,
                "foot_start_frame": -1,
                "landing_frame": -1,
                "hand_starts_before_foot": False,
                "max_ankle_velocity": round(max_ankle_velocity, 6),
                "max_wrist_velocity": round(max_wrist_velocity, 6),
                "ankle_peak_frame": segment["peak_frame"],
                "lunge_segment_start": segment["segment_start"],
                "lunge_segment_end": segment["segment_end"],
                "ankle_high_threshold": round(segment["high_threshold"], 6),
                "ankle_low_threshold": round(segment["low_threshold"], 6),
            }

            return {
                "total_score": 0,
                "metrics": metrics,
                "issues": ["Lunge segment could not be reliably detected."],
                "correction_plan": [
                    "Record a shorter side-view clip containing only guard, lunge, and recovery.",
                    "Make sure the whole body and front foot remain visible.",
                ],
            }

        segment_start = int(segment["segment_start"])
        segment_end = int(segment["segment_end"])
        ankle_peak_frame = int(segment["peak_frame"])

        foot_start = segment_start
        landing = segment_end

        wrist_threshold = max(0.003, max_wrist_velocity * 0.25)

        wrist_start = first_frame_above_threshold(
            wrist_x,
            start=max(1, segment_start - 10),
            end=segment_end,
            threshold=wrist_threshold,
        )

        hand_before_foot = (
            wrist_start != -1
            and foot_start != -1
            and wrist_start <= foot_start
        )

        landing_lm = valid[landing] if 0 <= landing < len(valid) else valid[-1]

        knee_angle = calculate_angle(
            landing_lm[f"{front_side}_hip"],
            landing_lm[f"{front_side}_knee"],
            landing_lm[f"{front_side}_ankle"],
        )

        shoulder = landing_lm[f"{front_side}_shoulder"]
        hip = landing_lm[f"{front_side}_hip"]

        hip_x = get_coord(hip, "x")
        hip_y = get_coord(hip, "y")

        vertical_ref = (hip_x, hip_y - 1.0)
        torso_lean = calculate_angle(vertical_ref, hip, shoulder)

        head_vertical_range = max(nose_y) - min(nose_y)

        metrics = {
            "front_side_estimate": front_side,
            "number_of_valid_frames": len(valid),

            "wrist_start_frame": wrist_start,
            "foot_start_frame": foot_start,
            "landing_frame": landing,
            "hand_starts_before_foot": hand_before_foot,

            "front_knee_angle_at_landing_deg": round(knee_angle, 2) if knee_angle is not None else None,
            "torso_lean_angle_at_landing_deg": round(torso_lean, 2) if torso_lean is not None else None,
            "head_vertical_movement_range_norm": round(head_vertical_range, 4),

            "max_ankle_velocity": round(max_ankle_velocity, 6),
            "max_wrist_velocity": round(max_wrist_velocity, 6),
            "ankle_peak_frame": ankle_peak_frame,
            "lunge_segment_start": segment_start,
            "lunge_segment_end": segment_end,
            "ankle_high_threshold": round(segment["high_threshold"], 6),
            "ankle_low_threshold": round(segment["low_threshold"], 6),
            "wrist_threshold": round(wrist_threshold, 6),
        }

        return self._evaluate(metrics)

    def _evaluate(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        issues: List[str] = []
        correction_plan: List[str] = []
        score = 100

        if metrics.get("wrist_start_frame", -1) == -1:
            issues.append("Weapon hand start could not be clearly detected.")
            correction_plan.append("Record a clearer side-view clip with the weapon arm visible from the start.")
            score -= 12
        elif not metrics.get("hand_starts_before_foot", False):
            issues.append("Hand appears to start late relative to front foot.")
            correction_plan.append("Practice initiating extension just before front foot push.")
            score -= 18

        knee = metrics.get("front_knee_angle_at_landing_deg")
        if knee is not None and (knee < 100 or knee > 150):
            issues.append("Front knee angle at landing looks outside typical stable lunge range.")
            correction_plan.append("Aim for a controlled landing with front knee roughly bent over front foot.")
            score -= 15

        lean = metrics.get("torso_lean_angle_at_landing_deg")
        if lean is not None and lean > 35:
            issues.append("Torso lean may be excessive at landing.")
            correction_plan.append("Keep shoulders quieter and drive from legs to reduce over-leaning.")
            score -= 12

        head_move = metrics.get("head_vertical_movement_range_norm")
        if head_move is not None and head_move > 0.18:
            issues.append("Head level changes a lot during the action.")
            correction_plan.append("Work on smoother level change; avoid bouncing before or through the lunge.")
            score -= 8

        segment_length = metrics.get("lunge_segment_end", -1) - metrics.get("lunge_segment_start", -1)
        if segment_length > 0 and segment_length < 3:
            issues.append("Detected lunge segment is very short and may be unreliable.")
            correction_plan.append("Use a clearer clip with a full preparation, lunge, and landing.")
            score -= 8

        if not issues:
            issues.append("No major beginner-level issues detected from current heuristics.")
            correction_plan.append("Continue practicing timing and consistency; record another angle for confirmation.")

        return {
            "total_score": max(score, 0),
            "metrics": metrics,
            "issues": issues,
            "correction_plan": correction_plan,
        }
