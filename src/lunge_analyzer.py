"""Core lunge analysis logic for v1 prototype."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .geometry import calculate_angle
from .timing import detect_start_frame


def get_coord(landmark: Any, axis: str) -> float:
    """Get a landmark coordinate from dict or tuple/list formats."""
    if axis not in {"x", "y"}:
        raise ValueError(f"Unsupported axis: {axis}")
    if isinstance(landmark, dict):
        return float(landmark[axis])
    idx = 0 if axis == "x" else 1
    return float(landmark[idx])


def to_xy(landmark: Any) -> Tuple[float, float]:
    return get_coord(landmark, "x"), get_coord(landmark, "y")


def moving_average(values: List[float], window: int = 5) -> List[float]:
    if not values:
        return []
    if window < 1:
        return values

    radius = window // 2
    smoothed: List[float] = []
    for i in range(len(values)):
        start = max(0, i - radius)
        end = min(len(values), i + radius + 1)
        smoothed.append(sum(values[start:end]) / (end - start))
    return smoothed


class LungeAnalyzer:
    def analyze(self, landmark_frames: List[Optional[Dict[str, Any]]]) -> Dict[str, Any]:
        valid = [lm for lm in landmark_frames if lm is not None]
        if len(valid) < 5:
            return {
                "error": "Not enough valid pose frames for analysis.",
                "metrics": {"number_of_valid_frames": len(valid)},
                "issues": ["Pose detection insufficient; improve framing/lighting."],
                "correction_plan": ["Record again from side view with whole body visible."],
            }

        left_wrist_x = [get_coord(lm["left_wrist"], "x") for lm in valid]
        right_wrist_x = [get_coord(lm["right_wrist"], "x") for lm in valid]
        left_disp = max(left_wrist_x) - min(left_wrist_x)
        right_disp = max(right_wrist_x) - min(right_wrist_x)
        front_side = "left" if left_disp > right_disp else "right"

        wrist_x = [get_coord(lm[f"{front_side}_wrist"], "x") for lm in valid]
        ankle_x_raw = [get_coord(lm[f"{front_side}_ankle"], "x") for lm in valid]
        nose_y = [get_coord(lm["nose"], "y") for lm in valid]

        ankle_x = moving_average(ankle_x_raw, window=5)

        wrist_delta = [0.0] + [abs(wrist_x[i] - wrist_x[i - 1]) for i in range(1, len(wrist_x))]
        ankle_delta = [0.0] + [abs(ankle_x[i] - ankle_x[i - 1]) for i in range(1, len(ankle_x))]
        _nose_delta = [0.0] + [abs(nose_y[i] - nose_y[i - 1]) for i in range(1, len(nose_y))]

        max_ankle_velocity = max(ankle_delta) if ankle_delta else 0.0
        max_wrist_velocity = max(wrist_delta) if wrist_delta else 0.0

        high_threshold = max(0.003, max_ankle_velocity * 0.35)
        low_threshold = max(0.0015, max_ankle_velocity * 0.15)

        peak_frame = ankle_delta.index(max_ankle_velocity) if max_ankle_velocity > 0 else -1

        segment_start = -1
        segment_end = -1
        if peak_frame != -1 and max_ankle_velocity >= high_threshold:
            segment_start = peak_frame
            while segment_start > 0 and ankle_delta[segment_start] >= low_threshold:
                segment_start -= 1
            if ankle_delta[segment_start] < low_threshold:
                segment_start += 1

            segment_end = peak_frame
            while segment_end < len(ankle_delta) - 1 and ankle_delta[segment_end] >= low_threshold:
                segment_end += 1
            if ankle_delta[segment_end] < low_threshold:
                segment_end -= 1

        if segment_start == -1 or segment_end == -1 or segment_end <= segment_start:
            return {
                "total_score": 0,
                "metrics": {
                    "front_side_estimate": front_side,
                    "wrist_start_frame": -1,
                    "foot_start_frame": -1,
                    "landing_frame": -1,
                    "hand_starts_before_foot": False,
                    "front_knee_angle_at_landing_deg": 0.0,
                    "torso_lean_angle_at_landing_deg": 0.0,
                    "head_vertical_movement_range_px": round(max(nose_y) - min(nose_y), 2),
                    "number_of_valid_frames": len(valid),
                    "max_ankle_velocity": round(max_ankle_velocity, 6),
                    "max_wrist_velocity": round(max_wrist_velocity, 6),
                    "ankle_peak_frame": peak_frame,
                    "lunge_segment_start": segment_start,
                    "lunge_segment_end": segment_end,
                    "ankle_high_threshold": round(high_threshold, 6),
                    "ankle_low_threshold": round(low_threshold, 6),
                },
                "issues": ["Lunge segment could not be reliably detected."],
                "correction_plan": ["Record a clearer full-body side-view clip with one clean lunge action."],
            }

        foot_start = segment_start
        landing = segment_end

        wrist_window_start = max(0, segment_start - 10)
        wrist_window = wrist_x[wrist_window_start : segment_end + 1]
        wrist_threshold = max(0.003, max_wrist_velocity * 0.25)
        wrist_local_start = detect_start_frame(wrist_window, threshold=wrist_threshold)
        wrist_start = wrist_window_start + wrist_local_start if wrist_local_start != -1 else -1

        hand_before_foot = wrist_start != -1 and wrist_start < foot_start

        landing_lm = valid[landing]
        knee_angle = calculate_angle(
            to_xy(landing_lm[f"{front_side}_hip"]),
            to_xy(landing_lm[f"{front_side}_knee"]),
            to_xy(landing_lm[f"{front_side}_ankle"]),
        )

        shoulder = to_xy(landing_lm[f"{front_side}_shoulder"])
        hip = to_xy(landing_lm[f"{front_side}_hip"])
        vertical_ref = (hip[0], hip[1] - 100)
        torso_lean = calculate_angle(vertical_ref, hip, shoulder)

        metrics = {
            "front_side_estimate": front_side,
            "wrist_start_frame": wrist_start,
            "foot_start_frame": foot_start,
            "landing_frame": landing,
            "hand_starts_before_foot": hand_before_foot,
            "front_knee_angle_at_landing_deg": round(knee_angle, 2),
            "torso_lean_angle_at_landing_deg": round(torso_lean, 2),
            "head_vertical_movement_range_px": round(max(nose_y) - min(nose_y), 2),
            "number_of_valid_frames": len(valid),
            "max_ankle_velocity": round(max_ankle_velocity, 6),
            "max_wrist_velocity": round(max_wrist_velocity, 6),
            "ankle_peak_frame": peak_frame,
            "lunge_segment_start": segment_start,
            "lunge_segment_end": segment_end,
            "ankle_high_threshold": round(high_threshold, 6),
            "ankle_low_threshold": round(low_threshold, 6),
        }

        return self._evaluate(metrics)

    def _evaluate(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        issues: List[str] = []
        correction_plan: List[str] = []
        score = 100

        if not metrics.get("hand_starts_before_foot", False):
            issues.append("Hand appears to start late relative to front foot.")
            correction_plan.append("Practice initiating extension just before front foot push.")
            score -= 18

        knee = metrics["front_knee_angle_at_landing_deg"]
        if knee < 100 or knee > 150:
            issues.append("Front knee angle at landing looks outside typical stable lunge range.")
            correction_plan.append("Aim for a controlled landing with front knee roughly bent over front foot.")
            score -= 15

        lean = metrics["torso_lean_angle_at_landing_deg"]
        if lean > 35:
            issues.append("Torso lean may be excessive at landing.")
            correction_plan.append("Keep shoulders quieter and drive from legs to reduce over-leaning.")
            score -= 12

        head_move = metrics["head_vertical_movement_range_px"]
        if head_move > 70:
            issues.append("Head level changes a lot during the action.")
            correction_plan.append("Work on smoother level change; avoid bouncing before/through lunge.")
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
