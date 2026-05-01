"""Core lunge analysis logic for v1 prototype."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .geometry import calculate_angle
from .timing import detect_landing_frame, detect_start_frame


def _xy(point: Any) -> Tuple[float, float]:
    """Support both tuple points and dict landmarks with x/y keys."""
    if isinstance(point, dict):
        return float(point["x"]), float(point["y"])
    return float(point[0]), float(point[1])


class LungeAnalyzer:
    """Analyze sequence of pose landmarks.

    Assumptions:
    - Side-view clip with one fencer and visible body joints.
    - Front side inferred by larger wrist horizontal excursion.
    """

    def analyze(self, landmark_frames: List[Optional[Dict[str, tuple[float, float]]]]) -> Dict[str, Any]:
        valid = [lm for lm in landmark_frames if lm is not None]
        if len(valid) < 5:
            return {
                "error": "Not enough valid pose frames for analysis.",
                "metrics": {},
                "issues": ["Pose detection insufficient; improve framing/lighting."],
                "correction_plan": ["Record again from side view with whole body visible."],
            }

        left_wrist_x = [_xy(lm["left_wrist"])[0] for lm in valid]
        right_wrist_x = [_xy(lm["right_wrist"])[0] for lm in valid]

        left_disp = max(left_wrist_x) - min(left_wrist_x)
        right_disp = max(right_wrist_x) - min(right_wrist_x)
        front_side = "left" if left_disp > right_disp else "right"

        wrist_x = [_xy(lm[f"{front_side}_wrist"])[0] for lm in valid]
        ankle_x = [_xy(lm[f"{front_side}_ankle"])[0] for lm in valid]
        nose_y = [_xy(lm["nose"])[1] for lm in valid]

        wrist_start = detect_start_frame(wrist_x, threshold=12.0)
        foot_start = detect_start_frame(ankle_x, threshold=10.0)
        landing = detect_landing_frame(ankle_x)
        hand_before_foot = wrist_start != -1 and foot_start != -1 and wrist_start < foot_start

        landing_lm = valid[landing] if 0 <= landing < len(valid) else valid[-1]

        knee_angle = calculate_angle(
            _xy(landing_lm[f"{front_side}_hip"]),
            _xy(landing_lm[f"{front_side}_knee"]),
            _xy(landing_lm[f"{front_side}_ankle"]),
        )

        shoulder = _xy(landing_lm[f"{front_side}_shoulder"])
        hip = _xy(landing_lm[f"{front_side}_hip"])
        vertical_ref = (hip[0], hip[1] - 100)
        torso_lean = calculate_angle(vertical_ref, hip, shoulder)

        head_vertical_range = max(nose_y) - min(nose_y)

        metrics = {
            "front_side_estimate": front_side,
            "wrist_start_frame": wrist_start,
            "foot_start_frame": foot_start,
            "landing_frame": landing,
            "hand_starts_before_foot": hand_before_foot,
            "front_knee_angle_at_landing_deg": round(knee_angle, 2),
            "torso_lean_angle_at_landing_deg": round(torso_lean, 2),
            "head_vertical_movement_range_px": round(head_vertical_range, 2),
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
