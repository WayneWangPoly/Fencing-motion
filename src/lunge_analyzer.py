"""Core lunge analysis logic for v1 prototype."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .geometry import calculate_angle, midpoint
from .timing import detect_start_frame


def get_coord(landmark: Any, axis: str) -> float:
    if axis not in {"x", "y"}:
        raise ValueError(f"Unsupported axis: {axis}")
    if isinstance(landmark, dict):
        return float(landmark[axis])
    return float(landmark[0] if axis == "x" else landmark[1])


def to_xy(landmark: Any) -> Tuple[float, float]:
    return get_coord(landmark, "x"), get_coord(landmark, "y")


def moving_average(values: List[float], window: int = 5) -> List[float]:
    if not values:
        return []
    r = max(0, window // 2)
    out: List[float] = []
    for i in range(len(values)):
        s = max(0, i - r)
        e = min(len(values), i + r + 1)
        out.append(sum(values[s:e]) / (e - s))
    return out


class LungeAnalyzer:
    def _detect_segments(self, velocity: List[float], low_threshold: float, gap_merge: int = 2) -> List[Tuple[int, int]]:
        raw: List[Tuple[int, int]] = []
        start = None
        for i, v in enumerate(velocity):
            if v >= low_threshold and start is None:
                start = i
            if v < low_threshold and start is not None:
                raw.append((start, i - 1))
                start = None
        if start is not None:
            raw.append((start, len(velocity) - 1))

        if not raw:
            return []

        merged = [raw[0]]
        for s, e in raw[1:]:
            ps, pe = merged[-1]
            if s - pe <= gap_merge:
                merged[-1] = (ps, e)
            else:
                merged.append((s, e))
        return merged

    def _segment_features(
        self,
        seg: Tuple[int, int],
        valid: List[Dict[str, Any]],
        front_side: str,
        front_ankle_x: List[float],
        back_ankle_x: List[float],
        wrist_x: List[float],
        torso_mid_x: List[float],
        front_ankle_vel: List[float],
    ) -> Dict[str, Any]:
        s, e = seg
        peak = max(range(s, e + 1), key=lambda i: front_ankle_vel[i])

        front_disp = abs(front_ankle_x[e] - front_ankle_x[s])
        back_disp = abs(back_ankle_x[e] - back_ankle_x[s])

        pre = max(0, s - 2)
        post = min(len(valid) - 1, e + 2)
        stance_before = abs(front_ankle_x[pre] - back_ankle_x[pre])
        stance_after = abs(front_ankle_x[post] - back_ankle_x[post])
        stance_delta = stance_after - stance_before

        knee_before = calculate_angle(
            to_xy(valid[pre][f"{front_side}_hip"]),
            to_xy(valid[pre][f"{front_side}_knee"]),
            to_xy(valid[pre][f"{front_side}_ankle"]),
        )
        knee_after = calculate_angle(
            to_xy(valid[post][f"{front_side}_hip"]),
            to_xy(valid[post][f"{front_side}_knee"]),
            to_xy(valid[post][f"{front_side}_ankle"]),
        )
        knee_drop = max(0.0, knee_before - knee_after)

        near_s = max(0, s - 3)
        near_e = min(len(valid) - 1, e + 3)
        wrist_disp = abs(wrist_x[near_e] - wrist_x[near_s])
        torso_disp = abs(torso_mid_x[near_e] - torso_mid_x[near_s])

        max_vel = max(front_ankle_vel[s : e + 1]) if e >= s else 0.0

        # Penalty if back-foot movement mirrors front-foot movement (likely advance step)
        mirror_penalty = 0.0
        if front_disp > 0:
            ratio = back_disp / front_disp
            if ratio > 0.75:
                mirror_penalty = min(1.0, ratio - 0.75)

        lunge_score = (
            2.0 * front_disp
            + 1.5 * max_vel
            + 1.3 * max(0.0, stance_delta)
            + 0.8 * knee_drop
            + 0.9 * wrist_disp
            - 1.6 * mirror_penalty
        )

        return {
            "start": s,
            "end": e,
            "peak_frame": peak,
            "front_ankle_displacement": round(front_disp, 6),
            "back_ankle_displacement": round(back_disp, 6),
            "stance_width_before": round(stance_before, 6),
            "stance_width_after": round(stance_after, 6),
            "stance_width_delta": round(stance_delta, 6),
            "front_knee_angle_before": round(knee_before, 2),
            "front_knee_angle_after": round(knee_after, 2),
            "front_knee_angle_drop": round(knee_drop, 2),
            "wrist_displacement_near_segment": round(wrist_disp, 6),
            "torso_displacement_near_segment": round(torso_disp, 6),
            "max_front_ankle_velocity": round(max_vel, 6),
            "lunge_score": round(lunge_score, 6),
        }

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
        front_side = "left" if (max(left_wrist_x) - min(left_wrist_x)) > (max(right_wrist_x) - min(right_wrist_x)) else "right"
        back_side = "right" if front_side == "left" else "left"

        wrist_x = [get_coord(lm[f"{front_side}_wrist"], "x") for lm in valid]
        front_ankle_x = moving_average([get_coord(lm[f"{front_side}_ankle"], "x") for lm in valid], window=5)
        back_ankle_x = moving_average([get_coord(lm[f"{back_side}_ankle"], "x") for lm in valid], window=5)
        nose_y = [get_coord(lm["nose"], "y") for lm in valid]

        torso_mid_x: List[float] = []
        for lm in valid:
            sh = to_xy(lm[f"{front_side}_shoulder"])
            hp = to_xy(lm[f"{front_side}_hip"])
            torso_mid_x.append(midpoint(sh, hp)[0])

        front_ankle_vel = [0.0] + [abs(front_ankle_x[i] - front_ankle_x[i - 1]) for i in range(1, len(front_ankle_x))]
        wrist_vel = [0.0] + [abs(wrist_x[i] - wrist_x[i - 1]) for i in range(1, len(wrist_x))]

        max_ankle_vel = max(front_ankle_vel) if front_ankle_vel else 0.0
        low_thr = max(0.0015, max_ankle_vel * 0.15)

        raw_segments = self._detect_segments(front_ankle_vel, low_threshold=low_thr, gap_merge=2)
        movement_segments = [
            self._segment_features(
                seg,
                valid,
                front_side,
                front_ankle_x,
                back_ankle_x,
                wrist_x,
                torso_mid_x,
                front_ankle_vel,
            )
            for seg in raw_segments
        ]

        if not movement_segments:
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
                    "movement_segments": [],
                    "selected_lunge_segment_index": -1,
                    "selected_lunge_score": 0.0,
                    "lunge_segment_start": -1,
                    "lunge_segment_end": -1,
                    "lunge_peak_frame": -1,
                    "post_landing_torso_drift": False,
                    "post_landing_ankle_delta": 0.0,
                    "post_landing_torso_delta": 0.0,
                },
                "issues": ["Lunge segment could not be reliably detected."],
                "correction_plan": ["Record a clearer full-body side-view clip with one clean lunge action."],
            }

        selected_idx = max(range(len(movement_segments)), key=lambda i: movement_segments[i]["lunge_score"])
        selected = movement_segments[selected_idx]

        foot_start = selected["start"]
        landing = selected["end"]
        lunge_peak = selected["peak_frame"]

        w_start = max(0, foot_start - 3)
        w_end = landing
        wrist_window = wrist_x[w_start : w_end + 1]
        wrist_thr = max(0.003, (max(wrist_vel) if wrist_vel else 0.0) * 0.25)
        wrist_local = detect_start_frame(wrist_window, wrist_thr)
        wrist_start = w_start + wrist_local if wrist_local != -1 else -1

        hand_before_lunge = wrist_start != -1 and wrist_start < foot_start

        landing_lm = valid[landing]
        knee_angle = calculate_angle(
            to_xy(landing_lm[f"{front_side}_hip"]),
            to_xy(landing_lm[f"{front_side}_knee"]),
            to_xy(landing_lm[f"{front_side}_ankle"]),
        )
        shoulder = to_xy(landing_lm[f"{front_side}_shoulder"])
        hip = to_xy(landing_lm[f"{front_side}_hip"])
        torso_lean = calculate_angle((hip[0], hip[1] - 100), hip, shoulder)

        post_end = min(len(valid) - 1, landing + 8)
        ankle_post_delta = abs(front_ankle_x[post_end] - front_ankle_x[landing]) if post_end > landing else 0.0
        torso_post_delta = abs(torso_mid_x[post_end] - torso_mid_x[landing]) if post_end > landing else 0.0
        post_landing_torso_drift = ankle_post_delta < max(0.002, low_thr * 2.0) and torso_post_delta > max(0.004, low_thr * 3.0)

        metrics = {
            "front_side_estimate": front_side,
            "wrist_start_frame": wrist_start,
            "foot_start_frame": foot_start,
            "landing_frame": landing,
            "hand_starts_before_foot": hand_before_lunge,
            "front_knee_angle_at_landing_deg": round(knee_angle, 2),
            "torso_lean_angle_at_landing_deg": round(torso_lean, 2),
            "head_vertical_movement_range_px": round(max(nose_y) - min(nose_y), 2),
            "number_of_valid_frames": len(valid),
            "movement_segments": movement_segments,
            "selected_lunge_segment_index": selected_idx,
            "selected_lunge_score": selected["lunge_score"],
            "lunge_segment_start": foot_start,
            "lunge_segment_end": landing,
            "lunge_peak_frame": lunge_peak,
            "post_landing_torso_drift": post_landing_torso_drift,
            "post_landing_ankle_delta": round(ankle_post_delta, 6),
            "post_landing_torso_delta": round(torso_post_delta, 6),
            "wrist_start_frame_global": wrist_start,
            "wrist_start_frame_in_lunge": wrist_start,
            "foot_start_frame_in_lunge": foot_start,
            "hand_starts_before_foot_in_lunge": hand_before_lunge,
        }

        return self._evaluate(metrics)

    def _evaluate(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        issues: List[str] = []
        correction_plan: List[str] = []
        score = 100

        if not metrics.get("hand_starts_before_foot_in_lunge", False):
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

        if metrics.get("post_landing_torso_drift", False):
            issues.append("Upper body continues drifting forward after the front foot has landed.")
            correction_plan.append("Practice landing with the front foot and torso arriving together, then recover immediately.")
            score -= 10

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
