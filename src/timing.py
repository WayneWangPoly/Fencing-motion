"""Timing utilities for movement onset / event detection."""

from __future__ import annotations

from typing import List


def detect_start_frame(positions: List[float], threshold: float) -> int:
    """Find first frame where motion exceeds threshold from baseline.

    Baseline is median-like early value using first 5 frames mean.
    """
    if not positions:
        return -1

    window = positions[: min(5, len(positions))]
    baseline = sum(window) / len(window)

    for idx, value in enumerate(positions):
        if abs(value - baseline) >= threshold:
            return idx
    return -1


def detect_landing_frame(positions: List[float]) -> int:
    """Approximate landing frame from foot trajectory.

    We detect highest speed region then choose first strong deceleration frame.
    Works best when front ankle has clear swing then plant.
    """
    if len(positions) < 4:
        return -1

    speeds = [abs(positions[i] - positions[i - 1]) for i in range(1, len(positions))]
    peak_speed = max(speeds)
    if peak_speed == 0:
        return -1

    for i in range(1, len(speeds)):
        if speeds[i - 1] > 0.6 * peak_speed and speeds[i] < 0.25 * peak_speed:
            return i + 1

    return speeds.index(peak_speed) + 1
