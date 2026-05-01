"""Geometry helpers for 2D landmark analysis.

Assumption: points are OpenCV pixel-space tuples (x, y).
"""

from __future__ import annotations

import math
from typing import Tuple

Point = Tuple[float, float]


def calculate_angle(a: Point, b: Point, c: Point) -> float:
    """Return angle ABC in degrees.

    Uses vector BA and BC with clamp to avoid floating-point drift.
    Returns 0.0 if vectors are degenerate.
    """
    ba = (a[0] - b[0], a[1] - b[1])
    bc = (c[0] - b[0], c[1] - b[1])

    norm_ba = math.hypot(ba[0], ba[1])
    norm_bc = math.hypot(bc[0], bc[1])
    if norm_ba == 0 or norm_bc == 0:
        return 0.0

    dot = ba[0] * bc[0] + ba[1] * bc[1]
    cosine = max(-1.0, min(1.0, dot / (norm_ba * norm_bc)))
    return math.degrees(math.acos(cosine))


def midpoint(a: Point, b: Point) -> Point:
    """Return midpoint between two points."""
    return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)
