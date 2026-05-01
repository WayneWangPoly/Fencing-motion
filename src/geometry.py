import math


def get_coord(point, axis: str) -> float:
    """
    Support both point formats:
    - dict: {"x": ..., "y": ..., "z": ..., "visibility": ...}
    - list/tuple: [x, y, z, visibility] or (x, y)
    """
    if isinstance(point, dict):
        return float(point[axis])

    axis_index = {
        "x": 0,
        "y": 1,
        "z": 2,
        "visibility": 3,
    }
    return float(point[axis_index[axis]])


def calculate_angle(a, b, c):
    """
    Calculate angle ABC in degrees.

    Supports both:
    - dict landmarks with x/y keys
    - tuple/list points
    """
    ax, ay = get_coord(a, "x"), get_coord(a, "y")
    bx, by = get_coord(b, "x"), get_coord(b, "y")
    cx, cy = get_coord(c, "x"), get_coord(c, "y")

    ba = (ax - bx, ay - by)
    bc = (cx - bx, cy - by)

    dot_product = ba[0] * bc[0] + ba[1] * bc[1]
    magnitude_ba = math.sqrt(ba[0] ** 2 + ba[1] ** 2)
    magnitude_bc = math.sqrt(bc[0] ** 2 + bc[1] ** 2)

    if magnitude_ba == 0 or magnitude_bc == 0:
        return None

    cosine_angle = dot_product / (magnitude_ba * magnitude_bc)
    cosine_angle = max(-1.0, min(1.0, cosine_angle))

    return math.degrees(math.acos(cosine_angle))


def midpoint(a, b):
    """
    Calculate midpoint between two points.

    Returns tuple: (x, y)
    """
    ax, ay = get_coord(a, "x"), get_coord(a, "y")
    bx, by = get_coord(b, "x"), get_coord(b, "y")

    return ((ax + bx) / 2, (ay + by) / 2)
