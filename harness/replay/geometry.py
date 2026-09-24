from __future__ import annotations

import math

Point = tuple[float, float]
Polygon = list[Point]


def point_in_polygon(p: Point, poly: Polygon) -> bool:
    """Ray casting. Points exactly on an edge may land either side; the debounced counter absorbs that."""
    x, y = p
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            x_cross = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < x_cross:
                inside = not inside
    return inside


def distance_to_edge(p: Point, poly: Polygon) -> float:
    """Shortest distance from p to the polygon's boundary, inside or out."""
    x, y = p
    best = math.inf
    for (x1, y1), (x2, y2) in zip(poly, poly[1:] + poly[:1]):
        dx, dy = x2 - x1, y2 - y1
        t = max(0.0, min(1.0, ((x - x1) * dx + (y - y1) * dy) / ((dx * dx + dy * dy) or 1)))
        best = min(best, math.hypot(x - x1 - t * dx, y - y1 - t * dy))
    return best
