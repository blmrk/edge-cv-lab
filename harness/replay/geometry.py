from __future__ import annotations

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
