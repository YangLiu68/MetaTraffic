"""Polygon helpers (port of impl/polygon_util.ts).

The TypeScript version used jsts (a JavaScript port of JTS) and PolyK. Here shapely (GEOS, the
C++ port of JTS) does the buffering, polygonising and slicing.
"""

from __future__ import annotations

import math
import random

from shapely import LineString, Point, Polygon
from shapely.ops import polygonize, split, unary_union

Vec = tuple[float, float]


class PolygonTester:
    """Ray-casting point-in-polygon test with a bounding-box shortcut.

    Matches PolygonUtil.insidePolygon (pnpoly), including its treatment of an empty polygon
    (never inside).
    """

    __slots__ = ("pts", "xs", "ys", "minx", "miny", "maxx", "maxy")

    def __init__(self, pts: list[Vec]) -> None:
        self.pts = list(pts)
        self.xs = [p[0] for p in self.pts]
        self.ys = [p[1] for p in self.pts]
        if self.pts:
            self.minx, self.maxx = min(self.xs), max(self.xs)
            self.miny, self.maxy = min(self.ys), max(self.ys)
        else:
            self.minx = self.miny = math.inf
            self.maxx = self.maxy = -math.inf

    def contains(self, x: float, y: float) -> bool:
        if x < self.minx or x > self.maxx or y < self.miny or y > self.maxy:
            return False
        xs, ys = self.xs, self.ys
        inside = False
        j = len(xs) - 1
        for i in range(len(xs)):
            xi, yi, xj, yj = xs[i], ys[i], xs[j], ys[j]
            if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
                inside = not inside
            j = i
        return inside


def inside_polygon(point: Vec, polygon: list[Vec]) -> bool:
    return PolygonTester(polygon).contains(point[0], point[1])


def polygon_area(polygon: list[Vec]) -> float:
    total = 0.0
    n = len(polygon)
    for i in range(n):
        x0, y0 = polygon[i]
        x1, y1 = polygon[0 if i == n - 1 else i + 1]
        total += x0 * y1 * 0.5 - x1 * y0 * 0.5
    return abs(total)


def average_point(polygon: list[Vec]) -> Vec:
    if not polygon:
        return (0.0, 0.0)
    n = len(polygon)
    return (sum(p[0] for p in polygon) / n, sum(p[1] for p in polygon) / n)


def _largest_polygon(geom) -> Polygon | None:
    if geom.is_empty:
        return None
    if geom.geom_type == "Polygon":
        return geom
    polys = [g for g in getattr(geom, "geoms", []) if g.geom_type == "Polygon" and not g.is_empty]
    return max(polys, key=lambda g: g.area) if polys else None


def resize_geometry(geometry: list[Vec], spacing: float, is_polygon: bool = True) -> list[Vec]:
    """Grow (spacing > 0) or shrink (spacing < 0) a polygon, or buffer a polyline (flat caps).

    Returns the outline of the result, or [] when it vanishes. jsts returned every coordinate of
    the result; for the rare multi-part result this keeps only the largest part instead.
    """
    try:
        if is_polygon:
            if len(geometry) < 3:
                return []
            geom = Polygon(geometry)
        else:
            if len(geometry) < 2:
                return []
            geom = LineString(geometry)
        resized = geom.buffer(spacing, cap_style="flat")
        if resized.is_empty or not resized.is_simple:
            return []
        poly = _largest_polygon(resized)
        if poly is None:
            return []
        return [(float(x), float(y)) for x, y in poly.exterior.coords]
    except (ValueError, TypeError):
        return []


def line_rectangle_polygon_intersection(origin: Vec, dims: Vec, line: list[Vec]) -> list[Vec]:
    """Split the world rectangle with a polyline and return the smaller piece (used for the sea)."""
    ox, oy = origin
    w, h = dims
    rect = Polygon([(ox, oy), (ox + w, oy), (ox + w, oy + h), (ox, oy + h)])
    if len(line) < 2:
        return []
    union = unary_union([rect.exterior, LineString(line)])
    polys = list(polygonize(union))
    if not polys:
        return []
    smallest = min(polys, key=lambda p: p.area)
    return [(float(x), float(y)) for x, y in smallest.exterior.coords]


def slice_polygon(polygon: list[Vec], a: Vec, b: Vec) -> list[list[Vec]]:
    """Cut a polygon with the segment a-b, like PolyK.Slice.

    As in PolyK, nothing is cut when either end of the segment lies inside the polygon; the
    polygon is then returned unchanged.
    """
    poly = Polygon(polygon)
    if not poly.is_valid:
        poly = poly.buffer(0)
        poly = _largest_polygon(poly)
        if poly is None:
            return []
    if poly.contains(Point(a)) or poly.contains(Point(b)):
        return [list(polygon)]
    pieces = split(poly, LineString([a, b]))
    out = []
    for g in pieces.geoms:
        if g.geom_type == "Polygon" and not g.is_empty:
            out.append([(float(x), float(y)) for x, y in list(g.exterior.coords)[:-1]])
    return out


def _cut_misses(polygon: list[Vec], a: Vec, b: Vec) -> bool:
    """True when segment a-b cannot slice the polygon: all corners lie on one side of its
    line, or an end of the segment is inside the polygon (PolyK then refuses to cut)."""
    ux, uy = b[0] - a[0], b[1] - a[1]
    left = right = False
    for x, y in polygon:
        c = ux * (y - a[1]) - uy * (x - a[0])
        if c > 0:
            left = True
        elif c < 0:
            right = True
        if left and right:
            break
    if not (left and right):
        return True
    tester = PolygonTester(polygon)
    return tester.contains(*a) or tester.contains(*b)


def subdivide_polygon(
    p: list[Vec], min_area: float, rng: random.Random, max_tries: int = 200
) -> list[list[Vec]]:
    """Recursively cut a polygon across its longest side until pieces reach ``min_area``.

    Port of PolygonUtil.subdividePolygon, keeping its quirks: the cut goes through
    (p0 + p1) * deviation (not the point at ``deviation`` along the side), and when the cut
    misses the polygon the JS version recursed with a new random deviation until it hit or the
    stack overflowed (then the lot was dropped). That retry is a bounded loop here.
    """
    for _ in range(max_tries):
        area = polygon_area(p)
        if area < 0.5 * min_area:
            return []

        longest_len = 0.0
        longest = (p[0], p[1])
        perimeter = 0.0
        n = len(p)
        for i in range(n):
            q0, q1 = p[i], p[(i + 1) % n]
            side = math.hypot(q0[0] - q1[0], q0[1] - q1[1])
            perimeter += side
            if side > longest_len:
                longest_len = side
                longest = (q0, q1)

        # Shape index: drop slivers (rectangle ratio of about 1:4)
        if perimeter == 0 or area / (perimeter * perimeter) < 0.04:
            return []
        if area < 2 * min_area:
            return [p]

        deviation = rng.random() * 0.2 + 0.4
        (x0, y0), (x1, y1) = longest
        ax, ay = (x0 + x1) * deviation, (y0 + y1) * deviation
        dx, dy = x0 - x1, y0 - y1
        norm = math.hypot(dy, -dx) or 1.0
        px, py = dy / norm * 100.0, -dx / norm * 100.0
        a, b = (ax + px, ay + py), (ax - px, ay - py)
        if _cut_misses(p, a, b):
            continue  # cheap check for the common miss; same outcome as a failed slice
        try:
            sliced = slice_polygon(p, a, b)
        except (ValueError, TypeError):
            return []
        if not sliced:
            return []
        if len(sliced) == 1:
            continue  # missed: retry with a new deviation
        out: list[list[Vec]] = []
        for s in sliced:
            out.extend(subdivide_polygon(s, min_area, rng, max_tries))
        return out
    return []
