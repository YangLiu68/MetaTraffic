"""City blocks, parks and building lots (port of impl/graph.ts + impl/polygon_finder.ts).

The TypeScript version built its own road graph (isect + a quadtree) and walked it, turning
right at every junction, to find the faces between roads. Here shapely nodes the road lines at
their intersections and ``polygonize`` returns the same faces; dangling dead-end roads are
ignored either way.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from shapely import LineString
from shapely.ops import polygonize, unary_union

from .polygon_util import average_point, resize_geometry, subdivide_polygon
from .tensor_field import TensorField

Vec = tuple[float, float]


@dataclass
class PolygonParams:
    max_length: int = 20  # faces with this many corners or more are skipped
    min_area: float = 80.0
    shrink_spacing: float = 4.0
    chance_no_divide: float = 1.0


def find_polygons(
    streamlines: list[list[Vec]], params: PolygonParams, tensor_field: TensorField
) -> list[list[Vec]]:
    """Faces enclosed by the roads, on land and outside parks."""
    lines = [LineString(s) for s in streamlines if len(set(s)) >= 2]
    if not lines:
        return []
    noded = unary_union(lines)
    parts = list(getattr(noded, "geoms", [noded]))
    out: list[list[Vec]] = []
    for face in polygonize(parts):
        ring = [(float(x), float(y)) for x, y in list(face.exterior.coords)[:-1]]
        if len(ring) >= params.max_length:
            continue
        ax, ay = average_point(ring)
        if tensor_field.on_land(ax, ay) and not tensor_field.in_parks(ax, ay):
            out.append(ring)
    return out


def shrink_polygons(polygons: list[list[Vec]], spacing: float) -> list[list[Vec]]:
    """Inset every polygon so its edges sit ``spacing`` back from the road centrelines."""
    out = []
    for p in polygons:
        shrunk = resize_geometry(p, -spacing)
        if shrunk:
            out.append(shrunk)
    return out


def divide_polygons(
    polygons: list[list[Vec]], params: PolygonParams, rng: random.Random
) -> list[list[Vec]]:
    """Split blocks into building lots; a few blocks are kept whole."""
    out: list[list[Vec]] = []
    for p in polygons:
        if params.chance_no_divide > 0 and rng.random() < params.chance_no_divide:
            out.append(p)
            continue
        out.extend(subdivide_polygon(p, params.min_area, rng))
    return out
