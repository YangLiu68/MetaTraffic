"""Triangle-mesh helpers and the backend-neutral scene description."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import shapely
import shapely.errors
from shapely.geometry.base import BaseGeometry

from .geometry import offset_points

RGBA = tuple[float, float, float, float]


@dataclass
class Mesh:
    """One static mesh in world metres (z up), drawn with a single colour."""

    name: str
    vertices: np.ndarray  # (n, 3) float
    faces: np.ndarray  # (m, 3) int, counter-clockwise seen from outside
    rgba: RGBA
    group: str  # ground | road | marking | sidewalk | building | water | park | bridge | tree | prop

    @property
    def n_faces(self) -> int:
        return len(self.faces)


@dataclass
class Part:
    """One coloured piece of an asset in its local frame (x forward, z up, origin on the ground)."""

    vertices: np.ndarray
    faces: np.ndarray
    rgba: RGBA
    tintable: bool = False  # body paint that can be recoloured per instance


@dataclass
class Scene:
    meshes: list[Mesh] = field(default_factory=list)
    bounds: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)  # xmin, ymin, xmax, ymax

    def stats(self) -> dict:
        by: dict[str, int] = {}
        for m in self.meshes:
            by[m.group] = by.get(m.group, 0) + m.n_faces
        return {"meshes": len(self.meshes), "faces_by_group": by,
                "faces": int(sum(m.n_faces for m in self.meshes))}  # fmt: skip


# --------------------------------------------------------------------- building blocks
def merge(parts: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    vs, fs, off = [], [], 0
    for v, f in parts:
        if len(v) == 0 or len(f) == 0:
            continue
        vs.append(np.asarray(v, float))
        fs.append(np.asarray(f, np.int64) + off)
        off += len(v)
    if not vs:
        return np.zeros((0, 3)), np.zeros((0, 3), np.int64)
    return np.vstack(vs), np.vstack(fs)


def _cdt(geom: BaseGeometry) -> np.ndarray:
    """Constrained Delaunay triangles; robust to the odd polygon GEOS cannot triangulate."""
    try:
        return shapely.get_parts(shapely.constrained_delaunay_triangles(geom))
    except shapely.errors.GEOSException:
        pass
    out = []
    for part in shapely.get_parts(geom):
        for candidate in (part, part.buffer(0).simplify(0.02), part.buffer(0).simplify(0.2)):
            try:
                out.extend(shapely.get_parts(shapely.constrained_delaunay_triangles(candidate)))
                break
            except shapely.errors.GEOSException:
                continue
        else:  # last resort: unconstrained Delaunay of the outline, keep triangles inside
            dense = shapely.segmentize(part, 2.0)
            tri = shapely.get_parts(shapely.delaunay_triangles(dense))
            keep = shapely.contains(part, shapely.centroid(tri))
            out.extend(tri[keep])
    return np.asarray(out, dtype=object)


def triangulate(geom: BaseGeometry, z: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Flat, upward-facing triangles covering a (Multi)Polygon at height ``z``."""
    if geom is None or geom.is_empty:
        return np.zeros((0, 3)), np.zeros((0, 3), np.int64)
    tris = _cdt(geom)
    if len(tris) == 0:
        return np.zeros((0, 3)), np.zeros((0, 3), np.int64)
    coords = shapely.get_coordinates(tris).reshape(len(tris), 4, 2)[:, :3, :]
    a, b, c = coords[:, 0], coords[:, 1], coords[:, 2]
    cross = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    flip = cross < 0
    coords[flip] = coords[flip][:, ::-1]
    coords = coords[np.abs(cross) > 1e-9]
    flat = coords.reshape(-1, 2)
    uniq, inv = np.unique(np.round(flat, 4), axis=0, return_inverse=True)
    v = np.column_stack([uniq, np.full(len(uniq), z)])
    return v, inv.reshape(-1, 3).astype(np.int64)


def _rings(geom: BaseGeometry):
    for poly in shapely.get_parts(geom):
        if poly.geom_type != "Polygon" or poly.is_empty:
            continue
        poly = shapely.geometry.polygon.orient(poly, 1.0)  # exterior CCW, holes CW
        yield np.asarray(poly.exterior.coords)[:-1, :2]
        for hole in poly.interiors:
            yield np.asarray(hole.coords)[:-1, :2]


def walls(geom: BaseGeometry, z0: float, z1: float, inward: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Vertical quads along every ring of a polygon, facing outwards (or inwards)."""
    parts = []
    for ring in _rings(geom):
        n = len(ring)
        if n < 2:
            continue
        nxt = np.roll(ring, -1, axis=0)
        v = np.vstack([np.column_stack([ring, np.full(n, z0)]), np.column_stack([nxt, np.full(n, z0)]),
                       np.column_stack([nxt, np.full(n, z1)]), np.column_stack([ring, np.full(n, z1)])])  # fmt: skip
        i = np.arange(n)
        f = np.vstack([np.column_stack([i, i + n, i + 2 * n]), np.column_stack([i, i + 2 * n, i + 3 * n])])
        if inward:
            f = f[:, ::-1]
        parts.append((v, f))
    return merge(parts)


def extrude(geom: BaseGeometry, z0: float, z1: float) -> tuple[np.ndarray, np.ndarray]:
    """Prism: walls plus a flat top (no bottom; nothing looks up at it)."""
    return merge([walls(geom, z0, z1), triangulate(geom, z1)])


def ribbon(pts: np.ndarray, width: float, z: float) -> tuple[np.ndarray, np.ndarray]:
    """Flat strip of the given width along a polyline."""
    p = np.asarray(pts, float)
    if len(p) < 2:
        return np.zeros((0, 3)), np.zeros((0, 3), np.int64)
    left = offset_points(p, -width / 2)
    right = offset_points(p, width / 2)
    n = len(p)
    v = np.vstack([np.column_stack([left, np.full(n, z)]), np.column_stack([right, np.full(n, z)])])
    i = np.arange(n - 1)
    f = np.vstack([np.column_stack([i, i + n, i + n + 1]), np.column_stack([i, i + n + 1, i + 1])])
    return v, f


def box(sx: float, sy: float, sz: float, center=(0.0, 0.0, 0.0)) -> tuple[np.ndarray, np.ndarray]:
    hx, hy, hz = sx / 2, sy / 2, sz / 2
    v = np.array([[-hx, -hy, -hz], [hx, -hy, -hz], [hx, hy, -hz], [-hx, hy, -hz],
                  [-hx, -hy, hz], [hx, -hy, hz], [hx, hy, hz], [-hx, hy, hz]]) + np.asarray(center)  # fmt: skip
    f = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4],
                  [1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]])  # fmt: skip
    return v, f


def cylinder(r: float, h: float, n: int = 8, center=(0.0, 0.0, 0.0), axis: str = "z", r_top: float | None = None):
    """Closed cylinder (or cone/frustum with r_top) along an axis, centred at ``center``."""
    r_top = r if r_top is None else r_top
    t = np.linspace(0, 2 * math.pi, n, endpoint=False)
    bot = np.column_stack([np.cos(t) * r, np.sin(t) * r, np.full(n, -h / 2)])
    top = np.column_stack([np.cos(t) * r_top, np.sin(t) * r_top, np.full(n, h / 2)])
    v = np.vstack([bot, top, [[0, 0, -h / 2], [0, 0, h / 2]]])
    i = np.arange(n)
    j = (i + 1) % n
    f = np.vstack([np.column_stack([i, j, j + n]), np.column_stack([i, j + n, i + n]),
                   np.column_stack([np.full(n, 2 * n), j, i]), np.column_stack([np.full(n, 2 * n + 1), i + n, j + n])])  # fmt: skip
    if axis == "y":
        v = v[:, [0, 2, 1]] * np.array([1, -1, 1])
        f = f[:, ::-1]
    elif axis == "x":
        v = v[:, [2, 1, 0]] * np.array([-1, 1, 1])
        f = f[:, ::-1]
    return v + np.asarray(center), f


def icosphere(r: float, center=(0.0, 0.0, 0.0), squash: float = 1.0):
    p = (1 + 5**0.5) / 2
    v = np.array([[-1, p, 0], [1, p, 0], [-1, -p, 0], [1, -p, 0], [0, -1, p], [0, 1, p],
                  [0, -1, -p], [0, 1, -p], [p, 0, -1], [p, 0, 1], [-p, 0, -1], [-p, 0, 1]], float)  # fmt: skip
    v = v / np.linalg.norm(v, axis=1, keepdims=True) * r
    v[:, 2] *= squash
    f = np.array([[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9], [5, 11, 4],
                  [11, 10, 2], [10, 7, 6], [7, 1, 8], [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8],
                  [3, 8, 9], [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]])  # fmt: skip
    return v + np.asarray(center), f


def transform(v: np.ndarray, x: float, y: float, z: float, yaw: float, scale: float = 1.0) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return (np.asarray(v) * scale) @ rot.T + np.array([x, y, z])


def write_obj(path: str, v: np.ndarray, f: np.ndarray, uv: np.ndarray | None = None) -> None:
    with open(path, "w") as fh:
        fh.write("".join(f"v {a:.3f} {b:.3f} {c:.3f}\n" for a, b, c in v))
        if uv is None:
            fh.write("".join(f"f {a + 1} {b + 1} {c + 1}\n" for a, b, c in f))
        else:
            fh.write("".join(f"vt {a:.5f} {b:.5f}\n" for a, b in uv))
            fh.write("".join(f"f {a + 1}/{a + 1} {b + 1}/{b + 1} {c + 1}/{c + 1}\n" for a, b, c in f))


class Palette:
    """A small texture of flat colours, so a many-coloured model can be a single textured mesh."""

    def __init__(self, size: int = 32) -> None:
        self.size = size
        self.colors: dict[tuple, int] = {}

    def uv(self, rgba) -> tuple[float, float]:
        key = tuple(int(round(c * 255)) for c in rgba[:3])
        if key not in self.colors:
            if len(self.colors) >= self.size * self.size:
                key = min(self.colors, key=lambda k: sum((a - b) ** 2 for a, b in zip(k, key)))
            else:
                self.colors[key] = len(self.colors)
        i = self.colors[key]
        r, c = divmod(i, self.size)
        return (c + 0.5) / self.size, 1.0 - (r + 0.5) / self.size

    def merged(self, parts) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Merge parts into one mesh with per-vertex UVs pointing at each part's colour."""
        vs, fs, uvs, off = [], [], [], 0
        for p in parts:
            vs.append(p.vertices)
            fs.append(p.faces + off)
            uvs.append(np.tile(self.uv(p.rgba), (len(p.vertices), 1)))
            off += len(p.vertices)
        return np.vstack(vs), np.vstack(fs), np.vstack(uvs)

    def save(self, path: str) -> None:
        from PIL import Image

        img = np.zeros((self.size, self.size, 3), np.uint8)
        for (r, g, b), i in self.colors.items():
            row, col = divmod(i, self.size)
            img[row, col] = (r, g, b)
        Image.fromarray(img).save(path)
