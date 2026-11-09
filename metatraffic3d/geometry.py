"""Small 2D polyline helpers (metres, x east, y north)."""

from __future__ import annotations

import math

import numpy as np

Vec = tuple[float, float]


class Polyline:
    """A polyline with arc-length lookup. ``pts`` is an (n, 2) float array, n >= 2."""

    __slots__ = ("pts", "s", "length", "_dirs")

    def __init__(self, pts) -> None:
        p = np.asarray(pts, dtype=float).reshape(-1, 2)
        if len(p) < 2:
            raise ValueError("polyline needs at least 2 points")
        seg = np.diff(p, axis=0)
        seg_len = np.hypot(seg[:, 0], seg[:, 1])
        keep = np.concatenate([[True], seg_len > 1e-9])
        if keep.sum() < 2:  # all points coincide: make a tiny segment so maths stays finite
            p = np.array([p[0], p[0] + (1e-6, 0.0)])
        else:
            p = p[keep]
        seg = np.diff(p, axis=0)
        seg_len = np.hypot(seg[:, 0], seg[:, 1])
        self.pts = p
        self.s = np.concatenate([[0.0], np.cumsum(seg_len)])
        self.length = float(self.s[-1])
        self._dirs = seg / seg_len[:, None]

    def __len__(self) -> int:
        return len(self.pts)

    def _seg(self, s: float) -> int:
        i = int(np.searchsorted(self.s, s, side="right")) - 1
        return min(max(i, 0), len(self.pts) - 2)

    def point_at(self, s: float) -> Vec:
        s = min(max(s, 0.0), self.length)
        i = self._seg(s)
        d = s - self.s[i]
        return (float(self.pts[i, 0] + self._dirs[i, 0] * d), float(self.pts[i, 1] + self._dirs[i, 1] * d))

    def heading_at(self, s: float) -> float:
        i = self._seg(min(max(s, 0.0), self.length))
        return math.atan2(self._dirs[i, 1], self._dirs[i, 0])

    def pose_at(self, s: float) -> tuple[float, float, float]:
        x, y = self.point_at(s)
        return x, y, self.heading_at(s)

    @property
    def start_heading(self) -> float:
        return math.atan2(self._dirs[0, 1], self._dirs[0, 0])

    @property
    def end_heading(self) -> float:
        return math.atan2(self._dirs[-1, 1], self._dirs[-1, 0])

    def project(self, p: Vec) -> tuple[float, float]:
        """Arc length of the closest point and the signed lateral offset (+ = left)."""
        a = self.pts[:-1]
        d = self._dirs
        seg_len = np.diff(self.s)
        rel = np.asarray(p, dtype=float) - a
        t = np.clip((rel * d).sum(1), 0.0, seg_len)
        closest = a + d * t[:, None]
        dist2 = ((np.asarray(p) - closest) ** 2).sum(1)
        i = int(np.argmin(dist2))
        lat = d[i, 0] * rel[i, 1] - d[i, 1] * rel[i, 0]
        return float(self.s[i] + t[i]), float(lat)

    def sub(self, s0: float, s1: float) -> np.ndarray:
        """Points of the piece between arc lengths s0 < s1."""
        s0 = min(max(s0, 0.0), self.length)
        s1 = min(max(s1, s0), self.length)
        inner = self.pts[(self.s > s0) & (self.s < s1)]
        return np.vstack([self.point_at(s0), inner, self.point_at(s1)])

    def reversed(self) -> Polyline:
        return Polyline(self.pts[::-1])

    def resample(self, step: float) -> np.ndarray:
        n = max(2, int(math.ceil(self.length / step)) + 1)
        return np.array([self.point_at(s) for s in np.linspace(0.0, self.length, n)])


def offset_points(pts: np.ndarray, d: float) -> np.ndarray:
    """Offset a polyline by ``d`` to the RIGHT of its direction (mitred, miter capped at 3x)."""
    p = np.asarray(pts, dtype=float)
    seg = np.diff(p, axis=0)
    n = np.hypot(seg[:, 0], seg[:, 1])
    n[n < 1e-12] = 1e-12
    t = seg / n[:, None]
    right = np.stack([t[:, 1], -t[:, 0]], axis=1)  # right normal of each segment
    normals = np.empty_like(p)
    normals[0] = right[0]
    normals[-1] = right[-1]
    if len(p) > 2:
        m = right[:-1] + right[1:]
        mn = np.hypot(m[:, 0], m[:, 1])
        mn[mn < 1e-9] = 1e-9
        m = m / mn[:, None]
        cos_half = (m * right[:-1]).sum(1)
        scale = 1.0 / np.clip(cos_half, 1.0 / 3.0, None)
        normals[1:-1] = m * scale[:, None]
    return p + normals * d


def hermite(p0: Vec, h0: float, p1: Vec, h1: float, n: int = 12, scale: float = 0.5) -> np.ndarray:
    """Cubic Hermite curve from p0 (heading h0) to p1 (heading h1)."""
    chord = math.dist(p0, p1)
    k = max(chord * scale * 2.0, 1e-3)
    m0 = np.array([math.cos(h0), math.sin(h0)]) * k
    m1 = np.array([math.cos(h1), math.sin(h1)]) * k
    a, b = np.asarray(p0, float), np.asarray(p1, float)
    t = np.linspace(0.0, 1.0, n)[:, None]
    h00 = 2 * t**3 - 3 * t**2 + 1
    h10 = t**3 - 2 * t**2 + t
    h01 = -2 * t**3 + 3 * t**2
    h11 = t**3 - t**2
    return h00 * a + h10 * m0 + h01 * b + h11 * m1


def wrap_angle(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def min_turn_radius(pts: np.ndarray) -> float:
    """Smallest circumradius over consecutive point triples (inf for straight lines)."""
    p = np.asarray(pts, dtype=float)
    if len(p) < 3:
        return math.inf
    a, b, c = p[:-2], p[1:-1], p[2:]
    ab = np.hypot(*(b - a).T)
    bc = np.hypot(*(c - b).T)
    ca = np.hypot(*(a - c).T)
    cross = np.abs((b - a)[:, 0] * (c - a)[:, 1] - (b - a)[:, 1] * (c - a)[:, 0])
    with np.errstate(divide="ignore", invalid="ignore"):
        r = ab * bc * ca / (2.0 * cross)
    r = r[np.isfinite(r)]
    return float(r.min()) if len(r) else math.inf
