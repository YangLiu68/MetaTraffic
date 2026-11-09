"""Hilly terrain (think San Francisco): a seeded height field the whole city sits on.

Height = a sum of random Gaussian hills plus a gentle tilt, faded to sea level near water so
shores, river banks and bridges stay flat. ``height(x, y)`` is vectorised; ``grid()`` samples
it on a regular grid for the MuJoCo height field and the ground mesh.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import shapely


@dataclass
class TerrainConfig:
    amplitude: float = 45.0  # tallest hills, metres (0 = flat city)
    hill_size: tuple[float, float] = (220.0, 520.0)  # Gaussian sigma range, metres
    hills_per_km2: float = 1.6
    tilt: float = 0.004  # overall slope across the map
    shore_flat: float = 160.0  # distance from water over which hills fade in
    mask_cell: float = 25.0
    seed_offset: int = 211


class Terrain:
    def __init__(self, bounds, water=None, seed: int = 0, cfg: TerrainConfig | None = None) -> None:
        self.cfg = cfg or TerrainConfig()
        c = self.cfg
        rng = np.random.default_rng(seed + c.seed_offset)
        xmin, ymin, xmax, ymax = bounds
        self.bounds = bounds
        area_km2 = (xmax - xmin) * (ymax - ymin) / 1e6
        k = max(1, int(round(c.hills_per_km2 * area_km2)))
        self.cx = rng.uniform(xmin, xmax, k)
        self.cy = rng.uniform(ymin, ymax, k)
        self.sig = rng.uniform(*c.hill_size, k)
        self.amp = c.amplitude * rng.uniform(0.35, 1.0, k)
        ang = rng.uniform(0, 2 * math.pi)
        self.tilt = (math.cos(ang) * c.tilt, math.sin(ang) * c.tilt)
        self.x0, self.y0 = (xmin + xmax) / 2, (ymin + ymax) / 2
        # fade-to-flat mask near water, sampled on a coarse grid
        self.mx = np.arange(xmin - c.mask_cell, xmax + 2 * c.mask_cell, c.mask_cell)
        self.my = np.arange(ymin - c.mask_cell, ymax + 2 * c.mask_cell, c.mask_cell)
        if water is not None and not water.is_empty and c.amplitude > 0:
            gx, gy = np.meshgrid(self.mx, self.my)
            d = shapely.distance(shapely.points(gx.ravel(), gy.ravel()), water).reshape(gx.shape)
            fade = max(c.shore_flat, c.amplitude / 0.08)  # keep shore slopes under ~12%
            t = np.clip(d / fade, 0.0, 1.0)
            self.mask = t * t * (3 - 2 * t)
        else:
            self.mask = np.ones((len(self.my), len(self.mx)))
        self._base = 0.0
        if c.amplitude > 0:  # shift so the lowest land point sits at 0
            gx, gy = np.meshgrid(np.linspace(xmin, xmax, 60), np.linspace(ymin, ymax, 40))
            self._base = float(np.min(self._raw(gx, gy)))

    def _raw(self, x, y):
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        h = np.zeros(np.broadcast(x, y).shape)
        for cx, cy, s, a in zip(self.cx, self.cy, self.sig, self.amp):
            h = h + a * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * s * s))
        return h + self.tilt[0] * (x - self.x0) + self.tilt[1] * (y - self.y0)

    def _mask(self, x, y):
        c = self.cfg.mask_cell
        fx = np.clip((np.asarray(x, float) - self.mx[0]) / c, 0, len(self.mx) - 1.001)
        fy = np.clip((np.asarray(y, float) - self.my[0]) / c, 0, len(self.my) - 1.001)
        ix, iy = fx.astype(int), fy.astype(int)
        tx, ty = fx - ix, fy - iy
        m = self.mask
        return ((m[iy, ix] * (1 - tx) + m[iy, ix + 1] * tx) * (1 - ty)
                + (m[iy + 1, ix] * (1 - tx) + m[iy + 1, ix + 1] * tx) * ty)  # fmt: skip

    def fast_height(self, x, y):
        """Bilinear lookup in a 4 m grid of ``height`` (for placing many things per frame)."""
        if self.cfg.amplitude <= 0:
            return np.zeros(np.shape(x))
        if not hasattr(self, "_fx"):
            self._fx, self._fy, self._fh = self.grid(4.0)
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        fx = np.clip((x - self._fx[0]) / 4.0, 0, len(self._fx) - 1.001)
        fy = np.clip((y - self._fy[0]) / 4.0, 0, len(self._fy) - 1.001)
        ix, iy = fx.astype(int), fy.astype(int)
        tx, ty = fx - ix, fy - iy
        h = self._fh
        return ((h[iy, ix] * (1 - tx) + h[iy, ix + 1] * tx) * (1 - ty)
                + (h[iy + 1, ix] * (1 - tx) + h[iy + 1, ix + 1] * tx) * ty)  # fmt: skip

    def height(self, x, y):
        if self.cfg.amplitude <= 0:
            return np.zeros(np.broadcast(np.asarray(x), np.asarray(y)).shape) if np.ndim(x) else 0.0
        h = np.maximum(self._raw(x, y) - self._base, 0.0) * self._mask(x, y)
        return h if np.ndim(h) else float(h)

    __call__ = height

    def slope(self, x, y, heading: float, d: float = 3.0) -> float:
        """sin of the road grade along a heading."""
        c, s = math.cos(heading), math.sin(heading)
        return float((self.height(x + d * c, y + d * s) - self.height(x - d * c, y - d * s)) / (2 * d))

    def grid(self, spacing: float = 8.0):
        """(xs, ys, H[ny, nx]) covering the bounds."""
        xmin, ymin, xmax, ymax = self.bounds
        xs = np.arange(xmin, xmax + spacing, spacing)
        ys = np.arange(ymin, ymax + spacing, spacing)
        gx, gy = np.meshgrid(xs, ys)
        return xs, ys, np.asarray(self.height(gx, gy))

    def stats(self) -> dict:
        xs, ys, h = self.grid(20.0)
        gy, gx = np.gradient(h, 20.0)
        g = np.hypot(gx, gy)
        return {"max_height_m": round(float(h.max()), 1), "mean_height_m": round(float(h.mean()), 1),
                "max_grade_pct": round(float(g.max() * 100), 1), "p95_grade_pct": round(float(np.percentile(g, 95) * 100), 1)}  # fmt: skip


def drape(v: np.ndarray, terrain: Terrain | None) -> np.ndarray:
    """Lift mesh vertices (z = offset above ground) onto the terrain."""
    if terrain is None or terrain.cfg.amplitude <= 0 or len(v) == 0:
        return v
    out = np.array(v, float)
    out[:, 2] += terrain.height(out[:, 0], out[:, 1])
    return out
