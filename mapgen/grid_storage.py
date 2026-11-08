"""Uniform grid of samples for fast separation tests (port of impl/grid_storage.ts)."""

from __future__ import annotations

import math

Vec = tuple[float, float]


class GridStorage:
    def __init__(self, world_dims: Vec, origin: Vec, dsep: float) -> None:
        self.wx, self.wy = world_dims
        self.ox, self.oy = origin
        self.dsep = dsep
        self.dsep_sq = dsep * dsep
        self.gx = self.wx / dsep  # grid dimensions (fractional, as in the original)
        self.gy = self.wy / dsep
        self.nx = math.ceil(self.gx)
        self.ny = math.ceil(self.gy)
        self.grid: list[list[list[Vec]]] = [[[] for _ in range(self.ny)] for _ in range(self.nx)]

    def add_all(self, other: GridStorage) -> None:
        for column in other.grid:
            for cell in column:
                for sample in cell:
                    self.add_sample(sample)

    def add_polyline(self, line: list[Vec]) -> None:
        for v in line:
            self.add_sample(v)

    def add_sample(self, v: Vec) -> None:
        cx, cy = self._coords(v[0], v[1])
        self.grid[cx][cy].append(v)

    def is_valid_sample(self, x: float, y: float, d_sq: float | None = None) -> bool:
        """True if no stored sample lies within sqrt(d_sq) of (x, y)."""
        if d_sq is None:
            d_sq = self.dsep_sq
        cx, cy = self._coords(x, y)
        gx, gy, grid = self.gx, self.gy, self.grid
        for i in (cx - 1, cx, cx + 1):
            if i < 0 or i >= gx:
                continue
            column = grid[i]
            for j in (cy - 1, cy, cy + 1):
                if j < 0 or j >= gy:
                    continue
                for sx, sy in column[j]:
                    dx = sx - x
                    dy = sy - y
                    if dx * dx + dy * dy < d_sq:
                        return False
        return True

    def get_nearby_points(self, v: Vec, distance: float) -> list[Vec]:
        radius = math.ceil(distance / self.dsep - 0.5)
        cx, cy = self._coords(v[0], v[1])
        out: list[Vec] = []
        for i in range(cx - radius, cx + radius + 1):
            if i < 0 or i >= self.gx:
                continue
            for j in range(cy - radius, cy + radius + 1):
                if j < 0 or j >= self.gy:
                    continue
                out.extend(self.grid[i][j])
        return out

    def _coords(self, x: float, y: float) -> tuple[int, int]:
        vx = x - self.ox
        vy = y - self.oy
        # The original maps out-of-bounds samples to cell (0, 0).
        if vx < 0 or vy < 0 or vx >= self.wx or vy >= self.wy:
            return (0, 0)
        return (int(vx // self.dsep), int(vy // self.dsep))
