"""Coastline and river (port of impl/water_generator.ts)."""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field

from .integrator import FieldIntegrator
from .polygon_util import PolygonTester, line_rectangle_polygon_intersection, resize_geometry
from .streamlines import StreamlineGenerator, StreamlineParams
from .tensor_field import TensorField

Vec = tuple[float, float]
log = logging.getLogger(__name__)


@dataclass
class NoiseStreamlineParams:
    noise_enabled: bool = True
    noise_size: float = 30.0
    noise_angle: float = 20.0  # degrees


@dataclass
class WaterParams(StreamlineParams):
    coast_noise: NoiseStreamlineParams = field(default_factory=NoiseStreamlineParams)
    river_noise: NoiseStreamlineParams = field(default_factory=NoiseStreamlineParams)
    river_bank_size: float = 10.0
    river_size: float = 30.0


class WaterGenerator(StreamlineGenerator):
    TRIES = 100

    def __init__(
        self,
        integrator: FieldIntegrator,
        origin: Vec,
        world_dims: Vec,
        params: WaterParams,
        tensor_field: TensorField,
        rng: random.Random,
    ) -> None:
        super().__init__(integrator, origin, world_dims, params, rng)
        self.params: WaterParams = self.params  # type: ignore[assignment]
        self.tensor_field = tensor_field
        self.coastline_major = True
        self.coastline: list[Vec] = []  # noisy line, drawn only
        self.sea_polygon: list[Vec] = []
        self.river_polygon: list[Vec] = []
        self.river_secondary_road: list[Vec] = []

    def create_coast(self) -> None:
        p = self.params
        if p.coast_noise.noise_enabled:
            cn = p.coast_noise
            self.tensor_field.enable_global_noise(cn.noise_angle, cn.noise_size)
        coast: list[Vec] = []
        major = True
        for _ in range(self.TRIES):
            major = self.rng.random() < 0.5
            seed = self.get_seed(major)
            if seed is None:
                continue
            coast = self.extend_streamline(self.integrate_streamline(seed, major))
            if self.reaches_edges(coast):
                break
        self.tensor_field.disable_global_noise()

        self.coastline = coast
        self.coastline_major = major
        if len(coast) < 2:
            return

        road = self.simplify_streamline(coast)
        self.sea_polygon = self.get_sea_polygon(road)
        self.all_streamlines_simple.append(road)
        self.tensor_field.sea = self.sea_polygon

        complex_ = self.complexify_streamline(road)
        self.grid(major).add_polyline(complex_)
        self.streamlines(major).append(complex_)
        self.all_streamlines.append(complex_)

    def create_river(self) -> None:
        p = self.params
        old_sea = self.tensor_field.sea
        self.tensor_field.sea = []  # ignore the sea when checking that the river reaches the edges
        if p.river_noise.noise_enabled:
            rn = p.river_noise
            self.tensor_field.enable_global_noise(rn.noise_angle, rn.noise_size)
        river: list[Vec] = []
        for i in range(self.TRIES):
            seed = self.get_seed(not self.coastline_major)
            if seed is None:
                continue
            traced = self.integrate_streamline(seed, not self.coastline_major)
            river = self.extend_streamline(traced)
            if self.reaches_edges(river):
                break
            if i == self.TRIES - 1:
                log.warning("Failed to find river reaching edge")
        self.tensor_field.sea = old_sea
        self.tensor_field.disable_global_noise()
        if len(river) < 2:
            return

        # River banks become roads
        expanded_noisy = self.complexify_streamline(resize_geometry(river, p.river_size, False))
        self.river_polygon = resize_geometry(river, p.river_size - p.river_bank_size, False)
        # Make sure the outline starts off screen
        first_off = next((i for i, v in enumerate(expanded_noisy) if self.vector_off_screen(v)), -1)
        if first_off > 0:
            expanded_noisy = expanded_noisy[first_off:] + expanded_noisy[:first_off]

        sea = PolygonTester(self.sea_polygon)
        split = PolygonTester(self.get_sea_polygon(river))
        road1 = [
            v
            for v in expanded_noisy
            if not sea.contains(*v) and not self.vector_off_screen(v) and split.contains(*v)
        ]
        road2 = [
            v
            for v in expanded_noisy
            if not sea.contains(*v) and not self.vector_off_screen(v) and not split.contains(*v)
        ]
        road1_simple = self.simplify_streamline(road1)
        road2_simple = self.simplify_streamline(road2)
        if not road1 or not road2:
            return

        def d2(a: Vec, b: Vec) -> float:
            return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2

        if d2(road1[0], road2[0]) < d2(road1[0], road2[-1]):
            road2_simple.reverse()

        self.tensor_field.river = road1_simple + road2_simple
        self.all_streamlines_simple.append(road1_simple)
        self.river_secondary_road = road2_simple

        minor = not self.coastline_major
        self.grid(minor).add_polyline(road1)
        self.grid(minor).add_polyline(road2)
        self.streamlines(minor).append(road1)
        self.streamlines(minor).append(road2)
        self.all_streamlines.append(road1)
        self.all_streamlines.append(road2)

    # --------------------------------------------------------------- helpers
    def get_sea_polygon(self, polyline: list[Vec]) -> list[Vec]:
        return line_rectangle_polygon_intersection(self.origin, self.world_dims, polyline)

    def complexify_streamline(self, s: list[Vec]) -> list[Vec]:
        """Insert samples so consecutive points are at most dstep apart (joints duplicated)."""
        out: list[Vec] = []
        for i in range(len(s) - 1):
            out.extend(self._complexify(s[i], s[i + 1]))
        return out

    def _complexify(self, v1: Vec, v2: Vec) -> list[Vec]:
        if (v1[0] - v2[0]) ** 2 + (v1[1] - v2[1]) ** 2 <= self.dstep_sq:
            return [v1, v2]
        half = (v1[0] + (v2[0] - v1[0]) * 0.5, v1[1] + (v2[1] - v1[1]) * 0.5)
        return self._complexify(v1, half) + self._complexify(half, v2)

    def extend_streamline(self, s: list[Vec]) -> list[Vec]:
        """Push both ends 5 steps further out (mutates and returns ``s``)."""
        if len(s) < 2:
            return s
        length = self.params.dstep * 5

        def ext(a: Vec, b: Vec) -> Vec:
            dx, dy = a[0] - b[0], a[1] - b[1]
            n = (dx * dx + dy * dy) ** 0.5 or 1.0
            return (a[0] + dx / n * length, a[1] + dy / n * length)

        s.insert(0, ext(s[0], s[1]))
        s.append(ext(s[-1], s[-2]))
        return s

    def reaches_edges(self, s: list[Vec]) -> bool:
        return bool(s) and self.vector_off_screen(s[0]) and self.vector_off_screen(s[-1])

    def vector_off_screen(self, v: Vec) -> bool:
        x = v[0] - self.origin[0]
        y = v[1] - self.origin[1]
        return x <= 0 or y <= 0 or x >= self.world_dims[0] or y >= self.world_dims[1]
