"""Tensor, basis fields and the combined tensor field.

Port of impl/tensor.ts, impl/basis_field.ts, impl/tensor_field.ts and the field setup in
ui/tensor_field_gui.ts. Behaviour (including the quirks) matches the TypeScript version.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from .noise import SimplexNoise
from .polygon_util import PolygonTester

Vec = tuple[float, float]
_PI = math.pi


class Tensor:
    """2x2 symmetric traceless tensor stored as r * [cos 2θ, sin 2θ]."""

    __slots__ = ("r", "m0", "m1", "_theta", "_stale")

    def __init__(self, r: float, m0: float, m1: float) -> None:
        self.r = r
        self.m0 = m0
        self.m1 = m1
        self._stale = False
        self._theta = self._calc_theta()

    @staticmethod
    def from_angle(angle: float) -> Tensor:
        return Tensor(1.0, math.cos(angle * 4), math.sin(angle * 4))

    @staticmethod
    def from_vector(x: float, y: float) -> Tensor:
        t1 = x * x - y * y
        t2 = 2 * x * y
        return Tensor(1.0, t1 * t1 - t2 * t2, 2 * t1 * t2)

    @staticmethod
    def zero() -> Tensor:
        return Tensor(0.0, 0.0, 0.0)

    @property
    def theta(self) -> float:
        if self._stale:
            self._theta = self._calc_theta()
            self._stale = False
        return self._theta

    def add(self, t: Tensor, smooth: bool) -> Tensor:
        self.m0 = self.m0 * self.r + t.m0 * t.r
        self.m1 = self.m1 * self.r + t.m1 * t.r
        if smooth:
            self.r = math.hypot(self.m0, self.m1)
            if self.r > 0:
                self.m0 /= self.r
                self.m1 /= self.r
        else:
            self.r = 2.0
        self._stale = True
        return self

    def scale(self, s: float) -> Tensor:
        self.r *= s
        self._stale = True
        return self

    def rotate(self, theta: float) -> Tensor:
        if theta == 0:
            return self
        new_theta = self.theta + theta
        if new_theta < _PI:
            new_theta += _PI
        if new_theta >= _PI:
            new_theta -= _PI
        self.m0 = math.cos(2 * new_theta) * self.r
        self.m1 = math.sin(2 * new_theta) * self.r
        self._theta = new_theta
        self._stale = False
        return self

    def major(self) -> Vec:
        if self.r == 0:
            return (0.0, 0.0)
        th = self.theta
        return (math.cos(th), math.sin(th))

    def minor(self) -> Vec:
        if self.r == 0:
            return (0.0, 0.0)
        a = self.theta + _PI / 2
        return (math.cos(a), math.sin(a))

    def _calc_theta(self) -> float:
        if self.r == 0:
            return 0.0
        return math.atan2(self.m1 / self.r, self.m0 / self.r) / 2


class BasisField:
    """Grid or radial field, weighted by distance to its centre."""

    def __init__(self, centre: Vec, size: float, decay: float) -> None:
        self.centre = centre
        self.size = size
        self.decay = decay

    def tensor(self, x: float, y: float) -> Tensor:
        raise NotImplementedError

    def weight(self, x: float, y: float, smooth: bool) -> float:
        d = math.hypot(x - self.centre[0], y - self.centre[1]) / self.size
        if smooth:
            return math.inf if d == 0 and self.decay > 0 else d ** (-self.decay)
        if self.decay == 0 and d >= 1:
            return 0.0
        return max(0.0, 1.0 - d) ** self.decay

    def weighted_tensor(self, x: float, y: float, smooth: bool) -> Tensor:
        return self.tensor(x, y).scale(self.weight(x, y, smooth))


class Grid(BasisField):
    def __init__(self, centre: Vec, size: float, decay: float, theta: float) -> None:
        super().__init__(centre, size, decay)
        self.theta = theta
        self._c = math.cos(2 * theta)
        self._s = math.sin(2 * theta)

    def tensor(self, x: float, y: float) -> Tensor:
        return Tensor(1.0, self._c, self._s)


class Radial(BasisField):
    def tensor(self, x: float, y: float) -> Tensor:
        tx = x - self.centre[0]
        ty = y - self.centre[1]
        return Tensor(1.0, ty * ty - tx * tx, -2 * tx * ty)


@dataclass
class NoiseParams:
    global_noise: bool = False
    noise_size_park: float = 20.0
    noise_angle_park: float = 90.0  # degrees
    noise_size_global: float = 30.0
    noise_angle_global: float = 20.0  # degrees


class TensorField:
    """Sum of basis fields; rotational noise inside parks; zero tensor in the sea / river."""

    def __init__(self, noise_params: NoiseParams, rng: random.Random) -> None:
        self.noise_params = noise_params
        self.basis_fields: list[BasisField] = []
        self.noise = SimplexNoise(rng)
        self.smooth = False
        self.ignore_river = False
        self._parks: list[list[Vec]] = []
        self._park_testers: list[PolygonTester] = []
        self._sea: list[Vec] = []
        self._sea_tester = PolygonTester([])
        self._river: list[Vec] = []
        self._river_tester = PolygonTester([])

    # polygons are wrapped in testers once, when they are set
    @property
    def parks(self) -> list[list[Vec]]:
        return self._parks

    @parks.setter
    def parks(self, parks: list[list[Vec]]) -> None:
        self._parks = list(parks)
        self._park_testers = [PolygonTester(p) for p in self._parks]

    @property
    def sea(self) -> list[Vec]:
        return self._sea

    @sea.setter
    def sea(self, sea: list[Vec]) -> None:
        self._sea = list(sea)
        self._sea_tester = PolygonTester(self._sea)

    @property
    def river(self) -> list[Vec]:
        return self._river

    @river.setter
    def river(self, river: list[Vec]) -> None:
        self._river = list(river)
        self._river_tester = PolygonTester(self._river)

    def enable_global_noise(self, angle: float, size: float) -> None:
        self.noise_params.global_noise = True
        self.noise_params.noise_angle_global = angle
        self.noise_params.noise_size_global = size

    def disable_global_noise(self) -> None:
        self.noise_params.global_noise = False

    def add_grid(self, centre: Vec, size: float, decay: float, theta: float) -> None:
        self.basis_fields.append(Grid(centre, size, decay, theta))

    def add_radial(self, centre: Vec, size: float, decay: float) -> None:
        self.basis_fields.append(Radial(centre, size, decay))

    def reset(self) -> None:
        self.basis_fields = []
        self.parks = []
        self.sea = []
        self.river = []

    def sample_point(self, x: float, y: float) -> Tensor:
        if not self.on_land(x, y):
            return Tensor.zero()
        if not self.basis_fields:
            return Tensor(1.0, 0.0, 0.0)

        smooth = self.smooth
        if smooth:
            acc = Tensor.zero()
            for f in self.basis_fields:
                acc.add(f.weighted_tensor(x, y, True), True)
        else:
            # Same result as repeated Tensor.add(..., smooth=False): after the first add r == 2,
            # so every later add doubles what was accumulated before it.
            m0 = m1 = 0.0
            r = 0.0
            for f in self.basis_fields:
                t = f.weighted_tensor(x, y, False)
                m0 = m0 * r + t.m0 * t.r
                m1 = m1 * r + t.m1 * t.r
                r = 2.0
            acc = Tensor(r, m0, m1)

        np_ = self.noise_params
        for tester in self._park_testers:
            if tester.contains(x, y):
                acc.rotate(self.rotational_noise(x, y, np_.noise_size_park, np_.noise_angle_park))
                break
        if np_.global_noise:
            acc.rotate(self.rotational_noise(x, y, np_.noise_size_global, np_.noise_angle_global))
        return acc

    def rotational_noise(self, x: float, y: float, size: float, angle_deg: float) -> float:
        return self.noise.noise2d(x / size, y / size) * angle_deg * _PI / 180

    def on_land(self, x: float, y: float) -> bool:
        if self._sea_tester.contains(x, y):
            return False
        if self.ignore_river:
            return True
        return not self._river_tester.contains(x, y)

    def in_parks(self, x: float, y: float) -> bool:
        return any(t.contains(x, y) for t in self._park_testers)

    # ------------------------------------------------- ui/tensor_field_gui.ts
    def set_recommended(
        self, origin: Vec, world_dims: Vec, rng: random.Random, spawn_scale: float = 0.7
    ) -> None:
        """Four grids at the corners of the central 70% box plus one random radial field."""
        self.reset()
        sx, sy = world_dims[0] * spawn_scale, world_dims[1] * spawn_scale
        ox = world_dims[0] * (1 - spawn_scale) / 2 + origin[0]
        oy = world_dims[1] * (1 - spawn_scale) / 2 + origin[1]
        width = world_dims[0]
        for cx, cy in ((ox, oy), (ox + sx, oy + sy), (ox + sx, oy), (ox, oy + sy)):
            self.add_grid(
                (cx, cy),
                _random_range(rng, width / 4, width),
                _random_range(rng, 50),
                _random_range(rng, _PI / 2),
            )
        loc = (ox + rng.random() * sx, oy + rng.random() * sy)
        self.add_radial(loc, _random_range(rng, width / 10, width / 5), _random_range(rng, 50))


def _random_range(rng: random.Random, max_: float, min_: float = 0.0) -> float:
    return rng.random() * (max_ - min_) + min_
