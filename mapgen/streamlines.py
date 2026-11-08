"""Road polylines traced through the tensor field (port of impl/streamlines.ts).

See the paper "Interactive Procedural Street Modeling" (Chen et al., 2008) for the method.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, replace

from .grid_storage import GridStorage
from .integrator import FieldIntegrator
from .simplify import simplify

Vec = tuple[float, float]


@dataclass
class StreamlineParams:
    dsep: float = 20.0  # seed separation distance
    dtest: float = 15.0  # integration separation distance
    dstep: float = 1.0  # step size
    dcirclejoin: float = 5.0  # how far to look to join circles
    dlookahead: float = 40.0  # how far to look ahead to join up dangling ends
    joinangle: float = 0.1  # angle (radians) to join roads
    path_iterations: int = 1000  # integration step limit
    seed_tries: int = 300  # max failed seeds
    simplify_tolerance: float = 0.5
    collide_early: float = 0.0  # chance of testing collision against both grids


@dataclass
class _Front:
    seed: Vec
    original_dir: Vec
    streamline: list[Vec]
    prev_dir: Vec
    prev_point: Vec
    valid: bool


def _angle_between(v1: Vec, v2: Vec) -> float:
    a = math.atan2(v1[1], v1[0]) - math.atan2(v2[1], v2[0])
    if a > math.pi:
        a -= 2 * math.pi
    elif a <= -math.pi:
        a += 2 * math.pi
    return a


def _set_length(v: Vec, length: float) -> Vec:
    n = math.hypot(v[0], v[1])
    if n == 0:
        return v
    return (v[0] / n * length, v[1] / n * length)


class StreamlineGenerator:
    SEED_AT_ENDPOINTS = False

    def __init__(
        self,
        integrator: FieldIntegrator,
        origin: Vec,
        world_dims: Vec,
        params: StreamlineParams,
        rng: random.Random,
    ) -> None:
        params = replace(params)
        params.dtest = min(params.dtest, params.dsep)  # enforce test < sep
        self.params = params
        self.integrator = integrator
        self.origin = origin
        self.world_dims = world_dims
        self.rng = rng

        self.dcollideself_sq = (params.dcirclejoin / 2) ** 2
        self.n_streamline_step = math.floor(params.dcirclejoin / params.dstep)
        self.n_streamline_look_back = 2 * self.n_streamline_step

        self.major_grid = GridStorage(world_dims, origin, params.dsep)
        self.minor_grid = GridStorage(world_dims, origin, params.dsep)

        self.dsep_sq = params.dsep**2
        self.dtest_sq = params.dtest**2
        self.dstep_sq = params.dstep**2
        self.dcirclejoin_sq = params.dcirclejoin**2

        self.candidate_seeds_major: list[Vec] = []
        self.candidate_seeds_minor: list[Vec] = []
        self.all_streamlines: list[list[Vec]] = []
        self.streamlines_major: list[list[Vec]] = []
        self.streamlines_minor: list[list[Vec]] = []
        self.all_streamlines_simple: list[list[Vec]] = []

    # --------------------------------------------------------------- helpers
    def grid(self, major: bool) -> GridStorage:
        return self.major_grid if major else self.minor_grid

    def streamlines(self, major: bool) -> list[list[Vec]]:
        return self.streamlines_major if major else self.streamlines_minor

    def candidate_seeds(self, major: bool) -> list[Vec]:
        return self.candidate_seeds_major if major else self.candidate_seeds_minor

    def clear_streamlines(self) -> None:
        self.all_streamlines_simple = []
        self.streamlines_major = []
        self.streamlines_minor = []
        self.all_streamlines = []

    def add_existing_streamlines(self, other: StreamlineGenerator) -> None:
        self.major_grid.add_all(other.major_grid)
        self.minor_grid.add_all(other.minor_grid)

    def simplify_streamline(self, streamline: list[Vec]) -> list[Vec]:
        return simplify(streamline, self.params.simplify_tolerance)

    def point_in_bounds(self, x: float, y: float) -> bool:
        ox, oy = self.origin
        return ox <= x < self.world_dims[0] + ox and oy <= y < self.world_dims[1] + oy

    def sample_point(self) -> Vec:
        return (
            self.rng.random() * self.world_dims[0] + self.origin[0],
            self.rng.random() * self.world_dims[1] + self.origin[1],
        )

    def is_valid_sample(
        self, major: bool, x: float, y: float, d_sq: float, both_grids: bool = False
    ) -> bool:
        if not self.grid(major).is_valid_sample(x, y, d_sq):
            return False
        if both_grids and not self.grid(not major).is_valid_sample(x, y, d_sq):
            return False
        return self.integrator.on_land(x, y)

    def get_seed(self, major: bool) -> Vec | None:
        if self.SEED_AT_ENDPOINTS:
            seeds = self.candidate_seeds(major)
            while seeds:
                seed = seeds.pop()
                if self.is_valid_sample(major, seed[0], seed[1], self.dsep_sq):
                    return seed
        seed = self.sample_point()
        i = 0
        while not self.is_valid_sample(major, seed[0], seed[1], self.dsep_sq):
            if i >= self.params.seed_tries:
                return None
            seed = self.sample_point()
            i += 1
        return seed

    # ------------------------------------------------------------ generation
    def create_all_streamlines(self) -> None:
        major = True
        while self.create_streamline(major):
            major = not major
        self.join_dangling_streamlines()

    def create_streamline(self, major: bool) -> bool:
        """Trace one streamline from a fresh seed. False once no valid seed can be found."""
        seed = self.get_seed(major)
        if seed is None:
            return False
        streamline = self.integrate_streamline(seed, major)
        if len(streamline) > 5:
            self.grid(major).add_polyline(streamline)
            self.streamlines(major).append(streamline)
            self.all_streamlines.append(streamline)
            self.all_streamlines_simple.append(self.simplify_streamline(streamline))
            if streamline[0] != streamline[-1]:
                self.candidate_seeds(not major).append(streamline[0])
                self.candidate_seeds(not major).append(streamline[-1])
        return True

    def streamline_turned(self, seed: Vec, original_dir: Vec, point: Vec, direction: Vec) -> bool:
        """True if the streamline has turned through more than 180 degrees."""
        if original_dir[0] * direction[0] + original_dir[1] * direction[1] < 0:
            px, py = original_dir[1], -original_dir[0]
            is_left = (point[0] - seed[0]) * px + (point[1] - seed[1]) * py < 0
            direction_up = direction[0] * px + direction[1] * py > 0
            return is_left == direction_up
        return False

    def _integration_step(self, f: _Front, major: bool, collide_both: bool) -> None:
        if not f.valid:
            return
        f.streamline.append(f.prev_point)
        dx, dy = self.integrator.integrate(f.prev_point[0], f.prev_point[1], major)
        if dx * dx + dy * dy < 0.01:  # degenerate point
            f.valid = False
            return
        if dx * f.prev_dir[0] + dy * f.prev_dir[1] < 0:  # keep travelling the same way
            dx, dy = -dx, -dy
        nx, ny = f.prev_point[0] + dx, f.prev_point[1] + dy
        if (
            self.point_in_bounds(nx, ny)
            and self.is_valid_sample(major, nx, ny, self.dtest_sq, collide_both)
            and not self.streamline_turned(f.seed, f.original_dir, (nx, ny), (dx, dy))
        ):
            f.prev_point = (nx, ny)
            f.prev_dir = (dx, dy)
        else:
            f.streamline.append((nx, ny))  # one more step
            f.valid = False

    def integrate_streamline(self, seed: Vec, major: bool) -> list[Vec]:
        """Integrate forwards and backwards at once so that loops join up cleanly."""
        count = 0
        points_escaped = False
        collide_both = self.rng.random() < self.params.collide_early

        d = self.integrator.integrate(seed[0], seed[1], major)
        fwd_start = (seed[0] + d[0], seed[1] + d[1])
        fwd = _Front(seed, d, [seed], d, fwd_start, self.point_in_bounds(*fwd_start))
        neg = (-d[0], -d[1])
        bwd_start = (seed[0] + neg[0], seed[1] + neg[1])
        bwd = _Front(seed, neg, [], neg, bwd_start, self.point_in_bounds(*bwd_start))

        while count < self.params.path_iterations and (fwd.valid or bwd.valid):
            self._integration_step(fwd, major, collide_both)
            self._integration_step(bwd, major, collide_both)

            ex = fwd.prev_point[0] - bwd.prev_point[0]
            ey = fwd.prev_point[1] - bwd.prev_point[1]
            sq = ex * ex + ey * ey
            if not points_escaped and sq > self.dcirclejoin_sq:
                points_escaped = True
            if points_escaped and sq <= self.dcirclejoin_sq:  # join up circles
                fwd.streamline.append(fwd.prev_point)
                fwd.streamline.append(bwd.prev_point)
                bwd.streamline.append(bwd.prev_point)
                break
            count += 1

        bwd.streamline.reverse()
        bwd.streamline.extend(fwd.streamline)
        return bwd.streamline

    # ------------------------------------------------------- dangling ends
    def join_dangling_streamlines(self) -> None:
        """Extend loose road ends to a nearby road when one lies roughly straight ahead."""
        for major in (True, False):
            for s in self.streamlines(major):
                if s[0] == s[-1]:  # ignore circles
                    continue
                new_start = self.get_best_next_point(s[0], s[4])
                if new_start is not None:
                    for p in self.points_between(s[0], new_start, self.params.dstep):
                        s.insert(0, p)
                        self.grid(major).add_sample(p)
                new_end = self.get_best_next_point(s[-1], s[-4])
                if new_end is not None:
                    for p in self.points_between(s[-1], new_end, self.params.dstep):
                        s.append(p)
                        self.grid(major).add_sample(p)
        self.all_streamlines_simple = [self.simplify_streamline(s) for s in self.all_streamlines]

    def points_between(self, v1: Vec, v2: Vec, dstep: float) -> list[Vec]:
        """Points from v1 towards v2 at most dstep apart (v1 excluded); same quirks as the TS."""
        d = math.hypot(v2[0] - v1[0], v2[1] - v1[1])
        n = math.floor(d / dstep)
        if n == 0:
            return []
        sx, sy = v2[0] - v1[0], v2[1] - v1[1]
        out: list[Vec] = []
        nxt = (v1[0] + sx * (1 / n), v1[1] + sy * (1 / n))
        for i in range(1, n + 1):
            ix, iy = self.integrator.integrate(nxt[0], nxt[1], True)
            if ix * ix + iy * iy > 0.001:  # stop at degenerate points
                out.append(nxt)
            else:
                return out
            nxt = (v1[0] + sx * (i / n), v1[1] + sy * (i / n))
        return out

    def get_best_next_point(self, point: Vec, previous: Vec) -> Vec | None:
        nearby = self.major_grid.get_nearby_points(point, self.params.dlookahead)
        nearby.extend(self.minor_grid.get_nearby_points(point, self.params.dlookahead))
        direction = (point[0] - previous[0], point[1] - previous[1])

        closest: Vec | None = None
        closest_dist = math.inf
        for sample in nearby:
            if sample == point or sample == previous:
                continue
            diff = (sample[0] - point[0], sample[1] - point[1])
            if diff[0] * direction[0] + diff[1] * direction[1] < 0:
                continue  # backwards
            dist_sq = diff[0] * diff[0] + diff[1] * diff[1]
            if dist_sq < 2 * self.dstep_sq:
                closest = sample
                break
            angle = abs(_angle_between(direction, diff))
            if angle < self.params.joinangle and dist_sq < closest_dist:
                closest_dist = dist_sq
                closest = sample

        if closest is not None:
            # Overshoot slightly so the joined end still crosses the road after simplification.
            ex, ey = _set_length(direction, self.params.simplify_tolerance * 4)
            closest = (closest[0] + ex, closest[1] + ey)
        return closest
