"""Quick checks for the Python port. Run: python -m pytest mapgen/tests"""

import random

from mapgen import CityConfig, generate_city, to_svg
from mapgen.noise import SimplexNoise
from mapgen.polygon_util import inside_polygon, polygon_area, subdivide_polygon
from mapgen.simplify import simplify
from mapgen.tensor_field import Tensor

SMALL = dict(width=300.0, height=200.0, zoom=1.0)


def test_simplify_keeps_ends_and_drops_collinear_points():
    line = [(float(i), 0.0) for i in range(50)] + [(49.0, 10.0)]
    out = simplify(line, 0.5)
    assert out[0] == line[0] and out[-1] == line[-1]
    assert out == [(0.0, 0.0), (49.0, 0.0), (49.0, 10.0)]


def test_noise_is_seeded_and_bounded():
    a = SimplexNoise(random.Random(3))
    b = SimplexNoise(random.Random(3))
    vals = [a.noise2d(x * 0.37, x * 0.11) for x in range(500)]
    assert vals == [b.noise2d(x * 0.37, x * 0.11) for x in range(500)]
    assert all(-1.0 <= v <= 1.0 for v in vals)


def test_tensor_major_and_minor_are_perpendicular():
    t = Tensor.from_angle(0.3)
    (ax, ay), (bx, by) = t.major(), t.minor()
    assert abs(ax * bx + ay * by) < 1e-9
    assert Tensor.zero().major() == (0.0, 0.0)


def test_subdivide_makes_non_overlapping_lots():
    from shapely import Polygon
    from shapely.ops import unary_union

    square = [(0.0, 0.0), (40.0, 0.0), (40.0, 40.0), (0.0, 40.0)]
    lots = subdivide_polygon(square, 50.0, random.Random(0))
    areas = [polygon_area(p) for p in lots]
    assert len(lots) > 4
    assert min(areas) >= 25.0  # pieces under half the minimum area are dropped
    union = unary_union([Polygon(p) for p in lots]).area
    assert abs(union - sum(areas)) < 1e-6  # no overlaps
    assert 0.9 * 1600.0 <= sum(areas) <= 1600.0 + 1e-6  # only slivers are lost
    assert inside_polygon((20.0, 20.0), square) and not inside_polygon((50.0, 20.0), square)


def test_same_seed_same_map_and_roads_exist():
    a = to_svg(generate_city(CityConfig(seed=11, **SMALL)))
    b = to_svg(generate_city(CityConfig(seed=11, **SMALL)))
    assert a == b
    city = generate_city(CityConfig(seed=12, **SMALL))
    assert city.minor_roads and city.lots
    assert to_svg(city) != a
