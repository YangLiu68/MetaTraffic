"""Trip demand per directed street: floor area of the buildings that front it."""

from __future__ import annotations

import numpy as np
import shapely
from shapely import LineString, STRtree


def demand_weights(world, base: float = 200.0) -> np.ndarray:
    """One weight per directed edge (router order). Taller blocks attract more trips."""
    net = world.net
    ids = sorted(net.dedges)
    index = {d: i for i, d in enumerate(ids)}
    w = np.full(len(ids), base)
    lines = [LineString(e.center.pts) for e in net.edges]
    if not world.buildings or not lines:
        return w
    tree = STRtree(lines)
    cents = shapely.centroid(np.asarray([b.footprint for b in world.buildings], dtype=object))
    nearest = tree.nearest(cents)
    for b, ei in zip(world.buildings, nearest):
        for d in (0, 1):
            w[index[2 * int(ei) + d]] += b.floor_area / 2
    return w
