from typing import List, Dict, Tuple
from .types import Poly

def nearest_point_on_segment(ax: float, ay: float, bx: float, by: float, px: float, py: float):
    abx = bx - ax
    aby = by - ay
    ab2 = abx*abx + aby*aby
    t = 0.0
    if ab2 > 0:
        t = max(0.0, min(1.0, ((px-ax)*abx + (py-ay)*aby) / ab2))
    projx = ax + t*abx
    projy = ay + t*aby
    vx = bx - ax
    vy = by - ay
    n = (vx*vx + vy*vy) ** 0.5
    dirx = vx/n if n > 1e-9 else 1.0
    diry = vy/n if n > 1e-9 else 0.0
    dist = ((px - projx)**2 + (py - projy)**2) ** 0.5
    return projx, projy, dist, dirx, diry

def nearest_on_polyline(pts: List[float], px: float, py: float):
    best = {'idx': -1, 'projx': 0.0, 'projy': 0.0, 'dist': float('inf'), 'dirx': 1.0, 'diry': 0.0}
    for i in range(0, len(pts) - 2, 2):
        ax, ay = pts[i], pts[i+1]
        bx, by = pts[i+2], pts[i+3]
        projx, projy, dist, dirx, diry = nearest_point_on_segment(ax, ay, bx, by, px, py)
        if dist < best['dist']:
            best = {'idx': i//2, 'projx': projx, 'projy': projy, 'dist': dist, 'dirx': dirx, 'diry': diry}
    return best

def find_nearest_road_segment(roads: Dict[str, List[Poly]], px: float, py: float):
    best = {'typ': None, 'pts': None, 'idx': -1, 'projx': 0.0, 'projy': 0.0, 'dirx': 1.0, 'diry': 0.0, 'dist': float('inf')}
    for typ in ('main','secondary','residential'):
        for poly in roads.get(typ, []):
            r = nearest_on_polyline(poly.pts, px, py)
            if r['dist'] < best['dist']:
                best = {'typ': typ, 'pts': poly.pts, 'idx': r['idx'], 'projx': r['projx'], 'projy': r['projy'], 'dirx': r['dirx'], 'diry': r['diry'], 'dist': r['dist']}
    return best
