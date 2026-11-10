import pygame as pg
from typing import Dict, Tuple, List
from .types import Poly, BBox
from .constants import ROAD_LANES, LANE_WIDTH

def build_road_mask(roads: Dict[str, List[Poly]], world_bbox: BBox, max_tex: int = 4096):
    minx, miny, maxx, maxy = world_bbox
    wx = maxx - minx
    wy = maxy - miny
    scale_x = max_tex / wx if wx > 0 else 1.0
    scale_y = max_tex / wy if wy > 0 else 1.0
    mask_scale = min(scale_x, scale_y) * 4.0
    W = max(8, int(wx * mask_scale))
    H = max(8, int(wy * mask_scale))

    surf = pg.Surface((W, H), flags=pg.SRCALPHA)
    surf.fill((0,0,0,0))

    def w2m(x: float, y: float):
        u = int((x - minx) * mask_scale)
        v = int((y - miny) * mask_scale)
        return u, v

    order = ('residential', 'secondary', 'main')
    margins = {'residential': 0.6, 'secondary': 1.0, 'main': 1.5}

    for typ in order:
        l, r = ROAD_LANES[typ]
        total_w = (l + r) * LANE_WIDTH
        margin = margins[typ]
        pxw = max(1, int((total_w + margin) * mask_scale))
        for poly in roads.get(typ, []):
            pts = poly.pts
            if len(pts) < 4: 
                continue
            path = [w2m(pts[0], pts[1])]
            for i in range(2, len(pts), 2):
                path.append(w2m(pts[i], pts[i+1]))
            # draw thick polyline as mask (white)
            if len(path) >= 2:
                pg.draw.lines(surf, (255,255,255,255), False, path, pxw)

    def get(x: float, y: float) -> bool:
        u = int((x - minx) * mask_scale)
        v = int((y - miny) * mask_scale)
        if u < 0 or v < 0 or u >= W or v >= H:
            return False
        c = surf.get_at((u, v))
        return c.a > 0 or c.r > 0

    return {'surface': surf, 'mask_scale': mask_scale, 'origin': (minx, miny), 'get': get}
