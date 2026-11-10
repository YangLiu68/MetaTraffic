import pygame as pg
from typing import Dict, List, Tuple
from .types import Poly, SvgData, BBox
from .constants import BG_COLOR, ROAD_DRAW_W, CENTERLINE_COLOR, CENTERLINE_WIDTH
from .utils import brighten_rgb_string, parse_rgb_triplet

def draw_filled_polys(screen: pg.Surface, polys: List[Poly] | None, to_screen, fill_override=None, stroke=True):
    if not polys:
        return
    for poly in polys:
        pts = poly.pts
        if len(pts) < 6:
            continue
        path = [to_screen(pts[0], pts[1])]
        for i in range(2, len(pts), 2):
            path.append(to_screen(pts[i], pts[i+1]))
        color_fill = parse_rgb_triplet(fill_override(poly.color) if fill_override else poly.color)
        pg.draw.polygon(screen, color_fill, path, 0)
        if stroke:
            pg.draw.polygon(screen, parse_rgb_triplet(poly.color), path, 1)

def draw_roads(screen: pg.Surface, data: SvgData, to_screen, zoom: float, base_scale: float):
    for typ in ('residential','secondary','main'):
        polys = data.get(typ, [])
        for poly in polys:
            pts = poly.pts
            if len(pts) < 4:
                continue
            path = [to_screen(pts[0], pts[1])]
            for i in range(2, len(pts), 2):
                path.append(to_screen(pts[i], pts[i+1]))
            # base stroke (road body)
            width = max(1, int(ROAD_DRAW_W[typ] * zoom))
            pg.draw.lines(screen, parse_rgb_triplet(poly.color), False, path, width)
            # centerline
            width2 = max(1, int(CENTERLINE_WIDTH * base_scale * zoom))
            pg.draw.lines(screen, CENTERLINE_COLOR[typ], False, path, width2)

def clear(screen: pg.Surface, W: int, H: int):
    screen.fill(BG_COLOR)
