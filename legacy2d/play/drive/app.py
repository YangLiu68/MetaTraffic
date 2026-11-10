import os, sys, math, random, time
import pygame as pg
from pygame import freetype
from typing import Optional, Tuple, Dict, List

from .types import SvgData, Poly, BBox
from .svg_parser import extract_paths_from_svg_string
from .constants import (
    COLOR_MAP, ROAD_LANES, LANE_WIDTH, CENTERLINE_WIDTH,
    road_half_width, CENTERLINE_COLOR
)
from .constants import BG_COLOR, CAR_LEN, CAR_WID
from .render import clear, draw_filled_polys, draw_roads
from .mask import build_road_mask
from .geometry import find_nearest_road_segment
from .physics import Car
from .utils import clamp, lerp, parse_rgb_triplet, brighten_rgb_string

class App:
    def __init__(self, size=(1280, 900)):
        pg.init()
        pg.display.set_caption('Drive One Car v4 — Python')
        self.screen = pg.display.set_mode(size, pg.RESIZABLE | pg.SCALED)
        try:
            freetype.init()
            self.font = freetype.SysFont(None, 16)
            self.font_big = freetype.SysFont(None, 18)
        except Exception:
            pg.font.init()
            self.font = pg.font.SysFont(None, 16)
            self.font_big = pg.font.SysFont(None, 18)

        self.svg_text: Optional[str] = None
        self.data: Optional[SvgData] = None
        self.bbox: Optional[BBox] = None
        self.base_scale: float = 1.0
        self.base_offset: Tuple[float,float] = (0.0, 0.0)

        self.zoom: float = 1.0
        self.pan: Tuple[float,float] = (0.0, 0.0)
        self.dragging: bool = False
        self.last_mouse: Tuple[int,int] = (0,0)

        self.show_buildings: bool = True
        self.follow_cam: bool = False
        self.debug_collision: bool = False

        self.keys = {'w':False,'s':False,'a':False,'d':False}

        self.car: Optional[Car] = None
        self.start_center: Optional[Tuple[float,float]] = None
        self.end_center: Optional[Tuple[float,float]] = None

        self.mask = None

        self.clock = pg.time.Clock()

    # --- Coordinate transforms ---
    def _recompute_base_view(self):
        if not self.bbox:
            return
        W, H = self.screen.get_size()
        minx, miny, maxx, maxy = self.bbox
        wx = maxx - minx
        wy = maxy - miny
        margin = 100.0
        # fit (W-2*margin, H-2*margin)
        scale = min(max((W - 2*margin) / wx, 0.0001), max((H - 2*margin) / wy, 0.0001))
        self.base_scale = scale
        self.base_offset = (margin - minx * scale, margin - miny * scale)

    def to_screen(self, x: float, y: float) -> tuple[int,int]:
        sx = x * self.base_scale * self.zoom + self.base_offset[0] + self.pan[0]
        sy = y * self.base_scale * self.zoom + self.base_offset[1] + self.pan[1]
        return int(round(sx)), int(round(sy))

    # --- Loading & Parsing ---
    def _load_svg_text(self, svg_text: str):
        data, bbox = extract_paths_from_svg_string(svg_text)
        self.data = data
        self.bbox = bbox
        self._recompute_base_view()

        roads = {
            'main': data.get('main', []),
            'secondary': data.get('secondary', []),
            'residential': data.get('residential', []),
        }

        # Build mask
        self.mask = build_road_mask(roads, bbox)

        # Random building -> spawn near the nearest road right side
        buildings = data.get('building', [])
        if not buildings:
            print('SVG 未解析到建筑')
            self.car = None
            self.start_center = None
            self.end_center = None
            return

        pick = random.choice(buildings)
        # rough centroid
        cx = 0.0; cy = 0.0; n = 0
        for i in range(0, len(pick.pts), 2):
            cx += pick.pts[i]; cy += pick.pts[i+1]; n += 1
        cx /= n; cy /= n
        self.start_center = (cx, cy)

        near = find_nearest_road_segment(roads, cx, cy)
        if (not near['typ']) or (not near['pts']):
            print('未找到临近道路')
            return
        right = (-near['diry'], near['dirx'])
        from .constants import road_half_width, LANE_WIDTH, CAR_WID
        spawn_offset = max(0.0, min(road_half_width(near['typ']) - CAR_WID/2 - 0.02, LANE_WIDTH * 0.75))
        spawn_pos = (near['projx'] + right[0]*spawn_offset, near['projy'] + right[1]*spawn_offset)
        heading = (near['dirx'], near['diry'])
        self.car = Car.spawn(spawn_pos, heading, near['typ'])

        if len(buildings) > 1:
            target = None
            for _ in range(10):
                b = random.choice(buildings)
                if b is not pick:
                    target = b
                    break
            if target:
                tx = ty = 0.0; tn = 0
                for i in range(0, len(target.pts), 2):
                    tx += target.pts[i]; ty += target.pts[i+1]; tn += 1
                tx /= tn; ty /= tn
                self.end_center = (tx, ty)
        # reset view
        self.zoom = 1.0
        self.pan = (0.0, 0.0)

    # --- Event handling ---
    def handle_event(self, e: pg.event.Event):
        if e.type == pg.QUIT:
            pg.quit(); sys.exit(0)
        elif e.type == pg.VIDEORESIZE:
            self._recompute_base_view()
        elif e.type == pg.KEYDOWN:
            if e.key == pg.K_w: self.keys['w'] = True
            if e.key == pg.K_s: self.keys['s'] = True
            if e.key == pg.K_a: self.keys['a'] = True
            if e.key == pg.K_d: self.keys['d'] = True
            if e.key == pg.K_b: self.show_buildings = not self.show_buildings
            if e.key == pg.K_v: self.follow_cam = not self.follow_cam
            if e.key == pg.K_c: self.debug_collision = not self.debug_collision
        elif e.type == pg.KEYUP:
            if e.key == pg.K_w: self.keys['w'] = False
            if e.key == pg.K_s: self.keys['s'] = False
            if e.key == pg.K_a: self.keys['a'] = False
            if e.key == pg.K_d: self.keys['d'] = False
        elif e.type == pg.MOUSEBUTTONDOWN and e.button == 1:
            self.dragging = True
            self.last_mouse = e.pos
        elif e.type == pg.MOUSEBUTTONUP and e.button == 1:
            self.dragging = False
        elif e.type == pg.MOUSEMOTION and self.dragging:
            dx = e.pos[0] - self.last_mouse[0]
            dy = e.pos[1] - self.last_mouse[1]
            self.pan = (self.pan[0] + dx, self.pan[1] + dy)
            self.last_mouse = e.pos
        elif e.type == pg.DROPFILE:
            try:
                path = e.file
            except Exception:
                path = getattr(e, 'file', None)
            if path and path.lower().endswith('.svg'):
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        svg_text = f.read()
                    self._load_svg_text(svg_text)
                except Exception as ex:
                    print('Load dropped file failed:', ex)
        elif e.type == pg.MOUSEWHEEL:
            # zoom around mouse position (screen coords)
            mx, my = pg.mouse.get_pos()
            if not self.bbox:
                return
            # world before
            bx = (mx - (self.base_offset[0] + self.pan[0])) / (self.base_scale * self.zoom)
            by = (my - (self.base_offset[1] + self.pan[1])) / (self.base_scale * self.zoom)
            factor = 1.1 if e.y > 0 else 1/1.1
            new_zoom = clamp(self.zoom * factor, 0.1, 50.0)
            # screen after keep world point invariant
            new_pan_x = mx - (bx * self.base_scale * new_zoom + self.base_offset[0])
            new_pan_y = my - (by * self.base_scale * new_zoom + self.base_offset[1])
            self.zoom = new_zoom
            self.pan = (new_pan_x, new_pan_y)

    # --- Simulation step & drawing ---
    def step_and_draw(self, dt: float):
        screen = self.screen
        W, H = screen.get_size()
        # Physics
        if self.car and self.data and self.bbox:
            car = self.car
            car.physics_input(dt, self.keys['w'], self.keys['s'], self.keys['a'], self.keys['d'])
            old_pos = car.pos
            old_yaw = car.yaw
            car.physics_step(dt)

            # Collision
            minx, miny, maxx, maxy = self.bbox
            eps = 0.02
            def is_on_road(x: float, y: float) -> bool:
                mask = self.mask
                if not mask: return False
                for dx in (-eps, 0.0, eps):
                    for dy in (-eps, 0.0, eps):
                        if mask['get'](x+dx, y+dy):
                            return True
                return False

            body, _ = car.body_points()
            samples = [
                *body,
                ((body[0][0] + body[1][0]) / 2, (body[0][1] + body[1][1]) / 2),
                ((body[1][0] + body[2][0]) / 2, (body[1][1] + body[2][1]) / 2),
                ((body[2][0] + body[3][0]) / 2, (body[2][1] + body[3][1]) / 2),
                ((body[3][0] + body[0][0]) / 2, (body[3][1] + body[0][1]) / 2),
                (car.pos[0], car.pos[1]),
            ]
            valid = 0
            for (x,y) in samples:
                if (x >= minx and x <= maxx and y >= miny and y <= maxy and is_on_road(x,y)):
                    valid += 1
            inside = (valid == len(samples))
            if not inside:
                car.pos = old_pos
                car.yaw = old_yaw
                car.v = 0.0

            # Follow cam
            if self.follow_cam:
                cx = car.pos[0] * self.base_scale * self.zoom + self.base_offset[0] + self.pan[0]
                cy = car.pos[1] * self.base_scale * self.zoom + self.base_offset[1] + self.pan[1]
                dx = W/2 - cx
                dy = H/2 - cy
                self.pan = (lerp(self.pan[0], self.pan[0] + dx, 0.15),
                            lerp(self.pan[1], self.pan[1] + dy, 0.15))

        # Draw
        clear(screen, W, H)
        if self.data and self.bbox:
            # water & park
            draw_filled_polys(screen, self.data.get('water'), self.to_screen,
                              fill_override=lambda s: 'rgb(55,68,100)')
            draw_filled_polys(screen, self.data.get('park'), self.to_screen,
                              fill_override=lambda s: 'rgb(60,90,90)')
            # buildings
            if self.show_buildings:
                def _bright(s): return brighten_rgb_string(s, 1.3)
                draw_filled_polys(screen, self.data.get('building'), self.to_screen, fill_override=_bright)

            # roads
            draw_roads(screen, self.data, self.to_screen, self.zoom, self.base_scale)

            # debug mask around car
            if self.debug_collision and self.car and self.mask:
                cx, cy = self.car.pos
                radius = 6.0
                mask = self.mask
                surf = mask['surface']
                origin = mask['origin']
                mask_scale = mask['mask_scale']
                u0 = int((cx - radius - origin[0]) * mask_scale)
                v0 = int((cy - radius - origin[1]) * mask_scale)
                size = int(radius * 2 * mask_scale)
                u0 = max(0, min(surf.get_width()-1, u0))
                v0 = max(0, min(surf.get_height()-1, v0))
                rect = pg.Rect(u0, v0, min(size, surf.get_width()-u0), min(size, surf.get_height()-v0))
                if rect.w > 0 and rect.h > 0:
                    sub = surf.subsurface(rect).copy()
                    sub.set_alpha(150)
                    # draw to screen
                    sx = (cx - radius) * self.base_scale * self.zoom + self.base_offset[0] + self.pan[0]
                    sy = (cy - radius) * self.base_scale * self.zoom + self.base_offset[1] + self.pan[1]
                    sw = (rect.w / mask_scale) * self.base_scale * self.zoom
                    sh = (rect.h / mask_scale) * self.base_scale * self.zoom
                    sub = pg.transform.smoothscale(sub, (int(sw), int(sh)))
                    screen.blit(sub, (sx, sy))

            # start marker
            if self.start_center:
                sx, sy = self.to_screen(self.start_center[0], self.start_center[1])
                pg.draw.circle(screen, (80,255,80), (sx, sy), 6)
            if self.end_center:
                ex, ey = self.to_screen(self.end_center[0], self.end_center[1])
                pg.draw.circle(screen, (255,80,80), (ex, ey), 10, 3)
                pg.draw.circle(screen, (255,80,80), (ex, ey), 4, 0)

            # car
            if self.car:
                body, nose = self.car.body_points()
                def _poly(pts, color):
                    path = [self.to_screen(p[0], p[1]) for p in pts]
                    pg.draw.polygon(screen, color, path, 0)
                _poly(body, self.car.color)
                _poly(nose, (255,255,180))

            # debug body outline & samples
            if self.debug_collision and self.car and self.mask:
                body, _ = self.car.body_points()
                path = [self.to_screen(p[0], p[1]) for p in body]
                pg.draw.polygon(screen, (255,255,180), path, max(1, int(0.01 * self.base_scale * self.zoom)))
                samples = [
                    *body,
                    ((body[0][0] + body[1][0]) / 2, (body[0][1] + body[1][1]) / 2),
                    ((body[1][0] + body[2][0]) / 2, (body[1][1] + body[2][1]) / 2),
                    ((body[2][0] + body[3][0]) / 2, (body[2][1] + body[3][1]) / 2),
                    ((body[3][0] + body[0][0]) / 2, (body[3][1] + body[0][1]) / 2),
                    (self.car.pos[0], self.car.pos[1]),
                ]
                for (x,y) in samples:
                    sx, sy = self.to_screen(x, y)
                    on_road = self.mask['get'](x, y)
                    pg.draw.circle(screen, (0,255,120) if on_road else (255,60,60), (sx, sy), 2)

        # overlay instructions
        self._draw_overlay()

        pg.display.flip()

    def _draw_text(self, surf, pos, text, color=(230,230,230)):
        if hasattr(self.font, 'render_to'):
            self.font.render_to(surf, pos, text, color)
        else:
            font = self.font
            img = font.render(text, True, color)
            surf.blit(img, pos)

    def _draw_overlay(self):
        surf = self.screen
        W, H = surf.get_size()
        # translucent panel
        panel = pg.Surface((360, 120), pg.SRCALPHA)
        # panel.fill((0,0,0,140))
        surf.blit(panel, (12, H-12-120))
        y = H - 12 - 110
        self._draw_text(surf, (20, y), 'Operations:', (220,220,220)); y += 20
        self._draw_text(surf, (20, y), 'W/S: Throttle/Brake, A/D: Turn Left/Right'); y += 20
        self._draw_text(surf, (20, y), 'Mouse wheel: Zoom, Left-click: Drag canvas'); y += 20
        self._draw_text(surf, (20, y), 'B: Show/Hide buildings, V: Camera follow, C: Collision debug'); y += 20
        self._draw_text(surf, (20, y), 'Drag SVG map into the window to start driving'); y += 20


        # simple top toolbar
        zoom_str = f"Zoom: {self.zoom:.2f}"
        panel2 = pg.Surface((W, 40), pg.SRCALPHA); panel2.fill((0,0,0,120))
        surf.blit(panel2, (0,0))
        self._draw_text(surf, (12, 12), 'CityMap')
        self._draw_text(surf, (W-140, 12), zoom_str, (180,180,180))

def run():
    app = App()
    last = time.perf_counter()
    while True:
        dt = min(0.05, time.perf_counter() - last)
        last = time.perf_counter()
        for e in pg.event.get():
            app.handle_event(e)
        app.step_and_draw(dt)
        pg.time.delay(1)
