import os, sys, time, math, argparse
import pygame as pg
import torch
import numpy as np

from drive.rl.env import DrivingEnv
from drive.rl.models import Actor
from drive.render import clear, draw_filled_polys, draw_roads
from drive.utils import brighten_rgb_string, lerp

def load_svg_text(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

def recompute_base_view(bbox, screen_size, margin: float = 100.0):
    W, H = screen_size
    minx, miny, maxx, maxy = bbox
    wx, wy = maxx - minx, maxy - miny
    wx = max(wx, 1e-4)
    wy = max(wy, 1e-4)
    scale = min(max((W - 2*margin) / wx, 1e-4), max((H - 2*margin) / wy, 1e-4))
    offset = (margin - minx * scale, margin - miny * scale)
    return scale, offset

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--svg', required=True, help='Path to map.svg')
    ap.add_argument('--ckpt', default='checkpoints', help='Directory with actor.pt')
    ap.add_argument('--follow', action='store_true', help='Start with follow-cam enabled')
    ap.add_argument('--seed', type=int, default=-1, help='Random seed for env; -1 => randomize every run')
    args = ap.parse_args()

    # --- RL policy ---
    svg_text = load_svg_text(args.svg)
    # Use a different seed each run unless user specifies one
    run_seed = args.seed if args.seed >= 0 else int.from_bytes(os.urandom(4), 'little')
    env = DrivingEnv(svg_text, seed=run_seed)
    obs_dim, act_dim = env.obs_dim, env.act_dim

    device = 'cpu'
    actor = Actor(obs_dim, act_dim).to(device)
    actor_path = os.path.join(args.ckpt, 'actor.pt')
    if os.path.isfile(actor_path):
        actor.load_state_dict(torch.load(actor_path, map_location=device))
        print(f'Loaded policy from {actor_path}')
    else:
        print(f'WARNING: {actor_path} not found. Using untrained policy likely fails.')
    actor.eval()

    # --- Pygame window ---
    pg.init()
    screen = pg.display.set_mode((1280, 900), pg.RESIZABLE | pg.SCALED | pg.DROPFILE)
    pg.display.set_caption('MetaTraffic — RL Demo')
    clock = pg.time.Clock()
    font = pg.font.SysFont(None, 18)

    # --- View transform: base_fit + (zoom, pan) ---
    base_scale, base_offset = recompute_base_view(env.bbox, screen.get_size())
    zoom = 1.0
    pan = [0.0, 0.0]
    dragging = False
    last_mouse = (0,0)
    follow_cam = args.follow

    def to_screen(x, y):
        return int(x * base_scale * zoom + base_offset[0] + pan[0]), int(y * base_scale * zoom + base_offset[1] + pan[1])

    # --- Episode ---
    obs = env.reset()
    ep = 0

    running = True
    show_buildings = True
    debug_collision = False

    while running:
        for e in pg.event.get():
            if e.type == pg.QUIT:
                running = False
            elif e.type == pg.VIDEORESIZE:
                base_scale, base_offset = recompute_base_view(env.bbox, e.size)
            elif e.type == pg.KEYDOWN:
                if e.key == pg.K_v:
                    follow_cam = not follow_cam
                elif e.key == pg.K_b:
                    show_buildings = not show_buildings
                elif e.key == pg.K_c:
                    debug_collision = not debug_collision
            elif e.type == pg.MOUSEBUTTONDOWN and e.button == 1:
                dragging = True
                last_mouse = e.pos
            elif e.type == pg.MOUSEBUTTONUP and e.button == 1:
                dragging = False
            elif e.type == pg.MOUSEMOTION and dragging:
                dx = e.pos[0] - last_mouse[0]
                dy = e.pos[1] - last_mouse[1]
                pan[0] += dx; pan[1] += dy
                last_mouse = e.pos
            elif e.type == pg.MOUSEWHEEL:
                mx, my = pg.mouse.get_pos()
                bx = (mx - (base_offset[0] + pan[0])) / (base_scale * zoom)
                by = (my - (base_offset[1] + pan[1])) / (base_scale * zoom)
                factor = 1.1 if e.y > 0 else 1/1.1
                zoom = max(0.1, min(50.0, zoom * factor))
                pan[0] = mx - (bx * base_scale * zoom + base_offset[0])
                pan[1] = my - (by * base_scale * zoom + base_offset[1])
            elif e.type == pg.DROPFILE:
                path = getattr(e, 'file', None)
                if path and path.lower().endswith('.svg') and os.path.isfile(path):
                    try:
                        new_text = load_svg_text(path)
                        # New env with fresh random seed to avoid fixed start/end
                        new_seed = int.from_bytes(os.urandom(4), 'little')
                        env = DrivingEnv(new_text, seed=new_seed)
                        base_scale, base_offset = recompute_base_view(env.bbox, screen.get_size())
                        zoom = 1.0
                        pan = [0.0, 0.0]
                        obs = env.reset()
                        ep = 0
                        print(f'Loaded new map: {path}')
                    except Exception as ex:
                        print('Load dropped file failed:', ex)

        # --- Policy action ---
        with torch.no_grad():
            a = actor(torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0))[0].cpu().numpy()
        obs2, r, d, info = env.step(a)

        # --- Follow camera (smooth) ---
        if follow_cam and env.car is not None:
            W, H = screen.get_size()
            cx = env.car.pos[0] * base_scale * zoom + base_offset[0] + pan[0]
            cy = env.car.pos[1] * base_scale * zoom + base_offset[1] + pan[1]
            pan[0] = lerp(pan[0], pan[0] + (W/2 - cx), 0.15)
            pan[1] = lerp(pan[1], pan[1] + (H/2 - cy), 0.15)

        # --- Draw ---
        clear(screen, *screen.get_size())
        data = env.data
        draw_filled_polys(screen, data.get('water'), to_screen, fill_override=lambda s: 'rgb(55,68,100)')
        draw_filled_polys(screen, data.get('park'), to_screen, fill_override=lambda s: 'rgb(60,90,90)')
        if show_buildings:
            draw_filled_polys(screen, data.get('building'), to_screen, fill_override=lambda s: brighten_rgb_string(s, 1.3))
        draw_roads(screen, data, to_screen, zoom=zoom, base_scale=base_scale)

        # target marker
        ex, ey = to_screen(env.target[0], env.target[1])
        pg.draw.circle(screen, (255,80,80), (ex,ey), max(2, int(10*zoom)), max(1, int(3*zoom)))
        pg.draw.circle(screen, (255,80,80), (ex,ey), max(2, int(4*zoom)), 0)

        # car
        body, nose = env.car.body_points()
        def poly(pts, color):
            path = [to_screen(p[0], p[1]) for p in pts]
            pg.draw.polygon(screen, color, path, 0)
        poly(body, (255,80,80))
        poly(nose, (255,255,180))
        if debug_collision and hasattr(env, 'mask') and env.mask is not None:
            body, _ = env.car.body_points()
            path = [to_screen(p[0], p[1]) for p in body]
            pg.draw.polygon(screen, (255,255,180), path, max(1, int(0.01 * base_scale * zoom)))
            samples = [
                *body,
                ((body[0][0] + body[1][0]) / 2, (body[0][1] + body[1][1]) / 2),
                ((body[1][0] + body[2][0]) / 2, (body[1][1] + body[2][1]) / 2),
                ((body[2][0] + body[3][0]) / 2, (body[2][1] + body[3][1]) / 2),
                ((body[3][0] + body[0][0]) / 2, (body[3][1] + body[0][1]) / 2),
                (env.car.pos[0], env.car.pos[1]),
            ]
            for (x,y) in samples:
                sx, sy = to_screen(x, y)
                on_road = env.mask['get'](x, y)
                pg.draw.circle(screen, (0,255,120) if on_road else (255,60,60), (sx, sy), 2)


        # HUD
        hud_panel = pg.Surface((760, 80), pg.SRCALPHA); hud_panel.fill((0,0,0,140))
        screen.blit(hud_panel, (12, 8))
        txt1 = font.render(f'Episode {ep}  Dist {info["dist"]:.2f}  Reward {r:.3f}', True, (230,230,230))
        txt2 = font.render('drag map.svg | V: Camera Follow | Scroll: Zoom | LMB: Drag | B: Building | C: Collision Debug', True, (200,200,200))
        screen.blit(txt1, (20, 16))
        screen.blit(txt2, (20, 44))

        pg.display.flip()
        clock.tick(60)

        # step end
        if d:
            # reseed RNG to ensure new random start/end each episode
            env.rng.seed(int.from_bytes(os.urandom(4), 'little'))
            obs = env.reset()
            ep += 1
        else:
            obs = obs2

    pg.quit()

if __name__ == '__main__':
    main()