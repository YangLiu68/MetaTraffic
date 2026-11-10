import os, math, random
import numpy as np
import pygame as pg

from typing import Optional, Tuple, Dict, List
from collections import deque

from ..types import SvgData, Poly, BBox
from ..svg_parser import extract_paths_from_svg_string
from ..mask import build_road_mask
from ..geometry import find_nearest_road_segment
from ..physics import Car
from ..constants import road_half_width, LANE_WIDTH, CAR_WID, MAX_SPEED_FWD
from ..constants import CENTERLINE_WIDTH

class DrivingEnv:
    """
    最小化 Gym-like 环境（不依赖 gym），用于 DDPG/TD3 训练。
    观测：mask 周围 32x32 补丁（0/1），+ 速度标量，朝向向量(cos,sin)，到终点的位移向量(dx,dy)（裁剪并归一化到 [-1,1]）。
    动作：2 维连续向量：throttle_brake ∈ [-1,1]，steer ∈ [-1,1]。
    dt: 0.05s，内部按动作幅度进行分组子步进以保持原物理逻辑一致。
    """
    def __init__(self, svg_text: str, patch_radius_world: float = 6.0, patch_size: int = 32, seed: int = 0):
        self.rng = random.Random(seed)
        self.svg_text = svg_text
        self.data, self.bbox = extract_paths_from_svg_string(svg_text)
        roads = {
            'main': self.data.get('main', []),
            'secondary': self.data.get('secondary', []),
            'residential': self.data.get('residential', []),
        }
        self.mask = build_road_mask(roads, self.bbox)
        self.buildings = self.data.get('building', [])
        self.patch_radius = patch_radius_world
        self.patch_size = patch_size
        self.dt = 0.05
        self.max_steps = 800

        self.car: Optional[Car] = None
        self.target: Optional[Tuple[float,float]] = None
        self.steps = 0
        self.prev_dist = None

        # ===== 新增：shaping/稳定用状态 =====
        self.gamma = 0.995                  # shaping 折扣
        self.prev_potential = 0.0
        self.last_action = np.zeros(2, dtype=np.float32)
        self.offroad_steps = 0
        self.stuck_hist = deque(maxlen=int(1.5 / self.dt))  # 1.5s 进步窗口

    # ---------- helpers ----------
    def _random_start_end(self):
        if not self.buildings:
            roads = self.data.get('main', []) or self.data.get('secondary', []) or self.data.get('residential', [])
            if not roads: raise RuntimeError('No roads found')
            poly = self.rng.choice(roads)
            pts = poly.pts
            i = self.rng.randrange(0, len(pts)-2, 2)
            x0,y0,x1,y1 = pts[i],pts[i+1],pts[i+2],pts[i+3]
            sx, sy = (x0+x1)/2, (y0+y1)/2
            ex, ey = sx + self.rng.uniform(-10,10), sy + self.rng.uniform(-10,10)
            return (sx,sy),(ex,ey)

        pick = self.rng.choice(self.buildings)
        cx = cy = 0.0; n = 0
        for i in range(0, len(pick.pts), 2):
            cx += pick.pts[i]; cy += pick.pts[i+1]; n += 1
        sx, sy = cx/n, cy/n

        target = None
        for _ in range(20):
            b = self.rng.choice(self.buildings)
            if b is not pick:
                tx = ty = 0.0; tn = 0
                for i in range(0, len(b.pts), 2):
                    tx += b.pts[i]; ty += b.pts[i+1]; tn += 1
                target = (tx/tn, ty/tn)
                break
        if not target:
            target = (sx + self.rng.uniform(5,15), sy + self.rng.uniform(5,15))
        return (sx, sy), target

    def _spawn_car_near(self, pos: Tuple[float,float]) -> Optional[Car]:
        roads = {
            'main': self.data.get('main', []),
            'secondary': self.data.get('secondary', []),
            'residential': self.data.get('residential', []),
        }
        near = find_nearest_road_segment(roads, pos[0], pos[1])
        if (not near['typ']) or (not near['pts']):
            return None
        right = (-near['diry'], near['dirx'])
        spawn_offset = max(0.0, min(road_half_width(near['typ']) - CAR_WID/2 - 0.02, LANE_WIDTH * 0.75))
        spawn_pos = (near['projx'] + right[0]*spawn_offset, near['projy'] + right[1]*spawn_offset)
        heading = (near['dirx'], near['diry'])
        return Car.spawn(spawn_pos, heading, near['typ'])

    def _on_road(self, x: float, y: float) -> bool:
        eps = 0.02
        for dx in (-eps, 0.0, eps):
            for dy in (-eps, 0.0, eps):
                if self.mask['get'](x+dx, y+dy):
                    return True
        return False

    def _collision(self, car: Car) -> bool:
        body, _ = car.body_points()
        samples = [
            *body,
            ((body[0][0] + body[1][0]) / 2, (body[0][1] + body[1][1]) / 2),
            ((body[1][0] + body[2][0]) / 2, (body[1][1] + body[2][1]) / 2),
            ((body[2][0] + body[3][0]) / 2, (body[2][1] + body[3][1]) / 2),
            ((body[3][0] + body[0][0]) / 2, (body[3][1] + body[0][1]) / 2),
            (car.pos[0], car.pos[1]),
        ]
        minx, miny, maxx, maxy = self.bbox
        for (x,y) in samples:
            if not (x >= minx and x <= maxx and y >= miny and y <= maxy and self._on_road(x,y)):
                return True
        return False

    def _mask_patch(self, cx: float, cy: float) -> np.ndarray:
        surf = self.mask['surface']
        origin = self.mask['origin']
        mask_scale = self.mask['mask_scale']
        radius = self.patch_radius
        u0 = int((cx - radius - origin[0]) * mask_scale)
        v0 = int((cy - radius - origin[1]) * mask_scale)
        size = int(radius * 2 * mask_scale)
        u0 = max(0, min(surf.get_width()-1, u0))
        v0 = max(0, min(surf.get_height()-1, v0))
        rect = pg.Rect(u0, v0, min(size, surf.get_width()-u0), min(size, surf.get_height()-v0))
        if rect.w <= 0 or rect.h <= 0:
            patch = pg.Surface((self.patch_size, self.patch_size), pg.SRCALPHA)
            arr = pg.surfarray.array3d(patch)[:,:,0]
            return (arr / 255.0).astype(np.float32)
        sub = surf.subsurface(rect).copy()
        sub = pg.transform.smoothscale(sub, (self.patch_size, self.patch_size))
        arr = pg.surfarray.array3d(sub)[:,:,0]  # red channel is enough
        return (arr / 255.0).astype(np.float32)

    def reset(self):
        start, target = self._random_start_end()
        car = self._spawn_car_near(start)
        for _ in range(10):
            if car: break
            start, target = self._random_start_end()
            car = self._spawn_car_near(start)
        if not car:
            raise RuntimeError('Failed to spawn car')
        self.car = car
        self.target = target
        self.steps = 0
        self.prev_dist = self._goal_dist()

        # ===== 新增：shaping/状态初始化 =====
        self.prev_potential = self._potential()
        self.last_action = np.zeros(2, dtype=np.float32)
        self.offroad_steps = 0
        self.stuck_hist.clear()

        return self._get_obs()

    def _goal_dist(self) -> float:
        dx = self.target[0] - self.car.pos[0]
        dy = self.target[1] - self.car.pos[1]
        return math.hypot(dx, dy)

    def _get_obs(self):
        car = self.car
        patch = self._mask_patch(car.pos[0], car.pos[1]).reshape(-1)  # flatten
        v_norm = np.array([car.v / MAX_SPEED_FWD], dtype=np.float32)
        heading = np.array([math.cos(car.yaw), math.sin(car.yaw)], dtype=np.float32)
        dx = self.target[0] - car.pos[0]
        dy = self.target[1] - car.pos[1]
        disp = np.array([dx / self.patch_radius, dy / self.patch_radius], dtype=np.float32)
        disp = np.clip(disp, -1.0, 1.0)
        return np.concatenate([patch.astype(np.float32), v_norm, heading, disp], axis=0)

    @property
    def obs_dim(self):
        return self.patch_size * self.patch_size + 1 + 2 + 2

    @property
    def act_dim(self):
        return 2

    # ====== 下面是奖励 shaping 相关的小工具（不影响外部接口） ======
    def _nearest_centerline_info(self):
        roads = {'main': self.data.get('main', []),
                 'secondary': self.data.get('secondary', []),
                 'residential': self.data.get('residential', [])}
        near = find_nearest_road_segment(roads, self.car.pos[0], self.car.pos[1])
        proj = np.array([near['projx'], near['projy']], dtype=np.float32)
        dirv = np.array([near['dirx'], near['diry']], dtype=np.float32)
        right = np.array([-dirv[1], dirv[0]], dtype=np.float32)
        carp = np.array(self.car.pos, dtype=np.float32)
        lateral = float(np.dot(carp - proj, right))
        half = road_half_width(near['typ'])
        norm_lat = np.clip(abs(lateral) / max(1e-3, half), 0.0, 2.0)
        return dirv, norm_lat

    def _heading_vec(self):
        return np.array([math.cos(self.car.yaw), math.sin(self.car.yaw)], dtype=np.float32)

    def _to_target_dir(self):
        d = np.array([self.target[0] - self.car.pos[0], self.target[1] - self.car.pos[1]], dtype=np.float32)
        n = np.linalg.norm(d) + 1e-6
        return d / n

    def _potential(self):
        dist = self._goal_dist()
        to_t = self._to_target_dir()
        head = self._heading_vec()
        head_align = float(np.dot(head, to_t))           # [-1,1]
        v_tang = float(self.car.v * head_align)          # 沿目标方向速度
        v_tang = np.clip(v_tang / MAX_SPEED_FWD, -1.0, 1.0)
        _, lane_lat = self._nearest_centerline_info()

        # 权重可按需要微调
        w_d, w_h, w_v, w_lane = 0.05, 0.3, 0.4, 0.5
        Phi = (-w_d * dist) + (w_h * head_align) + (w_v * v_tang) - (w_lane * lane_lat)
        return Phi
    def step(self, action: np.ndarray):
        """
        改进版 step():
        - 强化目标导向性 (v_tang)
        - 抑制高速转圈
        - 融入 potential-based shaping
        - 自动检测无进展高速旋转
        """
        t, s = float(np.clip(action[0], -1, 1)), float(np.clip(action[1], -1, 1))
        dt = self.dt
        car = self.car
        done = False

        prev_yaw = car.yaw
        prev_pos = np.array(car.pos, dtype=np.float32)

        # === 低速时削弱转向 ===
        speed = abs(car.v)
        if speed < 0.5:
            s *= (speed / 0.5)

        # === 执行物理步进 ===
        def substep(subdt, throttle, brake, steer_left, steer_right):
            nonlocal done
            if subdt <= 0 or done:
                return
            car.physics_input(subdt, throttle, brake, steer_left, steer_right)
            car.physics_step(subdt)
            if self._collision(car):
                done = True

        a, b = abs(t), abs(s)
        subdt_turn = dt * b
        subdt_drive = dt * a
        if a <= 1e-6:
            subdt_turn = 0.0

        substep(min(subdt_turn, subdt_drive), (t>0), (t<0), (s>0), (s<0))
        substep(max(0.0, subdt_turn - subdt_drive), False, False, (s>0), (s<0))
        substep(max(0.0, subdt_drive - subdt_turn), (t>0), (t<0), False, False)
        rem = dt - (min(subdt_turn, subdt_drive)
                    + max(0.0, subdt_turn - subdt_drive)
                    + max(0.0, subdt_drive - subdt_turn))
        substep(rem, False, False, False, False)

        # === 基础指标 ===
        new_dist = self._goal_dist()
        progress = self.prev_dist - new_dist
        head = self._heading_vec()
        to_t = self._to_target_dir()
        align = float(np.dot(head, to_t))  # [-1,1]
        v_norm = np.clip(car.v / MAX_SPEED_FWD, -1.0, 1.0)
        _, lane_lat = self._nearest_centerline_info()
        onroad = self._on_road(car.pos[0], car.pos[1])
        yaw_rate = abs((car.yaw - prev_yaw + math.pi) % (2*math.pi) - math.pi) / dt

        # === 状态更新 ===
        self.last_action[:] = (t, s)
        self.prev_dist = new_dist
        self.steps += 1
        self.stuck_hist.append(max(0.0, progress))

        # === stuck 检测 ===
        stuck = (
            len(self.stuck_hist) == self.stuck_hist.maxlen
            and sum(self.stuck_hist) < 2.0
            and abs(car.v) < 0.5
        )

        # === 1. 任务层 ===
        r_goal = 0.5 * progress
        r_time = -0.01
        r_success = 10.0 if (new_dist < 10.0 and abs(car.v) < 1.0) else 0.0

        # === 2. 目标导向速度 ===
        v_tang = car.v * align / MAX_SPEED_FWD
        r_v_tang = 0.5 * v_tang  # 奖励沿目标方向的前进速度

        # === 3. 姿态与稳定层 ===
        r_heading = 0.3 * align
        r_lane = -0.3 * (lane_lat ** 2)
        r_offroad = -1.0 if not onroad else 0.0

        # 高速打方向惩罚
        r_spin = -0.1 * abs(yaw_rate) * np.clip(car.v / 2.0, 0, 1)
        # 平滑动作
        r_smooth = -0.05 * abs(s - self.last_action[1])

        # === 4. 高速无进展检测 ===
        if self.steps % 20 == 0:
            progress_ratio = (self.prev_dist - new_dist) / max(1e-3, self.prev_dist)
            if progress_ratio < 0.001 and abs(car.v) > 2.0:
                reward = -5.0
                done = True
                obs = None
                info = {'looping': True, 'dist': new_dist, 'steps': self.steps}
                return obs, reward, done, info

        # === 5. 合并 reward ===
        reward = (
            r_goal + r_v_tang + r_heading + r_lane +
            r_spin + r_offroad + r_smooth + r_time + r_success
        )

        # === 6. potential shaping ===
        new_phi = self._potential()
        shaping = self.gamma * new_phi - self.prev_potential
        self.prev_potential = new_phi
        reward += 0.4 * shaping

        # === Done 条件 ===
        if not onroad:
            self.offroad_steps += 1
        else:
            self.offroad_steps = 0

        if self.offroad_steps >= int(1.0 / self.dt):
            done = True
            reward += -3.0
        if r_success > 0:
            done = True
        if stuck:
            done = True
        if self.steps >= self.max_steps:
            done = True

        reward = float(np.clip(reward, -5.0, 5.0))
        obs = self._get_obs() if not done else None
        info = {'success': r_success > 0, 'dist': new_dist, 'steps': self.steps}
        return obs, reward, done, info
