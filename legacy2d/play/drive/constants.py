from typing import Dict, Tuple

# 颜色映射（与原版一致，以 'r,g,b' 字符串匹配）
COLOR_MAP: Dict[str, str] = {
    "149,108,62": "main",
    "78,81,84": "secondary",
    "65,68,71": "residential",
    "52,54,56": "building",
    "47,49,51": "building",
    "55,68,100": "water",
    "40,56,56": "park",
}

BG_COLOR = (43, 45, 47)  # #2b2d2f

# 视觉/物理参数（世界单位）
CENTERLINE_WIDTH = 0.01
LANE_WIDTH = 0.6
ROAD_LANES: Dict[str, Tuple[int, int]] = {
    "main": (3, 3),
    "secondary": (2, 2),
    "residential": (1, 1),
}

def road_half_width(typ: str) -> float:
    l, r = ROAD_LANES[typ]
    return ((l + r) * LANE_WIDTH) / 2.0

# 线条像素宽（视觉底色：道路本身）
ROAD_DRAW_W: Dict[str, float] = {
    "main": 3.0,
    "secondary": 2.0,
    "residential": 1.0,
}

CENTERLINE_COLOR: Dict[str, Tuple[int, int, int]] = {
    "main": (220, 200, 120),
    "secondary": (180, 180, 180),
    "residential": (160, 160, 160),
}

# 车辆尺寸与动力学
CAR_LEN = 0.6
CAR_WID = 0.4
NOSE_DOT = 0.1

WHEELBASE = 0.45
MAX_SPEED_FWD = 20.0
MAX_SPEED_REV = -8.0
ACCEL = 5.0
BRAKE = 2.0
DRAG = 0.3
MAX_STEER_DEG = 32.0
STEER_RATE_DEG = 120.0
STEER_RETURN_DEG = 160.0
