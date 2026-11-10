import math
from typing import Tuple

def clamp(x: float, a: float, b: float) -> float:
    return a if x < a else b if x > b else x

def rad(d: float) -> float:
    return d * math.pi / 180.0

def brighten_rgb_string(rgb: str, factor: float = 1.3) -> str:
    # 'rgb(r,g,b)' -> brighten and return 'rgb(r,g,b)'
    import re
    m = re.findall(r"\d+", rgb)
    if len(m) != 3:
        return rgb
    r, g, b = [int(v) for v in m]
    r = min(255, int(r * factor))
    g = min(255, int(g * factor))
    b = min(255, int(b * factor))
    return f"rgb({r},{g},{b})"

def parse_rgb_triplet(s: str) -> tuple:
    # 'rgb(r,g,b)' -> (r,g,b)
    import re
    m = re.findall(r"\d+", s)
    if len(m) != 3:
        return (255, 255, 255)
    return (int(m[0]), int(m[1]), int(m[2]))

def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t
