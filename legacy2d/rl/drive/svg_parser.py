import re
import xml.etree.ElementTree as ET
from typing import List, Tuple, Dict
from .types import Poly, SvgData, BBox
from .constants import COLOR_MAP

def _parse_rgb(s: str) -> str | None:
    if not s:
        return None
    s = s.strip().lower()
    m = re.match(r"rgb\((\d+),\s*(\d+),\s*(\d+)\)", s)
    if m:
        r, g, b = [int(m.group(i)) for i in (1,2,3)]
        return f"{r},{g},{b}"
    if s.startswith('#'):
        hexs = s[1:]
        if len(hexs) == 3:
            hexs = ''.join(c*2 for c in hexs)
        try:
            r = int(hexs[0:2], 16)
            g = int(hexs[2:4], 16)
            b = int(hexs[4:6], 16)
            return f"{r},{g},{b}"
        except Exception:
            return None
    return None

def _parse_points(points_str: str | None) -> List[float]:
    if not points_str:
        return []
    nums = re.split(r"[ ,]+", points_str.strip())
    out: List[float] = []
    for x in nums:
        if x:
            try:
                out.append(float(x))
            except ValueError:
                pass
    return out

def extract_paths_from_svg_string(svg_text: str) -> tuple[SvgData, BBox]:
    # Parse XML
    root = ET.fromstring(svg_text)
    ns = ''
    data: Dict[str, List[Poly]] = {}
    for v in set(COLOR_MAP.values()):
        data[v] = []
    # Collect elements
    def walk(node):
        for e in node:
            tag = e.tag.split('}')[-1].lower()
            if tag in ('path', 'polyline', 'polygon', 'line'):
                style = (e.attrib.get('style', '') + ';')
                fill = e.attrib.get('fill', '')
                stroke = e.attrib.get('stroke', '')
                candidates = []
                candidates += [m.group(1) for m in re.finditer(r"fill\s*:\s*([^;]+)", style)]
                if fill: candidates.append(fill)
                if stroke: candidates.append(stroke)
                color_key = None
                rgb_used = None
                for c in candidates:
                    rgb = _parse_rgb(c)
                    if rgb and rgb in COLOR_MAP:
                        color_key = COLOR_MAP[rgb]
                        rgb_used = rgb
                        break
                if not color_key:
                    continue

                pts: List[float] = []
                if tag == 'line':
                    xs = [e.attrib.get('x1',''), e.attrib.get('y1',''), e.attrib.get('x2',''), e.attrib.get('y2','')]
                    try:
                        x1,y1,x2,y2 = map(float, xs)
                        pts = [x1,y1,x2,y2]
                    except Exception:
                        pts = []
                elif tag in ('polyline','polygon'):
                    pts = _parse_points(e.attrib.get('points'))
                elif tag == 'path':
                    d = e.attrib.get('d', '')
                    nums = [float(m.group(0)) for m in re.finditer(r"-?\d+\.?\d*", d)]
                    if len(nums) >= 4:
                        pts = nums
                if len(pts) >= 4:
                    data.setdefault(color_key, []).append(Poly(color=f"rgb({rgb_used})", pts=pts))
            walk(e)
    walk(root)

    # bbox
    all_pts: List[float] = []
    for arr in data.values():
        for p in arr:
            all_pts.extend(p.pts)
    xs = all_pts[0::2] or [0.0]
    ys = all_pts[1::2] or [0.0]
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    return data, (minx, miny, maxx, maxy)
