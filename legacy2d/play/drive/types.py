from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

@dataclass
class Poly:
    color: str               # e.g., 'rgb(52,54,56)'
    pts: List[float]         # flat list [x0, y0, x1, y1, ...]

# SvgData: keys 'main' | 'secondary' | 'residential' 必有，其余可选
SvgData = Dict[str, List[Poly]]
BBox = Tuple[float, float, float, float]
