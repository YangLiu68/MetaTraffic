"""Polyline simplification, ported from simplify-js 1.2 (radial distance + Douglas-Peucker)."""

from __future__ import annotations

Vec = tuple[float, float]


def _sq_seg_dist(p: Vec, p1: Vec, p2: Vec) -> float:
    x, y = p1
    dx = p2[0] - x
    dy = p2[1] - y
    if dx != 0 or dy != 0:
        t = ((p[0] - x) * dx + (p[1] - y) * dy) / (dx * dx + dy * dy)
        if t > 1:
            x, y = p2
        elif t > 0:
            x += dx * t
            y += dy * t
    dx = p[0] - x
    dy = p[1] - y
    return dx * dx + dy * dy


def _simplify_radial(points: list[Vec], sq_tol: float) -> list[Vec]:
    prev = points[0]
    out = [prev]
    point = prev
    for point in points[1:]:
        dx = point[0] - prev[0]
        dy = point[1] - prev[1]
        if dx * dx + dy * dy > sq_tol:
            out.append(point)
            prev = point
    if prev is not point:
        out.append(point)
    return out


def _simplify_dp(points: list[Vec], sq_tol: float) -> list[Vec]:
    """Same output order as the recursive simplify-js step, without Python's recursion limit."""
    last = len(points) - 1
    out = [points[0]]
    stack: list[tuple[int, int]] = [(0, last)]  # (first, last) segment, or (index, -1) = emit
    while stack:
        first, end = stack.pop()
        if end < 0:
            out.append(points[first])
            continue
        max_sq = sq_tol
        index = -1
        p_first, p_end = points[first], points[end]
        for i in range(first + 1, end):
            d = _sq_seg_dist(points[i], p_first, p_end)
            if d > max_sq:
                index = i
                max_sq = d
        if max_sq > sq_tol:
            if end - index > 1:
                stack.append((index, end))
            stack.append((index, -1))
            if index - first > 1:
                stack.append((first, index))
    out.append(points[last])
    return out


def simplify(points: list[Vec], tolerance: float = 1.0, highest_quality: bool = False) -> list[Vec]:
    if len(points) <= 2:
        return list(points)
    sq_tol = tolerance * tolerance
    pts = points if highest_quality else _simplify_radial(points, sq_tol)
    return _simplify_dp(pts, sq_tol)
