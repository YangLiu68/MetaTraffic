// src/geometry.js
import { clamp, LANE_WIDTH, roadHalfWidth, ROAD_LANES } from "./constants.js";

export function nearestPointOnSegment(ax, ay, bx, by, px, py) {
  const abx = bx - ax;
  const aby = by - ay;
  const ab2 = abx * abx + aby * aby;
  let t = 0;
  if (ab2 > 0) t = clamp(((px - ax) * abx + (py - ay) * aby) / ab2, 0, 1);
  const projx = ax + t * abx;
  const projy = ay + t * aby;
  const vx = bx - ax;
  const vy = by - ay;
  const n = Math.hypot(vx, vy);
  const dirx = n > 1e-9 ? vx / n : 1;
  const diry = n > 1e-9 ? vy / n : 0;
  const dist = Math.hypot(px - projx, py - projy);
  return { projx, projy, dist, dirx, diry };
}

export function nearestOnPolyline(pts, px, py) {
  let best = { idx: -1, projx: 0, projy: 0, dist: Infinity, dirx: 1, diry: 0 };
  for (let i = 0; i < pts.length - 2; i += 2) {
    const { projx, projy, dist, dirx, diry } = nearestPointOnSegment(
      pts[i], pts[i + 1], pts[i + 2], pts[i + 3], px, py
    );
    if (dist < best.dist) best = { idx: i / 2, projx, projy, dist, dirx, diry };
  }
  return best;
}

export function findNearestRoadSegment(roads, px, py) {
  let best = {
    typ: null, pts: null, idx: -1, projx: 0, projy: 0, dirx: 1, diry: 0, dist: Infinity
  };
  (["main","secondary","residential"]).forEach((typ) => {
    for (const poly of roads[typ]) {
      const r = nearestOnPolyline(poly.pts, px, py);
      if (r.dist < best.dist) best = { typ, pts: poly.pts, idx: r.idx, projx: r.projx, projy: r.projy, dirx: r.dirx, diry: r.diry, dist: r.dist };
    }
  });
  return best;
}
