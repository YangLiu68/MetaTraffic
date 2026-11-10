// src/mask.js
import { ROAD_LANES, LANE_WIDTH } from "./constants.js";

export function buildRoadMask(roads, worldBBox, maxTex = 4096) {
  const [minx, miny, maxx, maxy] = worldBBox;
  const wx = maxx - minx;
  const wy = maxy - miny;
  const scaleX = maxTex / wx;
  const scaleY = maxTex / wy;
  const maskScale = Math.min(scaleX, scaleY) * 4;
  const W = Math.max(8, Math.floor(wx * maskScale));
  const H = Math.max(8, Math.floor(wy * maskScale));

  const dpr = 1;
  const c = document.createElement("canvas");
  c.width = Math.max(8, Math.floor(W * dpr));
  c.height = Math.max(8, Math.floor(H * dpr));
  const g = c.getContext("2d", { willReadFrequently: true });

  g.clearRect(0, 0, c.width, c.height);

  const w2m = (x, y) => {
    const u = (x - minx) * maskScale * dpr;
    const v = (y - miny) * maskScale * dpr;
    return [u, v];
  };

  const order = ["residential","secondary","main"];
  g.strokeStyle = "white";
  g.lineCap = "round";
  g.lineJoin = "round";
  const margins = { "residential": 0.6, "secondary": 1.0, "main": 1.5 };
  for (const typ of order) {
    const [l, r] = ROAD_LANES[typ];
    const totalW = (l + r) * LANE_WIDTH;
    const margin = margins[typ];
    const pxw = Math.max(1, Math.floor((totalW + margin) * maskScale * dpr));
    g.lineWidth = pxw;
    for (const poly of roads[typ]) {
      const pts = poly.pts;
      if (pts.length < 4) continue;
      g.beginPath();
      g.moveTo(...w2m(pts[0], pts[1]));
      for (let i=2; i<pts.length; i+=2) g.lineTo(...w2m(pts[i], pts[i+1]));
      g.stroke();
    }
  }

  const get = (x, y) => {
    const u = Math.floor((x - minx) * maskScale * dpr);
    const v = Math.floor((y - miny) * maskScale * dpr);
    if (u < 0 || v < 0 || u >= c.width || v >= c.height) return false;
    const img = g.getImageData(u, v, 1, 1).data;
    return img[3] > 0 || img[0] > 0;
  };

  return { canvas: c, maskScale, origin: [minx, miny], get };
}
