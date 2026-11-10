// src/svg_parser.js
import { COLOR_MAP } from "./constants.js";

export function parseRGB(str) {
  if (!str) return null;
  const s = String(str).trim().toLowerCase();
  const m = s.match(/rgb\((\d+),\s*(\d+),\s*(\d+)\)/);
  if (m) return `${parseInt(m[1],10)},${parseInt(m[2],10)},${parseInt(m[3],10)}`;
  if (s.startsWith("#")) {
    let hex = s.slice(1);
    if (hex.length === 3) hex = hex.split("").map((c) => c + c).join("");
    const r = parseInt(hex.slice(0,2),16);
    const g = parseInt(hex.slice(2,4),16);
    const b = parseInt(hex.slice(4,6),16);
    if ([r,g,b].every((v)=>!Number.isNaN(v))) return `${r},${g},${b}`;
  }
  return null;
}

export function parsePoints(pointsStr) {
  if (!pointsStr) return new Float64Array();
  const nums = pointsStr.trim().split(/[ ,]+/).filter(Boolean).map((x)=>parseFloat(x));
  return new Float64Array(nums);
}

export function extractPathsFromSVGString(svgText) {
  const parser = new DOMParser();
  const doc = parser.parseFromString(svgText, "image/svg+xml");
  const elems = Array.from(doc.querySelectorAll("path, polyline, polygon, line"));
  const data = {};
  Object.values(COLOR_MAP).forEach((v) => data[v] = []);

  const pushPoly = (key, colorRgb, pts) => {
    if (pts.length < 4) return;
    data[key] = data[key] || [];
    data[key].push({ color: `rgb(${colorRgb})`, pts: new Float64Array(pts) });
  };

  for (const e of elems) {
    const style = (e.getAttribute("style") || "") + ";";
    const fill = e.getAttribute("fill") || "";
    const stroke = e.getAttribute("stroke") || "";
    const candidates = [
      ...Array.from(style.matchAll(/fill\s*:\s*([^;]+)/g)).map((m) => m[1]),
      fill,
      stroke,
    ];
    let colorKey = null;
    for (const c of candidates) {
      const rgb = parseRGB(c);
      if (rgb && COLOR_MAP[rgb]) { colorKey = COLOR_MAP[rgb]; break; }
    }
    if (!colorKey) continue;

    const tag = e.tagName.toLowerCase();
    let pts = [];
    if (tag === "line") {
      const x1 = parseFloat(e.getAttribute("x1") || "");
      const y1 = parseFloat(e.getAttribute("y1") || "");
      const x2 = parseFloat(e.getAttribute("x2") || "");
      const y2 = parseFloat(e.getAttribute("y2") || "");
      if ([x1,y1,x2,y2].every((n) => Number.isFinite(n))) pts = [x1,y1,x2,y2];
    } else if (tag === "polyline" || tag === "polygon") {
      const arr = Array.from(parsePoints(e.getAttribute("points")));
      pts = arr;
    } else if (tag === "path") {
      const d = e.getAttribute("d") || "";
      const nums = Array.from(d.matchAll(/-?\d+\.?\d*/g)).map((m)=>parseFloat(m[0]));
      if (nums.length >= 4) pts = nums;
    }
    if (pts.length >= 4) pushPoly(colorKey, parseRGB(candidates.find((c)=>parseRGB(c)) || "") || "0,0,0", pts);
  }

  // bbox
  const allPts = [];
  Object.values(data).forEach((arr) => {
    arr.forEach((p) => {
      for (let i=0; i<p.pts.length; i+=2) {
        allPts.push(p.pts[i], p.pts[i+1]);
      }
    });
  });
  const xs = allPts.filter((_,i)=>i%2===0);
  const ys = allPts.filter((_,i)=>i%2===1);
  const minx = Math.min(...xs);
  const miny = Math.min(...ys);
  const maxx = Math.max(...xs);
  const maxy = Math.max(...ys);

  return { data, bbox: [minx, miny, maxx, maxy] };
}
