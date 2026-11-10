// src/main.js
import {
  BG_COLOR, CENTERLINE_COLOR, CENTERLINE_WIDTH, ROAD_DRAW_W, ROAD_LANES,
  roadHalfWidth, clamp
} from "./constants.js";
import { extractPathsFromSVGString } from "./svg_parser.js";
import { buildRoadMask } from "./mask.js";
import { Car } from "./car.js";
import { findNearestRoadSegment } from "./geometry.js";

// ------- 状态 -------
const state = {
  svgText: null,
  debugCollision: false,
  data: null,
  bbox: null,
  zoom: 1,
  pan: [0, 0],
  dragging: false,
  showBuildings: true,
  followCam: false,
  baseView: null,
  car: null,
  startCenter: null,
  endCenter: null,
  mask: null,
  keys: { w: false, s: false, a: false, d: false },
  lastMouse: [0, 0],
};

// ------- DOM -------
const canvas = document.getElementById("canvas");
const fileInput = document.getElementById("fileInput");
const btnBuildings = document.getElementById("btnBuildings");
const btnFollow = document.getElementById("btnFollow");
const btnDebug = document.getElementById("btnDebug");
const zoomLabel = document.getElementById("zoomLabel");

function updateButtons() {
  btnBuildings.textContent = state.showBuildings ? "隐藏建筑 (B)" : "显示建筑 (B)";
  btnFollow.textContent = state.followCam ? "关闭跟随 (V)" : "开启跟随 (V)";
  btnDebug.textContent = state.debugCollision ? "关闭碰撞调试 (C)" : "开启碰撞调试 (C)";
  zoomLabel.textContent = state.zoom.toFixed(2);
}

btnBuildings.addEventListener("click", ()=>{ state.showBuildings = !state.showBuildings; updateButtons(); });
btnFollow.addEventListener("click", ()=>{ state.followCam = !state.followCam; updateButtons(); });
btnDebug.addEventListener("click", ()=>{ state.debugCollision = !state.debugCollision; updateButtons(); });

fileInput.addEventListener("change", async (e) => {
  const f = e.target.files && e.target.files[0];
  if (f) {
    const txt = await f.text();
    await loadSVG(txt);
  }
});

// 键盘
function onKey(e) {
  const down = e.type === "keydown";
  if (e.repeat) return;
  if (e.code === "KeyW") state.keys.w = down;
  if (e.code === "KeyS") state.keys.s = down;
  if (e.code === "KeyA") state.keys.a = down;
  if (e.code === "KeyD") state.keys.d = down;
  if (down && e.code === "KeyB") { state.showBuildings = !state.showBuildings; updateButtons(); }
  if (down && e.code === "KeyV") { state.followCam = !state.followCam; updateButtons(); }
  if (down && e.code === "KeyC") { state.debugCollision = !state.debugCollision; updateButtons(); }
}
window.addEventListener("keydown", onKey);
window.addEventListener("keyup", onKey);

// 鼠标交互（拖拽、缩放）
canvas.addEventListener("mousedown", (e) => {
  if (e.button === 0) {
    state.dragging = true;
    state.lastMouse = [e.clientX, e.clientY];
  }
});
window.addEventListener("mouseup", (e) => { if (e.button === 0) state.dragging = false; });
window.addEventListener("mousemove", (e) => {
  if (!state.dragging) return;
  const [lx, ly] = state.lastMouse;
  const dx = e.clientX - lx;
  const dy = e.clientY - ly;
  state.pan = [state.pan[0] + dx, state.pan[1] + dy];
  state.lastMouse = [e.clientX, e.clientY];
});

canvas.addEventListener("wheel", (e) => {
  if (!state.baseView) return;
  e.preventDefault();
  const rect = canvas.getBoundingClientRect();
  const mouseX = e.clientX - rect.left;
  const mouseY = e.clientY - rect.top;

  const before = [
    (mouseX - (state.baseView.offset[0] + state.pan[0])) / (state.baseView.scale * state.zoom),
    (mouseY - (state.baseView.offset[1] + state.pan[1])) / (state.baseView.scale * state.zoom),
  ];
  const factor = e.deltaY < 0 ? 1.1 : 1/1.1;
  const newZoom = clamp(state.zoom * factor, 0.1, 50);
  const after = before;
  const newPanX = mouseX - (after[0] * state.baseView.scale * newZoom + state.baseView.offset[0]);
  const newPanY = mouseY - (after[1] * state.baseView.scale * newZoom + state.baseView.offset[1]);
  state.zoom = newZoom;
  state.pan = [newPanX, newPanY];
  updateButtons();
}, { passive: false });

// 载入 SVG
async function loadSVG(svgText) {
  state.svgText = svgText;
  const { data, bbox } = extractPathsFromSVGString(svgText);
  state.data = data;
  state.bbox = bbox;

  const [minx, miny, maxx, maxy] = bbox;
  const wx = maxx - minx;
  const wy = maxy - miny;
  const scale = Math.min(1000 / wx, 800 / wy);
  const offset = [100 - minx * scale, 100 - miny * scale];
  state.baseView = { scale, offset };

  const roads = {
    main: data.main || [],
    secondary: data.secondary || [],
    residential: data.residential || [],
  };

  const mask = buildRoadMask(roads, bbox);
  state.mask = mask;

  // 随机建筑
  if (!data.building || data.building.length === 0) {
    console.error("SVG 未解析到建筑");
  } else {
    const buildings = data.building;
    const pick = buildings[Math.floor(Math.random() * buildings.length)];
    let cx = 0, cy = 0, n = 0;
    for (let i=0; i<pick.pts.length; i+=2) { cx += pick.pts[i]; cy += pick.pts[i+1]; n++; }
    cx /= n; cy /= n;
    state.startCenter = [cx, cy];

    const near = findNearestRoadSegment(roads, cx, cy);
    if (near.typ && near.pts) {
      const right = [-near.diry, near.dirx];
      const spawnOffset = Math.max(0, Math.min(roadHalfWidth(near.typ) - 0.4/2 - 0.02, 0.6 * 0.75));
      state.car = new Car([near.projx + right[0] * spawnOffset, near.projy + right[1] * spawnOffset], [near.dirx, near.diry], near.typ);
    }

    if (buildings.length > 1) {
      let target = null;
      for (let tries=0; tries<10; tries++) {
        const b = buildings[Math.floor(Math.random() * buildings.length)];
        if (b !== pick) { target = b; break; }
      }
      if (target) {
        let tx = 0, ty = 0, tn = 0;
        for (let i=0; i<target.pts.length; i+=2) { tx += target.pts[i]; ty += target.pts[i+1]; tn++; }
        tx /= tn; ty /= tn;
        state.endCenter = [tx, ty];
      }
    }
  }

  state.zoom = 1;
  state.pan = [0, 0];
  updateButtons();
}

// 绘制循环
let raf = 0;
let last = performance.now();

function loop() {
  raf = requestAnimationFrame(loop);
  const now = performance.now();
  const dt = Math.min(0.05, (now - last) / 1000);
  last = now;
  drawFrame(dt);
}
requestAnimationFrame(loop);

function drawFrame(dt) {
  const cvs = canvas;
  const data = state.data;
  const bbox = state.bbox;
  const baseView = state.baseView;
  if (!cvs || !data || !bbox || !baseView) return;
  const g = cvs.getContext("2d");

  const dpr = window.devicePixelRatio || 1;
  const W = Math.floor(cvs.clientWidth * dpr);
  const H = Math.floor(cvs.clientHeight * dpr);
  if (cvs.width !== W || cvs.height !== H) {
    cvs.width = W; cvs.height = H;
  }

  const { scale, offset } = baseView;

  const toScreen = (x, y) => [
    (x * scale * state.zoom + offset[0] + state.pan[0]) * dpr,
    (y * scale * state.zoom + offset[1] + state.pan[1]) * dpr,
  ];

  // 车辆输入 & 物理
  const car = state.car;
  if (car) {
    car.physicsInput(dt, state.keys.w, state.keys.s, state.keys.a, state.keys.d);
    const oldPos = new Float64Array([car.pos[0], car.pos[1]]);
    const oldYaw = car.yaw;
    car.physicsStep(dt);

    // 碰撞
    const [minx, miny, maxx, maxy] = bbox;
    function isOnRoad(x, y, mask) {
      const eps = 0.02;
      for (let dx = -eps; dx <= eps; dx += eps) {
        for (let dy = -eps; dy <= eps; dy += eps) {
          if (mask?.get(x + dx, y + dy)) return true;
        }
      }
      return false;
    }
    const { body } = car.bodyPoints();
    const samples = [
      ...body,
      [(body[0][0] + body[1][0]) / 2, (body[0][1] + body[1][1]) / 2],
      [(body[1][0] + body[2][0]) / 2, (body[1][1] + body[2][1]) / 2],
      [(body[2][0] + body[3][0]) / 2, (body[2][1] + body[3][1]) / 2],
      [(body[3][0] + body[0][0]) / 2, (body[3][1] + body[0][1]) / 2],
      [car.pos[0], car.pos[1]],
    ];
    const valid = samples.filter(([x,y]) => x>=minx && x<=maxx && y>=miny && y<=maxy && isOnRoad(x,y,state.mask)).length;
    const inside = valid === samples.length;
    if (!inside) {
      car.pos.set(oldPos);
      car.yaw = oldYaw;
      car.v = 0;
    }

    // 摄像机跟随
    if (state.followCam) {
      const screenCenter = [W / (2*dpr), H / (2*dpr)];
      const sx = car.pos[0] * scale * state.zoom + offset[0] + state.pan[0];
      const sy = car.pos[1] * scale * state.zoom + offset[1] + state.pan[1];
      const dx = screenCenter[0] - sx;
      const dy = screenCenter[1] - sy;
      const lerp = (a,b,t)=> a + (b-a)*t;
      state.pan = [lerp(state.pan[0], state.pan[0] + dx, 0.15), lerp(state.pan[1], state.pan[1] + dy, 0.15)];
    }
  }

  // 清屏
  g.fillStyle = BG_COLOR;
  g.fillRect(0, 0, W, H);

  // 水域、公园
  const drawFilled = (polys, fill) => {
    if (!polys) return;
    for (const poly of polys) {
      const pts = poly.pts;
      if (pts.length < 6) continue;
      g.beginPath();
      let [sx, sy] = toScreen(pts[0], pts[1]);
      g.moveTo(sx, sy);
      for (let i=2; i<pts.length; i+=2) {
        [sx, sy] = toScreen(pts[i], pts[i+1]);
        g.lineTo(sx, sy);
      }
      const [ex, ey] = toScreen(pts[0], pts[1]);
      g.lineTo(ex, ey);
      g.fillStyle = fill(poly.color);
      g.fill();
      g.lineWidth = 1 * dpr;
      g.strokeStyle = poly.color;
      g.stroke();
    }
  };
  drawFilled(state.data.water, () => "rgb(55,68,100)");
  drawFilled(state.data.park, () => "rgb(60,90,90)");

  // 建筑
  if (state.showBuildings) {
    const buildings = state.data.building;
    if (buildings) {
      for (const poly of buildings) {
        const pts = poly.pts;
        if (pts.length < 6) continue;
        g.beginPath();
        let [sx, sy] = toScreen(pts[0], pts[1]);
        g.moveTo(sx, sy);
        for (let i=2; i<pts.length; i+=2) {
          [sx, sy] = toScreen(pts[i], pts[i+1]);
          g.lineTo(sx, sy);
        }
        const [ex, ey] = toScreen(pts[0], pts[1]);
        g.lineTo(ex, ey);
        const rgb = poly.color.match(/\d+/g).map(Number);
        const brighten = rgb.map((c)=> Math.min(255, Math.floor(c*1.3)));
        g.fillStyle = `rgb(${brighten[0]},${brighten[1]},${brighten[2]})`;
        g.fill();
        g.lineWidth = 1 * dpr;
        g.strokeStyle = poly.color;
        g.stroke();
      }
    }
  }

  // 道路
  (["residential","secondary","main"]).forEach((typ) => {
    const polys = state.data[typ];
    if (!polys) return;
    g.lineCap = "round";
    g.lineJoin = "round";
    for (const poly of polys) {
      const pts = poly.pts;
      if (pts.length < 4) continue;
      // 底色
      g.beginPath();
      let [sx, sy] = toScreen(pts[0], pts[1]);
      g.moveTo(sx, sy);
      for (let i=2; i<pts.length; i+=2) {
        [sx, sy] = toScreen(pts[i], pts[i+1]);
        g.lineTo(sx, sy);
      }
      g.lineWidth = Math.max(1, ROAD_DRAW_W[typ] * state.zoom * dpr);
      g.strokeStyle = poly.color;
      g.stroke();

      // 中心线
      g.beginPath();
      [sx, sy] = toScreen(pts[0], pts[1]);
      g.moveTo(sx, sy);
      for (let i=2; i<pts.length; i+=2) {
        [sx, sy] = toScreen(pts[i], pts[i+1]);
        g.lineTo(sx, sy);
      }
      g.lineWidth = Math.max(1, CENTERLINE_WIDTH * baseView.scale * state.zoom * dpr);
      g.strokeStyle = CENTERLINE_COLOR[typ];
      g.stroke();
    }
  });

  // 掩膜调试
  if (state.debugCollision && state.car && state.mask) {
    const mask = state.mask;
    const ctx = g;
    const car = state.car;

    const cx = car.pos[0];
    const cy = car.pos[1];
    const radius = 6;

    const u0 = (cx - radius - mask.origin[0]) * mask.maskScale;
    const v0 = (cy - radius - mask.origin[1]) * mask.maskScale;
    const size = radius * 2 * mask.maskScale;

    const sx = (cx - radius) * baseView.scale * state.zoom + baseView.offset[0] + state.pan[0];
    const sy = (cy - radius) * baseView.scale * state.zoom + baseView.offset[1] + state.pan[1];
    const sw = size / mask.maskScale * baseView.scale * state.zoom;
    const sh = size / mask.maskScale * baseView.scale * state.zoom;

    ctx.save();
    ctx.globalAlpha = 0.6;
    ctx.filter = "hue-rotate(180deg) brightness(1.8)";
    ctx.drawImage(mask.canvas, u0, v0, size, size, sx * dpr, sy * dpr, sw * dpr, sh * dpr);
    ctx.restore();
  }

  // 起点/终点
  if (state.startCenter) {
    const [sx, sy] = toScreen(state.startCenter[0], state.startCenter[1]);
    g.beginPath();
    g.fillStyle = "rgb(80,255,80)";
    g.arc(sx, sy, 6 * dpr, 0, Math.PI * 2);
    g.fill();
  }

  if (state.endCenter) {
    const [ex, ey] = toScreen(state.endCenter[0], state.endCenter[1]);
    g.beginPath();
    g.arc(ex, ey, 10 * dpr, 0, Math.PI * 2);
    g.lineWidth = 3 * dpr;
    g.strokeStyle = "rgb(255,80,80)";
    g.stroke();

    g.beginPath();
    g.arc(ex, ey, 4 * dpr, 0, Math.PI * 2);
    g.fillStyle = "rgb(255,80,80)";
    g.fill();
  }

  // 车辆
  if (state.car) {
    const { body, nose } = state.car.bodyPoints();
    const drawPoly = (pts, fillStyle) => {
      g.beginPath();
      let [sx, sy] = toScreen(pts[0][0], pts[0][1]);
      g.moveTo(sx, sy);
      for (let i=1; i<pts.length; i++) {
        [sx, sy] = toScreen(pts[i][0], pts[i][1]);
        g.lineTo(sx, sy);
      }
      g.closePath();
      g.fillStyle = fillStyle;
      g.fill();
    };
    drawPoly(body, state.car.color);
    drawPoly(nose, "rgb(255,255,180)");
  }

  if (state.debugCollision && state.car && state.mask) {
    const { body } = state.car.bodyPoints();
    g.beginPath();
    let [sx, sy] = toScreen(body[0][0], body[0][1]);
    g.moveTo(sx, sy);
    for (let i=1; i<body.length; i++) {
      [sx, sy] = toScreen(body[i][0], body[i][1]);
      g.lineTo(sx, sy);
    }
    g.closePath();
    g.lineWidth = 0.01 * baseView.scale * state.zoom * dpr;
    g.strokeStyle = "rgba(255,255,180,0.8)";
    g.stroke();

    const pos = state.car.pos;
    const samples = [
      ...body,
      [(body[0][0] + body[1][0]) / 2, (body[0][1] + body[1][1]) / 2],
      [(body[1][0] + body[2][0]) / 2, (body[1][1] + body[2][1]) / 2],
      [(body[2][0] + body[3][0]) / 2, (body[2][1] + body[3][1]) / 2],
      [(body[3][0] + body[0][0]) / 2, (body[3][1] + body[0][1]) / 2],
      [pos[0], pos[1]],
    ];

    for (const [x,y] of samples) {
      const [dx, dy] = toScreen(x, y);
      const onRoad = state.mask.get(x, y);
      g.beginPath();
      g.fillStyle = onRoad ? "rgba(0,255,120,0.7)" : "rgba(255,60,60,0.7)";
      g.arc(dx, dy, 2 * dpr, 0, Math.PI * 2);
      g.fill();
    }
  }
}

// 初始按钮状态
updateButtons();
