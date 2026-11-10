// src/constants.js
export const COLOR_MAP = {
  "149,108,62": "main",
  "78,81,84": "secondary",
  "65,68,71": "residential",
  "52,54,56": "building",
  "47,49,51": "building",
  "55,68,100": "water",
  "40,56,56": "park",
};
export const BG_COLOR = "#2b2d2f";

// 视觉/物理参数（世界单位）
export const CENTERLINE_WIDTH = 0.01;
export const LANE_WIDTH = 0.6;
export const ROAD_LANES = {
  main: [3, 3],
  secondary: [2, 2],
  residential: [1, 1],
};
export const roadHalfWidth = (typ) => {
  const [l, r] = ROAD_LANES[typ];
  return ((l + r) * LANE_WIDTH) / 2.0;
};

export const ROAD_DRAW_W = {
  main: 3.0,
  secondary: 2.0,
  residential: 1.0,
};
export const CENTERLINE_COLOR = {
  main: "rgb(220,200,120)",
  secondary: "rgb(180,180,180)",
  residential: "rgb(160,160,160)",
};

export const CAR_LEN = 0.6;
export const CAR_WID = 0.4;
export const NOSE_DOT = 0.1;

export const WHEELBASE = 0.45;
export const MAX_SPEED_FWD = 20;
export const MAX_SPEED_REV = -8;
export const ACCEL = 5;
export const BRAKE = 2;
export const DRAG = 0.3;
export const MAX_STEER_DEG = 32.0;
export const STEER_RATE_DEG = 120.0;
export const STEER_RETURN_DEG = 160.0;

export const clamp = (x, a, b) => (x < a ? a : x > b ? b : x);
export const rad = (d) => (d * Math.PI) / 180.0;
