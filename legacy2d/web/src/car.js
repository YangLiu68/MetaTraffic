// src/car.js
import {
  CAR_LEN, CAR_WID, NOSE_DOT, WHEELBASE, MAX_SPEED_FWD, MAX_SPEED_REV,
  ACCEL, BRAKE, DRAG, MAX_STEER_DEG, STEER_RATE_DEG, STEER_RETURN_DEG, clamp, rad
} from "./constants.js";

export class Car {
  constructor(pos, heading, roadType) {
    this.pos = new Float64Array([pos[0], pos[1]]);
    this.yaw = Math.atan2(heading[1], heading[0]);
    this.v = 0;
    this.delta = 0;
    this.roadType = roadType;
    this.color = "rgb(255,80,80)";
  }

  physicsInput(dt, throttle, brake, steerLeft, steerRight) {
    if (throttle) this.v += ACCEL * dt;
    if (brake) this.v -= BRAKE * dt;
    if (!throttle && !brake) {
      if (Math.abs(this.v) > 1e-4) {
        const drag = DRAG * dt * (this.v > 0 ? 1 : -1);
        if (Math.abs(drag) > Math.abs(this.v)) this.v = 0; else this.v -= drag;
      }
    }
    this.v = clamp(this.v, MAX_SPEED_REV, MAX_SPEED_FWD);

    const steerRate = rad(STEER_RATE_DEG);
    if (steerLeft) this.delta += steerRate * dt;
    if (steerRight) this.delta -= steerRate * dt;

    if (!steerLeft && !steerRight) {
      const ret = rad(STEER_RETURN_DEG) * dt;
      if (this.delta > 0) this.delta = Math.max(0, this.delta - ret);
      else if (this.delta < 0) this.delta = Math.min(0, this.delta + ret);
    }
    this.delta = clamp(this.delta, -rad(MAX_STEER_DEG), rad(MAX_STEER_DEG));
  }

  physicsStep(dt) {
    if (Math.abs(this.v) < 1e-6) return;
    const beta = Math.tan(this.delta);
    this.pos[0] += Math.cos(this.yaw) * (this.v * dt);
    this.pos[1] += Math.sin(this.yaw) * (this.v * dt);
    this.yaw -= (this.v / WHEELBASE) * beta * dt;
  }

  bodyPoints() {
    const d = [Math.cos(this.yaw), Math.sin(this.yaw)];
    const r = [-d[1], d[0]];
    const L = CAR_LEN;
    const W = CAR_WID;
    const c = this.pos;
    const body = [
      [c[0] + d[0] * (L/2) + r[0]*(W/2), c[1] + d[1]*(L/2) + r[1]*(W/2)],
      [c[0] + d[0] * (L/2) - r[0]*(W/2), c[1] + d[1]*(L/2) - r[1]*(W/2)],
      [c[0] - d[0] * (L/2) - r[0]*(W/2), c[1] - d[1]*(L/2) - r[1]*(W/2)],
      [c[0] - d[0] * (L/2) + r[0]*(W/2), c[1] - d[1]*(L/2) + r[1]*(W/2)],
    ];
    const noseC = [c[0] + d[0]*(L/2 - NOSE_DOT/2), c[1] + d[1]*(L/2 - NOSE_DOT/2)];
    const h = NOSE_DOT/2;
    const nose = [
      [noseC[0] + r[0]*h + d[0]*h, noseC[1] + r[1]*h + d[1]*h],
      [noseC[0] - r[0]*h + d[0]*h, noseC[1] - r[1]*h + d[1]*h],
      [noseC[0] - r[0]*h - d[0]*h, noseC[1] - r[1]*h - d[1]*h],
      [noseC[0] + r[0]*h - d[0]*h, noseC[1] + r[1]*h - d[1]*h],
    ];
    return { body, nose };
  }
}
