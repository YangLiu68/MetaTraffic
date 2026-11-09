import math

import pytest

mujoco = pytest.importorskip("mujoco")

from metatraffic3d.backends.mujoco.vehicle import EgoCar, auto_gear, ego_actuators, ego_xml  # noqa: E402


def _run(seconds, policy, slope=0.0, speed=0.0):
    ground = '<geom type="plane" size="500 500 1" contype="1" conaffinity="1"/>'
    if slope:
        ground = f'<geom type="plane" size="500 500 1" contype="1" conaffinity="1" euler="0 {-math.atan(slope)} 0"/>'
    xml = (f'<mujoco><compiler angle="radian"/><option timestep="0.002" integrator="implicitfast"/>'
           f'<worldbody>{ground}{ego_xml()}</worldbody><actuator>{ego_actuators()}</actuator></mujoco>')
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    car = EgoCar(m, d)
    car.place(0, 0, 0.1, 0.0, speed)
    gear = "1"
    for i in range(int(seconds / 0.002)):
        if i % 25 == 0:
            st = car.state(0.05)
            steer, thr, brk, gear = policy(st, gear)
            car.act(steer, thr, brk, gear)
        car.apply(0.002)
        mujoco.mj_step(m, d)
    return car.state(0.05)


def test_accelerates_and_shifts_up():
    st = _run(6, lambda s, g: (0.0, 1.0, 0.0, auto_gear(s.gear, s.speed, 1.0)))
    assert st.speed > 12.0 and st.gear in ("3", "4")


def test_brakes_to_a_stop():
    st = _run(4, lambda s, g: (0.0, 0.0, 1.0, "2"), speed=14.0)
    assert abs(st.speed) < 0.3


def test_gravity_slows_it_uphill():
    flat = _run(5, lambda s, g: (0.0, 0.5, 0.0, auto_gear(s.gear, s.speed, 0.5)))
    hill = _run(5, lambda s, g: (0.0, 0.5, 0.0, auto_gear(s.gear, s.speed, 0.5)), slope=0.12)
    assert hill.speed < flat.speed - 2.0
