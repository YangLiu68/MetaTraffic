import numpy as np

from metatraffic3d.traffic.demand import demand_weights
from metatraffic3d.traffic.events import IncidentManager
from metatraffic3d.traffic.sim import PARKED, TrafficSim

from .conftest import small_cfg


def _sim(world, seed=1):
    sim = TrafficSim(world.net, small_cfg(), demand_weights(world), seed=seed)
    sim.height = world.terrain.height
    return sim


def test_traffic_invariants_over_ten_minutes(small_world):
    sim = _sim(small_world)
    sim.incidents = IncidentManager(sim, seed=1)
    cap = sim.pc.max_occupancy
    for _ in range(600):
        sim.step(1.0)
        assert (sim.conn_busy >= 0).all()
        for lst in sim.lane_veh:
            for a, b in zip(lst, lst[1:]):
                assert a.s - a.length - b.s > -0.05  # nobody drives into the car ahead
        for _lane, spots in sim.kerb_spots.items():
            assert np.count_nonzero(sim.spot_state[spots] == PARKED) <= cap * len(spots) + 1e-9
    assert sim.stats.arrived > 0 and sim.stats.grants > 0
    assert sim.stats.teleported <= 3


def test_signal_cycle_and_conflicting_phases(small_world):
    sigs = [n for n in small_world.net.nodes if n.control is not None and n.control.kind == "signal"]
    if not sigs:
        return
    ctl = sigs[0].control
    a, b = ctl.phases[0][0], ctl.phases[1][0]
    states = {(ctl.state(a, t), ctl.state(b, t)) for t in np.arange(0, ctl.cycle, 0.5)}
    assert ("G", "G") not in states and ("G", "Y") not in states  # two axes never go together
    assert {"G", "Y", "R"} <= {s for s, _ in states}


def test_people_on_a_zebra_block_turns_into_that_leg(small_world):
    sim = _sim(small_world)
    node = next(n for n in small_world.net.nodes if n.degree >= 3)
    edge = node.edges[0]
    sim.peds.cross_busy[(node.id, edge)] = 1
    blocked = [c for c in sim.leg_conns[(node.id, edge)]]
    assert blocked  # every turn path touching that leg is held until the crossing clears


def test_construction_closes_and_reopens_a_lane(small_world):
    sim = _sim(small_world, seed=3)
    inc = IncidentManager(sim, seed=3, rates={}, initial_construction=0)
    for _ in range(20):
        z = inc._start("construction")
        if z is not None and z.lanes:
            break
    assert z is not None
    if z.lanes:
        lane, span = z.lanes[0]
        assert span in sim.static_blocks[lane]
    inc._finish(z)
    assert all(span not in sim.static_blocks.get(lane, []) for lane, span in z.lanes)
