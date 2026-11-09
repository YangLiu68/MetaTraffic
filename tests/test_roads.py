from metatraffic3d.config import MAIN, MINOR
from metatraffic3d.roads import _lane_pairs


def test_every_driving_lane_has_a_way_out(small_world):
    net = small_world.net
    assert all(lane.out_conns for lane in net.lanes if lane.kind == "drive")


def test_lane_layout_main_8_lanes_sub_4_lanes():
    assert MAIN.drive_lanes * 2 == 8 and not MAIN.parking
    assert MINOR.drive_lanes * 2 + 2 == 4 and MINOR.parking  # 2 driving + 2 parking


def test_turn_lane_rules_on_a_4_lane_carriageway():
    right = _lane_pairs("R", 4, 4)
    left = _lane_pairs("L", 4, 4)
    straight = _lane_pairs("S", 4, 4)
    assert {k for k, _ in right} == {0}  # right turns from the kerb lane
    assert {k for k, _ in left} == {3, 2}  # dual left-turn lanes on the median side
    assert straight == [(k, k) for k in range(4)]  # straight traffic keeps its lane


def test_conflicts_are_symmetric_and_merges_conflict(small_world):
    net = small_world.net
    for c in net.connectors:
        for o in c.conflicts:
            assert c.id in net.connectors[o].conflicts
    for node in net.nodes:
        by_out = {}
        for cid in node.connectors:
            c = net.connectors[cid]
            by_out.setdefault(c.to_lane, []).append(c)
        for group in by_out.values():
            for a in group:
                for b in group:
                    if a.from_lane != b.from_lane:
                        assert b.id in a.conflicts


def test_no_parking_on_bridges(small_world):
    river = small_world.river
    if river is None:
        return
    from shapely import Point

    assert not any(river.contains(Point(s.pose[:2])) for s in small_world.net.spots)
